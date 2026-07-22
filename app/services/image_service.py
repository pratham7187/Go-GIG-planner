"""
ImageService: the business-logic layer for upload + async processing.

Controllers call into this service and never touch repositories or the
Processor directly. This is where retry logic, status transitions, and
orchestration between the file storage / DB / analyzers all live.
"""
import threading

from fastapi import UploadFile

from app.analysis.processor import Processor
from app.config.settings import get_settings
from app.database.session import db_session_scope
from app.models.enums import ProcessingStatus
from app.models.image import Image
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.image_repository import ImageRepository
from app.utils.exceptions import ImageNotFoundError
from app.utils.file_storage import delete_upload, save_upload
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
processor = Processor()

# FastAPI's BackgroundTasks run concurrently on a thread pool. Without
# serializing the "read existing hashes -> compute pHash -> write result"
# sequence, two near-simultaneous uploads of the same image can both read
# the hash table before either has committed, and neither will see the
# other as a duplicate. A single process-wide lock is a deliberately
# simple fix for a single-instance deployment; a multi-instance deployment
# would need a DB-level unique constraint or advisory lock instead (see
# README "Trade-offs").
_duplicate_check_lock = threading.Lock()


class ImageService:
    def __init__(self, image_repo: ImageRepository, analysis_repo: AnalysisRepository) -> None:
        self.image_repo = image_repo
        self.analysis_repo = analysis_repo

    def upload_image(self, file: UploadFile) -> Image:
        stored_filename, filepath = save_upload(file)
        image = self.image_repo.create(filename=stored_filename, filepath=filepath)
        logger.info("Image uploaded: id=%s filename=%s", image.id, stored_filename)
        return image

    def get_image_or_raise(self, image_id: str) -> Image:
        image = self.image_repo.get(image_id)
        if image is None:
            raise ImageNotFoundError(image_id)
        return image

    def get_analysis_or_none(self, image_id: str):
        return self.analysis_repo.get_by_image_id(image_id)

    def delete_image(self, image_id: str) -> None:
        """Delete an upload, its analysis, and its duplicate-detection hash."""
        # This is the same lock used while a worker calculates/persists a hash.
        # It prevents a deleted processing job from being written back later.
        with _duplicate_check_lock:
            image = self.get_image_or_raise(image_id)
            delete_upload(image.filepath)
            self.analysis_repo.delete_by_image_id(image.id, commit=False)
            self.image_repo.delete(image, commit=False)
            self.image_repo.db.commit()

    def clear_history(self) -> int:
        """Delete every upload and its analysis in one scoped operation."""
        with _duplicate_check_lock:
            images, _ = self.list_images(limit=100000, offset=0)
            for image in images:
                delete_upload(image.filepath)
            self.analysis_repo.delete_all(commit=False)
            deleted = self.image_repo.delete_all(commit=False)
            self.image_repo.db.commit()
            return deleted

    def list_images(self, limit: int, offset: int):
        return self.image_repo.list_all(limit=limit, offset=offset), self.image_repo.count_all()

    def get_dashboard_stats(self) -> dict:
        from sqlalchemy import select, func
        from app.models.analysis_result import AnalysisResult

        total = self.image_repo.count_all()
        pending = self.image_repo.count_by_status(ProcessingStatus.PENDING)
        processing = self.image_repo.count_by_status(ProcessingStatus.PROCESSING)
        completed = self.image_repo.count_by_status(ProcessingStatus.COMPLETED)
        failed = self.image_repo.count_by_status(ProcessingStatus.FAILED)

        db = self.image_repo.db
        duplicates = db.scalar(
            select(func.count()).select_from(AnalysisResult).where(AnalysisResult.duplicate.is_(True))
        ) or 0
        avg_confidence = db.scalar(select(func.avg(AnalysisResult.confidence_score)))
        valid_plates = db.scalar(
            select(func.count()).select_from(AnalysisResult).where(AnalysisResult.plate_valid.is_(True))
        ) or 0

        plate_rate = round(valid_plates / completed, 3) if completed else None

        return {
            "total_images": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "duplicates_detected": duplicates,
            "average_confidence_score": round(avg_confidence, 3) if avg_confidence is not None else None,
            "plate_validation_rate": plate_rate,
        }


def process_image_task(image_id: str) -> None:
    """
    Entry point for the background task. Opens its own DB session scope
    since BackgroundTasks run outside the request's session lifecycle.

    Retries the whole analyzer pipeline up to settings.retry_limit times on
    unexpected failures (e.g. transient file-read errors) before marking
    the image FAILED with the captured error message.
    """
    with db_session_scope() as db:
        image_repo = ImageRepository(db)
        analysis_repo = AnalysisRepository(db)

        image = image_repo.get(image_id)
        if image is None:
            logger.error("process_image_task: image %s not found", image_id)
            return

        image_repo.update_status(image, ProcessingStatus.PROCESSING)
        logger.info("Processing started: id=%s", image_id)

        attempt = 0
        last_error: str | None = None

        while attempt <= settings.retry_limit:
            try:
                with _duplicate_check_lock:
                    existing_hashes = image_repo.get_completed_hashes(exclude_id=image_id)
                    result = processor.process(image.filepath, existing_hashes)

                    computed_hash = result.pop("computed_hash", None)

                    # Persist analysis BEFORE flipping status to COMPLETED, so a
                    # write failure here can never leave the image marked
                    # COMPLETED with no corresponding analysis row.
                    analysis_repo.upsert(image_id, result)
                    image_repo.update_status(
                        image,
                        ProcessingStatus.COMPLETED,
                        image_hash=computed_hash,
                    )
                logger.info("Processing completed: id=%s status=%s", image_id, result["overall_status"])
                return
            except Exception as exc:  # noqa: BLE001 — top-level retry boundary
                attempt += 1
                last_error = str(exc)
                # A failed flush leaves the session's transaction unusable
                # until rolled back — without this, every subsequent retry
                # attempt (and even increment_retry itself) fails too.
                db.rollback()
                image_repo.increment_retry(image)
                logger.warning(
                    "Processing attempt %d/%d failed for id=%s: %s",
                    attempt, settings.retry_limit + 1, image_id, last_error,
                )

        image_repo.update_status(image, ProcessingStatus.FAILED, error_message=last_error)
        logger.error("Processing failed permanently: id=%s error=%s", image_id, last_error)
