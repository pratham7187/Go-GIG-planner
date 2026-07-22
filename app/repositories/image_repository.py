"""
Repository layer for Image records.

Keeps SQLAlchemy query logic out of services — services express business
operations ("mark this image as failed"), repositories express storage
operations ("update this row"). This split makes it possible to swap the
persistence layer without touching business logic.
"""
from datetime import datetime, timezone

from sqlalchemy import delete, select, func
from sqlalchemy.orm import Session

from app.models.image import Image
from app.models.enums import ProcessingStatus


class ImageRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, filename: str, filepath: str) -> Image:
        image = Image(filename=filename, filepath=filepath, status=ProcessingStatus.PENDING)
        self.db.add(image)
        self.db.commit()
        self.db.refresh(image)
        return image

    def get(self, image_id: str) -> Image | None:
        return self.db.get(Image, image_id)

    def list_all(self, limit: int = 50, offset: int = 0) -> list[Image]:
        stmt = select(Image).order_by(Image.uploaded_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def count_all(self) -> int:
        return self.db.scalar(select(func.count()).select_from(Image)) or 0

    def count_by_status(self, status: ProcessingStatus) -> int:
        stmt = select(func.count()).select_from(Image).where(Image.status == status)
        return self.db.scalar(stmt) or 0

    def get_completed_hashes(self, exclude_id: str) -> dict[str, str]:
        """Return {image_id: hash} for all previously completed images, used
        by DuplicateAnalyzer to compare against the current upload."""
        stmt = select(Image.id, Image.hash).where(
            Image.status == ProcessingStatus.COMPLETED,
            Image.hash.is_not(None),
            Image.id != exclude_id,
        )
        return {row.id: row.hash for row in self.db.execute(stmt)}

    def update_status(
        self,
        image: Image,
        status: ProcessingStatus,
        error_message: str | None = None,
        image_hash: str | None = None,
    ) -> Image:
        image.status = status
        if error_message is not None:
            image.error_message = error_message
        if image_hash is not None:
            image.hash = image_hash
        if status in (ProcessingStatus.COMPLETED, ProcessingStatus.FAILED):
            image.completed_at = datetime.now(timezone.utc)
        self.db.add(image)
        self.db.commit()
        self.db.refresh(image)
        return image

    def increment_retry(self, image: Image) -> Image:
        image.retry_count += 1
        self.db.add(image)
        self.db.commit()
        self.db.refresh(image)
        return image

    def delete(self, image: Image, commit: bool = True) -> None:
        """Delete one image row. Its analysis must be removed first."""
        self.db.delete(image)
        if commit:
            self.db.commit()

    def delete_all(self, commit: bool = True) -> int:
        """Delete all image rows. Their analyses must be removed first."""
        result = self.db.execute(delete(Image))
        if commit:
            self.db.commit()
        return result.rowcount or 0
