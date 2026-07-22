from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File, Query
from fastapi.responses import JSONResponse

from app.api.dependencies import get_image_service
from app.models.enums import ProcessingStatus
from app.schemas.image_schemas import (
    UploadResponse,
    StatusResponse,
    AnalysisResultResponse,
    PaginatedImages,
    DashboardStats,
    DeleteResponse,
)
from app.services.image_service import ImageService, process_image_task
from app.utils.exceptions import ImageNotFoundError
from app.utils.file_storage import UnsupportedFileTypeError, FileTooLargeError

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=202)
def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: ImageService = Depends(get_image_service),
):
    try:
        image = service.upload_image(file)
    except UnsupportedFileTypeError as exc:
        return JSONResponse(status_code=415, content={"error": "unsupported_file_type", "detail": str(exc)})
    except FileTooLargeError as exc:
        return JSONResponse(status_code=413, content={"error": "file_too_large", "detail": str(exc)})

    background_tasks.add_task(process_image_task, image.id)
    return UploadResponse(id=image.id, status=ProcessingStatus.PENDING)


@router.get("/status/{image_id}", response_model=StatusResponse)
def get_status(image_id: str, service: ImageService = Depends(get_image_service)):
    image = service.get_image_or_raise(image_id)  # raises ImageNotFoundError -> global handler
    return StatusResponse(
        id=image.id,
        status=image.status,
        uploaded_at=image.uploaded_at,
        completed_at=image.completed_at,
        error_message=image.error_message,
        retry_count=image.retry_count,
    )


@router.get("/result/{image_id}", response_model=AnalysisResultResponse)
def get_result(image_id: str, service: ImageService = Depends(get_image_service)):
    image = service.get_image_or_raise(image_id)
    analysis = service.get_analysis_or_none(image_id)

    if analysis is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "analysis_not_ready",
                "detail": f"Image status is '{image.status.value}'. Analysis is available once status is COMPLETED.",
            },
        )
    return AnalysisResultResponse.model_validate(analysis)


@router.get("/images", response_model=PaginatedImages)
def list_images(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: ImageService = Depends(get_image_service),
):
    items, total = service.list_images(limit=limit, offset=offset)
    return PaginatedImages(total=total, limit=limit, offset=offset, items=items)


@router.get("/stats", response_model=DashboardStats)
def get_stats(service: ImageService = Depends(get_image_service)):
    return DashboardStats(**service.get_dashboard_stats())


@router.delete("/images/{image_id}", response_model=DeleteResponse)
def delete_image(image_id: str, service: ImageService = Depends(get_image_service)):
    service.delete_image(image_id)
    return DeleteResponse()


@router.delete("/images", response_model=DeleteResponse)
def clear_history(service: ImageService = Depends(get_image_service)):
    deleted = service.clear_history()
    return DeleteResponse(deleted=deleted)
