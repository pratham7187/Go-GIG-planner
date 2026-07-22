from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.enums import ProcessingStatus, OverallStatus


class UploadResponse(BaseModel):
    id: str
    status: ProcessingStatus


class StatusResponse(BaseModel):
    id: str
    status: ProcessingStatus
    uploaded_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int


class AnalyzerFindingSchema(BaseModel):
    name: str
    status: str
    confidence: float
    findings: dict[str, Any]
    execution_time: float
    error: str | None = None


class AnalysisResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    image_id: str
    blur_score: float | None
    is_blurry: bool | None
    brightness: float | None
    is_low_light: bool | None
    duplicate: bool | None
    duplicate_of: str | None
    duplicate_type: str | None
    ocr_text: str | None
    vehicle_number: str | None
    plate_valid: bool | None
    image_metadata: dict[str, Any] | None
    screenshot: bool | None
    tampered: bool | None = None
    confidence_score: float | None
    overall_status: OverallStatus | None
    summary: str | None
    raw_findings: dict[str, Any] | None


class ImageListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: ProcessingStatus
    uploaded_at: datetime
    completed_at: datetime | None = None


class PaginatedImages(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ImageListItem]


class DashboardStats(BaseModel):
    total_images: int
    pending: int
    processing: int
    completed: int
    failed: int
    duplicates_detected: int
    average_confidence_score: float | None
    plate_validation_rate: float | None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class DeleteResponse(BaseModel):
    success: bool = True
    deleted: int = 1
