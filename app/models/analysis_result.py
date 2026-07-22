from sqlalchemy import String, Float, Boolean, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base
from app.models.enums import OverallStatus


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id"), unique=True, nullable=False, index=True
    )

    blur_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_blurry: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_low_light: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    duplicate: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(36), nullable=True)
    duplicate_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    ocr_text: Mapped[str | None] = mapped_column(String(512), nullable=True)
    vehicle_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    plate_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    image_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    screenshot: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    tampered: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_status: Mapped[str | None] = mapped_column(SAEnum(OverallStatus), nullable=True)
    summary: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Raw per-analyzer findings, keyed by analyzer name. Kept as JSON so we
    # don't need a migration every time we add/adjust an analyzer's output.
    raw_findings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
