from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.analysis_result import AnalysisResult


class AnalysisRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_image_id(self, image_id: str) -> AnalysisResult | None:
        stmt = select(AnalysisResult).where(AnalysisResult.image_id == image_id)
        return self.db.scalar(stmt)

    def upsert(self, image_id: str, data: dict) -> AnalysisResult:
        existing = self.get_by_image_id(image_id)
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
            result = existing
        else:
            result = AnalysisResult(image_id=image_id, **data)
            self.db.add(result)

        self.db.commit()
        self.db.refresh(result)
        return result

    def delete_by_image_id(self, image_id: str, commit: bool = True) -> None:
        self.db.execute(delete(AnalysisResult).where(AnalysisResult.image_id == image_id))
        if commit:
            self.db.commit()

    def delete_all(self, commit: bool = True) -> None:
        self.db.execute(delete(AnalysisResult))
        if commit:
            self.db.commit()
