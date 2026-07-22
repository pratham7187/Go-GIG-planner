from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.image_repository import ImageRepository
from app.services.image_service import ImageService


def get_image_service(db: Session = Depends(get_db)) -> ImageService:
    return ImageService(
        image_repo=ImageRepository(db),
        analysis_repo=AnalysisRepository(db),
    )
