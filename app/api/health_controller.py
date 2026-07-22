from fastapi import APIRouter

from app.config.settings import get_settings

router = APIRouter()


@router.get("/health")
def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "env": settings.env,
        "gemini_enabled": settings.gemini_enabled,
    }
