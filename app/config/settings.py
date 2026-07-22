"""
Centralized application configuration.

All environment-driven values are declared here so the rest of the codebase
never touches `os.environ` directly. This keeps configuration testable and
makes it obvious what the system depends on at runtime.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


# Common Tesseract install locations on Windows.
_TESSERACT_SEARCH_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    r"C:\Tesseract-OCR\tesseract.exe",
]


def _find_tesseract() -> str | None:
    """Auto-detect Tesseract binary on the system."""
    import shutil
    # Check if it's already on PATH.
    path_result = shutil.which("tesseract")
    if path_result:
        return path_result
    # Check common Windows install locations.
    for candidate in _TESSERACT_SEARCH_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None


class Settings(BaseSettings):
    app_name: str = "Vehicle Image Processing Pipeline"
    env: str = "development"
    log_level: str = "INFO"

    upload_dir: str = "uploads"
    max_upload_size_mb: int = 10

    database_url: str = "sqlite:///./data/app.db"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    # Tesseract OCR path — set explicitly if not on PATH.
    # Auto-detected at startup if left blank.
    tesseract_path: str | None = None

    retry_limit: int = 2
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def resolved_tesseract_path(self) -> str | None:
        """Return explicit path if set, otherwise auto-detect."""
        if self.tesseract_path:
            return self.tesseract_path
        return _find_tesseract()


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so we parse the environment exactly once."""
    return Settings()

