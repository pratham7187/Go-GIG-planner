"""
Local filesystem storage for uploaded images.

Kept as a thin, swappable module — a future S3/GCS-backed implementation
would expose the same `save()` signature so callers don't change.
"""
import os
import uuid

from fastapi import UploadFile

from app.config.settings import get_settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


def save_upload(file: UploadFile) -> tuple[str, str]:
    """
    Persists an UploadFile to disk under a UUID-prefixed name to avoid
    collisions, and returns (stored_filename, absolute_filepath).
    """
    settings = get_settings()

    original_name = file.filename or "upload"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file extension: {ext}")

    os.makedirs(settings.upload_dir, exist_ok=True)

    stored_filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(settings.upload_dir, stored_filename)

    size = 0
    too_large = False
    with open(filepath, "wb") as out_file:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_size_bytes:
                too_large = True
                break
            out_file.write(chunk)

    if too_large:
        os.remove(filepath)
        raise FileTooLargeError(
            f"File exceeds max upload size of {settings.max_upload_size_mb}MB"
        )

    return stored_filename, filepath


def delete_upload(filepath: str) -> None:
    """Remove a managed upload, refusing paths outside the upload directory."""
    upload_dir = os.path.realpath(get_settings().upload_dir)
    target = os.path.realpath(filepath)
    if os.path.commonpath([upload_dir, target]) != upload_dir:
        raise ValueError("Refusing to delete a file outside the upload directory")
    if os.path.isfile(target):
        os.remove(target)
