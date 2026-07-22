"""
Metadata analysis: file properties + EXIF data.

Absence of EXIF data (e.g. camera make/model, GPS) is a weak-but-useful
signal that an image has been re-saved, screenshotted, or stripped by a
messaging app — feeds into ScreenshotAnalyzer's heuristic as well.
"""
import os
from typing import Any

from PIL import Image as PILImage
from PIL.ExifTags import TAGS

from app.analysis.base import ImageAnalyzer


class MetadataAnalyzer(ImageAnalyzer):
    name = "metadata_analyzer"

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        file_size = os.path.getsize(image_path)

        with PILImage.open(image_path) as img:
            width, height = img.size
            fmt = img.format
            exif_raw = img.getexif()

        exif_data = {}
        for tag_id, value in (exif_raw or {}).items():
            tag = TAGS.get(tag_id, str(tag_id))
            # Keep only JSON-serializable primitives.
            if isinstance(value, (str, int, float)):
                exif_data[tag] = value

        has_exif = len(exif_data) > 0
        # Metadata extraction is deterministic, but EXIF richness is a signal:
        # photos with many EXIF fields (camera make, GPS, datetime) are more
        # likely genuine originals. Scale confidence with EXIF field count.
        if has_exif:
            # 1-3 fields → 0.72; 5-8 fields → 0.82-0.90; 10+ → ~0.92
            richness = min(1.0, len(exif_data) / 10.0)
            confidence = 0.70 + 0.22 * richness
        else:
            # No EXIF — could be a screenshot, stripped image, or re-save.
            # Vary slightly based on image dimensions (larger originals are
            # more likely genuine even without EXIF).
            megapixels = (width * height) / 1_000_000
            confidence = min(0.65, 0.55 + megapixels * 0.005)

        return round(confidence, 3), {
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "format": fmt,
            "has_exif": has_exif,
            "exif": exif_data,
        }

