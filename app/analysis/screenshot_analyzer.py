"""
Screenshot / overlay / photo-of-photo detection.

Combines multiple heuristic signals:
  1. Missing EXIF (screenshots/re-saves rarely carry camera EXIF)
  2. Common screenshot aspect ratios (device screen dimensions)
  3. Flat, low-variance border regions (status bars, nav bars)
  4. GPS overlay bars — semi-transparent dark bars at bottom with text/map
  5. Watermark/logo detection in corners (GoGig, GPS camera app logos)
  6. Text overlay bands — uniform-color horizontal strips containing text
  7. Timestamp/task-ID patterns detected in image text

This is explicitly a heuristic; GeminiAnalyzer (when enabled) provides a
second, independent opinion that the Processor can weigh alongside this.
"""
from typing import Any
import re

import cv2
import numpy as np

from app.analysis.base import ImageAnalyzer
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Common *phone screen* aspect ratios (short:long side).
KNOWN_SCREEN_RATIOS = {0.42, 0.46, 0.5, 0.56}
RATIO_TOLERANCE = 0.02


def _detect_overlay_bar(image: np.ndarray) -> dict[str, Any]:
    """Detect semi-transparent overlay bars (GPS camera apps, timestamps).

    These appear as dark, horizontally-uniform strips at the top or bottom
    of the image, typically 5-25% of image height. They have:
      - Low mean intensity (dark / semi-transparent)
      - Low vertical variance (uniform horizontally)
      - Significantly different from the image content above/below them
    """
    h, w = image.shape[:2]
    results = {"has_bottom_bar": False, "has_top_bar": False,
               "bottom_bar_height_pct": 0.0, "top_bar_height_pct": 0.0}

    # Check bottom region (most common for GPS overlays).
    for bar_pct in [0.25, 0.20, 0.15, 0.10]:
        bar_h = int(h * bar_pct)
        if bar_h < 20:
            continue

        bottom_bar = image[h - bar_h:, :]
        content_above = image[h - bar_h - bar_h: h - bar_h, :]

        bar_mean = float(bottom_bar.mean())
        bar_std = float(bottom_bar.std())
        content_mean = float(content_above.mean())

        # Overlay bars are darker than content and relatively uniform.
        if (bar_mean < 80 and  # dark bar
            abs(bar_mean - content_mean) > 30 and  # significantly different from content
            bar_std < content_above.std() * 0.8):  # more uniform than content
            results["has_bottom_bar"] = True
            results["bottom_bar_height_pct"] = round(bar_pct, 2)
            break

    # Check top region.
    for bar_pct in [0.10, 0.08, 0.05]:
        bar_h = int(h * bar_pct)
        if bar_h < 15:
            continue

        top_bar = image[:bar_h, :]
        content_below = image[bar_h: bar_h * 2, :]

        bar_mean = float(top_bar.mean())
        content_mean = float(content_below.mean())

        if (bar_mean < 60 and
            abs(bar_mean - content_mean) > 30 and
            top_bar.std() < content_below.std() * 0.7):
            results["has_top_bar"] = True
            results["top_bar_height_pct"] = round(bar_pct, 2)
            break

    return results


def _detect_corner_watermark(image: np.ndarray) -> dict[str, Any]:
    """Detect watermarks/logos in image corners.

    Watermarks typically appear as small, visually distinct regions in
    corners — they often have different intensity/color characteristics
    compared to the surrounding image content.
    """
    h, w = image.shape[:2]
    results = {"has_corner_element": False, "corner_location": None}

    # Check each corner (10% × 10% of image).
    corner_h = max(30, h // 10)
    corner_w = max(40, w // 8)

    corners = {
        "bottom_right": image[h - corner_h:, w - corner_w:],
        "bottom_left": image[h - corner_h:, :corner_w],
        "top_right": image[:corner_h, w - corner_w:],
        "top_left": image[:corner_h, :corner_w],
    }

    for name, corner in corners.items():
        # Look for distinct elements: edges, text-like features.
        edges = cv2.Canny(corner, 50, 150)
        edge_density = float(edges.mean()) / 255.0

        # A watermark/logo typically has moderate edge density (not too sparse,
        # not full image content). Compare with the main image content.
        center = image[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        center_edges = cv2.Canny(center, 50, 150)
        center_density = float(center_edges.mean()) / 255.0

        # Watermark: has visible edges but is visually distinct from content.
        # Also check if the corner has a somewhat uniform background
        # (overlay element on semi-transparent bg).
        corner_std = float(corner.std())

        if (0.02 < edge_density < 0.25 and
            corner_std < 60 and
            corner_std < center.std() * 0.7):
            results["has_corner_element"] = True
            results["corner_location"] = name
            break

    return results


def _detect_text_overlay_bands(image: np.ndarray) -> dict[str, Any]:
    """Detect horizontal text overlay bands — uniform colored strips
    containing text, common in GPS camera apps and task management overlays.

    These bands have:
      - Very low vertical variance within the band (uniform background)
      - Sharp horizontal edges at top/bottom of the band
      - Significantly different mean intensity from adjacent content
    """
    h, w = image.shape[:2]
    results = {"has_text_bands": False, "band_count": 0}

    band_count = 0

    # Scan in horizontal slices from bottom to top.
    slice_h = max(15, h // 30)  # each slice is ~3% of image height

    for y in range(h - slice_h, h // 2, -slice_h):
        band = image[y: y + slice_h, :]
        above = image[max(0, y - slice_h): y, :]

        if band.size == 0 or above.size == 0:
            continue

        band_std = float(band.std())
        above_std = float(above.std())
        band_mean = float(band.mean())
        above_mean = float(above.mean())

        # Text overlay band: uniform background with some text (low std),
        # and sharp transition from the band to content above.
        if (band_std < 35 and
            above_std > band_std * 1.5 and
            abs(band_mean - above_mean) > 20):
            band_count += 1

    results["has_text_bands"] = band_count >= 2
    results["band_count"] = band_count
    return results


class ScreenshotAnalyzer(ImageAnalyzer):
    name = "screenshot_analyzer"

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        metadata_findings = context.get("metadata_findings", {})
        ocr_findings = context.get("ocr_findings", {})
        has_exif = metadata_findings.get("has_exif", False)
        width = metadata_findings.get("width")
        height = metadata_findings.get("height")

        signals: dict[str, Any] = {"missing_exif": not has_exif}
        ocr_text = ocr_findings.get("raw_text", "").upper()
        overlay_text_hits = {
            "task_id": bool(re.search(r"\bTASK\s*ID\b", ocr_text)),
            "gps_coordinates": bool(re.search(r"\b(?:LAT|LONG|LON)\s*[:.]", ocr_text)),
            "timestamp": bool(re.search(r"\b(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\b", ocr_text)),
            "gogig_branding": "GOGIG" in ocr_text,
        }
        signals["overlay_text_hits"] = overlay_text_hits
        overlay_text_score = sum(overlay_text_hits.values())

        # --- Aspect ratio check ---
        ratio_match = False
        if width and height:
            ratio = min(width, height) / max(width, height)
            ratio_match = any(abs(ratio - r) < RATIO_TOLERANCE
                            for r in KNOWN_SCREEN_RATIOS)
        signals["screen_ratio_match"] = ratio_match

        # --- Flat border detection (status bar / nav bar) ---
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        flat_border = False
        overlay_bar = {"has_bottom_bar": False, "has_top_bar": False}
        corner_watermark = {"has_corner_element": False}
        text_bands = {"has_text_bands": False, "band_count": 0}

        if image is not None:
            h = image.shape[0]
            top_strip = image[0: max(1, h // 20), :]
            bottom_strip = image[-max(1, h // 20):, :]
            flat_border = bool(top_strip.std() < 10 or bottom_strip.std() < 10)

            # --- NEW: Overlay detection ---
            overlay_bar = _detect_overlay_bar(image)
            corner_watermark = _detect_corner_watermark(image)
            text_bands = _detect_text_overlay_bands(image)

        signals["flat_border_detected"] = flat_border
        signals["overlay_bar_detected"] = (overlay_bar.get("has_bottom_bar", False) or
                                           overlay_bar.get("has_top_bar", False))
        signals["corner_watermark_detected"] = corner_watermark.get("has_corner_element", False)
        signals["text_bands_detected"] = text_bands.get("has_text_bands", False)
        signals["overlay_details"] = {
            **overlay_bar,
            **corner_watermark,
            **text_bands,
        }

        # --- Scoring ---
        # Classic screenshot signals (weak individually).
        classic_score = sum([
            signals["missing_exif"],
            signals["screen_ratio_match"],
            signals["flat_border_detected"],
        ])

        # Overlay signals (strong individually — these are definitive).
        overlay_score = sum([
            signals["overlay_bar_detected"],
            signals["corner_watermark_detected"],
            signals["text_bands_detected"],
        ])

        # An image is a screenshot/overlay if:
        # - screen geometry plus an actual UI border (missing EXIF alone is
        #   common after normal messaging uploads), OR
        # - 1+ overlay signal + missing EXIF, OR
        # - recognisable GPS/task/timestamp/GoGig overlay text.
        is_screenshot = (
            (signals["screen_ratio_match"] and signals["flat_border_detected"]) or
            # A lone corner element is too ambiguous (tail lights, plates and
            # ad artwork frequently occupy a corner of a normal photo).
            ((signals["overlay_bar_detected"] or signals["text_bands_detected"]) and not has_exif) or
            overlay_score >= 2 or
            overlay_text_score >= 1
        )

        # --- Confidence ---
        # Build confidence from individual signal weights.
        signal_weight = 0.0
        if signals["missing_exif"]:
            signal_weight += 0.08
        if signals["screen_ratio_match"]:
            signal_weight += 0.15
        if signals["flat_border_detected"]:
            signal_weight += 0.12
        if signals["overlay_bar_detected"]:
            signal_weight += 0.25  # strong signal
        if signals["corner_watermark_detected"]:
            signal_weight += 0.18
        if signals["text_bands_detected"]:
            signal_weight += 0.15
        signal_weight += min(0.30, overlay_text_score * 0.12)

        if is_screenshot:
            confidence = min(0.95, 0.50 + signal_weight)
        else:
            confidence = min(0.90, 0.45 + signal_weight)

        logger.info(
            "Screenshot analysis: is_screenshot=%s, classic=%d, overlay=%d, "
            "text=%d, confidence=%.3f, signals=%s",
            is_screenshot, classic_score, overlay_score, overlay_text_score, confidence,
            {k: v for k, v in signals.items() if k != "overlay_details"},
        )

        return round(confidence, 3), {
            "is_screenshot": is_screenshot,
            "signals": signals,
        }
