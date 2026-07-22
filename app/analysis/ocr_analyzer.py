"""
OCR extraction + Indian vehicle number plate format validation.

Pipeline:
  1. Auto-detect and configure Tesseract executable path.
  2. Use plate_detector to localize license plate regions.
  3. Run OCR on plate regions specifically for plate number extraction.
  4. Run OCR on full image + text-rich regions for advertisement/general text.
  5. Aggregate ALL extracted text and return it.
  6. Regex-match against Indian plate format for vehicle number.
"""
import re
from typing import Any

import cv2
import numpy as np
import pytesseract

from app.analysis.base import ImageAnalyzer
from app.analysis.plate_detector import detect_plates
from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# --- Tesseract path configuration ---
# Done once at module load so every OCR call uses the resolved path.
_settings = get_settings()
_tess_path = _settings.resolved_tesseract_path
if _tess_path:
    pytesseract.pytesseract.tesseract_cmd = _tess_path
    logger.info("Tesseract configured at: %s", _tess_path)
else:
    logger.warning("Tesseract not found — OCR will fail. Install Tesseract and "
                   "set TESSERACT_PATH in .env or add it to system PATH.")


# State code (2 letters) + district code (1-2 digits) + series (1-3 letters)
# + number (4 digits). Spaces/hyphens/dots optional, case-insensitive.
PLATE_PATTERN = re.compile(
    r"([A-Z]{2})\s?[.\-]?\s?(\d{1,2})\s?[.\-]?\s?([A-Z]{1,3})\s?[.\-]?\s?(\d{4})",
    re.IGNORECASE,
)

# Also match plates with the format like "TN.05 BT5754" or "MH 12N W8556"
PLATE_PATTERN_ALT = re.compile(
    r"([A-Z]{2})\s?[.\-]?\s?(\d{1,2})\s?([A-Z])\s?[.\-]?\s?([A-Z]?\d{4})",
    re.IGNORECASE,
)

# Tesseract configs for different text layouts.
TESS_PLATE = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TESS_PLATE_BLOCK = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
TESS_BLOCK = r"--oem 3 --psm 6"
TESS_SPARSE = r"--oem 3 --psm 11"
TESS_AUTO = r"--oem 3 --psm 3"


# --- Preprocessing functions ---

def _preprocess_adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def _preprocess_clahe_otsu(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _preprocess_clahe_adaptive(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )


def _preprocess_otsu(gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _preprocess_inverted(gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def _preprocess_sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _preprocess_bilateral(gray: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _resize_if_small(gray: np.ndarray, min_height: int = 600) -> np.ndarray:
    h, w = gray.shape[:2]
    if h < min_height:
        scale = min_height / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _clean_text(raw: str) -> str:
    """Collapse whitespace and strip non-printable characters."""
    cleaned = re.sub(r"[^\x20-\x7E\n\t]", " ", raw)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def _extract_plate_number(text: str) -> str | None:
    """Try to extract an Indian vehicle registration number from text."""
    upper = text.upper()

    # Try primary pattern first.
    match = PLATE_PATTERN.search(upper)
    if match:
        state, district, series, number = match.groups()
        return f"{state}{district}{series}{number}"

    # Try alternative pattern.
    match = PLATE_PATTERN_ALT.search(upper)
    if match:
        groups = match.groups()
        return "".join(g for g in groups if g)

    # Two-line plates are common on auto-rickshaws.  Treat their two OCR
    # lines as one registration, rather than requiring all characters to be
    # on a single line.  A leading artefact before the second-line letter is
    # ignored when the first line already has a series letter.
    lines = [re.sub(r"[^A-Z0-9]", "", line) for line in upper.splitlines()]
    for first, second in zip(lines, lines[1:]):
        top = re.search(r"([A-Z]{2})([0-9IOZ]{1,2})([A-Z]{0,3})", first)
        bottom = re.search(r"([A-Z]{1,3})([0-9OILS]{4})", second)
        if not top or not bottom:
            continue
        state, district_raw, top_series = top.groups()
        bottom_series, number_raw = bottom.groups()
        district = district_raw.translate(str.maketrans({"I": "1", "O": "0", "Z": "2"}))
        number = number_raw.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5"}))
        # E.g. OCR "MH12N" + "LW8556" should retain N and W, not the
        # spurious leading L introduced by the plate border.
        if top_series and len(bottom_series) > 1:
            bottom_series = bottom_series[-1]
        series = top_series + bottom_series
        if 1 <= len(series) <= 3:
            return f"{state}{district}{series}{number}"

    # Plate OCR commonly confuses 1/I and 0/O, and occasionally inserts a
    # digit into the letter series.  Recover only from compact, plate-shaped
    # fragments; this deliberately does not loosen full-image OCR matching.
    for fragment in re.findall(r"[A-Z0-9]{8,14}", upper.replace(" ", "")):
        if len(fragment) < 8 or not fragment[:2].isalpha():
            continue
        rest = fragment[2:]
        district_match = re.match(r"([0-9IOZ]{1,2})(.*)", rest)
        if not district_match:
            continue
        district_raw, tail = district_match.groups()
        district = district_raw.translate(str.maketrans({"I": "1", "O": "0", "Z": "2"}))
        number_match = re.search(r"([0-9OILS]{4})(?:[OILS])?$", tail)
        if not number_match:
            continue
        number = number_match.group(1).translate(str.maketrans({
            "O": "0", "I": "1", "L": "1", "S": "5",
        }))
        series_raw = tail[:number_match.start()]
        # Series is 1-3 letters.  Discard a spurious non-letter rather than
        # converting it, which avoids manufacturing a plate from prose.
        series = "".join(ch for ch in series_raw if ch.isalpha())
        if 1 <= len(series) <= 3:
            return f"{fragment[:2]}{district}{series}{number}"

    return None


def _ocr_single_image(gray: np.ndarray, configs: list[str],
                       preprocess_fns: list) -> str:
    """Run OCR with multiple strategies on a single grayscale image.
    Returns the best (longest) cleaned text found."""
    best = ""
    for fn in preprocess_fns:
        for cfg in configs:
            try:
                src = fn(gray) if fn else gray
                raw = pytesseract.image_to_string(src, config=cfg)
                cleaned = _clean_text(raw)
                if len(cleaned) > len(best):
                    best = cleaned
            except Exception:
                continue
    return best


def _ocr_plate_region(plate_img: np.ndarray) -> str:
    """Specialized OCR for a cropped plate region.
    Uses character-whitelisted config and more aggressive preprocessing."""
    if plate_img is None or plate_img.size == 0:
        return ""

    # Upscale plate crops — they're typically small.
    plate_img = _resize_if_small(plate_img, min_height=150)

    # A small, complementary set is more reliable than repeatedly OCRing a
    # noisy crop with every threshold variant.  It also keeps an uploaded
    # image from spawning dozens of Tesseract processes.
    plate_preprocess = [None, _preprocess_otsu, _preprocess_clahe_otsu]
    plate_configs = [TESS_PLATE, TESS_PLATE_BLOCK]

    best = ""
    for fn in plate_preprocess:
        for cfg in plate_configs:
            try:
                src = fn(plate_img) if fn else plate_img
                raw = pytesseract.image_to_string(src, config=cfg)
                cleaned = _clean_text(raw)
                if len(cleaned) > len(best):
                    best = cleaned
                # Early exit if we found a plate number.
                if _extract_plate_number(best):
                    return best
            except Exception:
                continue
    return best


class OCRAnalyzer(ImageAnalyzer):
    name = "ocr_analyzer"

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        # Verify Tesseract is available.
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                f"Tesseract OCR not found. Searched PATH and common locations. "
                f"Install from https://github.com/UB-Mannheim/tesseract/wiki "
                f"or set TESSERACT_PATH in .env"
            )

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image for OCR analysis")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Light denoise — preserve text detail.
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        full_gray = _resize_if_small(denoised)

        # ---- Phase 1: Full-image OCR for general text extraction ----
        general_preprocess = [None, _preprocess_clahe_otsu]
        general_configs = [TESS_SPARSE, TESS_BLOCK]

        full_text = _ocr_single_image(full_gray, general_configs, general_preprocess)

        # ---- Phase 2: Region-based OCR (bottom half, center, top) ----
        all_texts = [full_text]
        regions = [
            ("bottom_half", denoised[h // 2:, :]),
            ("bottom_third", denoised[2 * h // 3:, :]),
            ("center_strip", denoised[h // 4: 3 * h // 4, w // 6: 5 * w // 6]),
            ("top_half", denoised[: h // 2, :]),
        ]

        for region_name, region_img in regions:
            if region_img.shape[0] < 50 or region_img.shape[1] < 50:
                continue
            region_img = _resize_if_small(region_img, min_height=400)
            region_text = _ocr_single_image(region_img, [TESS_SPARSE], [None])
            if region_text and region_text not in full_text:
                all_texts.append(region_text)

        # ---- Phase 3: License plate detection + plate-specific OCR ----
        plate_candidates = detect_plates(image_path)
        plate_texts = []
        vehicle_number = None

        # Candidates are confidence-ranked.  OCRing the top three preserves
        # recall while bounding work for images with many advertisement edges.
        for candidate in plate_candidates[:3]:
            if candidate.image is not None:
                plate_text = _ocr_plate_region(candidate.image)
                if plate_text:
                    plate_texts.append(plate_text)
                    # Try to extract plate number.
                    if not vehicle_number:
                        vehicle_number = _extract_plate_number(plate_text)

        # ---- Phase 4: Try to find plate in all collected text ----
        combined_text = "\n".join(all_texts)
        if not vehicle_number:
            vehicle_number = _extract_plate_number(combined_text)

        # Also check plate texts.
        if not vehicle_number:
            for pt in plate_texts:
                vehicle_number = _extract_plate_number(pt)
                if vehicle_number:
                    break

        plate_valid = vehicle_number is not None

        # ---- Aggregate all unique text ----
        # Combine full-image text with any plate-specific text that adds new info.
        all_unique_parts = list(dict.fromkeys(all_texts + plate_texts))
        aggregated_text = "\n".join(t for t in all_unique_parts if t)

        text_len = len(aggregated_text)

        # ---- Confidence ----
        if plate_valid and text_len > 50:
            confidence = min(0.95, 0.85 + (text_len / 3000))
        elif plate_valid:
            confidence = 0.82
        elif text_len > 200:
            confidence = min(0.82, 0.65 + (text_len / 5000))
        elif text_len > 80:
            confidence = 0.58 + (text_len / 3000)
        elif text_len > 20:
            confidence = 0.42 + (text_len / 2000)
        elif text_len > 5:
            confidence = 0.30 + (text_len / 1000)
        else:
            confidence = 0.55  # thorough scan found nothing — valid result

        logger.info(
            "OCR result: text_len=%d, plate_valid=%s, vehicle=%s, "
            "plates_detected=%d, confidence=%.3f",
            text_len, plate_valid, vehicle_number,
            len(plate_candidates), confidence,
        )

        return round(confidence, 3), {
            "raw_text": aggregated_text[:512],
            "vehicle_number": vehicle_number,
            "plate_valid": plate_valid,
            "text_length": text_len,
            "plates_detected": len(plate_candidates),
            "plate_texts": [t[:100] for t in plate_texts[:3]],
        }
