"""
License plate localization using OpenCV morphological operations + Haar cascade.

Strategy:
  1. Morphological approach: grayscale → bilateral filter → Sobel vertical
     edges → morphological closing → contour extraction → aspect ratio filter.
  2. Haar cascade (haarcascade_russian_plate_number.xml) as a secondary signal.
  3. Merge and deduplicate candidate regions from both methods.
  4. Return cropped plate regions sorted by confidence (area, aspect ratio fit).

Indian plates come in two layouts:
  - Single-line: aspect ratio ~2.5–5.5 (cars, buses, trucks)
  - Double-line: aspect ratio ~1.2–2.5 (auto-rickshaws, two-wheelers)
"""
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlateCandidate:
    """A detected license plate region."""
    x: int
    y: int
    w: int
    h: int
    confidence: float  # 0.0 - 1.0 heuristic confidence
    source: str  # "morphological" | "haar" | "contour"
    image: np.ndarray | None = None  # cropped plate image


# Aspect ratio ranges for Indian plates.
SINGLE_LINE_AR = (2.2, 6.0)
DOUBLE_LINE_AR = (1.0, 2.5)

# Minimum plate area relative to full image area.
MIN_PLATE_AREA_RATIO = 0.001
MAX_PLATE_AREA_RATIO = 0.15


def _load_haar_cascade() -> cv2.CascadeClassifier | None:
    """Load the Russian plate Haar cascade (closest available for plates)."""
    cascade_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
    for name in ["haarcascade_russian_plate_number.xml",
                 "haarcascade_license_plate_rus_16stages.xml"]:
        path = os.path.join(cascade_dir, name)
        if os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    return None


def _is_valid_plate_aspect_ratio(w: int, h: int) -> bool:
    """Check if dimensions match Indian plate aspect ratios."""
    if h == 0:
        return False
    ar = w / h
    return (SINGLE_LINE_AR[0] <= ar <= SINGLE_LINE_AR[1] or
            DOUBLE_LINE_AR[0] <= ar <= DOUBLE_LINE_AR[1])


def _morphological_detect(gray: np.ndarray, image_area: int) -> list[PlateCandidate]:
    """Detect plate candidates using morphological operations + Sobel edges."""
    candidates = []

    # Bilateral filter preserves edges while reducing noise.
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Sobel vertical edges — plates have dense vertical character transitions.
    sobel = cv2.Sobel(filtered, cv2.CV_8U, dx=1, dy=0, ksize=3)

    # Otsu threshold on Sobel output.
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to connect character edges into a plate-shaped blob.
    # Try multiple kernel sizes to catch both single-line and double-line plates.
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),   # wide, short (single-line)
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)),   # medium
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 11)),  # squarish (double-line)
    ]

    for kernel in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Dilate to fill gaps.
        closed = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area.
            if area < image_area * MIN_PLATE_AREA_RATIO:
                continue
            if area > image_area * MAX_PLATE_AREA_RATIO:
                continue

            # Filter by aspect ratio.
            if not _is_valid_plate_aspect_ratio(w, h):
                continue

            # Solidity check — plate-like contours should be fairly solid.
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area > 0:
                solidity = cv2.contourArea(contour) / hull_area
                if solidity < 0.3:
                    continue

            # Confidence based on how well the aspect ratio fits.
            ar = w / h
            if SINGLE_LINE_AR[0] <= ar <= SINGLE_LINE_AR[1]:
                ideal = (SINGLE_LINE_AR[0] + SINGLE_LINE_AR[1]) / 2
            else:
                ideal = (DOUBLE_LINE_AR[0] + DOUBLE_LINE_AR[1]) / 2
            ar_distance = abs(ar - ideal) / ideal
            conf = max(0.3, 0.85 - ar_distance)

            candidates.append(PlateCandidate(
                x=x, y=y, w=w, h=h,
                confidence=round(conf, 3),
                source="morphological",
            ))

    return candidates


def _haar_detect(gray: np.ndarray, image_area: int) -> list[PlateCandidate]:
    """Detect plate candidates using Haar cascade."""
    cascade = _load_haar_cascade()
    if cascade is None:
        return []

    candidates = []
    # Try multiple scale factors for different plate sizes.
    for scale_factor in [1.05, 1.1, 1.2]:
        plates = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=3,
            minSize=(60, 20),
            maxSize=(gray.shape[1] // 2, gray.shape[0] // 3),
        )
        for (x, y, w, h) in plates:
            area = w * h
            if area < image_area * MIN_PLATE_AREA_RATIO:
                continue
            if area > image_area * MAX_PLATE_AREA_RATIO:
                continue

            candidates.append(PlateCandidate(
                x=x, y=y, w=w, h=h,
                confidence=0.65,  # Haar is less reliable for Indian plates
                source="haar",
            ))

    return candidates


def _contour_detect(gray: np.ndarray, image_area: int) -> list[PlateCandidate]:
    """Simple contour-based detection: find rectangular contours with
    plate-like proportions."""
    candidates = []

    # Edge detection.
    edges = cv2.Canny(gray, 50, 200)
    # Dilate to close gaps in edges.
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Plates are roughly rectangular (4 corners).
        if len(approx) < 4 or len(approx) > 6:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h

        if area < image_area * MIN_PLATE_AREA_RATIO:
            continue
        if area > image_area * MAX_PLATE_AREA_RATIO:
            continue
        if not _is_valid_plate_aspect_ratio(w, h):
            continue

        candidates.append(PlateCandidate(
            x=x, y=y, w=w, h=h,
            confidence=0.50,  # contour detection is less reliable
            source="contour",
        ))

    return candidates


def _yellow_plate_detect(image: np.ndarray, image_area: int) -> list[PlateCandidate]:
    """Find the yellow reflective plates commonly used on commercial vehicles.

    Edge-only detection often promotes large advertisements and grille details
    above a clean yellow rear plate.  Colour is therefore used as an additional
    *localisation* signal, never as a replacement for the other detectors.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Broad enough for sun/shade, but excludes most white highlights.
    mask = cv2.inRange(hsv, (15, 80, 100), (45, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[PlateCandidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not (image_area * MIN_PLATE_AREA_RATIO <= area <= image_area * 0.04):
            continue
        if not _is_valid_plate_aspect_ratio(w, h):
            continue

        fill_ratio = cv2.contourArea(contour) / area if area else 0.0
        if fill_ratio < 0.45:
            continue
        candidates.append(PlateCandidate(
            x=x, y=y, w=w, h=h, confidence=0.90,
            source="yellow_colour",
        ))
    return candidates


def _deduplicate_candidates(candidates: list[PlateCandidate]) -> list[PlateCandidate]:
    """Remove overlapping candidates, keeping the highest-confidence one."""
    if not candidates:
        return []

    # Sort by confidence descending.
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    kept = []
    for candidate in candidates:
        overlap = False
        for existing in kept:
            # Check IoU (Intersection over Union).
            x1 = max(candidate.x, existing.x)
            y1 = max(candidate.y, existing.y)
            x2 = min(candidate.x + candidate.w, existing.x + existing.w)
            y2 = min(candidate.y + candidate.h, existing.y + existing.h)

            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                union = (candidate.w * candidate.h + existing.w * existing.h
                         - intersection)
                iou = intersection / union if union > 0 else 0
                if iou > 0.3:
                    overlap = True
                    break

        if not overlap:
            kept.append(candidate)

    return kept


def detect_plates(image_path: str) -> list[PlateCandidate]:
    """
    Detect license plate regions in an image.

    Returns a list of PlateCandidate objects sorted by confidence (highest first),
    with cropped plate images attached.
    """
    image = cv2.imread(image_path)
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = h * w

    # Run all detection methods.
    candidates = []
    candidates.extend(_morphological_detect(gray, image_area))
    candidates.extend(_haar_detect(gray, image_area))
    candidates.extend(_contour_detect(gray, image_area))
    candidates.extend(_yellow_plate_detect(image, image_area))

    # Deduplicate overlapping detections.
    candidates = _deduplicate_candidates(candidates)

    # Attach cropped plate images with padding.
    for candidate in candidates:
        pad_x = int(candidate.w * 0.1)
        pad_y = int(candidate.h * 0.2)
        x1 = max(0, candidate.x - pad_x)
        y1 = max(0, candidate.y - pad_y)
        x2 = min(w, candidate.x + candidate.w + pad_x)
        y2 = min(h, candidate.y + candidate.h + pad_y)
        candidate.image = gray[y1:y2, x1:x2]

    # Sort by confidence descending, limit to top 5.
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:5]
