"""
Brightness / low-light detection via mean grayscale intensity.

Pixel intensities range 0-255. Very low averages indicate underexposed /
low-light shots; very high averages can indicate blown-out / overexposed
shots. Both are flagged since either can make a vehicle number unreadable.
"""
from typing import Any

import cv2

from app.analysis.base import ImageAnalyzer

LOW_LIGHT_THRESHOLD = 60.0
OVEREXPOSED_THRESHOLD = 220.0


class BrightnessAnalyzer(ImageAnalyzer):
    name = "brightness_analyzer"

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read image for brightness analysis")

        mean_intensity = float(image.mean())
        is_low_light = mean_intensity < LOW_LIGHT_THRESHOLD
        is_overexposed = mean_intensity > OVEREXPOSED_THRESHOLD

        if is_low_light:
            # The darker it is, the more confident we are about the problem.
            # Intensity 0 → confidence 0.95; at threshold → confidence 0.55.
            confidence = 0.55 + 0.40 * max(0.0, 1.0 - mean_intensity / LOW_LIGHT_THRESHOLD)
        elif is_overexposed:
            # The brighter it is, the more confident we are about overexposure.
            excess = (mean_intensity - OVEREXPOSED_THRESHOLD) / (255 - OVEREXPOSED_THRESHOLD)
            confidence = min(0.95, 0.55 + 0.40 * excess)
        else:
            # Normal range: confidence reflects how "ideal" the brightness is.
            # Sweet spot around 120-140 gets highest confidence (~0.92).
            # Near thresholds (60 or 220) gets lower confidence (~0.65).
            # This creates genuine variation across the normal range.
            ideal = 130.0
            max_dist = max(ideal - LOW_LIGHT_THRESHOLD, OVEREXPOSED_THRESHOLD - ideal)
            distance_from_ideal = abs(mean_intensity - ideal)
            # Quadratic falloff for smoother variation
            normalized = (distance_from_ideal / max_dist) ** 1.5
            confidence = 0.92 - 0.27 * normalized

        return round(confidence, 3), {
            "mean_intensity": round(mean_intensity, 2),
            "is_low_light": bool(is_low_light),
            "is_overexposed": bool(is_overexposed),
        }

