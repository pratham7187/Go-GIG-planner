"""
Blur detection via Laplacian variance.

Heuristic: sharp images have high-frequency edges, which the Laplacian
operator responds to strongly. Blurry images produce a low-variance
Laplacian response. This is a well-known, cheap heuristic — not a learned
model — and the threshold below is a reasonable default, not a guarantee.
See README "Trade-offs" for why we didn't train a dedicated model here.
"""
from typing import Any

import cv2

from app.analysis.base import ImageAnalyzer

BLUR_THRESHOLD = 100.0  # Empirical cutoff; images below this are "blurry"


class BlurAnalyzer(ImageAnalyzer):
    name = "blur_analyzer"

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read image for blur analysis")

        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        is_blurry = variance < BLUR_THRESHOLD

        # Confidence reflects how clearly the image falls into blurry vs clear.
        # Uses a continuous mapping over the full practical range so different
        # images produce genuinely different confidence scores.
        if is_blurry:
            # Very blurry (var→0) = high confidence it's blurry (~0.95)
            # Borderline (var→100) = lower confidence (~0.55)
            confidence = 0.55 + 0.40 * max(0.0, 1.0 - variance / BLUR_THRESHOLD)
        else:
            # Barely clear (var≈100) = moderate confidence (~0.55)
            # Very sharp (var≈1000+) = high confidence (~0.95)
            # Use a log-scale mapping so the range 100-2000 spreads evenly.
            import math
            ratio = variance / BLUR_THRESHOLD  # 1.0 at threshold, 10+ for sharp
            confidence = min(0.97, 0.50 + 0.15 * math.log2(max(1.0, ratio)))

        return round(confidence, 3), {
            "laplacian_variance": round(float(variance), 2),
            "is_blurry": bool(is_blurry),
            "threshold": BLUR_THRESHOLD,
        }
