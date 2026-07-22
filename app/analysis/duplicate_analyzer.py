"""
Duplicate detection via perceptual hashing (pHash).

Unlike a cryptographic hash, pHash is designed so visually-similar images
produce hashes with a small Hamming distance — this catches re-saves,
mild recompression, and minor crops, not just byte-identical duplicates.

Tiered classification:
  - exact_duplicate (distance 0-3): Identical or trivially recompressed
  - near_duplicate (distance 4-10): Same image with minor edits
  - similar (distance 11-20): Same vehicle, different angle — NOT duplicate
  - different (distance 21+): Different images entirely

Only exact_duplicate and near_duplicate are treated as rejection-worthy.
'similar' is informational only.
"""
from typing import Any

import imagehash
from PIL import Image as PILImage

from app.analysis.base import ImageAnalyzer

# Thresholds for tiered classification.
EXACT_DUPLICATE_THRESHOLD = 3   # distance 0-3: exact duplicate
NEAR_DUPLICATE_THRESHOLD = 10   # distance 4-10: near duplicate
SIMILAR_THRESHOLD = 20          # distance 11-20: similar (same vehicle, different angle)


class DuplicateAnalyzer(ImageAnalyzer):
    name = "duplicate_analyzer"

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        existing_hashes: dict[str, str] = context.get("existing_hashes", {})

        with PILImage.open(image_path) as img:
            current_hash = imagehash.phash(img)

        best_match_id = None
        best_distance = None

        for image_id, hash_str in existing_hashes.items():
            try:
                other_hash = imagehash.hex_to_hash(hash_str)
            except ValueError:
                continue
            distance = current_hash - other_hash
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match_id = image_id

        # --- Tiered classification ---
        if best_distance is not None and best_distance <= EXACT_DUPLICATE_THRESHOLD:
            duplicate_type = "exact_duplicate"
            is_duplicate = True
        elif best_distance is not None and best_distance <= NEAR_DUPLICATE_THRESHOLD:
            duplicate_type = "near_duplicate"
            is_duplicate = True
        elif best_distance is not None and best_distance <= SIMILAR_THRESHOLD:
            duplicate_type = "similar"
            is_duplicate = False  # Similar but NOT a duplicate
        else:
            duplicate_type = "different"
            is_duplicate = False

        # --- Confidence ---
        if duplicate_type == "exact_duplicate":
            # Very confident — near-zero distance.
            confidence = 0.90 + 0.10 * (1.0 - best_distance / (EXACT_DUPLICATE_THRESHOLD + 1))
        elif duplicate_type == "near_duplicate":
            # Moderately confident — within near-dup range.
            position = (best_distance - EXACT_DUPLICATE_THRESHOLD) / (
                NEAR_DUPLICATE_THRESHOLD - EXACT_DUPLICATE_THRESHOLD
            )
            confidence = 0.80 - 0.15 * position  # 0.80 → 0.65
        elif duplicate_type == "similar":
            # Confident it's similar but not duplicate.
            position = (best_distance - NEAR_DUPLICATE_THRESHOLD) / (
                SIMILAR_THRESHOLD - NEAR_DUPLICATE_THRESHOLD
            )
            confidence = 0.75 - 0.10 * position  # 0.75 → 0.65
        elif best_distance is not None:
            # Confident it's different — large distance.
            margin = best_distance - SIMILAR_THRESHOLD
            confidence = min(0.92, 0.70 + margin * 0.01)
        else:
            # No existing images to compare — first upload.
            hash_str = str(current_hash)
            unique_chars = len(set(hash_str))
            diversity = unique_chars / 16.0
            confidence = 0.75 + 0.15 * diversity

        return round(confidence, 3), {
            "hash": str(current_hash),
            "is_duplicate": bool(is_duplicate),
            "duplicate_type": duplicate_type,
            "duplicate_of": best_match_id if is_duplicate else None,
            "hamming_distance": int(best_distance) if best_distance is not None else None,
        }
