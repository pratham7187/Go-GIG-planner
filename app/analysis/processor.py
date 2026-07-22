"""
Processor: runs every registered analyzer against one image and combines
their outputs into a single overall verdict.

Design notes:
  - Analyzers are independent; the Processor is the only place that knows
    about ordering/dependencies between them (e.g. ScreenshotAnalyzer wants
    MetadataAnalyzer's findings, DuplicateAnalyzer wants existing hashes).
  - One analyzer erroring never stops the others — each already isolates
    its own exceptions (see ImageAnalyzer.run), so the Processor just
    collects whatever came back.
  - Overall confidence is a weighted average of successful analyzers'
    confidences with penalty/bonus adjustments based on detected issues.
"""
from typing import Any

from app.analysis.base import AnalyzerResult
from app.analysis.blur_analyzer import BlurAnalyzer
from app.analysis.brightness_analyzer import BrightnessAnalyzer
from app.analysis.duplicate_analyzer import DuplicateAnalyzer
from app.analysis.ocr_analyzer import OCRAnalyzer
from app.analysis.metadata_analyzer import MetadataAnalyzer
from app.analysis.screenshot_analyzer import ScreenshotAnalyzer
from app.analysis.gemini_analyzer import GeminiAnalyzer
from app.models.enums import OverallStatus

# Weight reflects how strongly each check should influence the final verdict.
ANALYZER_WEIGHTS = {
    "blur_analyzer": 1.0,
    "brightness_analyzer": 1.0,
    "duplicate_analyzer": 1.5,
    "ocr_analyzer": 1.2,
    "metadata_analyzer": 0.5,
    "screenshot_analyzer": 1.0,
    "gemini_analyzer": 1.2,
}


class Processor:
    def __init__(self) -> None:
        # Order matters: metadata before screenshot (dependency), everything
        # else is independent.
        self.analyzers = [
            MetadataAnalyzer(),
            BlurAnalyzer(),
            BrightnessAnalyzer(),
            OCRAnalyzer(),
            ScreenshotAnalyzer(),
            DuplicateAnalyzer(),
            GeminiAnalyzer(),
        ]

    def process(self, image_path: str, existing_hashes: dict[str, str]) -> dict[str, Any]:
        context: dict[str, Any] = {"existing_hashes": existing_hashes}
        results: dict[str, AnalyzerResult] = {}

        for analyzer in self.analyzers:
            result = analyzer.run(image_path, context)
            results[analyzer.name] = result

            # Feed metadata findings forward to screenshot analyzer.
            if analyzer.name == "metadata_analyzer" and result.status == "success":
                context["metadata_findings"] = result.findings
            if analyzer.name == "ocr_analyzer" and result.status == "success":
                context["ocr_findings"] = result.findings

        return self._combine(results)

    def _combine(self, results: dict[str, AnalyzerResult]) -> dict[str, Any]:
        # Fallback for any analyzer that didn't run or errored out entirely.
        _empty = AnalyzerResult(name="<missing>", status="skipped", confidence=0.0, findings={})

        blur = results.get("blur_analyzer", _empty)
        brightness = results.get("brightness_analyzer", _empty)
        ocr = results.get("ocr_analyzer", _empty)
        metadata = results.get("metadata_analyzer", _empty)
        screenshot = results.get("screenshot_analyzer", _empty)
        duplicate = results.get("duplicate_analyzer", _empty)
        gemini = results.get("gemini_analyzer", _empty)

        is_blurry = blur.findings.get("is_blurry", False) if blur.status == "success" else False
        is_low_light = brightness.findings.get("is_low_light", False) if brightness.status == "success" else False
        is_overexposed = brightness.findings.get("is_overexposed", False) if brightness.status == "success" else False

        # Duplicate: only exact and near duplicates count.
        is_duplicate = duplicate.findings.get("is_duplicate", False) if duplicate.status == "success" else False
        duplicate_type = duplicate.findings.get("duplicate_type", "different") if duplicate.status == "success" else "different"

        is_screenshot = screenshot.findings.get("is_screenshot", False) if screenshot.status == "success" else False
        if gemini.status == "success":
            is_screenshot = is_screenshot or gemini.findings.get("is_screenshot", False)
        tampered = gemini.findings.get("tampered", False) if gemini.status == "success" else False

        plate_valid = ocr.findings.get("plate_valid", False) if ocr.status == "success" else False
        ocr_text_len = ocr.findings.get("text_length", 0) if ocr.status == "success" else 0

        # --- Weighted overall confidence across successful analyzers ---
        total_weight = 0.0
        weighted_sum = 0.0
        for name, result in results.items():
            if result.status != "success":
                continue
            weight = ANALYZER_WEIGHTS.get(name, 1.0)
            weighted_sum += result.confidence * weight
            total_weight += weight
        base_confidence = weighted_sum / total_weight if total_weight else 0.0

        # --- Penalty / bonus adjustments ---
        adjustment = 0.0

        # Penalties for issues.
        if is_duplicate:
            dup_conf = duplicate.confidence if duplicate.status == "success" else 0.5
            if duplicate_type == "exact_duplicate":
                adjustment -= 0.18 * dup_conf
            else:  # near_duplicate
                adjustment -= 0.12 * dup_conf
        if tampered:
            gem_conf = gemini.confidence if gemini.status == "success" else 0.5
            adjustment -= 0.15 * gem_conf
        if is_screenshot:
            scr_conf = screenshot.confidence if screenshot.status == "success" else 0.5
            adjustment -= 0.06 * scr_conf
        if is_blurry:
            blur_conf = blur.confidence if blur.status == "success" else 0.5
            adjustment -= 0.06 * blur_conf
        if is_low_light or is_overexposed:
            bright_conf = brightness.confidence if brightness.status == "success" else 0.5
            adjustment -= 0.03 * bright_conf

        # Bonuses for positive signals.
        if plate_valid and ocr.status == "success":
            adjustment += 0.05  # Successfully read a plate = strong signal
        if ocr_text_len > 50 and ocr.status == "success":
            adjustment += 0.02  # Good OCR yield
        if not plate_valid and ocr.status == "success":
            adjustment -= 0.02  # Mild penalty for no plate

        confidence_score = round(max(0.0, min(1.0, base_confidence + adjustment)), 3)

        # --- Overall verdict ---
        rejection_reasons = []
        if is_duplicate:
            rejection_reasons.append(f"duplicate image ({duplicate_type.replace('_', ' ')})")
        if tampered:
            rejection_reasons.append("possible tampering detected")

        review_reasons = []
        if is_blurry:
            review_reasons.append("image is blurry")
        if is_low_light:
            review_reasons.append("low light conditions")
        if is_overexposed:
            review_reasons.append("image is overexposed")
        if is_screenshot:
            review_reasons.append("screenshot or overlay detected")
        if not plate_valid and ocr.status == "success":
            review_reasons.append("vehicle number plate not confidently detected")

        if rejection_reasons:
            overall_status = OverallStatus.REJECTED
            summary = "Rejected: " + "; ".join(rejection_reasons)
        elif review_reasons:
            overall_status = OverallStatus.NEEDS_REVIEW
            summary = "Needs review: " + "; ".join(review_reasons)
        else:
            overall_status = OverallStatus.OK
            summary = "Image passed all checks"

        raw_findings = {name: self._result_to_dict(result) for name, result in results.items()}

        return {
            "blur_score": blur.findings.get("laplacian_variance"),
            "is_blurry": is_blurry,
            "brightness": brightness.findings.get("mean_intensity"),
            "is_low_light": is_low_light,
            "duplicate": is_duplicate,
            "duplicate_of": duplicate.findings.get("duplicate_of"),
            "duplicate_type": duplicate_type,
            "ocr_text": ocr.findings.get("raw_text"),
            "vehicle_number": ocr.findings.get("vehicle_number"),
            "plate_valid": plate_valid,
            "image_metadata": {
                "width": metadata.findings.get("width"),
                "height": metadata.findings.get("height"),
                "format": metadata.findings.get("format"),
                "has_exif": metadata.findings.get("has_exif"),
                "file_size_bytes": metadata.findings.get("file_size_bytes"),
            },
            "screenshot": is_screenshot,
            "tampered": tampered,
            "confidence_score": confidence_score,
            "overall_status": overall_status,
            "summary": summary,
            "raw_findings": raw_findings,
            "computed_hash": duplicate.findings.get("hash"),
        }

    @staticmethod
    def _result_to_dict(result: AnalyzerResult) -> dict[str, Any]:
        return {
            "name": result.name,
            "status": result.status,
            "confidence": result.confidence,
            "findings": result.findings,
            "execution_time": result.execution_time,
            "error": result.error,
        }
