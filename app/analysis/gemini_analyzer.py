"""
Gemini Vision analyzer — OPTIONAL, higher-level reasoning layer.

Deliberately kept pluggable rather than load-bearing:
  - If GEMINI_API_KEY is not configured, this analyzer returns a
    "skipped" result instead of failing the whole pipeline or blocking
    local setup on an external API key.
  - It provides a second opinion on tampering / screenshot signals that
    complements (not replaces) the cheap OpenCV heuristics, which remain
    the primary, always-available signal.

This keeps the core pipeline runnable and demoable with zero external
dependencies, while still showing how an LLM-based check would plug in.
"""
import base64
import json
from typing import Any

import httpx

from app.analysis.base import ImageAnalyzer, AnalyzerResult
from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)

PROMPT = (
    "You are reviewing a vehicle photo submitted by a field agent. "
    "Answer strictly as JSON with keys: "
    "\"tampered\" (bool), \"is_screenshot\" (bool), \"reasoning\" (short string, "
    "under 30 words). Do not include any text outside the JSON object."
)


class GeminiAnalyzer(ImageAnalyzer):
    name = "gemini_analyzer"

    def run(self, image_path: str, context: dict[str, Any]) -> AnalyzerResult:
        settings = get_settings()
        if not settings.gemini_enabled:
            return AnalyzerResult(
                name=self.name,
                status="skipped",
                confidence=0.0,
                findings={"reason": "GEMINI_API_KEY not configured"},
            )
        return super().run(image_path, context)

    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        settings = get_settings()

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        url = GEMINI_ENDPOINT_TEMPLATE.format(model=settings.gemini_model, key=settings.gemini_api_key)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": PROMPT},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    ]
                }
            ]
        }

        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = self._safe_parse_json(text)

        tampered = bool(parsed.get("tampered", False))
        is_screenshot = bool(parsed.get("is_screenshot", False))
        reasoning = parsed.get("reasoning", "")

        # Confidence varies based on what Gemini found and whether it
        # provided reasoning. Flagging issues = higher confidence (definitive),
        # clean report = moderate confidence (LLM can miss things).
        if tampered and is_screenshot:
            confidence = 0.82  # multiple flags → confident in findings
        elif tampered:
            confidence = 0.78
        elif is_screenshot:
            confidence = 0.76
        elif len(reasoning) > 15:
            confidence = 0.72  # clean with good reasoning
        else:
            confidence = 0.65  # clean but sparse reasoning

        return round(confidence, 3), {
            "tampered": tampered,
            "is_screenshot": is_screenshot,
            "reasoning": reasoning,
            "raw_response": text[:1024],
        }

    @staticmethod
    def _safe_parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Gemini response was not valid JSON; returning empty findings")
            return {}
