"""
Common interface every analyzer implements.

Design intent: each analyzer is a small, independent, side-effect-free unit
that takes an image path (+ shared context) and returns a normalized
AnalyzerResult. The Processor is responsible for orchestration, not the
analyzers themselves — this keeps each check independently testable and
means adding a new check never touches existing ones.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalyzerResult:
    name: str
    status: str  # "success" | "skipped" | "error"
    confidence: float  # 0.0 - 1.0, analyzer's confidence in its own finding
    findings: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error: str | None = None


class ImageAnalyzer(ABC):
    """Base interface for all image analyzers."""

    name: str = "base_analyzer"

    @abstractmethod
    def _analyze(self, image_path: str, context: dict[str, Any]) -> tuple[float, dict]:
        """
        Subclasses implement the actual check.
        Returns (confidence, findings). Raise on unrecoverable errors —
        the wrapping `run()` converts exceptions into an "error" result so
        one failing analyzer never crashes the whole pipeline.
        """
        raise NotImplementedError

    def run(self, image_path: str, context: dict[str, Any]) -> AnalyzerResult:
        start = time.perf_counter()
        try:
            confidence, findings = self._analyze(image_path, context)
            return AnalyzerResult(
                name=self.name,
                status="success",
                confidence=confidence,
                findings=findings,
                execution_time=round(time.perf_counter() - start, 4),
            )
        except Exception as exc:  # noqa: BLE001 — intentional: isolate analyzer failures
            return AnalyzerResult(
                name=self.name,
                status="error",
                confidence=0.0,
                findings={},
                execution_time=round(time.perf_counter() - start, 4),
                error=str(exc),
            )
