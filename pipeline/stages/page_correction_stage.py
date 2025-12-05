"""Page Correction Stage: Page-level text correction using VLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base import BaseStage

if TYPE_CHECKING:
    from pipeline.recognition import TextRecognizer

logger = logging.getLogger(__name__)


@dataclass
class PageCorrectionResult:
    """Result from page correction."""

    corrected_text: str
    correction_ratio: float
    should_stop: bool


class PageCorrectionStage(BaseStage[str, PageCorrectionResult]):
    """Stage 7: PageCorrection - Page-level text correction.

    This stage performs optional page-level text correction using VLM
    to fix OCR errors in the rendered text.
    """

    name = "page_correction"

    def __init__(self, recognizer: TextRecognizer, backend: str, enable: bool = True):
        """Initialize PageCorrectionStage.

        Args:
            recognizer: Text recognizer instance
            backend: Backend name (e.g., "openai", "gemini", "paddleocr-vl")
            enable: Whether to enable page-level correction
        """
        self.recognizer = recognizer
        self.backend = backend
        self.enable = enable

    def _process_impl(self, input_data: str, **context: Any) -> PageCorrectionResult:
        """Perform page-level text correction.

        Args:
            input_data: Raw rendered text (Markdown)
            **context: May include 'page_num' for logging

        Returns:
            PageCorrectionResult with corrected text and metadata
        """
        page_num = context.get("page_num", 0)

        if not self.enable:
            return PageCorrectionResult(input_data, 0.0, False)

        # Skip page correction for PaddleOCR-VL (it already extracts text directly)
        if self.backend == "paddleocr-vl":
            return PageCorrectionResult(input_data, 0.0, False)

        correction_result = self.recognizer.correct_text(input_data)

        if isinstance(correction_result, dict):
            corrected_text = correction_result.get("corrected_text", input_data)
            correction_ratio = float(correction_result.get("correction_ratio", 0.0))
            return PageCorrectionResult(corrected_text, correction_ratio, False)

        corrected_text = str(correction_result)
        rate_limit_indicators = ["RATE_LIMIT_EXCEEDED", "DAILY_LIMIT_EXCEEDED"]
        if any(indicator in corrected_text for indicator in rate_limit_indicators):
            logger.warning(
                "Rate limit detected during page text correction on page %d. Stopping processing.",
                page_num,
            )
            return PageCorrectionResult(corrected_text, 0.0, True)

        return PageCorrectionResult(corrected_text, 0.0, False)

    def correct_page(self, raw_text: str, page_num: int) -> tuple[str, float, bool]:
        """Perform page-level text correction.

        Legacy method for backward compatibility.

        Args:
            raw_text: Raw rendered text (Markdown)
            page_num: Page number (for logging)

        Returns:
            tuple: (corrected_text, correction_ratio, should_stop)
        """
        result = self.process(raw_text, page_num=page_num)
        return result.corrected_text, result.correction_ratio, result.should_stop
