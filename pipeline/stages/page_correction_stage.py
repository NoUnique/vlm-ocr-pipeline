"""Page Correction Stage: Page-level text correction using VLM."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PageCorrectionStage:
    """Stage 7: PageCorrection - Page-level text correction."""

    def __init__(self, recognizer: Any, backend: str, enable: bool = True):
        """Initialize PageCorrectionStage.

        Args:
            recognizer: Text recognizer instance (TextRecognizer)
            backend: Backend name (e.g., "openai", "gemini", "paddleocr-vl")
            enable: Whether to enable page-level correction
        """
        self.recognizer = recognizer
        self.backend = backend
        self.enable = enable

    def correct_page(self, raw_text: str, page_num: int) -> tuple[str, float, bool]:
        """Perform page-level text correction.

        Args:
            raw_text: Raw rendered text (Markdown)
            page_num: Page number (for logging)

        Returns:
            tuple: (corrected_text, correction_ratio, should_stop)
                correction_ratio: 0.0 = no change, 1.0 = completely different
                should_stop: True if rate limit detected
        """
        if not self.enable:
            return raw_text, 0.0, False

        # Skip page correction for PaddleOCR-VL (it already extracts text directly)
        if self.backend == "paddleocr-vl":
            return raw_text, 0.0, False

        correction_result = self.recognizer.correct_text(raw_text)

        if isinstance(correction_result, dict):
            corrected_text = correction_result.get("corrected_text", raw_text)
            correction_ratio = float(correction_result.get("correction_ratio", 0.0))
            return corrected_text, correction_ratio, False

        corrected_text = str(correction_result)
        rate_limit_indicators = ["RATE_LIMIT_EXCEEDED", "DAILY_LIMIT_EXCEEDED"]
        if any(indicator in corrected_text for indicator in rate_limit_indicators):
            logger.warning(
                "Rate limit detected during page text correction on page %d. Stopping processing.",
                page_num,
            )
            return corrected_text, 0.0, True

        return corrected_text, 0.0, False
