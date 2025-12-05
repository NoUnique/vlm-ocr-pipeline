"""Local model-based recognizers.

This module provides recognizers that run on local hardware (GPU/CPU),
as opposed to API-based recognizers that call external services.

Available recognizers:
- PaddleOCRVLRecognizer: PaddleOCR Vision-Language model
- DeepSeekOCRRecognizer: DeepSeek OCR model

These recognizers require local GPU resources and model weights.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "PaddleOCRVLRecognizer",
    "DeepSeekOCRRecognizer",
]

# Lazy imports to avoid loading heavy dependencies unless needed


def __getattr__(name: str):
    """Lazy import for local recognizers."""
    if name == "PaddleOCRVLRecognizer":
        try:
            from pipeline.recognition.paddleocr import PaddleOCRVLRecognizer

            return PaddleOCRVLRecognizer
        except ImportError as e:
            raise ImportError(
                f"PaddleOCR-VL not available. Install with: pip install paddlepaddle-gpu paddleocr\n"
                f"Error: {e}"
            ) from e

    elif name == "DeepSeekOCRRecognizer":
        try:
            from pipeline.recognition.deepseek import DeepSeekOCRRecognizer

            return DeepSeekOCRRecognizer
        except ImportError as e:
            raise ImportError(
                f"DeepSeek-OCR not available. Install required dependencies.\n"
                f"Error: {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

