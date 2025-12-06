"""Text recognition module for OCR and content extraction.

Recognizers organized by backend:
- text_recognizer.py: VLM-based recognizer (OpenAI/Gemini)
- paddleocr/: PaddleOCR-VL recognizer
- deepseek/: DeepSeek-OCR recognizer

Components:
- BaseRecognizer: Abstract base class for all recognizers
- RecognizerRegistry: Registry for recognizer management
- create_recognizer: Factory function for recognizer creation
"""

from __future__ import annotations

import logging
from typing import Any

from pipeline.types import Recognizer

from .base import BaseRecognizer
from .registry import RecognizerRegistry, recognizer_registry

logger = logging.getLogger(__name__)

__all__ = [
    # Base class
    "BaseRecognizer",
    # Registry
    "RecognizerRegistry",
    "recognizer_registry",
    # Factory functions
    "create_recognizer",
    "list_available_recognizers",
    # Recognizer classes (lazy loaded)
    "TextRecognizer",
    "PaddleOCRVLRecognizer",
    "DeepSeekOCRRecognizer",
]


def create_recognizer(name: str, **kwargs: Any) -> Recognizer:
    """Create a recognizer instance.

    Args:
        name: Recognizer name or model name
            - Recognizer names: "openai", "gemini", "paddleocr-vl", "deepseek-ocr"
            - Model names (auto-resolved):
                - Gemini models: "gemini-2.5-flash" → "gemini"
                - GPT models: "gpt-4o" → "openai"
        **kwargs: Arguments for recognizer constructor
            Common args:
                - cache_dir: Cache directory (default: ".cache")
                - use_cache: Enable caching (default: True)
                - model: Model name (backend-specific)
                - gemini_tier: Gemini API tier ("free", "tier1", etc.)

    Returns:
        Recognizer instance

    Raises:
        ValueError: If recognizer name is unknown
        ImportError: If recognizer dependencies not available

    Example:
        >>> recognizer = create_recognizer("gemini-2.5-flash")
        >>> blocks = recognizer.process_blocks(image, blocks)
    """
    return recognizer_registry.create(name, **kwargs)


def list_available_recognizers() -> list[str]:
    """List available recognizer names.

    Returns:
        Sorted list of available recognizer names

    Example:
        >>> recognizers = list_available_recognizers()
        >>> print(recognizers)
        ['deepseek-ocr', 'gemini', 'openai', 'paddleocr-vl']
    """
    return recognizer_registry.list_available()


def __getattr__(name: str) -> Any:
    """Lazy import for recognizer classes.

    This allows direct imports like:
        from pipeline.recognition import TextRecognizer

    While keeping the benefits of lazy loading (only import when needed).
    """
    if name == "TextRecognizer":
        from .text_recognizer import TextRecognizer

        return TextRecognizer
    elif name == "PaddleOCRVLRecognizer":
        from .paddleocr import PaddleOCRVLRecognizer

        return PaddleOCRVLRecognizer
    elif name == "DeepSeekOCRRecognizer":
        from .deepseek import DeepSeekOCRRecognizer

        return DeepSeekOCRRecognizer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
