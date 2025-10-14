"""olmOCR ordering implementations."""

from __future__ import annotations

__all__ = []

try:
    from .vlm import OlmOCRVLMSorter

    __all__.append("OlmOCRVLMSorter")
except ImportError:
    OlmOCRVLMSorter = None  # type: ignore[assignment, misc]
