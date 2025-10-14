"""MinerU detection implementations."""

from __future__ import annotations

__all__ = []

try:
    from .vlm import MinerUVLMDetector

    __all__.append("MinerUVLMDetector")
except ImportError:
    MinerUVLMDetector = None  # type: ignore[assignment, misc]

try:
    from .doclayout_yolo import MinerUDocLayoutYOLODetector

    __all__.append("MinerUDocLayoutYOLODetector")
except ImportError:
    MinerUDocLayoutYOLODetector = None  # type: ignore[assignment, misc]
