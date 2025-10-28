"""Layout detection module.

Detectors organized by framework:
- doclayout_yolo.py: This project's DocLayout-YOLO
- mineru/: MinerU detectors (DocLayout-YOLO, VLM)
- paddleocr/: PaddleOCR detectors (PP-DocLayoutV2)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pipeline.types import Detector

logger = logging.getLogger(__name__)

__all__ = [
    "LayoutDetector",
    "DocLayoutYOLODetector",
    "MinerUVLMDetector",
    "MinerUDocLayoutYOLODetector",
    "PPDocLayoutV2Detector",
    "create_detector",
    "list_available_detectors",
]

# Lazy import registry - imports detectors only when needed
_DETECTOR_REGISTRY: dict[str, Callable[..., Detector]] = {}


def _get_detector_class(name: str) -> Callable[..., Detector]:
    """Lazy load detector class."""
    if name == "doclayout-yolo":
        from .doclayout_yolo import DocLayoutYOLODetector  # noqa: PLC0415

        return DocLayoutYOLODetector
    elif name == "mineru-vlm":
        from .mineru import MinerUVLMDetector  # noqa: PLC0415

        return MinerUVLMDetector
    elif name == "mineru-doclayout-yolo":
        from .mineru import MinerUDocLayoutYOLODetector  # noqa: PLC0415

        return MinerUDocLayoutYOLODetector
    elif name == "paddleocr-doclayout-v2":
        try:
            from .paddleocr import PPDocLayoutV2Detector  # noqa: PLC0415

            return PPDocLayoutV2Detector
        except ImportError as e:
            raise ImportError(f"PaddleOCR detector not available: {e}") from e
    else:
        raise ValueError(f"Unknown detector: {name}")


def create_detector(name: str, **kwargs: Any) -> Detector:
    """Create a detector instance.

    Args:
        name: Detector name
        **kwargs: Arguments for detector

    Returns:
        Detector instance
    """
    detector_class = _get_detector_class(name)
    return detector_class(**kwargs)


def list_available_detectors() -> list[str]:
    """List available detector names."""
    return ["doclayout-yolo", "mineru-vlm", "mineru-doclayout-yolo", "paddleocr-doclayout-v2"]
