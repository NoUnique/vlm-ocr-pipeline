"""Layout detection module.

Detectors organized by framework:
- doclayout_yolo.py: This project's DocLayout-YOLO
- mineru/: MinerU detectors (DocLayout-YOLO, VLM)
- paddleocr/: PaddleOCR detectors (PP-DocLayoutV2)

Components:
- BaseDetector: Abstract base class for all detectors
- DetectorRegistry: Registry for detector management
- create_detector: Factory function for detector creation
"""

from __future__ import annotations

import logging
from typing import Any

from pipeline.types import Detector

from .base import BaseDetector
from .registry import DetectorRegistry, detector_registry

logger = logging.getLogger(__name__)

__all__ = [
    # Base class
    "BaseDetector",
    # Registry
    "DetectorRegistry",
    "detector_registry",
    # Factory functions
    "create_detector",
    "list_available_detectors",
    # Detector classes (lazy loaded)
    "DocLayoutYOLODetector",
    "MinerUVLMDetector",
    "MinerUDocLayoutYOLODetector",
    "PPDocLayoutV2Detector",
]


def create_detector(name: str, **kwargs: Any) -> Detector:
    """Create a detector instance.

    Args:
        name: Detector name (e.g., "doclayout-yolo", "paddleocr-doclayout-v2")
        **kwargs: Arguments for detector constructor

    Returns:
        Detector instance

    Raises:
        ValueError: If detector name is unknown
        ImportError: If detector dependencies not available

    Example:
        >>> detector = create_detector("doclayout-yolo", confidence_threshold=0.5)
        >>> blocks = detector.detect(image)
    """
    return detector_registry.create(name, **kwargs)


def list_available_detectors() -> list[str]:
    """List available detector names.

    Returns:
        Sorted list of available detector names
    """
    return detector_registry.list_available()


def __getattr__(name: str) -> Any:
    """Lazy import for detector classes.

    This allows direct imports like:
        from pipeline.layout.detection import DocLayoutYOLODetector

    While keeping the benefits of lazy loading (only import when needed).
    """
    if name == "DocLayoutYOLODetector":
        from .doclayout_yolo import DocLayoutYOLODetector  # noqa: PLC0415

        return DocLayoutYOLODetector
    elif name == "MinerUVLMDetector":
        from .mineru import MinerUVLMDetector  # noqa: PLC0415

        return MinerUVLMDetector
    elif name == "MinerUDocLayoutYOLODetector":
        from .mineru import MinerUDocLayoutYOLODetector  # noqa: PLC0415

        return MinerUDocLayoutYOLODetector
    elif name == "PPDocLayoutV2Detector":
        from .paddleocr import PPDocLayoutV2Detector  # noqa: PLC0415

        return PPDocLayoutV2Detector
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
