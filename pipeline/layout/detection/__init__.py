"""Layout detection module.

Detectors organized by framework:
- doclayout_yolo.py: This project's DocLayout-YOLO
- mineru/: MinerU detectors (DocLayout-YOLO, VLM)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pipeline.types import Detector

from .detector import LayoutDetector
from .doclayout_yolo import DocLayoutYOLODetector
from .mineru import MinerUDocLayoutYOLODetector, MinerUVLMDetector

logger = logging.getLogger(__name__)

__all__ = [
    "LayoutDetector",
    "DocLayoutYOLODetector",
    "MinerUVLMDetector",
    "MinerUDocLayoutYOLODetector",
    "create_detector",
    "list_available_detectors",
]

_DETECTOR_REGISTRY: dict[str, Callable[..., Detector]] = {
    "doclayout-yolo": DocLayoutYOLODetector,
}

if MinerUVLMDetector is not None:
    _DETECTOR_REGISTRY["mineru-vlm"] = MinerUVLMDetector

if MinerUDocLayoutYOLODetector is not None:
    _DETECTOR_REGISTRY["mineru-doclayout-yolo"] = MinerUDocLayoutYOLODetector


def create_detector(name: str, **kwargs: Any) -> Detector:
    """Create a detector instance.

    Args:
        name: Detector name
        **kwargs: Arguments for detector

    Returns:
        Detector instance
    """
    if name not in _DETECTOR_REGISTRY:
        available = ", ".join(_DETECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown detector: {name}. Available: {available}")

    return _DETECTOR_REGISTRY[name](**kwargs)


def list_available_detectors() -> list[str]:
    """List available detector names."""
    return list(_DETECTOR_REGISTRY.keys())
