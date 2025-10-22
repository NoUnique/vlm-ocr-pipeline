"""Reading order analysis module.

Sorters organized by framework:
- pymupdf.py: PyMuPDF multi-column
- mineru/: MinerU sorters (LayoutReader, XY-Cut, VLM)
- olmocr/: olmOCR sorters (VLM)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pipeline.types import Sorter

from .analyzer import ReadingOrderAnalyzer
from .mineru import MinerULayoutReaderSorter, MinerUVLMSorter, MinerUXYCutSorter
from .olmocr import OlmOCRVLMSorter
from .paddleocr import PPDocLayoutV2Sorter
from .pymupdf import MultiColumnSorter

logger = logging.getLogger(__name__)

__all__ = [
    "ReadingOrderAnalyzer",
    "MultiColumnSorter",
    "MinerULayoutReaderSorter",
    "MinerUXYCutSorter",
    "MinerUVLMSorter",
    "OlmOCRVLMSorter",
    "PPDocLayoutV2Sorter",
    "create_sorter",
    "list_available_sorters",
    "validate_combination",
]

_SORTER_REGISTRY: dict[str, Callable[..., Sorter]] = {
    "pymupdf": MultiColumnSorter,  # Legacy name for backward compatibility
}

if MinerULayoutReaderSorter is not None:
    _SORTER_REGISTRY["mineru-layoutreader"] = MinerULayoutReaderSorter

if MinerUXYCutSorter is not None:
    _SORTER_REGISTRY["mineru-xycut"] = MinerUXYCutSorter

if MinerUVLMSorter is not None:
    _SORTER_REGISTRY["mineru-vlm"] = MinerUVLMSorter

if OlmOCRVLMSorter is not None:
    _SORTER_REGISTRY["olmocr-vlm"] = OlmOCRVLMSorter

if PPDocLayoutV2Sorter is not None:
    _SORTER_REGISTRY["paddleocr-doclayout-v2"] = PPDocLayoutV2Sorter


def create_sorter(name: str, **kwargs: Any) -> Sorter:
    """Create a sorter instance.

    Args:
        name: Sorter name
        **kwargs: Arguments for sorter

    Returns:
        Sorter instance
    """
    if name not in _SORTER_REGISTRY:
        available = ", ".join(_SORTER_REGISTRY.keys())
        raise ValueError(f"Unknown sorter: {name}. Available: {available}")

    return _SORTER_REGISTRY[name](**kwargs)


def list_available_sorters() -> list[str]:
    """List available sorter names."""
    return list(_SORTER_REGISTRY.keys())


VALID_COMBINATIONS = {
    "doclayout-yolo": ["pymupdf", "mineru-layoutreader", "mineru-xycut", "olmocr-vlm"],
    "mineru-doclayout-yolo": ["pymupdf", "mineru-layoutreader", "mineru-xycut", "olmocr-vlm"],
    "mineru-vlm": ["mineru-vlm"],  # Only mineru-vlm sorter (tightly coupled)
    "paddleocr-doclayout-v2": ["paddleocr-doclayout-v2"],  # Only paddleocr-doclayout-v2 sorter (tightly coupled)
}

RECOMMENDED_COMBINATIONS = {
    ("mineru-vlm", "mineru-vlm"): "Tightly coupled - detection and ordering by same VLM!",
    ("doclayout-yolo", "mineru-xycut"): "Fast and accurate - good default!",
    ("mineru-doclayout-yolo", "mineru-xycut"): "MinerU detection with fast ordering!",
    ("paddleocr-doclayout-v2", "paddleocr-doclayout-v2"): "Tightly coupled - preserves pointer network ordering!",
}

REQUIRED_COMBINATIONS = {
    "mineru-vlm": "mineru-vlm",  # sorter → required detector
    "paddleocr-doclayout-v2": "paddleocr-doclayout-v2",  # sorter → required detector
}


def validate_combination(detector: str, sorter: str) -> tuple[bool, str]:
    """Validate detector + sorter combination.

    Returns:
        (is_valid, message)
    """
    if sorter in REQUIRED_COMBINATIONS:
        required_detector = REQUIRED_COMBINATIONS[sorter]
        if detector != required_detector:
            return False, f"Sorter '{sorter}' requires detector '{required_detector}' (tightly coupled)"

    if detector not in VALID_COMBINATIONS:
        return False, f"Unknown detector: {detector}"

    if sorter not in VALID_COMBINATIONS[detector]:
        valid = ", ".join(VALID_COMBINATIONS[detector])
        return False, f"Invalid combination. {detector} supports: {valid}"

    if (detector, sorter) in RECOMMENDED_COMBINATIONS:
        msg = RECOMMENDED_COMBINATIONS[(detector, sorter)]
        return True, f"✨ {msg}"

    return True, f"Valid: {detector} + {sorter}"
