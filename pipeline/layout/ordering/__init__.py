"""Reading order analysis module.

Sorters organized by framework:
- pymupdf.py: PyMuPDF multi-column
- mineru/: MinerU sorters (LayoutReader, XY-Cut, VLM)
- olmocr/: olmOCR sorters (VLM)
- paddleocr/: PaddleOCR sorters (PP-DocLayoutV2)
"""

from __future__ import annotations

import logging
from typing import Any

from pipeline.types import Sorter

from .analyzer import ReadingOrderAnalyzer
from .mineru import MinerULayoutReaderSorter, MinerUVLMSorter, MinerUXYCutSorter
from .olmocr import OlmOCRVLMSorter
from .paddleocr import PPDocLayoutV2Sorter
from .pymupdf import MultiColumnSorter
from .registry import SorterRegistry, sorter_registry

logger = logging.getLogger(__name__)

__all__ = [
    # Classes
    "ReadingOrderAnalyzer",
    "MultiColumnSorter",
    "MinerULayoutReaderSorter",
    "MinerUXYCutSorter",
    "MinerUVLMSorter",
    "OlmOCRVLMSorter",
    "PPDocLayoutV2Sorter",
    # Registry
    "SorterRegistry",
    "sorter_registry",
    # Functions
    "create_sorter",
    "list_available_sorters",
    "validate_combination",
]


def create_sorter(name: str, **kwargs: Any) -> Sorter:
    """Create a sorter instance.

    Args:
        name: Sorter name
        **kwargs: Arguments for sorter

    Returns:
        Sorter instance
    """
    return sorter_registry.create(name, **kwargs)


def list_available_sorters() -> list[str]:
    """List available sorter names."""
    return sorter_registry.list_available()


# Valid detector + sorter combinations
VALID_COMBINATIONS = {
    "doclayout-yolo": ["pymupdf", "mineru-layoutreader", "mineru-xycut", "olmocr-vlm"],
    "mineru-doclayout-yolo": ["pymupdf", "mineru-layoutreader", "mineru-xycut", "olmocr-vlm"],
    "mineru-vlm": ["mineru-vlm"],  # Tightly coupled
    "paddleocr-doclayout-v2": ["paddleocr-doclayout-v2"],  # Tightly coupled
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
