"""DocLayout-YOLO detector implementation.

DocLayout-YOLO BBox Format:
- Output: [x, y, width, height]
- Origin: Top-Left (0, 0)
- Coordinate Order: Left-Top + Size
- Example: [100, 50, 200, 150] means rectangle from (100,50) with width=200, height=150
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ...types import BBox, Region

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class DocLayoutYOLODetector:
    """Detector using DocLayout-YOLO model.

    This detector wraps the existing DocLayoutYOLO model and provides
    output in the unified Region format.

    BBox Format: [x, y, width, height] - Top-Left origin
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize DocLayout-YOLO detector.

        Args:
            model_path: Path to DocLayout-YOLO model weights
            confidence_threshold: Confidence threshold for detections
        """
        from models.doclayout_yolo import DocLayoutYOLO

        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.model = DocLayoutYOLO(model_path=model_path)

        logger.info("DocLayout-YOLO detector initialized")

    def detect(self, image: np.ndarray) -> list[Region]:
        """Detect layout regions in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected regions in unified format

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> regions = detector.detect(image)
            >>> regions[0]["type"]
            'plain text'
            >>> regions[0]["coords"]  # [x, y, w, h]
            [100, 50, 200, 150]
            >>> regions[0]["bbox"]  # BBox object
            BBox(x0=100, y0=50, x1=300, y1=200)
        """
        raw_results = self.model.predict(image, conf=self.confidence_threshold)
        logger.debug("Detected %d regions with DocLayout-YOLO", len(raw_results))

        return [self._to_region(r) for r in raw_results]

    def _to_region(self, raw_data: dict) -> Region:
        """Convert DocLayout-YOLO result to unified Region format.

        Args:
            raw_data: {"type": str, "coords": [x, y, w, h], "confidence": float}

        Returns:
            Unified Region dataclass instance with BBox
        """
        coords = raw_data["coords"]
        bbox = BBox.from_xywh(*coords[:4])

        return Region(
            type=raw_data["type"],
            bbox=bbox,
            confidence=float(raw_data["confidence"]),
            source="doclayout-yolo",
        )
