"""Layout detector for identifying document regions using DocLayout-YOLO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from models import DocLayoutYOLO

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Detects layout regions in document images.
    
    This class uses DocLayout-YOLO model to identify and classify
    regions such as text blocks, titles, tables, figures, etc.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the layout detector.
        
        Args:
            model_path: Path to DocLayout-YOLO model weights
            confidence_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.model = self._setup_model()

    def _setup_model(self) -> DocLayoutYOLO:
        """Setup DocLayout-YOLO model.
        
        Returns:
            Initialized DocLayoutYOLO model
        """
        model = DocLayoutYOLO(model_path=self.model_path)
        logger.info("DocLayout-YOLO model loaded successfully")
        return model

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect layout regions in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected regions with coordinates, types, and confidence scores
        """
        regions = self.model.predict(image, conf=self.confidence_threshold)
        logger.debug("Detected %d regions", len(regions))
        return regions

    def crop_region(self, image: np.ndarray, region: dict[str, Any]) -> np.ndarray:
        """Crop a region from an image.
        
        Args:
            image: Source image
            region: Region information with coordinates
            
        Returns:
            Cropped image region
        """
        coords = region["coords"]
        x, y, w, h = coords  # coords format is [x, y, width, height]

        # Convert to x1, y1, x2, y2
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Add small padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        # Ensure valid dimensions
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid region coordinates: x1=%s, y1=%s, x2=%s, y2=%s", x1, y1, x2, y2)
            return np.zeros((1, 1, 3), dtype=np.uint8)  # Return minimal image

        return image[y1:y2, x1:x2]

