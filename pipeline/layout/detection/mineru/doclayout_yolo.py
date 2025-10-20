"""MinerU DocLayout-YOLO detector implementation.

MinerU DocLayout-YOLO BBox Format:
- Output: poly [x1, y1, x2, y2, x3, y3, x4, y4] (8 points)
- Simplified to: [xmin, ymin, xmax, ymax]
- Origin: Top-Left (0, 0)
- Coordinate Order: Left-Top + Right-Bottom
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mineru.model.layout.doclayoutyolo import DocLayoutYOLOModel
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

from ....types import BBox, Block, BlockTypeMapper

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class MinerUDocLayoutYOLODetector:
    """Detector using MinerU's DocLayout-YOLO implementation.

    This uses MinerU's DocLayoutYOLOModel which may have different
    model weights or configurations compared to the main project's version.

    BBox Format: poly [x1,y1,x2,y2,x3,y3,x4,y4] → simplified to [xmin,ymin,xmax,ymax]
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        device: str = "cuda",
        imgsz: int = 1280,
    ):
        """Initialize MinerU DocLayout-YOLO detector.

        Args:
            model_path: Path to model weights (uses MinerU default if None)
            confidence_threshold: Confidence threshold for detections
            device: Device to use ("cuda" or "cpu")
            imgsz: Image size for inference
        """
        if model_path is None:
            model_root = auto_download_and_get_model_root_path(ModelPath.doclayout_yolo)
            model_path = str(Path(model_root) / ModelPath.doclayout_yolo)

        self.model = DocLayoutYOLOModel(
            weight=str(model_path),
            device=device,
            imgsz=imgsz,
            conf=confidence_threshold,
        )
        self.confidence_threshold = confidence_threshold

        logger.info("MinerU DocLayout-YOLO detector initialized")

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected blocks in unified format
        """
        try:
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError("PIL required for MinerU DocLayout-YOLO") from e

        pil_image = PILImage.fromarray(image)
        raw_results = self.model.predict(pil_image)

        logger.debug("Detected %d blocks with MinerU DocLayout-YOLO", len(raw_results))

        return [self._to_block(r) for r in raw_results]

    def _to_block(self, raw_data: dict[str, Any]) -> Block:
        """Convert MinerU DocLayout-YOLO result to unified Block format.

        Args:
            raw_data: {"category_id": int, "poly": [8 points], "score": float}

        Returns:
            Unified Block with BBox and standardized type
        """
        poly = raw_data["poly"]
        xmin = min(poly[0], poly[2], poly[4], poly[6])
        ymin = min(poly[1], poly[3], poly[5], poly[7])
        xmax = max(poly[0], poly[2], poly[4], poly[6])
        ymax = max(poly[1], poly[3], poly[5], poly[7])

        bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)

        category_id = raw_data["category_id"]
        original_type = self._category_to_type(category_id)

        # Map to standardized type
        standardized_type = BlockTypeMapper.map_type(original_type, "mineru-doclayout-yolo")

        return Block(
            type=standardized_type,
            bbox=bbox,
            detection_confidence=float(raw_data["score"]),
            source="mineru-doclayout-yolo",
        )

    def _category_to_type(self, category_id: int) -> str:
        """Convert MinerU category ID to block type name."""
        category_map = {
            0: "title",
            1: "plain text",
            2: "abandon",
            3: "figure",
            4: "figure_caption",
            5: "table",
            6: "table_caption",
            7: "table_footnote",
            8: "isolate_formula",
            9: "formula_caption",
        }
        return category_map.get(category_id, "unknown")
