"""MinerU VLM detector implementation.

MinerU VLM BBox Format:
- Input/Output: [x0, y0, x1, y1]
- Origin: Top-Left (0, 0)
- Coordinate Order: Left-Top + Right-Bottom
- Example: [100, 50, 300, 200] means rectangle from (100,50) to (300,200)
- Additional: May include "index" field for reading order
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ....types import BBox, Region

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class MinerUVLMDetector:
    """Detector using MinerU VLM model.

    MinerU VLM can be used in two modes:
    1. Detection only: Extract layout regions (step 1)
    2. Full pipeline: Detection + ordering + OCR (two-step)

    This detector provides flexible usage based on the detection_only flag.
    """

    def __init__(
        self,
        model: str | None = None,
        backend: str = "transformers",
        detection_only: bool = True,
        **vlm_kwargs: Any,
    ):
        """Initialize MinerU VLM detector.

        Args:
            model: Model path or name (e.g., "opendatalab/PDF-Extract-Kit-1.0")
            backend: VLM backend ("transformers", "vllm-engine", "vllm-async-engine")
            detection_only: If True, only perform detection (step 1).
                          If False, perform full two-step extract (detection + ordering)
            **vlm_kwargs: Additional arguments for VLM initialization

        Raises:
            ImportError: If MinerU dependencies not available
        """
        try:
            from mineru.backend.vlm.vlm_analyze import ModelSingleton
        except ImportError as e:
            raise ImportError(
                "MinerU VLM dependencies not available. "
                "Please install MinerU to use this detector."
            ) from e

        self.model_path = model
        self.backend = backend
        self.detection_only = detection_only

        model_singleton = ModelSingleton()
        self.vlm = model_singleton.get_model(
            backend=backend,
            model_path=model,
            server_url=None,
            **vlm_kwargs,
        )

        logger.info(
            "MinerU VLM detector initialized (backend=%s, detection_only=%s)",
            backend,
            detection_only,
        )

    def detect(self, image: np.ndarray) -> list[Region]:
        """Detect layout regions using MinerU VLM.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected regions in unified format

        Example:
            >>> detector = MinerUVLMDetector(model="opendatalab/PDF-Extract-Kit-1.0")
            >>> regions = detector.detect(image)
            >>> regions[0]["type"]
            'text'
            >>> regions[0]["bbox"]
            BBox(x0=100, y0=50, x1=300, y1=200)
            >>> # If detection_only=False, regions will have reading_order_rank
            >>> regions[0].get("reading_order_rank")
            0
        """
        try:
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError("PIL required for MinerU VLM") from e

        pil_image = PILImage.fromarray(image)

        if self.detection_only:
            # TODO: MinerU doesn't expose step 1 (detection only) separately
            # For now, use two-step and remove ordering info
            logger.warning(
                "MinerU VLM detection_only mode: using two-step extract but ignoring ordering"
            )
            results = self.vlm.batch_two_step_extract(images=[pil_image])
            raw_blocks = results[0]

            for block in raw_blocks:
                if "index" in block:
                    del block["index"]
        else:
            results = self.vlm.batch_two_step_extract(images=[pil_image])
            raw_blocks = results[0]

        logger.debug("Detected %d regions with MinerU VLM", len(raw_blocks))

        return [self._to_region(block) for block in raw_blocks]
    
    def _to_region(self, block: dict[str, Any]) -> Region:
        """Convert MinerU block to unified Region format.
        
        Args:
            block: {"type": str, "bbox": [x0, y0, x1, y1], "text": str (optional), "index": int (optional)}
            
        Returns:
            Unified Region dict with BBox
        """
        bbox = BBox.from_mineru_bbox(block["bbox"])
        x, y, w, h = bbox.to_xywh()

        region: Region = {
            "type": block["type"],
            "coords": [x, y, w, h],
            "confidence": float(block.get("confidence", 1.0)),
            "bbox": bbox,
            "source": "mineru-vlm",
        }

        if "text" in block:
            region["text"] = block["text"]

        if "index" in block:
            region["index"] = block["index"]
            region["reading_order_rank"] = block["index"]

        return region

