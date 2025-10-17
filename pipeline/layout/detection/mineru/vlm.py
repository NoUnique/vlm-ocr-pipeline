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

from mineru.backend.vlm.vlm_analyze import ModelSingleton

from ....types import BBox, Block, BlockTypeMapper

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
        self.model_path = model
        self.backend = backend
        self.detection_only = detection_only

        # WORKAROUND: MinerU's ModelSingleton doesn't pass model_path to MinerUClient
        # when model/processor are None. We need to pre-load them.
        if backend == "transformers" and model:
            try:
                from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
                from transformers import __version__ as transformers_version
                from packaging import version

                # Determine dtype parameter name based on transformers version
                if version.parse(transformers_version) >= version.parse("4.56.0"):
                    dtype_key = "dtype"
                else:
                    dtype_key = "torch_dtype"

                # Load model and processor with device_map="auto" (requires accelerate)
                vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model,
                    device_map="auto",
                    **{dtype_key: "auto"},
                )
                vlm_processor = AutoProcessor.from_pretrained(model, use_fast=True)

                # Pass loaded model/processor to avoid model_path issue
                vlm_kwargs["model"] = vlm_model
                vlm_kwargs["processor"] = vlm_processor

                logger.info("MinerU VLM model loaded successfully")
            except Exception as e:
                logger.warning("Failed to pre-load model, falling back to ModelSingleton: %s", e)

        # WORKAROUND PART 2: ModelSingleton.get_model() has a bug where it doesn't pass
        # model/processor from kwargs to MinerUClient. Bypass it entirely when we've pre-loaded.
        if "model" in vlm_kwargs and "processor" in vlm_kwargs:
            from mineru_vl_utils import MinerUClient

            self.vlm = MinerUClient(
                backend=backend,
                model_path=model,
                **vlm_kwargs,
            )
        else:
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

    def detect(self, image: np.ndarray) -> list[Block]:
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

        print(f"[BBOX-DEBUG] Image shape: {image.shape}, PIL size: {pil_image.size}")
        print(f"[BBOX-DEBUG] detection_only={self.detection_only}")

        if self.detection_only:
            # TODO: MinerU doesn't expose step 1 (detection only) separately
            # For now, use two-step and remove ordering info
            logger.warning("MinerU VLM detection_only mode: using two-step extract but ignoring ordering")
            results = self.vlm.batch_two_step_extract(images=[pil_image])
            raw_blocks = results[0]

            for block in raw_blocks:
                if "index" in block:
                    del block["index"]
        else:
            results = self.vlm.batch_two_step_extract(images=[pil_image])
            raw_blocks = results[0]

        print(f"[BBOX-DEBUG] MinerU returned {len(raw_blocks)} blocks")
        if raw_blocks:
            print(f"[BBOX-DEBUG] First raw block: {raw_blocks[0]}")
            print(f"[BBOX-DEBUG] Second raw block: {raw_blocks[1] if len(raw_blocks) > 1 else 'N/A'}")

        logger.debug("Detected %d regions with MinerU VLM", len(raw_blocks))

        # Get image dimensions for bbox scaling
        img_height, img_width = image.shape[:2]

        return [self._to_block(block, img_width, img_height) for block in raw_blocks]

    def _to_block(self, block: dict[str, Any], img_width: int, img_height: int) -> Block:
        """Convert MinerU block to unified Block format.

        Args:
            block: {"type": str, "bbox": [x0, y0, x1, y1], "text": str (optional), "index": int (optional)}
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Unified Block dataclass instance with BBox and standardized type
        """
        # MinerU returns normalized coordinates (0.0-1.0), convert to pixel coordinates
        normalized_bbox = block["bbox"]
        x0_norm, y0_norm, x1_norm, y1_norm = normalized_bbox[:4]

        # Scale to pixel coordinates
        x0_px = x0_norm * img_width
        y0_px = y0_norm * img_height
        x1_px = x1_norm * img_width
        y1_px = y1_norm * img_height

        print(f"[BBOX-DEBUG] Normalized: {normalized_bbox[:4]} -> Pixel: [{x0_px:.1f}, {y0_px:.1f}, {x1_px:.1f}, {y1_px:.1f}]")

        # Create BBox with pixel coordinates
        bbox = BBox.from_xyxy(x0_px, y0_px, x1_px, y1_px)

        # Map to standardized type (MinerU VLM already uses standardized types,
        # but we apply the mapper for consistency and future-proofing)
        original_type = block["type"]
        standardized_type = BlockTypeMapper.map_type(original_type, "mineru-vlm")

        # MinerU VLM returns 'content' field with OCR text
        # Note: This text will only be used if no separate recognizer is configured,
        # or if the recognizer is explicitly set to use detector's content
        # MinerU VLM 2.5 does not provide confidence scores
        confidence_value = block.get("confidence")
        block_obj = Block(
            type=standardized_type,
            bbox=bbox,
            detection_confidence=float(confidence_value) if confidence_value is not None else None,
            source="mineru-vlm",
            text=block.get("content"),  # MinerU uses 'content', not 'text'
            index=block.get("index"),
            order=block.get("index"),  # Use index as reading order
        )

        return block_obj
