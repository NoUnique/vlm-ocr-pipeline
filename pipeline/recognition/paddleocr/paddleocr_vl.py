"""PaddleOCR-VL (PaddleOCR-VL-0.9B) text recognition implementation.

PaddleOCR-VL-0.9B is a compact yet powerful Vision Language Model designed for
document element recognition. It combines a NaViT-style dynamic resolution vision
encoder (SiglipVisionModel) with ERNIE-4.5-0.3B language model.

Key Features:
- Model Size: 0.9B parameters
- Vision Encoder: SiglipVisionModel (NaViT-style dynamic resolution)
- Language Model: ERNIE-4.5-0.3B
- Multilingual: Supports 109 languages
- SOTA Performance: Excellent at recognizing text, tables, formulas, and charts
- Low Resource: Compact model with efficient inference

Architecture:
- Vision: SiglipVisionModel with dynamic resolution (adaptive to input size)
- Projector: Maps vision features to language model space
- Language: ERNIE-4.5-0.3B for text generation
- Training: ~5M document understanding multimodal datasets

Usage:
    >>> from pipeline.recognition.paddleocr import PaddleOCRVLRecognizer
    >>> recognizer = PaddleOCRVLRecognizer()
    >>> blocks_with_text = recognizer.process_blocks(image, blocks)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ...types import Block, Recognizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PaddleOCRVLRecognizer(Recognizer):
    """Text recognizer using PaddleOCR-VL-0.9B model.

    This class implements the Recognizer protocol using PaddleOCR's pipeline which includes:
    - Layout detection (PP-DocLayoutV2) - optional
    - VL Recognition (PaddleOCR-VL-0.9B) - 0.9B params, NaViT + ERNIE-4.5-0.3B

    Note: Text correction is not supported by this recognizer.

    Implements:
        Recognizer: Text recognition protocol with PaddleOCR-VL backend

    Example:
        >>> recognizer = PaddleOCRVLRecognizer()
        >>> blocks_with_text = recognizer.process_blocks(image, blocks)
    """

    def __init__(
        self,
        vl_rec_model_name: str = "PaddleOCR-VL-0.9B",
        vl_rec_model_dir: str | Path | None = None,
        device: str | None = None,
        vl_rec_backend: str = "native",
        use_layout_detection: bool = False,
        use_multi_gpu: bool = True,
        **kwargs,
    ):
        """Initialize PaddleOCR-VL recognizer.

        Args:
            vl_rec_model_name: VL recognition model name (default: "PaddleOCR-VL-0.9B")
            vl_rec_model_dir: Path to custom model directory (optional)
            device: Device to run model on ("cpu", "gpu", "gpu:0", etc.)
                Default: GPU 0 if available, otherwise CPU
            vl_rec_backend: Inference backend ("native", "vllm-server", "sglang-server")
                - "native": Direct PaddlePaddle inference (default)
                - "vllm-server": Use vLLM server for acceleration
                - "sglang-server": Use SGLang server for acceleration
            use_layout_detection: Whether to use layout detection (default: False)
                When False, the pipeline only performs VL recognition on provided blocks
            use_multi_gpu: Whether to use multiple GPUs if available (default: True)
            **kwargs: Additional arguments passed to PaddleOCRVL

        Example:
            >>> # Basic usage (recognition only)
            >>> recognizer = PaddleOCRVLRecognizer()

            >>> # With custom backend
            >>> recognizer = PaddleOCRVLRecognizer(vl_rec_backend="vllm-server")

            >>> # With layout detection enabled
            >>> recognizer = PaddleOCRVLRecognizer(use_layout_detection=True)

            >>> # With multi-GPU disabled
            >>> recognizer = PaddleOCRVLRecognizer(use_multi_gpu=False)
        """
        # Lazy import to avoid loading PaddleOCR unless needed
        try:
            import paddle  # noqa: PLC0415
            from paddleocr import PaddleOCRVL  # noqa: PLC0415  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "PaddleOCR is required for PaddleOCR-VL recognizer. Please install it: pip install paddleocr"
            ) from e

        self.vl_rec_model_name = vl_rec_model_name
        self.vl_rec_model_dir = Path(vl_rec_model_dir) if vl_rec_model_dir else None
        self.device = device
        self.vl_rec_backend = vl_rec_backend
        self.use_layout_detection = use_layout_detection
        self.use_multi_gpu = use_multi_gpu

        # Check for multi-GPU availability
        self.gpu_count = paddle.device.cuda.device_count() if paddle.device.cuda.device_count() > 0 else 0
        if self.gpu_count > 1 and use_multi_gpu and vl_rec_backend == "native":
            logger.info("Multi-GPU mode enabled: will use %d GPUs with multiprocessing", self.gpu_count)
            self.multi_gpu_enabled = True
        else:
            self.multi_gpu_enabled = False
            if use_multi_gpu and self.gpu_count <= 1:
                logger.warning("Multi-GPU requested but only %d GPU available", self.gpu_count)

        # Initialize PaddleOCRVL pipeline
        init_kwargs = {
            "vl_rec_model_name": vl_rec_model_name,
            "vl_rec_backend": vl_rec_backend,
            "use_layout_detection": use_layout_detection,
            "device": device,
            **kwargs,
        }
        if vl_rec_model_dir:
            init_kwargs["vl_rec_model_dir"] = str(vl_rec_model_dir)

        self.pipeline = PaddleOCRVL(**init_kwargs)

        logger.info(
            "PaddleOCR-VL recognizer initialized (model: %s, device: %s, backend: %s, multi_gpu: %s)",
            vl_rec_model_name,
            device or "auto",
            vl_rec_backend,
            self.multi_gpu_enabled,
        )

    def process_blocks(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
        **kwargs,
    ) -> list[Block]:
        """Process blocks and extract text using PaddleOCR-VL.

        PaddleOCR-VL is an end-to-end pipeline. We crop each block and process individually.

        Args:
            image: Input image as numpy array (HWC format, RGB/BGR)
            blocks: List of Block objects with bounding boxes
            **kwargs: Additional arguments (not used, for compatibility)

        Returns:
            List of Block objects with extracted text in the `text` field

        Example:
            >>> blocks_with_text = recognizer.process_blocks(image, blocks)
            >>> for block in blocks_with_text:
            ...     print(f"{block.type}: {block.text}")
        """
        logger.info("=== PaddleOCRVLRecognizer.process_blocks START ===")
        logger.info("Number of blocks: %d", len(blocks) if blocks else 0)

        if not blocks:
            logger.warning("No blocks provided for recognition")
            return []

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            logger.info("Converting PIL Image to numpy array")
            image = np.array(image)

        logger.info("Image shape: %s", image.shape)

        # Process each block individually by cropping
        output_blocks = []
        for idx, block in enumerate(blocks):
            logger.info("--- Processing block %d/%d (type=%s) ---", idx + 1, len(blocks), block.type)
            # Skip blocks without valid bbox
            if not block.bbox:
                logger.warning("Block %d without bbox, skipping", idx)
                output_blocks.append(block)
                continue

            # Crop image to block bbox
            x0, y0, x1, y1 = block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1

            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)

            # Check for valid crop dimensions
            if x1 <= x0 or y1 <= y0:
                logger.warning("Block %d has invalid dimensions: (%d,%d,%d,%d)", idx, x0, y0, x1, y1)
                output_blocks.append(block)
                continue

            # Crop the block region
            cropped = image[y0:y1, x0:x1]
            logger.info("Cropped block %d to shape: %s", idx, cropped.shape)

            # Process cropped image with PaddleOCR-VL
            try:
                logger.info("Calling PaddleOCR-VL predict() for block %d...", idx)
                results = self.pipeline.predict(cropped, use_layout_detection=False)
                logger.info("PaddleOCR-VL predict() completed for block %d", idx)

                # Extract text from PaddleOCRVL results
                # Results format: list of PaddleOCRVLResult (dict subclass) objects
                # Each result has 'parsing_res_list' key with list of PaddleOCRVLBlock objects
                text_parts = []
                if results and len(results) > 0:
                    result_item = results[0]  # First result (dict subclass)
                    logger.debug("Block %d result type: %s", idx, type(result_item))
                    # Access parsing_res_list as dict key (not attribute!)
                    parsing_res = result_item.get("parsing_res_list", [])
                    logger.debug("Block %d parsing_res_list length: %d", idx, len(parsing_res))
                    for parsed_block in parsing_res:
                        # Each block is PaddleOCRVLBlock with content attribute
                        logger.debug("Block %d parsed_block type: %s", idx, type(parsed_block))
                        if hasattr(parsed_block, "content"):
                            content = parsed_block.content
                            logger.debug("Block %d content: %s", idx, repr(content))
                            if content:
                                text_parts.append(content)
                        else:
                            logger.debug("Block %d parsed_block has no content attribute", idx)

                text = "\n".join(text_parts) if text_parts else ""
                logger.debug("Block %d final text: %s", idx, repr(text))

                # Create updated block with text
                output_blocks.append(
                    Block(
                        type=block.type,
                        bbox=block.bbox,
                        text=text,
                        detection_confidence=block.detection_confidence,
                        order=block.order,
                        column_index=block.column_index,
                        corrected_text=None,
                        correction_ratio=None,
                        source=block.source or "paddleocr-vl",
                    )
                )

            except Exception as e:
                logger.warning("Error processing block %d with PaddleOCR-VL: %s", idx, e, exc_info=True)
                # Return original block if processing fails
                output_blocks.append(block)

        logger.info("Processed %d blocks with PaddleOCR-VL", len(output_blocks))
        return output_blocks

    def correct_text(self, text: str) -> str:
        """Correct text (not supported for PaddleOCR-VL).

        PaddleOCR-VL performs direct text extraction, not correction.
        This method simply returns the original text unchanged.

        Args:
            text: Raw text to correct

        Returns:
            Original text unchanged (correction not supported)
        """
        logger.debug("PaddleOCR-VL does not support text correction, returning original text")
        return text
