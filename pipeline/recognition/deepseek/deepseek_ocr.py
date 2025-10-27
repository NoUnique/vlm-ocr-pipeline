"""DeepSeek-OCR text recognition implementation.

DeepSeek-OCR is a Vision Language Model designed for document understanding
from an LLM-centric viewpoint with contextual optical compression.

Key Features:
- Model: deepseek-ai/DeepSeek-OCR
- Multiple resolution modes: Tiny (512x512), Small (640x640), Base (1024x1024),
  Large (1280x1280), Gundam (dynamic resolution)
- Two inference backends: HuggingFace Transformers and vLLM
- Native markdown conversion and grounding support

Usage:
    >>> from pipeline.recognition.deepseek import DeepSeekOCRRecognizer
    >>> # HuggingFace backend
    >>> recognizer = DeepSeekOCRRecognizer(backend="hf")
    >>> blocks_with_text = recognizer.process_blocks(image, blocks)
    >>> # vLLM backend
    >>> recognizer = DeepSeekOCRRecognizer(backend="vllm")
    >>> blocks_with_text = recognizer.process_blocks(image, blocks)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from ...types import Block, Recognizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DeepSeekOCRRecognizer(Recognizer):
    """Text recognizer using DeepSeek-OCR model.

    This class implements the Recognizer protocol using DeepSeek-OCR which supports:
    - HuggingFace Transformers backend (flash_attention_2, trust_remote_code)
    - vLLM backend (AsyncLLMEngine with NGramLogitsProcessor)

    Note: Text correction is not supported by this recognizer.

    Implements:
        Recognizer: Text recognition protocol with DeepSeek-OCR backend

    Example:
        >>> # HuggingFace backend
        >>> recognizer = DeepSeekOCRRecognizer(backend="hf", base_size=1024, image_size=640)
        >>> blocks_with_text = recognizer.process_blocks(image, blocks)

        >>> # vLLM backend
        >>> recognizer = DeepSeekOCRRecognizer(backend="vllm")
        >>> blocks_with_text = recognizer.process_blocks(image, blocks)
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        backend: str = "hf",
        device: str | None = None,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        prompt_template: str = "<image>\\n<|grounding|>Convert the document to markdown. ",
        **kwargs: Any,
    ):
        """Initialize DeepSeek-OCR recognizer.

        Args:
            model_name: Model name/path (default: "deepseek-ai/DeepSeek-OCR")
            backend: Inference backend ("hf" or "vllm")
                - "hf": HuggingFace Transformers (flash_attention_2)
                - "vllm": vLLM AsyncLLMEngine (high-throughput)
            device: Device to run model on ("cuda:0", "cpu", etc.)
                Default: CUDA 0 if available, otherwise CPU
            base_size: Base resolution for image processing
                - 512: Tiny (64 tokens)
                - 640: Small (100 tokens)
                - 1024: Base (256 tokens)
                - 1280: Large (400 tokens)
            image_size: Image size for cropping (default: 640)
            crop_mode: Whether to use dynamic resolution (Gundam mode)
                True: n×640×640 + 1×1024×1024 (default)
                False: Single resolution mode
            prompt_template: Prompt template for OCR
                Default: "<image>\\n<|grounding|>Convert the document to markdown. "
                Alternatives:
                - "<image>\\n<|grounding|>OCR this image. "
                - "<image>\\nFree OCR. "
            **kwargs: Additional arguments passed to model initialization

        Example:
            >>> # Base mode (1024x1024, 256 tokens)
            >>> recognizer = DeepSeekOCRRecognizer(
            ...     backend="hf",
            ...     base_size=1024,
            ...     image_size=1024,
            ...     crop_mode=False
            ... )

            >>> # Gundam mode (dynamic resolution)
            >>> recognizer = DeepSeekOCRRecognizer(
            ...     backend="hf",
            ...     base_size=1024,
            ...     image_size=640,
            ...     crop_mode=True
            ... )

            >>> # vLLM backend
            >>> recognizer = DeepSeekOCRRecognizer(backend="vllm")
        """
        self.model_name = model_name
        self.backend = backend.lower()
        self.device = device or "cuda:0"
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.prompt_template = prompt_template
        self.kwargs = kwargs

        # Initialize backend-specific components
        if self.backend == "hf":
            self._init_hf_backend()
        elif self.backend == "vllm":
            self._init_vllm_backend()
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'hf', 'vllm'")

        logger.info(
            "DeepSeek-OCR recognizer initialized (model: %s, backend: %s, device: %s, "
            "base_size: %d, image_size: %d, crop_mode: %s)",
            model_name,
            self.backend,
            self.device,
            base_size,
            image_size,
            crop_mode,
        )

    def _init_hf_backend(self) -> None:
        """Initialize HuggingFace Transformers backend."""
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for HuggingFace backend. "
                "Please install them: pip install transformers torch"
            ) from e

        logger.info("Initializing HuggingFace backend for DeepSeek-OCR...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
            **self.kwargs,
        )
        self.model = self.model.eval().to(self.device).to(torch.bfloat16)
        logger.info("HuggingFace backend initialized successfully")

    def _init_vllm_backend(self) -> None:
        """Initialize vLLM backend."""
        try:
            from vllm import AsyncLLMEngine, SamplingParams  # noqa: PLC0415  # type: ignore[import-untyped]
            from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: PLC0415  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("vllm is required for vLLM backend. Please install it: pip install vllm") from e

        # Import custom DeepSeek-OCR components from external submodule
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        deepseek_vllm_path = Path("external/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        if deepseek_vllm_path.exists():
            sys.path.insert(0, str(deepseek_vllm_path))
        else:
            raise FileNotFoundError(
                f"DeepSeek-OCR-vllm directory not found at {deepseek_vllm_path}. "
                "Please ensure the DeepSeek-OCR submodule is initialized."
            )

        try:
            from deepseek_ocr import DeepseekOCRForCausalLM  # noqa: PLC0415  # type: ignore[import-not-found]
            from process.image_process import DeepseekOCRProcessor  # noqa: PLC0415  # type: ignore[import-not-found]
            from process.ngram_norepeat import (  # noqa: PLC0415
                NoRepeatNGramLogitsProcessor,  # type: ignore[import-not-found]
            )
            from vllm.model_executor.models.registry import (  # noqa: PLC0415
                ModelRegistry,  # type: ignore[import-untyped]
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import DeepSeek-OCR vLLM components. "
                "Please ensure external/DeepSeek-OCR submodule is properly initialized."
            ) from e

        logger.info("Initializing vLLM backend for DeepSeek-OCR...")

        # Register custom model
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

        # Store classes for later use
        self.SamplingParams = SamplingParams
        self.AsyncEngineArgs = AsyncEngineArgs
        self.AsyncLLMEngine = AsyncLLMEngine
        self.NoRepeatNGramLogitsProcessor = NoRepeatNGramLogitsProcessor
        self.DeepseekOCRProcessor = DeepseekOCRProcessor

        # Engine will be created lazily on first use
        self.engine = None
        logger.info("vLLM backend initialized successfully (engine will be created on first use)")

    async def _get_or_create_vllm_engine(self):
        """Get or create vLLM engine (lazy initialization)."""
        if self.engine is None:
            engine_args = self.AsyncEngineArgs(
                model=self.model_name,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                max_model_len=8192,
                enforce_eager=False,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,
                **self.kwargs,
            )
            self.engine = self.AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine created")
        return self.engine

    def process_blocks(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
        **kwargs: Any,
    ) -> list[Block]:
        """Process blocks and extract text using DeepSeek-OCR.

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
        logger.info("=== DeepSeekOCRRecognizer.process_blocks START ===")
        logger.info("Number of blocks: %d", len(blocks) if blocks else 0)
        logger.info("Backend: %s", self.backend)

        if not blocks:
            logger.warning("No blocks provided for recognition")
            return []

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            logger.info("Converting PIL Image to numpy array")
            image = np.array(image)

        logger.info("Image shape: %s", image.shape)

        if self.backend == "hf":
            return self._process_blocks_hf(image, blocks)
        elif self.backend == "vllm":
            # vLLM requires async, use asyncio.run
            import asyncio  # noqa: PLC0415

            return asyncio.run(self._process_blocks_vllm(image, blocks))
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _process_blocks_hf(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        """Process blocks using HuggingFace backend."""
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

            # Convert to PIL Image
            cropped_pil = Image.fromarray(cropped)

            # Save to temporary file for model.infer()
            import tempfile  # noqa: PLC0415

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cropped_pil.save(tmp_path)

            try:
                logger.info("Calling DeepSeek-OCR model.infer() for block %d...", idx)
                # Call model.infer() with proper parameters
                result = self.model.infer(
                    self.tokenizer,
                    prompt=self.prompt_template,
                    image_file=tmp_path,
                    output_path=None,  # Don't save intermediate results
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=self.crop_mode,
                    test_compress=False,
                    save_results=False,
                )
                logger.info("DeepSeek-OCR model.infer() completed for block %d", idx)

                # Extract text from result
                text = str(result) if result else ""
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
                        source=block.source or "deepseek-ocr",
                    )
                )

            except Exception as e:
                logger.warning("Error processing block %d with DeepSeek-OCR: %s", idx, e, exc_info=True)
                # Return original block if processing fails
                output_blocks.append(block)
            finally:
                # Clean up temporary file
                import os  # noqa: PLC0415

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        logger.info("Processed %d blocks with DeepSeek-OCR (HF)", len(output_blocks))
        return output_blocks

    async def _process_blocks_vllm(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        """Process blocks using vLLM backend (async)."""
        import time  # noqa: PLC0415

        engine = await self._get_or_create_vllm_engine()

        # Process each block individually
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

            # Crop the block region and convert to PIL
            cropped = image[y0:y1, x0:x1]
            cropped_pil = Image.fromarray(cropped)
            logger.info("Cropped block %d to shape: %s", idx, cropped.shape)

            try:
                logger.info("Processing block %d with vLLM...", idx)

                # Prepare image features
                processor = self.DeepseekOCRProcessor()
                image_features = processor.tokenize_with_images(
                    images=[cropped_pil], bos=True, eos=True, cropping=self.crop_mode
                )

                # Prepare logits processors
                logits_processors = [
                    self.NoRepeatNGramLogitsProcessor(
                        ngram_size=30,
                        window_size=90,
                        whitelist_token_ids={128821, 128822},  # <td>, </td>
                    )
                ]

                # Prepare sampling params
                sampling_params = self.SamplingParams(
                    temperature=0.0,
                    max_tokens=8192,
                    logits_processors=logits_processors,
                    skip_special_tokens=False,
                )

                # Prepare request
                request_id = f"block-{idx}-{int(time.time())}"
                request = {"prompt": self.prompt_template, "multi_modal_data": {"image": image_features}}

                # Generate text
                final_output = ""
                async for request_output in engine.generate(request, sampling_params, request_id):
                    if request_output.outputs:
                        final_output = request_output.outputs[0].text

                logger.info("vLLM processing completed for block %d", idx)
                logger.debug("Block %d final text: %s", idx, repr(final_output))

                # Create updated block with text
                output_blocks.append(
                    Block(
                        type=block.type,
                        bbox=block.bbox,
                        text=final_output,
                        detection_confidence=block.detection_confidence,
                        order=block.order,
                        column_index=block.column_index,
                        corrected_text=None,
                        correction_ratio=None,
                        source=block.source or "deepseek-ocr",
                    )
                )

            except Exception as e:
                logger.warning("Error processing block %d with DeepSeek-OCR (vLLM): %s", idx, e, exc_info=True)
                # Return original block if processing fails
                output_blocks.append(block)

        logger.info("Processed %d blocks with DeepSeek-OCR (vLLM)", len(output_blocks))
        return output_blocks

    def correct_text(self, text: str) -> str:
        """Correct text (not supported for DeepSeek-OCR).

        DeepSeek-OCR performs direct text extraction, not correction.
        This method simply returns the original text unchanged.

        Args:
            text: Raw text to correct

        Returns:
            Original text unchanged (correction not supported)
        """
        logger.debug("DeepSeek-OCR does not support text correction, returning original text")
        return text
