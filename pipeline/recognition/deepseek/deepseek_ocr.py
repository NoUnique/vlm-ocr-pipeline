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
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from ...types import Block, Recognizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def parse_deepseek_ocr_output(raw_output: str) -> str:
    """Parse DeepSeek-OCR output to extract actual text.

    DeepSeek-OCR outputs in the format:
    - Text: <|ref|>actual_text<|/ref|><|det|>[[x, y, w, h]]<|/det|>
    - Table: <|ref|>table<|/ref|><|det|>[[...]]<|/det|>\n<table>...</table>

    Args:
        raw_output: Raw output from DeepSeek-OCR model

    Returns:
        Extracted text content (or HTML table if present)
    """
    if not raw_output:
        return ""

    # Extract content between <|ref|> and <|/ref|>
    ref_match = re.search(r"<\|ref\|>(.*?)<\|/ref\|>", raw_output, re.DOTALL)
    if not ref_match:
        # No ref tags found, return raw output
        return raw_output.strip()

    ref_content = ref_match.group(1).strip()

    # Check if it's a table reference
    if ref_content.lower() == "table":
        # Look for <table>...</table> content after the tags
        table_match = re.search(r"<table>.*?</table>", raw_output, re.DOTALL)
        if table_match:
            return table_match.group(0)
        # No table HTML found, return "table" indicator
        return ref_content

    # Regular text content
    return ref_content


def _setup_gpu_for_worker(gpu_id: int) -> None:
    """Set up GPU environment for worker subprocess.

    Args:
        gpu_id: Physical GPU ID to use

    Raises:
        RuntimeError: If CUDA is not available
    """
    import os  # noqa: PLC0415

    import torch  # noqa: PLC0415

    # Set CUDA_VISIBLE_DEVICES BEFORE any torch operations
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU {gpu_id}: CUDA is not available in this subprocess")

    visible_gpus = torch.cuda.device_count()
    if visible_gpus != 1:
        print(f"WARNING: GPU {gpu_id}: Expected 1 visible GPU, but found {visible_gpus}")

    torch.cuda.set_device(0)


def _load_worker_model(model_name: str) -> tuple[Any, Any]:
    """Load model and tokenizer for worker.

    Args:
        model_name: HuggingFace model name

    Returns:
        (model, tokenizer) tuple
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    try:
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
    except (ImportError, ValueError):
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
        )

    model = model.eval().cuda().to(torch.bfloat16)
    return model, tokenizer


def _crop_block_image(image: np.ndarray, block: Block) -> np.ndarray | None:
    """Crop image to block bounding box.

    Args:
        image: Full page image
        block: Block with bbox

    Returns:
        Cropped image or None if invalid
    """
    if not block.bbox:
        return None

    x0, y0, x1, y1 = block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1
    h, w = image.shape[:2]

    # Clamp to image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    if x1 <= x0 or y1 <= y0:
        return None

    return image[y0:y1, x0:x1]


def _run_model_inference(
    model: Any,
    tokenizer: Any,
    cropped_image: np.ndarray,
    prompt_template: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
) -> str:
    """Run DeepSeek-OCR inference on cropped image.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        cropped_image: Cropped block image
        prompt_template: Prompt for OCR
        base_size: Base resolution
        image_size: Image size
        crop_mode: Dynamic resolution mode

    Returns:
        Extracted text
    """
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    cropped_pil = Image.fromarray(cropped_image)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        cropped_pil.save(tmp_path)

    tmp_output_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

    try:
        result = model.infer(
            tokenizer,
            prompt=prompt_template,
            image_file=tmp_path,
            output_path=tmp_output_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=False,
            save_results=True,
        )

        # Read output from result.mmd file
        mmd_file = os.path.join(tmp_output_dir, "result.mmd")
        if os.path.exists(mmd_file):
            with open(mmd_file, encoding="utf-8") as f:
                raw_output = f.read()
        else:
            raw_output = str(result) if result else ""

        return parse_deepseek_ocr_output(raw_output)

    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_output_dir, ignore_errors=True)
        except Exception:
            pass


def _process_on_gpu_worker(
    gpu_id: int,
    device: str,
    assigned_blocks: list[tuple[int, Block]],
    image: np.ndarray,
    model_name: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    prompt_template: str,
) -> list[tuple[int, Block]]:
    """Worker function to process blocks on a specific GPU.

    Args:
        gpu_id: GPU ID for logging
        device: Device string (e.g., "cuda:0")
        assigned_blocks: List of (index, block) tuples to process
        image: Full page image
        model_name: DeepSeek-OCR model name
        base_size: Base resolution
        image_size: Image size for cropping
        crop_mode: Whether to use dynamic resolution
        prompt_template: Prompt template for OCR

    Returns:
        List of (index, processed_block) tuples
    """
    # Set up GPU environment
    _setup_gpu_for_worker(gpu_id)

    # Load model
    model, tokenizer = _load_worker_model(model_name)

    results = []

    for idx, block in assigned_blocks:
        # Crop image
        cropped = _crop_block_image(image, block)
        if cropped is None:
            results.append((idx, block))
            continue

        try:
            # Run inference
            text = _run_model_inference(
                model, tokenizer, cropped, prompt_template, base_size, image_size, crop_mode
            )

            # Create updated block
            processed_block = Block(
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
            results.append((idx, processed_block))

        except Exception as e:
            logger.warning("[%s] Error processing block %d: %s", device, idx, e)
            results.append((idx, block))

    return results


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
        use_multi_gpu: bool = True,
        batch_size: int | None = None,
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
                Ignored if use_multi_gpu=True
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
            use_multi_gpu: Whether to use multiple GPUs for parallel processing
                True: Replicate model across all available GPUs (default)
                False: Use single GPU specified by device
            batch_size: Number of blocks to process in parallel per GPU
                None: Auto-detect based on GPU memory (default)
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
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size

        # Filter out kwargs that shouldn't be passed to AutoModel.from_pretrained()
        # Keep only transformers-related kwargs
        excluded_keys = {
            "backend",
            "model",
            "model_name",
            "device",
            "base_size",
            "image_size",
            "crop_mode",
            "prompt_template",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "use_bf16",
            "api_key",
            "gemini_tier",
            "rate_limit",
            "use_async",
            "use_multi_gpu",
            "batch_size",
        }
        self.kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}

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

        gpu_count = torch.cuda.device_count()

        if self.use_multi_gpu and gpu_count > 1:
            logger.info("Multi-GPU mode enabled: will use %d GPUs with multiprocessing", gpu_count)
            # Store GPU count for parallel processing
            # Models will be loaded separately in each process to avoid CUDA context issues
            self.gpu_count = gpu_count
            self.gpu_devices = [f"cuda:{i}" for i in range(gpu_count)]
            logger.info("Multi-GPU initialization complete (lazy loading per process)")

        else:
            # Single GPU mode
            if self.use_multi_gpu and gpu_count <= 1:
                logger.warning("Multi-GPU requested but only %d GPU available, using single GPU mode", gpu_count)

            logger.info("Single GPU mode: loading model on %s", self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # Try flash_attention_2, fallback to eager
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    _attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                    use_safetensors=True,
                    **self.kwargs,
                )
                self.model = self.model.eval().to(self.device).to(torch.bfloat16)
                logger.info("Using Flash Attention 2 on %s", self.device)
            except (ImportError, ValueError):
                logger.warning("Flash Attention 2 not available, using eager attention")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    _attn_implementation="eager",
                    trust_remote_code=True,
                    use_safetensors=True,
                    **self.kwargs,
                )
                self.model = self.model.eval().to(self.device).to(torch.bfloat16)
                logger.info("Using eager attention on %s", self.device)

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
        logger.info("Number of blocks: %d", len(blocks) if blocks is not None else 0)
        logger.info("Backend: %s", self.backend)

        if blocks is None or len(blocks) == 0:
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
        if hasattr(self, "gpu_devices"):
            # Multi-GPU parallel processing with multiprocessing
            return self._process_blocks_parallel(image, blocks)
        else:
            # Single GPU sequential processing
            return self._process_blocks_sequential(image, blocks)

    def _process_blocks_parallel(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        """Process blocks in parallel across multiple GPUs using multiprocessing.

        Note: Uses ProcessPoolExecutor with 'spawn' method to avoid
        CUDA context issues with DeepSeek-OCR's infer() method.
        """
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

        # Set start method to 'spawn' for CUDA compatibility
        ctx = mp.get_context("spawn")

        gpu_devices = self.gpu_devices
        num_gpus = len(gpu_devices)

        logger.info("Processing %d blocks across %d GPUs in parallel (multiprocessing)", len(blocks), num_gpus)

        # Assign blocks to GPUs (round-robin) with original indices
        blocks_with_indices = [(idx, block) for idx, block in enumerate(blocks)]
        gpu_assignments = [[] for _ in range(num_gpus)]
        for i, (idx, block) in enumerate(blocks_with_indices):
            gpu_id = i % num_gpus
            gpu_assignments[gpu_id].append((idx, block))

        # Prepare arguments for each process
        process_args = []
        for gpu_id, device in enumerate(gpu_devices):
            assigned = gpu_assignments[gpu_id]
            if assigned:  # Only submit if there are blocks for this GPU
                process_args.append(
                    (
                        gpu_id,
                        device,
                        assigned,
                        image,
                        self.model_name,
                        self.base_size,
                        self.image_size,
                        self.crop_mode,
                        self.prompt_template,
                    )
                )

        # Launch parallel processing with separate processes using spawn context
        all_results = []
        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            futures = {}
            for args in process_args:
                future = executor.submit(_process_on_gpu_worker, *args)
                futures[future] = args[0]  # gpu_id

            # Collect results
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info("GPU %d completed processing %d blocks", gpu_id, len(results))
                except Exception as e:
                    logger.error("Error on GPU %d: %s", gpu_id, e, exc_info=True)

        # Sort by original index to maintain order
        all_results.sort(key=lambda x: x[0])
        output_blocks = [block for _, block in all_results]

        logger.info("Processed %d blocks with DeepSeek-OCR (HF, multi-GPU)", len(output_blocks))
        return output_blocks

    def _process_single_block_hf(
        self, model, tokenizer, device: str, block: Block, image: np.ndarray, idx: int
    ) -> Block:
        """Process a single block on a specific GPU."""
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        logger.info("[%s] Processing block %d (type=%s)", device, idx, block.type)

        # Skip blocks without valid bbox
        if not block.bbox:
            logger.warning("[%s] Block %d without bbox, skipping", device, idx)
            return block

        # Crop image to block bbox
        x0, y0, x1, y1 = block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        # Check for valid crop dimensions
        if x1 <= x0 or y1 <= y0:
            logger.warning("[%s] Block %d has invalid dimensions: (%d,%d,%d,%d)", device, idx, x0, y0, x1, y1)
            return block

        # Crop the block region
        cropped = image[y0:y1, x0:x1]

        # Convert to PIL Image
        cropped_pil = Image.fromarray(cropped)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cropped_pil.save(tmp_path)

        # Create temporary output directory
        tmp_output_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

        try:
            # Call model.infer()
            result = model.infer(
                tokenizer,
                prompt=self.prompt_template,
                image_file=tmp_path,
                output_path=tmp_output_dir,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                test_compress=False,
                save_results=False,
            )

            # Extract text from result
            raw_output = str(result) if result else ""
            text = parse_deepseek_ocr_output(raw_output)

            # Create updated block with text
            return Block(
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

        except Exception as e:
            logger.warning("[%s] Error processing block %d: %s", device, idx, e, exc_info=True)
            return block
        finally:
            # Clean up temporary files
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            try:
                import shutil  # noqa: PLC0415

                shutil.rmtree(tmp_output_dir, ignore_errors=True)
            except Exception:
                pass

    def _process_blocks_sequential(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        """Process blocks sequentially on a single GPU."""
        output_blocks = []
        for idx, block in enumerate(blocks):
            logger.info("--- Processing block %d/%d (type=%s) ---", idx + 1, len(blocks), block.type)
            processed_block = self._process_single_block_hf(self.model, self.tokenizer, self.device, block, image, idx)
            output_blocks.append(processed_block)

        logger.info("Processed %d blocks with DeepSeek-OCR (HF, single GPU)", len(output_blocks))
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

                # Parse DeepSeek-OCR output format
                text = parse_deepseek_ocr_output(final_output)
                logger.info("Block %d raw output: %s", idx, repr(final_output[:200]))
                logger.info("Block %d extracted text: %s", idx, repr(text[:200] if len(text) > 200 else text))

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
