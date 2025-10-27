"""Text recognition module for OCR and content extraction."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..prompt import PromptManager
from ..resources import managed_numpy_array
from ..types import Block, Recognizer
from .api.gemini import GeminiClient
from .api.gemini_async import AsyncGeminiClient
from .api.openai import OpenAIClient
from .api.openai_async import AsyncOpenAIClient
from .cache import RecognitionCache

logger = logging.getLogger(__name__)

# Lazy import for PaddleOCR-VL (optional dependency)
try:
    from .paddleocr import PaddleOCRVLRecognizer  # noqa: PLC0415

    _HAS_PADDLEOCR_VL = True
except ImportError:
    PaddleOCRVLRecognizer = None  # type: ignore[misc, assignment]
    _HAS_PADDLEOCR_VL = False

# Lazy import for DeepSeek-OCR (optional dependency)
try:
    from .deepseek import DeepSeekOCRRecognizer  # noqa: PLC0415

    _HAS_DEEPSEEK_OCR = True
except ImportError:
    DeepSeekOCRRecognizer = None  # type: ignore[misc, assignment]
    _HAS_DEEPSEEK_OCR = False

__all__ = [
    "TextRecognizer",
    "PaddleOCRVLRecognizer",
    "DeepSeekOCRRecognizer",
    "create_recognizer",
    "list_available_recognizers",
]


class TextRecognizer(Recognizer):
    """Handles text recognition and extraction from document blocks.

    This class implements the Recognizer protocol and manages the text extraction
    pipeline using VLM APIs (OpenAI/OpenRouter or Gemini) with intelligent caching.

    Implements:
        Recognizer: Text recognition protocol with VLM backends (OpenAI/Gemini)
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        use_cache: bool = True,
        backend: str = "openai",
        model: str = "gemini-2.5-flash",
        gemini_tier: str = "free",
        use_async: bool = False,
    ):
        """Initialize text recognizer.

        Args:
            cache_dir: Directory for caching recognition results
            use_cache: Whether to use caching
            backend: Backend API ("openai" or "gemini")
            model: Model to use for text processing
            gemini_tier: Gemini API tier (only used with gemini backend)
            use_async: Whether to use async API clients for concurrent processing
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.backend = backend.lower()
        self.model = model
        self.gemini_tier = gemini_tier
        self.use_async = use_async

        # Initialize cache
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = RecognitionCache(cache_dir=self.cache_dir)
        else:
            self.cache = None

        # Initialize VLM clients (both sync and async if requested)
        if self.backend == "gemini":
            self.client = GeminiClient(gemini_model=model)
            self.async_client = AsyncGeminiClient(gemini_model=model) if use_async else None
        elif self.backend == "openai":
            self.client = OpenAIClient(model=model)
            self.async_client = AsyncOpenAIClient(model=model) if use_async else None
        elif self.backend == "paddleocr-vl":
            # PaddleOCR-VL is handled separately, no client needed
            self.client = None  # type: ignore[assignment]
            self.async_client = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Initialize prompt manager
        self.prompt_manager = PromptManager(model=model)

        logger.info(
            "TextRecognizer initialized: backend=%s, model=%s, cache=%s, async=%s",
            self.backend,
            model,
            use_cache,
            use_async,
        )

    def process_blocks(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        """Process all blocks to extract text.

        Args:
            image: Full page image
            blocks: List of detected blocks

        Returns:
            Blocks with text extracted
        """
        processed_blocks = []

        for block in blocks:
            processed_block = self._process_single_block(image, block)
            processed_blocks.append(processed_block)

        return processed_blocks

    def _process_single_block(self, image: np.ndarray, block: Block) -> Block:
        """Process a single block to extract text.

        IMPORTANT: This recognizer will ALWAYS extract text using the configured VLM backend,
        regardless of whether the detector already provided text content. This ensures that
        when a user selects a specific recognizer (e.g., Gemini), that recognizer is used
        for text extraction, not the detector's built-in OCR.

        Args:
            image: Full page image
            block: Block instance

        Returns:
            Block with text extracted
        """
        block_type = block.type

        # Clear any pre-existing text from detector
        # This ensures we use the recognizer's extraction, not the detector's content
        if block.text:
            logger.debug(
                "Clearing detector-provided text for block (source=%s). Will use recognizer instead.", block.source
            )
            block.text = None

        # Extract block image
        try:
            block_image = self._crop_block(image, block)
        except (IndexError, ValueError, TypeError) as e:
            logger.error("Failed to crop block: %s", e)
            # Note: error field not in Block dataclass, just log and continue
            return block

        # Get prompt for block type
        prompt = self._get_prompt_for_block_type(block_type)

        # Extract text with managed memory
        with managed_numpy_array(block_image) as (managed_image,):
            try:
                if self.client is None:
                    raise RuntimeError(
                        f"API client not initialized for backend '{self.backend}'. "
                        "Cannot extract text without a configured client."
                    )
                result = self.client.extract_text(managed_image, block.to_dict(), prompt)
                block.text = result.get("text", "") if isinstance(result, dict) else str(result)
            except Exception as e:
                # Fallback for unexpected errors - set empty text to continue processing
                # (allowed per ERROR_HANDLING.md section 3.3)
                logger.error("Failed to extract text: %s", e, exc_info=True)
                block.text = ""

        return block

    def _crop_block(self, image: np.ndarray, block: Block) -> np.ndarray:
        """Crop a block from the full image.

        Args:
            image: Full page image
            block: Block with bbox

        Returns:
            Cropped block image
        """
        # Use BBox.crop() method with padding
        return block.bbox.crop(image, padding=5)

    def _get_prompt_for_block_type(self, block_type: str) -> str:
        """Get prompt for specific block type.

        Args:
            block_type: Type of block

        Returns:
            Prompt string for the block type
        """
        if block_type in ["plain text", "title", "list"]:
            return self.prompt_manager.get_prompt("text_extraction", "user")
        elif block_type == "table":
            return self.prompt_manager.get_prompt("content_analysis", "table_analysis", "user")
        elif block_type in ["figure", "equation"]:
            return self.prompt_manager.get_prompt("content_analysis", "figure_analysis", "user")
        else:
            return self.prompt_manager.get_prompt("text_extraction", "user")

    def correct_text(self, text: str) -> str | dict[str, Any]:
        """Correct extracted text using VLM.

        Args:
            text: Raw text to correct

        Returns:
            Corrected text or dict with error info
        """
        if not text or not text.strip():
            return text

        try:
            if self.client is None:
                raise RuntimeError(
                    f"API client not initialized for backend '{self.backend}'. "
                    "Cannot correct text without a configured client."
                )
            system_prompt = self.prompt_manager.get_prompt("text_correction", "system")
            user_prompt = self.prompt_manager.get_prompt("text_correction", "user", text=text)
            corrected = self.client.correct_text(text, system_prompt, user_prompt)
            return corrected
        except Exception as e:
            # Fallback for unexpected errors - return error with original text
            # (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Text correction failed: %s", e, exc_info=True)
            return {"error": "correction_failed", "original_text": text}

    async def process_blocks_async(
        self, image: np.ndarray, blocks: Sequence[Block], max_concurrent: int = 5
    ) -> list[Block]:
        """Process all blocks to extract text concurrently (async).

        This method processes multiple blocks in parallel using async API clients,
        significantly improving performance by reducing overall processing time.

        Args:
            image: Full page image
            blocks: List of detected blocks
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            Blocks with text extracted

        Raises:
            RuntimeError: If async client is not initialized
        """
        if self.async_client is None:
            raise RuntimeError("Async client not initialized. Please set use_async=True when creating TextRecognizer.")

        # Prepare batch data: (image, region_info, prompt) tuples
        batch_data = []
        for block in blocks:
            # Clear any pre-existing text from detector
            if block.text:
                logger.debug(
                    "Clearing detector-provided text for block (source=%s). Will use recognizer instead.",
                    block.source,
                )
                block.text = None

            # Extract block image
            try:
                block_image = self._crop_block(image, block)
                prompt = self._get_prompt_for_block_type(block.type)
                batch_data.append((block_image, block.to_dict(), prompt))
            except (IndexError, ValueError, TypeError) as e:
                logger.error("Failed to crop block: %s", e)
                # Skip this block in batch processing
                continue

        # Process all blocks concurrently
        results = await self.async_client.extract_text_batch(batch_data, max_concurrent=max_concurrent)

        # Update blocks with extracted text
        processed_blocks = []
        result_idx = 0
        for block in blocks:
            if block.text is None:  # Block was included in batch
                if result_idx < len(results):
                    result = results[result_idx]
                    block.text = result.get("text", "") if isinstance(result, dict) else str(result)
                    result_idx += 1
                else:
                    block.text = ""  # Fallback
            processed_blocks.append(block)

        return processed_blocks

    async def correct_text_async(self, text: str) -> str | dict[str, Any]:
        """Correct extracted text using VLM (async).

        Args:
            text: Raw text to correct

        Returns:
            Corrected text or dict with error info

        Raises:
            RuntimeError: If async client is not initialized
        """
        if not text or not text.strip():
            return text

        if self.async_client is None:
            raise RuntimeError("Async client not initialized. Please set use_async=True when creating TextRecognizer.")

        try:
            system_prompt = self.prompt_manager.get_prompt("text_correction", "system")
            user_prompt = self.prompt_manager.get_prompt("text_correction", "user", text=text)
            corrected = await self.async_client.correct_text(text, system_prompt, user_prompt)
            return corrected
        except Exception as e:
            # Fallback for unexpected errors - return error with original text
            # (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Async text correction failed: %s", e, exc_info=True)
            return {"error": "correction_failed", "original_text": text}


# ==================== Factory Pattern ====================

# Registry mapping recognizer names to factory functions
_RECOGNIZER_REGISTRY: dict[str, Callable[..., Recognizer]] = {}


def _register_recognizers() -> None:
    """Register available recognizers."""

    # Always available: TextRecognizer with OpenAI/Gemini backends
    def create_openai_recognizer(**kwargs: Any) -> Recognizer:
        """Create OpenAI TextRecognizer."""
        kwargs.setdefault("backend", "openai")
        return TextRecognizer(**kwargs)

    def create_gemini_recognizer(**kwargs: Any) -> Recognizer:
        """Create Gemini TextRecognizer."""
        kwargs.setdefault("backend", "gemini")
        return TextRecognizer(**kwargs)

    _RECOGNIZER_REGISTRY["openai"] = create_openai_recognizer
    _RECOGNIZER_REGISTRY["gemini"] = create_gemini_recognizer

    # Optional: PaddleOCR-VL
    if _HAS_PADDLEOCR_VL and PaddleOCRVLRecognizer is not None:
        _RECOGNIZER_REGISTRY["paddleocr-vl"] = PaddleOCRVLRecognizer

    # Optional: DeepSeek-OCR
    if _HAS_DEEPSEEK_OCR and DeepSeekOCRRecognizer is not None:
        _RECOGNIZER_REGISTRY["deepseek-ocr"] = DeepSeekOCRRecognizer


# Register on module import
_register_recognizers()


def create_recognizer(name: str, **kwargs: Any) -> Recognizer:
    """Create a recognizer instance using factory pattern.

    Args:
        name: Recognizer name ("openai", "gemini", "paddleocr-vl")
        **kwargs: Additional arguments passed to the recognizer constructor
            Common args:
                - cache_dir: Cache directory (default: ".cache")
                - use_cache: Enable caching (default: True)
                - model: Model name (backend-specific)
                - gemini_tier: Gemini API tier ("free", "tier1", etc.)

    Returns:
        Recognizer instance

    Raises:
        ValueError: If recognizer name is not registered

    Example:
        >>> # Create OpenAI recognizer
        >>> recognizer = create_recognizer("openai", model="gpt-4o")

        >>> # Create Gemini recognizer
        >>> recognizer = create_recognizer("gemini", model="gemini-2.5-flash")

        >>> # Create PaddleOCR-VL recognizer (if available)
        >>> recognizer = create_recognizer("paddleocr-vl")
    """
    if name not in _RECOGNIZER_REGISTRY:
        available = ", ".join(_RECOGNIZER_REGISTRY.keys())
        raise ValueError(f"Unknown recognizer: {name}. Available: {available}")

    logger.info("Creating recognizer: %s", name)
    return _RECOGNIZER_REGISTRY[name](**kwargs)


def list_available_recognizers() -> list[str]:
    """List available recognizer names.

    Returns:
        List of registered recognizer names

    Example:
        >>> recognizers = list_available_recognizers()
        >>> print(recognizers)
        ['openai', 'gemini', 'paddleocr-vl']
    """
    return list(_RECOGNIZER_REGISTRY.keys())
