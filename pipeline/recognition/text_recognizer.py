"""VLM-based text recognizer for OCR and content extraction.

This module provides the TextRecognizer class that implements the Recognizer
protocol using VLM APIs (OpenAI/Gemini) for text extraction and correction.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.exceptions import APIClientError, InvalidConfigError
from pipeline.prompt import PromptManager
from pipeline.resources import managed_numpy_array
from pipeline.types import Block, Recognizer

from .api.gemini import GeminiClient
from .api.gemini_async import AsyncGeminiClient
from .api.openai import OpenAIClient
from .api.openai_async import AsyncOpenAIClient
from .cache import RecognitionCache

if TYPE_CHECKING:
    from .api.gemini import GeminiClient as GeminiClientType
    from .api.gemini_async import AsyncGeminiClient as AsyncGeminiClientType
    from .api.openai import OpenAIClient as OpenAIClientType
    from .api.openai_async import AsyncOpenAIClient as AsyncOpenAIClientType

logger = logging.getLogger(__name__)

__all__ = ["TextRecognizer"]


class TextRecognizer(Recognizer):
    """Handles text recognition and extraction from document blocks.

    This class implements the Recognizer protocol and manages the text extraction
    pipeline using VLM APIs (OpenAI/OpenRouter or Gemini) with intelligent caching.

    Attributes:
        name: Recognizer identifier
        supports_correction: Whether this recognizer supports text correction

    Example:
        >>> recognizer = TextRecognizer(backend="gemini", model="gemini-2.5-flash")
        >>> blocks_with_text = recognizer.process_blocks(image, blocks)
    """

    # Protocol attributes
    name: str = "text-recognizer"
    supports_correction: bool = True

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
        self.cache: RecognitionCache | None = None
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = RecognitionCache(cache_dir=self.cache_dir)

        # Initialize VLM clients
        self.client: GeminiClientType | OpenAIClientType | None = None
        self.async_client: AsyncGeminiClientType | AsyncOpenAIClientType | None = None
        self._initialize_clients(backend, model, use_async)

        # Initialize prompt manager
        self.prompt_manager = PromptManager(model=model)

        logger.info(
            "TextRecognizer initialized: backend=%s, model=%s, cache=%s, async=%s",
            self.backend,
            model,
            use_cache,
            use_async,
        )

    def _initialize_clients(self, backend: str, model: str, use_async: bool) -> None:
        """Initialize VLM API clients based on backend.

        Args:
            backend: Backend API name
            model: Model name to use
            use_async: Whether to initialize async client

        Raises:
            ValueError: If backend is unknown
        """
        if backend == "gemini":
            self.client = GeminiClient(gemini_model=model)
            self.async_client = AsyncGeminiClient(gemini_model=model) if use_async else None
        elif backend == "openai":
            self.client = OpenAIClient(model=model)
            self.async_client = AsyncOpenAIClient(model=model) if use_async else None
        elif backend == "paddleocr-vl":
            # PaddleOCR-VL is handled by a separate recognizer class
            self.client = None
            self.async_client = None
        else:
            raise InvalidConfigError(f"Unknown backend: {backend}")

    def process_blocks(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        """Process all blocks to extract text.

        Args:
            image: Full page image as numpy array (H, W, C)
            blocks: List of detected blocks with bounding boxes

        Returns:
            List of blocks with text field populated
        """
        processed_blocks = []

        for block in blocks:
            processed_block = self._process_single_block(image, block)
            processed_blocks.append(processed_block)

        return processed_blocks

    def _process_single_block(
        self,
        image: np.ndarray,
        block: Block,
        generate_figure_description: bool = True,
    ) -> Block:
        """Process a single block to extract text or description.

        IMPORTANT: This recognizer will ALWAYS extract text using the configured VLM backend,
        regardless of whether the detector already provided text content. This ensures that
        when a user selects a specific recognizer (e.g., Gemini), that recognizer is used
        for text extraction, not the detector's built-in OCR.

        For image/figure/chart blocks, generates a description instead of text extraction.

        Args:
            image: Full page image
            block: Block instance with bounding box
            generate_figure_description: Whether to generate descriptions for image blocks

        Returns:
            Block with text field populated (and description for image blocks)
        """
        block_type = block.type.lower()

        # Clear any pre-existing text from detector
        # This ensures we use the recognizer's extraction, not the detector's content
        if block.text:
            logger.debug(
                "Clearing detector-provided text for block (source=%s). Will use recognizer instead.",
                block.source,
            )
            block.text = None

        # Extract block image
        try:
            block_image = self._crop_block(image, block)
        except (IndexError, ValueError, TypeError) as e:
            logger.error("Failed to crop block: %s", e)
            return block

        # Check if this is an image/figure/chart block
        is_image_block = block_type in {
            "image", "image_body", "figure", "chart",
        }

        # Get prompt for block type
        prompt = self._get_prompt_for_block_type(block.type)

        # For image blocks, generate description if enabled
        if is_image_block and generate_figure_description:
            prompt = self._get_figure_description_prompt()

        # Extract text with managed memory
        with managed_numpy_array(block_image) as (managed_image,):
            try:
                if self.client is None:
                    raise APIClientError(
                        f"API client not initialized for backend '{self.backend}'. "
                        "Cannot extract text without a configured client."
                    )
                result = self.client.extract_text(managed_image, block.to_dict(), prompt)
                extracted_text = result.get("text", "") if isinstance(result, dict) else str(result)

                # For image blocks, store in description field
                if is_image_block and generate_figure_description:
                    block.description = extracted_text
                    block.text = ""  # Set empty text for image blocks
                else:
                    block.text = extracted_text
            except Exception as e:
                # Fallback for unexpected errors - set empty text to continue processing
                logger.error("Failed to extract text: %s", e, exc_info=True)
                if is_image_block:
                    block.description = ""
                block.text = ""

        return block

    def _get_figure_description_prompt(self) -> str:
        """Get prompt for generating figure/image description.

        Returns:
            Prompt string for figure description
        """
        # Try to get from prompt manager first
        try:
            return self.prompt_manager.get_prompt("content_analysis", "figure_description", "user")
        except (KeyError, AttributeError):
            # Fallback prompt for figure description
            return (
                "Describe this image or figure in detail. Include:\n"
                "1. What type of image/figure it is (photo, chart, graph, diagram, etc.)\n"
                "2. The main elements or content visible\n"
                "3. Any text, labels, or annotations visible\n"
                "4. Key information or data shown (if applicable)\n"
                "5. Any notable colors, patterns, or visual characteristics\n\n"
                "Provide a comprehensive description that allows someone to understand "
                "the content without seeing the image."
            )

    def _crop_block(self, image: np.ndarray, block: Block) -> np.ndarray:
        """Crop a block from the full image.

        Args:
            image: Full page image
            block: Block with bbox

        Returns:
            Cropped block image as numpy array
        """
        return block.bbox.crop(image, padding=5)

    def _get_prompt_for_block_type(self, block_type: str) -> str:
        """Get appropriate prompt for specific block type.

        Args:
            block_type: Type of block (e.g., "text", "table", "figure")

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
            Corrected text string, or dict with error info if correction fails
        """
        if not text or not text.strip():
            return text

        try:
            if self.client is None:
                raise APIClientError(
                    f"API client not initialized for backend '{self.backend}'. "
                    "Cannot correct text without a configured client."
                )
            system_prompt = self.prompt_manager.get_prompt("text_correction", "system")
            user_prompt = self.prompt_manager.get_prompt("text_correction", "user", text=text)
            corrected = self.client.correct_text(text, system_prompt, user_prompt)
            return corrected
        except Exception as e:
            logger.error("Text correction failed: %s", e, exc_info=True)
            return {"error": "correction_failed", "original_text": text}

    async def process_blocks_async(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
        max_concurrent: int = 5,
    ) -> list[Block]:
        """Process all blocks to extract text concurrently (async).

        This method processes multiple blocks in parallel using async API clients,
        significantly improving performance by reducing overall processing time.

        Args:
            image: Full page image
            blocks: List of detected blocks
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of blocks with text extracted

        Raises:
            APIClientError: If async client is not initialized
        """
        if self.async_client is None:
            raise APIClientError(
                "Async client not initialized. Please set use_async=True when creating TextRecognizer."
            )

        # Prepare batch data: (image, region_info, prompt) tuples
        batch_data: list[tuple[np.ndarray, dict[str, Any], str]] = []
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
            Corrected text string, or dict with error info if correction fails

        Raises:
            RuntimeError: If async client is not initialized
        """
        if not text or not text.strip():
            return text

        if self.async_client is None:
            raise RuntimeError(
                "Async client not initialized. Please set use_async=True when creating TextRecognizer."
            )

        try:
            system_prompt = self.prompt_manager.get_prompt("text_correction", "system")
            user_prompt = self.prompt_manager.get_prompt("text_correction", "user", text=text)
            corrected = await self.async_client.correct_text(text, system_prompt, user_prompt)
            return corrected
        except Exception as e:
            logger.error("Async text correction failed: %s", e, exc_info=True)
            return {"error": "correction_failed", "original_text": text}

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TextRecognizer(backend={self.backend!r}, model={self.model!r}, "
            f"use_cache={self.use_cache}, use_async={self.use_async})"
        )

