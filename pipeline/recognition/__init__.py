"""Text recognition module for OCR and content extraction."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..prompt import PromptManager
from ..types import Block, Recognizer
from .api.gemini import GeminiClient
from .api.openai import OpenAIClient
from .cache import RecognitionCache

logger = logging.getLogger(__name__)

# Lazy import for PaddleOCR-VL (optional dependency)
try:
    from .paddleocr import PaddleOCRVLRecognizer  # noqa: PLC0415

    _HAS_PADDLEOCR_VL = True
except ImportError:
    PaddleOCRVLRecognizer = None  # type: ignore[misc, assignment]
    _HAS_PADDLEOCR_VL = False

__all__ = ["TextRecognizer", "PaddleOCRVLRecognizer"]


class TextRecognizer(Recognizer):
    """Handles text recognition and extraction from document blocks.

    This class manages the text extraction pipeline using VLM APIs
    (OpenAI/OpenRouter or Gemini) with intelligent caching.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        use_cache: bool = True,
        backend: str = "openai",
        model: str = "gemini-2.5-flash",
        gemini_tier: str = "free",
    ):
        """Initialize text recognizer.

        Args:
            cache_dir: Directory for caching recognition results
            use_cache: Whether to use caching
            backend: Backend API ("openai" or "gemini")
            model: Model to use for text processing
            gemini_tier: Gemini API tier (only used with gemini backend)
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.backend = backend.lower()
        self.model = model
        self.gemini_tier = gemini_tier

        # Initialize cache
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = RecognitionCache(cache_dir=self.cache_dir)
        else:
            self.cache = None

        # Initialize VLM client
        if self.backend == "gemini":
            self.client = GeminiClient(gemini_model=model)
        elif self.backend == "openai":
            self.client = OpenAIClient(model=model)
        elif self.backend == "paddleocr-vl":
            # PaddleOCR-VL is handled separately, no client needed
            self.client = None  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Initialize prompt manager
        self.prompt_manager = PromptManager(model=model)

        logger.info(
            "TextRecognizer initialized: backend=%s, model=%s, cache=%s",
            self.backend,
            model,
            use_cache,
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

        # Extract text
        try:
            if self.client is None:
                raise RuntimeError(
                    f"API client not initialized for backend '{self.backend}'. "
                    "Cannot extract text without a configured client."
                )
            result = self.client.extract_text(block_image, block.to_dict(), prompt)
            block.text = result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            # Fallback for unexpected errors - set empty text to continue processing
            # (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Failed to extract text: %s", e, exc_info=True)
            block.text = ""

        # Cleanup
        del block_image
        gc.collect()

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
