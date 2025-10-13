"""Text recognition module for OCR and content extraction."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..prompt import PromptManager
from .api.gemini import GeminiClient
from .api.openai import OpenAIClient
from .cache import RecognitionCache

logger = logging.getLogger(__name__)

__all__ = ["TextRecognizer"]


class TextRecognizer:
    """Handles text recognition and extraction from document regions.

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
            self.client = GeminiClient(model=model, tier=gemini_tier)
        elif self.backend == "openai":
            self.client = OpenAIClient(model=model)
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

    def process_regions(
        self, image: np.ndarray, regions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process all regions to extract text.

        Args:
            image: Full page image
            regions: List of detected regions

        Returns:
            Regions with text extracted
        """
        processed_regions = []

        for region in regions:
            processed_region = self._process_single_region(image, region)
            processed_regions.append(processed_region)

        return processed_regions

    def _process_single_region(
        self, image: np.ndarray, region: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a single region to extract text.

        Args:
            image: Full page image
            region: Single region dict

        Returns:
            Region with text extracted
        """
        region_type = region.get("type", "unknown")

        # Extract region image
        try:
            region_image = self._crop_region(image, region)
        except Exception as e:
            logger.error("Failed to crop region: %s", e)
            region["error"] = "crop_failed"
            return region

        # Get prompt for region type
        prompt = self._get_prompt_for_region_type(region_type)

        # Extract text
        try:
            text = self.client.extract_text(region_image, prompt)
            region["text"] = text
        except Exception as e:
            logger.error("Failed to extract text: %s", e)
            region["error"] = "extraction_failed"
            region["text"] = ""

        # Cleanup
        del region_image
        gc.collect()

        return region

    def _crop_region(self, image: np.ndarray, region: dict[str, Any]) -> np.ndarray:
        """Crop a region from the full image.

        Args:
            image: Full page image
            region: Region with coords

        Returns:
            Cropped region image
        """
        coords = region.get("coords", [0, 0, 0, 0])
        x, y, w, h = coords

        # Convert to corners
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Add padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        # Validate
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid region coordinates")
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return image[y1:y2, x1:x2]

    def _get_prompt_for_region_type(self, region_type: str) -> dict[str, str]:
        """Get prompt for specific region type.

        Args:
            region_type: Type of region

        Returns:
            Prompt dict with system and user prompts
        """
        if region_type in ["plain text", "title", "list"]:
            return self.prompt_manager.get_prompt("text_extraction")
        elif region_type == "table":
            return self.prompt_manager.get_prompt("content_analysis")
        elif region_type in ["figure", "equation"]:
            return self.prompt_manager.get_prompt("content_analysis")
        else:
            return self.prompt_manager.get_prompt("text_extraction")

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
            prompt = self.prompt_manager.get_prompt("text_correction")
            corrected = self.client.correct_text(text, prompt)
            return corrected
        except Exception as e:
            logger.error("Text correction failed: %s", e)
            return {"error": "correction_failed", "original_text": text}


