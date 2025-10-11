"""Text recognizer for extracting and processing text from document regions."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..prompt import PromptManager
from .api import GeminiClient, OpenAIClient
from .cache import RecognitionCache

logger = logging.getLogger(__name__)


class TextRecognizer:
    """Handles text recognition and extraction from document regions.
    
    This class manages:
    - OCR text extraction using API clients
    - Special region processing (tables, figures)
    - Text correction
    - Result caching
    """

    def __init__(
        self,
        cache_dir: Path,
        use_cache: bool = True,
        backend: str = "openai",
        model: str = "gemini-2.5-flash",
        gemini_tier: str = "free",
    ):
        """Initialize the text recognizer.
        
        Args:
            cache_dir: Directory for cache files
            use_cache: Whether to use caching
            backend: Backend API to use ("openai" or "gemini")
            model: Model name to use
            gemini_tier: Gemini API tier (for rate limiting)
        """
        self.backend = backend.lower()
        self.model = model
        self.cache = RecognitionCache(cache_dir, use_cache)
        self.prompt_manager = PromptManager(model=model, backend=backend)

        # Initialize API clients
        if self.backend == "gemini":
            self.gemini_client = GeminiClient(gemini_model=model)
            self.ai_client = self.gemini_client
        else:  # OpenAI backend
            self.openai_client = OpenAIClient(model=model)
            self.ai_client = self.openai_client
            # Still initialize Gemini client for fallback if needed
            self.gemini_client = GeminiClient(gemini_model="gemini-2.5-flash")

        logger.info("Text recognizer initialized: %s (model=%s)", self.backend.upper(), self.model)

    def process_regions(
        self,
        image: np.ndarray,
        regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process all detected regions to extract text.
        
        Args:
            image: Source image
            regions: List of detected regions
            
        Returns:
            List of processed regions with extracted text
        """
        processed_regions = []
        processed_coords = set()

        for region in regions:
            region_key = f"{region['coords'][0]}_{region['coords'][1]}_{region['coords'][2]}_{region['coords'][3]}"

            if region_key in processed_coords:
                continue

            region_img = self._crop_region(image, region)

            if region["type"] in ["table", "figure"]:
                processed_region = self._process_special_region(region_img, region)
            else:
                processed_region = self._extract_text(region_img, region)

            processed_regions.append(processed_region)
            processed_coords.add(region_key)

            del region_img
            gc.collect()

        return processed_regions

    def _crop_region(self, image: np.ndarray, region: dict[str, Any]) -> np.ndarray:
        """Crop region from image (delegated to maintain encapsulation).
        
        Note: In a full refactoring, this might be moved to a shared utility,
        but keeping here for now to match the original structure.
        """
        coords = region["coords"]
        x, y, w, h = coords

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid region coordinates: x1=%s, y1=%s, x2=%s, y2=%s", x1, y1, x2, y2)
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return image[y1:y2, x1:x2]

    def _extract_text(self, region_img: np.ndarray, region_info: dict[str, Any]) -> dict[str, Any]:
        """Extract text from a region using the configured API backend.
        
        Args:
            region_img: Cropped region image
            region_info: Region metadata
            
        Returns:
            Dictionary with extracted text and metadata
        """
        image_hash = self.cache.calculate_image_hash(region_img)
        cache_type = f"{self.backend}_ocr"

        cached_result = self.cache.get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result["coords"] = region_info["coords"]
            return cached_result

        # Get prompt from PromptManager
        prompt = self.prompt_manager.get_prompt("text_extraction", "user")

        # Use AI client to extract text
        result = self.ai_client.extract_text(region_img, region_info, prompt)

        # Save to cache if successful
        if "error" not in result:
            self.cache.save_to_cache(image_hash, cache_type, result)

        return result

    def _process_special_region(self, region_img: np.ndarray, region_info: dict[str, Any]) -> dict[str, Any]:
        """Process special regions (tables, figures) with configured AI backend.
        
        Args:
            region_img: Cropped region image
            region_info: Region metadata
            
        Returns:
            Dictionary with processed content and metadata
        """
        image_hash = self.cache.calculate_image_hash(region_img)
        cache_type = f"{region_info['type']}_{self.backend}"

        cached_result = self.cache.get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result["coords"] = region_info["coords"]
            return cached_result

        if not self.ai_client.is_available():
            logger.warning("%s API client not initialized, falling back to text extraction", self.backend.upper())
            return self._extract_text(region_img, region_info)

        # Get prompt from PromptManager
        if self.backend == "gemini":
            prompt = self.prompt_manager.get_gemini_prompt_for_region_type(region_info["type"])
        else:
            prompt = self.prompt_manager.get_prompt_for_region_type(region_info["type"])

        # Use AI client to process special region
        result = self.ai_client.process_special_region(region_img, region_info, prompt)

        # Save to cache if successful
        if "error" not in result:
            self.cache.save_to_cache(image_hash, cache_type, result)

        return result

    def correct_text(self, text: str) -> str:
        """Correct OCR text using configured AI backend.
        
        Args:
            text: Raw OCR text to correct
            
        Returns:
            Corrected text
        """
        if not self.ai_client.is_available() or not text:
            return text

        # Get prompts from PromptManager
        system_prompt = self.prompt_manager.get_prompt("text_correction", "system")
        user_prompt = self.prompt_manager.get_prompt("text_correction", "user", text=text)

        # Use AI client to correct text
        result = self.ai_client.correct_text(text, system_prompt, user_prompt)

        # Handle different return types
        if isinstance(result, dict):
            return result.get("corrected_text", text)
        else:
            return str(result)

