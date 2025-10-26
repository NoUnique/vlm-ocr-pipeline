"""Caching utilities for recognition results."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RecognitionCache:
    """Manages caching of recognition results.

    This class handles:
    - Image hashing for cache keys
    - Reading and writing cached results
    - Cache invalidation
    """

    def __init__(self, cache_dir: Path, use_cache: bool = True):
        """Initialize the recognition cache.

        Args:
            cache_dir: Directory for cache files
            use_cache: Whether caching is enabled
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def calculate_image_hash(self, image: np.ndarray) -> str:
        """Calculate hash for image caching.

        Args:
            image: Input image

        Returns:
            MD5 hash of the resized image
        """
        small_img = cv2.resize(image, (32, 32))
        success, encoded = cv2.imencode(".jpg", small_img, (cv2.IMWRITE_JPEG_QUALITY, 50))
        if not success:
            raise ValueError("Failed to encode image for hashing")
        image_hash = hashlib.md5(encoded.tobytes()).hexdigest()
        del small_img, encoded
        return image_hash

    def get_cached_result(self, image_hash: str, cache_type: str) -> dict[str, Any] | None:
        """Get cached result if exists.

        Args:
            image_hash: Hash of the image
            cache_type: Type of cache (e.g., 'gemini_ocr', 'table')

        Returns:
            Cached result or None if not found
        """
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_type}_{image_hash}.json"

        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.debug("Cache hit for %s: %s", cache_type, image_hash)
                return cached_data
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                logger.warning("Failed to load cache file %s: %s", cache_file, e)

        return None

    def save_to_cache(self, image_hash: str, cache_type: str, result: dict[str, Any]) -> None:
        """Save result to cache.

        Args:
            image_hash: Hash of the image
            cache_type: Type of cache
            result: Result to cache
        """
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_type}_{image_hash}.json"

        try:
            # Remove coords before caching (will be added back when retrieved)
            cache_data = {k: v for k, v in result.items() if k != "coords"}

            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug("Cached result for %s: %s", cache_type, image_hash)
        except (OSError, TypeError) as e:
            logger.warning("Failed to save cache file %s: %s", cache_file, e)
