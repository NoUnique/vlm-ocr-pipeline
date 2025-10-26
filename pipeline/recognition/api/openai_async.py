"""
Async OpenAI VLM API client for parallel OCR text extraction.

This module provides async versions of the OpenAI API client for concurrent processing
of multiple blocks, significantly improving performance by allowing parallel API calls
within rate limits.
"""

import asyncio
import base64
import gc
import io
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openai
import yaml
from openai import AsyncOpenAI
from PIL import Image

from pipeline.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_IMAGE_DIMENSION,
    SPECIAL_BLOCK_MAX_TOKENS,
    TEXT_CORRECTION_MAX_TOKENS,
    TEXT_CORRECTION_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class AsyncOpenAIClient:
    """Async OpenAI VLM API client for concurrent OCR text processing."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, base_url: str | None = None):
        """
        Initialize async OpenAI API client.

        Args:
            model: Model to use (can be OpenRouter format like 'openai/gpt-4')
            api_key: API key (if not provided, reads from environment)
            base_url: Base URL for API (for OpenRouter or custom endpoints)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Set default base URL for OpenRouter if using openrouter models
        if not self.base_url and (
            "/" in self.model or "anthropic" in self.model.lower() or "google" in self.model.lower()
        ):
            self.base_url = "https://openrouter.ai/api/v1"
            # Try OpenRouter API key if OpenAI key not available
            if not self.api_key:
                self.api_key = os.environ.get("OPENROUTER_API_KEY")

        # Load API config for max_tokens and temperature
        self._load_api_config()

        # Use AsyncOpenAI client
        self.client: Any = self._setup_openai_client()

    def _load_api_config(self) -> None:
        """Load API configuration from settings/api_config.yaml."""
        try:
            config_path = Path("settings") / "api_config.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    api_config = yaml.safe_load(f) or {}
                    openai_config = api_config.get("openai", {})

                    # Extract config values with fallbacks
                    text_extraction = openai_config.get("text_extraction", {})
                    special_blocks = openai_config.get("special_blocks", {})
                    text_correction = openai_config.get("text_correction", {})

                    self.text_extraction_max_tokens = text_extraction.get("max_tokens", DEFAULT_MAX_TOKENS)
                    self.text_extraction_temperature = text_extraction.get("temperature", DEFAULT_TEMPERATURE)

                    self.special_blocks_max_tokens = special_blocks.get("max_tokens", SPECIAL_BLOCK_MAX_TOKENS)
                    self.special_blocks_temperature = special_blocks.get("temperature", DEFAULT_TEMPERATURE)

                    self.text_correction_max_tokens = text_correction.get("max_tokens", TEXT_CORRECTION_MAX_TOKENS)
                    self.text_correction_temperature = text_correction.get("temperature", TEXT_CORRECTION_TEMPERATURE)

                    logger.debug("Loaded API config from %s", config_path)
            else:
                # Use default constants
                self._set_default_config()
                logger.debug("API config file not found, using defaults")
        except (yaml.YAMLError, OSError, KeyError, TypeError) as e:
            logger.warning("Failed to load API config: %s. Using defaults.", e)
            self._set_default_config()

    def _set_default_config(self) -> None:
        """Set default API configuration values."""
        self.text_extraction_max_tokens = DEFAULT_MAX_TOKENS
        self.text_extraction_temperature = DEFAULT_TEMPERATURE
        self.special_blocks_max_tokens = SPECIAL_BLOCK_MAX_TOKENS
        self.special_blocks_temperature = DEFAULT_TEMPERATURE
        self.text_correction_max_tokens = TEXT_CORRECTION_MAX_TOKENS
        self.text_correction_temperature = TEXT_CORRECTION_TEMPERATURE

    def _setup_openai_client(self) -> AsyncOpenAI | None:
        """
        Setup async OpenAI API client.

        Returns:
            AsyncOpenAI client instance or None if setup fails
        """
        if not self.api_key:
            logger.warning("OpenAI API key not found (model=%s, base_url=%s)", self.model, self.base_url or "default")
            return None

        try:
            # Create async client
            if self.base_url:
                client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.debug(
                    "AsyncOpenAI client initialized (model=%s, base_url=%s)", self.model, self.base_url or "default"
                )
            else:
                client = AsyncOpenAI(api_key=self.api_key)
                logger.debug("AsyncOpenAI client initialized (model=%s)", self.model)

            return client

        except Exception as e:
            logger.error("Failed to initialize AsyncOpenAI client: %s", e)
            return None

    def is_available(self) -> bool:
        """
        Check if client is available for use.

        Returns:
            True if client is initialized and ready
        """
        return self.client is not None and self.api_key is not None

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.

        Args:
            image: Image as numpy array

        Returns:
            Base64 encoded image string
        """
        # Resize if too large
        height, width = image.shape[:2]
        if max(height, width) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            image_resized = image

        pil_image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG", quality=85, optimize=True)
        img_bytes = img_byte_arr.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    async def extract_text(
        self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str
    ) -> dict[str, Any]:  # noqa: PLR0911
        """
        Extract text from block using OpenAI API (async).

        Args:
            region_img: Image block as numpy array
            region_info: Block metadata including type and coordinates
            prompt: Prompt for text extraction

        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_available():
            logger.warning(
                "OpenAI API client not initialized (model=%s, base_url=%s)", self.model, self.base_url or "default"
            )
            return {"type": region_info["type"], "xywh": region_info["xywh"], "text": "", "confidence": 0.0}

        try:
            base64_image = self._encode_image(region_img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]

            logger.debug(
                "Requesting async OpenAI extract_text (model=%s, base_url=%s)", self.model, self.base_url or "default"
            )
            client = self.client
            if client is None:
                logger.warning(
                    "OpenAI API client became unavailable (model=%s, base_url=%s)",
                    self.model,
                    self.base_url or "default",
                )
                return {"type": region_info["type"], "xywh": region_info["xywh"], "text": "", "confidence": 0.0}

            # Async API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.text_extraction_max_tokens,
                temperature=self.text_extraction_temperature,
            )

            text = response.choices[0].message.content.strip()

            result = {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": text,
                "confidence": region_info.get("confidence", 1.0),
            }

            # Clean up
            del base64_image
            gc.collect()

            return result

        except openai.RateLimitError as e:
            # 429 Rate limit errors
            logger.error("OpenAI rate limit exceeded: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[RATE_LIMIT_EXCEEDED]",
                "confidence": 0.0,
                "error": "openai_rate_limit",
                "error_message": str(e),
            }
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            # Network/timeout errors
            logger.error("OpenAI connection/timeout error: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[OPENAI_CONNECTION_ERROR]",
                "confidence": 0.0,
                "error": "openai_connection_error",
                "error_message": str(e),
            }
        except openai.APIError as e:
            # Other OpenAI API errors (4xx, 5xx)
            logger.error("OpenAI API error: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[OPENAI_API_ERROR]",
                "confidence": 0.0,
                "error": "openai_api_error",
                "error_message": str(e),
            }
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error in async extract_text: %s", e, exc_info=True)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[EXTRACTION_ERROR]",
                "confidence": 0.0,
                "error": "unexpected_error",
                "error_message": str(e),
            }

    async def extract_text_batch(
        self, regions: list[tuple[np.ndarray, dict[str, Any], str]], max_concurrent: int = 5
    ) -> list[dict[str, Any]]:
        """
        Extract text from multiple blocks concurrently with rate limiting.

        Args:
            regions: List of (image, region_info, prompt) tuples
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of extraction results in the same order as input
        """
        if not regions:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(
            img: np.ndarray, info: dict[str, Any], prompt: str
        ) -> dict[str, Any]:
            async with semaphore:
                return await self.extract_text(img, info, prompt)

        # Create tasks for all regions
        tasks = [extract_with_semaphore(img, info, prompt) for img, info, prompt in regions]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Task %d failed with exception: %s", i, result)
                _, info, _ = regions[i]
                final_results.append({
                    "type": info["type"],
                    "xywh": info["xywh"],
                    "text": "[BATCH_ERROR]",
                    "confidence": 0.0,
                    "error": "batch_processing_error",
                })
            else:
                final_results.append(result)

        return final_results

    async def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> str:
        """
        Correct raw text using VLM (async).

        Args:
            text: Raw text to correct
            system_prompt: System prompt for correction task
            user_prompt: User prompt with text to correct

        Returns:
            Corrected text
        """
        if not self.is_available():
            logger.warning("OpenAI API client not available for text correction")
            return text

        try:
            client = self.client
            if client is None:
                return text

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            logger.debug("Requesting async OpenAI text correction (model=%s)", self.model)

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.text_correction_max_tokens,
                temperature=self.text_correction_temperature,
            )

            corrected = response.choices[0].message.content.strip()
            return corrected

        except Exception as e:
            logger.error("Text correction failed (async): %s", e, exc_info=True)
            return text

    async def close(self) -> None:
        """Close the async client connection."""
        if self.client:
            await self.client.close()
