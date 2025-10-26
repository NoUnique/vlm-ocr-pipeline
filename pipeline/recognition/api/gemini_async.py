"""
Async Google Gemini VLM API client for parallel OCR text extraction.

This module provides async versions of the Gemini API client for concurrent processing
of multiple blocks, significantly improving performance by allowing parallel API calls
within rate limits.
"""

import asyncio
import difflib
import gc
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import yaml
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types
from PIL import Image

from pipeline.constants import ESTIMATED_IMAGE_TOKENS

logger = logging.getLogger(__name__)


class AsyncGeminiClient:
    """Async Google Gemini VLM API client for concurrent OCR text processing."""

    def __init__(self, gemini_model: str = "gemini-2.5-flash", api_key: str | None = None):
        """
        Initialize async Gemini API client.

        Args:
            gemini_model: Gemini model to use
            api_key: Gemini API key (if not provided, reads from environment)
        """
        self.gemini_model = gemini_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Load API config for estimated tokens
        self._load_api_config()

        self.client = self._setup_gemini_api()

    def _load_api_config(self) -> None:
        """Load API configuration from settings/api_config.yaml."""
        try:
            config_path = Path("settings") / "api_config.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    api_config = yaml.safe_load(f) or {}
                    gemini_config = api_config.get("gemini", {})

                    # Extract estimated tokens values
                    text_extraction = gemini_config.get("text_extraction", {})
                    special_blocks = gemini_config.get("special_blocks", {})
                    text_correction = gemini_config.get("text_correction", {})

                    self.text_extraction_estimated_tokens = text_extraction.get(
                        "estimated_tokens", ESTIMATED_IMAGE_TOKENS
                    )
                    self.special_blocks_estimated_tokens = special_blocks.get(
                        "estimated_tokens", ESTIMATED_IMAGE_TOKENS
                    )
                    self.text_correction_estimated_tokens = text_correction.get(
                        "estimated_tokens", ESTIMATED_IMAGE_TOKENS
                    )

                    logger.debug("Loaded Gemini async API config from %s", config_path)
            else:
                # Use default constants
                self._set_default_config()
                logger.debug("Gemini async API config file not found, using defaults")
        except (yaml.YAMLError, OSError, KeyError, TypeError) as e:
            logger.warning("Failed to load Gemini async API config: %s. Using defaults.", e)
            self._set_default_config()

    def _set_default_config(self) -> None:
        """Set default API configuration from constants."""
        self.text_extraction_estimated_tokens = ESTIMATED_IMAGE_TOKENS
        self.special_blocks_estimated_tokens = ESTIMATED_IMAGE_TOKENS
        self.text_correction_estimated_tokens = ESTIMATED_IMAGE_TOKENS

    def _setup_gemini_api(self) -> genai.Client | None:
        """Setup Gemini API client."""
        try:
            if not self.api_key:
                logger.warning("GEMINI_API_KEY environment variable not set")
                return None

            client = genai.Client(api_key=self.api_key)
            logger.info("Async Gemini API client initialized successfully")
            return client
        except (TypeError, ValueError) as e:
            logger.error("Failed to initialize async Gemini client with invalid configuration: %s", e)
            return None
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error initializing async Gemini client: %s", e, exc_info=True)
            return None

    def is_available(self) -> bool:
        """Check if Gemini API client is available."""
        return self.client is not None

    async def extract_text(  # noqa: PLR0911
        self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """
        Extract text from block using Gemini API (async).

        Args:
            region_img: Image block as numpy array
            region_info: Block metadata including type and coordinates
            prompt: Prompt for text extraction

        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_available():
            logger.warning("Async Gemini API client not initialized")
            return {"type": region_info["type"], "xywh": region_info["xywh"], "text": "", "confidence": 0.0}

        try:
            # Resize image if too large
            h, w = region_img.shape[:2]
            max_dim = 1024

            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                region_img_resized = cv2.resize(region_img, (new_w, new_h))
            else:
                region_img_resized = region_img

            pil_image = Image.fromarray(cv2.cvtColor(region_img_resized, cv2.COLOR_BGR2RGB))

            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="JPEG", quality=85, optimize=True)
            img_bytes = img_byte_arr.getvalue()

            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": types.Blob(
                                mime_type="image/jpeg",
                                data=img_bytes,
                            )
                        },
                    ],
                }
            ]

            logger.debug("Requesting async Gemini extract_text (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Async Gemini API client became unavailable")
                return {
                    "type": region_info["type"],
                    "xywh": region_info["xywh"],
                    "text": "",
                    "confidence": 0.0,
                    "error": "client_unavailable",
                    "error_message": "Gemini client not initialized",
                }

            # Async API call
            response = await client.aio.models.generate_content(
                model=self.gemini_model,
                contents=cast(types.ContentListUnionDict, contents),
            )

            del pil_image, img_byte_arr, img_bytes, region_img_resized
            gc.collect()

            text = (response.text or "").strip()

            result = {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": text,
                "confidence": region_info.get("confidence", 1.0),
            }

            del response
            gc.collect()

            return result

        except google_exceptions.ResourceExhausted as e:
            # 429 Rate limit errors (RESOURCE_EXHAUSTED)
            logger.error("Gemini rate limit exceeded: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[RATE_LIMIT_EXCEEDED]",
                "confidence": 0.0,
                "error": "gemini_rate_limit",
                "error_message": str(e),
            }
        except google_exceptions.RetryError as e:
            # Retry failures (network issues, timeouts)
            logger.error("Gemini retry error: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[GEMINI_RETRY_FAILED]",
                "confidence": 0.0,
                "error": "gemini_retry_error",
                "error_message": str(e),
            }
        except google_exceptions.GoogleAPIError as e:
            # Other Google API errors (4xx, 5xx)
            logger.error("Gemini API error: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[GEMINI_API_ERROR]",
                "confidence": 0.0,
                "error": "gemini_api_error",
                "error_message": str(e),
            }
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error during async Gemini text extraction: %s", e, exc_info=True)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[GEMINI_EXTRACTION_FAILED]",
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

        async def extract_with_semaphore(img: np.ndarray, info: dict[str, Any], prompt: str) -> dict[str, Any]:
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
                final_results.append(
                    {
                        "type": info["type"],
                        "xywh": info["xywh"],
                        "text": "[BATCH_ERROR]",
                        "confidence": 0.0,
                        "error": "batch_processing_error",
                    }
                )
            else:
                final_results.append(result)

        return final_results

    async def process_special_region(  # noqa: PLR0911
        self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """
        Process special blocks (tables, figures) with Gemini API (async).

        Args:
            region_img: Image block as numpy array
            region_info: Block metadata including type and coordinates
            prompt: Prompt for special content analysis

        Returns:
            Dictionary containing processed content and metadata
        """
        if not self.is_available():
            logger.warning("Async Gemini API client not initialized")
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "Gemini API not available",
                "analysis": "Client not initialized",
                "confidence": 0.0,
            }

        try:
            # Resize image if too large
            h, w = region_img.shape[:2]
            max_dim = 1024

            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                region_img_resized = cv2.resize(region_img, (new_w, new_h))
            else:
                region_img_resized = region_img

            pil_image = Image.fromarray(cv2.cvtColor(region_img_resized, cv2.COLOR_BGR2RGB))

            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="JPEG", quality=85, optimize=True)
            img_bytes = img_byte_arr.getvalue()

            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": types.Blob(
                                mime_type="image/jpeg",
                                data=img_bytes,
                            )
                        },
                    ],
                }
            ]

            logger.debug("Requesting async Gemini process_special_region (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Async Gemini API client became unavailable")
                return {
                    "type": region_info["type"],
                    "xywh": region_info["xywh"],
                    "content": "Gemini API not available",
                    "analysis": "Client not initialized",
                    "confidence": 0.0,
                }

            # Async API call
            response = await client.aio.models.generate_content(
                model=self.gemini_model,
                contents=cast(types.ContentListUnionDict, contents),
            )

            del pil_image, img_byte_arr, img_bytes, region_img_resized
            gc.collect()

            response_text = (response.text or "").strip()
            parsed_result = self._parse_gemini_response(response_text, region_info)

            del response
            gc.collect()

            return parsed_result

        except google_exceptions.ResourceExhausted as e:
            # 429 Rate limit errors (RESOURCE_EXHAUSTED)
            logger.error("Gemini rate limit exceeded: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "[RATE_LIMIT_EXCEEDED]",
                "analysis": "Rate limit exceeded",
                "confidence": 0.0,
                "error": "gemini_rate_limit",
                "error_message": str(e),
            }
        except google_exceptions.RetryError as e:
            # Retry failures (network issues, timeouts)
            logger.error("Gemini retry error: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "[GEMINI_RETRY_FAILED]",
                "analysis": "Retry failed",
                "confidence": 0.0,
                "error": "gemini_retry_error",
                "error_message": str(e),
            }
        except google_exceptions.GoogleAPIError as e:
            # Other Google API errors (4xx, 5xx)
            logger.error("Gemini API error: %s", e)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "[GEMINI_API_ERROR]",
                "analysis": "Gemini API error",
                "confidence": 0.0,
                "error": "gemini_api_error",
                "error_message": str(e),
            }
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error during async Gemini special block processing: %s", e, exc_info=True)
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "[GEMINI_PROCESSING_FAILED]",
                "analysis": f"Processing failed: {str(e)}",
                "confidence": 0.0,
                "error": "unexpected_error",
                "error_message": str(e),
            }

    async def process_special_region_batch(
        self, regions: list[tuple[np.ndarray, dict[str, Any], str]], max_concurrent: int = 3
    ) -> list[dict[str, Any]]:
        """
        Process multiple special blocks concurrently with rate limiting.

        Args:
            regions: List of (image, region_info, prompt) tuples
            max_concurrent: Maximum number of concurrent API calls (lower for special blocks)

        Returns:
            List of processing results in the same order as input
        """
        if not regions:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(img: np.ndarray, info: dict[str, Any], prompt: str) -> dict[str, Any]:
            async with semaphore:
                return await self.process_special_region(img, info, prompt)

        # Create tasks for all regions
        tasks = [process_with_semaphore(img, info, prompt) for img, info, prompt in regions]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Special region task %d failed with exception: %s", i, result)
                _, info, _ = regions[i]
                final_results.append(
                    {
                        "type": info["type"],
                        "xywh": info["xywh"],
                        "content": "[BATCH_ERROR]",
                        "analysis": "Batch processing error",
                        "confidence": 0.0,
                        "error": "batch_processing_error",
                    }
                )
            else:
                final_results.append(result)

        return final_results

    async def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Correct OCR text using Gemini API (async).

        Args:
            text: Text to correct
            system_prompt: System instruction prompt
            user_prompt: User prompt with text formatting

        Returns:
            Dictionary containing corrected text and confidence
        """
        if not self.is_available() or not text:
            return self._text_correction_result(text, 0.0)

        result: dict[str, Any]

        try:
            contents = [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                }
            ]

            logger.debug("Requesting async Gemini correct_text (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Async Gemini API client became unavailable")
                result = self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_FAILED]",
                    error_message="Gemini client not initialized",
                )
            else:
                # Async API call
                response = await client.aio.models.generate_content(
                    model=self.gemini_model,
                    contents=cast(types.ContentListUnionDict, contents),
                )

                corrected_text = (response.text or "").strip()
                similarity = difflib.SequenceMatcher(None, text, corrected_text).ratio()
                # How much was changed (0.0 = no change, 1.0 = completely different)
                correction_ratio = 1.0 - similarity
                result = self._text_correction_result(corrected_text, correction_ratio)

        except google_exceptions.ResourceExhausted as e:
            # 429 Rate limit errors (RESOURCE_EXHAUSTED)
            logger.error("Gemini rate limit exceeded during text correction: %s", e)
            result = self._text_correction_result(
                text,
                0.0,
                error="[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]",
                error_message=str(e),
            )
        except google_exceptions.ServiceUnavailable as e:
            # 503 Service unavailable errors
            logger.error("Gemini service unavailable during text correction: %s", e)
            result = self._text_correction_result(
                text,
                0.0,
                error="[TEXT_CORRECTION_SERVICE_UNAVAILABLE]",
                error_message=str(e),
            )
        except google_exceptions.RetryError as e:
            # Retry failures (network issues, timeouts)
            logger.error("Gemini retry error during text correction: %s", e)
            result = self._text_correction_result(
                text,
                0.0,
                error="[TEXT_CORRECTION_RETRY_FAILED]",
                error_message=str(e),
            )
        except google_exceptions.GoogleAPIError as e:
            # Other Google API errors (4xx, 5xx)
            logger.error("Gemini API error during text correction: %s", e)
            result = self._text_correction_result(
                text,
                0.0,
                error="[TEXT_CORRECTION_API_ERROR]",
                error_message=str(e),
            )
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error during async text correction: %s", e, exc_info=True)
            result = self._text_correction_result(
                text,
                0.0,
                error="[TEXT_CORRECTION_FAILED]",
                error_message=str(e),
            )

        return result

    def _text_correction_result(
        self,
        corrected_text: str,
        correction_ratio: float,
        error: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"corrected_text": corrected_text, "correction_ratio": correction_ratio}
        if error:
            result["error"] = error
        if error_message:
            result["error_message"] = error_message
        return result

    def _parse_gemini_response(self, response_text: str, region_info: dict[str, Any]) -> dict[str, Any]:
        """Parse Gemini response for special blocks."""
        try:
            parsed = json.loads(response_text)

            result = {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "confidence": region_info.get("confidence", 1.0),
            }

            if region_info["type"] == "table":
                result["content"] = parsed.get("markdown_table", "")
                result["analysis"] = parsed.get("summary", "")
                result["educational_value"] = parsed.get("educational_value", "")
                result["related_topics"] = parsed.get("related_topics", [])
            else:  # figure, formula, etc.
                result["content"] = parsed.get("description", "")
                result["analysis"] = parsed.get("educational_value", "")
                result["related_topics"] = parsed.get("related_topics", [])
                result["exam_relevance"] = parsed.get("exam_relevance", "")

            return result

        except json.JSONDecodeError:
            logger.warning("Failed to parse async Gemini JSON response, using as plain text")
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": response_text,
                "analysis": "Direct response (JSON parsing failed)",
                "confidence": region_info.get("confidence", 1.0),
            }

    def reload_client(self, api_key: str | None = None) -> bool:
        """
        Reload the Gemini API client (useful after API key updates).

        Args:
            api_key: New API key to use (optional)

        Returns:
            True if client was successfully reloaded
        """
        if api_key:
            self.api_key = api_key

        self.client = self._setup_gemini_api()
        return self.is_available()
