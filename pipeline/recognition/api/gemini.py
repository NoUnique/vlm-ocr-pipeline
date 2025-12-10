"""
Google Gemini VLM API client for advanced OCR text extraction and processing.
Handles text extraction, special content analysis, and text correction using Gemini Vision Language Models.
"""

import difflib
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from pipeline.constants import ESTIMATED_IMAGE_TOKENS

from .base import BaseVLMClient
from .image_utils import prepare_image_for_api
from .ratelimit import rate_limiter
from .types import (
    create_extraction_error,
    create_special_content_error,
)

logger = logging.getLogger(__name__)


class GeminiClient(BaseVLMClient):
    """Google Gemini VLM API client for OCR text processing."""

    PROVIDER_NAME = "gemini"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, gemini_model: str = "gemini-2.5-flash", api_key: str | None = None):
        """
        Initialize Gemini API client.

        Args:
            gemini_model: Gemini model to use
            api_key: Gemini API key (if not provided, reads from environment)
        """
        super().__init__(model=gemini_model)
        self.gemini_model = gemini_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Load API config for estimated tokens
        self._load_api_config()

        self.client = self._setup_gemini_api()

    def _load_api_config(self) -> None:
        """Load API configuration from settings/api_config.yaml"""
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

                    logger.debug("Loaded Gemini API config from %s", config_path)
            else:
                # Use default constants
                self._set_default_config()
                logger.debug("Gemini API config file not found, using defaults")
        except (yaml.YAMLError, OSError, KeyError, TypeError) as e:
            logger.warning("Failed to load Gemini API config: %s. Using defaults.", e)
            self._set_default_config()

    def _set_default_config(self) -> None:
        """Set default API configuration from constants"""
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
            logger.info("Gemini API client initialized successfully")
            return client
        except (TypeError, ValueError) as e:
            logger.error("Failed to initialize Gemini client with invalid configuration: %s", e)
            return None
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error initializing Gemini client: %s", e, exc_info=True)
            return None

    def _setup_client(self) -> genai.Client | None:
        """Set up and return the API client instance (BaseVLMClient interface)."""
        return self._setup_gemini_api()

    def is_available(self) -> bool:
        """Check if Gemini API client is available"""
        return self.client is not None

    def extract_text(self, block_img: np.ndarray, block_info: dict[str, Any], prompt: str) -> dict[str, Any]:  # noqa: PLR0911
        """
        Extract text from block using Gemini API

        Args:
            block_img: Image block as numpy array
            block_info: Block metadata including type and coordinates
            prompt: Prompt for text extraction

        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_available():
            logger.warning("Gemini API client not initialized")
            return {"type": block_info["type"], "xywh": block_info["xywh"], "text": "", "confidence": 0.0}

        try:
            # Prepare image for API (resize and convert to JPEG)
            img_bytes = prepare_image_for_api(block_img)

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

            # Apply rate limiting
            estimated_tokens = self.text_extraction_estimated_tokens
            if not rate_limiter.wait_if_needed(estimated_tokens):
                return create_extraction_error(
                    block_info,
                    "[DAILY_LIMIT_EXCEEDED]",
                    "rate_limit_daily",
                    "Daily rate limit exceeded",
                )

            logger.info("Requesting Gemini extract_text (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Gemini API client became unavailable")
                return create_extraction_error(
                    block_info,
                    "",
                    "client_unavailable",
                    "Gemini client not initialized",
                )

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=cast(types.ContentListUnionDict, contents),
            )

            text = (response.text or "").strip()

            result = {
                "type": block_info["type"],
                "xywh": block_info["xywh"],
                "text": text,
                "confidence": block_info.get("confidence", 1.0),
            }

            return result

        except google_exceptions.ResourceExhausted as e:
            # 429 Rate limit errors (RESOURCE_EXHAUSTED)
            logger.error("Gemini rate limit exceeded: %s", e)
            return create_extraction_error(
                block_info, "[RATE_LIMIT_EXCEEDED]", "gemini_rate_limit", str(e)
            )
        except google_exceptions.RetryError as e:
            # Retry failures (network issues, timeouts)
            logger.error("Gemini retry error: %s", e)
            return create_extraction_error(
                block_info, "[GEMINI_RETRY_FAILED]", "gemini_retry_error", str(e)
            )
        except google_exceptions.GoogleAPIError as e:
            # Other Google API errors (4xx, 5xx)
            logger.error("Gemini API error: %s", e)
            return create_extraction_error(
                block_info, "[GEMINI_API_ERROR]", "gemini_api_error", str(e)
            )
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error during Gemini text extraction: %s", e, exc_info=True)
            return create_extraction_error(
                block_info, "[GEMINI_EXTRACTION_FAILED]", "unexpected_error", str(e)
            )

    def process_special_block(  # noqa: PLR0911
        self, block_img: np.ndarray, block_info: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """
        Process special blocks (tables, figures) with Gemini API

        Args:
            block_img: Image block as numpy array
            block_info: Block metadata including type and coordinates
            prompt: Prompt for special content analysis

        Returns:
            Dictionary containing processed content and metadata
        """
        if not self.is_available():
            logger.warning("Gemini API client not initialized")
            return {
                "type": block_info["type"],
                "xywh": block_info["xywh"],
                "content": "Gemini API not available",
                "analysis": "Client not initialized",
                "confidence": 0.0,
            }

        try:
            # Prepare image for API (resize and convert to JPEG)
            img_bytes = prepare_image_for_api(block_img)

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

            # Apply rate limiting
            estimated_tokens = self.special_blocks_estimated_tokens
            if not rate_limiter.wait_if_needed(estimated_tokens):
                return {
                    "type": block_info["type"],
                    "xywh": block_info["xywh"],
                    "content": "[DAILY_LIMIT_EXCEEDED]",
                    "analysis": "Daily rate limit exceeded",
                    "confidence": 0.0,
                    "error": "rate_limit_daily",
                    "error_message": "Daily rate limit exceeded",
                }

            logger.info("Requesting Gemini process_special_block (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Gemini API client became unavailable")
                return {
                    "type": block_info["type"],
                    "xywh": block_info["xywh"],
                    "content": "Gemini API not available",
                    "analysis": "Client not initialized",
                    "confidence": 0.0,
                }

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=cast(types.ContentListUnionDict, contents),
            )

            response_text = (response.text or "").strip()
            parsed_result = self._parse_gemini_response(response_text, block_info)

            return parsed_result

        except google_exceptions.ResourceExhausted as e:
            # 429 Rate limit errors (RESOURCE_EXHAUSTED)
            logger.error("Gemini rate limit exceeded: %s", e)
            return create_special_content_error(
                block_info, "[RATE_LIMIT_EXCEEDED]", "Rate limit exceeded",
                "gemini_rate_limit", str(e),
            )
        except google_exceptions.RetryError as e:
            # Retry failures (network issues, timeouts)
            logger.error("Gemini retry error: %s", e)
            return create_special_content_error(
                block_info, "[GEMINI_RETRY_FAILED]", "Retry failed",
                "gemini_retry_error", str(e),
            )
        except google_exceptions.GoogleAPIError as e:
            # Other Google API errors (4xx, 5xx)
            logger.error("Gemini API error: %s", e)
            return create_special_content_error(
                block_info, "[GEMINI_API_ERROR]", "Gemini API error",
                "gemini_api_error", str(e),
            )
        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
            logger.error("Unexpected error during Gemini special block processing: %s", e, exc_info=True)
            return create_special_content_error(
                block_info, "[GEMINI_PROCESSING_FAILED]", f"Processing failed: {e!s}",
                "unexpected_error", str(e),
            )

    def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Correct OCR text using Gemini API

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

            # Apply rate limiting
            estimated_tokens = len(text.split()) * 2  # Rough estimate based on input text
            if not rate_limiter.wait_if_needed(estimated_tokens):
                result = self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_DAILY_LIMIT_EXCEEDED]",
                    error_message="Daily rate limit exceeded",
                )
            else:
                logger.info("Requesting Gemini correct_text (model=%s)", self.gemini_model)
                client = self.client
                if client is None:
                    logger.warning("Gemini API client became unavailable")
                    result = self._text_correction_result(
                        text,
                        0.0,
                        error="[TEXT_CORRECTION_FAILED]",
                        error_message="Gemini client not initialized",
                    )
                else:
                    response = client.models.generate_content(
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
            logger.error("Unexpected error during text correction: %s", e, exc_info=True)
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

    def _parse_gemini_response(self, response_text: str, block_info: dict[str, Any]) -> dict[str, Any]:
        """Parse Gemini response for special blocks"""
        try:
            parsed = json.loads(response_text)

            result = {
                "type": block_info["type"],
                "xywh": block_info["xywh"],
                "confidence": block_info.get("confidence", 1.0),
            }

            if block_info["type"] == "table":
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
            logger.warning("Failed to parse Gemini JSON response, using as plain text")
            return {
                "type": block_info["type"],
                "xywh": block_info["xywh"],
                "content": response_text,
                "analysis": "Direct response (JSON parsing failed)",
                "confidence": block_info.get("confidence", 1.0),
            }

    def reload_client(self, api_key: str | None = None) -> bool:
        """
        Reload the Gemini API client (useful after API key updates)

        Args:
            api_key: New API key to use (optional)

        Returns:
            True if client was successfully reloaded
        """
        if api_key:
            self.api_key = api_key

        self.client = self._setup_gemini_api()
        return self.is_available()
