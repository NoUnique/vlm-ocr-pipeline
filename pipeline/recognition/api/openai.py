"""
OpenAI VLM API client for advanced OCR text extraction and processing.
Compatible with OpenAI and OpenRouter Vision Language Model APIs.
Handles text extraction, special content analysis, and text correction.
"""

import base64
import difflib
import gc
import io
import json
import logging
import os
from typing import Any

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI VLM API client for OCR text processing"""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, base_url: str | None = None):
        """
        Initialize OpenAI API client

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

        self.client: Any = self._setup_openai_client()  # Use Any to avoid strict type checking

    def _setup_openai_client(self) -> OpenAI | None:
        """Setup OpenAI API client"""
        try:
            if not self.api_key:
                logger.warning("OPENAI_API_KEY or OPENROUTER_API_KEY environment variable not set")
                return None

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
            logger.info("OpenAI API client initialized successfully (base_url: %s)", self.base_url or "default")
            return client
        except Exception as e:
            logger.error("Failed to initialize OpenAI API client: %s", e)
            return None

    def is_available(self) -> bool:
        """Check if OpenAI API client is available"""
        return self.client is not None

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 for OpenAI API"""
        # Resize image if too large
        h, w = image.shape[:2]
        max_dim = 1024

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image

        pil_image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG", quality=85, optimize=True)
        img_bytes = img_byte_arr.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    def extract_text(self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str) -> dict[str, Any]:
        """
        Extract text from region using OpenAI API

        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
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

            logger.info(
                "Requesting OpenAI extract_text (model=%s, base_url=%s)", self.model, self.base_url or "default"
            )
            client = self.client
            if client is None:
                logger.warning(
                    "OpenAI API client became unavailable (model=%s, base_url=%s)",
                    self.model,
                    self.base_url or "default",
                )
                return {"type": region_info["type"], "xywh": region_info["xywh"], "text": "", "confidence": 0.0}

            response = client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=2000, temperature=0.1
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

        except Exception as e:
            error_str = str(e)
            logger.error("OpenAI text extraction error: %s", e)

            # Handle rate limit errors specifically
            if "429" in error_str or "rate_limit" in error_str.lower():
                logger.error("Rate limit exceeded. Please wait before retrying or check your API quota.")
                return {
                    "type": region_info["type"],
                    "xywh": region_info["xywh"],
                    "text": "[RATE_LIMIT_EXCEEDED]",
                    "confidence": 0.0,
                    "error": "openai_rate_limit",
                    "error_message": "OpenAI API rate limit exceeded",
                }

            # Handle other OpenAI API errors
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "text": "[OPENAI_EXTRACTION_FAILED]",
                "confidence": 0.0,
                "error": "openai_api_error",
                "error_message": str(e),
            }

    def process_special_region(
        self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """
        Process special regions (tables, figures) with OpenAI API

        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for special content analysis

        Returns:
            Dictionary containing processed content and metadata
        """
        if not self.is_available():
            logger.warning(
                "OpenAI API client not initialized (model=%s, base_url=%s)", self.model, self.base_url or "default"
            )
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "OpenAI API not available",
                "analysis": "Client not initialized",
                "confidence": 0.0,
            }

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

            logger.info(
                "Requesting OpenAI process_special_region (model=%s, base_url=%s)",
                self.model,
                self.base_url or "default",
            )
            client = self.client
            if client is None:
                logger.warning(
                    "OpenAI API client became unavailable (model=%s, base_url=%s)",
                    self.model,
                    self.base_url or "default",
                )
                return {
                    "type": region_info["type"],
                    "xywh": region_info["xywh"],
                    "content": "OpenAI API not available",
                    "analysis": "Client not initialized",
                    "confidence": 0.0,
                }

            response = client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=3000, temperature=0.1
            )

            response_text = response.choices[0].message.content.strip()
            parsed_result = self._parse_openai_response(response_text, region_info)

            # Clean up
            del base64_image
            gc.collect()

            return parsed_result

        except Exception as e:
            error_str = str(e)
            logger.error("OpenAI special region processing error: %s", e)

            # Handle rate limit errors
            if "429" in error_str or "rate_limit" in error_str.lower():
                return {
                    "type": region_info["type"],
                    "xywh": region_info["xywh"],
                    "content": "[RATE_LIMIT_EXCEEDED]",
                    "analysis": "Rate limit exceeded",
                    "confidence": 0.0,
                    "error": "openai_rate_limit",
                    "error_message": "OpenAI API rate limit exceeded",
                }

            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": "[OPENAI_PROCESSING_FAILED]",
                "analysis": f"Processing failed: {str(e)}",
                "confidence": 0.0,
                "error": "openai_api_error",
                "error_message": str(e),
            }

    def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Correct OCR text using OpenAI API

        Args:
            text: Text to correct
            system_prompt: System instruction prompt
            user_prompt: User prompt with text formatting

        Returns:
            Dictionary containing corrected text and confidence
        """
        if not self.is_available() or not text:
            return self._text_correction_result(text, 0.0)

        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            logger.info(
                "Requesting OpenAI correct_text (model=%s, base_url=%s)", self.model, self.base_url or "default"
            )
            client = self.client
            if client is None:
                logger.warning(
                    "OpenAI API client became unavailable (model=%s, base_url=%s)",
                    self.model,
                    self.base_url or "default",
                )
                return self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_FAILED]",
                    error_message="OpenAI client not initialized",
                )

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=len(text.split()) * 3,  # Allow for expansion
                temperature=0.1,
            )

            corrected_text = response.choices[0].message.content.strip()

            similarity = difflib.SequenceMatcher(None, text, corrected_text).ratio()
            correction_ratio = 1.0 - similarity  # How much was changed (0.0 = no change, 1.0 = completely different)

            return self._text_correction_result(corrected_text, correction_ratio)

        except Exception as e:
            error_str = str(e)
            logger.error("Text correction error: %s", e)

            # Handle rate limit errors specifically
            if "429" in error_str or "rate_limit" in error_str.lower():
                logger.error("Rate limit exceeded during text correction")
                return self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]",
                    error_message=error_str,
                )

            # Handle service unavailable errors
            elif "503" in error_str or "unavailable" in error_str.lower():
                logger.error("Service unavailable during text correction")
                return self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_SERVICE_UNAVAILABLE]",
                    error_message=error_str,
                )

            # For other errors, return original text with error indicator
            else:
                logger.error("Text correction failed with other error")
                return self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_FAILED]",
                    error_message=error_str,
                )

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

    def _parse_openai_response(self, response_text: str, region_info: dict[str, Any]) -> dict[str, Any]:
        """Parse OpenAI response for special regions"""
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
            logger.warning("Failed to parse OpenAI JSON response, using as plain text")
            return {
                "type": region_info["type"],
                "xywh": region_info["xywh"],
                "content": response_text,
                "analysis": "Direct response (JSON parsing failed)",
                "confidence": region_info.get("confidence", 1.0),
            }

    def reload_client(self, api_key: str | None = None, base_url: str | None = None) -> bool:
        """
        Reload the OpenAI API client (useful after API key updates)

        Args:
            api_key: New API key to use (optional)
            base_url: New base URL to use (optional)

        Returns:
            True if client was successfully reloaded
        """
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url

        self.client = self._setup_openai_client()
        return self.is_available()
