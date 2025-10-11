"""
Google Gemini VLM API client for advanced OCR text extraction and processing.
Handles text extraction, special content analysis, and text correction using Gemini Vision Language Models.
"""

import difflib
import gc
import io
import json
import logging
import os
from typing import Any, cast

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from .ratelimit import rate_limiter

logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini VLM API client for OCR text processing"""

    def __init__(self, gemini_model: str = "gemini-2.5-flash", api_key: str | None = None):
        """
        Initialize Gemini API client

        Args:
            gemini_model: Gemini model to use
            api_key: Gemini API key (if not provided, reads from environment)
        """
        self.gemini_model = gemini_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = self._setup_gemini_api()

    def _setup_gemini_api(self) -> genai.Client | None:
        """Setup Gemini API client"""
        try:
            if not self.api_key:
                logger.warning("GEMINI_API_KEY environment variable not set")
                return None

            client = genai.Client(api_key=self.api_key)
            logger.info("Gemini API client initialized successfully")
            return client
        except Exception as e:
            logger.error("Failed to initialize Gemini API client: %s", e)
            return None

    def is_available(self) -> bool:
        """Check if Gemini API client is available"""
        return self.client is not None

    def extract_text(self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str) -> dict[str, Any]:
        """
        Extract text from region using Gemini API

        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for text extraction

        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_available():
            logger.warning("Gemini API client not initialized")
            return {"type": region_info["type"], "coords": region_info["coords"], "text": "", "confidence": 0.0}

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

            # Apply rate limiting
            estimated_tokens = 2000  # Estimated tokens for image + text
            if not rate_limiter.wait_if_needed(estimated_tokens):
                return {
                    "type": region_info["type"],
                    "coords": region_info["coords"],
                    "text": "[DAILY_LIMIT_EXCEEDED]",
                    "confidence": 0.0,
                    "error": "rate_limit_daily",
                    "error_message": "Daily rate limit exceeded",
                }

            logger.info("Requesting Gemini extract_text (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Gemini API client became unavailable")
                return {
                    "type": region_info["type"],
                    "coords": region_info["coords"],
                    "text": "",
                    "confidence": 0.0,
                    "error": "client_unavailable",
                    "error_message": "Gemini client not initialized",
                }

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=cast(types.ContentListUnionDict, contents),
            )

            del pil_image, img_byte_arr, img_bytes, region_img_resized
            gc.collect()

            text = (response.text or "").strip()

            result = {
                "type": region_info["type"],
                "coords": region_info["coords"],
                "text": text,
                "confidence": region_info.get("confidence", 1.0),
            }

            del response
            gc.collect()

            return result

        except Exception as e:
            error_str = str(e)
            logger.error("Gemini text extraction error: %s", e)

            # Handle rate limit errors specifically
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.error("Rate limit exceeded. Please wait before retrying or check your API quota.")
                return {
                    "type": region_info["type"],
                    "coords": region_info["coords"],
                    "text": "[RATE_LIMIT_EXCEEDED]",
                    "confidence": 0.0,
                    "error": "gemini_rate_limit",
                    "error_message": "Gemini API rate limit exceeded",
                }

            # Handle other Gemini API errors
            return {
                "type": region_info["type"],
                "coords": region_info["coords"],
                "text": "[GEMINI_EXTRACTION_FAILED]",
                "confidence": 0.0,
                "error": "gemini_api_error",
                "error_message": str(e),
            }

    def process_special_region(
        self, region_img: np.ndarray, region_info: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """
        Process special regions (tables, figures) with Gemini API

        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for special content analysis

        Returns:
            Dictionary containing processed content and metadata
        """
        if not self.is_available():
            logger.warning("Gemini API client not initialized")
            return {
                "type": region_info["type"],
                "coords": region_info["coords"],
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

            # Apply rate limiting
            estimated_tokens = 2500  # Estimated tokens for special content analysis
            if not rate_limiter.wait_if_needed(estimated_tokens):
                return {
                    "type": region_info["type"],
                    "coords": region_info["coords"],
                    "content": "[DAILY_LIMIT_EXCEEDED]",
                    "analysis": "Daily rate limit exceeded",
                    "confidence": 0.0,
                    "error": "rate_limit_daily",
                    "error_message": "Daily rate limit exceeded",
                }

            logger.info("Requesting Gemini process_special_region (model=%s)", self.gemini_model)
            client = self.client
            if client is None:
                logger.warning("Gemini API client became unavailable")
                return {
                    "type": region_info["type"],
                    "coords": region_info["coords"],
                    "content": "Gemini API not available",
                    "analysis": "Client not initialized",
                    "confidence": 0.0,
                }

            response = client.models.generate_content(
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

        except Exception as e:
            error_str = str(e)
            logger.error("Gemini special region processing error: %s", e)

            # Handle rate limit errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return {
                    "type": region_info["type"],
                    "coords": region_info["coords"],
                    "content": "[RATE_LIMIT_EXCEEDED]",
                    "analysis": "Rate limit exceeded",
                    "confidence": 0.0,
                    "error": "gemini_rate_limit",
                    "error_message": "Gemini API rate limit exceeded",
                }

            return {
                "type": region_info["type"],
                "coords": region_info["coords"],
                "content": "[GEMINI_PROCESSING_FAILED]",
                "analysis": f"Processing failed: {str(e)}",
                "confidence": 0.0,
                "error": "gemini_api_error",
                "error_message": str(e),
            }

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
                    confidence = difflib.SequenceMatcher(None, text, corrected_text).ratio()
                    result = self._text_correction_result(corrected_text, confidence)

        except Exception as e:
            error_str = str(e)
            logger.error("Text correction error: %s", e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.error("Rate limit exceeded during text correction")
                result = self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]",
                    error_message=error_str,
                )
            elif "503" in error_str or "UNAVAILABLE" in error_str:
                logger.error("Service unavailable during text correction")
                result = self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_SERVICE_UNAVAILABLE]",
                    error_message=error_str,
                )
            else:
                logger.error("Text correction failed with other error")
                result = self._text_correction_result(
                    text,
                    0.0,
                    error="[TEXT_CORRECTION_FAILED]",
                    error_message=error_str,
                )

        return result

    def _text_correction_result(
        self,
        corrected_text: str,
        confidence: float,
        error: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"corrected_text": corrected_text, "confidence": confidence}
        if error:
            result["error"] = error
        if error_message:
            result["error_message"] = error_message
        return result

    def _parse_gemini_response(self, response_text: str, region_info: dict[str, Any]) -> dict[str, Any]:
        """Parse Gemini response for special regions"""
        try:
            parsed = json.loads(response_text)

            result = {
                "type": region_info["type"],
                "coords": region_info["coords"],
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
            logger.warning("Failed to parse Gemini JSON response, using as plain text")
            return {
                "type": region_info["type"],
                "coords": region_info["coords"],
                "content": response_text,
                "analysis": "Direct response (JSON parsing failed)",
                "confidence": region_info.get("confidence", 1.0),
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
