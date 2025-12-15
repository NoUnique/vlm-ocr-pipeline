"""Tests for text recognizers and API clients."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import openai
from google.api_core import exceptions as google_exceptions

from pipeline.recognition.api.gemini import GeminiClient
from pipeline.recognition.api.openai import OpenAIClient


class TestOpenAIClient:
    """Tests for OpenAI API client."""

    def test_client_initialization_with_api_key(self):
        """Test OpenAI client initializes with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4o")

            assert client.model == "gpt-4o"
            assert client.api_key == "test-key"

    def test_client_initialization_without_api_key(self):
        """Test OpenAI client handles missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = OpenAIClient(model="gpt-4o")

            assert not client.is_available()

    def test_extract_text_returns_proper_structure(self):
        """Test extract_text returns properly structured result."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4o")

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Extracted text from block"

            with patch.object(client, "client") as mock_client:
                mock_client.chat.completions.create.return_value = mock_response

                block_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {
                    "type": "text",
                    "xywh": [10, 20, 80, 50],
                    "confidence": 0.95,
                }
                result = client.extract_text(block_image, block_info, "Extract text from this image")

                assert result["type"] == "text"  # type: ignore[typeddict-item]
                assert result["xywh"] == [10, 20, 80, 50]  # type: ignore[typeddict-item]
                assert result["text"] == "Extracted text from block"  # type: ignore[typeddict-item]
                assert result["confidence"] == 0.95  # type: ignore[typeddict-item]

    def test_extract_text_handles_rate_limit_error(self):
        """Test extract_text handles rate limit errors gracefully."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4o")

            # Mock rate limit error - create actual exception
            with patch.object(client, "client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 429
                # Create real exception with minimal required params
                mock_client.chat.completions.create.side_effect = openai.RateLimitError(
                    message="Rate limit exceeded",
                    response=mock_response,
                    body=None,
                )

                block_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {"type": "text", "xywh": [10, 20, 80, 50]}
                result = client.extract_text(block_image, block_info, "Extract text")

                assert result["text"] == "[RATE_LIMIT_EXCEEDED]"  # type: ignore[typeddict-item]
                assert result["error"] == "openai_rate_limit"

    def test_correct_text_calculates_correction_ratio(self):
        """Test correct_text calculates correction ratio."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4o")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "corrected text here"

            with patch.object(client, "client") as mock_client:
                mock_client.chat.completions.create.return_value = mock_response

                result = client.correct_text(
                    "original text here",
                    "You are a text corrector",
                    "Correct this text: original text here",
                )

                assert "corrected_text" in result
                assert "correction_ratio" in result
                assert 0.0 <= result["correction_ratio"] <= 1.0


class TestGeminiClient:
    """Tests for Gemini API client."""

    def test_client_initialization_with_api_key(self):
        """Test Gemini client initializes with API key."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            assert client.gemini_model == "gemini-2.5-flash"
            assert client.api_key == "test-key"

    def test_client_initialization_without_api_key(self):
        """Test Gemini client handles missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            assert not client.is_available()

    def test_extract_text_returns_proper_structure(self):
        """Test extract_text returns properly structured result."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            # Mock the Gemini client
            mock_response = MagicMock()
            mock_response.text = "Extracted text from block"

            with patch.object(client, "client") as mock_client:
                mock_client.models.generate_content.return_value = mock_response

                block_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {
                    "type": "text",
                    "xywh": [10, 20, 80, 50],
                    "confidence": 0.95,
                }

                # Mock rate limiter
                with patch("pipeline.recognition.api.gemini.rate_limiter") as mock_limiter:
                    mock_limiter.wait_if_needed.return_value = True

                    result = client.extract_text(block_image, block_info, "Extract text from this image")

                    assert result["type"] == "text"  # type: ignore[typeddict-item]
                    assert result["xywh"] == [10, 20, 80, 50]  # type: ignore[typeddict-item]
                    assert result["text"] == "Extracted text from block"  # type: ignore[typeddict-item]
                    assert result["confidence"] == 0.95  # type: ignore[typeddict-item]

    def test_extract_text_handles_resource_exhausted(self):
        """Test extract_text handles resource exhausted errors."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            # Mock resource exhausted error
            with patch.object(client, "client") as mock_client:
                mock_client.models.generate_content.side_effect = google_exceptions.ResourceExhausted("Rate limit")

                block_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {"type": "text", "xywh": [10, 20, 80, 50]}

                with patch("pipeline.recognition.api.gemini.rate_limiter") as mock_limiter:
                    mock_limiter.wait_if_needed.return_value = True

                    result = client.extract_text(block_image, block_info, "Extract text")

                    assert result["text"] == "[RATE_LIMIT_EXCEEDED]"  # type: ignore[typeddict-item]
                    assert result["error"] == "gemini_rate_limit"  # type: ignore[typeddict-item]

    def test_correct_text_calculates_correction_ratio(self):
        """Test correct_text calculates correction ratio."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            mock_response = MagicMock()
            mock_response.text = "corrected text here"

            with patch.object(client, "client") as mock_client:
                mock_client.models.generate_content.return_value = mock_response

                with patch("pipeline.recognition.api.gemini.rate_limiter") as mock_limiter:
                    mock_limiter.wait_if_needed.return_value = True

                    result = client.correct_text(
                        "original text here",
                        "You are a text corrector",
                        "Correct this text: original text here",
                    )

                    assert "corrected_text" in result
                    assert "correction_ratio" in result
                    assert 0.0 <= result["correction_ratio"] <= 1.0

    def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Test that rate limiter prevents API calls when daily limit exceeded."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            block_image = np.zeros((100, 100, 3), dtype=np.uint8)
            block_info = {"type": "text", "xywh": [10, 20, 80, 50]}

            # Mock rate limiter to return False (limit exceeded)
            with patch("pipeline.recognition.api.gemini.rate_limiter") as mock_limiter:
                mock_limiter.wait_if_needed.return_value = False

                result = client.extract_text(block_image, block_info, "Extract text")

                assert result["text"] == "[DAILY_LIMIT_EXCEEDED]"  # type: ignore[typeddict-item]
                assert result["error"] == "rate_limit_daily"  # type: ignore[typeddict-item]


class TestAPIClientExceptionHandling:
    """Tests for API client exception handling."""

    def test_openai_handles_connection_errors(self):
        """Test OpenAI client handles connection errors."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4o")

            with patch.object(client, "client") as mock_client:
                # Create real exception with required request parameter
                mock_request = MagicMock()
                mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
                    message="Connection failed", request=mock_request
                )

                block_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {"type": "text", "xywh": [10, 20, 80, 50]}
                result = client.extract_text(block_image, block_info, "Extract")

                assert "[OPENAI_CONNECTION_ERROR]" in result["text"]
                assert result["error"] == "openai_connection_error"

    def test_gemini_handles_retry_errors(self):
        """Test Gemini client handles retry errors."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(gemini_model="gemini-2.5-flash")

            with patch.object(client, "client") as mock_client:
                mock_client.models.generate_content.side_effect = google_exceptions.RetryError(
                    "Retry failed", cause=Exception("Network error")
                )

                block_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {"type": "text", "xywh": [10, 20, 80, 50]}

                with patch("pipeline.recognition.api.gemini.rate_limiter") as mock_limiter:
                    mock_limiter.wait_if_needed.return_value = True

                    result = client.extract_text(block_image, block_info, "Extract")

                    assert "[GEMINI_RETRY_FAILED]" in result["text"]  # type: ignore[typeddict-item]
                    assert result["error"] == "gemini_retry_error"  # type: ignore[typeddict-item]

    def test_openai_handles_server_errors(self):
        """Test OpenAI client handles server errors."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4o")

            with patch.object(client, "client") as mock_client:
                # Create real exception
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_client.chat.completions.create.side_effect = openai.InternalServerError(
                    message="Internal server error",
                    response=mock_response,
                    body=None,
                )

                result = client.correct_text("text", "system", "user")

                assert "[TEXT_CORRECTION_SERVICE_UNAVAILABLE]" in result["error"]
                assert result["correction_ratio"] == 0.0
