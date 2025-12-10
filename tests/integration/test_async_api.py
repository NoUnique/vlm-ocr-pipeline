"""Tests for async API clients (OpenAI and Gemini)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pipeline.recognition.api.gemini_async import AsyncGeminiClient
from pipeline.recognition.api.openai_async import AsyncOpenAIClient


class TestAsyncOpenAIClient:
    """Test AsyncOpenAIClient functionality."""

    @pytest.mark.anyio
    async def test_client_initialization_with_api_key(self):
        """Test async client initializes with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = AsyncOpenAIClient(model="gpt-4o")
            assert client.is_available()
            assert client.model == "gpt-4o"

    @pytest.mark.anyio
    async def test_client_initialization_without_api_key(self):
        """Test async client handles missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = AsyncOpenAIClient(model="gpt-4o")
            assert not client.is_available()
            assert client.client is None

    @pytest.mark.anyio
    async def test_extract_text_returns_proper_structure(self):
        """Test async extract_text returns correct structure."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = AsyncOpenAIClient(model="gpt-4o")

            # Mock async API response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Extracted text"))]

            with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_response

                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {"type": "text", "xywh": [10, 10, 50, 20]}
                prompt = "Extract text from this image."

                result = await client.extract_text(test_image, block_info, prompt)

                assert "text" in result
                assert result["text"] == "Extracted text"
                assert result["type"] == "text"
                assert result["xywh"] == [10, 10, 50, 20]
                assert "confidence" in result
                mock_create.assert_awaited_once()

    @pytest.mark.anyio
    async def test_extract_text_batch_processes_multiple_blocks(self):
        """Test extract_text_batch processes multiple blocks concurrently."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = AsyncOpenAIClient(model="gpt-4o")

            # Mock async API responses
            mock_response1 = MagicMock()
            mock_response1.choices = [MagicMock(message=MagicMock(content="Text 1"))]
            mock_response2 = MagicMock()
            mock_response2.choices = [MagicMock(message=MagicMock(content="Text 2"))]

            with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                mock_create.side_effect = [mock_response1, mock_response2]

                # Prepare batch data
                test_image1 = np.zeros((100, 100, 3), dtype=np.uint8)
                test_image2 = np.zeros((100, 100, 3), dtype=np.uint8)
                regions = [
                    (test_image1, {"type": "text", "xywh": [10, 10, 50, 20]}, "Extract text 1"),
                    (test_image2, {"type": "title", "xywh": [10, 50, 50, 20]}, "Extract text 2"),
                ]

                results = await client.extract_text_batch(regions, max_concurrent=2)

                assert len(results) == 2
                assert results[0]["text"] == "Text 1"
                assert results[1]["text"] == "Text 2"
                assert mock_create.await_count == 2

    @pytest.mark.anyio
    async def test_extract_text_batch_handles_exceptions(self):
        """Test extract_text_batch handles exceptions gracefully."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = AsyncOpenAIClient(model="gpt-4o")

            # Mock one success and one failure
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]

            with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                mock_create.side_effect = [mock_response, Exception("API Error")]

                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                regions = [
                    (test_image, {"type": "text", "xywh": [10, 10, 50, 20]}, "Extract 1"),
                    (test_image, {"type": "text", "xywh": [10, 50, 50, 20]}, "Extract 2"),
                ]

                results = await client.extract_text_batch(regions, max_concurrent=2)

                assert len(results) == 2
                assert results[0]["text"] == "Success"
                # Second result should contain error (extract_text handles it)
                assert results[1]["text"] == "[EXTRACTION_ERROR]"
                assert "error" in results[1]

    @pytest.mark.anyio
    async def test_correct_text_async(self):
        """Test async text correction."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = AsyncOpenAIClient(model="gpt-4o")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Corrected text"))]

            with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_response

                result = await client.correct_text("Original text", "System prompt", "User prompt with {text}")

                assert result == "Corrected text"
                mock_create.assert_awaited_once()

    @pytest.mark.anyio
    async def test_close_client(self):
        """Test async client cleanup."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = AsyncOpenAIClient(model="gpt-4o")

            with patch.object(client.client, "close", new_callable=AsyncMock) as mock_close:
                await client.close()
                mock_close.assert_awaited_once()


class TestAsyncGeminiClient:
    """Test AsyncGeminiClient functionality."""

    @pytest.mark.anyio
    async def test_client_initialization_with_api_key(self):
        """Test async Gemini client initializes with API key."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")
            assert client.is_available()
            assert client.gemini_model == "gemini-2.5-flash"

    @pytest.mark.anyio
    async def test_client_initialization_without_api_key(self):
        """Test async Gemini client handles missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")
            assert not client.is_available()
            assert client.client is None

    @pytest.mark.anyio
    async def test_extract_text_returns_proper_structure(self):
        """Test async Gemini extract_text returns correct structure."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

            # Mock async API response
            mock_response = MagicMock()
            mock_response.text = "Extracted text from Gemini"

            with patch.object(client.client.aio.models, "generate_content", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = mock_response

                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                block_info = {"type": "text", "xywh": [10, 10, 50, 20]}
                prompt = "Extract text from this image."

                result = await client.extract_text(test_image, block_info, prompt)

                assert "text" in result
                assert result["text"] == "Extracted text from Gemini"
                assert result["type"] == "text"
                assert result["xywh"] == [10, 10, 50, 20]
                mock_gen.assert_awaited_once()

    @pytest.mark.anyio
    async def test_extract_text_batch_processes_multiple_blocks(self):
        """Test Gemini extract_text_batch processes multiple blocks concurrently."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

            # Mock async API responses
            mock_response1 = MagicMock()
            mock_response1.text = "Gemini text 1"
            mock_response2 = MagicMock()
            mock_response2.text = "Gemini text 2"

            with patch.object(client.client.aio.models, "generate_content", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = [mock_response1, mock_response2]

                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                regions = [
                    (test_image, {"type": "text", "xywh": [10, 10, 50, 20]}, "Extract 1"),
                    (test_image, {"type": "title", "xywh": [10, 50, 50, 20]}, "Extract 2"),
                ]

                results = await client.extract_text_batch(regions, max_concurrent=2)

                assert len(results) == 2
                assert results[0]["text"] == "Gemini text 1"
                assert results[1]["text"] == "Gemini text 2"
                assert mock_gen.await_count == 2

    @pytest.mark.anyio
    async def test_process_special_block_batch(self):
        """Test Gemini process_special_block_batch processes multiple special blocks."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

            # Mock async API responses
            mock_response1 = MagicMock()
            mock_response1.text = '{"markdown_table": "Table content"}'
            mock_response2 = MagicMock()
            mock_response2.text = '{"description": "Figure content"}'

            with patch.object(client.client.aio.models, "generate_content", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = [mock_response1, mock_response2]

                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                regions = [
                    (test_image, {"type": "table", "xywh": [10, 10, 50, 20]}, "Analyze table"),
                    (test_image, {"type": "figure", "xywh": [10, 50, 50, 20]}, "Analyze figure"),
                ]

                results = await client.process_special_block_batch(regions, max_concurrent=2)

                assert len(results) == 2
                assert results[0]["content"] == "Table content"
                assert results[1]["content"] == "Figure content"
                assert mock_gen.await_count == 2

    @pytest.mark.anyio
    async def test_correct_text_async(self):
        """Test async Gemini text correction."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

            mock_response = MagicMock()
            mock_response.text = "Corrected by Gemini"

            with patch.object(client.client.aio.models, "generate_content", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = mock_response

                result = await client.correct_text("Original text", "System prompt", "User prompt with text")

                assert result["corrected_text"] == "Corrected by Gemini"
                assert "correction_ratio" in result
                mock_gen.assert_awaited_once()


class TestAsyncRateLimiter:
    """Test AsyncRateLimitManager functionality."""

    @pytest.mark.anyio
    async def test_wait_if_needed_allows_request(self):
        """Test async rate limiter allows requests within limits."""
        from pipeline.recognition.api.ratelimit_async import AsyncRateLimitManager

        # Create new instance for testing
        limiter = AsyncRateLimitManager()
        limiter.set_tier_and_model("tier1", "gemini-2.5-flash")

        # Should allow request within limits
        can_proceed = await limiter.wait_if_needed(estimated_tokens=1000)
        assert can_proceed is True

    @pytest.mark.anyio
    async def test_concurrent_requests_respect_limits(self):
        """Test concurrent async requests respect rate limits."""
        from pipeline.recognition.api.ratelimit_async import AsyncRateLimitManager

        limiter = AsyncRateLimitManager()
        limiter.set_tier_and_model("free", "gemini-2.5-flash")

        async def make_request():
            return await limiter.wait_if_needed(estimated_tokens=100)

        # Make multiple concurrent requests
        results = await asyncio.gather(*[make_request() for _ in range(5)])

        # All should succeed (within limits)
        assert all(results)

    @pytest.mark.anyio
    async def test_get_status_async(self):
        """Test async status retrieval."""
        from pipeline.recognition.api.ratelimit_async import AsyncRateLimitManager

        # Create fresh instance to avoid state from other tests
        limiter = AsyncRateLimitManager.__new__(AsyncRateLimitManager)
        limiter._initialized = False
        limiter.__init__()
        limiter.set_tier_and_model("tier1", "gemini-2.5-flash")

        # Make a request first
        await limiter.wait_if_needed(estimated_tokens=500)

        # Get status
        status = await limiter.get_status()

        assert "tier" in status
        assert "model" in status
        assert "limits" in status
        assert "current" in status
        assert status["tier"] == "tier1"
        # Should be 1 since we created a fresh instance
        assert status["current"]["rpm"] >= 1
