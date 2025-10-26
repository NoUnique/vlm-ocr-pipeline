"""Integration tests for async API with real API keys.

NOTE: These tests require actual API keys and make real API calls.
Run with: pytest tests/test_integration_async.py -v -m integration

Requirements:
- OPENAI_API_KEY or GEMINI_API_KEY environment variable must be set
- Tests will consume API quota
- Tests may take longer due to actual API calls
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from pipeline.recognition.api.gemini_async import AsyncGeminiClient
from pipeline.recognition.api.openai_async import AsyncOpenAIClient

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def _create_test_image() -> np.ndarray:
    """Create a simple test image with white text on black background."""
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    # Add some white pixels to simulate text
    image[40:60, 50:250] = 255  # Horizontal white bar
    return image


class TestAsyncOpenAIIntegration:
    """Integration tests for AsyncOpenAIClient with real API."""

    def setup_method(self):
        """Setup for each test."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            pytest.skip("OPENAI_API_KEY not set - skipping OpenAI integration tests")

    @pytest.mark.anyio
    async def test_real_extract_text_single(self):
        """Test real API call for single text extraction."""
        client = AsyncOpenAIClient(model="gpt-4o")

        if not client.is_available():
            pytest.skip("OpenAI client not available")

        test_image = _create_test_image()
        region_info = {"type": "text", "xywh": [50, 40, 200, 20]}
        prompt = "Extract any text you see in this image."

        start_time = time.perf_counter()
        result = await client.extract_text(test_image, region_info, prompt)
        elapsed_time = time.perf_counter() - start_time

        # Assertions
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "text" in result, "Result should contain 'text' key"
        assert "type" in result, "Result should contain 'type' key"
        assert result["type"] == "text", "Type should match input"
        assert elapsed_time < 30.0, f"API call took too long: {elapsed_time:.2f}s"

        print(f"\n✅ Single extraction completed in {elapsed_time:.2f}s")
        print(f"   Extracted text: {result['text'][:100]}")

    @pytest.mark.anyio
    async def test_real_extract_text_batch_concurrent(self):
        """Test real API calls with concurrent batch processing."""
        client = AsyncOpenAIClient(model="gpt-4o")

        if not client.is_available():
            pytest.skip("OpenAI client not available")

        # Create multiple test regions
        test_image = _create_test_image()
        num_regions = 3
        regions = [
            (test_image, {"type": "text", "xywh": [50, 40, 200, 20]}, f"Extract text from region {i}")
            for i in range(num_regions)
        ]

        start_time = time.perf_counter()
        results = await client.extract_text_batch(regions, max_concurrent=3)
        elapsed_time = time.perf_counter() - start_time

        # Assertions
        assert len(results) == num_regions, f"Should return {num_regions} results"
        assert all(isinstance(r, dict) for r in results), "All results should be dicts"
        assert all("text" in r for r in results), "All results should have 'text' key"
        assert elapsed_time < 60.0, f"Batch processing took too long: {elapsed_time:.2f}s"

        # Calculate average time per request
        avg_time = elapsed_time / num_regions
        print(f"\n✅ Batch extraction completed in {elapsed_time:.2f}s")
        print(f"   Average per request: {avg_time:.2f}s")
        print(f"   Speedup from concurrency: ~{num_regions / elapsed_time * avg_time:.2f}x (estimated)")

    @pytest.mark.anyio
    async def test_real_text_correction(self):
        """Test real API call for text correction."""
        client = AsyncOpenAIClient(model="gpt-4o")

        if not client.is_available():
            pytest.skip("OpenAI client not available")

        test_text = "Ths is a sampel txt with som errors."
        system_prompt = "You are a text correction assistant. Fix spelling and grammar errors."
        user_prompt = f"Correct this text: {test_text}"

        start_time = time.perf_counter()
        corrected = await client.correct_text(test_text, system_prompt, user_prompt)
        elapsed_time = time.perf_counter() - start_time

        # Assertions
        assert isinstance(corrected, str), "Corrected text should be a string"
        assert len(corrected) > 0, "Corrected text should not be empty"
        assert elapsed_time < 30.0, f"Text correction took too long: {elapsed_time:.2f}s"

        print(f"\n✅ Text correction completed in {elapsed_time:.2f}s")
        print(f"   Original: {test_text}")
        print(f"   Corrected: {corrected}")


class TestAsyncGeminiIntegration:
    """Integration tests for AsyncGeminiClient with real API."""

    def setup_method(self):
        """Setup for each test."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            pytest.skip("GEMINI_API_KEY not set - skipping Gemini integration tests")

    @pytest.mark.anyio
    async def test_real_extract_text_single(self):
        """Test real Gemini API call for single text extraction."""
        client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

        if not client.is_available():
            pytest.skip("Gemini client not available")

        test_image = _create_test_image()
        region_info = {"type": "text", "xywh": [50, 40, 200, 20]}
        prompt = "Extract any text you see in this image."

        start_time = time.perf_counter()
        result = await client.extract_text(test_image, region_info, prompt)
        elapsed_time = time.perf_counter() - start_time

        # Assertions
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "text" in result, "Result should contain 'text' key"
        assert "type" in result, "Result should contain 'type' key"
        assert result["type"] == "text", "Type should match input"
        assert elapsed_time < 30.0, f"API call took too long: {elapsed_time:.2f}s"

        print(f"\n✅ Single extraction completed in {elapsed_time:.2f}s")
        print(f"   Extracted text: {result['text'][:100]}")

    @pytest.mark.anyio
    async def test_real_extract_text_batch_concurrent(self):
        """Test real Gemini API calls with concurrent batch processing."""
        client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

        if not client.is_available():
            pytest.skip("Gemini client not available")

        # Create multiple test regions
        test_image = _create_test_image()
        num_regions = 3
        regions = [
            (test_image, {"type": "text", "xywh": [50, 40, 200, 20]}, f"Extract text from region {i}")
            for i in range(num_regions)
        ]

        start_time = time.perf_counter()
        results = await client.extract_text_batch(regions, max_concurrent=3)
        elapsed_time = time.perf_counter() - start_time

        # Assertions
        assert len(results) == num_regions, f"Should return {num_regions} results"
        assert all(isinstance(r, dict) for r in results), "All results should be dicts"
        assert all("text" in r for r in results), "All results should have 'text' key"
        assert elapsed_time < 60.0, f"Batch processing took too long: {elapsed_time:.2f}s"

        # Calculate average time per request
        avg_time = elapsed_time / num_regions
        print(f"\n✅ Batch extraction completed in {elapsed_time:.2f}s")
        print(f"   Average per request: {avg_time:.2f}s")
        print(f"   Speedup from concurrency: ~{num_regions / elapsed_time * avg_time:.2f}x (estimated)")

    @pytest.mark.anyio
    async def test_real_text_correction(self):
        """Test real Gemini API call for text correction."""
        client = AsyncGeminiClient(gemini_model="gemini-2.5-flash")

        if not client.is_available():
            pytest.skip("Gemini client not available")

        test_text = "Ths is a sampel txt with som errors."
        system_prompt = "You are a text correction assistant. Fix spelling and grammar errors."
        user_prompt = f"Correct this text: {test_text}"

        start_time = time.perf_counter()
        result = await client.correct_text(test_text, system_prompt, user_prompt)
        elapsed_time = time.perf_counter() - start_time

        # Assertions
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "corrected_text" in result, "Result should contain 'corrected_text' key"
        assert len(result["corrected_text"]) > 0, "Corrected text should not be empty"
        assert elapsed_time < 30.0, f"Text correction took too long: {elapsed_time:.2f}s"

        print(f"\n✅ Text correction completed in {elapsed_time:.2f}s")
        print(f"   Original: {test_text}")
        print(f"   Corrected: {result['corrected_text']}")


class TestAsyncRateLimiterIntegration:
    """Integration tests for AsyncRateLimitManager with real API."""

    @pytest.mark.anyio
    async def test_real_rate_limiter_gemini(self):
        """Test rate limiter with actual Gemini API calls."""
        from pipeline.recognition.api.ratelimit_async import AsyncRateLimitManager

        # Create new rate limiter instance
        limiter = AsyncRateLimitManager()
        limiter.set_tier_and_model("free", "gemini-2.5-flash")

        # Make multiple requests to test rate limiting
        request_times = []
        for i in range(5):
            start = time.perf_counter()
            can_proceed = await limiter.wait_if_needed(estimated_tokens=1000)
            elapsed = time.perf_counter() - start
            request_times.append(elapsed)

            assert can_proceed is True, f"Request {i} should be allowed"

        # Get status
        status = await limiter.get_status()

        print(f"\n✅ Rate limiter test completed")
        print(f"   Requests made: 5")
        print(f"   Current RPM: {status['current']['rpm']}")
        print(f"   Wait times: {[f'{t:.3f}s' for t in request_times]}")
