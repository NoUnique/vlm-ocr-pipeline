"""API integration tests.

Run with:
  - Gemini: uv run pytest tests/integration/test_env_api.py --run-api-gemini
  - OpenAI: uv run pytest tests/integration/test_env_api.py --run-api-openai
  - All APIs: uv run pytest tests/integration/test_env_api.py --run-api
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.types import BBox, Block


@pytest.mark.api_gemini
class TestGeminiAPI:
    """Tests for Gemini API integration."""

    def test_gemini_api_key_available(self, has_gemini_api: bool):
        """Test that Gemini API key is available."""
        assert has_gemini_api is True, "GEMINI_API_KEY should be set"

    def test_gemini_recognizer_creation(self):
        """Test creating Gemini recognizer."""
        from pipeline.recognition import create_recognizer

        recognizer = create_recognizer("gemini-2.5-flash")
        assert recognizer is not None
        assert recognizer.backend == "gemini"

    def test_gemini_text_extraction(self, sample_image):
        """Test Gemini text extraction from image."""
        from pipeline.recognition import create_recognizer

        recognizer = create_recognizer("gemini-2.5-flash")
        blocks = [Block(type="text", bbox=BBox(50, 50, 750, 550), order=0)]
        result = recognizer.process_blocks(sample_image, blocks)
        assert len(result) == 1
        print(f"Gemini extracted text: '{result[0].text}'")

    def test_gemini_text_correction(self):
        """Test Gemini text correction."""
        from pipeline.recognition import create_recognizer

        recognizer = create_recognizer("gemini-2.5-flash")
        test_text = "Ths is a tset of txt correcton."
        result = recognizer.correct_text(test_text)
        assert isinstance(result, dict)
        assert "corrected_text" in result
        print(f"Original: {test_text}")
        print(f"Corrected: {result['corrected_text']}")

    def test_gemini_rate_limiter(self):
        """Test Gemini rate limiter functionality."""
        from pipeline.recognition.api.ratelimit import rate_limiter

        rate_limiter.set_tier_and_model("free", "gemini-2.5-flash")
        status = rate_limiter.get_status()
        assert status["tier"] == "free"
        assert "limits" in status


@pytest.mark.api_gemini
class TestGeminiAsyncAPI:
    """Tests for Gemini async API integration."""

    @pytest.mark.anyio
    async def test_gemini_async_client(self):
        """Test Gemini async client initialization."""
        from pipeline.recognition.api.gemini_async import AsyncGeminiClient

        client = AsyncGeminiClient()
        assert client is not None


@pytest.mark.api_openai
class TestOpenAIAPI:
    """Tests for OpenAI API integration."""

    def test_openai_api_key_available(self, has_openai_api: bool):
        """Test that OpenAI API key is available."""
        assert has_openai_api is True, "OPENAI_API_KEY should be set"

    def test_openai_recognizer_creation(self):
        """Test creating OpenAI recognizer."""
        from pipeline.recognition import create_recognizer

        recognizer = create_recognizer("gpt-4o")
        assert recognizer is not None

    def test_openai_text_extraction(self, sample_image):
        """Test OpenAI text extraction from image."""
        from pipeline.recognition import create_recognizer

        recognizer = create_recognizer("gpt-4o")
        blocks = [Block(type="text", bbox=BBox(50, 50, 750, 550), order=0)]
        result = recognizer.process_blocks(sample_image, blocks)
        assert len(result) == 1

    def test_openai_text_correction(self):
        """Test OpenAI text correction."""
        from pipeline.recognition import create_recognizer

        recognizer = create_recognizer("gpt-4o")
        test_text = "Ths is a tset of txt correcton."
        result = recognizer.correct_text(test_text)
        assert isinstance(result, dict)
        assert "corrected_text" in result


@pytest.mark.api_openai
class TestOpenAIAsyncAPI:
    """Tests for OpenAI async API integration."""

    @pytest.mark.anyio
    async def test_openai_async_client(self):
        """Test OpenAI async client initialization."""
        from pipeline.recognition.api.openai_async import AsyncOpenAIClient

        client = AsyncOpenAIClient()
        assert client is not None


@pytest.mark.api_gemini
@pytest.mark.api_openai
class TestAPIComparison:
    """Tests comparing different API backends."""

    def test_compare_text_extraction(self, sample_image):
        """Compare text extraction between Gemini and OpenAI."""
        from pipeline.recognition import create_recognizer

        blocks = [Block(type="text", bbox=BBox(50, 50, 750, 550), order=0)]

        gemini = create_recognizer("gemini-2.5-flash")
        gemini_result = gemini.process_blocks(sample_image, blocks.copy())

        openai = create_recognizer("gpt-4o")
        openai_result = openai.process_blocks(sample_image, blocks.copy())

        assert len(gemini_result) == 1
        assert len(openai_result) == 1


@pytest.mark.api_gemini
class TestRealDocumentGemini:
    """Tests with real document samples using Gemini."""

    def test_process_sample_pdf_page(self, fixtures_dir: Path):
        """Test processing a real PDF page with Gemini."""
        sample_pdf = fixtures_dir / "sample.pdf"
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found in fixtures")

        from pipeline.io.input.pdf import load_pdf_page
        from pipeline.layout.detection import create_detector
        from pipeline.recognition import create_recognizer

        page_image = load_pdf_page(sample_pdf, page_num=1, dpi=150)
        detector = create_detector("paddleocr-doclayout-v2")
        blocks = detector.detect(page_image)

        if not blocks:
            pytest.skip("No blocks detected in sample PDF")

        recognizer = create_recognizer("gemini-2.5-flash")
        result_blocks = recognizer.process_blocks(page_image, blocks[:5])
        assert all(block.text is not None for block in result_blocks)
