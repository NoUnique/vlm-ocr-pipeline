"""Tests for BaseRecognizer abstract class."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from pipeline.exceptions import RecognitionError
from pipeline.recognition.base import BaseRecognizer
from pipeline.types import BBox, Block


class ConcreteRecognizer(BaseRecognizer):
    """Concrete implementation for testing."""

    name = "test-recognizer"
    supports_correction = True

    def __init__(self, return_blocks: list[Block] | None = None, correction_result: str | dict | None = None):
        self._return_blocks = return_blocks
        self._correction_result = correction_result

    def _process_blocks_impl(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        if self._return_blocks is not None:
            return self._return_blocks
        # Default: add text to each block
        result = []
        for block in blocks:
            result.append(replace(block, text=f"Text for {block.type}"))
        return result

    def _correct_text_impl(self, text: str) -> str | dict:
        if self._correction_result is not None:
            return self._correction_result
        return f"[Corrected] {text}"


class TestBaseRecognizerInit:
    """Tests for BaseRecognizer initialization."""

    def test_default_attributes(self):
        """Test default attribute values."""
        recognizer = ConcreteRecognizer()
        
        assert recognizer.name == "test-recognizer"
        assert recognizer.supports_correction is True
        assert recognizer.supports_batch is False


class TestBaseRecognizerProcessBlocks:
    """Tests for process_blocks method."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_blocks(self):
        """Create sample blocks."""
        return [
            Block(type="text", bbox=BBox(10, 10, 50, 50)),
            Block(type="title", bbox=BBox(10, 60, 90, 90)),
        ]

    def test_process_blocks_empty(self, sample_image):
        """Test processing empty block list."""
        recognizer = ConcreteRecognizer()
        result = recognizer.process_blocks(sample_image, [])
        assert result == []

    def test_process_blocks_adds_text(self, sample_image, sample_blocks):
        """Test that processing adds text to blocks."""
        recognizer = ConcreteRecognizer()
        result = recognizer.process_blocks(sample_image, sample_blocks)
        
        assert len(result) == 2
        assert result[0].text == "Text for text"
        assert result[1].text == "Text for title"

    def test_process_blocks_none_image(self, sample_blocks):
        """Test processing with None image."""
        recognizer = ConcreteRecognizer()
        with pytest.raises((ValueError, RecognitionError), match="Image cannot be None"):
            recognizer.process_blocks(None, sample_blocks)  # type: ignore[arg-type]

    def test_process_blocks_none_blocks(self, sample_image):
        """Test processing with None blocks."""
        recognizer = ConcreteRecognizer()
        with pytest.raises((ValueError, RecognitionError), match="Blocks cannot be None"):
            recognizer.process_blocks(sample_image, None)  # type: ignore[arg-type]


class TestBaseRecognizerCorrectText:
    """Tests for correct_text method."""

    def test_correct_text_basic(self):
        """Test basic text correction."""
        recognizer = ConcreteRecognizer()
        result = recognizer.correct_text("Hello World")
        assert result == "[Corrected] Hello World"

    def test_correct_text_empty(self):
        """Test correction of empty text."""
        recognizer = ConcreteRecognizer()
        result = recognizer.correct_text("")
        assert result == ""

    def test_correct_text_whitespace_only(self):
        """Test correction of whitespace-only text."""
        recognizer = ConcreteRecognizer()
        result = recognizer.correct_text("   ")
        assert result == "   "

    def test_correct_text_none(self):
        """Test correction with None."""
        recognizer = ConcreteRecognizer()
        with pytest.raises((ValueError, RecognitionError), match="Text cannot be None"):
            recognizer.correct_text(None)  # type: ignore[arg-type]

    def test_correct_text_dict_result(self):
        """Test correction returning dict result."""
        recognizer = ConcreteRecognizer(
            correction_result={"corrected_text": "Fixed", "correction_ratio": 0.2}
        )
        result = recognizer.correct_text("Original")
        
        assert isinstance(result, dict)
        assert result["corrected_text"] == "Fixed"
        assert result["correction_ratio"] == 0.2


class TestBaseRecognizerNoCorrection:
    """Tests for recognizer without correction support."""

    def test_no_correction_support(self):
        """Test recognizer that doesn't support correction."""
        recognizer = ConcreteRecognizer()
        recognizer.supports_correction = False
        
        result = recognizer.correct_text("Hello")
        assert result == "Hello"  # Should return unchanged


class TestBaseRecognizerBatch:
    """Tests for batch processing."""

    def test_process_blocks_batch(self):
        """Test batch processing."""
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        blocks_list = [
            [Block(type="text", bbox=BBox(10, 10, 50, 50))],
            [Block(type="title", bbox=BBox(10, 10, 90, 90))],
        ]
        
        recognizer = ConcreteRecognizer()
        results = recognizer.process_blocks_batch(images, blocks_list)  # type: ignore[arg-type]  # type: ignore[arg-type]
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    def test_process_blocks_batch_mismatched(self):
        """Test batch processing with mismatched lengths."""
        images = [np.zeros((100, 100, 3), dtype=np.uint8)]
        blocks_list = [[], []]  # 2 items vs 1 image
        
        recognizer = ConcreteRecognizer()
        with pytest.raises((ValueError, RecognitionError), match="Mismatched lengths"):
            recognizer.process_blocks_batch(images, blocks_list)  # type: ignore[arg-type]


class TestBaseRecognizerRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        recognizer = ConcreteRecognizer()
        repr_str = repr(recognizer)
        
        assert "ConcreteRecognizer" in repr_str
        assert "test-recognizer" in repr_str
        assert "supports_correction=True" in repr_str

