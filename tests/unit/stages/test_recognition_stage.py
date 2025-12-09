"""Tests for RecognitionStage.

Tests cover:
- Stage initialization
- Block text recognition
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from pipeline.stages.recognition_stage import RecognitionStage
from pipeline.types import BBox, Block


class TestRecognitionStageInit:
    """Tests for RecognitionStage initialization."""

    def test_init(self):
        """Test RecognitionStage initialization."""
        mock_recognizer = Mock()
        stage = RecognitionStage(recognizer=mock_recognizer)
        assert stage.recognizer == mock_recognizer


class TestRecognitionStageRecognize:
    """Tests for RecognitionStage recognition."""

    def test_recognize_blocks(self):
        """Test recognizing text in blocks."""
        # Setup
        mock_recognizer = Mock()
        input_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text=None,
            )
        ]
        processed_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Extracted text",
            )
        ]
        mock_recognizer.process_blocks.return_value = processed_blocks

        stage = RecognitionStage(recognizer=mock_recognizer)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.process(input_blocks, image=image)

        # Verify
        mock_recognizer.process_blocks.assert_called_once()
        call_args = mock_recognizer.process_blocks.call_args.args
        assert np.array_equal(call_args[0], image)  # First arg is image
        assert call_args[1] == input_blocks  # Second arg is blocks
        assert result == processed_blocks
        assert result[0].text == "Extracted text"

    def test_recognize_multiple_blocks(self):
        """Test recognizing text in multiple blocks."""
        # Setup
        mock_recognizer = Mock()
        input_blocks = [
            Block(
                type="title",
                bbox=BBox(100, 50, 300, 80),
                detection_confidence=0.98,
                order=0,
            ),
            Block(
                type="text",
                bbox=BBox(100, 100, 300, 200),
                detection_confidence=0.95,
                order=1,
            ),
        ]
        processed_blocks = [
            Block(
                type="title",
                bbox=BBox(100, 50, 300, 80),
                detection_confidence=0.98,
                order=0,
                text="Chapter 1",
            ),
            Block(
                type="text",
                bbox=BBox(100, 100, 300, 200),
                detection_confidence=0.95,
                order=1,
                text="Content paragraph",
            ),
        ]
        mock_recognizer.process_blocks.return_value = processed_blocks

        stage = RecognitionStage(recognizer=mock_recognizer)
        image = np.zeros((300, 400, 3), dtype=np.uint8)

        # Execute
        result = stage.process(input_blocks, image=image)

        # Verify
        assert len(result) == 2
        assert result[0].text == "Chapter 1"
        assert result[1].text == "Content paragraph"

    def test_recognize_empty_blocks(self):
        """Test recognizing empty blocks list."""
        # Setup
        mock_recognizer = Mock()
        mock_recognizer.process_blocks.return_value = []

        stage = RecognitionStage(recognizer=mock_recognizer)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.process([], image=image)

        # Verify
        assert result == []

