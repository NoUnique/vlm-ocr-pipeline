"""Tests for DetectionStage.

Tests cover:
- Stage initialization
- Block detection
- Column layout extraction
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from pipeline.stages.detection_stage import DetectionStage
from pipeline.types import BBox, Block


class TestDetectionStageInit:
    """Tests for DetectionStage initialization."""

    def test_init(self):
        """Test DetectionStage initialization."""
        mock_detector = Mock()
        stage = DetectionStage(detector=mock_detector)
        assert stage.detector == mock_detector


class TestDetectionStageDetect:
    """Tests for DetectionStage detection."""

    def test_detect(self):
        """Test detecting blocks."""
        # Setup
        mock_detector = Mock()
        mock_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=None,
            )
        ]
        mock_detector.detect.return_value = mock_blocks

        stage = DetectionStage(detector=mock_detector)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.process(image)

        # Verify
        mock_detector.detect.assert_called_once()
        assert result == mock_blocks

    def test_detect_multiple_blocks(self):
        """Test detecting multiple blocks."""
        # Setup
        mock_detector = Mock()
        mock_blocks = [
            Block(
                type="title",
                bbox=BBox(100, 50, 400, 100),
                detection_confidence=0.98,
            ),
            Block(
                type="text",
                bbox=BBox(100, 120, 400, 300),
                detection_confidence=0.95,
            ),
            Block(
                type="figure",
                bbox=BBox(100, 320, 400, 500),
                detection_confidence=0.92,
            ),
        ]
        mock_detector.detect.return_value = mock_blocks

        stage = DetectionStage(detector=mock_detector)
        image = np.zeros((600, 500, 3), dtype=np.uint8)

        # Execute
        result = stage.process(image)

        # Verify
        assert len(result) == 3
        assert result[0].type == "title"
        assert result[1].type == "text"
        assert result[2].type == "figure"

    def test_detect_empty(self):
        """Test detection returning empty result."""
        # Setup
        mock_detector = Mock()
        mock_detector.detect.return_value = []

        stage = DetectionStage(detector=mock_detector)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.process(image)

        # Verify
        assert result == []


class TestDetectionStageColumnLayout:
    """Tests for DetectionStage column layout extraction."""

    def test_extract_column_layout_no_columns(self):
        """Test extracting column layout when no columns exist."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                column_index=None,
            )
        ]

        stage = DetectionStage(detector=Mock())

        # Execute
        result = stage.extract_column_layout(blocks)

        # Verify
        assert result is None

    def test_extract_column_layout_with_columns(self):
        """Test extracting column layout when columns exist."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                column_index=0,
            ),
            Block(
                type="text",
                bbox=BBox(300, 100, 400, 200),
                detection_confidence=0.95,
                order=1,
                column_index=1,
            ),
        ]

        stage = DetectionStage(detector=Mock())

        # Execute
        result = stage.extract_column_layout(blocks)

        # Verify
        assert result is not None
        assert "columns" in result
        assert len(result["columns"]) == 2
        assert result["columns"][0]["index"] == 0
        assert result["columns"][0]["x0"] == 100
        assert result["columns"][1]["index"] == 1
        assert result["columns"][1]["x0"] == 300

    def test_extract_column_layout_three_columns(self):
        """Test extracting column layout with three columns."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(50, 100, 150, 200),
                detection_confidence=0.95,
                order=0,
                column_index=0,
            ),
            Block(
                type="text",
                bbox=BBox(200, 100, 300, 200),
                detection_confidence=0.95,
                order=1,
                column_index=1,
            ),
            Block(
                type="text",
                bbox=BBox(350, 100, 450, 200),
                detection_confidence=0.95,
                order=2,
                column_index=2,
            ),
        ]

        stage = DetectionStage(detector=Mock())

        # Execute
        result = stage.extract_column_layout(blocks)

        # Verify
        assert result is not None
        assert len(result["columns"]) == 3

