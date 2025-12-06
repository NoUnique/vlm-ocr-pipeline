"""Tests for OrderingStage.

Tests cover:
- Stage initialization
- Block sorting
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from pipeline.stages.ordering_stage import OrderingStage
from pipeline.types import BBox, Block


class TestOrderingStageInit:
    """Tests for OrderingStage initialization."""

    def test_init(self):
        """Test OrderingStage initialization."""
        mock_sorter = Mock()
        stage = OrderingStage(sorter=mock_sorter)
        assert stage.sorter == mock_sorter


class TestOrderingStageSort:
    """Tests for OrderingStage sorting."""

    def test_sort(self):
        """Test sorting blocks."""
        # Setup
        mock_sorter = Mock()
        input_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                detection_confidence=0.95,
                order=None,
            ),
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=None,
            ),
        ]
        sorted_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
            ),
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                detection_confidence=0.95,
                order=1,
            ),
        ]
        mock_sorter.sort.return_value = sorted_blocks

        stage = OrderingStage(sorter=mock_sorter)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.sort(input_blocks, image)

        # Verify - sorter.sort is called with input_blocks, image, and any kwargs from context
        mock_sorter.sort.assert_called_once()
        call_args = mock_sorter.sort.call_args
        assert list(call_args[0][0]) == input_blocks  # First arg is blocks
        np.testing.assert_array_equal(call_args[0][1], image)  # Second arg is image
        assert result == sorted_blocks
        assert result[0].order == 0
        assert result[1].order == 1

    def test_sort_empty_blocks(self):
        """Test sorting empty blocks list."""
        # Setup
        mock_sorter = Mock()
        mock_sorter.sort.return_value = []

        stage = OrderingStage(sorter=mock_sorter)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.sort([], image)

        # Verify
        mock_sorter.sort.assert_called_once()
        assert result == []

    def test_sort_single_block(self):
        """Test sorting a single block."""
        # Setup
        mock_sorter = Mock()
        input_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=None,
            ),
        ]
        sorted_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
            ),
        ]
        mock_sorter.sort.return_value = sorted_blocks

        stage = OrderingStage(sorter=mock_sorter)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.sort(input_blocks, image)

        # Verify
        assert len(result) == 1
        assert result[0].order == 0

