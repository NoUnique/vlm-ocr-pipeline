"""Tests for MinerU XY-Cut sorter."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.layout.ordering.mineru.xycut import MinerUXYCutSorter
from pipeline.types import BBox, Block


class TestMinerUXYCutSorterInit:
    """Tests for MinerUXYCutSorter initialization."""

    def test_init_success(self):
        """Test successful sorter initialization."""
        sorter = MinerUXYCutSorter()
        assert sorter is not None

    def test_sorter_has_sort_method(self):
        """Test sorter has sort method."""
        sorter = MinerUXYCutSorter()
        assert hasattr(sorter, "sort")
        assert callable(sorter.sort)


class TestMinerUXYCutSorterSort:
    """Tests for MinerUXYCutSorter.sort method."""

    @pytest.fixture
    def sorter(self):
        """Create sorter instance."""
        return MinerUXYCutSorter()

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.zeros((600, 800, 3), dtype=np.uint8)

    def test_sort_empty_blocks(self, sorter, sample_image):
        """Test sorting empty block list."""
        result = sorter.sort([], sample_image)
        assert result == []

    def test_sort_single_block(self, sorter, sample_image):
        """Test sorting single block."""
        blocks = [Block(type="text", bbox=BBox(100, 100, 200, 200))]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 1
        assert result[0].order == 0

    def test_sort_vertical_order(self, sorter, sample_image):
        """Test sorting blocks in vertical order."""
        blocks = [
            Block(type="text", bbox=BBox(100, 200, 200, 300)),  # Bottom
            Block(type="text", bbox=BBox(100, 50, 200, 100)),   # Top
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 2
        # Top block should come first
        assert result[0].bbox.y0 < result[1].bbox.y0

    def test_sort_horizontal_order(self, sorter, sample_image):
        """Test sorting blocks in horizontal order (same row)."""
        blocks = [
            Block(type="text", bbox=BBox(300, 100, 400, 150)),  # Right
            Block(type="text", bbox=BBox(100, 100, 200, 150)),  # Left
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 2
        # Both should have order assigned
        assert result[0].order is not None
        assert result[1].order is not None

    def test_sort_grid_layout(self, sorter, sample_image):
        """Test sorting blocks in 2x2 grid layout."""
        blocks = [
            Block(type="text", bbox=BBox(300, 200, 400, 300)),  # Bottom right
            Block(type="text", bbox=BBox(100, 200, 200, 300)),  # Bottom left
            Block(type="text", bbox=BBox(300, 50, 400, 100)),   # Top right
            Block(type="text", bbox=BBox(100, 50, 200, 100)),   # Top left
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 4
        # All should have order assigned
        for i, block in enumerate(result):
            assert block.order == i

    def test_sort_preserves_block_data(self, sorter, sample_image):
        """Test that sorting preserves other block fields."""
        blocks = [
            Block(
                type="title",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                source="test-detector",
                text="Test Title",
            )
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert result[0].type == "title"
        assert result[0].detection_confidence == 0.95
        assert result[0].source == "test-detector"
        assert result[0].text == "Test Title"

    def test_sort_multiple_block_types(self, sorter, sample_image):
        """Test sorting blocks with different types."""
        blocks = [
            Block(type="table", bbox=BBox(100, 300, 400, 500)),
            Block(type="title", bbox=BBox(100, 50, 400, 100)),
            Block(type="text", bbox=BBox(100, 150, 400, 250)),
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 3
        # Title should be first (top)
        assert result[0].type == "title"
        # Text should be second
        assert result[1].type == "text"
        # Table should be last (bottom)
        assert result[2].type == "table"


class TestMinerUXYCutSorterEdgeCases:
    """Edge case tests for MinerUXYCutSorter."""

    @pytest.fixture
    def sorter(self):
        """Create sorter instance."""
        return MinerUXYCutSorter()

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.zeros((600, 800, 3), dtype=np.uint8)

    def test_sort_overlapping_blocks(self, sorter, sample_image):
        """Test sorting overlapping blocks."""
        blocks = [
            Block(type="text", bbox=BBox(100, 100, 250, 200)),
            Block(type="text", bbox=BBox(200, 150, 350, 250)),  # Overlaps
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 2
        # Should still assign order
        assert result[0].order is not None
        assert result[1].order is not None

    def test_sort_small_blocks(self, sorter, sample_image):
        """Test sorting very small blocks."""
        blocks = [
            Block(type="text", bbox=BBox(10, 10, 15, 15)),
            Block(type="text", bbox=BBox(20, 10, 25, 15)),
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 2

    def test_sort_large_block_list(self, sorter, sample_image):
        """Test sorting large number of blocks."""
        blocks = [
            Block(type="text", bbox=BBox(i * 50, i * 30, i * 50 + 40, i * 30 + 20))
            for i in range(20)
        ]
        result = sorter.sort(blocks, sample_image)
        
        assert len(result) == 20
        # All should have unique order values
        orders = [b.order for b in result]
        assert len(set(orders)) == 20

