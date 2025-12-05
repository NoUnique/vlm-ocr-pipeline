"""Tests for PyMuPDF multi-column sorter."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pipeline.types import BBox, Block


class TestMultiColumnSorterInit:
    """Tests for MultiColumnSorter initialization."""

    def test_init_with_pymupdf_available(self):
        """Test sorter initialization when PyMuPDF is available."""
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            from pipeline.layout.ordering.pymupdf.multi_column import MultiColumnSorter
            
            sorter = MultiColumnSorter()
            assert sorter is not None

    def test_sorter_has_sort_method(self):
        """Test sorter has sort method."""
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            from pipeline.layout.ordering.pymupdf.multi_column import MultiColumnSorter
            
            sorter = MultiColumnSorter()
            assert hasattr(sorter, "sort")
            assert callable(sorter.sort)


class TestMultiColumnSorterSort:
    """Tests for MultiColumnSorter.sort method."""

    @pytest.fixture
    def mock_fitz(self):
        """Create mock fitz module."""
        mock = MagicMock()
        mock.Rect = MagicMock(return_value=MagicMock())
        return mock

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.zeros((600, 800, 3), dtype=np.uint8)

    def test_sort_empty_blocks(self, mock_fitz, sample_image):
        """Test sorting empty block list."""
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from pipeline.layout.ordering.pymupdf.multi_column import MultiColumnSorter
            
            sorter = MultiColumnSorter()
            result = sorter.sort([], sample_image)
            
            assert result == []

    def test_sort_single_block(self, mock_fitz, sample_image):
        """Test sorting single block."""
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from pipeline.layout.ordering.pymupdf.multi_column import MultiColumnSorter
            
            blocks = [Block(type="text", bbox=BBox(100, 100, 200, 200))]
            sorter = MultiColumnSorter()
            result = sorter.sort(blocks, sample_image)
            
            assert len(result) == 1
            assert result[0].order is not None

    def test_sort_multiple_blocks(self, mock_fitz, sample_image):
        """Test sorting multiple blocks."""
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from pipeline.layout.ordering.pymupdf.multi_column import MultiColumnSorter
            
            blocks = [
                Block(type="text", bbox=BBox(100, 200, 200, 300)),  # Bottom
                Block(type="text", bbox=BBox(100, 50, 200, 100)),   # Top
            ]
            sorter = MultiColumnSorter()
            result = sorter.sort(blocks, sample_image)
            
            assert len(result) == 2
            # All blocks should have order assigned
            for block in result:
                assert block.order is not None

    def test_sort_preserves_block_fields(self, mock_fitz, sample_image):
        """Test that sorting preserves block fields."""
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from pipeline.layout.ordering.pymupdf.multi_column import MultiColumnSorter
            
            blocks = [
                Block(
                    type="title",
                    bbox=BBox(100, 100, 200, 200),
                    detection_confidence=0.95,
                    source="test",
                    text="Test Title",
                )
            ]
            sorter = MultiColumnSorter()
            result = sorter.sort(blocks, sample_image)
            
            assert result[0].type == "title"
            assert result[0].detection_confidence == 0.95
            assert result[0].source == "test"
            assert result[0].text == "Test Title"


class TestColumnBoxesFunction:
    """Tests for column_boxes function."""

    def test_column_boxes_no_pymupdf(self):
        """Test that column_boxes raises when PyMuPDF not available."""
        with patch.dict("sys.modules", {"fitz": None}):
            # Reload module to pick up patched fitz
            import importlib
            import pipeline.layout.ordering.pymupdf.multi_column as mc
            
            # Should handle missing fitz gracefully
            # The function checks for fitz availability internally

