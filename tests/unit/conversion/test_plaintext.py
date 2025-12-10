"""Tests for plaintext output conversion."""

from __future__ import annotations

from pipeline.io.output.plaintext import blocks_to_plaintext, page_to_plaintext
from pipeline.types import BBox, Block, Page


class TestBlocksToPlaintext:
    """Tests for blocks_to_plaintext function."""

    def test_empty_blocks(self):
        """Test conversion with empty block list."""
        result = blocks_to_plaintext([])
        assert result == ""

    def test_single_text_block(self):
        """Test conversion with single text block."""
        blocks = [
            Block(type="text", bbox=BBox(0, 0, 100, 50), text="Hello World", order=0)
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "Hello World"

    def test_multiple_text_blocks(self):
        """Test conversion with multiple text blocks."""
        blocks = [
            Block(type="title", bbox=BBox(0, 0, 100, 30), text="Title", order=0),
            Block(type="text", bbox=BBox(0, 50, 100, 100), text="Paragraph 1", order=1),
            Block(type="text", bbox=BBox(0, 120, 100, 170), text="Paragraph 2", order=2),
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "Title\n\nParagraph 1\n\nParagraph 2"

    def test_preserves_block_order(self):
        """Test that blocks are sorted by order field."""
        blocks = [
            Block(type="text", bbox=BBox(0, 100, 100, 150), text="Second", order=1),
            Block(type="text", bbox=BBox(0, 0, 100, 50), text="First", order=0),
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "First\n\nSecond"

    def test_excludes_non_text_blocks(self):
        """Test that non-text blocks are excluded."""
        blocks = [
            Block(type="title", bbox=BBox(0, 0, 100, 30), text="Title", order=0),
            Block(type="table", bbox=BBox(0, 50, 100, 150), text="Table content", order=1),
            Block(type="text", bbox=BBox(0, 170, 100, 200), text="Body text", order=2),
            Block(type="figure", bbox=BBox(0, 220, 100, 300), text="Figure", order=3),
        ]
        result = blocks_to_plaintext(blocks)
        # Only title and text types are included
        assert result == "Title\n\nBody text"

    def test_includes_list_type(self):
        """Test that list type blocks are included."""
        blocks = [
            Block(type="list", bbox=BBox(0, 0, 100, 50), text="• Item 1\n• Item 2", order=0),
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "• Item 1\n• Item 2"

    def test_empty_text_blocks_excluded(self):
        """Test that blocks with empty text are excluded."""
        blocks = [
            Block(type="text", bbox=BBox(0, 0, 100, 50), text="Valid", order=0),
            Block(type="text", bbox=BBox(0, 60, 100, 100), text="", order=1),
            Block(type="text", bbox=BBox(0, 110, 100, 150), text="   ", order=2),
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "Valid"

    def test_whitespace_trimmed(self):
        """Test that outer whitespace is trimmed."""
        blocks = [
            Block(type="text", bbox=BBox(0, 0, 100, 50), text="  Trimmed text  ", order=0),
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "Trimmed text"

    def test_internal_newlines_preserved(self):
        """Test that internal newlines are preserved."""
        blocks = [
            Block(type="text", bbox=BBox(0, 0, 100, 100), text="Line 1\nLine 2\nLine 3", order=0),
        ]
        result = blocks_to_plaintext(blocks)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_fallback_to_position_sort(self):
        """Test sorting by position when order is None."""
        blocks = [
            Block(type="text", bbox=BBox(0, 100, 100, 150), text="Bottom"),  # y=100
            Block(type="text", bbox=BBox(0, 0, 100, 50), text="Top"),  # y=0
        ]
        result = blocks_to_plaintext(blocks)
        # Should sort by y position
        assert result == "Top\n\nBottom"


class TestPageToPlaintext:
    """Tests for page_to_plaintext function."""

    def test_page_conversion(self):
        """Test page conversion to plaintext."""
        page = Page(
            page_num=1,
            blocks=[
                Block(type="title", bbox=BBox(0, 0, 100, 30), text="Page Title", order=0),
                Block(type="text", bbox=BBox(0, 50, 100, 100), text="Page content", order=1),
            ],
        )
        result = page_to_plaintext(page)
        assert result == "Page Title\n\nPage content"

    def test_empty_page(self):
        """Test empty page conversion."""
        page = Page(page_num=1, blocks=[])
        result = page_to_plaintext(page)
        assert result == ""

    def test_page_with_mixed_block_types(self):
        """Test page with mixed block types."""
        page = Page(
            page_num=1,
            blocks=[
                Block(type="title", bbox=BBox(0, 0, 100, 30), text="Title", order=0),
                Block(type="figure", bbox=BBox(0, 40, 100, 100), text="Figure", order=1),
                Block(type="text", bbox=BBox(0, 110, 100, 150), text="Text", order=2),
            ],
        )
        result = page_to_plaintext(page)
        # Figure should be excluded
        assert result == "Title\n\nText"

