"""Tests for Block class and related utilities.

Tests cover:
- Block creation
- Block utilities (anchor text generation)
"""

from __future__ import annotations

from pipeline.types import BBox, Block, blocks_to_olmocr_anchor_text


class TestBlockCreation:
    """Test Block creation."""

    def test_basic_creation(self):
        """Test basic Block creation."""
        bbox = BBox(100, 50, 300, 200)
        block = Block(
            type="text",
            bbox=bbox,
            detection_confidence=0.95,
        )

        assert block.type == "text"
        assert block.bbox == bbox
        assert block.detection_confidence == 0.95
        assert block.order is None
        assert block.text is None

    def test_creation_with_all_fields(self):
        """Test Block creation with all fields."""
        bbox = BBox(100, 50, 300, 200)
        block = Block(
            type="title",
            bbox=bbox,
            detection_confidence=0.98,
            order=0,
            column_index=1,
            text="Chapter 1",
            corrected_text="Chapter 1",
            correction_ratio=0.0,
            source="doclayout-yolo",
        )

        assert block.type == "title"
        assert block.order == 0
        assert block.column_index == 1
        assert block.text == "Chapter 1"
        assert block.corrected_text == "Chapter 1"
        assert block.correction_ratio == 0.0
        assert block.source == "doclayout-yolo"


class TestBlockUtilities:
    """Test Block utility functions."""

    def test_blocks_to_olmocr_anchor_text(self):
        """Test blocks to olmOCR anchor text conversion."""
        blocks: list[Block] = [
            Block(
                type="title",
                bbox=BBox(100, 50, 300, 80),
                detection_confidence=0.9,
                text="Chapter 1",
            ),
            Block(
                type="figure",
                bbox=BBox(100, 100, 300, 250),
                detection_confidence=0.95,
            ),
            Block(
                type="plain text",
                bbox=BBox(100, 300, 500, 350),
                detection_confidence=0.9,
                text="Content here with some long text that might be truncated",
            ),
        ]

        anchor_text = blocks_to_olmocr_anchor_text(blocks, 800, 600)

        # Check header
        assert "Page dimensions: 800x600" in anchor_text

        # Check text region (with partial content)
        assert "[100x50]Chapter 1" in anchor_text

        # Check image region
        assert "[Image 100x100 to 300x250]" in anchor_text

        # Check another text region
        assert "[100x300]Content here" in anchor_text

    def test_blocks_to_olmocr_anchor_text_max_length(self):
        """Test anchor text respects max_length limit."""
        # Create many blocks
        blocks: list[Block] = [
            Block(
                type="text",
                bbox=BBox(i * 10, i * 10, i * 10 + 100, i * 10 + 20),
                detection_confidence=0.9,
            )
            for i in range(100)
        ]

        # Set short max_length
        anchor_text = blocks_to_olmocr_anchor_text(blocks, 800, 600, max_length=200)

        # Should be limited
        assert len(anchor_text) <= 250  # Some buffer for formatting

    def test_blocks_to_olmocr_anchor_text_empty(self):
        """Test anchor text with empty blocks list."""
        anchor_text = blocks_to_olmocr_anchor_text([], 800, 600)

        # Should still have header
        assert "Page dimensions: 800x600" in anchor_text

    def test_blocks_to_olmocr_anchor_text_table(self):
        """Test anchor text with table blocks."""
        blocks: list[Block] = [
            Block(
                type="table",
                bbox=BBox(50, 100, 500, 400),
                detection_confidence=0.92,
            ),
        ]

        anchor_text = blocks_to_olmocr_anchor_text(blocks, 800, 600)

        # Check table region
        assert "[Table 50x100 to 500x400]" in anchor_text


class TestBlockModification:
    """Test Block modification (dataclass replace)."""

    def test_block_replace(self):
        """Test creating modified Block using dataclasses.replace."""
        from dataclasses import replace

        bbox = BBox(100, 50, 300, 200)
        original = Block(
            type="text",
            bbox=bbox,
            detection_confidence=0.95,
            order=None,
            text=None,
        )

        # Create modified copy with order and text
        modified = replace(original, order=1, text="Recognized text")

        # Original unchanged
        assert original.order is None
        assert original.text is None

        # Modified has new values
        assert modified.order == 1
        assert modified.text == "Recognized text"
        assert modified.type == "text"  # Preserved
        assert modified.bbox == bbox  # Preserved

    def test_block_replace_correction(self):
        """Test adding correction to Block."""
        from dataclasses import replace

        bbox = BBox(100, 50, 300, 200)
        block = Block(
            type="text",
            bbox=bbox,
            detection_confidence=0.95,
            order=0,
            text="Originl txt",  # With typos
        )

        corrected = replace(
            block,
            corrected_text="Original text",
            correction_ratio=0.15,
        )

        assert corrected.text == "Originl txt"  # Original preserved
        assert corrected.corrected_text == "Original text"
        assert corrected.correction_ratio == 0.15

