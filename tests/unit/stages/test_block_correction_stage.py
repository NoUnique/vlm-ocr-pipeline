"""Tests for BlockCorrectionStage."""

from __future__ import annotations

from pipeline.stages.block_correction_stage import BlockCorrectionStage
from pipeline.types import BBox, Block


class TestBlockCorrectionStageInit:
    """Tests for BlockCorrectionStage initialization."""

    def test_init_enabled(self):
        """Test BlockCorrectionStage initialization with enabled=True."""
        stage = BlockCorrectionStage(enable=True)
        assert stage.enable is True

    def test_init_disabled(self):
        """Test BlockCorrectionStage initialization with enabled=False."""
        stage = BlockCorrectionStage(enable=False)
        assert stage.enable is False


class TestBlockCorrectionStageCorrect:
    """Tests for BlockCorrectionStage.correct_blocks method."""

    def test_correct_blocks_disabled(self):
        """Test block correction when disabled."""
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Original text",
                corrected_text=None,
            )
        ]

        stage = BlockCorrectionStage(enable=False)
        result = stage.correct_blocks(blocks)

        assert result[0].corrected_text == "Original text"

    def test_correct_blocks_enabled(self):
        """Test block correction when enabled (currently placeholder)."""
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Original text",
                corrected_text=None,
            )
        ]

        stage = BlockCorrectionStage(enable=True)
        result = stage.correct_blocks(blocks)

        # Currently just copies text as-is
        assert result[0].corrected_text == "Original text"
        assert result[0].correction_ratio == 0.0

    def test_correct_blocks_empty_list(self):
        """Test block correction with empty list."""
        stage = BlockCorrectionStage(enable=True)
        result = stage.correct_blocks([])
        assert result == []

    def test_correct_blocks_preserves_other_fields(self):
        """Test that correction preserves other block fields."""
        blocks = [
            Block(
                type="title",
                bbox=BBox(50, 50, 150, 100),
                detection_confidence=0.99,
                order=1,
                text="Title Text",
                source="test-detector",
            )
        ]

        stage = BlockCorrectionStage(enable=True)
        result = stage.correct_blocks(blocks)

        assert result[0].type == "title"
        assert result[0].bbox == BBox(50, 50, 150, 100)
        assert result[0].detection_confidence == 0.99
        assert result[0].order == 1
        assert result[0].source == "test-detector"

