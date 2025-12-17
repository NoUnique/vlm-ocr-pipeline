"""Tests for RenderingStage.

Tests cover:
- Stage initialization
- Markdown rendering
- Unsupported renderer handling
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from pipeline.stages.rendering_stage import RenderingStage
from pipeline.types import BBox, Block


class TestRenderingStageInit:
    """Tests for RenderingStage initialization."""

    def test_init_markdown(self):
        """Test RenderingStage initialization with markdown."""
        stage = RenderingStage(renderer="markdown")
        assert stage.renderer == "markdown"

    def test_init_plaintext(self):
        """Test RenderingStage initialization with plaintext."""
        stage = RenderingStage(renderer="plaintext")
        assert stage.renderer == "plaintext"


class TestRenderingStageRender:
    """Tests for RenderingStage rendering."""

    @patch("pipeline.stages.rendering_stage.blocks_to_markdown")
    def test_render_markdown(self, mock_blocks_to_markdown: Mock):
        """Test rendering blocks to markdown."""
        # Setup
        blocks = [
            Block(
                type="title",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Title",
                corrected_text="Title",
            ),
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                detection_confidence=0.95,
                order=1,
                text="Body text",
                corrected_text="Body text",
            ),
        ]
        # blocks_to_markdown is called once per block
        mock_blocks_to_markdown.side_effect = ["# Title", "Body text"]

        stage = RenderingStage(renderer="markdown")

        # Execute
        result = stage.process(blocks)

        # Verify - called once per block (2 blocks = 2 calls)
        assert mock_blocks_to_markdown.call_count == 2
        assert "# Title" in result
        assert "Body text" in result

    def test_render_unsupported_renderer(self):
        """Test rendering with unsupported renderer."""
        from pipeline.stages.base import StageError

        # Setup
        blocks = []
        stage = RenderingStage(renderer="html")

        # Execute & Verify - StageError wraps the ValueError
        with pytest.raises(StageError) as exc_info:
            stage.process(blocks)

        assert "Unsupported renderer: html" in str(exc_info.value)
        assert exc_info.value.stage_name == "rendering"

    def test_render_empty_blocks(self):
        """Test rendering empty blocks list."""
        stage = RenderingStage(renderer="markdown")

        # Execute
        result = stage.process([])

        # Verify - empty blocks should return empty string
        assert result == ""

    def test_render_plaintext(self):
        """Test rendering blocks to plaintext."""
        blocks = [
            Block(
                type="title",
                bbox=BBox(100, 100, 200, 200),
                order=0,
                text="Title",
                corrected_text="Title",
            ),
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                order=1,
                text="Body text",
                corrected_text="Body text",
            ),
        ]

        stage = RenderingStage(renderer="plaintext")
        result = stage.process(blocks)

        assert "Title" in result
        assert "Body text" in result

    def test_render_plaintext_empty_blocks(self):
        """Test plaintext rendering with empty blocks."""
        stage = RenderingStage(renderer="plaintext")
        result = stage.process([])
        assert result == ""


class TestRenderingStageImageMode:
    """Tests for RenderingStage image rendering modes."""

    def test_image_render_mode_default(self):
        """Test default image render mode."""
        stage = RenderingStage(renderer="markdown")
        assert stage.image_render_mode == "image_and_description"

    def test_image_render_mode_image_only(self):
        """Test image_only render mode."""
        stage = RenderingStage(renderer="markdown", image_render_mode="image_only")
        assert stage.image_render_mode == "image_only"

    def test_image_render_mode_description_only(self):
        """Test description_only render mode."""
        stage = RenderingStage(renderer="markdown", image_render_mode="description_only")
        assert stage.image_render_mode == "description_only"

    def test_render_image_block_with_description(self):
        """Test rendering image block with description."""
        blocks = [
            Block(
                type="image",
                bbox=BBox(100, 100, 400, 400),
                order=0,
                image_path="images/page_1_block_0_image.png",
                description="A chart showing sales data",
            ),
        ]

        stage = RenderingStage(renderer="markdown", image_render_mode="image_and_description")
        result = stage.process(blocks)

        assert "images/page_1_block_0_image.png" in result
        assert "A chart showing sales data" in result

    def test_render_image_block_image_only(self):
        """Test rendering image block with image_only mode."""
        blocks = [
            Block(
                type="image",
                bbox=BBox(100, 100, 400, 400),
                order=0,
                image_path="images/page_1_block_0_image.png",
                description="A chart showing sales data",
            ),
        ]

        stage = RenderingStage(renderer="markdown", image_render_mode="image_only")
        result = stage.process(blocks)

        # Image path should be in output
        assert "images/page_1_block_0_image.png" in result
        # Description may be used as alt text in markdown, which is acceptable
        # Just verify there's no separate description paragraph
        assert result.count("A chart showing sales data") <= 1

    def test_render_image_block_description_only(self):
        """Test rendering image block with description_only mode."""
        blocks = [
            Block(
                type="image",
                bbox=BBox(100, 100, 400, 400),
                order=0,
                image_path="images/page_1_block_0_image.png",
                description="A chart showing sales data",
            ),
        ]

        stage = RenderingStage(renderer="markdown", image_render_mode="description_only")
        result = stage.process(blocks)

        # Image path should not be in output
        assert "images/page_1_block_0_image.png" not in result
        assert "A chart showing sales data" in result

