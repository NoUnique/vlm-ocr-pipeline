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

