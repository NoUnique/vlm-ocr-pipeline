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
        expected_markdown = "# Title\n\nBody text"
        mock_blocks_to_markdown.return_value = expected_markdown

        stage = RenderingStage(renderer="markdown")

        # Execute
        result = stage.render(blocks)

        # Verify
        mock_blocks_to_markdown.assert_called_once_with(blocks)
        assert result == expected_markdown

    def test_render_unsupported_renderer(self):
        """Test rendering with unsupported renderer."""
        # Setup
        blocks = []
        stage = RenderingStage(renderer="html")

        # Execute & Verify
        with pytest.raises(ValueError, match="Unsupported renderer: html"):
            stage.render(blocks)

    @patch("pipeline.stages.rendering_stage.blocks_to_markdown")
    def test_render_empty_blocks(self, mock_blocks_to_markdown: Mock):
        """Test rendering empty blocks list."""
        # Setup
        mock_blocks_to_markdown.return_value = ""

        stage = RenderingStage(renderer="markdown")

        # Execute
        result = stage.render([])

        # Verify
        mock_blocks_to_markdown.assert_called_once_with([])
        assert result == ""

