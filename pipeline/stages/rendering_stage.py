"""Rendering Stage: Convert blocks to output format (Markdown, HTML, etc.)."""

from __future__ import annotations

from typing import Any

from pipeline.conversion.output.markdown import blocks_to_markdown
from pipeline.types import Block


class RenderingStage:
    """Stage 6: Rendering - Convert blocks to output format."""

    def __init__(self, renderer: str = "markdown"):
        """Initialize RenderingStage.

        Args:
            renderer: Output format renderer (currently only "markdown" supported)
        """
        self.renderer = renderer

    def render(self, blocks: list[Block], auxiliary_info: dict[str, Any] | None = None) -> str:
        """Render blocks to output format.

        Args:
            blocks: List of blocks with text
            auxiliary_info: Additional metadata (e.g., text_spans for font-based headers)

        Returns:
            Rendered text in the specified format
        """
        if self.renderer == "markdown":
            return self._render_markdown(blocks, auxiliary_info)
        else:
            # Future: support HTML, LaTeX, etc.
            raise ValueError(f"Unsupported renderer: {self.renderer}")

    def _render_markdown(self, blocks: list[Block], auxiliary_info: dict[str, Any] | None) -> str:
        """Render blocks as Markdown.

        Args:
            blocks: List of blocks with text
            auxiliary_info: Additional metadata

        Returns:
            Markdown-formatted text
        """
        # Use existing blocks_to_markdown function
        # auxiliary_info is passed through kwargs if needed
        markdown_text = blocks_to_markdown(blocks)
        return markdown_text
