"""Rendering Stage: Convert blocks to output format (Markdown, HTML, etc.)."""

from __future__ import annotations

from typing import Any

from pipeline.conversion.output.markdown import blocks_to_markdown
from pipeline.types import Block

from .base import BaseStage


class RenderingStage(BaseStage[list[Block], str]):
    """Stage 6: Rendering - Convert blocks to output format.

    This stage converts processed blocks into a human-readable format
    such as Markdown. Future support for HTML, LaTeX, etc.
    """

    name = "rendering"

    def __init__(self, renderer: str = "markdown"):
        """Initialize RenderingStage.

        Args:
            renderer: Output format renderer (currently only "markdown" supported)
        """
        self.renderer = renderer

    def _process_impl(self, input_data: list[Block], **context: Any) -> str:
        """Render blocks to output format.

        Args:
            input_data: List of blocks with text
            **context: May include 'auxiliary_info' for font-based headers

        Returns:
            Rendered text in the specified format
        """
        auxiliary_info = context.get("auxiliary_info")

        if self.renderer == "markdown":
            return self._render_markdown(input_data, auxiliary_info)
        else:
            raise ValueError(f"Unsupported renderer: {self.renderer}")

    def render(self, blocks: list[Block], auxiliary_info: dict[str, Any] | None = None) -> str:
        """Render blocks to output format.

        Legacy method for backward compatibility.

        Args:
            blocks: List of blocks with text
            auxiliary_info: Additional metadata (e.g., text_spans for font-based headers)

        Returns:
            Rendered text in the specified format
        """
        return self.process(blocks, auxiliary_info=auxiliary_info)

    def _render_markdown(self, blocks: list[Block], auxiliary_info: dict[str, Any] | None) -> str:
        """Render blocks as Markdown.

        Args:
            blocks: List of blocks with text
            auxiliary_info: Additional metadata

        Returns:
            Markdown-formatted text
        """
        markdown_text = blocks_to_markdown(blocks)
        return markdown_text
