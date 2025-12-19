"""Rendering Stage: Convert blocks to output format (Markdown, Plaintext, etc.)."""

from __future__ import annotations

from typing import Any

from pipeline.io.output.markdown import blocks_to_markdown
from pipeline.types import Block

from .base import BaseStage


class RenderingStage(BaseStage[list[Block], str]):
    """Stage 6: Rendering - Convert blocks to output format.

    This stage converts processed blocks into a human-readable format
    such as Markdown or Plaintext.

    Supports image render modes for figure/image blocks:
    - image_only: Include only image links
    - image_and_description: Include both image links and descriptions
    - description_only: Include only descriptions (no image links)
    """

    name = "rendering"

    def __init__(
        self,
        renderer: str = "markdown",
        image_render_mode: str = "image_and_description",
    ):
        """Initialize RenderingStage.

        Args:
            renderer: Output format ("markdown" or "plaintext")
            image_render_mode: How to render image blocks
                - "image_only": Only image link
                - "image_and_description": Image link + description
                - "description_only": Only description text
        """
        self.renderer = renderer.lower()
        self.image_render_mode = image_render_mode.lower()

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
        elif self.renderer == "plaintext":
            return self._render_plaintext(input_data, auxiliary_info)
        else:
            raise ValueError(f"Unsupported renderer: {self.renderer}")

    def _render_markdown(self, blocks: list[Block], auxiliary_info: dict[str, Any] | None) -> str:
        """Render blocks as Markdown.

        Args:
            blocks: List of blocks with text
            auxiliary_info: Additional metadata

        Returns:
            Markdown-formatted text
        """
        lines: list[str] = []

        # Sort blocks by reading order
        sorted_blocks = self._sort_blocks(blocks)

        for block in sorted_blocks:
            rendered = self._render_block_markdown(block)
            if rendered:
                lines.append(rendered)

        return "\n\n".join(lines).strip()

    def _render_block_markdown(self, block: Block) -> str:
        """Render a single block as Markdown.

        Args:
            block: Block to render

        Returns:
            Markdown-formatted string
        """
        block_type = block.type.lower()

        # Check if this is an image/figure block
        is_image_block = block_type in {"image", "image_body", "figure", "chart"}

        if is_image_block:
            return self._render_image_block_markdown(block)

        # Standard text rendering using existing logic
        return blocks_to_markdown([block])

    def _render_image_block_markdown(self, block: Block) -> str:
        """Render an image block as Markdown based on image_render_mode.

        Args:
            block: Image block to render

        Returns:
            Markdown-formatted string
        """
        parts: list[str] = []

        # Image link
        if self.image_render_mode in {"image_only", "image_and_description"}:
            if block.image_path:
                # Use relative path for markdown image link
                alt_text = block.description[:50] if block.description else "Figure"
                parts.append(f"![{alt_text}]({block.image_path})")

        # Description
        if self.image_render_mode in {"description_only", "image_and_description"}:
            if block.description:
                parts.append(f"**Figure:**\n\n{block.description}")
            elif not block.image_path:
                # No image path and no description, use text if available
                if block.text:
                    parts.append(f"**Figure:**\n\n{block.text}")

        return "\n\n".join(parts)

    def _render_plaintext(self, blocks: list[Block], auxiliary_info: dict[str, Any] | None) -> str:
        """Render blocks as plaintext.

        Args:
            blocks: List of blocks with text
            auxiliary_info: Additional metadata

        Returns:
            Plain text output
        """
        lines: list[str] = []

        # Sort blocks by reading order
        sorted_blocks = self._sort_blocks(blocks)

        for block in sorted_blocks:
            rendered = self._render_block_plaintext(block)
            if rendered:
                lines.append(rendered)

        return "\n\n".join(lines).strip()

    def _render_block_plaintext(self, block: Block) -> str:
        """Render a single block as plaintext.

        Args:
            block: Block to render

        Returns:
            Plain text string
        """
        block_type = block.type.lower()

        # Check if this is an image/figure block
        is_image_block = block_type in {"image", "image_body", "figure", "chart"}

        if is_image_block:
            return self._render_image_block_plaintext(block)

        # Get text content (prefer corrected_text over text)
        text = block.corrected_text or block.text or ""

        # For title blocks, add a simple prefix
        if block_type == "title":
            return f"[TITLE] {text}"

        # For list blocks, ensure proper formatting
        if block_type in {"list", "list_item"}:
            if not text.startswith(("-", "*", "1.", "2.", "3.")):
                return f"- {text}"

        return text

    def _render_image_block_plaintext(self, block: Block) -> str:
        """Render an image block as plaintext based on image_render_mode.

        Args:
            block: Image block to render

        Returns:
            Plain text string
        """
        parts: list[str] = []

        # Image reference
        if self.image_render_mode in {"image_only", "image_and_description"}:
            if block.image_path:
                parts.append(f"[IMAGE: {block.image_path}]")

        # Description
        if self.image_render_mode in {"description_only", "image_and_description"}:
            if block.description:
                parts.append(f"[FIGURE DESCRIPTION] {block.description}")
            elif not block.image_path:
                if block.text:
                    parts.append(f"[FIGURE] {block.text}")

        return "\n".join(parts)

    def _sort_blocks(self, blocks: list[Block]) -> list[Block]:
        """Sort blocks by reading order.

        Args:
            blocks: List of blocks

        Returns:
            Sorted list of blocks
        """
        # Filter blocks with order
        ranked_blocks = [b for b in blocks if b.order is not None]
        unranked_blocks = [b for b in blocks if b.order is None]

        if ranked_blocks:
            sorted_blocks = sorted(ranked_blocks, key=lambda b: b.order)  # type: ignore
            sorted_blocks.extend(unranked_blocks)
            return sorted_blocks

        return blocks
