"""Reading order analyzer for composing page text from blocks."""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.types import Block

__all__ = ["ReadingOrderAnalyzer", "ColumnOrderingInfo"]


class ReadingOrderAnalyzer:
    """Analyzes and composes text from document blocks in reading order."""

    def compose_page_text(self, processed_blocks: Sequence[Block]) -> str:
        """Compose page-level raw text from processed blocks in reading order.

        Reading order: Uses order field if available, otherwise top-to-bottom (y),
        then left-to-right (x). Includes text-like blocks only and preserves internal
        newlines within each block's text.

        Args:
            processed_blocks: List of processed Block objects with text content

        Returns:
            Composed text from all text-like blocks in reading order
        """
        if not processed_blocks:
            return ""

        # Filter text-like blocks (excludes table, figure, equation, etc.)
        text_like_types = {"plain text", "text", "title", "list"}
        text_blocks = [b for b in processed_blocks if b.type in text_like_types and b.text]

        if not text_blocks:
            return ""

        # Sort by reading order (order field) if available, otherwise by position
        def sort_key(block: Block) -> tuple[float, int, int]:
            rank: float = float(block.order) if block.order is not None else float("inf")
            y = block.bbox.y0
            x = block.bbox.x0
            return (rank, y, x)

        sorted_blocks = sorted(text_blocks, key=sort_key)

        # Compose text
        texts = []
        for block in sorted_blocks:
            text = (block.text or "").strip()
            if text:
                texts.append(text)

        return "\n\n".join(texts)


class ColumnOrderingInfo:
    """Information about column ordering for multi-column layouts."""

    def __init__(
        self,
        column_count: int = 0,
        column_boundaries: list[tuple[float, float]] | None = None,
    ):
        """Initialize column ordering info.

        Args:
            column_count: Number of columns detected
            column_boundaries: List of (left, right) boundaries for each column
        """
        self.column_count = column_count
        self.column_boundaries = column_boundaries or []
