"""Reading order analyzer for composing page text from regions."""

from __future__ import annotations

from typing import Any

__all__ = ["ReadingOrderAnalyzer", "ColumnOrderingInfo"]


class ReadingOrderAnalyzer:
    """Analyzes and composes text from document regions in reading order."""

    def compose_page_text(self, processed_blocks: list[dict[str, Any]]) -> str:
        """Compose page-level raw text from processed regions in reading order.

        Reading order: Uses reading_order_rank if available, otherwise top-to-bottom (y),
        then left-to-right (x). Includes text-like regions only and preserves internal
        newlines within each region's text.

        Args:
            processed_blocks: List of processed regions with text content

        Returns:
            Composed text from all text-like regions in reading order
        """
        if not processed_blocks:
            return ""

        # Filter text-like regions (excludes table, figure, equation, etc.)
        text_like_types = {"plain text", "text", "title", "list"}
        text_regions = [
            r for r in processed_blocks if isinstance(r, dict) and r.get("type") in text_like_types and r.get("text")
        ]

        if not text_regions:
            return ""

        # Sort by reading order rank if available, otherwise by position
        def sort_key(region: dict[str, Any]) -> tuple[int, float, float]:
            rank = region.get("reading_order_rank", float("inf"))
            coords = region.get("coords", [0, 0, 0, 0])
            y = coords[1] if len(coords) > 1 else 0
            x = coords[0] if len(coords) > 0 else 0
            return (rank, y, x)

        sorted_blocks = sorted(text_regions, key=sort_key)

        # Compose text
        texts = []
        for block in sorted_blocks:
            text = block.get("text", "").strip()
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
