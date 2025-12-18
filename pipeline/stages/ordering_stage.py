"""Ordering Stage: Reading order analysis."""

from __future__ import annotations

from typing import Any

from pipeline.types import Block, Sorter

from .base import BaseStage


class OrderingStage(BaseStage[list[Block], list[Block]]):
    """Stage 3: BlockOrdering - Reading order analysis.

    This stage determines the reading order of detected blocks,
    optionally detecting multi-column layouts.
    """

    name = "ordering"

    def __init__(self, sorter: Sorter):
        """Initialize OrderingStage.

        Args:
            sorter: Block sorter instance
        """
        self.sorter = sorter

    def _process_impl(self, input_data: list[Block], **context: Any) -> list[Block]:
        """Sort blocks by reading order.

        Args:
            input_data: List of detected blocks
            **context: Must include 'image' (page image as numpy array)

        Returns:
            List of blocks with order and optionally column_index
        """
        image = context.get("image")
        if image is None:
            raise ValueError("OrderingStage requires 'image' in context")

        # Remove 'image' from context to avoid duplicate argument error
        remaining_context = {k: v for k, v in context.items() if k != "image"}
        sorted_blocks = self.sorter.sort(input_data, image, **remaining_context)
        return sorted_blocks
