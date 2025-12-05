"""Ordering Stage: Reading order analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

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

        sorted_blocks = self.sorter.sort(input_data, image, **context)
        return sorted_blocks

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Sort blocks by reading order.

        Legacy method for backward compatibility.

        Args:
            blocks: List of detected blocks
            image: Page image as numpy array
            **kwargs: Additional arguments for sorter

        Returns:
            List of blocks with order and optionally column_index
        """
        return self.process(blocks, image=image, **kwargs)
