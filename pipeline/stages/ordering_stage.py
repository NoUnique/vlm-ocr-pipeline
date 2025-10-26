"""Ordering Stage: Reading order analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.types import Block, Sorter


class OrderingStage:
    """Stage 3: BlockOrdering - Reading order analysis."""

    def __init__(self, sorter: Sorter):
        """Initialize OrderingStage.

        Args:
            sorter: Block sorter instance
        """
        self.sorter = sorter

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Sort blocks by reading order.

        Args:
            blocks: List of detected blocks
            image: Page image as numpy array
            **kwargs: Additional arguments for sorter

        Returns:
            List of blocks with order and optionally column_index
        """
        sorted_blocks = self.sorter.sort(blocks, image, **kwargs)
        return sorted_blocks
