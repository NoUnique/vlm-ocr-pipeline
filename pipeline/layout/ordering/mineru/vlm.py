"""MinerU VLM sorter implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pipeline.types import Block, Sorter

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class MinerUVLMSorter(Sorter):
    """Sorter using MinerU VLM model's built-in ordering.

    This sorter is designed to work with regions that already have
    ordering information from MinerU VLM detection. It simply extracts
    and applies the ordering index.

    Note: This should typically be used with MinerUVLMDetector
    to ensure ordering information is available.
    """

    def __init__(self) -> None:
        """Initialize MinerU VLM sorter."""
        logger.info("MinerU VLM sorter initialized")

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Sort regions using MinerU VLM's ordering information.

        This sorter expects regions to have "index" field from MinerU VLM.
        If index is not present, falls back to simple geometric sorting.

        Args:
            regions: Detected regions (should be from MinerUVLMDetector)
            image: Page image (unused)
            **kwargs: Additional context (unused)

        Returns:
            Sorted blocks with reading_order_rank added/updated

        Example:
            >>> sorter = MinerUVLMSorter()
            >>> regions = [
            ...     {"type": "text", "coords": [...], "index": 1, "confidence": 0.9},
            ...     {"type": "text", "coords": [...], "index": 0, "confidence": 0.9},
            ... ]
            >>> sorted_blocks = sorter.sort(regions, image)
            >>> [r["reading_order_rank"] for r in sorted_blocks]
            [0, 1]
        """
        if not blocks:
            return blocks

        has_index = all(r.index is not None for r in blocks)

        if not has_index:
            logger.warning(
                "MinerU VLM sorter: blocks missing 'index' field. "
                "Did you use MinerUVLMDetector with detection_only=False? "
                "Falling back to simple sort."
            )
            return self._fallback_sort(blocks)

        sorted_blocks = sorted(blocks, key=lambda r: r.index if r.index is not None else float("inf"))

        for rank, block in enumerate(sorted_blocks):
            block.order = rank

        logger.debug("Sorted %d blocks using MinerU VLM ordering", len(sorted_blocks))

        return sorted_blocks

    def _fallback_sort(self, blocks: list[Block]) -> list[Block]:
        """Fallback to simple geometric sorting."""
        sorted_blocks = sorted(blocks, key=lambda r: (r.bbox.y0, r.bbox.x0))

        for rank, block in enumerate(sorted_blocks):
            block.order = rank

        return sorted_blocks
