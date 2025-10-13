"""MinerU VLM sorter implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..types import Region

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class MinerUVLMSorter:
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

    def sort(self, regions: list[Region], image: np.ndarray, **kwargs: Any) -> list[Region]:
        """Sort regions using MinerU VLM's ordering information.

        This sorter expects regions to have "index" field from MinerU VLM.
        If index is not present, falls back to simple geometric sorting.

        Args:
            regions: Detected regions (should be from MinerUVLMDetector)
            image: Page image (unused)
            **kwargs: Additional context (unused)

        Returns:
            Sorted regions with reading_order_rank added/updated

        Example:
            >>> sorter = MinerUVLMSorter()
            >>> regions = [
            ...     {"type": "text", "coords": [...], "index": 1, "confidence": 0.9},
            ...     {"type": "text", "coords": [...], "index": 0, "confidence": 0.9},
            ... ]
            >>> sorted_regions = sorter.sort(regions, image)
            >>> [r["reading_order_rank"] for r in sorted_regions]
            [0, 1]
        """
        if not regions:
            return regions

        has_index = all("index" in r for r in regions)

        if not has_index:
            logger.warning(
                "MinerU VLM sorter: regions missing 'index' field. "
                "Did you use MinerUVLMDetector with detection_only=False? "
                "Falling back to simple sort."
            )
            return self._fallback_sort(regions)

        sorted_regions = sorted(regions, key=lambda r: r.get("index", float("inf")))

        for rank, region in enumerate(sorted_regions):
            region["reading_order_rank"] = rank

        logger.debug("Sorted %d regions using MinerU VLM ordering", len(sorted_regions))

        return sorted_regions

    def _fallback_sort(self, regions: list[Region]) -> list[Region]:
        """Fallback to simple geometric sorting."""
        from ..types import ensure_bbox_in_region

        regions = [ensure_bbox_in_region(r) for r in regions]
        sorted_regions = sorted(regions, key=lambda r: (r["bbox"].y0, r["bbox"].x0))

        for rank, region in enumerate(sorted_regions):
            region["reading_order_rank"] = rank

        return sorted_regions

