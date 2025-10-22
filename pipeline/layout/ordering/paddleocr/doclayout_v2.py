"""PP-DocLayoutV2 sorter.

PP-DocLayoutV2 detector uses a lightweight pointer network (6 Transformer layers)
to restore reading order during detection. This sorter preserves that ordering
instead of re-sorting with a different algorithm.

This sorter also includes overlap filtering to remove redundant duplicate detections,
matching the behavior of PaddleOCR-VL's original implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pipeline.types import BBox, Block, Sorter

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class PPDocLayoutV2Sorter(Sorter):
    """Sorter for PP-DocLayoutV2 detector that preserves detector's ordering.

    PP-DocLayoutV2 detector sorts blocks by Y-coordinate (top-to-bottom) during detection.
    This sorter simply preserves that pre-sorted ordering without re-sorting.

    Note on Pointer Network:
    While the PP-DocLayoutV2 model includes a pointer network for reading order,
    PaddleOCR/PaddleX Python API does NOT expose this information.
    The detector falls back to Y-coordinate sorting, which this sorter preserves.

    Coupling:
        This sorter is tightly coupled with the paddleocr-doclayout-v2 detector.
        It assumes blocks already have their 'order' field set by the detector.
    """

    def __init__(self, overlap_threshold: float = 0.7) -> None:
        """Initialize PP-DocLayoutV2 sorter.

        Args:
            overlap_threshold: Overlap ratio threshold for filtering duplicate detections.
                             Default 0.7 matches PaddleOCR-VL's implementation.
        """
        self.overlap_threshold = overlap_threshold
        logger.info(
            "PP-DocLayoutV2 sorter initialized (preserves detector's Y-coordinate ordering, "
            "overlap_threshold=%.2f)",
            overlap_threshold,
        )

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Preserve Y-coordinate ordering from PP-DocLayoutV2 detector.

        This method performs two steps:
        1. Filter overlapping blocks (remove redundant duplicate detections)
        2. Preserve Y-coordinate ordering from detector

        Args:
            blocks: Detected blocks with 'order' already set by PP-DocLayoutV2 detector
            image: Page image (unused, included for interface compatibility)
            **kwargs: Additional context (unused)

        Returns:
            Blocks in their original order (Y-coordinate sorted by detector),
            with overlapping duplicates removed.

        Note:
            If blocks don't have 'order' set (e.g., from a different detector),
            this sorter will apply a simple top-to-bottom, left-to-right fallback.
        """
        if not blocks:
            return blocks

        # Step 1: Filter overlapping blocks (matching PaddleOCR-VL's behavior)
        original_count = len(blocks)
        blocks = self._filter_overlap_blocks(blocks)
        if len(blocks) < original_count:
            logger.debug(
                "Filtered %d overlapping blocks (%d → %d)",
                original_count - len(blocks),
                original_count,
                len(blocks),
            )

        # Step 2: Check if blocks already have order (from PP-DocLayoutV2)
        has_order = all(block.order is not None for block in blocks)

        if has_order:
            # Blocks are already ordered by PP-DocLayoutV2 detector (Y-coordinate sorting)
            # Just ensure they're sorted by the order field
            sorted_blocks = sorted(blocks, key=lambda b: b.order if b.order is not None else 0)
            logger.debug(
                "Preserved Y-coordinate ordering from PP-DocLayoutV2 for %d blocks",
                len(sorted_blocks),
            )
            return sorted_blocks

        # Fallback: If order is not set, apply simple geometric sorting
        logger.warning(
            "Blocks from non-PP-DocLayoutV2 detector detected. "
            "Applying fallback top-to-bottom, left-to-right sorting."
        )
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

        # Set order for consistency
        for rank, block in enumerate(sorted_blocks):
            block.order = rank

        return sorted_blocks

    def _filter_overlap_blocks(self, blocks: list[Block]) -> list[Block]:
        """Filter overlapping blocks to remove redundant duplicate detections.

        Matches PaddleOCR-VL's filter_overlap_boxes behavior:
        - Excludes 'ref_text' blocks from overlap checking (equivalent to 'reference')
        - Calculates overlap ratio for all block pairs
        - If overlap > threshold, removes smaller block
        - Exception: preserves overlaps where one block is 'image' and the other isn't

        Args:
            blocks: List of detected blocks

        Returns:
            List of blocks with overlapping duplicates removed
        """
        if not blocks:
            return blocks

        # Exclude 'ref_text' blocks from overlap filtering (equivalent to 'reference' in PaddleOCR)
        filterable_blocks = [b for b in blocks if b.type != "ref_text"]
        excluded_blocks = [b for b in blocks if b.type == "ref_text"]

        dropped_indexes: set[int] = set()

        # Check all pairs of blocks for overlap
        for i in range(len(filterable_blocks)):
            for j in range(i + 1, len(filterable_blocks)):
                # Skip if either block is already marked for removal
                if i in dropped_indexes or j in dropped_indexes:
                    continue

                block_i = filterable_blocks[i]
                block_j = filterable_blocks[j]

                # Calculate overlap ratio (relative to smaller box)
                overlap_ratio = self._calculate_overlap_ratio(block_i.bbox, block_j.bbox)

                if overlap_ratio > self.overlap_threshold:
                    # Calculate areas to determine which is smaller
                    area_i = self._calculate_bbox_area(block_i.bbox)
                    area_j = self._calculate_bbox_area(block_j.bbox)

                    # Exception: preserve overlaps between 'image' and other types
                    if (block_i.type == "image" or block_j.type == "image") and block_i.type != block_j.type:
                        continue

                    # Remove smaller box
                    if area_i >= area_j:
                        dropped_indexes.add(j)
                        logger.debug(
                            "Dropping block %d (area=%.0f) due to overlap %.2f with block %d (area=%.0f)",
                            j,
                            area_j,
                            overlap_ratio,
                            i,
                            area_i,
                        )
                    else:
                        dropped_indexes.add(i)
                        logger.debug(
                            "Dropping block %d (area=%.0f) due to overlap %.2f with block %d (area=%.0f)",
                            i,
                            area_i,
                            overlap_ratio,
                            j,
                            area_j,
                        )

        # Filter out dropped blocks
        filtered_blocks = [
            block for idx, block in enumerate(filterable_blocks) if idx not in dropped_indexes
        ]

        # Re-add excluded blocks (ref_text)
        return filtered_blocks + excluded_blocks

    @staticmethod
    def _calculate_bbox_area(bbox: BBox) -> float:
        """Calculate bounding box area.

        Args:
            bbox: Bounding box

        Returns:
            Area in pixels
        """
        return abs((bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0))

    @staticmethod
    def _calculate_overlap_ratio(bbox1: BBox, bbox2: BBox) -> float:
        """Calculate overlap ratio between two bounding boxes.

        Uses "small" mode: overlap area / smaller box area.
        This matches PaddleOCR-VL's calculate_overlap_ratio(..., mode="small").

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            Overlap ratio (0.0 to 1.0+)
        """
        # Calculate intersection
        x_min_inter = max(bbox1.x0, bbox2.x0)
        y_min_inter = max(bbox1.y0, bbox2.y0)
        x_max_inter = min(bbox1.x1, bbox2.x1)
        y_max_inter = min(bbox1.y1, bbox2.y1)

        # Calculate intersection area
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        if inter_area == 0:
            return 0.0

        # Calculate areas
        area1 = abs((bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0))
        area2 = abs((bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0))

        # Return overlap ratio relative to smaller box ("small" mode)
        smaller_area = min(area1, area2)
        if smaller_area == 0:
            return 0.0

        return inter_area / smaller_area
