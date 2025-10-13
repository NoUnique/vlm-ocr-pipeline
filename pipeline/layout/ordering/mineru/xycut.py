"""MinerU XY-Cut sorter implementation.

XY-Cut Algorithm:
- Recursive projection-based sorting
- No specific bbox format requirement (works with unified BBox)
- Splits regions by alternating X and Y projections
- Fast and dependency-light (only numpy)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..types import Region, ensure_bbox_in_region

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class MinerUXYCutSorter:
    """Sorter using MinerU's XY-Cut algorithm.

    XY-Cut is a recursive projection-based algorithm that splits regions
    by alternating between horizontal (X) and vertical (Y) projections.
    It's fast and doesn't require model loading.
    """

    def __init__(self) -> None:
        """Initialize XY-Cut sorter.

        Raises:
            ImportError: If numpy is not available
        """
        try:
            import numpy as np  # noqa: F401
        except ImportError as e:
            raise ImportError("numpy required for XY-Cut sorter") from e

        logger.info("MinerU XY-Cut sorter initialized")

    def sort(self, regions: list[Region], image: np.ndarray, **kwargs: Any) -> list[Region]:
        """Sort regions using XY-Cut algorithm.

        Args:
            regions: Detected regions in unified format
            image: Page image (unused, included for interface compatibility)
            **kwargs: Additional context (unused)

        Returns:
            Sorted regions with reading_order_rank added

        Example:
            >>> sorter = MinerUXYCutSorter()
            >>> sorted_regions = sorter.sort(regions, image)
            >>> sorted_regions[0]["reading_order_rank"]
            0
        """
        if not regions:
            return regions

        regions = [ensure_bbox_in_region(r) for r in regions]

        try:
            import numpy as np

            bboxes = np.array(
                [[r["bbox"].x0, r["bbox"].y0, r["bbox"].x1, r["bbox"].y1] for r in regions],
                dtype=int,
            )

            indices = list(range(len(regions)))
            result_indices: list[int] = []
            self._recursive_xy_cut(bboxes, indices, result_indices)

            sorted_regions = [regions[i] for i in result_indices]

            for rank, region in enumerate(sorted_regions):
                region["reading_order_rank"] = rank

            logger.debug("Sorted %d regions using XY-Cut algorithm", len(sorted_regions))

            return sorted_regions

        except Exception as e:
            logger.error("XY-Cut sorting failed: %s, falling back to simple sort", e)
            return self._fallback_sort(regions)

    def _recursive_xy_cut(
        self, boxes: np.ndarray, indices: list[int], result: list[int]
    ) -> None:
        """Recursive XY-Cut algorithm.

        Adapted from MinerU's implementation.

        Args:
            boxes: Array of bboxes (N, 4) in [x0, y0, x1, y1] format
            indices: Current indices mapping to original regions
            result: Output list to accumulate sorted indices
        """

        if len(boxes) == 0:
            return

        y_sorted_idx = boxes[:, 1].argsort()
        y_sorted_boxes = boxes[y_sorted_idx]
        y_sorted_indices = [indices[i] for i in y_sorted_idx]

        y_projection = self._projection_by_bboxes(y_sorted_boxes, axis=1)
        pos_y = self._split_projection_profile(y_projection, min_value=0, min_gap=1)

        if pos_y is None:
            return

        arr_y0, arr_y1 = pos_y

        for r0, r1 in zip(arr_y0, arr_y1, strict=False):
            mask = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)
            band_boxes = y_sorted_boxes[mask]
            band_indices = [y_sorted_indices[i] for i, m in enumerate(mask) if m]

            if len(band_boxes) == 0:
                continue

            x_sorted_idx = band_boxes[:, 0].argsort()
            x_sorted_boxes = band_boxes[x_sorted_idx]
            x_sorted_indices = [band_indices[i] for i in x_sorted_idx]

            x_projection = self._projection_by_bboxes(x_sorted_boxes, axis=0)
            pos_x = self._split_projection_profile(x_projection, min_value=0, min_gap=1)

            if pos_x is None:
                result.extend(x_sorted_indices)
                continue

            arr_x0, arr_x1 = pos_x

            if len(arr_x0) == 1:
                result.extend(x_sorted_indices)
                continue

            for c0, c1 in zip(arr_x0, arr_x1, strict=False):
                mask = (c0 <= x_sorted_boxes[:, 0]) & (x_sorted_boxes[:, 0] < c1)
                column_boxes = x_sorted_boxes[mask]
                column_indices = [x_sorted_indices[i] for i, m in enumerate(mask) if m]

                self._recursive_xy_cut(column_boxes, column_indices, result)

    def _projection_by_bboxes(self, boxes: np.ndarray, axis: int) -> np.ndarray:
        """Get projection histogram along specified axis.

        Args:
            boxes: Array of bboxes (N, 4)
            axis: 0 for X-axis (horizontal), 1 for Y-axis (vertical)

        Returns:
            1D projection histogram
        """
        import numpy as np

        assert axis in [0, 1], "axis must be 0 (X) or 1 (Y)"

        length = int(np.max(boxes[:, axis::2]))
        result = np.zeros(length, dtype=int)

        for start, end in boxes[:, axis::2]:
            start_idx = int(start)
            end_idx = int(end)
            if 0 <= start_idx < length and 0 <= end_idx <= length:
                result[start_idx:end_idx] += 1

        return result

    def _split_projection_profile(
        self, arr_values: np.ndarray, min_value: float, min_gap: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Split projection profile into groups.

        Args:
            arr_values: 1D projection histogram
            min_value: Minimum projection value threshold
            min_gap: Minimum gap size between groups

        Returns:
            Tuple of (start_indices, end_indices) or None if no groups found
        """
        import numpy as np

        arr_index = np.where(arr_values > min_value)[0]

        if len(arr_index) == 0:
            return None

        arr_diff = arr_index[1:] - arr_index[:-1]
        arr_diff_index = np.where(arr_diff > min_gap)[0]

        arr_zero_intvl_start = arr_index[arr_diff_index]
        arr_zero_intvl_end = arr_index[arr_diff_index + 1]

        arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
        arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
        arr_end += 1

        return arr_start, arr_end

    def _fallback_sort(self, regions: list[Region]) -> list[Region]:
        """Fallback to simple geometric sorting."""
        if not regions:
            return regions

        regions = [ensure_bbox_in_region(r) for r in regions]
        sorted_regions = sorted(regions, key=lambda r: (r["bbox"].y0, r["bbox"].x0))

        for rank, region in enumerate(sorted_regions):
            region["reading_order_rank"] = rank

        return sorted_regions

