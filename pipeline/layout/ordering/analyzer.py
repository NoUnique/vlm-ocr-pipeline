"""Reading order analyzer for determining the natural reading flow of document regions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np

from .multi_column import column_boxes

logger = logging.getLogger(__name__)

COLUMN_ORDER_EPSILON = 1e-6


@dataclass
class ColumnBand:
    """Represents a provisional column band before serialization."""

    x0: float
    x1: float
    centers: list[float]

    def add_rect(self, x0: float, x1: float, center: float) -> None:
        """Add a rectangle to this column band."""
        self.x0 = min(self.x0, x0)
        self.x1 = max(self.x1, x1)
        self.centers.append(center)

    @property
    def center(self) -> float:
        """Get the average center of all rectangles in this band."""
        return sum(self.centers) / len(self.centers)

    @property
    def width(self) -> float:
        """Get the width of this column band."""
        return self.x1 - self.x0

    def to_output_dict(self, index: int) -> ColumnOrderingInfo:
        """Convert to output dictionary format."""
        return cast(
            ColumnOrderingInfo,
            {
                "index": index,
                "x0": float(self.x0),
                "x1": float(self.x1),
                "center": float(self.center),
                "width": float(self.width),
            },
        )


class ColumnOrderingInfo(TypedDict):
    """Information about a column for ordering purposes."""

    index: int
    x0: float
    x1: float
    center: float
    width: float


@dataclass
class RegionColumnMetrics:
    """Metrics for assigning a region to a column."""

    column_index: int
    order_key: tuple[float, float, float, float]
    overlap_ratio: float


class ReadingOrderAnalyzer:
    """Analyzes and determines reading order of document regions.
    
    This class handles:
    - Multi-column detection using PyMuPDF
    - Region sorting based on columns
    - Reading order rank assignment
    - Composition of page-level text in natural reading order
    """

    def __init__(self, enable_multi_column: bool = False):
        """Initialize the reading order analyzer.
        
        Args:
            enable_multi_column: Whether to enable multi-column detection and ordering
        """
        self.enable_multi_column = enable_multi_column

    def analyze_reading_order(
        self,
        regions: list[dict[str, Any]],
        page_image: np.ndarray,
        pymupdf_page: Any | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Analyze and determine reading order for regions.
        
        Args:
            regions: List of detected regions
            page_image: Page image for dimension reference
            pymupdf_page: PyMuPDF page object for column detection (optional)
            
        Returns:
            Tuple of (sorted regions with reading order, column layout info)
        """
        if not self.enable_multi_column or pymupdf_page is None:
            return regions, None

        column_layout = self._detect_column_layout(pymupdf_page, page_image)
        
        if column_layout is None:
            return regions, None

        ordering_columns = column_layout.get("_ordering_columns")
        if not ordering_columns:
            return regions, None

        typed_columns = cast(Sequence[ColumnOrderingInfo], ordering_columns)
        sorted_regions = self._sort_regions_by_columns(
            regions,
            typed_columns,
            assign_rank=True,
            store_column_index=True,
        )

        # Remove internal ordering columns from output
        column_layout.pop("_ordering_columns", None)
        
        return sorted_regions, column_layout

    def _detect_column_layout(
        self,
        pymupdf_page: Any,
        page_image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Detect column layout using PyMuPDF."""
        try:
            detected_boxes = column_boxes(pymupdf_page)
        except Exception as exc:  # pragma: no cover - PyMuPDF failure path
            logger.debug("PyMuPDF column detection failed: %s", exc)
            return None

        if not detected_boxes:
            return None

        page_rect = pymupdf_page.rect
        if page_rect.width == 0 or page_rect.height == 0:
            return None

        image_height, image_width = page_image.shape[0], page_image.shape[1]
        scale_x = image_width / float(page_rect.width)
        scale_y = image_height / float(page_rect.height)

        image_rects: list[tuple[int, int, int, int]] = []
        for rect in detected_boxes:
            image_rects.append(
                (
                    int(round(rect.x0 * scale_x)),
                    int(round(rect.y0 * scale_y)),
                    int(round(rect.x1 * scale_x)),
                    int(round(rect.y1 * scale_y)),
                )
            )

        column_bands = self._merge_column_rects(image_rects, image_width)
        if len(column_bands) <= 1:
            return None

        ordering_columns: list[ColumnOrderingInfo] = []
        columns_for_output: list[dict[str, float | int]] = []
        for idx, band in enumerate(column_bands):
            ordering_columns.append(band.to_output_dict(idx))
            columns_for_output.append(
                {
                    "index": idx,
                    "x0": int(round(band.x0)),
                    "x1": int(round(band.x1)),
                    "center": float(band.center),
                    "width": float(band.width),
                }
            )

        return {
            "columns": columns_for_output,
            "_ordering_columns": ordering_columns,
            "image_rects": [
                {"x0": rect[0], "y0": rect[1], "x1": rect[2], "y1": rect[3]} for rect in image_rects
            ],
            "scale": {"x": scale_x, "y": scale_y},
        }

    def _merge_column_rects(
        self,
        rects: list[tuple[int, int, int, int]],
        page_width: int,
    ) -> list[ColumnBand]:
        """Merge rectangles with similar horizontal centers into column bands."""
        if not rects:
            return []

        columns: list[ColumnBand] = []
        grouping_threshold = max(int(page_width * 0.05), 25)

        for rect in rects:
            x0, _, x1, _ = rect
            center = (x0 + x1) / 2.0
            added = False

            for column in columns:
                column_center = column.center
                column_width = column.width
                threshold = max(grouping_threshold, column_width)
                if abs(center - column_center) <= threshold:
                    column.add_rect(float(x0), float(x1), center)
                    added = True
                    break

            if not added:
                columns.append(ColumnBand(float(x0), float(x1), [center]))

        columns.sort(key=lambda col: col.x0)
        return columns

    def _compute_region_column_metrics(
        self,
        region: dict[str, Any],
        columns: Sequence[ColumnOrderingInfo],
    ) -> RegionColumnMetrics:
        """Compute metrics for assigning a region to a column."""
        coords = region.get("coords") or [0, 0, 0, 0]
        try:
            x, y, w, _ = (float(value) for value in coords[:4])
        except Exception:
            x = y = w = 0.0

        x1 = x + w
        region_center_x = x + (w / 2.0)
        region_width = max(w, 1.0)

        best_index = int(columns[0]["index"])
        best_overlap = -1.0
        best_distance = float("inf")

        for column in columns:
            col_x0 = column["x0"]
            col_x1 = column["x1"]
            overlap = max(0.0, min(x1, col_x1) - max(x, col_x0))
            overlap_ratio = overlap / region_width
            distance = abs(region_center_x - column["center"])

            if overlap_ratio > best_overlap or (
                abs(overlap_ratio - best_overlap) <= COLUMN_ORDER_EPSILON and distance < best_distance
            ):
                best_overlap = overlap_ratio
                best_distance = distance
                best_index = int(column["index"])

        if best_overlap <= 0:
            nearest_column = min(columns, key=lambda col: abs(region_center_x - col["center"]))
            best_index = int(nearest_column["index"])
            best_distance = abs(region_center_x - nearest_column["center"])

        order_key = (
            float(best_index),
            float(y),
            float(x),
            float(best_distance),
        )

        return RegionColumnMetrics(
            column_index=best_index,
            order_key=order_key,
            overlap_ratio=float(max(best_overlap, 0.0)),
        )

    def _sort_regions_by_columns(
        self,
        regions: list[dict[str, Any]],
        columns: Sequence[ColumnOrderingInfo],
        assign_rank: bool = False,
        store_column_index: bool = False,
    ) -> list[dict[str, Any]]:
        """Sort regions by column-aware reading order."""
        if not regions:
            return regions

        keyed_regions: list[tuple[tuple[float, float, float, float], dict[str, Any], int]] = []

        for region in regions:
            metrics = self._compute_region_column_metrics(region, columns)
            keyed_regions.append((metrics.order_key, region, metrics.column_index))

        keyed_regions.sort(key=lambda item: item[0])

        sorted_regions: list[dict[str, Any]] = []
        for rank, (_, region, column_index) in enumerate(keyed_regions):
            if assign_rank:
                region["reading_order_rank"] = rank
            if store_column_index:
                region["column_index"] = int(column_index)
            sorted_regions.append(region)

        return sorted_regions

    def compose_page_text(self, processed_regions: list[dict[str, Any]]) -> str:
        """Compose page-level raw text from processed regions in reading order.

        Reading order: Uses reading_order_rank if available, otherwise top-to-bottom (y),
        then left-to-right (x). Includes text-like regions only and preserves internal
        newlines within each region's text.
        
        Args:
            processed_regions: List of processed regions with text content
            
        Returns:
            Composed text in natural reading order
        """
        if not isinstance(processed_regions, list):
            return ""
            
        text_like_types = {"plain text", "title", "list"}
        sortable_regions: list[tuple[int, int, str, int | None]] = []
        
        for region in processed_regions:
            if not isinstance(region, dict):
                continue
            region_type = region.get("type")
            if region_type not in text_like_types:
                continue
            coords = region.get("coords") or [0, 0, 0, 0]
            try:
                x, y = int(coords[0]), int(coords[1])
            except Exception:
                x, y = 0, 0
            text_value = region.get("text")
            if isinstance(text_value, str) and text_value.strip():
                # Keep internal newlines; trim outer whitespace only
                order_rank = region.get("reading_order_rank")
                sortable_regions.append((y, x, text_value.strip(), order_rank))
                
        # Sort by reading order rank if available, otherwise by y then x
        if sortable_regions:
            if any(item[3] is not None for item in sortable_regions):
                sortable_regions.sort(
                    key=lambda item: (
                        0 if item[3] is not None else 1,
                        item[3] if item[3] is not None else item[0],
                        item[0],
                        item[1],
                    )
                )
            else:
                sortable_regions.sort(key=lambda item: (item[0], item[1]))

        # Join with a blank line between regions to separate blocks
        return "\n\n".join(item[2] for item in sortable_regions)

