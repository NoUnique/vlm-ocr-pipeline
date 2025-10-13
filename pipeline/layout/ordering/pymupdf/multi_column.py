"""Multi-column detection and sorting using PyMuPDF.

PyMuPDF BBox Format:
- Input/Output: Rect(x0, y0, x1, y1) or IRect(x0, y0, x1, y1)
- Origin: Top-Left (0, 0)
- Coordinate Order: Left-Top + Right-Bottom
- Example: Rect(100, 50, 300, 200) means rectangle from (100,50) to (300,200)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    import numpy as np

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["column_boxes", "MultiColumnSorter"]

_MIN_NBLOCK_LENGTH = 2
_BOTTOM_TOLERANCE = 10

PyMuPDFModule = Any
PyMuPDFPage = Any
PyMuPDFRect = Any
PyMuPDFIRect = Any


def column_boxes(
    page: PyMuPDFPage,
    footer_margin: int = 50,
    header_margin: int = 50,
    no_image_text: bool = True,
) -> list[PyMuPDFIRect]:
    """Determine bounding boxes that wrap text columns on a page."""

    fitz_module = _ensure_pymupdf_available()

    clip = _create_clip_rect(fitz_module, page, footer_margin, header_margin)
    path_bboxes = _extract_path_bboxes(fitz_module, page)
    img_bboxes = _extract_image_bboxes(fitz_module, page)
    text_bboxes, vert_bboxes = _extract_text_bboxes(
        fitz_module,
        page,
        clip,
        img_bboxes,
        no_image_text,
    )

    if not text_bboxes:
        return []

    text_bboxes = _sort_text_bboxes(text_bboxes, path_bboxes)
    text_bboxes = _extend_right(
        text_bboxes,
        int(page.rect.width),
        path_bboxes,
        vert_bboxes,
        img_bboxes,
    )

    if not text_bboxes:
        return []

    joined_bboxes = _join_bboxes(text_bboxes, path_bboxes, vert_bboxes)
    return _clean_nblocks(joined_bboxes)


def _ensure_pymupdf_available() -> PyMuPDFModule:
    if fitz is None:  # pragma: no cover - guard optional dependency
        raise ImportError("PyMuPDF (fitz) is required for column detection")
    return cast(PyMuPDFModule, fitz)


def _create_clip_rect(
    _fitz: PyMuPDFModule,
    page: PyMuPDFPage,
    footer_margin: int,
    header_margin: int,
) -> PyMuPDFRect:
    clip = +page.rect
    clip.y1 -= footer_margin
    clip.y0 += header_margin
    return clip


def _extract_path_bboxes(_fitz: PyMuPDFModule, page: PyMuPDFPage) -> list[PyMuPDFIRect]:
    path_bboxes: list[PyMuPDFIRect] = []
    for path in page.get_drawings():
        path_bboxes.append(cast(PyMuPDFIRect, path["rect"].irect))
    path_bboxes.sort(key=lambda rect: (rect.y0, rect.x0))
    return path_bboxes


def _extract_image_bboxes(_fitz: PyMuPDFModule, page: PyMuPDFPage) -> list[PyMuPDFIRect]:
    img_bboxes: list[PyMuPDFIRect] = []
    for item in page.get_images():
        img_bboxes.extend(cast(Sequence[PyMuPDFIRect], page.get_image_rects(item[0])))
    return img_bboxes


def _extract_text_bboxes(
    fitz_module: PyMuPDFModule,
    page: PyMuPDFPage,
    clip: PyMuPDFRect,
    img_bboxes: Sequence[PyMuPDFIRect],
    no_image_text: bool,
) -> tuple[list[PyMuPDFIRect], list[PyMuPDFIRect]]:
    bboxes: list[PyMuPDFIRect] = []
    vert_bboxes: list[PyMuPDFIRect] = []

    blocks = page.get_text(
        "dict",
        flags=fitz_module.TEXTFLAGS_TEXT,
        clip=clip,
    )["blocks"]

    for block in blocks:
        bbox = fitz_module.IRect(block["bbox"])

        if no_image_text and _in_bbox(bbox, img_bboxes):
            continue

        first_line = block["lines"][0]
        if first_line["dir"] != (1, 0):
            vert_bboxes.append(bbox)
            continue

        srect = fitz_module.EMPTY_IRECT()
        for line in block["lines"]:
            line_bbox = fitz_module.IRect(line["bbox"])
            text = "".join(span["text"].strip() for span in line["spans"])
            if len(text) > 1:
                srect |= line_bbox

        bbox = +srect
        if not bbox.is_empty:
            bboxes.append(bbox)

    return bboxes, vert_bboxes


def _sort_text_bboxes(
    bboxes: list[PyMuPDFIRect],
    path_bboxes: Sequence[PyMuPDFIRect],
) -> list[PyMuPDFIRect]:
    bboxes.sort(key=lambda rect: (_in_bbox(rect, path_bboxes), rect.y0, rect.x0))
    return bboxes


def _extend_right(
    bboxes: list[PyMuPDFIRect],
    width: int,
    path_bboxes: Sequence[PyMuPDFIRect],
    vert_bboxes: Sequence[PyMuPDFIRect],
    img_bboxes: Sequence[PyMuPDFIRect],
) -> list[PyMuPDFIRect]:
    for index, bbox in enumerate(bboxes):
        if _in_bbox(bbox, path_bboxes):
            continue

        if _in_bbox(bbox, img_bboxes):
            continue

        temp_rect = +bbox
        temp_rect.x1 = width

        combined_blocks = list(path_bboxes) + list(vert_bboxes) + list(img_bboxes)

        if _intersects_bboxes(temp_rect, combined_blocks):
            continue

        if _can_extend(temp_rect, bbox, bboxes, vert_bboxes):
            bboxes[index] = temp_rect

    return [rect for rect in bboxes if rect is not None]


def _join_bboxes(
    bboxes: list[PyMuPDFIRect],
    path_bboxes: Sequence[PyMuPDFIRect],
    vert_bboxes: Sequence[PyMuPDFIRect],
) -> list[PyMuPDFIRect]:
    nblocks: list[PyMuPDFIRect] = [bboxes[0]]
    remaining: list[PyMuPDFIRect | None] = list(bboxes[1:])

    for index, bbox in enumerate(remaining):
        if bbox is None:
            continue

        merged, merge_index, temp_rect = _find_merge_candidate(
            bbox,
            nblocks,
            path_bboxes,
            vert_bboxes,
        )

        if not merged:
            nblocks.append(bbox)
            merge_index = len(nblocks) - 1
            temp_rect = nblocks[merge_index]

        if not _can_extend(temp_rect, bbox, remaining, vert_bboxes):
            nblocks.append(bbox)
        else:
            nblocks[merge_index] = temp_rect

        remaining[index] = None

    return nblocks


def _find_merge_candidate(
    bbox: PyMuPDFIRect,
    nblocks: Sequence[PyMuPDFIRect],
    path_bboxes: Sequence[PyMuPDFIRect],
    vert_bboxes: Sequence[PyMuPDFIRect],
) -> tuple[bool, int, PyMuPDFIRect]:
    for new_index, new_bbox in enumerate(nblocks):
        if new_bbox.x1 < bbox.x0 or bbox.x1 < new_bbox.x0:
            continue

        if _in_bbox(new_bbox, path_bboxes) != _in_bbox(bbox, path_bboxes):
            continue

        candidate = bbox | new_bbox
        if _can_extend(candidate, new_bbox, nblocks, vert_bboxes):
            return True, new_index, candidate

    return False, 0, bbox


def _clean_nblocks(nblocks: list[PyMuPDFIRect]) -> list[PyMuPDFIRect]:
    blen = len(nblocks)
    if blen < _MIN_NBLOCK_LENGTH:
        return nblocks

    start = blen - 1
    for idx in range(start, -1, -1):
        bb1 = nblocks[idx]
        bb0 = nblocks[idx - 1]
        if bb0 == bb1:
            del nblocks[idx]

    y1 = nblocks[0].y1
    i0 = 0
    i1 = -1

    for idx in range(1, len(nblocks)):
        bbox1 = nblocks[idx]
        if abs(bbox1.y1 - y1) > _BOTTOM_TOLERANCE:
            if i1 > i0:
                nblocks[i0 : i1 + 1] = sorted(
                    nblocks[i0 : i1 + 1],
                    key=lambda rect: rect.x0,
                )
            y1 = bbox1.y1
            i0 = idx
        i1 = idx

    if i1 > i0:
        nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda rect: rect.x0)
    return nblocks


def _in_bbox(bb: PyMuPDFIRect, target_boxes: Sequence[PyMuPDFIRect]) -> int:
    for index, bbox in enumerate(target_boxes):
        if bb in bbox:
            return index + 1
    return 0


def _intersects_bboxes(bb: PyMuPDFIRect, bboxes_to_check: Sequence[PyMuPDFIRect]) -> bool:
    for bbox in bboxes_to_check:
        if not (bb & bbox).is_empty:
            return True
    return False


def _can_extend(
    temp: PyMuPDFIRect,
    bbox: PyMuPDFIRect,
    bbox_list: Sequence[PyMuPDFIRect | None],
    vert_bboxes: Sequence[PyMuPDFIRect],
) -> bool:
    for existing in bbox_list:
        if not _intersects_bboxes(temp, vert_bboxes) and (
            existing is None or existing is bbox or (temp & existing).is_empty
        ):
            continue
        return False

    return True


COLUMN_ORDER_EPSILON = 1e-6


class ColumnInfo(TypedDict):
    """Column information for ordering."""
    index: int
    bbox: Any  # BBox from types module
    center: float
    width: float


class MultiColumnSorter:
    """Multi-column aware reading order sorter using PyMuPDF.
    
    Detects columns using PyMuPDF text block extraction (column_boxes),
    then sorts regions by column-aware reading order:
    left column top-to-bottom, then right column top-to-bottom.
    """
    
    def __init__(self) -> None:
        """Initialize PyMuPDF sorter."""
        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for PyMuPDF sorter. "
                "Install with: pip install pymupdf"
            )
        logger.info("PyMuPDF sorter initialized")
    
    def sort(self, regions: list[Any], image: np.ndarray, **kwargs: Any) -> list[Any]:
        """Sort regions using PyMuPDF multi-column detection.
        
        Args:
            regions: Detected regions in unified format
            image: Page image for dimension reference
            **kwargs: Required context:
                - pymupdf_page: PyMuPDF page object for column detection
                
        Returns:
            Sorted regions with reading_order_rank and column_index added
        """
        
        pymupdf_page = kwargs.get("pymupdf_page")
        
        if not pymupdf_page:
            logger.warning("PyMuPDF page not provided, falling back to simple ordering")
            return self._fallback_sort(regions)
        
        column_layout = self._detect_column_layout(pymupdf_page, image)
        
        if not column_layout:
            logger.debug("No multi-column layout detected, using simple ordering")
            return self._fallback_sort(regions)
        
        columns = column_layout["columns"]
        
        if len(columns) <= 1:
            logger.debug("Single column detected, using simple ordering")
            return self._fallback_sort(regions)
        
        sorted_regions = self._sort_by_columns(regions, columns)
        logger.debug("Sorted %d regions by %d columns", len(sorted_regions), len(columns))
        
        return sorted_regions
    
    def _detect_column_layout(self, pymupdf_page: Any, page_image: np.ndarray) -> dict[str, Any] | None:
        """Detect column layout using column_boxes utility function."""
        from pipeline.types import BBox
        
        try:
            detected_boxes = column_boxes(pymupdf_page)
        except Exception as exc:
            logger.debug("PyMuPDF column detection failed: %s", exc)
            return None
        
        if not detected_boxes:
            return None
        
        page_rect = pymupdf_page.rect
        if page_rect.width == 0 or page_rect.height == 0:
            return None
        
        image_height, image_width = page_image.shape[:2]
        scale_x = image_width / float(page_rect.width)
        scale_y = image_height / float(page_rect.height)
        
        column_regions = []
        for rect in detected_boxes:
            x0 = rect.x0 * scale_x
            y0 = rect.y0 * scale_y
            x1 = rect.x1 * scale_x
            y1 = rect.y1 * scale_y
            
            bbox = BBox.from_xyxy(x0, y0, x1, y1)
            column_regions.append({"bbox": bbox})
        
        columns = self._merge_column_boxes(column_regions, image_width)
        
        if len(columns) <= 1:
            return None
        
        return {"columns": columns, "scale": {"x": scale_x, "y": scale_y}}
    
    def _merge_column_boxes(self, column_boxes: list[dict[str, Any]], page_width: int) -> list[ColumnInfo]:
        """Merge column boxes with similar horizontal centers."""
        if not column_boxes:
            return []
        
        columns: list[dict[str, Any]] = []
        grouping_threshold = max(int(page_width * 0.05), 25)
        
        for box in column_boxes:
            bbox = box["bbox"]
            center_x = bbox.center[0]
            added = False
            
            for column in columns:
                col_center = column["center"]
                col_width = column["width"]
                threshold = max(grouping_threshold, col_width)
                
                if abs(center_x - col_center) <= threshold:
                    from pipeline.types import BBox
                    column["bbox"] = BBox(
                        x0=min(column["bbox"].x0, bbox.x0),
                        y0=min(column["bbox"].y0, bbox.y0),
                        x1=max(column["bbox"].x1, bbox.x1),
                        y1=max(column["bbox"].y1, bbox.y1),
                    )
                    column["centers"].append(center_x)
                    column["center"] = sum(column["centers"]) / len(column["centers"])
                    column["width"] = column["bbox"].width
                    added = True
                    break
            
            if not added:
                columns.append({
                    "bbox": bbox,
                    "centers": [center_x],
                    "center": center_x,
                    "width": bbox.width,
                })
        
        columns.sort(key=lambda col: col["bbox"].x0)
        
        column_infos: list[ColumnInfo] = []
        for idx, col in enumerate(columns):
            column_infos.append({
                "index": idx,
                "bbox": col["bbox"],
                "center": col["center"],
                "width": col["width"],
            })
        
        return column_infos
    
    def _sort_by_columns(self, regions: list[Any], columns: list[ColumnInfo]) -> list[Any]:
        """Sort regions by column-aware reading order."""
        from pipeline.types import ensure_bbox_in_region
        
        if not regions:
            return regions
        
        regions = [ensure_bbox_in_region(r) for r in regions]
        keyed_regions: list[tuple[tuple[float, float, float], Any, int]] = []
        
        for region in regions:
            bbox = region["bbox"]
            region_center_x, _ = bbox.center
            
            best_col_idx = 0
            best_overlap = 0.0
            best_distance = float("inf")
            
            for col in columns:
                col_bbox = col["bbox"]
                overlap = bbox.intersect(col_bbox)
                overlap_ratio = overlap / bbox.area if bbox.area > 0 else 0.0
                distance = abs(region_center_x - col["center"])
                
                if overlap_ratio > best_overlap or (
                    abs(overlap_ratio - best_overlap) <= COLUMN_ORDER_EPSILON
                    and distance < best_distance
                ):
                    best_overlap = overlap_ratio
                    best_distance = distance
                    best_col_idx = col["index"]
            
            if best_overlap <= 0:
                nearest_col = min(columns, key=lambda c: abs(region_center_x - c["center"]))
                best_col_idx = nearest_col["index"]
            
            sort_key = (float(best_col_idx), bbox.y0, bbox.x0)
            keyed_regions.append((sort_key, region, best_col_idx))
        
        keyed_regions.sort(key=lambda item: item[0])
        
        sorted_regions = []
        for rank, (_, region, col_idx) in enumerate(keyed_regions):
            region["reading_order_rank"] = rank
            region["column_index"] = col_idx
            sorted_regions.append(region)
        
        return sorted_regions
    
    def _fallback_sort(self, regions: list[Any]) -> list[Any]:
        """Fallback to simple geometric sorting."""
        from pipeline.types import ensure_bbox_in_region
        
        if not regions:
            return regions
        
        regions = [ensure_bbox_in_region(r) for r in regions]
        sorted_regions = sorted(regions, key=lambda r: (r["bbox"].y0, r["bbox"].x0))
        
        for rank, region in enumerate(sorted_regions):
            region["reading_order_rank"] = rank
        
        return sorted_regions
