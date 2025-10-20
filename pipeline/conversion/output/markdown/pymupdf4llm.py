"""PyMuPDF4LLM-style Markdown conversion using font sizes.

This module implements the PyMuPDF4LLM approach with proper separation:
- Regions: Layout detection results (from YOLO, etc.)
- Text Spans: PDF text objects with font info (from PyMuPDF parser)
- Matching happens at markdown conversion time (not at extraction time)

Uses PyMuPDF terminology:
- span: Text fragment with consistent formatting
- size: Font size in points (not font_size)
- font: Font name (not font_name)
- bbox: Bounding box [x0, y0, x1, y1]
"""

from __future__ import annotations

import logging
from typing import Any

from pipeline.types import BBox

logger = logging.getLogger(__name__)


class FontSizeHeaderIdentifier:
    """Identify header levels based on font sizes."""

    def __init__(
        self,
        font_sizes: list[float] | None = None,
        min_header_font_size: float = 12.0,
        max_header_levels: int = 3,
    ) -> None:
        """Initialize font size-based header identifier."""
        self.min_header_font_size = min_header_font_size
        self.max_header_levels = max_header_levels
        self.font_size_to_level: dict[float, int] = {}

        if font_sizes:
            self._build_font_size_mapping(font_sizes)

    def _build_font_size_mapping(self, font_sizes: list[float]) -> None:
        """Build mapping from font size to header level."""
        unique_sizes = sorted(set(font_sizes), reverse=True)
        header_sizes = [size for size in unique_sizes if size >= self.min_header_font_size]

        for i, font_size in enumerate(header_sizes[: self.max_header_levels]):
            self.font_size_to_level[font_size] = i + 1

        logger.info("Built font size mapping: %s", self.font_size_to_level)

    def set_font_size_mapping(self, mapping: dict[float, int]) -> None:
        """Manually set font size to header level mapping."""
        self.font_size_to_level = mapping

    def get_header_level(self, font_size: float | None) -> int | None:
        """Get header level for a font size."""
        if font_size is None:
            return None
        return self.font_size_to_level.get(font_size)

    def get_header_prefix(self, font_size: float | None) -> str:
        """Get Markdown header prefix for a font size."""
        level = self.get_header_level(font_size)
        if level is None:
            return ""
        return "#" * level + " "


def match_block_with_text_spans(
    block: dict[str, Any],
    text_spans: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any] | None:
    """Match a block with text spans using IoU.

    Args:
        block: Block dictionary with bbox
        text_spans: List of text spans (PyMuPDF spans with size, font)
        iou_threshold: Minimum IoU to consider a match (default: 0.3)

    Returns:
        Best matching text span or None
    """
    block_bbox_list = block.get("bbox", [])
    if len(block_bbox_list) < 4:  # noqa: PLR2004 - bbox always needs 4 coordinates
        return None

    # Block bbox is in xywh format, convert to xyxy
    block_bbox = BBox.from_xywh(*block_bbox_list[:4])

    best_match = None
    best_iou = 0.0

    for span in text_spans:
        span_bbox_list = span.get("bbox", [])
        if len(span_bbox_list) < 4:  # noqa: PLR2004 - bbox always needs 4 coordinates
            continue

        # Text span bbox is in xyxy format (PyMuPDF format)
        span_bbox = BBox(*span_bbox_list[:4])

        # Calculate IoU
        iou = _calculate_iou(block_bbox, span_bbox)

        if iou > best_iou:
            best_iou = iou
            best_match = span

    if best_match and best_iou >= iou_threshold:
        return best_match
    return None


def _calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x_left = max(bbox1.x0, bbox2.x0)
    y_top = max(bbox1.y0, bbox2.y0)
    x_right = min(bbox1.x1, bbox2.x1)
    y_bottom = min(bbox1.y1, bbox2.y1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = bbox1.area + bbox2.area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def block_to_markdown_with_font(  # noqa: PLR0911
    block: dict[str, Any],
    text_spans: list[dict[str, Any]],
    header_identifier: FontSizeHeaderIdentifier | None = None,
    iou_threshold: float = 0.3,
) -> str:
    """Convert a block to Markdown using font size from matched text spans.

    Args:
        block: Block dictionary
        text_spans: List of text spans with font info (PyMuPDF spans)
        header_identifier: Font size-based header identifier
        iou_threshold: IoU threshold for matching (default: 0.3)

    Returns:
        Markdown-formatted string
    """
    if header_identifier is None:
        header_identifier = FontSizeHeaderIdentifier()

    block_type = block.get("type", "").lower()
    text = block.get("corrected_text") or block.get("text", "")

    if not text:
        return ""

    # Match block with text span to get font size
    matched_span = match_block_with_text_spans(block, text_spans, iou_threshold)
    font_size = matched_span.get("size") if matched_span else None  # PyMuPDF uses 'size'

    # Try to determine header based on font size
    header_prefix = header_identifier.get_header_prefix(font_size)
    if header_prefix:
        return f"{header_prefix}{text}"

    # Fallback to block type-based formatting for special types
    if block_type in ["list", "list_item"]:
        if not text.startswith(("-", "*")):
            return f"- {text}"
        return text

    if block_type == "table":
        if "|" in text:
            return text
        return f"**Table:**\n\n{text}"

    if block_type in ["figure", "image"]:
        return f"**Figure:** {text}"

    if block_type == "equation":
        if text.startswith(("$$", "$")):
            return text
        return f"$${text}$$"

    return text


def regions_to_markdown_with_fonts(
    regions: list[dict[str, Any]],
    auxiliary_info: dict[str, Any] | None = None,
    header_identifier: FontSizeHeaderIdentifier | None = None,
    auto_detect_headers: bool = True,
    preserve_reading_order: bool = True,
    iou_threshold: float = 0.3,
) -> str:
    """Convert regions to Markdown using font size information from auxiliary info.

    This is the main function that implements PyMuPDF4LLM-style conversion
    with proper separation of concerns.

    Args:
        regions: List of region dictionaries (from layout detection)
        auxiliary_info: Auxiliary info dict containing 'text_spans'
        header_identifier: Font size-based header identifier
        auto_detect_headers: Auto-detect header levels from font sizes
        preserve_reading_order: Sort by order
        iou_threshold: IoU threshold for region-span matching

    Returns:
        Markdown-formatted string

    Example:
        >>> regions = [{"bbox": [100, 50, 300, 80], "text": "Chapter 1"}]
        >>> auxiliary_info = {"text_spans": [{"bbox": [100, 50, 300, 80], "size": 24.0}]}
        >>> md = regions_to_markdown_with_fonts(regions, auxiliary_info)
        >>> print(md)
        # Chapter 1
    """
    # Extract text spans from auxiliary info
    text_spans = []
    if auxiliary_info and "text_spans" in auxiliary_info:
        text_spans = auxiliary_info["text_spans"]

    # Auto-detect header levels from font sizes if requested
    if header_identifier is None and auto_detect_headers and text_spans:
        font_sizes = [span.get("size") for span in text_spans if span.get("size")]  # PyMuPDF 'size'
        if font_sizes:
            header_identifier = FontSizeHeaderIdentifier(font_sizes=font_sizes)
        else:
            logger.warning("No font size information found in text spans")
            header_identifier = FontSizeHeaderIdentifier()
    elif header_identifier is None:
        header_identifier = FontSizeHeaderIdentifier()

    # Sort by reading order if available
    sorted_blocks = regions
    if preserve_reading_order:
        ranked = [r for r in regions if r.get("order") is not None]
        unranked = [r for r in regions if r.get("order") is None]

        if ranked:
            sorted_blocks = sorted(ranked, key=lambda r: r["order"])
            sorted_blocks.extend(unranked)

    lines: list[str] = []
    prev_was_header = False

    for block in sorted_blocks:
        md_text = block_to_markdown_with_font(block, text_spans, header_identifier, iou_threshold)

        if not md_text:
            continue

        # Check if current is a header
        is_header = md_text.strip().startswith("#")

        # Add spacing before headers
        if is_header and prev_was_header:
            lines.append("")

        lines.append(md_text)
        prev_was_header = is_header

    return "\n\n".join(lines).strip()


def to_markdown(
    json_data: dict[str, Any] | list[dict[str, Any]],
    header_identifier: FontSizeHeaderIdentifier | None = None,
    auto_detect_headers: bool = True,
    iou_threshold: float = 0.3,
) -> str:
    """Convert pipeline JSON to Markdown using font size information.

    This is the main API function that implements PyMuPDF4LLM-style conversion.

    Args:
        json_data: Pipeline result dict (with auxiliary_info) or regions list
        header_identifier: Custom font size-based header identifier
        auto_detect_headers: Auto-detect header levels from font sizes
        iou_threshold: IoU threshold for region-span matching

    Returns:
        Markdown-formatted string

    Example:
        >>> # From page result with auxiliary_info
        >>> page_result = {
        ...     "processed_blocks": [...],
        ...     "auxiliary_info": {
        ...         "text_spans": [...]  # PyMuPDF spans with size, font
        ...     }
        ... }
        >>> md = to_markdown(page_result)
    """
    if isinstance(json_data, list):
        # Direct list of regions (no auxiliary info)
        return regions_to_markdown_with_fonts(
            json_data,
            auxiliary_info=None,
            header_identifier=header_identifier,
            auto_detect_headers=auto_detect_headers,
            iou_threshold=iou_threshold,
        )

    elif isinstance(json_data, dict):
        # Extract regions and auxiliary_info
        auxiliary_info = json_data.get("auxiliary_info")

        if "regions" in json_data:
            regions = json_data["regions"]
        elif "pages" in json_data:
            # Multi-page result - concatenate all regions and auxiliary info
            all_regions = []
            all_text_spans = []
            for page_data in json_data["pages"]:
                page_regions = page_data.get("blocks", [])
                all_regions.extend(page_regions)

                # Extract text spans from page's auxiliary_info
                page_aux = page_data.get("auxiliary_info", {})
                page_spans = page_aux.get("text_spans", [])
                all_text_spans.extend(page_spans)

            regions = all_regions
            auxiliary_info = {"text_spans": all_text_spans} if all_text_spans else None
        else:
            raise ValueError("Dict must contain 'pages', 'regions', or 'processed_blocks' key")

        return regions_to_markdown_with_fonts(
            regions,
            auxiliary_info=auxiliary_info,
            header_identifier=header_identifier,
            auto_detect_headers=auto_detect_headers,
            iou_threshold=iou_threshold,
        )

    else:
        raise TypeError(f"Unsupported JSON data type: {type(json_data)}")
