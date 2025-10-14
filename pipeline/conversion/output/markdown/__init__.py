"""Markdown output conversion utilities.

This module provides region type-based Markdown conversion,
the default conversion strategy for this pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RegionTypeHeaderIdentifier:
    """Identify header levels based on region types.

    This is the default header identifier that uses region type classification
    from layout detection to determine Markdown header levels.
    """

    # Region type to header level mapping
    HEADER_MAPPING: dict[str, int] = {
        "title": 1,  # #
        "heading": 1,  # #
        "section_header": 1,  # #
        "subtitle": 2,  # ##
        "subsection_header": 2,  # ##
        "section_title": 2,  # ##
        "subheading": 3,  # ###
    }

    def get_header_level(self, region_type: str) -> int | None:
        """Get header level for a region type.

        Args:
            region_type: Region type string

        Returns:
            Header level (1-6) or None if not a header

        Example:
            >>> identifier = RegionTypeHeaderIdentifier()
            >>> identifier.get_header_level("title")
            1
            >>> identifier.get_header_level("text")
            None
        """
        return self.HEADER_MAPPING.get(region_type.lower())

    def get_header_prefix(self, region_type: str) -> str:
        """Get Markdown header prefix for a region type.

        Args:
            region_type: Region type string

        Returns:
            Markdown header prefix (e.g., "# ", "## ") or empty string

        Example:
            >>> identifier = RegionTypeHeaderIdentifier()
            >>> identifier.get_header_prefix("title")
            '# '
            >>> identifier.get_header_prefix("text")
            ''
        """
        level = self.get_header_level(region_type)
        if level is None:
            return ""
        return "#" * level + " "


def region_to_markdown(  # noqa: PLR0911
    region: dict[str, Any],
    header_identifier: RegionTypeHeaderIdentifier | None = None,
) -> str:
    """Convert a single region to Markdown format using region type.

    Args:
        region: Region dictionary with 'type', 'text', etc.
        header_identifier: Header identifier (uses default if None)

    Returns:
        Markdown-formatted string for the region

    Example:
        >>> region = {"type": "title", "text": "Introduction"}
        >>> region_to_markdown(region)
        '# Introduction'
    """
    if header_identifier is None:
        header_identifier = RegionTypeHeaderIdentifier()

    region_type = region.get("type", "").lower()
    text = region.get("corrected_text") or region.get("text", "")

    if not text:
        return ""

    # Check if it's a header based on region type
    header_prefix = header_identifier.get_header_prefix(region_type)
    if header_prefix:
        return f"{header_prefix}{text}"

    # Handle special types
    if region_type in ["list", "list_item"]:
        # Ensure proper list formatting
        if not text.startswith(("-", "*")):
            return f"- {text}"
        return text

    if region_type == "table":
        # Table regions might already contain markdown table or need formatting
        if "|" in text:  # Already markdown table
            return text
        return f"**Table:**\n\n{text}"

    if region_type in ["figure", "image"]:
        # Image descriptions
        return f"**Figure:** {text}"

    if region_type == "equation":
        # Math equations
        if text.startswith(("$$", "$")):
            return text
        return f"$${text}$$"

    # Default: plain text
    return text


def regions_to_markdown(
    regions: list[dict[str, Any]],
    include_bbox: bool = False,
    include_confidence: bool = False,
    header_identifier: RegionTypeHeaderIdentifier | None = None,
    preserve_reading_order: bool = True,
) -> str:
    """Convert regions to Markdown format using region types.

    This function processes regions in reading order (if available) and
    applies appropriate Markdown formatting based on region types.

    Args:
        regions: List of region dictionaries with 'type', 'text', 'bbox', etc.
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)
        header_identifier: Custom header identifier (uses default if None)
        preserve_reading_order: Sort by reading_order_rank if True (default: True)

    Returns:
        Markdown-formatted string

    Example:
        >>> regions = [
        ...     {"type": "title", "text": "Chapter 1", "reading_order_rank": 0},
        ...     {"type": "text", "text": "Introduction.", "reading_order_rank": 1},
        ... ]
        >>> md = regions_to_markdown(regions)
        >>> print(md)
        # Chapter 1
        <BLANKLINE>
        Introduction.
    """
    if header_identifier is None:
        header_identifier = RegionTypeHeaderIdentifier()

    # Sort by reading order if available and requested
    sorted_regions = regions
    if preserve_reading_order:
        # Filter regions with reading_order_rank
        ranked_regions = [r for r in regions if r.get("reading_order_rank") is not None]
        unranked_regions = [r for r in regions if r.get("reading_order_rank") is None]

        if ranked_regions:
            sorted_regions = sorted(ranked_regions, key=lambda r: r["reading_order_rank"])
            # Append unranked regions at the end
            sorted_regions.extend(unranked_regions)

    lines: list[str] = []
    prev_type = None

    for region in sorted_regions:
        md_text = region_to_markdown(region, header_identifier)

        if not md_text:
            continue

        current_type = region.get("type", "").lower()

        # Add spacing between different types
        if prev_type is not None and current_type != prev_type:
            # Add extra blank line between different section types
            if current_type in ["title", "heading", "section_header", "subtitle"]:
                lines.append("")  # Extra spacing before headers

        lines.append(md_text)

        # Add metadata if requested
        metadata_parts: list[str] = []
        if include_bbox and "bbox" in region:
            bbox = region["bbox"]
            metadata_parts.append(f"bbox: {bbox}")
        if include_confidence and "confidence" in region:
            conf = region["confidence"]
            metadata_parts.append(f"confidence: {conf:.2f}")

        if metadata_parts:
            lines.append(f"*({', '.join(metadata_parts)})*")

        prev_type = current_type

    return "\n\n".join(lines).strip()


def save_regions_to_markdown(
    regions: list[dict[str, Any]],
    output_path: Path,
    include_bbox: bool = False,
    include_confidence: bool = False,
    add_header: bool = True,
) -> None:
    """Save regions to Markdown file.

    Args:
        regions: List of region dictionaries
        output_path: Output Markdown file path
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)
        add_header: Add document header with metadata (default: True)

    Raises:
        OSError: If file cannot be written

    Example:
        >>> regions = [
        ...     {"type": "title", "text": "Document Title"},
        ...     {"type": "text", "text": "Content here."},
        ... ]
        >>> save_regions_to_markdown(regions, Path("output.md"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Add header
    if add_header:
        lines.append("# OCR Result")
        lines.append("")
        lines.append(f"Total regions: {len(regions)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Convert regions to markdown
    md_content = regions_to_markdown(
        regions,
        include_bbox=include_bbox,
        include_confidence=include_confidence,
    )
    lines.append(md_content)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved %d regions to Markdown: %s", len(regions), output_path)


def pipeline_result_to_markdown(
    result: dict[str, Any],
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> str:
    """Convert pipeline result to Markdown format.

    Handles both single-page and multi-page results.

    Args:
        result: Pipeline result dictionary
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)

    Returns:
        Markdown-formatted string

    Example:
        >>> result = {
        ...     "metadata": {"source": "doc.pdf", "total_pages": 2},
        ...     "pages": [
        ...         {"page_num": 1, "regions": [{"type": "title", "text": "Page 1"}]},
        ...         {"page_num": 2, "regions": [{"type": "title", "text": "Page 2"}]},
        ...     ]
        ... }
        >>> md = pipeline_result_to_markdown(result)
    """
    lines: list[str] = []

    # Add metadata if present
    if "metadata" in result:
        metadata = result["metadata"]
        lines.append("# Document Information")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Handle different result formats
    if "pages" in result:
        # Multi-page format
        for page_data in result["pages"]:
            page_num = page_data.get("page_num", "?")
            regions = page_data.get("regions", [])

            lines.append(f"## Page {page_num}")
            lines.append("")

            md_content = regions_to_markdown(
                regions,
                include_bbox=include_bbox,
                include_confidence=include_confidence,
            )
            lines.append(md_content)
            lines.append("")
            lines.append("---")
            lines.append("")

    elif "regions" in result or "processed_regions" in result:
        # Single-page format
        regions = result.get("processed_regions") or result.get("regions", [])
        md_content = regions_to_markdown(
            regions,
            include_bbox=include_bbox,
            include_confidence=include_confidence,
        )
        lines.append(md_content)

    else:
        raise ValueError("Result must contain 'pages', 'regions', or 'processed_regions' key")

    return "\n".join(lines).strip()


def save_pipeline_result_to_markdown(
    result: dict[str, Any],
    output_path: Path,
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> None:
    """Save pipeline result to Markdown file.

    Args:
        result: Pipeline result dictionary
        output_path: Output Markdown file path
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)

    Raises:
        OSError: If file cannot be written

    Example:
        >>> result = {"pages": [...]}
        >>> save_pipeline_result_to_markdown(result, Path("output.md"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = pipeline_result_to_markdown(
        result,
        include_bbox=include_bbox,
        include_confidence=include_confidence,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved pipeline result to Markdown: %s", output_path)


def json_to_markdown(
    json_data: dict[str, Any] | list[dict[str, Any]],
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> str:
    """Convert JSON data to Markdown format.

    Automatically detects format (regions list, single result, or pipeline result).

    Args:
        json_data: JSON data (list of regions or result dict)
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)

    Returns:
        Markdown-formatted string

    Example:
        >>> json_data = [{"type": "title", "text": "Hello"}]
        >>> md = json_to_markdown(json_data)
        >>> print(md)
        # Hello
    """
    if isinstance(json_data, list):
        # Direct list of regions
        return regions_to_markdown(
            json_data,
            include_bbox=include_bbox,
            include_confidence=include_confidence,
        )
    elif isinstance(json_data, dict):
        # Pipeline result or wrapped format
        if "pages" in json_data or "regions" in json_data or "processed_regions" in json_data:
            return pipeline_result_to_markdown(
                json_data,
                include_bbox=include_bbox,
                include_confidence=include_confidence,
            )
        else:
            raise ValueError("Dict must contain 'pages', 'regions', or 'processed_regions' key")
    else:
        raise TypeError(f"Unsupported JSON data type: {type(json_data)}")
