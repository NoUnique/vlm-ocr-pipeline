"""Markdown output conversion utilities.

This module provides region type-based Markdown conversion,
the default conversion strategy for this pipeline.

Key principles:
1. Object-first: Primary functions work with Region/Page/Document objects
2. Dict wrappers: Convenience functions for dict inputs
3. Clear naming: Function names indicate whether they process objects or dicts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline.types import Block, Document, Page

logger = logging.getLogger(__name__)


class RegionTypeHeaderIdentifier:
    """Identify header levels based on region types.

    This is the default header identifier that uses region type classification
    from layout detection to determine Markdown header levels.

    Note: Only "title" is actively used by detectors. Other mappings are
    reserved for future use or custom detectors.
    """

    # Region type to header level mapping
    HEADER_MAPPING: dict[str, int] = {
        "title": 1,  # # - Used by all detectors
        # Reserved for future use:
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


# ==================== Core: Object → Markdown ====================


def block_to_markdown(  # noqa: PLR0911, PLR0912, PLR0915
    block: Block,
    header_identifier: RegionTypeHeaderIdentifier | None = None,
) -> str:
    """Convert a Block object to Markdown format using block type.

    Supports all MinerU 2.5 VLM block types (25+ types) plus legacy types.

    Args:
        block: Block object (not dict!)
        header_identifier: Header identifier (uses default if None)

    Returns:
        Markdown-formatted string for the block

    Example:
        >>> from pipeline.types import Block, BBox
        >>> block = Block(type="title", bbox=BBox(0, 0, 100, 20), detection_confidence=0.9, text="Introduction")
        >>> block_to_markdown(block)
        '# Introduction'
    """
    if header_identifier is None:
        header_identifier = RegionTypeHeaderIdentifier()

    block_type = block.type.lower()
    text = block.corrected_text or block.text or ""

    if not text:
        return ""

    # Check if it's a header based on block type
    header_prefix = header_identifier.get_header_prefix(block_type)
    if header_prefix:
        return f"{header_prefix}{text}"

    # ==================== Content Types ====================
    if block_type in ["text", "plain text"]:
        return text

    # ==================== List Types ====================
    if block_type in ["list", "list_item"]:
        # Ensure proper list formatting
        if not text.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            return f"- {text}"
        return text

    # ==================== Table Types ====================
    if block_type in ["table", "table_body"]:
        # Table blocks might already contain markdown table or need formatting
        if "|" in text:  # Already markdown table
            return text
        return f"**Table:**\n\n{text}"

    if block_type == "table_caption":
        return f"**Table:** {text}"

    if block_type == "table_footnote":
        return f"*{text}*"

    # ==================== Figure/Image Types ====================
    if block_type in ["figure", "image", "image_body"]:
        # Image descriptions
        return f"**Figure:** {text}"

    if block_type == "image_caption":
        return f"**Figure:** {text}"

    if block_type == "image_footnote":
        return f"*{text}*"

    # ==================== Equation Types ====================
    if block_type in ["equation", "interline_equation"]:
        # Display equations (on their own line)
        if text.startswith(("$$", "$")):
            return text
        return f"$${text}$$"

    if block_type == "inline_equation":
        # Inline equations
        if text.startswith("$") and text.endswith("$"):
            return text
        return f"${text}$"

    # Legacy MinerU DocLayoutYOLO types
    if block_type == "isolate_formula":
        if text.startswith(("$$", "$")):
            return text
        return f"$${text}$$"

    if block_type == "formula_caption":
        return f"*Formula: {text}*"

    if block_type == "figure_caption":
        return f"**Figure:** {text}"

    # ==================== Code Types ====================
    if block_type in ["code", "code_body", "algorithm"]:
        # Code blocks
        if text.startswith("```") and text.endswith("```"):
            return text
        # Try to detect language from first line
        lines = text.split("\n", 1)
        if len(lines) == 1:
            return f"```\n{text}\n```"
        return f"```\n{text}\n```"

    if block_type == "code_caption":
        return f"**Code:** {text}"

    # ==================== Page Elements (Skip) ====================
    if block_type in ["header", "footer", "page_number"]:
        # Skip page headers/footers/numbers - they're not content
        return ""

    # ==================== Reference/Metadata Types ====================
    if block_type == "ref_text":
        # References - render as normal text
        return text

    if block_type in ["phonetic", "aside_text"]:
        # Side notes - render in italics
        return f"*{text}*"

    if block_type == "page_footnote":
        # Footnotes - render in italics
        return f"*{text}*"

    if block_type == "index":
        # Index entries - render as normal text or skip
        return text

    # ==================== Discarded Types ====================
    if block_type in ["discarded", "abandon"]:
        # Skip discarded content
        return ""

    # ==================== Default: Plain Text ====================
    return text


def blocks_to_markdown(
    blocks: list[Block],
    include_bbox: bool = False,
    include_confidence: bool = False,
    header_identifier: RegionTypeHeaderIdentifier | None = None,
    preserve_reading_order: bool = True,
) -> str:
    """Convert list of Block objects to Markdown format using block types.

    This function processes blocks in reading order (if available) and
    applies appropriate Markdown formatting based on block types.

    Args:
        blocks: List of Block objects (not dicts!)
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)
        header_identifier: Custom header identifier (uses default if None)
        preserve_reading_order: Sort by order if True (default: True)

    Returns:
        Markdown-formatted string

    Example:
        >>> from pipeline.types import Block, BBox
        >>> blocks = [
        ...     Block(type="title", bbox=BBox(0, 0, 100, 20), detection_confidence=0.9,
        ...            text="Chapter 1", order=0),
        ...     Block(type="text", bbox=BBox(0, 30, 100, 50), detection_confidence=0.95,
        ...            text="Introduction.", order=1),
        ... ]
        >>> md = blocks_to_markdown(blocks)
        >>> print(md)
        # Chapter 1
        <BLANKLINE>
        Introduction.
    """
    if header_identifier is None:
        header_identifier = RegionTypeHeaderIdentifier()

    # Sort by reading order if available and requested
    sorted_blocks = blocks
    if preserve_reading_order:
        # Filter blocks with order
        ranked_blocks = [b for b in blocks if b.order is not None]
        unranked_blocks = [b for b in blocks if b.order is None]

        if ranked_blocks:
            sorted_blocks = sorted(ranked_blocks, key=lambda b: b.order)  # type: ignore
            # Append unranked blocks at the end
            sorted_blocks.extend(unranked_blocks)

    lines: list[str] = []
    prev_type = None

    for block in sorted_blocks:
        md_text = block_to_markdown(block, header_identifier)

        if not md_text:
            continue

        current_type = block.type.lower()

        # Add spacing between different types
        if prev_type is not None and current_type != prev_type:
            # Add extra blank line between different section types
            if current_type in ["title", "heading", "section_header", "subtitle"]:
                lines.append("")  # Extra spacing before headers

        lines.append(md_text)

        # Add metadata if requested
        metadata_parts: list[str] = []
        if include_bbox:
            bbox = block.bbox.to_xywh_list()
            metadata_parts.append(f"bbox: {bbox}")
        if include_confidence and block.detection_confidence is not None:
            conf = block.detection_confidence
            metadata_parts.append(f"confidence: {conf:.2f}")

        if metadata_parts:
            lines.append(f"*({', '.join(metadata_parts)})*")

        prev_type = current_type

    return "\n\n".join(lines).strip()


def page_to_markdown(
    page: Page,
    include_page_header: bool = True,
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> str:
    """Convert a Page object to Markdown format.

    Args:
        page: Page object (not dict!)
        include_page_header: Include "## Page N" header
        include_bbox: Include bbox metadata
        include_confidence: Include confidence metadata

    Returns:
        Markdown-formatted string

    Example:
        >>> from pipeline.types import Page, Block, BBox
        >>> page = Page(
        ...     page_num=1,
        ...     regions=[Region(type="title", bbox=BBox(0, 0, 100, 20), confidence=0.9, text="Title")]
        ... )
        >>> md = page_to_markdown(page)
        >>> print(md)
        ## Page 1
        <BLANKLINE>
        # Title
    """
    lines: list[str] = []

    if include_page_header:
        lines.append(f"## Page {page.page_num}")
        lines.append("")

    md_content = blocks_to_markdown(
        page.blocks,
        include_bbox=include_bbox,
        include_confidence=include_confidence,
    )
    lines.append(md_content)

    return "\n".join(lines).strip()


def document_to_markdown(
    doc: Document,
    include_metadata: bool = True,
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> str:
    """Convert a Document object to Markdown format.

    Args:
        doc: Document object (not dict!)
        include_metadata: Include document metadata section
        include_bbox: Include bbox metadata
        include_confidence: Include confidence metadata

    Returns:
        Markdown-formatted string

    Example:
        >>> from pipeline.types import Document, Page, Block, BBox
        >>> doc = Document(
        ...     pdf_name="test",
        ...     pdf_path="/test.pdf",
        ...     num_pages=1,
        ...     processed_pages=1,
        ...     pages=[Page(page_num=1, regions=[])]
        ... )
        >>> md = document_to_markdown(doc)
    """
    lines: list[str] = []

    # Metadata section
    if include_metadata:
        lines.append("# Document Information")
        lines.append("")
        lines.append(f"- **Source:** {doc.pdf_name}")
        lines.append(f"- **Total pages:** {doc.num_pages}")
        lines.append(f"- **Processed pages:** {doc.processed_pages}")
        if doc.processed_at:
            lines.append(f"- **Processed at:** {doc.processed_at}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Process each page
    for page in doc.pages:
        page_md = page_to_markdown(
            page,
            include_page_header=True,
            include_bbox=include_bbox,
            include_confidence=include_confidence,
        )
        lines.append(page_md)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip()


# ==================== Wrapper: Dict → Object → Markdown ====================


def region_dict_to_markdown(data: dict[str, Any], **kwargs: Any) -> str:
    """Convert region dict to Markdown (convenience wrapper).

    Args:
        data: Region dictionary
        **kwargs: Additional arguments for region_to_markdown

    Returns:
        Markdown-formatted string

    Example:
        >>> data = {"type": "title", "xywh": [0, 0, 100, 20], "confidence": 0.9, "text": "Hello"}
        >>> region_dict_to_markdown(data)
        '# Hello'
    """
    block = Block.from_dict(data)
    return block_to_markdown(block, **kwargs)


def regions_dict_to_markdown(data: list[dict[str, Any]], **kwargs: Any) -> str:
    """Convert list of region dicts to Markdown (convenience wrapper).

    Args:
        data: List of region dictionaries
        **kwargs: Additional arguments for regions_to_markdown

    Returns:
        Markdown-formatted string

    Example:
        >>> data = [{"type": "title", "xywh": [0, 0, 100, 20], "confidence": 0.9, "text": "Hello"}]
        >>> regions_dict_to_markdown(data)
        '# Hello'
    """
    blocks = [Block.from_dict(b) for b in data]
    return blocks_to_markdown(blocks, **kwargs)


def page_dict_to_markdown(data: dict[str, Any], **kwargs: Any) -> str:
    """Convert page dict to Markdown (convenience wrapper).

    Args:
        data: Page dictionary
        **kwargs: Additional arguments for page_to_markdown

    Returns:
        Markdown-formatted string

    Example:
        >>> data = {
        ...     "page_num": 1,
        ...     "blocks": [{"type": "title", "xywh": [0, 0, 100, 20], "confidence": 0.9, "text": "Hello"}]
        ... }
        >>> page_dict_to_markdown(data)
        '## Page 1\\n\\n# Hello'
    """
    page = Page.from_dict(data)
    return page_to_markdown(page, **kwargs)


def document_dict_to_markdown(data: dict[str, Any], **kwargs: Any) -> str:
    """Convert document dict to Markdown (convenience wrapper).

    Args:
        data: Document dictionary
        **kwargs: Additional arguments for document_to_markdown

    Returns:
        Markdown-formatted string

    Example:
        >>> data = {
        ...     "pdf_name": "test",
        ...     "pdf_path": "/test.pdf",
        ...     "num_pages": 1,
        ...     "processed_pages": 1,
        ...     "pages": [{"page_num": 1, "blocks": []}]
        ... }
        >>> md = document_dict_to_markdown(data)
    """
    doc = Document.from_dict(data)
    return document_to_markdown(doc, **kwargs)




# ==================== File I/O ====================


def save_blocks_to_markdown(
    blocks: list[Block],
    output_path: Path,
    include_bbox: bool = False,
    include_confidence: bool = False,
    add_header: bool = True,
) -> None:
    """Save blocks to Markdown file.

    Args:
        blocks: List of Block objects
        output_path: Output Markdown file path
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)
        add_header: Add document header with metadata (default: True)

    Raises:
        OSError: If file cannot be written

    Example:
        >>> from pipeline.types import Block, BBox
        >>> blocks = [
        ...     Block(type="title", bbox=BBox(0, 0, 100, 20), detection_confidence=0.9, text="Document Title"),
        ...     Block(type="text", bbox=BBox(0, 30, 100, 50), detection_confidence=0.95, text="Content here."),
        ... ]
        >>> save_blocks_to_markdown(blocks, Path("output.md"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Add header
    if add_header:
        lines.append("# OCR Result")
        lines.append("")
        lines.append(f"Total blocks: {len(blocks)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Convert blocks to markdown
    md_content = blocks_to_markdown(
        blocks,
        include_bbox=include_bbox,
        include_confidence=include_confidence,
    )
    lines.append(md_content)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved %d blocks to Markdown: %s", len(blocks), output_path)


# Backward compatibility alias
save_regions_to_markdown = save_blocks_to_markdown


def save_page_to_markdown(
    page: Page,
    output_path: Path,
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> None:
    """Save page to Markdown file.

    Args:
        page: Page object
        output_path: Output Markdown file path
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)

    Raises:
        OSError: If file cannot be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = page_to_markdown(
        page,
        include_bbox=include_bbox,
        include_confidence=include_confidence,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved page %d to Markdown: %s", page.page_num, output_path)


def save_document_to_markdown(
    doc: Document,
    output_path: Path,
    include_bbox: bool = False,
    include_confidence: bool = False,
) -> None:
    """Save document to Markdown file.

    Args:
        doc: Document object
        output_path: Output Markdown file path
        include_bbox: Include bounding box information (default: False)
        include_confidence: Include confidence scores (default: False)

    Raises:
        OSError: If file cannot be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = document_to_markdown(
        doc,
        include_bbox=include_bbox,
        include_confidence=include_confidence,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved document '%s' to Markdown: %s", doc.pdf_name, output_path)


# ==================== Backward Compatibility Aliases ====================
# For backward compatibility with code that uses the old naming

region_to_markdown = block_to_markdown
regions_to_markdown = blocks_to_markdown
