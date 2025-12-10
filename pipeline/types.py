"""Unified type definitions for the VLM OCR pipeline.

This module provides:
- BBox: Integer-based bounding box (internal: xyxy, JSON: xywh)
- Block: Document block with required bbox field
- Page: Single page processing result
- Document: Multi-page document processing result
- Detector/Sorter: Protocol interfaces
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

# ==================== Constants ====================

RGB_IMAGE_NDIM = 3  # RGB image has 3 dimensions (H, W, C)


# ==================== Auxiliary Type Definitions ====================


class TextSpan(TypedDict):
    """PyMuPDF text span with font information.

    Extracted from PDF using PyMuPDF's get_text("dict") method.
    Uses PyMuPDF terminology: 'size' for font size, 'font' for font name.
    """

    bbox: list[int]  # [x0, y0, x1, y1] in xyxy format
    text: str  # Text content
    size: float  # Font size (PyMuPDF terminology)
    font: str  # Font name (PyMuPDF terminology)


class AuxiliaryInfo(TypedDict, total=False):
    """Auxiliary information extracted from PDF.

    This contains metadata used for enhanced markdown conversion,
    such as font information for auto-detecting headers.
    """

    text_spans: list[TextSpan]  # Text spans with font metadata


class ColumnInfo(TypedDict):
    """Column information for multi-column documents."""

    index: int  # Column index (0-based)
    x0: int  # Left boundary
    x1: int  # Right boundary
    center: float  # Center X coordinate
    width: int  # Column width


class ColumnLayout(TypedDict):
    """Column layout information for a page."""

    columns: list[ColumnInfo]  # List of columns on the page


# ==================== External Library Protocols ====================


class PyMuPDFRect(Protocol):
    """Protocol for PyMuPDF Rect object.

    This defines the minimal interface we need from fitz.Rect
    without requiring PyMuPDF as a dependency for type checking.
    """

    x0: float  # Left boundary
    y0: float  # Top boundary
    x1: float  # Right boundary
    y1: float  # Bottom boundary


class PyMuPDFPage(Protocol):
    """Protocol for PyMuPDF Page object.

    This defines the minimal interface we need from fitz.Page
    without requiring PyMuPDF as a dependency for type checking.
    """

    rect: PyMuPDFRect  # Page rectangle


# ==================== Standardized Block Types ====================
# Based on MinerU 2.5 VLM (most comprehensive type system)


class BlockType:
    """Standardized block types based on MinerU 2.5 VLM.

    This provides a canonical set of block types that all detectors
    should map to for consistent processing throughout the pipeline.
    """

    # Content
    TEXT = "text"
    TITLE = "title"

    # Figures
    IMAGE = "image"
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"
    IMAGE_FOOTNOTE = "image_footnote"

    # Tables
    TABLE = "table"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"

    # Equations
    EQUATION = "equation"  # Alias for interline_equation
    INTERLINE_EQUATION = "interline_equation"
    INLINE_EQUATION = "inline_equation"

    # Code
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"

    # Lists
    LIST = "list"

    # Page Elements
    HEADER = "header"  # Page header (not heading)
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    PAGE_FOOTNOTE = "page_footnote"

    # References
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    ASIDE_TEXT = "aside_text"
    INDEX = "index"

    # Special
    DISCARDED = "discarded"
    ABANDON = "abandon"  # MinerU DocLayoutYOLO legacy

    # Aliases for backward compatibility
    PLAIN_TEXT = "plain text"  # Legacy type
    FIGURE = "figure"  # Alias for image
    ISOLATE_FORMULA = "isolate_formula"  # MinerU DocLayoutYOLO legacy
    FORMULA_CAPTION = "formula_caption"  # MinerU DocLayoutYOLO legacy
    FIGURE_CAPTION = "figure_caption"  # MinerU DocLayoutYOLO legacy
    LIST_ITEM = "list_item"  # Project-specific type


# Type mapping dictionaries for each detector
class BlockTypeMapper:
    """Maps detector-specific block types to standardized types."""

    # DocLayout-YOLO (this project) - model-dependent, these are common mappings
    DOCLAYOUT_YOLO_MAP: dict[str, str] = {
        "title": BlockType.TITLE,
        "plain text": BlockType.TEXT,
        "text": BlockType.TEXT,
        "figure": BlockType.IMAGE,
        "image": BlockType.IMAGE,
        "table": BlockType.TABLE,
        "equation": BlockType.INTERLINE_EQUATION,
        "list": BlockType.LIST,
        "list_item": BlockType.LIST,
    }

    # MinerU DocLayout-YOLO (fixed mapping)
    MINERU_DOCLAYOUT_YOLO_MAP: dict[str, str] = {
        "title": BlockType.TITLE,
        "plain text": BlockType.TEXT,
        "abandon": BlockType.DISCARDED,
        "figure": BlockType.IMAGE,
        "figure_caption": BlockType.IMAGE_CAPTION,
        "table": BlockType.TABLE,
        "table_caption": BlockType.TABLE_CAPTION,
        "table_footnote": BlockType.TABLE_FOOTNOTE,
        "isolate_formula": BlockType.INTERLINE_EQUATION,
        "formula_caption": BlockType.IMAGE_CAPTION,  # Treat as caption
    }

    # MinerU VLM 2.5 (already standardized, identity mapping)
    MINERU_VLM_MAP: dict[str, str] = {
        "text": BlockType.TEXT,
        "title": BlockType.TITLE,
        "image": BlockType.IMAGE,
        "image_body": BlockType.IMAGE_BODY,
        "image_caption": BlockType.IMAGE_CAPTION,
        "image_footnote": BlockType.IMAGE_FOOTNOTE,
        "table": BlockType.TABLE,
        "table_body": BlockType.TABLE_BODY,
        "table_caption": BlockType.TABLE_CAPTION,
        "table_footnote": BlockType.TABLE_FOOTNOTE,
        "interline_equation": BlockType.INTERLINE_EQUATION,
        "inline_equation": BlockType.INLINE_EQUATION,
        "code": BlockType.CODE,
        "code_body": BlockType.CODE_BODY,
        "code_caption": BlockType.CODE_CAPTION,
        "algorithm": BlockType.ALGORITHM,
        "list": BlockType.LIST,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "page_number": BlockType.PAGE_NUMBER,
        "page_footnote": BlockType.PAGE_FOOTNOTE,
        "ref_text": BlockType.REF_TEXT,
        "phonetic": BlockType.PHONETIC,
        "aside_text": BlockType.ASIDE_TEXT,
        "index": BlockType.INDEX,
        "discarded": BlockType.DISCARDED,
    }

    # olmOCR VLM (always returns "text")
    OLMOCR_VLM_MAP: dict[str, str] = {
        "text": BlockType.TEXT,
    }

    # PaddleOCR PP-DocLayoutV2 (PP-DocLayout-L/M/S with 23 categories)
    PADDLEOCR_DOCLAYOUT_V2_MAP: dict[str, str] = {
        # Titles and text
        "doc_title": BlockType.TITLE,
        "paragraph_title": BlockType.TITLE,
        "text": BlockType.TEXT,
        "sidebar_text": BlockType.ASIDE_TEXT,
        # Page elements
        "page_number": BlockType.PAGE_NUMBER,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "header_image": BlockType.HEADER,
        "footer_image": BlockType.FOOTER,
        # Structural elements
        "abstract": BlockType.TEXT,
        "contents": BlockType.TEXT,
        "reference": BlockType.REF_TEXT,
        "reference_content": BlockType.REF_TEXT,
        "footnote": BlockType.PAGE_FOOTNOTE,
        # Math and code
        "formula": BlockType.INTERLINE_EQUATION,
        "formula_number": BlockType.INTERLINE_EQUATION,
        "algorithm": BlockType.ALGORITHM,
        # Figures and images
        "image": BlockType.IMAGE,
        # Tables
        "table": BlockType.TABLE,
        "table_title": BlockType.TABLE_CAPTION,
        # Charts
        "chart": BlockType.IMAGE,
        "chart_title": BlockType.IMAGE_CAPTION,
        # Special
        "seal": BlockType.IMAGE,
    }

    @classmethod
    def map_type(cls, region_type: str, detector_name: str) -> str:
        """Map a detector-specific type to standardized type.

        Args:
            region_type: Original block type from detector
            detector_name: Name of detector ("doclayout-yolo", "mineru-doclayout-yolo",
                          "mineru-vlm", "olmocr-vlm", "paddleocr-doclayout-v2")

        Returns:
            Standardized block type (falls back to original if no mapping found)

        Example:
            >>> BlockTypeMapper.map_type("plain text", "doclayout-yolo")
            'text'
            >>> BlockTypeMapper.map_type("abandon", "mineru-doclayout-yolo")
            'discarded'
            >>> BlockTypeMapper.map_type("doc_title", "paddleocr-doclayout-v2")
            'title'
        """
        mapping_dict: dict[str, str] | None = None

        if detector_name == "doclayout-yolo":
            mapping_dict = cls.DOCLAYOUT_YOLO_MAP
        elif detector_name == "mineru-doclayout-yolo":
            mapping_dict = cls.MINERU_DOCLAYOUT_YOLO_MAP
        elif detector_name == "mineru-vlm":
            mapping_dict = cls.MINERU_VLM_MAP
        elif detector_name == "olmocr-vlm":
            mapping_dict = cls.OLMOCR_VLM_MAP
        elif detector_name == "paddleocr-doclayout-v2":
            mapping_dict = cls.PADDLEOCR_DOCLAYOUT_V2_MAP

        if mapping_dict:
            return mapping_dict.get(region_type.lower(), region_type)

        # No mapping found - return original
        return region_type


# ==================== BBox Format Definitions ====================

"""
BBox formats used by different frameworks:

1. Internal representation (this project):
   BBox(x0, y0, x1, y1) - integers, xyxy format
   Example: BBox(100, 50, 300, 200) - top-left to bottom-right corners

2. JSON output (this project):
   [x, y, width, height] - xywh format for human readability
   Example: [100, 50, 200, 150]

3. YOLO internal:
   [x1, y1, x2, y2] - xyxy format
   Example: [100, 50, 300, 200]

4. PyMuPDF Rect:
   Rect(x0, y0, x1, y1)
   Example: Rect(100, 50, 300, 200)

5. MinerU blocks:
   bbox: [x0, y0, x1, y1]
   Example: [100, 50, 300, 200]

6. PyPDF (bottom-left origin!):
   [x0, y0, x1, y1] - Y-axis is inverted
   Example: [100, 642, 300, 742] when page height is 792

7. olmOCR anchor text:
   - Text: "[x, y]text"
   - Image: "[Image x0, y0 to x1, y1]"
   Example: "[100x50]Chapter 1" or "[Image 100x50 to 300x200]"
"""


@dataclass(frozen=True)
class BBox:
    """Pixel-based bounding box with integer coordinates.

    Internal format: (x0, y0, x1, y1) - Top-left and bottom-right corners (xyxy)
    Origin: Top-left corner of image (0, 0) - standard computer vision convention
    Coordinates: Integer pixel values (not normalized)

    This class provides conversion methods to/from all major bbox formats
    used in document processing frameworks.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    # ==================== FROM Conversions (Format → BBox) ====================

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create from (x, y, width, height) format.

        Format: [x, y, width, height]
        Used by: JSON output, detection models

        Args:
            x: Left x coordinate (will be rounded to int)
            y: Top y coordinate (will be rounded to int)
            w: Width (will be rounded to int)
            h: Height (will be rounded to int)

        Returns:
            BBox object with integer coordinates

        Example:
            >>> bbox = BBox.from_xywh(100.5, 50.2, 200.1, 150.8)
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 301, 201)
        """
        return cls(
            x0=round(x),
            y0=round(y),
            x1=round(x + w),
            y1=round(y + h),
        )

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float) -> BBox:
        """Create from (x0, y0, x1, y1) format.

        Format: [x0, y0, x1, y1] - corners
        Used by: YOLO internal, MinerU, PyMuPDF

        Args:
            x0: Left x coordinate (will be rounded to int)
            y0: Top y coordinate (will be rounded to int)
            x1: Right x coordinate (will be rounded to int)
            y1: Bottom y coordinate (will be rounded to int)

        Returns:
            BBox object with integer coordinates

        Example:
            >>> bbox = BBox.from_xyxy(100.5, 50.2, 300.8, 200.1)
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 301, 200)
        """
        return cls(
            x0=round(x0),
            y0=round(y0),
            x1=round(x1),
            y1=round(y1),
        )

    @classmethod
    def from_list(cls, coords: Sequence[float], coord_format: str = "xywh") -> BBox:
        """Create from coordinate list.

        Args:
            coords: Coordinate list (at least 4 elements)
            coord_format: "xywh" or "xyxy"

        Returns:
            Unified BBox object

        Raises:
            ValueError: If coord_format is unknown

        Example:
            >>> bbox = BBox.from_list([100, 50, 200, 150], coord_format="xywh")
            >>> bbox = BBox.from_list([100, 50, 300, 200], coord_format="xyxy")
        """
        if coord_format == "xywh":
            return cls.from_xywh(*coords[:4])
        elif coord_format == "xyxy":
            return cls.from_xyxy(*coords[:4])
        else:
            raise ValueError(f"Unknown bbox coord_format: {coord_format}. Use 'xywh' or 'xyxy'.")

    @classmethod
    def from_pymupdf_rect(cls, rect: PyMuPDFRect) -> BBox:
        """Create from PyMuPDF Rect or IRect object.

        Format: Rect(x0, y0, x1, y1)
        Used by: PyMuPDF (fitz)

        Args:
            rect: PyMuPDF Rect or IRect object

        Returns:
            BBox object with integer coordinates

        Example:
            >>> import fitz
            >>> rect = fitz.Rect(100.5, 50.2, 300.8, 200.1)
            >>> bbox = BBox.from_pymupdf_rect(rect)
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 301, 200)
        """
        return cls(
            x0=round(rect.x0),
            y0=round(rect.y0),
            x1=round(rect.x1),
            y1=round(rect.y1),
        )

    @classmethod
    def from_mineru_bbox(cls, bbox: Sequence[float]) -> BBox:
        """Create from MinerU bbox format.

        Format: [x0, y0, x1, y1]
        Used by: MinerU blocks

        Args:
            bbox: MinerU bbox list

        Returns:
            BBox object with integer coordinates

        Example:
            >>> bbox = BBox.from_mineru_bbox([100.5, 50.2, 300.8, 200.1])
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 301, 200)
        """
        return cls.from_xyxy(*bbox[:4])

    @classmethod
    def from_pypdf_rect(cls, rect: list[float], page_height: float) -> BBox:
        """Create from PyPDF rect (requires Y-axis flip).

        Format: [x0, y0, x1, y1] with bottom-left origin
        Used by: PyPDF
        Note: PyPDF uses bottom-left origin, need to flip Y-axis

        Args:
            rect: PyPDF rectangle [x0, y0, x1, y1]
            page_height: Page height for Y-axis conversion

        Returns:
            BBox object with top-left origin and integer coordinates

        Example:
            >>> # Page height: 792, PyPDF rect at bottom
            >>> bbox = BBox.from_pypdf_rect([100, 642, 300, 742], page_height=792)
            >>> bbox.y0, bbox.y1  # Flipped to top-left origin
            (50, 150)
        """
        x0, y0_bottom, x1, y1_bottom = rect[:4]
        # Y-axis flip: bottom-left → top-left
        y0_top = page_height - y1_bottom
        y1_top = page_height - y0_bottom
        return cls(
            x0=round(x0),
            y0=round(y0_top),
            x1=round(x1),
            y1=round(y1_top),
        )

    @classmethod
    def from_olmocr_anchor_coords(cls, x: float, y: float) -> BBox:
        """Create from olmOCR anchor text coordinates (point format).

        Format: "[x, y]" - single point
        Used by: olmOCR anchor text for text elements

        Args:
            x: X coordinate (will be rounded to int)
            y: Y coordinate (will be rounded to int)

        Returns:
            BBox representing a point (x0=x1, y0=y1)

        Example:
            >>> bbox = BBox.from_olmocr_anchor_coords(100.5, 50.2)
            >>> bbox.x0, bbox.y0
            (100, 50)
        """
        x_int = round(x)
        y_int = round(y)
        return cls(x0=x_int, y0=y_int, x1=x_int, y1=y_int)

    @classmethod
    def from_olmocr_anchor_box(cls, x0: float, y0: float, x1: float, y1: float) -> BBox:
        """Create from olmOCR anchor image/table bbox.

        Format: "[Image x0, y0 to x1, y1]"
        Used by: olmOCR anchor text for images/tables

        Args:
            x0, y0: Top-left coordinates (will be rounded to int)
            x1, y1: Bottom-right coordinates (will be rounded to int)

        Returns:
            BBox object with integer coordinates

        Example:
            >>> bbox = BBox.from_olmocr_anchor_box(100.5, 50.2, 300.8, 200.1)
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 301, 200)
        """
        return cls.from_xyxy(x0, y0, x1, y1)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Create from center coordinates format.

        Format: [center_x, center_y, width, height]
        Used by: YOLO training, Anchor boxes

        Args:
            cx: Center x coordinate (will be rounded to int)
            cy: Center y coordinate (will be rounded to int)
            w: Width (will be rounded to int)
            h: Height (will be rounded to int)

        Returns:
            BBox object with integer coordinates

        Example:
            >>> bbox = BBox.from_cxcywh(200, 125, 200, 150)
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 300, 200)
        """
        x0 = cx - w / 2
        y0 = cy - h / 2
        x1 = cx + w / 2
        y1 = cy + h / 2
        return cls(
            x0=round(x0),
            y0=round(y0),
            x1=round(x1),
            y1=round(y1),
        )

    # ==================== TO Conversions (BBox → Format) ====================

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Convert to (x, y, width, height) format.

        Format: (x, y, width, height)
        Used by: JSON serialization, display

        Returns:
            Tuple of (x, y, width, height) as integers

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_xywh()
            (100, 50, 200, 150)
        """
        return (self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (x0, y0, x1, y1) format.

        Format: (x0, y0, x1, y1) - corners
        Used by: Internal operations

        Returns:
            Tuple of (x0, y0, x1, y1) as integers

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_xyxy()
            (100, 50, 300, 200)
        """
        return (self.x0, self.y0, self.x1, self.y1)

    def to_cxcywh(self) -> tuple[float, float, int, int]:
        """Convert to center format.

        Format: (center_x, center_y, width, height)
        Used by: YOLO training, Anchor boxes

        Returns:
            Tuple of (cx, cy, width, height) - center is float, size is int

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_cxcywh()
            (200.0, 125.0, 200, 150)
        """
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        w = self.x1 - self.x0
        h = self.y1 - self.y0
        return (cx, cy, w, h)

    def to_list(self) -> list[int]:
        """Convert to list in internal format (xyxy).

        Returns:
            List [x0, y0, x1, y1]

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_list()
            [100, 50, 300, 200]
        """
        return [self.x0, self.y0, self.x1, self.y1]

    def to_dict(self) -> dict[str, int]:
        """Convert to dict with explicit keys.

        Returns:
            Dict with keys x0, y0, x1, y1

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_dict()
            {'x0': 100, 'y0': 50, 'x1': 300, 'y1': 200}
        """
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }

    def to_xywh_list(self) -> list[int]:
        """Convert to [x, y, w, h] list (for JSON serialization).

        Returns:
            List [x, y, width, height]

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_xywh_list()
            [100, 50, 200, 150]
        """
        return [self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0]

    def to_mineru_bbox(self) -> list[int]:
        """Convert to MinerU bbox format.

        Format: [x0, y0, x1, y1]
        Used by: MinerU blocks

        Returns:
            List [x0, y0, x1, y1]

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_mineru_bbox()
            [100, 50, 300, 200]
        """
        return self.to_list()

    def to_pypdf_rect(self, page_height: float) -> list[int]:
        """Convert to PyPDF rect format (requires Y-axis flip).

        Format: [x0, y0, x1, y1] with bottom-left origin
        Used by: PyPDF

        Args:
            page_height: Page height for Y-axis conversion

        Returns:
            List [x0, y0, x1, y1] with bottom-left origin

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_pypdf_rect(page_height=792)
            [100, 592, 300, 742]
        """
        # Y-axis flip: top-left → bottom-left
        y0_bottom = round(page_height - self.y1)
        y1_bottom = round(page_height - self.y0)
        return [self.x0, y0_bottom, self.x1, y1_bottom]

    def to_olmocr_anchor(self, content_type: str = "text", text_content: str = "") -> str:
        """Convert to olmOCR anchor text format.

        Format varies by content type:
        - text: "[x, y]content"
        - image: "[Image x0, y0 to x1, y1]"
        - table: "[Table x0, y0 to x1, y1]"

        Used by: olmOCR anchor text generation

        Args:
            content_type: Type of content ("text", "image", "figure", "table")
            text_content: Text content for text blocks (optional)

        Returns:
            olmOCR anchor text string

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_olmocr_anchor("text", "Chapter 1")
            '[100x50]Chapter 1'
            >>> bbox.to_olmocr_anchor("image")
            '[Image 100x50 to 300x200]'
        """
        if content_type in ["text", "title", "list", "plain text"]:
            return f"[{self.x0:.0f}x{self.y0:.0f}]{text_content}"
        elif content_type in ["image", "figure", "equation"]:
            return f"[Image {self.x0:.0f}x{self.y0:.0f} to {self.x1:.0f}x{self.y1:.0f}]"
        elif content_type == "table":
            return f"[Table {self.x0:.0f}x{self.y0:.0f} to {self.x1:.0f}x{self.y1:.0f}]"
        else:
            # Unknown type: default to point format
            return f"[{self.x0:.0f}x{self.y0:.0f}]"

    # ==================== Properties ====================

    @property
    def left(self) -> int:
        """Get left x coordinate (alias for x0).

        Returns:
            Left x coordinate
        """
        return self.x0

    @property
    def top(self) -> int:
        """Get top y coordinate (alias for y0).

        Returns:
            Top y coordinate
        """
        return self.y0

    @property
    def right(self) -> int:
        """Get right x coordinate (alias for x1).

        Returns:
            Right x coordinate
        """
        return self.x1

    @property
    def bottom(self) -> int:
        """Get bottom y coordinate (alias for y1).

        Returns:
            Bottom y coordinate
        """
        return self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Get center point (cx, cy).

        Returns:
            Tuple of (center_x, center_y)

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.center
            (200.0, 125.0)
        """
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def width(self) -> int:
        """Get width.

        Returns:
            Width of the bounding box (integer)

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.width
            200
        """
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Get height.

        Returns:
            Height of the bounding box (integer)

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.height
            150
        """
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        """Get bbox area.

        Returns:
            Area of the bounding box (integer)

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.area
            30000
        """
        return max(0, self.width) * max(0, self.height)

    # ==================== Geometric Operations ====================

    def intersect(self, other: BBox) -> int:
        """Calculate intersection area with another bbox.

        Args:
            other: Another BBox object

        Returns:
            Intersection area (integer)

        Example:
            >>> bbox1 = BBox(100, 50, 300, 200)
            >>> bbox2 = BBox(200, 100, 400, 250)
            >>> bbox1.intersect(bbox2)
            5000
        """
        x_left = max(self.x0, other.x0)
        y_top = max(self.y0, other.y0)
        x_right = min(self.x1, other.x1)
        y_bottom = min(self.y1, other.y1)

        if x_right < x_left or y_bottom < y_top:
            return 0

        return (x_right - x_left) * (y_bottom - y_top)

    def iou(self, other: BBox) -> float:
        """Calculate IoU (Intersection over Union) with another bbox.

        Args:
            other: Another BBox object

        Returns:
            IoU value (0.0 to 1.0)

        Example:
            >>> bbox1 = BBox(100, 50, 300, 200)
            >>> bbox2 = BBox(100, 50, 300, 200)
            >>> bbox1.iou(bbox2)
            1.0
        """
        intersection = self.intersect(other)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def overlap_ratio(self, other: BBox) -> float:
        """Calculate overlap ratio relative to this bbox.

        Args:
            other: Another BBox object

        Returns:
            Ratio of intersection to this bbox's area (0.0 to 1.0)

        Example:
            >>> bbox1 = BBox(100, 50, 200, 150)
            >>> bbox2 = BBox(100, 50, 300, 200)
            >>> bbox1.overlap_ratio(bbox2)  # bbox1 fully contained
            1.0
        """
        intersection = self.intersect(other)
        return intersection / self.area if self.area > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bbox.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside bbox
        """
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def expand(self, padding: int) -> BBox:
        """Expand bbox by padding.

        Args:
            padding: Padding to add on all sides (integer pixels)

        Returns:
            New expanded BBox

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> expanded = bbox.expand(10)
            >>> expanded
            BBox(x0=90, y0=40, x1=310, y1=210)
        """
        return BBox(
            x0=self.x0 - padding,
            y0=self.y0 - padding,
            x1=self.x1 + padding,
            y1=self.y1 + padding,
        )

    def clip(self, max_width: int, max_height: int) -> BBox:
        """Clip bbox to image boundaries.

        Args:
            max_width: Maximum width (image width)
            max_height: Maximum height (image height)

        Returns:
            Clipped BBox

        Example:
            >>> bbox = BBox(100, 50, 1000, 900)
            >>> clipped = bbox.clip(800, 600)
            >>> clipped
            BBox(x0=100, y0=50, x1=800, y1=600)
        """
        return BBox(
            x0=max(0, self.x0),
            y0=max(0, self.y0),
            x1=min(max_width, self.x1),
            y1=min(max_height, self.y1),
        )

    def crop(self, image: np.ndarray, padding: int = 0) -> np.ndarray:
        """Crop this bbox area from an image with optional padding.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            padding: Padding to add on all sides (default: 0)

        Returns:
            Cropped area as numpy array

        Example:
            >>> import numpy as np
            >>> image = np.zeros((600, 800, 3), dtype=np.uint8)
            >>> bbox = BBox(100, 50, 300, 200)
            >>> cropped = bbox.crop(image, padding=5)
            >>> cropped.shape
            (160, 210, 3)
        """
        # Apply padding
        x0 = max(0, self.x0 - padding)
        y0 = max(0, self.y0 - padding)
        x1 = min(image.shape[1], self.x1 + padding)
        y1 = min(image.shape[0], self.y1 + padding)

        # Validate
        if x1 <= x0 or y1 <= y0:
            # Return small empty image if invalid
            if image.ndim == RGB_IMAGE_NDIM:
                return np.zeros((1, 1, image.shape[2]), dtype=image.dtype)
            else:
                return np.zeros((1, 1), dtype=image.dtype)

        # Crop using NumPy slicing
        return image[y0:y1, x0:x1]


# ==================== Block Dataclass ====================


@dataclass
class Block:
    """Document block with bounding box.

    This dataclass provides type safety and better IDE support for
    block data throughout the pipeline.

    Core fields:
    - type: Block type string. Should use standardized types from BlockType class
            (e.g., "text", "title", "image", "table", "code", etc.)
            Detectors may use detector-specific types which should be mapped using
            BlockTypeMapper.map_type() before further processing.
    - bbox: BBox object (required, always present)
    - detection_confidence: Detection confidence (0.0 to 1.0), optional

    Optional fields added by various pipeline stages:
    - order: Reading order rank (Added by sorters)
    - column_index: Column index (Added by multi-column sorters)
    - text: Recognized text (Added by recognizers)
    - corrected_text: VLM-corrected text (Added by text correction)
    - correction_ratio: Block-level correction ratio (0.0 = no change, 1.0 = completely different)
      (Added by text correction)
    - corrected_by: Model name that performed text correction (Added by text correction)
    - image_path: Path to extracted image file for image/figure blocks (Added by image extraction)
    - description: VLM-generated description for image/figure/chart blocks (Added by recognition)
    - source: Which detector/sorter produced this block (internal use only, not serialized)
    - index: Internal index (MinerU VLM)

    Note:
        Block types should ideally use the standardized types defined in BlockType.
        Use BlockTypeMapper.map_type(block.type, detector_name) to normalize
        detector-specific types to standardized types.
    """

    # Core fields (always present)
    type: str  # Ideally from BlockType constants
    bbox: BBox  # Required, no longer optional
    detection_confidence: float | None = None  # Optional, not all detectors provide confidence

    # Optional fields added by pipeline stages
    order: int | None = None  # Reading order rank (renamed from reading_order_rank)
    column_index: int | None = None

    # Text content
    text: str | None = None
    corrected_text: str | None = None
    correction_ratio: float | None = None  # Block-level correction ratio (0.0 = no change, 1.0 = completely different)
    corrected_by: str | None = None  # Model name that performed text correction

    # Image/Figure block fields
    image_path: str | None = None  # Path to extracted image file for image/figure/chart blocks
    description: str | None = None  # VLM-generated description for image/figure/chart blocks

    # Metadata (internal use, not serialized to JSON)
    source: str | None = None  # "doclayout-yolo", "mineru-vlm", etc.
    index: int | None = None  # MinerU VLM internal index

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Field order:
            order → type → xywh → detection_confidence → column_index → text → corrected_text
            → correction_ratio → corrected_by → image_path → description
        This order prioritizes reading order first, then type, then position, then content.

        Bbox is serialized as xywh list [x, y, width, height] for human readability.
        The 'source' field is excluded from serialization as it's for internal use only.

        Returns:
            Dict with xywh bbox format

        Example:
            >>> block = Block(type="text", bbox=BBox(100, 50, 300, 200), detection_confidence=0.95, order=0)
            >>> block.to_dict()
            {'order': 0, 'type': 'text', 'xywh': [100, 50, 200, 150], 'detection_confidence': 0.95}
        """
        result: dict[str, Any] = {}

        # Add fields in desired order (Python 3.7+ preserves insertion order)
        # 1. Reading order (most important for document reconstruction)
        if self.order is not None:
            result["order"] = self.order

        # 2. Type (what kind of block)
        result["type"] = self.type

        # 3. Position (where it is)
        result["xywh"] = self.bbox.to_xywh_list()

        # 4. Detection confidence (optional)
        if self.detection_confidence is not None:
            result["detection_confidence"] = self.detection_confidence

        # 5. Column index (optional, for multi-column layouts)
        if self.column_index is not None:
            result["column_index"] = self.column_index

        # 6. Text content (extracted)
        if self.text is not None:
            result["text"] = self.text

        # 7. Corrected text (VLM-corrected)
        if self.corrected_text is not None:
            result["corrected_text"] = self.corrected_text

        # 8. Correction ratio (block-level)
        if self.correction_ratio is not None:
            result["correction_ratio"] = self.correction_ratio

        # 9. Corrected by (model name that performed correction)
        if self.corrected_by is not None:
            result["corrected_by"] = self.corrected_by

        # 10. Image/figure block fields
        if self.image_path is not None:
            result["image_path"] = self.image_path

        if self.description is not None:
            result["description"] = self.description

        # Note: 'source' and 'index' are intentionally excluded (internal metadata)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        """Create Block from dict.

        Args:
            data: Dictionary with block data (must have "xywh" field)

        Returns:
            Block object

        Example:
            >>> data = {"type": "text", "xywh": [100, 50, 200, 150], "detection_confidence": 0.95}
            >>> block = Block.from_dict(data)
            >>> block.bbox
            BBox(x0=100, y0=50, x1=300, y1=200)
        """
        data = data.copy()  # Don't modify original

        # Parse xywh field (required)
        if "xywh" not in data:
            raise ValueError("Block dict must have 'xywh' field")

        xywh_value = data.pop("xywh")
        if not isinstance(xywh_value, (list, tuple)):
            raise ValueError(f"Invalid xywh type: {type(xywh_value)}")

        bbox = BBox.from_xywh(*xywh_value[:4])

        # Build Block with remaining fields
        return cls(
            type=data.get("type", "unknown"),
            bbox=bbox,
            detection_confidence=data.get("detection_confidence"),
            order=data.get("order"),
            column_index=data.get("column_index"),
            text=data.get("text"),
            corrected_text=data.get("corrected_text"),
            correction_ratio=data.get("correction_ratio"),
            corrected_by=data.get("corrected_by"),
            image_path=data.get("image_path"),
            description=data.get("description"),
            source=data.get("source"),
            index=data.get("index"),
        )


# ==================== Page and Document Types ====================


@dataclass
class Page:
    """Single page processing result.

    Represents the result of processing one page, containing detected blocks
    and associated metadata. This provides type safety for page-level operations.

    Core fields:
    - page_num: Page number (1-indexed)
    - blocks: List of detected Block objects

    Optional metadata fields:
    - image_path: Path to rendered page image
    - auxiliary_info: Additional info (e.g., text spans for markdown conversion)
    - status: Processing status ("completed", "failed", etc.)
    - processed_at: ISO timestamp
    - page_path: Path to saved JSON output
    """

    # Core fields
    page_num: int
    blocks: list[Block]

    # Optional metadata
    image_path: str | None = None
    auxiliary_info: dict[str, Any] | None = None  # Flexible storage for various metadata
    status: str = "completed"
    processed_at: str | None = None
    page_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dict with blocks converted to dicts

        Example:
            >>> page = Page(page_num=1, blocks=[...])
            >>> page.to_dict()
            {'page_num': 1, 'blocks': [...], 'status': 'completed'}
        """
        result: dict[str, Any] = {
            "page_num": self.page_num,
            "blocks": [b.to_dict() for b in self.blocks],
        }

        # Add optional fields (only if not None)
        if self.image_path is not None:
            result["image_path"] = self.image_path
        if self.auxiliary_info is not None:
            result["auxiliary_info"] = self.auxiliary_info
        # Always include status for consistency
        result["status"] = self.status
        if self.processed_at is not None:
            result["processed_at"] = self.processed_at
        if self.page_path is not None:
            result["page_path"] = self.page_path

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        """Create Page from dict.

        Args:
            data: Dictionary with page data

        Returns:
            Page object

        Example:
            >>> data = {"page_num": 1, "blocks": [...]}
            >>> page = Page.from_dict(data)
        """
        return cls(
            page_num=data["page_num"],
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            image_path=data.get("image_path"),
            auxiliary_info=data.get("auxiliary_info"),
            status=data.get("status", "completed"),
            processed_at=data.get("processed_at"),
            page_path=data.get("page_path"),
        )


@dataclass
class Document:
    """Multi-page document processing result.

    Represents the complete result of processing a PDF or multi-page document.
    This provides type safety for document-level operations.

    Core fields:
    - pdf_name: Document name (without extension)
    - pdf_path: Full path to source PDF
    - num_pages: Total number of pages in document
    - processed_pages: Number of pages actually processed
    - pages: List of Page objects

    Optional metadata fields:
    - output_directory: Directory where results are saved
    - processed_at: ISO timestamp
    - status_summary: Count of pages by status
    """

    # Core fields
    pdf_name: str
    pdf_path: str
    num_pages: int
    processed_pages: int
    pages: list[Page]

    # Optional metadata
    detected_by: str | None = None  # Detector name
    ordered_by: str | None = None  # Sorter name
    recognized_by: str | None = None  # Backend/model name
    corrected_by: str | None = None  # Model name for text correction (block/page level)
    rendered_by: str | None = None  # Renderer name (markdown/plaintext)
    output_directory: str | None = None
    processed_at: str | None = None
    status_summary: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dict with pages converted to dicts

        Example:
            >>> doc = Document(pdf_name="test", pdf_path="/test.pdf", ...)
            >>> doc.to_dict()
            {'pdf_name': 'test', 'pdf_path': '/test.pdf', ...}
        """
        result: dict[str, Any] = {
            "pdf_name": self.pdf_name,
            "pdf_path": self.pdf_path,
            "num_pages": self.num_pages,
            "processed_pages": self.processed_pages,
            "pages": [p.to_dict() for p in self.pages],
        }

        # Add optional fields (only if not None)
        if self.detected_by is not None:
            result["detected_by"] = self.detected_by
        if self.ordered_by is not None:
            result["ordered_by"] = self.ordered_by
        if self.recognized_by is not None:
            result["recognized_by"] = self.recognized_by
        if self.corrected_by is not None:
            result["corrected_by"] = self.corrected_by
        if self.rendered_by is not None:
            result["rendered_by"] = self.rendered_by
        if self.output_directory is not None:
            result["output_directory"] = self.output_directory
        if self.processed_at is not None:
            result["processed_at"] = self.processed_at
        if self.status_summary is not None:
            result["status_summary"] = self.status_summary

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create Document from dict.

        Args:
            data: Dictionary with document data

        Returns:
            Document object

        Example:
            >>> data = {"pdf_name": "test", "pdf_path": "/test.pdf", ...}
            >>> doc = Document.from_dict(data)
        """
        return cls(
            pdf_name=data["pdf_name"],
            pdf_path=data["pdf_path"],
            num_pages=data["num_pages"],
            processed_pages=data["processed_pages"],
            pages=[Page.from_dict(p) for p in data.get("pages", [])],
            detected_by=data.get("detected_by"),
            ordered_by=data.get("ordered_by"),
            recognized_by=data.get("recognized_by"),
            corrected_by=data.get("corrected_by"),
            rendered_by=data.get("rendered_by"),
            output_directory=data.get("output_directory"),
            processed_at=data.get("processed_at"),
            status_summary=data.get("status_summary"),
        )


# ==================== Protocol Interfaces ====================


@runtime_checkable
class Detector(Protocol):
    """Layout detection interface.

    All detectors must implement this interface and return blocks
    in the unified Block format with bbox field.

    Attributes:
        name: Detector identifier (e.g., "doclayout-yolo", "paddleocr-doclayout-v2")
        source: Source identifier for blocks (used in Block.source field)

    Methods:
        detect: Detect blocks in a single image
        detect_batch: Detect blocks in multiple images (optional, default: sequential)

    Example:
        >>> detector = DocLayoutYOLODetector()
        >>> detector.name
        'doclayout-yolo'
        >>> blocks = detector.detect(image)
    """

    name: str
    source: str

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect blocks in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected Block objects with bbox

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> blocks = detector.detect(image)
            >>> blocks[0].type
            'text'
            >>> blocks[0].bbox
            BBox(x0=100, y0=50, x1=300, y1=200)
        """
        ...

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
        """Detect blocks in multiple images.

        Default implementation processes images sequentially.
        Subclasses may override for true batch/parallel processing.

        Args:
            images: List of input images as numpy arrays (H, W, C)

        Returns:
            List of block lists, one per image

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> results = detector.detect_batch([image1, image2])
            >>> len(results)
            2
        """
        ...


@runtime_checkable
class Sorter(Protocol):
    """Reading order sorting interface.

    All sorters must implement this interface and add ordering information
    to blocks (order field) while maintaining the unified Block format.

    Attributes:
        name: Sorter identifier (e.g., "pymupdf", "mineru-xycut")

    Methods:
        sort: Sort blocks by reading order

    Example:
        >>> sorter = PyMuPDFSorter()
        >>> sorter.name
        'pymupdf'
        >>> sorted_blocks = sorter.sort(blocks, image)
    """

    name: str

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Sort blocks by reading order.

        Args:
            blocks: Detected blocks with bbox
            image: Page image for analysis (H, W, C)
            **kwargs: Additional context (e.g., pymupdf_page, pdf_path)

        Returns:
            Sorted blocks with order field added

        Example:
            >>> sorter = PyMuPDFSorter()
            >>> sorted_blocks = sorter.sort(blocks, image, pymupdf_page=page)
            >>> sorted_blocks[0].order
            0
        """
        ...


@runtime_checkable
class Recognizer(Protocol):
    """Text recognition interface.

    All recognizers must implement this interface to extract and correct
    text from blocks using various OCR or VLM backends.

    Attributes:
        name: Recognizer identifier (e.g., "gemini", "openai", "paddleocr-vl")
        supports_correction: Whether the recognizer supports text correction

    Methods:
        process_blocks: Extract text from blocks in an image
        correct_text: Correct raw text using VLM (optional for some backends)
        process_blocks_batch: Process multiple sets of blocks (optional)

    Example:
        >>> recognizer = GeminiClient(model="gemini-2.5-flash")
        >>> recognizer.name
        'gemini'
        >>> recognizer.supports_correction
        True
        >>> blocks_with_text = recognizer.process_blocks(image, blocks)
    """

    name: str
    supports_correction: bool

    def process_blocks(self, image: np.ndarray | None, blocks: Sequence[Block]) -> list[Block]:
        """Process blocks to extract text.

        This method processes each block in the input image to extract text content.
        The returned blocks should have their `text` field populated.

        Args:
            image: Full page image as numpy array (H, W, C) in RGB format.
                   Can be None for recognizers that don't need the image.
            blocks: Detected blocks with bbox field populated

        Returns:
            List of blocks with text field populated

        Example:
            >>> recognizer = GeminiClient(model="gemini-2.5-flash")
            >>> blocks_with_text = recognizer.process_blocks(image, blocks)
            >>> blocks_with_text[0].text
            'Sample text content'
        """
        ...

    def correct_text(self, text: str) -> str | dict[str, Any]:
        """Correct extracted text using VLM.

        This method takes raw extracted text and applies correction using a VLM.
        Some recognizers may not support correction (e.g., PaddleOCR-VL).
        Check `supports_correction` attribute before calling.

        Args:
            text: Raw extracted text to correct

        Returns:
            Corrected text string, or dict with correction metadata:
            - {"corrected_text": str, "correction_ratio": float}
            If correction is not supported, returns the original text unchanged.

        Example:
            >>> recognizer = GeminiClient(model="gemini-2.5-flash")
            >>> if recognizer.supports_correction:
            ...     corrected = recognizer.correct_text("sampel txt")
            ...     print(corrected)
            'sample text'

            >>> # For recognizers without correction support
            >>> recognizer = PaddleOCRVLRecognizer()
            >>> recognizer.supports_correction
            False
        """
        ...

    def process_blocks_batch(
        self,
        images: Sequence[np.ndarray | None],
        blocks_list: Sequence[Sequence[Block]],
    ) -> list[list[Block]]:
        """Process multiple sets of blocks in a batch.

        Default implementation processes sequentially.
        Subclasses may override for true batch/parallel processing.

        Args:
            images: Sequence of input images
            blocks_list: Sequence of block lists, one per image

        Returns:
            List of processed block lists

        Example:
            >>> results = recognizer.process_blocks_batch(
            ...     [image1, image2],
            ...     [blocks1, blocks2]
            ... )
        """
        ...


class Renderer(Protocol):
    """Output rendering interface.

    Renderers convert processed blocks/pages/documents to various output
    formats (markdown, plaintext, HTML, etc.).

    This is a callable protocol - any function matching the signature can be used.
    """

    def __call__(self, blocks: Sequence[Block], **kwargs: Any) -> str:
        """Render blocks to output format.

        Args:
            blocks: Processed blocks with text
            **kwargs: Additional rendering options

        Returns:
            Rendered output string

        Example:
            >>> renderer = blocks_to_markdown
            >>> output = renderer(blocks)
            >>> print(output)
            # Title

            Sample content...
        """
        ...


# ==================== Pipeline Result Types ====================


@dataclass
class StageTimingInfo:
    """Timing information for a pipeline stage.

    Attributes:
        stage_name: Name of the stage
        processing_time_ms: Processing time in milliseconds
        items_processed: Number of items processed (e.g., blocks, pages)
    """

    stage_name: str
    processing_time_ms: float
    items_processed: int = 0

    @property
    def processing_time_sec(self) -> float:
        """Get processing time in seconds."""
        return self.processing_time_ms / 1000.0


@dataclass
class PipelineResult:
    """Complete pipeline processing result.

    This dataclass captures the full result of processing a document
    through the OCR pipeline, including timing and metadata.

    Attributes:
        document: Processed Document object with all pages
        stage_timings: Timing information for each stage
        total_time_ms: Total processing time in milliseconds
        success: Whether processing completed successfully
        error: Error message if processing failed

    Example:
        >>> result = pipeline.process_pdf("document.pdf")
        >>> result.success
        True
        >>> result.total_time_sec
        12.5
        >>> result.get_stage_timings()
        {'detection': 2000.0, 'ordering': 500.0, 'recognition': 8000.0, ...}
    """

    document: Document | None
    stage_timings: list[StageTimingInfo]
    total_time_ms: float
    success: bool = True
    error: str | None = None

    @property
    def total_time_sec(self) -> float:
        """Get total processing time in seconds."""
        return self.total_time_ms / 1000.0

    def get_stage_timings(self) -> dict[str, float]:
        """Get timing for each stage as a dictionary.

        Returns:
            Dictionary mapping stage names to processing times in ms
        """
        return {timing.stage_name: timing.processing_time_ms for timing in self.stage_timings}

    def get_slowest_stage(self) -> StageTimingInfo | None:
        """Get the slowest stage.

        Returns:
            StageTimingInfo for the slowest stage, or None if no stages
        """
        if not self.stage_timings:
            return None
        return max(self.stage_timings, key=lambda t: t.processing_time_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dictionary with all result information
        """
        result: dict[str, Any] = {
            "success": self.success,
            "total_time_ms": self.total_time_ms,
            "total_time_sec": self.total_time_sec,
            "stage_timings": {t.stage_name: t.processing_time_ms for t in self.stage_timings},
        }

        if self.document is not None:
            result["document"] = self.document.to_dict()

        if self.error is not None:
            result["error"] = self.error

        return result


# Type alias for renderer functions (alternative to Protocol)
# Use this when you need a simple type hint without Protocol overhead
RendererFunc = Callable[[Sequence[Block]], str]


# ==================== Utility Functions ====================


def blocks_to_olmocr_anchor_text(
    blocks: Sequence[Block],
    page_width: int,
    page_height: int,
    max_length: int = 4000,
) -> str:
    """Convert blocks to olmOCR anchor text format.

    Args:
        blocks: List of Block instances with bbox
        page_width: Page width in pixels
        page_height: Page height in pixels
        max_length: Maximum anchor text length (approximate)

    Returns:
        olmOCR anchor text string

    Example:
        >>> blocks = [
        ...     Block(type="title", bbox=BBox(100, 50, 300, 80), detection_confidence=0.9),
        ...     Block(type="figure", bbox=BBox(100, 100, 300, 250), detection_confidence=0.95),
        ... ]
        >>> anchor = blocks_to_olmocr_anchor_text(blocks, 800, 600)
        >>> print(anchor)
        Page dimensions: 800x600
        [100x50]
        [Image 100x100 to 300x250]
    """
    # Header
    lines = [f"Page dimensions: {page_width}x{page_height}"]

    # Convert each block
    for block in blocks:
        bbox = block.bbox
        text_content = (block.text or "")[:50] if block.type in ["text", "title", "plain text"] else ""
        anchor_line = bbox.to_olmocr_anchor(content_type=block.type, text_content=text_content)
        lines.append(anchor_line)

        # Check length limit
        current_length = sum(len(line) for line in lines)
        if current_length > max_length:
            break

    return "\n".join(lines)
