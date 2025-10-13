"""Unified type definitions for the VLM OCR pipeline.

This module provides:
- BBox: Unified bounding box with format conversions
- Region: TypedDict for document regions
- Detector/Sorter: Protocol interfaces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, Sequence, TypedDict

if TYPE_CHECKING:
    import numpy as np


# ==================== BBox Format Definitions ====================

"""
BBox formats used by different frameworks:

1. Current project (DocLayout-YOLO output):
   coords: [x, y, width, height]
   Example: [100, 50, 200, 150]

2. YOLO internal (xyxy):
   [x1, y1, x2, y2]
   Example: [100, 50, 300, 200] - top-left to bottom-right

3. PyMuPDF Rect:
   Rect(x0, y0, x1, y1)
   Example: Rect(100, 50, 300, 200)

4. MinerU blocks:
   bbox: [x0, y0, x1, y1]
   Example: [100, 50, 300, 200]

5. PyPDF (bottom-left origin!):
   [x0, y0, x1, y1] - Y-axis is inverted
   Example: [100, 642, 300, 742] when page height is 792

6. olmOCR anchor text:
   - Text: "[x, y]text"
   - Image: "[Image x0, y0 to x1, y1]"
   Example: "[100x50]Chapter 1" or "[Image 100x50 to 300x200]"
"""


@dataclass(frozen=True)
class BBox:
    """Unified bounding box representation with format conversions.

    Internal format: (x0, y0, x1, y1) - Top-left and bottom-right corners
    Origin: Top-left corner of image (0, 0) - standard computer vision convention

    This class provides conversion methods to/from all major bbox formats
    used in document processing frameworks.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    # ==================== FROM Conversions (Format → BBox) ====================

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create from (x, y, width, height) format.

        Format: [x, y, width, height]
        Used by: Current project (DocLayout-YOLO output)

        Args:
            x: Left x coordinate
            y: Top y coordinate
            w: Width
            h: Height

        Returns:
            Unified BBox object

        Example:
            >>> bbox = BBox.from_xywh(100, 50, 200, 150)
            >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
            (100, 50, 300, 200)
        """
        return cls(x0=x, y0=y, x1=x + w, y1=y + h)

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float) -> BBox:
        """Create from (x0, y0, x1, y1) format.

        Format: [x0, y0, x1, y1] - corners
        Used by: YOLO internal, MinerU, PyMuPDF

        Args:
            x0: Left x coordinate
            y0: Top y coordinate
            x1: Right x coordinate
            y1: Bottom y coordinate

        Returns:
            Unified BBox object

        Example:
            >>> bbox = BBox.from_xyxy(100, 50, 300, 200)
            >>> bbox.width, bbox.height
            (200, 150)
        """
        return cls(x0=x0, y0=y0, x1=x1, y1=y1)

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
    def from_pymupdf_rect(cls, rect: Any) -> BBox:
        """Create from PyMuPDF Rect or IRect object.

        Format: Rect(x0, y0, x1, y1)
        Used by: PyMuPDF (fitz)

        Args:
            rect: PyMuPDF Rect or IRect object

        Returns:
            Unified BBox object

        Example:
            >>> import fitz
            >>> rect = fitz.Rect(100, 50, 300, 200)
            >>> bbox = BBox.from_pymupdf_rect(rect)
        """
        return cls(x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1))

    @classmethod
    def from_mineru_bbox(cls, bbox: Sequence[float]) -> BBox:
        """Create from MinerU bbox format.

        Format: [x0, y0, x1, y1]
        Used by: MinerU blocks

        Args:
            bbox: MinerU bbox list

        Returns:
            Unified BBox object

        Example:
            >>> bbox = BBox.from_mineru_bbox([100, 50, 300, 200])
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
            Unified BBox object with top-left origin

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
        return cls(x0=x0, y0=y0_top, x1=x1, y1=y1_top)

    @classmethod
    def from_olmocr_anchor_coords(cls, x: float, y: float) -> BBox:
        """Create from olmOCR anchor text coordinates (point format).

        Format: "[x, y]" - single point
        Used by: olmOCR anchor text for text elements

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            BBox representing a point (x0=x1, y0=y1)

        Example:
            >>> bbox = BBox.from_olmocr_anchor_coords(100, 50)
            >>> bbox.x0, bbox.y0
            (100, 50)
        """
        return cls(x0=x, y0=y, x1=x, y1=y)

    @classmethod
    def from_olmocr_anchor_box(cls, x0: float, y0: float, x1: float, y1: float) -> BBox:
        """Create from olmOCR anchor image/table bbox.

        Format: "[Image x0, y0 to x1, y1]"
        Used by: olmOCR anchor text for images/tables

        Args:
            x0, y0: Top-left coordinates
            x1, y1: Bottom-right coordinates

        Returns:
            Unified BBox object

        Example:
            >>> bbox = BBox.from_olmocr_anchor_box(100, 50, 300, 200)
        """
        return cls.from_xyxy(x0, y0, x1, y1)

    # ==================== TO Conversions (BBox → Format) ====================

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format.

        Format: [x, y, width, height]
        Used by: Current project output

        Returns:
            Tuple of (x, y, width, height)

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_xywh()
            (100, 50, 200, 150)
        """
        return (self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x0, y0, x1, y1) format.

        Format: [x0, y0, x1, y1] - corners
        Used by: MinerU, YOLO, PyMuPDF

        Returns:
            Tuple of (x0, y0, x1, y1)

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_xyxy()
            (100, 50, 300, 200)
        """
        return (self.x0, self.y0, self.x1, self.y1)

    def to_list_xywh(self) -> list[float]:
        """Convert to [x, y, w, h] list.

        Returns:
            List of [x, y, width, height]
        """
        return list(self.to_xywh())

    def to_list_xyxy(self) -> list[float]:
        """Convert to [x0, y0, x1, y1] list.

        Returns:
            List of [x0, y0, x1, y1]
        """
        return list(self.to_xyxy())

    def to_mineru_bbox(self) -> list[float]:
        """Convert to MinerU bbox format.

        Format: [x0, y0, x1, y1]
        Used by: MinerU blocks

        Returns:
            List of [x0, y0, x1, y1]
        """
        return self.to_list_xyxy()

    def to_pypdf_rect(self, page_height: float) -> list[float]:
        """Convert to PyPDF rect format (requires Y-axis flip).

        Format: [x0, y0, x1, y1] with bottom-left origin
        Used by: PyPDF

        Args:
            page_height: Page height for Y-axis conversion

        Returns:
            List of [x0, y0, x1, y1] with bottom-left origin

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.to_pypdf_rect(page_height=792)
            [100, 592, 300, 742]  # Y-axis flipped
        """
        # Y-axis flip: top-left → bottom-left
        y0_bottom = page_height - self.y1
        y1_bottom = page_height - self.y0
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
            text_content: Text content for text regions (optional)

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
    def width(self) -> float:
        """Get width.

        Returns:
            Width of the bounding box

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.width
            200
        """
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Get height.

        Returns:
            Height of the bounding box

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.height
            150
        """
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Get bbox area.

        Returns:
            Area of the bounding box

        Example:
            >>> bbox = BBox(100, 50, 300, 200)
            >>> bbox.area
            30000
        """
        return max(0, self.width) * max(0, self.height)

    # ==================== Geometric Operations ====================

    def intersect(self, other: BBox) -> float:
        """Calculate intersection area with another bbox.

        Args:
            other: Another BBox object

        Returns:
            Intersection area

        Example:
            >>> bbox1 = BBox(100, 50, 300, 200)
            >>> bbox2 = BBox(200, 100, 400, 250)
            >>> bbox1.intersect(bbox2)
            5000.0
        """
        x_left = max(self.x0, other.x0)
        y_top = max(self.y0, other.y0)
        x_right = min(self.x1, other.x1)
        y_bottom = min(self.y1, other.y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

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

    def expand(self, padding: float) -> BBox:
        """Expand bbox by padding.

        Args:
            padding: Padding to add on all sides

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

    def clip(self, max_width: float, max_height: float) -> BBox:
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


# ==================== Region TypedDict ====================

class Region(TypedDict):
    """Unified region format used throughout the pipeline.

    This TypedDict maintains backward compatibility with existing dict-based
    code while providing type safety.

    Standard fields:
    - type: Region type (e.g., "plain text", "title", "figure", "table")
    - coords: [x, y, w, h] - Legacy format for backward compatibility
    - confidence: Detection confidence (0.0 to 1.0)

    Optional fields added by various pipeline stages:
    - bbox: Unified BBox object (for type-safe operations)
    - reading_order_rank: Added by sorters
    - column_index: Added by multi-column sorters
    - text: Added by recognizers
    - corrected_text: Added by text correction
    - source: Which detector/sorter produced this region
    """

    # Core fields (always present after detection)
    type: str
    coords: list[float]  # [x, y, w, h] - legacy format
    confidence: float

    # Optional bbox object (for type-safe operations)
    bbox: NotRequired[BBox]

    # Added by sorters
    reading_order_rank: NotRequired[int]
    column_index: NotRequired[int]

    # Added by recognizers
    text: NotRequired[str]
    corrected_text: NotRequired[str]

    # Metadata
    source: NotRequired[str]  # "doclayout-yolo", "mineru-vlm", etc.
    index: NotRequired[int]  # MinerU VLM internal index


# ==================== Protocol Interfaces ====================

class Detector(Protocol):
    """Layout detection interface.

    All detectors must implement this interface and return regions
    in the unified Region format with coords and bbox fields.
    """

    def detect(self, image: np.ndarray) -> list[Region]:
        """Detect regions in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected regions in unified format

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> regions = detector.detect(image)
            >>> regions[0]["type"]
            'plain text'
            >>> regions[0]["coords"]
            [100, 50, 200, 150]
        """
        ...


class Sorter(Protocol):
    """Reading order sorting interface.

    All sorters must implement this interface and add ordering information
    to regions (reading_order_rank) while maintaining the unified Region format.
    """

    def sort(self, regions: list[Region], image: np.ndarray, **kwargs: Any) -> list[Region]:
        """Sort regions by reading order.

        Args:
            regions: Detected regions in unified format
            image: Page image for analysis (H, W, C)
            **kwargs: Additional context (e.g., pymupdf_page, pdf_path)

        Returns:
            Sorted regions with reading_order_rank field added

        Example:
            >>> sorter = PyMuPDFSorter()
            >>> sorted_regions = sorter.sort(regions, image, pymupdf_page=page)
            >>> sorted_regions[0]["reading_order_rank"]
            0
        """
        ...


# ==================== Utility Functions ====================

def ensure_bbox_in_region(region: dict[str, Any]) -> dict[str, Any]:
    """Ensure region has bbox field populated from coords.

    This function is idempotent and safe to call multiple times.

    Args:
        region: Region dict (may or may not have bbox)

    Returns:
        Region dict with bbox field added

    Example:
        >>> region = {"type": "text", "coords": [100, 50, 200, 150], "confidence": 0.9}
        >>> region = ensure_bbox_in_region(region)
        >>> region["bbox"]
        BBox(x0=100, y0=50, x1=300, y1=200)
    """
    if "bbox" not in region or region["bbox"] is None:
        region["bbox"] = BBox.from_list(region["coords"], coord_format="xywh")
    return region


def regions_to_olmocr_anchor_text(
    regions: Sequence[dict[str, Any]],
    page_width: int,
    page_height: int,
    max_length: int = 4000,
) -> str:
    """Convert regions to olmOCR anchor text format.

    Args:
        regions: List of regions with bbox information
        page_width: Page width in pixels
        page_height: Page height in pixels
        max_length: Maximum anchor text length (approximate)

    Returns:
        olmOCR anchor text string

    Example:
        >>> regions = [
        ...     {"type": "title", "coords": [100, 50, 200, 30], "confidence": 0.9},
        ...     {"type": "figure", "coords": [100, 100, 200, 150], "confidence": 0.95},
        ... ]
        >>> anchor = regions_to_olmocr_anchor_text(regions, 800, 600)
        >>> print(anchor)
        Page dimensions: 800.0x600.0
        [100x50]
        [Image 100x100 to 300x250]
    """
    # Ensure all regions have bbox
    regions_with_bbox: Sequence[dict[str, Any]] = [ensure_bbox_in_region(r) for r in regions]

    # Header
    lines = [f"Page dimensions: {page_width:.1f}x{page_height:.1f}"]

    # Convert each region
    for region in regions_with_bbox:
        if "bbox" not in region:
            continue
        bbox: BBox = region["bbox"]
        text_content = region.get("text", "")[:50] if region["type"] in ["text", "title"] else ""
        anchor_line = bbox.to_olmocr_anchor(content_type=region["type"], text_content=text_content)
        lines.append(anchor_line)

        # Check length limit
        current_length = sum(len(line) for line in lines)
        if current_length > max_length:
            break

    return "\n".join(lines)


def normalize_region_coords(region: dict[str, Any]) -> Region:
    """Normalize a region dict to ensure it has proper coords and bbox.

    Handles various input formats and ensures consistent output.

    Args:
        region: Region dict with various possible formats

    Returns:
        Normalized Region with coords and bbox

    Example:
        >>> raw = {"type": "text", "bbox": [100, 50, 300, 200], "confidence": 0.9}
        >>> normalized = normalize_region_coords(raw)
        >>> normalized["coords"]
        [100, 50, 200, 150]
    """
    # If has coords, use it
    if "coords" in region:
        coords = region["coords"]
        if "bbox" not in region:
            region["bbox"] = BBox.from_list(coords, coord_format="xywh")
    # If has bbox field (MinerU format), convert to coords
    elif "bbox" in region and isinstance(region["bbox"], (list, tuple)):
        bbox = BBox.from_list(region["bbox"], coord_format="xyxy")
        region["coords"] = bbox.to_list_xywh()
        region["bbox"] = bbox
    else:
        raise ValueError("Region must have either 'coords' or 'bbox' field")

    return region  # type: ignore[return-value]

