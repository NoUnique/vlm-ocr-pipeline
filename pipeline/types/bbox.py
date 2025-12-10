"""BBox class for bounding box operations.

Internal format: (x0, y0, x1, y1) - xyxy corners
JSON output: [x, y, width, height] - xywh format
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .external import PyMuPDFRect

# Constants
RGB_IMAGE_NDIM = 3  # RGB image has 3 dimensions (H, W, C)


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

