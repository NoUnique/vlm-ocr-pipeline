# Types API

Core type definitions and data structures used throughout the pipeline.

## Overview

The pipeline uses a unified type system for representing document elements:

- **BBox**: Pixel-based bounding box with automatic format conversion
- **Block**: Document block (text, table, figure, etc.) with metadata
- **Page**: Processed page with blocks and text
- **Document**: Complete document with multiple pages

## BBox

Unified bounding box representation with automatic format conversion between 6+ formats.

**Key Features:**

- Internal format: `(x0, y0, x1, y1)` - integer pixel coordinates (xyxy corners)
- JSON output: `[x, y, w, h]` - position + size for human readability
- Automatic conversion from YOLO, MinerU, PyMuPDF, PyPDF, olmOCR formats

### Quick Example

```python
from pipeline.types import BBox

# Create from different formats
bbox = BBox.from_xywh(100, 50, 200, 150)      # [x, y, w, h]
bbox = BBox.from_xyxy(100, 50, 300, 200)      # corners
bbox = BBox.from_pypdf_rect([100, 592, 300, 742], page_height=792)

# Convert to formats
xywh = bbox.to_xywh_list()     # [100, 50, 200, 150]
xyxy = bbox.to_list()          # [100, 50, 300, 200]

# Properties
print(bbox.width, bbox.height)  # 200, 150
print(bbox.area)                # 30000
print(bbox.center)              # (200.0, 125.0)

# Operations
cropped = bbox.crop(image, padding=5)
iou = bbox1.iou(bbox2)
```

### BBox Class Reference

```python
@dataclass(frozen=True)
class BBox:
    """Pixel-based bounding box with integer coordinates.

    Internal format: (x0, y0, x1, y1) - Top-left and bottom-right corners (xyxy)
    Origin: Top-left corner of image (0, 0)
    Coordinates: Integer pixel values (not normalized)
    """

    x0: int  # Left
    y0: int  # Top
    x1: int  # Right
    y1: int  # Bottom
```

### Factory Methods

| Method | Format | Example |
|--------|--------|---------|
| `from_xywh(x, y, w, h)` | Position + Size | `BBox.from_xywh(100, 50, 200, 150)` |
| `from_xyxy(x0, y0, x1, y1)` | Corners | `BBox.from_xyxy(100, 50, 300, 200)` |
| `from_list(coords, format)` | List | `BBox.from_list([100, 50, 200, 150], "xywh")` |
| `from_pymupdf_rect(rect)` | PyMuPDF Rect | `BBox.from_pymupdf_rect(rect)` |
| `from_mineru_bbox(bbox)` | MinerU list | `BBox.from_mineru_bbox([100, 50, 300, 200])` |
| `from_pypdf_rect(rect, height)` | PyPDF (Y-flip) | `BBox.from_pypdf_rect([100, 592, 300, 742], 792)` |
| `from_cxcywh(cx, cy, w, h)` | Center format | `BBox.from_cxcywh(200, 125, 200, 150)` |

### Conversion Methods

| Method | Output | Example |
|--------|--------|---------|
| `to_xywh()` | Tuple | `(100, 50, 200, 150)` |
| `to_xyxy()` | Tuple | `(100, 50, 300, 200)` |
| `to_xywh_list()` | List (JSON) | `[100, 50, 200, 150]` |
| `to_list()` | List (xyxy) | `[100, 50, 300, 200]` |
| `to_dict()` | Dict | `{"x0": 100, "y0": 50, "x1": 300, "y1": 200}` |
| `to_mineru_bbox()` | List | `[100, 50, 300, 200]` |
| `to_pypdf_rect(height)` | List (Y-flip) | `[100, 592, 300, 742]` |
| `to_olmocr_anchor(type)` | String | `"[100x50]text"` |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `width` | int | Width of bbox |
| `height` | int | Height of bbox |
| `area` | int | Area (width × height) |
| `center` | tuple[float, float] | Center point (cx, cy) |
| `left`, `top`, `right`, `bottom` | int | Edge coordinates |

### Geometric Operations

| Method | Description | Returns |
|--------|-------------|---------|
| `intersect(other)` | Intersection area | int |
| `iou(other)` | Intersection over Union | float (0-1) |
| `overlap_ratio(other)` | Overlap relative to self | float (0-1) |
| `contains_point(x, y)` | Point inside bbox? | bool |
| `expand(padding)` | Expand by padding | BBox |
| `clip(max_w, max_h)` | Clip to bounds | BBox |
| `crop(image, padding)` | Crop from image | np.ndarray |

## Block

Document block representing a detected layout element.

### Quick Example

```python
from pipeline.types import Block, BBox

# Create block
block = Block(
    type="text",
    bbox=BBox(100, 50, 500, 200),
    detection_confidence=0.95,
    order=0,
    text="Extracted text content",
)

# Serialize to JSON
data = block.to_dict()
# {"order": 0, "type": "text", "xywh": [100, 50, 400, 150], ...}

# Deserialize from JSON
block = Block.from_dict(data)
```

### Block Class Reference

```python
@dataclass
class Block:
    """Document block with bounding box."""

    # Core fields (always present)
    type: str              # "text", "title", "table", "image", etc.
    bbox: BBox             # Required, always present
    detection_confidence: float | None = None

    # Added by sorters
    order: int | None = None           # Reading order rank
    column_index: int | None = None    # Column index

    # Added by recognizers
    text: str | None = None            # Extracted text
    corrected_text: str | None = None  # VLM-corrected text
    correction_ratio: float | None = None
    corrected_by: str | None = None

    # Image/figure blocks
    image_path: str | None = None      # Path to extracted image
    description: str | None = None     # VLM description

    # Metadata (internal use)
    source: str | None = None          # Detector name
    index: int | None = None           # Internal index
```

### Methods

| Method | Description |
|--------|-------------|
| `to_dict()` | Convert to JSON-serializable dict (bbox → xywh) |
| `from_dict(data)` | Create Block from dict (xywh → BBox) |

## BlockType

Standardized block type constants for consistent processing.

```python
class BlockType:
    # Content
    TEXT = "text"
    TITLE = "title"

    # Figures
    IMAGE = "image"
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"

    # Tables
    TABLE = "table"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"

    # Equations
    EQUATION = "equation"
    INTERLINE_EQUATION = "interline_equation"
    INLINE_EQUATION = "inline_equation"

    # Code
    CODE = "code"
    ALGORITHM = "algorithm"

    # Lists
    LIST = "list"

    # Page Elements
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
```

## Protocol Interfaces

### Detector Protocol

All detectors must implement this interface:

```python
from typing import Protocol
import numpy as np
from pipeline.types import Block

class Detector(Protocol):
    """Protocol for layout detectors."""

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected blocks
        """
        ...
```

### Sorter Protocol

All sorters must implement this interface:

```python
class Sorter(Protocol):
    """Protocol for reading order sorters."""

    def sort(
        self,
        blocks: list[Block],
        image: np.ndarray,
        **kwargs,
    ) -> list[Block]:
        """Sort blocks in reading order.

        Args:
            blocks: Detected blocks
            image: Original page image
            **kwargs: Additional arguments

        Returns:
            Sorted blocks with order field
        """
        ...
```

### Recognizer Protocol

All recognizers must implement this interface:

```python
class Recognizer(Protocol):
    """Protocol for text recognizers."""

    def process_blocks(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
    ) -> list[Block]:
        """Extract text from blocks.

        Args:
            image: Full page image
            blocks: Blocks to process

        Returns:
            Blocks with text field
        """
        ...

    def correct_text(self, text: str) -> str | dict[str, Any]:
        """Correct extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Corrected text or dict with metadata
        """
        ...
```

## Page and Document

### Page

```python
@dataclass
class Page:
    """Processed page with all metadata."""

    page_num: int
    width: int
    height: int
    blocks: list[Block]
    text: str                      # Rendered text
    corrected_text: str | None     # Page-level corrected
    correction_ratio: float | None
    processing_time_seconds: float
    processed_at: str              # ISO timestamp
    status: str                    # "complete", "partial", "error"
```

### Document

```python
@dataclass
class Document:
    """Complete document with all pages."""

    name: str
    path: Path
    num_pages: int
    processed_pages: int
    pages: list[Page]
    output_directory: Path
    processed_at: str
    status_summary: dict[str, int]
```

## See Also

- [BBox Formats Guide](../guides/bbox-formats.md) - Detailed format conversion examples
- [Detector Block Types](../guides/detector-block-types.md) - Block type mappings per detector
