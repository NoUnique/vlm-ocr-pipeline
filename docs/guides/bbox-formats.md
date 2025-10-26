# BBox Format Reference

## Coordinate Systems

All frameworks use **Top-Left origin (0,0)** except PyPDF.

### Format Comparison

| Framework | Format | Example | Notes |
|-----------|--------|---------|-------|
| **This Project (Internal)** | `BBox(x0, y0, x1, y1)` | `BBox(100, 50, 300, 200)` | Integer coordinates, xyxy corners |
| **This Project (JSON)** | `[x, y, w, h]` | `[100, 50, 200, 150]` | Position + Size (human-readable) |
| **YOLO** | `[x1, y1, x2, y2]` | `[100, 50, 300, 200]` | Top-Left + Bottom-Right |
| **MinerU** | `[x0, y0, x1, y1]` | `[100, 50, 300, 200]` | Top-Left + Bottom-Right |
| **PyMuPDF** | `Rect(x0, y0, x1, y1)` | `Rect(100, 50, 300, 200)` | Top-Left + Bottom-Right |
| **PyPDF** ⚠️ | `[x0, y0, x1, y1]` | `[100, 592, 300, 742]` | **Bottom-Left origin** |
| **olmOCR** | `"[x, y]text"` | `"[100x50]Chapter 1"` | Text format |

### Visual Example

Same rectangle at visual position (100, 50) with size 200×150 on page height 792:

```
Computer Vision (Top-Left origin):
┌─────────────────┐
│ (0,0)           │
│                 │
│   [100,50]      │ ← Region here
│   ┌────────┐    │
│   │        │    │
│   └────────┘    │
│          [300,200]
└─────────────────┘

PDF (Bottom-Left origin):
┌─────────────────┐
│          [300,742] ← Y is flipped!
│   ┌────────┐    │
│   │        │    │
│   [100,592]     │ ← Same region
│                 │
│                 │
└─────────────────┘
  (0,0)
```

## Format Conversions

### This Project ↔ MinerU/YOLO

```python
# [x, y, w, h] → [x0, y0, x1, y1]
x0, y0, x1, y1 = x, y, x + w, y + h

# [x0, y0, x1, y1] → [x, y, w, h]
x, y, w, h = x0, y0, x1 - x0, y1 - y0
```

### PyPDF ↔ This Project (Y-axis flip required!)

```python
# PyPDF → This Project
y_top = page_height - y1_pypdf       # 792 - 742 = 50
y_bottom = page_height - y0_pypdf    # 792 - 592 = 200

# This Project → PyPDF
y0_pypdf = page_height - y_bottom    # 792 - 200 = 592
y1_pypdf = page_height - y_top       # 792 - 50 = 742
```

### olmOCR Anchor Format

```python
# Text region
anchor = f"[{x:.0f}x{y:.0f}]{text_content}"
# Example: "[100x50]Chapter 1"

# Image/Figure region
anchor = f"[Image {x0:.0f}x{y0:.0f} to {x1:.0f}x{y1:.0f}]"
# Example: "[Image 100x50 to 300x200]"

# Table region
anchor = f"[Table {x0:.0f}x{y0:.0f} to {x1:.0f}x{y1:.0f}]"
# Example: "[Table 100x450 to 500x600]"
```

## BBox Class Usage

Our `BBox` class handles all conversions automatically with **integer coordinates**:

```python
from pipeline.types import BBox

# Create from any format (accepts float, converts to int)
bbox = BBox.from_xywh(100, 50, 200, 150)      # xywh format
bbox = BBox.from_xyxy(100, 50, 300, 200)      # xyxy format (corners)
bbox = BBox.from_cxcywh(200, 125, 200, 150)   # Center format (YOLO training)
bbox = BBox.from_mineru_bbox([100, 50, 300, 200])
bbox = BBox.from_pymupdf_rect(rect)
bbox = BBox.from_pypdf_rect([100, 592, 300, 742], page_height=792)

# Convert to any format
coords = bbox.to_xywh_list()          # [100, 50, 200, 150] (for JSON)
coords = bbox.to_list()                # [100, 50, 300, 200] (xyxy)
coords = bbox.to_dict()                # {"x0": 100, "y0": 50, "x1": 300, "y1": 200}
cx, cy, w, h = bbox.to_cxcywh()        # Center format
coords = bbox.to_mineru_bbox()         # [100, 50, 300, 200]
coords = bbox.to_pypdf_rect(792)       # [100, 592, 300, 742]
anchor = bbox.to_olmocr_anchor("image") # "[Image 100x50 to 300x200]"

# Geometric operations (integer results)
center_x, center_y = bbox.center  # (float, float)
area = bbox.area                  # int
width = bbox.width                # int
height = bbox.height              # int
overlap = bbox1.intersect(bbox2)  # int
iou = bbox1.iou(bbox2)            # float

# Clear aliases
left = bbox.left     # = bbox.x0
top = bbox.top       # = bbox.y0
right = bbox.right   # = bbox.x1
bottom = bbox.bottom # = bbox.y1

# NumPy convenience
cropped = bbox.crop(image, padding=5)  # Direct image cropping
```

## Region Structure

The `Region` dataclass represents detected document regions:

```python
from pipeline.types import Block, BBox

block = Block(
    type="text",
    bbox=BBox(100, 50, 300, 200),  # Required, always present
    detection_confidence=0.95,
    # Optional fields
    order=0,
    column_index=1,
    text="Extracted text...",
)

# Serialize to JSON (bbox → xywh format for readability)
data = block.to_dict()
# {"order": 0, "type": "text", "xywh": [100, 50, 200, 150], "detection_confidence": 0.95, ...}

# Deserialize from JSON (supports xywh)
block = Block.from_dict(data)
```

## Key Points

1. **Internal representation**: `BBox(x0, y0, x1, y1)` with **integer coordinates** (xyxy format)
2. **JSON output**: `[x, y, w, h]` (xywh format) for human readability
3. **Most frameworks use the same internal format**: `[x0, y0, x1, y1]` with Top-Left origin
4. **Only PyPDF is different**: Bottom-Left origin requires Y-axis flip
5. **Region.bbox is always present**: No longer optional (required field)

