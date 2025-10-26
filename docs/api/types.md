# Types API

Core type definitions and data structures.

## BBox

Unified bounding box representation with automatic format conversion.

```python
from pipeline.types import BBox

# Create from xyxy coordinates
bbox = BBox(x0=100, y0=50, x1=500, y1=200)

# Convert from different formats
bbox = BBox.from_yolo([0.5, 0.3, 0.4, 0.2], width=1000, height=800)
bbox = BBox.from_mineru([100, 50, 400, 150])  # xywh
bbox = BBox.from_pypdf_rect([100, 550, 500, 600], page_height=792)

# Export
xywh = bbox.to_xywh_list()  # [100, 50, 400, 150]
```

## Block

Represents a detected layout block with metadata.

```python
from pipeline.types import Block, BBox

block = Block(
    type="text",
    bbox=BBox(100, 50, 500, 200),
    detection_confidence=0.95,
    order=0,
    text="Extracted text",
    source="doclayout-yolo"
)
```

## Protocols

### Detector

```python
class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Block]:
        ...
```

### Sorter

```python
class Sorter(Protocol):
    def sort(self, blocks: list[Block], image: np.ndarray) -> list[Block]:
        ...
```

### Recognizer

```python
class Recognizer(Protocol):
    def process_blocks(self, image: np.ndarray, blocks: Sequence[Block]) -> list[Block]:
        ...

    def correct_text(self, text: str) -> str | dict[str, Any]:
        ...
```

!!! note "Full API Reference"
    Detailed API reference coming soon. See [BBox Formats](../guides/bbox-formats.md) for conversion examples.
