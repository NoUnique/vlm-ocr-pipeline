# Detection API

Factory function and interface for layout detectors.

## Overview

The detection module provides:

- **Factory function**: `create_detector()` for creating detector instances
- **Protocol interface**: `Detector` protocol for implementing custom detectors
- **Built-in detectors**: DocLayout-YOLO, PaddleOCR PP-DocLayoutV2, MinerU VLM

## Quick Start

```python
from pipeline.layout.detection import create_detector
import numpy as np

# Create detector
detector = create_detector("doclayout-yolo")

# Detect layout blocks
image = np.zeros((1000, 800, 3), dtype=np.uint8)  # Your image
blocks = detector.detect(image)

for block in blocks:
    print(f"Type: {block.type}, BBox: {block.bbox}, Confidence: {block.detection_confidence}")
```

## Factory Function

### `create_detector`

```python
def create_detector(
    name: str,
    model_path: str | None = None,
    confidence_threshold: float = 0.5,
    auto_batch_size: bool = False,
    batch_size: int | None = None,
    target_memory_fraction: float = 0.8,
    backend: str | None = None,
    **kwargs,
) -> Detector:
    """Create a detector instance.

    Args:
        name: Detector name ("doclayout-yolo", "paddleocr-doclayout-v2", etc.)
        model_path: Custom model path (optional)
        confidence_threshold: Detection threshold (0.0-1.0)
        auto_batch_size: Enable automatic batch size optimization
        batch_size: Manual batch size
        target_memory_fraction: Target GPU memory usage for auto batch
        backend: Inference backend ("pytorch", "hf", "vllm")
        **kwargs: Additional detector-specific arguments

    Returns:
        Detector instance

    Raises:
        ValueError: If detector name is unknown
    """
```

## Available Detectors

| Detector | Description | Speed | Block Types |
|----------|-------------|-------|-------------|
| `doclayout-yolo` | This project's DocLayout-YOLO | Fast | 7 types |
| `mineru-doclayout-yolo` | MinerU's DocLayout-YOLO | Fast | 10 types |
| `paddleocr-doclayout-v2` | PaddleOCR PP-DocLayoutV2 | Medium | 25 types |
| `mineru-vlm` | MinerU VLM-based detection | Slow | 25+ types |

### DocLayout-YOLO

Fast YOLO-based detector for common document layouts.

**Block Types**: title, plain text, figure, table, equation, list, text

```python
detector = create_detector(
    "doclayout-yolo",
    model_path="path/to/model.pt",  # Optional custom model
    confidence_threshold=0.5,
)
```

### PaddleOCR PP-DocLayoutV2

High-quality detector with 25 block type categories and pointer network-based ordering.

**Block Types**: doc_title, paragraph_title, text, table, image, formula, algorithm, and more

```python
detector = create_detector(
    "paddleocr-doclayout-v2",
    confidence_threshold=0.5,
)
```

### MinerU VLM

VLM-based detector for complex document understanding.

**Block Types**: Full MinerU 2.5 block type hierarchy (25+ types)

```python
detector = create_detector(
    "mineru-vlm",
    model_name="opendatalab/MinerU2.5-2509-1.2B",
    backend="hf",  # or "vllm"
)
```

## Detector Protocol

All detectors implement the `Detector` protocol:

```python
from typing import Protocol
import numpy as np
from pipeline.types import Block

class Detector(Protocol):
    """Protocol for layout detectors."""

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in an image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of detected blocks with bounding boxes, types, and confidence scores
        """
        ...
```

## Implementing Custom Detectors

```python
from pipeline.types import Block, BBox
import numpy as np

class MyDetector:
    """Custom detector implementation."""

    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks."""
        results = self.model.predict(image)

        blocks = []
        for r in results:
            block = Block(
                type=r.class_name,
                bbox=BBox.from_xyxy(*r.bbox),
                detection_confidence=r.confidence,
                source="my-detector",
            )
            blocks.append(block)

        return blocks
```

## CLI Usage

```bash
# Default detector
python main.py --input doc.pdf

# Specific detector
python main.py --input doc.pdf --detector doclayout-yolo

# With backend
python main.py --input doc.pdf --detector mineru-vlm --detector-backend vllm

# With confidence threshold
python main.py --input doc.pdf --detector doclayout-yolo --confidence 0.7
```

## See Also

- [Detectors Architecture](../architecture/detectors.md) - Detailed detector comparison
- [Detector Block Types](../guides/detector-block-types.md) - Block type mappings
- [Types API](types.md) - BBox and Block classes
