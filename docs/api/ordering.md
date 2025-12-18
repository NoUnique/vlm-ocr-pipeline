# Ordering API

Factory function and interface for reading order sorters.

## Overview

The ordering module provides:

- **Factory function**: `create_sorter()` for creating sorter instances
- **Validation**: `validate_combination()` for checking detector/sorter compatibility
- **Protocol interface**: `Sorter` protocol for implementing custom sorters
- **Built-in sorters**: PyMuPDF, XY-Cut, LayoutReader, VLM-based

## Quick Start

```python
from pipeline.layout.ordering import create_sorter, validate_combination
from pipeline.layout.detection import create_detector
import numpy as np

# Validate detector/sorter combination
validate_combination("doclayout-yolo", "mineru-xycut")

# Create components
detector = create_detector("doclayout-yolo")
sorter = create_sorter("mineru-xycut")

# Detect and sort
image = np.zeros((1000, 800, 3), dtype=np.uint8)
blocks = detector.detect(image)
sorted_blocks = sorter.sort(blocks, image)

for block in sorted_blocks:
    print(f"Order: {block.order}, Type: {block.type}")
```

## Factory Function

### `create_sorter`

```python
def create_sorter(
    name: str,
    backend: str | None = None,
    **kwargs,
) -> Sorter:
    """Create a sorter instance.

    Args:
        name: Sorter name ("pymupdf", "mineru-xycut", etc.)
        backend: Inference backend for VLM sorters
        **kwargs: Additional sorter-specific arguments

    Returns:
        Sorter instance

    Raises:
        ValueError: If sorter name is unknown
    """
```

### `validate_combination`

```python
def validate_combination(
    detector_name: str,
    sorter_name: str,
) -> None:
    """Validate detector/sorter combination.

    Args:
        detector_name: Name of detector
        sorter_name: Name of sorter

    Raises:
        ValueError: If combination is invalid
    """
```

## Available Sorters

| Sorter | Algorithm | Multi-Column | Speed | Notes |
|--------|-----------|--------------|-------|-------|
| `pymupdf` | Font analysis | Yes | Fast | Best for standard documents |
| `mineru-xycut` | XY-Cut | No | Fast | Simple and reliable |
| `mineru-layoutreader` | LayoutLMv3 | Yes | Medium | ML-based ordering |
| `mineru-vlm` | VLM reasoning | Yes | Slow | Requires mineru-vlm detector |
| `olmocr-vlm` | VLM reasoning | Yes | Slow | Flexible VLM ordering |
| `paddleocr-doclayout-v2` | Pointer network | Yes | Medium | Preserves detector ordering |

### PyMuPDF Sorter

Multi-column aware sorter using PyMuPDF font analysis.

```python
sorter = create_sorter("pymupdf")
sorted_blocks = sorter.sort(blocks, image, pymupdf_page=page)
```

### XY-Cut Sorter

Classic recursive XY-Cut algorithm for reading order.

```python
sorter = create_sorter("mineru-xycut")
sorted_blocks = sorter.sort(blocks, image)
```

### LayoutReader Sorter

LayoutLMv3-based reading order prediction.

```python
sorter = create_sorter("mineru-layoutreader")
sorted_blocks = sorter.sort(blocks, image)
```

### VLM Sorters

VLM-based sorters use visual reasoning for complex layouts.

```python
# olmOCR VLM (independent)
sorter = create_sorter("olmocr-vlm")

# MinerU VLM (requires mineru-vlm detector)
sorter = create_sorter("mineru-vlm")
```

## Sorter Protocol

All sorters implement the `Sorter` protocol:

```python
from typing import Protocol
import numpy as np
from pipeline.types import Block

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
            blocks: Detected blocks to sort
            image: Original page image for context
            **kwargs: Additional arguments (e.g., pymupdf_page)

        Returns:
            Blocks sorted by reading order with `order` field populated
        """
        ...
```

## Implementing Custom Sorters

```python
from pipeline.types import Block
import numpy as np

class MySorter:
    """Custom sorter implementation."""

    def sort(
        self,
        blocks: list[Block],
        image: np.ndarray,
        **kwargs,
    ) -> list[Block]:
        """Sort blocks by custom logic."""
        # Sort by y-coordinate then x-coordinate
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

        # Add order field
        for i, block in enumerate(sorted_blocks):
            block.order = i

        return sorted_blocks
```

## Detector/Sorter Compatibility

Some combinations have specific requirements:

| Detector | Compatible Sorters | Notes |
|----------|-------------------|-------|
| `doclayout-yolo` | All except `mineru-vlm` | Most flexible |
| `mineru-doclayout-yolo` | All except `mineru-vlm` | - |
| `paddleocr-doclayout-v2` | All | Best with `paddleocr-doclayout-v2` sorter |
| `mineru-vlm` | `mineru-vlm` only | Tightly coupled |

## CLI Usage

```bash
# Default sorter (auto-selected)
python main.py --input doc.pdf

# Specific sorter
python main.py --input doc.pdf --sorter mineru-xycut

# With backend
python main.py --input doc.pdf --sorter olmocr-vlm --sorter-backend vllm

# Detector + sorter combination
python main.py --input doc.pdf --detector doclayout-yolo --sorter pymupdf
```

## See Also

- [Sorters Architecture](../architecture/sorters.md) - Detailed sorter comparison
- [Detection API](detection.md) - Detector documentation
- [Types API](types.md) - Block class reference
