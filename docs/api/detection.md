# Detection API

Factory function and interface for layout detectors.

## Factory Function

```python
from pipeline.layout.detection import create_detector

detector = create_detector("doclayout-yolo")
```

## Detector Protocol

```python
from typing import Protocol
import numpy as np
from pipeline.types import Block

class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in image."""
        ...
```

!!! note "Full API Reference"
    Detailed API reference coming soon. See [Detectors](../architecture/detectors.md) for available detectors.
