# Ordering API

Factory function and interface for reading order sorters.

## Factory Function

```python
from pipeline.layout.ordering import create_sorter, validate_combination

sorter = create_sorter("mineru-xycut")
validate_combination("doclayout-yolo", "mineru-xycut")  # Check compatibility
```

## Sorter Protocol

```python
from typing import Protocol
import numpy as np
from pipeline.types import Block

class Sorter(Protocol):
    def sort(self, blocks: list[Block], image: np.ndarray) -> list[Block]:
        """Sort blocks in reading order."""
        ...
```

!!! note "Full API Reference"
    Detailed API reference coming soon. See [Sorters](../architecture/sorters.md) for available sorters.
