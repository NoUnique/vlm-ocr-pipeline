# Pipeline API

Main pipeline class for document processing.

## Overview

The `Pipeline` class orchestrates the entire 8-stage document processing workflow.

```python
from pathlib import Path
from pipeline import Pipeline

# Create pipeline
pipeline = Pipeline(
    detector_name="doclayout-yolo",
    sorter_name="mineru-xycut",
    backend="gemini",
    model="gemini-2.5-flash"
)

# Process PDF
result = pipeline.process_single_pdf(Path("document.pdf"))
```

## Key Methods

### `__init__`

Initialize the pipeline with detector, sorter, and recognizer configurations.

### `process_single_pdf`

Process a single PDF file and return results.

### `process_directory`

Batch process all PDFs in a directory.

### `process_page`

Process a single page (used internally).

!!! note "Full API Reference"
    Detailed API reference coming soon. See [Advanced Examples](../guides/advanced-examples.md) for usage patterns.
