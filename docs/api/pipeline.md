# Pipeline API

Main pipeline class for document processing.

## Overview

The `Pipeline` class orchestrates the entire 8-stage document processing workflow:

1. **Input**: Load documents and extract auxiliary information
2. **Detection**: Detect layout blocks using selected detector
3. **Ordering**: Analyze reading order using selected sorter
4. **Recognition**: Extract text from blocks using VLM or local model
5. **Block Correction**: Block-level text correction (optional)
6. **Rendering**: Convert to Markdown or plaintext
7. **Page Correction**: Page-level VLM correction (optional)
8. **Output**: Save results and generate summaries

## Quick Start

```python
from pathlib import Path
from pipeline import Pipeline
from pipeline.config import PipelineConfig

# Create pipeline with configuration
config = PipelineConfig(
    detector="paddleocr-doclayout-v2",
    sorter="mineru-xycut",
    recognizer="gemini-2.5-flash",
)
pipeline = Pipeline(config=config)

# Process PDF
document = pipeline.process_pdf(Path("document.pdf"))

# Process single image
result = pipeline.process_single_image(Path("image.jpg"))

# Process directory
summary = pipeline.process_directory(Path("pdfs/"), "output/")
```

## Pipeline Class

### Constructor

```python
class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        """Initialize VLM OCR processing pipeline.

        Args:
            config: Pipeline configuration object. If None, uses default configuration.
        """
```

### Key Methods

#### `process_pdf`

```python
def process_pdf(
    self,
    pdf_path: Path,
    max_pages: int | None = None,
    page_range: tuple[int, int] | None = None,
    pages: list[int] | None = None,
) -> Document:
    """Process PDF file with page limiting options.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to process
        page_range: Range of pages to process (start, end)
        pages: Specific list of page numbers to process

    Returns:
        Document object with full processing results
    """
```

#### `process_image`

```python
def process_image(
    self,
    image_path: str | Path,
    max_pages: int | None = None,
    page_range: tuple[int, int] | None = None,
    pages: list[int] | None = None,
) -> Document | dict[str, Any]:
    """Process single image or PDF.

    Args:
        image_path: Path to image or PDF file
        max_pages: Maximum number of pages to process (PDF only)
        page_range: Range of pages to process (PDF only)
        pages: Specific pages to process (PDF only)

    Returns:
        Document for PDF, dict for single image
    """
```

#### `process_single_image`

```python
def process_single_image(self, image_path: Path) -> dict[str, Any]:
    """Process a single image file.

    Args:
        image_path: Path to image file

    Returns:
        Processing results with blocks and extracted text
    """
```

#### `process_directory`

```python
def process_directory(
    self,
    input_dir: Path,
    output_dir: str,
    max_pages: int | None = None,
    page_range: tuple[int, int] | None = None,
    specific_pages: list[int] | None = None,
) -> dict[str, Any]:
    """Process all PDFs in a directory using staged batch processing.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Output directory
        max_pages: Maximum pages per PDF to process
        page_range: Page range per PDF to process
        specific_pages: Specific pages per PDF to process

    Returns:
        Processing summary dictionary
    """
```

## PipelineConfig

Configuration class for the pipeline.

### Constructor

```python
@dataclass
class PipelineConfig:
    # Detector settings
    detector: str = "paddleocr-doclayout-v2"
    detector_backend: str | None = None
    detector_model_path: str | None = None
    confidence_threshold: float = 0.5

    # Sorter settings
    sorter: str | None = None  # Auto-selected based on detector
    sorter_backend: str | None = None

    # Recognizer settings
    recognizer: str = "gemini-2.5-flash"
    recognizer_backend: str | None = None

    # DPI settings
    dpi: int | None = None
    detection_dpi: int | None = 150
    recognition_dpi: int | None = 300
    use_dual_resolution: bool = True

    # Correction settings
    enable_block_correction: bool = False
    enable_page_correction: bool = False

    # Output settings
    output_dir: Path = Path("output")
    renderer: str = "block-type"

    # Caching
    use_cache: bool = True
    cache_dir: Path = Path(".cache")

    # Batch processing
    auto_batch_size: bool = False
    batch_size: int | None = None
```

### Example Usage

```python
from pipeline.config import PipelineConfig

# Minimal configuration
config = PipelineConfig(
    recognizer="gemini-2.5-flash",
)

# Full configuration
config = PipelineConfig(
    detector="paddleocr-doclayout-v2",
    sorter="mineru-xycut",
    recognizer="paddleocr-vl",
    recognizer_backend="vllm",
    detection_dpi=150,
    recognition_dpi=300,
    enable_page_correction=True,
    output_dir=Path("results"),
)
```

## See Also

- [Architecture Overview](../architecture/overview.md) - Understanding the pipeline stages
- [Detectors](../architecture/detectors.md) - Available detection models
- [Sorters](../architecture/sorters.md) - Reading order algorithms
- [Recognizers](../architecture/recognizers.md) - Text extraction backends
