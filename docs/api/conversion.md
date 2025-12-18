# Conversion API

PDF/Image loading and output format conversion utilities.

## Overview

The conversion module provides:

- **Input**: PDF and image loading with auxiliary information extraction
- **Output**: Markdown and JSON generation from processed blocks

## Input Conversion

### PDF Loading

```python
from pipeline.io.input import pdf

# Get PDF information
info = pdf.get_pdf_info(Path("document.pdf"))
print(f"Pages: {info['Pages']}")

# Determine pages to process
pages = pdf.determine_pages_to_process(
    total_pages=10,
    max_pages=5,        # Optional: limit pages
    page_range=(1, 5),  # Optional: range
    pages=[1, 3, 5],    # Optional: specific pages
)

# Render page to image
image = pdf.render_pdf_page(
    pdf_path=Path("document.pdf"),
    page_num=1,
    dpi=200,
)
```

### Image Loading

```python
from pipeline.io.input import image

# Load image as numpy array
img = image.load_image(Path("document.jpg"))
```

### InputLoader Class

The `InputLoader` class provides a high-level interface for loading documents:

```python
from pipeline.io import InputLoader
from pipeline.stages import InputStage

input_stage = InputStage(dpi=200)
loader = InputLoader(
    input_stage=input_stage,
    use_dual_resolution=True,
    detection_dpi=150,
    recognition_dpi=300,
)

# Load all pages from PDF
page_images, recognition_images, auxiliary_infos, _, _ = loader.load_page_images(
    pdf_path=Path("document.pdf"),
    pages_to_process=[1, 2, 3],
)
```

## Output Conversion

### Markdown Generation

Two strategies are available for Markdown conversion:

#### Block Type-Based (Default)

Fast conversion using block type classification:

```python
from pipeline.io.output.markdown import blocks_to_markdown

blocks = [
    Block(type="title", bbox=..., text="Document Title"),
    Block(type="text", bbox=..., text="Content here."),
]

markdown = blocks_to_markdown(blocks)
# # Document Title
#
# Content here.
```

**Block Type → Markdown Mapping:**

| Block Type | Markdown |
|------------|----------|
| `title` | `# Heading` |
| `text` | Paragraph |
| `table` | Table (if markdown content) |
| `image` | `![Image](path)` |
| `code` | Code block |
| `list` | List items |
| `equation` | Math block |

#### Font Size-Based (PyMuPDF4LLM Style)

Advanced conversion using font size information from PDF text spans:

```python
from pipeline.io.output.markdown.pymupdf4llm import to_markdown

# page_result contains auxiliary_info with text_spans
markdown = to_markdown(page_result, auto_detect_headers=True)
```

**Features:**

- Auto-detects headers from font sizes (largest → H1, 2nd largest → H2)
- Uses IoU matching to link text spans to blocks
- Preserves original PDF formatting

### JSON Output

The `OutputSaver` class handles JSON serialization:

```python
from pipeline.io import OutputSaver

saver = OutputSaver(
    detector_name="doclayout-yolo",
    sorter_name="mineru-xycut",
    backend="gemini",
    model="gemini-2.5-flash",
)

# Save intermediate results
saver.save_intermediate_results(
    pdf_path=pdf_path,
    pages_to_process=[1, 2, 3],
    page_output_dir=output_dir,
    detected_blocks=blocks_dict,
    stage="detection",
)

# Save final results
saver.save_final_results(result, output_path)
```

## Output Directory Structure

```
output/
└── {model}/
    └── {document}/
        ├── page_1.json           # Page data
        ├── page_1.md             # Markdown output
        ├── page_2.json
        ├── page_2.md
        └── summary.json          # Document summary
```

### Page JSON Format

```json
{
  "page_num": 1,
  "width": 1920,
  "height": 1080,
  "blocks": [
    {
      "order": 0,
      "type": "title",
      "xywh": [100, 50, 400, 80],
      "detection_confidence": 0.95,
      "text": "Document Title"
    }
  ],
  "text": "Rendered markdown text",
  "corrected_text": "VLM-corrected text",
  "correction_ratio": 0.05,
  "processing_time_seconds": 2.5,
  "processed_at": "2024-12-19T10:30:00"
}
```

### Summary JSON Format

```json
{
  "pdf_name": "document",
  "pdf_path": "/path/to/document.pdf",
  "num_pages": 10,
  "processed_pages": 10,
  "output_directory": "output/gemini-2.5-flash/document",
  "processed_at": "2024-12-19T10:30:00",
  "status_summary": {"complete": 10},
  "pages": [
    {"page": 1, "status": "complete", "file_suffix": ""}
  ]
}
```

## See Also

- [BBox Formats](../guides/bbox-formats.md) - Coordinate system reference
- [Pipeline API](pipeline.md) - Main pipeline class
- [Types API](types.md) - Data structures
