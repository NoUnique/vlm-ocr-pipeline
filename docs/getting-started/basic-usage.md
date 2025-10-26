# Basic Usage

Learn the core features and command-line options of VLM OCR Pipeline.

## Command-Line Interface

The main entry point is `main.py`, which provides a comprehensive CLI.

### Basic Syntax

```bash
python main.py [OPTIONS]
```

## Input Options

### Single File

```bash
# Process a PDF
python main.py --input document.pdf --backend gemini

# Process an image
python main.py --input photo.jpg --backend gemini
```

### Batch Processing

```bash
# Process all PDFs in a directory
python main.py --input documents/ --backend gemini
```

### Page Limiting

Control which pages to process:

=== "Max Pages"

    Process first N pages only:
    ```bash
    python main.py --input doc.pdf --backend gemini --max-pages 5
    ```

=== "Page Range"

    Process a specific range (inclusive):
    ```bash
    python main.py --input doc.pdf --backend gemini --page-range 10-20
    ```

=== "Specific Pages"

    Process selected pages:
    ```bash
    python main.py --input doc.pdf --backend gemini --pages 1,5,10,15
    ```

## Backend Selection

### Cloud VLM APIs

=== "Gemini"

    Google's Gemini API (free tier available):
    ```bash
    export GEMINI_API_KEY="your_key"
    python main.py --input doc.pdf --backend gemini --model gemini-2.5-flash
    ```

    **Tier Options**: `free`, `tier1`, `tier2`, `tier3`
    ```bash
    python main.py --input doc.pdf --backend gemini --gemini-tier free
    ```

=== "OpenAI"

    OpenAI's GPT-4 Vision:
    ```bash
    export OPENAI_API_KEY="your_key"
    python main.py --input doc.pdf --backend openai --model gpt-4o
    ```

=== "OpenRouter"

    Access multiple VLMs through OpenRouter:
    ```bash
    export OPENROUTER_API_KEY="your_key"
    python main.py --input doc.pdf --backend openai --model google/gemini-2.5-flash
    ```

### Local Recognition

PaddleOCR-VL (no API required):
```bash
python main.py --input doc.pdf --recognizer paddleocr-vl
```

## Detector Selection

Choose the layout detection model:

| Detector | Source | Speed | Quality | Use Case |
|----------|--------|-------|---------|----------|
| `doclayout-yolo` | This project | ⚡⚡⚡ | ⭐⭐⭐ | Default, fast |
| `mineru-doclayout-yolo` | MinerU | ⚡⚡ | ⭐⭐⭐ | MinerU pipeline |
| `paddleocr-doclayout-v2` | PaddleOCR | ⚡⚡ | ⭐⭐⭐⭐ | High quality |
| `mineru-vlm` | MinerU | ⚡ | ⭐⭐⭐⭐ | VLM-based |
| `olmocr-vlm` | olmOCR | ⚡ | ⭐⭐⭐⭐ | VLM-based |

**Examples**:

```bash
# Default detector (doclayout-yolo)
python main.py --input doc.pdf --backend gemini

# High-quality detector
python main.py --input doc.pdf --detector paddleocr-doclayout-v2 --backend gemini

# VLM-based detection
python main.py --input doc.pdf --detector mineru-vlm --backend gemini
```

## Sorter Selection

Choose the reading order algorithm:

| Sorter | Algorithm | Speed | Multi-Column | Use Case |
|--------|-----------|-------|--------------|----------|
| `pymupdf` | Font analysis | ⚡⚡⚡ | ✅ | Multi-column docs |
| `mineru-xycut` | XY-Cut | ⚡⚡⚡ | ❌ | Simple layouts |
| `mineru-layoutreader` | LayoutLMv3 | ⚡⚡ | ✅ | Complex layouts |
| `mineru-vlm` | VLM reasoning | ⚡ | ✅ | Very complex |
| `olmocr-vlm` | VLM reasoning | ⚡ | ✅ | Research papers |
| `paddleocr-doclayout-v2` | Pointer network | ⚡⚡ | ✅ | With PP-DocLayoutV2 |

**Examples**:

```bash
# Multi-column documents
python main.py --input doc.pdf --sorter pymupdf --backend gemini

# Complex academic papers
python main.py --input paper.pdf --sorter mineru-layoutreader --backend gemini

# VLM-based ordering
python main.py --input doc.pdf --sorter olmocr-vlm --backend gemini
```

## Detector + Sorter Combinations

Not all combinations are valid. The pipeline validates compatibility:

### Recommended Combinations

```bash
# Fast general-purpose
python main.py --input doc.pdf \
    --detector doclayout-yolo \
    --sorter mineru-xycut \
    --backend gemini

# High quality multi-column
python main.py --input doc.pdf \
    --detector paddleocr-doclayout-v2 \
    --sorter pymupdf \
    --backend gemini

# Maximum quality (slower)
python main.py --input doc.pdf \
    --detector mineru-vlm \
    --sorter mineru-vlm \
    --backend gemini

# Full PaddleOCR pipeline (local)
python main.py --input doc.pdf \
    --detector paddleocr-doclayout-v2 \
    --recognizer paddleocr-vl
```

!!! warning "Invalid Combinations"
    - `paddleocr-doclayout-v2` detector auto-selects its sorter (cannot override)
    - VLM detectors (`mineru-vlm`, `olmocr-vlm`) require matching VLM sorters

## Output Options

### Output Directory

```bash
# Custom output directory
python main.py --input doc.pdf --backend gemini --output results/
```

**Default**: `output/{model}/{filename}/`

Example: `output/gemini-2.5-flash/document/page_1.json`

### Cache Control

```bash
# Disable caching
python main.py --input doc.pdf --backend gemini --no-cache

# Custom cache directory
python main.py --input doc.pdf --backend gemini --cache-dir .my-cache/
```

## Rate Limiting (Gemini)

### Check Status

```bash
python main.py --rate-limit-status --backend gemini --gemini-tier free
```

**Output**:
```
=== Gemini API Rate Limit Status ===
Tier: free
Model: gemini-2.5-flash

Current Limits:
  RPM (Requests Per Minute): 2 / 15 (13.3%)
  TPM (Tokens Per Minute): 45,234 / 1,500,000 (3.0%)
  RPD (Requests Per Day): 156 / 1,500 (10.4%)
```

### Tier Configuration

```bash
# Free tier (default)
python main.py --input doc.pdf --backend gemini --gemini-tier free

# Paid tiers (higher limits)
python main.py --input doc.pdf --backend gemini --gemini-tier tier1
python main.py --input doc.pdf --backend gemini --gemini-tier tier2
```

## Advanced Options

### DPI Settings

For PDF rendering quality:

```bash
# Higher DPI = better quality, larger images
python main.py --input doc.pdf --backend gemini --dpi 300  # Default: 200
```

### Temporary Files

```bash
# Custom temp directory
python main.py --input doc.pdf --backend gemini --temp-dir /tmp/ocr/
```

### Logging

```bash
# Verbose output
python main.py --input doc.pdf --backend gemini -v

# Very verbose (debug level)
python main.py --input doc.pdf --backend gemini -vv
```

## Common Workflows

### Academic Papers

```bash
# High-quality processing for research papers
python main.py --input paper.pdf \
    --detector paddleocr-doclayout-v2 \
    --sorter mineru-layoutreader \
    --backend gemini \
    --dpi 300
```

### Multi-Column Magazines

```bash
# Multi-column layout detection
python main.py --input magazine.pdf \
    --detector doclayout-yolo \
    --sorter pymupdf \
    --backend gemini
```

### Large Batch Processing

```bash
# Process many PDFs with local model
python main.py --input documents/ \
    --detector paddleocr-doclayout-v2 \
    --recognizer paddleocr-vl \
    --max-pages 10  # Limit for testing
```

### Cost-Optimized Processing

```bash
# Use Gemini free tier + caching
python main.py --input doc.pdf \
    --backend gemini \
    --gemini-tier free \
    --cache-dir .cache/
```

## Output Structure

After processing, you'll find:

```
output/
└── {model}/              # e.g., gemini-2.5-flash/
    └── {document}/       # e.g., research_paper/
        ├── page_1.json   # Detailed page data
        ├── page_1.md     # Markdown output
        ├── page_2.json
        ├── page_2.md
        └── {document}_summary.json  # Processing metadata
```

### JSON Structure

```json
{
  "page_num": 1,
  "image_size": [1650, 2200],
  "text": "# Title\n\nBody text...",
  "corrected_text": "# Title\n\nBody text...",
  "correction_ratio": 0.02,
  "processing_stopped": false,
  "blocks": [
    {
      "type": "title",
      "bbox": [100, 50, 500, 120],
      "detection_confidence": 0.95,
      "order": 0,
      "column_index": null,
      "text": "Title",
      "corrected_text": "Title",
      "source": "doclayout-yolo"
    }
  ],
  "auxiliary_info": {
    "text_spans": [...]  # Font metadata for markdown
  }
}
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) - Understand the pipeline
- [Advanced Examples](../guides/advanced-examples.md) - Complex use cases
- [API Reference](../api/pipeline.md) - Programmatic usage
