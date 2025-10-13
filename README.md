# VLM OCR Pipeline

A unified OCR processing pipeline that leverages Vision Language Models (VLMs) for document layout detection, text extraction, and AI-powered text correction. This system processes images and PDFs locally using multiple VLM backends (OpenAI/OpenRouter, Gemini).

> **Based on**: This project is based on and modified from [Versatile-OCR-Program](https://github.com/ses4255/Versatile-OCR-Program)

## Features

- **Document Layout Detection**: Automatically detects text, tables, figures, and other elements using DocLayout-YOLO
- **Multi-VLM Backend Support**: Support for OpenAI, OpenRouter, and Gemini VLM APIs for text extraction and processing
- **Modular Detection & Ordering**: Flexible detector and sorter combinations (DocLayout-YOLO, MinerU, olmOCR)
- **Advanced Ordering Algorithms**: Support for multi-column, LayoutReader (LayoutLMv3), XY-Cut, and VLM-based ordering
- **Unified BBox System**: Automatic conversion between 6+ different bbox formats (YOLO, MinerU, PyMuPDF, PyPDF, olmOCR)
- **VLM-Powered Text Extraction**: Advanced text extraction using Vision Language Models with intelligent context understanding
- **Multi-Language Support**: Supports English, Korean, and Japanese text extraction
- **AI-Powered Correction**: Intelligent text correction and content analysis
- **Special Content Processing**: Enhanced analysis of tables and figures with structured output
- **Model-Specific Prompts**: YAML-based prompt templates organized by model family for optimal results
- **Local Processing**: All processing runs locally with results stored on your filesystem
- **Caching System**: Intelligent caching to avoid reprocessing identical content
- **Flexible Input**: Supports single images, PDFs, or batch processing of directories

## Project Structure

```
vlm-ocr-pipeline/
├── main.py                     # CLI entry point
├── pipeline/                   # Modular VLM OCR Pipeline
│   ├── __init__.py            # Main Pipeline class
│   ├── types.py               # Unified BBox and Region types
│   ├── constants.py
│   ├── misc.py
│   ├── prompt.py
│   │
│   ├── layout/
│   │   ├── detection/         # Layout detection strategies
│   │   │   ├── __init__.py    # create_detector()
│   │   │   ├── doclayout_yolo.py  # This project's DocLayout-YOLO
│   │   │   └── mineru/        # MinerU detectors
│   │   │       ├── doclayout_yolo.py  # MinerU's DocLayout-YOLO
│   │   │       └── vlm.py     # MinerU VLM
│   │   │
│   │   └── ordering/          # Reading order strategies
│   │       ├── __init__.py    # create_sorter(), validate_combination()
│   │       ├── pymupdf/       # PyMuPDF sorters
│   │       │   └── multi_column.py  # Multi-column detection & sorting
│   │       ├── mineru/        # MinerU sorters
│   │       │   ├── layoutreader.py  # LayoutLMv3
│   │       │   ├── xycut.py   # XY-Cut algorithm
│   │       │   └── vlm.py     # VLM ordering
│   │       └── olmocr/        # olmOCR sorters
│   │           └── vlm.py     # VLM ordering
│   │
│   ├── conversion/            # PDF/Image conversion
│   │   └── converter.py
│   │
│   └── recognition/           # Text recognition and correction
│       ├── __init__.py        # TextRecognizer
│       ├── cache.py
│       └── api/               # VLM API clients (OpenAI, Gemini)
│
├── models/
│   └── doclayout_yolo.py      # DocLayout-YOLO wrapper
│
├── external/                  # External frameworks (git submodules)
│   ├── MinerU/                # MinerU 2.5
│   └── olmocr/                # olmOCR
│
├── settings/
│   └── prompts/               # YAML prompt templates by model
│
├── tests/                     # Unit tests
├── requirements.txt
├── README.md                  # This file
├── BBOX_FORMATS.md            # BBox format reference
│
├── .tmp/                      # Temporary files (auto-created)
├── .cache/                    # Recognition cache (auto-created)
├── .logs/                     # Log files (auto-created)
└── output/                    # Processing results (auto-created)
```

## BBox Format Reference

This project integrates multiple frameworks (DocLayout-YOLO, MinerU, PyMuPDF, PyPDF, olmOCR), each using different bounding box formats. We provide a **unified BBox conversion system** that handles all formats automatically.

### **Coordinate Systems**

| Framework | Format | Coordinate Order | Origin | Example |
|-----------|--------|------------------|--------|---------|
| **Current Project** | `[x, y, w, h]` | Top-Left + Size | Top-Left (0,0) | `[100, 50, 200, 150]` |
| **YOLO** | `[x1, y1, x2, y2]` | Top-Left + Bottom-Right | Top-Left (0,0) | `[100, 50, 300, 200]` |
| **MinerU** | `[x0, y0, x1, y1]` | Top-Left + Bottom-Right | Top-Left (0,0) | `[100, 50, 300, 200]` |
| **PyMuPDF** | `Rect(x0, y0, x1, y1)` | Top-Left + Bottom-Right | Top-Left (0,0) | `Rect(100, 50, 300, 200)` |
| **PyPDF** ⚠️ | `[x0, y0, x1, y1]` | **Bottom-Left + Top-Right** ⚠️ | **Bottom-Left (0,0)** ⚠️ | `[100, 592, 300, 742]` |
| **olmOCR** | `"[x, y]text"` | Text format | Top-Left (0,0) | `"[100x50]Chapter 1"` |

**Key Points:**
- ✅ Most frameworks use **Top-Left origin** (like images)
- ⚠️ **PyPDF uses Bottom-Left origin** (traditional PDF coordinate system)
- 📦 Our **BBox class** handles all conversions automatically

**Example Conversion:**
```python
from pipeline.types import BBox

# Current project → MinerU
bbox = BBox.from_xywh(100, 50, 200, 150)  # [x, y, w, h]
mineru = bbox.to_mineru_bbox()             # [100, 50, 300, 200]

# MinerU → olmOCR anchor
bbox = BBox.from_mineru_bbox([100, 50, 300, 200])
anchor = bbox.to_olmocr_anchor("image")    # "[Image 100x50 to 300x200]"

# PyPDF → Current (Y-axis flip!)
bbox = BBox.from_pypdf_rect([100, 592, 300, 742], page_height=792)
coords = bbox.to_list_xywh()               # [100, 50, 200, 150]
```

For detailed format specifications and conversion examples, see [BBOX_FORMATS.md](BBOX_FORMATS.md).

---

## Installation

### 1. Python Environment Setup

```bash
# Clone or download the project
cd vlm-ocr-pipeline

# Create virtual environment (recommended with Python 3.10 for best compatibility)
uv venv --python 3.10 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

 Run setup script to fix DocLayout-YOLO compatibility issues
python setup.py
```

### 2. Gemini API Setup (Required)

#### Step 1: Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key
5. Set environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### What are Vision Language Models (VLMs)?

Vision Language Models are advanced AI systems that can understand both visual and textual information simultaneously. Unlike traditional OCR that simply extracts text, VLMs can:

- **Understand context**: Analyze the relationship between text and visual elements
- **Intelligent text correction**: Fix OCR errors based on contextual understanding  
- **Content analysis**: Describe images, analyze tables, and extract meaningful insights
- **Multi-modal reasoning**: Combine visual and textual information for better results

Popular VLMs supported by this pipeline include:
- **GPT-4 Vision (OpenAI)**: Industry-leading multimodal capabilities
- **Gemini Vision (Google)**: Advanced visual understanding and reasoning
- **Claude Vision (Anthropic)**: Strong analytical and reasoning capabilities

### 3. OpenAI/OpenRouter API Setup

For OpenAI backend:

1. Visit [OpenAI API](https://platform.openai.com/api-keys)
2. Create an API key
3. Set environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

For OpenRouter backend (supports multiple models including Gemini):

1. Visit [OpenRouter](https://openrouter.ai/keys)
2. Create an API key
3. Set environment variable:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Choose your preferred backend (openai is default)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Custom OpenAI base URL (for OpenRouter or other compatible services)
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
```


## Usage

### Command Line Interface

#### Basic Usage

```bash
# Process a single PDF (uses OpenAI backend by default)
python main.py --input document.pdf

# Use Gemini backend specifically
python main.py --input document.pdf --backend gemini

# Use specific OpenAI model
python main.py --input document.pdf --model gpt-4o

# Use OpenRouter with Gemini model
python main.py --input document.pdf --model google/gemini-2.5-flash

# Process a directory of PDFs
python main.py --input /path/to/pdfs/

# Process a single image
python main.py --input image.jpg

# Specify custom output directory
python main.py --input document.pdf --output /custom/output/
```

#### Advanced Options

```bash
# Disable caching for fresh processing
python main.py --input document.pdf --no-cache

# Use custom prompts directory (overrides auto-detection)
python main.py --input document.pdf --prompts-dir custom_prompts/

# Backend and model combinations
python main.py --input document.pdf --backend openai --model gpt-4o-mini
python main.py --input document.pdf --backend gemini --model gemini-2.5-flash

# Page limiting options (mutually exclusive)
python main.py --input document.pdf --max-pages 5
python main.py --input document.pdf --page-range 10-20
python main.py --input document.pdf --pages 1,3,5,10,15

# Adjust detection confidence threshold
python main.py --input document.pdf --confidence 0.7

# Use custom model path
python main.py --input document.pdf --model-path /path/to/custom/model.pt

# Enable debug logging
python main.py --input document.pdf --log-level DEBUG

# Combined advanced usage
python main.py --input /docs/ --max-pages 3 --confidence 0.8
```

### Python API Usage

```python
from pipeline import Pipeline

# Initialize pipeline with default settings (Gemini API)
pipeline = Pipeline(
    confidence_threshold=0.5,
    use_cache=True,
    cache_dir=".cache",
    output_dir="output"
)

# Process a single image
result = pipeline.process_image("document.jpg")
print(f"Extracted text: {result['corrected_text']}")

# Process a PDF
result = pipeline.process_pdf("document.pdf")
print(f"Processed {result['num_pages']} pages")

# Process PDF with page limits
result = pipeline.process_pdf(
    "document.pdf", 
    max_pages=5  # Process only first 5 pages
)

result = pipeline.process_pdf(
    "document.pdf",
    page_range=(10, 20)  # Process pages 10-20
)

result = pipeline.process_pdf(
    "document.pdf",
    specific_pages=[1, 5, 10, 15]  # Process specific pages
)

# Process a directory
result = pipeline.process_directory("input_folder/")
print(f"Processed {result['total_pdfs']} PDF files")
```

## Output Format

### Single Image/Page Output

Each single image (or individual PDF page) is written as `page_<number>.json` under `<output>/<model>/<document_stem>/`. The payload includes:

- `image_path`: Path to the rendered page image (or original image if supplied)
- `width` / `height`: Pixel dimensions of the rendered page
- `regions`: Raw DocLayout-YOLO detections (bounding boxes and labels)
- `processed_regions`: Post-processed regions with extracted text, table summaries, etc.
- `raw_text`: Natural reading-order text composed from text-like regions
- `corrected_text`: Text after VLM correction (falls back to `raw_text` on failure)
- `correction_confidence`: Similarity score between raw and corrected text (0–1)
- `processing_time_seconds`: Total latency spent on the page
- `processed_at`: ISO-8601 timestamp for when processing completed

```json
{
  "image_path": "output/tmp/document_page_1.jpg",
  "width": 1920,
  "height": 1080,
  "regions": [...],
  "processed_regions": [...],
  "raw_text": "Original OCR text...",
  "corrected_text": "AI-corrected text...",
  "correction_confidence": 0.95,
  "processing_time_seconds": 12.34,
  "processed_at": "2024-12-19T10:30:00"
}
```

### PDF Summary Output

PDF runs emit a summary file alongside the page outputs: `summary.json` (all pages succeeded), `summary_partial.json` (some failures), or `summary_incomplete.json` (stopped early). The schema captures:

- `pdf_name` / `pdf_path`: Original filename and absolute path
- `num_pages`: Number of pages in the source PDF
- `processed_pages`: Count of pages processed (including fallbacks)
- `output_directory`: Folder that contains per-page and summary JSON artifacts
- `processed_at`: ISO-8601 timestamp for completion
- `status_summary`: Totals of `complete`, `partial`, and `incomplete` pages
- `pages`: Array of page status objects with optional file suffix (e.g., `partial` → `page_2_partial.json`)
- `processing_stopped`: Indicates an early stop due to rate limits or unexpected errors

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
    {"page": 1, "status": "complete", "file_suffix": ""},
    {"page": 2, "status": "partial", "file_suffix": "partial"}
  ],
  "processing_stopped": false
}
```

## Text Extraction

All text extraction is performed by the configured VLM backend (Gemini by default, or OpenAI/OpenRouter if selected). The model receives both rendered page images and prompt instructions tailored to the backend. Rate limiting and caching ensure the pipeline stays within API quotas while avoiding repeated work on identical regions.

## Page Limiting Options

For testing purposes, cost control, or processing specific sections, you can limit which pages to process:

### 1. Maximum Pages (`--max-pages`)
Process only the first N pages from the beginning:
```bash
python main.py --input document.pdf --max-pages 5
```

### 2. Page Range (`--page-range`)
Process a specific range of pages:
```bash
python main.py --input document.pdf --page-range 10-20
python main.py --input document.pdf --page-range 1-5
```

### 3. Specific Pages (`--pages`)
Process only specified pages (comma-separated):
```bash
python main.py --input document.pdf --pages 1,5,10,15
python main.py --input document.pdf --pages 3,7,12
```

### Important Notes
- **Mutually Exclusive**: You can only use one page limiting option at a time
- **1-Indexed**: Page numbers start from 1 (not 0)
- **Directory Mode**: When processing a directory, page limits apply to each PDF individually
- **Invalid Pages**: Invalid page numbers are automatically filtered out with warnings
- **Cost Savings**: Use `--max-pages 1` to test with just the first page before processing entire documents

### Use Cases
- **Testing**: `--max-pages 1` to test pipeline with single page
- **Cost Control**: `--max-pages 10` to limit API calls for large documents
- **Specific Sections**: `--page-range 5-15` to process only content pages
- **Table of Contents**: `--pages 1,2` to process only first few pages
- **Selective Processing**: `--pages 10,25,50` to process sample pages

## Prompt Management

### Model-Specific Prompt System

Prompts are organized by model family for optimal results. The system automatically selects the appropriate prompt directory based on the backend and model:

```
settings/prompts/
├── gemini/                   # Gemini-specific prompts
│   ├── text_extraction.yaml
│   ├── content_analysis.yaml
│   └── text_correction.yaml
├── openai/                   # OpenAI/GPT-specific prompts
│   ├── text_extraction.yaml
│   ├── content_analysis.yaml
│   └── text_correction.yaml
├── internvl/                 # InternVL-specific prompts
├── qwen/                     # Qwen-specific prompts
└── phi4/                     # Phi-4-specific prompts
```

### Automatic Prompt Selection

The system automatically detects the appropriate prompt directory:

- `--backend gemini` → `settings/prompts/gemini/`
- `--backend openai --model gpt-4o` → `settings/prompts/openai/`
- `--backend openai --model google/gemini-2.5-flash` → `settings/prompts/gemini/`
- `--model internvl/internvl2-5` → `settings/prompts/internvl/`

### Customizing Prompts

1. **Copy existing prompts**: `cp -r settings/prompts/gemini custom_prompts`
2. **Edit YAML files**: Modify prompts according to your needs
3. **Use custom prompts**: `python main.py --input doc.pdf --prompts-dir custom_prompts/`

### Prompt Structure

```yaml
# Example: settings/prompts/text_extraction.yaml
text_extraction:
  system: |
    You are an expert OCR system...
  user: |
    Please extract all text from this image...
  fallback: |
    Extract all visible text accurately...
```

### Fallback System
- **First**: Load prompts from YAML files
- **Second**: Use hardcoded fallback prompts
- **Error handling**: Graceful degradation if prompts are missing

## Special Content Processing

### Tables

Tables are automatically detected and processed with structured analysis:

```
[TableStart]

## Table Structure:
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| Data1   | Data2   | Data3   |

## Summary:
Brief description of table content

## Educational Significance:
Importance and context

## Related Topics:
Topic1, Topic2, Topic3

[TableEnd]
```

### Figures

Figures and images receive detailed analysis:

```
[FigureStart]

## Image Description:
Detailed description of visual content

## Educational Significance:
Educational importance

## Related Topics:
Related learning topics

## Exam Relevance:
How this could be used in exams

[FigureEnd]
```

## Free Tier Limits and Cost Management

### Gemini API
- **Free Tier**: Very generous limits
- **Rate Limits**: 15 requests/minute, 1,500 requests/day
- **Cost**: Free for most use cases
- **Optimization**: Enable caching to reduce API calls

### Best Practices for Cost Control

1. **Enable Caching**: Use `--cache` option (default) to avoid reprocessing
2. **Batch Processing**: Process multiple files in one session
3. **Monitor Usage**: Review rate-limit logs and provider dashboards regularly
4. **Set Billing Alerts**: Configure alerts with your API provider to avoid surprises

## Logging

The OCR Pipeline automatically creates detailed log files with timestamps for tracking and debugging.

### Log File Structure

- **Location**: `.logs/` (hidden directory, auto-created)
- **Format**: `YYYY-MM-DD_HH-MM-SS_ocr_pipeline.log`
- **Encoding**: UTF-8 with full Unicode support

### Example Log Files

```
.logs/
├── 2025-07-29_18-06-53_ocr_pipeline.log    # Rate limit status check
├── 2025-07-29_18-07-23_ocr_pipeline.log    # Full OCR processing run
└── 2025-07-29_19-15-42_ocr_pipeline.log    # Another processing session
```

### Log Levels

- **INFO**: General processing information and progress
- **DEBUG**: Detailed debugging information (use `--log-level DEBUG`)
- **WARNING**: Non-critical issues that don't stop processing
- **ERROR**: Critical errors that may stop processing

### Example Usage

```bash
# Standard logging (INFO level)
python main.py --input document.pdf

# Detailed debugging logs
python main.py --input document.pdf --log-level DEBUG

# Minimal logging (ERROR only)
python main.py --input document.pdf --log-level ERROR
```

### Log Cleanup

Log files are automatically organized by timestamp but not automatically deleted. You can:

```bash
# View recent logs
ls -la .logs/

# Remove old logs (older than 7 days)
find .logs/ -name "*.log" -mtime +7 -delete

# Archive logs by month
mkdir -p .logs/archive/2025-07/
mv .logs/2025-07-* .logs/archive/2025-07/
```

## Troubleshooting

### Common Issues

#### "No module named 'models'"
```bash
# Ensure you're in the project root directory
cd gemini_ocr
python main.py --input document.pdf
```

#### "Gemini API client not initialized"
```bash
# Verify environment variable
echo $GEMINI_API_KEY

# Or check .env file
cat .env
```

#### GPU/CUDA Issues
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# CPU-only operation is supported if GPU unavailable
```

### Performance Optimization

1. **Use SSD Storage**: For better I/O performance with large PDFs
2. **Increase RAM**: Processing large documents requires sufficient memory
3. **GPU Acceleration**: CUDA-compatible GPU improves DocLayout-YOLO performance
4. **Batch Size**: Process multiple files in sequence for better resource utilization

## Development

### Architecture

The pipeline follows a modular design:

1. **Document Layout Detection**: Uses DocLayout-YOLO to identify regions
2. **Text Extraction**: Gemini or OpenAI/OpenRouter VLM APIs for OCR
3. **Content Analysis**: Gemini/OpenAI APIs for tables and figures
4. **Text Correction**: AI-powered post-processing with the configured VLM
5. **Local Storage**: All results saved locally

### Extending the Pipeline

#### Adding New Region Types

```python
# In ocr_pipeline.py
def _get_gemini_prompt_for_region_type(self, region_type: str) -> str:
    if region_type == 'new_type':
        return "Custom prompt for new region type..."
    # ... existing code
```

#### Custom Post-Processing

```python
# Inherit from Pipeline
class CustomPipeline(Pipeline):
    def _correct_text_with_gemini(self, text: str) -> Dict[str, Any]:
        # Custom correction logic
        return super()._correct_text_with_gemini(text)
```

## License

This project is for educational and research purposes. Please ensure compliance with the respective API provider's terms of service and usage limits.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Verify API credentials and environment variables (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`)
3. Review terminal/log output for detailed error messages
4. Inspect the latest file in `.logs/`

---

**Note**: This system requires internet connectivity for API calls to the selected VLM provider. All processing results are stored locally for privacy and offline access. 
