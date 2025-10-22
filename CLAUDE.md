# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Guidelines

**IMPORTANT: Respond in Korean (한국어) when interacting with users.** Code, documentation, and technical terms should remain in English, but explanations and conversations should be in Korean for consistency and better understanding.

## Project Overview

VLM OCR Pipeline: A modular document processing system that combines layout detection (DocLayout-YOLO, MinerU, olmOCR) with Vision Language Models (OpenAI, Gemini) for intelligent text extraction and correction. The system processes PDFs and images through a four-stage pipeline: document conversion, layout detection, reading order analysis, and text recognition with VLM-powered correction.

## Commands

### Development
```bash
# Environment setup (uses uv for dependency management)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python setup.py  # Fix DocLayout-YOLO compatibility

# Type checking (use npx pyright, not pyright directly)
npx pyright

# Linting and formatting
uv run ruff check .
uv run ruff check . --fix

# Testing
uv run pytest
uv run pytest tests/test_types.py  # Single test file
uv run pytest tests/test_types.py::test_bbox_from_xywh  # Single test
```

### Running the Pipeline
```bash
# Basic usage
python main.py --input document.pdf --backend gemini
python main.py --input document.pdf --backend openai --model gpt-4o

# Modular detector/sorter combinations
python main.py --input doc.pdf --detector doclayout-yolo --sorter mineru-xycut
python main.py --input doc.pdf --detector mineru-vlm --sorter mineru-vlm \
    --mineru-model opendatalab/PDF-Extract-Kit-1.0

# PaddleOCR pipeline (PP-DocLayoutV2 detector + PaddleOCR-VL-0.9B recognizer)
# Note: paddleocr-doclayout-v2 detector auto-selects its sorter (preserves pointer network ordering)
python main.py --input doc.pdf --detector paddleocr-doclayout-v2 \
    --recognizer paddleocr-vl

# Page limiting (for testing/cost control)
python main.py --input doc.pdf --max-pages 5
python main.py --input doc.pdf --page-range 10-20
python main.py --input doc.pdf --pages 1,5,10

# Check rate limits (Gemini only)
python main.py --rate-limit-status --backend gemini --gemini-tier free
```

## Core Architecture

### 5-Stage Pipeline Design

The system follows a clear separation of concerns through five distinct stages:

1. **Document Conversion (Input)** (`pipeline/conversion/input/`)
   - Converts PDFs to images (pdf2image) or loads images directly
   - Extracts auxiliary info (text spans with font metadata) for markdown conversion
   - Functions: `render_pdf_page()`, `load_image()`, `extract_text_spans_from_pdf()`

2. **Layout Detection** (`pipeline/layout/detection/`)
   - Factory pattern: `create_detector(name, **kwargs)` in `__init__.py`
   - Returns `list[Block]` with detected bounding boxes
   - Detectors: `doclayout-yolo` (this project's YOLO), `mineru-doclayout-yolo`, `mineru-vlm`, `paddleocr-doclayout-v2` (PP-DocLayoutV2)
   - Protocol interface: `Detector.detect(image) -> list[Block]`

3. **Reading Order Analysis** (`pipeline/layout/ordering/`)
   - Factory pattern: `create_sorter(name, **kwargs)` in `__init__.py`
   - Adds `order` and optionally `column_index` to blocks
   - Sorters: `pymupdf` (multi-column), `mineru-layoutreader` (LayoutLMv3), `mineru-xycut` (fast, default), `mineru-vlm`, `olmocr-vlm`, `paddleocr-doclayout-v2` (passthrough for pointer network ordering)
   - Protocol interface: `Sorter.sort(blocks, image, **kwargs) -> list[Block]`
   - Combination validator: `validate_combination(detector, sorter)` ensures compatibility

4. **Text Recognition & Correction** (`pipeline/recognition/`)
   - VLM-powered text extraction per block: `TextRecognizer.process_blocks()`
   - Page-level correction: `TextRecognizer.correct_text()`
   - Multi-backend support (OpenAI/Gemini/PaddleOCR-VL) with prompt templates in `settings/prompts/{model}/`
   - **PaddleOCR-VL Recognizer**: Uses PaddleOCR-VL-0.9B model for block-level text recognition
     - **Architecture**: NaViT-style dynamic resolution vision encoder (SiglipVisionModel) + ERNIE-4.5-0.3B language model
     - **Model Size**: 0.9B parameters (compact but powerful)
     - **Features**: 109 languages support, SOTA on OmniDocBench v1.5 (text, formula, table, reading order)
     - **Query-based**: Different prompts for different block types ("OCR:", "Table Recognition:", "Formula Recognition:", "Chart Recognition:")
     - **Backends**: native (default), vllm-server, sglang-server
   - Intelligent caching system to avoid re-processing identical content

5. **Result Conversion (Output)** (`pipeline/conversion/output/`)
   - Converts processed blocks to various output formats
   - Currently implements Markdown conversion with two strategies (see below)
   - Extensible design for future formats (HTML, LaTeX, etc.)

### Key Design Patterns

**Unified BBox System**: All bounding boxes use the `BBox` dataclass from `pipeline/types.py`:
- Internal format: `BBox(x0, y0, x1, y1)` with **integer** coordinates (xyxy corners)
- JSON output: `[x, y, w, h]` (xywh format for human readability)
- Automatic conversion between 6+ formats (YOLO, MinerU, PyMuPDF, PyPDF, olmOCR)
- Only PyPDF uses bottom-left origin (requires Y-axis flip via `from_pypdf_rect()`)

**Block Dataclass**: Document blocks combine detection and processing results:
```python
@dataclass
class Block:
    type: str              # "text", "title", "table", "image", etc.
    bbox: BBox             # Required, integer coordinates
    detection_confidence: float | None  # Detection confidence
    order: int | None                   # Reading order rank (added by sorters)
    column_index: int | None            # Column index (added by multi-column sorters)
    text: str | None                    # Extracted text (added by recognizers)
    corrected_text: str | None          # VLM-corrected text (added by correction)
    correction_ratio: float | None      # Block-level correction ratio
    source: str | None                  # Detector name (internal use)
```

**Factory Pattern for Detectors/Sorters**:
- Central registration: `pipeline/layout/detection/__init__.py` and `pipeline/layout/ordering/__init__.py`
- Late imports: Lazy loading of optional dependencies (MinerU, olmOCR) using `PLC0415`
- Validation: `validate_combination()` ensures detector/sorter compatibility
- Examples in `tests/test_factory.py`, `tests/test_validator.py`

**Protocol Interfaces**: Type-safe plugin system via `Detector` and `Sorter` protocols in `pipeline/types.py`

### Important File Locations

**Core Pipeline**: `pipeline/__init__.py` - Main `Pipeline` class orchestrating all stages

**Type System**: `pipeline/types.py` - BBox, Region, Detector/Sorter protocols (read this first!)

**Detectors**:
- `pipeline/layout/detection/doclayout_yolo.py` - This project's DocLayout-YOLO
- `pipeline/layout/detection/mineru/` - MinerU detectors (DocLayout-YOLO, VLM)
- `pipeline/layout/detection/paddleocr/` - PaddleOCR PP-DocLayoutV2

**Sorters**:
- `pipeline/layout/ordering/pymupdf/multi_column.py` - Multi-column detection
- `pipeline/layout/ordering/mineru/` - LayoutReader, XY-Cut, VLM
- `pipeline/layout/ordering/olmocr/` - olmOCR VLM ordering
- `pipeline/layout/ordering/paddleocr/` - PP-DocLayoutV2 passthrough (preserves pointer network ordering)

**Recognition**:
- `pipeline/recognition/__init__.py` - TextRecognizer with VLM backends (OpenAI, Gemini)
- `pipeline/recognition/paddleocr/` - PaddleOCR-VL Recognizer
  - `paddleocr_vl.py` - PaddleOCRVLRecognizer using PaddleOCR-VL-0.9B (0.9B params, NaViT + ERNIE-4.5-0.3B)
  - Requires PaddleX v3.3.1 in `external/PaddleX/`

**Conversion** (Stages 1 & 5):
- Input (Stage 1): `pipeline/conversion/input/` - PDF/image loading
- Output (Stage 5): `pipeline/conversion/output/markdown/` - Markdown generation (two strategies: region-based and font-based)

**Prompt Management**: `settings/prompts/{gemini,openai,internvl,qwen,phi4}/` - Model-specific YAML templates

## Development Guidelines

### Code Quality Standards
- **Type annotations**: Required for all functions/methods (use `typing` module)
- **Docstrings**: Google style for all public functions/classes
- **Line length**: 120 characters (ruff.toml)
- **Import order**: isort with first-party: ["pipeline", "models"]
- **Testing**: pytest with 90%+ coverage goal

### BBox Handling Rules
1. **Always use BBox class** - Never use raw lists/tuples for coordinates
2. **Internal operations use xyxy** - Access via `bbox.x0, bbox.y0, bbox.x1, bbox.y1`
3. **JSON serialization uses xywh** - Call `bbox.to_xywh_list()` or `region.to_dict()`
4. **Accept floats, output integers** - All BBox methods round to nearest integer
5. **PyPDF requires page height** - Use `BBox.from_pypdf_rect(rect, page_height)` for Y-flip

### Adding New Detectors/Sorters
1. Implement `Detector` or `Sorter` protocol from `pipeline/types.py`
2. Register in `create_detector()` or `create_sorter()` factory
3. Add validation rule in `validate_combination()` if needed
4. Use lazy imports (`PLC0415` exception in ruff.toml) for optional dependencies
5. Write tests in `tests/test_detectors.py` or `tests/test_sorters.py`

### Prompt Engineering
- Prompts live in `settings/prompts/{model}/` as YAML files
- Auto-detection: `--backend gemini` → `prompts/gemini/`, `--model google/gemini-2.5-flash` → `prompts/gemini/`
- Each YAML has `system`, `user`, and `fallback` keys
- Fallback system: YAML → hardcoded prompts → graceful degradation
- See `pipeline/prompt.py` for PromptManager implementation

### Testing Guidelines
- **Unit tests**: `tests/test_*.py` - Focus on BBox conversions, factory patterns, validators
- **OCR accuracy**: `tests/test_ocr_accuracy.py` - Levenshtein distance metrics
- **Fixtures**: `tests/fixtures/` - Sample images/PDFs (use `.gitkeep` for empty dirs)
- **Timezone handling**: Always use `pipeline.misc.tz_now()` for timestamps (includes tzinfo)
- **Magic numbers**: Allowed in test files (`PLR2004` exception)

### Environment & Dependencies
- **Python 3.11+** required (use `uv` for all package management)
- **Never use `pip` directly** - Always `uv pip install <package>`
- **After install**: Update `requirements.txt` with exact versions
- **Run commands**: Prefix with `uv run` (e.g., `uv run pytest`, `uv run python main.py`)
- **External frameworks**: `external/` contains git submodules - excluded from type checking
  - `external/PaddleOCR/` - PaddleOCR v3.3.0 (for PP-DocLayoutV2 detector)
  - `external/PaddleX/` - PaddleX v3.3.1 (for PaddleOCR-VL-0.9B recognizer)
  - `external/MinerU/` - MinerU framework (for detectors and sorters)
  - `external/olmOCR/` - olmOCR framework (for VLM sorter)
- **Type checking**: Use `npx pyright`, not global pyright

### Rate Limiting & Caching
- **Gemini tiers**: `free`, `tier1`, `tier2`, `tier3` - set via `--gemini-tier`
- **Rate limiter**: `pipeline/recognition/api/ratelimit.py` - auto-throttles requests
- **Cache system**: `pipeline/recognition/cache.py` - content-based hashing to avoid reprocessing
- **Check status**: `python main.py --rate-limit-status --backend gemini`

### Result Conversion: Markdown Strategies

The Result Conversion stage (Stage 5) currently implements two Markdown conversion strategies:

**1. Region Type-Based** (Default - `pipeline/conversion/output/markdown/__init__.py`):
- Fast, no PDF parsing required
- Maps region types (title, subtitle) to Markdown headers
- Use for quick conversion: `json_to_markdown(regions)`

**2. Font Size-Based** (`pipeline/conversion/output/markdown/pymupdf4llm.py`):
- Requires PyMuPDF text span parsing (stored in `auxiliary_info.text_spans` from Stage 1)
- Auto-detects headers from font sizes (largest → H1, 2nd largest → H2)
- Uses IoU matching to link text spans to regions
- **PyMuPDF terminology**: Uses `size` and `font` (not `font_size`, `font_name`)
- Use for precise header detection: `to_markdown(page_result, auto_detect_headers=True)`

### Common Pitfalls
- **Don't create empty `__init__.py`** - Use PEP 420 namespace packages (no `__init__.py` unless it has logic)
- **Don't use bare except** - Catch specific exceptions (see `main.py` for `# noqa: BLE001` exceptions)
- **Don't install with `-e`** - Never use editable mode from external directories
- **Don't mix xywh/xyxy** - Always convert via BBox methods
- **Don't forget page_height for PyPDF** - Y-axis flip required

## Reference Documentation

- **BBox Formats**: See `BBOX_FORMATS.md` for detailed conversion examples
- **Block Types**: See `DETECTOR_BLOCK_TYPES.md` for detector-specific type mappings
- **README.md**: User-facing documentation with installation, usage, API examples
- **Cursor Rules**: `.cursorrules` contains detailed coding standards (Google docstrings, type hints, testing requirements)
