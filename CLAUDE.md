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
# Basic usage (default: paddleocr-doclayout-v2 detector + paddleocr-vl recognizer, balanced DPI)
python main.py --input document.pdf

# Different recognizers (backend auto-selected)
python main.py --input document.pdf --recognizer deepseek-ocr  # → hf backend
python main.py --input document.pdf --recognizer gemini-2.5-flash  # → gemini backend
python main.py --input document.pdf --recognizer gpt-4o  # → openai backend
python main.py --input document.pdf --recognizer meta-llama/Llama-3-8b  # → openai backend (OpenRouter)

# Alternative local recognizers (backend auto-selected)
python main.py --input document.pdf --recognizer deepseek-ocr  # → hf backend

# Manual backend selection (optional, for advanced users)
python main.py --input doc.pdf --recognizer paddleocr-vl --recognizer-backend vllm
python main.py --input doc.pdf --recognizer deepseek-ocr --recognizer-backend vllm

# Full pipeline with custom backends
python main.py --input doc.pdf \
    --detector mineru-vlm --detector-backend vllm \
    --sorter mineru-vlm --sorter-backend vllm \
    --recognizer paddleocr-vl --recognizer-backend sglang

# Modular detector/sorter combinations
python main.py --input doc.pdf --detector doclayout-yolo --sorter mineru-xycut
python main.py --input doc.pdf --detector mineru-vlm --sorter mineru-vlm

# Alternative detectors and recognizers
python main.py --input doc.pdf --detector doclayout-yolo --recognizer gemini-2.5-flash
python main.py --input doc.pdf --detector paddleocr-doclayout-v2 --recognizer paddleocr-vl

# Page limiting (for testing/cost control)
python main.py --input doc.pdf --max-pages 5
python main.py --input doc.pdf --page-range 10-20
python main.py --input doc.pdf --pages 1,5,10

# DPI configuration (PDF-to-image conversion quality)
python main.py --input doc.pdf --dpi fast      # 150 DPI - fastest
python main.py --input doc.pdf --dpi balanced  # 150→300 dual (recommended)
python main.py --input doc.pdf --dpi quality   # 300 DPI - best quality
python main.py --input doc.pdf --dpi 200       # Custom single DPI
python main.py --input doc.pdf --dpi 150,300   # Custom dual (detection,recognition)

# Adaptive batch size (automatic GPU memory optimization)
python main.py --input doc.pdf --auto-batch-size  # Auto-calibrate optimal batch size
python main.py --input doc.pdf --batch-size 16    # Manual batch size
python main.py --input doc.pdf --auto-batch-size --target-memory-fraction 0.9  # Use 90% GPU memory

# Check rate limits (Gemini only)
python main.py --rate-limit-status --recognizer gemini-2.5-flash --gemini-tier free
```

### Documentation

```bash
# Build documentation locally
uv run mkdocs build

# Serve documentation locally (with live reload)
uv run mkdocs serve  # Visit http://127.0.0.1:8000

# Deploy to GitHub Pages (manual)
uv run mkdocs gh-deploy --force
```

**GitHub Pages Auto-Deployment**:
- Documentation is automatically built and deployed to GitHub Pages when changes to `docs/` or `mkdocs.yml` are pushed to `main`
- Workflow: `.github/workflows/docs.yml`
- Site URL: `https://nounique.github.io/vlm-ocr-pipeline/` (once enabled)

**Enable GitHub Pages**:
1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

### Benchmarking

```bash
# Benchmark pipeline performance
python scripts/benchmark.py --input document.pdf --max-pages 5

# Different detector/backend combinations
python scripts/benchmark.py --input doc.pdf --detector doclayout-yolo --backend gemini
python scripts/benchmark.py --input doc.pdf --detector mineru-vlm --backend openai

# Save results to JSON
python scripts/benchmark.py --input doc.pdf --max-pages 3 --output results.json

# With cache enabled
python scripts/benchmark.py --input doc.pdf --use-cache
```

### Pre-Commit Checklist

Before committing, run the pre-commit check script:

```bash
./scripts/pre-commit-check.sh
```

This script automatically runs:
1. ✅ **Code formatting** (`ruff format --check`)
2. ✅ **Linting** (`ruff check`)
3. ✅ **Type checking** (`npx pyright`)
4. ✅ **Documentation build** (`mkdocs build --strict`)

**Manual Pre-Commit Workflow**:
```bash
# 1. Format code
uv run ruff format .

# 2. Fix linting issues
uv run ruff check . --fix

# 3. Type check
npx pyright

# 4. Test documentation build
uv run mkdocs build --strict

# 5. Commit
git add .
git commit -m "your message"
git push
```

After push to `main`, GitHub Actions will automatically:
- Build documentation
- Deploy to GitHub Pages (if docs/ or mkdocs.yml changed)

### CI/CD Pipeline

The project uses GitHub Actions for automated testing and deployment.

**Workflows** (`.github/workflows/`):
- `ci.yml`: Main CI pipeline (lint, type check, test, build docs)
- `pr-checks.yml`: PR analysis and automated labeling
- `docs.yml`: Documentation deployment to GitHub Pages

**CI Checks on Every Push/PR**:
1. ✅ **Lint and Format** - ruff format & check
2. ✅ **Type Check** - pyright (Python 3.11)
3. ✅ **Tests** - pytest (Python 3.11 & 3.12)
4. ✅ **Documentation** - mkdocs build --strict

**PR Automation**:
- Automatic size labeling (XS/S/M/L/XL based on lines changed)
- Change analysis comment (Python/test/doc files count)
- Status checks required before merge

**View CI Status**:
```bash
# Check workflow runs
https://github.com/<user>/<repo>/actions

# Add status badge to README
[![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)](...)
```

**Local CI Simulation**:
```bash
# Run all checks locally (same as CI)
./scripts/pre-commit-check.sh

# Or run individually
uv run ruff format --check .  # Format check
uv run ruff check .           # Lint
npx pyright                   # Type check
uv run pytest                 # Tests
uv run mkdocs build --strict  # Docs
```

**Branch Protection** (Recommended):
- Require PR reviews
- Require status checks to pass:
  - Lint and Format Check
  - Type Check
  - Test (3.11 & 3.12)
  - Build Documentation
- Settings → Branches → Add rule for `main`

See [docs/CI_CD.md](docs/CI_CD.md) for detailed documentation.

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
   - Sorters: `pymupdf` (multi-column), `mineru-layoutreader` (LayoutLMv3), `mineru-xycut` (fast, default), `mineru-vlm`, `olmocr-vlm`, `paddleocr-doclayout-v2` (preserves pointer network ordering)
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

**Backend System**: Stage-specific inference backends for detectors, sorters, and recognizers
- Configured via `settings/models.yaml` - maps models to supported backends
- Validation: `pipeline/backend_validator.py` - validates backend compatibility
- **Available backends**:
  - `pytorch`: Native PyTorch (DocLayout-YOLO, PaddleOCR)
  - `hf`: HuggingFace Transformers single-GPU (MinerU, LayoutReader)
  - `pt-ray`: PyTorch + Ray multi-GPU
  - `hf-ray`: HuggingFace + Ray multi-GPU
  - `vllm`: vLLM inference engine (fast VLM inference)
  - `sglang`: SGLang inference engine (fast VLM inference)
  - `openai`: OpenAI API (GPT-4o, GPT-4 Turbo)
  - `gemini`: Google Gemini API (Gemini 2.5 Flash, 2.0 Pro)
- **Auto-selection**: Backends auto-selected based on model when not specified
- **Examples**:
  ```bash
  # Auto-select backends (recommended)
  python main.py --input doc.pdf --detector mineru-vlm --recognizer gpt-4o

  # Explicit backends for performance tuning
  python main.py --input doc.pdf \
      --detector mineru-vlm --detector-backend vllm \
      --sorter mineru-vlm --sorter-backend vllm \
      --recognizer paddleocr-vl --recognizer-backend sglang
  ```

### Important File Locations

**Core Pipeline**: `pipeline/__init__.py` - Main `Pipeline` class orchestrating all stages

**Type System**: `pipeline/types.py` - BBox, Region, Detector/Sorter protocols (read this first!)

**Backend Validation**: `pipeline/backend_validator.py` - Backend compatibility validation and resolution

**Detectors**:
- `pipeline/layout/detection/doclayout_yolo.py` - This project's DocLayout-YOLO
- `pipeline/layout/detection/mineru/` - MinerU detectors (DocLayout-YOLO, VLM)
- `pipeline/layout/detection/paddleocr/` - PaddleOCR PP-DocLayoutV2

**Sorters**:
- `pipeline/layout/ordering/pymupdf/multi_column.py` - Multi-column detection
- `pipeline/layout/ordering/mineru/` - LayoutReader, XY-Cut, VLM
- `pipeline/layout/ordering/olmocr/` - olmOCR VLM ordering
- `pipeline/layout/ordering/paddleocr/` - PP-DocLayoutV2 sorter (preserves pointer network ordering)

**Recognition**:
- `pipeline/recognition/__init__.py` - TextRecognizer with VLM backends (OpenAI, Gemini)
- `pipeline/recognition/paddleocr/` - PaddleOCR-VL Recognizer
  - `paddleocr_vl.py` - PaddleOCRVLRecognizer using PaddleOCR-VL-0.9B (0.9B params, NaViT + ERNIE-4.5-0.3B)
  - Requires PaddleX v3.3.1 in `external/PaddleX/`

**Conversion** (Stages 1 & 5):
- Input (Stage 1): `pipeline/conversion/input/` - PDF/image loading
- Output (Stage 5): `pipeline/conversion/output/markdown/` - Markdown generation (two strategies: region-based and font-based)

**Optimization**:
- `pipeline/optimization/batch_size.py` - Adaptive batch size calibration with GPU memory profiling
  - `BatchSizeCalibrator` - Binary search-based optimal batch size finder
  - `calibrate_batch_size()` - Convenience function for one-off calibration
  - `get_optimal_batch_size()` - Read cached batch size without calibration
  - Cache location: `~/.cache/vlm-ocr-pipeline/batch_size_cache.json`

**Prompt Management**: `settings/prompts/{gemini,openai,internvl,qwen,phi4}/` - Model-specific YAML templates

## Development Guidelines

### Refactoring Principles

**IMPORTANT: This project has not been officially released yet. Pre-1.0 development philosophy:**
- **No backward compatibility concerns** - This is pre-release software, breaking changes are acceptable
- **Aggressive refactoring** - Remove deprecated code completely, don't keep migration logic
- **Clean slate approach** - If a feature is removed or replaced, delete all related code without hesitation
- **Focus on correctness over compatibility** - Prioritize code quality and maintainability over preserving old APIs

When refactoring:
1. Remove deprecated arguments/functions completely (no migration logic, no warnings)
2. Update documentation immediately to reflect changes
3. Don't version-gate features - just implement the best solution
4. If unsure whether to keep something, remove it (can always restore from git if needed)

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

### PyTorch 2.6 Compatibility

**Issue**: PyTorch 2.6 changed the default `weights_only` parameter in `torch.load()` from `False` to `True`, breaking models with custom classes like `YOLOv10DetectionModel`.

**Solution**: The `setup.py` script automatically patches `doclayout_yolo/nn/tasks.py` to use `weights_only=False`:
```bash
python setup.py  # Applies both DocLayout-YOLO and PyTorch 2.6 patches
```

**Additional Requirements**:
- `dill==0.4.0` - Required for loading models serialized with dill (added to `requirements.txt`)

**Manual Patch** (if needed):
```bash
# Apply patch manually
sed -i 's/torch\.load(file, map_location="cpu")/torch.load(file, map_location="cpu", weights_only=False)/g' \
  .venv/lib/python3.11/site-packages/doclayout_yolo/nn/tasks.py

# Install dill
uv pip install dill
```

**How It Works**:
- `setup.py` includes `fix_pytorch26_compatibility()` function
- Automatically called during environment setup
- Patches `torch.load()` call at line 753 of `tasks.py`
- Enables loading of custom model classes safely

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
