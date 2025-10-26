# Contributing Guide

Thank you for considering contributing to VLM OCR Pipeline! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/vlm-ocr-pipeline.git
cd vlm-ocr-pipeline
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install pytest pytest-cov ruff pyright

# Run setup script
python setup.py
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

## Code Quality Standards

### Type Annotations

All functions and methods must have type annotations:

```python
def process_blocks(
    self,
    image: np.ndarray,
    blocks: Sequence[Block]
) -> list[Block]:
    """Process blocks to extract text."""
    ...
```

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def detect_layout(image: np.ndarray, confidence_threshold: float = 0.5) -> list[Block]:
    """Detect layout blocks in an image.

    Args:
        image: Input image as numpy array (H, W, C)
        confidence_threshold: Minimum confidence score for detection

    Returns:
        List of detected blocks with bounding boxes

    Raises:
        DetectionError: If detection fails

    Example:
        >>> detector = DocLayoutYOLO()
        >>> blocks = detector.detect(image, confidence_threshold=0.7)
        >>> len(blocks)
        15
    """
```

### Code Style

We use **ruff** for linting and formatting:

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix
```

**Configuration** (ruff.toml):
- Line length: 120 characters
- Import order: isort compatible
- First-party modules: `["pipeline", "models"]`

### Type Checking

We use **pyright** for static type checking:

```bash
# Type check entire project
npx pyright

# Type check specific files
npx pyright pipeline/__init__.py
```

**Note**: Use `npx pyright`, not global `pyright`

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_types.py

# Run with coverage
uv run pytest --cov=pipeline --cov-report=term-missing

# Run verbose
uv run pytest -v
```

### Writing Tests

Place tests in `tests/` directory with naming convention `test_*.py`:

```python
def test_bbox_from_yolo():
    """Test BBox conversion from YOLO format."""
    bbox = BBox.from_yolo([0.5, 0.5, 0.3, 0.4], 1000, 800)
    assert bbox.x0 == 350
    assert bbox.y0 == 240
    assert bbox.x1 == 650
    assert bbox.y1 == 640
```

**Coverage Goal**: 90%+ for new code

### Test Fixtures

Use fixtures in `tests/fixtures/`:
- Sample images
- Sample PDFs
- Expected outputs

## BBox Handling Rules

!!! warning "Critical: BBox Standards"
    1. **Always use BBox class** - Never use raw lists/tuples
    2. **Internal operations use xyxy** - Access via `bbox.x0, bbox.y0, bbox.x1, bbox.y1`
    3. **JSON serialization uses xywh** - Call `bbox.to_xywh_list()`
    4. **Accept floats, output integers** - All BBox methods round to nearest integer
    5. **PyPDF requires page height** - Use `BBox.from_pypdf_rect(rect, page_height)`

## Adding New Components

### Adding a Detector

1. Create detector file in `pipeline/layout/detection/`
2. Implement `Detector` protocol from `pipeline/types.py`
3. Register in `create_detector()` factory
4. Add validation rule in `validate_combination()` if needed
5. Write tests in `tests/test_detectors.py`

**Example**:

```python
# pipeline/layout/detection/my_detector.py
from pipeline.types import Block, Detector

class MyDetector:
    """My custom detector implementation."""

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks.

        Args:
            image: Input image (H, W, C)

        Returns:
            List of detected blocks
        """
        # Your detection logic
        return blocks

# pipeline/layout/detection/__init__.py
def create_detector(name: str, **kwargs) -> Detector:
    if name == "my-detector":
        from .my_detector import MyDetector  # noqa: PLC0415
        return MyDetector(**kwargs)
    ...
```

### Adding a Sorter

Similar process in `pipeline/layout/ordering/`:

```python
from pipeline.types import Block, Sorter

class MySorter:
    """My custom sorter implementation."""

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs) -> list[Block]:
        """Sort blocks in reading order.

        Args:
            blocks: Detected blocks
            image: Original image for context

        Returns:
            Sorted blocks with order field
        """
        # Your sorting logic
        return sorted_blocks
```

### Adding Prompts

Place YAML prompts in `settings/prompts/{model}/`:

```yaml
# settings/prompts/my-model/text_extraction.yaml
system: |
  You are an expert OCR system.

user: |
  Extract text from this image.
  Preserve formatting and structure.

fallback: |
  [OCR failed]
```

## Error Handling

Follow the error handling policy (see [Error Handling Guide](error-handling.md)):

### Custom Exceptions

Use specific exception types from `pipeline/exceptions.py`:

```python
from pipeline.exceptions import DetectionError, InvalidConfigError

if confidence < 0 or confidence > 1:
    raise InvalidConfigError(f"Confidence must be between 0 and 1, got {confidence}")

try:
    blocks = self.detector.detect(image)
except Exception as e:
    raise DetectionError(f"Detection failed: {e}") from e
```

### Error Logging

Use proper logging with `%s` formatting (not f-strings):

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Good
logger.error("Failed to load file %s: %s", file_path, error)

# ❌ Bad
logger.error(f"Failed to load file {file_path}: {error}")
```

Add `exc_info=True` for unexpected errors:

```python
except Exception as e:
    logger.error("Unexpected error: %s", e, exc_info=True)
```

## Commit Guidelines

### Commit Messages

Follow conventional commits:

```
feat: add new detector for layout analysis
fix: resolve type error in BBox conversion
docs: update installation guide
test: add tests for multi-column detection
refactor: simplify block sorting logic
perf: optimize image preprocessing
```

### Before Committing

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Run tests
uv run pytest

# Type check
npx pyright
```

### Creating a Pull Request

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG (if exists)
4. Create PR with clear description:

```markdown
## Summary

Brief description of changes

## Changes

- Added feature X
- Fixed bug Y
- Updated documentation Z

## Testing

- Tested on Python 3.11
- All existing tests pass
- Added new tests for feature X

## Breaking Changes

None (or list if applicable)
```

## Common Pitfalls

!!! danger "Avoid These Mistakes"
    - **Don't use bare except**: Catch specific exceptions
    - **Don't create empty `__init__.py`**: Use PEP 420 namespace packages
    - **Don't install with `-e`**: Never use editable mode from external directories
    - **Don't mix xywh/xyxy**: Always convert via BBox methods
    - **Don't forget page_height for PyPDF**: Y-axis flip required

## Documentation

### Building Docs Locally

```bash
# Install MkDocs
uv pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

### Writing Docs

- Use Markdown with admonitions
- Include code examples
- Add mermaid diagrams where helpful
- Cross-reference related pages

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/NoUnique/vlm-ocr-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NoUnique/vlm-ocr-pipeline/discussions)
- **Documentation**: This site

## Code Review Process

1. All PRs require review
2. Address review comments
3. Keep PRs focused (one feature/fix per PR)
4. Maintain backward compatibility when possible
5. Update tests and docs

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
