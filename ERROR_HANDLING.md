# Error Handling Guidelines

This document defines the error handling policy for the VLM OCR Pipeline project.

## Table of Contents

- [Error Handling Guidelines](#error-handling-guidelines)
  - [Table of Contents](#table-of-contents)
  - [1. Custom Exception Hierarchy](#1-custom-exception-hierarchy)
  - [2. When to Use Each Exception](#2-when-to-use-each-exception)
    - [ConfigurationError](#configurationerror)
    - [APIError](#apierror)
    - [ProcessingError](#processingerror)
    - [FileError](#fileerror)
    - [DependencyError](#dependencyerror)
  - [3. Exception Handling Best Practices](#3-exception-handling-best-practices)
    - [3.1. Catch Specific Exceptions](#31-catch-specific-exceptions)
    - [3.2. Exception Chaining](#32-exception-chaining)
    - [3.3. When to Use Broad Exception Handlers](#33-when-to-use-broad-exception-handlers)
    - [3.4. Re-raising Exceptions](#34-re-raising-exceptions)
  - [4. Error Logging Standards](#4-error-logging-standards)
    - [4.1. Logging Levels](#41-logging-levels)
    - [4.2. Log Message Format](#42-log-message-format)
    - [4.3. Exception Stack Traces](#43-exception-stack-traces)
  - [5. Error Recovery Strategies](#5-error-recovery-strategies)
    - [5.1. Graceful Degradation](#51-graceful-degradation)
    - [5.2. Retry Logic](#52-retry-logic)
    - [5.3. Fallback Values](#53-fallback-values)
  - [6. Testing Error Handling](#6-testing-error-handling)
  - [7. Migration Guide](#7-migration-guide)
    - [Step 1: Identify the error type](#step-1-identify-the-error-type)
    - [Step 2: Choose the appropriate custom exception](#step-2-choose-the-appropriate-custom-exception)
    - [Step 3: Add proper error context](#step-3-add-proper-error-context)
    - [Step 4: Update logging](#step-4-update-logging)

---

## 1. Custom Exception Hierarchy

All custom exceptions inherit from `PipelineError` and are defined in `pipeline/exceptions.py`:

```
PipelineError (base)
├── ConfigurationError
│   ├── InvalidConfigError
│   └── MissingConfigError
├── APIError
│   ├── APIClientError
│   ├── APIAuthenticationError
│   ├── APIRateLimitError
│   └── APITimeoutError
├── ProcessingError
│   ├── PageProcessingError
│   ├── DetectionError
│   ├── RecognitionError
│   └── RenderingError
├── FileError
│   ├── FileLoadError
│   ├── FileSaveError
│   └── FileFormatError
└── DependencyError
```

**Import all exceptions from:**
```python
from pipeline.exceptions import (
    PipelineError,
    ConfigurationError,
    InvalidConfigError,
    MissingConfigError,
    APIError,
    APIClientError,
    APIAuthenticationError,
    APIRateLimitError,
    APITimeoutError,
    ProcessingError,
    PageProcessingError,
    DetectionError,
    RecognitionError,
    RenderingError,
    FileError,
    FileLoadError,
    FileSaveError,
    FileFormatError,
    DependencyError,
)
```

---

## 2. When to Use Each Exception

### ConfigurationError

Use when dealing with configuration files, settings, or initialization parameters.

**InvalidConfigError:**
```python
# Example: Invalid tier name
if tier not in ["free", "tier1", "tier2", "tier3"]:
    raise InvalidConfigError(f"Invalid tier: {tier}. Must be one of: free, tier1, tier2, tier3")

# Example: Malformed YAML
try:
    config = yaml.safe_load(f)
except yaml.YAMLError as e:
    raise InvalidConfigError(f"Malformed YAML in {config_file}: {e}") from e
```

**MissingConfigError:**
```python
# Example: Missing API key
if not api_key:
    raise MissingConfigError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

# Example: Required config file not found
if not config_file.exists():
    raise MissingConfigError(f"Configuration file not found: {config_file}")
```

### APIError

Use when interacting with external APIs (OpenAI, Gemini, PaddleOCR-VL, etc.).

**APIClientError:**
```python
# Example: Client initialization failure
try:
    self.client = OpenAI(api_key=api_key, base_url=base_url)
except TypeError as e:
    raise APIClientError(f"Failed to initialize OpenAI client: {e}") from e
```

**APIAuthenticationError:**
```python
# Example: Invalid API key (from external library)
try:
    response = self.client.chat.completions.create(...)
except AuthenticationError as e:
    raise APIAuthenticationError(f"OpenAI authentication failed: {e}") from e
```

**APIRateLimitError:**
```python
# Example: Rate limit exceeded
try:
    response = self.client.chat.completions.create(...)
except RateLimitError as e:
    raise APIRateLimitError(f"OpenAI rate limit exceeded: {e}") from e
```

**APITimeoutError:**
```python
# Example: Request timeout
try:
    response = requests.get(url, timeout=30)
except requests.Timeout as e:
    raise APITimeoutError(f"API request timed out: {e}") from e
```

### ProcessingError

Use when processing documents through the pipeline stages.

**PageProcessingError:**
```python
# Example: Page rendering failure
try:
    page_image = render_pdf_page(pdf_path, page_num)
except Exception as e:
    raise PageProcessingError(f"Failed to process page {page_num}: {e}") from e
```

**DetectionError:**
```python
# Example: Layout detection failure
try:
    blocks = self.detector.detect(page_image)
except Exception as e:
    raise DetectionError(f"Layout detection failed: {e}") from e
```

**RecognitionError:**
```python
# Example: Text recognition failure
try:
    text = self.recognizer.extract_text(block_image)
except Exception as e:
    raise RecognitionError(f"Text recognition failed for block: {e}") from e
```

**RenderingError:**
```python
# Example: Markdown conversion error
try:
    markdown = block_to_markdown(block)
except Exception as e:
    raise RenderingError(f"Failed to render block to markdown: {e}") from e
```

### FileError

Use for file I/O operations.

**FileLoadError:**
```python
# Example: File not found
if not file_path.exists():
    raise FileLoadError(f"File not found: {file_path}")

# Example: PDF loading failure
try:
    images = convert_from_path(str(pdf_path))
except Exception as e:
    raise FileLoadError(f"Failed to load PDF: {e}") from e
```

**FileSaveError:**
```python
# Example: Permission denied
try:
    with open(output_file, "w") as f:
        json.dump(data, f)
except PermissionError as e:
    raise FileSaveError(f"Permission denied writing to {output_file}: {e}") from e

# Example: Disk full
except OSError as e:
    raise FileSaveError(f"Failed to save file {output_file}: {e}") from e
```

**FileFormatError:**
```python
# Example: Invalid PDF
if not is_valid_pdf(file_path):
    raise FileFormatError(f"Invalid PDF file: {file_path}")

# Example: Unsupported image format
if file_path.suffix not in [".png", ".jpg", ".jpeg"]:
    raise FileFormatError(f"Unsupported image format: {file_path.suffix}")
```

### DependencyError

Use when optional dependencies are missing or incompatible.

```python
# Example: Missing optional dependency
try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise DependencyError("PyMuPDF is required for multi-column detection. Install with: uv pip install pymupdf") from e

# Example: Incompatible version
if version.parse(mineru.__version__) < version.parse("0.8.0"):
    raise DependencyError(f"MinerU version {mineru.__version__} is not supported. Requires >= 0.8.0")
```

---

## 3. Exception Handling Best Practices

### 3.1. Catch Specific Exceptions

**✅ Good:**
```python
try:
    config = yaml.safe_load(f)
except yaml.YAMLError as e:
    raise InvalidConfigError(f"Malformed YAML: {e}") from e
except OSError as e:
    raise FileLoadError(f"Failed to read config file: {e}") from e
```

**❌ Bad:**
```python
try:
    config = yaml.safe_load(f)
except Exception as e:  # Too broad!
    logger.error("Error: %s", e)
```

### 3.2. Exception Chaining

Always use `from e` to preserve the original exception context:

**✅ Good:**
```python
try:
    response = self.client.chat.completions.create(...)
except AuthenticationError as e:
    raise APIAuthenticationError(f"Authentication failed: {e}") from e
```

**❌ Bad:**
```python
try:
    response = self.client.chat.completions.create(...)
except AuthenticationError as e:
    raise APIAuthenticationError(f"Authentication failed: {e}")  # Lost context!
```

### 3.3. When to Use Broad Exception Handlers

Broad exception handlers (`except Exception`) are **only allowed** in these cases:

1. **Top-level CLI entry points** (with `# noqa: BLE001` comment):
   ```python
   # main.py
   try:
       result = pipeline.process(pdf_path)
   except Exception as exc:  # noqa: BLE001 - retain broad logging for CLI
       logger.error("Unexpected error: %s", exc, exc_info=True)
       return 1
   ```

2. **Optional dependency guards** (with `# pragma: no cover` comment):
   ```python
   try:
       import fitz
   except Exception:  # pragma: no cover - optional dependency guard
       fitz = None
   ```

3. **Fallback for unexpected errors** (after catching specific errors):
   ```python
   try:
       response = self.client.chat.completions.create(...)
   except AuthenticationError as e:
       raise APIAuthenticationError(f"Authentication failed: {e}") from e
   except RateLimitError as e:
       raise APIRateLimitError(f"Rate limit exceeded: {e}") from e
   except Exception as e:
       # Fallback for unexpected errors (document why!)
       logger.error("Unexpected API error: %s", e)
       return {"error": "api_error", "message": str(e)}
   ```

### 3.4. Re-raising Exceptions

When you want to log an error but still propagate it:

```python
try:
    page_result = self._process_pdf_page(pdf_path, page_num)
except PageProcessingError as e:
    logger.error("Page %d processing failed: %s", page_num, e)
    raise  # Re-raise the same exception
```

---

## 4. Error Logging Standards

### 4.1. Logging Levels

Use appropriate logging levels:

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Detailed diagnostic information | `logger.debug("Processing block %d of %d", i, total)` |
| `INFO` | General informational messages | `logger.info("Loaded %d pages from %s", len(pages), pdf_path)` |
| `WARNING` | Recoverable errors, fallback used | `logger.warning("PyMuPDF not available, using basic sorter")` |
| `ERROR` | Serious errors, operation failed | `logger.error("Failed to process page %d: %s", page_num, e)` |
| `CRITICAL` | Critical errors, system cannot continue | `logger.critical("API key not found, cannot proceed")` |

### 4.2. Log Message Format

**Format:** `"Action failed: %s", error_details`

**✅ Good:**
```python
logger.error("Failed to load config file %s: %s", config_path, e)
logger.warning("Rate limit reached. Waiting %.2f seconds...", wait_time)
logger.info("Processed %d pages in %.2f seconds", page_count, elapsed)
```

**❌ Bad:**
```python
logger.error(f"Failed to load config file {config_path}: {e}")  # Don't use f-strings!
logger.error("Error!")  # Not descriptive!
logger.error(str(e))  # Missing context!
```

**Why avoid f-strings in logging?**
- f-strings are evaluated before the log level check (performance overhead)
- `%s` formatting is only evaluated if the log level is enabled
- Better for structured logging

### 4.3. Exception Stack Traces

Use `exc_info=True` to include full stack trace:

**✅ Good:**
```python
try:
    result = process_page(page_num)
except PageProcessingError as e:
    logger.error("Error processing page %d: %s", page_num, e, exc_info=True)
```

**When to use `exc_info=True`?**
- For unexpected errors at top-level handlers
- When debugging complex issues
- For errors that should never happen

**When NOT to use `exc_info=True`?**
- For expected errors (rate limits, missing files)
- For informational warnings
- When stack trace would be too verbose

---

## 5. Error Recovery Strategies

### 5.1. Graceful Degradation

Continue processing with reduced functionality:

```python
try:
    import fitz  # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF not available. Multi-column detection disabled.")
    fitz = None

# Later in code
if fitz is not None:
    # Use advanced multi-column detection
    layout = detect_multi_column_layout(page)
else:
    # Fallback to basic processing
    layout = None
```

### 5.2. Retry Logic

Retry on transient failures:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def api_call_with_retry():
    try:
        return self.client.chat.completions.create(...)
    except APITimeoutError:
        logger.warning("API timeout, retrying...")
        raise
```

### 5.3. Fallback Values

Return safe defaults on error:

```python
def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration file with fallback to empty dict."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.debug("Config file not found: %s", config_path)
        return {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
```

---

## 6. Testing Error Handling

Always test error paths:

```python
def test_invalid_tier_raises_error():
    """Test that InvalidConfigError is raised for invalid tier."""
    with pytest.raises(InvalidConfigError, match="Invalid tier"):
        rate_limiter.set_tier_and_model("invalid_tier", "gemini-2.5-flash")

def test_missing_api_key_raises_error():
    """Test that MissingConfigError is raised when API key is missing."""
    with pytest.raises(MissingConfigError, match="API key not found"):
        OpenAIClient(api_key=None)

def test_file_not_found_raises_error():
    """Test that FileLoadError is raised for non-existent file."""
    with pytest.raises(FileLoadError, match="File not found"):
        load_pdf("/nonexistent/file.pdf")
```

---

## 7. Migration Guide

When migrating from `except Exception` to specific exceptions:

### Step 1: Identify the error type

Look at what can go wrong in the try block:
```python
# Before
try:
    config = yaml.safe_load(f)
except Exception as e:
    logger.error("Error: %s", e)
    return {}
```

### Step 2: Choose the appropriate custom exception

- YAML parsing error → `InvalidConfigError`
- File I/O error → `FileLoadError`

### Step 3: Add proper error context

```python
# After
try:
    config = yaml.safe_load(f)
except yaml.YAMLError as e:
    raise InvalidConfigError(f"Malformed YAML in {config_file}: {e}") from e
except OSError as e:
    raise FileLoadError(f"Failed to read config file {config_file}: {e}") from e
```

### Step 4: Update logging

```python
# Caller code
try:
    config = load_config(config_path)
except InvalidConfigError as e:
    logger.error("Configuration error: %s", e)
    return {}
except FileLoadError as e:
    logger.warning("Config file not found: %s", e)
    return {}
```

---

**Last Updated:** 2025-01-26

**See Also:**
- `pipeline/exceptions.py` - Custom exception definitions
- `CLAUDE.md` - Project coding standards
- `.cursorrules` - Detailed coding guidelines
