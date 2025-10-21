# OpenCV Contrib Python Dummy Package

## Problem

PaddleX (required by PaddleOCR) has a hardcoded dependency check for `opencv-contrib-python`:

```python
# From PaddleX source: paddlex/utils/deps.py
def require_deps(*deps, obj_name=None):
    unavailable_deps = [dep for dep in deps if not is_dep_available(dep)]
    if len(unavailable_deps) > 0:
        raise DependencyError(...)
```

The `PDFReaderBackend` class specifically requires `opencv-contrib-python`:
```python
@requires_deps(("opencv-contrib-python", "pypdfium2"))
class PDFReaderBackend:
    ...
```

However, `opencv-contrib-python` has GUI dependencies (libGL.so.1, libgthread-2.0.so.0) that are not available in headless server environments.

## Solution

This dummy package satisfies PaddleX's metadata check while using `opencv-python-headless` for actual cv2 functionality.

### How It Works

1. **PaddleX checks package metadata**:
   ```python
   version = importlib.metadata.version("opencv-contrib-python")
   ```

2. **This dummy package provides that metadata**:
   ```python
   name="opencv-contrib-python"
   version="4.7.0.72"
   ```

3. **Actual cv2 module comes from opencv-python-headless**:
   ```python
   import cv2  # Works via opencv-python-headless
   ```

## Installation

```bash
# Install the dummy package in editable mode
uv pip install -e workarounds/opencv-contrib-python-dummy

# Verify it worked
python -c "import importlib.metadata; print(importlib.metadata.version('opencv-contrib-python'))"
# Should output: 4.7.0.72

python -c "import cv2; print(cv2.__version__)"
# Should work via opencv-python-headless
```

## Why Not Just Install opencv-contrib-python?

Installing `opencv-contrib-python` would bring GUI dependencies:
- libGL.so.1 (OpenGL)
- libgthread-2.0.so.0 (GTK)
- X11 libraries

These are not needed for our use case (image processing only) and may not be available on headless servers.

## Alternative Solutions Considered

1. **Install system libraries** (rejected - requires sudo, pollutes system)
2. **Modify PaddleX source** (rejected - harder to maintain, git submodule conflicts)
3. **Use opencv-contrib-python-headless** (doesn't exist as a package)
4. **This dummy package** (chosen - clean, documented, reversible)

## Maintenance

Keep the version number in sync with `opencv-python-headless` version in requirements.txt.

Current version: **4.7.0.72**
