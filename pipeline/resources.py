"""Resource management utilities with context managers.

This module provides context managers for automatic resource cleanup
to prevent memory leaks and improve performance.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Optional torch import for GPU memory management
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# Optional PyMuPDF import
try:
    import fitz  # type: ignore[import-untyped]

    _HAS_PYMUPDF = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    _HAS_PYMUPDF = False


# ==================== PyMuPDF Document Context Manager ====================


@contextmanager
def open_pdf_document(pdf_path: str | Path) -> Iterator[Any]:
    """Context manager for PyMuPDF document.

    Automatically closes the document after use to prevent memory leaks.
    PyMuPDF documents can hold significant memory, especially for large PDFs.

    Args:
        pdf_path: Path to PDF file

    Yields:
        PyMuPDF document object (fitz.Document)

    Raises:
        RuntimeError: If PyMuPDF is not available
        FileNotFoundError: If PDF file does not exist

    Example:
        >>> with open_pdf_document("doc.pdf") as doc:
        ...     print(f"Pages: {doc.page_count}")
        ...     page = doc.load_page(0)
        ...     # Document automatically closed after this block
    """
    if not _HAS_PYMUPDF or fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not available. Install it with: pip install pymupdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        logger.debug("Opened PDF document: %s (pages: %d)", pdf_path.name, doc.page_count)
        yield doc
    finally:
        if doc is not None:
            doc.close()
            logger.debug("Closed PDF document: %s", pdf_path.name)
        # Force garbage collection to free memory
        gc.collect()


# ==================== Image Resource Context Manager ====================


@contextmanager
def managed_numpy_array(*arrays: np.ndarray) -> Iterator[tuple[np.ndarray, ...]]:
    """Context manager for numpy arrays to ensure cleanup.

    Helps manage memory for large image arrays by explicitly deleting
    them and running garbage collection after use. Also clears GPU cache
    if torch is available.

    Args:
        *arrays: One or more numpy arrays to manage

    Yields:
        Tuple of managed numpy arrays (same as input)

    Example:
        >>> import numpy as np
        >>> large_image = np.zeros((4000, 3000, 3), dtype=np.uint8)
        >>> with managed_numpy_array(large_image) as (image,):
        ...     # Process image
        ...     result = some_processing(image)
        ... # Image automatically cleaned up after this block
    """
    try:
        yield arrays if len(arrays) > 1 else (arrays[0],)  # type: ignore[return-value]
    finally:
        # Delete all arrays
        for arr in arrays:
            del arr
        logger.debug("Cleaned up %d numpy array(s)", len(arrays))

        # Clean up GPU memory if available
        if _HAS_TORCH and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")

        gc.collect()


@contextmanager
def managed_image_processing() -> Iterator[None]:
    """Context manager for image processing operations.

    Ensures garbage collection runs after image processing to free memory.
    Also clears GPU cache if torch is available. Useful for wrapping blocks
    of code that create many temporary images.

    Example:
        >>> with managed_image_processing():
        ...     for block in blocks:
        ...         cropped = block.bbox.crop(image, padding=5)
        ...         result = process(cropped)
        ... # All temporary images cleaned up here
    """
    try:
        yield
    finally:
        # Clean up GPU memory if available
        if _HAS_TORCH and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache after image processing")

        gc.collect()


# ==================== Generic Resource Manager ====================


class ManagedResource:
    """Context manager for any resource with cleanup function.

    This is a generic wrapper that can manage any resource that needs cleanup.

    Args:
        resource: The resource object to manage
        cleanup_fn: Function to call for cleanup (receives resource as argument)
        name: Optional name for logging

    Example:
        >>> def cleanup_model(model):
        ...     model.clear_cache()
        ...     del model
        >>> with ManagedResource(my_model, cleanup_model, "YOLOv8") as model:
        ...     predictions = model.predict(image)
        ... # Model automatically cleaned up
    """

    def __init__(
        self,
        resource: Any,
        cleanup_fn: callable,  # type: ignore[valid-type]
        name: str | None = None,
    ):
        """Initialize managed resource.

        Args:
            resource: Resource to manage
            cleanup_fn: Cleanup function
            name: Optional name for logging
        """
        self.resource = resource
        self.cleanup_fn = cleanup_fn
        self.name = name or "resource"

    def __enter__(self) -> Any:
        """Enter context."""
        logger.debug("Acquired resource: %s", self.name)
        return self.resource

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and cleanup."""
        try:
            self.cleanup_fn(self.resource)
            logger.debug("Cleaned up resource: %s", self.name)
        except Exception as e:  # noqa: BLE001 - catch all for cleanup
            logger.warning("Error cleaning up resource %s: %s", self.name, e)
        finally:
            # Clean up GPU memory if available
            if _HAS_TORCH and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU cache for resource: %s", self.name)

            gc.collect()
