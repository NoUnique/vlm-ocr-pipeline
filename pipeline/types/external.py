"""Protocol definitions for external library types.

These protocols define minimal interfaces for external library objects
(e.g., PyMuPDF) without requiring the actual library as a dependency.
"""

from __future__ import annotations

from typing import Protocol


class PyMuPDFRect(Protocol):
    """Protocol for PyMuPDF Rect object.

    This defines the minimal interface we need from fitz.Rect
    without requiring PyMuPDF as a dependency for type checking.
    """

    x0: float  # Left boundary
    y0: float  # Top boundary
    x1: float  # Right boundary
    y1: float  # Bottom boundary


class PyMuPDFPage(Protocol):
    """Protocol for PyMuPDF Page object.

    This defines the minimal interface we need from fitz.Page
    without requiring PyMuPDF as a dependency for type checking.
    """

    rect: PyMuPDFRect  # Page rectangle
