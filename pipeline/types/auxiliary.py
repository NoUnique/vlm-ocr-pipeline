"""Auxiliary type definitions for document processing.

TypedDict classes for text spans, column layout, and other metadata.
"""

from __future__ import annotations

from typing import TypedDict


class TextSpan(TypedDict):
    """PyMuPDF text span with font information.

    Extracted from PDF using PyMuPDF's get_text("dict") method.
    Uses PyMuPDF terminology: 'size' for font size, 'font' for font name.
    """

    bbox: list[int]  # [x0, y0, x1, y1] in xyxy format
    text: str  # Text content
    size: float  # Font size (PyMuPDF terminology)
    font: str  # Font name (PyMuPDF terminology)


class AuxiliaryInfo(TypedDict, total=False):
    """Auxiliary information extracted from PDF.

    This contains metadata used for enhanced markdown conversion,
    such as font information for auto-detecting headers.
    """

    text_spans: list[TextSpan]  # Text spans with font metadata


class ColumnInfo(TypedDict):
    """Column information for multi-column documents."""

    index: int  # Column index (0-based)
    x0: int  # Left boundary
    x1: int  # Right boundary
    center: float  # Center X coordinate
    width: int  # Column width


class ColumnLayout(TypedDict):
    """Column layout information for a page."""

    columns: list[ColumnInfo]  # List of columns on the page
