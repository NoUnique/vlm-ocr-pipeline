"""PyMuPDF ordering implementations."""

from __future__ import annotations

__all__ = []

try:
    from .multi_column import MultiColumnSorter

    __all__.append("MultiColumnSorter")
except ImportError:
    MultiColumnSorter = None  # type: ignore[assignment, misc]
