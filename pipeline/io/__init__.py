"""IO module for input loading and output saving.

This module provides unified interfaces for:
- Loading documents (PDF, images)
- Saving results (JSON, Markdown, plaintext)
"""

from .input import InputLoader
from .output import OutputSaver

__all__ = ["InputLoader", "OutputSaver"]
