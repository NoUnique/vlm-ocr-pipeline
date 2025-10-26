"""Input Stage: Document loading and auxiliary info extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class InputStage:
    """Stage 1: Input - Load documents and extract auxiliary information."""

    def __init__(self, temp_dir: Path):
        """Initialize InputStage.

        Args:
            temp_dir: Temporary directory for intermediate files
        """
        self.temp_dir = temp_dir

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array
        """
        from pipeline.conversion.input.image import load_image

        return load_image(image_path)

    def load_pdf_page(self, pdf_path: Path, page_num: int) -> np.ndarray:
        """Render PDF page as image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)

        Returns:
            Page image as numpy array
        """
        from pipeline.conversion.input.pdf import render_pdf_page

        page_image, _temp_path = render_pdf_page(pdf_path, page_num, temp_dir=self.temp_dir)
        return page_image

    def extract_auxiliary_info(self, pdf_path: Path, page_num: int) -> dict[str, Any] | None:
        """Extract auxiliary information from PDF page.

        This includes text spans with font information for pymupdf4llm-style
        markdown conversion. Uses PyMuPDF terminology.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)

        Returns:
            Dictionary with auxiliary info or None if extraction fails
        """
        try:
            from pipeline.conversion.input.pdf import extract_text_spans_from_pdf

            text_spans = extract_text_spans_from_pdf(pdf_path, page_num)
            if text_spans:
                return {
                    "text_spans": text_spans,  # PyMuPDF spans with size, font
                }
            return None
        except Exception as exc:
            logger.warning("Failed to extract auxiliary info from page %d: %s", page_num, exc)
            return None
