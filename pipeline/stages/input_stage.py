"""Input Stage: Document loading and auxiliary info extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InputResult:
    """Result from input stage processing.

    Attributes:
        image: Page image as numpy array
        auxiliary_info: Optional auxiliary information (text spans, etc.)
        pdf_path: Path to source PDF (if applicable)
        page_num: Page number (if applicable)
    """

    image: np.ndarray
    auxiliary_info: dict[str, Any] | None = None
    pdf_path: Path | None = None
    page_num: int | None = None


class InputStage:
    """Stage 1: Input - Load documents and extract auxiliary information.

    This stage handles document loading (images, PDFs) and extracts
    auxiliary information like text spans for downstream processing.
    """

    name = "input"

    def __init__(
        self,
        temp_dir: Path,
        dpi: int = 200,
        detection_dpi: int = 150,
        recognition_dpi: int = 300,
        use_dual_resolution: bool = False,
    ):
        """Initialize InputStage.

        Args:
            temp_dir: Temporary directory for intermediate files
            dpi: Default DPI for PDF-to-image conversion
            detection_dpi: DPI for detection stage (used when use_dual_resolution=True)
            recognition_dpi: DPI for recognition stage (used when use_dual_resolution=True)
            use_dual_resolution: Use different DPIs for detection and recognition
        """
        self.temp_dir = temp_dir
        self.dpi = dpi
        self.detection_dpi = detection_dpi
        self.recognition_dpi = recognition_dpi
        self.use_dual_resolution = use_dual_resolution

    def process_pdf_page(self, pdf_path: Path, page_num: int, dpi: int | None = None) -> InputResult:
        """Process a PDF page: load image and extract auxiliary info.

        This is the main entry point for processing PDF pages.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            dpi: DPI for rendering (default: use self.dpi)

        Returns:
            InputResult with image and auxiliary information
        """
        image = self.load_pdf_page(pdf_path, page_num, dpi)
        auxiliary_info = self.extract_auxiliary_info(pdf_path, page_num)
        return InputResult(
            image=image,
            auxiliary_info=auxiliary_info,
            pdf_path=pdf_path,
            page_num=page_num,
        )

    def process_image(self, image_path: Path) -> InputResult:
        """Process an image file.

        Args:
            image_path: Path to image file

        Returns:
            InputResult with image
        """
        image = self.load_image(image_path)
        return InputResult(image=image)

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array
        """
        from pipeline.io.input.image import load_image

        return load_image(image_path)

    def load_pdf_page(self, pdf_path: Path, page_num: int, dpi: int | None = None) -> np.ndarray:
        """Render PDF page as image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            dpi: DPI for rendering (default: use self.dpi)

        Returns:
            Page image as numpy array
        """
        from pipeline.io.input.pdf import render_pdf_page

        effective_dpi = dpi if dpi is not None else self.dpi
        page_image, _temp_path = render_pdf_page(pdf_path, page_num, temp_dir=self.temp_dir, dpi=effective_dpi)
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
            from pipeline.io.input.pdf import extract_text_spans_from_pdf

            text_spans = extract_text_spans_from_pdf(pdf_path, page_num)
            if text_spans:
                return {
                    "text_spans": text_spans,  # PyMuPDF spans with size, font
                }
            return None
        except Exception as exc:
            logger.warning("Failed to extract auxiliary info from page %d: %s", page_num, exc)
            return None
