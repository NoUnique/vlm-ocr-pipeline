"""Document converter for PDF and image file processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pdf2image.pdf2image import convert_from_path, pdfinfo_from_path

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    fitz = None  # type: ignore


class DocumentConverter:
    """Handles conversion of document files to images.
    
    This class is responsible for:
    - Converting PDF pages to images
    - Loading and processing image files
    - Managing PyMuPDF documents for advanced PDF operations
    - Determining page ranges and counts
    """

    def __init__(self, temp_dir: Path):
        """Initialize the document converter.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf_info(self, pdf_path: Path) -> dict[str, Any]:
        """Get PDF document information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata including page count
        """
        return pdfinfo_from_path(str(pdf_path))

    def render_pdf_page(self, pdf_path: Path, page_num: int, dpi: int = 200) -> tuple[np.ndarray, Path]:
        """Render a PDF page to an image.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to render (1-indexed)
            dpi: DPI for rendering
            
        Returns:
            Tuple of (image array, temporary file path)
        """
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)
        page_image = np.array(images[0])

        temp_image_path = self.temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"
        cv2.imwrite(str(temp_image_path), cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR))
        logger.info("Rendered PDF page to: %s", temp_image_path)

        return page_image, temp_image_path

    def open_pymupdf_document(self, pdf_path: Path) -> Any | None:
        """Open a PDF document using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PyMuPDF document object or None if PyMuPDF is not available
        """
        if fitz is None:
            return None

        try:
            return fitz.open(str(pdf_path))
        except Exception as exc:  # pragma: no cover - PyMuPDF failure path
            logger.warning(
                "Failed to open PDF with PyMuPDF for multi-column ordering: %s",
                exc,
            )
            return None

    def determine_pages_to_process(
        self,
        total_pages: int,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
        pages: list[int] | None = None,
    ) -> list[int]:
        """Determine which pages to process based on limiting options.
        
        Args:
            total_pages: Total number of pages in the document
            max_pages: Maximum number of pages to process
            page_range: Range of pages to process (start, end)
            pages: Specific list of page numbers to process
            
        Returns:
            List of page numbers to process
        """
        if pages is not None:
            # Specific pages specified
            valid_pages = [p for p in pages if 1 <= p <= total_pages]
            if len(valid_pages) != len(pages):
                invalid_pages = [p for p in pages if p not in valid_pages]
                logger.warning("Invalid page numbers (outside 1-%d): %s", total_pages, invalid_pages)
            return sorted(valid_pages)

        elif page_range is not None:
            # Page range specified
            start, end = page_range
            start = max(1, start)
            end = min(total_pages, end)
            return list(range(start, end + 1))

        elif max_pages is not None:
            # Max pages specified
            return list(range(1, min(max_pages + 1, total_pages + 1)))

        else:
            # Process all pages
            return list(range(1, total_pages + 1))

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array
            
        Raises:
            ValueError: If image cannot be loaded
        """
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image_np

