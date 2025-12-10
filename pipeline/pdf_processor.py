"""PDF processing utilities for the VLM OCR Pipeline.

This module handles PDF-specific operations like loading pages,
extracting images, and managing PyMuPDF document lifecycle.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .stages import InputStage

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF loading and page extraction.

    This class encapsulates PDF-specific logic including:
    - Loading PDF documents
    - Extracting page images at different DPIs
    - Managing PyMuPDF document lifecycle
    - Extracting auxiliary information

    Attributes:
        input_stage: Input stage for PDF conversion
        use_dual_resolution: Whether to use different DPIs for detection/recognition
        detection_dpi: DPI for detection stage
        recognition_dpi: DPI for recognition stage
    """

    def __init__(
        self,
        input_stage: InputStage,
        use_dual_resolution: bool = False,
        detection_dpi: int = 150,
        recognition_dpi: int = 300,
    ):
        """Initialize PDFProcessor.

        Args:
            input_stage: Input stage for PDF conversion
            use_dual_resolution: Use different DPIs for detection and recognition
            detection_dpi: DPI for detection stage
            recognition_dpi: DPI for recognition stage
        """
        self.input_stage = input_stage
        self.use_dual_resolution = use_dual_resolution
        self.detection_dpi = detection_dpi
        self.recognition_dpi = recognition_dpi

    def load_page_images(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, dict[str, Any]],
        dict[int, Any | None],
        Any | None,
    ]:
        """Load page images and auxiliary information from PDF.

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers to process

        Returns:
            Tuple of:
                - page_images: Detection images by page number
                - recognition_images: Recognition images by page number
                - auxiliary_infos: Auxiliary info by page number
                - pymupdf_pages: PyMuPDF page objects by page number
                - pymupdf_doc: PyMuPDF document (caller must close)
        """
        page_images: dict[int, np.ndarray] = {}
        recognition_images: dict[int, np.ndarray] = {}
        auxiliary_infos: dict[int, dict[str, Any]] = {}
        pymupdf_pages: dict[int, Any | None] = {}
        pymupdf_doc = None

        # Prepare PyMuPDF document if pymupdf-based sorter is used
        try:
            import pymupdf

            pymupdf_doc = pymupdf.open(str(pdf_path))
        except ImportError:
            pass

        for page_num in pages_to_process:
            # Load detection image
            page_image = self.input_stage.load_pdf_page(pdf_path, page_num, dpi=self.detection_dpi)
            page_images[page_num] = page_image
            aux_info = self.input_stage.extract_auxiliary_info(pdf_path, page_num)
            auxiliary_infos[page_num] = aux_info if aux_info is not None else {}

            # Load recognition image (if dual resolution)
            if self.use_dual_resolution and self.detection_dpi != self.recognition_dpi:
                recognition_image = self.input_stage.load_pdf_page(
                    pdf_path,
                    page_num,
                    dpi=self.recognition_dpi,
                )
                recognition_images[page_num] = recognition_image
            else:
                recognition_images[page_num] = page_image

            # Load PyMuPDF page if document is available
            if pymupdf_doc is not None:
                pymupdf_pages[page_num] = pymupdf_doc[page_num - 1]
            else:
                pymupdf_pages[page_num] = None

        return page_images, recognition_images, auxiliary_infos, pymupdf_pages, pymupdf_doc

    def get_total_pages(self, pdf_path: Path) -> int:
        """Get total number of pages in PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Total number of pages
        """
        try:
            import pymupdf

            with pymupdf.open(str(pdf_path)) as doc:
                return len(doc)
        except ImportError:
            # Fallback: use pdf2image (slower)
            try:
                from pdf2image.pdf2image import pdfinfo_from_path

                info = pdfinfo_from_path(str(pdf_path))
                return info.get("Pages", 0)
            except ImportError:
                logger.warning("Neither pymupdf nor pdf2image available for page count")
                return 0

    def close_document(self, pymupdf_doc: Any | None) -> None:
        """Close PyMuPDF document and release resources.

        Args:
            pymupdf_doc: PyMuPDF document to close
        """
        if pymupdf_doc is not None:
            try:
                pymupdf_doc.close()
            except Exception as e:
                logger.debug("Error closing PyMuPDF document: %s", e)
            gc.collect()

    def scale_blocks_for_recognition(
        self,
        blocks: list[Any],
        scale_factor: float,
    ) -> list[Any]:
        """Scale block bounding boxes for different resolution.

        Args:
            blocks: List of blocks to scale
            scale_factor: Factor to scale by (recognition_dpi / detection_dpi)

        Returns:
            Scaled blocks
        """
        from dataclasses import replace

        from .types import BBox

        scaled_blocks = []
        for block in blocks:
            if block.bbox is not None:
                scaled_bbox = BBox(
                    x0=int(block.bbox.x0 * scale_factor),
                    y0=int(block.bbox.y0 * scale_factor),
                    x1=int(block.bbox.x1 * scale_factor),
                    y1=int(block.bbox.y1 * scale_factor),
                )
                scaled_blocks.append(replace(block, bbox=scaled_bbox))
            else:
                scaled_blocks.append(block)
        return scaled_blocks

