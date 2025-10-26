"""PDF document conversion utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pdf2image.pdf2image import convert_from_path, pdfinfo_from_path

from ...resources import open_pdf_document

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    fitz = None  # type: ignore


def get_pdf_info(pdf_path: Path) -> dict[str, Any]:
    """Get PDF document information.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing PDF metadata including page count

    Example:
        >>> info = get_pdf_info(Path("document.pdf"))
        >>> info["Pages"]
        10
    """
    return pdfinfo_from_path(str(pdf_path))


def render_pdf_page(
    pdf_path: Path,
    page_num: int,
    temp_dir: Path,
    dpi: int = 200,
) -> tuple[np.ndarray, Path]:
    """Render a PDF page to an image.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to render (1-indexed)
        temp_dir: Directory for temporary files
        dpi: DPI for rendering (default: 200)

    Returns:
        Tuple of (image array, temporary file path)

    Raises:
        ValueError: If page_num is invalid or rendering fails

    Example:
        >>> temp_dir = Path("/tmp/pdf_temp")
        >>> image, temp_path = render_pdf_page(
        ...     Path("doc.pdf"), page_num=1, temp_dir=temp_dir, dpi=200
        ... )
        >>> image.shape
        (1650, 1275, 3)
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=dpi,
    )

    if not images:
        raise ValueError(f"Failed to render page {page_num} from {pdf_path}")

    page_image = np.array(images[0])

    temp_image_path = temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"
    cv2.imwrite(str(temp_image_path), cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR))
    logger.info("Rendered PDF page to: %s", temp_image_path)

    return page_image, temp_image_path


def open_pymupdf_document(pdf_path: Path) -> Any | None:
    """Open a PDF document using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        PyMuPDF document object or None if PyMuPDF is not available

    Example:
        >>> doc = open_pymupdf_document(Path("document.pdf"))
        >>> if doc:
        ...     print(f"Pages: {doc.page_count}")
        ...     doc.close()
    """
    if fitz is None:
        logger.warning("PyMuPDF (fitz) is not available")
        return None

    try:
        return fitz.open(str(pdf_path))
    except Exception as exc:  # pragma: no cover - PyMuPDF failure path
        logger.warning(
            "Failed to open PDF with PyMuPDF for multi-column ordering: %s",
            exc,
        )
        return None


def extract_text_spans_from_pdf(
    pdf_path: Path,
    page_num: int,
) -> list[dict[str, Any]]:
    """Extract text spans with font information from a PDF page.

    This function uses PyMuPDF to parse the PDF and extract actual text objects
    (spans) with their font properties. Uses PyMuPDF terminology.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)

    Returns:
        List of text spans with bbox, text, size, font (PyMuPDF terminology)

    Example:
        >>> spans = extract_text_spans_from_pdf(Path("doc.pdf"), page_num=1)
        >>> spans[0]
        {
            'bbox': [100, 50, 300, 80],
            'text': 'Chapter 1',
            'size': 24.0,  # PyMuPDF uses 'size'
            'font': 'Times-Bold'  # PyMuPDF uses 'font'
        }
    """
    if fitz is None:
        logger.warning("PyMuPDF (fitz) not available - cannot extract text spans")
        return []

    try:
        with open_pdf_document(pdf_path) as doc:
            if page_num < 1 or page_num > doc.page_count:
                logger.error("Invalid page number: %d (total pages: %d)", page_num, doc.page_count)
                return []

            page = doc.load_page(page_num - 1)  # 0-indexed in PyMuPDF
            text_spans: list[dict[str, Any]] = []

            # Get text with detailed formatting information (PyMuPDF's dict format)
            text_dict = page.get_text("dict")  # type: ignore[attr-defined]
            blocks = text_dict["blocks"]

            for block in blocks:
                # Skip image blocks (type 0 = text, type 1 = image)
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        # Get font information (PyMuPDF keys)
                        size = span.get("size", 12.0)  # PyMuPDF uses 'size'
                        font = span.get("font", "Unknown")  # PyMuPDF uses 'font'

                        # Get bounding box (PyMuPDF uses xyxy format)
                        bbox_tuple = span.get("bbox", (0, 0, 0, 0))
                        x0, y0, x1, y1 = bbox_tuple

                        text_spans.append(
                            {
                                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                                "text": text,
                                "size": float(size),  # PyMuPDF terminology
                                "font": font,  # PyMuPDF terminology
                            }
                        )

            logger.info("Extracted %d text spans from page %d", len(text_spans), page_num)
            return text_spans

    except FileNotFoundError:
        logger.error("PDF file not found: %s", pdf_path)
        return []
    except RuntimeError as exc:
        # PyMuPDF not available
        logger.warning("Cannot extract text spans: %s", exc)
        return []
    except Exception as exc:
        # Fallback for unexpected errors - text spans are optional (allowed per ERROR_HANDLING.md section 3.3)
        logger.error("Failed to extract text spans: %s", exc, exc_info=True)
        return []


def determine_pages_to_process(
    total_pages: int,
    max_pages: int | None = None,
    page_range: tuple[int, int] | None = None,
    pages: list[int] | None = None,
) -> list[int]:
    """Determine which pages to process based on limiting options.

    Priority order:
    1. Specific pages list (if provided)
    2. Page range (if provided)
    3. Max pages (if provided)
    4. All pages (default)

    Args:
        total_pages: Total number of pages in the document
        max_pages: Maximum number of pages to process (from start)
        page_range: Range of pages to process (start, end) - both inclusive
        pages: Specific list of page numbers to process (1-indexed)

    Returns:
        List of page numbers to process (1-indexed)

    Example:
        >>> determine_pages_to_process(100, max_pages=5)
        [1, 2, 3, 4, 5]
        >>> determine_pages_to_process(100, page_range=(10, 15))
        [10, 11, 12, 13, 14, 15]
        >>> determine_pages_to_process(100, pages=[1, 5, 10])
        [1, 5, 10]
    """
    if pages is not None:
        # Specific pages specified
        valid_pages = [p for p in pages if 1 <= p <= total_pages]
        if len(valid_pages) != len(pages):
            invalid_pages = [p for p in pages if p not in valid_pages]
            logger.warning(
                "Invalid page numbers (outside 1-%d): %s",
                total_pages,
                invalid_pages,
            )
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
