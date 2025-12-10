"""Tests for PDF conversion utilities.

Tests the PDF conversion functions which handle:
- PDF metadata extraction
- PDF page rendering to images
- PyMuPDF document opening
- Text span extraction with font information
- Page selection logic
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pipeline.io.input.pdf import (
    determine_pages_to_process,
    extract_text_spans_from_pdf,
    get_pdf_info,
    open_pymupdf_document,
    render_pdf_page,
)


class TestGetPdfInfo:
    """Tests for get_pdf_info function."""

    @patch("pipeline.conversion.input.pdf.pdfinfo_from_path")
    def test_get_pdf_info_success(self, mock_pdfinfo: Mock, tmp_path: Path):
        """Test successful PDF info extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        expected_info = {
            "Pages": 10,
            "Title": "Test Document",
            "Author": "Test Author",
            "Creator": "Test Creator",
            "Producer": "Test Producer",
        }
        mock_pdfinfo.return_value = expected_info

        result = get_pdf_info(pdf_path)

        assert result == expected_info
        mock_pdfinfo.assert_called_once_with(str(pdf_path))

    @patch("pipeline.conversion.input.pdf.pdfinfo_from_path")
    def test_get_pdf_info_minimal(self, mock_pdfinfo: Mock, tmp_path: Path):
        """Test PDF info with minimal metadata."""
        pdf_path = tmp_path / "minimal.pdf"
        pdf_path.touch()

        minimal_info = {"Pages": 1}
        mock_pdfinfo.return_value = minimal_info

        result = get_pdf_info(pdf_path)

        assert result == minimal_info
        assert result["Pages"] == 1

    @patch("pipeline.conversion.input.pdf.pdfinfo_from_path")
    def test_get_pdf_info_with_unicode(self, mock_pdfinfo: Mock, tmp_path: Path):
        """Test PDF info with unicode characters."""
        pdf_path = tmp_path / "unicode.pdf"
        pdf_path.touch()

        unicode_info = {
            "Pages": 5,
            "Title": "测试文档",
            "Author": "作者",
        }
        mock_pdfinfo.return_value = unicode_info

        result = get_pdf_info(pdf_path)

        assert result["Title"] == "测试文档"
        assert result["Author"] == "作者"


class TestRenderPdfPage:
    """Tests for render_pdf_page function."""

    @patch("pipeline.conversion.input.pdf.convert_from_path")
    @patch("cv2.imwrite")
    def test_render_pdf_page_success(
        self,
        mock_imwrite: Mock,
        mock_convert: Mock,
        tmp_path: Path,
    ):
        """Test successful PDF page rendering."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        temp_dir = tmp_path / "temp"

        # Create mock PIL image
        mock_pil_image = MagicMock()
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_pil_image.__array__ = lambda: test_array
        mock_convert.return_value = [mock_pil_image]

        image, temp_image_path = render_pdf_page(pdf_path, 1, temp_dir, dpi=200)

        assert temp_dir.exists()
        assert image.shape == (100, 100, 3)
        mock_convert.assert_called_once_with(
            pdf_path,
            first_page=1,
            last_page=1,
            dpi=200,
        )
        assert temp_image_path == temp_dir / "test_page_1.jpg"

    @patch("pipeline.conversion.input.pdf.convert_from_path")
    def test_render_pdf_page_custom_dpi(self, mock_convert: Mock, tmp_path: Path):
        """Test rendering with custom DPI."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        temp_dir = tmp_path / "temp"

        mock_pil_image = MagicMock()
        test_array = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_pil_image.__array__ = lambda: test_array
        mock_convert.return_value = [mock_pil_image]

        with patch("cv2.imwrite"):
            image, _ = render_pdf_page(pdf_path, 1, temp_dir, dpi=300)

        assert image.shape == (200, 200, 3)
        mock_convert.assert_called_once_with(
            pdf_path,
            first_page=1,
            last_page=1,
            dpi=300,
        )

    @patch("pipeline.conversion.input.pdf.convert_from_path")
    def test_render_pdf_page_empty_result(self, mock_convert: Mock, tmp_path: Path):
        """Test rendering failure when no images returned."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        temp_dir = tmp_path / "temp"

        mock_convert.return_value = []

        with pytest.raises(ValueError, match="Failed to render page 1"):
            render_pdf_page(pdf_path, 1, temp_dir)

    @patch("pipeline.conversion.input.pdf.convert_from_path")
    def test_render_pdf_page_creates_temp_dir(
        self,
        mock_convert: Mock,
        tmp_path: Path,
    ):
        """Test that temp directory is created if it doesn't exist."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        temp_dir = tmp_path / "nested" / "temp"

        mock_pil_image = MagicMock()
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_pil_image.__array__ = lambda: test_array
        mock_convert.return_value = [mock_pil_image]

        with patch("cv2.imwrite"):
            _, _ = render_pdf_page(pdf_path, 1, temp_dir)

        assert temp_dir.exists()

    @patch("pipeline.conversion.input.pdf.convert_from_path")
    def test_render_pdf_page_different_pages(
        self,
        mock_convert: Mock,
        tmp_path: Path,
    ):
        """Test rendering different page numbers."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        temp_dir = tmp_path / "temp"

        mock_pil_image = MagicMock()
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_pil_image.__array__ = lambda: test_array

        for page_num in [1, 5, 10]:
            mock_convert.return_value = [mock_pil_image]
            with patch("cv2.imwrite"):
                _, temp_path = render_pdf_page(pdf_path, page_num, temp_dir)

            assert temp_path == temp_dir / f"test_page_{page_num}.jpg"


class TestOpenPymupdfDocument:
    """Tests for open_pymupdf_document function."""

    @patch("pipeline.conversion.input.pdf.fitz")
    def test_open_pymupdf_document_success(self, mock_fitz: Mock, tmp_path: Path):
        """Test successful document opening."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_fitz.open.return_value = mock_doc

        result = open_pymupdf_document(pdf_path)

        assert result == mock_doc
        mock_fitz.open.assert_called_once_with(str(pdf_path))

    def test_open_pymupdf_document_no_fitz(self, tmp_path: Path, monkeypatch):
        """Test when PyMuPDF is not available."""
        # Simulate fitz not being available
        monkeypatch.setattr("pipeline.conversion.input.pdf.fitz", None)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = open_pymupdf_document(pdf_path)

        assert result is None

    @patch("pipeline.conversion.input.pdf.fitz")
    def test_open_pymupdf_document_failure(self, mock_fitz: Mock, tmp_path: Path):
        """Test document opening failure."""
        pdf_path = tmp_path / "invalid.pdf"
        pdf_path.touch()

        mock_fitz.open.side_effect = Exception("Invalid PDF")

        result = open_pymupdf_document(pdf_path)

        assert result is None


class TestExtractTextSpansFromPdf:
    """Tests for extract_text_spans_from_pdf function."""

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_success(self, mock_open_pdf: Mock, tmp_path: Path):
        """Test successful text span extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        # Mock PyMuPDF document structure
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_page = MagicMock()

        # Create mock text dict structure
        mock_text_dict = {
            "blocks": [
                {
                    "type": 0,  # Text block
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Hello World",
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                },
                                {
                                    "text": "Test Text",
                                    "size": 14.0,
                                    "font": "Times-Bold",
                                    "bbox": (10.0, 50.0, 120.0, 70.0),
                                },
                            ]
                        }
                    ],
                }
            ]
        }

        mock_page.get_text.return_value = mock_text_dict
        mock_doc.load_page.return_value = mock_page
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert len(spans) == 2
        assert spans[0]["text"] == "Hello World"
        assert spans[0]["size"] == 12.0
        assert spans[0]["font"] == "Arial"
        assert spans[0]["bbox"] == [10, 20, 100, 40]

        assert spans[1]["text"] == "Test Text"
        assert spans[1]["size"] == 14.0
        assert spans[1]["font"] == "Times-Bold"

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_with_image_blocks(
        self,
        mock_open_pdf: Mock,
        tmp_path: Path,
    ):
        """Test that image blocks are skipped."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_page = MagicMock()

        mock_text_dict = {
            "blocks": [
                {"type": 1},  # Image block - should be skipped
                {
                    "type": 0,  # Text block
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Text content",
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                }
                            ]
                        }
                    ],
                },
            ]
        }

        mock_page.get_text.return_value = mock_text_dict
        mock_doc.load_page.return_value = mock_page
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert len(spans) == 1
        assert spans[0]["text"] == "Text content"

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_empty_text(self, mock_open_pdf: Mock, tmp_path: Path):
        """Test that empty/whitespace text is skipped."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_page = MagicMock()

        mock_text_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "   ",  # Whitespace - should be skipped
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                },
                                {
                                    "text": "",  # Empty - should be skipped
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                },
                                {
                                    "text": "Valid text",
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                },
                            ]
                        }
                    ],
                }
            ]
        }

        mock_page.get_text.return_value = mock_text_dict
        mock_doc.load_page.return_value = mock_page
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert len(spans) == 1
        assert spans[0]["text"] == "Valid text"

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_invalid_page(self, mock_open_pdf: Mock, tmp_path: Path):
        """Test extraction with invalid page number."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        # Page number too high
        spans = extract_text_spans_from_pdf(pdf_path, page_num=10)
        assert spans == []

        # Page number too low
        spans = extract_text_spans_from_pdf(pdf_path, page_num=0)
        assert spans == []

    def test_extract_text_spans_no_fitz(self, tmp_path: Path, monkeypatch):
        """Test when PyMuPDF is not available."""
        monkeypatch.setattr("pipeline.conversion.input.pdf.fitz", None)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert spans == []

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_exception_handling(
        self,
        mock_open_pdf: Mock,
        tmp_path: Path,
    ):
        """Test graceful handling of exceptions."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_open_pdf.return_value.__enter__.side_effect = Exception("Unexpected error")

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert spans == []

    @patch("pipeline.conversion.input.pdf.fitz")
    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_with_default_values(
        self,
        mock_open_pdf: Mock,
        tmp_path: Path,
    ):
        """Test spans with missing font information use defaults."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_page = MagicMock()

        mock_text_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Text without font info",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                    # Missing 'size' and 'font' keys
                                }
                            ]
                        }
                    ],
                }
            ]
        }

        mock_page.get_text.return_value = mock_text_dict
        mock_doc.load_page.return_value = mock_page
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert len(spans) == 1
        assert spans[0]["size"] == 12.0  # Default
        assert spans[0]["font"] == "Unknown"  # Default

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_unicode(self, mock_open_pdf: Mock, tmp_path: Path):
        """Test extraction with unicode text."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_page = MagicMock()

        mock_text_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "한글 텍스트",
                                    "size": 12.0,
                                    "font": "NanumGothic",
                                    "bbox": (10.0, 20.0, 100.0, 40.0),
                                },
                                {
                                    "text": "日本語",
                                    "size": 12.0,
                                    "font": "MSGothic",
                                    "bbox": (10.0, 50.0, 100.0, 70.0),
                                },
                            ]
                        }
                    ],
                }
            ]
        }

        mock_page.get_text.return_value = mock_text_dict
        mock_doc.load_page.return_value = mock_page
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert len(spans) == 2
        assert spans[0]["text"] == "한글 텍스트"
        assert spans[1]["text"] == "日本語"

    @patch("pipeline.conversion.input.pdf.open_pdf_document")
    def test_extract_text_spans_multiple_lines(self, mock_open_pdf: Mock, tmp_path: Path):
        """Test extraction from multiple lines."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_page = MagicMock()

        mock_text_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Line 1 Span 1",
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (10.0, 10.0, 100.0, 20.0),
                                },
                                {
                                    "text": "Line 1 Span 2",
                                    "size": 12.0,
                                    "font": "Arial",
                                    "bbox": (110.0, 10.0, 200.0, 20.0),
                                },
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Line 2 Span 1",
                                    "size": 14.0,
                                    "font": "Times",
                                    "bbox": (10.0, 30.0, 100.0, 45.0),
                                }
                            ]
                        },
                    ],
                }
            ]
        }

        mock_page.get_text.return_value = mock_text_dict
        mock_doc.load_page.return_value = mock_page
        mock_open_pdf.return_value.__enter__.return_value = mock_doc

        spans = extract_text_spans_from_pdf(pdf_path, page_num=1)

        assert len(spans) == 3
        assert spans[0]["text"] == "Line 1 Span 1"
        assert spans[1]["text"] == "Line 1 Span 2"
        assert spans[2]["text"] == "Line 2 Span 1"


class TestDeterminePagesToProcess:
    """Tests for determine_pages_to_process function."""

    def test_determine_pages_all(self):
        """Test processing all pages."""
        pages = determine_pages_to_process(total_pages=10)

        assert pages == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_determine_pages_max_pages(self):
        """Test processing with max_pages limit."""
        pages = determine_pages_to_process(total_pages=100, max_pages=5)

        assert pages == [1, 2, 3, 4, 5]

    def test_determine_pages_max_pages_exceeds_total(self):
        """Test max_pages larger than total."""
        pages = determine_pages_to_process(total_pages=3, max_pages=10)

        assert pages == [1, 2, 3]

    def test_determine_pages_page_range(self):
        """Test processing specific page range."""
        pages = determine_pages_to_process(total_pages=100, page_range=(10, 15))

        assert pages == [10, 11, 12, 13, 14, 15]

    def test_determine_pages_page_range_start_before_1(self):
        """Test page range with start < 1."""
        pages = determine_pages_to_process(total_pages=100, page_range=(-5, 5))

        assert pages == [1, 2, 3, 4, 5]

    def test_determine_pages_page_range_end_exceeds_total(self):
        """Test page range with end > total."""
        pages = determine_pages_to_process(total_pages=10, page_range=(5, 20))

        assert pages == [5, 6, 7, 8, 9, 10]

    def test_determine_pages_specific_pages(self):
        """Test processing specific page list."""
        pages = determine_pages_to_process(total_pages=100, pages=[1, 5, 10, 20])

        assert pages == [1, 5, 10, 20]

    def test_determine_pages_specific_pages_sorted(self):
        """Test that specific pages are sorted."""
        pages = determine_pages_to_process(total_pages=100, pages=[20, 5, 1, 10])

        assert pages == [1, 5, 10, 20]

    def test_determine_pages_specific_pages_invalid(self, caplog):
        """Test specific pages with invalid page numbers."""
        pages = determine_pages_to_process(
            total_pages=10,
            pages=[1, 5, 15, 20],  # 15, 20 are invalid
        )

        assert pages == [1, 5]
        assert "Invalid page numbers" in caplog.text

    def test_determine_pages_priority_specific_pages(self):
        """Test that specific pages takes priority over range."""
        pages = determine_pages_to_process(
            total_pages=100,
            max_pages=5,
            page_range=(10, 20),
            pages=[1, 50, 100],
        )

        # Should use pages, not range or max_pages
        assert pages == [1, 50, 100]

    def test_determine_pages_priority_page_range(self):
        """Test that page range takes priority over max_pages."""
        pages = determine_pages_to_process(
            total_pages=100,
            max_pages=5,
            page_range=(10, 20),
        )

        # Should use page_range, not max_pages
        assert pages == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def test_determine_pages_empty_document(self):
        """Test with document that has 0 pages."""
        pages = determine_pages_to_process(total_pages=0)

        assert pages == []

    def test_determine_pages_single_page(self):
        """Test with single page document."""
        pages = determine_pages_to_process(total_pages=1)

        assert pages == [1]

    def test_determine_pages_max_pages_zero(self):
        """Test with max_pages=0."""
        pages = determine_pages_to_process(total_pages=10, max_pages=0)

        assert pages == []

    def test_determine_pages_empty_specific_pages(self):
        """Test with empty specific pages list."""
        pages = determine_pages_to_process(total_pages=10, pages=[])

        assert pages == []
