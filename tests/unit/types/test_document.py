"""Tests for Document and Page classes.

Tests cover:
- Page creation and serialization
- Document creation and serialization
"""

from __future__ import annotations

from pipeline.misc import tz_now
from pipeline.types import BBox, Block, Document, Page


class TestPageDataclass:
    """Test Page dataclass functionality."""

    def test_page_creation(self):
        """Test creating a Page object with blocks."""
        blocks = [
            Block(type="text", bbox=BBox(100, 50, 300, 200), text="Hello"),
            Block(type="title", bbox=BBox(50, 10, 250, 40), text="Title"),
        ]

        page = Page(
            page_num=1,
            blocks=blocks,
            status="completed",
        )

        assert page.page_num == 1
        assert len(page.blocks) == 2
        assert page.status == "completed"
        assert page.auxiliary_info is None

    def test_page_to_dict(self):
        """Test Page.to_dict() converts blocks correctly."""
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 50, 300, 200),
                text="Hello",
                order=0,
            ),
        ]

        page = Page(
            page_num=1,
            blocks=blocks,
            auxiliary_info={"text": "Hello"},
            status="completed",
        )

        result = page.to_dict()

        assert result["page_num"] == 1
        assert result["status"] == "completed"
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["type"] == "text"
        assert result["blocks"][0]["xywh"] == [100, 50, 200, 150]  # xywh format
        assert result["auxiliary_info"]["text"] == "Hello"

    def test_page_with_failed_status(self):
        """Test creating a failed Page."""
        page = Page(
            page_num=5,
            blocks=[],
            status="failed",
            auxiliary_info={"error": "Rate limit exceeded"},
        )

        assert page.status == "failed"
        assert page.blocks == []
        assert page.auxiliary_info is not None
        assert "error" in page.auxiliary_info


class TestDocumentDataclass:
    """Test Document dataclass functionality."""

    def test_document_creation(self):
        """Test creating a Document object with pages."""
        blocks1 = [Block(type="text", bbox=BBox(10, 10, 100, 50), text="Page 1")]
        blocks2 = [Block(type="text", bbox=BBox(20, 20, 120, 60), text="Page 2")]

        page1 = Page(page_num=1, blocks=blocks1, status="completed")
        page2 = Page(page_num=2, blocks=blocks2, status="completed")

        document = Document(
            pdf_name="test",
            pdf_path="/path/to/test.pdf",
            num_pages=2,
            processed_pages=2,
            pages=[page1, page2],
        )

        assert document.pdf_name == "test"
        assert document.num_pages == 2
        assert len(document.pages) == 2

    def test_document_with_metadata(self):
        """Test Document with all metadata fields."""
        page = Page(page_num=1, blocks=[], status="completed")

        document = Document(
            pdf_name="sample",
            pdf_path="/docs/sample.pdf",
            num_pages=10,
            processed_pages=5,
            pages=[page],
            detected_by="doclayout-yolo",
            ordered_by="mineru-xycut",
            recognized_by="gemini/gemini-2.5-flash",
            rendered_by="markdown",
            output_directory="/output/gemini-2.5-flash/sample",
            processed_at=tz_now().isoformat(),
            status_summary={"completed": 5},
        )

        assert document.detected_by == "doclayout-yolo"
        assert document.ordered_by == "mineru-xycut"
        assert document.recognized_by == "gemini/gemini-2.5-flash"
        assert document.rendered_by == "markdown"
        assert document.status_summary is not None
        assert document.status_summary["completed"] == 5

    def test_document_to_dict(self):
        """Test Document.to_dict() includes all fields."""
        blocks = [Block(type="text", bbox=BBox(10, 10, 100, 50), text="Test")]
        page = Page(page_num=1, blocks=blocks, status="completed")

        document = Document(
            pdf_name="test",
            pdf_path="/test.pdf",
            num_pages=1,
            processed_pages=1,
            pages=[page],
            detected_by="doclayout-yolo",
            ordered_by="pymupdf",
            recognized_by="gemini/gemini-2.5-flash",
            rendered_by="markdown",
        )

        result = document.to_dict()

        assert result["pdf_name"] == "test"
        assert result["num_pages"] == 1
        assert result["detected_by"] == "doclayout-yolo"
        assert result["ordered_by"] == "pymupdf"
        assert result["recognized_by"] == "gemini/gemini-2.5-flash"
        assert result["rendered_by"] == "markdown"
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_num"] == 1

    def test_document_to_dict_optional_fields(self):
        """Test Document.to_dict() omits None fields."""
        page = Page(page_num=1, blocks=[], status="completed")

        document = Document(
            pdf_name="minimal",
            pdf_path="/minimal.pdf",
            num_pages=1,
            processed_pages=1,
            pages=[page],
            # All optional fields are None
        )

        result = document.to_dict()

        # Required fields present
        assert "pdf_name" in result
        assert "num_pages" in result

        # Optional fields not present when None
        assert "detected_by" not in result
        assert "ordered_by" not in result
        assert "output_directory" not in result
        assert "status_summary" not in result

