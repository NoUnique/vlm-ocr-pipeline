"""Tests for Markdown conversion utilities."""

from __future__ import annotations

import pytest

from pipeline.conversion.output.markdown import (
    RegionTypeHeaderIdentifier,
    block_to_markdown,
    blocks_to_markdown,
    document_to_markdown,
    page_to_markdown,
)
from pipeline.types import BBox, Block, Document, Page


class TestRegionTypeHeaderIdentifier:
    """Tests for RegionTypeHeaderIdentifier."""

    def test_get_header_level_title(self):
        """Test getting header level for title."""
        identifier = RegionTypeHeaderIdentifier()
        assert identifier.get_header_level("title") == 1
        assert identifier.get_header_level("Title") == 1  # Case insensitive

    def test_get_header_level_subtitle(self):
        """Test getting header level for subtitle."""
        identifier = RegionTypeHeaderIdentifier()
        assert identifier.get_header_level("subtitle") == 2

    def test_get_header_level_non_header(self):
        """Test getting header level for non-header type."""
        identifier = RegionTypeHeaderIdentifier()
        assert identifier.get_header_level("text") is None
        assert identifier.get_header_level("table") is None

    def test_get_header_prefix_title(self):
        """Test getting header prefix for title."""
        identifier = RegionTypeHeaderIdentifier()
        assert identifier.get_header_prefix("title") == "# "

    def test_get_header_prefix_subtitle(self):
        """Test getting header prefix for subtitle."""
        identifier = RegionTypeHeaderIdentifier()
        assert identifier.get_header_prefix("subtitle") == "## "

    def test_get_header_prefix_non_header(self):
        """Test getting header prefix for non-header type."""
        identifier = RegionTypeHeaderIdentifier()
        assert identifier.get_header_prefix("text") == ""


class TestBlockToMarkdown:
    """Tests for block_to_markdown function."""

    def test_title_block(self):
        """Test converting title block."""
        block = Block(
            type="title",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Introduction",
        )
        result = block_to_markdown(block)
        assert result == "# Introduction"

    def test_text_block(self):
        """Test converting text block."""
        block = Block(
            type="text",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="This is body text.",
        )
        result = block_to_markdown(block)
        assert result == "This is body text."

    def test_plain_text_block(self):
        """Test converting plain text block (legacy)."""
        block = Block(
            type="plain text",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Plain text content.",
        )
        result = block_to_markdown(block)
        assert result == "Plain text content."

    def test_list_block(self):
        """Test converting list block."""
        block = Block(
            type="list",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="First item",
        )
        result = block_to_markdown(block)
        assert result == "- First item"

    def test_list_block_already_formatted(self):
        """Test converting list block that's already formatted."""
        block = Block(
            type="list",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="- Already formatted",
        )
        result = block_to_markdown(block)
        assert result == "- Already formatted"

    def test_table_block_with_markdown(self):
        """Test converting table block with markdown table."""
        block = Block(
            type="table",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="| Header |\n| ------ |\n| Cell   |",
        )
        result = block_to_markdown(block)
        assert "| Header |" in result

    def test_table_block_without_markdown(self):
        """Test converting table block without markdown table."""
        block = Block(
            type="table",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Simple table text",
        )
        result = block_to_markdown(block)
        assert result == "**Table:**\n\nSimple table text"

    def test_table_caption(self):
        """Test converting table caption."""
        block = Block(
            type="table_caption",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Table 1: Results",
        )
        result = block_to_markdown(block)
        assert result == "**Table:** Table 1: Results"

    def test_table_footnote(self):
        """Test converting table footnote."""
        block = Block(
            type="table_footnote",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="n=100",
        )
        result = block_to_markdown(block)
        assert result == "*n=100*"

    def test_image_block(self):
        """Test converting image block."""
        block = Block(
            type="image",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Image description",
        )
        result = block_to_markdown(block)
        assert result == "**Figure:** Image description"

    def test_image_caption(self):
        """Test converting image caption."""
        block = Block(
            type="image_caption",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Figure 1: Chart",
        )
        result = block_to_markdown(block)
        assert result == "**Figure:** Figure 1: Chart"

    def test_image_footnote(self):
        """Test converting image footnote."""
        block = Block(
            type="image_footnote",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Source: 2024",
        )
        result = block_to_markdown(block)
        assert result == "*Source: 2024*"

    def test_interline_equation(self):
        """Test converting interline equation."""
        block = Block(
            type="interline_equation",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="E = mc^2",
        )
        result = block_to_markdown(block)
        assert result == "$$E = mc^2$$"

    def test_interline_equation_already_formatted(self):
        """Test converting interline equation that's already formatted."""
        block = Block(
            type="interline_equation",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="$$E = mc^2$$",
        )
        result = block_to_markdown(block)
        assert result == "$$E = mc^2$$"

    def test_inline_equation(self):
        """Test converting inline equation."""
        block = Block(
            type="inline_equation",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="x^2 + y^2",
        )
        result = block_to_markdown(block)
        assert result == "$x^2 + y^2$"

    def test_code_block(self):
        """Test converting code block."""
        block = Block(
            type="code",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="def hello():\n    print('Hello')",
        )
        result = block_to_markdown(block)
        assert result.startswith("```")
        assert "def hello():" in result
        assert result.endswith("```")

    def test_algorithm_block(self):
        """Test converting algorithm block."""
        block = Block(
            type="algorithm",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Algorithm 1: Sort\nInput: array\nOutput: sorted array",
        )
        result = block_to_markdown(block)
        assert result.startswith("```")

    def test_code_caption(self):
        """Test converting code caption."""
        block = Block(
            type="code_caption",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Listing 1: Example",
        )
        result = block_to_markdown(block)
        assert result == "**Code:** Listing 1: Example"

    def test_header_element(self):
        """Test converting header element (skipped)."""
        block = Block(
            type="header",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Page Header",
        )
        result = block_to_markdown(block)
        assert result == ""  # Headers are skipped

    def test_footer_element(self):
        """Test converting footer element (skipped)."""
        block = Block(
            type="footer",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Page Footer",
        )
        result = block_to_markdown(block)
        assert result == ""  # Footers are skipped

    def test_page_number_element(self):
        """Test converting page number element (skipped)."""
        block = Block(
            type="page_number",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="1",
        )
        result = block_to_markdown(block)
        assert result == ""  # Page numbers are skipped

    def test_discarded_block(self):
        """Test converting discarded block (skipped)."""
        block = Block(
            type="discarded",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Discarded content",
        )
        result = block_to_markdown(block)
        assert result == ""  # Discarded blocks are skipped

    def test_ref_text(self):
        """Test converting reference text."""
        block = Block(
            type="ref_text",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="[1] Smith et al., 2024",
        )
        result = block_to_markdown(block)
        assert result == "[1] Smith et al., 2024"

    def test_page_footnote(self):
        """Test converting page footnote."""
        block = Block(
            type="page_footnote",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="This is a footnote",
        )
        result = block_to_markdown(block)
        assert result == "*This is a footnote*"

    def test_aside_text(self):
        """Test converting aside text."""
        block = Block(
            type="aside_text",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Note: See appendix",
        )
        result = block_to_markdown(block)
        assert result == "*Note: See appendix*"

    def test_empty_text(self):
        """Test converting block with empty text."""
        block = Block(
            type="text",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="",
        )
        result = block_to_markdown(block)
        assert result == ""

    def test_corrected_text_priority(self):
        """Test that corrected_text takes priority over text."""
        block = Block(
            type="text",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Original text",
            corrected_text="Corrected text",
        )
        result = block_to_markdown(block)
        assert result == "Corrected text"


class TestBlocksToMarkdown:
    """Tests for blocks_to_markdown function."""

    def test_multiple_blocks(self):
        """Test converting multiple blocks to markdown."""
        blocks = [
            Block(
                type="title",
                bbox=BBox(0, 0, 100, 20),
                detection_confidence=0.9,
                order=0,
                text="Title",
            ),
            Block(
                type="text",
                bbox=BBox(0, 30, 100, 50),
                detection_confidence=0.9,
                order=1,
                text="Paragraph 1",
            ),
            Block(
                type="text",
                bbox=BBox(0, 60, 100, 80),
                detection_confidence=0.9,
                order=2,
                text="Paragraph 2",
            ),
        ]
        result = blocks_to_markdown(blocks)
        assert "# Title" in result
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result
        # Check proper spacing (double newline between blocks)
        assert "\n\n" in result

    def test_empty_blocks_list(self):
        """Test converting empty blocks list."""
        result = blocks_to_markdown([])
        assert result == ""

    def test_blocks_with_skipped_types(self):
        """Test converting blocks with some skipped types."""
        blocks = [
            Block(
                type="title",
                bbox=BBox(0, 0, 100, 20),
                detection_confidence=0.9,
                order=0,
                text="Title",
            ),
            Block(
                type="header",
                bbox=BBox(0, 5, 100, 10),
                detection_confidence=0.9,
                order=1,
                text="Page Header",
            ),  # Skipped
            Block(
                type="text",
                bbox=BBox(0, 30, 100, 50),
                detection_confidence=0.9,
                order=2,
                text="Content",
            ),
        ]
        result = blocks_to_markdown(blocks)
        assert "# Title" in result
        assert "Content" in result
        assert "Page Header" not in result


class TestPageToMarkdown:
    """Tests for page_to_markdown function."""

    def test_page_to_markdown(self):
        """Test converting Page object to markdown."""
        page = Page(
            page_num=1,
            blocks=[
                Block(
                    type="title",
                    bbox=BBox(0, 0, 100, 20),
                    detection_confidence=0.9,
                    order=0,
                    text="Page Title",
                ),
                Block(
                    type="text",
                    bbox=BBox(0, 30, 100, 50),
                    detection_confidence=0.9,
                    order=1,
                    text="Page content",
                ),
            ],
            auxiliary_info={"text": "raw text"},
            status="completed",
        )
        result = page_to_markdown(page)
        assert "# Page Title" in result
        assert "Page content" in result

    def test_page_with_empty_blocks(self):
        """Test converting Page with empty blocks."""
        page = Page(
            page_num=1,
            blocks=[],
            auxiliary_info={},
            status="completed",
        )
        result = page_to_markdown(page)
        # Even with empty blocks, page header is included
        assert result == "## Page 1"


class TestDocumentToMarkdown:
    """Tests for document_to_markdown function."""

    def test_document_to_markdown(self):
        """Test converting Document object to markdown."""
        document = Document(
            pdf_name="test",
            pdf_path="/path/to/test.pdf",
            num_pages=2,
            processed_pages=2,
            pages=[
                Page(
                    page_num=1,
                    blocks=[
                        Block(
                            type="title",
                            bbox=BBox(0, 0, 100, 20),
                            detection_confidence=0.9,
                            order=0,
                            text="Page 1 Title",
                        ),
                    ],
                    auxiliary_info={},
                    status="completed",
                ),
                Page(
                    page_num=2,
                    blocks=[
                        Block(
                            type="text",
                            bbox=BBox(0, 0, 100, 20),
                            detection_confidence=0.9,
                            order=0,
                            text="Page 2 content",
                        ),
                    ],
                    auxiliary_info={},
                    status="completed",
                ),
            ],
            detected_by="detector",
            ordered_by="sorter",
            recognized_by="recognizer",
            rendered_by="markdown",
            output_directory="/output",
            processed_at="2024-01-01T00:00:00Z",
        )
        result = document_to_markdown(document)
        assert "# Page 1 Title" in result
        assert "Page 2 content" in result

    def test_document_with_empty_pages(self):
        """Test converting Document with empty pages."""
        document = Document(
            pdf_name="test",
            pdf_path="/path/to/test.pdf",
            num_pages=0,
            processed_pages=0,
            pages=[],
            detected_by="detector",
            ordered_by="sorter",
            recognized_by="recognizer",
            rendered_by="markdown",
            output_directory="/output",
            processed_at="2024-01-01T00:00:00Z",
        )
        result = document_to_markdown(document)
        # Even with empty pages, metadata header is included
        assert "# Document Information" in result
        assert "**Source:** test" in result
        assert "**Total pages:** 0" in result


class TestLegacyTypes:
    """Tests for legacy block types (MinerU DocLayout-YOLO)."""

    def test_isolate_formula(self):
        """Test converting isolate_formula (legacy)."""
        block = Block(
            type="isolate_formula",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="E = mc^2",
        )
        result = block_to_markdown(block)
        assert result == "$$E = mc^2$$"

    def test_formula_caption(self):
        """Test converting formula_caption (legacy)."""
        block = Block(
            type="formula_caption",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Equation 1",
        )
        result = block_to_markdown(block)
        assert result == "*Formula: Equation 1*"

    def test_figure_caption_legacy(self):
        """Test converting figure_caption (legacy)."""
        block = Block(
            type="figure_caption",
            bbox=BBox(0, 0, 100, 20),
            detection_confidence=0.9,
            text="Figure 1",
        )
        result = block_to_markdown(block)
        assert result == "**Figure:** Figure 1"
