"""Tests for OutputStage.

Tests cover:
- Stage initialization
- Page result building
- Page output saving
- Summary filename determination
- Pages summary building
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.stages.output_stage import OutputStage
from pipeline.types import BBox, Block, Page


class TestOutputStageInit:
    """Tests for OutputStage initialization."""

    def test_init(self, tmp_path: Path):
        """Test OutputStage initialization."""
        stage = OutputStage(temp_dir=tmp_path)
        assert stage.temp_dir == tmp_path


class TestOutputStageBuildResult:
    """Tests for OutputStage result building."""

    def test_build_page_result(self, tmp_path: Path):
        """Test building page result."""
        # Setup
        stage = OutputStage(temp_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"
        page_num = 1
        page_image = np.zeros((792, 612, 3), dtype=np.uint8)
        detected_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=None,
            )
        ]
        processed_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Text",
                corrected_text="Text",
            )
        ]
        text = "Text"
        corrected_text = "Text"
        correction_ratio = 0.0
        column_layout = None

        # Execute
        result = stage.build_page_result(
            pdf_path=pdf_path,
            page_num=page_num,
            page_image=page_image,
            detected_blocks=detected_blocks,
            processed_blocks=processed_blocks,
            text=text,
            corrected_text=corrected_text,
            correction_ratio=correction_ratio,
            column_layout=column_layout,
        )

        # Verify
        assert isinstance(result, Page)
        assert result.page_num == page_num
        assert result.status == "completed"
        assert len(result.blocks) == 1
        assert result.blocks[0].text == "Text"
        assert result.auxiliary_info is not None
        assert result.auxiliary_info["width"] == 612
        assert result.auxiliary_info["height"] == 792
        assert result.auxiliary_info["text"] == text
        assert result.auxiliary_info["corrected_text"] == corrected_text


class TestOutputStageSummary:
    """Tests for OutputStage summary operations."""

    def test_determine_summary_filename_complete(self, tmp_path: Path):
        """Test determining summary filename for complete processing."""
        stage = OutputStage(temp_dir=tmp_path)

        assert (
            stage._determine_summary_filename(
                processing_stopped=False,
                has_errors=False,
            )
            == "summary.json"
        )

    def test_determine_summary_filename_partial(self, tmp_path: Path):
        """Test determining summary filename for partial processing."""
        stage = OutputStage(temp_dir=tmp_path)

        assert (
            stage._determine_summary_filename(
                processing_stopped=False,
                has_errors=True,
            )
            == "summary_partial.json"
        )

    def test_determine_summary_filename_incomplete(self, tmp_path: Path):
        """Test determining summary filename for incomplete processing."""
        stage = OutputStage(temp_dir=tmp_path)

        assert (
            stage._determine_summary_filename(
                processing_stopped=True,
                has_errors=False,
            )
            == "summary_incomplete.json"
        )

    def test_build_pages_summary(self, tmp_path: Path):
        """Test building pages summary."""
        # Setup
        stage = OutputStage(temp_dir=tmp_path)
        processed_pages = [
            Page(
                page_num=1,
                blocks=[],
                auxiliary_info={},
                status="completed",
                processed_at="2024-01-01T00:00:00Z",
            ),
            Page(
                page_num=2,
                blocks=[],
                auxiliary_info={},
                status="failed",
                processed_at="2024-01-01T00:00:01Z",
            ),
        ]

        # Execute
        pages_summary, status_counts = stage._build_pages_summary(processed_pages)

        # Verify
        assert len(pages_summary) == 2
        assert pages_summary[0]["id"] == 1
        assert pages_summary[0]["status"] == "complete"
        assert pages_summary[1]["id"] == 2
        assert pages_summary[1]["status"] == "partial"
        assert status_counts["complete"] == 1
        assert status_counts["partial"] == 1
        assert status_counts["incomplete"] == 0


class TestOutputStageSaveOutput:
    """Tests for OutputStage output saving."""

    def test_save_page_output(self, tmp_path: Path):
        """Test saving page output."""
        stage = OutputStage(temp_dir=tmp_path)

        page = Page(
            page_num=1,
            blocks=[
                Block(
                    type="text",
                    bbox=BBox(100, 100, 200, 200),
                    order=0,
                    text="Hello",
                    corrected_text="Hello",
                )
            ],
            auxiliary_info={"width": 612, "height": 792, "text": "# Hello\n\nWorld"},
            status="completed",
            processed_at="2024-01-01T00:00:00Z",
        )

        stage.save_page_output(
            page_output_dir=tmp_path,
            page_num=1,
            page=page,
        )

        # Verify JSON file created
        json_file = tmp_path / "json" / "page_1.json"
        assert json_file.exists()

        # Verify markdown file created
        md_file = tmp_path / "page_1.md"
        assert md_file.exists()

    def test_save_page_output_with_corrected_text(self, tmp_path: Path):
        """Test saving page output with corrected text."""
        stage = OutputStage(temp_dir=tmp_path)

        page = Page(
            page_num=1,
            blocks=[],
            auxiliary_info={"text": "Original", "corrected_text": "Corrected"},
            status="completed",
            processed_at="2024-01-01T00:00:00Z",
        )

        stage.save_page_output(
            page_output_dir=tmp_path,
            page_num=1,
            page=page,
        )

        # Verify markdown file uses corrected_text
        md_file = tmp_path / "page_1.md"
        assert md_file.exists()
        assert md_file.read_text() == "Corrected"


class TestOutputStageCreateSummary:
    """Tests for OutputStage summary creation."""

    def test_create_pdf_summary(self, tmp_path: Path):
        """Test creating PDF summary."""
        from pipeline.types import Document

        stage = OutputStage(temp_dir=tmp_path)

        processed_pages = [
            Page(
                page_num=1,
                blocks=[],
                auxiliary_info={},
                status="completed",
                processed_at="2024-01-01T00:00:00Z",
            ),
        ]

        document = stage.create_pdf_summary(
            pdf_path=tmp_path / "test.pdf",
            total_pages=1,
            processed_pages=processed_pages,
            processing_stopped=False,
            summary_output_dir=tmp_path,
            detector_name="doclayout-yolo",
            sorter_name="mineru-xycut",
            backend="gemini",
            model="gemini-2.5-flash",
            renderer="markdown",
        )

        assert isinstance(document, Document)
        assert document.pdf_name == "test"
        assert document.num_pages == 1
        assert document.processed_pages == 1
        assert document.detected_by == "doclayout-yolo"
        assert document.ordered_by == "mineru-xycut"
        assert document.recognized_by == "gemini/gemini-2.5-flash"
        assert document.rendered_by == "markdown"

        # Verify summary file created
        summary_file = tmp_path / "summary.json"
        assert summary_file.exists()

