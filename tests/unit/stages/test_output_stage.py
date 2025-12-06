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

