"""Tests for PipelineResult and StageTimingInfo.

Tests cover:
- StageTimingInfo creation and properties
- PipelineResult creation and methods
- Serialization to dict
"""

from __future__ import annotations

import pytest

from pipeline.types import (
    BBox,
    Block,
    Document,
    Page,
    PipelineResult,
    StageTimingInfo,
)


class TestStageTimingInfo:
    """Tests for StageTimingInfo dataclass."""

    def test_creation(self):
        """Test basic creation."""
        timing = StageTimingInfo(
            stage_name="detection",
            processing_time_ms=150.5,
            items_processed=10,
        )
        assert timing.stage_name == "detection"
        assert timing.processing_time_ms == 150.5
        assert timing.items_processed == 10

    def test_default_items_processed(self):
        """Test default value for items_processed."""
        timing = StageTimingInfo(
            stage_name="ordering",
            processing_time_ms=50.0,
        )
        assert timing.items_processed == 0

    def test_processing_time_sec_property(self):
        """Test processing_time_sec property."""
        timing = StageTimingInfo(
            stage_name="recognition",
            processing_time_ms=2500.0,
        )
        assert timing.processing_time_sec == 2.5

    def test_zero_processing_time(self):
        """Test with zero processing time."""
        timing = StageTimingInfo(
            stage_name="fast-stage",
            processing_time_ms=0.0,
        )
        assert timing.processing_time_sec == 0.0


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample Document for testing."""
        pages = [
            Page(
                page_num=1,
                blocks=[Block(type="text", bbox=BBox(0, 0, 100, 100), text="Hello")],
            )
        ]
        return Document(
            pdf_name="test",
            pdf_path="/path/to/test.pdf",
            num_pages=1,
            processed_pages=1,
            pages=pages,
        )

    @pytest.fixture
    def sample_timings(self):
        """Create sample stage timings."""
        return [
            StageTimingInfo("detection", 100.0, 5),
            StageTimingInfo("ordering", 50.0, 5),
            StageTimingInfo("recognition", 500.0, 5),
            StageTimingInfo("rendering", 20.0, 5),
        ]

    def test_creation_success(self, sample_document, sample_timings):
        """Test successful result creation."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=sample_timings,
            total_time_ms=1000.0,
            success=True,
        )
        assert result.document == sample_document
        assert len(result.stage_timings) == 4
        assert result.total_time_ms == 1000.0
        assert result.success is True
        assert result.error is None

    def test_creation_failure(self):
        """Test failed result creation."""
        result = PipelineResult(
            document=None,
            stage_timings=[StageTimingInfo("detection", 100.0)],
            total_time_ms=100.0,
            success=False,
            error="Detection failed: model not found",
        )
        assert result.document is None
        assert result.success is False
        assert result.error == "Detection failed: model not found"

    def test_total_time_sec_property(self, sample_document, sample_timings):
        """Test total_time_sec property."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=sample_timings,
            total_time_ms=5000.0,
        )
        assert result.total_time_sec == 5.0

    def test_get_stage_timings(self, sample_document, sample_timings):
        """Test get_stage_timings method."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=sample_timings,
            total_time_ms=1000.0,
        )
        timings = result.get_stage_timings()

        assert timings["detection"] == 100.0
        assert timings["ordering"] == 50.0
        assert timings["recognition"] == 500.0
        assert timings["rendering"] == 20.0

    def test_get_stage_timings_empty(self, sample_document):
        """Test get_stage_timings with no stages."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=[],
            total_time_ms=0.0,
        )
        timings = result.get_stage_timings()
        assert timings == {}

    def test_get_slowest_stage(self, sample_document, sample_timings):
        """Test get_slowest_stage method."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=sample_timings,
            total_time_ms=1000.0,
        )
        slowest = result.get_slowest_stage()

        assert slowest is not None
        assert slowest.stage_name == "recognition"
        assert slowest.processing_time_ms == 500.0

    def test_get_slowest_stage_empty(self, sample_document):
        """Test get_slowest_stage with no stages."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=[],
            total_time_ms=0.0,
        )
        assert result.get_slowest_stage() is None

    def test_to_dict_success(self, sample_document, sample_timings):
        """Test to_dict for successful result."""
        result = PipelineResult(
            document=sample_document,
            stage_timings=sample_timings,
            total_time_ms=1000.0,
            success=True,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["total_time_ms"] == 1000.0
        assert data["total_time_sec"] == 1.0
        assert "stage_timings" in data
        assert data["stage_timings"]["detection"] == 100.0
        assert "document" in data
        assert data["document"]["pdf_name"] == "test"
        assert "error" not in data

    def test_to_dict_failure(self):
        """Test to_dict for failed result."""
        result = PipelineResult(
            document=None,
            stage_timings=[StageTimingInfo("detection", 100.0)],
            total_time_ms=100.0,
            success=False,
            error="Failed to process",
        )
        data = result.to_dict()

        assert data["success"] is False
        assert data["error"] == "Failed to process"
        assert "document" not in data

    def test_to_dict_with_all_stages(self):
        """Test to_dict with all pipeline stages."""
        timings = [
            StageTimingInfo("input", 50.0, 1),
            StageTimingInfo("detection", 200.0, 10),
            StageTimingInfo("ordering", 30.0, 10),
            StageTimingInfo("recognition", 800.0, 10),
            StageTimingInfo("block_correction", 100.0, 10),
            StageTimingInfo("rendering", 20.0, 1),
            StageTimingInfo("page_correction", 150.0, 1),
            StageTimingInfo("output", 10.0, 1),
        ]
        result = PipelineResult(
            document=None,
            stage_timings=timings,
            total_time_ms=1360.0,
            success=True,
        )
        data = result.to_dict()

        assert len(data["stage_timings"]) == 8
        assert data["stage_timings"]["input"] == 50.0
        assert data["stage_timings"]["output"] == 10.0


class TestPipelineResultIntegration:
    """Integration tests for PipelineResult with real data."""

    def test_full_pipeline_result(self):
        """Test creating a full pipeline result with all components."""
        # Create blocks
        blocks = [
            Block(type="title", bbox=BBox(100, 50, 500, 100), text="Document Title", order=0),
            Block(type="text", bbox=BBox(100, 120, 500, 300), text="Paragraph content.", order=1),
        ]

        # Create page
        page = Page(
            page_num=1,
            blocks=blocks,
            auxiliary_info={"text": "# Document Title\n\nParagraph content."},
            status="completed",
        )

        # Create document
        document = Document(
            pdf_name="sample",
            pdf_path="/docs/sample.pdf",
            num_pages=1,
            processed_pages=1,
            pages=[page],
            detected_by="doclayout-yolo",
            ordered_by="pymupdf",
            recognized_by="gemini/gemini-2.5-flash",
            rendered_by="markdown",
        )

        # Create timings
        timings = [
            StageTimingInfo("detection", 150.0, 2),
            StageTimingInfo("ordering", 25.0, 2),
            StageTimingInfo("recognition", 450.0, 2),
            StageTimingInfo("rendering", 10.0, 1),
        ]

        # Create result
        result = PipelineResult(
            document=document,
            stage_timings=timings,
            total_time_ms=635.0,
            success=True,
        )

        # Verify
        assert result.success
        assert result.document is not None
        assert result.document.pdf_name == "sample"
        assert len(result.document.pages) == 1
        assert len(result.document.pages[0].blocks) == 2

        # Check timings
        assert result.get_slowest_stage().stage_name == "recognition"
        assert result.total_time_sec == 0.635

        # Check serialization
        data = result.to_dict()
        assert data["success"] is True
        assert "document" in data
        assert data["document"]["detected_by"] == "doclayout-yolo"

