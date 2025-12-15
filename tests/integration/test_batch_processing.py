"""Tests for staged batch processing."""

from __future__ import annotations

from pathlib import Path

from pipeline.batch import PageInfo, StagedBatchProcessor
from pipeline.batch.types import BatchProgress
from pipeline.config import PipelineConfig


def test_page_info_creation():
    """Test PageInfo dataclass creation."""
    page_info = PageInfo(
        pdf_path=Path("test.pdf"),
        page_num=1,
        total_pages=10,
    )

    assert page_info.pdf_path == Path("test.pdf")
    assert page_info.page_num == 1
    assert page_info.total_pages == 10
    assert page_info.status == "pending"
    assert page_info.image is None
    assert page_info.blocks is None


def test_page_info_properties():
    """Test PageInfo properties."""
    page_info = PageInfo(
        pdf_path=Path("document.pdf"),
        page_num=5,
        total_pages=20,
    )

    assert page_info.pdf_stem == "document"
    assert page_info.page_id == "document_page_5"


def test_page_info_mark_completed():
    """Test marking page as completed."""
    page_info = PageInfo(
        pdf_path=Path("test.pdf"),
        page_num=1,
        total_pages=10,
    )

    page_info.mark_completed()

    assert page_info.status == "completed"
    assert page_info.error is None


def test_page_info_mark_failed():
    """Test marking page as failed."""
    page_info = PageInfo(
        pdf_path=Path("test.pdf"),
        page_num=1,
        total_pages=10,
    )

    page_info.mark_failed("Test error")

    assert page_info.status == "failed"
    assert page_info.error == "Test error"


def test_batch_progress_creation():
    """Test BatchProgress dataclass creation."""
    progress = BatchProgress(total_pages=100)

    assert progress.total_pages == 100
    assert progress.completed_pages == 0
    assert progress.failed_pages == 0
    assert progress.current_stage == 0
    assert progress.stage_name == "Not Started"


def test_batch_progress_properties():
    """Test BatchProgress properties."""
    progress = BatchProgress(total_pages=100)
    progress.update(1, "Conversion", 50, 5)

    assert progress.current_stage == 1
    assert progress.stage_name == "Conversion"
    assert progress.completed_pages == 50
    assert progress.failed_pages == 5
    assert progress.progress_pct == 50.0
    assert not progress.is_complete


def test_batch_progress_completion():
    """Test BatchProgress completion detection."""
    progress = BatchProgress(total_pages=100)
    progress.update(5, "Output", 95, 5)

    assert progress.is_complete


def test_staged_batch_processor_initialization():
    """Test StagedBatchProcessor initialization."""
    from pipeline import Pipeline

    config = PipelineConfig(recognizer="gemini-2.5-flash")
    pipeline = Pipeline(config=config)
    processor = StagedBatchProcessor(pipeline)

    assert processor.pipeline == pipeline
    assert processor.progress is None


def test_staged_batch_processor_directory_not_found():
    """Test StagedBatchProcessor with non-existent directory."""
    from pipeline import Pipeline

    config = PipelineConfig(recognizer="gemini-2.5-flash")
    pipeline = Pipeline(config=config)
    processor = StagedBatchProcessor(pipeline)

    result = processor.process_directory(
        Path("/nonexistent/directory"),
        "output",
    )

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_pipeline_process_directory_uses_staged_processor(tmp_path):
    """Test that Pipeline.process_directory uses StagedBatchProcessor."""
    from pipeline import Pipeline

    # Create empty directory (no PDFs)
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    config = PipelineConfig(recognizer="gemini-2.5-flash")
    pipeline = Pipeline(config=config)

    result = pipeline.process_directory(
        input_dir,
        str(tmp_path / "output"),
    )

    # Should get error for no PDF files (proves StagedBatchProcessor is being used)
    assert "error" in result
    assert "No PDF files found" in result["error"]
