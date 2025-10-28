"""Tests for checkpoint and smart resume functionality."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def test_progress_tracker_initialization(temp_checkpoint_dir: Path) -> None:
    """Test ProgressTracker initialization."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir, input_file="test.pdf")

    assert tracker.output_dir == temp_checkpoint_dir
    assert tracker.input_file == "test.pdf"
    # Progress file created on first _save() call (e.g., start_stage)
    assert tracker.data["status"] == "in_progress"
    assert tracker.data["completed_stages"] == []
    assert tracker.data["input_file"] == "test.pdf"

    # After starting a stage, file should exist
    tracker.start_stage("test_stage")
    assert tracker.progress_file.exists()


def test_progress_tracker_stage_lifecycle(temp_checkpoint_dir: Path) -> None:
    """Test stage start/complete/fail lifecycle."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir)

    # Start stage
    tracker.start_stage("page_1")
    assert tracker.data["current_stage"] == "page_1"
    assert "page_1" in tracker.data["stages"]
    assert tracker.data["stages"]["page_1"]["status"] == "in_progress"

    # Complete stage
    output_file = temp_checkpoint_dir / "page1.json"
    tracker.complete_stage("page_1", output_file)
    assert "page_1" in tracker.data["completed_stages"]
    assert tracker.data["stages"]["page_1"]["status"] == "completed"
    assert tracker.data["stages"]["page_1"]["output_file"] == str(output_file)
    assert tracker.data["current_stage"] is None


def test_progress_tracker_fail_stage(temp_checkpoint_dir: Path) -> None:
    """Test stage failure tracking."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir)

    tracker.start_stage("page_2")
    error = RuntimeError("Test error")
    tracker.fail_stage("page_2", error)

    assert tracker.data["status"] == "failed"
    assert tracker.data["failed_stage"] == "page_2"
    assert "RuntimeError: Test error" in tracker.data["error"]
    assert tracker.data["stages"]["page_2"]["status"] == "failed"


def test_progress_tracker_resume_point(temp_checkpoint_dir: Path) -> None:
    """Test get_resume_point logic."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir)

    # No completed stages
    assert tracker.get_resume_point() is None

    # Complete first stage
    output_file = temp_checkpoint_dir / "page1.json"
    # Create dummy checkpoint file
    output_file.write_text('{"test": "data"}')

    tracker.start_stage("page_1")
    tracker.complete_stage("page_1", output_file)

    # Start and fail second stage
    tracker.start_stage("page_2")
    tracker.fail_stage("page_2", RuntimeError("Test"))

    # Should resume from failed stage
    resume_point = tracker.get_resume_point()
    assert resume_point is not None
    stage_name, checkpoint_file = resume_point
    assert stage_name == "page_2"
    assert checkpoint_file == output_file


def test_progress_tracker_mark_complete(temp_checkpoint_dir: Path) -> None:
    """Test marking pipeline as complete."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir)

    tracker.start_stage("page_1")
    tracker.complete_stage("page_1", temp_checkpoint_dir / "page1.json")
    tracker.mark_complete()

    assert tracker.data["status"] == "completed"
    assert tracker.data["current_stage"] is None
    assert tracker.data["failed_stage"] is None
    assert tracker.data["error"] is None


def test_progress_tracker_validate_input(temp_checkpoint_dir: Path) -> None:
    """Test input file validation."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir, input_file="test.pdf")

    # Same input file
    assert tracker.validate_input("test.pdf") is True

    # Different input file
    assert tracker.validate_input("other.pdf") is False


def test_progress_tracker_persistence(temp_checkpoint_dir: Path) -> None:
    """Test that progress is persisted to disk."""
    from pipeline.checkpoint import ProgressTracker

    # Create and modify tracker
    tracker1 = ProgressTracker(temp_checkpoint_dir, input_file="test.pdf")
    tracker1.start_stage("page_1")
    tracker1.complete_stage("page_1", temp_checkpoint_dir / "page1.json")

    # Load from disk
    tracker2 = ProgressTracker(temp_checkpoint_dir, input_file="test.pdf")
    assert "page_1" in tracker2.data["completed_stages"]
    assert tracker2.data["input_file"] == "test.pdf"


def test_checkpoint_serialization_page(temp_checkpoint_dir: Path) -> None:
    """Test Page object serialization/deserialization."""
    from pipeline.checkpoint import deserialize_stage_result, serialize_stage_result
    from pipeline.types import BBox, Block, Page

    # Create test Page
    blocks = [
        Block(
            type="text",
            bbox=BBox(10, 20, 100, 50),
            text="Hello",
            order=1,
        ),
        Block(
            type="title",
            bbox=BBox(10, 60, 100, 80),
            text="Title",
            order=0,
        ),
    ]
    page = Page(page_num=1, blocks=blocks)

    # Serialize
    output_file = temp_checkpoint_dir / "test_page.json"
    serialize_stage_result("detection", page, output_file)
    assert output_file.exists()

    # Deserialize
    loaded_page = deserialize_stage_result("detection", output_file)
    assert isinstance(loaded_page, Page)
    assert loaded_page.page_num == 1
    assert len(loaded_page.blocks) == 2
    assert loaded_page.blocks[0].text == "Hello"
    assert loaded_page.blocks[1].text == "Title"


def test_checkpoint_serialization_output(temp_checkpoint_dir: Path) -> None:
    """Test output stage (markdown) serialization."""
    from pipeline.checkpoint import deserialize_stage_result, serialize_stage_result

    # Serialize markdown
    markdown = "# Title\n\nThis is a test document."
    output_file = temp_checkpoint_dir / "test_output.json"
    serialize_stage_result("output", markdown, output_file)

    # Check both JSON and .md files created
    assert output_file.exists()
    md_file = temp_checkpoint_dir / "test_output.md"
    assert md_file.exists()

    # Deserialize
    loaded_markdown = deserialize_stage_result("output", output_file)
    assert loaded_markdown == markdown


def test_save_and_load_checkpoint(temp_checkpoint_dir: Path) -> None:
    """Test save_checkpoint and load_checkpoint convenience functions."""
    from pipeline.checkpoint import load_checkpoint, save_checkpoint
    from pipeline.types import BBox, Block, Page

    # Create and save checkpoint
    page = Page(
        page_num=3,
        blocks=[
            Block(type="text", bbox=BBox(0, 0, 100, 100), text="Test", order=0),
        ],
    )

    checkpoint_file = save_checkpoint("detection", page, temp_checkpoint_dir, page_num=3)
    assert checkpoint_file.exists()
    assert checkpoint_file.name == "stage2_detection_page3.json"

    # Load checkpoint
    loaded_page = load_checkpoint("detection", temp_checkpoint_dir, page_num=3)
    assert isinstance(loaded_page, Page)
    assert loaded_page.page_num == 3
    assert len(loaded_page.blocks) == 1


def test_checkpoint_unknown_stage_error(temp_checkpoint_dir: Path) -> None:
    """Test that unknown stage raises ValueError."""
    from pipeline.checkpoint import serialize_stage_result

    with pytest.raises(ValueError, match="Unknown stage"):
        serialize_stage_result("unknown_stage", None, temp_checkpoint_dir / "test.json")


def test_checkpoint_file_not_found_error(temp_checkpoint_dir: Path) -> None:
    """Test that missing checkpoint file raises FileNotFoundError."""
    from pipeline.checkpoint import deserialize_stage_result

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        deserialize_stage_result("detection", temp_checkpoint_dir / "missing.json")


def test_progress_tracker_print_resume_info(temp_checkpoint_dir: Path, capsys: pytest.CaptureFixture) -> None:
    """Test print_resume_info output."""
    from pipeline.checkpoint import ProgressTracker

    tracker = ProgressTracker(temp_checkpoint_dir, input_file="test.pdf")

    # Create dummy checkpoint file
    output_file = temp_checkpoint_dir / "page1.json"
    output_file.write_text('{"test": "data"}')

    tracker.start_stage("page_1")
    tracker.complete_stage("page_1", output_file)
    tracker.start_stage("page_2")
    tracker.fail_stage("page_2", RuntimeError("Test error"))

    tracker.print_resume_info()
    captured = capsys.readouterr()

    assert "Found checkpoint in" in captured.out
    assert "Completed stages: page_1" in captured.out
    assert "Failed at: page_2" in captured.out
    assert "Resuming from 'page_2' stage" in captured.out
