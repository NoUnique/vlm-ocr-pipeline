"""Tests for staged batch processor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from pipeline.batch.processor import StagedBatchProcessor
from pipeline.batch.types import BatchProgress, PageInfo


class TestStagedBatchProcessorInit:
    """Tests for StagedBatchProcessor initialization."""

    def test_init(self):
        """Test processor initialization."""
        mock_pipeline = Mock()
        mock_pipeline.model = "test-model"
        
        processor = StagedBatchProcessor(mock_pipeline)
        
        assert processor.pipeline == mock_pipeline
        assert processor.progress is None

    def test_init_stores_pipeline_reference(self):
        """Test that processor stores pipeline reference."""
        mock_pipeline = Mock()
        mock_pipeline.model = "gemini-2.5-flash"
        
        processor = StagedBatchProcessor(mock_pipeline)
        
        assert processor.pipeline is mock_pipeline


class TestStagedBatchProcessorDirectory:
    """Tests for process_directory method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline."""
        pipeline = Mock()
        pipeline.model = "test-model"
        pipeline.detector = Mock()
        pipeline.sorter = Mock()
        pipeline.recognizer = Mock()
        return pipeline

    def test_process_directory_not_found(self, mock_pipeline, tmp_path):
        """Test error when directory not found."""
        processor = StagedBatchProcessor(mock_pipeline)
        
        result = processor.process_directory(
            tmp_path / "nonexistent",
            str(tmp_path / "output"),
        )
        
        assert "error" in result
        assert "not found" in result["error"]

    def test_process_directory_no_pdfs(self, mock_pipeline, tmp_path):
        """Test error when no PDF files found."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        processor = StagedBatchProcessor(mock_pipeline)
        
        result = processor.process_directory(
            input_dir,
            str(tmp_path / "output"),
        )
        
        assert "error" in result
        assert "No PDF files found" in result["error"]


class TestPageInfo:
    """Tests for PageInfo dataclass."""

    def test_page_info_creation(self, tmp_path):
        """Test PageInfo creation."""
        pdf_path = tmp_path / "test.pdf"
        
        info = PageInfo(
            pdf_path=pdf_path,
            page_num=1,
            total_pages=10,
        )
        
        assert info.pdf_path == pdf_path
        assert info.page_num == 1
        assert info.total_pages == 10
        assert info.image is None
        assert info.blocks is None
        assert info.status == "pending"

    def test_page_info_with_all_fields(self, tmp_path):
        """Test PageInfo with all fields populated."""
        import numpy as np
        from pipeline.types import BBox, Block
        
        pdf_path = tmp_path / "test.pdf"
        blocks = [Block(type="text", bbox=BBox(0, 0, 100, 100))]
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        info = PageInfo(
            pdf_path=pdf_path,
            page_num=1,
            total_pages=10,
            image=image,
            blocks=blocks,
            status="completed",
        )
        
        assert info.image is not None
        assert len(info.blocks) == 1
        assert info.status == "completed"


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_batch_progress_creation(self):
        """Test BatchProgress creation."""
        progress = BatchProgress(total_pages=10)
        
        assert progress.total_pages == 10
        assert progress.completed_pages == 0
        assert progress.failed_pages == 0
        assert progress.current_stage == 0  # 0 = not started
        assert progress.stage_name == "Not Started"

    def test_batch_progress_update(self):
        """Test BatchProgress can be updated."""
        progress = BatchProgress(total_pages=10)
        
        progress.completed_pages = 5
        progress.failed_pages = 1
        progress.current_stage = 4
        progress.stage_name = "Recognition"
        
        assert progress.completed_pages == 5
        assert progress.failed_pages == 1
        assert progress.current_stage == 4
        assert progress.stage_name == "Recognition"

    def test_batch_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = BatchProgress(total_pages=100)
        progress.completed_pages = 25
        
        # Use property method
        assert progress.progress_pct == 25.0

    def test_batch_progress_is_complete(self):
        """Test is_complete property."""
        progress = BatchProgress(total_pages=10)
        progress.completed_pages = 8
        progress.failed_pages = 2
        
        assert progress.is_complete is True

