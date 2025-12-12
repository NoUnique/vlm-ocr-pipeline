"""Tests for InputStage.

Tests cover:
- Stage initialization
- Image loading
- PDF page loading
- Auxiliary info extraction
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from pipeline.stages.input_stage import InputStage


class TestInputStageInit:
    """Tests for InputStage initialization."""

    def test_init_basic(self, tmp_path: Path):
        """Test basic InputStage initialization."""
        stage = InputStage(temp_dir=tmp_path)
        assert stage.temp_dir == tmp_path

    def test_init_with_dpi(self, tmp_path: Path):
        """Test InputStage initialization with DPI settings."""
        stage = InputStage(
            temp_dir=tmp_path,
            dpi=300,
            detection_dpi=150,
            recognition_dpi=400,
            use_dual_resolution=True,
        )

        assert stage.temp_dir == tmp_path
        assert stage.dpi == 300
        assert stage.detection_dpi == 150
        assert stage.recognition_dpi == 400
        assert stage.use_dual_resolution is True


class TestInputStageLoadImage:
    """Tests for InputStage image loading."""

    @patch("pipeline.io.input.image.load_image")
    def test_load_image(self, mock_load_image: Mock, tmp_path: Path):
        """Test loading image file."""
        # Setup
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_load_image.return_value = mock_image

        stage = InputStage(temp_dir=tmp_path)
        image_path = tmp_path / "test.jpg"

        # Execute
        result = stage.load_image(image_path)

        # Verify
        mock_load_image.assert_called_once_with(image_path)
        assert np.array_equal(result, mock_image)


class TestInputStageLoadPDF:
    """Tests for InputStage PDF loading."""

    @patch("pipeline.io.input.pdf.render_pdf_page")
    def test_load_pdf_page(self, mock_render: Mock, tmp_path: Path):
        """Test loading PDF page."""
        # Setup
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_temp_path = tmp_path / "temp.jpg"
        mock_render.return_value = (mock_image, mock_temp_path)

        stage = InputStage(temp_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"

        # Execute
        result = stage.load_pdf_page(pdf_path, page_num=1)

        # Verify
        mock_render.assert_called_once_with(pdf_path, 1, temp_dir=tmp_path, dpi=200)
        assert np.array_equal(result, mock_image)


class TestInputStageAuxiliaryInfo:
    """Tests for InputStage auxiliary info extraction."""

    @patch("pipeline.io.input.pdf.extract_text_spans_from_pdf")
    def test_extract_auxiliary_info_success(self, mock_extract: Mock, tmp_path: Path):
        """Test extracting auxiliary info successfully."""
        # Setup
        mock_spans = [{"text": "Sample", "size": 12, "font": "Arial"}]
        mock_extract.return_value = mock_spans

        stage = InputStage(temp_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"

        # Execute
        result = stage.extract_auxiliary_info(pdf_path, page_num=1)

        # Verify
        mock_extract.assert_called_once_with(pdf_path, 1)
        assert result is not None
        assert result["text_spans"] == mock_spans

    @patch("pipeline.io.input.pdf.extract_text_spans_from_pdf")
    def test_extract_auxiliary_info_no_spans(self, mock_extract: Mock, tmp_path: Path):
        """Test extracting auxiliary info when no spans found."""
        # Setup
        mock_extract.return_value = []

        stage = InputStage(temp_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"

        # Execute
        result = stage.extract_auxiliary_info(pdf_path, page_num=1)

        # Verify
        assert result is None

    @patch("pipeline.io.input.pdf.extract_text_spans_from_pdf")
    def test_extract_auxiliary_info_exception(self, mock_extract: Mock, tmp_path: Path):
        """Test extracting auxiliary info when exception occurs."""
        # Setup
        mock_extract.side_effect = RuntimeError("Test error")

        stage = InputStage(temp_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"

        # Execute
        result = stage.extract_auxiliary_info(pdf_path, page_num=1)

        # Verify
        assert result is None

