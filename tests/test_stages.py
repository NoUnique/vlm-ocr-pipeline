"""Tests for pipeline stages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pipeline.stages.block_correction_stage import BlockCorrectionStage
from pipeline.stages.detection_stage import DetectionStage
from pipeline.stages.input_stage import InputStage
from pipeline.stages.ordering_stage import OrderingStage
from pipeline.stages.output_stage import OutputStage
from pipeline.stages.page_correction_stage import PageCorrectionStage
from pipeline.stages.recognition_stage import RecognitionStage
from pipeline.stages.rendering_stage import RenderingStage
from pipeline.types import BBox, Block, Page


class TestInputStage:
    """Tests for InputStage."""

    def test_init(self, tmp_path: Path):
        """Test InputStage initialization."""
        stage = InputStage(temp_dir=tmp_path)
        assert stage.temp_dir == tmp_path

    @patch("pipeline.conversion.input.image.load_image")
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

    @patch("pipeline.conversion.input.pdf.render_pdf_page")
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

    @patch("pipeline.conversion.input.pdf.extract_text_spans_from_pdf")
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

    @patch("pipeline.conversion.input.pdf.extract_text_spans_from_pdf")
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

    @patch("pipeline.conversion.input.pdf.extract_text_spans_from_pdf")
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


class TestDetectionStage:
    """Tests for DetectionStage."""

    def test_init(self):
        """Test DetectionStage initialization."""
        mock_detector = Mock()
        stage = DetectionStage(detector=mock_detector)
        assert stage.detector == mock_detector

    def test_detect(self):
        """Test detecting blocks."""
        # Setup
        mock_detector = Mock()
        mock_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=None,
            )
        ]
        mock_detector.detect.return_value = mock_blocks

        stage = DetectionStage(detector=mock_detector)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.detect(image)

        # Verify
        mock_detector.detect.assert_called_once()
        assert result == mock_blocks

    def test_extract_column_layout_no_columns(self):
        """Test extracting column layout when no columns exist."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                column_index=None,
            )
        ]

        stage = DetectionStage(detector=Mock())

        # Execute
        result = stage.extract_column_layout(blocks)

        # Verify
        assert result is None

    def test_extract_column_layout_with_columns(self):
        """Test extracting column layout when columns exist."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                column_index=0,
            ),
            Block(
                type="text",
                bbox=BBox(300, 100, 400, 200),
                detection_confidence=0.95,
                order=1,
                column_index=1,
            ),
        ]

        stage = DetectionStage(detector=Mock())

        # Execute
        result = stage.extract_column_layout(blocks)

        # Verify
        assert result is not None
        assert "columns" in result
        assert len(result["columns"]) == 2
        assert result["columns"][0]["index"] == 0
        assert result["columns"][0]["x0"] == 100
        assert result["columns"][1]["index"] == 1
        assert result["columns"][1]["x0"] == 300


class TestOrderingStage:
    """Tests for OrderingStage."""

    def test_init(self):
        """Test OrderingStage initialization."""
        mock_sorter = Mock()
        stage = OrderingStage(sorter=mock_sorter)
        assert stage.sorter == mock_sorter

    def test_sort(self):
        """Test sorting blocks."""
        # Setup
        mock_sorter = Mock()
        input_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                detection_confidence=0.95,
                order=None,
            ),
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=None,
            ),
        ]
        sorted_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
            ),
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                detection_confidence=0.95,
                order=1,
            ),
        ]
        mock_sorter.sort.return_value = sorted_blocks

        stage = OrderingStage(sorter=mock_sorter)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.sort(input_blocks, image)

        # Verify
        mock_sorter.sort.assert_called_once_with(input_blocks, image)
        assert result == sorted_blocks
        assert result[0].order == 0
        assert result[1].order == 1


class TestRecognitionStage:
    """Tests for RecognitionStage."""

    def test_init(self):
        """Test RecognitionStage initialization."""
        mock_recognizer = Mock()
        stage = RecognitionStage(recognizer=mock_recognizer)
        assert stage.recognizer == mock_recognizer

    def test_recognize_blocks(self):
        """Test recognizing text in blocks."""
        # Setup
        mock_recognizer = Mock()
        input_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text=None,
            )
        ]
        processed_blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Extracted text",
            )
        ]
        mock_recognizer.process_blocks.return_value = processed_blocks

        stage = RecognitionStage(recognizer=mock_recognizer)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Execute
        result = stage.recognize_blocks(input_blocks, image)

        # Verify
        mock_recognizer.process_blocks.assert_called_once()
        call_args = mock_recognizer.process_blocks.call_args.args
        assert np.array_equal(call_args[0], image)  # First arg is image
        assert call_args[1] == input_blocks  # Second arg is blocks
        assert result == processed_blocks
        assert result[0].text == "Extracted text"


class TestBlockCorrectionStage:
    """Tests for BlockCorrectionStage."""

    def test_init(self):
        """Test BlockCorrectionStage initialization."""
        stage = BlockCorrectionStage(enable=True)
        assert stage.enable is True

    def test_correct_blocks_disabled(self):
        """Test block correction when disabled."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Original text",
                corrected_text=None,
            )
        ]

        stage = BlockCorrectionStage(enable=False)

        # Execute
        result = stage.correct_blocks(blocks)

        # Verify
        assert result[0].corrected_text == "Original text"

    def test_correct_blocks_enabled_placeholder(self):
        """Test block correction when enabled (currently placeholder)."""
        # Setup
        blocks = [
            Block(
                type="text",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Original text",
                corrected_text=None,
            )
        ]

        stage = BlockCorrectionStage(enable=True)

        # Execute
        result = stage.correct_blocks(blocks)

        # Verify (currently just copies text as-is)
        assert result[0].corrected_text == "Original text"
        assert result[0].correction_ratio == 0.0


class TestRenderingStage:
    """Tests for RenderingStage."""

    def test_init(self):
        """Test RenderingStage initialization."""
        stage = RenderingStage(renderer="markdown")
        assert stage.renderer == "markdown"

    @patch("pipeline.stages.rendering_stage.blocks_to_markdown")
    def test_render_markdown(self, mock_blocks_to_markdown: Mock):
        """Test rendering blocks to markdown."""
        # Setup
        blocks = [
            Block(
                type="title",
                bbox=BBox(100, 100, 200, 200),
                detection_confidence=0.95,
                order=0,
                text="Title",
                corrected_text="Title",
            ),
            Block(
                type="text",
                bbox=BBox(100, 200, 200, 300),
                detection_confidence=0.95,
                order=1,
                text="Body text",
                corrected_text="Body text",
            ),
        ]
        expected_markdown = "# Title\n\nBody text"
        mock_blocks_to_markdown.return_value = expected_markdown

        stage = RenderingStage(renderer="markdown")

        # Execute
        result = stage.render(blocks)

        # Verify
        mock_blocks_to_markdown.assert_called_once_with(blocks)
        assert result == expected_markdown

    def test_render_unsupported_renderer(self):
        """Test rendering with unsupported renderer."""
        # Setup
        blocks = []
        stage = RenderingStage(renderer="html")

        # Execute & Verify
        with pytest.raises(ValueError, match="Unsupported renderer: html"):
            stage.render(blocks)


class TestPageCorrectionStage:
    """Tests for PageCorrectionStage."""

    def test_init(self):
        """Test PageCorrectionStage initialization."""
        mock_recognizer = Mock()
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=True,
        )
        assert stage.recognizer == mock_recognizer
        assert stage.backend == "gemini"
        assert stage.enable is True

    def test_correct_page_disabled(self):
        """Test page correction when disabled."""
        # Setup
        mock_recognizer = Mock()
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=False,
        )
        raw_text = "# Title\n\nRaw text"

        # Execute
        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        # Verify
        assert corrected_text == raw_text
        assert ratio == 0.0
        assert should_stop is False
        mock_recognizer.correct_text.assert_not_called()

    def test_correct_page_paddleocr_vl_backend(self):
        """Test page correction with PaddleOCR-VL backend (skipped)."""
        # Setup
        mock_recognizer = Mock()
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="paddleocr-vl",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        # Execute
        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        # Verify
        assert corrected_text == raw_text
        assert ratio == 0.0
        assert should_stop is False
        mock_recognizer.correct_text.assert_not_called()

    def test_correct_page_dict_result(self):
        """Test page correction with dict result."""
        # Setup
        mock_recognizer = Mock()
        mock_recognizer.correct_text.return_value = {
            "corrected_text": "# Title\n\nCorrected text",
            "correction_ratio": 0.15,
        }
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        # Execute
        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        # Verify
        assert corrected_text == "# Title\n\nCorrected text"
        assert ratio == 0.15
        assert should_stop is False

    def test_correct_page_rate_limit_detected(self):
        """Test page correction with rate limit detection."""
        # Setup
        mock_recognizer = Mock()
        mock_recognizer.correct_text.return_value = "RATE_LIMIT_EXCEEDED: Please try again later"
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        # Execute
        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        # Verify
        assert "RATE_LIMIT_EXCEEDED" in corrected_text
        assert ratio == 0.0
        assert should_stop is True


class TestOutputStage:
    """Tests for OutputStage."""

    def test_init(self, tmp_path: Path):
        """Test OutputStage initialization."""
        stage = OutputStage(temp_dir=tmp_path)
        assert stage.temp_dir == tmp_path

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

    def test_save_page_output(self, tmp_path: Path):
        """Test saving page output."""
        # Setup
        stage = OutputStage(temp_dir=tmp_path)
        page_output_dir = tmp_path / "output"
        page_output_dir.mkdir()
        page_num = 1
        page = Page(
            page_num=page_num,
            blocks=[],
            auxiliary_info={"text": "Test"},
            status="completed",
            processed_at="2024-01-01T00:00:00Z",
        )

        # Execute
        stage.save_page_output(page_output_dir, page_num, page)

        # Verify JSON is saved to json/ subdirectory
        json_file = page_output_dir / "json" / f"page_{page_num}.json"
        assert json_file.exists()

        # Verify Markdown is saved to main directory
        md_file = page_output_dir / f"page_{page_num}.md"
        assert md_file.exists()

    def test_determine_summary_filename(self, tmp_path: Path):
        """Test determining summary filename."""
        # Setup
        stage = OutputStage(temp_dir=tmp_path)

        # Test complete processing
        assert (
            stage._determine_summary_filename(
                processing_stopped=False,
                has_errors=False,
            )
            == "summary.json"
        )

        # Test partial processing (has errors)
        assert (
            stage._determine_summary_filename(
                processing_stopped=False,
                has_errors=True,
            )
            == "summary_partial.json"
        )

        # Test incomplete processing (stopped early)
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
