"""Tests for PageCorrectionStage."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from pipeline.stages.page_correction_stage import PageCorrectionStage


class TestPageCorrectionStageInit:
    """Tests for PageCorrectionStage initialization."""

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

    def test_init_disabled(self):
        """Test PageCorrectionStage initialization with disabled."""
        mock_recognizer = Mock()
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="openai",
            enable=False,
        )
        assert stage.enable is False


class TestPageCorrectionStageCorrect:
    """Tests for PageCorrectionStage.correct_page method."""

    def test_correct_page_disabled(self):
        """Test page correction when disabled."""
        mock_recognizer = Mock()
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=False,
        )
        raw_text = "# Title\n\nRaw text"

        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        assert corrected_text == raw_text
        assert ratio == 0.0
        assert should_stop is False
        mock_recognizer.correct_text.assert_not_called()

    def test_correct_page_paddleocr_vl_backend(self):
        """Test page correction with PaddleOCR-VL backend (skipped)."""
        mock_recognizer = Mock()
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="paddleocr-vl",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        assert corrected_text == raw_text
        assert ratio == 0.0
        assert should_stop is False
        mock_recognizer.correct_text.assert_not_called()

    def test_correct_page_dict_result(self):
        """Test page correction with dict result."""
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

        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        assert corrected_text == "# Title\n\nCorrected text"
        assert ratio == 0.15
        assert should_stop is False

    def test_correct_page_string_result(self):
        """Test page correction with string result."""
        mock_recognizer = Mock()
        mock_recognizer.correct_text.return_value = "# Title\n\nCorrected text"
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        assert corrected_text == "# Title\n\nCorrected text"
        # ratio is calculated from text difference
        assert should_stop is False

    def test_correct_page_rate_limit_detected(self):
        """Test page correction with rate limit detection."""
        mock_recognizer = Mock()
        mock_recognizer.correct_text.return_value = "RATE_LIMIT_EXCEEDED: Please try again later"
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        corrected_text, ratio, should_stop = stage.correct_page(raw_text, page_num=1)

        assert "RATE_LIMIT_EXCEEDED" in corrected_text
        assert ratio == 0.0
        assert should_stop is True

    def test_correct_page_exception_propagates(self):
        """Test page correction exception propagates to caller."""
        mock_recognizer = Mock()
        mock_recognizer.correct_text.side_effect = Exception("API Error")
        stage = PageCorrectionStage(
            recognizer=mock_recognizer,
            backend="gemini",
            enable=True,
        )
        raw_text = "# Title\n\nRaw text"

        # Exception should propagate (not caught internally)
        with pytest.raises(Exception, match="API Error"):
            stage.correct_page(raw_text, page_num=1)

