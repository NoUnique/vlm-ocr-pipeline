"""Tests for factory functions.

These are unit tests focused on factory logic, not integration tests.
Models are mocked to avoid heavy initialization.
Uses lazy imports to reduce test startup time.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline.exceptions import InvalidConfigError


def test_list_available_detectors():
    """Test list_available_detectors function."""
    from pipeline.layout.detection import list_available_detectors

    available = list_available_detectors()

    assert "doclayout-yolo" in available
    # PaddleOCR detector should be available if installed
    # assert "paddleocr-ppdoclayout" in available  # Optional dependency


@patch("pipeline.layout.detection.doclayout_yolo.DocLayoutYOLODetector.__init__", return_value=None)
def test_create_detector_doclayout(mock_init):
    """Test create_detector creates DocLayout-YOLO detector."""
    from pipeline.layout.detection import create_detector

    detector = create_detector("doclayout-yolo", confidence_threshold=0.5)

    assert detector is not None
    assert hasattr(detector, "detect")
    mock_init.assert_called_once()


def test_create_detector_unknown():
    """Test create_detector raises error for unknown detector."""
    from pipeline.layout.detection import create_detector

    with pytest.raises(InvalidConfigError, match="Unknown detector"):
        create_detector("unknown-detector")


def test_list_available_sorters():
    """Test list_available_sorters function."""
    from pipeline.layout.ordering import list_available_sorters

    available = list_available_sorters()

    assert "pymupdf" in available
    assert "mineru-xycut" in available


def test_create_sorter_pymupdf():
    """Test create_sorter creates PyMuPDF sorter."""
    from pipeline.layout.ordering import create_sorter

    sorter = create_sorter("pymupdf")

    assert sorter is not None
    assert hasattr(sorter, "sort")


def test_create_sorter_xycut():
    """Test create_sorter creates XY-Cut sorter."""
    from pipeline.layout.ordering import create_sorter

    sorter = create_sorter("mineru-xycut")

    assert sorter is not None
    assert hasattr(sorter, "sort")


def test_create_sorter_unknown():
    """Test create_sorter raises error for unknown sorter."""
    from pipeline.layout.ordering import create_sorter

    with pytest.raises(InvalidConfigError, match="Unknown sorter"):
        create_sorter("unknown-sorter")


# ==================== Recognizer Factory Tests ====================


@pytest.mark.skip(reason="Slow test (41s) - imports all recognizer modules. Covered by factory tests.")
def test_list_available_recognizers():
    """Test list_available_recognizers function."""
    from pipeline.recognition import list_available_recognizers

    available = list_available_recognizers()

    assert "openai" in available
    assert "gemini" in available
    assert "deepseek-ocr" in available
    # PaddleOCR-VL recognizer should be available if installed
    # assert "paddleocr-vl" in available  # Optional dependency


@patch("pipeline.recognition.TextRecognizer.__init__", return_value=None)
def test_create_recognizer_openai(mock_init):
    """Test create_recognizer creates OpenAI TextRecognizer."""
    from pipeline.recognition import create_recognizer

    recognizer = create_recognizer("openai", model="gpt-4o", use_cache=False)

    assert recognizer is not None
    assert hasattr(recognizer, "process_blocks")
    assert hasattr(recognizer, "correct_text")
    mock_init.assert_called_once()


@patch("pipeline.recognition.TextRecognizer.__init__", return_value=None)
def test_create_recognizer_gemini(mock_init):
    """Test create_recognizer creates Gemini TextRecognizer."""
    from pipeline.recognition import create_recognizer

    recognizer = create_recognizer("gemini", model="gemini-2.5-flash", use_cache=False)

    assert recognizer is not None
    assert hasattr(recognizer, "process_blocks")
    assert hasattr(recognizer, "correct_text")
    mock_init.assert_called_once()


@patch("pipeline.recognition.deepseek.deepseek_ocr.DeepSeekOCRRecognizer.__init__", return_value=None)
def test_create_recognizer_deepseek_hf(mock_init):
    """Test create_recognizer creates DeepSeek-OCR recognizer with HF backend."""
    from pipeline.recognition import create_recognizer

    recognizer = create_recognizer("deepseek-ocr", backend="hf")

    assert recognizer is not None
    assert hasattr(recognizer, "process_blocks")
    assert hasattr(recognizer, "correct_text")
    mock_init.assert_called_once()


@patch("pipeline.recognition.deepseek.deepseek_ocr.DeepSeekOCRRecognizer.__init__", return_value=None)
def test_create_recognizer_deepseek_vllm(mock_init):
    """Test create_recognizer creates DeepSeek-OCR recognizer with vLLM backend."""
    from pipeline.recognition import create_recognizer

    recognizer = create_recognizer("deepseek-ocr", backend="vllm")

    assert recognizer is not None
    assert hasattr(recognizer, "process_blocks")
    assert hasattr(recognizer, "correct_text")
    mock_init.assert_called_once()


def test_create_recognizer_unknown():
    """Test create_recognizer raises error for unknown recognizer."""
    from pipeline.recognition import create_recognizer

    with pytest.raises(InvalidConfigError, match="Unknown recognizer"):
        create_recognizer("unknown-recognizer")
