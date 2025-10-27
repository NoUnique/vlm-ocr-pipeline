"""Tests for factory functions."""

from __future__ import annotations

import pytest

from pipeline.layout.detection import create_detector, list_available_detectors
from pipeline.layout.ordering import create_sorter, list_available_sorters
from pipeline.recognition import create_recognizer, list_available_recognizers


def test_list_available_detectors():
    """Test list_available_detectors function."""
    available = list_available_detectors()

    assert "doclayout-yolo" in available
    # PaddleOCR detector should be available if installed
    # assert "paddleocr-ppdoclayout" in available  # Optional dependency


def test_create_detector_doclayout():
    """Test create_detector creates DocLayout-YOLO detector."""
    detector = create_detector("doclayout-yolo", confidence_threshold=0.5)

    assert detector is not None
    assert hasattr(detector, "detect")


def test_create_detector_unknown():
    """Test create_detector raises error for unknown detector."""
    with pytest.raises(ValueError, match="Unknown detector"):
        create_detector("unknown-detector")


def test_list_available_sorters():
    """Test list_available_sorters function."""
    available = list_available_sorters()

    assert "pymupdf" in available
    assert "mineru-xycut" in available


def test_create_sorter_pymupdf():
    """Test create_sorter creates PyMuPDF sorter."""
    sorter = create_sorter("pymupdf")

    assert sorter is not None
    assert hasattr(sorter, "sort")


def test_create_sorter_xycut():
    """Test create_sorter creates XY-Cut sorter."""
    sorter = create_sorter("mineru-xycut")

    assert sorter is not None
    assert hasattr(sorter, "sort")


def test_create_sorter_unknown():
    """Test create_sorter raises error for unknown sorter."""
    with pytest.raises(ValueError, match="Unknown sorter"):
        create_sorter("unknown-sorter")


# ==================== Recognizer Factory Tests ====================


def test_list_available_recognizers():
    """Test list_available_recognizers function."""
    available = list_available_recognizers()

    assert "openai" in available
    assert "gemini" in available
    assert "deepseek-ocr" in available
    # PaddleOCR-VL recognizer should be available if installed
    # assert "paddleocr-vl" in available  # Optional dependency


def test_create_recognizer_openai():
    """Test create_recognizer creates OpenAI TextRecognizer."""
    recognizer = create_recognizer("openai", model="gpt-4o", use_cache=False)

    assert recognizer is not None
    assert hasattr(recognizer, "process_blocks")
    assert hasattr(recognizer, "correct_text")


def test_create_recognizer_gemini():
    """Test create_recognizer creates Gemini TextRecognizer."""
    recognizer = create_recognizer("gemini", model="gemini-2.5-flash", use_cache=False)

    assert recognizer is not None
    assert hasattr(recognizer, "process_blocks")
    assert hasattr(recognizer, "correct_text")


def test_create_recognizer_deepseek_hf():
    """Test create_recognizer creates DeepSeek-OCR recognizer with HF backend."""
    # Will fail at model loading (expected), but tests factory registration
    try:
        recognizer = create_recognizer("deepseek-ocr", backend="hf")
        # If we reach here, factory worked but model not available
        assert hasattr(recognizer, "process_blocks")
        assert hasattr(recognizer, "correct_text")
    except (ImportError, OSError, RuntimeError) as e:
        # Expected: missing dependencies or model not downloaded
        assert "deepseek" in str(e).lower() or "transformers" in str(e).lower() or "addict" in str(e).lower()


def test_create_recognizer_deepseek_vllm():
    """Test create_recognizer creates DeepSeek-OCR recognizer with vLLM backend."""
    # Will fail at model loading (expected), but tests factory registration
    try:
        recognizer = create_recognizer("deepseek-ocr", backend="vllm")
        # If we reach here, factory worked but model not available
        assert hasattr(recognizer, "process_blocks")
        assert hasattr(recognizer, "correct_text")
    except (ImportError, OSError, RuntimeError) as e:
        # Expected: missing dependencies or model not downloaded
        assert "deepseek" in str(e).lower() or "vllm" in str(e).lower() or "addict" in str(e).lower()


def test_create_recognizer_unknown():
    """Test create_recognizer raises error for unknown recognizer."""
    with pytest.raises(ValueError, match="Unknown recognizer"):
        create_recognizer("unknown-recognizer")
