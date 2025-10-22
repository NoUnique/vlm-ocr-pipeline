"""Tests for bidirectional auto-selection of detector/sorter pairs."""

from __future__ import annotations

import pytest

from pipeline import Pipeline


def test_sorter_auto_selects_detector_paddleocr():
    """Test that specifying paddleocr-doclayout-v2 sorter auto-selects the detector."""
    # When only sorter is specified, detector should be auto-selected
    pipeline = Pipeline(
        sorter="paddleocr-doclayout-v2",
        backend="openai",  # Need to specify a backend
    )

    assert pipeline.detector_name == "paddleocr-doclayout-v2"
    assert pipeline.sorter_name == "paddleocr-doclayout-v2"


def test_sorter_auto_selects_detector_mineru_vlm():
    """Test that specifying mineru-vlm sorter auto-selects the detector."""
    pipeline = Pipeline(
        sorter="mineru-vlm",
        backend="openai",
    )

    assert pipeline.detector_name == "mineru-vlm"
    assert pipeline.sorter_name == "mineru-vlm"


def test_detector_auto_selects_sorter_paddleocr():
    """Test that specifying paddleocr-doclayout-v2 detector auto-selects the sorter."""
    pipeline = Pipeline(
        detector="paddleocr-doclayout-v2",
        backend="openai",
    )

    assert pipeline.detector_name == "paddleocr-doclayout-v2"
    assert pipeline.sorter_name == "paddleocr-doclayout-v2"


def test_detector_auto_selects_sorter_mineru_vlm():
    """Test that specifying mineru-vlm detector auto-selects the sorter."""
    pipeline = Pipeline(
        detector="mineru-vlm",
        backend="openai",
    )

    assert pipeline.detector_name == "mineru-vlm"
    assert pipeline.sorter_name == "mineru-vlm"


def test_incompatible_explicit_detector_sorter_raises_error():
    """Test that explicitly specifying incompatible detector/sorter raises error."""
    # User explicitly specifies incompatible combination (non-default detector)
    with pytest.raises(ValueError, match="tightly coupled"):
        Pipeline(
            detector="mineru-doclayout-yolo",  # Non-default detector
            sorter="paddleocr-doclayout-v2",
            backend="openai",
        )


def test_compatible_explicit_detector_sorter_succeeds():
    """Test that explicitly specifying compatible detector/sorter succeeds."""
    pipeline = Pipeline(
        detector="paddleocr-doclayout-v2",
        sorter="paddleocr-doclayout-v2",
        backend="openai",
    )

    assert pipeline.detector_name == "paddleocr-doclayout-v2"
    assert pipeline.sorter_name == "paddleocr-doclayout-v2"


def test_default_detector_sorter_without_specification():
    """Test that default detector/sorter are used when nothing is specified."""
    pipeline = Pipeline(backend="openai")

    # Default: doclayout-yolo detector with mineru-xycut sorter
    assert pipeline.detector_name == "doclayout-yolo"
    assert pipeline.sorter_name == "mineru-xycut"


def test_non_tightly_coupled_sorter_uses_default_detector():
    """Test that non-tightly-coupled sorter doesn't change default detector."""
    pipeline = Pipeline(
        sorter="mineru-xycut",
        backend="openai",
    )

    # mineru-xycut is not tightly coupled, so default detector should be used
    assert pipeline.detector_name == "doclayout-yolo"
    assert pipeline.sorter_name == "mineru-xycut"
