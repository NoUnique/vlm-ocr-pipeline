"""Tests for factory pattern."""

from __future__ import annotations

import pytest

from pipeline.factory import DetectorFactory, SorterFactory


def test_detector_factory_lists_available():
    """Test DetectorFactory lists available detectors."""
    available = DetectorFactory.list_available()
    
    assert "doclayout-yolo" in available


def test_detector_factory_creates_doclayout():
    """Test DetectorFactory creates DocLayout-YOLO detector."""
    detector = DetectorFactory.create("doclayout-yolo", confidence_threshold=0.5)
    
    assert detector is not None
    assert hasattr(detector, "detect")


def test_detector_factory_unknown_detector():
    """Test DetectorFactory raises error for unknown detector."""
    with pytest.raises(ValueError, match="Unknown detector"):
        DetectorFactory.create("unknown-detector")


def test_sorter_factory_lists_available():
    """Test SorterFactory lists available sorters."""
    available = SorterFactory.list_available()
    
    assert "pymupdf" in available
    assert "mineru-xycut" in available


def test_sorter_factory_creates_pymupdf():
    """Test SorterFactory creates PyMuPDF sorter."""
    sorter = SorterFactory.create("pymupdf")
    
    assert sorter is not None
    assert hasattr(sorter, "sort")


def test_sorter_factory_creates_xycut():
    """Test SorterFactory creates XY-Cut sorter."""
    sorter = SorterFactory.create("mineru-xycut")
    
    assert sorter is not None
    assert hasattr(sorter, "sort")


def test_sorter_factory_unknown_sorter():
    """Test SorterFactory raises error for unknown sorter."""
    with pytest.raises(ValueError, match="Unknown sorter"):
        SorterFactory.create("unknown-sorter")


def test_detector_factory_is_available():
    """Test DetectorFactory.is_available()."""
    assert DetectorFactory.is_available("doclayout-yolo") is True
    assert DetectorFactory.is_available("unknown") is False


def test_sorter_factory_is_available():
    """Test SorterFactory.is_available()."""
    assert SorterFactory.is_available("pymupdf") is True
    assert SorterFactory.is_available("mineru-xycut") is True
    assert SorterFactory.is_available("unknown") is False

