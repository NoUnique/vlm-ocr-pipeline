"""Tests for factory functions."""

from __future__ import annotations

import pytest

from pipeline.layout.detection import create_detector, list_available_detectors
from pipeline.layout.ordering import create_sorter, list_available_sorters


def test_list_available_detectors():
    """Test list_available_detectors function."""
    available = list_available_detectors()
    
    assert "doclayout-yolo" in available


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

