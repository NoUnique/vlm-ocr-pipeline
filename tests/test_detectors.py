"""Tests for detector implementations."""

from __future__ import annotations

import numpy as np

from pipeline.layout.detection import DocLayoutYOLODetector


def test_doclayout_detector_has_detect_method():
    """Test DocLayoutYOLODetector has detect method."""
    detector = DocLayoutYOLODetector()

    assert hasattr(detector, "detect")
    assert callable(detector.detect)


def test_doclayout_detector_returns_regions_with_bbox():
    """Test DocLayoutYOLODetector returns regions with bbox."""
    detector = DocLayoutYOLODetector()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Note: This will likely return empty since it's a blank image
    regions = detector.detect(image)

    # Should return list (empty or with regions)
    assert isinstance(regions, list)

    # If regions exist, they should have proper format
    for region in regions:
        assert hasattr(region, "type")
        assert hasattr(region, "detection_confidence")
        assert hasattr(region, "source")
        assert block.bbox is not None
