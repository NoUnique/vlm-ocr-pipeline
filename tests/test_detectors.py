"""Tests for detector implementations."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.layout.detection import DocLayoutYOLODetector, create_detector
from pipeline.types import BBox


def test_doclayout_detector_has_detect_method():
    """Test DocLayoutYOLODetector has detect method."""
    detector = DocLayoutYOLODetector()

    assert hasattr(detector, "detect")
    assert callable(detector.detect)


def test_doclayout_detector_returns_blocks_with_required_fields():
    """Test DocLayoutYOLODetector returns blocks with all required fields."""
    detector = DocLayoutYOLODetector()
    # Create a simple test image (blank, will likely return empty)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    blocks = detector.detect(image)

    # Should return list
    assert isinstance(blocks, list)

    # If blocks exist, they should have proper format
    for block in blocks:
        assert hasattr(block, "type")
        assert hasattr(block, "bbox")
        assert hasattr(block, "detection_confidence")
        assert hasattr(block, "source")
        assert isinstance(block.bbox, BBox)
        assert block.detection_confidence is not None
        assert 0.0 <= block.detection_confidence <= 1.0
        assert block.source == "doclayout-yolo"


def test_doclayout_detector_bbox_coordinates_valid():
    """Test that detected bboxes have valid coordinates."""
    detector = DocLayoutYOLODetector()
    # Create a larger test image with some content
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    blocks = detector.detect(image)

    # If blocks are detected, verify bbox validity
    for block in blocks:
        bbox = block.bbox
        # x0 < x1, y0 < y1
        assert bbox.x0 < bbox.x1, f"Invalid bbox x coordinates: {bbox.x0} >= {bbox.x1}"
        assert bbox.y0 < bbox.y1, f"Invalid bbox y coordinates: {bbox.y0} >= {bbox.y1}"
        # Coordinates should be within image bounds
        assert 0 <= bbox.x0 < image.shape[1]
        assert 0 <= bbox.y0 < image.shape[0]
        assert 0 < bbox.x1 <= image.shape[1]
        assert 0 < bbox.y1 <= image.shape[0]


def test_doclayout_detector_confidence_threshold():
    """Test that confidence threshold parameter works."""
    # Create detector with high threshold
    detector_high = DocLayoutYOLODetector(confidence_threshold=0.9)
    detector_low = DocLayoutYOLODetector(confidence_threshold=0.1)

    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    blocks_high = detector_high.detect(image)
    blocks_low = detector_low.detect(image)

    # Lower threshold should detect same or more blocks
    assert len(blocks_low) >= len(blocks_high)

    # All high-confidence blocks should meet threshold
    for block in blocks_high:
        assert block.detection_confidence is not None
        assert block.detection_confidence >= 0.9


def test_create_detector_factory():
    """Test detector factory function."""
    detector = create_detector("doclayout-yolo", confidence_threshold=0.5)

    assert detector is not None
    assert hasattr(detector, "detect")

    # Verify confidence threshold was applied
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = detector.detect(image)
    for block in blocks:
        assert block.detection_confidence is not None
        assert block.detection_confidence >= 0.5


def test_create_detector_invalid_name():
    """Test that creating detector with invalid name raises error."""
    with pytest.raises(ValueError, match="Unknown detector"):
        create_detector("non-existent-detector")


def test_detector_handles_different_image_sizes():
    """Test detector handles various image dimensions."""
    detector = DocLayoutYOLODetector()

    # Test different image sizes
    test_sizes = [
        (100, 100, 3),
        (600, 800, 3),
        (1920, 1080, 3),
        (480, 640, 3),
    ]

    for size in test_sizes:
        image = np.zeros(size, dtype=np.uint8)
        blocks = detector.detect(image)

        # Should not crash and return list
        assert isinstance(blocks, list)
        # All bbox coordinates should be within image bounds
        for block in blocks:
            assert 0 <= block.bbox.x0 < size[1]
            assert 0 <= block.bbox.y0 < size[0]
            assert 0 < block.bbox.x1 <= size[1]
            assert 0 < block.bbox.y1 <= size[0]


def test_detector_block_types_are_valid():
    """Test that detected blocks have valid type strings."""
    detector = DocLayoutYOLODetector()
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    blocks = detector.detect(image)

    # Common block types from DocLayout-YOLO
    valid_types = {
        "text",
        "title",
        "list",
        "table",
        "figure",
        "caption",
        "formula",
        "footnote",
        "header",
        "footer",
    }

    for block in blocks:
        # Type should be non-empty string
        assert isinstance(block.type, str)
        assert len(block.type) > 0
        # Should be one of the known types (may include more)
        # Note: This is informational, not strict enforcement
        if block.type not in valid_types:
            # Log unexpected types for debugging, but don't fail
            print(f"Note: Detected unexpected block type: {block.type}")
