"""Tests for detector implementations."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import BBox

# Note: Uses session-scoped 'doclayout_yolo_detector' fixture from conftest.py
# to avoid TORCH_LIBRARY registration conflicts across test files


@pytest.mark.slow
class TestDocLayoutYOLODetector:
    """Tests for DocLayoutYOLODetector.

    These tests require model loading and are marked as slow.
    Run with: pytest -m slow
    """

    def test_doclayout_detector_has_detect_method(self, doclayout_yolo_detector):
        """Test DocLayoutYOLODetector has detect method."""
        assert hasattr(doclayout_yolo_detector, "detect")
        assert callable(doclayout_yolo_detector.detect)

    def test_doclayout_detector_returns_blocks_with_required_fields(self, doclayout_yolo_detector):
        """Test DocLayoutYOLODetector returns blocks with all required fields."""
        # Create a simple test image (blank, will likely return empty)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        blocks = doclayout_yolo_detector.detect(image)

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

    def test_doclayout_detector_bbox_coordinates_valid(self, doclayout_yolo_detector):
        """Test that detected bboxes have valid coordinates."""
        # Create a larger test image with some content
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255

        blocks = doclayout_yolo_detector.detect(image)

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

    def test_doclayout_detector_confidence_threshold(self, doclayout_yolo_detector):
        """Test that confidence threshold parameter works.

        Note: Using session-scoped detector with default threshold (0.5).
        We verify detection works rather than comparing thresholds.
        """
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255

        blocks = doclayout_yolo_detector.detect(image)

        # All blocks should meet the detector's threshold
        for block in blocks:
            assert block.detection_confidence is not None
            assert block.detection_confidence >= 0.5

    def test_detector_handles_different_image_sizes(self, doclayout_yolo_detector):
        """Test detector handles various image dimensions."""
        # Test different image sizes
        test_sizes = [
            (100, 100, 3),
            (600, 800, 3),
            (480, 640, 3),
        ]

        for size in test_sizes:
            image = np.zeros(size, dtype=np.uint8)
            blocks = doclayout_yolo_detector.detect(image)

            # Should not crash and return list
            assert isinstance(blocks, list)
            # All bbox coordinates should be within image bounds
            for block in blocks:
                assert 0 <= block.bbox.x0 < size[1]
                assert 0 <= block.bbox.y0 < size[0]
                assert 0 < block.bbox.x1 <= size[1]
                assert 0 < block.bbox.y1 <= size[0]

    def test_detector_block_types_are_valid(self, doclayout_yolo_detector):
        """Test that detected blocks have valid type strings."""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255

        blocks = doclayout_yolo_detector.detect(image)

        # Note: Block types vary by model, so we only check for non-empty strings
        # Common types include: text, title, list, table, figure, caption, etc.

        for block in blocks:
            # Type should be non-empty string
            assert isinstance(block.type, str)
            assert len(block.type) > 0


@pytest.mark.slow
class TestDetectorFactory:
    """Tests for detector factory function."""

    def test_create_detector_factory(self, doclayout_yolo_detector):
        """Test detector factory function.

        Uses session-scoped fixture to avoid TORCH_LIBRARY conflicts.
        """
        detector = doclayout_yolo_detector

        assert detector is not None
        assert hasattr(detector, "detect")

        # Verify detection works
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        blocks = detector.detect(image)
        for block in blocks:
            assert block.detection_confidence is not None
            assert block.detection_confidence >= 0.5

    def test_create_detector_invalid_name(self):
        """Test that creating detector with invalid name raises error."""
        from pipeline.layout.detection import create_detector

        with pytest.raises(ValueError, match="Unknown detector"):
            create_detector("non-existent-detector")
