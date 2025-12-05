"""Tests for BaseDetector abstract class."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.layout.detection.base import BaseDetector
from pipeline.types import BBox, Block


class ConcreteDetector(BaseDetector):
    """Concrete implementation for testing."""

    name = "test-detector"
    source = "test-detector"

    def __init__(self, confidence_threshold: float = 0.5, return_blocks: list[Block] | None = None):
        super().__init__(confidence_threshold)
        self._return_blocks = return_blocks or []

    def _detect_impl(self, image: np.ndarray) -> list[Block]:
        return self._return_blocks


class TestBaseDetectorInit:
    """Tests for BaseDetector initialization."""

    def test_init_default_threshold(self):
        """Test default confidence threshold."""
        detector = ConcreteDetector()
        assert detector.confidence_threshold == 0.5

    def test_init_custom_threshold(self):
        """Test custom confidence threshold."""
        detector = ConcreteDetector(confidence_threshold=0.7)
        assert detector.confidence_threshold == 0.7

    def test_init_invalid_threshold_low(self):
        """Test invalid threshold below 0."""
        with pytest.raises(ValueError, match="confidence_threshold must be"):
            ConcreteDetector(confidence_threshold=-0.1)

    def test_init_invalid_threshold_high(self):
        """Test invalid threshold above 1."""
        with pytest.raises(ValueError, match="confidence_threshold must be"):
            ConcreteDetector(confidence_threshold=1.5)


class TestBaseDetectorDetect:
    """Tests for BaseDetector.detect method."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def test_detect_empty_blocks(self, sample_image):
        """Test detection returning empty blocks."""
        detector = ConcreteDetector()
        result = detector.detect(sample_image)
        assert result == []

    def test_detect_with_blocks(self, sample_image):
        """Test detection returning blocks."""
        blocks = [
            Block(type="text", bbox=BBox(10, 10, 50, 50), detection_confidence=0.9, source="test")
        ]
        detector = ConcreteDetector(return_blocks=blocks)
        result = detector.detect(sample_image)
        assert len(result) == 1
        assert result[0].type == "text"

    def test_detect_none_image(self):
        """Test detection with None image."""
        detector = ConcreteDetector()
        with pytest.raises(ValueError, match="Image cannot be None"):
            detector.detect(None)

    def test_detect_invalid_dimensions(self):
        """Test detection with wrong image dimensions."""
        detector = ConcreteDetector()
        image = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(ValueError, match="3 dimensions"):
            detector.detect(image)

    def test_detect_empty_image(self):
        """Test detection with empty image."""
        detector = ConcreteDetector()
        image = np.zeros((0, 100, 3), dtype=np.uint8)
        result = detector.detect(image)
        assert result == []


class TestBaseDetectorValidation:
    """Tests for block validation in BaseDetector."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def test_filter_low_confidence(self, sample_image):
        """Test filtering blocks below confidence threshold."""
        blocks = [
            Block(type="text", bbox=BBox(10, 10, 50, 50), detection_confidence=0.3, source="test"),
            Block(type="text", bbox=BBox(60, 10, 100, 50), detection_confidence=0.8, source="test"),
        ]
        detector = ConcreteDetector(confidence_threshold=0.5, return_blocks=blocks)
        result = detector.detect(sample_image)
        
        assert len(result) == 1
        assert result[0].detection_confidence == 0.8

    def test_clip_negative_coords(self, sample_image):
        """Test clipping negative coordinates."""
        blocks = [
            Block(type="text", bbox=BBox(-10, -5, 50, 50), detection_confidence=0.9, source="test"),
        ]
        detector = ConcreteDetector(return_blocks=blocks)
        result = detector.detect(sample_image)
        
        assert len(result) == 1
        assert result[0].bbox.x0 == 0
        assert result[0].bbox.y0 == 0

    def test_clip_exceeding_bounds(self, sample_image):
        """Test clipping coordinates exceeding image bounds."""
        blocks = [
            Block(type="text", bbox=BBox(10, 10, 250, 150), detection_confidence=0.9, source="test"),
        ]
        detector = ConcreteDetector(return_blocks=blocks)
        result = detector.detect(sample_image)
        
        assert len(result) == 1
        assert result[0].bbox.x1 == 200  # image width
        assert result[0].bbox.y1 == 100  # image height

    def test_skip_zero_area(self, sample_image):
        """Test skipping zero-area blocks."""
        blocks = [
            Block(type="text", bbox=BBox(10, 10, 10, 50), detection_confidence=0.9, source="test"),  # width=0
        ]
        detector = ConcreteDetector(return_blocks=blocks)
        result = detector.detect(sample_image)
        
        assert len(result) == 0


class TestBaseDetectorBatch:
    """Tests for batch detection."""

    def test_detect_batch(self):
        """Test batch detection."""
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        blocks = [Block(type="text", bbox=BBox(10, 10, 50, 50), detection_confidence=0.9, source="test")]
        detector = ConcreteDetector(return_blocks=blocks)
        
        results = detector.detect_batch(images)
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1


class TestBaseDetectorHelpers:
    """Tests for helper methods."""

    def test_create_block(self):
        """Test _create_block helper."""
        detector = ConcreteDetector()
        
        block = detector._create_block(
            block_type="title",
            bbox=(10, 20, 100, 80),
            confidence=0.95,
        )
        
        assert block.type == "title"
        assert block.bbox == BBox(10, 20, 100, 80)
        assert block.detection_confidence == 0.95
        assert block.source == "test-detector"

    def test_repr(self):
        """Test string representation."""
        detector = ConcreteDetector(confidence_threshold=0.7)
        repr_str = repr(detector)
        
        assert "ConcreteDetector" in repr_str
        assert "test-detector" in repr_str
        assert "0.7" in repr_str

