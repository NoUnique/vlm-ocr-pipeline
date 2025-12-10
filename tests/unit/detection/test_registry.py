"""Tests for DetectorRegistry."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from pipeline.exceptions import InvalidConfigError
from pipeline.layout.detection.registry import DetectorRegistry, detector_registry


class TestDetectorRegistryInit:
    """Tests for DetectorRegistry initialization."""

    def test_init(self):
        """Test registry initialization."""
        registry = DetectorRegistry()
        assert len(registry._custom_detectors) == 0
        assert len(registry._loaded_classes) == 0

    def test_global_instance(self):
        """Test global registry instance exists."""
        assert detector_registry is not None
        assert isinstance(detector_registry, DetectorRegistry)


class TestDetectorRegistryListAvailable:
    """Tests for listing available detectors."""

    def test_list_available(self):
        """Test listing available detectors."""
        registry = DetectorRegistry()
        available = registry.list_available()
        
        assert "doclayout-yolo" in available
        assert "paddleocr-doclayout-v2" in available
        assert "mineru-doclayout-yolo" in available
        assert "mineru-vlm" in available

    def test_is_available(self):
        """Test checking detector availability."""
        registry = DetectorRegistry()
        
        assert registry.is_available("doclayout-yolo")
        assert registry.is_available("paddleocr-doclayout-v2")
        assert not registry.is_available("non-existent")

    def test_contains(self):
        """Test __contains__ method."""
        registry = DetectorRegistry()
        
        assert "doclayout-yolo" in registry
        assert "non-existent" not in registry


class TestDetectorRegistryRegister:
    """Tests for custom detector registration."""

    def test_register_custom(self):
        """Test registering custom detector."""
        registry = DetectorRegistry()
        mock_class = Mock()
        
        registry.register("custom-detector", mock_class)
        
        assert "custom-detector" in registry.list_available()
        assert registry.is_available("custom-detector")

    def test_register_override_builtin(self):
        """Test overriding built-in detector logs warning."""
        registry = DetectorRegistry()
        mock_class = Mock()
        
        # Should log warning when overriding
        with patch("pipeline.layout.detection.registry.logger") as mock_logger:
            registry.register("doclayout-yolo", mock_class)
            mock_logger.warning.assert_called()


class TestDetectorRegistryCreate:
    """Tests for detector creation."""

    def test_create_unknown(self):
        """Test creating unknown detector raises InvalidConfigError."""
        registry = DetectorRegistry()
        
        with pytest.raises(InvalidConfigError, match="Unknown detector"):
            registry.create("non-existent-detector")

    def test_create_custom(self):
        """Test creating custom detector."""
        registry = DetectorRegistry()
        mock_instance = Mock()
        mock_class = Mock(return_value=mock_instance)
        
        registry.register("custom-detector", mock_class)
        result = registry.create("custom-detector", confidence_threshold=0.7)
        
        mock_class.assert_called_once_with(confidence_threshold=0.7)
        assert result == mock_instance


class TestDetectorRegistryGetClass:
    """Tests for getting detector class."""

    def test_get_class_caching(self):
        """Test that classes are cached after loading."""
        registry = DetectorRegistry()
        
        # Mock the import to track loading
        mock_class = Mock()
        with patch.dict(registry._loaded_classes, {"doclayout-yolo": mock_class}):
            result = registry.get_class("doclayout-yolo")
            assert result == mock_class


class TestDetectorRegistryRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        registry = DetectorRegistry()
        repr_str = repr(registry)
        
        assert "DetectorRegistry" in repr_str
        assert "available=" in repr_str

