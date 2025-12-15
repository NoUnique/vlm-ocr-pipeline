"""Tests for RecognizerRegistry."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from pipeline.exceptions import InvalidConfigError
from pipeline.recognition.registry import RecognizerRegistry, recognizer_registry


class TestRecognizerRegistryInit:
    """Tests for RecognizerRegistry initialization."""

    def test_init(self):
        """Test registry initialization."""
        registry = RecognizerRegistry()
        assert len(registry._custom_recognizers) == 0
        assert len(registry._loaded_classes) == 0

    def test_global_instance(self):
        """Test global registry instance exists."""
        assert recognizer_registry is not None
        assert isinstance(recognizer_registry, RecognizerRegistry)


class TestRecognizerRegistryListAvailable:
    """Tests for listing available recognizers."""

    def test_list_available(self):
        """Test listing available recognizers."""
        registry = RecognizerRegistry()
        available = registry.list_available()
        
        assert "openai" in available
        assert "gemini" in available
        assert "paddleocr-vl" in available
        assert "deepseek-ocr" in available

    def test_is_available(self):
        """Test checking recognizer availability."""
        registry = RecognizerRegistry()
        
        assert registry.is_available("openai")
        assert registry.is_available("gemini")
        # Unknown names fallback to gemini, so is_available returns True
        # This is by design for flexibility with model names
        assert registry.is_available("gemini-2.5-flash")

    def test_contains(self):
        """Test __contains__ method."""
        registry = RecognizerRegistry()
        
        assert "gemini" in registry
        assert "openai" in registry


class TestRecognizerRegistryResolveName:
    """Tests for name resolution."""

    def test_resolve_direct_name(self):
        """Test resolving direct recognizer name."""
        registry = RecognizerRegistry()
        
        name, kwargs = registry.resolve_name("gemini")
        assert name == "gemini"
        assert kwargs == {}

    def test_resolve_model_name_gemini(self):
        """Test resolving Gemini model name."""
        registry = RecognizerRegistry()
        
        name, kwargs = registry.resolve_name("gemini-2.5-flash")
        assert name == "gemini"
        assert kwargs == {"model": "gemini-2.5-flash"}

    def test_resolve_model_name_gpt(self):
        """Test resolving GPT model name."""
        registry = RecognizerRegistry()
        
        name, kwargs = registry.resolve_name("gpt-4o")
        assert name == "openai"
        assert kwargs == {"model": "gpt-4o"}

    def test_resolve_model_pattern(self):
        """Test resolving model name patterns."""
        registry = RecognizerRegistry()
        
        # gemini pattern
        name, kwargs = registry.resolve_name("gemini-2.5-flash")
        assert name == "gemini"
        
        # gpt pattern -> openai
        name, kwargs = registry.resolve_name("gpt-4o")
        assert name == "openai"


class TestRecognizerRegistryRegister:
    """Tests for custom recognizer registration."""

    def test_register_custom(self):
        """Test registering custom recognizer."""
        registry = RecognizerRegistry()
        mock_class = Mock()
        
        registry.register("custom-recognizer", mock_class)
        
        assert "custom-recognizer" in registry.list_available()
        assert registry.is_available("custom-recognizer")

    def test_register_override_builtin(self):
        """Test overriding built-in recognizer logs warning."""
        registry = RecognizerRegistry()
        mock_class = Mock()
        
        with patch("pipeline.recognition.registry.logger") as mock_logger:
            registry.register("gemini", mock_class)
            mock_logger.warning.assert_called()


class TestRecognizerRegistryCreate:
    """Tests for recognizer creation."""

    def test_create_unknown(self):
        """Test creating unknown recognizer raises InvalidConfigError."""
        registry = RecognizerRegistry()
        
        # Unknown names should raise InvalidConfigError (explicit error, no silent fallback)
        with pytest.raises(InvalidConfigError, match="Unknown recognizer: 'unknown-model'"):
            registry.create("unknown-model")

    def test_create_custom(self):
        """Test creating custom recognizer."""
        registry = RecognizerRegistry()
        mock_instance = Mock()
        mock_class = Mock(return_value=mock_instance)
        
        registry.register("custom-recognizer", mock_class)
        result = registry.create("custom-recognizer", use_cache=False)
        
        mock_class.assert_called_once_with(use_cache=False)
        assert result == mock_instance


class TestRecognizerRegistryRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        registry = RecognizerRegistry()
        repr_str = repr(registry)
        
        assert "RecognizerRegistry" in repr_str
        assert "available=" in repr_str

