"""Recognizer registry for managing recognizer implementations.

This module provides a centralized registry for recognizer classes,
enabling dynamic recognizer discovery and lazy loading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.types import Recognizer

logger = logging.getLogger(__name__)

__all__ = ["RecognizerRegistry", "recognizer_registry"]


class RecognizerRegistry:
    """Registry for recognizer implementations.

    Provides:
    - Lazy loading of recognizer classes
    - Dynamic registration of new recognizers
    - Unified access to all available recognizers
    - Model name pattern matching for auto-detection

    Example:
        >>> from pipeline.recognition.registry import recognizer_registry
        >>> recognizer = recognizer_registry.create("gemini-2.5-flash")
        >>> available = recognizer_registry.list_available()
    """

    # Built-in recognizer mappings (name -> (module, class, default_kwargs))
    _BUILTIN_RECOGNIZERS: dict[str, tuple[str, str, dict[str, Any]]] = {
        # API-based recognizers
        "openai": (
            "pipeline.recognition",
            "TextRecognizer",
            {"backend": "openai"},
        ),
        "gemini": (
            "pipeline.recognition",
            "TextRecognizer",
            {"backend": "gemini"},
        ),
        # Local recognizers
        "paddleocr-vl": (
            "pipeline.recognition.paddleocr",
            "PaddleOCRVLRecognizer",
            {},
        ),
        "deepseek-ocr": (
            "pipeline.recognition.deepseek",
            "DeepSeekOCRRecognizer",
            {},
        ),
    }

    # Model name patterns for auto-detection
    _MODEL_PATTERNS: dict[str, str] = {
        "gemini": "gemini",  # gemini-*, gemini-2.5-flash, etc.
        "gpt": "openai",  # gpt-4, gpt-4o, etc.
        "claude": "openai",  # claude-3, etc. (via OpenRouter)
        "deepseek": "deepseek-ocr",
        "paddleocr": "paddleocr-vl",
    }

    # Aliases for backward compatibility
    _ALIASES: dict[str, str] = {
        "text-recognizer": "gemini",
        "vlm": "gemini",
        "ocr": "paddleocr-vl",
    }

    def __init__(self) -> None:
        """Initialize recognizer registry."""
        self._custom_recognizers: dict[str, Callable[..., Recognizer]] = {}
        self._loaded_classes: dict[str, type] = {}

    def register(
        self,
        name: str,
        recognizer_class: type | Callable[..., Recognizer],
    ) -> None:
        """Register a custom recognizer.

        Args:
            name: Recognizer name for lookup
            recognizer_class: Recognizer class or factory function

        Example:
            >>> registry.register("my-recognizer", MyRecognizerClass)
        """
        if name in self._BUILTIN_RECOGNIZERS:
            logger.warning("Overriding built-in recognizer: %s", name)

        self._custom_recognizers[name] = recognizer_class
        logger.debug("Registered recognizer: %s", name)

    def resolve_name(self, name: str) -> tuple[str, dict[str, Any]]:
        """Resolve recognizer name and extract model info.

        Handles:
        - Direct recognizer names ("openai", "gemini")
        - Model names with patterns ("gemini-2.5-flash", "gpt-4o")
        - Aliases ("vlm" -> "gemini")

        Args:
            name: Recognizer or model name

        Returns:
            Tuple of (resolved_name, extra_kwargs)
        """
        # Check aliases first
        if name in self._ALIASES:
            return self._ALIASES[name], {}

        # Check direct recognizer names
        if name in self._BUILTIN_RECOGNIZERS or name in self._custom_recognizers:
            return name, {}

        # Try pattern matching for model names
        name_lower = name.lower()
        for pattern, recognizer_name in self._MODEL_PATTERNS.items():
            if pattern in name_lower:
                return recognizer_name, {"model": name}

        # Default to gemini for unknown names
        logger.warning(
            "Unknown recognizer '%s', defaulting to 'gemini' with model='%s'",
            name,
            name,
        )
        return "gemini", {"model": name}

    def get_class(self, name: str) -> tuple[type, dict[str, Any]]:
        """Get recognizer class by name (lazy loading).

        Args:
            name: Recognizer name

        Returns:
            Tuple of (recognizer_class, default_kwargs)

        Raises:
            ValueError: If recognizer not found
        """
        resolved_name, extra_kwargs = self.resolve_name(name)

        # Check custom recognizers first
        if resolved_name in self._custom_recognizers:
            return self._custom_recognizers[resolved_name], extra_kwargs

        # Check cache
        if resolved_name in self._loaded_classes:
            _, _, default_kwargs = self._BUILTIN_RECOGNIZERS[resolved_name]
            return self._loaded_classes[resolved_name], {**default_kwargs, **extra_kwargs}

        # Lazy load built-in recognizer
        if resolved_name in self._BUILTIN_RECOGNIZERS:
            module_path, class_name, default_kwargs = self._BUILTIN_RECOGNIZERS[resolved_name]
            try:
                import importlib

                module = importlib.import_module(module_path)
                recognizer_class = getattr(module, class_name)
                self._loaded_classes[resolved_name] = recognizer_class
                return recognizer_class, {**default_kwargs, **extra_kwargs}
            except ImportError as e:
                raise ImportError(
                    f"Failed to import recognizer '{resolved_name}': {e}"
                ) from e
            except AttributeError as e:
                raise ValueError(
                    f"Recognizer class '{class_name}' not found in '{module_path}': {e}"
                ) from e

        raise ValueError(
            f"Unknown recognizer: '{name}'. "
            f"Available: {', '.join(self.list_available())}"
        )

    def create(self, name: str, **kwargs: Any) -> Recognizer:
        """Create recognizer instance.

        Args:
            name: Recognizer name or model name
            **kwargs: Arguments for recognizer constructor

        Returns:
            Recognizer instance

        Example:
            >>> recognizer = registry.create("gemini-2.5-flash")
            >>> recognizer = registry.create("openai", model="gpt-4o")
        """
        recognizer_class, default_kwargs = self.get_class(name)
        merged_kwargs = {**default_kwargs, **kwargs}
        return recognizer_class(**merged_kwargs)

    def list_available(self) -> list[str]:
        """List all available recognizer names.

        Returns:
            Sorted list of recognizer names
        """
        all_names = set(self._BUILTIN_RECOGNIZERS.keys())
        all_names.update(self._custom_recognizers.keys())
        return sorted(all_names)

    def list_aliases(self) -> dict[str, str]:
        """List all recognizer aliases.

        Returns:
            Dictionary mapping alias to canonical name
        """
        return dict(self._ALIASES)

    def is_available(self, name: str) -> bool:
        """Check if recognizer is available.

        Args:
            name: Recognizer name

        Returns:
            True if recognizer is available
        """
        resolved_name, _ = self.resolve_name(name)
        return (
            resolved_name in self._BUILTIN_RECOGNIZERS
            or resolved_name in self._custom_recognizers
        )

    def __contains__(self, name: str) -> bool:
        """Check if recognizer is in registry."""
        return self.is_available(name)

    def __repr__(self) -> str:
        """String representation."""
        return f"RecognizerRegistry(available={self.list_available()})"


# Global registry instance
recognizer_registry = RecognizerRegistry()

