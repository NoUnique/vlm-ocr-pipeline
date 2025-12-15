"""Sorter registry for managing sorter implementations.

This module provides a centralized registry for sorter classes,
enabling dynamic sorter discovery and lazy loading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pipeline.exceptions import InvalidConfigError

if TYPE_CHECKING:
    from pipeline.types import Sorter

logger = logging.getLogger(__name__)

__all__ = ["SorterRegistry", "sorter_registry"]


class SorterRegistry:
    """Registry for sorter implementations.

    Provides:
    - Lazy loading of sorter classes
    - Dynamic registration of new sorters
    - Unified access to all available sorters

    Example:
        >>> from pipeline.layout.ordering.registry import sorter_registry
        >>> sorter = sorter_registry.create("mineru-xycut")
        >>> available = sorter_registry.list_available()
    """

    # Built-in sorter mappings (name -> (module_path, class_name))
    _BUILTIN_SORTERS: dict[str, tuple[str, str]] = {
        "pymupdf": (
            "pipeline.layout.ordering.pymupdf",
            "MultiColumnSorter",
        ),
        "mineru-layoutreader": (
            "pipeline.layout.ordering.mineru",
            "MinerULayoutReaderSorter",
        ),
        "mineru-xycut": (
            "pipeline.layout.ordering.mineru",
            "MinerUXYCutSorter",
        ),
        "mineru-vlm": (
            "pipeline.layout.ordering.mineru",
            "MinerUVLMSorter",
        ),
        "olmocr-vlm": (
            "pipeline.layout.ordering.olmocr",
            "OlmOCRVLMSorter",
        ),
        "paddleocr-doclayout-v2": (
            "pipeline.layout.ordering.paddleocr",
            "PPDocLayoutV2Sorter",
        ),
    }

    def __init__(self) -> None:
        """Initialize sorter registry."""
        self._custom_sorters: dict[str, Callable[..., Sorter]] = {}
        self._loaded_classes: dict[str, type] = {}

    def register(
        self,
        name: str,
        sorter_class: type | Callable[..., Sorter],
    ) -> None:
        """Register a custom sorter.

        Args:
            name: Sorter name for lookup
            sorter_class: Sorter class or factory function
        """
        if name in self._BUILTIN_SORTERS:
            logger.warning("Overriding built-in sorter: %s", name)

        self._custom_sorters[name] = sorter_class
        logger.debug("Registered sorter: %s", name)

    def get_class(self, name: str) -> type[Sorter] | Callable[..., Sorter]:
        """Get sorter class by name (lazy loading).

        Args:
            name: Sorter name

        Returns:
            Sorter class

        Raises:
            InvalidConfigError: If sorter not found
        """
        # Check custom sorters first
        if name in self._custom_sorters:
            return self._custom_sorters[name]

        # Check cached loaded classes
        if name in self._loaded_classes:
            return self._loaded_classes[name]

        # Try to load built-in sorter
        if name not in self._BUILTIN_SORTERS:
            available = ", ".join(self.list_available())
            raise InvalidConfigError(f"Unknown sorter: {name}. Available: {available}")

        module_path, class_name = self._BUILTIN_SORTERS[name]
        try:
            import importlib

            module = importlib.import_module(module_path)
            sorter_class = getattr(module, class_name)

            if sorter_class is None:
                raise InvalidConfigError(f"Sorter '{name}' is not available (dependency missing)")

            self._loaded_classes[name] = sorter_class
            return sorter_class

        except ImportError as e:
            raise InvalidConfigError(f"Failed to load sorter '{name}': {e}") from e

    def create(self, name: str, **kwargs: Any) -> Sorter:
        """Create a sorter instance.

        Args:
            name: Sorter name
            **kwargs: Arguments passed to sorter constructor

        Returns:
            Sorter instance
        """
        sorter_class = self.get_class(name)
        return sorter_class(**kwargs)

    def list_available(self) -> list[str]:
        """List all available sorter names."""
        names = set(self._BUILTIN_SORTERS.keys())
        names.update(self._custom_sorters.keys())
        return sorted(names)

    def is_available(self, name: str) -> bool:
        """Check if a sorter is available."""
        try:
            self.get_class(name)
            return True
        except InvalidConfigError:
            return False


# Global singleton instance
sorter_registry = SorterRegistry()
