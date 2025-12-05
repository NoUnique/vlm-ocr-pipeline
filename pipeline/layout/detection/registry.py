"""Detector registry for managing detector implementations.

This module provides a centralized registry for detector classes,
enabling dynamic detector discovery and lazy loading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.types import Detector

logger = logging.getLogger(__name__)

__all__ = ["DetectorRegistry", "detector_registry"]


class DetectorRegistry:
    """Registry for detector implementations.

    Provides:
    - Lazy loading of detector classes
    - Dynamic registration of new detectors
    - Unified access to all available detectors

    Example:
        >>> from pipeline.layout.detection.registry import detector_registry
        >>> detector = detector_registry.create("doclayout-yolo", confidence_threshold=0.5)
        >>> available = detector_registry.list_available()
    """

    # Built-in detector mappings (name -> import path)
    _BUILTIN_DETECTORS: dict[str, tuple[str, str]] = {
        "doclayout-yolo": (
            "pipeline.layout.detection.doclayout_yolo",
            "DocLayoutYOLODetector",
        ),
        "mineru-doclayout-yolo": (
            "pipeline.layout.detection.mineru",
            "MinerUDocLayoutYOLODetector",
        ),
        "mineru-vlm": (
            "pipeline.layout.detection.mineru",
            "MinerUVLMDetector",
        ),
        "paddleocr-doclayout-v2": (
            "pipeline.layout.detection.paddleocr",
            "PPDocLayoutV2Detector",
        ),
    }

    # Aliases for backward compatibility
    _ALIASES: dict[str, str] = {
        "layout-detector": "doclayout-yolo",
        "docLayoutYolo": "doclayout-yolo",
        "mineru": "mineru-doclayout-yolo",
        "paddleocr": "paddleocr-doclayout-v2",
    }

    def __init__(self) -> None:
        """Initialize detector registry."""
        self._custom_detectors: dict[str, Callable[..., Detector]] = {}
        self._loaded_classes: dict[str, type] = {}

    def register(
        self,
        name: str,
        detector_class: type | Callable[..., Detector],
    ) -> None:
        """Register a custom detector.

        Args:
            name: Detector name for lookup
            detector_class: Detector class or factory function

        Example:
            >>> registry.register("my-detector", MyDetectorClass)
        """
        if name in self._BUILTIN_DETECTORS:
            logger.warning("Overriding built-in detector: %s", name)

        self._custom_detectors[name] = detector_class
        logger.debug("Registered detector: %s", name)

    def get_class(self, name: str) -> type:
        """Get detector class by name (lazy loading).

        Args:
            name: Detector name

        Returns:
            Detector class

        Raises:
            ValueError: If detector not found
        """
        # Resolve aliases
        resolved_name = self._ALIASES.get(name, name)

        # Check custom detectors first
        if resolved_name in self._custom_detectors:
            return self._custom_detectors[resolved_name]

        # Check cache
        if resolved_name in self._loaded_classes:
            return self._loaded_classes[resolved_name]

        # Lazy load built-in detector
        if resolved_name in self._BUILTIN_DETECTORS:
            module_path, class_name = self._BUILTIN_DETECTORS[resolved_name]
            try:
                import importlib

                module = importlib.import_module(module_path)
                detector_class = getattr(module, class_name)
                self._loaded_classes[resolved_name] = detector_class
                return detector_class
            except ImportError as e:
                raise ImportError(
                    f"Failed to import detector '{resolved_name}': {e}"
                ) from e
            except AttributeError as e:
                raise ValueError(
                    f"Detector class '{class_name}' not found in '{module_path}': {e}"
                ) from e

        raise ValueError(
            f"Unknown detector: '{name}'. "
            f"Available: {', '.join(self.list_available())}"
        )

    def create(self, name: str, **kwargs: Any) -> Detector:
        """Create detector instance.

        Args:
            name: Detector name
            **kwargs: Arguments for detector constructor

        Returns:
            Detector instance

        Example:
            >>> detector = registry.create("doclayout-yolo", confidence_threshold=0.5)
        """
        detector_class = self.get_class(name)
        return detector_class(**kwargs)

    def list_available(self) -> list[str]:
        """List all available detector names.

        Returns:
            Sorted list of detector names
        """
        all_names = set(self._BUILTIN_DETECTORS.keys())
        all_names.update(self._custom_detectors.keys())
        return sorted(all_names)

    def list_aliases(self) -> dict[str, str]:
        """List all detector aliases.

        Returns:
            Dictionary mapping alias to canonical name
        """
        return dict(self._ALIASES)

    def is_available(self, name: str) -> bool:
        """Check if detector is available.

        Args:
            name: Detector name

        Returns:
            True if detector is available
        """
        resolved_name = self._ALIASES.get(name, name)
        return (
            resolved_name in self._BUILTIN_DETECTORS
            or resolved_name in self._custom_detectors
        )

    def __contains__(self, name: str) -> bool:
        """Check if detector is in registry."""
        return self.is_available(name)

    def __repr__(self) -> str:
        """String representation."""
        return f"DetectorRegistry(available={self.list_available()})"


# Global registry instance
detector_registry = DetectorRegistry()

