"""Base layout detector class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

__all__ = ["LayoutDetector"]


class LayoutDetector:
    """Base class for layout detection."""

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect layout regions in an image.

        Args:
            image: Input image

        Returns:
            List of detected regions
        """
        raise NotImplementedError("Subclasses must implement detect()")
