"""Base detector class and interface.

This module defines the abstract base class for all layout detectors,
providing a consistent interface and common functionality.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pipeline.types import BBox, Block

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["BaseDetector"]


class BaseDetector(ABC):
    """Abstract base class for all layout detectors.

    All detector implementations should inherit from this class and implement
    the `_detect_impl` method. This base class provides:

    - Consistent interface (detect, detect_batch)
    - Common block creation helper
    - Input validation
    - Logging and error handling

    Attributes:
        name: Detector name (e.g., "doclayout-yolo")
        source: Source identifier for Block.source field
        confidence_threshold: Minimum confidence for detections

    Example:
        >>> class MyDetector(BaseDetector):
        ...     name = "my-detector"
        ...     source = "my-detector"
        ...
        ...     def _detect_impl(self, image: np.ndarray) -> list[Block]:
        ...         # Implementation here
        ...         pass
    """

    # Subclasses should override these
    name: str = "base-detector"
    source: str = "base-detector"

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize base detector.

        Args:
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be 0.0-1.0, got {confidence_threshold}")

        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def _detect_impl(self, image: np.ndarray) -> list[Block]:
        """Internal detection implementation.

        Subclasses must implement this method to perform actual detection.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected Block objects
        """
        pass

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in image.

        This method wraps _detect_impl with validation and error handling.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected Block objects

        Raises:
            ValueError: If image is invalid
            DetectionError: If detection fails
        """
        # Validate input
        if image is None:
            raise ValueError("Image cannot be None")

        if len(image.shape) != 3:
            raise ValueError(f"Image must have 3 dimensions (H, W, C), got {len(image.shape)}")

        h, w, c = image.shape
        if c not in (1, 3, 4):
            raise ValueError(f"Image must have 1, 3, or 4 channels, got {c}")

        if h == 0 or w == 0:
            logger.warning("Empty image (size=%dx%d), returning empty blocks", w, h)
            return []

        try:
            blocks = self._detect_impl(image)

            # Validate and filter blocks
            valid_blocks = self._validate_blocks(blocks, image.shape)

            logger.debug(
                "%s detected %d blocks (filtered from %d)",
                self.name,
                len(valid_blocks),
                len(blocks),
            )

            return valid_blocks

        except Exception as e:
            logger.error("%s detection failed: %s", self.name, e)
            raise

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
        """Detect layout blocks in multiple images.

        Default implementation processes images sequentially.
        Subclasses may override for parallel/batch processing.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of block lists, one per image
        """
        return [self.detect(image) for image in images]

    def _validate_blocks(
        self,
        blocks: list[Block],
        image_shape: tuple[int, int, int],
    ) -> list[Block]:
        """Validate and filter blocks.

        Args:
            blocks: List of detected blocks
            image_shape: Image shape (H, W, C)

        Returns:
            Filtered list of valid blocks
        """
        h, w, _ = image_shape
        valid_blocks = []

        for block in blocks:
            # Filter by confidence
            if block.detection_confidence is not None:
                if block.detection_confidence < self.confidence_threshold:
                    continue

            # Validate bbox bounds
            bbox = block.bbox
            if bbox.x0 < 0 or bbox.y0 < 0:
                logger.debug("Clipping negative bbox coordinates")
                block = replace(
                    block, bbox=BBox(max(0, bbox.x0), max(0, bbox.y0), bbox.x1, bbox.y1)
                )
                bbox = block.bbox  # Update bbox reference

            if bbox.x1 > w or bbox.y1 > h:
                logger.debug("Clipping bbox exceeding image bounds")
                block = replace(
                    block, bbox=BBox(bbox.x0, bbox.y0, min(w, bbox.x1), min(h, bbox.y1))
                )

            # Skip zero-area blocks
            if block.bbox.width <= 0 or block.bbox.height <= 0:
                logger.debug("Skipping zero-area block")
                continue

            valid_blocks.append(block)

        return valid_blocks

    def _create_block(
        self,
        block_type: str,
        bbox: tuple[int, int, int, int],
        confidence: float,
        **kwargs: Any,
    ) -> Block:
        """Helper to create Block with consistent source.

        Args:
            block_type: Block type (e.g., "text", "title", "table")
            bbox: Bounding box as (x0, y0, x1, y1)
            confidence: Detection confidence (0.0-1.0)
            **kwargs: Additional Block fields

        Returns:
            Block object with source set to self.source
        """
        return Block(
            type=block_type,
            bbox=BBox.from_xyxy(*bbox),
            detection_confidence=confidence,
            source=self.source,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r}, threshold={self.confidence_threshold})"

