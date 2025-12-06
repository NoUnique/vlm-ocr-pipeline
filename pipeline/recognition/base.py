"""Base recognizer class and interface.

This module defines the abstract base class for all text recognizers,
providing a consistent interface and common functionality.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pipeline.types import Block

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["BaseRecognizer"]


class BaseRecognizer(ABC):
    """Abstract base class for all text recognizers.

    All recognizer implementations should inherit from this class and implement
    the abstract methods. This base class provides:

    - Consistent interface (process_blocks, correct_text)
    - Common text processing helpers
    - Input validation
    - Logging and error handling

    Attributes:
        name: Recognizer name (e.g., "gemini", "openai", "paddleocr-vl")
        supports_correction: Whether this recognizer supports text correction
        supports_batch: Whether this recognizer supports batch processing

    Example:
        >>> class MyRecognizer(BaseRecognizer):
        ...     name = "my-recognizer"
        ...     supports_correction = True
        ...
        ...     def _process_blocks_impl(self, image, blocks):
        ...         # Implementation here
        ...         pass
        ...
        ...     def _correct_text_impl(self, text):
        ...         # Implementation here
        ...         pass
    """

    # Subclasses should override these
    name: str = "base-recognizer"
    supports_correction: bool = True
    supports_batch: bool = False

    @abstractmethod
    def _process_blocks_impl(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
    ) -> list[Block]:
        """Internal implementation of block processing.

        Subclasses must implement this method to extract text from blocks.

        Args:
            image: Input image as numpy array (H, W, C)
            blocks: Sequence of blocks to process

        Returns:
            List of blocks with text populated
        """

    def process_blocks(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
    ) -> list[Block]:
        """Extract text from blocks in image.

        This method wraps _process_blocks_impl with validation and error handling.

        Args:
            image: Input image as numpy array (H, W, C)
            blocks: Sequence of blocks to process

        Returns:
            List of blocks with text field populated

        Raises:
            ValueError: If image or blocks are invalid
        """
        # Validate inputs
        if image is None:
            raise ValueError("Image cannot be None")

        if blocks is None:
            raise ValueError("Blocks cannot be None")

        if len(blocks) == 0:
            logger.debug("No blocks to process")
            return []

        try:
            result = self._process_blocks_impl(image, blocks)
            logger.debug(
                "%s processed %d blocks",
                self.name,
                len(result),
            )
            return result

        except Exception as e:
            logger.error("%s processing failed: %s", self.name, e)
            raise

    def _correct_text_impl(self, text: str) -> str | dict[str, Any]:
        """Internal implementation of text correction.

        Default implementation returns text unchanged.
        Subclasses can override for actual correction.

        Args:
            text: Text to correct

        Returns:
            Corrected text or dict with corrected_text and correction_ratio
        """
        return text

    def correct_text(self, text: str) -> str | dict[str, Any]:
        """Perform text correction.

        This method wraps _correct_text_impl with validation.

        Args:
            text: Text to correct

        Returns:
            Corrected text (str) or dict with keys:
            - corrected_text: The corrected text
            - correction_ratio: Ratio of changes (0.0 = no change, 1.0 = complete change)

        Raises:
            ValueError: If text is None
        """
        if not self.supports_correction:
            logger.debug("%s does not support text correction", self.name)
            return text

        if text is None:
            raise ValueError("Text cannot be None")

        if not text.strip():
            return text

        try:
            result = self._correct_text_impl(text)
            return result

        except Exception as e:
            logger.error("%s text correction failed: %s", self.name, e)
            raise

    def process_blocks_batch(
        self,
        images: list[np.ndarray],
        blocks_list: list[Sequence[Block]],
    ) -> list[list[Block]]:
        """Process multiple images and their blocks.

        Default implementation processes images sequentially.
        Subclasses may override for parallel/batch processing.

        Args:
            images: List of images
            blocks_list: List of block sequences, one per image

        Returns:
            List of block lists, one per image
        """
        if len(images) != len(blocks_list):
            raise ValueError(
                f"Mismatched lengths: {len(images)} images vs {len(blocks_list)} block lists"
            )

        return [
            self.process_blocks(image, blocks)
            for image, blocks in zip(images, blocks_list, strict=False)
        ]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"supports_correction={self.supports_correction})"
        )

