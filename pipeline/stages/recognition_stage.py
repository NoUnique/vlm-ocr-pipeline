"""Recognition Stage: Block-level text extraction using VLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.types import Block

if TYPE_CHECKING:
    from pipeline.distributed import RayRecognizerPool


class RecognitionStage:
    """Stage 4: Recognition - Text extraction from blocks."""

    def __init__(self, recognizer, ray_recognizer_pool: RayRecognizerPool | None = None):
        """Initialize RecognitionStage.

        Args:
            recognizer: Text recognizer instance (TextRecognizer)
            ray_recognizer_pool: Optional Ray recognizer pool for multi-GPU parallelization
        """
        self.recognizer = recognizer
        self.ray_recognizer_pool = ray_recognizer_pool

    def recognize_blocks(self, blocks: list[Block], image: np.ndarray) -> list[Block]:
        """Extract text from blocks using VLM.

        Args:
            blocks: List of blocks with bounding boxes
            image: Page image as numpy array

        Returns:
            List of blocks with text field populated
        """
        # Use Ray pool if available, otherwise use regular recognizer
        if self.ray_recognizer_pool is not None:
            processed_blocks = self.ray_recognizer_pool.recognize_blocks(image, blocks)
        else:
            # Use existing TextRecognizer.process_blocks method
            # Note: process_blocks expects (image, blocks) order
            processed_blocks = self.recognizer.process_blocks(image, blocks)
        return processed_blocks
