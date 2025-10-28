"""Recognition Stage: Block-level text extraction using VLM."""

from __future__ import annotations

import numpy as np

from pipeline.types import Block


class RecognitionStage:
    """Stage 4: Recognition - Text extraction from blocks."""

    def __init__(self, recognizer):
        """Initialize RecognitionStage.

        Args:
            recognizer: Text recognizer instance (TextRecognizer)
        """
        self.recognizer = recognizer

    def recognize_blocks(self, blocks: list[Block], image: np.ndarray) -> list[Block]:
        """Extract text from blocks using VLM.

        Args:
            blocks: List of blocks with bounding boxes
            image: Page image as numpy array

        Returns:
            List of blocks with text field populated
        """
        # Use existing TextRecognizer.process_blocks method
        # Note: process_blocks expects (image, blocks) order
        processed_blocks = self.recognizer.process_blocks(image, blocks)
        return processed_blocks
