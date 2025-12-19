"""Recognition Stage: Block-level text extraction using VLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipeline.types import Block

from .base import BaseStage

if TYPE_CHECKING:
    from pipeline.distributed import RayRecognizerPool


class RecognitionStage(BaseStage[list[Block], list[Block]]):
    """Stage 4: Recognition - Text extraction from blocks.

    This stage extracts text from detected blocks using VLM (Vision Language Model).
    Supports both local recognition and Ray-based distributed recognition.

    For image/figure/chart blocks, generates descriptions when enable_figure_description
    is True (default). The description is stored in the block's `description` field.
    """

    name = "recognition"

    def __init__(
        self,
        recognizer,
        ray_recognizer_pool: RayRecognizerPool | None = None,
        enable_figure_description: bool = True,
    ):
        """Initialize RecognitionStage.

        Args:
            recognizer: Text recognizer instance (TextRecognizer)
            ray_recognizer_pool: Optional Ray recognizer pool for multi-GPU parallelization
            enable_figure_description: Whether to generate descriptions for image/figure blocks.
                When True, image blocks will have their `description` field populated.
                Default: True.
        """
        self.recognizer = recognizer
        self.ray_recognizer_pool = ray_recognizer_pool
        self.enable_figure_description = enable_figure_description

    def _process_impl(self, input_data: list[Block], **context: Any) -> list[Block]:
        """Extract text from blocks using VLM.

        Args:
            input_data: List of blocks with bounding boxes
            **context: Must include 'image' (page image as numpy array)

        Returns:
            List of blocks with text field populated (and description for image blocks)
        """
        image = context.get("image")
        if image is None:
            raise ValueError("RecognitionStage requires 'image' in context")

        # Use Ray pool if available, otherwise use regular recognizer
        if self.ray_recognizer_pool is not None:
            processed_blocks = self.ray_recognizer_pool.recognize_blocks(
                image, input_data, enable_figure_description=self.enable_figure_description
            )
        else:
            # Use existing TextRecognizer.process_blocks method
            # Note: process_blocks expects (image, blocks) order
            processed_blocks = self.recognizer.process_blocks(
                image, input_data, enable_figure_description=self.enable_figure_description
            )
        return processed_blocks
