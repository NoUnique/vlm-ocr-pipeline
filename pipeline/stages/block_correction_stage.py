"""Block Correction Stage: Block-level text correction (placeholder for future)."""

from __future__ import annotations

import logging
from typing import Any

from pipeline.types import Block

from .base import BaseStage

logger = logging.getLogger(__name__)


class BlockCorrectionStage(BaseStage[list[Block], list[Block]]):
    """Stage 5: BlockCorrection - Block-level text correction (optional).

    This stage performs optional block-level text correction using VLM.
    Currently a placeholder that copies text to corrected_text.
    """

    name = "block_correction"

    def __init__(self, enable: bool = False):
        """Initialize BlockCorrectionStage.

        Args:
            enable: Whether to enable block-level correction (default: False)
        """
        self.enable = enable

    def _process_impl(self, input_data: list[Block], **context: Any) -> list[Block]:
        """Correct text in each block individually.

        Args:
            input_data: List of blocks with text field
            **context: Additional context (unused)

        Returns:
            List of blocks with corrected_text field
        """
        if not self.enable:
            # If disabled, just copy text to corrected_text
            for block in input_data:
                if block.text is not None:
                    block.corrected_text = block.text
            return input_data

        # TODO: Implement block-level correction with VLM
        # For now, just copy text to corrected_text
        logger.warning("Block-level correction is not yet implemented. Using text as-is.")
        for block in input_data:
            if block.text is not None:
                block.corrected_text = block.text
                block.correction_ratio = 0.0

        return input_data
