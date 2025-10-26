"""Block Correction Stage: Block-level text correction (placeholder for future)."""

from __future__ import annotations

import logging

from pipeline.types import Block

logger = logging.getLogger(__name__)


class BlockCorrectionStage:
    """Stage 5: BlockCorrection - Block-level text correction (optional)."""

    def __init__(self, enable: bool = False):
        """Initialize BlockCorrectionStage.

        Args:
            enable: Whether to enable block-level correction (default: False)
        """
        self.enable = enable

    def correct_blocks(self, blocks: list[Block]) -> list[Block]:
        """Correct text in each block individually.

        Args:
            blocks: List of blocks with text field

        Returns:
            List of blocks with corrected_text field
        """
        if not self.enable:
            # If disabled, just copy text to corrected_text
            for block in blocks:
                if block.text is not None:
                    block.corrected_text = block.text
            return blocks

        # TODO: Implement block-level correction with VLM
        # For now, just copy text to corrected_text
        logger.warning("Block-level correction is not yet implemented. Using text as-is.")
        for block in blocks:
            if block.text is not None:
                block.corrected_text = block.text
                block.correction_ratio = 0.0

        return blocks
