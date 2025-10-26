"""Detection Stage: Layout block detection."""

from __future__ import annotations

import numpy as np

from pipeline.types import Block, ColumnLayout, Detector


class DetectionStage:
    """Stage 2: Detection - Layout block detection."""

    def __init__(self, detector: Detector):
        """Initialize DetectionStage.

        Args:
            detector: Layout detector instance
        """
        self.detector = detector

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in image.

        Args:
            image: Page image as numpy array

        Returns:
            List of detected blocks with bbox, type, confidence
        """
        blocks = self.detector.detect(image)
        return blocks

    def extract_column_layout(self, blocks: list[Block]) -> ColumnLayout | None:
        """Extract column layout information from sorted blocks.

        Args:
            blocks: Sorted blocks (may have column_index)

        Returns:
            Column layout dict or None
        """
        # Check if any blocks have column_index
        has_columns = any(r.column_index is not None for r in blocks)

        if not has_columns:
            return None

        # Extract unique columns
        column_indices = {r.column_index for r in blocks if r.column_index is not None}

        if not column_indices:
            return None

        # Build column layout info (filter out None values)
        columns = []
        for col_idx in sorted(column_indices):
            col_blocks = [r for r in blocks if r.column_index == col_idx]
            if col_blocks:
                # Get bbox if available
                first_block = col_blocks[0]
                if first_block.bbox:
                    bbox = first_block.bbox
                    columns.append(
                        {
                            "index": col_idx,
                            "x0": int(bbox.x0),
                            "x1": int(bbox.x1),
                            "center": bbox.center[0],
                            "width": bbox.width,
                        }
                    )

        if not columns:
            return None

        return {"columns": columns}
