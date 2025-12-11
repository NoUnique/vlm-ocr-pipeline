"""Detection Stage: Layout block detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.types import Block, ColumnLayout, Detector

from .base import BaseStage

if TYPE_CHECKING:
    from pipeline.distributed import RayDetectorPool


class DetectionStage(BaseStage[np.ndarray, list[Block]]):
    """Stage 2: Detection - Layout block detection.

    This stage detects layout blocks (text, tables, figures, etc.) in page images
    using a configured detector. Supports both local detection and Ray-based
    distributed detection.
    """

    name = "detection"

    def __init__(self, detector: Detector, ray_detector_pool: RayDetectorPool | None = None):
        """Initialize DetectionStage.

        Args:
            detector: Layout detector instance
            ray_detector_pool: Optional Ray detector pool for multi-GPU parallelization
        """
        self.detector = detector
        self.ray_detector_pool = ray_detector_pool

    def _process_impl(self, input_data: np.ndarray, **context: Any) -> list[Block]:
        """Detect layout blocks in image.

        Args:
            input_data: Page image as numpy array
            **context: Additional context (unused)

        Returns:
            List of detected blocks with bbox, type, confidence
        """
        # Use Ray pool if available, otherwise use regular detector
        if self.ray_detector_pool is not None:
            blocks = self.ray_detector_pool.detect(input_data)
        else:
            blocks = self.detector.detect(input_data)
        return blocks

    @staticmethod
    def extract_column_layout(blocks: list[Block]) -> ColumnLayout | None:
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
