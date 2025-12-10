"""MinerU LayoutReader sorter implementation.

MinerU LayoutReader BBox Format:
- Input: Unified BBox (any format) → converted to scaled [x0, y0, x1, y1]
- Model Input: Scaled to 1000x1000 coordinate space
- Example: [100, 50, 300, 200] → [125, 62, 375, 250] (if page is 800x800)
- Model: LayoutLMv3ForTokenClassification
- Limitation: Maximum 200 lines per page
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Any

from pipeline.constants import LAYOUTREADER_SCALE
from pipeline.types import Block, BlockType, Sorter

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

MAX_LINES = 200  # LayoutReader maximum line limit


class MinerULayoutReaderSorter(Sorter):
    """Sorter using MinerU's LayoutReader (LayoutLMv3) model.

    LayoutReader uses a transformer-based model to predict reading order
    for document blocks. It's more accurate than heuristic methods but
    requires model loading and has a line limit (~200 lines).

    The sorter first splits blocks into lines, then uses LayoutReader
    to order the lines, and finally assigns block-level ordering based
    on line ordering.
    """

    def __init__(self, device: str | None = None) -> None:
        """Initialize LayoutReader sorter.

        Args:
            device: Device for model inference ("cpu", "cuda", etc.)
                   If None, auto-detected

        Raises:
            ImportError: If required dependencies not available
        """
        try:
            import torch  # noqa: F401
            from mineru.utils.block_sort import ModelSingleton
        except ImportError as e:
            raise ImportError(
                "MinerU LayoutReader dependencies not available. Install with: pip install torch transformers"
            ) from e

        self.device = device
        self.model_manager = ModelSingleton()

        logger.info("MinerU LayoutReader sorter initialized")

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Sort blocks using LayoutReader model.

        Args:
            blocks: Detected blocks in unified format
            image: Page image for dimension reference
            **kwargs: Additional context (unused)

        Returns:
            Sorted blocks with order field added

        Example:
            >>> sorter = MinerULayoutReaderSorter()
            >>> sorted_blocks = sorter.sort(blocks, image)
            >>> sorted_blocks[0].order
            0
        """
        if not blocks:
            return blocks

        page_height, page_width = image.shape[:2]

        line_height = self._estimate_line_height(blocks)
        lines_data = self._split_blocks_into_lines(blocks, line_height, page_width, page_height)

        if len(lines_data) > MAX_LINES:
            logger.warning("Too many lines (%d > %d), falling back to simple sort", len(lines_data), MAX_LINES)
            return self._fallback_sort(blocks)

        try:
            sorted_line_indices = self._sort_lines_with_layoutreader(lines_data, page_width, page_height)
        except Exception as e:
            logger.error("LayoutReader failed: %s, falling back to simple sort", e)
            return self._fallback_sort(blocks)

        sorted_blocks = self._assign_block_ordering(blocks, lines_data, sorted_line_indices)

        logger.debug("Sorted %d blocks using LayoutReader", len(sorted_blocks))

        return sorted_blocks

    def _estimate_line_height(self, blocks: list[Block]) -> float:
        """Estimate typical line height from text blocks."""
        text_types = {BlockType.PLAIN_TEXT, BlockType.TEXT, BlockType.TITLE}
        heights = []

        for block in blocks:
            if block.type in text_types and block.bbox is not None:
                heights.append(block.bbox.height)

        if heights:
            return statistics.median(heights)
        else:
            return 10.0  # Default fallback

    def _split_blocks_into_lines(
        self,
        blocks: list[Block],
        line_height: float,
        page_width: int,
        page_height: int,
    ) -> list[dict[str, Any]]:
        """Split blocks into line-level boxes.

        For large text blocks, split into multiple lines.
        For other blocks (images, tables), use as is.
        """
        lines_data = []

        for block_idx, block in enumerate(blocks):
            bbox = block.bbox
            if bbox is None:
                continue  # Skip blocks without bbox

            block_type = block.type

            if block_type in {BlockType.PLAIN_TEXT, BlockType.TEXT, BlockType.TITLE}:
                block_height = bbox.height

                if block_height > line_height * 2:
                    num_lines = max(2, int(block_height / line_height))
                    line_h = block_height / num_lines

                    for i in range(num_lines):
                        y_start = bbox.y0 + i * line_h
                        y_end = bbox.y0 + (i + 1) * line_h
                        lines_data.append(
                            {
                                "bbox": [int(bbox.x0), int(y_start), int(bbox.x1), int(y_end)],
                                "block_idx": block_idx,
                            }
                        )
                else:
                    lines_data.append(
                        {
                            "bbox": [int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)],
                            "block_idx": block_idx,
                        }
                    )
            else:
                block_height = bbox.height
                num_lines = min(3, max(1, int(block_height / line_height)))
                line_h = block_height / num_lines

                for i in range(num_lines):
                    y_start = bbox.y0 + i * line_h
                    y_end = bbox.y0 + (i + 1) * line_h
                    lines_data.append(
                        {
                            "bbox": [int(bbox.x0), int(y_start), int(bbox.x1), int(y_end)],
                            "block_idx": block_idx,
                        }
                    )

        return lines_data

    def _sort_lines_with_layoutreader(
        self, lines_data: list[dict[str, Any]], page_width: int, page_height: int
    ) -> list[int]:
        """Sort lines using LayoutReader model.

        Returns:
            List of line indices in reading order
        """
        import torch

        x_scale = float(LAYOUTREADER_SCALE) / page_width
        y_scale = float(LAYOUTREADER_SCALE) / page_height

        boxes = []
        for line in lines_data:
            x0, y0, x1, y1 = line["bbox"]

            x0 = max(0, min(x0, page_width))
            x1 = max(0, min(x1, page_width))
            y0 = max(0, min(y0, page_height))
            y1 = max(0, min(y1, page_height))

            x0_scaled = max(0, min(round(x0 * x_scale), LAYOUTREADER_SCALE))
            y0_scaled = max(0, min(round(y0 * y_scale), LAYOUTREADER_SCALE))
            x1_scaled = max(0, min(round(x1 * x_scale), LAYOUTREADER_SCALE))
            y1_scaled = max(0, min(round(y1 * y_scale), LAYOUTREADER_SCALE))

            boxes.append([x0_scaled, y0_scaled, x1_scaled, y1_scaled])

        model = self.model_manager.get_model("layoutreader")

        from mineru.model.reading_order.layout_reader import boxes2inputs, parse_logits, prepare_inputs

        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, model)

        with torch.no_grad():
            logits = model(**inputs).logits.cpu().squeeze(0)

        return parse_logits(logits, len(boxes))

    def _assign_block_ordering(
        self,
        blocks: list[Block],
        lines_data: list[dict[str, Any]],
        sorted_line_indices: list[int],
    ) -> list[Block]:
        """Assign block-level ordering based on line-level ordering.

        Each block may have multiple lines. We use the median line index
        as the block's ordering.
        """
        block_to_line_positions: dict[int, list[int]] = {}

        for line_position, line_idx in enumerate(sorted_line_indices):
            line = lines_data[line_idx]
            block_idx = line["block_idx"]

            if block_idx not in block_to_line_positions:
                block_to_line_positions[block_idx] = []

            block_to_line_positions[block_idx].append(line_position)

        block_orders = []
        for block_idx, block in enumerate(blocks):
            if block_idx in block_to_line_positions:
                line_positions = block_to_line_positions[block_idx]
                median_position = statistics.median(line_positions)
            else:
                median_position = float("inf")

            block_orders.append((median_position, block_idx, block))

        block_orders.sort(key=lambda x: x[0])

        sorted_blocks = []
        for rank, (_, _, block) in enumerate(block_orders):
            block.order = rank
            sorted_blocks.append(block)

        return sorted_blocks

    def _fallback_sort(self, blocks: list[Block]) -> list[Block]:
        """Fallback to simple geometric sorting."""
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

        for rank, block in enumerate(sorted_blocks):
            block.order = rank

        return sorted_blocks
