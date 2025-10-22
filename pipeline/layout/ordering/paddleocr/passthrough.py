"""PP-DocLayoutV2 passthrough sorter.

PP-DocLayoutV2 already provides reading order via its pointer network,
so this sorter simply preserves the order from the detector.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pipeline.types import Block, Sorter

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class PPDocLayoutV2Sorter(Sorter):
    """Passthrough sorter for PP-DocLayoutV2 detector.

    PP-DocLayoutV2 uses a lightweight pointer network (6 Transformer layers)
    to restore reading order during detection. This sorter preserves that
    ordering instead of re-sorting with a different algorithm.

    Note: This sorter is tightly coupled with the paddleocr-doclayout-v2 detector.
    It assumes blocks already have their 'order' field set by the detector.
    """

    def __init__(self) -> None:
        """Initialize PP-DocLayoutV2 passthrough sorter."""
        logger.info("PP-DocLayoutV2 passthrough sorter initialized (preserves pointer network ordering)")

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Preserve reading order from PP-DocLayoutV2 detector.

        Args:
            blocks: Detected blocks with 'order' already set by PP-DocLayoutV2
            image: Page image (unused, included for interface compatibility)
            **kwargs: Additional context (unused)

        Returns:
            Blocks in their original order (as provided by PP-DocLayoutV2)

        Note:
            If blocks don't have 'order' set (e.g., from a different detector),
            this sorter will apply a simple top-to-bottom, left-to-right fallback.
        """
        if not blocks:
            return blocks

        # Check if blocks already have order (from PP-DocLayoutV2)
        has_order = all(block.order is not None for block in blocks)

        if has_order:
            # Blocks are already ordered by PP-DocLayoutV2's pointer network
            # Just ensure they're sorted by the order field
            sorted_blocks = sorted(blocks, key=lambda b: b.order if b.order is not None else 0)
            logger.debug(
                "Preserved reading order from PP-DocLayoutV2 for %d blocks (pointer network ordering)",
                len(sorted_blocks),
            )
            return sorted_blocks

        # Fallback: If order is not set, apply simple geometric sorting
        logger.warning(
            "Blocks from non-PP-DocLayoutV2 detector detected. "
            "Applying fallback top-to-bottom, left-to-right sorting."
        )
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

        # Set order for consistency
        for rank, block in enumerate(sorted_blocks):
            block.order = rank

        return sorted_blocks
