"""Pipeline result types and utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .block import Block
    from .document import Document


@dataclass
class StageTimingInfo:
    """Timing information for a pipeline stage.

    Attributes:
        stage_name: Name of the stage
        processing_time_ms: Processing time in milliseconds
        items_processed: Number of items processed (e.g., blocks, pages)
    """

    stage_name: str
    processing_time_ms: float
    items_processed: int = 0

    @property
    def processing_time_sec(self) -> float:
        """Get processing time in seconds."""
        return self.processing_time_ms / 1000.0


@dataclass
class PipelineResult:
    """Complete pipeline processing result.

    This dataclass captures the full result of processing a document
    through the OCR pipeline, including timing and metadata.

    Attributes:
        document: Processed Document object with all pages
        stage_timings: Timing information for each stage
        total_time_ms: Total processing time in milliseconds
        success: Whether processing completed successfully
        error: Error message if processing failed

    Example:
        >>> result = pipeline.process_pdf("document.pdf")
        >>> result.success
        True
        >>> result.total_time_sec
        12.5
        >>> result.get_stage_timings()
        {'detection': 2000.0, 'ordering': 500.0, 'recognition': 8000.0, ...}
    """

    document: Document | None
    stage_timings: list[StageTimingInfo]
    total_time_ms: float
    success: bool = True
    error: str | None = None

    @property
    def total_time_sec(self) -> float:
        """Get total processing time in seconds."""
        return self.total_time_ms / 1000.0

    def get_stage_timings(self) -> dict[str, float]:
        """Get timing for each stage as a dictionary.

        Returns:
            Dictionary mapping stage names to processing times in ms
        """
        return {timing.stage_name: timing.processing_time_ms for timing in self.stage_timings}

    def get_slowest_stage(self) -> StageTimingInfo | None:
        """Get the slowest stage.

        Returns:
            StageTimingInfo for the slowest stage, or None if no stages
        """
        if not self.stage_timings:
            return None
        return max(self.stage_timings, key=lambda t: t.processing_time_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dictionary with all result information
        """
        result: dict[str, Any] = {
            "success": self.success,
            "total_time_ms": self.total_time_ms,
            "total_time_sec": self.total_time_sec,
            "stage_timings": {t.stage_name: t.processing_time_ms for t in self.stage_timings},
        }

        if self.document is not None:
            result["document"] = self.document.to_dict()

        if self.error is not None:
            result["error"] = self.error

        return result


# Type alias for renderer functions (alternative to Protocol)
# Use this when you need a simple type hint without Protocol overhead
RendererFunc = Callable[[Sequence[Block]], str]


# ==================== Utility Functions ====================


def blocks_to_olmocr_anchor_text(
    blocks: Sequence[Block],
    page_width: int,
    page_height: int,
    max_length: int = 4000,
) -> str:
    """Convert blocks to olmOCR anchor text format.

    Args:
        blocks: List of Block instances with bbox
        page_width: Page width in pixels
        page_height: Page height in pixels
        max_length: Maximum anchor text length (approximate)

    Returns:
        olmOCR anchor text string

    Example:
        >>> blocks = [
        ...     Block(type="title", bbox=BBox(100, 50, 300, 80), detection_confidence=0.9),
        ...     Block(type="figure", bbox=BBox(100, 100, 300, 250), detection_confidence=0.95),
        ... ]
        >>> anchor = blocks_to_olmocr_anchor_text(blocks, 800, 600)
        >>> print(anchor)
        Page dimensions: 800x600
        [100x50]
        [Image 100x100 to 300x250]
    """
    # Header
    lines = [f"Page dimensions: {page_width}x{page_height}"]

    # Convert each block
    for block in blocks:
        bbox = block.bbox
        text_content = (block.text or "")[:50] if block.type in ["text", "title", "plain text"] else ""
        anchor_line = bbox.to_olmocr_anchor(content_type=block.type, text_content=text_content)
        lines.append(anchor_line)

        # Check length limit
        current_length = sum(len(line) for line in lines)
        if current_length > max_length:
            break

    return "\n".join(lines)
