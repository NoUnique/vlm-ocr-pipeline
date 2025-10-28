"""Data types for staged batch processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from pipeline.types import Block


@dataclass
class PageInfo:
    """Information about a single page in batch processing.

    Tracks page state through all 5 stages:
    1. Conversion: PDF → image
    2. Detection: image → blocks
    3. Ordering: blocks → sorted_blocks
    4. Recognition: sorted_blocks → blocks with text
    5. Output: generate Page object

    Attributes:
        pdf_path: Path to source PDF file
        page_num: Page number (1-indexed)
        total_pages: Total pages in this PDF
        image: Page image (populated in Stage 1)
        blocks: Detected blocks (populated in Stage 2)
        sorted_blocks: Ordered blocks (populated in Stage 3)
        recognized_blocks: Blocks with text (populated in Stage 4)
        auxiliary_info: Auxiliary information (text spans, etc.)
        status: Processing status ("pending", "in_progress", "completed", "failed")
        error: Error message if status == "failed"
    """

    pdf_path: Path
    page_num: int
    total_pages: int

    # Stage 1: Conversion
    image: np.ndarray | None = None

    # Stage 2: Detection
    blocks: list[Block] | None = None

    # Stage 3: Ordering
    sorted_blocks: list[Block] | None = None

    # Stage 4: Recognition
    recognized_blocks: list[Block] | None = None

    # Auxiliary data
    auxiliary_info: dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    error: str | None = None

    @property
    def pdf_stem(self) -> str:
        """PDF filename without extension."""
        return self.pdf_path.stem

    @property
    def page_id(self) -> str:
        """Unique page identifier: <pdf_stem>_page_<num>."""
        return f"{self.pdf_stem}_page_{self.page_num}"

    def mark_completed(self) -> None:
        """Mark page as successfully completed."""
        self.status = "completed"
        self.error = None

    def mark_failed(self, error: str) -> None:
        """Mark page as failed with error message."""
        self.status = "failed"
        self.error = error


@dataclass
class BatchProgress:
    """Progress tracking for batch processing.

    Attributes:
        total_pages: Total number of pages to process
        completed_pages: Number of completed pages
        failed_pages: Number of failed pages
        current_stage: Current processing stage (1-5)
        stage_name: Name of current stage
    """

    total_pages: int
    completed_pages: int = 0
    failed_pages: int = 0
    current_stage: int = 0  # 0 = not started, 1-5 = stages
    stage_name: str = "Not Started"

    @property
    def progress_pct(self) -> float:
        """Progress percentage (0-100)."""
        if self.total_pages == 0:
            return 0.0
        return (self.completed_pages / self.total_pages) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all pages are processed."""
        return (self.completed_pages + self.failed_pages) == self.total_pages

    def update(self, stage: int, stage_name: str, completed: int, failed: int) -> None:
        """Update progress.

        Args:
            stage: Current stage number (1-5)
            stage_name: Name of current stage
            completed: Number of completed pages
            failed: Number of failed pages
        """
        self.current_stage = stage
        self.stage_name = stage_name
        self.completed_pages = completed
        self.failed_pages = failed
