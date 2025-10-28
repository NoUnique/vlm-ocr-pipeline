"""Staged batch processing for efficient multi-file processing.

This module implements stage-wise batch processing to maximize GPU utilization
by processing all files through each stage before moving to the next stage.

Architecture:
    Sequential per-file (OLD):
        PDF1 → [convert→detect→sort→recognize] → Done
        PDF2 → [convert→detect→sort→recognize] → Done

    Staged batch (NEW):
        All PDFs → [convert] → All pages
        All pages → [detect] → All results (detector loaded once)
        All pages → [sort] → All results
        All pages → [recognize] → All results (recognizer loaded once)

Benefits:
    - Maximum GPU utilization (models loaded once per stage)
    - Memory efficient (intermediate results stored on disk)
    - Easy to resume (restart from any stage)
    - Better for Ray multi-GPU (distribute pages across GPUs)

Usage:
    from pipeline.batch import StagedBatchProcessor

    processor = StagedBatchProcessor(pipeline)
    results = processor.process_directory(input_dir, output_dir)
"""

from __future__ import annotations

from pipeline.batch.processor import StagedBatchProcessor
from pipeline.batch.types import PageInfo

__all__ = [
    "StagedBatchProcessor",
    "PageInfo",
]
