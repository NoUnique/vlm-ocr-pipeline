"""Checkpoint and resume functionality for VLM OCR Pipeline.

This module provides smart resume capabilities:
- Automatic saving of intermediate results after each stage
- Progress tracking with _progress.json
- Resume from last successful stage after failure
- No re-computation of completed stages

Usage:
    >>> from pipeline.checkpoint import ProgressTracker
    >>>
    >>> tracker = ProgressTracker(output_dir=Path("results/doc1"))
    >>> tracker.start_stage("detection")
    >>> # ... run detection ...
    >>> tracker.complete_stage("detection", Path("results/doc1/stage2_detection.json"))
"""

from __future__ import annotations

from pipeline.checkpoint.progress import ProgressTracker
from pipeline.checkpoint.serializer import (
    deserialize_stage_result,
    load_checkpoint,
    save_checkpoint,
    serialize_stage_result,
)

__all__ = [
    "ProgressTracker",
    "serialize_stage_result",
    "deserialize_stage_result",
    "save_checkpoint",
    "load_checkpoint",
]
