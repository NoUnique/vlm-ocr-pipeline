"""Optimization utilities for VLM OCR Pipeline.

This module provides performance optimization tools including:
- Adaptive batch size calibration
- GPU memory profiling
- Auto-tuning for different hardware configurations
"""

from __future__ import annotations

from pipeline.optimization.batch_size import (
    BatchSizeCalibrator,
    calibrate_batch_size,
    get_optimal_batch_size,
)

__all__ = [
    "BatchSizeCalibrator",
    "calibrate_batch_size",
    "get_optimal_batch_size",
]
