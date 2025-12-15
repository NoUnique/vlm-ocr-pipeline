"""Distributed computing support for VLM OCR Pipeline.

This module provides Ray-based multi-GPU parallelization for efficient
processing across multiple GPUs and machines.

Components:
- ray_pool: Actor pool management and load balancing
- ray_detector: Ray-wrapped detector actors
- ray_recognizer: Ray-wrapped recognizer actors

Usage:
    from pipeline.distributed import RayDetectorPool, RayRecognizerPool

    # Create actor pools
    detector_pool = RayDetectorPool(
        detector_name="doclayout-yolo",
        num_actors=4,  # 4 GPUs
    )

    # Process images in parallel
    results = detector_pool.detect_batch(images)
"""

from __future__ import annotations

from pipeline.distributed.ray_detector import RayDetectorPool
from pipeline.distributed.ray_recognizer import RayRecognizerPool

__all__ = [
    "RayDetectorPool",
    "RayRecognizerPool",
    "is_ray_available",
]


def is_ray_available() -> bool:
    """Check if Ray is available for distributed processing.

    Returns:
        True if Ray is installed and can be imported
    """
    try:
        import ray  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False
