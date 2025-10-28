"""Adaptive batch size calibration for optimal GPU utilization.

This module automatically finds the optimal batch size for a given model
and GPU configuration through binary search and memory profiling.

Key Features:
- Auto-calibration: Measures GPU memory and finds max batch size
- Caching: Stores results per (GPU model, model name, input shape)
- OOM handling: Graceful fallback when memory exceeded
- Cross-session persistence: Cache survives across runs

Usage:
    >>> from pipeline.optimization import calibrate_batch_size
    >>>
    >>> # Auto-calibrate (with caching)
    >>> batch_size = calibrate_batch_size(
    ...     model=detector,
    ...     input_shape=(1920, 1080, 3),
    ...     target_memory_fraction=0.85
    ... )
    >>>
    >>> # Use the calibrated batch size
    >>> for batch in batched(images, batch_size):
    ...     results = detector.detect_batch(batch)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class BatchSizeCalibrator:
    """Calibrates optimal batch size for a model on current GPU.

    This class uses binary search to find the maximum batch size that fits
    in GPU memory while maintaining a safety margin.

    The calibration process:
    1. Check cache for previous results
    2. If not cached, run binary search:
       - Start with min_batch_size and max_batch_size
       - Test batch size by running inference
       - If OOM, reduce batch size
       - If success, try larger batch size
    3. Cache result for future use

    Attributes:
        cache_dir: Directory for storing calibration cache
        target_memory_fraction: Target GPU memory usage (0.0-1.0)
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        use_cache: Whether to use cached results
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        target_memory_fraction: float = 0.85,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        use_cache: bool = True,
    ):
        """Initialize calibrator.

        Args:
            cache_dir: Cache directory (default: ~/.cache/vlm-ocr-pipeline)
            target_memory_fraction: Target GPU memory usage (0.0-1.0)
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            use_cache: Whether to use cached results
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "vlm-ocr-pipeline"

        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "batch_size_cache.json"
        self.target_memory_fraction = target_memory_fraction
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.use_cache = use_cache

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(
        self,
        inference_fn: Callable[[int], Any],
        model_name: str,
        input_shape: tuple[int, ...],
        gpu_model: str | None = None,
    ) -> int:
        """Calibrate optimal batch size for given model and input shape.

        Args:
            inference_fn: Function that runs inference with batch size
                         Must accept batch_size as first argument
                         Should raise RuntimeError/CUDA OOM on memory error
            model_name: Name of model being calibrated
            input_shape: Shape of single input (H, W, C) or (C, H, W)
            gpu_model: GPU model name (auto-detected if None)

        Returns:
            Optimal batch size that fits in GPU memory

        Examples:
            >>> def run_inference(batch_size):
            ...     batch = torch.randn(batch_size, 3, 1920, 1080).cuda()
            ...     return model(batch)
            >>>
            >>> calibrator = BatchSizeCalibrator()
            >>> optimal = calibrator.calibrate(
            ...     inference_fn=run_inference,
            ...     model_name="doclayout-yolo",
            ...     input_shape=(1920, 1080, 3)
            ... )
        """
        # Generate cache key
        if gpu_model is None:
            gpu_model = self._detect_gpu_model()

        cache_key = self._generate_cache_key(model_name, input_shape, gpu_model)

        # Check cache
        if self.use_cache:
            cached_batch_size = self._load_from_cache(cache_key)
            if cached_batch_size is not None:
                logger.info(
                    "Using cached batch size: %d (model=%s, gpu=%s)",
                    cached_batch_size,
                    model_name,
                    gpu_model,
                )
                return cached_batch_size

        # Run calibration
        logger.info(
            "Calibrating optimal batch size for %s on %s (input_shape=%s)",
            model_name,
            gpu_model,
            input_shape,
        )

        optimal_batch_size = self._binary_search(inference_fn)

        logger.info(
            "Calibration complete: optimal_batch_size=%d (%.0f%% GPU memory)",
            optimal_batch_size,
            self.target_memory_fraction * 100,
        )

        # Save to cache
        if self.use_cache:
            self._save_to_cache(cache_key, optimal_batch_size, model_name, gpu_model)

        return optimal_batch_size

    def _binary_search(self, inference_fn: Callable[[int], Any]) -> int:
        """Binary search for maximum batch size that fits in memory.

        Args:
            inference_fn: Function to test with batch size

        Returns:
            Maximum batch size that doesn't OOM
        """
        left = self.min_batch_size
        right = self.max_batch_size
        best_batch_size = left

        while left <= right:
            mid = (left + right) // 2

            logger.debug("Testing batch_size=%d", mid)

            # Test if this batch size fits
            if self._test_batch_size(inference_fn, mid):
                # Success - try larger
                best_batch_size = mid
                left = mid + 1
                logger.debug("batch_size=%d succeeded, trying larger", mid)
            else:
                # OOM - try smaller
                right = mid - 1
                logger.debug("batch_size=%d failed (OOM), trying smaller", mid)

        return best_batch_size

    def _test_batch_size(self, inference_fn: Callable[[int], Any], batch_size: int) -> bool:
        """Test if batch size fits in GPU memory.

        Args:
            inference_fn: Function to run inference
            batch_size: Batch size to test

        Returns:
            True if inference succeeds, False if OOM
        """
        try:
            # Import torch here to avoid import at module level
            import torch

            # Clear GPU cache before testing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Run inference
            _ = inference_fn(batch_size)

            # Clear cache after test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        except (RuntimeError, Exception) as e:
            # Check if it's OOM error
            error_msg = str(e).lower()
            is_oom = any(
                keyword in error_msg
                for keyword in ["out of memory", "oom", "cuda", "memory"]
            )

            if is_oom:
                # Expected OOM - batch size too large
                logger.debug("OOM at batch_size=%d: %s", batch_size, str(e)[:100])

                # Clear GPU cache after OOM
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                except Exception:
                    pass

                return False
            else:
                # Unexpected error - re-raise
                logger.error("Unexpected error during calibration: %s", e)
                raise

    def _detect_gpu_model(self) -> str:
        """Detect current GPU model name.

        Returns:
            GPU model name or 'CPU' if no GPU
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return "CPU"

            # Get first GPU name
            return torch.cuda.get_device_name(0)

        except Exception:
            return "Unknown"

    def _generate_cache_key(
        self,
        model_name: str,
        input_shape: tuple[int, ...],
        gpu_model: str,
    ) -> str:
        """Generate cache key for model + input + GPU combination.

        Args:
            model_name: Model name
            input_shape: Input shape
            gpu_model: GPU model name

        Returns:
            Cache key hash
        """
        # Create deterministic key
        key_str = f"{model_name}_{input_shape}_{gpu_model}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> int | None:
        """Load cached batch size.

        Args:
            cache_key: Cache key

        Returns:
            Cached batch size or None if not found
        """
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file) as f:
                cache_data = json.load(f)

            if cache_key in cache_data:
                entry = cache_data[cache_key]
                return entry["batch_size"]

        except Exception as e:
            logger.warning("Failed to load cache: %s", e)

        return None

    def _save_to_cache(
        self,
        cache_key: str,
        batch_size: int,
        model_name: str,
        gpu_model: str,
    ) -> None:
        """Save batch size to cache.

        Args:
            cache_key: Cache key
            batch_size: Batch size to save
            model_name: Model name (for metadata)
            gpu_model: GPU model (for metadata)
        """
        try:
            # Load existing cache
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    cache_data = json.load(f)
            else:
                cache_data = {}

            # Add/update entry
            cache_data[cache_key] = {
                "batch_size": batch_size,
                "model_name": model_name,
                "gpu_model": gpu_model,
                "target_memory_fraction": self.target_memory_fraction,
            }

            # Save cache
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug("Saved batch size to cache: %s", self.cache_file)

        except Exception as e:
            logger.warning("Failed to save cache: %s", e)


def calibrate_batch_size(
    inference_fn: Callable[[int], Any],
    model_name: str,
    input_shape: tuple[int, ...],
    target_memory_fraction: float = 0.85,
    min_batch_size: int = 1,
    max_batch_size: int = 128,
    use_cache: bool = True,
) -> int:
    """Convenience function to calibrate batch size.

    This is a simplified interface to BatchSizeCalibrator for one-off calibration.

    Args:
        inference_fn: Function that runs inference with batch size
        model_name: Name of model being calibrated
        input_shape: Shape of single input
        target_memory_fraction: Target GPU memory usage (0.0-1.0)
        min_batch_size: Minimum batch size to test
        max_batch_size: Maximum batch size to test
        use_cache: Whether to use cached results

    Returns:
        Optimal batch size

    Examples:
        >>> def run_inference(batch_size):
        ...     images = [np.random.rand(1920, 1080, 3) for _ in range(batch_size)]
        ...     return detector.detect_batch(images)
        >>>
        >>> optimal = calibrate_batch_size(
        ...     inference_fn=run_inference,
        ...     model_name="doclayout-yolo",
        ...     input_shape=(1920, 1080, 3)
        ... )
    """
    calibrator = BatchSizeCalibrator(
        target_memory_fraction=target_memory_fraction,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        use_cache=use_cache,
    )

    return calibrator.calibrate(
        inference_fn=inference_fn,
        model_name=model_name,
        input_shape=input_shape,
    )


def get_optimal_batch_size(
    model_name: str,
    input_shape: tuple[int, ...],
    gpu_model: str | None = None,
) -> int | None:
    """Get cached optimal batch size without running calibration.

    This function only reads from cache and never triggers calibration.
    Useful for checking if calibration has been done before.

    Args:
        model_name: Name of model
        input_shape: Shape of single input
        gpu_model: GPU model name (auto-detected if None)

    Returns:
        Cached batch size or None if not calibrated yet

    Examples:
        >>> batch_size = get_optimal_batch_size(
        ...     model_name="doclayout-yolo",
        ...     input_shape=(1920, 1080, 3)
        ... )
        >>> if batch_size is None:
        ...     print("Not calibrated yet - run calibration first")
    """
    calibrator = BatchSizeCalibrator(use_cache=True)

    if gpu_model is None:
        gpu_model = calibrator._detect_gpu_model()

    cache_key = calibrator._generate_cache_key(model_name, input_shape, gpu_model)
    return calibrator._load_from_cache(cache_key)
