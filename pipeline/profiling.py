"""Performance profiling utilities for the pipeline.

This module provides decorators and context managers for measuring
execution time and memory usage of pipeline stages.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic function
F = TypeVar("F", bound=Callable[..., Any])


class PerformanceMetrics:
    """Collects and stores performance metrics for pipeline stages."""

    def __init__(self):
        """Initialize metrics collector."""
        self.timings: dict[str, float] = {}
        self.call_counts: dict[str, int] = {}
        self.enabled: bool = False

    def enable(self) -> None:
        """Enable metrics collection."""
        self.enabled = True
        logger.info("Performance metrics collection enabled")

    def disable(self) -> None:
        """Disable metrics collection."""
        self.enabled = False

    def record_timing(self, stage: str, elapsed: float) -> None:
        """Record timing for a stage.

        Args:
            stage: Stage name
            elapsed: Elapsed time in seconds
        """
        if not self.enabled:
            return

        if stage in self.timings:
            # Average with previous timings
            count = self.call_counts[stage]
            self.timings[stage] = (self.timings[stage] * count + elapsed) / (count + 1)
            self.call_counts[stage] = count + 1
        else:
            self.timings[stage] = elapsed
            self.call_counts[stage] = 1

    def get_report(self) -> dict[str, Any]:
        """Get performance report.

        Returns:
            Dictionary with stage timings and call counts
        """
        total_time = sum(self.timings.values())
        return {
            "total_time": total_time,
            "stages": [
                {
                    "name": stage,
                    "avg_time": elapsed,
                    "calls": self.call_counts.get(stage, 0),
                    "total_time": elapsed * self.call_counts.get(stage, 0),
                    "percentage": (elapsed * self.call_counts.get(stage, 0) / total_time * 100)
                    if total_time > 0
                    else 0,
                }
                for stage, elapsed in sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
            ],
        }

    def print_report(self) -> None:
        """Print formatted performance report."""
        if not self.timings:
            print("No performance data collected.")
            return

        report = self.get_report()
        total_time = report["total_time"]

        print("\n" + "=" * 90)
        print("PIPELINE PERFORMANCE REPORT")
        print("=" * 90)
        print(f"\n{'Stage':<45} {'Calls':<8} {'Avg (s)':<10} {'Total (s)':<10} {'%':<8}")
        print("-" * 90)

        for stage_data in report["stages"]:
            print(
                f"{stage_data['name']:<45} "
                f"{stage_data['calls']:<8} "
                f"{stage_data['avg_time']:>8.3f}s "
                f"{stage_data['total_time']:>8.3f}s "
                f"{stage_data['percentage']:>6.1f}%"
            )

        print("-" * 90)
        print(f"{'TOTAL':<45} {'':<8} {'':<10} {total_time:>8.3f}s {'100.0%':>6}")
        print("=" * 90)

    def reset(self) -> None:
        """Reset all metrics."""
        self.timings.clear()
        self.call_counts.clear()


# Global metrics instance
_global_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    """Get global metrics instance.

    Returns:
        Global PerformanceMetrics instance
    """
    return _global_metrics


@contextmanager
def measure_time(stage_name: str, metrics: PerformanceMetrics | None = None):
    """Context manager to measure execution time of a code block.

    Args:
        stage_name: Name of the stage being measured
        metrics: Optional metrics instance (uses global if None)

    Yields:
        None

    Example:
        >>> with measure_time("pdf_rendering"):
        ...     render_pdf_page(pdf_path, page_num)
    """
    if metrics is None:
        metrics = _global_metrics

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        metrics.record_timing(stage_name, elapsed)
        if metrics.enabled:
            logger.debug("Stage '%s' took %.3fs", stage_name, elapsed)


def timed(stage_name: str | None = None) -> Callable[[F], F]:
    """Decorator to measure execution time of a function.

    Args:
        stage_name: Optional custom stage name (uses function name if None)

    Returns:
        Decorated function

    Example:
        >>> @timed("pdf_conversion")
        ... def convert_pdf(path):
        ...     # conversion logic
        ...     pass
    """

    def decorator(func: F) -> F:
        name = stage_name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with measure_time(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
