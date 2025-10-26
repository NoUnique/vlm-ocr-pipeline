#!/usr/bin/env python3
"""Performance profiling script for the OCR pipeline.

This script profiles the pipeline execution and identifies performance bottlenecks.
It measures execution time for each stage and reports memory usage.

Usage:
    python scripts/profile_pipeline.py --input <pdf_file> [--detector <name>] [--max-pages <n>]
    python scripts/profile_pipeline.py --input document.pdf --detector doclayout-yolo --max-pages 1
"""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
import time
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for the OCR pipeline."""

    def __init__(self):
        """Initialize profiler."""
        self.timings: dict[str, float] = {}
        self.stage_start: float = 0.0

    def start_stage(self, stage_name: str) -> None:
        """Start timing a pipeline stage.

        Args:
            stage_name: Name of the stage to profile
        """
        self.stage_start = time.perf_counter()
        logger.info("Starting stage: %s", stage_name)

    def end_stage(self, stage_name: str) -> float:
        """End timing a pipeline stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Elapsed time in seconds
        """
        elapsed = time.perf_counter() - self.stage_start
        self.timings[stage_name] = elapsed
        logger.info("Completed stage: %s (%.3fs)", stage_name, elapsed)
        return elapsed

    def print_report(self) -> None:
        """Print performance report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE PROFILE REPORT")
        print("=" * 70)

        total_time = sum(self.timings.values())

        print(f"\n{'Stage':<40} {'Time (s)':<12} {'% Total':<10}")
        print("-" * 70)

        for stage, elapsed in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"{stage:<40} {elapsed:>10.3f}s {percentage:>8.1f}%")

        print("-" * 70)
        print(f"{'TOTAL':<40} {total_time:>10.3f}s {'100.0%':>8}")
        print("=" * 70)


def profile_with_cprofile(func, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    """Profile a function using cProfile.

    Args:
        func: Function to profile
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Tuple of (function result, profiling stats as string)
    """
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    # Get stats as string
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(30)  # Top 30 functions

    return result, s.getvalue()


def run_profiled_pipeline(
    input_path: str,
    detector: str = "doclayout-yolo",
    sorter: str | None = None,
    backend: str = "gemini",
    max_pages: int | None = None,
) -> None:
    """Run pipeline with profiling.

    Args:
        input_path: Path to input PDF or image
        detector: Detector name
        sorter: Sorter name (optional)
        backend: Recognition backend
        max_pages: Maximum pages to process
    """
    from pipeline import Pipeline

    profiler = PerformanceProfiler()

    # Initialize pipeline
    profiler.start_stage("1. Pipeline Initialization")
    pipeline = Pipeline(
        detector=detector,
        sorter=sorter,
        backend=backend,
        use_cache=False,  # Disable cache for fair profiling
    )
    profiler.end_stage("1. Pipeline Initialization")

    # Process document
    profiler.start_stage("2. Total Pipeline Execution")

    # Stage-by-stage profiling
    profiler.start_stage("2a. Document Conversion (PDF → Images)")
    # Note: We'll instrument the pipeline code to get finer-grained timing
    input_file = Path(input_path)
    if input_file.suffix.lower() == ".pdf":
        document = pipeline.process_pdf(input_file, max_pages=max_pages)
        pages = document.pages
    else:
        single_result = pipeline.process_single_image(input_file)
        pages = [single_result] if single_result else []

    profiler.end_stage("2. Total Pipeline Execution")

    # Print results summary
    print(f"\nProcessed {len(pages)} page(s)")

    # Print performance report
    profiler.print_report()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile OCR pipeline performance")
    parser.add_argument("--input", required=True, help="Input PDF or image file")
    parser.add_argument("--detector", default="doclayout-yolo", help="Detector name")
    parser.add_argument("--sorter", help="Sorter name (optional)")
    parser.add_argument("--backend", default="gemini", help="Recognition backend")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed cProfile report",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return

    if args.detailed:
        # Run with detailed cProfile
        logger.info("Running with detailed cProfile...")
        _, stats = profile_with_cprofile(
            run_profiled_pipeline,
            str(input_path),
            args.detector,
            args.sorter,
            args.backend,
            args.max_pages,
        )
        print("\n" + "=" * 70)
        print("DETAILED cProfile REPORT (Top 30 functions)")
        print("=" * 70)
        print(stats)
    else:
        # Run with stage-level profiling
        run_profiled_pipeline(
            str(input_path),
            args.detector,
            args.sorter,
            args.backend,
            args.max_pages,
        )


if __name__ == "__main__":
    main()
