#!/usr/bin/env python3
"""Detailed startup profiling script - measures each init stage.

Measures time for each Pipeline.__init__() stage to identify bottlenecks.

Usage:
    python scripts/profile_startup_detailed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig


def measure_stage(name: str):
    """Context manager to measure stage execution time."""

    class Timer:
        def __init__(self, stage_name: str):
            self.stage_name = stage_name
            self.start = 0.0

        def __enter__(self):
            print(f"  ⏱️  {self.stage_name}...", end="", flush=True)
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f" {elapsed:.3f}s")

    return Timer(name)


def main():
    """Profile Pipeline initialization in detail."""
    print("=" * 70)
    print("DETAILED PIPELINE INITIALIZATION PROFILER")
    print("=" * 70)

    total_start = time.perf_counter()

    # Measure imports
    print("\n[1/3] Import Stage:")
    with measure_stage("Import Pipeline module"):
        from pipeline import Pipeline

    # Measure Pipeline creation with instrumentation
    print("\n[2/3] Pipeline.__init__() Stage:")
    print("  Creating Pipeline with doclayout-yolo + gemini-2.5-flash...")

    init_start = time.perf_counter()

    # Monkey-patch to measure internal stages
    original_init = Pipeline.__init__

    timings = {}

    def instrumented_init(self, *args, **kwargs):
        # Stage 1: Config loading
        stage_start = time.perf_counter()
        # Original init will load configs
        original_init(self, *args, **kwargs)
        timings["total_init"] = time.perf_counter() - stage_start

    Pipeline.__init__ = instrumented_init

    config = PipelineConfig(
        detector="doclayout-yolo",
        recognizer="gemini-2.5-flash",
        use_cache=False,
    )
    pipeline = Pipeline(config=config)

    init_elapsed = time.perf_counter() - init_start

    # Restore original
    Pipeline.__init__ = original_init

    print(f"  → Total __init__ time: {init_elapsed:.3f}s")

    # Measure first detection (if test image exists)
    print("\n[3/3] First Detection (Cold Start):")
    test_image_path = Path("tests/fixtures/sample.png")
    if test_image_path.exists():
        with measure_stage("Load test image"):
            import numpy as np
            from PIL import Image

            image = Image.open(test_image_path)
            image_array = np.array(image)

        with measure_stage("First detection call"):
            if pipeline.detection_stage:
                blocks = pipeline.detection_stage.process(image_array)
                print(f"    → Detected {len(blocks)} blocks")
            else:
                print("    → Detection stage not initialized")
    else:
        print(f"  ⚠️  Test image not found at {test_image_path}")
        print("  Skipping detection test")

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {total_elapsed:.3f}s")
    print("=" * 70)

    # Recommendations
    print("\n💡 Optimization Recommendations:")
    print("  1. Defer detector loading until first detect() call")
    print("  2. Defer recognizer loading until first recognize() call")
    print("  3. Use lazy property pattern for heavy components")


if __name__ == "__main__":
    main()
