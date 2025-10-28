#!/usr/bin/env python3
"""Quick startup profiling script.

Measures time for each initialization stage to identify bottlenecks.

Usage:
    python scripts/profile_startup_quick.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def measure_stage(name: str):
    """Context manager to measure stage execution time."""

    class Timer:
        def __init__(self, stage_name: str):
            self.stage_name = stage_name
            self.start = 0.0

        def __enter__(self):
            print(f"\n⏱️  {self.stage_name}...", end="", flush=True)
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f" {elapsed:.3f}s")

    return Timer(name)


def main():
    """Profile startup time."""
    print("=" * 70)
    print("STARTUP TIME PROFILER")
    print("=" * 70)

    total_start = time.perf_counter()

    # Stage 1: Import Pipeline
    with measure_stage("Import Pipeline"):
        from pipeline import Pipeline

    # Stage 2: Import other dependencies
    with measure_stage("Import numpy"):
        import numpy as np

    # Stage 3: Create Pipeline (detector + recognizer initialization)
    with measure_stage("Create Pipeline (doclayout-yolo + gemini-2.5-flash)"):
        pipeline = Pipeline(
            detector="doclayout-yolo",
            recognizer="gemini-2.5-flash",
            use_cache=False,
        )

    # Stage 4: Load sample image (if exists)
    test_image_path = Path("tests/fixtures/sample.png")
    if test_image_path.exists():
        with measure_stage("Load test image"):
            from PIL import Image

            image = Image.open(test_image_path)
            image_array = np.array(image)

        # Stage 5: First detection (cold start)
        with measure_stage("First detection (cold start)"):
            blocks = pipeline.detector.detect(image_array)
            print(f"    → Detected {len(blocks)} blocks")
    else:
        print(f"\n⚠️  Test image not found at {test_image_path}")
        print("   Skipping detection test")

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print(f"TOTAL STARTUP TIME: {total_elapsed:.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
