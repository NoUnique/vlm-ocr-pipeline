#!/usr/bin/env python3
"""Simple benchmark script for measuring pipeline performance.

Usage:
    python scripts/benchmark.py --input document.pdf --max-pages 1 --detector doclayout-yolo
    python scripts/benchmark.py --input document.pdf --max-pages 5 --backend gemini
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import Pipeline
from pipeline.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleBenchmark:
    """Simple benchmark to measure pipeline execution time."""

    def __init__(self):
        """Initialize benchmark."""
        self.results = {
            "timings": {},
            "metadata": {},
        }

    def run(
        self,
        input_path: str,
        detector: str = "doclayout-yolo",
        sorter: str | None = None,
        backend: str = "gemini",
        max_pages: int | None = None,
        use_cache: bool = False,
    ) -> dict:
        """Run benchmark.

        Args:
            input_path: Path to input PDF
            detector: Detector name
            sorter: Sorter name
            backend: Backend name
            max_pages: Maximum pages to process
            use_cache: Whether to use cache

        Returns:
            Benchmark results dict
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Store metadata
        self.results["metadata"] = {
            "input": str(input_file),
            "detector": detector,
            "sorter": sorter,
            "backend": backend,
            "max_pages": max_pages,
            "use_cache": use_cache,
        }

        logger.info("=" * 70)
        logger.info("BENCHMARK CONFIGURATION")
        logger.info("=" * 70)
        logger.info("Input: %s", input_path)
        logger.info("Detector: %s", detector)
        logger.info("Sorter: %s", sorter or "auto")
        logger.info("Backend: %s", backend)
        logger.info("Max Pages: %s", max_pages or "all")
        logger.info("Cache: %s", use_cache)
        logger.info("=" * 70)

        # Initialize pipeline
        logger.info("\n[1/3] Initializing pipeline...")
        init_start = time.perf_counter()
        config = PipelineConfig(
            detector=detector,
            sorter=sorter,
            recognizer_backend=backend,
            use_cache=use_cache,
        )
        pipeline = Pipeline(config=config)
        init_time = time.perf_counter() - init_start
        self.results["timings"]["initialization"] = init_time
        logger.info("✓ Pipeline initialized in %.3fs", init_time)

        # Process document
        logger.info("\n[2/3] Processing document...")
        process_start = time.perf_counter()

        if input_file.suffix.lower() == ".pdf":
            document = pipeline.process_pdf(input_file, max_pages=max_pages)
            pages = document.pages
            num_pages = len(pages)
            # Count blocks (Page objects have blocks attribute)
            total_blocks = sum(len(page.blocks) for page in pages)
        else:
            # process_single_image returns a dict, not a Page object
            result_dict = pipeline.process_single_image(input_file)
            num_pages = 1
            # Count blocks from dict
            total_blocks = len(result_dict.get("blocks", []))

        process_time = time.perf_counter() - process_start
        self.results["timings"]["processing"] = process_time

        # Calculate statistics
        logger.info("\n[3/3] Calculating statistics...")

        avg_time_per_page = process_time / num_pages if num_pages > 0 else 0

        self.results["timings"]["total"] = init_time + process_time
        self.results["timings"]["avg_per_page"] = avg_time_per_page
        self.results["stats"] = {
            "pages_processed": num_pages,
            "total_blocks": total_blocks,
            "avg_blocks_per_page": total_blocks / num_pages if num_pages > 0 else 0,
        }

        return self.results

    def print_report(self) -> None:
        """Print formatted benchmark report."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        # Metadata
        print("\nConfiguration:")
        for key, value in self.results["metadata"].items():
            print(f"  {key}: {value}")

        # Timings
        print("\nTiming Results:")
        timings = self.results["timings"]
        print(f"  Initialization:     {timings.get('initialization', 0):>8.3f}s")
        print(f"  Processing:         {timings.get('processing', 0):>8.3f}s")
        print(f"  {'─' * 40}")
        print(f"  Total:              {timings.get('total', 0):>8.3f}s")

        # Per-page stats
        if "avg_per_page" in timings:
            print(f"\n  Avg per page:       {timings['avg_per_page']:>8.3f}s")

        # Statistics
        if "stats" in self.results:
            print("\nStatistics:")
            stats = self.results["stats"]
            print(f"  Pages processed:    {stats.get('pages_processed', 0):>8}")
            if "total_blocks" in stats:
                print(f"  Total blocks:       {stats.get('total_blocks', 0):>8}")
                print(f"  Avg blocks/page:    {stats.get('avg_blocks_per_page', 0):>8.1f}")

        print("=" * 70)

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", output_file)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark OCR pipeline performance")
    parser.add_argument("--input", required=True, help="Input PDF or image file")
    parser.add_argument("--detector", default="doclayout-yolo", help="Detector name")
    parser.add_argument("--sorter", help="Sorter name (optional)")
    parser.add_argument("--backend", default="gemini", help="Recognition backend")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument("--use-cache", action="store_true", help="Enable caching")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    try:
        benchmark = SimpleBenchmark()
        benchmark.run(
            input_path=args.input,
            detector=args.detector,
            sorter=args.sorter,
            backend=args.backend,
            max_pages=args.max_pages,
            use_cache=args.use_cache,
        )
        benchmark.print_report()

        if args.output:
            benchmark.save_results(args.output)

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
