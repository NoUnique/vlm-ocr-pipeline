#!/usr/bin/env python3
"""Benchmark script for testing different concurrency levels.

This script tests various max_concurrent values to find the optimal
concurrency level for async API processing.

Usage:
    python scripts/benchmark_concurrency.py --input document.pdf --backend gemini --max-pages 3
    python scripts/benchmark_concurrency.py --input document.pdf --backend openai --concurrency-levels 1,5,10,20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.conversion.input.image import load_image
from pipeline.conversion.input.pdf import render_pdf_page
from pipeline.layout.detection import create_detector
from pipeline.recognition import TextRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ConcurrencyBenchmark:
    """Benchmark for testing different concurrency levels."""

    def __init__(self):
        """Initialize benchmark."""
        self.results: list[dict[str, Any]] = []

    async def run_with_concurrency(
        self,
        input_path: str,
        detector: str,
        backend: str,
        model: str,
        max_pages: int | None,
        max_concurrent: int,
    ) -> dict[str, Any]:
        """Run benchmark with specific concurrency level.

        Args:
            input_path: Path to input PDF/image
            detector: Detector name
            backend: Backend name
            model: Model name
            max_pages: Maximum pages to process
            max_concurrent: Max concurrent API calls

        Returns:
            Benchmark results for this concurrency level
        """
        logger.info("\n" + "=" * 70)
        logger.info("TESTING CONCURRENCY LEVEL: %d", max_concurrent)
        logger.info("=" * 70)

        # Initialize components
        start_init = time.perf_counter()
        detector_obj = create_detector(detector)
        recognizer = TextRecognizer(
            backend=backend,
            model=model,
            use_cache=False,
            use_async=True,
        )
        init_time = time.perf_counter() - start_init

        # Process document
        start_process = time.perf_counter()
        input_file = Path(input_path)

        total_blocks = 0
        pages_processed = 0

        if input_file.suffix.lower() == ".pdf":
            # Process PDF pages
            import fitz

            doc = fitz.open(str(input_file))
            max_pages_to_process = min(max_pages, doc.page_count) if max_pages else doc.page_count

            for page_num in range(1, max_pages_to_process + 1):
                # Render page
                image, _ = render_pdf_page(input_file, page_num, Path(".cache"))

                # Detect blocks
                blocks = detector_obj.detect(image)

                # Extract text async
                blocks = await recognizer.process_blocks_async(image, blocks, max_concurrent=max_concurrent)

                total_blocks += len(blocks)
                pages_processed += 1

            doc.close()
        else:
            # Single image
            image = load_image(input_file)
            blocks = detector_obj.detect(image)
            blocks = await recognizer.process_blocks_async(image, blocks, max_concurrent=max_concurrent)
            total_blocks = len(blocks)
            pages_processed = 1

        process_time = time.perf_counter() - start_process

        # Calculate metrics
        throughput = total_blocks / process_time if process_time > 0 else 0
        avg_time_per_block = process_time / total_blocks if total_blocks > 0 else 0

        results = {
            "max_concurrent": max_concurrent,
            "init_time": init_time,
            "process_time": process_time,
            "total_time": init_time + process_time,
            "pages": pages_processed,
            "total_blocks": total_blocks,
            "throughput": throughput,  # blocks per second
            "avg_time_per_block": avg_time_per_block,
        }

        logger.info("Initialization: %.3fs", init_time)
        logger.info("Processing: %.3fs", process_time)
        logger.info("Total: %.3fs", results["total_time"])
        logger.info("Blocks: %d", total_blocks)
        logger.info("Throughput: %.2f blocks/sec", throughput)
        logger.info("Avg per block: %.3fs", avg_time_per_block)

        return results

    def compare_results(self) -> dict[str, Any]:
        """Compare results across different concurrency levels.

        Returns:
            Comparison results
        """
        if not self.results:
            return {}

        # Find best configuration
        best_throughput = max(self.results, key=lambda x: x["throughput"])
        best_time = min(self.results, key=lambda x: x["process_time"])
        baseline = self.results[0]  # Assuming first is baseline (lowest concurrency)

        comparison = {
            "baseline": {
                "max_concurrent": baseline["max_concurrent"],
                "process_time": baseline["process_time"],
                "throughput": baseline["throughput"],
            },
            "best_throughput": {
                "max_concurrent": best_throughput["max_concurrent"],
                "throughput": best_throughput["throughput"],
                "speedup": best_throughput["throughput"] / baseline["throughput"] if baseline["throughput"] > 0 else 0,
            },
            "best_time": {
                "max_concurrent": best_time["max_concurrent"],
                "process_time": best_time["process_time"],
                "speedup": baseline["process_time"] / best_time["process_time"] if best_time["process_time"] > 0 else 0,
            },
            "all_results": self.results,
        }

        return comparison

    def print_comparison(self):
        """Print formatted comparison report."""
        comparison = self.compare_results()

        if not comparison:
            print("No comparison data available")
            return

        print("\n" + "=" * 70)
        print("CONCURRENCY LEVEL COMPARISON")
        print("=" * 70)

        print("\n📊 All Results:")
        print(f"{'Concurrency':<15} {'Process Time':<15} {'Throughput':<20} {'Avg/Block':<15}")
        print("-" * 70)
        for result in self.results:
            print(
                f"{result['max_concurrent']:<15} "
                f"{result['process_time']:>8.3f}s      "
                f"{result['throughput']:>8.2f} blocks/s    "
                f"{result['avg_time_per_block']:>8.3f}s"
            )

        print("\n🏆 Best Configurations:")
        baseline = comparison["baseline"]
        best_throughput = comparison["best_throughput"]
        best_time = comparison["best_time"]

        print(f"\nBaseline (concurrency={baseline['max_concurrent']}):")
        print(f"  Process Time:  {baseline['process_time']:>8.3f}s")
        print(f"  Throughput:    {baseline['throughput']:>8.2f} blocks/s")

        print(f"\nBest Throughput (concurrency={best_throughput['max_concurrent']}):")
        print(f"  Throughput:    {best_throughput['throughput']:>8.2f} blocks/s")
        print(f"  Speedup:       {best_throughput['speedup']:>8.2f}x")

        print(f"\nBest Time (concurrency={best_time['max_concurrent']}):")
        print(f"  Process Time:  {best_time['process_time']:>8.3f}s")
        print(f"  Speedup:       {best_time['speedup']:>8.2f}x")

        # Recommendation
        print("\n💡 Recommendation:")
        if best_throughput["max_concurrent"] == best_time["max_concurrent"]:
            print(f"  Optimal concurrency level: {best_throughput['max_concurrent']}")
        else:
            print(f"  For best throughput: {best_throughput['max_concurrent']}")
            print(f"  For best time: {best_time['max_concurrent']}")

        print("=" * 70)

    def save_results(self, output_path: str):
        """Save results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        comparison = self.compare_results()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", output_file)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark different concurrency levels")
    parser.add_argument("--input", required=True, help="Input PDF or image file")
    parser.add_argument("--detector", default="doclayout-yolo", help="Detector name")
    parser.add_argument("--backend", default="gemini", help="Recognition backend")
    parser.add_argument("--model", help="Model name (optional, uses backend default)")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,3,5,10,15,20",
        help="Comma-separated list of concurrency levels to test (default: 1,3,5,10,15,20)",
    )
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Set default model based on backend
    if not args.model:
        args.model = "gemini-2.5-flash" if args.backend == "gemini" else "gpt-4o"

    # Parse concurrency levels
    try:
        concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    except ValueError as e:
        logger.error("Invalid concurrency levels: %s", e)
        sys.exit(1)

    logger.info("Testing concurrency levels: %s", concurrency_levels)

    try:
        benchmark = ConcurrencyBenchmark()

        # Test each concurrency level
        for max_concurrent in concurrency_levels:
            result = await benchmark.run_with_concurrency(
                input_path=args.input,
                detector=args.detector,
                backend=args.backend,
                model=args.model,
                max_pages=args.max_pages,
                max_concurrent=max_concurrent,
            )
            benchmark.results.append(result)

            # Small delay between tests to avoid rate limiting
            await asyncio.sleep(2)

        # Print comparison
        benchmark.print_comparison()

        # Save results
        if args.output:
            benchmark.save_results(args.output)

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
