#!/usr/bin/env python3
"""Benchmark script comparing sync vs async performance.

Usage:
    python scripts/benchmark_async.py --input document.pdf --max-pages 3 --backend gemini
    python scripts/benchmark_async.py --input document.pdf --max-pages 5 --backend openai --max-concurrent 10
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

from pipeline import Pipeline
from pipeline.conversion.input import load_image, render_pdf_page
from pipeline.recognition import TextRecognizer
from pipeline.types import Block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AsyncBenchmark:
    """Benchmark comparing sync vs async recognition performance."""

    def __init__(self):
        """Initialize benchmark."""
        self.results = {
            "sync": {},
            "async": {},
            "comparison": {},
            "metadata": {},
        }

    def run_sync_benchmark(
        self,
        input_path: str,
        detector: str,
        backend: str,
        model: str,
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """Run sync benchmark.

        Args:
            input_path: Path to input PDF/image
            detector: Detector name
            backend: Backend name
            model: Model name
            max_pages: Maximum pages to process

        Returns:
            Sync benchmark results
        """
        logger.info("\n" + "=" * 70)
        logger.info("SYNC BENCHMARK")
        logger.info("=" * 70)

        # Initialize pipeline with sync recognizer
        start = time.perf_counter()
        pipeline = Pipeline(
            detector=detector,
            backend=backend,
            model=model,
            use_cache=False,  # Disable cache for fair comparison
        )
        init_time = time.perf_counter() - start

        # Process document
        start = time.perf_counter()
        input_file = Path(input_path)

        if input_file.suffix.lower() == ".pdf":
            document = pipeline.process_pdf(input_file, max_pages=max_pages)
            pages = document.pages
            total_blocks = sum(len(page.blocks) for page in pages)
        else:
            result_dict = pipeline.process_single_image(input_file)
            pages = 1
            total_blocks = len(result_dict.get("blocks", []))

        process_time = time.perf_counter() - start

        results = {
            "init_time": init_time,
            "process_time": process_time,
            "total_time": init_time + process_time,
            "pages": len(pages) if input_file.suffix.lower() == ".pdf" else 1,
            "total_blocks": total_blocks,
            "avg_time_per_block": process_time / total_blocks if total_blocks > 0 else 0,
        }

        logger.info("Initialization: %.3fs", init_time)
        logger.info("Processing: %.3fs", process_time)
        logger.info("Total: %.3fs", results["total_time"])
        logger.info("Blocks: %d", total_blocks)
        logger.info("Avg per block: %.3fs", results["avg_time_per_block"])

        return results

    async def run_async_benchmark(
        self,
        input_path: str,
        detector: str,
        backend: str,
        model: str,
        max_pages: int | None = None,
        max_concurrent: int = 5,
    ) -> dict[str, Any]:
        """Run async benchmark.

        Args:
            input_path: Path to input PDF/image
            detector: Detector name
            backend: Backend name
            model: Model name
            max_pages: Maximum pages to process
            max_concurrent: Max concurrent API calls

        Returns:
            Async benchmark results
        """
        logger.info("\n" + "=" * 70)
        logger.info("ASYNC BENCHMARK (max_concurrent=%d)", max_concurrent)
        logger.info("=" * 70)

        # Initialize pipeline components
        start = time.perf_counter()

        # Create detector
        from pipeline.layout.detection import create_detector

        detector_obj = create_detector(detector)

        # Create async recognizer
        recognizer = TextRecognizer(
            backend=backend,
            model=model,
            use_cache=False,
            use_async=True,  # Enable async
        )

        init_time = time.perf_counter() - start

        # Process document
        start = time.perf_counter()
        input_file = Path(input_path)

        total_blocks = 0
        pages_processed = 0

        if input_file.suffix.lower() == ".pdf":
            # Process multiple pages
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

        process_time = time.perf_counter() - start

        results = {
            "init_time": init_time,
            "process_time": process_time,
            "total_time": init_time + process_time,
            "pages": pages_processed,
            "total_blocks": total_blocks,
            "avg_time_per_block": process_time / total_blocks if total_blocks > 0 else 0,
            "max_concurrent": max_concurrent,
        }

        logger.info("Initialization: %.3fs", init_time)
        logger.info("Processing: %.3fs", process_time)
        logger.info("Total: %.3fs", results["total_time"])
        logger.info("Blocks: %d", total_blocks)
        logger.info("Avg per block: %.3fs", results["avg_time_per_block"])

        return results

    def compare_results(self) -> dict[str, Any]:
        """Compare sync and async results.

        Returns:
            Comparison results
        """
        sync = self.results["sync"]
        async_res = self.results["async"]

        if not sync or not async_res:
            return {}

        speedup = sync["process_time"] / async_res["process_time"] if async_res["process_time"] > 0 else 0
        improvement_pct = (1 - async_res["process_time"] / sync["process_time"]) * 100 if sync["process_time"] > 0 else 0

        comparison = {
            "speedup": speedup,
            "improvement_percent": improvement_pct,
            "time_saved": sync["process_time"] - async_res["process_time"],
            "sync_time_per_block": sync["avg_time_per_block"],
            "async_time_per_block": async_res["avg_time_per_block"],
            "blocks_processed": sync["total_blocks"],
        }

        return comparison

    def print_comparison(self):
        """Print formatted comparison report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)

        sync = self.results["sync"]
        async_res = self.results["async"]
        comp = self.results["comparison"]

        if not comp:
            print("No comparison data available")
            return

        print("\nProcessing Time:")
        print(f"  Sync:              {sync['process_time']:>8.3f}s")
        print(f"  Async:             {async_res['process_time']:>8.3f}s")
        print(f"  {'─' * 40}")
        print(f"  Time Saved:        {comp['time_saved']:>8.3f}s")
        print(f"  Speedup:           {comp['speedup']:>8.2f}x")
        print(f"  Improvement:       {comp['improvement_percent']:>8.1f}%")

        print("\nPer-Block Performance:")
        print(f"  Sync:              {comp['sync_time_per_block']:>8.3f}s/block")
        print(f"  Async:             {comp['async_time_per_block']:>8.3f}s/block")

        print("\nConfiguration:")
        print(f"  Blocks processed:  {comp['blocks_processed']:>8}")
        print(f"  Max concurrent:    {async_res['max_concurrent']:>8}")

        print("=" * 70)

    def save_results(self, output_path: str):
        """Save results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", output_file)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark sync vs async performance")
    parser.add_argument("--input", required=True, help="Input PDF or image file")
    parser.add_argument("--detector", default="doclayout-yolo", help="Detector name")
    parser.add_argument("--backend", default="gemini", help="Recognition backend")
    parser.add_argument("--model", help="Model name (optional, uses backend default)")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API calls for async")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Set default model based on backend
    if not args.model:
        args.model = "gemini-2.5-flash" if args.backend == "gemini" else "gpt-4o"

    try:
        benchmark = AsyncBenchmark()

        # Store metadata
        benchmark.results["metadata"] = {
            "input": args.input,
            "detector": args.detector,
            "backend": args.backend,
            "model": args.model,
            "max_pages": args.max_pages,
            "max_concurrent": args.max_concurrent,
        }

        # Run sync benchmark
        logger.info("\n🔄 Running SYNC benchmark...")
        benchmark.results["sync"] = benchmark.run_sync_benchmark(
            input_path=args.input,
            detector=args.detector,
            backend=args.backend,
            model=args.model,
            max_pages=args.max_pages,
        )

        # Run async benchmark
        logger.info("\n🚀 Running ASYNC benchmark...")
        benchmark.results["async"] = await benchmark.run_async_benchmark(
            input_path=args.input,
            detector=args.detector,
            backend=args.backend,
            model=args.model,
            max_pages=args.max_pages,
            max_concurrent=args.max_concurrent,
        )

        # Compare results
        benchmark.results["comparison"] = benchmark.compare_results()
        benchmark.print_comparison()

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
