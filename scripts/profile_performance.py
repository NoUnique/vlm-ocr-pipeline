#!/usr/bin/env python3
"""Performance profiling script for CPU and execution time analysis.

This script profiles the pipeline execution using cProfile to identify
performance bottlenecks and optimization opportunities.

Usage:
    # Profile full pipeline run
    python scripts/profile_performance.py --input document.pdf --backend gemini

    # Profile with limited pages
    python scripts/profile_performance.py --input doc.pdf --max-pages 3 --backend openai

    # Generate flamegraph data
    python scripts/profile_performance.py --input doc.pdf --flamegraph

    # Compare sync vs async performance
    python scripts/profile_performance.py --input doc.pdf --compare-async
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import Pipeline

# Try to import flamegraph support (optional)
try:
    import flameprof  # type: ignore

    FLAMEPROF_AVAILABLE = True
except ImportError:
    FLAMEPROF_AVAILABLE = False


class PerformanceProfiler:
    """Performance profiler for pipeline execution."""

    def __init__(self, output_dir: str = "profiling_results"):
        """Initialize profiler.

        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {}

    def profile_pipeline(
        self,
        input_path: str,
        backend: str,
        model: str | None = None,
        detector: str = "doclayout-yolo",
        sorter: str | None = None,
        max_pages: int | None = None,
        use_async: bool = False,
    ) -> tuple[float, pstats.Stats]:
        """Profile pipeline execution.

        Args:
            input_path: Path to input document
            backend: Recognition backend
            model: Model name (optional)
            detector: Detector name
            sorter: Sorter name (optional)
            max_pages: Maximum pages to process
            use_async: Use async processing

        Returns:
            Tuple of (execution_time, profile_stats)
        """
        print(f"\n{'=' * 70}")
        print(f"Profiling: {'Async' if use_async else 'Sync'} Pipeline")
        print(f"{'=' * 70}")

        # Create profiler
        profiler = cProfile.Profile()

        # Profile pipeline execution
        start_time = time.perf_counter()
        profiler.enable()

        try:
            # Set default model if not specified
            if model is None:
                model = "gemini-2.5-flash" if backend == "gemini" else "gpt-4o"

            pipeline = Pipeline(
                backend=backend,
                model=model,
                detector=detector,
                sorter=sorter,
                use_async=use_async,
                use_cache=False,  # Disable cache for accurate profiling
            )

            # Process document based on file type
            input_file = Path(input_path)
            if input_file.suffix.lower() == ".pdf":
                result = pipeline.process_pdf(
                    pdf_path=input_file,
                    max_pages=max_pages,
                )
                # PDF returns Document
                num_pages = len(result.pages)
                num_blocks = sum(len(p.blocks) for p in result.pages)
            else:
                result = pipeline.process_image(input_file)
                # Image returns dict[str, Any]
                num_pages = 1
                num_blocks = len(result.get("regions", []))  # type: ignore

            profiler.disable()
            execution_time = time.perf_counter() - start_time

            # Collect stats
            stats = pstats.Stats(profiler)

            print(f"✅ Execution completed in {execution_time:.3f}s")
            print(f"   Pages processed: {num_pages}")
            print(f"   Total blocks: {num_blocks}")

            return execution_time, stats

        except Exception as e:
            profiler.disable()
            print(f"❌ Profiling failed: {e}")
            raise

    def analyze_stats(
        self,
        stats: pstats.Stats,
        top_n: int = 20,
        sort_by: str = "cumulative",
    ) -> dict[str, Any]:
        """Analyze profiling statistics.

        Args:
            stats: Profile statistics
            top_n: Number of top functions to show
            sort_by: Sort key (cumulative, time, calls)

        Returns:
            Analysis results
        """
        # Capture stats output
        stream = StringIO()
        ps = pstats.Stats(stats, stream=stream)  # type: ignore[arg-type]
        ps.strip_dirs()
        ps.sort_stats(sort_by)
        ps.print_stats(top_n)

        stats_output = stream.getvalue()

        # Extract key metrics (using type: ignore for internal attributes)
        total_calls = ps.total_calls  # type: ignore
        prim_calls = ps.prim_calls  # type: ignore
        total_time = ps.total_tt  # type: ignore

        # Find hotspots (functions taking >1% of total time)
        hotspots = []
        stats_dict = ps.stats  # type: ignore
        for func, (cc, nc, tt, ct, callers) in stats_dict.items():
            if ct > total_time * 0.01:  # >1% of total time
                func_name = f"{func[0]}:{func[1]}:{func[2]}"
                hotspots.append(
                    {
                        "function": func_name,
                        "calls": nc,
                        "total_time": tt,
                        "cumulative_time": ct,
                        "percent": (ct / total_time) * 100,
                    }
                )

        # Sort hotspots by cumulative time
        hotspots.sort(key=lambda x: x["cumulative_time"], reverse=True)

        return {
            "total_calls": total_calls,
            "primitive_calls": prim_calls,
            "total_time": total_time,
            "hotspots": hotspots[:top_n],
            "stats_output": stats_output,
        }

    def save_results(
        self,
        stats: pstats.Stats,
        execution_time: float,
        config: dict[str, Any],
        suffix: str = "",
    ) -> None:
        """Save profiling results to files.

        Args:
            stats: Profile statistics
            execution_time: Total execution time
            config: Configuration used
            suffix: Filename suffix
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"profile_{timestamp}{suffix}"

        # Save raw profile data
        profile_file = self.output_dir / f"{base_name}.prof"
        stats.dump_stats(str(profile_file))
        print(f"\n📊 Profile data saved: {profile_file}")

        # Save text report
        report_file = self.output_dir / f"{base_name}.txt"
        with open(report_file, "w") as f:
            ps = pstats.Stats(stats, stream=f)  # type: ignore[arg-type]
            ps.strip_dirs()
            ps.sort_stats("cumulative")
            ps.print_stats(50)
        print(f"📊 Text report saved: {report_file}")

        # Save JSON analysis
        analysis = self.analyze_stats(stats)
        analysis["execution_time"] = execution_time
        analysis["config"] = config

        json_file = self.output_dir / f"{base_name}.json"
        with open(json_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"📊 JSON analysis saved: {json_file}")

        # Print hotspots
        print(f"\n🔥 Top {len(analysis['hotspots'])} Hotspots:")
        print(f"{'Function':<60} {'Calls':<10} {'Time %':<10}")
        print("-" * 80)
        for hotspot in analysis["hotspots"]:
            func_short = hotspot["function"][-57:]  # Truncate long names
            print(f"{func_short:<60} {hotspot['calls']:<10} {hotspot['percent']:<10.2f}")

    def compare_async(
        self,
        input_path: str,
        backend: str,
        model: str | None = None,
        detector: str = "doclayout-yolo",
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """Compare sync vs async performance.

        Args:
            input_path: Path to input document
            backend: Recognition backend
            model: Model name (optional)
            detector: Detector name
            max_pages: Maximum pages to process

        Returns:
            Comparison results
        """
        print(f"\n{'=' * 70}")
        print("COMPARING SYNC vs ASYNC PERFORMANCE")
        print(f"{'=' * 70}")

        # Profile sync
        print("\n1️⃣ Profiling SYNC mode...")
        sync_time, sync_stats = self.profile_pipeline(
            input_path=input_path,
            backend=backend,
            model=model,
            detector=detector,
            max_pages=max_pages,
            use_async=False,
        )
        self.save_results(
            sync_stats,
            sync_time,
            {"mode": "sync", "backend": backend, "detector": detector},
            suffix="_sync",
        )

        # Profile async
        print("\n2️⃣ Profiling ASYNC mode...")
        async_time, async_stats = self.profile_pipeline(
            input_path=input_path,
            backend=backend,
            model=model,
            detector=detector,
            max_pages=max_pages,
            use_async=True,
        )
        self.save_results(
            async_stats,
            async_time,
            {"mode": "async", "backend": backend, "detector": detector},
            suffix="_async",
        )

        # Compare
        speedup = sync_time / async_time if async_time > 0 else 0
        improvement = ((sync_time - async_time) / sync_time * 100) if sync_time > 0 else 0

        comparison = {
            "sync_time": sync_time,
            "async_time": async_time,
            "speedup": speedup,
            "improvement_percent": improvement,
        }

        # Print comparison
        print(f"\n{'=' * 70}")
        print("📊 COMPARISON RESULTS")
        print(f"{'=' * 70}")
        print(f"Sync time:   {sync_time:>10.3f}s")
        print(f"Async time:  {async_time:>10.3f}s")
        print(f"Speedup:     {speedup:>10.2f}x")
        print(f"Improvement: {improvement:>10.1f}%")
        print(f"{'=' * 70}")

        # Save comparison
        comparison_file = self.output_dir / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n📊 Comparison saved: {comparison_file}")

        return comparison


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile pipeline performance")
    parser.add_argument("--input", required=True, help="Input PDF or image file")
    parser.add_argument("--backend", default="gemini", help="Recognition backend")
    parser.add_argument("--model", help="Model name (optional)")
    parser.add_argument("--detector", default="doclayout-yolo", help="Detector name")
    parser.add_argument("--sorter", help="Sorter name (optional)")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument(
        "--output-dir",
        default="profiling_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--compare-async",
        action="store_true",
        help="Compare sync vs async performance",
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Use async processing (ignored if --compare-async)",
    )
    parser.add_argument(
        "--flamegraph",
        action="store_true",
        help="Generate flamegraph data (requires flameprof)",
    )

    args = parser.parse_args()

    # Check flamegraph
    if args.flamegraph and not FLAMEPROF_AVAILABLE:
        print("⚠️  Warning: flameprof not installed. Install with: pip install flameprof")
        print("   Continuing without flamegraph generation...")

    try:
        profiler = PerformanceProfiler(output_dir=args.output_dir)

        if args.compare_async:
            # Compare sync vs async
            profiler.compare_async(
                input_path=args.input,
                backend=args.backend,
                model=args.model,
                detector=args.detector,
                max_pages=args.max_pages,
            )
        else:
            # Single profile run
            execution_time, stats = profiler.profile_pipeline(
                input_path=args.input,
                backend=args.backend,
                model=args.model,
                detector=args.detector,
                sorter=args.sorter,
                max_pages=args.max_pages,
                use_async=args.use_async,
            )

            # Save results
            config = {
                "mode": "async" if args.use_async else "sync",
                "backend": args.backend,
                "detector": args.detector,
                "sorter": args.sorter,
            }
            profiler.save_results(stats, execution_time, config)

        print(f"\n✅ Profiling complete. Results saved to: {profiler.output_dir}")

    except KeyboardInterrupt:
        print("\n⚠️  Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
