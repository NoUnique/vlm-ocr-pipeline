#!/usr/bin/env python3
"""Memory profiling script for tracking memory usage and leaks.

This script profiles memory allocation during pipeline execution using
Python's built-in tracemalloc module to identify memory-intensive operations.

Usage:
    # Profile memory usage
    python scripts/profile_memory.py --input document.pdf --backend gemini

    # Profile with limited pages
    python scripts/profile_memory.py --input doc.pdf --max-pages 3

    # Compare sync vs async memory usage
    python scripts/profile_memory.py --input doc.pdf --compare-async

    # Take memory snapshots at intervals
    python scripts/profile_memory.py --input doc.pdf --snapshot-interval 10
"""

from __future__ import annotations

import argparse
import gc
import json
import linecache
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import Pipeline

# Try to import psutil for system memory info (optional)
try:
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import torch for GPU memory tracking (optional)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MemoryProfiler:
    """Memory profiler for pipeline execution."""

    def __init__(self, output_dir: str = "profiling_results"):
        """Initialize profiler.

        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[tuple[float, tracemalloc.Snapshot]] = []

    def get_system_memory(self) -> dict[str, Any] | None:
        """Get current system memory info.

        Returns:
            Memory info dict or None if psutil not available
        """
        if not PSUTIL_AVAILABLE:
            return None

        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "used": mem.used,
            "free": mem.free,
        }

    def get_gpu_memory(self) -> dict[str, Any] | None:
        """Get current GPU memory info.

        Returns:
            GPU memory info dict or None if torch/CUDA not available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        # Get GPU memory for all available devices
        num_gpus = torch.cuda.device_count()
        gpu_info = {
            "num_gpus": num_gpus,
            "devices": [],
        }

        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            max_allocated = torch.cuda.max_memory_allocated(i)
            max_reserved = torch.cuda.max_memory_reserved(i)

            # Get total memory (requires calling a CUDA function)
            total_memory = torch.cuda.get_device_properties(i).total_memory

            gpu_info["devices"].append(
                {
                    "index": i,
                    "name": device_name,
                    "allocated": allocated,
                    "allocated_mb": allocated / 1024 / 1024,
                    "reserved": reserved,
                    "reserved_mb": reserved / 1024 / 1024,
                    "max_allocated": max_allocated,
                    "max_allocated_mb": max_allocated / 1024 / 1024,
                    "max_reserved": max_reserved,
                    "max_reserved_mb": max_reserved / 1024 / 1024,
                    "total": total_memory,
                    "total_mb": total_memory / 1024 / 1024,
                    "utilization_percent": (allocated / total_memory * 100) if total_memory > 0 else 0,
                }
            )

        return gpu_info

    def profile_pipeline(
        self,
        input_path: str,
        backend: str,
        model: str | None = None,
        detector: str = "doclayout-yolo",
        sorter: str | None = None,
        max_pages: int | None = None,
        use_async: bool = False,
        snapshot_interval: int | None = None,
    ) -> dict[str, Any]:
        """Profile pipeline memory usage.

        Args:
            input_path: Path to input document
            backend: Recognition backend
            model: Model name (optional)
            detector: Detector name
            sorter: Sorter name (optional)
            max_pages: Maximum pages to process
            use_async: Use async processing
            snapshot_interval: Take snapshots every N seconds (optional)

        Returns:
            Memory profiling results
        """
        print(f"\n{'=' * 70}")
        print(f"Memory Profiling: {'Async' if use_async else 'Sync'} Pipeline")
        print(f"{'=' * 70}")

        # Start tracking allocations
        tracemalloc.start()
        gc.collect()  # Start with clean slate

        # Reset GPU memory stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Get initial memory state
        start_mem = self.get_system_memory()
        start_gpu_mem = self.get_gpu_memory()
        start_snapshot = tracemalloc.take_snapshot()
        start_time = time.perf_counter()

        # Initialize snapshots list
        self.snapshots = [(0.0, start_snapshot)]

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
                num_pages = len(result.pages)
                num_blocks = sum(len(p.blocks) for p in result.pages)
            else:
                result = pipeline.process_image(input_file)
                num_pages = 1
                num_blocks = len(result.get("regions", []))  # type: ignore

            # Get final memory state
            execution_time = time.perf_counter() - start_time
            end_snapshot = tracemalloc.take_snapshot()
            end_mem = self.get_system_memory()
            end_gpu_mem = self.get_gpu_memory()

            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()

            # Stop tracking
            tracemalloc.stop()

            print(f"✅ Execution completed in {execution_time:.3f}s")
            print(f"   Pages processed: {num_pages}")
            print(f"   Total blocks: {num_blocks}")
            print(f"   Peak memory: {peak / 1024 / 1024:.2f} MB")
            print(f"   Current memory: {current / 1024 / 1024:.2f} MB")

            # Print GPU memory stats if available
            if end_gpu_mem and end_gpu_mem["num_gpus"] > 0:
                print("\n   GPU Memory:")
                for device in end_gpu_mem["devices"]:
                    print(f"   - GPU {device['index']} ({device['name']}):")
                    print(f"     Peak allocated: {device['max_allocated_mb']:.2f} MB")
                    print(f"     Current allocated: {device['allocated_mb']:.2f} MB")
                    print(f"     Utilization: {device['utilization_percent']:.1f}%")

            # Analyze memory growth
            memory_diff = self._analyze_snapshot_diff(start_snapshot, end_snapshot)

            results = {
                "execution_time": execution_time,
                "pages": num_pages,
                "blocks": num_blocks,
                "memory": {
                    "peak": peak,
                    "current": current,
                    "peak_mb": peak / 1024 / 1024,
                    "current_mb": current / 1024 / 1024,
                },
                "system_memory": {
                    "start": start_mem,
                    "end": end_mem,
                },
                "gpu_memory": {
                    "start": start_gpu_mem,
                    "end": end_gpu_mem,
                },
                "top_allocations": memory_diff,
            }

            return results

        except Exception as e:
            tracemalloc.stop()
            print(f"❌ Profiling failed: {e}")
            raise

    def _analyze_snapshot_diff(
        self,
        snapshot1: tracemalloc.Snapshot,
        snapshot2: tracemalloc.Snapshot,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """Analyze difference between two memory snapshots.

        Args:
            snapshot1: Start snapshot
            snapshot2: End snapshot
            top_n: Number of top allocations to return

        Returns:
            List of top memory allocations
        """
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        allocations = []
        for stat in top_stats[:top_n]:
            frame = stat.traceback[0]
            # Load the line of code
            line = linecache.getline(frame.filename, frame.lineno).strip()

            allocations.append(
                {
                    "filename": frame.filename,
                    "lineno": frame.lineno,
                    "size": stat.size,
                    "size_mb": stat.size / 1024 / 1024,
                    "size_diff": stat.size_diff,
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count": stat.count,
                    "count_diff": stat.count_diff,
                    "code": line,
                }
            )

        return allocations

    def save_results(
        self,
        results: dict[str, Any],
        config: dict[str, Any],
        suffix: str = "",
    ) -> None:
        """Save memory profiling results.

        Args:
            results: Profile results
            config: Configuration used
            suffix: Filename suffix
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"memory_{timestamp}{suffix}"

        # Save JSON report
        report = {
            "config": config,
            "results": results,
        }

        json_file = self.output_dir / f"{base_name}.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Memory profile saved: {json_file}")

        # Save text report
        text_file = self.output_dir / f"{base_name}.txt"
        with open(text_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MEMORY PROFILING REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nExecution:\n")
            f.write(f"  Time: {results['execution_time']:.3f}s\n")
            f.write(f"  Pages: {results['pages']}\n")
            f.write(f"  Blocks: {results['blocks']}\n")

            f.write("\nMemory Usage:\n")
            f.write(f"  Peak: {results['memory']['peak_mb']:.2f} MB\n")
            f.write(f"  Current: {results['memory']['current_mb']:.2f} MB\n")

            if results["system_memory"]["start"]:
                start_mem = results["system_memory"]["start"]
                end_mem = results["system_memory"]["end"]
                f.write("\nSystem Memory:\n")
                f.write(f"  Start: {start_mem['used'] / 1024 / 1024:.2f} MB used\n")
                f.write(f"  End: {end_mem['used'] / 1024 / 1024:.2f} MB used\n")
                f.write(f"  Diff: {(end_mem['used'] - start_mem['used']) / 1024 / 1024:.2f} MB\n")

            # GPU memory section
            if results["gpu_memory"]["end"] and results["gpu_memory"]["end"]["num_gpus"] > 0:
                f.write("\nGPU Memory:\n")
                for device in results["gpu_memory"]["end"]["devices"]:
                    f.write(f"  GPU {device['index']} ({device['name']}):\n")
                    f.write(f"    Peak allocated: {device['max_allocated_mb']:.2f} MB\n")
                    f.write(f"    Current allocated: {device['allocated_mb']:.2f} MB\n")
                    f.write(f"    Total memory: {device['total_mb']:.2f} MB\n")
                    f.write(f"    Utilization: {device['utilization_percent']:.1f}%\n")

            f.write(f"\nTop {len(results['top_allocations'])} Memory Allocations:\n")
            f.write(f"{'File:Line':<50} {'Size':<15} {'Diff':<15}\n")
            f.write("-" * 80 + "\n")
            for alloc in results["top_allocations"]:
                location = f"{Path(alloc['filename']).name}:{alloc['lineno']}"
                size_str = f"{alloc['size_mb']:.2f} MB"
                diff_str = (
                    f"+{alloc['size_diff_mb']:.2f} MB" if alloc["size_diff"] > 0 else f"{alloc['size_diff_mb']:.2f} MB"
                )
                f.write(f"{location:<50} {size_str:<15} {diff_str:<15}\n")

        print(f"📊 Text report saved: {text_file}")

        # Print top allocations
        print(f"\n🔥 Top {len(results['top_allocations'])} Memory Allocations:")
        print(f"{'Location':<60} {'Size':<15} {'Diff':<15}")
        print("-" * 90)
        for alloc in results["top_allocations"][:10]:
            location = f"{Path(alloc['filename']).name}:{alloc['lineno']}"
            size_str = f"{alloc['size_mb']:.3f} MB"
            diff_str = f"+{alloc['size_diff_mb']:.3f}" if alloc["size_diff"] > 0 else f"{alloc['size_diff_mb']:.3f}"
            print(f"{location:<60} {size_str:<15} {diff_str:<15}")

    def compare_async(
        self,
        input_path: str,
        backend: str,
        model: str | None = None,
        detector: str = "doclayout-yolo",
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """Compare sync vs async memory usage.

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
        print("COMPARING SYNC vs ASYNC MEMORY USAGE")
        print(f"{'=' * 70}")

        # Profile sync
        print("\n1️⃣ Profiling SYNC mode...")
        sync_results = self.profile_pipeline(
            input_path=input_path,
            backend=backend,
            model=model,
            detector=detector,
            max_pages=max_pages,
            use_async=False,
        )
        self.save_results(
            sync_results,
            {"mode": "sync", "backend": backend, "detector": detector},
            suffix="_sync",
        )

        # Profile async
        print("\n2️⃣ Profiling ASYNC mode...")
        async_results = self.profile_pipeline(
            input_path=input_path,
            backend=backend,
            model=model,
            detector=detector,
            max_pages=max_pages,
            use_async=True,
        )
        self.save_results(
            async_results,
            {"mode": "async", "backend": backend, "detector": detector},
            suffix="_async",
        )

        # Compare
        sync_peak = sync_results["memory"]["peak_mb"]
        async_peak = async_results["memory"]["peak_mb"]
        memory_diff = async_peak - sync_peak
        memory_percent = (memory_diff / sync_peak * 100) if sync_peak > 0 else 0

        comparison = {
            "sync_peak_mb": sync_peak,
            "async_peak_mb": async_peak,
            "memory_diff_mb": memory_diff,
            "memory_diff_percent": memory_percent,
        }

        # Print comparison
        print(f"\n{'=' * 70}")
        print("📊 MEMORY COMPARISON RESULTS")
        print(f"{'=' * 70}")
        print(f"Sync peak:   {sync_peak:>10.2f} MB")
        print(f"Async peak:  {async_peak:>10.2f} MB")
        print(f"Difference:  {memory_diff:>10.2f} MB ({memory_percent:>+6.1f}%)")
        print(f"{'=' * 70}")

        # Save comparison
        comparison_file = self.output_dir / f"memory_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n📊 Comparison saved: {comparison_file}")

        return comparison


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile pipeline memory usage")
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
        help="Compare sync vs async memory usage",
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Use async processing (ignored if --compare-async)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        help="Take memory snapshots every N seconds",
    )

    args = parser.parse_args()

    # Check psutil
    if not PSUTIL_AVAILABLE:
        print("⚠️  Warning: psutil not installed. System memory tracking unavailable.")
        print("   Install with: uv pip install psutil")

    # Check torch
    if not TORCH_AVAILABLE:
        print("⚠️  Warning: torch not installed. GPU memory tracking unavailable.")
        print("   Install with: uv pip install torch")
    elif not torch.cuda.is_available():
        print("ℹ️  Info: CUDA not available. GPU memory tracking disabled.")

    try:
        profiler = MemoryProfiler(output_dir=args.output_dir)

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
            results = profiler.profile_pipeline(
                input_path=args.input,
                backend=args.backend,
                model=args.model,
                detector=args.detector,
                sorter=args.sorter,
                max_pages=args.max_pages,
                use_async=args.use_async,
                snapshot_interval=args.snapshot_interval,
            )

            # Save results
            config = {
                "mode": "async" if args.use_async else "sync",
                "backend": args.backend,
                "detector": args.detector,
                "sorter": args.sorter,
            }
            profiler.save_results(results, config)

        print(f"\n✅ Memory profiling complete. Results saved to: {profiler.output_dir}")

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
