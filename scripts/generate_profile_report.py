#!/usr/bin/env python3
"""Generate HTML report from profiling results.

This script analyzes profiling results (CPU and memory) and generates
a comprehensive HTML report with visualizations.

Usage:
    # Generate report from profiling results directory
    python scripts/generate_profile_report.py --input profiling_results

    # Generate report from specific result files
    python scripts/generate_profile_report.py --cpu profile_20250127_143000.json --memory memory_20250127_143000.json

    # Open report in browser after generation
    python scripts/generate_profile_report.py --input profiling_results --open
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ProfileReportGenerator:
    """Generate HTML reports from profiling results."""

    def __init__(self):
        """Initialize report generator."""
        self.cpu_results: dict[str, Any] | None = None
        self.memory_results: dict[str, Any] | None = None

    def load_results(self, results_dir: Path) -> bool:
        """Load profiling results from directory.

        Args:
            results_dir: Directory containing profiling results

        Returns:
            True if at least one result file was loaded
        """
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return False

        # Find latest CPU and memory results
        cpu_files = sorted(results_dir.glob("profile_*.json"), reverse=True)
        memory_files = sorted(results_dir.glob("memory_*.json"), reverse=True)

        loaded = False

        if cpu_files:
            with open(cpu_files[0]) as f:
                self.cpu_results = json.load(f)
            print(f"✅ Loaded CPU results: {cpu_files[0].name}")
            loaded = True

        if memory_files:
            with open(memory_files[0]) as f:
                self.memory_results = json.load(f)
            print(f"✅ Loaded memory results: {memory_files[0].name}")
            loaded = True

        return loaded

    def load_specific_files(self, cpu_file: Path | None, memory_file: Path | None) -> bool:
        """Load specific profiling result files.

        Args:
            cpu_file: Path to CPU profiling results
            memory_file: Path to memory profiling results

        Returns:
            True if at least one file was loaded
        """
        loaded = False

        if cpu_file and cpu_file.exists():
            with open(cpu_file) as f:
                self.cpu_results = json.load(f)
            print(f"✅ Loaded CPU results: {cpu_file.name}")
            loaded = True

        if memory_file and memory_file.exists():
            with open(memory_file) as f:
                self.memory_results = json.load(f)
            print(f"✅ Loaded memory results: {memory_file.name}")
            loaded = True

        return loaded

    def generate_html_report(self, output_path: Path) -> None:
        """Generate HTML report.

        Args:
            output_path: Path to save HTML report
        """
        if not self.cpu_results and not self.memory_results:
            print("❌ No profiling results loaded")
            return

        html = self._build_html()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n✅ HTML report generated: {output_path}")

    def _build_html(self) -> str:
        """Build HTML report content.

        Returns:
            HTML string
        """
        sections = []

        # Header
        sections.append(self._build_header())

        # Summary section
        sections.append(self._build_summary())

        # CPU profiling section
        if self.cpu_results:
            sections.append(self._build_cpu_section())

        # Memory profiling section
        if self.memory_results:
            sections.append(self._build_memory_section())

        # Footer
        sections.append(self._build_footer())

        # Combine all sections
        body = "\n".join(sections)

        # Complete HTML document
        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Profiling Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px 8px 0 0;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        section {{
            padding: 40px;
            border-bottom: 1px solid #eee;
        }}
        section:last-child {{
            border-bottom: none;
        }}
        h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 0.6em;
            color: #666;
            font-weight: normal;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        thead {{
            background: #f8f9fa;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            font-weight: 600;
            color: #667eea;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .code-block {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }}
    </style>
</head>
<body>
    {body}
</body>
</html>"""

    def _build_header(self) -> str:
        """Build header section."""
        import time

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"""
    <div class="container">
        <header>
            <h1>📊 Performance Profiling Report</h1>
            <p class="subtitle">Generated on {timestamp}</p>
        </header>"""

    def _build_summary(self) -> str:
        """Build summary section."""
        metrics = []

        # CPU metrics
        if self.cpu_results:
            exec_time = self.cpu_results.get("execution_time", 0)
            throughput = self.cpu_results.get("throughput", 0)
            metrics.append(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Execution Time</div>
                    <div class="metric-value">{exec_time:.2f} <span class="metric-unit">seconds</span></div>
                </div>"""
            )
            if throughput > 0:
                metrics.append(
                    f"""
                <div class="metric-card">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value">{throughput:.2f} <span class="metric-unit">blocks/sec</span></div>
                </div>"""
                )

        # Memory metrics
        if self.memory_results:
            peak_mb = self.memory_results.get("results", {}).get("memory", {}).get("peak_mb", 0)
            metrics.append(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Memory</div>
                    <div class="metric-value">{peak_mb:.2f} <span class="metric-unit">MB</span></div>
                </div>"""
            )

        # Pages and blocks
        if self.cpu_results:
            total_time = self.cpu_results.get("total_time", 0)
            pages = self.cpu_results.get("pages", 0)
            blocks = self.cpu_results.get("total_blocks", 0)
            metrics.append(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Time</div>
                    <div class="metric-value">{total_time:.2f} <span class="metric-unit">seconds</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pages Processed</div>
                    <div class="metric-value">{pages}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Blocks Processed</div>
                    <div class="metric-value">{blocks}</div>
                </div>"""
            )

        return f"""
        <section>
            <h2>Summary</h2>
            <div class="metric-grid">
                {''.join(metrics) if metrics else '<div class="no-data">No summary metrics available</div>'}
            </div>
        </section>"""

    def _build_cpu_section(self) -> str:
        """Build CPU profiling section."""
        if not self.cpu_results:
            return ""

        hotspots = self.cpu_results.get("hotspots", [])

        # Build hotspots table
        rows = []
        for hotspot in hotspots[:15]:
            func = hotspot["function"].split(":")[-1]  # Get function name
            percent = hotspot["percent"]
            cum_time = hotspot["cumulative_time"]
            calls = hotspot["calls"]

            rows.append(
                f"""
                <tr>
                    <td><code>{func}</code></td>
                    <td>{calls:,}</td>
                    <td>{cum_time:.3f}s</td>
                    <td><strong>{percent:.1f}%</strong></td>
                </tr>"""
            )

        table_html = f"""
            <table>
                <thead>
                    <tr>
                        <th>Function</th>
                        <th>Calls</th>
                        <th>Cumulative Time</th>
                        <th>% of Total</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows) if rows else '<tr><td colspan="4" class="no-data">No hotspots found</td></tr>'}
                </tbody>
            </table>"""

        return f"""
        <section>
            <h2>CPU Profiling</h2>
            <p><span class="badge badge-info">Top Performance Hotspots</span></p>
            {table_html}
        </section>"""

    def _build_memory_section(self) -> str:
        """Build memory profiling section."""
        if not self.memory_results:
            return ""

        allocations = self.memory_results.get("results", {}).get("top_allocations", [])

        # Build allocations table
        rows = []
        for alloc in allocations[:15]:
            filename = Path(alloc["filename"]).name
            lineno = alloc["lineno"]
            size_mb = alloc["size_mb"]
            diff_mb = alloc.get("size_diff_mb", 0)

            badge = "badge-success" if diff_mb < 1 else "badge-warning" if diff_mb < 10 else "badge-danger"
            rows.append(
                f"""
                <tr>
                    <td><code>{filename}:{lineno}</code></td>
                    <td>{size_mb:.2f} MB</td>
                    <td><span class="badge {badge}">{diff_mb:+.2f} MB</span></td>
                </tr>"""
            )

        table_html = f"""
            <table>
                <thead>
                    <tr>
                        <th>Location</th>
                        <th>Size</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows) if rows else '<tr><td colspan="3" class="no-data">No allocations found</td></tr>'}
                </tbody>
            </table>"""

        return f"""
        <section>
            <h2>Memory Profiling</h2>
            <p><span class="badge badge-warning">Top Memory Allocations</span></p>
            {table_html}
        </section>"""

    def _build_footer(self) -> str:
        """Build footer section."""
        return """
        <footer>
            <p>Generated with VLM OCR Pipeline Profiler</p>
            <p>🤖 Powered by Claude Code</p>
        </footer>
    </div>"""


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate profiling HTML report")
    parser.add_argument(
        "--input",
        type=Path,
        help="Directory containing profiling results",
    )
    parser.add_argument(
        "--cpu",
        type=Path,
        help="Specific CPU profiling result file",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        help="Specific memory profiling result file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_results/report.html"),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open report in browser after generation",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input and not args.cpu and not args.memory:
        parser.error("Must specify either --input or --cpu/--memory")

    try:
        generator = ProfileReportGenerator()

        # Load results
        if args.input:
            if not generator.load_results(args.input):
                sys.exit(1)
        else:
            if not generator.load_specific_files(args.cpu, args.memory):
                sys.exit(1)

        # Generate report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generator.generate_html_report(args.output)

        # Open in browser
        if args.open:
            print(f"\n🌐 Opening report in browser...")
            webbrowser.open(f"file://{args.output.absolute()}")

    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
