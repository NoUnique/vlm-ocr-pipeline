#!/usr/bin/env python3
"""Show GPU environment and auto-optimization configuration."""

if __name__ == "__main__":
    # Direct import to avoid pipeline dependencies
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))

    from pipeline.gpu_environment import print_gpu_info

    print_gpu_info()
