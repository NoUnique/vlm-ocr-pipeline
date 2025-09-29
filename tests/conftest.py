from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running tests via uv or python -m pytest
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
