"""PaddleOCR reading order sorters.

PP-DocLayoutV2 passthrough sorter (preserves pointer network ordering).
"""

from __future__ import annotations

from .passthrough import PPDocLayoutV2Sorter

__all__ = ["PPDocLayoutV2Sorter"]
