"""PaddleOCR reading order sorters.

PP-DocLayoutV2 sorter (preserves pointer network ordering from detector).
"""

from __future__ import annotations

from .doclayout_v2 import PPDocLayoutV2Sorter

__all__ = ["PPDocLayoutV2Sorter"]
