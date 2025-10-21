"""PaddleOCR layout detectors.

PP-DocLayoutV2 Family:
- PP-DocLayout-L: Highest accuracy (90.4 mAP), 23 categories
- PP-DocLayout-M: Balanced (75.2 mAP), 23 categories
- PP-DocLayout-S: Fastest (70.9 mAP), 23 categories
"""

from .doclayout_v2 import PPDocLayoutV2Detector

__all__ = ["PPDocLayoutV2Detector"]
