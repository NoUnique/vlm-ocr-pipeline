"""PaddleOCR recognition modules."""

from __future__ import annotations

__all__ = ["PaddleOCRVLRecognizer"]

try:
    from .paddleocr_vl import PaddleOCRVLRecognizer  # noqa: PLC0415
except ImportError:
    PaddleOCRVLRecognizer = None  # type: ignore[assignment, misc]
