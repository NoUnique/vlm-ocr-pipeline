"""Standalone tests for BBox format conversions.

These tests verify bbox format conversions between different frameworks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.types import BBox


class TestBBoxFormatConversions:
    """Test various bbox format conversions."""

    def test_doclayout_to_mineru(self):
        """Test DocLayout-YOLO format to MinerU format."""
        # DocLayout: [x, y, w, h]
        doclayout_coords = [100, 50, 200, 150]
        bbox = BBox.from_list(doclayout_coords, coord_format="xywh")

        # Convert to MinerU: [x0, y0, x1, y1]
        mineru_coords = bbox.to_mineru_bbox()

        assert mineru_coords == [100, 50, 300, 200]

    def test_mineru_to_doclayout(self):
        """Test MinerU format to DocLayout-YOLO format."""
        # MinerU: [x0, y0, x1, y1]
        mineru_coords = [100, 50, 300, 200]
        bbox = BBox.from_mineru_bbox(mineru_coords)

        # Convert to DocLayout: [x, y, w, h]
        doclayout_coords = bbox.to_list_xywh()

        assert doclayout_coords == [100, 50, 200, 150]

    def test_doclayout_to_olmocr_anchor(self):
        """Test DocLayout-YOLO to olmOCR anchor text."""
        doclayout_coords = [100, 50, 200, 150]
        bbox = BBox.from_list(doclayout_coords, coord_format="xywh")

        # Text region
        anchor_text = bbox.to_olmocr_anchor("text", "Chapter 1")
        assert anchor_text == "[100x50]Chapter 1"

        # Image region
        anchor_image = bbox.to_olmocr_anchor("image")
        assert anchor_image == "[Image 100x50 to 300x200]"

    def test_mineru_to_olmocr_anchor(self):
        """Test MinerU bbox to olmOCR anchor text."""
        mineru_bbox = [100, 50, 300, 200]
        bbox = BBox.from_mineru_bbox(mineru_bbox)

        anchor = bbox.to_olmocr_anchor("table")
        assert anchor == "[Table 100x50 to 300x200]"

