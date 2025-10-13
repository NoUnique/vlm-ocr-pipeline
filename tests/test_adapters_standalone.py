"""Standalone tests for framework adapters.

These tests import adapters directly to avoid cv2 dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.types import BBox


class TestDocLayoutYOLOConversion:
    """Test DocLayout-YOLO bbox conversions."""

    def test_to_region(self):
        """Test DocLayout-YOLO raw data to Region conversion."""
        from pipeline.layout.detection import DocLayoutYOLODetector
        detector = DocLayoutYOLODetector()

        raw_data = {
            "type": "plain text",
            "coords": [100, 50, 200, 150],
            "confidence": 0.95,
        }

        region = adapter.to_region(raw_data)

        assert region["type"] == "plain text"
        assert region["coords"] == [100, 50, 200, 150]
        assert region["confidence"] == 0.95
        assert "bbox" in region
        assert region["bbox"].x0 == 100
        assert region["bbox"].x1 == 300
        assert region["source"] == "doclayout-yolo"

    def test_from_region(self):
        """Test Region to DocLayout-YOLO format conversion."""
        adapter = DocLayoutYOLOAdapter()

        region = {
            "type": "text",
            "coords": [100, 50, 200, 150],
            "confidence": 0.9,
            "bbox": BBox(100, 50, 300, 200),
            "source": "doclayout-yolo",
        }

        raw_data = adapter.from_region(region)

        assert raw_data["type"] == "text"
        assert raw_data["coords"] == [100, 50, 200, 150]
        assert raw_data["confidence"] == 0.9
        # source should not be in raw data
        assert "source" not in raw_data

    def test_to_region_with_custom_source(self):
        """Test adapter with custom source name."""
        adapter = DocLayoutYOLOAdapter()

        raw_data = {
            "type": "title",
            "coords": [50, 25, 100, 50],
            "confidence": 0.98,
        }

        region = adapter.to_region(raw_data, source="custom-detector")

        assert region["source"] == "custom-detector"


class TestMinerUAdapter:
    """Test MinerU adapter conversions."""

    def test_to_region(self):
        """Test MinerU block to Region conversion."""
        adapter = MinerUAdapter()

        raw_data = {
            "type": "text",
            "bbox": [100, 50, 300, 200],
            "text": "Hello world",
            "index": 0,
            "confidence": 0.95,
        }

        region = adapter.to_region(raw_data)

        assert region["type"] == "text"
        assert region["coords"] == [100, 50, 200, 150]  # Converted to xywh
        assert region["confidence"] == 0.95
        assert region["text"] == "Hello world"
        assert region["index"] == 0
        assert region["reading_order_rank"] == 0  # Copied from index
        assert region["source"] == "mineru-vlm"

    def test_to_region_without_text(self):
        """Test MinerU block without text field."""
        adapter = MinerUAdapter()

        raw_data = {
            "type": "image",
            "bbox": [100, 50, 300, 200],
            "index": 1,
        }

        region = adapter.to_region(raw_data)

        assert region["type"] == "image"
        assert "text" not in region
        assert region["index"] == 1
        assert region["reading_order_rank"] == 1
        assert region["confidence"] == 1.0  # Default

    def test_from_region(self):
        """Test Region to MinerU format conversion."""
        adapter = MinerUAdapter()

        region = {
            "type": "text",
            "coords": [100, 50, 200, 150],
            "confidence": 0.9,
            "bbox": BBox(100, 50, 300, 200),
            "text": "Hello",
            "reading_order_rank": 0,
        }

        raw_data = adapter.from_region(region)

        assert raw_data["type"] == "text"
        assert raw_data["bbox"] == [100, 50, 300, 200]  # Converted to xyxy
        assert raw_data["text"] == "Hello"
        assert raw_data["index"] == 0  # Copied from reading_order_rank

    def test_from_region_without_bbox(self):
        """Test conversion when region doesn't have bbox field."""
        adapter = MinerUAdapter()

        region = {
            "type": "text",
            "coords": [100, 50, 200, 150],
            "confidence": 0.9,
            "reading_order_rank": 5,
        }

        raw_data = adapter.from_region(region)

        # Should reconstruct bbox from coords
        assert raw_data["bbox"] == [100, 50, 300, 200]
        assert raw_data["index"] == 5

    def test_to_region_default_confidence(self):
        """Test MinerU block without confidence uses default."""
        adapter = MinerUAdapter()

        raw_data = {
            "type": "table",
            "bbox": [50, 100, 500, 400],
        }

        region = adapter.to_region(raw_data)

        assert region["confidence"] == 1.0


class TestBBoxFormatConversions:
    """Test various bbox format conversions."""

    def test_doclayout_to_mineru(self):
        """Test DocLayout-YOLO format to MinerU format."""
        # DocLayout: [x, y, w, h]
        doclayout_coords = [100, 50, 200, 150]
        bbox = BBox.from_list(doclayout_coords, format="xywh")

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
        bbox = BBox.from_list(doclayout_coords, format="xywh")

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

