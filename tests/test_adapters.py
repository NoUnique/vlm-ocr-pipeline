"""Tests for framework adapters."""

from __future__ import annotations

from pipeline.adapters import DocLayoutYOLOAdapter, MinerUAdapter
from pipeline.types import BBox


def test_doclayout_yolo_adapter_to_region():
    """Test DocLayout-YOLO adapter to_region conversion."""
    adapter = DocLayoutYOLOAdapter()
    
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


def test_doclayout_yolo_adapter_from_region():
    """Test DocLayout-YOLO adapter from_region conversion."""
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


def test_mineru_adapter_to_region():
    """Test MinerU adapter to_region conversion."""
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


def test_mineru_adapter_from_region():
    """Test MinerU adapter from_region conversion."""
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

