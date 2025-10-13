"""Tests for pipeline types (BBox, Region)."""

from __future__ import annotations

import pytest

from pipeline.types import BBox, ensure_bbox_in_region, regions_to_olmocr_anchor_text


def test_bbox_from_xywh():
    """Test BBox creation from xywh format."""
    bbox = BBox.from_xywh(100, 50, 200, 150)
    
    assert bbox.x0 == 100
    assert bbox.y0 == 50
    assert bbox.x1 == 300
    assert bbox.y1 == 200


def test_bbox_from_xyxy():
    """Test BBox creation from xyxy format."""
    bbox = BBox.from_xyxy(100, 50, 300, 200)
    
    assert bbox.x0 == 100
    assert bbox.y0 == 50
    assert bbox.x1 == 300
    assert bbox.y1 == 200


def test_bbox_to_xywh():
    """Test BBox conversion to xywh format."""
    bbox = BBox(100, 50, 300, 200)
    x, y, w, h = bbox.to_xywh()
    
    assert x == 100
    assert y == 50
    assert w == 200
    assert h == 150


def test_bbox_to_xyxy():
    """Test BBox conversion to xyxy format."""
    bbox = BBox(100, 50, 300, 200)
    x0, y0, x1, y1 = bbox.to_xyxy()
    
    assert x0 == 100
    assert y0 == 50
    assert x1 == 300
    assert y1 == 200


def test_bbox_properties():
    """Test BBox geometric properties."""
    bbox = BBox(100, 50, 300, 200)
    
    assert bbox.width == 200
    assert bbox.height == 150
    assert bbox.area == 30000
    assert bbox.center == (200.0, 125.0)


def test_bbox_intersect():
    """Test BBox intersection calculation."""
    bbox1 = BBox(100, 50, 300, 200)
    bbox2 = BBox(200, 100, 400, 250)
    
    intersection = bbox1.intersect(bbox2)
    assert intersection == 5000.0  # 100 * 50


def test_bbox_iou():
    """Test BBox IoU calculation."""
    bbox1 = BBox(100, 50, 300, 200)
    bbox2 = BBox(100, 50, 300, 200)
    
    iou = bbox1.iou(bbox2)
    assert iou == 1.0  # Perfect overlap


def test_bbox_to_olmocr_anchor_text():
    """Test BBox to olmOCR anchor text conversion."""
    bbox = BBox(100, 50, 300, 200)
    
    # Text format
    anchor = bbox.to_olmocr_anchor("text", "Chapter 1")
    assert anchor == "[100x50]Chapter 1"
    
    # Image format
    anchor = bbox.to_olmocr_anchor("image")
    assert anchor == "[Image 100x50 to 300x200]"
    
    # Table format
    anchor = bbox.to_olmocr_anchor("table")
    assert anchor == "[Table 100x50 to 300x200]"


def test_ensure_bbox_in_region():
    """Test ensure_bbox_in_region utility."""
    region = {
        "type": "text",
        "coords": [100, 50, 200, 150],
        "confidence": 0.9,
    }
    
    region = ensure_bbox_in_region(region)
    
    assert "bbox" in region
    assert region["bbox"].x0 == 100
    assert region["bbox"].y0 == 50
    assert region["bbox"].x1 == 300
    assert region["bbox"].y1 == 200


def test_regions_to_olmocr_anchor_text():
    """Test regions to olmOCR anchor text conversion."""
    regions = [
        {"type": "title", "coords": [100, 50, 200, 30], "confidence": 0.9, "text": "Chapter 1"},
        {"type": "figure", "coords": [100, 100, 200, 150], "confidence": 0.95},
        {"type": "plain text", "coords": [100, 300, 400, 50], "confidence": 0.9, "text": "Content here"},
    ]
    
    anchor_text = regions_to_olmocr_anchor_text(regions, 800, 600)
    
    assert "Page dimensions: 800.0x600.0" in anchor_text
    assert "[100x50]Chapter 1" in anchor_text
    assert "[Image 100x100 to 300x250]" in anchor_text
    assert "[100x300]Content here" in anchor_text


def test_bbox_from_list_xywh():
    """Test BBox creation from list with xywh format."""
    bbox = BBox.from_list([100, 50, 200, 150], format="xywh")
    
    assert bbox.x0 == 100
    assert bbox.x1 == 300


def test_bbox_from_list_xyxy():
    """Test BBox creation from list with xyxy format."""
    bbox = BBox.from_list([100, 50, 300, 200], format="xyxy")
    
    assert bbox.x0 == 100
    assert bbox.x1 == 300


def test_bbox_from_list_invalid_format():
    """Test BBox creation with invalid format."""
    with pytest.raises(ValueError, match="Unknown bbox format"):
        BBox.from_list([100, 50, 200, 150], format="invalid")


def test_bbox_clip():
    """Test BBox clipping to image boundaries."""
    bbox = BBox(100, 50, 1000, 900)
    clipped = bbox.clip(800, 600)
    
    assert clipped.x0 == 100
    assert clipped.y0 == 50
    assert clipped.x1 == 800
    assert clipped.y1 == 600


def test_bbox_expand():
    """Test BBox expansion with padding."""
    bbox = BBox(100, 50, 300, 200)
    expanded = bbox.expand(10)
    
    assert expanded.x0 == 90
    assert expanded.y0 == 40
    assert expanded.x1 == 310
    assert expanded.y1 == 210

