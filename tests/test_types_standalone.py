"""Standalone tests for pipeline types (BBox, Region).

These tests import types.py directly without going through pipeline/__init__.py
to avoid cv2 dependency issues in test environments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Direct import from types module
from pipeline.types import BBox, Region, ensure_bbox_in_region, regions_to_olmocr_anchor_text


class TestBBoxCreation:
    """Test BBox creation from various formats."""

    def test_from_xywh(self):
        """Test BBox creation from xywh format."""
        bbox = BBox.from_xywh(100, 50, 200, 150)

        assert bbox.x0 == 100
        assert bbox.y0 == 50
        assert bbox.x1 == 300
        assert bbox.y1 == 200

    def test_from_xyxy(self):
        """Test BBox creation from xyxy format."""
        bbox = BBox.from_xyxy(100, 50, 300, 200)

        assert bbox.x0 == 100
        assert bbox.y0 == 50
        assert bbox.x1 == 300
        assert bbox.y1 == 200

    def test_from_list_xywh(self):
        """Test BBox creation from list with xywh format."""
        bbox = BBox.from_list([100, 50, 200, 150], coord_format="xywh")

        assert bbox.x0 == 100
        assert bbox.x1 == 300

    def test_from_list_xyxy(self):
        """Test BBox creation from list with xyxy format."""
        bbox = BBox.from_list([100, 50, 300, 200], coord_format="xyxy")

        assert bbox.x0 == 100
        assert bbox.x1 == 300

    def test_from_list_invalid_format(self):
        """Test BBox creation with invalid format raises error."""
        with pytest.raises(ValueError, match="Unknown bbox format"):
            BBox.from_list([100, 50, 200, 150], coord_format="invalid")

    def test_from_mineru_bbox(self):
        """Test BBox creation from MinerU bbox format."""
        bbox = BBox.from_mineru_bbox([100, 50, 300, 200])

        assert bbox.x0 == 100
        assert bbox.y0 == 50
        assert bbox.x1 == 300
        assert bbox.y1 == 200


class TestBBoxConversion:
    """Test BBox conversion to various formats."""

    def test_to_xywh(self):
        """Test BBox conversion to xywh format."""
        bbox = BBox(100, 50, 300, 200)
        x, y, w, h = bbox.to_xywh()

        assert x == 100
        assert y == 50
        assert w == 200
        assert h == 150

    def test_to_xyxy(self):
        """Test BBox conversion to xyxy format."""
        bbox = BBox(100, 50, 300, 200)
        x0, y0, x1, y1 = bbox.to_xyxy()

        assert x0 == 100
        assert y0 == 50
        assert x1 == 300
        assert y1 == 200

    def test_to_list_xywh(self):
        """Test BBox conversion to list xywh."""
        bbox = BBox(100, 50, 300, 200)
        coords = bbox.to_list_xywh()

        assert coords == [100, 50, 200, 150]

    def test_to_list_xyxy(self):
        """Test BBox conversion to list xyxy."""
        bbox = BBox(100, 50, 300, 200)
        coords = bbox.to_list_xyxy()

        assert coords == [100, 50, 300, 200]

    def test_to_mineru_bbox(self):
        """Test BBox conversion to MinerU format."""
        bbox = BBox(100, 50, 300, 200)
        mineru_bbox = bbox.to_mineru_bbox()

        assert mineru_bbox == [100, 50, 300, 200]

    def test_to_olmocr_anchor_text(self):
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

        # Figure format
        anchor = bbox.to_olmocr_anchor("figure")
        assert anchor == "[Image 100x50 to 300x200]"


class TestBBoxProperties:
    """Test BBox geometric properties."""

    def test_width(self):
        """Test BBox width property."""
        bbox = BBox(100, 50, 300, 200)
        assert bbox.width == 200

    def test_height(self):
        """Test BBox height property."""
        bbox = BBox(100, 50, 300, 200)
        assert bbox.height == 150

    def test_area(self):
        """Test BBox area property."""
        bbox = BBox(100, 50, 300, 200)
        assert bbox.area == 30000

    def test_center(self):
        """Test BBox center property."""
        bbox = BBox(100, 50, 300, 200)
        assert bbox.center == (200.0, 125.0)


class TestBBoxGeometricOperations:
    """Test BBox geometric operations."""

    def test_intersect(self):
        """Test BBox intersection calculation."""
        bbox1 = BBox(100, 50, 300, 200)
        bbox2 = BBox(200, 100, 400, 250)

        intersection = bbox1.intersect(bbox2)
        # Overlap: x[200, 300] * y[100, 200] = 100 * 100 = 10000
        assert intersection == 10000.0

    def test_intersect_no_overlap(self):
        """Test BBox intersection with no overlap."""
        bbox1 = BBox(100, 50, 200, 150)
        bbox2 = BBox(300, 200, 400, 300)

        intersection = bbox1.intersect(bbox2)
        assert intersection == 0.0

    def test_iou_identical(self):
        """Test BBox IoU with identical boxes."""
        bbox1 = BBox(100, 50, 300, 200)
        bbox2 = BBox(100, 50, 300, 200)

        iou = bbox1.iou(bbox2)
        assert iou == 1.0

    def test_iou_partial_overlap(self):
        """Test BBox IoU with partial overlap."""
        bbox1 = BBox(100, 50, 300, 200)
        bbox2 = BBox(200, 100, 400, 250)

        iou = bbox1.iou(bbox2)
        # Intersection: 10000, Union: 30000 + 30000 - 10000 = 50000
        assert abs(iou - 0.2) < 0.01

    def test_overlap_ratio(self):
        """Test BBox overlap ratio calculation."""
        bbox1 = BBox(100, 50, 200, 150)  # Area: 10000
        bbox2 = BBox(100, 50, 300, 200)  # Fully contains bbox1

        overlap = bbox1.overlap_ratio(bbox2)
        assert overlap == 1.0  # bbox1 fully contained

    def test_contains_point(self):
        """Test BBox contains_point method."""
        bbox = BBox(100, 50, 300, 200)

        assert bbox.contains_point(200, 125) is True
        assert bbox.contains_point(100, 50) is True
        assert bbox.contains_point(300, 200) is True
        assert bbox.contains_point(50, 25) is False
        assert bbox.contains_point(400, 300) is False

    def test_expand(self):
        """Test BBox expansion with padding."""
        bbox = BBox(100, 50, 300, 200)
        expanded = bbox.expand(10)

        assert expanded.x0 == 90
        assert expanded.y0 == 40
        assert expanded.x1 == 310
        assert expanded.y1 == 210

    def test_clip(self):
        """Test BBox clipping to image boundaries."""
        bbox = BBox(100, 50, 1000, 900)
        clipped = bbox.clip(800, 600)

        assert clipped.x0 == 100
        assert clipped.y0 == 50
        assert clipped.x1 == 800
        assert clipped.y1 == 600

    def test_clip_negative_coords(self):
        """Test BBox clipping with negative coordinates."""
        bbox = BBox(-50, -20, 300, 200)
        clipped = bbox.clip(800, 600)

        assert clipped.x0 == 0
        assert clipped.y0 == 0
        assert clipped.x1 == 300
        assert clipped.y1 == 200


class TestBBoxRoundTrip:
    """Test BBox format round-trip conversions."""

    def test_xywh_roundtrip(self):
        """Test xywh → BBox → xywh."""
        original = [100, 50, 200, 150]
        bbox = BBox.from_list(original, coord_format="xywh")
        converted = bbox.to_list_xywh()

        assert converted == original

    def test_xyxy_roundtrip(self):
        """Test xyxy → BBox → xyxy."""
        original = [100, 50, 300, 200]
        bbox = BBox.from_list(original, coord_format="xyxy")
        converted = bbox.to_list_xyxy()

        assert converted == original

    def test_mineru_roundtrip(self):
        """Test MinerU format round-trip."""
        original = [100, 50, 300, 200]
        bbox = BBox.from_mineru_bbox(original)
        converted = bbox.to_mineru_bbox()

        assert converted == original


class TestRegionUtilities:
    """Test Region utility functions."""

    def test_ensure_bbox_in_region(self):
        """Test ensure_bbox_in_region adds bbox from coords."""
        region: Region = {
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

    def test_ensure_bbox_idempotent(self):
        """Test ensure_bbox_in_region is idempotent."""
        bbox_obj = BBox(100, 50, 300, 200)
        region: Region = {
            "type": "text",
            "coords": [100, 50, 200, 150],
            "confidence": 0.9,
            "bbox": bbox_obj,
        }

        region = ensure_bbox_in_region(region)

        # Should keep the same bbox object
        assert "bbox" in region
        assert region["bbox"] is bbox_obj

    def test_regions_to_olmocr_anchor_text(self):
        """Test regions to olmOCR anchor text conversion."""
        regions: list[Region] = [
            {
                "type": "title",
                "coords": [100, 50, 200, 30],
                "confidence": 0.9,
                "text": "Chapter 1",
            },
            {
                "type": "figure",
                "coords": [100, 100, 200, 150],
                "confidence": 0.95,
            },
            {
                "type": "plain text",
                "coords": [100, 300, 400, 50],
                "confidence": 0.9,
                "text": "Content here with some long text that might be truncated",
            },
        ]

        anchor_text = regions_to_olmocr_anchor_text(regions, 800, 600)

        # Check header
        assert "Page dimensions: 800.0x600.0" in anchor_text

        # Check text region (with partial content)
        assert "[100x50]Chapter 1" in anchor_text

        # Check image region
        assert "[Image 100x100 to 300x250]" in anchor_text

        # Check another text region
        assert "[100x300]Content here" in anchor_text

    def test_regions_to_olmocr_anchor_text_max_length(self):
        """Test anchor text respects max_length limit."""
        # Create many regions
        regions: list[Region] = [
            {
                "type": "text",
                "coords": [i * 10, i * 10, 100, 20],
                "confidence": 0.9,
            }
            for i in range(100)
        ]

        # Set short max_length
        anchor_text = regions_to_olmocr_anchor_text(regions, 800, 600, max_length=200)

        # Should be limited
        assert len(anchor_text) <= 250  # Some buffer for formatting


class TestBBoxEdgeCases:
    """Test BBox edge cases and special scenarios."""

    def test_zero_area_bbox(self):
        """Test BBox with zero area."""
        bbox = BBox(100, 50, 100, 50)

        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0

    def test_negative_size_bbox(self):
        """Test BBox with inverted coordinates."""
        # x1 < x0, y1 < y0
        bbox = BBox(300, 200, 100, 50)

        # Width and height should handle this
        assert bbox.width == -200
        assert bbox.height == -150
        # Area should be 0 (max(0, width) * max(0, height))
        assert bbox.area == 0

    def test_pypdf_rect_conversion(self):
        """Test PyPDF rect conversion with Y-axis flip."""
        # PyPDF uses bottom-left origin
        # Page height: 792
        # Bottom-left origin: [100, 642, 300, 742]
        # Should convert to top-left: [100, 50, 300, 200]
        bbox = BBox.from_pypdf_rect([100, 642, 300, 742], page_height=792)

        assert bbox.x0 == 100
        assert bbox.y0 == 50
        assert bbox.x1 == 300
        assert bbox.y1 == 150

    def test_to_pypdf_rect(self):
        """Test conversion to PyPDF rect format."""
        bbox = BBox(100, 50, 300, 200)
        pypdf_rect = bbox.to_pypdf_rect(page_height=792)

        # Should flip Y-axis back to bottom-left origin
        assert pypdf_rect[0] == 100  # x0 unchanged
        assert pypdf_rect[1] == 592  # y0 flipped: 792 - 200 = 592
        assert pypdf_rect[2] == 300  # x1 unchanged
        assert pypdf_rect[3] == 742  # y1 flipped: 792 - 50 = 742


class TestBBoxImmutability:
    """Test BBox immutability (frozen dataclass)."""

    def test_bbox_is_immutable(self):
        """Test BBox cannot be modified after creation."""
        bbox = BBox(100, 50, 300, 200)

        with pytest.raises(AttributeError):
            bbox.x0 = 200  # type: ignore[misc]

    def test_bbox_hashable(self):
        """Test BBox is hashable (can be used in sets/dicts)."""
        bbox1 = BBox(100, 50, 300, 200)
        bbox2 = BBox(100, 50, 300, 200)
        bbox3 = BBox(200, 100, 400, 300)

        bbox_set = {bbox1, bbox2, bbox3}

        # bbox1 and bbox2 are equal, so set should have 2 elements
        assert len(bbox_set) == 2

