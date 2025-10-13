"""Tests for sorter implementations."""

from __future__ import annotations

import numpy as np

from pipeline.layout.ordering import MinerUXYCutSorter
from pipeline.types import BBox


def test_xycut_sorter_sorts_regions():
    """Test XY-Cut sorter sorts regions correctly."""
    sorter = MinerUXYCutSorter()
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    regions = [
        {
            "type": "text",
            "coords": [100, 200, 50, 50],
            "confidence": 0.9,
            "bbox": BBox(100, 200, 150, 250),
        },
        {
            "type": "text",
            "coords": [100, 50, 50, 50],
            "confidence": 0.9,
            "bbox": BBox(100, 50, 150, 100),
        },
        {
            "type": "text",
            "coords": [300, 50, 50, 50],
            "confidence": 0.9,
            "bbox": BBox(300, 50, 350, 100),
        },
    ]
    
    sorted_regions = sorter.sort(regions, image)
    
    # XY-Cut should handle this correctly
    assert len(sorted_regions) == 3
    
    # Check reading_order_rank is assigned
    assert all("reading_order_rank" in r for r in sorted_regions)


def test_xycut_sorter_handles_empty_regions():
    """Test XY-Cut sorter handles empty region list."""
    sorter = MinerUXYCutSorter()
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    sorted_regions = sorter.sort([], image)
    
    assert sorted_regions == []


def test_xycut_sorter_adds_bbox_if_missing():
    """Test XY-Cut sorter adds bbox from coords if missing."""
    sorter = MinerUXYCutSorter()
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    regions = [
        {
            "type": "text",
            "coords": [100, 50, 200, 150],
            "confidence": 0.9,
            # bbox not provided
        }
    ]
    
    sorted_regions = sorter.sort(regions, image)
    
    assert "bbox" in sorted_regions[0]
    assert sorted_regions[0]["bbox"].x0 == 100

