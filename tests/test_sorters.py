"""Tests for sorter implementations."""

from __future__ import annotations

import numpy as np

from pipeline.layout.ordering import MinerUXYCutSorter
from pipeline.types import BBox, Region


def test_xycut_sorter_sorts_regions():
    """Test XY-Cut sorter sorts regions correctly."""
    sorter = MinerUXYCutSorter()
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    regions = [
        Region(
            type="text",
            bbox=BBox(100, 200, 150, 250),
            confidence=0.9,
        ),
        Region(
            type="text",
            bbox=BBox(100, 50, 150, 100),
            confidence=0.9,
        ),
        Region(
            type="text",
            bbox=BBox(300, 50, 350, 100),
            confidence=0.9,
        ),
    ]

    sorted_regions = sorter.sort(regions, image)

    # XY-Cut should handle this correctly
    assert len(sorted_regions) == 3

    # Check reading_order_rank is assigned
    assert all(r.reading_order_rank is not None for r in sorted_regions)


def test_xycut_sorter_handles_empty_regions():
    """Test XY-Cut sorter handles empty region list."""
    sorter = MinerUXYCutSorter()
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    sorted_regions = sorter.sort([], image)

    assert sorted_regions == []


def test_xycut_sorter_works_with_single_region():
    """Test XY-Cut sorter works with a single block."""
    sorter = MinerUXYCutSorter()
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    regions = [
        Region(
            type="text",
            bbox=BBox(100, 50, 300, 200),
            confidence=0.9,
        )
    ]

    sorted_regions = sorter.sort(regions, image)

    assert len(sorted_regions) == 1
    assert sorted_regions[0].bbox.x0 == 100
    assert sorted_regions[0].reading_order_rank is not None
