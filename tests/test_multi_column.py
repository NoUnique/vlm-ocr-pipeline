"""Tests for multi-column ordering helpers."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np

from pipeline import ColumnOrderingInfo, Pipeline

EXPECTED_COLUMN_COUNT = 2


class _Box:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


def _make_pipeline_stub() -> Pipeline:
    return Pipeline.__new__(Pipeline)  # type: ignore[call-arg]


def test_sort_regions_by_columns_assigns_reading_order() -> None:
    pipeline = _make_pipeline_stub()
    columns = [
        {"index": 0, "x0": 0.0, "x1": 200.0, "center": 100.0, "width": 200.0},
        {"index": 1, "x0": 200.0, "x1": 400.0, "center": 300.0, "width": 200.0},
    ]
    regions = [
        {"coords": [210, 50, 50, 50], "text": "B1", "type": "plain text"},
        {"coords": [10, 100, 50, 50], "text": "A2", "type": "plain text"},
        {"coords": [10, 20, 50, 50], "text": "A1", "type": "plain text"},
        {"coords": [210, 120, 50, 50], "text": "B2", "type": "plain text"},
    ]

    ordered = pipeline._sort_regions_by_columns(
        regions,
        cast(Sequence[ColumnOrderingInfo], columns),
        assign_rank=True,
        store_column_index=True,
    )

    assert [region["text"] for region in ordered] == ["A1", "A2", "B1", "B2"]
    assert [region["reading_order_rank"] for region in ordered] == [0, 1, 2, 3]
    assert [region["column_index"] for region in ordered] == [0, 0, 1, 1]


def test_compose_page_respects_reading_order_rank() -> None:
    pipeline = _make_pipeline_stub()
    processed_regions = [
        {
            "type": "plain text",
            "coords": [150, 10, 20, 20],
            "text": "Second",
            "reading_order_rank": 1,
        },
        {
            "type": "plain text",
            "coords": [10, 100, 20, 20],
            "text": "Third",
        },
        {
            "type": "plain text",
            "coords": [10, 10, 20, 20],
            "text": "First",
            "reading_order_rank": 0,
        },
    ]

    text = pipeline._compose_page_raw_text(processed_regions)
    assert text.split("\n\n") == ["First", "Second", "Third"]


def test_detect_column_layout_maps_boxes_to_columns(monkeypatch) -> None:
    pipeline = _make_pipeline_stub()

    fake_page = SimpleNamespace(rect=SimpleNamespace(width=400, height=800))
    fake_image = np.zeros((800, 400, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "pipeline.column_boxes",
        lambda page: [_Box(0, 0, 150, 200), _Box(200, 0, 350, 200)],
    )

    layout = pipeline._detect_column_layout(fake_page, fake_image)

    assert layout is not None
    assert len(layout["columns"]) == EXPECTED_COLUMN_COUNT
    ordering = cast(Sequence[ColumnOrderingInfo], layout["_ordering_columns"])
    assert ordering[0]["index"] == 0
    assert ordering[1]["index"] == 1
    assert ordering[0]["x0"] < ordering[1]["x0"]
