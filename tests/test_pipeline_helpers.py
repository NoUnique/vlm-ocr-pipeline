"""Tests for pipeline helper functions.

Tests helper methods and utilities used by the Pipeline class:
- Text composition from blocks (ordering, filtering)
- Page processing selection (page ranges, max pages)
- Error detection (rate limits, processing failures)
"""

import pytest

from pipeline import Pipeline
from pipeline.conversion.input import pdf as pdf_converter
from pipeline.layout.ordering import ReadingOrderAnalyzer
from pipeline.types import BBox, Block


def make_pipeline():
    return Pipeline.__new__(Pipeline)


def test_compose_page_raw_text_orders_and_preserves_blocks():
    analyzer = ReadingOrderAnalyzer()
    blocks = [
        Block(type="plain text", bbox=BBox(30, 20, 70, 30), text="Second block"),
        Block(type="plain text", bbox=BBox(10, 20, 50, 30), text="First block"),
        Block(type="title", bbox=BBox(5, 80, 45, 90), text=" Title heading\nSubtitle "),
        Block(type="figure", bbox=BBox(0, 0, 10, 10), text="Ignored"),
    ]

    result = analyzer.compose_page_text(blocks)

    assert result == "First block\n\nSecond block\n\nTitle heading\nSubtitle"


def test_compose_page_raw_text_skips_invalid_entries():
    analyzer = ReadingOrderAnalyzer()
    blocks = [
        Block(type="plain text", bbox=BBox(0, 0, 0, 0), text=""),
        Block(type="table", bbox=BBox(0, 0, 0, 0), text="Table"),
        Block(type="plain text", bbox=BBox(0, 0, 0, 0)),  # Missing text
    ]

    result = analyzer.compose_page_text(blocks)

    assert result == ""


def test_determine_pages_to_process_honors_priority_and_bounds():
    total_pages = 10

    result_specific = pdf_converter.determine_pages_to_process(total_pages, pages=[5, 1, 12, 0])
    result_range = pdf_converter.determine_pages_to_process(total_pages, page_range=(0, 12))
    result_max = pdf_converter.determine_pages_to_process(total_pages, max_pages=3)
    result_all = pdf_converter.determine_pages_to_process(total_pages)

    assert result_specific == [1, 5]
    assert result_range == list(range(1, 11))
    assert result_max == [1, 2, 3]
    assert result_all == list(range(1, 11))


@pytest.mark.parametrize(
    "page_result, expected",
    [
        ({"blocks": [{"error": "gemini_rate_limit"}]}, True),
        ({"corrected_text": {"error": "rate_limit_daily"}}, True),
        ({"corrected_text": "RATE_LIMIT_EXCEEDED"}, True),
        ({"blocks": [{"type": "plain text"}], "corrected_text": "ok"}, False),
    ],
)
def test_check_for_rate_limit_errors(page_result, expected):
    pipeline = make_pipeline()

    assert Pipeline._check_for_rate_limit_errors(pipeline, page_result) is expected


@pytest.mark.parametrize(
    "summary, expected",
    [
        ({"pages_data": [{"page_number": 1, "blocks": [], "corrected_text": "ok"}]}, False),
        ({"pages_data": [{"page_number": 1, "error": "failed"}]}, True),
        ({"pages_data": [{"page_number": 1, "blocks": [{"error": "bad"}]}]}, True),
        ({"pages_data": [{"page_number": 1, "corrected_text": {"error": "oops"}}]}, True),
        (
            {
                "pages_data": [{"page_number": 1, "corrected_text": "[RATE_LIMIT_EXCEEDED]"}],
                "processing_stopped": False,
            },
            True,
        ),
        (
            {
                "pages_data": [{"page_number": 1, "blocks": []}],
                "processing_stopped": True,
            },
            True,
        ),
    ],
)
def test_check_for_any_errors(summary, expected):
    pipeline = make_pipeline()

    assert Pipeline._check_for_any_errors(pipeline, summary) is expected
