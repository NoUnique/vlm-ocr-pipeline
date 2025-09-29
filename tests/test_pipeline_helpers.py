import pytest

from pipeline import Pipeline


def make_pipeline():
    return Pipeline.__new__(Pipeline)


def test_compose_page_raw_text_orders_and_preserves_blocks():
    pipeline = make_pipeline()
    regions = [
        {"type": "plain text", "coords": [30, 20, 40, 10], "text": "Second block"},
        {"type": "plain text", "coords": [10, 20, 40, 10], "text": "First block"},
        {"type": "title", "coords": [5, 80, 40, 10], "text": " Title heading\nSubtitle "},
        {"type": "figure", "coords": [0, 0, 10, 10], "text": "Ignored"},
    ]

    result = Pipeline._compose_page_raw_text(pipeline, regions)

    assert result == "First block\n\nSecond block\n\nTitle heading\nSubtitle"


def test_compose_page_raw_text_skips_invalid_entries():
    pipeline = make_pipeline()
    regions = [
        "not-a-dict",
        {"type": "plain text", "coords": [0, 0, 0, 0], "text": ""},
        {"type": "table", "coords": [0, 0, 0, 0], "text": "Table"},
        {"coords": [0, 0, 0, 0], "text": "Missing type"},
    ]

    result = Pipeline._compose_page_raw_text(pipeline, regions)

    assert result == ""


def test_determine_pages_to_process_honors_priority_and_bounds():
    pipeline = make_pipeline()
    total_pages = 10

    result_specific = Pipeline._determine_pages_to_process(
        pipeline, total_pages, pages=[5, 1, 12, 0]
    )
    result_range = Pipeline._determine_pages_to_process(
        pipeline, total_pages, page_range=(0, 12)
    )
    result_max = Pipeline._determine_pages_to_process(
        pipeline, total_pages, max_pages=3
    )
    result_all = Pipeline._determine_pages_to_process(pipeline, total_pages)

    assert result_specific == [1, 5]
    assert result_range == list(range(1, 11))
    assert result_max == [1, 2, 3]
    assert result_all == list(range(1, 11))


@pytest.mark.parametrize(
    "page_result, expected",
    [
        ({"regions": [{"error": "gemini_rate_limit"}]}, True),
        ({"corrected_text": {"error": "rate_limit_daily"}}, True),
        ({"corrected_text": "RATE_LIMIT_EXCEEDED"}, True),
        ({"regions": [{"type": "plain text"}], "corrected_text": "ok"}, False),
    ],
)
def test_check_for_rate_limit_errors(page_result, expected):
    pipeline = make_pipeline()

    assert Pipeline._check_for_rate_limit_errors(pipeline, page_result) is expected


@pytest.mark.parametrize(
    "summary, expected",
    [
        ({"pages_data": [{"page_number": 1, "regions": [], "corrected_text": "ok"}]}, False),
        ({"pages_data": [{"page_number": 1, "error": "failed"}]}, True),
        ({"pages_data": [{"page_number": 1, "regions": [{"error": "bad"}]}]}, True),
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
                "pages_data": [{"page_number": 1, "regions": []}],
                "processing_stopped": True,
            },
            True,
        ),
    ],
)
def test_check_for_any_errors(summary, expected):
    pipeline = make_pipeline()

    assert Pipeline._check_for_any_errors(pipeline, summary) is expected
