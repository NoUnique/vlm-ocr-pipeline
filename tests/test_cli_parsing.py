"""Tests for CLI argument parsing."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_paddleocr_detector_recognized():
    """Test that paddleocr-doclayout-v2 detector is recognized."""
    # Import here to avoid import errors
    from main import _build_argument_parser  # noqa: PLC0415

    parser = _build_argument_parser()

    # Test that paddleocr-doclayout-v2 is a valid detector choice
    test_args = [
        "--input",
        "test.pdf",
        "--detector",
        "paddleocr-doclayout-v2",
    ]

    with patch.object(sys, "argv", ["main.py", *test_args]):
        args = parser.parse_args(test_args)
        assert args.detector == "paddleocr-doclayout-v2"


def test_paddleocr_recognizer_recognized():
    """Test that paddleocr-vl recognizer is recognized."""
    from main import _build_argument_parser  # noqa: PLC0415

    parser = _build_argument_parser()

    # Test that paddleocr-vl is a valid recognizer choice
    test_args = [
        "--input",
        "test.pdf",
        "--recognizer",
        "paddleocr-vl",
    ]

    with patch.object(sys, "argv", ["main.py", *test_args]):
        args = parser.parse_args(test_args)
        assert args.recognizer == "paddleocr-vl"


def test_paddleocr_full_pipeline_args():
    """Test full PaddleOCR pipeline arguments."""
    from main import _build_argument_parser  # noqa: PLC0415

    parser = _build_argument_parser()

    # Test complete PaddleOCR pipeline command
    test_args = [
        "--input",
        "samples/98A-004.pdf",
        "--detector",
        "paddleocr-doclayout-v2",
        "--recognizer",
        "paddleocr-vl",
        "--output",
        "output/",
    ]

    with patch.object(sys, "argv", ["main.py", *test_args]):
        args = parser.parse_args(test_args)
        assert args.input == "samples/98A-004.pdf"
        assert args.detector == "paddleocr-doclayout-v2"
        assert args.recognizer == "paddleocr-vl"
        assert args.output == "output/"


def test_invalid_detector_raises_error():
    """Test that invalid detector raises error."""
    from main import _build_argument_parser  # noqa: PLC0415

    parser = _build_argument_parser()

    test_args = [
        "--input",
        "test.pdf",
        "--detector",
        "invalid-detector",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(test_args)


def test_invalid_recognizer_raises_error():
    """Test that invalid recognizer raises error."""
    from main import _build_argument_parser  # noqa: PLC0415

    parser = _build_argument_parser()

    test_args = [
        "--input",
        "test.pdf",
        "--recognizer",
        "invalid-recognizer",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(test_args)
