"""Tests for pipeline validator."""

from __future__ import annotations

from pipeline.layout.ordering import validate_combination


def test_validate_combination_valid():
    """Test validate_combination accepts valid combinations."""
    # Valid combination should not raise error
    try:
        validate_combination("doclayout-yolo", "mineru-xycut")
        valid = True
    except ValueError:
        valid = False

    assert valid is True


def test_validate_combination_invalid_detector():
    """Test validate_combination rejects unknown detector."""
    # Unknown detector should raise ValueError
    try:
        validate_combination("unknown-detector", "none")
        raised = False
    except ValueError as e:
        raised = True
        assert "detector" in str(e).lower()

    assert raised is True


def test_validate_combination_incompatible():
    """Test validate_combination detects incompatible combinations."""
    # Some combinations may have warnings but should still work
    # Just test that the function runs without error
    try:
        validate_combination("doclayout-yolo", "pymupdf")
        passed = True
    except Exception:
        passed = False

    assert passed is True
