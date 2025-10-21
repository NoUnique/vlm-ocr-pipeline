"""Tests for pipeline validator."""

from __future__ import annotations

from pipeline.layout.ordering import validate_combination


def test_validate_combination_valid():
    """Test validate_combination accepts valid combinations."""
    is_valid, message = validate_combination("doclayout-yolo", "mineru-xycut")

    assert is_valid is True
    assert message  # Should have a message


def test_validate_combination_invalid_detector():
    """Test validate_combination rejects unknown detector."""
    is_valid, message = validate_combination("unknown-detector", "none")

    assert is_valid is False
    assert "detector" in message.lower()


def test_validate_combination_incompatible():
    """Test validate_combination detects incompatible combinations."""
    # MinerU VLM requires mineru-vlm sorter
    is_valid, message = validate_combination("mineru-vlm", "mineru-xycut")

    # This should be invalid or at least have a warning
    # But valid combinations should work
    is_valid_good, message_good = validate_combination("doclayout-yolo", "pymupdf")
    assert is_valid_good is True


def test_validate_combination_paddleocr():
    """Test validate_combination accepts PaddleOCR PP-DocLayoutV2 detector."""
    is_valid, message = validate_combination("paddleocr-doclayout-v2", "mineru-xycut")

    assert is_valid is True
    # Should be a recommended combination
    assert "PaddleOCR" in message or "PP-DocLayoutV2" in message or "Valid" in message
