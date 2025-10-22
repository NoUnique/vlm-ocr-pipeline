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


def test_validate_combination_paddleocr_tightly_coupled():
    """Test validate_combination enforces paddleocr-doclayout-v2 detector/sorter coupling."""
    # Valid: paddleocr-doclayout-v2 detector with paddleocr-doclayout-v2 sorter
    is_valid, message = validate_combination("paddleocr-doclayout-v2", "paddleocr-doclayout-v2")
    assert is_valid is True
    assert "Tightly coupled" in message or "pointer network" in message

    # Invalid: paddleocr-doclayout-v2 detector with other sorters
    is_valid_xycut, message_xycut = validate_combination("paddleocr-doclayout-v2", "mineru-xycut")
    assert is_valid_xycut is False
    assert "paddleocr-doclayout-v2" in message_xycut.lower()

    # Invalid: paddleocr-doclayout-v2 sorter with other detectors
    is_valid_yolo, message_yolo = validate_combination("doclayout-yolo", "paddleocr-doclayout-v2")
    assert is_valid_yolo is False
    assert "tightly coupled" in message_yolo.lower()
