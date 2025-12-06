"""Integration tests for detector and sorter combinations.

Tests verify that valid detector/sorter combinations work together correctly.
"""

from __future__ import annotations

from pipeline.config import PipelineConfig
from pipeline.layout.ordering import validate_combination


class TestDetectorSorterCombinations:
    """Tests for detector/sorter combination validation."""

    def test_paddleocr_detector_sorter_combination(self):
        """Test PaddleOCR detector with its sorter."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            sorter="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
        )
        config.validate()

        assert config.detector == "paddleocr-doclayout-v2"
        assert config.sorter == "paddleocr-doclayout-v2"

    def test_mineru_vlm_detector_sorter_combination(self):
        """Test MinerU VLM detector with its sorter."""
        config = PipelineConfig(
            detector="mineru-vlm",
            sorter="mineru-vlm",
            recognizer="paddleocr-vl",
        )
        config.validate()

        assert config.detector == "mineru-vlm"
        assert config.sorter == "mineru-vlm"

    def test_doclayout_yolo_with_xycut_sorter(self):
        """Test DocLayout-YOLO detector with XY-Cut sorter."""
        config = PipelineConfig(
            detector="doclayout-yolo",
            sorter="mineru-xycut",
            recognizer="paddleocr-vl",
        )
        config.validate()

        assert config.detector == "doclayout-yolo"
        assert config.sorter == "mineru-xycut"

    def test_doclayout_yolo_with_layoutreader_sorter(self):
        """Test DocLayout-YOLO detector with LayoutReader sorter."""
        config = PipelineConfig(
            detector="doclayout-yolo",
            sorter="mineru-layoutreader",
            recognizer="paddleocr-vl",
        )
        config.validate()

        assert config.detector == "doclayout-yolo"
        assert config.sorter == "mineru-layoutreader"

    def test_invalid_sorter_with_wrong_detector(self):
        """Test that tightly coupled sorter with wrong detector raises error."""
        is_valid, message = validate_combination("doclayout-yolo", "paddleocr-doclayout-v2")
        assert is_valid is False
        assert "requires" in message.lower() or "must be used" in message.lower()

    def test_sorter_none_with_paddleocr(self):
        """Test that sorter=None is valid (auto-selection happens in Pipeline)."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            sorter=None,  # Auto-selection happens in Pipeline, not config
            recognizer="paddleocr-vl",
        )
        config.validate()
        # Config keeps sorter as None; Pipeline will auto-select
        assert config.sorter is None

    def test_sorter_none_with_mineru_vlm(self):
        """Test that sorter=None is valid with MinerU VLM detector."""
        config = PipelineConfig(
            detector="mineru-vlm",
            sorter=None,  # Auto-selection happens in Pipeline, not config
            recognizer="paddleocr-vl",
        )
        config.validate()
        # Config keeps sorter as None; Pipeline will auto-select
        assert config.sorter is None

    def test_sorter_none_with_doclayout_yolo(self):
        """Test that sorter=None is valid with doclayout-yolo detector."""
        config = PipelineConfig(
            detector="doclayout-yolo",
            sorter=None,  # Auto-selection happens in Pipeline, not config
            recognizer="paddleocr-vl",
        )
        config.validate()
        # Config keeps sorter as None; Pipeline will auto-select
        assert config.sorter is None


class TestValidateCombination:
    """Tests for validate_combination function."""

    def test_validate_valid_combination(self):
        """Test validating a valid combination."""
        is_valid, message = validate_combination("doclayout-yolo", "mineru-xycut")
        assert is_valid is True
        assert "valid" in message.lower() or len(message) > 0

    def test_validate_tightly_coupled_correct(self):
        """Test validating tightly coupled components with correct detector."""
        is_valid, message = validate_combination(
            "paddleocr-doclayout-v2", "paddleocr-doclayout-v2"
        )
        assert is_valid is True

    def test_validate_tightly_coupled_incorrect(self):
        """Test validating tightly coupled components with wrong detector."""
        is_valid, message = validate_combination(
            "doclayout-yolo", "paddleocr-doclayout-v2"
        )
        assert is_valid is False
        assert "requires" in message.lower() or "invalid" in message.lower()

