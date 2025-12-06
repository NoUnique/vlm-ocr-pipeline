"""Tests for bidirectional auto-selection of detector/sorter pairs."""

from __future__ import annotations

import pytest

from pipeline.config import PipelineConfig


class TestSorterAutoSelectsDetector:
    """Tests for sorter auto-selecting the required detector."""

    def test_paddleocr_sorter_requires_detector(self):
        """Test that paddleocr-doclayout-v2 sorter requires matching detector."""
        config = PipelineConfig(
        sorter="paddleocr-doclayout-v2",
            detector="paddleocr-doclayout-v2",  # Must match
            recognizer="gemini-2.5-flash",  # API-based (no model loading)
        )
        config.validate()

        assert config.detector == "paddleocr-doclayout-v2"
        assert config.sorter == "paddleocr-doclayout-v2"

    def test_mineru_vlm_sorter_requires_detector(self):
        """Test that mineru-vlm sorter requires matching detector."""
        config = PipelineConfig(
        sorter="mineru-vlm",
            detector="mineru-vlm",  # Must match
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        assert config.detector == "mineru-vlm"
        assert config.sorter == "mineru-vlm"


class TestDetectorAutoSelectsSorter:
    """Tests for detector auto-selecting the compatible sorter."""

    def test_paddleocr_detector_with_matching_sorter(self):
        """Test paddleocr-doclayout-v2 detector with matching sorter."""
        config = PipelineConfig(
        detector="paddleocr-doclayout-v2",
            sorter="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        assert config.detector == "paddleocr-doclayout-v2"
        assert config.sorter == "paddleocr-doclayout-v2"

    def test_mineru_vlm_detector_with_matching_sorter(self):
        """Test mineru-vlm detector with matching sorter."""
        config = PipelineConfig(
        detector="mineru-vlm",
            sorter="mineru-vlm",
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        assert config.detector == "mineru-vlm"
        assert config.sorter == "mineru-vlm"


class TestIncompatibleCombinations:
    """Tests for incompatible detector/sorter combinations."""

    def test_incompatible_detector_sorter_raises_error(self):
        """Test that incompatible detector/sorter raises validation error.
        
        Note: doclayout-yolo is the 'default' detector, so it would be auto-corrected.
        We use mineru-vlm as an explicitly incompatible detector for paddleocr sorter.
        """
        # paddleocr-doclayout-v2 sorter requires paddleocr-doclayout-v2 detector
        config = PipelineConfig(
            detector="mineru-vlm",  # Wrong detector for paddleocr sorter (not default)
            sorter="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
        )

        with pytest.raises(ValueError, match="(tightly coupled|requires|must be used)"):
            config.validate()

    def test_mineru_vlm_sorter_with_wrong_detector_raises_error(self):
        """Test that mineru-vlm sorter with wrong detector raises error.
        
        Note: doclayout-yolo is treated as 'default', so it would be auto-corrected.
        We use paddleocr-doclayout-v2 as an explicitly incompatible detector.
        """
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",  # Wrong detector for mineru-vlm sorter
            sorter="mineru-vlm",
            recognizer="gemini-2.5-flash",
        )

        with pytest.raises(ValueError, match="(tightly coupled|requires|must be used)"):
            config.validate()


class TestCompatibleCombinations:
    """Tests for compatible detector/sorter combinations."""

    def test_doclayout_yolo_with_xycut(self):
        """Test doclayout-yolo detector with mineru-xycut sorter."""
        config = PipelineConfig(
            detector="doclayout-yolo",
            sorter="mineru-xycut",
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        assert config.detector == "doclayout-yolo"
        assert config.sorter == "mineru-xycut"

    def test_doclayout_yolo_with_layoutreader(self):
        """Test doclayout-yolo detector with mineru-layoutreader sorter."""
        config = PipelineConfig(
            detector="doclayout-yolo",
            sorter="mineru-layoutreader",
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        assert config.detector == "doclayout-yolo"
        assert config.sorter == "mineru-layoutreader"


class TestDefaultConfiguration:
    """Tests for default detector/sorter configuration."""

    def test_default_configuration(self):
        """Test default detector/sorter values."""
        config = PipelineConfig(
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        # Default: paddleocr-doclayout-v2 detector
        assert config.detector == "paddleocr-doclayout-v2"
        # Sorter can be None (auto-selected in Pipeline)
        # or paddleocr-doclayout-v2 if explicitly set

    def test_non_tightly_coupled_sorter_with_any_detector(self):
        """Test that non-tightly-coupled sorters work with any detector."""
        config = PipelineConfig(
            detector="doclayout-yolo",
            sorter="mineru-xycut",  # Not tightly coupled
            recognizer="gemini-2.5-flash",
        )
        config.validate()

        assert config.detector == "doclayout-yolo"
        assert config.sorter == "mineru-xycut"
