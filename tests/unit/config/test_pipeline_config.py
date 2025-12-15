"""Tests for PipelineConfig class.

Tests cover:
- Basic creation and defaults
- Configuration validation
- Backend resolution
- Detector/sorter combination validation
- from_yaml and from_cli methods
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.config import PipelineConfig
from pipeline.exceptions import InvalidConfigError


class TestPipelineConfigCreation:
    """Tests for PipelineConfig creation and defaults."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.detector == "paddleocr-doclayout-v2"
        assert config.recognizer == "paddleocr-vl"
        assert config.sorter is None  # Auto-selected during validation
        assert config.renderer == "markdown"
        assert config.use_cache is True
        assert config.dpi is None  # Resolved during validation
        assert config.use_dual_resolution is False

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = PipelineConfig(
            detector="mineru-vlm",
            recognizer="gemini-2.5-flash",
            dpi=300,
            use_cache=False,
        )

        assert config.detector == "mineru-vlm"
        assert config.recognizer == "gemini-2.5-flash"
        assert config.dpi == 300
        assert config.use_cache is False

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = PipelineConfig(
            cache_dir=Path("/tmp/cache"),
            output_dir=Path("/tmp/output"),
            temp_dir=Path("/tmp/temp"),
        )

        assert isinstance(config.cache_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.temp_dir, Path)
        assert config.cache_dir == Path("/tmp/cache")


class TestPipelineConfigValidation:
    """Tests for PipelineConfig validation."""

    def test_validate_success(self):
        """Test successful validation."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
        )

        # Should not raise
        config.validate()

        # Check resolved values
        assert config.resolved_sorter == "paddleocr-doclayout-v2"
        assert config.resolved_recognizer_backend == "pytorch"

    def test_validate_gemini_backend(self):
        """Test validation with Gemini recognizer."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
        )

        config.validate()

        assert config.resolved_recognizer_backend == "gemini"

    def test_validate_openai_backend(self):
        """Test validation with OpenAI recognizer."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="gpt-4o",
        )

        config.validate()

        assert config.resolved_recognizer_backend == "openai"

    def test_validate_invalid_renderer(self):
        """Test validation with invalid renderer."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
            renderer="html",  # Invalid
        )

        with pytest.raises((ValueError, InvalidConfigError), match="Invalid renderer"):
            config.validate()

    def test_dpi_resolution(self):
        """Test DPI resolution from config."""
        config = PipelineConfig()
        config.validate()

        # DPI should be resolved from config or defaults
        assert config.dpi is not None
        assert config.detection_dpi is not None
        assert config.recognition_dpi is not None


class TestDetectorSorterCombination:
    """Tests for detector/sorter combination validation."""

    def test_auto_select_sorter_for_paddleocr(self):
        """Test auto-selection of sorter for paddleocr-doclayout-v2."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            sorter=None,  # Auto-select
        )

        config.validate()

        assert config.resolved_sorter == "paddleocr-doclayout-v2"

    def test_auto_select_sorter_for_mineru_vlm(self):
        """Test auto-selection of sorter for mineru-vlm detector."""
        config = PipelineConfig(
            detector="mineru-vlm",
            sorter=None,  # Auto-select
            recognizer="paddleocr-vl",
        )

        config.validate()

        assert config.resolved_sorter == "mineru-vlm"

    def test_tightly_coupled_sorter_requires_detector(self):
        """Test that tightly coupled sorter auto-selects detector."""
        config = PipelineConfig(
            detector="doclayout-yolo",  # Default, can be overridden
            sorter="paddleocr-doclayout-v2",  # Requires paddleocr detector
            recognizer="paddleocr-vl",
        )

        config.validate()

        # Detector should be auto-selected to paddleocr-doclayout-v2
        assert config.detector == "paddleocr-doclayout-v2"

    def test_incompatible_combination_raises(self):
        """Test that incompatible detector/sorter combination raises error."""
        config = PipelineConfig(
            detector="mineru-vlm",  # Explicitly set
            sorter="paddleocr-doclayout-v2",  # Requires paddleocr detector
            recognizer="paddleocr-vl",
        )

        with pytest.raises((ValueError, InvalidConfigError), match="requires detector"):
            config.validate()


class TestPipelineConfigFromCLI:
    """Tests for PipelineConfig.from_cli method."""

    def test_from_cli_basic(self):
        """Test creating config from CLI arguments."""
        args = Mock()
        args.detector = "paddleocr-doclayout-v2"
        args.recognizer = "gemini-2.5-flash"
        args.output = "/tmp/output"
        args.no_cache = True
        args.dpi = 300

        # Set other attrs to None or missing
        args.detector_backend = None
        args.detector_model_path = None
        args.confidence = None
        args.auto_batch_size = False
        args.batch_size = None
        args.target_memory_fraction = None
        args.sorter = None
        args.sorter_backend = None
        args.sorter_model_path = None
        args.recognizer_backend = None
        args.gemini_tier = None
        args.cache_dir = None
        args.temp_dir = None
        args.renderer = None
        args.detection_dpi = None
        args.recognition_dpi = None
        args.use_dual_resolution = False

        config = PipelineConfig.from_cli(args)

        assert config.detector == "paddleocr-doclayout-v2"
        assert config.recognizer == "gemini-2.5-flash"
        assert config.output_dir == Path("/tmp/output")
        assert config.use_cache is False
        assert config.dpi == 300


class TestPipelineConfigToDict:
    """Tests for PipelineConfig.to_dict method."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
            dpi=200,
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["detector"] == "paddleocr-doclayout-v2"
        assert result["recognizer"] == "paddleocr-vl"
        assert result["dpi"] == 200
        assert "cache_dir" in result


class TestResolvedProperties:
    """Tests for resolved property accessors."""

    def test_resolved_properties_before_validation(self):
        """Test that accessing resolved properties before validation raises."""
        config = PipelineConfig()

        with pytest.raises(RuntimeError, match="not validated"):
            _ = config.resolved_sorter

        with pytest.raises(RuntimeError, match="not validated"):
            _ = config.resolved_recognizer_backend

    def test_resolved_properties_after_validation(self):
        """Test resolved properties after validation."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
        )
        config.validate()

        assert config.resolved_sorter == "paddleocr-doclayout-v2"
        assert config.resolved_recognizer_backend == "pytorch"
        assert config.resolved_detector_backend is None  # paddleocr doesn't use backend


