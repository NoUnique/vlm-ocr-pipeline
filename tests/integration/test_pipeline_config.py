"""Integration tests for PipelineConfig.

Tests verify configuration loading and validation across components.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import PipelineConfig


class TestPipelineConfigValidation:
    """Tests for PipelineConfig validation."""

    def test_config_default_values(self):
        """Test PipelineConfig has correct default values."""
        config = PipelineConfig()
        config.validate()

        assert config.detector == "paddleocr-doclayout-v2"
        assert config.recognizer == "paddleocr-vl"
        assert config.renderer == "markdown"
        assert config.use_cache is True

    def test_config_dpi_resolution(self):
        """Test DPI values are correctly resolved from settings."""
        config = PipelineConfig()
        config.validate()

        # DPI should be resolved from settings/config.yaml or defaults
        assert config.dpi > 0
        assert config.detection_dpi > 0
        assert config.recognition_dpi > 0

    def test_config_backend_resolution(self):
        """Test backend resolution for different recognizers."""
        # PaddleOCR-VL should get pytorch backend
        config_paddle = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
        )
        config_paddle.validate()
        assert config_paddle.resolved_recognizer_backend in ["pytorch", "vllm", "sglang"]

        # Gemini should get gemini backend
        config_gemini = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
        )
        config_gemini.validate()
        assert config_gemini.resolved_recognizer_backend == "gemini"

    def test_config_invalid_renderer(self):
        """Test invalid renderer raises error."""
        with pytest.raises(ValueError, match="Invalid renderer"):
            config = PipelineConfig(
                detector="paddleocr-doclayout-v2",
                recognizer="paddleocr-vl",
                renderer="html",  # Invalid
            )
            config.validate()

    def test_config_path_conversion(self, tmp_path: Path):
        """Test path values are converted to Path objects."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="paddleocr-vl",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            temp_dir=tmp_path / "temp",
        )
        config.validate()

        assert isinstance(config.cache_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.temp_dir, Path)


class TestPipelineConfigGemini:
    """Tests for Gemini-specific configuration."""

    def test_gemini_tier_free(self):
        """Test Gemini with free tier."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
            gemini_tier="free",
        )
        config.validate()

        assert config.gemini_tier == "free"
        assert config.resolved_recognizer_backend == "gemini"

    def test_gemini_tier_paid(self):
        """Test Gemini with paid tier."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
            gemini_tier="paid",
        )
        config.validate()

        assert config.gemini_tier == "paid"

    def test_gemini_async_mode(self):
        """Test Gemini with async mode."""
        config = PipelineConfig(
            detector="paddleocr-doclayout-v2",
            recognizer="gemini-2.5-flash",
            use_async=True,
        )
        config.validate()

        assert config.use_async is True

