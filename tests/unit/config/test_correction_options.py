"""Tests for correction enable/disable options in PipelineConfig."""

from __future__ import annotations

import argparse

from pipeline.config import PipelineConfig


class TestCorrectionConfigDefaults:
    """Tests for correction option defaults."""

    def test_block_correction_disabled_by_default(self):
        """Test that block correction is disabled by default."""
        config = PipelineConfig()
        assert config.enable_block_correction is False

    def test_page_correction_disabled_by_default(self):
        """Test that page correction is disabled by default."""
        config = PipelineConfig()
        assert config.enable_page_correction is False


class TestCorrectionConfigExplicit:
    """Tests for explicit correction option setting."""

    def test_enable_block_correction(self):
        """Test enabling block correction explicitly."""
        config = PipelineConfig(enable_block_correction=True)
        assert config.enable_block_correction is True

    def test_enable_page_correction(self):
        """Test enabling page correction explicitly."""
        config = PipelineConfig(enable_page_correction=True)
        assert config.enable_page_correction is True

    def test_enable_both_corrections(self):
        """Test enabling both corrections."""
        config = PipelineConfig(
            enable_block_correction=True,
            enable_page_correction=True,
        )
        assert config.enable_block_correction is True
        assert config.enable_page_correction is True


class TestCorrectionConfigFromCLI:
    """Tests for correction options from CLI arguments."""

    def test_block_correction_from_cli(self):
        """Test --block-correction CLI argument."""
        args = argparse.Namespace(
            block_correction=True,
            page_correction=False,
            no_cache=False,
        )
        config = PipelineConfig.from_cli(args)
        assert config.enable_block_correction is True
        assert config.enable_page_correction is False

    def test_page_correction_from_cli(self):
        """Test --page-correction CLI argument."""
        args = argparse.Namespace(
            block_correction=False,
            page_correction=True,
            no_cache=False,
        )
        config = PipelineConfig.from_cli(args)
        assert config.enable_block_correction is False
        assert config.enable_page_correction is True

    def test_both_corrections_from_cli(self):
        """Test both correction CLI arguments."""
        args = argparse.Namespace(
            block_correction=True,
            page_correction=True,
            no_cache=False,
        )
        config = PipelineConfig.from_cli(args)
        assert config.enable_block_correction is True
        assert config.enable_page_correction is True

    def test_no_corrections_from_cli(self):
        """Test no correction CLI arguments (defaults)."""
        args = argparse.Namespace(
            block_correction=False,
            page_correction=False,
            no_cache=False,
        )
        config = PipelineConfig.from_cli(args)
        assert config.enable_block_correction is False
        assert config.enable_page_correction is False

