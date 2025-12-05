"""Tests for pipeline constants."""

from __future__ import annotations

import pytest


class TestConstantsImport:
    """Tests for constants module import."""

    def test_rate_limiting_constants(self):
        """Test rate limiting constants are defined."""
        from pipeline.constants import REQUEST_WINDOW_SECONDS
        
        assert isinstance(REQUEST_WINDOW_SECONDS, int)
        assert REQUEST_WINDOW_SECONDS > 0

    def test_layoutreader_constants(self):
        """Test LayoutReader constants are defined."""
        from pipeline.constants import LAYOUTREADER_SCALE
        
        assert isinstance(LAYOUTREADER_SCALE, int)
        assert LAYOUTREADER_SCALE == 1000

    def test_api_token_constants(self):
        """Test API token constants are defined."""
        from pipeline.constants import (
            DEFAULT_MAX_TOKENS,
            SPECIAL_BLOCK_MAX_TOKENS,
            TEXT_CORRECTION_MAX_TOKENS,
            DEFAULT_TEMPERATURE,
            ESTIMATED_IMAGE_TOKENS,
            DEFAULT_ESTIMATED_TOKENS,
        )
        
        assert DEFAULT_MAX_TOKENS > 0
        assert SPECIAL_BLOCK_MAX_TOKENS > 0
        assert TEXT_CORRECTION_MAX_TOKENS > 0
        assert 0.0 <= DEFAULT_TEMPERATURE <= 1.0
        assert ESTIMATED_IMAGE_TOKENS > 0
        assert DEFAULT_ESTIMATED_TOKENS > 0

    def test_detection_constants(self):
        """Test detection constants are defined."""
        from pipeline.constants import (
            DEFAULT_CONFIDENCE_THRESHOLD,
            DEFAULT_OVERLAP_THRESHOLD,
        )
        
        assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 <= DEFAULT_OVERLAP_THRESHOLD <= 1.0

    def test_global_settings(self):
        """Test global settings are defined."""
        from pipeline.constants import MIN_BOX_SIZE, MAX_IMAGE_DIMENSION
        
        assert MIN_BOX_SIZE > 0
        assert MAX_IMAGE_DIMENSION > 0

    def test_text_correction_temperature(self):
        """Test text correction temperature constant."""
        from pipeline.constants import TEXT_CORRECTION_TEMPERATURE
        
        assert TEXT_CORRECTION_TEMPERATURE == 0.0

