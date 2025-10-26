"""Tests for pipeline constants.

Tests that all constants are defined with correct types and reasonable values.
"""

from __future__ import annotations

import pipeline.constants as const


class TestRateLimitingConstants:
    """Tests for rate limiting constants."""

    def test_request_window_seconds_exists(self):
        """Test REQUEST_WINDOW_SECONDS is defined."""
        assert hasattr(const, "REQUEST_WINDOW_SECONDS")

    def test_request_window_seconds_type(self):
        """Test REQUEST_WINDOW_SECONDS is an integer."""
        assert isinstance(const.REQUEST_WINDOW_SECONDS, int)

    def test_request_window_seconds_value(self):
        """Test REQUEST_WINDOW_SECONDS has reasonable value."""
        assert const.REQUEST_WINDOW_SECONDS > 0
        assert const.REQUEST_WINDOW_SECONDS == 60


class TestLayoutReaderConstants:
    """Tests for LayoutReader constants."""

    def test_layoutreader_scale_exists(self):
        """Test LAYOUTREADER_SCALE is defined."""
        assert hasattr(const, "LAYOUTREADER_SCALE")

    def test_layoutreader_scale_type(self):
        """Test LAYOUTREADER_SCALE is an integer."""
        assert isinstance(const.LAYOUTREADER_SCALE, int)

    def test_layoutreader_scale_value(self):
        """Test LAYOUTREADER_SCALE has reasonable value."""
        assert const.LAYOUTREADER_SCALE > 0
        assert const.LAYOUTREADER_SCALE == 1000


class TestAPITokenConstants:
    """Tests for API token-related constants."""

    def test_default_max_tokens_exists(self):
        """Test DEFAULT_MAX_TOKENS is defined."""
        assert hasattr(const, "DEFAULT_MAX_TOKENS")

    def test_default_max_tokens_type(self):
        """Test DEFAULT_MAX_TOKENS is an integer."""
        assert isinstance(const.DEFAULT_MAX_TOKENS, int)

    def test_default_max_tokens_value(self):
        """Test DEFAULT_MAX_TOKENS has reasonable value."""
        assert const.DEFAULT_MAX_TOKENS > 0
        assert const.DEFAULT_MAX_TOKENS == 2000

    def test_special_block_max_tokens_exists(self):
        """Test SPECIAL_BLOCK_MAX_TOKENS is defined."""
        assert hasattr(const, "SPECIAL_BLOCK_MAX_TOKENS")

    def test_special_block_max_tokens_type(self):
        """Test SPECIAL_BLOCK_MAX_TOKENS is an integer."""
        assert isinstance(const.SPECIAL_BLOCK_MAX_TOKENS, int)

    def test_special_block_max_tokens_value(self):
        """Test SPECIAL_BLOCK_MAX_TOKENS is larger than DEFAULT_MAX_TOKENS."""
        assert const.SPECIAL_BLOCK_MAX_TOKENS > const.DEFAULT_MAX_TOKENS
        assert const.SPECIAL_BLOCK_MAX_TOKENS == 3000

    def test_text_correction_max_tokens_exists(self):
        """Test TEXT_CORRECTION_MAX_TOKENS is defined."""
        assert hasattr(const, "TEXT_CORRECTION_MAX_TOKENS")

    def test_text_correction_max_tokens_type(self):
        """Test TEXT_CORRECTION_MAX_TOKENS is an integer."""
        assert isinstance(const.TEXT_CORRECTION_MAX_TOKENS, int)

    def test_text_correction_max_tokens_value(self):
        """Test TEXT_CORRECTION_MAX_TOKENS is largest token limit."""
        assert const.TEXT_CORRECTION_MAX_TOKENS > const.SPECIAL_BLOCK_MAX_TOKENS
        assert const.TEXT_CORRECTION_MAX_TOKENS == 4000

    def test_default_temperature_exists(self):
        """Test DEFAULT_TEMPERATURE is defined."""
        assert hasattr(const, "DEFAULT_TEMPERATURE")

    def test_default_temperature_type(self):
        """Test DEFAULT_TEMPERATURE is a number."""
        assert isinstance(const.DEFAULT_TEMPERATURE, (int, float))

    def test_default_temperature_value(self):
        """Test DEFAULT_TEMPERATURE is in valid range."""
        assert 0.0 <= const.DEFAULT_TEMPERATURE <= 2.0
        assert const.DEFAULT_TEMPERATURE == 0.1

    def test_estimated_image_tokens_exists(self):
        """Test ESTIMATED_IMAGE_TOKENS is defined."""
        assert hasattr(const, "ESTIMATED_IMAGE_TOKENS")

    def test_estimated_image_tokens_type(self):
        """Test ESTIMATED_IMAGE_TOKENS is an integer."""
        assert isinstance(const.ESTIMATED_IMAGE_TOKENS, int)

    def test_estimated_image_tokens_value(self):
        """Test ESTIMATED_IMAGE_TOKENS has reasonable value."""
        assert const.ESTIMATED_IMAGE_TOKENS > 0
        assert const.ESTIMATED_IMAGE_TOKENS == 2000

    def test_default_estimated_tokens_exists(self):
        """Test DEFAULT_ESTIMATED_TOKENS is defined."""
        assert hasattr(const, "DEFAULT_ESTIMATED_TOKENS")

    def test_default_estimated_tokens_type(self):
        """Test DEFAULT_ESTIMATED_TOKENS is an integer."""
        assert isinstance(const.DEFAULT_ESTIMATED_TOKENS, int)

    def test_default_estimated_tokens_value(self):
        """Test DEFAULT_ESTIMATED_TOKENS has reasonable value."""
        assert const.DEFAULT_ESTIMATED_TOKENS > 0
        assert const.DEFAULT_ESTIMATED_TOKENS == 1000


class TestDetectionConstants:
    """Tests for detection-related constants."""

    def test_default_confidence_threshold_exists(self):
        """Test DEFAULT_CONFIDENCE_THRESHOLD is defined."""
        assert hasattr(const, "DEFAULT_CONFIDENCE_THRESHOLD")

    def test_default_confidence_threshold_type(self):
        """Test DEFAULT_CONFIDENCE_THRESHOLD is a number."""
        assert isinstance(const.DEFAULT_CONFIDENCE_THRESHOLD, (int, float))

    def test_default_confidence_threshold_value(self):
        """Test DEFAULT_CONFIDENCE_THRESHOLD is in valid probability range."""
        assert 0.0 <= const.DEFAULT_CONFIDENCE_THRESHOLD <= 1.0
        assert const.DEFAULT_CONFIDENCE_THRESHOLD == 0.5

    def test_default_overlap_threshold_exists(self):
        """Test DEFAULT_OVERLAP_THRESHOLD is defined."""
        assert hasattr(const, "DEFAULT_OVERLAP_THRESHOLD")

    def test_default_overlap_threshold_type(self):
        """Test DEFAULT_OVERLAP_THRESHOLD is a number."""
        assert isinstance(const.DEFAULT_OVERLAP_THRESHOLD, (int, float))

    def test_default_overlap_threshold_value(self):
        """Test DEFAULT_OVERLAP_THRESHOLD is in valid IoU range."""
        assert 0.0 <= const.DEFAULT_OVERLAP_THRESHOLD <= 1.0
        assert const.DEFAULT_OVERLAP_THRESHOLD == 0.7


class TestGlobalSettings:
    """Tests for global settings constants."""

    def test_min_box_size_exists(self):
        """Test MIN_BOX_SIZE is defined."""
        assert hasattr(const, "MIN_BOX_SIZE")

    def test_min_box_size_type(self):
        """Test MIN_BOX_SIZE is an integer."""
        assert isinstance(const.MIN_BOX_SIZE, int)

    def test_min_box_size_value(self):
        """Test MIN_BOX_SIZE has reasonable value."""
        assert const.MIN_BOX_SIZE > 0
        assert const.MIN_BOX_SIZE == 10


class TestImageProcessingConstants:
    """Tests for image processing constants."""

    def test_max_image_dimension_exists(self):
        """Test MAX_IMAGE_DIMENSION is defined."""
        assert hasattr(const, "MAX_IMAGE_DIMENSION")

    def test_max_image_dimension_type(self):
        """Test MAX_IMAGE_DIMENSION is an integer."""
        assert isinstance(const.MAX_IMAGE_DIMENSION, int)

    def test_max_image_dimension_value(self):
        """Test MAX_IMAGE_DIMENSION has reasonable value."""
        assert const.MAX_IMAGE_DIMENSION > 0
        assert const.MAX_IMAGE_DIMENSION == 1024


class TestTextCorrectionConstants:
    """Tests for text correction constants."""

    def test_text_correction_temperature_exists(self):
        """Test TEXT_CORRECTION_TEMPERATURE is defined."""
        assert hasattr(const, "TEXT_CORRECTION_TEMPERATURE")

    def test_text_correction_temperature_type(self):
        """Test TEXT_CORRECTION_TEMPERATURE is a number."""
        assert isinstance(const.TEXT_CORRECTION_TEMPERATURE, (int, float))

    def test_text_correction_temperature_value(self):
        """Test TEXT_CORRECTION_TEMPERATURE is deterministic."""
        assert const.TEXT_CORRECTION_TEMPERATURE == 0.0


class TestConstantsConsistency:
    """Integration tests for constants consistency."""

    def test_token_limits_ordering(self):
        """Test that token limits are correctly ordered."""
        assert const.DEFAULT_MAX_TOKENS < const.SPECIAL_BLOCK_MAX_TOKENS < const.TEXT_CORRECTION_MAX_TOKENS

    def test_temperature_consistency(self):
        """Test that temperatures are consistent."""
        # Text correction should be fully deterministic
        assert const.TEXT_CORRECTION_TEMPERATURE == 0.0
        # Default temperature should allow some variation
        assert const.DEFAULT_TEMPERATURE > 0.0

    def test_threshold_ranges(self):
        """Test that all thresholds are in valid ranges."""
        assert 0.0 <= const.DEFAULT_CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 <= const.DEFAULT_OVERLAP_THRESHOLD <= 1.0

    def test_all_constants_positive(self):
        """Test that all numeric constants are positive."""
        assert const.REQUEST_WINDOW_SECONDS > 0
        assert const.LAYOUTREADER_SCALE > 0
        assert const.DEFAULT_MAX_TOKENS > 0
        assert const.SPECIAL_BLOCK_MAX_TOKENS > 0
        assert const.TEXT_CORRECTION_MAX_TOKENS > 0
        assert const.ESTIMATED_IMAGE_TOKENS > 0
        assert const.DEFAULT_ESTIMATED_TOKENS > 0
        assert const.MIN_BOX_SIZE > 0
        assert const.MAX_IMAGE_DIMENSION > 0

    def test_temperature_values_valid(self):
        """Test that all temperature values are valid."""
        assert 0.0 <= const.DEFAULT_TEMPERATURE <= 2.0
        assert 0.0 <= const.TEXT_CORRECTION_TEMPERATURE <= 2.0
