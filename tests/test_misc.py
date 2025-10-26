"""Tests for miscellaneous helpers.

Tests the timezone utilities which handle:
- Timezone coercion (string to tzinfo)
- Default timezone management
- Timezone-aware datetime generation
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo

from pipeline.misc import _coerce_timezone, get_default_timezone, set_default_timezone, tz_now


class TestCoerceTimezone:
    """Tests for _coerce_timezone function."""

    def test_coerce_string_to_timezone(self):
        """Test converting string timezone to ZoneInfo."""
        result = _coerce_timezone("UTC")

        assert isinstance(result, ZoneInfo)
        assert str(result) == "UTC"

    def test_coerce_america_new_york(self):
        """Test converting America/New_York string."""
        result = _coerce_timezone("America/New_York")

        assert isinstance(result, ZoneInfo)
        assert str(result) == "America/New_York"

    def test_coerce_asia_seoul(self):
        """Test converting Asia/Seoul string."""
        result = _coerce_timezone("Asia/Seoul")

        assert isinstance(result, ZoneInfo)
        assert str(result) == "Asia/Seoul"

    def test_coerce_tzinfo_passthrough(self):
        """Test that tzinfo objects are returned unchanged."""
        tz = ZoneInfo("UTC")
        result = _coerce_timezone(tz)

        assert result is tz

    def test_coerce_utc_passthrough(self):
        """Test that UTC object is returned unchanged."""
        result = _coerce_timezone(UTC)

        assert result is UTC


class TestSetDefaultTimezone:
    """Tests for set_default_timezone function."""

    def setup_method(self):
        """Reset timezone to UTC before each test."""
        set_default_timezone(UTC)

    def test_set_timezone_string(self):
        """Test setting timezone from string."""
        set_default_timezone("America/New_York")
        tz = get_default_timezone()

        assert isinstance(tz, ZoneInfo)
        assert str(tz) == "America/New_York"

    def test_set_timezone_tzinfo(self):
        """Test setting timezone from tzinfo object."""
        tz = ZoneInfo("Asia/Tokyo")
        set_default_timezone(tz)
        result = get_default_timezone()

        assert result is tz

    def test_set_timezone_utc(self):
        """Test setting timezone to UTC."""
        set_default_timezone(UTC)
        result = get_default_timezone()

        assert result is UTC

    def test_set_timezone_multiple_times(self):
        """Test changing timezone multiple times."""
        set_default_timezone("America/New_York")
        assert str(get_default_timezone()) == "America/New_York"

        set_default_timezone("Europe/London")
        assert str(get_default_timezone()) == "Europe/London"

        set_default_timezone(UTC)
        assert get_default_timezone() is UTC


class TestGetDefaultTimezone:
    """Tests for get_default_timezone function."""

    def setup_method(self):
        """Reset timezone to UTC before each test."""
        set_default_timezone(UTC)

    def test_get_default_timezone_initial(self):
        """Test getting initial default timezone."""
        tz = get_default_timezone()

        assert isinstance(tz, tzinfo)

    def test_get_default_timezone_after_set(self):
        """Test getting timezone after setting it."""
        set_default_timezone("America/New_York")
        tz = get_default_timezone()

        assert str(tz) == "America/New_York"

    def test_get_default_timezone_returns_same_object(self):
        """Test that get_default_timezone returns the same object."""
        tz1 = get_default_timezone()
        tz2 = get_default_timezone()

        assert tz1 is tz2


class TestTzNow:
    """Tests for tz_now function."""

    def setup_method(self):
        """Reset timezone to UTC before each test."""
        set_default_timezone(UTC)

    def test_tz_now_returns_datetime(self):
        """Test that tz_now returns a datetime object."""
        result = tz_now()

        assert isinstance(result, datetime)

    def test_tz_now_has_timezone(self):
        """Test that returned datetime has timezone info."""
        result = tz_now()

        assert result.tzinfo is not None

    def test_tz_now_utc(self):
        """Test tz_now with UTC timezone."""
        set_default_timezone(UTC)
        result = tz_now()

        assert result.tzinfo is UTC

    def test_tz_now_custom_timezone(self):
        """Test tz_now with custom timezone."""
        set_default_timezone("America/New_York")
        result = tz_now()

        assert str(result.tzinfo) == "America/New_York"

    def test_tz_now_different_timezones(self):
        """Test tz_now returns different timezone info based on setting."""
        set_default_timezone("Asia/Tokyo")
        tokyo_time = tz_now()

        set_default_timezone("Europe/London")
        london_time = tz_now()

        assert str(tokyo_time.tzinfo) == "Asia/Tokyo"
        assert str(london_time.tzinfo) == "Europe/London"

    def test_tz_now_is_current_time(self):
        """Test that tz_now returns approximately current time."""
        before = datetime.now(UTC)
        result = tz_now()
        after = datetime.now(UTC)

        # Convert all to UTC for comparison
        result_utc = result.astimezone(UTC)

        assert before <= result_utc <= after


class TestTimezoneIntegration:
    """Integration tests for timezone utilities."""

    def setup_method(self):
        """Reset timezone to UTC before each test."""
        set_default_timezone(UTC)

    def test_full_timezone_workflow(self):
        """Test complete workflow of setting timezone and getting current time."""
        # Set timezone
        set_default_timezone("America/Los_Angeles")

        # Verify it was set
        tz = get_default_timezone()
        assert str(tz) == "America/Los_Angeles"

        # Get current time
        now = tz_now()
        assert str(now.tzinfo) == "America/Los_Angeles"
        assert isinstance(now, datetime)

    def test_timezone_independence(self):
        """Test that timezone changes don't affect previously created datetimes."""
        set_default_timezone(UTC)
        utc_time = tz_now()

        set_default_timezone("Asia/Seoul")
        seoul_time = tz_now()

        # Original datetime should still have UTC timezone
        assert utc_time.tzinfo is UTC
        # New datetime should have Seoul timezone
        assert str(seoul_time.tzinfo) == "Asia/Seoul"

    def test_timezone_state_persistence(self):
        """Test that timezone setting persists across function calls."""
        set_default_timezone("Europe/Paris")

        # Multiple calls should all use the same timezone
        time1 = tz_now()
        time2 = tz_now()
        time3 = tz_now()

        assert str(time1.tzinfo) == "Europe/Paris"
        assert str(time2.tzinfo) == "Europe/Paris"
        assert str(time3.tzinfo) == "Europe/Paris"

    def test_timezone_reset_to_utc(self):
        """Test resetting timezone back to UTC."""
        # Change to non-UTC
        set_default_timezone("America/Chicago")
        assert str(get_default_timezone()) == "America/Chicago"

        # Reset to UTC
        set_default_timezone(UTC)
        assert get_default_timezone() is UTC

        # Verify tz_now uses UTC
        now = tz_now()
        assert now.tzinfo is UTC
