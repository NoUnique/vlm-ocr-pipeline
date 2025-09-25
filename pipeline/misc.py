"""Miscellaneous helpers for the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo

TimeZoneLike = str | tzinfo

_STATE: dict[str, tzinfo] = {"tz": UTC}


def _coerce_timezone(tz: TimeZoneLike) -> tzinfo:
    if isinstance(tz, str):
        return ZoneInfo(tz)
    return tz


def set_default_timezone(tz: TimeZoneLike) -> None:
    """Set the default timezone used by tz_now."""
    _STATE["tz"] = _coerce_timezone(tz)


def get_default_timezone() -> tzinfo:
    """Return the currently configured default timezone."""
    return _STATE["tz"]


def tz_now() -> datetime:
    """Return the current time in the default timezone."""
    return datetime.now(_STATE["tz"])
