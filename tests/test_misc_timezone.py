from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from pipeline.misc import get_default_timezone, set_default_timezone, tz_now


@pytest.fixture(autouse=True)
def reset_timezone():
    set_default_timezone("UTC")
    yield
    set_default_timezone("UTC")


def test_tz_now_reflects_configured_timezone():
    set_default_timezone("Asia/Seoul")

    now = tz_now()

    assert now.utcoffset() == timedelta(hours=9)


def test_set_default_timezone_accepts_zoneinfo_instance():
    tz = ZoneInfo("America/New_York")
    set_default_timezone(tz)

    assert get_default_timezone() is tz
    assert tz_now().tzinfo is tz
