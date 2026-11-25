"""Unit tests for time handling.

Time is the quietest place to be wrong: a plausible answer about the wrong
day looks exactly like a correct one. These tests pin the venue clock,
the local-day boundary, and the "media offset with no clock" case.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.timeutil import (
    DateRange,
    absolute_to_offset,
    iso_utc,
    local_day_range,
    offset_to_absolute,
    parse_timestamp,
    resolve_local_hour,
    resolve_relative,
    window_bounds,
)

RIYADH = "Asia/Riyadh"


def test_yesterday_is_the_whole_local_day_not_24_hours():
    now = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)  # 12:00 Riyadh
    rng = resolve_relative("yesterday", now, RIYADH)

    # 24 Nov 00:00 Riyadh == 24 Nov 21:00 UTC the previous day
    assert rng.start_utc == datetime(2026, 11, 23, 21, 0, tzinfo=timezone.utc)
    assert rng.end_utc == datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)
    assert rng.duration() == timedelta(days=1)


def test_yesterday_crosses_midnight_correctly_when_asked_after_local_midnight():
    # 00:30 Riyadh on the 25th == 21:30 UTC on the 24th.
    now = datetime(2026, 11, 24, 21, 30, tzinfo=timezone.utc)
    rng = resolve_relative("yesterday", now, RIYADH)
    assert rng.start_utc == datetime(2026, 11, 23, 21, 0, tzinfo=timezone.utc)
    assert rng.end_utc == datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)


def test_relative_resolution_records_what_it_was_resolved_from():
    now = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
    rng = resolve_relative("yesterday", now, RIYADH)
    assert rng.was_relative is True
    assert rng.resolved_from == "yesterday"
    assert rng.timezone_used == RIYADH


def test_unknown_relative_phrase_is_rejected_not_guessed():
    now = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        resolve_relative("last tuesday", now, RIYADH)


def test_explicit_day_range_is_not_marked_relative():
    rng = local_day_range("2026-11-20", RIYADH)
    assert rng.was_relative is False
    assert rng.resolved_from is None
    assert rng.duration() == timedelta(hours=24)


def test_local_hour_converts_to_absolute_utc():
    day = local_day_range("2026-11-20", RIYADH)
    moment = resolve_local_hour(16, day.start_utc, RIYADH)
    # 16:00 Riyadh == 13:00 UTC
    assert moment == datetime(2026, 11, 20, 13, 0, tzinfo=timezone.utc)


def test_window_bounds_snap_to_ten_minute_blocks():
    moment = datetime(2026, 11, 20, 6, 17, 42, tzinfo=timezone.utc)
    start, end = window_bounds(moment, 10)
    assert start == datetime(2026, 11, 20, 6, 10, tzinfo=timezone.utc)
    assert end == datetime(2026, 11, 20, 6, 20, tzinfo=timezone.utc)
    assert end - start == timedelta(minutes=10)


def test_window_bounds_roll_over_the_hour():
    moment = datetime(2026, 11, 20, 6, 59, 59, tzinfo=timezone.utc)
    start, end = window_bounds(moment, 10)
    assert start == datetime(2026, 11, 20, 6, 50, tzinfo=timezone.utc)
    assert end == datetime(2026, 11, 20, 7, 0, tzinfo=timezone.utc)
    assert end.minute == 0


def test_window_bounds_rejects_non_positive_length():
    with pytest.raises(ValueError):
        window_bounds(datetime.now(timezone.utc), 0)


def test_offset_without_a_clock_returns_none_not_a_guess():
    assert offset_to_absolute(120.0, None) is None


def test_offset_maps_to_absolute_time():
    started = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    assert offset_to_absolute(545.0, started) == datetime(
        2026, 11, 20, 6, 9, 5, tzinfo=timezone.utc
    )


def test_negative_offset_is_rejected():
    started = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        offset_to_absolute(-1.0, started)


def test_offset_round_trip():
    started = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    moment = offset_to_absolute(540.0, started)
    assert absolute_to_offset(moment, started) == pytest.approx(540.0)


def test_iso_utc_formats_with_z():
    moment = datetime(2026, 11, 20, 6, 9, 5, tzinfo=timezone.utc)
    assert iso_utc(moment) == "2026-11-20T06:09:05Z"


def test_parse_timestamp_requires_a_timezone():
    assert parse_timestamp("2026-11-20T06:09:05Z") == datetime(
        2026, 11, 20, 6, 9, 5, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError):
        parse_timestamp("2026-11-20T06:09:05")


def test_date_range_contains_is_half_open():
    rng = DateRange(
        start_utc=datetime(2026, 11, 24, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 11, 25, 0, tzinfo=timezone.utc),
        timezone_used=RIYADH,
        resolved_from=None,
        was_relative=False,
    )
    assert rng.contains(datetime(2026, 11, 24, 0, 0, tzinfo=timezone.utc))
    assert not rng.contains(datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc))
