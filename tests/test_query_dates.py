"""Tests for explicit date parsing.

Regression guard: the app once silently fell back to today when a user
wrote "20 November 2026", then said "no footage available" — which is
technically true and completely useless. These pin every format.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.query import find_explicit_day, parse

NOW = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text",
    [
        "lost it on 20 November 2026",
        "lost it on 20 Nov 2026",
        "lost it November 20 2026",
        "lost it on 2026-11-20",
        "lost 20/11/2026",
        "gray jacket 20th Nov 2026",
    ],
)
def test_every_date_format_resolves_to_the_same_local_day(text):
    parsed = parse(text, NOW)
    assert parsed.date_range is not None
    # 20 Nov local in Riyadh is 19 Nov 21:00Z .. 20 Nov 21:00Z
    assert parsed.date_range.start_utc == datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)


def test_explicit_date_is_not_marked_assumed():
    parsed = parse("gray jacket on 20 November 2026", NOW)
    assert parsed.assumed_date is False


def test_explicit_date_beats_a_relative_phrase():
    parsed = parse("20 November 2026, not yesterday", NOW)
    assert parsed.date_range.start_utc == datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)


def test_no_date_is_marked_assumed():
    parsed = parse("have you seen a gray jacket", NOW)
    assert parsed.assumed_date is True


def test_unparseable_month_falls_back_rather_than_crashing():
    assert find_explicit_day("20 Smarch 2026") is None


def test_day_month_year_is_read_in_venue_order():
    # 11/12/2026 in the venue convention is 11 December, not 12 November.
    assert find_explicit_day("11/12/2026") == (2026, 12, 11)
