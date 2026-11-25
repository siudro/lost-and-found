"""Time handling: the venue clock.

Every "yesterday" and "around 4 PM" in this project resolves through here.
Getting this wrong is the quietest way to be wrong about everything
downstream, because a plausible-looking answer about the wrong day is
indistinguishable from a correct one at a glance.

The rules, stated once:

* Storage is UTC. Rendering is venue-local.
* "Yesterday" means the whole local day, not a rolling 24 hours.
* Venue is Asia/Riyadh, UTC+03:00, and it has no daylight saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Asia/Riyadh"

RELATIVE_DAYS = {
    "today": 0,
    "yesterday": -1,
    "the day before yesterday": -2,
}


@dataclass(frozen=True)
class DateRange:
    """An absolute half-open range, plus how it was derived."""

    start_utc: datetime
    end_utc: datetime
    timezone_used: str
    resolved_from: str | None
    was_relative: bool

    def duration(self) -> timedelta:
        return self.end_utc - self.start_utc

    def contains(self, moment: datetime) -> bool:
        return self.start_utc <= moment < self.end_utc

    def to_dict(self) -> dict:
        return {
            "start": iso_utc(self.start_utc),
            "end": iso_utc(self.end_utc),
            "was_relative": self.was_relative,
            "resolved_from": self.resolved_from,
            "timezone_used": self.timezone_used,
        }


def venue_zone(name: str = DEFAULT_TIMEZONE) -> ZoneInfo:
    return ZoneInfo(name)


def iso_utc(moment: datetime) -> str:
    """Serialise to UTC ISO-8601 with a Z suffix, the storage format."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value}")
    return parsed.astimezone(timezone.utc)


def local_day_range(
    day: datetime | str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> DateRange:
    """The half-open UTC range covering one local calendar day."""
    zone = venue_zone(timezone_name)
    if isinstance(day, str):
        local = datetime.fromisoformat(day).replace(tzinfo=zone)
    elif day.tzinfo is None:
        local = day.replace(tzinfo=zone)
    else:
        local = day.astimezone(zone)

    local_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    local_end = local_start + timedelta(days=1)
    return DateRange(
        start_utc=local_start.astimezone(timezone.utc),
        end_utc=local_end.astimezone(timezone.utc),
        timezone_used=timezone_name,
        resolved_from=None,
        was_relative=False,
    )


def resolve_relative(
    phrase: str,
    now: datetime,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> DateRange:
    """Resolve 'yesterday' / 'today' against the venue clock."""
    key = phrase.strip().lower()
    if key not in RELATIVE_DAYS:
        raise ValueError(f"not a supported relative day: {phrase!r}")

    zone = venue_zone(timezone_name)
    local_now = now.astimezone(zone)
    target = local_now + timedelta(days=RELATIVE_DAYS[key])
    base = local_day_range(target, timezone_name)
    return DateRange(
        start_utc=base.start_utc,
        end_utc=base.end_utc,
        timezone_used=timezone_name,
        resolved_from=key,
        was_relative=True,
    )


def resolve_local_hour(
    hour: int,
    day: datetime,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> datetime:
    """Convert '4 PM on some local day' into an absolute UTC instant."""
    if not 0 <= hour <= 23:
        raise ValueError(f"hour out of range: {hour}")
    zone = venue_zone(timezone_name)
    local = day.astimezone(zone).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    return local.astimezone(timezone.utc)


def window_bounds(start: datetime, minutes: int = 10) -> tuple[datetime, datetime]:
    """The 10-minute report window containing an instant, aligned to the hour.

    Alignment matters: windows must tile the footage deterministically so
    reprocessing a clip produces the same window ids and the same records.
    """
    if minutes <= 0:
        raise ValueError("window length must be positive")
    zone = timezone.utc
    moment = start.astimezone(zone)
    floor_minute = (moment.minute // minutes) * minutes
    beginning = moment.replace(minute=floor_minute, second=0, microsecond=0)
    return beginning, beginning + timedelta(minutes=minutes)


def offset_to_absolute(
    media_offset_s: float,
    recording_started_at: datetime | None,
) -> datetime | None:
    """Map a media-relative offset onto the wall clock, when a clock exists.

    Returns None rather than guessing: a source with no recorded start time
    has no absolute time, and inventing one would put a false timestamp in
    front of a user.
    """
    if recording_started_at is None:
        return None
    if media_offset_s < 0:
        raise ValueError("media offset cannot be negative")
    return recording_started_at + timedelta(seconds=media_offset_s)


def absolute_to_offset(
    moment: datetime,
    recording_started_at: datetime | None,
) -> float | None:
    """Inverse of offset_to_absolute, for translating a query into a seek."""
    if recording_started_at is None:
        return None
    delta = moment - recording_started_at
    return delta.total_seconds()
