"""Query understanding: turn a sentence into search filters.

Handles the ways people actually talk about lost items — "grey" and "gray",
"hoodie" and "coat", "yesterday" and "this morning" — and refuses to invent
a time when the user did not give one. The distinction between "the user
said yesterday" and "we assumed yesterday" decides whether the answer is
allowed to claim a date at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .timeutil import (
    DEFAULT_TIMEZONE,
    DateRange,
    local_day_range,
    resolve_local_hour,
    resolve_relative,
)

# Synonyms collapse to one canonical term, so a search for "hoodie" finds a
# record stored as "jacket". Ordering matters: longer phrases first.
ITEM_SYNONYMS: dict[str, str] = {
    "hoodie": "jacket",
    "hooded sweatshirt": "jacket",
    "sweatshirt": "jacket",
    "coat": "jacket",
    "raincoat": "jacket",
    "parka": "jacket",
    "windbreaker": "jacket",
    "rucksack": "backpack",
    "school bag": "backpack",
    "schoolbag": "backpack",
    "bag": "backpack",
    "satchel": "backpack",
    "rucksack ": "backpack",
    "water bottle": "bottle",
    "flask": "bottle",
}

CANONICAL_ITEMS = {
    "jacket",
    "backpack",
    "bottle",
    "umbrella",
    "book",
    "phone",
}

COLOUR_SYNONYMS: dict[str, str] = {
    "grey": "gray",
    "charcoal": "gray",
    "silver": "gray",
    "navy": "navy blue",
    "sky blue": "blue",
    "royal blue": "blue",
    "maroon": "red",
    "burgundy": "red",
    "crimson": "red",
    "mustard": "yellow",
    "cream": "white",
    "ivory": "white",
}

CANONICAL_COLOURS = {
    "black", "white", "gray", "red", "blue", "navy blue", "green",
    "yellow", "orange", "purple", "pink", "brown", "beige",
}

FEATURES = {
    "hood", "zipper", "buttons", "pattern", "logo", "stripes",
    "pocket", "collar", "quilted", "rain shell",
}

RELATIVE_PHRASES = [
    "the day before yesterday",
    "yesterday",
    "today",
]

HOUR_PATTERN = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# "20 November 2026", "20 Nov 2026", "November 20 2026"
DATE_TEXT_PATTERN = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?\s+(\d{4})\b"
    r"|\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})\b",
    re.IGNORECASE,
)

# "2026-11-20" or "20/11/2026" (day/month/year, the venue convention)
DATE_NUMERIC_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b|\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


@dataclass
class ParsedQuery:
    """Structured interpretation of one question."""

    raw_text: str
    item_terms: list[str] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    location_terms: list[str] = field(default_factory=list)
    date_range: DateRange | None = None
    resolution: str = "none"
    assumed_date: bool = False
    hour_hint: int | None = None

    def to_dict(self) -> dict:
        return {
            "item_terms": self.item_terms,
            "colors": self.colours,
            "features": self.features,
            "location_terms": self.location_terms,
            "date_range": self.date_range.to_dict() if self.date_range else None,
            "resolution": self.resolution,
            "assumed_date": self.assumed_date,
            "hour_hint": self.hour_hint,
        }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def expand_items(text: str) -> list[str]:
    """Find every item term, expanding synonyms to canonical form."""
    found: list[str] = []
    for phrase, canonical in sorted(ITEM_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            if canonical not in found:
                found.append(canonical)
    for canonical in CANONICAL_ITEMS:
        if re.search(rf"\b{re.escape(canonical)}\b", text) and canonical not in found:
            found.insert(0, canonical)
    return found


def expand_colours(text: str) -> list[str]:
    found: list[str] = []
    for phrase, canonical in sorted(COLOUR_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", text) and canonical not in found:
            found.append(canonical)
    for canonical in CANONICAL_COLOURS:
        if re.search(rf"\b{re.escape(canonical)}\b", text) and canonical not in found:
            found.insert(0, canonical)
    return found


def find_features(text: str) -> list[str]:
    return [f for f in FEATURES if re.search(rf"\b{re.escape(f)}\b", text)]


def find_hour(text: str) -> int | None:
    """'4 PM' -> 16. Returns None when no hour is stated."""
    match = HOUR_PATTERN.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    meridiem = match.group(3).lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour if 0 <= hour <= 23 else None


def find_explicit_day(text: str) -> tuple[int, int, int] | None:
    """Extract an explicit calendar date, if the user gave one.

    Returns (year, month, day) or None. Deliberately strict: a wrong guess
    here produces a confident answer about the wrong day.
    """
    numeric = DATE_NUMERIC_PATTERN.search(text)
    if numeric:
        if numeric.group(1):
            return int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
        return int(numeric.group(6)), int(numeric.group(5)), int(numeric.group(4))

    textual = DATE_TEXT_PATTERN.search(text)
    if textual:
        if textual.group(1):
            day, month_name, year = textual.group(1), textual.group(2), textual.group(3)
        else:
            month_name, day, year = textual.group(4), textual.group(5), textual.group(6)
        month = MONTHS.get(month_name.lower().rstrip("."))
        if month is None:
            return None
        return int(year), month, int(day)

    return None


def parse(
    text: str,
    now: datetime,
    timezone_name: str = DEFAULT_TIMEZONE,
    *,
    default_to_today: bool = True,
) -> ParsedQuery:
    """Parse a question.

    If the user named no day, we search today's footage but record that the
    day was assumed — the answer must say so rather than presenting today as
    something the user asked for.
    """
    normalised = _normalise(text)
    parsed = ParsedQuery(raw_text=text)
    parsed.item_terms = expand_items(normalised)
    parsed.colours = expand_colours(normalised)
    parsed.features = find_features(normalised)
    parsed.hour_hint = find_hour(normalised)

    explicit = find_explicit_day(normalised)
    if explicit is not None:
        year, month, day = explicit
        try:
            parsed.date_range = local_day_range(f"{year:04d}-{month:02d}-{day:02d}", timezone_name)
            parsed.resolution = "day"
            parsed.date_range = DateRange(
                start_utc=parsed.date_range.start_utc,
                end_utc=parsed.date_range.end_utc,
                timezone_used=timezone_name,
                resolved_from=f"{day} {month} {year}",
                was_relative=False,
            )
        except ValueError:
            parsed.date_range = None
        if parsed.date_range is not None:
            return parsed

    for phrase in RELATIVE_PHRASES:
        if phrase in normalised:
            parsed.date_range = resolve_relative(phrase, now, timezone_name)
            parsed.resolution = "day"
            break

    if parsed.date_range is None and parsed.hour_hint is not None:
        # "around 4 PM" with no day: today, one hour either side.
        base = resolve_relative("today", now, timezone_name)
        centre = resolve_local_hour(parsed.hour_hint, base.start_utc, timezone_name)
        from datetime import timedelta

        parsed.date_range = DateRange(
            start_utc=centre - timedelta(hours=1),
            end_utc=centre + timedelta(hours=1),
            timezone_used=timezone_name,
            resolved_from="unstated day with an hour",
            was_relative=True,
        )
        parsed.resolution = "hour"
        parsed.assumed_date = True
        return parsed

    if parsed.date_range is None:
        if default_to_today:
            parsed.date_range = resolve_relative("today", now, timezone_name)
            parsed.resolution = "day"
            parsed.assumed_date = True
        else:
            parsed.date_range = None

    return parsed


def looks_like_scene_question(text: str) -> bool:
    """'What happened near the entrance?' is not an item search."""
    normalised = _normalise(text)
    scene_starts = ("what happened", "what was going on", "who was there")
    return normalised.startswith(scene_starts)
