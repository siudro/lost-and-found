"""Answer composition.

The rules the chatbot must obey live here rather than in a system prompt,
because a prompt is a request and this is enforcement. The system prompt
explains the rules to the model; this module makes the answer structurally
incapable of breaking them.

Three things it will not do:

1. Say "no match" without saying what it could not check.
2. Say "it is not there" when the footage was missing.
3. Say a handover happened when no intake record exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .timeutil import iso_utc, venue_zone

CLAIM_SIGHTING = "sighting"
CLAIM_LIKELY_UNATTENDED = "likely_unattended"
CLAIM_VERIFIED_HANDOVER = "verified_handover"


@dataclass
class Match:
    event_id: str
    observation_id: str
    evidence_id: str | None
    rank: int
    score: float
    item_type: str | None
    colour: str | None
    state: str
    onset_certainty: str | None
    event_time: datetime | None
    event_offset_s: float
    location_label: str | None
    difference_note: str
    evidence_available: bool
    local_time: str
    media_url: str | None = None
    seek_to_s: float | None = None
    caption: str | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "observation_id": self.observation_id,
            "evidence_id": self.evidence_id,
            "rank": self.rank,
            "score": round(self.score, 2),
            "item_type": self.item_type,
            "colour": self.colour,
            "state": self.state,
            "state_claim": claim_for_state(self.state),
            "onset_certainty": self.onset_certainty,
            "event_time": iso_utc(self.event_time) if self.event_time else None,
            "local_time": self.local_time,
            "event_offset_s": self.event_offset_s,
            "location_label": self.location_label,
            "difference_note": self.difference_note,
            "evidence_available": self.evidence_available,
            "media_url": self.media_url,
            "seek_to_s": self.seek_to_s,
            "caption": self.caption,
        }


def claim_for_state(state: str) -> str:
    """What a state is allowed to be called in an answer.

    Only 'likely_unattended' supports an unattended claim. Everything else
    is a sighting — the footage shows the item, and nothing more.
    """
    if state == "likely_unattended":
        return CLAIM_LIKELY_UNATTENDED
    return CLAIM_SIGHTING


@dataclass
class Limitation:
    kind: str
    description: str
    start: datetime | None = None
    end: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "description": self.description,
            "start": iso_utc(self.start) if self.start else None,
            "end": iso_utc(self.end) if self.end else None,
        }


@dataclass
class Answer:
    answer_mode: str
    headline: str
    body: str
    matches: list[Match] = field(default_factory=list)
    limitations: list[Limitation] = field(default_factory=list)
    scope_statement: str = ""
    clarifying_question: str | None = None
    handover_status: str = "no_intake_records"
    handover_note: str = ""
    timezone_used: str = "Asia/Riyadh"
    lead: str = ""
    """The main sentences only. ``body`` is the full answer; a chat
    interface that shows matches, limitations and the handover note as
    their own blocks can show ``lead`` instead of repeating them."""
    available_dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer_mode": self.answer_mode,
            "headline": self.headline,
            "body": self.body,
            "lead": self.lead or self.body,
            "available_dates": self.available_dates,
            "matches": [m.to_dict() for m in self.matches],
            "coverage_limitations": [l.to_dict() for l in self.limitations],
            "scope_statement": self.scope_statement,
            "clarifying_question": self.clarifying_question,
            "handover": {
                "status": self.handover_status,
                "note": self.handover_note,
            },
            "timezone_used": self.timezone_used,
        }


def local_clock(moment: datetime, timezone_name: str) -> str:
    return moment.astimezone(venue_zone(timezone_name)).strftime("%H:%M")


def local_day_label(moment: datetime, timezone_name: str) -> str:
    return moment.astimezone(venue_zone(timezone_name)).strftime("%d %b %Y")


def compose_matches(
    matches: list[Match],
    *,
    scope_statement: str,
    limitations: list[Limitation],
    handover_status: str,
    handover_note: str,
    timezone_name: str,
) -> Answer:
    """Answer where at least one candidate was found."""
    if not matches:
        raise ValueError("compose_matches called with no matches")

    lines: list[str] = []
    best = matches[0]
    state_word = {
        "likely_unattended": "was left unattended",
        "possibly_unattended": "may have been left unattended (nobody was near it for a while)",
        "moved": "was later picked up or moved",
        "stationary": "was put down with people close by",
    }.get(best.state, "appears in the footage")
    item = best.item_type or "item"
    colour = f"{best.colour} " if best.colour else ""
    where = f" {best.location_label}" if best.location_label else ""
    article = "an" if (colour or item)[0].lower() in "aeiou" else "a"

    lines.append(
        f"Found {len(matches)} possible match"
        f"{'es' if len(matches) > 1 else ''}. The closest is {article} "
        f"{colour}{item}{where} at {best.local_time} that {state_word}."
    )

    if best.item_type is None:
        lines.append(
            "The system can tell an item's colour and where it was, but not what "
            "kind of item it is, so watch the clip to check whether it is yours."
        )

    if best.onset_certainty == "uncertain":
        lines.append(
            "It was already there when the recording started, so when it was "
            "left cannot be determined from this footage."
        )
    elif best.onset_certainty == "inferred":
        lines.append(
            "The unattended reading is inferred from how long it sat still "
            "with no one nearby; the footage cannot show whose it was."
        )

    lead = " ".join(lines)

    if len(matches) > 1:
        others = ", ".join(
            f"{m.local_time} ({m.difference_note})" for m in matches[1:4]
        )
        lines.append(f"Also worth checking: {others}.")

    if limitations:
        lines.append(
            "Caveat: " + " ".join(l.description for l in limitations)
        )

    if handover_status == "no_intake_records":
        lines.append(handover_note)

    return Answer(
        answer_mode="matches_found",
        headline=f"Possible match at {best.local_time}",
        body=" ".join(lines),
        lead=lead,
        matches=matches,
        limitations=limitations,
        scope_statement=scope_statement,
        handover_status=handover_status,
        handover_note=handover_note,
        timezone_used=timezone_name,
    )


def compose_no_match(
    *,
    scope_statement: str,
    limitations: list[Limitation],
    handover_status: str,
    handover_note: str,
    timezone_name: str,
    footage_present: bool,
    available_dates: list[str] | None = None,
) -> Answer:
    """Answer when nothing matched.

    The distinction this function protects: if footage existed and we
    searched it, say "no match in the footage checked". If footage did not
    exist, say that instead — never imply the item was not there.
    """
    if footage_present:
        mode = "no_match_in_footage_checked"
        headline = "No match in the footage checked"
        body = (
            "Nothing matching that description appears in the footage I searched. "
            "That is not the same as the item not being there — it means it was "
            "not visible in what I could check."
        )
    else:
        mode = "footage_not_available"
        headline = "No footage available for that period"
        body = (
            "There is no processed footage covering the time you asked about, so "
            "I cannot tell you whether the item appears. This is a gap in "
            "coverage, not a negative result."
        )

    lead = body
    if limitations:
        body += " " + " ".join(l.description for l in limitations)

    if not footage_present:
        if available_dates:
            extra = (
                " Footage has been processed for: " + ", ".join(available_dates)
                + ". Ask about one of those days, or upload footage for the day you need."
            )
        else:
            extra = " No footage has been uploaded yet. Upload a video to search it."
        body += extra
        lead += extra

    return Answer(
        answer_mode=mode,
        headline=headline,
        body=body,
        lead=lead,
        available_dates=list(available_dates or []) if not footage_present else [],
        matches=[],
        limitations=limitations,
        scope_statement=scope_statement,
        handover_status=handover_status,
        handover_note=handover_note,
        timezone_used=timezone_name,
    )


HANDOVER_ABSENT_NOTE = (
    "No staff intake record exists for this item, so whether it reached the "
    "front desk cannot be verified from the footage. What is shown above is "
    "what the camera recorded, not confirmation of a handover."
)


def handover_state(records: list[dict]) -> tuple[str, str]:
    """Summarise the intake log.

    Empty is the honest and expected default for this demo.
    """
    if not records:
        return "no_intake_records", HANDOVER_ABSENT_NOTE
    verified = [r for r in records if r.get("verification", {}).get("status") == "verified"]
    if verified:
        return "verified_present", (
            f"{len(verified)} verified intake record(s) exist. A handover may be "
            "stated only where a record matches the item."
        )
    return "unverified_present", (
        "Intake records exist but none is verified against the footage, so a "
        "handover cannot be confirmed."
    )
