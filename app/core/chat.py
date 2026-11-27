"""The chat service.

Wires parsing, retrieval, ranking and answer composition together. This is
the module the HTTP layer calls, and the module the tests drive directly —
so a test of the answer is a test of the real path, not a stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .answer import (
    Answer,
    Limitation,
    Match,
    compose_matches,
    compose_no_match,
    handover_state,
    local_clock,
)
from .query import ParsedQuery, expand_item_family, looks_like_scene_question, parse
from .store import Store
from .timeutil import DEFAULT_TIMEZONE, DateRange, iso_utc, parse_timestamp, venue_zone


@dataclass
class ChatResult:
    query: ParsedQuery
    answer: Answer
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query.to_dict(),
            "answer": self.answer.to_dict(),
            "latency_ms": round(self.latency_ms, 1),
        }


class ChatService:
    def __init__(
        self,
        store: Store,
        *,
        timezone_name: str = DEFAULT_TIMEZONE,
        now_provider=None,
        media_url_for=None,
        crop_url_for=None,
    ) -> None:
        self.store = store
        self.media_url_for = media_url_for
        self.crop_url_for = crop_url_for
        self.timezone_name = timezone_name
        self._now = now_provider or (lambda: datetime.now(timezone.utc))

    def ask(self, text: str) -> ChatResult:
        import time

        started = time.perf_counter()
        now = self._now()
        parsed = parse(text, now, self.timezone_name)

        if parsed.date_range is None:
            answer = self._needs_date(parsed)
            return ChatResult(parsed, answer, (time.perf_counter() - started) * 1000)

        coverage = self.store.coverage_between(
            parsed.date_range.start_utc, parsed.date_range.end_utc
        )
        limitations = self._limitations_for(parsed.date_range, coverage, parsed)
        scope = self._scope_statement(parsed)
        status, note = handover_state(self.store.handovers())

        if looks_like_scene_question(text):
            answer = self._scene_answer(parsed, coverage, limitations, scope, status, note)
            return ChatResult(parsed, answer, (time.perf_counter() - started) * 1000)

        rows = self.store.search_events(
            start=parsed.date_range.start_utc,
            end=parsed.date_range.end_utc,
            item_terms=expand_item_family(parsed.item_terms),
            colours=parsed.colours,
        )

        if rows:
            matches = self._rank(rows, parsed)
            answer = compose_matches(
                matches,
                scope_statement=scope,
                limitations=limitations,
                handover_status=status,
                handover_note=note,
                timezone_name=self.timezone_name,
            )
        else:
            answer = compose_no_match(
                scope_statement=scope,
                limitations=limitations,
                handover_status=status,
                handover_note=note,
                timezone_name=self.timezone_name,
                footage_present=bool(coverage),
                available_dates=self._available_dates(),
            )
        return ChatResult(parsed, answer, (time.perf_counter() - started) * 1000)

    # -- helpers ---------------------------------------------------------

    def _needs_date(self, parsed: ParsedQuery) -> Answer:
        return Answer(
            answer_mode="insufficient_information",
            headline="Which day should I look at?",
            body=(
                "I did not catch a day in your question. Tell me when you lost it "
                "and I will search that day's footage."
            ),
            clarifying_question="Which day did you lose it?",
        )

    def _scope_statement(self, parsed: ParsedQuery) -> str:
        date_range = parsed.date_range
        assert date_range is not None
        zone = venue_zone(self.timezone_name)
        start = date_range.start_utc.astimezone(zone)
        end = date_range.end_utc.astimezone(zone)
        if date_range.was_relative and (end - start) == timedelta(days=1):
            return (
                f"Searched {start.strftime('%d %b %Y')} "
                f"({date_range.resolved_from or 'that day'}) in {self.timezone_name}."
            )
        if (end - start) == timedelta(days=1) and start.hour == 0 and start.minute == 0:
            return f"Searched the whole of {start.strftime('%d %b %Y')} in {self.timezone_name}."
        return (
            f"Searched {start.strftime('%d %b %Y %H:%M')} to "
            f"{end.strftime('%H:%M')} in {self.timezone_name}."
        )

    def _limitations_for(
        self,
        date_range: DateRange,
        coverage: list[dict],
        parsed: ParsedQuery,
    ) -> list[Limitation]:
        limitations: list[Limitation] = []

        if not coverage:
            limitations.append(Limitation(
                kind="unprocessed_period",
                description=(
                    "No footage from that period has been processed, so nothing "
                    "was searchable."
                ),
                start=date_range.start_utc,
                end=date_range.end_utc,
            ))
            return limitations

        for window in coverage:
            cov = window.get("coverage", {})
            for gap in cov.get("gaps", []):
                limitations.append(Limitation(
                    kind="camera_offline" if gap.get("reason") == "camera_offline" else "coverage_gap",
                    description=(
                        f"The camera was not recording for part of that period "
                        f"({gap.get('reason', 'gap')})."
                    ),
                    start=None,
                    end=None,
                ))

        observed = sum(float(w.get("coverage", {}).get("observed_seconds") or 0) for w in coverage)
        requested = (date_range.end_utc - date_range.start_utc).total_seconds()
        if observed < requested - 60:
            zone = venue_zone(self.timezone_name)
            labels = list(dict.fromkeys(
                f"{parse_timestamp(w['window_start']).astimezone(zone).strftime('%H:%M')}–"
                f"{parse_timestamp(w['window_end']).astimezone(zone).strftime('%H:%M')}"
                for w in coverage
            ))
            spans = ", ".join(labels[:6]) + ("…" if len(labels) > 6 else "")
            minutes = observed / 60
            amount = f"{minutes:.0f} minutes" if minutes >= 2 else f"{observed:.0f} seconds"
            limitations.append(Limitation(
                kind="unprocessed_period",
                description=(
                    f"Only {amount} of footage exist for the time you asked about "
                    f"(within {spans}). "
                    "Anything outside that could not be checked."
                ),
            ))

        if parsed.assumed_date:
            limitations.append(Limitation(
                kind="assumed_date",
                description=(
                    "You did not give a date, so today was searched. Say which "
                    "day if that is wrong."
                ),
            ))
        return limitations

    def _scene_answer(
        self,
        parsed: ParsedQuery,
        coverage: list[dict],
        limitations: list[Limitation],
        scope: str,
        handover_status: str,
        handover_note: str,
    ) -> Answer:
        if not coverage:
            return compose_no_match(
                scope_statement=scope,
                limitations=limitations,
                handover_status=handover_status,
                handover_note=handover_note,
                timezone_name=self.timezone_name,
                footage_present=False,
                available_dates=self._available_dates(),
            )
        summaries = [w["scene_text"] for w in coverage if w.get("scene_text")]
        body = " ".join(summaries[:3]) if summaries else (
            "The footage for that period exists but no scene summary was written."
        )
        if limitations:
            body += " " + " ".join(l.description for l in limitations)
        return Answer(
            answer_mode="matches_found",
            headline=f"Scene summary for {len(coverage)} window(s)",
            body=body,
            limitations=limitations,
            scope_statement=scope,
            handover_status=handover_status,
            handover_note=handover_note,
            timezone_used=self.timezone_name,
        )

    def _available_dates(self) -> list[str]:
        """Local calendar days that have processed footage, for a helpful
        'no footage that day' answer."""
        zone = venue_zone(self.timezone_name)
        days: list[str] = []
        for start, end in self.store.footage_spans():
            if not start:
                continue
            first = parse_timestamp(start).astimezone(zone).date()
            last = (parse_timestamp(end) - timedelta(seconds=1)).astimezone(zone).date()
            day = first
            while day <= last:
                label = day.strftime("%d %B %Y").lstrip("0")
                if label not in days:
                    days.append(label)
                day += timedelta(days=1)
        return days[-8:]

    def _media_for(self, source_id: str) -> str | None:
        if self.media_url_for is None:
            return None
        return self.media_url_for(source_id)

    def _rank(self, rows, parsed: ParsedQuery) -> list[Match]:
        wanted_state = parsed.item_terms or parsed.colours
        scored = []
        for row in rows:
            observation = self.store.observation(row.observation_id) or {}
            attrs = dict(observation.get("attributes") or {})
            score = 0.4
            evidence = self.store.evidence_for(row.evidence_id) if row.evidence_id else None
            available = bool(evidence and evidence.get("availability") == "available")

            if row.state_at_event == "likely_unattended":
                score += 0.2
            elif row.state_at_event == "possibly_unattended":
                score += 0.1
            if parsed.item_terms and attrs.get("item_type") in expand_item_family(parsed.item_terms):
                score += 0.15
            if parsed.features and set(parsed.features) & {
                f.replace("_", " ") for f in attrs.get("visible_features") or []
            }:
                score += 0.1
            if row.onset_certainty == "observed":
                score += 0.1
            if available:
                score += 0.1
            if wanted_state:
                score += 0.1
            if row.is_worn:
                score -= 1.0

            local = (
                row.event_time.astimezone(venue_zone(self.timezone_name)).strftime("%H:%M")
                if row.event_time else "an unknown time"
            )
            note = {
                "likely_unattended": "left unattended",
                "possibly_unattended": "possibly left unattended",
                "moved": "picked up or moved",
            }.get(row.state_at_event, "seen but not confirmed unattended")
            if attrs.get("item_type") is None:
                note += ", item type unknown"
            seek = evidence.get("seek_to_offset_s") if evidence else None
            media = self._media_for(row.source_id) if available else None
            caption = observation.get("caption_text")
            crop = self.crop_url_for(row.observation_id) if self.crop_url_for else None
            scored.append((score, row, local, note, available, attrs, media, seek, caption, crop))

        scored.sort(key=lambda item: (-item[0], item[2]))
        matches: list[Match] = []
        for rank, (score, row, local, note, available, attrs, media, seek, caption, crop) in enumerate(scored, start=1):
            if score <= 0:
                continue
            matches.append(Match(
                event_id=row.event_id,
                observation_id=row.observation_id,
                evidence_id=row.evidence_id,
                rank=rank,
                score=score,
                item_type=attrs.get("item_type"),
                colour=attrs.get("primary_color"),
                state=row.state_at_event,
                onset_certainty=row.onset_certainty,
                event_time=row.event_time,
                event_offset_s=row.event_offset_s,
                location_label=attrs.get("location_label"),
                difference_note=note,
                evidence_available=available,
                local_time=local,
                media_url=media,
                seek_to_s=seek if seek is not None else row.event_offset_s,
                caption=caption,
                crop_url=crop,
            ))
        return matches
