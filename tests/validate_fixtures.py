"""Validate the Lost and Found shared interface.

Two layers of checking:

1. Schema conformance — every fixture is validated against its JSON Schema.
2. Cross-record invariants — the rules that a schema alone cannot express,
   because they relate records to each other. These encode the project's
   hard constraints. Each one is here because breaking it would let the
   system make a claim the footage does not support.

Run: python tests/validate_fixtures.py
Exit code 0 means every check passed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
FIXTURES = ROOT / "fixtures"

EVENT_SCHEMA_FOR = {
    "video_sources.json": "video_source.schema.json",
    "observations.json": "observation.schema.json",
    "item_events.json": "item_event.schema.json",
    "report_windows.json": "report_window.schema.json",
    "evidence.json": "evidence.schema.json",
    "search_results.json": "search_result.schema.json",
    "handover_records.json": "handover_record.schema.json",
}

UNATTENDED_STATES = {"possibly_unattended", "likely_unattended"}


def load(name: str):
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes += 1
        else:
            self.failures.append(message)

    def report(self) -> int:
        for failure in self.failures:
            print(f"FAIL  {failure}")
        print(f"\n{self.passes} checks passed, {len(self.failures)} failed")
        return 0 if not self.failures else 1


def schema_checks(c: Checker, data: dict[str, list]) -> None:
    for fixture, schema_name in EVENT_SCHEMA_FOR.items():
        schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        for index, record in enumerate(data[fixture]):
            label = record.get("event_id") or record.get("observation_id") or record.get(
                "window_id"
            ) or record.get("evidence_id") or record.get("query_id") or record.get(
                "source_id"
            ) or record.get("handover_id") or f"{fixture}[{index}]"
            for error in validator.iter_errors(record):
                path = "/".join(str(p) for p in error.absolute_path) or "<root>"
                c.check(False, f"{schema_name} · {label} · {path}: {error.message}")
            c.check(True, f"schema ok: {label}")


def index_by(records: list, key: str) -> dict:
    return {record[key]: record for record in records}


def referential_checks(c: Checker, data: dict[str, list]) -> None:
    sources = index_by(data["video_sources.json"], "source_id")
    windows = index_by(data["report_windows.json"], "window_id")
    observations = index_by(data["observations.json"], "observation_id")
    events = index_by(data["item_events.json"], "event_id")
    evidence = index_by(data["evidence.json"], "evidence_id")

    for obs in data["observations.json"]:
        oid = obs["observation_id"]
        c.check(obs["source_id"] in sources, f"observation {oid} points at an unknown source")
        c.check(
            obs["last_frame_offset_s"] >= obs["first_frame_offset_s"],
            f"observation {oid} ends before it starts",
        )
        if obs["source_id"] in sources and sources[obs["source_id"]]["duration_seconds"]:
            duration = sources[obs["source_id"]]["duration_seconds"]
            c.check(
                obs["last_frame_offset_s"] <= duration,
                f"observation {oid} extends past the end of its source",
            )

    for evt in data["item_events.json"]:
        eid = evt["event_id"]
        c.check(evt["window_id"] in windows, f"event {eid} points at an unknown window")
        c.check(evt["observation_id"] in observations, f"event {eid} points at an unknown observation")
        c.check(evt["evidence_id"] in evidence, f"event {eid} has no evidence record")
        if evt["observation_id"] in observations:
            c.check(
                evt["source_id"] == observations[evt["observation_id"]]["source_id"],
                f"event {eid} claims a different source than its observation",
            )

    for win in data["report_windows.json"]:
        wid = win["window_id"]
        for eid in win["event_ids"]:
            c.check(eid in events, f"window {wid} lists unknown event {eid}")
        for oid in win["observation_ids"]:
            c.check(oid in observations, f"window {wid} lists unknown observation {oid}")
        length = parse_ts(win["window_end"]) - parse_ts(win["window_start"])
        c.check(
            length == timedelta(minutes=10),
            f"window {wid} is {length}, not the required 10 minutes",
        )
        coverage = win["coverage"]
        c.check(
            coverage["observed_seconds"] <= coverage["expected_seconds"],
            f"window {wid} observed more seconds than it expected",
        )
        if coverage["status"] == "full":
            c.check(not coverage["gaps"], f"window {wid} is 'full' but lists gaps")
        else:
            c.check(
                len(coverage["gaps"]) > 0 or coverage["observed_seconds"] < coverage["expected_seconds"],
                f"window {wid} is '{coverage['status']}' but shows no gap or shortfall",
            )
        for gap in coverage["gaps"]:
            c.check(
                parse_ts(gap["end"]) > parse_ts(gap["start"]),
                f"window {wid} has a gap that ends before it starts",
            )

    for evd in data["evidence.json"]:
        ref = evd["evidence_id"]
        c.check(evd["event_id"] in events, f"evidence {ref} supports an unknown event")
        c.check(
            evd["clip_end_offset_s"] > evd["clip_start_offset_s"],
            f"evidence {ref} has a non-positive clip length",
        )
        c.check(
            evd["clip_start_offset_s"] <= evd["seek_to_offset_s"] <= evd["clip_end_offset_s"],
            f"evidence {ref} seeks outside its own excerpt",
        )
        if evd["event_id"] in events:
            c.check(
                abs(evd["seek_to_offset_s"] - events[evd["event_id"]]["event_time_offset_s"]) < 1.0,
                f"evidence {ref} does not seek to its event moment",
            )

    for result in data["search_results.json"]:
        qid = result["query_id"]
        for match in result["matches"]:
            c.check(match["event_id"] in events, f"{qid} returns an unknown event")
            if match.get("evidence_id"):
                c.check(match["evidence_id"] in evidence, f"{qid} returns an unknown evidence record")
        c.check(
            result["answer_mode"] != "insufficient_information"
            or result["clarifying_question"] is not None,
            f"{qid} asks for clarification without providing a question",
        )
        c.check(
            result["answer_mode"] == "insufficient_information"
            or result["clarifying_question"] is None,
            f"{qid} supplies a clarifying question outside insufficient_information",
        )
        if result["answer_mode"] in {"no_match_in_footage_checked", "footage_not_available"}:
            c.check(
                not result["matches"],
                f"{qid} claims no match but returns matches",
            )
        if result["answer_mode"] == "no_match_in_footage_checked":
            c.check(
                len(result["coverage_limitations"]) > 0,
                f"{qid} says 'no match' without stating what could not be searched",
            )
        if result["answer_mode"] == "footage_not_available":
            c.check(
                any(
                    limit["kind"] in {"unprocessed_period", "camera_offline", "footage_expired", "access_denied"}
                    for limit in result["coverage_limitations"]
                ),
                f"{qid} says footage is unavailable without naming a footage problem",
            )
        parsed = result["parsed"]
        c.check(
            parse_ts(parsed["date_range"]["end"]) > parse_ts(parsed["date_range"]["start"]),
            f"{qid} has an empty or inverted date range",
        )
        if parsed["date_range"]["was_relative"]:
            c.check(
                bool(parsed["date_range"]["resolved_from"]),
                f"{qid} resolved a relative date without recording the phrase",
            )

    handovers = data["handover_records.json"]
    for rec in handovers:
        c.check(
            rec["verification"]["status"] == "verified"
            or rec["verification"]["method"] != "staff_confirmed",
            f"handover {rec['handover_id']} claims staff confirmation but is not verified",
        )
        for oid in rec["verification"].get("linked_observation_ids", []):
            c.check(oid in observations, f"handover {rec['handover_id']} links an unknown observation")


def rule_checks(c: Checker, data: dict[str, list]) -> None:
    """The project's hard rules, expressed as failures rather than prose."""

    # A worn jacket is never a lost jacket.
    for obs in data["observations.json"]:
        if obs["is_worn"]:
            c.check(
                obs["state"] not in UNATTENDED_STATES,
                f"observation {obs['observation_id']} is worn but marked {obs['state']}",
            )
    for evt in data["item_events.json"]:
        if evt.get("is_worn"):
            c.check(
                evt["state_at_event"] not in UNATTENDED_STATES,
                f"event {evt['event_id']} is worn but marked unattended",
            )
            c.check(
                evt["event_type"] in {"worn_seen", "left_frame", "reappeared", "first_observed"},
                f"event {evt['event_id']} treats a worn jacket as {evt['event_type']}",
            )

    # An onset that was already in progress at clip start must say so.
    for obs in data["observations.json"]:
        if obs["first_frame_offset_s"] == 0 and not obs["is_worn"]:
            if obs["state"] in UNATTENDED_STATES:
                c.check(
                    obs["onset_certainty"] == "uncertain",
                    f"observation {obs['observation_id']} starts unattended at offset 0 "
                    "without marking onset uncertain",
                )
            c.check(
                "present_at_clip_start" in obs["uncertainty"]
                or obs["state"] == "seen",
                f"observation {obs['observation_id']} was present at clip start "
                "but does not record that uncertainty",
            )

    # Onset, confirmation and first sighting are three different instants.
    for obs in data["observations.json"]:
        if obs["unattended_onset_estimate"] and obs["confirmed_at"]:
            c.check(
                parse_ts(obs["confirmed_at"]) >= parse_ts(obs["unattended_onset_estimate"]),
                f"observation {obs['observation_id']} confirmed unattended before the onset estimate",
            )
        if obs["unattended_onset_estimate"] and obs["first_seen_at"]:
            c.check(
                parse_ts(obs["unattended_onset_estimate"]) >= parse_ts(obs["first_seen_at"]),
                f"observation {obs['observation_id']} is unattended before it was first seen",
            )

    # A worn jacket must never surface as a retrieval candidate.
    events = index_by(data["item_events.json"], "event_id")
    observations = index_by(data["observations.json"], "observation_id")
    for result in data["search_results.json"]:
        for match in result["matches"]:
            obs = observations.get(match["observation_id"])
            if obs:
                c.check(
                    not obs["is_worn"],
                    f"{result['query_id']} returns worn garment {match['observation_id']} as a lost-item match",
                )

    # No pipeline output may express a handover.
    for evt in data["item_events.json"]:
        c.check(
            "handover" not in evt["event_type"],
            f"event {evt['event_id']} is a handover claim, which only intake records may make",
        )

    # Evidence must point at footage that existed.
    sources = index_by(data["video_sources.json"], "source_id")
    for evd in data["evidence.json"]:
        source = sources.get(evd["source_id"])
        if source and source["duration_seconds"]:
            c.check(
                evd["clip_end_offset_s"] <= source["duration_seconds"],
                f"evidence {evd['evidence_id']} runs past the end of its source",
            )


def main() -> int:
    data = {name: load(name) for name in EVENT_SCHEMA_FOR}
    checker = Checker()
    schema_checks(checker, data)
    referential_checks(checker, data)
    rule_checks(checker, data)
    print("Shared interface validation")
    print(f"  sources:    {len(data['video_sources.json'])}")
    print(f"  windows:    {len(data['report_windows.json'])}")
    print(f"  observations: {len(data['observations.json'])}")
    print(f"  events:     {len(data['item_events.json'])}")
    print(f"  evidence:   {len(data['evidence.json'])}")
    print(f"  queries:    {len(data['search_results.json'])}")
    print(f"  handovers:  {len(data['handover_records.json'])}")
    print()
    return checker.report()


if __name__ == "__main__":
    sys.exit(main())
