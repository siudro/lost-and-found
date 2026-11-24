"""Prove the validator rejects broken records.

A validator that passes on correct fixtures might simply never fail. These
tests mutate a copy of the good fixtures into each forbidden state and assert
the validator catches it. Each case maps to a rule in the project brief, so a
passing suite means the rule is actually enforced rather than just written
down.

Run: python tests/test_rules_catch_violations.py
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import validate_fixtures  # noqa: E402


def run_with(mutate) -> list[str]:
    """Copy fixtures, apply a mutation, and return the failure messages."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "fixtures"
        shutil.copytree(ROOT / "fixtures", target)
        data = {name: validate_fixtures.load(name) for name in validate_fixtures.EVENT_SCHEMA_FOR}
        mutate(data)
        for name, records in data.items():
            (target / name).write_text(json.dumps(records), encoding="utf-8")

        original = validate_fixtures.FIXTURES
        validate_fixtures.FIXTURES = target
        try:
            loaded = {name: validate_fixtures.load(name) for name in validate_fixtures.EVENT_SCHEMA_FOR}
            checker = validate_fixtures.Checker()
            validate_fixtures.schema_checks(checker, loaded)
            validate_fixtures.referential_checks(checker, loaded)
            validate_fixtures.rule_checks(checker, loaded)
            return checker.failures
        finally:
            validate_fixtures.FIXTURES = original


def find(records: list, key: str, value: str) -> dict:
    return next(r for r in records if r[key] == value)


CASES = []


def case(name: str, must_contain: str):
    def register(fn):
        CASES.append((name, must_contain, fn))
        return fn

    return register


@case("worn jacket marked likely_unattended", "is worn but marked")
def _(data):
    obs = find(data["observations.json"], "observation_id", "obs_jacket_worn")
    obs["state"] = "likely_unattended"


@case("worn event marked as an unattended pickup", "treats a worn jacket as")
def _(data):
    evt = find(data["item_events.json"], "event_id", "evt_jacket_worn_seen")
    evt["event_type"] = "picked_up"


@case("unattended at offset 0 claimed with certain onset", "starts unattended at offset 0")
def _(data):
    obs = find(data["observations.json"], "observation_id", "obs_backpack_blue_present_at_start")
    obs["state"] = "likely_unattended"
    obs["onset_certainty"] = "observed"


@case("unattended confirmed before its own onset", "confirmed unattended before the onset estimate")
def _(data):
    obs = find(data["observations.json"], "observation_id", "obs_jacket_gray_unattended")
    obs["confirmed_at"] = "2026-11-20T06:05:00Z"


@case("event without evidence record", "has no evidence record")
def _(data):
    evt = find(data["item_events.json"], "event_id", "evt_jacket_gray_placed")
    evt["evidence_id"] = "evd_does_not_exist"


@case("evidence that seeks to the wrong moment", "does not seek to its event moment")
def _(data):
    evd = find(data["evidence.json"], "evidence_id", "evd_jacket_gray_placed")
    evd["seek_to_offset_s"] = 560.0


@case("window that is not ten minutes", "not the required 10 minutes")
def _(data):
    win = find(data["report_windows.json"], "window_id", "win_src_demo_classroom_20261120T060000Z")
    win["window_end"] = "2026-11-20T06:15:00Z"


@case("window marked full while listing a gap", "is 'full' but lists gaps")
def _(data):
    win = find(data["report_windows.json"], "window_id", "win_src_demo_classroom_20261120T060000Z")
    win["coverage"]["gaps"] = [
        {"start": "2026-11-20T06:02:00Z", "end": "2026-11-20T06:03:00Z", "reason": "frames_missing"}
    ]


@case("no-match answer hiding unsearched footage", "without stating what could not be searched")
def _(data):
    result = find(data["search_results.json"], "query_id", "qry_scene_entrance_4pm")
    result["coverage_limitations"] = []


@case("footage-unavailable answer that blames the item", "without naming a footage problem")
def _(data):
    result = find(data["search_results.json"], "query_id", "qry_red_umbrella_unprocessed")
    result["coverage_limitations"] = []


@case("clarifying question asked without permission", "supplies a clarifying question")
def _(data):
    result = find(data["search_results.json"], "query_id", "qry_gray_jacket_yesterday")
    result["clarifying_question"] = "Which classroom?"


@case("worn garment returned as a lost-item match", "as a lost-item match")
def _(data):
    result = find(data["search_results.json"], "query_id", "qry_gray_jacket_yesterday")
    result["matches"].append(
        {
            "event_id": "evt_jacket_worn_seen",
            "observation_id": "obs_jacket_worn",
            "evidence_id": "evd_jacket_worn_seen",
            "rank": 3,
            "score": 0.5,
            "match_kind": "text_semantic",
            "state_at_event": "seen",
            "onset_certainty": "not_applicable",
            "difference_note": "a jacket, but being worn",
            "evidence_availability": "available",
        }
    )


@case("evidence clip running past the end of its source", "runs past the end of its source")
def _(data):
    evd = data["evidence.json"][0]
    evd["clip_end_offset_s"] = 99_999.0
    evd["seek_to_offset_s"] = 99_999.0


@case("handover claimed as staff-confirmed but unverified", "claims staff confirmation but is not verified")
def _(data):
    data["handover_records.json"].append(
        {
            "handover_id": "hnd_test_unverified",
            "intake_reference": "INTAKE-0001",
            "recorded_by": "front desk",
            "recorded_at": "2026-11-20T09:00:00Z",
            "received_at": "2026-11-20T08:50:00Z",
            "item_description": {"item_type": "jacket", "primary_color": "gray", "notes": None},
            "verification": {
                "method": "staff_confirmed",
                "status": "probable",
                "linked_observation_ids": ["obs_jacket_gray_unattended"],
                "linked_event_ids": [],
                "verified_by": None,
                "verified_at": None,
            },
        }
    )


@case("event record attempting a handover claim", "which only intake records may make")
def _(data):
    evt = find(data["item_events.json"], "event_id", "evt_jacket_gray_placed")
    evt["event_type"] = "handover_to_staff"


def main() -> int:
    failures = []
    for name, must_contain, mutate in CASES:
        messages = run_with(mutate)
        caught = any(must_contain in message for message in messages)
        if caught:
            print(f"ok    {name}")
        else:
            print(f"MISS  {name} — expected a failure containing {must_contain!r}")
            for message in messages:
                print(f"        got: {message}")
            failures.append(name)

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} forbidden states rejected")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
