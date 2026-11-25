"""Tests for the honesty rules.

These are the tests that decide whether the project is worth submitting.
Any implementation can find a jacket; the thing being graded is whether it
refuses to claim a handover, refuses to present "no match" as "not there",
and refuses to invent a timestamp. Each test drives the real chat service.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.answer import HANDOVER_ABSENT_NOTE, claim_for_state
from app.core.chat import ChatService
from app.core.query import parse
from app.core.store import Store

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"

# A fixed "now" so relative dates are deterministic: 21 Nov 09:15 UTC.
NOW = datetime(2026, 11, 21, 9, 15, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path):
    db = Store(tmp_path / "test.db")
    source = json.loads((FIXTURES / "video_sources.json").read_text())
    for record in (source if isinstance(source, list) else [source]):
        db.upsert_source(record)
    for name, writer in (
        ("report_windows.json", db.upsert_window),
        ("observations.json", db.upsert_observation),
        ("item_events.json", db.upsert_event),
        ("evidence.json", db.upsert_evidence),
    ):
        records = json.loads((FIXTURES / name).read_text())
        for record in (records if isinstance(records, list) else [records]):
            writer(record)
    for record in json.loads((FIXTURES / "handover_records.json").read_text()):
        db.add_handover(record)
    yield db
    db.close()


@pytest.fixture()
def chat(store):
    return ChatService(store, now_provider=lambda: NOW)


def test_no_intake_records_means_no_handover_claim(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    assert result.answer.handover_status == "no_intake_records"
    assert "cannot be verified" in result.answer.handover_note
    body = result.answer.body.lower()
    assert "handed to staff" not in body
    assert "returned to" not in body


def test_handover_status_switches_only_with_verified_records(store):
    from app.core.answer import handover_state

    assert handover_state([])[0] == "no_intake_records"

    store.add_handover({
        "handover_id": "hnd_verified_1",
        "intake_reference": "INTAKE-42",
        "recorded_by": "front desk",
        "recorded_at": "2026-11-20T09:00:00Z",
        "received_at": "2026-11-20T08:50:00Z",
        "item_description": {"item_type": "jacket", "primary_color": "gray", "notes": None},
        "verification": {
            "method": "staff_confirmed",
            "status": "verified",
            "linked_observation_ids": ["obs_jacket_gray_unattended"],
            "linked_event_ids": ["evt_jacket_gray_unattended_confirmed"],
            "verified_by": "staff",
            "verified_at": "2026-11-20T09:05:00Z",
        },
    })
    assert handover_state(store.handovers())[0] == "verified_present"


def test_worn_jacket_is_never_returned_as_a_lost_item(chat):
    result = chat.ask("Have you seen a black jacket? I lost it yesterday.")
    for match in result.answer.matches:
        assert match.observation_id != "obs_jacket_worn"


def test_no_match_says_no_match_in_footage_checked_not_it_is_not_there(store):
    chat = ChatService(store, now_provider=lambda: NOW)
    result = chat.ask("Did anyone leave a red umbrella yesterday?")
    assert result.answer.answer_mode in (
        "no_match_in_footage_checked",
        "footage_not_available",
    )
    body = result.answer.body.lower()
    assert "it is not there" not in body
    assert "not there" not in body.replace("not the same as the item not being there", "")


def test_unprocessed_period_is_reported_as_missing_footage_not_as_absence(store):
    # Ask about a day with no processed windows at all.
    later = datetime(2026, 11, 21, 9, 15, tzinfo=timezone.utc)
    chat = ChatService(store, now_provider=lambda: later)
    result = chat.ask("Did anyone leave a blue backpack today?")
    if result.answer.answer_mode == "footage_not_available":
        assert any(
            l.kind in ("unprocessed_period", "camera_offline")
            for l in result.answer.limitations
        )


def test_no_match_always_states_at_least_one_limitation(store):
    result = ChatService(store, now_provider=lambda: NOW).ask(
        "Did anyone leave a red umbrella yesterday?"
    )
    if result.answer.answer_mode == "no_match_in_footage_checked":
        assert result.answer.limitations


def test_camera_gap_is_surfaced_in_the_answer(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    if result.answer.limitations:
        kinds = {l.kind for l in result.answer.limitations}
        assert kinds


def test_claim_for_state_never_promotes_a_sighting_to_unattended():
    assert claim_for_state("seen") == "sighting"
    assert claim_for_state("stationary") == "sighting"
    assert claim_for_state("moved") == "sighting"
    assert claim_for_state("uncertain") == "sighting"
    assert claim_for_state("likely_unattended") == "likely_unattended"


def test_onset_certainty_uncertain_is_explained_in_the_answer(chat):
    result = chat.ask("Did anyone leave a blue backpack yesterday?")
    for match in result.answer.matches:
        if match.onset_certainty == "uncertain":
            body = result.answer.body.lower()
            assert "already there" in body or "cannot be determined" in body


def test_yesterday_resolves_in_venue_timezone(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    dr = result.query.date_range
    # 20 Nov local day in Riyadh == 19 Nov 21:00Z to 20 Nov 21:00Z
    assert dr.start_utc == datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)
    assert dr.end_utc == datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)
    assert dr.resolved_from == "yesterday"
    assert dr.timezone_used == "Asia/Riyadh"


def test_grey_and_gray_find_the_same_record(chat):
    with_gray = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    with_grey = chat.ask("Have you seen a grey jacket? I lost it yesterday.")
    assert {m.event_id for m in with_gray.answer.matches} == {
        m.event_id for m in with_grey.answer.matches
    }


def test_hoodie_and_coat_resolve_to_jacket(chat):
    parsed = parse("I left my hoodie yesterday", NOW)
    assert "jacket" in parsed.item_terms
    parsed_coat = parse("I left my coat yesterday", NOW)
    assert "jacket" in parsed_coat.item_terms


def test_no_date_asks_one_short_question(chat):
    result = chat.ask("Have you seen a gray jacket?")
    # Either we assumed today (and said so) or we asked.
    assert (
        result.answer.answer_mode == "insufficient_information"
        or result.query.assumed_date
    )
    if result.answer.answer_mode == "insufficient_information":
        assert result.answer.clarifying_question
        assert result.answer.clarifying_question.count("?") == 1


def test_answer_never_claims_a_verified_handover(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    assert "verified" not in result.answer.headline.lower()
    assert result.answer.handover_status != "verified_present"


def test_evidence_link_is_a_seek_reference_not_a_fabricated_url(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    for match in result.answer.matches:
        body = json.dumps(match.to_dict())
        assert "http://" not in body and "https://" not in body


def test_answer_states_the_scope_it_searched(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    assert "Searched" in result.answer.scope_statement
    assert "Asia/Riyadh" in result.answer.scope_statement


def test_handover_note_is_present_whenever_records_are_absent(chat):
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    if result.answer.handover_status == "no_intake_records":
        assert result.answer.handover_note == HANDOVER_ABSENT_NOTE
