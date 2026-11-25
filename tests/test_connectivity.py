"""Connectivity tests: do the parts actually talk to each other?

Unit tests prove each piece works in isolation. These prove the seams
hold — ingest writes what search reads, evidence points at a real event,
and the HTTP layer returns the same answer the service does. Seams are
where projects break quietly.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.chat import ChatService
from app.core.pipeline import (
    IngestError,
    SourceSpec,
    ffmpeg_available,
    make_window_id,
    probe_duration,
    run_pipeline,
)
from app.core.store import Store
from app.core.timeutil import window_bounds

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
NOW = datetime(2026, 11, 21, 9, 15, tzinfo=timezone.utc)


@pytest.fixture()
def seeded_store(tmp_path):
    db = Store(tmp_path / "conn.db")
    source = json.loads((FIXTURES / "video_sources.json").read_text())
    for record in (source if isinstance(source, list) else [source]):
        db.upsert_source(record)
    for name, writer in (
        ("report_windows.json", db.upsert_window),
        ("observations.json", db.upsert_observation),
        ("item_events.json", db.upsert_event),
        ("evidence.json", db.upsert_evidence),
    ):
        for record in json.loads((FIXTURES / name).read_text()):
            writer(record)
    yield db
    db.close()


def test_every_event_resolves_to_a_stored_evidence_record(seeded_store):
    for event_payload in [
        json.loads(r["payload"])
        for r in seeded_store._conn.execute("SELECT payload FROM events")
    ]:
        evidence = seeded_store.evidence_for(event_payload["evidence_id"])
        assert evidence is not None, f"event {event_payload['event_id']} has no evidence"
        assert evidence["event_id"] == event_payload["event_id"]


def test_every_event_belongs_to_a_window_that_exists(seeded_store):
    windows = {w["window_id"] for w in seeded_store.coverage_between(
        NOW - timedelta(days=30), NOW + timedelta(days=1)
    )}
    for row in seeded_store._conn.execute("SELECT event_id, window_id FROM events"):
        assert row["window_id"] in windows, f"{row['event_id']} points at a missing window"


def test_search_reads_what_ingest_wrote(seeded_store):
    """The seam that matters most: writer and reader agree on column names."""
    rows = seeded_store.search_events(
        start=datetime(2026, 11, 20, 0, tzinfo=timezone.utc),
        end=datetime(2026, 11, 21, 0, tzinfo=timezone.utc),
    )
    assert rows, "search found nothing after fixtures were ingested"


def test_chat_service_and_store_agree_on_item_names(seeded_store):
    chat = ChatService(seeded_store, now_provider=lambda: NOW)
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    assert result.answer.matches
    first = result.answer.matches[0]
    assert first.event_id


def test_reprocessing_the_same_window_does_not_duplicate(seeded_store):
    windows = json.loads((FIXTURES / "report_windows.json").read_text())
    before = seeded_store.counts()["windows"]
    for window in windows:
        seeded_store.upsert_window(window)
    assert seeded_store.counts()["windows"] == before

    events = json.loads((FIXTURES / "item_events.json").read_text())
    before_events = seeded_store.counts()["events"]
    for event in events:
        seeded_store.upsert_event(event)
    assert seeded_store.counts()["events"] == before_events


def test_upsert_window_reports_a_repeat_as_not_new(seeded_store):
    window = json.loads((FIXTURES / "report_windows.json").read_text())[0]
    assert seeded_store.upsert_window(window) is False


def test_fixture_evidence_availability_reaches_the_answer(seeded_store):
    chat = ChatService(seeded_store, now_provider=lambda: NOW)
    result = chat.ask("Have you seen a gray jacket? I lost it yesterday.")
    assert all(isinstance(m.evidence_available, bool) for m in result.answer.matches)


def test_window_id_is_deterministic_for_the_same_instant():
    moment = datetime(2026, 11, 20, 6, 3, tzinfo=timezone.utc)
    first, _ = window_bounds(moment)
    second, _ = window_bounds(moment + timedelta(minutes=4))
    assert make_window_id("src_demo", first) == make_window_id("src_demo", second)


def test_windows_tile_without_overlap_or_hole():
    start = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    current = start
    for _ in range(6):
        beginning, end = window_bounds(current)
        assert beginning <= current < end
        current = end
    assert current == start + timedelta(hours=1)


# --------------------------------------------------------------------------
# Real video, when ffmpeg is present
# --------------------------------------------------------------------------


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_pipeline_processes_a_real_generated_clip(tmp_path):
    """Real encoded video, real ffmpeg decode, all the way to stored, searchable events.

    The clip has an exact answer: a gray bag put down about 6 s in, owner
    gone by 14 s, bag still there at 60 s. Accuracy across scenarios is in
    test_detector.py; this proves the plumbing carries it to the chat.
    """
    from tests.scenes import left_behind

    clip = left_behind(tmp_path / "left.mp4")
    assert probe_duration(clip) > 50

    started = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    source = SourceSpec(
        source_id="src_test_clip",
        camera_id="cam_test",
        venue_id="venue_test",
        kind="replay",
        clock="source_clock",
        location_label="synthetic test scene",
        recording_started_at=started,
        media_path=str(clip),
    )
    log: list[str] = []
    results = run_pipeline(source, progress=log.append)

    assert results, "pipeline produced no windows"
    assert any("decoded" in line for line in log)
    observations = [o for r in results for o in r.observations]
    assert len(observations) == 1
    assert observations[0]["state"] == "likely_unattended"
    assert observations[0]["attributes"]["primary_color"] == "gray"

    db = Store(tmp_path / "ingest.db")
    db.upsert_source(source.to_record())
    for result in results:
        db.upsert_window(result.window)
        for record in result.observations:
            db.upsert_observation(record)
        for record in result.events:
            db.upsert_event(record)
        for record in result.evidence:
            db.upsert_evidence(record)
    service = ChatService(db, now_provider=lambda: started + timedelta(hours=1))
    answer = service.ask("Have you seen a grey bag today?").answer
    assert answer.answer_mode == "matches_found"
    assert answer.matches[0].colour == "gray"
    assert answer.matches[0].item_type is None
    assert "not what kind of item" in answer.body


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_pipeline_rejects_a_source_without_a_clock(tmp_path):
    clip = tmp_path / "noclk.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=black:s=160x90:d=2", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)],
        capture_output=True, timeout=120,
    )
    source = SourceSpec(
        source_id="src_noclock",
        camera_id="cam_x",
        venue_id="venue_x",
        kind="recorded_upload",
        clock="media_relative_only",
        recording_started_at=None,
        media_path=str(clip),
    )
    with pytest.raises(IngestError, match="no recording_started_at"):
        run_pipeline(source)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_pipeline_rejects_a_missing_file(tmp_path):
    source = SourceSpec(
        source_id="src_missing",
        camera_id="cam_x",
        venue_id="venue_x",
        kind="recorded_upload",
        clock="source_clock",
        recording_started_at=datetime(2026, 11, 20, tzinfo=timezone.utc),
        media_path=str(tmp_path / "nope.mp4"),
    )
    with pytest.raises(IngestError, match="not found"):
        run_pipeline(source)
