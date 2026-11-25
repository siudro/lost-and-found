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
    """Generate a clip with a stationary object and prove the pipeline reads it.

    This is the genuine ingest path: real H.264, real ffmpeg decode, real
    frame analysis. It does not test the detector's accuracy — only that the
    plumbing carries real pixels all the way to a stored event.
    """
    clip = tmp_path / "stationary.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=6",
            "-vf", "drawbox=x=140:y=80:w=40:h=30:color=red@1:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not synthesise a test clip: {result.stderr[:120]}")

    assert probe_duration(clip) > 5

    source = SourceSpec(
        source_id="src_test_clip",
        camera_id="cam_test",
        venue_id="venue_test",
        kind="replay",
        clock="source_clock",
        location_label="synthetic test scene",
        recording_started_at=datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc),
        media_path=str(clip),
    )
    log: list[str] = []
    results = run_pipeline(source, sampling_seconds=1.0, progress=log.append)

    assert results, "pipeline produced no windows"
    assert any("decoded" in line for line in log)
    first = results[0]
    assert first.window["window_id"].startswith("win_")
    assert first.observations
    assert first.events
    assert first.evidence
    assert first.observations[0]["state"]


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
