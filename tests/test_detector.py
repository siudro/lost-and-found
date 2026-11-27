"""Detector accuracy on clips with an exact answer.

Each scene in ``tests/scenes.py`` is rendered to a real MP4, decoded by
ffmpeg and run through the full pipeline. The assertions are the ground
truth of the scene, with tolerances of a couple of seconds.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.pipeline import SourceSpec, ffmpeg_available, run_pipeline
from tests import scenes
from tests import validate_fixtures as vf

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")

START = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)


def _run(tmp_path, scene):
    clip = scene(tmp_path / f"{scene.__name__}.mp4")
    source = SourceSpec(
        source_id=f"src_{scene.__name__}",
        camera_id="cam_test",
        venue_id="venue_test",
        kind="replay",
        clock="source_clock",
        location_label="test lobby",
        recording_started_at=START,
        media_path=str(clip),
    )
    results = run_pipeline(source)
    data = {
        "video_sources.json": [source.to_record()],
        "report_windows.json": [r.window for r in results],
        "observations.json": [o for r in results for o in r.observations],
        "item_events.json": [e for r in results for e in r.events],
        "evidence.json": [d for r in results for d in r.evidence],
        "search_results.json": [],
        "handover_records.json": [],
    }
    checker = vf.Checker()
    vf.schema_checks(checker, data)
    vf.referential_checks(checker, data)
    vf.rule_checks(checker, data)
    assert not checker.failures, checker.failures
    return data


def _seconds(stamp: str) -> float:
    return (vf.parse_ts(stamp) - START).total_seconds()


def test_left_behind_is_likely_unattended(tmp_path):
    data = _run(tmp_path, scenes.left_behind)
    (obs,) = data["observations.json"]
    assert obs["state"] == "likely_unattended"
    assert obs["is_worn"] is False
    assert obs["attributes"]["item_type"] is None  # never guessed
    assert obs["attributes"]["primary_color"] == "gray"
    assert abs(_seconds(obs["first_seen_at"]) - 6) <= 2       # put down at 6 s
    onset = _seconds(obs["unattended_onset_estimate"])
    assert 8 <= onset <= 15                                    # owner gone 9–14 s
    # Three timestamps stay distinct and ordered.
    assert _seconds(obs["first_seen_at"]) <= onset < _seconds(obs["confirmed_at"])
    (event,) = data["item_events.json"]
    assert event["event_type"] == "unattended_onset_estimated"
    assert event["event_time"] == obs["unattended_onset_estimate"]  # not the confirmation


def test_attended_item_is_not_reported_unattended(tmp_path):
    data = _run(tmp_path, scenes.attended)
    states = {o["state"] for o in data["observations.json"]}
    assert not states & {"possibly_unattended", "likely_unattended"}


def test_pickup_is_reported_with_the_time_alone_before_it(tmp_path):
    data = _run(tmp_path, scenes.picked_up)
    (obs,) = data["observations.json"]
    assert obs["state"] == "moved"
    kinds = {e["event_type"]: e for e in data["item_events.json"]}
    assert "picked_up" in kinds
    assert abs(_seconds(kinds["picked_up"]["event_time"]) - 40) <= 3   # collected at 40 s
    assert kinds["unattended_onset_estimated"]["state_at_event"] == "possibly_unattended"


def test_people_walking_past_leave_nothing(tmp_path):
    data = _run(tmp_path, scenes.passers_by)
    assert data["observations.json"] == []
    assert data["item_events.json"] == []
    (window,) = data["report_windows.json"]
    assert window["coverage"]["status"] == "partial"
    assert window["coverage"]["observed_seconds"] == pytest.approx(25, abs=1)


def test_people_in_shot_at_the_start_do_not_become_items(tmp_path):
    data = _run(tmp_path, scenes.busy_start)
    (obs,) = data["observations.json"]           # only the real bag
    assert obs["attributes"]["primary_color"] == "black"
    assert abs(_seconds(obs["first_seen_at"]) - 12) <= 2
    assert obs["state"] in {"possibly_unattended", "likely_unattended"}


def test_light_changing_on_the_floor_is_not_an_item(tmp_path):
    data = _run(tmp_path, scenes.sunlight)
    (obs,) = data["observations.json"]           # only the bag
    assert obs["attributes"]["primary_color"] == "black"
    assert abs(_seconds(obs["first_seen_at"]) - 10) <= 2


def test_ignore_zone_hides_items_only_inside_it(tmp_path):
    from app.core.detector import DetectorConfig, detect_left_items

    clip = scenes.left_behind(tmp_path / "zone.mp4")
    over_bag = (((0.4, 0.6), (0.9, 0.6), (0.9, 1.0), (0.4, 1.0)),)
    elsewhere = (((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)),)
    assert detect_left_items(clip, DetectorConfig(ignore_polygons=over_bag)).tracks == []
    assert len(detect_left_items(clip, DetectorConfig(ignore_polygons=elsewhere)).tracks) == 1


def test_simulated_classroom_jacket_is_the_only_item(tmp_path):
    """The demo clip (shortened to 4 minutes): a class sits through a lesson,
    one student hangs a gray jacket on a chair and leaves it at the bell.
    Seated students must not be reported; the jacket must."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "tools" / "make_classroom_clip.py"
    spec = importlib.util.spec_from_file_location("make_classroom_clip", script)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    clip = tmp_path / "classroom.mp4"
    story = module.render(str(clip), minutes=4)

    source = SourceSpec(
        source_id="src_sim_classroom", camera_id="cam_test", venue_id="venue_test", kind="replay",
        clock="source_clock", location_label="simulated classroom", recording_started_at=START,
        media_path=str(clip),
    )
    results = run_pipeline(source)
    observations = [o for r in results for o in r.observations]
    assert len(observations) == 1, [o["caption_text"] for o in observations]
    (jacket,) = observations
    assert jacket["attributes"]["primary_color"] == "gray"
    assert jacket["state"] == "likely_unattended"
    assert abs(_seconds(jacket["first_seen_at"]) - story["jacket_placed_s"]) <= 6
    assert _seconds(jacket["unattended_onset_estimate"]) >= story["owner_leaves_s"]
