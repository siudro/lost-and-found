"""HTTP layer tests.

Mounted routes are a seam where a working service can still look broken:
wrong status codes, missing fields, a 500 where a 400 belongs. These drive
the real app object.
"""

from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "http.db"))
    import app.main as main_module

    importlib.reload(main_module)
    return TestClient(main_module.app)


def test_health_reports_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert "checks" in body
    assert isinstance(body["ok"], bool)


def test_readiness_lists_individual_checks(client):
    body = client.get("/api/readiness").json()
    names = {c["name"] for c in body["checks"]}
    assert "python" in names
    assert "schemas" in names
    assert "storage" in names


def test_providers_answer_which_ai_is_needed(client):
    providers = client.get("/api/providers").json()["providers"]
    by_name = {p["name"]: p for p in providers}
    # The baseline detector must never require a key.
    assert by_name["rule_based_detector"]["requirement"] == "none"
    # Every provider must say what happens without it.
    for provider in providers:
        assert provider["without_it"]


def test_chat_endpoint_returns_an_answer(client):
    response = client.post(
        "/api/chat", json={"message": "Have you seen a gray jacket? I lost it yesterday."}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["answer_mode"]
    assert "handover" in body["answer"]
    assert body["answer"]["handover"]["status"] == "no_intake_records"


def test_chat_rejects_an_empty_message(client):
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400


def test_stats_show_demo_data_was_loaded(client):
    counts = client.get("/api/stats").json()["counts"]
    assert counts["events"] > 0
    assert counts["windows"] > 0


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FoundIt" in response.text


def test_ingest_rejects_an_unparseable_timestamp(client):
    response = client.post(
        "/api/ingest",
        files={"file": ("clip.mp4", b"not really a video", "video/mp4")},
        data={"recorded_at": "not-a-time"},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# Footage: upload, background job, playback, search
# --------------------------------------------------------------------------


def _wait(client, job_id, timeout=180):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_upload_is_processed_played_back_and_searchable(client, tmp_path):
    from datetime import datetime, timedelta, timezone

    from app.core.pipeline import ffmpeg_available
    from tests.scenes import left_behind

    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")
    clip = left_behind(tmp_path / "upload.mp4")
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    with clip.open("rb") as handle:
        response = client.post(
            "/api/ingest",
            files={"file": ("upload.mp4", handle, "video/mp4")},
            data={"recorded_at": started.isoformat(), "location_label": "Test hall"},
        )
    assert response.status_code == 200, response.text
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "done", job["error"]
    result = job["result"]
    assert len(result["items"]) == 1
    assert result["items"][0]["state"] == "likely_unattended"

    media = client.get(result["media_url"], headers={"Range": "bytes=0-99"})
    assert media.status_code == 206
    assert media.headers["content-type"] == "video/mp4"

    listed = client.get("/api/footage").json()["footage"]
    assert any(f["source_id"] == result["source_id"] for f in listed)

    answer = client.post("/api/chat", json={"message": "Did anyone leave a gray bag today?"}).json()["answer"]
    assert answer["answer_mode"] == "matches_found"
    assert answer["matches"][0]["media_url"] == result["media_url"]
    assert answer["matches"][0]["seek_to_s"] > 0


def test_media_route_refuses_unknown_and_malformed_names(client):
    assert client.get("/media/src_nothing_here.mp4").status_code == 404
    assert client.get("/media/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/media/src_demo_classroom.txt").status_code == 404


def test_samples_are_listed_with_credit(client):
    body = client.get("/api/samples").json()
    assert "CAVIAR" in body["credit"]
    assert {s["id"] for s in body["samples"]} >= {"left_bag_at_chair", "left_bag_picked_up"}
    assert client.post("/api/samples/not_a_sample").status_code == 404


def test_question_about_a_day_without_footage_names_days_that_have_it(client):
    answer = client.post(
        "/api/chat", json={"message": "Have you seen a gray jacket? I lost it yesterday."}
    ).json()["answer"]
    if answer["answer_mode"] == "footage_not_available":
        assert "20 November 2026" in answer["body"]


def test_footage_can_be_deleted_but_demo_records_cannot(client, tmp_path):
    from datetime import datetime, timedelta, timezone

    from app.core.pipeline import ffmpeg_available
    from tests.scenes import passers_by

    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")
    clip = passers_by(tmp_path / "walk.mp4")
    with clip.open("rb") as handle:
        job_id = client.post(
            "/api/ingest",
            files={"file": ("walk.mp4", handle, "video/mp4")},
            data={"recorded_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()},
        ).json()["job_id"]
    job = _wait(client, job_id)
    assert job["status"] == "done", job["error"]
    source_id = job["result"]["source_id"]
    assert client.get(job["result"]["media_url"]).status_code == 200

    assert client.delete(f"/api/footage/{source_id}").status_code == 200
    assert client.get(job["result"]["media_url"]).status_code == 404
    assert all(f["source_id"] != source_id for f in client.get("/api/footage").json()["footage"])
    assert client.delete("/api/footage/src_demo_classroom").status_code == 400
