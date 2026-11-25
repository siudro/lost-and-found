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
