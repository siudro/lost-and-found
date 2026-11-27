"""The OpenAI connection: cheap, fact-checked, and never leaks the key.

No real API call is made here: the HTTP transport is replaced.
"""

from __future__ import annotations

import json

import pytest

from app.ai.llm import LLMClient, _facts, _supported_by

FAKE_KEY = "sk-test-not-a-real-key-123"

ANSWER = {
    "headline": "Possible match at 18:31",
    "lead": "Found 2 possible matches. The closest is a gray item at the bottom of the frame at 18:31 that was left unattended.",
    "body": "Found 2 possible matches. ...",
    "matches": [
        {"local_time": "18:31", "colour": "gray", "item_type": None, "state": "likely_unattended",
         "difference_note": "left unattended", "location_label": "Library entrance"},
        {"local_time": "18:40", "colour": "gray", "item_type": None, "state": "moved",
         "difference_note": "picked up or moved", "location_label": "Library entrance"},
    ],
    "coverage_limitations": [{"kind": "unprocessed_period", "description": "Only 3 minutes of footage exist."}],
    "scope_statement": "Searched 27 Nov 2026 (today) in Asia/Riyadh.",
}


class FakeTransport:
    def __init__(self, reply="A gray item was left near the library entrance at 18:31. Watch the clip below."):
        self.reply = reply
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body)})
        return {"choices": [{"message": {"content": self.reply}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 40}}


def make_client(transport, **kw):
    return LLMClient(api_key=FAKE_KEY, transport=transport, **kw)


def test_disabled_without_a_key_and_makes_no_call():
    t = FakeTransport()
    c = LLMClient(api_key=None, transport=t)
    assert c.phrase_answer("gray bag?", ANSWER) is None
    assert c.chat_reply("how are you") is None
    assert t.calls == []


def test_good_reply_is_used_and_request_is_small():
    t = FakeTransport()
    c = make_client(t)
    reply = c.phrase_answer("Did anyone leave a gray bag?", ANSWER)
    assert reply and "18:31" in reply
    body = t.calls[0]["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["max_completion_tokens"] <= 160
    assert "max_tokens" not in body
    assert len(json.dumps(body["messages"])) < 4000  # a short prompt, not the whole database


def test_reply_with_an_invented_time_is_thrown_away():
    c = make_client(FakeTransport("Someone left a gray bag at 09:15."))
    assert c.phrase_answer("gray bag?", ANSWER) is None
    assert c.usage.rejected == 1


def test_reply_with_an_invented_number_is_thrown_away():
    c = make_client(FakeTransport("I found 7 gray items for you."))
    assert c.phrase_answer("gray bag?", ANSWER) is None


def test_reply_claiming_a_handover_is_thrown_away():
    c = make_client(FakeTransport("Good news: it was handed in and is at the front desk."))
    assert c.phrase_answer("gray bag?", ANSWER) is None


def test_chat_reply_may_not_claim_a_sighting():
    c = make_client(FakeTransport("I saw a bag in the footage earlier!"))
    assert c.chat_reply("my name is Sara") is None
    ok = make_client(FakeTransport("Nice to meet you, Sara. What did you lose, and what colour is it?"))
    assert ok.chat_reply("my name is Sara")


def test_identical_questions_are_cached():
    t = FakeTransport()
    c = make_client(t)
    c.phrase_answer("gray bag?", ANSWER)
    c.phrase_answer("Gray bag?", ANSWER)
    assert len(t.calls) == 1
    assert c.usage.cache_hits == 1


def test_daily_limit_stops_calls():
    t = FakeTransport()
    c = make_client(t, daily_call_limit=2)
    for n in range(5):
        c.chat_reply(f"hello number {n}")
    assert len(t.calls) == 2


def test_failures_fall_back_silently_without_details():
    def broken(url, headers, body, timeout):
        raise RuntimeError(f"connection failed {headers['Authorization']}")

    c = make_client(broken)
    assert c.phrase_answer("gray bag?", ANSWER) is None
    assert c.usage.failures == 1


def test_status_never_contains_the_key():
    c = make_client(FakeTransport())
    c.phrase_answer("gray bag?", ANSWER)
    assert FAKE_KEY not in json.dumps(c.status())
    assert c.status()["enabled"] is True


def test_supported_by_accepts_times_from_the_facts_only():
    facts = _facts(ANSWER)
    assert _supported_by("It was there at 18:31 and moved at 18:40.", facts)
    assert not _supported_by("It was there at 18:32.", facts)


# ---------------------------------------------------------------- endpoint


def test_chat_endpoint_uses_llm_reply_and_keeps_body(client_with_llm):
    api, transport = client_with_llm
    data = api.post("/api/chat", json={"message": "Have you seen a gray jacket on 20 November 2026?"}).json()
    answer = data["answer"]
    assert answer["phrased_by"] in {"llm", "template"}
    assert answer["body"].startswith("Found")  # the composed record is untouched
    assert FAKE_KEY not in json.dumps(data)


def test_non_search_chat_goes_to_the_llm(client_with_llm):
    api, transport = client_with_llm
    transport.reply = "Nice to meet you! What did you lose, and what colour is it?"
    data = api.post("/api/chat", json={"message": "my name is Sara"}).json()
    assert data["conversation"]["phrased_by"] == "llm"
    assert "colour" in data["conversation"]["text"]


def test_non_search_chat_without_llm_gets_a_steer(client):
    data = client.post("/api/chat", json={"message": "how are you today"}).json()
    assert data["conversation"]["kind"] in {"steer", "llm"}


def test_llm_status_endpoint_hides_the_key(client_with_llm):
    api, _ = client_with_llm
    body = api.get("/api/llm/status").text
    assert FAKE_KEY not in body


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "llm.db"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import app.main as main_module

    importlib.reload(main_module)
    return TestClient(main_module.app)


@pytest.fixture
def client_with_llm(client, monkeypatch):
    import app.main as main

    transport = FakeTransport("A gray jacket was seen. Watch the clip below.")
    monkeypatch.setattr(main, "LLM", LLMClient(api_key=FAKE_KEY, transport=transport))
    return client, transport
