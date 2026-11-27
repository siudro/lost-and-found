"""AI captioner, photo search, prompt files and the OpenAI key check.

No real API call is made: the transport is a fake, and the key check uses a
fake opener.
"""

from __future__ import annotations

import importlib
import io
import json
import urllib.error

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ai import captioner, photo, prompts
from app.ai.llm import LLMClient
from app.core.diagnostics import openai_key_status

FAKE_KEY = "sk-test-not-a-real-key-123"


class JsonTransport:
    def __init__(self, reply: dict | str):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(json.loads(body))
        return {"choices": [{"message": {"content": self.reply}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 50}}


GOOD = {"is_item": True, "is_worn": False, "item_type": "hoodie", "primary_color": "gray",
        "secondary_color": None, "visible_features": ["hood", "zipper"], "location_hint": "on a chair",
        "confidence": "high"}


def solid(colour_bgr, size=(120, 160)) -> np.ndarray:
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = colour_bgr
    return img


def png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


# ------------------------------------------------------------------ prompts

def test_every_required_prompt_has_a_version():
    versions = prompts.prompt_versions()
    for name in prompts.REQUIRED:
        assert versions[name].count(".") == 2


def test_prompt_text_has_no_header():
    body = prompts.text("captioner")
    assert not body.startswith("---")
    assert "untrusted" in body.lower() or "never" in body.lower()


# ---------------------------------------------------------------- captioner

def test_clean_drops_values_outside_the_schema():
    out = captioner.clean({"is_item": True, "item_type": "spaceship", "primary_color": "grey-ish",
                           "visible_features": ["hood", "laser", "hood"], "confidence": "sure",
                           "location_hint": "  " })
    assert out["item_type"] is None
    assert out["primary_color"] is None
    assert out["visible_features"] == ["hood"]
    assert out["confidence"] == "low"
    assert out["location_hint"] is None
    assert captioner.clean("not a dict") is None


def test_phrase_reads_naturally():
    assert captioner.phrase(GOOD) == "gray hoodie with a hood, a zipper"


def test_vision_call_is_small_and_structured():
    t = JsonTransport(GOOD)
    llm = LLMClient(api_key=FAKE_KEY, transport=t)
    out = llm.vision_json("sys", "look", [b"\xff\xd8fake"], captioner.description_schema(), cache_key="c1")
    assert out == GOOD
    body = t.calls[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["max_completion_tokens"] <= 220
    images = [p for p in body["messages"][1]["content"] if p["type"] == "image_url"]
    assert images and all(p["image_url"]["detail"] == "low" for p in images)
    # Same images again: answered from the cache, no second call.
    llm.vision_json("sys", "look", [b"\xff\xd8fake"], captioner.description_schema(), cache_key="c1")
    assert len(t.calls) == 1


def test_vision_rejects_broken_json():
    llm = LLMClient(api_key=FAKE_KEY, transport=JsonTransport("not json {"))
    assert llm.vision_json("s", "u", [b"x"], captioner.description_schema(), cache_key="k") is None


def test_no_key_means_no_vision_call():
    t = JsonTransport(GOOD)
    llm = LLMClient(api_key=None, transport=t)
    assert llm.vision_json("s", "u", [b"x"], {}, cache_key="k") is None
    assert t.calls == []


def observation():
    return {"attributes": {"item_type": None, "primary_color": "gray", "secondary_color": None,
                           "visible_features": [], "location_label": "Bottom of the frame"},
            "uncertainty": ["ambiguous_attributes"], "caption_text": "A gray item.",
            "caption_version": "rule_based"}


def test_apply_merges_a_confident_description():
    desc = dict(GOOD, prompt_version="1.0.0")
    obs = captioner.apply(observation(), desc, "gpt-4o-mini")
    assert obs["attributes"]["item_type"] == "hoodie"
    assert obs["attributes"]["visible_features"] == ["hood", "zipper"]
    assert "ambiguous_attributes" not in obs["uncertainty"]
    assert obs["caption_version"] == "openai_vision:gpt-4o-mini:captioner-1.0.0"
    assert "AI description" in obs["caption_text"]
    assert obs["attributes"]["location_label"].startswith("on a chair")


def test_apply_keeps_uncertainty_when_not_an_item():
    desc = dict(GOOD, is_item=False, prompt_version="1.0.0")
    obs = captioner.apply(observation(), desc, "m")
    assert obs["attributes"]["item_type"] is None
    assert "ambiguous_attributes" in obs["uncertainty"]
    assert "may not be an item" in obs["caption_text"]


def test_caption_budget_and_crops(tmp_path, monkeypatch):
    clip = tmp_path / "c.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
    for _ in range(10):
        writer.write(solid((128, 128, 128)))
    writer.release()
    views = [captioner.ItemView(f"obs_{i}", 1.0, (40, 30, 30, 30), (160, 120)) for i in range(3)]
    monkeypatch.setenv("CAPTIONS_PER_VIDEO", "2")
    t = JsonTransport(GOOD)
    llm = LLMClient(api_key=FAKE_KEY, transport=t)
    out = captioner.describe_items(llm, clip, views, tmp_path / "crops")
    assert len(t.calls) <= 2          # budget respected (cache may save more)
    assert set(out) <= {"obs_0", "obs_1", "obs_2"}
    assert sorted(p.name for p in (tmp_path / "crops").iterdir()) == ["obs_0.jpg", "obs_1.jpg", "obs_2.jpg"]


def test_crops_saved_without_a_key(tmp_path):
    clip = tmp_path / "c.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
    for _ in range(10):
        writer.write(solid((40, 40, 200)))
    writer.release()
    out = captioner.describe_items(LLMClient(api_key=None), clip,
                                   [captioner.ItemView("obs_a", 1.0, (40, 30, 30, 30), (160, 120))],
                                   tmp_path / "crops")
    assert out == {}
    assert (tmp_path / "crops" / "obs_a.jpg").exists()


# -------------------------------------------------------------------- photo

def test_photo_decode_rejects_bad_input():
    with pytest.raises(photo.PhotoError):
        photo.decode(b"")
    with pytest.raises(photo.PhotoError):
        photo.decode(b"this is not an image")
    with pytest.raises(photo.PhotoError):
        photo.decode(b"x" * (photo.MAX_PHOTO_BYTES + 1))


def test_photo_without_key_uses_colour_only():
    img = photo.decode(png(solid((200, 60, 30))))   # BGR: blue
    seen = photo.describe_photo(LLMClient(api_key=None), img)
    assert seen["source"] == "colour"
    assert seen["description"]["primary_color"] in ("blue", "navy blue")
    assert seen["description"]["item_type"] is None


def test_photo_with_key_uses_the_ai_description():
    t = JsonTransport(GOOD)
    seen = photo.describe_photo(LLMClient(api_key=FAKE_KEY, transport=t), solid((128, 128, 128)))
    assert seen["source"] == "ai"
    assert seen["phrase"].startswith("gray hoodie")
    assert photo.search_text("lost it today", seen["description"]) == "lost it today (gray hoodie hood zipper)"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "p.db"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import app.main as main_module

    importlib.reload(main_module)
    return TestClient(main_module.app), tmp_path


def test_photo_endpoint_answers_and_stores_nothing(api):
    client, tmp_path = api
    before = {p for p in tmp_path.rglob("*")}
    r = client.post("/api/chat/photo", files={"photo": ("j.png", png(solid((128, 128, 128))), "image/png")},
                    data={"message": "I lost this yesterday"})
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert answer["photo"]["described_by"] == "colour"
    assert "possible match" in answer["lead"]
    after = {p for p in tmp_path.rglob("*")}
    assert not [p for p in after - before if p.suffix.lower() in (".png", ".jpg", ".jpeg")]


def test_photo_endpoint_rejects_non_images(api):
    client, _ = api
    r = client.post("/api/chat/photo", files={"photo": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


# ---------------------------------------------------------------- key check

class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener_raising(error):
    def opener(request, timeout):
        raise error
    return opener


def http_error(code):
    return urllib.error.HTTPError("https://api.openai.com/v1/models", code, "x", {}, io.BytesIO(b""))


def test_key_check_outcomes_never_echo_the_key():
    results = [
        openai_key_status(FAKE_KEY, opener=lambda request, timeout: FakeResponse()),
        openai_key_status(FAKE_KEY, opener=opener_raising(http_error(401))),
        openai_key_status(FAKE_KEY, opener=opener_raising(http_error(429))),
        openai_key_status(FAKE_KEY, opener=opener_raising(urllib.error.URLError("down"))),
    ]
    assert [ok for ok, _ in results] == [True, False, False, False]
    assert "401" in results[1][1] and "429" in results[2][1] and "network" in results[3][1]
    assert all(FAKE_KEY not in detail for _, detail in results)


def test_docker_image_includes_every_folder_the_app_reads():
    """The prompts folder once missed the image: the chat broke in production only."""
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
    for folder in ("app", "schemas", "fixtures", "tools", "prompts"):
        assert f"COPY {folder} ./{folder}" in dockerfile, folder


def test_crop_is_saved_even_when_the_seek_lands_past_the_end(tmp_path):
    clip = tmp_path / "c.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
    for _ in range(10):  # 2 seconds
        writer.write(solid((60, 60, 60)))
    writer.release()
    view = captioner.ItemView("obs_end", 2.4, (40, 30, 30, 30), (160, 120))
    captioner.describe_items(LLMClient(api_key=None), clip, [view], tmp_path / "crops")
    assert (tmp_path / "crops" / "obs_end.jpg").exists()
