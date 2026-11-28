"""The web page's upload: chunked, retry-safe, and the recording time read
from the video when the person leaves it alone."""

from __future__ import annotations

import importlib
import subprocess
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.pipeline import ffmpeg_available, probe_creation_time, probe_width

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "u.db"))
    monkeypatch.setenv("UPLOAD_TMP_DIR", str(tmp_path / "scratch"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    import app.main as main_module

    importlib.reload(main_module)
    return TestClient(main_module.app)


def wait(client, job_id):
    for _ in range(400):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.25)
    raise AssertionError("job did not finish")


def make_clip(path, size="320x240", seconds=4, creation=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10", "-t", str(seconds),
           "-pix_fmt", "yuv420p"]
    if creation:
        cmd += ["-metadata", f"creation_time={creation}"]
    subprocess.run(cmd + [str(path)], check=True, timeout=60)
    return path


@needs_ffmpeg
def test_recording_time_is_read_from_the_video(tmp_path):
    clip = make_clip(tmp_path / "a.mp4", creation="2026-11-28T13:05:00.000000Z")
    assert probe_creation_time(clip) == datetime(2026, 11, 28, 13, 5, tzinfo=timezone.utc)
    assert probe_creation_time(make_clip(tmp_path / "b.mp4")) is None


@needs_ffmpeg
def test_width_accounts_for_size(tmp_path):
    assert probe_width(make_clip(tmp_path / "w.mp4", size="1280x720", seconds=1)) == 1280


@needs_ffmpeg
def test_chunked_upload_end_to_end_with_retry(client, tmp_path):
    from tests.scenes import left_behind

    clip = left_behind(tmp_path / "phone.mp4")
    data = clip.read_bytes()
    start = client.post("/api/uploads", data={"filename": "phone.mp4", "size": str(len(data))}).json()
    upload_id, piece = start["upload_id"], 100_000
    for offset in range(0, len(data), piece):
        r = client.put(f"/api/uploads/{upload_id}?offset={offset}", content=data[offset:offset + piece])
        assert r.status_code == 200, r.text
        if offset == 0:  # the same piece again, as after a dropped reply
            again = client.put(f"/api/uploads/{upload_id}?offset=0", content=data[:piece])
            assert again.json()["received"] == min(piece, len(data))
    gap = client.put(f"/api/uploads/{upload_id}?offset={len(data) + 5}", content=b"x")
    assert gap.status_code == 409
    finish = client.post(f"/api/uploads/{upload_id}/finish",
                         data={"filename": "phone.mp4", "file_modified_ms": "1764334800000"})
    job = wait(client, finish.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    assert len(job["result"]["items"]) == 1
    # No time typed and none in the file: the phone's file date was used.
    assert any("file's date" in line for line in job["log"])


def test_unknown_or_finished_upload_is_refused(client):
    r = client.put("/api/uploads/src_upload_20260101000000_abcd?offset=0", content=b"x")
    assert r.status_code == 404
    assert client.post("/api/uploads/../../etc/finish").status_code == 404


def test_too_big_is_refused_before_uploading(client):
    r = client.post("/api/uploads", data={"filename": "big.mov", "size": str(600 * 1024 * 1024)})
    assert r.status_code == 413
    assert "1080p" in r.json()["detail"]


@needs_ffmpeg
def test_large_video_is_analysed_from_the_small_copy(client, tmp_path):
    clip = make_clip(tmp_path / "hd.mp4", size="1280x720", seconds=3, creation="2026-11-28T10:00:00Z")
    with clip.open("rb") as handle:
        r = client.post("/api/ingest", files={"file": ("hd.mp4", handle, "video/mp4")})
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    assert any("read from the video file" in line for line in job["log"])
    assert job["result"]["recorded_local"]


@needs_ffmpeg
def test_playback_copy_caps_frame_rate_at_15(tmp_path):
    from app.core.pipeline import make_playback_copy, probe_fps

    fast = tmp_path / "fast.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "2",
                    "-pix_fmt", "yuv420p", str(fast)], check=True, timeout=60)
    slow = make_clip(tmp_path / "slow.mp4", seconds=2)  # 10 fps stays 10
    assert round(probe_fps(make_playback_copy(fast, tmp_path / "f_out.mp4"))) == 15
    assert round(probe_fps(make_playback_copy(slow, tmp_path / "s_out.mp4"))) == 10
