"""Live and replay feeds go through the same ingest path as uploads."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.pipeline import ffmpeg_available

ROOT = Path(__file__).resolve().parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location("live_ingest", ROOT / "tools" / "live_ingest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live = load_tool()


def test_camera_password_is_never_printed():
    assert live.redact("rtsp://admin:secret@10.0.0.5/stream") == "rtsp://***@10.0.0.5/stream"


def test_segment_names_carry_their_start_time():
    assert live.segment_start(Path("seg_20261120T060500Z.mp4")) == datetime(2026, 11, 20, 6, 5, tzinfo=timezone.utc)
    assert live.segment_start(Path("other.mp4")) is None


def test_replay_reads_at_real_speed_and_drops_audio(tmp_path):
    cmd = live.segment_command("clip.mp4", tmp_path, 300, replay=True)
    assert "-re" in cmd and "-an" in cmd
    assert cmd[cmd.index("-segment_time") + 1] == "300"
    rtsp = live.segment_command("rtsp://cam/stream", tmp_path, 60, replay=False)
    assert "-re" not in rtsp and "tcp" in rtsp


def test_newest_segment_is_held_back_while_writing(tmp_path):
    for name in ("seg_20261120T060000Z.mp4", "seg_20261120T060500Z.mp4"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in live.finished_segments(tmp_path, False)] == ["seg_20261120T060000Z.mp4"]
    assert len(live.finished_segments(tmp_path, True)) == 2


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_replay_is_cut_into_timed_segments_in_real_time(tmp_path):
    clip = tmp_path / "feed.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-t", "7", "-g", "1000", str(clip)], check=True, timeout=60)
    out = tmp_path / "segs"
    out.mkdir()
    started = time.monotonic()
    subprocess.run(live.segment_command(str(clip), out, 3, replay=True), check=True, timeout=60)
    assert time.monotonic() - started >= 6  # read at camera speed
    segments = live.finished_segments(out, True)
    assert len(segments) >= 2, [p.name for p in segments]
    starts = [live.segment_start(p) for p in segments]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "live.db"))
    import importlib as il

    import app.main as main_module

    il.reload(main_module)
    from fastapi.testclient import TestClient

    return TestClient(main_module.app)


def test_ingest_rejects_an_unknown_kind(client):
    r = client.post("/api/ingest", files={"file": ("a.mp4", b"x", "video/mp4")},
                    data={"recorded_at": "2026-11-20T06:00:00Z", "kind": "webcam"})
    assert r.status_code == 400
    assert "kind" in r.json()["detail"]


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_live_segment_is_stored_as_a_live_feed_on_the_ingestion_clock(client, tmp_path):
    import time

    from tests.scenes import left_behind

    clip = left_behind(tmp_path / "seg.mp4")
    with clip.open("rb") as handle:
        r = client.post("/api/ingest", files={"file": ("seg_20261120T060000Z.mp4", handle, "video/mp4")},
                        data={"recorded_at": "2026-11-20T06:00:00Z", "kind": "live_feed",
                              "camera_id": "cam_live_test", "location_label": "Hall"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    for _ in range(240):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    source_id = job["result"]["source_id"]
    assert source_id.startswith("src_live_")
    footage = {f["source_id"]: f for f in client.get("/api/footage").json()["footage"]}
    assert source_id in footage
    assert len(job["result"]["items"]) == 1


def test_raw_uploads_wait_on_scratch_disk_not_the_volume(client, tmp_path, monkeypatch):
    import app.main as main_module

    scratch = tmp_path / "scratch"
    monkeypatch.setenv("UPLOAD_TMP_DIR", str(scratch))
    assert main_module.incoming_dir() == scratch / "foundit_incoming"
    assert main_module.media_dir() not in main_module.incoming_dir().parents


def test_a_full_disk_gives_a_clear_message_not_a_500(client, monkeypatch):
    import errno

    import app.main as main_module

    class FullPath:
        def __init__(self, *a):
            pass

        def mkdir(self, **k):
            pass

        def __truediv__(self, other):
            return self

        def open(self, *a, **k):
            raise OSError(errno.ENOSPC, "No space left on device")

        def unlink(self, missing_ok=False):
            pass

    monkeypatch.setattr(main_module, "incoming_dir", lambda: FullPath())
    r = client.post("/api/ingest", files={"file": ("a.mp4", b"x" * 10, "video/mp4")},
                    data={"recorded_at": "2026-11-20T06:00:00Z"})
    assert r.status_code == 507
    assert "ran out of room" in r.json()["detail"]


def test_leftover_raw_uploads_are_cleared(client, tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setenv("UPLOAD_TMP_DIR", str(tmp_path / "scratch"))
    for folder in (main_module.incoming_dir(), main_module.media_dir() / "incoming"):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "src_upload_old.mp4").write_bytes(b"x")
    assert main_module.clear_leftover_uploads() == 2
    assert not list(main_module.incoming_dir().iterdir())
