"""Feed a live camera, or replay a file as if it were live, into FoundIt.

    # A real camera (RTSP or HTTP), 10-minute segments, to the deployed site:
    python tools/live_ingest.py rtsp://user:pass@192.168.1.20/stream \\
        --server https://your-app.up.railway.app --camera cam_classroom_3 \\
        --location "Classroom 3"

    # Replay a recorded file through the same live path (for demos):
    python tools/live_ingest.py classroom.mp4 --replay --server http://localhost:8000

How it works: ffmpeg reads the stream and cuts it into segments without
re-encoding. Each segment is named by the UTC time it started. When a
segment is finished, it is posted to POST /api/ingest, the same endpoint as
an upload, with kind=live_feed (or replay). Everything downstream (detector,
three timestamps, windows, evidence, chat) is identical.

Honest limits (checked by running a 60 s replay in two 30 s segments):
- Each segment is processed on its own; track state is not carried across
  segments yet. The detector learns each segment's empty room from its
  first seconds, so an item already lying there when a segment starts
  becomes part of that segment's background. It is reported in the segment
  where it was put down (and stays searchable), but a later pickup in a
  later segment is not seen. Use segments at least as long as the 10-minute
  report window (the default) so most events finish inside one segment.
- An item put down near the end of a segment may only reach "possibly
  unattended" before the segment ends.
- Live segments use the ingestion clock (when ffmpeg read them), which is
  absolute but late by the network and buffer delay.
- Audio is dropped.

Camera passwords in the URL never leave this machine and are not logged.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

NAME = re.compile(r"^seg_(\d{8}T\d{6}Z)\.mp4$")


def redact(url: str) -> str:
    return re.sub(r"//[^/@]+@", "//***@", url)


def segment_command(source: str, out_dir: Path, seconds: int, replay: bool) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if replay:
        cmd += ["-re"]  # read at real speed, like a camera
    elif source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [
        "-i", source, "-an",
    ]
    if replay:
        # Files may have few keyframes; re-encode so every cut lands on time.
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-force_key_frames", f"expr:gte(t,n_forced*{seconds})"]
    else:
        # Cameras send keyframes every second or two: copy, no CPU cost.
        cmd += ["-c:v", "copy"]
    cmd += [
        "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1",
        "-segment_format", "mp4", "-strftime", "1",
        str(out_dir / "seg_%Y%m%dT%H%M%SZ.mp4"),
    ]
    return cmd


def segment_start(path: Path) -> datetime | None:
    match = NAME.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def post_segment(server: str, path: Path, started: datetime, camera: str, location: str, kind: str,
                 timeout: float = 120) -> dict:
    """multipart/form-data POST with the standard library only."""
    boundary = uuid.uuid4().hex
    fields = {"recorded_at": started.isoformat(), "camera_id": camera, "location_label": location,
              "kind": kind}
    body = bytearray()
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
             "Content-Type: video/mp4\r\n\r\n").encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        server.rstrip("/") + "/api/ingest", data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def finished_segments(folder: Path, writing_done: bool) -> list[Path]:
    """All segments except the newest one, which ffmpeg may still be writing."""
    segments = sorted(p for p in folder.glob("seg_*.mp4") if segment_start(p))
    return segments if writing_done else segments[:-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="rtsp:// or http(s):// camera URL, or a video file with --replay")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--camera", default="cam_live_1")
    ap.add_argument("--location", default="")
    ap.add_argument("--segment-minutes", type=float, default=10)
    ap.add_argument("--replay", action="store_true", help="treat the source as a file played back live")
    ap.add_argument("--keep", action="store_true", help="keep segments after upload")
    args = ap.parse_args(argv)

    if not shutil.which("ffmpeg"):
        print("ffmpeg is not installed", file=sys.stderr)
        return 2
    kind = "replay" if args.replay else "live_feed"
    seconds = max(30, int(args.segment_minutes * 60))
    work = Path(tempfile.mkdtemp(prefix="foundit_live_"))
    print(f"reading {redact(args.source)} as {kind}, {seconds} s segments, posting to {args.server}")
    ffmpeg = subprocess.Popen(segment_command(args.source, work, seconds, args.replay))
    sent: set[str] = set()
    retries = 0
    try:
        while True:
            done = ffmpeg.poll() is not None
            for seg in finished_segments(work, done):
                if seg.name in sent:
                    continue
                started = segment_start(seg)
                try:
                    reply = post_segment(args.server, seg, started, args.camera, args.location, kind)
                    print(f"{seg.name}: posted, job {reply.get('job_id')}")
                except Exception as error:  # noqa: BLE001 - keep the feed running
                    print(f"{seg.name}: upload failed ({error}); will retry", file=sys.stderr)
                    continue
                sent.add(seg.name)
                if not args.keep:
                    seg.unlink(missing_ok=True)
            if done:
                left = [p for p in finished_segments(work, True) if p.name not in sent]
                retries += 1
                if not left or retries > 3:
                    for p in left:
                        print(f"{p.name}: not sent after retries; kept in {work}", file=sys.stderr)
                    return ffmpeg.returncode or (1 if left else 0)
            time.sleep(5)
    except KeyboardInterrupt:
        ffmpeg.terminate()
        print("stopped; the unfinished last segment was not sent")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
