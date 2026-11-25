"""FastAPI application.

One process, no external services required. Demo mode reads precomputed
fixtures; ingest mode genuinely decodes uploaded video. Both are mounted
so a judge can use one and an engineer can prove the other.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .ai.providers import PROVIDERS
from .core.chat import ChatService
from .core.diagnostics import provider_summary, run_checks
from .core.pipeline import IngestError, SourceSpec, make_playback_copy, probe_duration, run_pipeline
from .core.store import Store
from .core.timeutil import iso_utc, parse_timestamp

ROOT = Path(__file__).resolve().parent.parent
DEMO_DB = ROOT / "data" / "foundit.db"

STARTUP_REPORT = run_checks()
print(STARTUP_REPORT.render())

app = FastAPI(title="FoundIt", version="0.1.0")

templates_dir = ROOT / "app" / "web" / "templates"
static_dir = ROOT / "app" / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None


def get_store() -> Store:
    """Open the store, seeding demo fixtures when it is empty.

    Seeding is keyed on emptiness rather than on the path, so a fresh
    database anywhere — a temp file in tests, a Railway volume, a local
    run — behaves the same way.
    """
    path = os.environ.get("DATABASE_PATH", str(DEMO_DB))
    store = Store(path)
    ensure_demo_data(store)
    return store


def ensure_demo_data(store: Store) -> None:
    """Load the precomputed fixtures the first time the database is empty."""
    if store.counts()["events"] > 0:
        return
    source_file = ROOT / "fixtures" / "video_sources.json"
    if not source_file.exists():
        return
    payload = json.loads(source_file.read_text())
    sources = payload if isinstance(payload, list) else [payload]
    for source in sources:
        store.upsert_source(source)

    for name, writer in (
        ("report_windows.json", store.upsert_window),
        ("observations.json", store.upsert_observation),
        ("item_events.json", store.upsert_event),
        ("evidence.json", store.upsert_evidence),
    ):
        path = ROOT / "fixtures" / name
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        for record in records if isinstance(records, list) else [records]:
            if name == "observations.json":
                pass
            writer(record)

    handover_file = ROOT / "fixtures" / "handover_records.json"
    if handover_file.exists():
        for record in json.loads(handover_file.read_text()):
            store.add_handover(record)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="templates unavailable")
    return templates.TemplateResponse(
        request, "index.html", {"title": "FoundIt", "timezone": "Asia/Riyadh"}
    )


@app.get("/api/health")
def health():
    report = run_checks()
    return JSONResponse(report.to_dict())


@app.get("/api/readiness")
def readiness():
    return {
        "checks": [
            {"name": c.name, "ok": c.ok, "fatal": c.fatal, "detail": c.detail}
            for c in STARTUP_REPORT.checks
        ],
        "ok": not STARTUP_REPORT.fatal_failures,
    }


@app.get("/api/providers")
def providers():
    return {"providers": provider_summary()}


@app.get("/api/stats")
def stats():
    store = get_store()
    try:
        return {
            "counts": store.counts(),
            "sources": [
                {
                    "source_id": s["source_id"],
                    "camera_id": s["camera_id"],
                    "location_label": s.get("location_label"),
                    "visibility": s.get("access", {}).get("visibility"),
                }
                for s in store.sources()
            ],
        }
    finally:
        store.close()


def media_dir() -> Path:
    """Playback copies live next to the database, on the persistent volume."""
    db = Path(os.environ.get("DATABASE_PATH", str(DEMO_DB)))
    return db.parent / "media"


SOURCE_ID = re.compile(r"^src_[a-z0-9_]+$")


def media_url_for(source_id: str) -> str | None:
    if SOURCE_ID.match(source_id) and (media_dir() / f"{source_id}.mp4").exists():
        return f"/media/{source_id}.mp4"
    return None


@app.post("/api/chat")
def chat(payload: dict):
    text = (payload or {}).get("message", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")
    store = get_store()
    try:
        service = ChatService(store, media_url_for=media_url_for)
        result = service.ask(text)
        return result.to_dict()
    finally:
        store.close()


@app.get("/media/{name}")
def media(name: str):
    source_id = name.removesuffix(".mp4")
    path = media_dir() / f"{source_id}.mp4"
    if not name.endswith(".mp4") or not SOURCE_ID.match(source_id) or not path.exists():
        raise HTTPException(status_code=404, detail="no playback copy for that footage")
    return FileResponse(path, media_type="video/mp4")


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
WORKER = threading.Semaphore(1)  # one video at a time: 1 GB RAM, 2 vCPU


def start_job(title: str, work) -> dict:
    """Run ``work(log)`` in the background; the page polls /api/jobs/{id}."""
    job_id = secrets.token_hex(6)
    job = {"job_id": job_id, "title": title, "status": "queued", "log": [], "result": None, "error": None}
    with JOBS_LOCK:
        JOBS[job_id] = job
        for old in list(JOBS)[:-50]:
            JOBS.pop(old, None)

    def log(line: str) -> None:
        job["log"].append(line)
        del job["log"][:-12]

    def run() -> None:
        with WORKER:
            job["status"] = "running"
            try:
                job["result"] = work(log)
                job["status"] = "done"
            except HTTPException as error:
                job["status"], job["error"] = "failed", str(error.detail)
            except IngestError as error:
                job["status"], job["error"] = "failed", str(error)
            except Exception as error:  # surfaced to the user, never swallowed
                job["status"], job["error"] = "failed", f"{type(error).__name__}: {error}"

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "status": job["status"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job (the server may have restarted)")
    return job


MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024


def _new_source_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"src_{prefix}_{stamp}_{secrets.token_hex(2)}"


def ingest_file(
    media_path: Path,
    *,
    source_id: str,
    camera_id: str,
    location_label: str | None,
    recorded_at: datetime | None,
    kind: str,
    clock: str,
    visibility: str,
    log=None,
) -> dict:
    """Process one video file and store everything it produces."""
    log = log or (lambda line: None)
    log("reading the video")
    duration = probe_duration(media_path)
    if recorded_at is None:
        # Replayed sample footage: placed so that it ends now.
        recorded_at = datetime.now(timezone.utc) - timedelta(seconds=duration)
    source = SourceSpec(
        source_id=source_id,
        camera_id=camera_id,
        venue_id="venue_upload",
        kind=kind,
        clock=clock,
        location_label=location_label,
        recording_started_at=recorded_at,
        media_path=str(media_path),
        duration_seconds=duration,
        visibility=visibility,
    )
    lines: list[str] = []

    def progress(line: str) -> None:
        lines.append(line)
        log(line)

    log(f"looking for items left behind in {duration:.0f} s of video")
    results = run_pipeline(source, progress=progress)
    log("preparing the video for playback")
    playback = make_playback_copy(media_path, media_dir() / f"{source_id}.mp4")
    source.playback_path = str(playback)

    store = get_store()
    try:
        store.upsert_source(source.to_record())
        for result in results:
            store.upsert_window(result.window)
            for obs in result.observations:
                store.upsert_observation(obs)
            for event in result.events:
                store.upsert_event(event)
            for evd in result.evidence:
                store.upsert_evidence(evd)
        return footage_summary(store, source.to_record()) | {"log": [l for l in lines if not l.startswith("analysed")]}
    finally:
        store.close()


def footage_summary(store: Store, source: dict) -> dict:
    zone = ZoneInfo(source.get("timezone") or "Asia/Riyadh")
    started = source.get("recording_started_at")
    items = []
    for obs in store.observations_for_source(source["source_id"]):
        events = [e for e in store.events_for_source(source["source_id"])
                  if e["observation_id"] == obs["observation_id"]]
        items.append({
            "observation_id": obs["observation_id"],
            "state": obs["state"],
            "colour": obs["attributes"].get("primary_color"),
            "item_type": obs["attributes"].get("item_type"),
            "location": obs["attributes"].get("location_label"),
            "caption": obs.get("caption_text"),
            "events": [
                {
                    "type": e["event_type"],
                    "state": e["state_at_event"],
                    "offset_s": e["event_time_offset_s"],
                    "local_time": parse_timestamp(e["event_time"]).astimezone(zone).strftime("%H:%M:%S")
                    if e.get("event_time") else None,
                }
                for e in events
            ],
        })
    local_start = parse_timestamp(started).astimezone(zone) if started else None
    return {
        "source_id": source["source_id"],
        "location_label": source.get("location_label"),
        "kind": source.get("kind"),
        "recorded_at": started,
        "recorded_local": local_start.strftime("%d %B %Y, %H:%M") if local_start else None,
        "recorded_day": local_start.strftime("%d %B %Y").lstrip("0") if local_start else None,
        "duration_s": source.get("duration_seconds"),
        "media_url": media_url_for(source["source_id"]),
        "items": items,
    }


@app.post("/api/ingest")
def ingest(
    file: UploadFile = File(...),
    camera_id: str = Form("cam_upload_1"),
    location_label: str = Form(""),
    recorded_at: str = Form(...),
    visibility: str = Form("approved_private"),
):
    """Upload a clip: decode it, find items that were left, keep a playback copy."""
    try:
        started = parse_timestamp(recorded_at)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"recorded_at: {error}") from error
    if visibility not in {"public_demo", "approved_private", "internal_only"}:
        raise HTTPException(status_code=400, detail="unknown visibility")

    source_id = _new_source_id("upload")
    suffix = Path(file.filename or "clip.mp4").suffix.lower()[:8] or ".mp4"
    uploads = media_dir() / "incoming"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / f"{source_id}{suffix}"
    size = 0
    try:
        with target.open("wb") as handle:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="video is larger than the upload limit")
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="the uploaded file is empty")
    except BaseException:
        target.unlink(missing_ok=True)
        raise

    def work(log):
        try:
            return ingest_file(
                target,
                source_id=source_id,
                camera_id=camera_id or "cam_upload_1",
                location_label=location_label.strip() or None,
                recorded_at=started,
                kind="recorded_upload",
                clock="source_clock",
                visibility=visibility,
                log=log,
            )
        finally:
            target.unlink(missing_ok=True)

    return start_job(file.filename or "uploaded video", work)


# Real footage that may be shown publicly: CAVIAR, Creative Commons BY-SA.
# Acknowledgement requested by the dataset: EC Funded CAVIAR project/IST 2001 37540,
# http://homepages.inf.ed.ac.uk/rbf/CAVIAR/
CAVIAR_BASE = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1"
SAMPLES = {
    "left_bag_at_chair": ("LeftBag_AtChair/LeftBag_AtChair.mpg", "Person leaves a bag at the chairs"),
    "left_bag_picked_up": ("LeftBag_PickedUp/LeftBag_PickedUp.mpg", "Person leaves a bag, then picks it up"),
    "left_bag": ("LeftBag/LeftBag.mpg", "Person leaves a bag by the wall"),
    "left_box": ("LeftBox/LeftBox.mpg", "Person leaves a box"),
}


@app.get("/api/samples")
def samples():
    return {
        "credit": "CAVIAR dataset (EC Funded CAVIAR project/IST 2001 37540), CC BY-SA",
        "samples": [{"id": key, "title": title} for key, (_, title) in SAMPLES.items()],
    }


@app.post("/api/samples/{sample_id}")
def ingest_sample(sample_id: str):
    """Fetch a real CAVIAR clip on the server and process it like an upload.

    The clips were recorded in 2003. They are replayed with timestamps from
    the moment of ingest (clock = ingestion_clock) so they can be searched
    as today's footage, and the location label says so.
    """
    if sample_id not in SAMPLES:
        raise HTTPException(status_code=404, detail="unknown sample")
    path, title = SAMPLES[sample_id]

    def work(log):
        incoming = media_dir() / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        target = incoming / f"{sample_id}_{secrets.token_hex(3)}.mpg"
        try:
            log("downloading the sample clip from the CAVIAR archive")
            request = urllib.request.Request(f"{CAVIAR_BASE}/{path}", headers={"User-Agent": "FoundIt/0.2"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise HTTPException(status_code=502, detail=f"could not download the sample clip: {error}") from error
            return ingest_file(
                target,
                source_id=_new_source_id(f"caviar_{sample_id}"),
                camera_id="cam_caviar_inria",
                location_label=f"CAVIAR sample: {title} (INRIA lobby, 2003, replayed as today)",
                recorded_at=None,
                kind="replay",
                clock="ingestion_clock",
                visibility="public_demo",
                log=log,
            )
        finally:
            target.unlink(missing_ok=True)

    return start_job(title, work)


@app.get("/api/footage/{source_id}/tracks")
def footage_tracks(source_id: str):
    """What the detector sees in stored footage, track by track.

    Read-only diagnostics for tuning on real clips: re-runs detection on
    the playback copy and returns every track, including the ones the
    pipeline then sets aside as people, with the per-moment distance to the
    nearest moving person.
    """
    from statistics import median

    from .core.detector import detect_left_items
    from .core.pipeline import is_person_like

    path = media_dir() / f"{source_id}.mp4"
    if not SOURCE_ID.match(source_id) or not path.exists():
        raise HTTPException(status_code=404, detail="no playback copy for that footage")
    with WORKER:
        result = detect_left_items(path)
    return {
        "frames": result.frames_analysed,
        "frame_size": result.frame_size,
        "ghosts_healed": result.ghosts_healed,
        "scene_resets": result.scene_resets,
        "tracks": [
            {
                "id": t.track_id,
                "placed_s": round(t.placed_offset_s, 1),
                "detected_s": round(t.detected_offset_s, 1),
                "last_seen_s": round(t.last_seen_offset_s, 1),
                "end": t.end_reason,
                "colour": t.colour_name(),
                "bbox": t.bbox,
                "person_like": is_person_like(t),
                "motion_median": round(median(p.internal_motion for p in t.points), 3),
                "distance": [
                    [round(p.offset_s, 1), None if p.person_distance_ratio is None else round(p.person_distance_ratio, 3)]
                    for p in t.points[::5]
                ],
            }
            for t in result.tracks
        ],
    }


@app.get("/api/footage")
def footage():
    """Everything that has been processed, newest first, with what was found."""
    store = get_store()
    try:
        sources = sorted(
            store.sources(), key=lambda s: s.get("recording_started_at") or "", reverse=True
        )
        return {"footage": [footage_summary(store, s) for s in sources]}
    finally:
        store.close()


@app.exception_handler(IngestError)
def ingest_error_handler(request: Request, exc: IngestError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
