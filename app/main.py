"""FastAPI application.

One process, no external services required. Demo mode reads precomputed
fixtures; ingest mode genuinely decodes uploaded video. Both are mounted
so a judge can use one and an engineer can prove the other.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .ai.providers import PROVIDERS
from .core.chat import ChatService
from .core.diagnostics import provider_summary, run_checks
from .core.pipeline import IngestError, SourceSpec, run_pipeline
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


@app.post("/api/chat")
def chat(payload: dict):
    text = (payload or {}).get("message", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")
    store = get_store()
    try:
        service = ChatService(store)
        result = service.ask(text)
        return result.to_dict()
    finally:
        store.close()


@app.post("/api/ingest")
async def ingest(
    file: UploadFile = File(...),
    camera_id: str = Form("cam_upload_1"),
    location_label: str = Form("Uploaded footage"),
    recorded_at: str = Form(...),
    visibility: str = Form("approved_private"),
):
    """Genuine processing path: decode a real clip and store what it finds."""
    try:
        started = parse_timestamp(recorded_at)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"recorded_at: {error}") from error

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await file.read())
        media_path = handle.name

    source = SourceSpec(
        source_id=f"src_upload_{int(datetime.now(timezone.utc).timestamp())}",
        camera_id=camera_id,
        venue_id="venue_upload",
        kind="recorded_upload",
        clock="source_clock",
        location_label=location_label,
        recording_started_at=started,
        media_path=media_path,
        visibility=visibility,
    )

    log: list[str] = []
    try:
        results = run_pipeline(source, progress=log.append)
    except IngestError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

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
        return {
            "source_id": source.source_id,
            "windows": len(results),
            "log": log,
            "events": [e for r in results for e in r.events],
        }
    finally:
        store.close()


@app.exception_handler(IngestError)
def ingest_error_handler(request: Request, exc: IngestError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
