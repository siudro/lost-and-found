"""Measure the pipeline against clips whose truth is known.

    python tools/evaluate.py                 # synthetic scenes + 10-minute classroom
    python tools/evaluate.py --classroom 4   # shorter classroom clip
    python tools/evaluate.py --json out.json

Ground truth comes from the scripts that draw the clips (tests/scenes.py and
tools/make_classroom_clip.py), so every number here is reproducible and
nothing is hand-labelled. These are SIMULATED clips: they show the logic is
right, not how well it works on real cameras. Real-footage results (CAVIAR,
checked by eye) are reported separately in docs/evaluation.md.

No API key is used. Photo queries use the colour-only path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from app.ai.photo import describe_photo, search_text  # noqa: E402
from app.ai.captioner import grab_frame  # noqa: E402
from app.ai.llm import LLMClient  # noqa: E402
from app.core.chat import ChatService  # noqa: E402
from app.core.pipeline import SourceSpec, run_pipeline  # noqa: E402
from app.core.store import Store  # noqa: E402
from tests import scenes  # noqa: E402

START = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
UNATTENDED = {"possibly_unattended", "likely_unattended"}

# What really happens in each synthetic scene (seconds from clip start).
# owner_leaves: the moment the owner starts walking away from the item.
TRUTH = {
    "left_behind": {"items": 1, "should_alert": True, "placed": 6, "owner_leaves": 9, "colour": "gray"},
    "attended": {"items": 0, "should_alert": False},
    "picked_up": {"items": 1, "should_alert": True, "placed": 6, "owner_leaves": 9, "picked_up": 40, "colour": "gray"},
    "passers_by": {"items": 0, "should_alert": False},
    "busy_start": {"items": 1, "should_alert": True, "placed": 12, "owner_leaves": 15, "colour": "black"},
    "sunlight": {"items": 1, "should_alert": True, "placed": 10, "owner_leaves": 13, "colour": "black"},
}


def seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return (datetime.fromisoformat(stamp.replace("Z", "+00:00")) - START).total_seconds()


def load_classroom():
    script = ROOT / "tools" / "make_classroom_clip.py"
    spec = importlib.util.spec_from_file_location("make_classroom_clip", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_clip(name, truth, results, elapsed, duration):
    observations = [o for r in results for o in r.observations]
    events = [e for r in results for e in r.events]
    alerted = [o for o in observations if o["state"] in UNATTENDED or any(
        e["observation_id"] == o["observation_id"] and e["event_type"] == "unattended_onset_estimated"
        for e in events)]
    row = {
        "clip": name,
        "duration_s": round(duration, 1),
        "processing_s": round(elapsed, 1),
        "speed_x_realtime": round(duration / elapsed, 1) if elapsed else None,
        "items_true": truth["items"],
        "items_reported": len(observations),
        "tp": 0, "fp": 0, "fn": 0,
    }
    if truth["should_alert"]:
        row["tp"] = min(1, len(alerted))
        row["fn"] = 1 - row["tp"]
        row["fp"] = max(0, len(alerted) - 1)
    else:
        row["fp"] = len(alerted)
    if alerted and truth.get("owner_leaves") is not None:
        obs = alerted[0]
        first = seconds(obs["first_seen_at"])
        onset = seconds(obs.get("unattended_onset_estimate"))
        confirmed = seconds(obs.get("confirmed_at"))
        row["first_seen_error_s"] = round(first - truth["placed"], 1) if first is not None else None
        row["onset_error_s"] = round(onset - truth["owner_leaves"], 1) if onset is not None else None
        row["detection_delay_s"] = round(confirmed - onset, 1) if None not in (confirmed, onset) else None
        row["colour_right"] = obs["attributes"].get("primary_color") == truth.get("colour")
        if truth.get("picked_up") is not None:
            picks = [seconds(e["event_time"]) for e in events if e["event_type"] == "picked_up"]
            row["pickup_error_s"] = round(picks[0] - truth["picked_up"], 1) if picks else None
    return row


def store_results(store: Store, source, results):
    store.upsert_source(source.to_record())
    for result in results:
        store.upsert_window(result.window)
        for o in result.observations:
            store.upsert_observation(o)
        for e in result.events:
            store.upsert_event(e)
        for d in result.evidence:
            store.upsert_evidence(d)


def retrieval_eval(store: Store, targets: list[dict]) -> dict:
    """Text and photo queries for each alerted item; is the right one first?"""
    service = ChatService(store, now_provider=lambda: START.replace(hour=20))
    offline = LLMClient(api_key=None)
    text_hits = photo_hits = text_top3 = photo_top3 = 0
    latencies = []
    rows = []
    for t in targets:
        query = f"Did anyone leave a {t['colour']} {t['noun']} today?"
        result = service.ask(query)
        latencies.append(result.latency_ms)
        ranked = [m.observation_id for m in result.answer.matches]
        text_rank = ranked.index(t["observation_id"]) + 1 if t["observation_id"] in ranked else None
        # Photo stand-in: the item cut tightly from the footage, so it fills
        # the picture the way a visitor's own photo of it would.
        frame = grab_frame(t["media"], t["offset_s"])
        sx, sy = frame.shape[1] / t["work_size"][0], frame.shape[0] / t["work_size"][1]
        x, y, w, h = t["bbox"]
        crop = frame[int(y * sy): int((y + h) * sy), int(x * sx): int((x + w) * sx)]
        seen = describe_photo(offline, crop)
        p_result = service.ask(search_text("I lost this today", seen["description"]))
        p_ranked = [m.observation_id for m in p_result.answer.matches]
        photo_rank = p_ranked.index(t["observation_id"]) + 1 if t["observation_id"] in p_ranked else None
        text_hits += text_rank == 1
        photo_hits += photo_rank == 1
        text_top3 += bool(text_rank and text_rank <= 3)
        photo_top3 += bool(photo_rank and photo_rank <= 3)
        rows.append({"item": t["name"], "text_query": query, "text_rank": text_rank,
                     "photo_colour_seen": seen["phrase"], "photo_rank": photo_rank,
                     "candidates": len(ranked)})
    n = len(targets) or 1
    latencies.sort()
    return {
        "items": len(targets),
        "text_top1": round(text_hits / n, 2), "text_top3": round(text_top3 / n, 2),
        "photo_colour_only_top1": round(photo_hits / n, 2), "photo_colour_only_top3": round(photo_top3 / n, 2),
        "search_latency_ms_median": round(latencies[len(latencies) // 2], 1) if latencies else None,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classroom", type=float, default=10, help="classroom clip minutes (0 to skip)")
    ap.add_argument("--json", help="write the full results here")
    args = ap.parse_args()

    rows, targets = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = Store(tmp / "eval.db")
        clips = [(name, getattr(scenes, name), TRUTH[name]) for name in TRUTH]
        if args.classroom:
            module = load_classroom()
            clips.append(("classroom_jacket", module, None))
        for name, maker, truth in clips:
            clip = tmp / f"{name}.mp4"
            if name == "classroom_jacket":
                story = maker.render(str(clip), minutes=args.classroom)
                truth = {"items": 1, "should_alert": True, "placed": story["jacket_placed_s"],
                         "owner_leaves": story["owner_leaves_s"], "colour": "gray"}
            else:
                maker(clip)
            cap = cv2.VideoCapture(str(clip))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            cap.release()
            print(f"running {name} ({duration:.0f} s)...", flush=True)
            views: list = []
            source = SourceSpec(
                source_id=f"src_eval_{name}", camera_id="cam_eval", venue_id="venue_eval", kind="replay",
                clock="source_clock", location_label=f"evaluation {name}", recording_started_at=START,
                media_path=str(clip),
            )
            started = time.perf_counter()
            results = run_pipeline(source, views=views)
            elapsed = time.perf_counter() - started
            rows.append(score_clip(name, truth, results, elapsed, duration))
            store_results(store, source, results)
            if truth["should_alert"]:
                for v in views:
                    targets.append({"name": name, "observation_id": v.observation_id, "colour": truth["colour"],
                                    "noun": "jacket" if name == "classroom_jacket" else "bag",
                                    "media": clip, "offset_s": v.offset_s, "bbox": v.bbox,
                                    "work_size": v.work_size})
                    break
        retrieval = retrieval_eval(store, targets)
        store.close()

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    summary = {
        "unattended_precision": round(tp / (tp + fp), 2) if tp + fp else None,
        "unattended_recall": round(tp / (tp + fn), 2) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
    }
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "simulated": True, "summary": summary, "clips": rows, "retrieval": retrieval}

    print("\nclip               items  TP FP FN  first-seen  onset   delay  pickup  colour  speed")
    for r in rows:
        print(f"{r['clip']:<18} {r['items_reported']}/{r['items_true']}    {r['tp']}  {r['fp']}  {r['fn']}  "
              f"{str(r.get('first_seen_error_s', '-')):>9}  {str(r.get('onset_error_s', '-')):>6}  "
              f"{str(r.get('detection_delay_s', '-')):>6}  {str(r.get('pickup_error_s', '-')):>6}  "
              f"{str(r.get('colour_right', '-')):>6}  {r['speed_x_realtime']}x")
    print(f"\nprecision {summary['unattended_precision']}  recall {summary['unattended_recall']}  "
          f"peak memory {summary['peak_memory_mb']:.0f} MB")
    print(f"retrieval: text top-1 {retrieval['text_top1']}, top-3 {retrieval['text_top3']}; "
          f"photo (colour only) top-1 {retrieval['photo_colour_only_top1']}, top-3 {retrieval['photo_colour_only_top3']}; "
          f"median search {retrieval['search_latency_ms_median']} ms")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
