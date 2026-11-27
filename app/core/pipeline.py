"""Video ingest: from a video file to report windows, observations, events
and evidence that the chat can search.

The pixels are read by ``detector.py`` (left-object detection with two
background models). This module turns each item track into the shared data
contract and judges it with the rule-based logic in ``unattended.py``.

The three timestamps stay separate, as the contract requires:

* ``first_seen_at`` — when the item's pixels first stopped changing, i.e.
  when it was put down;
* ``unattended_onset_estimate`` — from when nobody was near it;
* ``confirmed_at`` — when it had been alone long enough to call it
  "likely unattended". Confirmed minus onset is detection delay, never the
  event time.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from .detector import DetectionResult, DetectorConfig, DetectorError, ItemTrack, detect_left_items
from .timeutil import DEFAULT_TIMEZONE, iso_utc, window_bounds
from .unattended import Inference, State, Thresholds, TrackSample, infer

PIPELINE_VERSION = "leftobject_v2"
CAPTIONER = "colour_and_position_v1"

PERSON_LIKE_MOTION = 0.05
"""A 'still' shape whose pixels keep moving this much is a person standing,
not an item. Measured: bags ~0.00, a person standing still ~0.15."""


def demo_thresholds() -> Thresholds:
    """Unattended policy for ingested footage.

    The library defaults (2 min still, 5 min alone) suit a real venue. Test
    clips are a minute or two long, so ingest uses shorter, configurable
    values: still for 10 s, alone for 15 s = possibly unattended, alone for
    30 s = likely unattended.
    """
    return Thresholds(
        stationary_seconds=float(os.environ.get("UNATTENDED_STATIONARY_S", 10)),
        deserted_seconds=float(os.environ.get("UNATTENDED_DESERTED_S", 15)),
    )


class IngestError(RuntimeError):
    """Raised when footage cannot be decoded or has no usable clock."""


@dataclass
class SourceSpec:
    source_id: str
    camera_id: str
    venue_id: str
    kind: str
    clock: str
    timezone: str = DEFAULT_TIMEZONE
    location_label: str | None = None
    recording_started_at: datetime | None = None
    media_path: str | None = None
    duration_seconds: float | None = None
    visibility: str = "internal_only"
    playback_path: str | None = None

    def to_record(self) -> dict:
        return {
            "source_id": self.source_id,
            "camera_id": self.camera_id,
            "kind": self.kind,
            "clock": self.clock,
            "timezone": self.timezone,
            "venue_id": self.venue_id,
            "location_label": self.location_label,
            "recording_started_at": (
                iso_utc(self.recording_started_at) if self.recording_started_at else None
            ),
            "recording_ended_at": (
                iso_utc(self.recording_started_at + timedelta(seconds=self.duration_seconds))
                if self.recording_started_at and self.duration_seconds
                else None
            ),
            "media_path": self.playback_path or self.media_path,
            "duration_seconds": self.duration_seconds,
            "access": {"visibility": self.visibility, "approved_by": None, "expires_at": None},
        }


# --------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(path: str | Path) -> float:
    """Duration in seconds. Raises IngestError rather than guessing."""
    if not ffmpeg_available():
        raise IngestError("ffmpeg is not installed; cannot decode video")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise IngestError(f"ffprobe could not read this file: {result.stderr.strip()[:300]}")
    try:
        return float(result.stdout.strip())
    except ValueError as error:
        raise IngestError(f"could not read the duration of {Path(path).name}") from error


def make_playback_copy(source: str | Path, target: str | Path) -> Path:
    """Re-encode to H.264 MP4 that every browser can play and seek.

    Audio is dropped: the project keeps audio off by default.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-an",
            "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        raise IngestError(f"could not prepare the video for playback: {result.stderr.strip()[:300]}")
    return target


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class WindowResult:
    window: dict
    observations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "observations": self.observations,
            "events": self.events,
            "evidence": self.evidence,
        }


def make_window_id(source_id: str, start: datetime) -> str:
    stamp = start.strftime("%Y%m%dT%H%M%SZ")
    return f"win_{source_id[4:]}_{stamp}"


@dataclass
class ItemReport:
    """One item track, judged and placed in time."""

    observation: dict
    events: list[dict]
    evidence: list[dict]
    verdict: Inference
    summary: str


# --------------------------------------------------------------------------
# Track -> contract
# --------------------------------------------------------------------------


def is_person_like(track: ItemTrack) -> bool:
    return median(p.internal_motion for p in track.points) >= PERSON_LIKE_MOTION


def _clock(local_tz: ZoneInfo, moment: datetime) -> str:
    return moment.astimezone(local_tz).strftime("%H:%M:%S")


def _judge(track: ItemTrack, thresholds: Thresholds, present_at_start: bool) -> Inference:
    # The detector decides whether the item moved (by watching the empty
    # background come back); here the position is held steady so box jitter
    # from people walking in front is not mistaken for a pickup.
    cx = median(p.centre[0] for p in track.points)
    cy = median(p.centre[1] for p in track.points)
    samples = [TrackSample(offset_s=track.placed_offset_s, position=(cx, cy),
                           nearest_person_distance_ratio=0.0, visible=True)]
    samples += [
        TrackSample(
            offset_s=p.offset_s,
            position=(cx, cy),
            nearest_person_distance_ratio=p.person_distance_ratio,
            visible=True,
        )
        for p in track.points
    ]

    def judge(prefix: list[TrackSample]) -> Inference:
        return infer(
            prefix,
            is_worn=False,
            present_at_clip_start=present_at_start,
            thresholds=thresholds,
            frame_width=float(track.frame_size[0] or 1),
        )

    # The rules judge the end of whatever they are given. An item left alone
    # for a minute and then collected would be judged at the moment of
    # collection, with its owner beside it, and the minute alone would be
    # lost. So judge the end of every spell alone, and the end of the
    # track, and report the most serious reading.
    def alone(sample: TrackSample) -> bool:
        near = sample.nearest_person_distance_ratio
        return near is None or near > thresholds.person_proximity_ratio

    cuts = [
        i for i in range(len(samples))
        if i == len(samples) - 1 or (alone(samples[i]) and not alone(samples[i + 1]))
    ]
    severity = {
        State.LIKELY_UNATTENDED: 4, State.POSSIBLY_UNATTENDED: 3,
        State.STATIONARY: 2, State.SEEN: 1, State.UNCERTAIN: 0, State.MOVED: 0,
    }
    verdicts = [judge(samples[: i + 1]) for i in cuts]
    return max(verdicts, key=lambda v: (severity[v.state], v.seconds_deserted))


KEPT_WITH_OWNER_S = 60.0
"""An item that had someone right beside it, never alone for ALONE_BREAK_S,
for at least this long, and then went away with them, is treated as
belonging to that person rather than as lost property: a student's bag at
their feet through a lesson, taken at the bell. The same rule removes the
still back of a seated person, which background subtraction cannot tell
from an object. Items still there at the end are always reported."""
ALONE_BREAK_S = 3.0


def kept_with_owner(track: ItemTrack, verdict: Inference) -> bool:
    if track.end_reason == "removed":
        return (
            verdict.seconds_deserted < ALONE_BREAK_S
            and track.last_seen_offset_s - track.placed_offset_s >= KEPT_WITH_OWNER_S
        )
    if track.end_reason == "lost":
        # It went out of view without ever being left: people were around
        # it until it vanished, so it was never lost property.
        return verdict.state not in (State.POSSIBLY_UNATTENDED, State.LIKELY_UNATTENDED)
    return False


def build_item_report(
    source: SourceSpec,
    track: ItemTrack,
    index: int,
    thresholds: Thresholds,
    window_minutes: int,
) -> ItemReport | None:
    """None for an item that stayed with its owner the whole time (see
    KEPT_WITH_OWNER_S): nothing was left, so there is nothing to report."""
    assert source.recording_started_at is not None
    start = source.recording_started_at
    local_tz = ZoneInfo(source.timezone)
    present_at_start = track.placed_offset_s <= 1.5
    verdict = _judge(track, thresholds, present_at_start)
    if kept_with_owner(track, verdict):
        return None

    def at(offset: float) -> datetime:
        return start + timedelta(seconds=offset)

    colour = track.colour_name()
    location = track.location_phrase()
    if source.location_label:
        location = f"{location} ({source.location_label})"

    base = f"{source.source_id[4:]}_{index:03d}"
    observation_id = f"obs_{base}"
    unattended = verdict.state in (State.POSSIBLY_UNATTENDED, State.LIKELY_UNATTENDED)
    onset_offset = verdict.deserted_since_s if unattended else None
    confirmed_offset = (
        verdict.deserted_since_s + 2 * thresholds.deserted_seconds
        if verdict.state is State.LIKELY_UNATTENDED and verdict.deserted_since_s is not None
        else None
    )
    removed = track.end_reason == "removed"
    final_state = State.MOVED if removed else verdict.state

    uncertainty = list(dict.fromkeys(verdict.uncertainty))
    if track.end_reason == "lost":
        uncertainty.append("occluded")
    if track.frame_size[0] and track.frame_size[0] < 400:
        uncertainty.append("low_resolution")
    uncertainty.append("ambiguous_attributes")  # item type is never determined

    seconds_alone = int(verdict.seconds_deserted)
    pieces = [
        f"An item ({colour or 'colour unclear'}) was put down {track.location_phrase()} at "
        f"{_clock(local_tz, at(track.placed_offset_s))}."
    ]
    if unattended:
        pieces.append(
            f"Nobody was near it from {_clock(local_tz, at(onset_offset))} "
            f"(about {seconds_alone} s in total)."
        )
    elif verdict.state is State.STATIONARY and seconds_alone >= 3:
        pieces.append(
            f"Nobody was near it for about {seconds_alone} s at most, which is not long "
            f"enough to count as unattended (that needs {int(thresholds.deserted_seconds)} s)."
        )
    elif verdict.state is State.STATIONARY:
        pieces.append(
            "There was nearly always someone close to it (not necessarily its owner), "
            "so it does not count as unattended."
        )
    if removed:
        pieces.append(f"It was picked up or moved at {_clock(local_tz, at(track.ended_offset_s))}.")
    elif track.end_reason == "lost":
        pieces.append("It later went out of view (something may be standing in front of it).")
    else:
        pieces.append("It was still there when the recording ended.")
    caption = " ".join(pieces)

    observation = {
        "observation_id": observation_id,
        "source_id": source.source_id,
        "camera_id": source.camera_id,
        "track_ref": f"{PIPELINE_VERSION}:{source.source_id}:{track.track_id}",
        "first_frame_offset_s": round(track.placed_offset_s, 2),
        "last_frame_offset_s": round(track.last_seen_offset_s, 2),
        "first_seen_at": iso_utc(at(track.placed_offset_s)),
        "is_worn": False,
        "state": final_state.value,
        "unattended_onset_estimate": iso_utc(at(onset_offset)) if onset_offset is not None else None,
        "onset_certainty": verdict.onset_certainty.value,
        "confirmed_at": iso_utc(at(confirmed_offset)) if confirmed_offset is not None else None,
        "confidence": verdict.confidence,
        "uncertainty": uncertainty,
        "attributes": {
            "item_type": None,
            "primary_color": colour,
            "secondary_color": None,
            "visible_features": [],
            "location_label": location,
        },
        "caption_text": caption,
        "caption_version": CAPTIONER,
    }

    events: list[dict] = []
    evidence: list[dict] = []

    def add_event(kind: str, offset: float, state: State, certainty: str, note: str) -> None:
        moment = at(offset)
        win_start, _ = window_bounds(moment, window_minutes)
        window_id = make_window_id(source.source_id, win_start)
        event_id = f"evt_{base}_{kind}"
        evidence_id = f"evd_{base}_{kind}"
        clip_start = max(0.0, offset - 30)
        clip_end = offset + 30
        if source.duration_seconds:
            clip_end = min(clip_end, source.duration_seconds)
        events.append({
            "event_id": event_id,
            "event_type": kind,
            "observation_id": observation_id,
            "source_id": source.source_id,
            "camera_id": source.camera_id,
            "event_time_offset_s": round(offset, 2),
            "event_time": iso_utc(moment),
            "window_id": window_id,
            "report_id": window_id.replace("win_", "rep_"),
            "state_at_event": state.value,
            "onset_certainty": certainty,
            "uncertainty": uncertainty,
            "confidence": verdict.confidence,
            "evidence_id": evidence_id,
            "is_worn": False,
            "note": note,
        })
        evidence.append({
            "evidence_id": evidence_id,
            "event_id": event_id,
            "source_id": source.source_id,
            "camera_id": source.camera_id,
            "clip_start_offset_s": round(clip_start, 2),
            "clip_end_offset_s": round(clip_end, 2),
            "clip_start_at": iso_utc(at(clip_start)),
            "clip_end_at": iso_utc(at(clip_end)),
            "window_id": window_id,
            "seek_to_offset_s": round(offset, 2),
            "player_ref": f"player:{source.source_id}?t={offset:.1f}",
            "availability": "available",
            "retention": {"retain_until": None},
        })

    rules = ", ".join(verdict.rules_fired)
    if unattended:
        add_event("unattended_onset_estimated", onset_offset, verdict.state,
                  verdict.onset_certainty.value, f"{caption} Rules: {rules}.")
    else:
        add_event("became_stationary", track.placed_offset_s, verdict.state,
                  "not_applicable", f"{caption} Rules: {rules}.")
    if removed:
        add_event("picked_up", track.ended_offset_s, State.MOVED, "observed",
                  f"The empty background reappeared where the item had been. {caption}")

    return ItemReport(observation, events, evidence, verdict, caption)


def build_windows(
    source: SourceSpec,
    reports: list[ItemReport],
    detection: DetectionResult,
    window_minutes: int,
) -> list[WindowResult]:
    assert source.recording_started_at is not None and source.duration_seconds
    start = source.recording_started_at
    end = start + timedelta(seconds=source.duration_seconds)
    local_tz = ZoneInfo(source.timezone)
    window_seconds = window_minutes * 60
    content_hash = "sha256:" + hashlib.sha256(
        f"{source.source_id}:{source.duration_seconds}:{detection.frames_analysed}".encode()
    ).hexdigest()[:32]

    results: list[WindowResult] = []
    cursor, _ = window_bounds(start, window_minutes)
    while cursor < end:
        win_start, win_end = cursor, cursor + timedelta(seconds=window_seconds)
        overlap = (min(win_end, end) - max(win_start, start)).total_seconds()
        window_id = make_window_id(source.source_id, win_start)
        events = [e for r in reports for e in r.events if e["window_id"] == window_id]
        evidence = [d for r in reports for d in r.evidence if d["window_id"] == window_id]
        observation_ids = sorted({e["observation_id"] for e in events})
        observations = [r.observation for r in reports if r.observation["observation_id"] in observation_ids]

        span = (
            f"{max(win_start, start).astimezone(local_tz).strftime('%H:%M')}–"
            f"{min(win_end, end).astimezone(local_tz).strftime('%H:%M')}"
        )
        place = source.location_label or source.camera_id
        if observations:
            body = " ".join(r.summary for r in reports if r.observation["observation_id"] in observation_ids)
            text = f"Footage from {place}, {span} ({source.timezone}). {body}"
        else:
            text = (
                f"Footage from {place}, {span} ({source.timezone}). "
                "No item was put down and left in this part of the recording."
            )
        text += " Items already in place when the recording began cannot be detected."
        offset = (win_start - start).total_seconds()

        results.append(WindowResult(
            window={
                "window_id": window_id,
                "source_id": source.source_id,
                "camera_id": source.camera_id,
                "window_start": iso_utc(win_start),
                "window_end": iso_utc(win_end),
                "window_start_offset_s": offset if offset >= 0 else None,
                "report_id": window_id.replace("win_", "rep_"),
                "report_date": win_start.astimezone(local_tz).strftime("%Y-%m-%d"),
                "coverage": {
                    "status": "full" if overlap >= window_seconds - 1 else "partial",
                    "observed_seconds": round(max(0.0, overlap), 1),
                    "expected_seconds": float(window_seconds),
                    "gaps": [],
                },
                "scene_summary": {"text": text, "caption_version": CAPTIONER},
                "event_ids": [e["event_id"] for e in events],
                "observation_ids": observation_ids,
                "sampling": {"base_interval_s": 0.2, "dense_interval_s": None, "dense_intervals": []},
                "processing": {
                    "pipeline_version": PIPELINE_VERSION,
                    "detector": "dual_background_left_object",
                    "segmenter": None,
                    "captioner": CAPTIONER,
                    "content_hash": content_hash,
                    "processed_at": iso_utc(datetime.now(timezone.utc)),
                },
            },
            observations=observations,
            events=events,
            evidence=evidence,
        ))
        cursor = win_end
    return results


def run_pipeline(
    source: SourceSpec,
    *,
    window_minutes: int = 10,
    thresholds: Thresholds | None = None,
    detector_config: DetectorConfig | None = None,
    progress=None,
    sampling_seconds: float | None = None,
    views: list | None = None,
) -> list[WindowResult]:
    """Process a whole recording.

    Raises IngestError for undecodable video or a source with no wall clock;
    both are things the user must be told, not papered over.
    """
    if not source.media_path:
        raise IngestError("source has no media_path")
    media = Path(source.media_path)
    if not media.exists():
        raise IngestError(f"media file not found: {media}")
    if source.recording_started_at is None:
        raise IngestError(
            "source has no recording_started_at; calendar questions cannot be "
            "answered without a wall clock"
        )

    config = detector_config or DetectorConfig()
    if sampling_seconds:
        config = DetectorConfig(**{**config.__dict__, "sample_fps": 1.0 / sampling_seconds})
    policy = thresholds or demo_thresholds()

    duration = source.duration_seconds or probe_duration(media)
    source.duration_seconds = duration
    try:
        detection = detect_left_items(media, config, progress=progress)
    except DetectorError as error:
        raise IngestError(str(error)) from error
    if progress:
        progress(f"decoded {detection.frames_analysed} frames from {media.name}")

    tracks = [t for t in detection.tracks if not is_person_like(t)]
    skipped = len(detection.tracks) - len(tracks)
    if progress and skipped:
        progress(f"ignored {skipped} still shape(s) that moved like a person")

    reports = []
    kept = 0
    for track in tracks:
        report = build_item_report(source, track, len(reports) + 1, policy, window_minutes)
        if report is None:
            kept += 1
            continue
        reports.append(report)
        if views is not None:
            # The clearest look at the item: nobody near it, preferably late.
            best = max(
                track.points,
                key=lambda p: (p.person_distance_ratio is None, p.person_distance_ratio or 0.0, p.offset_s),
            )
            from ..ai.captioner import ItemView

            views.append(ItemView(report.observation["observation_id"], best.offset_s, best.bbox,
                                  track.frame_size))
    if progress and kept:
        progress(f"set aside {kept} thing(s) that stayed with someone the whole time and left with them")
    for report in reports:
        if progress:
            progress(report.summary)
    windows = build_windows(source, reports, detection, window_minutes)
    if progress:
        progress(f"{len(reports)} item(s) found across {len(windows)} report window(s)")
    return windows
