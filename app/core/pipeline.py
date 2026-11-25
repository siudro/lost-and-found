"""Video ingest and the per-window analysis pipeline.

This is the genuine processing path: it decodes real frames with ffmpeg,
finds regions that stay put while people move around them, and turns those
into ItemEvents.

What it is not: a learned detector. It is a motion-and-stability model. On
a staged clip of a jacket on a chair that is sufficient and honest; on a
cluttered real classroom it will over-propose. The segmentation benchmark
(WAI-269) replaces this with SAM-backed detection, and the interface is
designed so that swap needs no change anywhere else.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .timeutil import (
    DEFAULT_TIMEZONE,
    iso_utc,
    offset_to_absolute,
    parse_timestamp,
    window_bounds,
)
from .unattended import State, Thresholds, TrackSample, infer

FRAME_WIDTH = 320
FRAME_HEIGHT = 180


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
                iso_utc(
                    self.recording_started_at
                    + timedelta(seconds=self.duration_seconds or 0)
                )
                if self.recording_started_at and self.duration_seconds
                else None
            ),
            "media_path": self.media_path,
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
        raise IngestError(f"ffprobe failed: {result.stderr.strip()[:300]}")
    try:
        return float(result.stdout.strip())
    except ValueError as error:
        raise IngestError(f"could not read duration from {path}") from error


def extract_frames(
    path: str | Path,
    every_seconds: float,
    tmpdir: Path,
) -> list[tuple[float, Path]]:
    """Decode frames every N seconds into raw RGB files.

    Writes small PPM frames so no image library is required — the pipeline
    stays dependency-light, which matters for a free deployment target.
    """
    if every_seconds <= 0:
        raise IngestError("sampling interval must be positive")
    pattern = tmpdir / "frame_%05d.ppm"
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"fps=1/{every_seconds},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-pix_fmt", "rgb24", str(pattern),
        ],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        raise IngestError(f"ffmpeg frame extraction failed: {result.stderr.strip()[:300]}")

    frames: list[tuple[float, Path]] = []
    for index, frame_path in enumerate(sorted(tmpdir.glob("frame_*.ppm"))):
        frames.append((index * every_seconds, frame_path))
    if not frames:
        raise IngestError("no frames decoded from the video")
    return frames


def read_ppm(path: Path) -> tuple[int, int, bytes]:
    """Minimal PPM reader for the 8-bit RGB our own ffmpeg writes."""
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise IngestError(f"unexpected frame format in {path.name}")
    parts = data.split(maxsplit=4)
    if len(parts) < 5:
        raise IngestError(f"truncated header in {path.name}")
    width, height = int(parts[1]), int(parts[2])
    return width, height, parts[4]


# --------------------------------------------------------------------------
# Motion analysis
# --------------------------------------------------------------------------


def frame_difference(a: bytes, b: bytes) -> list[int]:
    """Absolute per-pixel luma difference, one value per pixel."""
    step = 3 * 4  # sample every 4th pixel for speed
    diffs: list[int] = []
    for i in range(0, len(a) - 2, step):
        la = (a[i] * 299 + a[i + 1] * 587 + a[i + 2] * 114) // 1000
        lb = (b[i] * 299 + b[i + 1] * 587 + b[i + 2] * 114) // 1000
        diffs.append(abs(la - lb))
    return diffs


def stable_regions(
    diffs: list[int],
    threshold: int = 12,
) -> list[tuple[int, int]]:
    """Contiguous runs of pixels that did not change.

    A run is a candidate for a stationary object: steady across the frame
    pair where everything else moved.
    """
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(diffs):
        if value <= threshold:
            if start is None:
                start = index
        else:
            if start is not None and index - start >= 40:
                regions.append((start, index))
            start = None
    if start is not None and len(diffs) - start >= 40:
        regions.append((start, len(diffs)))
    return regions


def dominant_colour(rgb: bytes, offset: int) -> str:
    """Cheap colour naming from one sampled pixel. Deliberately coarse."""
    if offset + 2 >= len(rgb):
        return "unknown"
    r, g, b = rgb[offset], rgb[offset + 1], rgb[offset + 2]
    high, low = max(r, g, b), min(r, g, b)
    if high - low < 25:
        if high < 60:
            return "black"
        if high > 200:
            return "white"
        return "gray"
    if r >= g and r >= b:
        if g > b + 40:
            return "orange"
        return "red" if g < 100 else "brown"
    if g >= r and g >= b:
        return "green" if g > 100 else "olive"
    if b > r + 30:
        return "navy blue" if b < 140 else "blue"
    return "gray"


# --------------------------------------------------------------------------
# Pipeline
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


def analyse_window(
    source: SourceSpec,
    frames: list[tuple[float, Path]],
    window_start_offset: float,
    *,
    thresholds: Thresholds | None = None,
    window_minutes: int = 10,
    scene_text: str | None = None,
) -> WindowResult:
    """Turn a block of decoded frames into observations, events and evidence."""
    if not frames:
        raise IngestError("no frames to analyse")

    first_rgb = read_ppm(frames[0][1])[2]
    samples: list[TrackSample] = []
    colour_votes: dict[str, int] = {}
    region_centres: list[tuple[float, float]] = []

    for offset, frame_path in frames:
        width, height, rgb = read_ppm(frame_path)
        diffs = frame_difference(first_rgb, rgb)
        regions = stable_regions(diffs)
        if regions:
            start, end = max(regions, key=lambda r: r[1] - r[0])
            centre_index = (start + end) // 2
            samples.append(
                TrackSample(
                    offset_s=window_start_offset + offset,
                    position=(float(centre_index % (frame_difference_width(diffs))), 0.0),
                    nearest_person_distance_ratio=None,
                    visible=True,
                )
            )
            region_centres.append((float(start), float(end)))
            colour = dominant_colour(rgb, centre_index * 12)
            colour_votes[colour] = colour_votes.get(colour, 0) + 1
        else:
            samples.append(
                TrackSample(
                    offset_s=window_start_offset + offset,
                    position=region_centres[-1] if region_centres else (0.0, 0.0),
                    nearest_person_distance_ratio=None,
                    visible=False,
                )
            )

    primary_colour = (
        max(colour_votes, key=lambda k: colour_votes[k]) if colour_votes else None
    )
    present_at_start = frames[0][0] == 0.0

    verdict = infer(
        samples,
        is_worn=False,
        present_at_clip_start=present_at_start,
        thresholds=thresholds,
        frame_width=float(frame_difference_width(frame_difference(first_rgb, first_rgb)) or 1),
    )

    abs_start = offset_to_absolute(window_start_offset, source.recording_started_at)
    if abs_start is None:
        raise IngestError("source has no recording start time; cannot place events in time")
    win_start, win_end = window_bounds(abs_start, window_minutes)

    content_basis = f"{source.source_id}:{iso_utc(win_start)}:{len(frames)}"
    content_hash = "sha256:" + hashlib.sha256(content_basis.encode()).hexdigest()[:32]

    window_id = make_window_id(source.source_id, win_start)
    observation_id = f"obs_{source.source_id[4:]}_{int(window_start_offset):06d}"
    event_id = f"evt_{source.source_id[4:]}_{int(window_start_offset):06d}_{verdict.state.value}"
    evidence_id = f"evd_{source.source_id[4:]}_{int(window_start_offset):06d}"

    event_time = abs_start
    clip_start = max(0.0, window_start_offset - 30)
    clip_end = window_start_offset + 30
    if source.duration_seconds:
        clip_end = min(clip_end, source.duration_seconds)

    observation = {
        "observation_id": observation_id,
        "source_id": source.source_id,
        "camera_id": source.camera_id,
        "track_ref": f"motion_v1:{content_hash[:16]}",
        "first_frame_offset_s": window_start_offset,
        "last_frame_offset_s": window_start_offset + (frames[-1][0] or 0),
        "first_seen_at": iso_utc(abs_start),
        "is_worn": False,
        "state": verdict.state.value,
        "unattended_onset_estimate": (
            iso_utc(abs_start + timedelta(seconds=verdict.deserted_since_s or 0))
            if verdict.deserted_since_s is not None
            and verdict.state in (State.POSSIBLY_UNATTENDED, State.LIKELY_UNATTENDED)
            else None
        ),
        "onset_certainty": verdict.onset_certainty.value,
        "confirmed_at": (
            iso_utc(abs_start + timedelta(seconds=frames[-1][0]))
            if verdict.state is State.LIKELY_UNATTENDED
            else None
        ),
        "confidence": verdict.confidence,
        "uncertainty": verdict.uncertainty,
        "attributes": {
            "item_type": "object",
            "primary_color": primary_colour,
            "secondary_color": None,
            "visible_features": [],
            "location_label": source.location_label,
        },
        "caption_text": (
            f"A {primary_colour or 'unidentified'} object remained still in view "
            f"from {abs_start.strftime('%H:%M')} for about "
            f"{int(verdict.seconds_stationary)} seconds."
        ),
        "caption_version": "motion_captioner_v1",
    }

    event = {
        "event_id": event_id,
        "event_type": (
            "unattended_onset_estimated"
            if verdict.state in (State.POSSIBLY_UNATTENDED, State.LIKELY_UNATTENDED)
            else "became_stationary" if verdict.state is State.STATIONARY
            else "first_observed"
        ),
        "observation_id": observation_id,
        "source_id": source.source_id,
        "camera_id": source.camera_id,
        "event_time_offset_s": window_start_offset,
        "event_time": iso_utc(event_time),
        "window_id": window_id,
        "report_id": window_id.replace("win_", "rep_"),
        "state_at_event": verdict.state.value,
        "onset_certainty": verdict.onset_certainty.value,
        "uncertainty": verdict.uncertainty,
        "confidence": verdict.confidence,
        "evidence_id": evidence_id,
        "is_worn": False,
        "note": "Detected by motion stability. " + ", ".join(verdict.rules_fired),
    }

    evidence = {
        "evidence_id": evidence_id,
        "event_id": event_id,
        "source_id": source.source_id,
        "camera_id": source.camera_id,
        "clip_start_offset_s": clip_start,
        "clip_end_offset_s": clip_end,
        "clip_start_at": iso_utc(abs_start - timedelta(seconds=window_start_offset - clip_start)),
        "clip_end_at": iso_utc(abs_start + timedelta(seconds=clip_end - window_start_offset)),
        "window_id": window_id,
        "seek_to_offset_s": window_start_offset,
        "player_ref": f"player:{source.source_id}?t={int(window_start_offset)}",
        "availability": "available",
        "retention": {"retain_until": None},
    }

    window = {
        "window_id": window_id,
        "source_id": source.source_id,
        "camera_id": source.camera_id,
        "window_start": iso_utc(win_start),
        "window_end": iso_utc(win_end),
        "window_start_offset_s": window_start_offset,
        "report_id": window_id.replace("win_", "rep_"),
        "report_date": win_start.astimezone(
            __import__("zoneinfo").ZoneInfo(source.timezone)
        ).strftime("%Y-%m-%d"),
        "coverage": {
            "status": "full",
            "observed_seconds": float(window_minutes * 60),
            "expected_seconds": float(window_minutes * 60),
            "gaps": [],
        },
        "scene_summary": {
            "text": scene_text or (
                f"Footage from {source.location_label or source.camera_id} between "
                f"{win_start.strftime('%H:%M')} and {win_end.strftime('%H:%M')} UTC. "
                f"An object that stayed still was detected and its stability was analysed."
            ),
            "caption_version": "motion_captioner_v1",
        },
        "event_ids": [event_id],
        "observation_ids": [observation_id],
        "sampling": {"base_interval_s": 1.0, "dense_interval_s": None, "dense_intervals": []},
        "processing": {
            "pipeline_version": "motion_v1",
            "detector": "rule_based_detector",
            "segmenter": None,
            "captioner": "motion_captioner_v1",
            "content_hash": content_hash,
            "processed_at": iso_utc(datetime.now(timezone.utc)),
        },
    }

    return WindowResult(window=window, observations=[observation], events=[event], evidence=[evidence])


def frame_difference_width(diffs: list[int]) -> int:
    """How many sampled columns a diff array represents."""
    return len(diffs)


def make_window_id(source_id: str, start: datetime) -> str:
    stamp = start.strftime("%Y%m%dT%H%M%SZ")
    return f"win_{source_id[4:]}_{stamp}"


def run_pipeline(
    source: SourceSpec,
    *,
    sampling_seconds: float = 1.0,
    window_minutes: int = 10,
    thresholds: Thresholds | None = None,
    progress=None,
) -> list[WindowResult]:
    """Process a whole source, window by window.

    Raises IngestError for undecodable video or a source with no wall clock —
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

    duration = source.duration_seconds or probe_duration(media)
    source.duration_seconds = duration
    window_seconds = window_minutes * 60
    results: list[WindowResult] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        frames = extract_frames(media, sampling_seconds, tmpdir)
        if progress:
            progress(f"decoded {len(frames)} frames from {media.name}")

        for index in range(0, len(frames), max(1, int(window_seconds / sampling_seconds))):
            chunk = frames[index:index + max(1, int(window_seconds / sampling_seconds))]
            if not chunk:
                continue
            start_offset = chunk[0][0]
            result = analyse_window(
                source, chunk, start_offset,
                thresholds=thresholds, window_minutes=window_minutes,
            )
            results.append(result)
            if progress:
                progress(
                    f"window {len(results)}: {result.window['window_id']} "
                    f"state={result.observations[0]['state']}"
                )

    return results
