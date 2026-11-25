"""Left-object detection on real video.

How it works, in one paragraph: keep two background models of the scene.
The *long* model changes slowly and only where nothing is in front of it,
so it remembers the empty room. The *short* model catches up within a
second or two. A pixel that differs from the long model but already matches
the short model belongs to something that arrived and then stopped moving:
a bag put down, a jacket left on a chair. Pixels that differ from both are
things still moving, which here means people. Static pixels are grouped
into items, items are tracked over time, and for every moment we record how
far the nearest moving person was. That track is what the rule-based
unattended logic in ``unattended.py`` then judges.

What this cannot do, stated so nobody has to discover it:

* It does not know *what* an item is. ``item_type`` is reported as unknown
  (``None``) rather than guessed.
* An item already in the room when the recording starts becomes part of the
  long background and is not detected at all.
* Somebody who stands perfectly still for several seconds can look like an
  item. The size limits reject the largest such shapes, not all of them.
* "Nearest person" means nearest moving foreground. It is anonymous,
  local to this camera, and never identifies anyone.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class DetectorError(RuntimeError):
    """The video could not be read."""


@dataclass(frozen=True)
class DetectorConfig:
    sample_fps: float = 5.0
    """Frames analysed per second. Five is plenty for objects that sit still."""

    work_width: int = 480
    """Frames wider than this are shrunk to it. Smaller videos keep their
    size: low-resolution footage (CAVIAR is 384 px wide) cannot afford to
    lose pixels when a bag is only a dozen pixels across."""

    diff_threshold: float = 30.0
    """Colour distance (0–441) above which a pixel differs from a background."""

    static_seconds: float = 3.0
    """How long a pixel must stay changed-but-still before it counts as an item."""

    bg_init_seconds: float = 8.0
    """The first picture of the empty room is the per-pixel median of this
    many seconds. People walking through are removed by the median; with a
    single first frame, every spot where someone stood became a fake item
    the moment they walked off."""

    long_alpha: float = 0.02
    """Per-frame learning rate of the long background, where it is visible."""

    short_alpha: float = 0.25
    """Per-frame learning rate of the short background."""

    min_area_ratio: float = 0.0004
    """Smallest item, as a fraction of the frame."""

    max_area_ratio: float = 0.10
    """Largest item. Bigger static shapes are people, furniture moves or lighting."""

    mover_min_area_ratio: float = 0.0012
    """Smallest moving shape that counts as a person nearby."""

    lost_seconds: float = 2.0
    """A track not seen for this long is closed."""

    ghost_edge_ratio: float = 1.3
    """A new 'item' whose background has clearly more edges than the live
    frame is a ghost: something that was there and left, revealing the floor.
    Ghosts are healed into the background instead of reported."""

    light_ncc: float = 0.75
    """Correlation above which a changed patch is treated as the same floor
    under different light, not an object."""

    scene_change_ratio: float = 0.45
    """If this fraction of the frame differs from the background, the camera
    moved or the lights changed. The background is rebuilt rather than
    reporting the whole room as one giant item."""


@dataclass
class TrackPoint:
    offset_s: float
    centre: tuple[float, float]
    bbox: tuple[int, int, int, int]
    person_distance_ratio: float | None
    internal_motion: float = 0.0
    """Fraction of the box's pixels still moving. A bag is ~0; a person
    standing still keeps swaying, breathing and gesturing."""


@dataclass
class ItemTrack:
    track_id: int
    placed_offset_s: float
    """Best estimate of when the item was put down: when its pixels first
    stopped changing, not when the detector became sure."""
    detected_offset_s: float
    points: list[TrackPoint] = field(default_factory=list)
    last_seen_offset_s: float = 0.0
    end_reason: str = "active"
    """'active' (still there at the end), 'removed' (the empty background
    came back: someone took it), or 'lost' (it vanished from view without the
    background returning, e.g. something now stands in front of it)."""
    ended_offset_s: float | None = None
    colour_samples: list[tuple[int, int, int]] = field(default_factory=list)
    frame_size: tuple[int, int] = (0, 0)
    surround_activity: float = 0.0
    """Share of the whole clip during which the ring around the item was
    moving. A spot seen through a glass railing onto a busy floor below, or
    swaying plants, is busy all the time; a floor where a bag is left is
    quiet apart from the person who left it."""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.points[-1].bbox

    def duration_s(self) -> float:
        return self.last_seen_offset_s - self.placed_offset_s

    def colour_name(self) -> str | None:
        if not self.colour_samples:
            return None
        arr = np.array(self.colour_samples, dtype=np.float32)
        b, g, r = (float(v) for v in np.median(arr, axis=0))
        return name_colour(r, g, b)

    def location_phrase(self) -> str:
        width, height = self.frame_size
        x, y, w, h = self.bbox
        cx = (x + w / 2) / max(width, 1)
        cy = (y + h / 2) / max(height, 1)
        horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
        vertical = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
        if horizontal == "centre" and vertical == "middle":
            return "in the centre of the frame"
        if horizontal == "centre":
            return f"at the {vertical} of the frame"
        if vertical == "middle":
            return f"on the {horizontal} side of the frame"
        return f"in the {vertical} {horizontal} of the frame"


@dataclass
class DetectionResult:
    tracks: list[ItemTrack]
    frames_analysed: int
    duration_s: float
    frame_size: tuple[int, int]
    scene_resets: list[float] = field(default_factory=list)
    ghosts_healed: int = 0


# --------------------------------------------------------------------------
# Colour naming
# --------------------------------------------------------------------------


def name_colour(r: float, g: float, b: float) -> str:
    """Name a colour in the vocabulary the query parser understands."""
    pixel = np.uint8([[[int(b), int(g), int(r)]]])
    h, s, v = (int(x) for x in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])
    # OpenCV: h in 0..179, s and v in 0..255.
    if v < 50:
        return "black"
    if s < 40:
        if v > 200:
            return "white"
        return "gray"
    if s < 70 and v < 110:
        return "gray"
    hue = h * 2  # degrees
    if hue < 15 or hue >= 340:
        return "red" if v > 90 else "brown"
    if hue < 40:
        return "orange" if v > 150 and s > 120 else "brown"
    if hue < 70:
        return "yellow" if v > 120 else "brown"
    if hue < 165:
        return "green"
    if hue < 255:
        return "blue" if v > 110 else "navy blue"
    if hue < 290:
        return "purple"
    return "pink"


# --------------------------------------------------------------------------
# Frame source
# --------------------------------------------------------------------------


def _probe_scaled_size(path: Path, width: int) -> tuple[int, int]:
    """Size of one decoded, scaled frame — respects phone rotation metadata."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"scale={width}:-2", "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "ppm", "-",
        ],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0 or not result.stdout.startswith(b"P6"):
        message = result.stderr.decode(errors="replace").strip()[:300]
        raise DetectorError(f"could not decode a frame: {message or 'unknown error'}")
    parts = result.stdout.split(maxsplit=3)
    return int(parts[1]), int(parts[2])


def _native_width(path: Path) -> int | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def iter_frames(path: Path, config: DetectorConfig) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (offset_seconds, BGR frame) at ``config.sample_fps``."""
    native = _native_width(path) or config.work_width
    target = min(config.work_width, native)
    target -= target % 2
    width, height = _probe_scaled_size(path, target)
    frame_bytes = width * height * 3
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-an",
            "-vf", f"fps={config.sample_fps},scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
            yield index / config.sample_fps, frame
            index += 1
    finally:
        process.stdout.close()
        process.wait(timeout=60)
    if index == 0:
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        raise DetectorError(f"no frames decoded: {error.strip()[:300]}")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _colour_distance(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    diff = frame.astype(np.float32) - background
    return np.sqrt(np.sum(diff * diff, axis=2))


def _boxes(mask: np.ndarray, min_area: int, max_area: int) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if min_area <= area <= max_area:
            boxes.append((x, y, w, h))
    return boxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Pixel distance between two boxes' edges; 0 when they touch or overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return float((dx * dx + dy * dy) ** 0.5)


def _edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def _is_ghost(frame: np.ndarray, background: np.ndarray, mask: np.ndarray, box, ratio: float) -> bool:
    """Did something *leave* here, rather than arrive?

    Look only along the outline of the changed region. A real item has a
    sharp outline in the live frame. A ghost (the floor revealed where
    something used to stand) has its outline only in the old background,
    and the live frame is smooth across it.
    """
    x, y, w, h = box
    pad = 4
    y0, y1 = max(0, y - pad), min(frame.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(frame.shape[1], x + w + pad)
    region = mask[y0:y1, x0:x1].astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    band = (cv2.dilate(region, kernel) - cv2.erode(region, kernel)).astype(bool)
    if not band.any():
        return False
    live = float(_edges(frame[y0:y1, x0:x1])[band].mean())
    past = float(_edges(background[y0:y1, x0:x1])[band].mean())
    return past > ratio * max(live, 1.0)


def _has_object_edges(frame: np.ndarray, background: np.ndarray, box) -> bool:
    """A real object adds a clear outline that the background lacks."""
    x, y, w, h = box
    live = float(_edges(frame[y:y + h, x:x + w]).mean())
    past = float(_edges(background[y:y + h, x:x + w]).mean())
    return live > 1.5 * past + 10


def _crisp_outline(frame: np.ndarray, background: np.ndarray, box, mask: np.ndarray | None) -> bool:
    x, y, w, h = box
    pad = 3
    y0, y1 = max(0, y - pad), min(frame.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(frame.shape[1], x + w + pad)
    if mask is None:
        region = np.zeros((y1 - y0, x1 - x0), np.uint8)
        region[y - y0:y - y0 + h, x - x0:x - x0 + w] = 1
    else:
        region = mask[y0:y1, x0:x1].astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    band = (cv2.dilate(region, kernel) - cv2.erode(region, kernel)).astype(bool)
    if not band.any():
        return False
    live = float(_edges(frame[y0:y1, x0:x1])[band].mean())
    past = float(_edges(background[y0:y1, x0:x1])[band].mean())
    return live > 1.5 * past + 12


def _only_light_changed(frame: np.ndarray, background: np.ndarray, box, min_ncc: float,
                        mask: np.ndarray | None = None) -> bool:
    """Is this a lighting change rather than an object?

    When sunlight, a shadow or the camera's exposure shifts, the floor's
    pattern survives and only gets brighter or darker, so inside the changed
    region the live pixels correlate strongly with the background. A real
    object replaces the pattern and the correlation collapses. Textureless
    regions fall back to a colour test: a brightness change keeps the hue.
    """
    x, y, w, h = box
    live_bgr = frame[y:y + h, x:x + w].astype(np.float32)
    past_bgr = background[y:y + h, x:x + w].astype(np.float32)
    inside = np.ones((h, w), bool) if mask is None else mask[y:y + h, x:x + w].astype(bool)
    if inside.sum() >= 30:
        inside = cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) if inside.sum() > 80 else inside
    if inside.sum() < 12:
        return False
    live = cv2.cvtColor(live_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[inside]
    past = cv2.cvtColor(past_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[inside]
    if live.mean() > past.mean() + 15 and float((live >= 240).mean()) > 0.5:
        return True  # sunlight blew the patch out to white; no object does that
    live_c, past_c = live - live.mean(), past - past.mean()
    live_sd, past_sd = float(live_c.std()), float(past_c.std())
    if past_sd < 3:
        # Flat background: compare hue (chromaticity) instead of pattern.
        a = live_bgr[inside].mean(axis=0)
        b = past_bgr[inside].mean(axis=0)
        same_hue = float(np.abs(a / max(a.sum(), 1) - b / max(b.sum(), 1)).max()) < 0.02
        if not (same_hue and live_sd < 6):
            return False
        # A gray bag on a smooth gray floor also keeps the hue. What a bag
        # has and a soft shadow lacks is a crisp outline.
        return not _crisp_outline(frame, background, box, mask)
    if live_sd < 2:
        return False  # pattern vanished: something flat now covers it
    ncc = float((live_c * past_c).mean() / (live_sd * past_sd))
    return ncc >= min_ncc


def detect_left_items(
    path: str | Path,
    config: DetectorConfig | None = None,
    progress=None,
) -> DetectionResult:
    """Find items that were put down and stayed put."""
    config = config or DetectorConfig()
    media = Path(path)
    if not media.exists():
        raise DetectorError(f"media file not found: {media}")

    dt = 1.0 / config.sample_fps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    init_count = max(2, int(config.bg_init_seconds * config.sample_fps))

    active: list[ItemTrack] = []
    finished: list[ItemTrack] = []
    next_id = 1
    frames = 0
    offset = 0.0
    frame_size = (0, 0)
    resets: list[float] = []
    big_change_since: float | None = None
    ghosts = 0

    source = iter_frames(media, config)
    buffered: list[tuple[float, np.ndarray]] = []
    for item in source:
        buffered.append(item)
        if len(buffered) >= init_count:
            break
    if len(buffered) < 2:
        raise DetectorError("the video is too short to analyse (under half a second)")
    # Median over the opening seconds (thinned to at most 40 frames).
    step = max(1, len(buffered) // 40)
    init_frames = [cv2.GaussianBlur(f, (5, 5), 0).astype(np.float32) for _, f in buffered[::step]]
    long_bg = np.median(np.stack(init_frames), axis=0).astype(np.float32)
    short_bg = cv2.GaussianBlur(buffered[0][1], (5, 5), 0).astype(np.float32)
    static_age = np.zeros(long_bg.shape[:2], dtype=np.float32)
    activity = np.zeros(long_bg.shape[:2], dtype=np.float32)

    def all_frames():
        yield from buffered
        yield from source

    for offset, frame in all_frames():
        frames += 1
        height, width = frame.shape[:2]
        frame_size = (width, height)
        frame_area = width * height
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        assert short_bg is not None and static_age is not None
        changed_long = _colour_distance(blurred, long_bg) > config.diff_threshold
        moving = _colour_distance(blurred, short_bg) > config.diff_threshold

        # Camera moved or lights changed: rebuild instead of reporting the room.
        if changed_long.mean() > config.scene_change_ratio:
            if big_change_since is None:
                big_change_since = offset
            elif offset - big_change_since >= 2.0:
                long_bg = blurred.astype(np.float32)
                short_bg = long_bg.copy()
                static_age[:] = 0
                resets.append(offset)
                for track in active:
                    track.end_reason = "lost"
                    track.ended_offset_s = offset
                finished.extend(active)
                active = []
                big_change_since = None
                continue
        else:
            big_change_since = None

        activity += moving
        still = changed_long & ~moving
        occluded = changed_long & moving
        static_age = np.where(still, static_age + dt, np.where(occluded, static_age, 0.0))
        static_age = static_age.astype(np.float32)

        # Backgrounds learn. The long one only where the room is visible and
        # nothing has settled, so a left item is never absorbed into it.
        visible_bg = ~changed_long & (static_age == 0)
        long_bg[visible_bg] += config.long_alpha * (blurred[visible_bg] - long_bg[visible_bg])
        short_bg += config.short_alpha * (blurred.astype(np.float32) - short_bg)

        item_mask = (static_age >= config.static_seconds).astype(np.uint8)
        item_mask = cv2.morphologyEx(item_mask, cv2.MORPH_CLOSE, kernel_close)
        item_mask = cv2.morphologyEx(item_mask, cv2.MORPH_OPEN, kernel_open)
        item_boxes = _boxes(
            item_mask,
            int(config.min_area_ratio * frame_area),
            int(config.max_area_ratio * frame_area),
        )

        mover_mask = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
        mover_mask = cv2.dilate(mover_mask, kernel_close)
        mover_boxes = _boxes(mover_mask, int(config.mover_min_area_ratio * frame_area), frame_area)

        matched: set[int] = set()
        for box in item_boxes:
            best: ItemTrack | None = None
            best_score = 0.0
            for track in active:
                score = _iou(box, track.bbox)
                if score > best_score:
                    best, best_score = track, score
            if best is None or best_score < 0.2:
                x, y, w, h = box
                if _is_ghost(blurred, long_bg, item_mask, box, config.ghost_edge_ratio) or \
                        _only_light_changed(blurred, long_bg, box, config.light_ncc, item_mask):
                    region = item_mask[y:y + h, x:x + w].astype(bool)
                    long_bg[y:y + h, x:x + w][region] = blurred[y:y + h, x:x + w][region]
                    static_age[y:y + h, x:x + w][region] = 0
                    ghosts += 1
                    continue
                age = float(static_age[y:y + h, x:x + w].max())
                best = ItemTrack(
                    track_id=next_id,
                    placed_offset_s=max(0.0, offset - age),
                    detected_offset_s=offset,
                    frame_size=frame_size,
                )
                next_id += 1
                active.append(best)
            matched.add(best.track_id)

            others = [m for m in mover_boxes if _iou(m, box) < 0.6]
            distance = min((_gap(box, m) for m in others), default=None)
            x, y, w, h = box
            inside = float(moving[y:y + h, x:x + w].mean())
            best.points.append(TrackPoint(
                offset_s=offset,
                centre=(x + w / 2, y + h / 2),
                bbox=box,
                person_distance_ratio=None if distance is None else distance / width,
                internal_motion=inside,
            ))
            best.last_seen_offset_s = offset

            region = item_mask[y:y + h, x:x + w].astype(bool)
            if region.any() and not moving[y:y + h, x:x + w][region].any():
                pixels = frame[y:y + h, x:x + w][region]
                best.colour_samples.append(tuple(int(v) for v in np.median(pixels, axis=0)))
                if len(best.colour_samples) > 50:
                    best.colour_samples.pop(0)

        # Light can creep over an item's region after it was first accepted
        # (a sunbeam moving across the floor). Re-check every two seconds.
        if frames % max(1, int(config.sample_fps * 2)) == 0:
            for track in list(active):
                if track.track_id in matched and (
                    _is_ghost(blurred, long_bg, item_mask, track.bbox, config.ghost_edge_ratio)
                    or (_only_light_changed(blurred, long_bg, track.bbox, config.light_ncc, item_mask)
                        and not _has_object_edges(blurred, long_bg, track.bbox))
                ):
                    x, y, w, h = track.bbox
                    region = item_mask[y:y + h, x:x + w].astype(bool)
                    long_bg[y:y + h, x:x + w][region] = blurred[y:y + h, x:x + w][region]
                    static_age[y:y + h, x:x + w][region] = 0
                    active.remove(track)
                    matched.discard(track.track_id)
                    ghosts += 1

        still_active = []
        for track in active:
            if track.track_id in matched:
                still_active.append(track)
                continue
            if offset - track.last_seen_offset_s < config.lost_seconds:
                still_active.append(track)
                continue
            x, y, w, h = track.bbox
            changed_here = changed_long[y:y + h, x:x + w].mean()
            track.end_reason = "removed" if changed_here < 0.25 else "lost"
            track.ended_offset_s = track.last_seen_offset_s + dt
            finished.append(track)
        active = still_active

        if progress and frames % int(config.sample_fps * 30) == 0:
            progress(f"analysed {offset:.0f}s of video, {len(active)} item(s) in view")

    finished.extend(active)
    rate = activity / max(frames, 1)
    for track in finished:
        if not track.points:
            continue
        x, y, w, h = track.bbox
        pad = max(4, min(w, h) // 2)
        y0, y1 = max(0, y - pad), min(rate.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(rate.shape[1], x + w + pad)
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        track.surround_activity = float(rate[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
    tracks = [
        t for t in finished
        if len(t.points) >= 2 and t.last_seen_offset_s - t.detected_offset_s >= 1.0
    ]
    tracks.sort(key=lambda t: t.placed_offset_s)
    return DetectionResult(
        tracks=tracks,
        frames_analysed=frames,
        duration_s=offset + dt,
        frame_size=frame_size,
        scene_resets=resets,
        ghosts_healed=ghosts,
    )


def summary(result: DetectionResult) -> str:
    """One-line JSON summary, for logs."""
    return json.dumps({
        "frames": result.frames_analysed,
        "ghosts_healed": result.ghosts_healed,
        "duration_s": round(result.duration_s, 1),
        "tracks": [
            {
                "id": t.track_id,
                "placed": round(t.placed_offset_s, 1),
                "last_seen": round(t.last_seen_offset_s, 1),
                "end": t.end_reason,
                "colour": t.colour_name(),
                "bbox": t.bbox,
            }
            for t in result.tracks
        ],
    })
