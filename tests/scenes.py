"""Synthetic test footage with known ground truth.

Not a stand-in for real footage — CAVIAR and the staged jacket shoot are
that — but a way to pin down the detector's behaviour in tests: every clip
here has an exact answer. The scenes include the things that break naive
detectors: sensor noise, slow lighting drift, people who sway while
standing, and people who walk past without leaving anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 384, 288, 25


@dataclass
class Person:
    path: list[tuple[float, float, float]]
    """(time_s, x, y) waypoints of the feet; linear motion between them."""
    colour: tuple[int, int, int] = (60, 40, 30)

    def where(self, t: float) -> tuple[float, float] | None:
        if t < self.path[0][0] or t > self.path[-1][0]:
            return None
        for (t0, x0, y0), (t1, x1, y1) in zip(self.path, self.path[1:]):
            if t0 <= t <= t1:
                k = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                return x0 + k * (x1 - x0), y0 + k * (y1 - y0)
        return self.path[-1][1], self.path[-1][2]


@dataclass
class Item:
    placed_at: float
    removed_at: float | None
    x: int
    y: int
    w: int = 26
    h: int = 20
    colour: tuple[int, int, int] = (105, 105, 110)  # BGR, a gray bag


def _room(rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((H, W, 3), np.float32)
    ys = np.linspace(0, 1, H)[:, None]
    base[..., 0] = 150 + 30 * ys   # beige floor, lighter towards camera
    base[..., 1] = 175 + 25 * ys
    base[..., 2] = 195 + 20 * ys
    base[: H // 3] = (200, 205, 210)  # back wall
    texture = rng.normal(0, 6, (H // 4, W // 4, 1)).astype(np.float32)
    base += cv2.resize(texture, (W, H))[..., None]
    cv2.rectangle(base, (40, 120), (110, 150), (70, 90, 120), -1)    # bench
    cv2.rectangle(base, (290, 140), (340, 200), (60, 70, 90), -1)    # chair
    return np.clip(base, 0, 255)


def render(
    out: Path,
    duration: float,
    people: list[Person],
    items: list[Item],
    seed: int = 1,
    carried_by: dict[int, int] | None = None,
) -> Path:
    """Write a clip. ``carried_by`` maps item index -> person index: the
    item travels beside that person while not placed."""
    rng = np.random.default_rng(seed)
    room = _room(rng)
    carried_by = carried_by or {}
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for n in range(int(duration * FPS)):
        t = n / FPS
        frame = room * (1 + 0.03 * np.sin(t / 7))  # slow lighting drift
        for index, item in enumerate(items):
            placed = item.placed_at <= t and (item.removed_at is None or t < item.removed_at)
            if placed:
                cv2.rectangle(frame, (item.x, item.y), (item.x + item.w, item.y + item.h),
                              item.colour, -1)
            elif index in carried_by:
                where = people[carried_by[index]].where(t)
                if where:
                    px, py = where
                    x, y = int(px + 14), int(py - 30)
                    cv2.rectangle(frame, (x, y), (x + item.w, y + item.h), item.colour, -1)
        for person in people:
            where = person.where(t)
            if not where:
                continue
            px, py = where
            sway = 2.5 * np.sin(t * 3.1 + px)
            cx = int(px + sway)
            cv2.ellipse(frame, (cx, int(py - 40)), (13, 38), 0, 0, 360, person.colour, -1)
            cv2.circle(frame, (cx, int(py - 86)), 9, (90, 120, 170), -1)
        frame = frame + rng.normal(0, 4, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    return out


def left_behind(out: Path) -> Path:
    """Someone puts a bag down at ~6 s, waits, walks away at ~9 s. It stays."""
    walker = Person([(1, -20, 250), (6, 190, 250), (9, 190, 250), (14, 420, 250)])
    bag = Item(placed_at=6, removed_at=None, x=200, y=225)
    return render(out, 60, [walker], [bag], carried_by={0: 0})


def attended(out: Path) -> Path:
    """Someone puts a bag down and stays right beside it the whole clip."""
    stays = Person([(1, -20, 250), (6, 190, 250), (30, 186, 250), (60, 192, 250)])
    bag = Item(placed_at=6, removed_at=None, x=200, y=225)
    return render(out, 60, [stays], [bag], carried_by={0: 0})


def picked_up(out: Path) -> Path:
    """Left at ~6 s, collected again at ~40 s."""
    walker = Person([
        (1, -20, 250), (6, 190, 250), (9, 190, 250), (14, 420, 250),
        (34, 420, 250), (39, 190, 250), (41, 190, 250), (47, -40, 250),
    ])
    bag = Item(placed_at=6, removed_at=40, x=200, y=225)
    return render(out, 55, [walker], [bag], carried_by={0: 0})


def passers_by(out: Path) -> Path:
    """Three people walk through. Nothing is left."""
    people = [
        Person([(1, -20, 250), (9, 420, 250)]),
        Person([(5, 420, 230), (14, -30, 230)], colour=(30, 30, 90)),
        Person([(12, -20, 270), (19, 420, 270)], colour=(90, 60, 20)),
    ]
    return render(out, 25, people, [])


def busy_start(out: Path) -> Path:
    """What broke the first version on real footage: someone is standing in
    shot when recording starts and then walks off (their spot must not
    become a fake item), and later a small dark bag is left by somebody
    else, who walks away at ~16 s."""
    stander = Person([(0, 90, 200), (4, 92, 200), (9, -40, 200)], colour=(40, 60, 40))
    walker = Person([(0, 420, 270), (8, -30, 270)], colour=(30, 30, 90))
    leaver = Person([(8, 420, 240), (12, 260, 240), (15, 260, 240), (20, 420, 240)])
    bag = Item(placed_at=12, removed_at=None, x=270, y=222, w=14, h=10, colour=(35, 35, 40))
    return render(out, 50, [stander, walker, leaver], [bag], carried_by={0: 2})


def sunlight(out: Path) -> Path:
    """A soft-edged patch of floor brightens over three seconds (sun through
    a window) and stays bright; another darkens (a shadow). Separately a
    dark bag is left at ~10 s. Only the bag is an item."""
    rng = np.random.default_rng(3)
    room = _room(rng)
    leaver = Person([(5, 420, 250), (10, 250, 250), (13, 250, 250), (18, 420, 250)])
    bag = Item(placed_at=10, removed_at=None, x=262, y=226, w=18, h=13, colour=(40, 35, 35))
    # Window light has soft edges at this distance: blur the patches.
    light = np.zeros((H, W), np.float32)
    light[150:220, 60:150] = 0.35    # sun patch
    light[100:140, 200:260] = -0.30  # shadow
    light = cv2.GaussianBlur(light, (31, 31), 0)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for n in range(int(45 * FPS)):
        t = n / FPS
        frame = room.copy()
        k = min(1.0, max(0.0, (t - 15) / 3))
        frame *= (1 + k * light)[..., None]
        if t >= bag.placed_at:
            cv2.rectangle(frame, (bag.x, bag.y), (bag.x + bag.w, bag.y + bag.h), bag.colour, -1)
        where = leaver.where(t)
        if where:
            px, py = where
            if t < bag.placed_at:
                cv2.rectangle(frame, (int(px + 14), int(py - 30)), (int(px + 32), int(py - 17)), bag.colour, -1)
            cx = int(px + 2.5 * np.sin(t * 3.1))
            cv2.ellipse(frame, (cx, int(py - 40)), (13, 38), 0, 0, 360, leaver.colour, -1)
            cv2.circle(frame, (cx, int(py - 86)), 9, (90, 120, 170), -1)
        frame = frame + rng.normal(0, 4, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    return out
