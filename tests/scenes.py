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
