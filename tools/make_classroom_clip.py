"""Render a long SIMULATED classroom clip with a forgotten gray jacket.

This is computer-drawn footage for demonstrating FoundIt when no real
footage is available. It is not evidence that the detector works on real
jackets; the staged shoot (WAI-267) is still needed for that. The output
file name and an on-screen corner label both say "SIMULATED".

Story (times from the start of the clip, default length 10 minutes):

  0:00  empty classroom, camera fixed at the back of the room
  0:12  students arrive one by one and sit down
  0:30  the student in the gray hooded jacket arrives at desk 1 (middle row),
        takes the jacket off, hangs it on the back of the chair and sits down
  0:50  the teacher arrives
  5:00  the bell: everyone leaves; the jacket stays on the chair
  5:35  the room is empty
  7:20  a cleaner sweeps along the far (window) side, then leaves
 10:00  end: the jacket has been alone for about 4.5 minutes

Usage:
  python tools/make_classroom_clip.py out.mp4 [--minutes 10] [--fps 15]
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

W, H = 960, 540
VP_Y = 210  # where the floor meets the front wall


# ---------------------------------------------------------------- geometry

def floor_xy(u: float, v: float) -> tuple[float, float]:
    """Floor point to screen. u: -1 (left wall) .. 1 (window wall); v: 0 front .. 1 camera."""
    y = VP_Y + v * (H - VP_Y)
    half = 280 + v * 200
    return 480 + u * half, y


def scale_at(v: float) -> float:
    return 0.55 + 0.75 * v


# ---------------------------------------------------------------- room

def draw_room(rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    # front wall
    img[:] = (196, 206, 212)
    # left wall and window wall (slightly darker / lighter)
    cv2.fillPoly(img, [np.array([[0, 0], [200, 30], [200, VP_Y], [0, H]])], (178, 188, 196))
    cv2.fillPoly(img, [np.array([[W, 0], [760, 30], [760, VP_Y], [W, H]])], (205, 214, 220))
    # ceiling band
    cv2.fillPoly(img, [np.array([[0, 0], [W, 0], [760, 30], [200, 30]])], (226, 230, 232))
    # floor with tiles
    floor = np.array([[200, VP_Y], [760, VP_Y], [W, H], [0, H]])
    cv2.fillPoly(img, [floor], (150, 160, 168))
    for i in range(1, 12):
        v = (i / 12) ** 1.4
        x0, y = floor_xy(-1, v)
        x1, _ = floor_xy(1, v)
        cv2.line(img, (int(x0), int(y)), (int(x1), int(y)), (138, 148, 156), 1)
    for u in np.linspace(-1, 1, 15):
        a = floor_xy(u, 0)
        b = floor_xy(u, 1)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (138, 148, 156), 1)
    # whiteboard + tray
    cv2.rectangle(img, (300, 62), (660, 150), (238, 240, 240), -1)
    cv2.rectangle(img, (300, 62), (660, 150), (120, 125, 130), 2)
    cv2.rectangle(img, (300, 150), (660, 155), (110, 115, 120), -1)
    cv2.line(img, (330, 90), (460, 90), (90, 60, 40), 2)       # faint old writing
    cv2.line(img, (330, 108), (520, 108), (90, 60, 40), 2)
    cv2.line(img, (540, 92), (620, 120), (40, 40, 160), 2)
    # clock (hands fixed: real second hands would be motion the camera sees)
    cv2.circle(img, (720, 70), 16, (245, 245, 245), -1)
    cv2.circle(img, (720, 70), 16, (60, 60, 60), 2)
    cv2.line(img, (720, 70), (720, 58), (30, 30, 30), 2)
    cv2.line(img, (720, 70), (729, 74), (30, 30, 30), 2)
    # room sign and noticeboard on the left wall
    cv2.fillPoly(img, [np.array([[40, 120], [150, 105], [150, 175], [40, 200]])], (60, 110, 160))
    for k in range(4):
        x = 55 + k * 22
        cv2.fillPoly(img, [np.array([[x, 128 - k * 3], [x + 14, 126 - k * 3], [x + 14, 150 - k * 3], [x, 153 - k * 3]])],
                     [(220, 220, 240), (200, 240, 220), (240, 220, 200), (230, 230, 200)][k])
    # windows on the right wall
    for k in range(3):
        top = 70 + k * 6
        x0 = 790 + k * 55
        cv2.fillPoly(img, [np.array([[x0, top], [x0 + 40, top - 8], [x0 + 40, top + 150 + k * 30], [x0, top + 140 + k * 28]])],
                     (235, 225, 205))
    # gentle texture so the floor and walls are not flat colour
    texture = cv2.resize(rng.normal(0, 4, (H // 6, W // 6, 1)).astype(np.float32), (W, H))
    img += texture[..., None]
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------- furniture

DESK_ROWS = [0.22, 0.47, 0.75]          # v of each desk row
DESK_COLS = [-0.62, -0.21, 0.21, 0.62]  # u of each desk column
CHAIR_DV = 0.07


def desk_points(u: float, v: float):
    s = scale_at(v)
    x, y = floor_xy(u, v)
    w, d = 120 * s, 34 * s
    top = y - 58 * s
    return x, y, s, w, d, top


def draw_desk(img, u, v):
    x, y, s, w, d, top = desk_points(u, v)
    for lx in (x - w / 2 + 6 * s, x + w / 2 - 6 * s):
        cv2.line(img, (int(lx), int(top)), (int(lx), int(y)), (70, 70, 74), max(1, int(3 * s)))
    cv2.fillPoly(img, [np.array([[x - w / 2, top], [x + w / 2, top], [x + w / 2 + 4 * s, top + d / 3],
                                 [x - w / 2 - 4 * s, top + d / 3]], np.int32)], (92, 132, 170))
    cv2.rectangle(img, (int(x - w / 2 - 4 * s), int(top + d / 3)), (int(x + w / 2 + 4 * s), int(top + d / 3 + 5 * s)),
                  (70, 100, 130), -1)


def chair_geom(u, v):
    cv = v + CHAIR_DV
    s = scale_at(cv)
    x, y = floor_xy(u, cv)
    return x, y, s


def draw_chair_seat(img, u, v):
    x, y, s = chair_geom(u, v)
    seat_y = y - 42 * s
    for lx in (x - 16 * s, x + 16 * s):
        cv2.line(img, (int(lx), int(seat_y)), (int(lx), int(y)), (60, 60, 64), max(1, int(2.5 * s)))
    cv2.rectangle(img, (int(x - 20 * s), int(seat_y - 4 * s)), (int(x + 20 * s), int(seat_y + 3 * s)), (150, 90, 40), -1)


def draw_chair_back(img, u, v):
    x, y, s = chair_geom(u, v)
    seat_y = y - 42 * s
    cv2.line(img, (int(x - 17 * s), int(seat_y)), (int(x - 17 * s), int(seat_y - 34 * s)), (60, 60, 64), max(1, int(2.5 * s)))
    cv2.line(img, (int(x + 17 * s), int(seat_y)), (int(x + 17 * s), int(seat_y - 34 * s)), (60, 60, 64), max(1, int(2.5 * s)))
    cv2.rectangle(img, (int(x - 20 * s), int(seat_y - 36 * s)), (int(x + 20 * s), int(seat_y - 20 * s)), (160, 98, 45), -1)


def draw_jacket_on_chair(img, u, v):
    """A gray hooded jacket hung over the chair back, sleeves hanging down."""
    x, y, s = chair_geom(u, v)
    seat_y = y - 42 * s
    top = seat_y - 40 * s
    body = np.array([[x - 23 * s, top + 4 * s], [x + 23 * s, top + 4 * s], [x + 25 * s, seat_y + 10 * s],
                     [x - 25 * s, seat_y + 10 * s]], np.int32)
    cv2.fillPoly(img, [body], (118, 118, 122))
    cv2.ellipse(img, (int(x), int(top + 4 * s)), (int(13 * s), int(8 * s)), 0, 180, 360, (105, 105, 110), -1)  # hood
    for side in (-1, 1):  # sleeves
        sx = x + side * 25 * s
        sleeve = np.array([[sx, top + 8 * s], [sx + side * 7 * s, top + 10 * s], [sx + side * 8 * s, seat_y + 22 * s],
                           [sx + side * 1 * s, seat_y + 22 * s]], np.int32)
        cv2.fillPoly(img, [sleeve], (108, 108, 112))
    cv2.line(img, (int(x), int(top + 8 * s)), (int(x), int(seat_y + 8 * s)), (80, 80, 85), max(1, int(1.5 * s)))  # zip
    cv2.line(img, (int(x - 22 * s), int(seat_y + 4 * s)), (int(x + 22 * s), int(seat_y + 4 * s)), (95, 95, 100), max(1, int(2 * s)))


# ---------------------------------------------------------------- people

@dataclass
class Person:
    shirt: tuple
    trousers: tuple
    hair: tuple
    skin: tuple
    path: list = field(default_factory=list)   # (t, u, v, mode) mode: walk|stand|sit
    jacket_off_at: float | None = None
    seed: float = 0.0

    def state(self, t: float):
        if not self.path or t < self.path[0][0] or t > self.path[-1][0]:
            return None
        for (t0, u0, v0, m0), (t1, u1, v1, _m1) in zip(self.path, self.path[1:]):
            if t0 <= t <= t1:
                k = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                return u0 + k * (u1 - u0), v0 + k * (v1 - v0), m0, (u0, v0) != (u1, v1)
        t_last, u, v, m = self.path[-1]
        return u, v, m, False

    def wearing_jacket(self, t: float) -> bool:
        return self.jacket_off_at is not None and t < self.jacket_off_at


def draw_standing(img, p: Person, u, v, t, moving):
    s = scale_at(v)
    x, y = floor_xy(u, v)
    phase = t * 7 + p.seed
    swing = math.sin(phase) * (9 * s if moving else 1.5 * s)
    bob = abs(math.sin(phase)) * (2 * s if moving else 0)
    sway = math.sin(t * 1.3 + p.seed) * 2.5 * s
    x += sway
    hip = y - 78 * s - bob
    for side, k in ((-1, 1), (1, -1)):
        foot_x = x + side * 6 * s + k * swing
        cv2.line(img, (int(x + side * 6 * s), int(hip)), (int(foot_x), int(y)), p.trousers, max(2, int(10 * s)))
    torso_col = (118, 118, 122) if p.wearing_jacket(t) else p.shirt
    shoulder = hip - 62 * s
    cv2.rectangle(img, (int(x - 17 * s), int(shoulder)), (int(x + 17 * s), int(hip + 4 * s)), torso_col, -1)
    if p.wearing_jacket(t):
        cv2.ellipse(img, (int(x), int(shoulder + 2 * s)), (int(12 * s), int(7 * s)), 0, 180, 360, (105, 105, 110), -1)
    arm_col = torso_col
    for side in (-1, 1):
        hand_x = x + side * 21 * s - side * swing * 0.6
        cv2.line(img, (int(x + side * 19 * s), int(shoulder + 6 * s)), (int(hand_x), int(hip - 4 * s)), arm_col, max(2, int(8 * s)))
        cv2.circle(img, (int(hand_x), int(hip - 2 * s)), max(1, int(4 * s)), p.skin, -1)
    head_y = shoulder - 13 * s
    cv2.circle(img, (int(x), int(head_y)), max(2, int(12 * s)), p.skin, -1)
    cv2.circle(img, (int(x), int(head_y - 2 * s)), max(2, int(12 * s)), p.hair, -1)  # seen from behind: mostly hair


def draw_seated(img, p: Person, u, v, t):
    """Seated at the chair behind desk (u, v), seen from behind. Real people
    never sit perfectly still: they lean, turn their heads and write."""
    x, y, s = chair_geom(u, v)
    seat_y = y - 42 * s
    lean = math.sin(t * 0.9 + p.seed) * 4 * s + math.sin(t * 2.3 + p.seed * 2) * 2 * s
    turn = math.sin(t * 0.5 + p.seed * 3) * 5 * s
    write = math.sin(t * 6 + p.seed) * 3 * s if math.sin(t * 0.2 + p.seed) > 0 else 0
    torso_col = (118, 118, 122) if p.wearing_jacket(t) else p.shirt
    shoulder = seat_y - 52 * s
    cx = x + lean
    cv2.rectangle(img, (int(cx - 17 * s), int(shoulder)), (int(cx + 17 * s), int(seat_y)), torso_col, -1)
    if p.wearing_jacket(t):
        cv2.ellipse(img, (int(cx), int(shoulder + 2 * s)), (int(12 * s), int(7 * s)), 0, 180, 360, (105, 105, 110), -1)
    for side in (-1, 1):  # arms forward onto the desk
        hx = cx + side * 15 * s + (write if side == 1 else 0)
        cv2.line(img, (int(cx + side * 18 * s), int(shoulder + 6 * s)), (int(hx), int(shoulder + 30 * s)), torso_col, max(2, int(7 * s)))
    head_x, head_y = cx + turn, shoulder - 12 * s
    cv2.circle(img, (int(head_x), int(head_y)), max(2, int(12 * s)), p.skin, -1)
    cv2.circle(img, (int(head_x - turn * 0.3), int(head_y - 2 * s)), max(2, int(12 * s)), p.hair, -1)


# ---------------------------------------------------------------- story

def build_story(minutes: float):
    end = minutes * 60
    bell = min(300.0, end * 0.5)
    rng = np.random.default_rng(7)
    # Only the jacket is gray, so a search for "gray" has one right answer.
    shirts = [(170, 70, 40), (40, 130, 60), (60, 60, 180), (40, 160, 210), (190, 110, 70), (120, 40, 140), (30, 110, 200)]
    trousers = [(50, 40, 30), (40, 40, 40), (90, 70, 50), (30, 30, 60)]
    hairs = [(20, 20, 25), (30, 40, 60), (15, 15, 15), (40, 60, 90)]
    skins = [(150, 175, 215), (110, 140, 185), (80, 105, 145), (170, 190, 225)]

    seats = [(c, r) for r in range(3) for c in range(4)]
    taken = [(1, 0), (2, 0), (0, 1), (3, 1), (1, 1), (2, 2), (0, 2), (3, 2)]  # (col, row)
    owner_seat = (0, 1)
    people: list[Person] = []
    arrival = 12.0
    order = [s for s in taken if s != owner_seat]
    order.insert(3, owner_seat)
    aisle = {0: -0.42, 1: -0.42, 2: 0.42, 3: 0.42}
    for index, (col, row) in enumerate(order):
        u_seat, v_desk = DESK_COLS[col], DESK_ROWS[row]
        v_chair = v_desk + CHAIR_DV
        a_u = aisle[col]
        t0 = arrival + index * 4.5
        walk_in = 4 + (1 - v_chair) * 6
        leave_start = bell + 3 + index * 3.2
        p = Person(
            shirt=shirts[index % len(shirts)], trousers=trousers[index % len(trousers)],
            hair=hairs[index % len(hairs)], skin=skins[index % len(skins)], seed=float(rng.uniform(0, 6)),
        )
        p.path = [
            (t0, a_u, 1.12, "walk"),
            (t0 + walk_in, a_u, v_chair + 0.02, "walk"),
            (t0 + walk_in + 1.2, u_seat, v_chair + 0.02, "walk"),
            (t0 + walk_in + 2.0, u_seat, v_chair + 0.02, "sit"),
        ]
        if (col, row) == owner_seat:
            # Arrives in the gray jacket, takes it off straight away, hangs it on the chair.
            p.shirt = (235, 235, 238)        # white shirt under the jacket
            seated_at = t0 + walk_in + 1.2
            p.jacket_off_at = seated_at + 3.0
            p.path[-1] = (seated_at + 0.1, u_seat, v_chair + 0.02, "stand")
            p.path += [
                (seated_at + 5.0, u_seat, v_chair + 0.02, "sit"),
            ]
        p.path += [
            (leave_start, u_seat, v_chair + 0.02, "stand"),
            (leave_start + 1.5, a_u, v_chair + 0.02, "walk"),
            (leave_start + 1.5 + (1.12 - v_chair) * 6, a_u, 1.15, "walk"),
        ]
        people.append(p)

    teacher = Person(shirt=(90, 50, 30), trousers=(40, 40, 45), hair=(30, 30, 30), skin=(120, 150, 195), seed=1.3)
    teacher.path = [(50, 0.0, 1.12, "walk"), (58, 0.0, 0.08, "walk")]
    t = 58.0
    k = 0
    while t < bell + 25:
        target = [-0.35, 0.3, -0.1, 0.4, 0.0][k % 5]
        teacher.path.append((t + 6, target, 0.08, "walk"))
        teacher.path.append((t + 18, target, 0.08, "stand"))
        t += 18
        k += 1
    teacher.path += [(t + 2, 0.0, 0.08, "walk"), (t + 9, 0.0, 1.15, "walk")]
    people.append(teacher)

    if end > bell + 120:
        cleaner = Person(shirt=(40, 110, 150), trousers=(60, 60, 60), hair=(25, 25, 25), skin=(110, 140, 180), seed=2.2)
        c0 = bell + 140
        cleaner.path = [
            (c0, 0.88, 1.12, "walk"), (c0 + 9, 0.88, 0.12, "walk"), (c0 + 20, 0.86, 0.12, "stand"),
            (c0 + 23, 0.5, 0.06, "walk"), (c0 + 30, 0.5, 0.06, "stand"), (c0 + 33, 0.88, 0.12, "walk"),
            (c0 + 43, 0.88, 1.15, "walk"),
        ]
        people.append(cleaner)

    owner = next(p for p in people if p.jacket_off_at is not None)
    return people, owner_seat, owner, bell, end


# ---------------------------------------------------------------- render

def render(out: str, minutes: float = 10.0, fps: int = 15) -> dict:
    rng = np.random.default_rng(11)
    room = draw_room(rng)
    with_desks = room.copy()
    for row in range(3):
        for col in range(4):
            draw_desk(with_desks, DESK_COLS[col], DESK_ROWS[row])
    people, owner_seat, owner, bell, end = build_story(minutes)
    noise_bank = [rng.normal(0, 3.0, (H, W, 3)).astype(np.float32) for _ in range(12)]
    window_mask = np.zeros((H, W), np.float32)
    cv2.fillPoly(window_mask, [np.array([[W, 0], [700, 30], [700, H], [W, H]])], 1.0)
    window_mask = cv2.GaussianBlur(window_mask, (151, 151), 0)[..., None]

    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    jacket_col, jacket_row = owner_seat
    frames = int(end * fps)
    for n in range(frames):
        t = n / fps
        frame = with_desks.copy()
        # the sun slowly brightens the window side during the second half
        sun = 0.06 * max(0.0, min(1.0, (t - end * 0.55) / (end * 0.4)))
        if sun:
            frame *= 1 + sun * window_mask

        drawables = []
        for row in range(3):
            for col in range(4):
                v = DESK_ROWS[row]
                drawables.append((v + CHAIR_DV - 0.001, "seat", (col, row)))
                drawables.append((v + CHAIR_DV + 0.001, "back", (col, row)))
        for p in people:
            st = p.state(t)
            if st is None:
                continue
            u, v, mode, moving = st
            if mode == "sit":
                col = min(range(4), key=lambda c: abs(DESK_COLS[c] - u))
                row = min(range(3), key=lambda r: abs(DESK_ROWS[r] + CHAIR_DV - v))
                drawables.append((DESK_ROWS[row] + CHAIR_DV, "seated", (p, col, row)))
            else:
                drawables.append((v + 0.005, "standing", (p, u, v, moving)))
        drawables.sort(key=lambda d: d[0])
        for _, kind, data in drawables:
            if kind == "seat":
                draw_chair_seat(frame, DESK_COLS[data[0]], DESK_ROWS[data[1]])
            elif kind == "back":
                draw_chair_back(frame, DESK_COLS[data[0]], DESK_ROWS[data[1]])
                if data == (jacket_col, jacket_row) and t >= owner.jacket_off_at:
                    draw_jacket_on_chair(frame, DESK_COLS[jacket_col], DESK_ROWS[jacket_row])
            elif kind == "seated":
                p, col, row = data
                draw_seated(frame, p, DESK_COLS[col], DESK_ROWS[row], t)
            else:
                p, u, v, moving = data
                draw_standing(frame, p, u, v, t, moving)

        frame = cv2.GaussianBlur(frame, (3, 3), 0) + noise_bank[n % len(noise_bank)]
        out_frame = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.putText(out_frame, "SIMULATED", (W - 132, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        encoder.stdin.write(out_frame.tobytes())
    encoder.stdin.close()
    encoder.wait()
    return {
        "jacket_placed_s": owner.jacket_off_at,
        "owner_leaves_s": next(t for t, *_ in owner.path if t > bell),
        "bell_s": bell,
        "duration_s": end,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    print(render(args.out, args.minutes, args.fps))
