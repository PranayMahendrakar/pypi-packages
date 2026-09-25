"""Synthetic frame sequences, made with numpy only: nothing is downloaded.

Every scene is a fixed background (flat or textured) plus sensor noise from a
seeded generator, with rectangles drawn on top: ones that move, one that
collapses from tall to wide, one that is put down and left.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

H, W = 120, 160


def texture(
    height: int = H, width: int = W, seed: int = 1, margin: int = 0, mean: float = 110.0, spread: float = 25.0
) -> np.ndarray:
    """A smooth random texture, by default mean 110 and spread 25 grey levels."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (height + 2 * margin, width + 2 * margin))
    for _ in range(3):
        padded = np.pad(base, 1, mode="edge")
        rows = padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]
        base = (rows[:-2] + rows[1:-1] + rows[2:]) / 9.0
    base = (base - base.mean()) / base.std()
    return mean + spread * base


def flat(level: float = 70.0, height: int = H, width: int = W) -> np.ndarray:
    """A plain background."""
    return np.full((height, width), float(level))


def _finish(stack: np.ndarray, rng: np.random.Generator, sigma: float) -> List[np.ndarray]:
    """Add seeded sensor noise to a float stack and return uint8 frames."""
    stack += rng.standard_normal(stack.shape, dtype=np.float32) * np.float32(sigma)
    np.round(stack, out=stack)
    np.clip(stack, 0, 255, out=stack)
    return list(stack.astype(np.uint8))


def noisy(background: np.ndarray, count: int, seed: int = 0, sigma: float = 2.0) -> List[np.ndarray]:
    """``count`` uint8 frames of ``background`` plus seeded Gaussian noise."""
    stack = np.repeat(background[None].astype(np.float32), count, axis=0)
    return _finish(stack, np.random.default_rng(seed), sigma)


def draw(frame: np.ndarray, top: int, left: int, height: int, width: int, value: float) -> None:
    """Paint a rectangle into ``frame`` in place (only on our own test frames)."""
    t, l = max(0, top), max(0, left)
    b, r = min(frame.shape[0], top + height), min(frame.shape[1], left + width)
    if b > t and r > l:
        frame[t:b, l:r] = value


def scene(
    background: np.ndarray,
    count: int,
    boxes: Sequence[Sequence[Optional[Tuple[int, int, int, int, float]]]],
    seed: int = 0,
    sigma: float = 2.0,
) -> List[np.ndarray]:
    """Frames with, per frame, a list of ``(top, left, height, width, value)`` boxes."""
    stack = np.repeat(background[None].astype(np.float32), count, axis=0)
    for index in range(min(count, len(boxes))):
        for box in boxes[index]:
            if box is not None:
                top, left, height, width, value = box
                draw(stack[index], top, left, height, width, value)
    return _finish(stack, np.random.default_rng(seed), sigma)


def empty_boxes(count: int) -> List[list]:
    """A per-frame box list with nothing in it."""
    return [[] for _ in range(count)]


def fall_sequence(seed: int = 3, textured: bool = False, fall: bool = True, fps: float = 10.0):
    """A 'person' walks in, collapses from 36x12 to 12x36 over three frames, lies still.

    With ``fall=False`` it just keeps walking (the control).
    """
    count = 95
    background = texture(seed=seed) if textured else flat(60.0)
    boxes = empty_boxes(count)
    bottom = 100
    for i in range(40, count):
        step = i - 40
        if not fall or i < 61:
            left = 20 + 2 * min(step, 45)
            boxes[i].append((bottom - 36, left, 36, 12, 185.0))
        else:
            left = 20 + 2 * 20
            shape = {61: (26, 20), 62: (16, 28), 63: (12, 36)}.get(i, (12, 36))
            h, w = shape
            boxes[i].append((bottom - h, left, h, w, 185.0))
    return scene(background, count, boxes, seed=seed), fps


def crouch_sequence(seed: int = 4):
    """The same walker dips to a crouch for two frames and stands straight back up."""
    count = 90
    background = flat(60.0)
    boxes = empty_boxes(count)
    bottom = 100
    for i in range(40, count):
        left = 20 + 2 * min(i - 40, 45)
        height = {60: 22, 61: 14, 62: 14, 63: 24}.get(i, 36)
        boxes[i].append((bottom - height, left, height, 14, 185.0))
    return scene(background, count, boxes, seed=seed)


def abandoned_sequence(seed: int = 5, pick_up_at: Optional[int] = None, count: int = 110):
    """A walker carries a bag in, puts it down at frame 50 and walks off to the right.

    With ``pick_up_at`` the bag disappears again at that frame.
    """
    background = flat(80.0)
    boxes = empty_boxes(count)
    for i in range(35, count):
        walker_left = 10 + 3 * (i - 35)
        boxes[i].append((50, walker_left, 40, 12, 200.0))
        if i < 50:
            boxes[i].append((78, walker_left + 12, 10, 10, 30.0))  # carried at the side
        elif pick_up_at is None or i < pick_up_at:
            boxes[i].append((78, 10 + 3 * 15 + 12, 10, 10, 30.0))  # left on the floor
    return scene(background, count, boxes, seed=seed)


def removed_sequence(seed: int = 6, count: int = 100):
    """An object sits in view through learning and is taken away at frame 45."""
    background = flat(90.0)
    boxes = empty_boxes(count)
    for i in range(count):
        if i < 45:
            boxes[i].append((40, 60, 20, 24, 20.0))
    return scene(background, count, boxes, seed=seed)


def crowd_sequence(seed: int = 7, people: int = 12, count: int = 90):
    """``people`` rectangles drift in one every two frames and wander about."""
    background = flat(70.0)
    rng = np.random.default_rng(seed)
    boxes = empty_boxes(count)
    starts = [36 + 2 * k for k in range(people)]
    pos = [(float(rng.integers(5, H - 25)), float(rng.integers(5, W - 25))) for _ in range(people)]
    vel = [(float(rng.choice([-1.5, 1.5])), float(rng.choice([-2.0, 2.0]))) for _ in range(people)]
    for i in range(count):
        for k in range(people):
            if i < starts[k]:
                continue
            y, x = pos[k]
            vy, vx = vel[k]
            y, x = y + vy, x + vx
            if y < 2 or y > H - 22:
                vy, y = -vy, min(max(y, 2), H - 22)
            if x < 2 or x > W - 18:
                vx, x = -vx, min(max(x, 2), W - 18)
            pos[k], vel[k] = (y, x), (vy, vx)
            boxes[i].append((int(y), int(x), 20, 16, 190.0 if k % 2 else 150.0))
    return scene(background, count, boxes, seed=seed)


def sudden_sequence(seed: int = 8, big: bool = True, count: int = 80):
    """A calm scene until frame 50, when a large (or small) block bursts in and moves."""
    background = texture(seed=seed)
    boxes = empty_boxes(count)
    for i in range(50, count):
        if big:
            boxes[i].append((25 + (i - 50) % 6, 30 + 2 * (i - 50), 50, 60, 230.0))
        else:
            boxes[i].append((50, 20 + 2 * (i - 50), 10, 8, 230.0))
    return scene(background, count, boxes, seed=seed)


def shaken_sequence(seed: int = 9, count: int = 80, bump: bool = False):
    """A still textured scene seen by a camera that shakes for five frames."""
    margin = 12
    big = texture(seed=seed, margin=margin)
    rng = np.random.default_rng(seed)
    shakes = {45: (3, 2), 46: (-2, 4), 47: (4, -3), 48: (-3, -2), 49: (2, 3)}
    stack = np.empty((count, H, W), dtype=np.float32)
    for i in range(count):
        if bump and i >= 45:
            dy, dx = (3, -2)
        else:
            dy, dx = shakes.get(i, (0, 0))
        stack[i] = big[margin + dy: margin + dy + H, margin + dx: margin + dx + W]
    return _finish(stack, rng, 2.0)


def lighting_sequence(seed: int = 10, count: int = 100, textured: bool = True):
    """A still scene whose lighting rises steadily by about 60 grey levels (gain and offset)."""
    background = texture(seed=seed, mean=80.0, spread=18.0) if textured else flat(60.0)
    rng = np.random.default_rng(seed)
    stack = np.empty((count,) + background.shape, dtype=np.float32)
    for i in range(count):
        ramp = max(0.0, (i - 35) / float(count - 35))
        stack[i] = background * (1.0 + 0.4 * ramp) + 30.0 * ramp
    return _finish(stack, rng, 2.0)
