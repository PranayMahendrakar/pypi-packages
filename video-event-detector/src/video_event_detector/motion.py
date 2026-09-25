"""The running background model and the per-frame motion measurement.

Per frame, after the background has been learned:

1. **Camera motion.** If much of the frame differs from the background, a
   phase correlation asks whether the whole picture simply moved. A shift is
   accepted only when undoing it removes most of the difference; the frame is
   then flagged as camera motion and not used for scene rules at all, so a
   shaking camera cannot raise a crowd of events.
2. **Lighting.** A robust gain-and-offset fit maps the frame onto the
   background's brightness, so a sunrise or a dimmer is not read as motion.
3. **Foreground** is where the lighting-matched frame differs from the
   background by more than a noise-scaled threshold; **changed** is where it
   differs from the previous frame. A foreground region with no changed pixels
   is still; one with changed pixels is moving.
4. **Update.** The background follows the scene only where nothing is in
   front of it, so a parked object is not absorbed in a few frames.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ._imageops import box3, clean, phase_correlate, shift_image

log = logging.getLogger(__name__)

MIN_LEVEL = 8.0
"""Smallest difference, in 0-255 grey levels, ever treated as motion (at sensitivity 0.5)."""

NOISE_FLOOR = 0.5
"""Lowest noise estimate used, in grey levels of the 3x3-smoothed difference."""


def _median(values: np.ndarray) -> float:
    """Median of a large array from every third value: robust, and three times cheaper."""
    flat = values.ravel()
    if flat.size > 3000:
        flat = flat[::3]
    return float(np.median(flat))


@dataclass
class FrameMotion:
    """What the motion model measured on one frame."""

    ready: bool
    camera_motion: bool = False
    shift: Tuple[float, float] = (0.0, 0.0)
    reregistered: bool = False
    frame: Optional[np.ndarray] = None       # lighting-matched frame, background coordinates
    raw: Optional[np.ndarray] = None         # the working frame as it arrived (aligned)
    foreground: Optional[np.ndarray] = None  # differs from the background
    changed: Optional[np.ndarray] = None     # differs from the previous frame
    difference: Optional[np.ndarray] = None  # |smoothed frame - background|
    threshold: float = 0.0
    gain: float = 1.0
    offset: float = 0.0


@dataclass
class MotionStats:
    """Running totals kept for the report."""

    camera_frames: int = 0
    largest_shift: float = 0.0
    reregistrations: List[int] = field(default_factory=list)
    max_offset: float = 0.0
    max_gain_change: float = 0.0


class MotionModel:
    """A median-initialised, selectively updated background with alignment.

    Args:
        frames_needed: frames used to learn the first background (their median).
        scale: threshold multiplier from the detector's sensitivity (1.0 at 0.5).
    """

    def __init__(self, frames_needed: int, scale: float) -> None:
        self.frames_needed = int(frames_needed)
        self.scale = float(scale)
        self.rate = 1.0 / max(2.0, float(frames_needed))
        self.background: Optional[np.ndarray] = None
        self.noise = NOISE_FLOOR
        self.previous: Optional[np.ndarray] = None
        self.learned_at: Optional[int] = None
        self.stats = MotionStats()
        self._buffer: List[np.ndarray] = []
        self._base_level: Optional[float] = None
        self._bg_level: Optional[float] = None
        self._tick = 0
        self.camera_gate = 0.04 * self.scale
        self._shift_run: List[Tuple[float, float]] = []
        self.persist = max(5, self.frames_needed // 3)

    # -- learning ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        """True once the background has been learned."""
        return self.background is not None

    @property
    def frames_seen_learning(self) -> int:
        """Frames collected towards the first background so far."""
        return self.frames_needed if self.ready else len(self._buffer)

    def _finish_learning(self, index: int) -> None:
        stack = np.stack(self._buffer)
        background = np.median(stack, axis=0).astype(np.float32)
        spreads = []
        for frame in self._buffer:
            diff = np.abs(box3(frame - background))
            spreads.append(float(np.median(diff)) / 0.6745)
        self.background = background
        self._base_level = float(np.median(background))
        self.noise = max(NOISE_FLOOR, float(np.median(spreads)))
        self.previous = self._buffer[-1]
        self._buffer = []
        self.learned_at = index
        log.debug("background learned at frame %d, noise %.2f", index, self.noise)

    # -- thresholds -------------------------------------------------------
    @property
    def threshold(self) -> float:
        """Grey-level difference from the background that counts as foreground."""
        return self.scale * max(MIN_LEVEL, 4.0 * self.noise)

    @property
    def change_threshold(self) -> float:
        """Grey-level difference from the previous frame that counts as change."""
        # Frame-to-frame differences carry the noise of two frames, not one.
        return self.scale * max(MIN_LEVEL, 4.0 * math.sqrt(2.0) * self.noise)

    # -- per frame --------------------------------------------------------
    def step(self, gray: np.ndarray, index: int) -> FrameMotion:
        """Measure one working frame against the background."""
        if self.background is None:
            self._buffer.append(gray)
            self.previous = gray
            if len(self._buffer) >= self.frames_needed:
                self._finish_learning(index)
            return FrameMotion(ready=False)

        background = self.background
        threshold = self.threshold
        rows, cols = gray.shape

        self._tick += 1
        plain = gray - background
        centre = _median(plain)
        raw_diff = box3(plain)
        busy = float(np.mean(np.abs(raw_diff - centre) > threshold))

        aligned = gray
        # Only a frame that differs over a good part of the view can be camera
        # motion; below this share a shift could not raise an event anyway.
        if busy > self.camera_gate and min(rows, cols) >= 16:
            shift = self._camera_shift(gray, raw_diff, centre)
            if shift is not None:
                return self._camera_frame(gray, shift)
        self._shift_run = []

        gain, offset = self._lighting(aligned, background, threshold, plain, centre)
        matched = (aligned - offset) / gain if gain != 1.0 else aligned - offset
        if self._base_level is not None:
            if self._bg_level is None or self._tick % 10 == 0:
                self._bg_level = _median(background)
            drift = abs(gain * self._bg_level + offset - self._base_level)
            self.stats.max_offset = max(self.stats.max_offset, drift)
        self.stats.max_gain_change = max(self.stats.max_gain_change, abs(gain - 1.0))

        difference = np.abs(box3(matched - background))
        foreground = clean(difference > threshold)
        share = float(foreground.mean())
        if share < 0.3 and self._tick % 4 == 1:
            quiet = difference[~foreground]
            if quiet.size:
                estimate = max(NOISE_FLOOR, _median(quiet) / 0.6745)
                self.noise = 0.95 * self.noise + 0.05 * estimate

        previous = self.previous if self.previous is not None else matched
        changed = clean(np.abs(box3(matched - previous)) > self.change_threshold)
        self.previous = matched
        return FrameMotion(
            ready=True,
            frame=matched,
            raw=aligned,
            foreground=foreground,
            changed=changed,
            difference=difference,
            threshold=threshold,
            gain=gain,
            offset=offset,
        )

    def _camera_shift(
        self, gray: np.ndarray, raw_diff: np.ndarray, centre: float
    ) -> Optional[Tuple[float, float]]:
        """The global shift of this frame, if one explains most of the difference."""
        background = self.background
        assert background is not None
        rows, cols = gray.shape
        dy, dx, peak = phase_correlate(background, gray)
        size = math.hypot(dy, dx)
        limit = 0.15 * min(rows, cols)
        if size < 0.5 or size > limit or peak < 0.05:
            return None
        moved, valid = shift_image(gray, -dy, -dx)
        if valid.sum() < 0.25 * rows * cols:
            return None
        before = float(np.mean(np.abs(raw_diff - centre)[valid]))
        after_diff = box3(moved - background)
        after = float(np.mean(np.abs(after_diff - float(np.median(after_diff[valid])))[valid]))
        if before <= 1e-6 or after > 0.6 * before:
            return None
        return (dy, dx)

    def _camera_frame(self, gray: np.ndarray, shift: Tuple[float, float]) -> FrameMotion:
        """Book-keeping for a frame on which the camera, not the scene, moved."""
        dy, dx = shift
        self.stats.camera_frames += 1
        self.stats.largest_shift = max(self.stats.largest_shift, math.hypot(dy, dx))
        run = self._shift_run
        run.append(shift)
        reregistered = False
        if len(run) >= self.persist:
            ys = [s[0] for s in run[-self.persist:]]
            xs = [s[1] for s in run[-self.persist:]]
            if max(ys) - min(ys) <= 1.0 and max(xs) - min(xs) <= 1.0:
                # The camera has settled in a new position: move the background
                # with it instead of flagging every frame from now on.
                my, mx = float(np.mean(ys)), float(np.mean(xs))
                assert self.background is not None
                moved, valid = shift_image(self.background, my, mx)
                self.background = np.where(valid, moved, gray).astype(np.float32)
                self.previous = gray
                self._shift_run = []
                reregistered = True
        return FrameMotion(ready=True, camera_motion=True, shift=shift, reregistered=reregistered)

    @staticmethod
    def _lighting(
        frame: np.ndarray, background: np.ndarray, threshold: float, diff: np.ndarray, offset: float
    ) -> Tuple[float, float]:
        """Robust ``frame ~= gain * background + offset`` over background-like pixels.

        ``diff`` is ``frame - background`` and ``offset`` its median. Every third
        pixel is plenty for two numbers and keeps this cheap.
        """
        step = 3 if diff.size > 3000 else 1
        d = diff.ravel()[::step]
        inliers = np.abs(d - offset) < 3.0 * threshold
        if inliers.sum() < 0.2 * d.size:
            return 1.0, offset
        base = background.ravel()[::step][inliers].astype(np.float64)
        seen = frame.ravel()[::step][inliers].astype(np.float64)
        spread = float(base.std())
        if spread < 4.0:
            return 1.0, float(np.median(seen - base))
        gain = float(np.mean((base - base.mean()) * (seen - seen.mean())) / (spread * spread))
        gain = min(2.0, max(0.5, gain))
        if abs(gain - 1.0) < 0.01:
            return 1.0, float(np.median(seen - base))
        return gain, float(np.median(seen - gain * base))

    def update(self, motion: FrameMotion, absorb: Optional[np.ndarray] = None) -> None:
        """Let the background follow the scene where nothing is in front of it.

        Args:
            motion: the measurement returned by :meth:`step` for this frame.
            absorb: pixels to copy straight into the background (long-still
                regions, or a hole left by an object that was taken away).
        """
        if self.background is None or motion.foreground is None or motion.raw is None:
            return
        background = self.background
        free = ~motion.foreground
        background[free] += self.rate * (motion.raw[free] - background[free])
        if absorb is not None and absorb.any():
            background[absorb] = motion.raw[absorb]
