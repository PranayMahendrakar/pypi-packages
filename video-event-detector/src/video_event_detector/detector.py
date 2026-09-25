"""The :class:`Detector`: feed it frames one at a time, get events as they are confirmed."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ._frames import FramePrep
from ._imageops import label
from .events import Event, EventReport, format_range, format_time
from .motion import MotionModel
from .rules import (
    AbandonedRule,
    CrowdingRule,
    FallRule,
    Region,
    Scene,
    Span,
    SuddenMotionRule,
)

log = logging.getLogger(__name__)

MOVING_CHANGE = 0.03
"""A foreground region with at least this share of changed pixels is moving."""


def _scale(sensitivity: float) -> float:
    """Threshold multiplier: 2.0 at sensitivity 0, 1.0 at 0.5, 0.5 at 1."""
    return float(2.0 ** ((0.5 - sensitivity) * 2.0))


def _duration(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a number or None; got {!r}".format(name, value))
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("{} must be a finite number >= 0; got {}".format(name, value))
    return value


class Detector:
    """Streaming motion-event detector over a running background model.

    Args:
        sensitivity: 0-1. Higher finds weaker motion and smaller events and
            raises more false alarms; 0.5 is the default balance.
        background_frames: frames used to learn the empty scene before any
            event can be reported (their per-pixel median). Also sets the pace
            at which the background follows slow changes.
        dwell: how long a new region must stay still to be reported as
            ``abandoned``, on the detector's clock (seconds when you pass
            timestamps, frame numbers otherwise). None means
            ``background_frames`` frames.
        hold: how long crowding, or a collapsed shape, must last before it is
            reported (same clock). None means ``max(3, background_frames // 6)``
            frames.

    Call :meth:`update` once per frame, in order. It returns the events
    confirmed on that frame; the detector keeps extending ``end`` on those same
    objects while the condition lasts. :attr:`events` holds all of them.
    """

    def __init__(
        self,
        *,
        sensitivity: float = 0.5,
        background_frames: int = 30,
        dwell: Optional[float] = None,
        hold: Optional[float] = None,
    ) -> None:
        if isinstance(sensitivity, bool) or not isinstance(sensitivity, (int, float)):
            raise ValueError("sensitivity must be a number from 0 to 1; got {!r}".format(sensitivity))
        if not (0.0 <= float(sensitivity) <= 1.0):
            raise ValueError("sensitivity must be from 0 to 1; got {}".format(sensitivity))
        if isinstance(background_frames, bool) or not isinstance(background_frames, (int, np.integer)):
            raise ValueError("background_frames must be a whole number >= 1; got {!r}".format(background_frames))
        if int(background_frames) < 1:
            raise ValueError("background_frames must be >= 1; got {}".format(background_frames))
        self.sensitivity = float(sensitivity)
        self.background_frames = int(background_frames)
        self.dwell = _duration(dwell, "dwell")
        self.hold = _duration(hold, "hold")

        scale = _scale(self.sensitivity)
        self._scale = scale
        bf = self.background_frames
        self._dwell = Span(clock=self.dwell) if self.dwell is not None else Span(frames=bf)
        self._hold = Span(clock=self.hold) if self.hold is not None else Span(frames=max(3, bf // 6))
        self._window = max(3, bf // 3)
        self._jump = min(0.9, 0.08 * scale)
        self._crowd = min(0.8, 0.20 * scale)
        self._fall_ratio = 0.45 + 0.2 * self.sensitivity

        self._prep = FramePrep()
        self._motion = MotionModel(bf, scale)
        self._rules_ready = False
        self._events: List[Event] = []
        self._index = -1
        self._clock: Optional[str] = None
        self._first_time: Optional[float] = None
        self._last_time: Optional[float] = None
        self._learned_time: Optional[float] = None
        self._analysed = 0
        self._share_peak = 0.0
        self._share_sum = 0.0
        self._reregistered: List[float] = []
        self._skipped = 0
        self._patched = 0
        self._source: Optional[str] = None
        self._fps: Optional[float] = None

    # -- set-up that needs the first frame --------------------------------
    def _build_rules(self) -> None:
        rows, cols = self._prep.working_shape  # type: ignore[misc]
        area = rows * cols
        self._min_area = max(6, int(round(max(12.0, 0.001 * area) * self._scale)))
        self._sudden = SuddenMotionRule(self._jump, self._hold, history=max(5, self._window))
        self._crowding = CrowdingRule(self._crowd, self._hold)
        self._falls = FallRule(
            self._fall_ratio,
            self._hold,
            self._window,
            min_height=max(6, int(round(0.08 * rows))),
        )
        self._abandoned = AbandonedRule(
            self._dwell,
            grace=self._hold,
            min_area=self._min_area,
            max_share=0.35,
            absorb_frames=10 * self.background_frames,
        )
        self._rules_ready = True

    # -- the clock --------------------------------------------------------
    def _time_for(self, timestamp: Optional[float]) -> float:
        index = self._index + 1
        clock = self._clock or ("frames" if timestamp is None else "seconds")
        if clock == "frames":
            if timestamp is not None:
                raise ValueError(
                    "the first frame had no timestamp, so this detector counts frames; "
                    "pass a timestamp for every frame or for none"
                )
            return float(index)
        if timestamp is None:
            raise ValueError(
                "earlier frames had timestamps; pass a timestamp for every frame or for none"
            )
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float, np.integer, np.floating)):
            raise ValueError("timestamp must be a number of seconds; got {!r}".format(timestamp))
        value = float(timestamp)
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite; got {}".format(timestamp))
        if self._last_time is not None and value < self._last_time:
            raise ValueError(
                "timestamps must not go backwards: got {} after {}".format(value, self._last_time)
            )
        return value

    # -- per frame --------------------------------------------------------
    def update(self, frame: Any, *, timestamp: Optional[float] = None) -> List[Event]:
        """Analyse the next frame; return the events confirmed on it.

        Args:
            frame: a numpy array (grey or colour), a PIL image, or an image path.
            timestamp: seconds, for every frame or for none. Without them the
                detector's clock is the frame number.

        Raises:
            ValueError: unreadable frame, or timestamps missing or going backwards.
        """
        return self._update(frame, timestamp, None)

    def _update(self, frame: Any, timestamp: Optional[float], name: Optional[str]) -> List[Event]:
        time = self._time_for(timestamp)
        gray = self._prep(frame, self._index + 1, name)
        self._index += 1
        index = self._index
        if self._clock is None:
            self._clock = "frames" if timestamp is None else "seconds"
            self._first_time = time
        self._last_time = time
        if not self._rules_ready:
            self._build_rules()

        unknown = ~np.isfinite(gray)
        if unknown.any():
            if unknown.mean() > 0.5:
                self._skipped += 1
                return []
            self._patched += 1
            if self._motion.background is not None:
                fill = self._motion.background
            elif self._motion.previous is not None:
                fill = self._motion.previous
            else:
                fill = np.full(gray.shape, float(np.nanmedian(gray)), dtype=np.float32)
            gray = np.where(unknown, fill, gray).astype(np.float32)

        was_ready = self._motion.ready
        motion = self._motion.step(gray, index)
        if not was_ready:
            if self._motion.ready:
                self._learned_time = time
            return []
        if motion.reregistered:
            self._reregistered.append(time)
            self._sudden.reset()
            self._crowding.reset()
            self._falls.reset()
            self._abandoned.reset()
        if motion.camera_motion:
            return []

        assert motion.foreground is not None and motion.changed is not None
        assert motion.frame is not None and self._motion.background is not None
        self._analysed += 1
        labels, blobs = label(motion.foreground)
        regions: List[Region] = []
        if blobs:
            changed_per = np.bincount(
                labels[motion.changed & (labels >= 0)], minlength=len(blobs)
            )
        moving_mask = np.zeros(gray.shape, dtype=bool)
        moving_labels = []
        for blob in blobs:
            if blob.area < self._min_area:
                continue
            changed = int(changed_per[blob.label])
            moving = changed >= max(2, MOVING_CHANGE * blob.area)
            regions.append(Region(blob=blob, changed=changed, moving=moving))
            if moving:
                moving_labels.append(blob.label)
        if moving_labels:
            moving_mask = np.isin(labels, np.array(moving_labels, dtype=np.int32))
        share = float(moving_mask.mean())
        self._share_peak = max(self._share_peak, share)
        self._share_sum += share

        scene = Scene(
            index=index,
            time=time,
            unit="seconds" if self._clock == "seconds" else "frames",
            frame=motion.frame,
            background=self._motion.background,
            foreground=motion.foreground,
            changed=motion.changed,
            labels=labels,
            regions=regions,
            moving_mask=moving_mask,
            moving_share=share,
            threshold=motion.threshold,
            factor=self._prep.factor,
            frame_size=self._prep.frame_size,  # type: ignore[arg-type]
        )
        raised: List[Event] = []
        raised += self._sudden.update(scene)
        raised += self._crowding.update(scene)
        raised += self._falls.update(scene, regions)
        found, absorb = self._abandoned.update(scene, self._falls.active_boxes())
        raised += found
        self._motion.update(motion, absorb)
        for event in raised:
            log.debug("%s at %s", event.kind, event.start)
        self._events.extend(raised)
        return raised

    # -- results ----------------------------------------------------------
    @property
    def events(self) -> List[Event]:
        """Every event confirmed so far, in the order it was confirmed."""
        return list(self._events)

    def _unit(self) -> str:
        return "seconds" if self._clock == "seconds" else "frames"

    def _background(self) -> Dict[str, Any]:
        unit = self._unit()
        needed = self.background_frames
        if self._index < 0:
            return {"state": "no frames", "frames_needed": needed, "frames_seen": 0,
                    "text": "no frames were given, so nothing was analysed"}
        seen = self._motion.frames_seen_learning
        if not self._motion.ready:
            suggestion = max(1, (self._index + 1) // 3)
            return {
                "state": "learning",
                "frames_needed": needed,
                "frames_seen": seen,
                "learned_at": None,
                "text": (
                    "STILL LEARNING - it has seen {} of the {} frames it needs to learn the "
                    "empty scene, so no events could be reported yet; give more frames, or "
                    "pass background_frames={} for a clip this short".format(seen, needed, suggestion)
                ),
            }
        learned = self._learned_time
        first = self._first_time if self._first_time is not None else 0.0
        return {
            "state": "learned",
            "frames_needed": needed,
            "frames_seen": seen,
            "learned_at": learned,
            "text": "learned from the first {} frame{} ({}); events can start after {}".format(
                needed,
                "" if needed == 1 else "s",
                format_range(first, learned if learned is not None else first, unit),
                format_time(learned if learned is not None else first, unit),
            ),
        }

    def _notes(self) -> List[str]:
        unit = self._unit()
        notes = list(self._prep.notes())
        if self._rules_ready:
            rule = self._abandoned
            if rule.removed:
                notes.append(
                    "{} still change{} looked like something taken away rather than left "
                    "behind (the edges were in the background, not the frame); not reported "
                    "as abandoned".format(rule.removed, "s" if rule.removed != 1 else "")
                )
            if rule.oversize:
                notes.append(
                    "{} still change{} covered more than 35% of the view - more likely the "
                    "scene or the lighting changed than an object was left; not reported as "
                    "abandoned".format(rule.oversize, "s" if rule.oversize != 1 else "")
                )
            if rule.skipped_as_fall:
                notes.append(
                    "a still region that was part of a reported fall was not reported again "
                    "as abandoned"
                )
        if self._skipped:
            notes.append(
                "{} frame{} had no usable pixels (mostly NaN or infinite) and {} skipped".format(
                    self._skipped, "s" if self._skipped != 1 else "", "were" if self._skipped != 1 else "was"
                )
            )
        if self._patched:
            notes.append(
                "{} frame{} held some NaN or infinite pixels; those pixels were treated as "
                "unchanged".format(self._patched, "s" if self._patched != 1 else "")
            )
        stats = self._motion.stats
        if stats.max_offset >= 10.0:
            notes.append(
                "overall brightness changed by up to {:.0f} grey levels; it was matched as "
                "lighting, not counted as motion".format(stats.max_offset)
            )
        for time in self._reregistered:
            notes.append(
                "the camera appears to have moved to a new position at {}; the background "
                "was moved with it and the rules restarted".format(format_time(time, unit))
            )
        if self._index >= 0 and stats.camera_frames > 0.5 * max(1, self._index + 1 - self.background_frames):
            notes.append(
                "the camera moved on most frames; this package expects a fixed camera and "
                "did not analyse frames on which the whole picture moved"
            )
        if self._prep.colour:
            notes.append("colour frames were read as brightness only")
        return notes

    def report(self) -> EventReport:
        """A snapshot :class:`EventReport` of everything seen so far."""
        stats: Dict[str, Any] = {
            "peak_moving_share": self._share_peak,
            "mean_moving_share": self._share_sum / self._analysed if self._analysed else 0.0,
            "largest_camera_shift_px": self._motion.stats.largest_shift * self._prep.factor,
            "max_brightness_change": self._motion.stats.max_offset,
            "noise_level": self._motion.noise,
            "pixel_threshold": self._motion.threshold,
        }
        unit = self._unit()
        settings: Dict[str, Any] = {
            "sensitivity": self.sensitivity,
            "background_frames": self.background_frames,
            "dwell": self._dwell.to_dict(),
            "hold": self._hold.to_dict(),
            "collapse_window_frames": self._window,
            "sudden_jump_share": self._jump,
            "crowding_share": self._crowd,
            "fall_height_ratio": self._fall_ratio,
            "min_region_px": (self._min_area * self._prep.factor ** 2) if self._rules_ready else None,
            "dwell_text": self._dwell.describe(unit),
            "hold_text": self._hold.describe(unit),
        }
        return EventReport(
            events=[copy.copy(e) for e in self._events],
            frames=self._index + 1,
            time_unit=unit,
            fps=self._fps,
            start=self._first_time,
            end=self._last_time,
            frame_size=self._prep.frame_size,
            working_size=self._prep.working_size,
            background=self._background(),
            camera_motion_frames=self._motion.stats.camera_frames,
            analysed_frames=self._analysed,
            notes=self._notes(),
            settings=settings,
            stats=stats,
            source=self._source,
        )

    def summary(self) -> str:
        """Human-readable account of everything seen so far."""
        return self.report().summary()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary of everything seen so far."""
        return self.report().to_dict()


__all__ = ["Detector"]
