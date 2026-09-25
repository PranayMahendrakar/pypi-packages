"""Result objects: one :class:`Event`, and the :class:`EventReport` for a sequence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EVENT_KINDS: Tuple[str, ...] = ("sudden_motion", "fall", "crowding", "abandoned")
"""Every kind of event this package looks for, in report order."""

CAVEAT = (
    "these are motion heuristics, not activity recognition: every event is a "
    "hypothesis for a person to review, not a verdict"
)

_WHAT = {
    "sudden_motion": "a large, abrupt jump in the moving area",
    "fall": "a moving region whose height collapsed while its width held, then stayed low",
    "crowding": "the share of the view in motion rose past a level and stayed there",
    "abandoned": "a new region appeared and stayed still for the dwell time",
}


def format_time(value: float, unit: str) -> str:
    """``4.0 s`` on a seconds clock, ``frame 40`` on a frame clock."""
    if unit == "frames":
        return "frame {}".format(int(round(value)))
    return "{} s".format(_number(value))


def format_span(value: float, unit: str) -> str:
    """A duration: ``2.0 s`` or ``20 frames``."""
    if unit == "frames":
        count = int(round(value))
        return "{} frame{}".format(count, "" if count == 1 else "s")
    return "{} s".format(_number(value))


def format_range(start: float, end: float, unit: str) -> str:
    """``4.0-7.9 s`` or ``frames 40-79``."""
    if unit == "frames":
        a, b = int(round(start)), int(round(end))
        return "frame {}".format(a) if a == b else "frames {}-{}".format(a, b)
    if abs(end - start) < 1e-9:
        return "{} s".format(_number(start))
    return "{}-{} s".format(_number(start), _number(end))


def _number(value: float) -> str:
    if abs(value - round(value, 1)) < 1e-6:
        return "{:.1f}".format(value)
    if abs(value - round(value, 2)) < 1e-6:
        return "{:.2f}".format(value)
    return "{:.3f}".format(value)


def _clean(value: Any) -> Any:
    """Make a value JSON-safe: plain floats, no NaN or infinity, tuples as lists."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(float(value), 4)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    try:  # numpy scalars
        return _clean(value.item())
    except AttributeError:
        return str(value)


@dataclass
class Event:
    """One detected event: a hypothesis about what happened, and where.

    Attributes:
        kind: ``sudden_motion``, ``fall``, ``crowding`` or ``abandoned``.
        start: when it began, on the detector's clock (seconds when timestamps
            or ``fps`` were given, otherwise frame numbers).
        end: the last time it was still true. While an event is ongoing the
            detector keeps moving ``end`` forward on this same object.
        confidence: 0-1 strength of the evidence. It is a heuristic score, not
            a calibrated probability.
        region: ``(x, y, width, height)`` in the pixels of the first frame, or
            None when the event has no single place.
        message: a plain-language account of what was measured.
    """

    kind: str
    start: float
    end: float
    confidence: float
    region: Optional[Tuple[int, int, int, int]]
    message: str

    @property
    def duration(self) -> float:
        """``end - start`` on the detector's clock."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary of the event."""
        region = None
        if self.region is not None:
            x, y, w, h = self.region
            region = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        return {
            "kind": self.kind,
            "start": _clean(float(self.start)),
            "end": _clean(float(self.end)),
            "duration": _clean(float(self.duration)),
            "confidence": _clean(float(self.confidence)),
            "region": region,
            "message": self.message,
        }

    def describe(self, unit: str = "seconds") -> str:
        """One line: kind, time range, confidence and region."""
        where = ""
        if self.region is not None:
            x, y, w, h = self.region
            where = "  at x={} y={} w={} h={}".format(x, y, w, h)
        return "{:<14} {:<16} confidence {:.2f}{}".format(
            self.kind, format_range(self.start, self.end, unit), self.confidence, where
        )


@dataclass
class EventReport:
    """Everything :func:`detect_events` (or :meth:`Detector.report`) found.

    Attributes:
        events: every event, in the order it was confirmed.
        frames: frames read.
        time_unit: ``"seconds"`` or ``"frames"`` - the clock of every time here.
        fps: the frame rate used to turn frame numbers into seconds, if any.
        start: time of the first frame; ``end`` of the last. None when no frames.
        frame_size: ``(width, height)`` of the first frame.
        working_size: ``(width, height)`` of the image the rules ran on.
        background: how far the background model got (``state`` is
            ``"learned"``, ``"learning"`` or ``"no frames"``).
        camera_motion_frames: frames on which the whole picture moved; they were
            treated as camera shake and not used for scene rules.
        analysed_frames: frames the rules actually ran on.
        notes: plain-language remarks about the input and the run.
        settings: the thresholds in force.
        stats: sequence-level measurements (moving share, lighting drift ...).
        source: where the frames came from, when known.
    """

    events: List[Event] = field(default_factory=list)
    frames: int = 0
    time_unit: str = "frames"
    fps: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None
    frame_size: Optional[Tuple[int, int]] = None
    working_size: Optional[Tuple[int, int]] = None
    background: Dict[str, Any] = field(default_factory=dict)
    camera_motion_frames: int = 0
    analysed_frames: int = 0
    notes: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    @property
    def by_kind(self) -> Dict[str, List[Event]]:
        """Events grouped by kind; every kind is present, possibly empty."""
        out: Dict[str, List[Event]] = {kind: [] for kind in EVENT_KINDS}
        for event in self.events:
            out.setdefault(event.kind, []).append(event)
        return out

    @property
    def timeline(self) -> List[Tuple[float, str]]:
        """``(start time, kind)`` for every event, in time order."""
        order = {kind: i for i, kind in enumerate(EVENT_KINDS)}
        pairs = [(float(e.start), e.kind) for e in self.events]
        return sorted(pairs, key=lambda p: (p[0], order.get(p[1], len(order)), p[1]))

    @property
    def found(self) -> bool:
        """True when at least one event was reported."""
        return bool(self.events)

    def summary(self) -> str:
        """A human-readable account, plain ASCII punctuation."""
        unit = self.time_unit
        lines: List[str] = []
        count = len(self.events)
        span = ""
        if self.start is not None and self.end is not None:
            span = ", " + format_range(self.start, self.end, unit)
            if self.fps:
                span += " at {} fps".format(_number(float(self.fps)).rstrip("0").rstrip("."))
        lines.append(
            "video events: {} found in {} frame{}{}".format(
                count, self.frames, "" if self.frames == 1 else "s", span
            )
        )
        if self.source:
            lines.append("source: {}".format(self.source))
        for event in sorted(self.events, key=lambda e: (e.start, e.kind)):
            lines.append("  " + event.describe(unit))
            lines.append("      " + event.message)
        state = self.background.get("state")
        if state == "learned":
            missing = [kind for kind in EVENT_KINDS if not self.by_kind.get(kind)]
            if missing:
                lines.append("looked for, not seen: {}".format(", ".join(missing)))
        lines.append("background: {}".format(self.background.get("text", "not started")))
        if self.camera_motion_frames:
            lines.append(
                "camera: the whole picture moved on {} frame{} (largest shift {:.1f} px); "
                "treated as camera shake, not scene motion, and not used for events".format(
                    self.camera_motion_frames,
                    "" if self.camera_motion_frames == 1 else "s",
                    float(self.stats.get("largest_camera_shift_px", 0.0) or 0.0),
                )
            )
        if self.notes:
            lines.append("notes:")
            for note in self.notes:
                lines.append("  - " + note)
        lines.append(CAVEAT)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary of the whole report."""
        return {
            "found": self.found,
            "events": [event.to_dict() for event in self.events],
            "by_kind": {kind: len(events) for kind, events in self.by_kind.items()},
            "timeline": [[_clean(t), kind] for t, kind in self.timeline],
            "frames": int(self.frames),
            "analysed_frames": int(self.analysed_frames),
            "time_unit": self.time_unit,
            "fps": _clean(self.fps),
            "start": _clean(self.start),
            "end": _clean(self.end),
            "frame_size": _clean(list(self.frame_size)) if self.frame_size else None,
            "working_size": _clean(list(self.working_size)) if self.working_size else None,
            "background": _clean(self.background),
            "camera_motion_frames": int(self.camera_motion_frames),
            "notes": list(self.notes),
            "settings": _clean(self.settings),
            "stats": _clean(self.stats),
            "source": self.source,
            "kinds": {kind: _WHAT[kind] for kind in EVENT_KINDS},
            "caveat": CAVEAT,
        }
