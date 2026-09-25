"""``Counter``: counts per frame of a stream, and objects crossing a line.

Objects are followed from frame to frame by nearest-centroid matching: each new
detection is paired with the closest track of the same label that is within
``max_distance`` pixels, closest pairs first. A track crossing a counting line
is counted once per line, however long it lingers on the line or however often
it wobbles back and forth over it.

This is simple tracking. It is right when objects move less than about their
own size between frames and do not pass through each other. Two objects that
swap places between frames, or one that vanishes for longer than
``max_missed`` frames, can be miscounted.
"""
from __future__ import annotations

import math
from collections import Counter as _Tally
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._core import Detector, _RegionCache, check_options, run_count
from ._region import parse_region
from ._result import METHOD_CLASSICAL, CountResult

Point = Tuple[float, float]


def _point(value: Any, name: str) -> Point:
    try:
        x, y = value
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an (x, y) pair of numbers, got {value!r}") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x, y


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


@dataclass
class _Line:
    name: str
    a: Point
    b: Point
    forward: int = 0
    backward: int = 0
    by_label: _Tally = field(default_factory=_Tally)
    recrossed: int = 0

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    def signed(self, p: Point) -> float:
        """Signed distance in pixels; positive is to the right of a->b as seen on screen."""
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        return (dx * (p[1] - self.a[1]) - dy * (p[0] - self.a[0])) / self.length

    def within(self, p: Point, q: Point, margin: float) -> bool:
        """Does the move p -> q cross the line inside its two end points?"""
        sp, sq = self.signed(p), self.signed(q)
        if sp == sq:
            return False
        t = sp / (sp - sq)
        cx = p[0] + t * (q[0] - p[0])
        cy = p[1] + t * (q[1] - p[1])
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        u = ((cx - self.a[0]) * dx + (cy - self.a[1]) * dy) / (self.length ** 2)
        slack = margin / self.length
        return -slack <= u <= 1.0 + slack

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": [self.a[0], self.a[1]],
            "b": [self.b[0], self.b[1]],
            "crossed": self.forward + self.backward,
            "forward": self.forward,
            "backward": self.backward,
            "by_label": dict(sorted(self.by_label.items(), key=lambda kv: (-kv[1], kv[0]))),
            "recrossed": self.recrossed,
        }


@dataclass
class _Track:
    id: int
    label: str
    centre: Point
    size: float
    first_frame: int
    last_frame: int
    sides: Dict[str, Tuple[int, Point]] = field(default_factory=dict)
    counted: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Count objects frame by frame, and objects crossing lines.

    ``Counter(detector=None, min_area=None, max_area=None, region=None,
    max_distance=None, max_missed=3, line_margin=2.0)``. The first four mean
    the same as in :func:`count`. ``max_distance`` is the furthest (pixels) an
    object may move between frames and still be the same object; by default it
    is 1.5 times the median box diagonal. ``max_missed`` is how many frames a
    track survives unseen. ``line_margin`` is how far (pixels) past a line an
    object's centre must be to be on the other side.
    """

    def __init__(self, detector: Optional[Detector] = None, *, min_area: Optional[float] = None,
                 max_area: Optional[float] = None, region: Any = None,
                 max_distance: Optional[float] = None, max_missed: int = 3,
                 line_margin: float = 2.0) -> None:
        self.detector, self.min_area, self.max_area = check_options(detector, min_area, max_area)
        if max_distance is not None:
            if isinstance(max_distance, bool) or not isinstance(max_distance, (int, float)) \
                    or not max_distance > 0:
                raise ValueError(f"max_distance must be a positive number of pixels, got {max_distance!r}")
            max_distance = float(max_distance)
        if isinstance(max_missed, bool) or not isinstance(max_missed, int) or max_missed < 0:
            raise ValueError(f"max_missed must be a whole number >= 0, got {max_missed!r}")
        if isinstance(line_margin, bool) or not isinstance(line_margin, (int, float)) \
                or not line_margin >= 0:
            raise ValueError(f"line_margin must be a number of pixels >= 0, got {line_margin!r}")
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.line_margin = float(line_margin)
        self._regions = _RegionCache(parse_region(region))
        self._lines: Dict[str, _Line] = {}
        self.reset()

    # ------------------------------------------------------------------ setup
    def line(self, a: Any, b: Any, name: Optional[str] = None) -> "Counter":
        """Add a counting line from point ``a`` to point ``b`` (x, y in pixels); returns self.

        Crossing it from the left of a->b to the right (as seen on screen, y down)
        is ``forward``; the other way is ``backward``. For a line drawn left to
        right, moving down the image is forward.
        """
        pa, pb = _point(a, "a"), _point(b, "b")
        if pa == pb:
            raise ValueError("a counting line needs two different points")
        if name is None:
            name = f"line {len(self._lines) + 1}"
        name = str(name)
        if name in self._lines:
            raise ValueError(f"there is already a line called {name!r}")
        self._lines[name] = _Line(name, pa, pb)
        for track in self._tracks:
            self._init_side(track, self._lines[name])
        return self

    def reset(self) -> None:
        """Forget every frame, track and crossing; keep the lines and options."""
        self._tracks: List[_Track] = []
        self._next_id = 1
        self.history: List[CountResult] = []
        self.frames = 0
        self.failed_frames = 0
        self.tracks_started = 0
        for ln in self._lines.values():
            ln.forward = ln.backward = ln.recrossed = 0
            ln.by_label = _Tally()

    # ------------------------------------------------------------------ results
    @property
    def totals(self) -> Dict[str, int]:
        """Objects counted crossing each line, both directions together."""
        return {name: ln.forward + ln.backward for name, ln in self._lines.items()}

    @property
    def lines(self) -> Dict[str, Dict[str, Any]]:
        """Per line: end points, crossed, forward, backward and by_label."""
        return {name: ln.to_dict() for name, ln in self._lines.items()}

    @property
    def peak(self) -> int:
        """The most things counted in any one frame."""
        return max((r.count for r in self.history if r.ok), default=0)

    # ------------------------------------------------------------------ work
    def _init_side(self, track: _Track, ln: _Line) -> None:
        s = ln.signed(track.centre)
        if abs(s) > self.line_margin:
            track.sides[ln.name] = (1 if s > 0 else -1, track.centre)

    def _gate(self, sizes: List[float]) -> float:
        if self.max_distance is not None:
            return self.max_distance
        if not sizes:
            return 0.0
        return max(4.0, 1.5 * float(np.median(sizes)))

    def update(self, frame: Any) -> CountResult:
        """Count one frame, follow the objects in it, and update the line totals."""
        result = run_count(frame, self.detector, self.min_area, self.max_area, self._regions)
        index = self.frames
        self.frames += 1
        result.frame = index
        result.crossings = {name: 0 for name in self._lines}
        if not result.ok:
            # Tracks are kept as they were: a failed frame is missing, not empty.
            self.failed_frames += 1
            self.history.append(result)
            return result

        centres = [(float(c[0]), float(c[1])) for c in result.centroids]
        sizes = [math.hypot(b[2] - b[0], b[3] - b[1]) for b in result.boxes]
        gate = self._gate(sizes + [t.size for t in self._tracks])
        assignment: Dict[int, int] = {}
        if self._tracks and centres:
            tc = np.array([t.centre for t in self._tracks], dtype=np.float64)
            dc = np.array(centres, dtype=np.float64)
            dist = np.hypot(tc[:, None, 0] - dc[None, :, 0], tc[:, None, 1] - dc[None, :, 1])
            gaps = np.array([max(1, index - t.last_frame) for t in self._tracks], dtype=np.float64)
            same = (np.array([t.label for t in self._tracks], dtype=object)[:, None]
                    == np.array(result.labels, dtype=object)[None, :])
            ti_all, di_all = np.nonzero(same & (dist <= gate * gaps[:, None]))
            track_ids = np.array([t.id for t in self._tracks], dtype=np.int64)[ti_all]
            order = np.lexsort((di_all, track_ids, dist[ti_all, di_all]))
            used_t, used_d = set(), set()
            for k in order:
                ti, di = int(ti_all[k]), int(di_all[k])
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                assignment[di] = ti

        ids: List[int] = []
        for di, centre in enumerate(centres):
            if di in assignment:
                track = self._tracks[assignment[di]]
                track.centre = centre
                track.size = sizes[di]
                track.last_frame = index
                for ln in self._lines.values():
                    if self._step(track, ln, centre):
                        result.crossings[ln.name] += 1
            else:
                track = _Track(self._next_id, result.labels[di], centre, sizes[di], index, index)
                self._next_id += 1
                self.tracks_started += 1
                for ln in self._lines.values():
                    self._init_side(track, ln)
                self._tracks.append(track)
            ids.append(track.id)
        self._tracks = [t for t in self._tracks if index - t.last_frame <= self.max_missed]
        result.track_ids = ids
        self.history.append(result)
        return result

    def _step(self, track: _Track, ln: _Line, centre: Point) -> bool:
        s = ln.signed(centre)
        if abs(s) <= self.line_margin:
            return False            # on the line: wait until it is clearly on one side
        side = 1 if s > 0 else -1
        before = track.sides.get(ln.name)
        track.sides[ln.name] = (side, centre)
        if before is None or before[0] == side:
            return False
        if not ln.within(before[1], centre, self.line_margin):
            return False            # passed beyond the end of the line
        if ln.name in track.counted:
            ln.recrossed += 1       # the same object again: never counted twice
            return False
        direction = "forward" if side > 0 else "backward"
        track.counted[ln.name] = direction
        if side > 0:
            ln.forward += 1
        else:
            ln.backward += 1
        ln.by_label[track.label] += 1
        return True

    # ------------------------------------------------------------------ text
    def summary(self) -> str:
        """Human-readable account of the stream so far, plain ASCII."""
        ok = [r for r in self.history if r.ok]
        method = ok[0].method if ok else ("detector" if self.detector else METHOD_CLASSICAL)
        how = "classical blob counter" if method == METHOD_CLASSICAL else "your detector"
        lines = [f"object-counter-ai stream: {self.frames} frame(s) ({how})"]
        if self.failed_frames:
            lines.append(f"  failed      {self.failed_frames} frame(s) not counted: the detector failed")
        if ok:
            counts = [r.count for r in ok]
            lines.append(
                f"  per frame   min {min(counts)}, max {max(counts)}, mean {np.mean(counts):.1f}"
            )
        gate = "auto (1.5 x box diagonal)" if self.max_distance is None else f"{_fmt(self.max_distance)} px"
        lines.append(
            f"  tracks      {self.tracks_started} started (nearest-centroid matching, "
            f"max distance {gate}, max missed {self.max_missed})"
        )
        if not self._lines:
            lines.append("  lines       none; add one with .line(a, b) to count crossings")
        for name, ln in self._lines.items():
            label_text = ""
            if ln.by_label and (len(ln.by_label) > 1 or "blob" not in ln.by_label):
                label_text = "; " + ", ".join(
                    f"{k} {v}" for k, v in sorted(ln.by_label.items(), key=lambda kv: (-kv[1], kv[0])))
            lines.append(
                f"  {name:<11} ({_fmt(ln.a[0])}, {_fmt(ln.a[1])}) to ({_fmt(ln.b[0])}, {_fmt(ln.b[1])}): "
                f"{ln.forward + ln.backward} crossed ({ln.forward} forward, {ln.backward} backward)"
                f"{label_text}"
            )
            if ln.recrossed:
                lines.append(f"  {'':<11} {ln.recrossed} re-crossing(s) by already-counted objects ignored")
        lines.append("  note        each object is counted once per line; tracking is simple "
                     "nearest-centroid matching")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe account of the stream, including every frame."""
        return {
            "frames": self.frames,
            "failed_frames": self.failed_frames,
            "tracks_started": self.tracks_started,
            "peak": self.peak,
            "totals": self.totals,
            "lines": self.lines,
            "options": {
                "min_area": self.min_area,
                "max_area": self.max_area,
                "region": self._regions.region.to_json() if self._regions.region else None,
                "max_distance": self.max_distance,
                "max_missed": self.max_missed,
                "line_margin": self.line_margin,
                "detector": getattr(self.detector, "__name__", type(self.detector).__name__)
                if self.detector is not None else None,
            },
            "history": [r.to_dict() for r in self.history],
        }
