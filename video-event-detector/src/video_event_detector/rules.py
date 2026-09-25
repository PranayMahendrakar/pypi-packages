"""The four event rules. Each one reads a :class:`Scene` and may raise events.

All of them are heuristics on the moving and still regions the motion model
finds. None of them knows what a person, a bag or a crowd is.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ._imageops import Blob, box_intersection, box_iou, dilate, erode, gradient, label
from .events import Event, format_span, format_time

Box = Tuple[int, int, int, int]  # (top, left, bottom, right), inclusive, working pixels


class Span:
    """A duration counted in frames, or on the caller's clock.

    ``Span(frames=30)`` is reached 30 frames after it starts whatever the
    timestamps say; ``Span(clock=2.0)`` after 2.0 units of the caller's clock
    (seconds when timestamps or fps are given, otherwise frames).
    """

    def __init__(self, *, frames: Optional[int] = None, clock: Optional[float] = None) -> None:
        if (frames is None) == (clock is None):
            raise ValueError("a Span is either frames or clock")
        self.frames = frames
        self.clock = clock

    def reached(self, start_index: int, start_time: float, index: int, time: float) -> bool:
        """True once this much has passed since ``start``."""
        if self.frames is not None:
            return index - start_index >= self.frames
        return time - start_time >= float(self.clock) - 1e-9  # type: ignore[arg-type]

    def reached_array(
        self, start_index: np.ndarray, start_time: np.ndarray, index: int, time: float
    ) -> np.ndarray:
        """Vectorised :meth:`reached`; entries with ``start_index < 0`` are False."""
        live = start_index >= 0
        if self.frames is not None:
            return live & (index - start_index >= self.frames)
        return live & (time - start_time >= float(self.clock) - 1e-9)  # type: ignore[arg-type]

    def describe(self, unit: str) -> str:
        """Human text, e.g. ``30 frames`` or ``2.0 s``."""
        if self.frames is not None:
            return "{} frame{}".format(self.frames, "" if self.frames == 1 else "s")
        return format_span(float(self.clock), unit)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, object]:
        """``{"frames": 30}`` or ``{"clock": 2.0}``."""
        if self.frames is not None:
            return {"frames": int(self.frames)}
        return {"clock": float(self.clock)}  # type: ignore[arg-type]


@dataclass
class Region:
    """One foreground region of a frame."""

    blob: Blob
    changed: int
    moving: bool

    @property
    def box(self) -> Box:
        """(top, left, bottom, right) in working pixels."""
        return self.blob.box


@dataclass
class Scene:
    """Everything the rules need about the current frame."""

    index: int
    time: float
    unit: str
    frame: np.ndarray            # lighting-matched working frame
    background: np.ndarray
    foreground: np.ndarray
    changed: np.ndarray
    labels: np.ndarray
    regions: List[Region]
    moving_mask: np.ndarray
    moving_share: float
    threshold: float
    factor: int                  # working pixel -> frame pixels
    frame_size: Tuple[int, int]  # (width, height) of the first frame

    @property
    def shape(self) -> Tuple[int, int]:
        """(rows, cols) of the working image."""
        return (int(self.foreground.shape[0]), int(self.foreground.shape[1]))

    def to_frame(self, box: Box) -> Tuple[int, int, int, int]:
        """Working (top, left, bottom, right) -> frame (x, y, width, height).

        The foreground mask runs about one working pixel past an object's own
        edge (the difference image is smoothed 3x3), so that pixel is trimmed.
        """
        k = self.factor
        top, left, bottom, right = box
        if bottom - top >= 2:
            top, bottom = top + 1, bottom - 1
        if right - left >= 2:
            left, right = left + 1, right - 1
        width, height = self.frame_size
        x, y = left * k, top * k
        w = min(width - x, (right - left + 1) * k)
        h = min(height - y, (bottom - top + 1) * k)
        return (int(x), int(y), int(max(1, w)), int(max(1, h)))

    def px(self, value: float) -> int:
        """A working-pixel length in frame pixels."""
        return int(round(value * self.factor))

    def extent(self, value: float) -> int:
        """A mask height or width in frame pixels, less the smoothing fringe."""
        return int(round(max(1.0, value - 2.0 if value >= 3 else value) * self.factor))

    def when(self, time: float) -> str:
        """Format a time on this clock."""
        return format_time(time, self.unit)


def mask_box(mask: np.ndarray) -> Optional[Box]:
    """Bounding box of a mask, or None when it is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))


def _union(a: Optional[Box], b: Optional[Box]) -> Optional[Box]:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _pct(value: float) -> str:
    return "{:.0f}%".format(100.0 * value) if value >= 0.095 else "{:.1f}%".format(100.0 * value)


# ---------------------------------------------------------------------------
# sudden_motion
# ---------------------------------------------------------------------------
class SuddenMotionRule:
    """A large jump in moving area within three frames of a calm baseline."""

    kind = "sudden_motion"

    def __init__(self, jump: float, settle: Span, history: int = 10) -> None:
        self.jump = float(jump)
        self.settle = settle
        self.history: Deque[Tuple[int, float, float]] = deque(maxlen=history + 2)
        self.active: Optional[Dict[str, object]] = None

    def reset(self) -> None:
        """Forget the recent history (after the camera moved to a new view)."""
        self.history.clear()
        self.active = None

    def update(self, scene: Scene) -> List[Event]:
        """Raise ``sudden_motion`` when the moving share jumps abruptly."""
        share = scene.moving_share
        raised: List[Event] = []
        active = self.active
        if active is not None:
            event: Event = active["event"]  # type: ignore[assignment]
            base = float(active["base"])  # type: ignore[arg-type]
            size = float(active["jump"])  # type: ignore[arg-type]
            settled = self.settle.reached(
                int(active["index"]), float(active["time"]), scene.index, scene.time  # type: ignore[arg-type]
            )
            if share >= base + 0.5 * size and not settled:
                event.end = scene.time
            else:
                self.active = None
        elif len(self.history) >= 5:
            shares = [s for _, _, s in self.history]
            before = statistics.median(shares[:-2])
            three_back = shares[-3]
            rise = share - before
            needed = max(self.jump, before)
            if rise >= needed and share - three_back >= 0.75 * rise and scene.moving_mask.any():
                onset_index, onset_time = scene.index, scene.time
                for index, time, value in list(self.history)[-2:]:
                    if value >= before + 0.25 * rise:
                        onset_index, onset_time = index, time
                        break
                box = mask_box(scene.moving_mask)
                confidence = min(0.99, 0.5 + 0.5 * min(1.0, (rise - needed) / needed))
                message = (
                    "the moving area jumped from {} to {} of the view within {}; "
                    "anything that suddenly fills the view qualifies (a door, a vehicle, "
                    "a light going on over a moving scene)".format(
                        _pct(before),
                        _pct(share),
                        "one frame" if onset_index == scene.index
                        else "{} frames".format(scene.index - onset_index + 1),
                    )
                )
                event = Event(
                    kind=self.kind,
                    start=onset_time,
                    end=scene.time,
                    confidence=round(confidence, 3),
                    region=scene.to_frame(box) if box else None,
                    message=message,
                )
                self.active = {
                    "event": event,
                    "base": before,
                    "jump": rise,
                    "index": onset_index,
                    "time": onset_time,
                }
                raised.append(event)
        self.history.append((scene.index, scene.time, share))
        return raised


# ---------------------------------------------------------------------------
# crowding
# ---------------------------------------------------------------------------
class CrowdingRule:
    """The moving share of the view staying above a level for ``hold``."""

    kind = "crowding"

    def __init__(self, level: float, hold: Span) -> None:
        self.level = float(level)
        self.hold = hold
        self.start: Optional[Tuple[int, float]] = None
        self.shares: List[float] = []
        self.box: Optional[Box] = None
        self.event: Optional[Event] = None
        self.low = 0

    def reset(self) -> None:
        """Drop a half-built candidate (after the camera moved to a new view)."""
        self.start = None
        self.shares = []
        self.box = None
        self.event = None
        self.low = 0

    def update(self, scene: Scene) -> List[Event]:
        """Raise ``crowding`` once the moving share has stayed high for ``hold``."""
        share = scene.moving_share
        floor = 0.8 * self.level if self.event is not None else self.level
        if share < floor:
            self.low += 1
            if self.event is None:
                self.start, self.shares, self.box = None, [], None
            elif self.low >= 2:
                self.reset()
            return []
        self.low = 0
        if self.start is None:
            self.start = (scene.index, scene.time)
        self.shares.append(share)
        self.box = _union(self.box, mask_box(scene.moving_mask))
        mean = float(np.mean(self.shares))
        peak = float(np.max(self.shares))
        confidence = round(min(0.99, max(0.05, 0.5 + 0.5 * (mean / self.level - 1.0))), 3)
        if self.event is not None:
            self.event.end = scene.time
            self.event.confidence = confidence
            if self.box is not None:
                self.event.region = scene.to_frame(self.box)
            self.event.message = self._message(scene, mean, peak)
            return []
        if self.hold.reached(self.start[0], self.start[1], scene.index, scene.time):
            self.event = Event(
                kind=self.kind,
                start=self.start[1],
                end=scene.time,
                confidence=confidence,
                region=scene.to_frame(self.box) if self.box else None,
                message=self._message(scene, mean, peak),
            )
            return [self.event]
        return []

    def _message(self, scene: Scene, mean: float, peak: float) -> str:
        return (
            "{} of the view was in motion on average (peak {}), above the {} level "
            "for {}; this measures how much of the picture moves, not how many "
            "people are in it".format(
                _pct(mean),
                _pct(peak),
                _pct(self.level),
                format_span(scene.time - float(self.start[1]), scene.unit)  # type: ignore[index]
                if scene.unit != "frames"
                else "{} frames".format(scene.index - int(self.start[0]) + 1),  # type: ignore[index]
            )
        )


# ---------------------------------------------------------------------------
# fall
# ---------------------------------------------------------------------------
@dataclass
class _Point:
    index: int
    time: float
    top: int
    left: int
    bottom: int
    right: int
    area: int
    moving: bool

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def box(self) -> Box:
        return (self.top, self.left, self.bottom, self.right)


@dataclass
class _Track:
    ident: int
    points: Deque[_Point]
    last_index: int
    interaction: int = -10 ** 9
    candidate: Optional[Dict[str, float]] = None
    event: Optional[Event] = None

    @property
    def last(self) -> _Point:
        return self.points[-1]


class FallRule:
    """A tracked moving region whose height collapses while its width holds.

    The rule follows foreground regions from frame to frame. A fall candidate
    is a region that was moving, at least as tall as it was wide, and then
    within ``window`` frames lost enough height (to ``ratio`` of it) while its
    width held and its lowest point did not rise. It becomes an event only if
    the shape then *stays* low for ``hold``. Regions that merge with or split
    from another region are skipped for a while: a crossing of two people
    changes box heights in exactly the same way.
    """

    kind = "fall"

    def __init__(self, ratio: float, hold: Span, window: int, min_height: int) -> None:
        self.ratio = float(ratio)
        self.recover = min(0.9, self.ratio + 0.15)
        self.hold = hold
        self.window = int(window)
        self.min_height = int(min_height)
        self.tracks: List[_Track] = []
        self._next = 0

    def reset(self) -> None:
        """Forget all tracks (after the camera moved to a new view)."""
        for track in self.tracks:
            self._finish(track)
        self.tracks = []

    def active_boxes(self) -> List[Box]:
        """Working boxes of falls that are ongoing (so they are not also 'abandoned')."""
        return [t.last.box for t in self.tracks if t.event is not None or t.candidate is not None]

    def _finish(self, track: _Track) -> None:
        track.event = None
        track.candidate = None

    def _associate(self, scene: Scene, regions: List[Region]) -> List[Tuple[_Track, Region]]:
        live = [t for t in self.tracks if t.last_index >= scene.index - 3]
        pairs = []
        for ti, track in enumerate(live):
            box = track.last.box
            size = max(track.last.height, track.last.width)
            cy = (box[0] + box[2]) / 2.0
            cx = (box[1] + box[3]) / 2.0
            for ri, region in enumerate(regions):
                iou = box_iou(box, region.box)
                dist = math.hypot(region.blob.cy - cy, region.blob.cx - cx)
                if iou >= 0.1 or dist <= 0.75 * size:
                    pairs.append((-iou, dist, ti, ri))
        pairs.sort()
        used_t, used_r = set(), set()
        matches: List[Tuple[_Track, Region]] = []
        for _, _, ti, ri in pairs:
            if ti in used_t or ri in used_r:
                continue
            used_t.add(ti)
            used_r.add(ri)
            matches.append((live[ti], regions[ri]))

        # A track box touching two regions is a split; a region touching two
        # track boxes is a merge. Either makes box heights meaningless for a while.
        for track in live:
            touching = sum(1 for r in regions if box_intersection(track.last.box, r.box) > 0)
            if touching >= 2:
                track.interaction = scene.index
        for region in regions:
            owners = [t for t in live if box_intersection(t.last.box, region.box) > 0]
            if len(owners) >= 2:
                for track in owners:
                    track.interaction = scene.index

        for ri, region in enumerate(regions):
            if ri not in used_r:
                track = _Track(ident=self._next, points=deque(maxlen=self.window + 2), last_index=scene.index)
                self._next += 1
                self.tracks.append(track)
                matches.append((track, region))
        return matches

    def update(self, scene: Scene, regions: List[Region]) -> List[Event]:
        """Follow the regions and raise ``fall`` for a confirmed collapse."""
        raised: List[Event] = []
        rows = scene.shape[0]
        for track, region in self._associate(scene, regions):
            blob = region.blob
            track.points.append(
                _Point(scene.index, scene.time, blob.top, blob.left, blob.bottom, blob.right, blob.area, region.moving)
            )
            track.last_index = scene.index
            event = self._check(track, scene, rows)
            if event is not None:
                raised.append(event)
        kept = []
        for track in self.tracks:
            if track.last_index >= scene.index - 3:
                kept.append(track)
            else:
                self._finish(track)
        self.tracks = kept
        return raised

    def _check(self, track: _Track, scene: Scene, rows: int) -> Optional[Event]:
        now = track.last
        if track.event is not None:
            if now.height <= self.recover * track.candidate["h_ref"]:  # type: ignore[index]
                track.event.end = scene.time
                track.event.region = scene.to_frame(now.box)
            else:
                self._finish(track)
            return None

        if track.candidate is not None:
            cand = track.candidate
            if now.height > self.recover * cand["h_ref"]:
                track.candidate = None  # a brief dip that recovered: not a fall
                return None
            if now.height < cand["low_height"]:
                cand["low_height"] = float(now.height)
                cand["low_width"] = float(now.width)
                cand["ratio"] = now.height / cand["h_ref"]
            if self.hold.reached(int(cand["collapse_index"]), cand["collapse_time"], scene.index, scene.time):
                return self._confirm(track, scene)
            return None

        points = [p for p in track.points if p.index >= scene.index - self.window]
        if len(points) < 3 or track.interaction >= scene.index - self.window:
            return None
        earlier = points[:-1]
        ref = max(earlier, key=lambda p: (p.height, p.index))
        h_ref, w_ref = ref.height, ref.width
        if h_ref < self.min_height or h_ref < 0.9 * w_ref:
            return None
        if not any(p.moving for p in points):
            return None
        ratio = now.height / float(h_ref)
        if ratio > self.ratio:
            return None
        if now.width < 0.75 * w_ref or now.area < 0.3 * ref.area:
            return None
        if now.bottom < ref.bottom - 0.25 * h_ref:
            return None  # the lowest point rose: the region lifted or left upwards
        if now.top <= 0 or now.bottom >= rows - 1 or ref.top <= 0 or ref.bottom >= rows - 1:
            return None  # clipped by the frame edge: it may just be leaving the view
        onset = ref
        for p in earlier:
            if p.index >= ref.index and p.height >= 0.9 * h_ref:
                onset = p
        track.candidate = {
            "h_ref": float(h_ref),
            "w_ref": float(w_ref),
            "ratio": ratio,
            "onset_index": float(onset.index),
            "onset_time": onset.time,
            "collapse_index": float(scene.index),
            "collapse_time": scene.time,
            "low_width": float(now.width),
            "low_height": float(now.height),
        }
        if self.hold.reached(scene.index, scene.time, scene.index, scene.time):
            return self._confirm(track, scene)
        return None

    def _confirm(self, track: _Track, scene: Scene) -> Event:
        cand = track.candidate
        assert cand is not None
        now = track.last
        frames_taken = max(1, int(cand["collapse_index"] - cand["onset_index"]))
        depth = min(1.0, max(0.0, (self.ratio - cand["ratio"]) / (0.5 * self.ratio)))
        speed = max(0.0, 1.0 - (frames_taken - 1) / float(max(1, self.window)))
        confidence = round(min(0.95, max(0.05, 0.5 + 0.3 * depth + 0.15 * speed)), 3)
        took = (
            format_span(cand["collapse_time"] - cand["onset_time"], scene.unit)
            if scene.unit != "frames"
            else "{} frame{}".format(frames_taken, "" if frames_taken == 1 else "s")
        )
        low_frames = scene.index - int(cand["collapse_index"])
        stayed = (
            format_span(scene.time - cand["collapse_time"], scene.unit)
            if scene.unit != "frames"
            else "{} frame{}".format(low_frames, "" if low_frames == 1 else "s")
        )
        message = (
            "a moving region {} px tall dropped to {} px ({} of its height) within {} "
            "while its width held ({} -> {} px), and stayed low for {}; a fall here is "
            "only this shape change - sitting, bending or lying down look the same, "
            "so review it".format(
                scene.extent(cand["h_ref"]),
                scene.extent(cand["low_height"]),
                _pct(scene.extent(cand["low_height"]) / float(max(1, scene.extent(cand["h_ref"])))),
                took,
                scene.extent(cand["w_ref"]),
                scene.extent(cand["low_width"]),
                stayed,
            )
        )
        track.event = Event(
            kind=self.kind,
            start=cand["onset_time"],
            end=scene.time,
            confidence=confidence,
            region=scene.to_frame(now.box),
            message=message,
        )
        return track.event


# ---------------------------------------------------------------------------
# abandoned
# ---------------------------------------------------------------------------
@dataclass
class _Left:
    event: Event
    mask: np.ndarray
    box: Box
    missing_since: Optional[Tuple[int, float]] = None
    dwell_text: str = ""
    appeared: float = 0.0


class AbandonedRule:
    """A new region that appears and then stays still for ``dwell``.

    Per pixel it keeps two clocks: since when the pixel has differed from the
    background, and since when it has also been unchanged frame to frame.
    Once enough adjacent pixels have been still for ``dwell``, the region is
    checked: an object *put down* shows its edges in the current frame, while
    the hole left by an object *taken away* shows them only in the background,
    so the second case is not reported and is folded back into the background.
    """

    kind = "abandoned"

    def __init__(
        self, dwell: Span, grace: Span, min_area: int, max_share: float, absorb_frames: int
    ) -> None:
        self.dwell = dwell
        self.grace = grace
        self.min_area = int(min_area)
        self.max_share = float(max_share)
        self.absorb_frames = int(absorb_frames)
        self.shape: Optional[Tuple[int, int]] = None
        self.left: List[_Left] = []
        self.removed = 0
        self.oversize = 0
        self.skipped_as_fall = 0

    def _ensure(self, shape: Tuple[int, int]) -> None:
        if self.shape != shape:
            self.shape = shape
            self.fg_index = np.full(shape, -1, dtype=np.int64)
            self.fg_time = np.zeros(shape, dtype=np.float64)
            self.still_index = np.full(shape, -1, dtype=np.int64)
            self.still_time = np.zeros(shape, dtype=np.float64)
            self.protect = np.zeros(shape, dtype=bool)
            self.handled = np.zeros(shape, dtype=bool)

    def reset(self) -> None:
        """Restart the pixel clocks (after the camera moved to a new view)."""
        if self.shape is not None:
            self.fg_index[:] = -1
            self.still_index[:] = -1
            self.handled[:] = False
        # Objects already reported cannot be followed into a new view.
        self.left = []
        if self.shape is not None:
            self.protect[:] = False

    def update(self, scene: Scene, fall_boxes: List[Box]) -> Tuple[List[Event], np.ndarray]:
        """Advance the clocks; return new ``abandoned`` events and pixels to absorb."""
        self._ensure(scene.shape)
        fg = scene.foreground
        still = fg & ~scene.changed
        index, time = scene.index, scene.time

        fresh = fg & (self.fg_index < 0)
        self.fg_index[fresh] = index
        self.fg_time[fresh] = time
        self.fg_index[~fg] = -1

        begin = still & (self.still_index < 0)
        self.still_index[begin] = index
        self.still_time[begin] = time
        self.still_index[~still] = -1
        self.handled &= still

        absorb = np.zeros(scene.shape, dtype=bool)
        raised: List[Event] = []
        self._follow(scene)

        ready = self.dwell.reached_array(self.still_index, self.still_time, index, time)
        ready &= ~self.protect & ~self.handled
        if int(ready.sum()) >= self.min_area:
            # Grow each group of long-still pixels to the whole still region it sits
            # in, so one object is one event even if its pixels settled unevenly.
            still_now = fg & (self.still_index >= 0) & ~self.protect & ~self.handled
            still_labels, _ = label(still_now)
            labels, blobs = label(ready)
            view = float(fg.size)
            taken = np.zeros(scene.shape, dtype=bool)
            for blob in sorted(blobs, key=lambda b: (-b.area, b.label)):
                if blob.area < self.min_area:
                    continue
                seed = labels == blob.label
                if (seed & taken).any():
                    continue
                ids = np.unique(still_labels[seed & (still_labels >= 0)])
                mask = np.isin(still_labels, ids) | seed if ids.size else seed
                taken |= mask
                box = mask_box(mask)
                if box is None:
                    continue
                area = int(mask.sum())
                if area > self.max_share * view:
                    self.oversize += 1
                    self.handled |= mask
                    continue
                if any(
                    box_intersection(box, fb) > 0.3 * area or box_iou(box, fb) > 0.3
                    for fb in fall_boxes
                ):
                    self.skipped_as_fall += 1
                    self.handled |= mask
                    continue
                # The foreground edge sits about a pixel outside the object's own
                # edge (the difference is smoothed), so measure a band across it.
                band = dilate(mask) & ~erode(erode(mask))
                if not band.any():
                    continue
                edges_now = float(gradient(scene.frame)[band].mean())
                edges_before = float(gradient(scene.background)[band].mean())
                if edges_now < 0.8 * edges_before:
                    # The background has the edges and the frame does not: something
                    # that used to be here was taken away. Heal the background.
                    self.removed += 1
                    self.handled |= mask
                    absorb |= dilate(mask) & fg
                    continue
                if self._merge(scene, mask, box):
                    continue
                raised.append(self._raise(scene, mask, box, edges_now, edges_before))

        long_still = (self.still_index >= 0) & (index - self.still_index >= self.absorb_frames)
        if self.dwell.clock is not None:
            long_still &= time - self.still_time >= 2.0 * float(self.dwell.clock)
        absorb |= long_still & ~self.protect
        return raised, absorb

    def _merge(self, scene: Scene, mask: np.ndarray, box: Box) -> bool:
        """Fold a still region into a reported object it touches (within 3 px)."""
        for left in self.left:
            rows_gap = max(box[0] - left.box[2], left.box[0] - box[2], 0)
            cols_gap = max(box[1] - left.box[3], left.box[1] - box[3], 0)
            if max(rows_gap, cols_gap) <= 3:
                left.mask |= mask
                left.box = _union(left.box, box)  # type: ignore[assignment]
                left.event.region = scene.to_frame(left.box)
                self.protect |= mask
                return True
        return False

    def _raise(
        self, scene: Scene, mask: np.ndarray, box: Box, edges_now: float, edges_before: float
    ) -> Event:
        seen = mask & (self.fg_index >= 0)
        appeared = float(np.median(self.fg_time[seen])) if seen.any() else scene.time
        settled = mask & (self.still_index >= 0)
        still_from = float(np.median(self.still_time[settled])) if settled.any() else scene.time
        contrast = float(np.median(np.abs(scene.frame - scene.background)[mask]))
        strength = min(1.0, max(0.0, (contrast / max(1e-6, scene.threshold) - 1.0) / 3.0))
        edge_score = min(1.0, max(0.0, edges_now / max(1e-6, edges_before) - 0.8))
        confidence = round(min(0.95, max(0.05, 0.5 + 0.3 * strength + 0.15 * edge_score)), 3)
        core = erode(mask)
        area = int(core.sum()) if core.any() else int(mask.sum())
        frame_area = scene.px(1) ** 2 * area
        event = Event(
            kind=self.kind,
            start=appeared,
            end=scene.time,
            confidence=confidence,
            region=scene.to_frame(box),
            message="",
        )
        left = _Left(
            event=event,
            mask=mask.copy(),
            box=box,
            dwell_text=self.dwell.describe(scene.unit),
            appeared=appeared,
        )
        left_share = area / float(mask.size)
        left.event.message = (
            "a new region of about {} px ({} of the view) appeared at {} and has not "
            "moved since {} (dwell {}); a person standing still, a parked vehicle or "
            "a moved piece of furniture look the same, so review it".format(
                frame_area,
                _pct(left_share),
                scene.when(appeared),
                scene.when(still_from),
                left.dwell_text,
            )
        )
        self.left.append(left)
        self.protect |= mask
        return event

    def _follow(self, scene: Scene) -> None:
        """Extend reported objects that are still there; close ones that left."""
        kept: List[_Left] = []
        for left in self.left:
            present = float(scene.foreground[left.mask].mean()) >= 0.5
            if present:
                left.event.end = scene.time
                left.missing_since = None
                kept.append(left)
                continue
            if left.missing_since is None:
                left.missing_since = (scene.index, scene.time)
            if self.grace.reached(left.missing_since[0], left.missing_since[1], scene.index, scene.time):
                self.protect &= ~left.mask
                continue
            kept.append(left)
        self.left = kept
