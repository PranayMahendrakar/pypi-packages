"""Watch one camera over time.

A single frame cannot tell a frozen feed from a dropped frame, and it cannot see
a camera that is slowly being nudged off target. Monitor keeps a rolling history
so both become visible: a repeat only counts as frozen once it has happened
``freeze_frames`` times in a row, and drift is a fall in the match to the
reference across the whole window rather than in any one frame.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from ._frames import Frame, as_frame
from .checks import _assess, score_from
from .faults import CRITICAL, DRIFT, TAMPERING, WARNING, Fault, sort_faults
from .result import FAULT, NOT_CHECKABLE, FrameHealth
from .thresholds import Thresholds, resolve

log = logging.getLogger(__name__)

HEALTHY = "healthy"
DEGRADED = "degraded"
FAILED = "failed"

STATES = (HEALTHY, DEGRADED, FAILED)
"""The three states a monitored camera can be in."""

RECOVERED = "recovered"
"""The alert kind raised when a camera comes back from degraded or failed."""


@dataclass(frozen=True)
class Alert:
    """Something that changed, worth telling somebody about.

    Attributes:
        when_index: the frame number the alert was raised on, counting from 0.
        kind: a fault kind, or "recovered".
        message: one plain sentence.
    """

    when_index: int
    kind: str
    message: str

    def line(self) -> str:
        """The one-line form used in Monitor.summary()."""
        return "  frame {:<5} {:<12} {}".format(self.when_index, self.kind, self.message)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of the alert."""
        return {"when_index": int(self.when_index), "kind": self.kind, "message": self.message}


class Monitor:
    """A running health check on one camera.

    Feed it frames as they arrive and it returns the same FrameHealth that
    ``check`` would, plus the things only a sequence can show: a frozen feed, a
    slow drift off target, and how much of the run the camera was usable for.

        >>> monitor = Monitor(reference_frame, freeze_frames=5)
        >>> for frame in incoming:
        ...     health = monitor.update(frame)
        >>> monitor.state, monitor.uptime
    """

    def __init__(
        self,
        reference: Any = None,
        *,
        freeze_frames: int = 5,
        history: int = 30,
        thresholds: Any = None,
    ) -> None:
        """Start watching a camera.

        Args:
            reference: how this view is supposed to look. When None, the first
                frame handed to ``update`` is adopted as the reference and a note
                records that, so tampering and drift are checkable from the
                second frame on.
            freeze_frames: how many identical frames in a row make the feed
                frozen. Below that a repeat is only a warning, because a still
                scene plus one dropped frame looks exactly the same. A live
                sensor watching a motionless room never reaches this count at
                all: its read noise moves most pixels every frame, so those
                frames are not repeats to begin with.
            history: how many frames to keep for the drift comparison.
            thresholds: a Thresholds, or a dict of overrides.

        Raises:
            ValueError: freeze_frames or history is below its minimum.
        """
        if int(freeze_frames) < 2:
            raise ValueError(
                "freeze_frames counts identical frames in a row, so it must be 2 or more; "
                "got {}".format(freeze_frames)
            )
        if int(history) < 2:
            raise ValueError("history must be 2 or more frames; got {}".format(history))
        self.thresholds: Thresholds = resolve(thresholds)
        self.freeze_frames = int(freeze_frames)
        self.history_size = int(history)
        self.history: Deque[FrameHealth] = deque(maxlen=self.history_size)
        self.alerts: List[Alert] = []
        self.notes: List[str] = []
        self.state: str = HEALTHY
        self.n_frames: int = 0
        self.n_ok: int = 0
        self.last: Optional[FrameHealth] = None

        self._reference: Optional[Frame] = None
        self._adopted_reference = False
        if reference is not None:
            self._reference = as_frame(reference, analysis_pixels=self.thresholds.analysis_pixels)
        self._previous: Optional[Frame] = None
        self._repeats = 0
        self._correlations: Deque[float] = deque(maxlen=self.history_size)
        self._reported: Dict[str, int] = {}
        self._bad_run = 0

    @property
    def uptime(self) -> float:
        """Share of every frame seen so far that was ok. 1.0 before any frame."""
        if self.n_frames == 0:
            return 1.0
        return self.n_ok / float(self.n_frames)

    @property
    def reference_adopted(self) -> bool:
        """True when the first frame was taken as the reference."""
        return self._adopted_reference

    def update(self, frame: Any) -> FrameHealth:
        """Check the next frame and fold it into the running state.

        Args:
            frame: an array, a PIL image, or a path to an image file.

        Returns:
            The FrameHealth for this frame, with frozen and drift decided using
            the history rather than this frame alone.

        Raises:
            ValueError: the frame could not be read.
            FileNotFoundError: a path was given and there is no file there.
        """
        limits = self.thresholds
        current = as_frame(frame, analysis_pixels=limits.analysis_pixels)
        index = self.n_frames
        notes: List[str] = []
        if self._reference is None:
            self._reference = current
            self._adopted_reference = True
            note = (
                "no reference was given, so frame 0 was adopted as the reference; "
                "tampering and drift are measured against it from frame 1 on"
            )
            notes.append(note)
            if note not in self.notes:
                self.notes.append(note)

        health, repeats, correlation = _assess(
            current,
            reference=self._reference if index > 0 or not self._adopted_reference else None,
            previous=self._previous,
            limits=limits,
            repeats=self._repeats,
            freeze_frames=self.freeze_frames,
            index=index,
            drift_window=len(self._correlations),
            extra_notes=notes,
        )
        if self._adopted_reference and index == 0:
            health.checks[DRIFT] = NOT_CHECKABLE
            health.not_checkable[DRIFT] = (
                "this frame is the adopted reference, so it cannot drift from itself"
            )
            health.checks[TAMPERING] = NOT_CHECKABLE
            health.not_checkable[TAMPERING] = (
                "this frame is the adopted reference, so it cannot differ from itself"
            )

        self._repeats = repeats
        self._previous = current
        if correlation is not None:
            self._correlations.append(correlation)
        self._add_drift(health, index, correlation, limits)

        self.n_frames += 1
        if health.ok:
            self.n_ok += 1
        self.history.append(health)
        self.last = health
        self._update_state(health, index)
        return health

    def _add_drift(
        self,
        health: FrameHealth,
        index: int,
        correlation: Optional[float],
        limits: Thresholds,
    ) -> None:
        """Raise drift when the match to the reference has fallen across the window."""
        if correlation is None or health.checks.get(DRIFT) == NOT_CHECKABLE:
            return
        if health.has("tampering"):
            # Already reported, and more strongly, as tampering.
            return
        window = list(self._correlations)
        if len(window) < limits.drift_min_frames or correlation >= limits.drift_correlation:
            return
        third = max(2, len(window) // 3)
        early = sorted(window[:third])[third // 2]
        late = sorted(window[-third:])[third // 2]
        if early - late < limits.drift_delta:
            return
        health.faults = sort_faults(
            health.faults
            + [
                Fault(
                    DRIFT,
                    WARNING,
                    0.7,
                    "the view has drifted away from the reference over the last {} frames "
                    "(match {:.2f}, down from {:.2f}); the camera is slowly moving or the "
                    "scene is being rearranged".format(len(window), late, early),
                )
            ]
        )
        health.checks[DRIFT] = FAULT
        health.score = score_from(health.faults, limits)
        health.ok = not any(fault.severity == CRITICAL for fault in health.faults)

    def _update_state(self, health: FrameHealth, index: int) -> None:
        """Move between healthy, degraded and failed, and raise alerts on the way.

        The state describes the camera now, judged over the frames still in the
        history window - not over the whole run. A camera that failed an hour ago
        and has been clean since is healthy; ``uptime`` is what remembers the
        hour ago. "Degraded" covers both a frame with warnings on it and a
        camera that has only just come back, because one clean frame after a
        failure is not yet proof of anything.
        """
        previous_state = self.state
        recent_failure = any(not seen.ok for seen in self.history)
        if not health.ok:
            state = FAILED
        elif health.faults or recent_failure:
            state = DEGRADED
        else:
            state = HEALTHY
        self.state = state

        # One alert when a fault kind appears, and one more if it later gets
        # worse. Repeating the same fault every frame would bury the change,
        # which is the only part an operator has to act on.
        present = {fault.kind: fault for fault in health.faults}
        for kind, fault in present.items():
            alerted_at = self._reported.get(kind)
            if alerted_at is not None and fault.rank >= alerted_at:
                continue
            self._reported[kind] = fault.rank
            self.alerts.append(
                Alert(index, kind, "[{}] {}".format(fault.severity, fault.message))
            )
        for kind in [kind for kind in self._reported if kind not in present]:
            del self._reported[kind]

        if previous_state == FAILED and state != FAILED:
            self.alerts.append(
                Alert(
                    index,
                    RECOVERED,
                    "the camera is usable again at frame {}, after {} unusable "
                    "frame(s) in a row".format(index, self._bad_run),
                )
            )
        self._bad_run = self._bad_run + 1 if not health.ok else 0

    def summary(self) -> str:
        """A short plain-text report on the run so far."""
        lines = [
            "camera monitor: {} after {} frame(s) (uptime {:.1%})".format(
                self.state, self.n_frames, self.uptime
            )
        ]
        if self._adopted_reference:
            lines.append("reference: frame 0 of this run (none was given)")
        if self.last is not None:
            lines.append("last frame: " + self.last.summary().splitlines()[0])
            for fault in self.last.faults:
                lines.append(fault.line())
        else:
            lines.append("no frames seen yet")
        if self.alerts:
            lines.append("alerts ({}):".format(len(self.alerts)))
            lines.extend(alert.line() for alert in self.alerts)
        else:
            lines.append("no alerts")
        if self.notes:
            lines.append("notes ({}):".format(len(self.notes)))
            lines.extend("  " + note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of the monitor state and the frames it still holds."""
        return {
            "state": self.state,
            "uptime": round(self.uptime, 4),
            "frames_seen": int(self.n_frames),
            "frames_ok": int(self.n_ok),
            "freeze_frames": int(self.freeze_frames),
            "history": int(self.history_size),
            "reference_adopted": bool(self._adopted_reference),
            "alerts": [alert.to_dict() for alert in self.alerts],
            "notes": list(self.notes),
            "last": self.last.to_dict() if self.last is not None else None,
            "recent": [health.to_dict() for health in self.history],
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return "Monitor(state={!r}, frames={}, uptime={:.3f})".format(
            self.state, self.n_frames, self.uptime
        )
