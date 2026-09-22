"""The objects a check hands back. They are meant to be printed and believed.

Every result carries three things beyond the verdict: the numbers it was reached
from (``metrics``), the checks that could not run and why (``not_checkable``),
and the decisions taken on the caller's behalf (``notes``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .faults import CRITICAL, FAULT_KINDS, Fault, canonical_kind, sort_faults

PASS = "pass"
FAULT = "fault"
NOT_CHECKABLE = "not_checkable"

_METRIC_ORDER = (
    ("brightness", "brightness", "{:.1f}"),
    ("contrast", "contrast", "{:.1f}"),
    ("focus", "focus", "{:.3f}"),
    ("noise", "noise", "{:.2f}"),
    ("clipped_high", "blown pixels", "{:.1%}"),
    ("obstruction_area", "blocked area", "{:.1%}"),
    ("mean_diff", "motion vs previous", "{:.3f}"),
    ("changed_fraction", "pixels moving", "{:.1%}"),
    ("scene_correlation", "match to reference", "{:.3f}"),
    ("colour_cast", "colour cast", "{:.3f}"),
)


def _round(value: Any) -> Any:
    """Round floats for JSON without turning ints into floats."""
    if isinstance(value, float):
        return round(value, 4)
    return value


@dataclass
class FrameHealth:
    """The verdict on one frame.

    Attributes:
        ok: True when nothing critical is wrong. Warnings lower the score but
            leave ``ok`` True, because a noisy or slightly soft camera is still
            a working camera.
        score: 0-100. 100 is a clean frame; every fault costs points in
            proportion to its severity and the confidence behind it.
        faults: what is wrong, worst first.
        metrics: every number the checks measured, fault or no fault.
        checks: each fault kind mapped to "pass", "fault" or "not_checkable".
        not_checkable: the checks that could not run, and the reason for each.
        notes: decisions taken for you, such as resizing a mismatched reference.
        size: the frame as it was handed in, ``(width, height)``.
        source: the file name, when the frame came from disk.
        index: position in the stream, when the frame came from one.
    """

    ok: bool
    score: float
    faults: List[Fault] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, str] = field(default_factory=dict)
    not_checkable: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)
    source: Optional[str] = None
    index: Optional[int] = None

    @property
    def kinds(self) -> List[str]:
        """The kinds of every fault found, worst first."""
        return [fault.kind for fault in self.faults]

    @property
    def critical(self) -> List[Fault]:
        """Only the faults that make the frame unusable."""
        return [fault for fault in self.faults if fault.severity == CRITICAL]

    @property
    def worst(self) -> Optional[Fault]:
        """The fault that cost the most, or None for a clean frame."""
        return self.faults[0] if self.faults else None

    def has(self, kind: str) -> bool:
        """True when a fault of this kind was raised. Accepts common spellings."""
        return canonical_kind(kind) in self.kinds

    def fault(self, kind: str) -> Optional[Fault]:
        """The fault of this kind, or None. Accepts common spellings."""
        wanted = canonical_kind(kind)
        for found in self.faults:
            if found.kind == wanted:
                return found
        return None

    def was_checked(self, kind: str) -> bool:
        """True when this check actually ran on this frame."""
        return self.checks.get(canonical_kind(kind)) != NOT_CHECKABLE

    def _headline(self) -> str:
        label = "ok" if self.ok else "NOT OK"
        where = "frame {}x{}".format(self.size[0], self.size[1])
        if self.index is not None:
            where = "frame {}, {}x{}".format(self.index, self.size[0], self.size[1])
        if self.source:
            where = "{}, {}".format(where, self.source)
        return "camera health: {} (score {:.1f} / 100)  [{}]".format(label, self.score, where)

    def summary(self) -> str:
        """A short plain-text report; this is what the CLI prints."""
        lines = [self._headline()]
        if self.faults:
            lines.append("faults ({}):".format(len(self.faults)))
            lines.extend(fault.line() for fault in self.faults)
        else:
            lines.append("no faults found")
        if self.not_checkable:
            lines.append("not checkable ({}):".format(len(self.not_checkable)))
            for kind in FAULT_KINDS:
                if kind in self.not_checkable:
                    lines.append("  {:<12} {}".format(kind, self.not_checkable[kind]))
        if self.notes:
            lines.append("notes ({}):".format(len(self.notes)))
            lines.extend("  " + note for note in self.notes)
        shown = [
            "{} {}".format(label, form.format(self.metrics[key]))
            for key, label, form in _METRIC_ORDER
            if key in self.metrics
        ]
        if shown:
            lines.append("measured: " + ", ".join(shown))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of the whole verdict."""
        return {
            "ok": bool(self.ok),
            "score": round(float(self.score), 2),
            "faults": [fault.to_dict() for fault in self.faults],
            "metrics": {key: _round(value) for key, value in sorted(self.metrics.items())},
            "checks": dict(sorted(self.checks.items())),
            "not_checkable": dict(sorted(self.not_checkable.items())),
            "notes": list(self.notes),
            "size": {"width": int(self.size[0]), "height": int(self.size[1])},
            "source": self.source,
            "index": self.index,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


@dataclass
class StreamReport:
    """What a run of frames looked like as a whole.

    Attributes:
        by_frame: the FrameHealth of every frame, in order.
        faults: one Fault per kind seen anywhere in the run - the worst instance
            of that kind, with a message saying how many frames it hit.
        first_failure: index of the first frame that was not ok, else None.
        fault_counts: how many frames each kind hit.
        frames_checked: how many frames were read.
        uptime: share of frames that were ok; 0.0 for an empty run, where
            nothing was proven healthy.
        notes: decisions taken for the run as a whole.
    """

    by_frame: List[FrameHealth] = field(default_factory=list)
    faults: List[Fault] = field(default_factory=list)
    first_failure: Optional[int] = None
    fault_counts: Dict[str, int] = field(default_factory=dict)
    first_seen: Dict[str, int] = field(default_factory=dict)
    frames_checked: int = 0
    uptime: float = 1.0
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every frame in the run was ok."""
        return self.first_failure is None and self.frames_checked > 0

    @property
    def score(self) -> float:
        """Mean frame score over the run; 100.0 for an empty run."""
        if not self.by_frame:
            return 100.0
        return float(sum(health.score for health in self.by_frame) / len(self.by_frame))

    @property
    def worst_frame(self) -> Optional[FrameHealth]:
        """The lowest scoring frame in the run."""
        if not self.by_frame:
            return None
        return min(self.by_frame, key=lambda health: (health.score, health.index or 0))

    def frames_with(self, kind: str) -> List[int]:
        """Indexes of every frame that carried this fault kind."""
        wanted = canonical_kind(kind)
        return [
            health.index if health.index is not None else position
            for position, health in enumerate(self.by_frame)
            if wanted in health.kinds
        ]

    def summary(self) -> str:
        """A short plain-text report over the whole run."""
        if self.frames_checked == 0:
            lines = ["camera health: no frames were given, so nothing was checked"]
            lines.extend("  " + note for note in self.notes)
            return "\n".join(lines)
        good = sum(1 for health in self.by_frame if health.ok)
        bad = self.frames_checked - good
        lines = [
            "camera health over {} frame(s): {} ok, {} with faults "
            "(uptime {:.1%}, mean score {:.1f})".format(
                self.frames_checked, good, bad, self.uptime, self.score
            )
        ]
        if self.first_failure is not None:
            lines.append("first failure: frame {}".format(self.first_failure))
        if self.faults:
            lines.append("faults seen ({}):".format(len(self.faults)))
            for fault in self.faults:
                lines.append(
                    "  [{:<8}] {:<12} {:>4} frame(s), first at frame {:<4} {}".format(
                        fault.severity,
                        fault.kind,
                        self.fault_counts.get(fault.kind, 0),
                        self.first_seen.get(fault.kind, 0),
                        fault.message,
                    )
                )
        else:
            lines.append("no faults found")
        if self.notes:
            lines.append("notes ({}):".format(len(self.notes)))
            lines.extend("  " + note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping, including every frame."""
        return {
            "ok": bool(self.ok),
            "frames_checked": int(self.frames_checked),
            "uptime": round(float(self.uptime), 4),
            "score": round(float(self.score), 2),
            "first_failure": self.first_failure,
            "faults": [fault.to_dict() for fault in self.faults],
            "fault_counts": dict(sorted(self.fault_counts.items())),
            "first_seen": dict(sorted(self.first_seen.items())),
            "notes": list(self.notes),
            "by_frame": [health.to_dict() for health in self.by_frame],
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


def summarise_faults(
    by_frame: List[FrameHealth],
) -> Tuple[List[Fault], Dict[str, int], Dict[str, int]]:
    """Roll per-frame faults up into one entry per kind, worst instance kept."""
    worst: Dict[str, Fault] = {}
    counts: Dict[str, int] = {}
    first: Dict[str, int] = {}
    for position, health in enumerate(by_frame):
        index = health.index if health.index is not None else position
        for fault in health.faults:
            counts[fault.kind] = counts.get(fault.kind, 0) + 1
            if fault.kind not in first:
                first[fault.kind] = index
            current = worst.get(fault.kind)
            if current is None or (fault.rank, -fault.confidence) < (
                current.rank,
                -current.confidence,
            ):
                worst[fault.kind] = fault
    return sort_faults(worst.values()), counts, first
