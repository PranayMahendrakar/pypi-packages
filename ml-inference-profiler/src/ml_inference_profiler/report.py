"""The result objects: :class:`Stage` (one timed step) and :class:`ProfileReport`.

Cumulative time vs self time
----------------------------
Every stage records two numbers, and mixing them up is the classic way to read a profile
wrong:

* ``total_ms`` is **cumulative**: the wall-clock time between entering the stage and
  leaving it, including every nested stage inside it. A parent that only calls two
  children has a large ``total_ms`` and does nothing itself.
* ``self_ms`` is **self time**: ``total_ms`` minus the cumulative time of its direct
  children. This is the time actually spent in that stage's own code.

``ProfileReport.bottleneck`` is the stage with the largest *self* time, because that is
the code you would have to make faster. Sorting by ``total_ms`` instead would always
crown the outermost stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .suggest import build_suggestions

#: Column order used by :meth:`ProfileReport.to_frame`.
FRAME_COLUMNS = (
    "label",
    "path",
    "depth",
    "parent",
    "calls",
    "total_ms",
    "self_ms",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "share",
    "pct_of_total",
    "errors",
)


@dataclass(frozen=True)
class Stage:
    """One timed stage, aggregated over every call that used its label at its position.

    Attributes:
        label: The name passed to :meth:`Profiler.stage`.
        calls: How many times this stage was entered.
        total_ms: Cumulative time, nested stages included.
        mean_ms: ``total_ms / calls``.
        p95_ms: 95th percentile of the per-call cumulative times; equals the single
            measurement when the stage ran once.
        share: Percent of this stage's **level** - the stages that share its parent add
            up to 100 percent (within floating point rounding).
        depth: 0 for a top-level stage, 1 for a stage nested in it, and so on.
        parent: The parent stage's path, or ``None`` at the top level.
        self_ms: ``total_ms`` minus the cumulative time of the direct children: the time
            spent in this stage's own code.
        pct_of_total: Cumulative time as a percent of the whole run.
        self_pct_of_total: Self time as a percent of the whole run.
        median_ms: Median per-call cumulative time.
        min_ms: Fastest call.
        max_ms: Slowest call.
        first_ms: The very first call, which is where cold caches show up.
        errors: How many calls left the stage by raising. The time is still counted and
            the exception still propagates.
        path: Full ``parent/child`` path; unique within a report.
    """

    label: str
    calls: int
    total_ms: float
    mean_ms: float
    p95_ms: float
    share: float
    depth: int
    parent: Optional[str]
    self_ms: float = 0.0
    pct_of_total: float = 0.0
    self_pct_of_total: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    first_ms: float = 0.0
    errors: int = 0
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of every field."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = round(value, 6) if isinstance(value, float) else value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stage":
        """Rebuild a stage from :meth:`to_dict` output, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(repr=False)
class ProfileReport:
    """What a :class:`Profiler` found: the stage tree, the bottleneck and plain advice.

    ``print(report.summary())`` is the intended way to read one. ``report.stages`` is the
    same data as a list, ``to_frame()`` as a DataFrame and ``to_dict()`` as JSON-safe
    Python.
    """

    name: str = "pipeline"
    stages: List[Stage] = field(default_factory=list)
    total_ms: float = 0.0
    repeats: int = 1
    warmup: int = 0
    overhead_ms_per_stage: float = 0.0
    threads: int = 1
    suggestions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.suggestions:
            self.suggestions = build_suggestions(self)

    # -- derived numbers ------------------------------------------------------------
    @property
    def bottleneck(self) -> Optional[Stage]:
        """The stage with the largest self time, or ``None`` when nothing was recorded."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: (s.self_ms, s.total_ms))

    @property
    def stage_calls(self) -> int:
        """Total number of stage entries recorded."""
        return sum(s.calls for s in self.stages)

    @property
    def overhead_total_ms(self) -> float:
        """What the timing itself cost: the per-stage floor times the number of calls."""
        return self.overhead_ms_per_stage * self.stage_calls

    def find(self, label: str) -> Optional[Stage]:
        """The first stage whose label or full path is ``label``, or ``None``."""
        for stage in self.stages:
            if stage.label == label or stage.path == label:
                return stage
        return None

    # -- rendering ------------------------------------------------------------------
    def tree(self) -> str:
        """The stage tree as indented text, with each stage's share of its level."""
        if not self.stages:
            return "{0}: no stages recorded".format(self.name)
        head = (
            f"{self.name}: {len(self.stages)} stage(s), {self.total_ms:.2f} ms total, "
            f"{self.repeats} repeat(s), {self.warmup} warmup"
        )
        names = [("  " * (s.depth + 1)) + s.label for s in self.stages]
        width = max(len(n) for n in names)
        lines = [head]
        for stage, name in zip(self.stages, names):
            line = (
                f"{name.ljust(width)}  {stage.total_ms:9.2f} ms  {stage.share:5.1f}%  "
                f"x{stage.calls}"
            )
            if stage.total_ms - stage.self_ms > 0.005:
                line += f"  (self {stage.self_ms:.2f} ms)"
            if stage.errors:
                line += f"  ({stage.errors} raised)"
            lines.append(line)
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable text: the tree, the bottleneck, the timing floor and advice."""
        lines = [f"ml-inference-profiler: {self.name}", self.tree()]
        if self.stages:
            lines.append(
                "Percent is the share of its own level; the stages under one parent add"
                " up to 100%."
            )
            top = self.bottleneck
            if top is not None:
                lines.append(
                    f"Bottleneck: {top.path or top.label} - {top.self_ms:.2f} ms of self"
                    f" time, {top.self_pct_of_total:.1f}% of the run, over"
                    f" {top.calls} call(s)."
                )
            lines.append(
                f"Timing floor: about {self.overhead_ms_per_stage:.4f} ms per stage call"
                f" (measured); {self.stage_calls} call(s) cost roughly"
                f" {self.overhead_total_ms:.3f} ms of the {self.total_ms:.2f} ms measured."
            )
        if self.threads > 1:
            lines.append(
                f"Recorded from {self.threads} threads: these are per-stage wall-clock"
                " totals, not a breakdown of elapsed time."
            )
        lines.append("Suggestions:")
        lines.extend(f"  - {text}" for text in self.suggestions)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()

    def __repr__(self) -> str:
        top = self.bottleneck
        where = f", bottleneck {top.path or top.label}" if top is not None else ""
        return (
            f"<ProfileReport {self.name!r}: {len(self.stages)} stage(s),"
            f" {self.total_ms:.2f} ms{where}>"
        )

    # -- exports --------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """One row per stage, in tree order, with :data:`FRAME_COLUMNS` as columns."""
        rows = [s.to_dict() for s in self.stages]
        return pd.DataFrame(rows, columns=list(FRAME_COLUMNS))

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of the whole report."""
        top = self.bottleneck
        return {
            "name": self.name,
            "total_ms": round(self.total_ms, 6),
            "repeats": self.repeats,
            "warmup": self.warmup,
            "threads": self.threads,
            "stage_calls": self.stage_calls,
            "overhead_ms_per_stage": round(self.overhead_ms_per_stage, 9),
            "overhead_total_ms": round(self.overhead_total_ms, 6),
            "bottleneck": (top.path or top.label) if top is not None else None,
            "stages": [s.to_dict() for s in self.stages],
            "suggestions": list(self.suggestions),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """:meth:`to_dict` as a JSON string (``ensure_ascii=False``)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> Path:
        """Write :meth:`to_json` to ``path`` as UTF-8 and return the path."""
        target = Path(path)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileReport":
        """Rebuild a report from :meth:`to_dict` output."""
        if not isinstance(data, dict):
            raise ValueError(
                "a profile report must be a JSON object, got " + type(data).__name__
            )
        stages = [Stage.from_dict(s) for s in data.get("stages", []) or []]
        return cls(
            name=str(data.get("name", "pipeline")),
            stages=stages,
            total_ms=float(data.get("total_ms", 0.0) or 0.0),
            repeats=int(data.get("repeats", 1) or 1),
            warmup=int(data.get("warmup", 0) or 0),
            overhead_ms_per_stage=float(data.get("overhead_ms_per_stage", 0.0) or 0.0),
            threads=int(data.get("threads", 1) or 1),
            suggestions=list(data.get("suggestions", []) or []),
        )


def load_report(path: Union[str, Path]) -> ProfileReport:
    """Read a report written by :meth:`ProfileReport.save`."""
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read profile report {source}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc
    return ProfileReport.from_dict(data)


__all__ = ["Stage", "ProfileReport", "load_report", "FRAME_COLUMNS"]
