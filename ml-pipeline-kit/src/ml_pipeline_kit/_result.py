"""What ``Pipeline.run()`` hands back: one row per step, and a report that
explains itself without anyone opening the source."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_CHECK_KINDS = ("validate", "schema", "range")


def _ms(value: Optional[float]) -> str:
    """Fixed-width milliseconds for the summary table."""
    if value is None:
        return "".rjust(10)
    return "{0:.3f} ms".format(value).rjust(10)


def _rows(count: Optional[int]) -> str:
    return "?" if count is None else "{0:,}".format(count)


@dataclass
class StepRun:
    """One step or one check, and what happened when it ran.

    ``rows_in`` and ``rows_out`` are filled in only when the data has a length;
    for anything else they stay ``None`` rather than being guessed. A check
    never changes the data, so its two counts are always equal.
    """

    name: str
    ok: bool = True
    duration_ms: Optional[float] = None
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    error: Optional[str] = None
    kind: str = "step"
    severity: Optional[str] = None
    skipped: bool = False

    @property
    def is_check(self) -> bool:
        """Was this a check rather than a step that changed the data?"""
        return self.kind in _CHECK_KINDS

    @property
    def status(self) -> str:
        """``ok``, ``warn``, ``skip`` or ``FAIL``, as shown in the summary."""
        if self.ok:
            return "ok"
        if self.severity == "warn":
            return "warn"
        if self.skipped:
            return "skip"
        return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of this row."""
        return {
            "name": self.name,
            "kind": self.kind,
            "ok": bool(self.ok),
            "status": self.status,
            "duration_ms": self.duration_ms,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "error": self.error,
            "severity": self.severity,
            "skipped": bool(self.skipped),
        }


@dataclass
class Result:
    """The output of a run, plus the log of how it got there.

    ``output`` is the value the last step produced. ``ok`` is True when nothing
    failed; a check with ``severity="warn"`` records itself in ``warnings`` and
    leaves ``ok`` alone. ``stopped_at`` names the step or check that ended the
    run early, and is ``None`` when every step ran.
    """

    output: Any = None
    name: str = "pipeline"
    steps: List[StepRun] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stopped_at: Optional[str] = None
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    n_planned: int = 0
    collect_timings: bool = True

    # ----------------------------------------------------------------- basics
    @property
    def ok(self) -> bool:
        """True when no step raised and no ``severity="error"`` check failed."""
        return not self.failures

    @property
    def n_steps(self) -> int:
        """How many steps and checks actually ran."""
        return len(self.steps)

    @property
    def timings(self) -> Dict[str, float]:
        """``{step name: milliseconds}`` for every timed step, in run order."""
        return {run.name: run.duration_ms for run in self.steps if run.duration_ms is not None}

    @property
    def duration_ms(self) -> Optional[float]:
        """Total milliseconds across every timed step, or None when untimed."""
        timed = [run.duration_ms for run in self.steps if run.duration_ms is not None]
        if not timed:
            return None
        return round(float(sum(timed)), 3)

    @property
    def failed_steps(self) -> List[StepRun]:
        """The rows that did not pass, checks and steps alike."""
        return [run for run in self.steps if not run.ok]

    def step(self, name: str) -> StepRun:
        """The row for one step by name."""
        for run in self.steps:
            if run.name == name:
                return run
        known = ", ".join(run.name for run in self.steps) or "(nothing ran)"
        raise KeyError("no step named '{0}' in this result; it has: {1}".format(name, known))

    # ------------------------------------------------------------- reporting
    def summary(self) -> str:
        """The plain-text report, safe for any console."""
        if self.ok:
            headline = "ml-pipeline-kit: pipeline '{0}' ok, {1} of {2} steps ran".format(
                self.name, self.n_steps, self.n_planned
            )
        else:
            headline = "ml-pipeline-kit: pipeline '{0}' FAILED, {1} of {2} steps ran".format(
                self.name, self.n_steps, self.n_planned
            )
        total = self.duration_ms
        if total is not None:
            headline += " in {0:.3f} ms".format(total)
        lines = [headline]
        if self.stopped_at is not None:
            lines.append("  stopped at: '{0}'".format(self.stopped_at))
        lines.append("  rows      : {0} in, {1} out".format(_rows(self.rows_in), _rows(self.rows_out)))
        if self.steps:
            width = max(len(run.name) for run in self.steps)
            kind_width = max(len(run.kind) for run in self.steps)
            lines.append("  steps     :")
            for position, run in enumerate(self.steps, start=1):
                row = "    {0}. {1}  {2}  {3}{4}".format(
                    position,
                    run.name.ljust(width),
                    run.kind.ljust(kind_width),
                    run.status.ljust(4),
                    _ms(run.duration_ms),
                )
                if not run.is_check and run.rows_in is not None:
                    row += "  {0} rows in, {1} rows out".format(_rows(run.rows_in), _rows(run.rows_out))
                lines.append(row)
        if self.failures:
            lines.append("  failures  :")
            lines.extend("    - " + text for text in self.failures)
        if self.warnings:
            lines.append("  warnings  :")
            lines.extend("    - " + text for text in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole run.

        ``output`` is deliberately left out: it is your data, in whatever shape
        your steps produced, and is reached through ``result.output``.
        """
        return {
            "name": self.name,
            "ok": self.ok,
            "stopped_at": self.stopped_at,
            "n_planned": self.n_planned,
            "n_steps": self.n_steps,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duration_ms": self.duration_ms,
            "steps": [run.to_dict() for run in self.steps],
            "timings": self.timings,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return "Result(name={0!r}, ok={1}, steps={2}, failures={3})".format(
            self.name, self.ok, self.n_steps, len(self.failures)
        )
