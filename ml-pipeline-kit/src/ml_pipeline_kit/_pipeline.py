"""The pipeline: add steps, add checks, run, and get a result that explains itself."""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._checks import (
    describe_schema,
    dtype_label,
    normalize_schema,
    range_problems,
    run_check,
    schema_problems,
)
from ._errors import PipelineError, StepError, ValidationError
from ._result import Result, StepRun

LOG = logging.getLogger(__name__)

ON_ERROR = ("raise", "skip", "stop")
SEVERITIES = ("error", "warn")
_ROLE_KINDS = ("preprocess", "predict", "postprocess")
_STEP_KINDS = ("step",) + _ROLE_KINDS
_CHECK_KINDS = ("validate", "schema", "range")
_KINDS = _STEP_KINDS + _CHECK_KINDS

_FORMAT = "ml-pipeline-kit/pipeline"
_FORMAT_VERSION = 1

_LOAD_NOTE = (
    "This file records step names and configuration only, never the callables. "
    "After Pipeline.load(), re-register each step with pipeline.bind(name, func); "
    "pipeline.unbound lists the ones still waiting."
)


# --------------------------------------------------------------------- helpers
def _row_count(data: Any) -> Optional[int]:
    """How many rows this value has, or None when it does not have rows.

    Text is a single value, not one row per character, so ``str`` and ``bytes``
    report None like any other scalar.
    """
    if data is None or isinstance(data, (str, bytes, bytearray)):
        return None
    try:
        return int(len(data))
    except (TypeError, ValueError):
        return None


def _defensive_copy(data: Any) -> Any:
    """A copy the steps may mutate freely, so the caller's value never changes.

    pandas, numpy and the builtin containers are copied. Anything else is passed
    through untouched, because copying an arbitrary object is not this library's
    decision to make - and that includes an arbitrary object stored inside a
    DataFrame cell, which ``DataFrame.copy(deep=True)`` does not copy either.
    """
    if isinstance(data, (pd.DataFrame, pd.Series, pd.Index)):
        return data.copy(deep=True)
    if isinstance(data, np.ndarray):
        return data.copy()
    if isinstance(data, (list, tuple, dict, set, bytearray)):
        try:
            return copy.deepcopy(data)
        except Exception:  # noqa: BLE001 - an uncopyable member, shallow is still better than nothing
            LOG.debug("deep copy of %s failed, falling back to a shallow copy", type(data).__name__)
            return copy.copy(data)
    return data


def _guard_duplicate_columns(data: Any, where: str) -> None:
    """Duplicate column names are reported here, not as an AttributeError later."""
    if not isinstance(data, pd.DataFrame):
        return
    counts = pd.Series(list(data.columns)).value_counts()
    duplicates = sorted(str(name) for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "ml-pipeline-kit: {0} was given a DataFrame with duplicate column names: {1}. "
            "Rename or drop them first; every check reads columns by name.".format(
                where, ", ".join(duplicates)
            )
        )


def _callable_name(func: Any, fallback: str) -> str:
    """A readable default name for a step."""
    name = getattr(func, "__name__", None) or type(func).__name__
    name = str(name).strip()
    if not name or name.startswith("<") or name in ("partial", "function", "method"):
        return fallback
    return name


def _callable_label(func: Any) -> Optional[str]:
    """What the callable was called, recorded in the saved file as a hint.

    Anonymous callables have nothing worth recording, so they record nothing.
    """
    label = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if label is None:
        return None
    label = str(label)
    return None if "<" in label else label


def _elapsed(start: Optional[float]) -> Optional[float]:
    return None if start is None else round((perf_counter() - start) * 1000.0, 3)


def _number(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "expect_range() needs a number for {0}, got {1!r}".format(field_name, value)
        ) from None
    if not np.isfinite(out):
        raise ValueError("expect_range() needs a finite number for {0}, got {1!r}".format(field_name, value))
    return out


@dataclass
class _Entry:
    """One registered step or check, with the configuration that can be saved."""

    name: str
    kind: str = "step"
    func: Optional[Callable[[Any], Any]] = None
    on_error: str = "raise"
    severity: str = "error"
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_check(self) -> bool:
        return self.kind in _CHECK_KINDS

    @property
    def needs_callable(self) -> bool:
        """Schema and range checks are pure configuration; the rest need code."""
        return self.kind not in ("schema", "range")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "on_error": self.on_error,
            "severity": self.severity,
            "config": dict(self.config),
        }


class Pipeline:
    """A preprocess, predict, validate and log pipeline you build in a few lines.

    Every step is timed, its row counts recorded and its failure reported with
    the step's name, so none of that has to be written by hand::

        pipe = Pipeline("scoring").preprocess(clean).predict(score)
        result = pipe.run(frame)
        print(result.summary())

    Steps run in the order they were added, each one taking the value the
    previous one returned. Checks are steps too: they look at the data at that
    point and never change it.
    """

    def __init__(self, name: str = "pipeline") -> None:
        text = str(name).strip() if name is not None else ""
        self.name = text or "pipeline"
        self._entries: List[_Entry] = []

    # ------------------------------------------------------------ registering
    def _unique(self, name: Optional[str], fallback: str) -> str:
        wanted = str(name).strip() if name is not None else ""
        wanted = wanted or fallback
        taken = {entry.name for entry in self._entries}
        if wanted not in taken:
            return wanted
        suffix = 2
        while "{0}-{1}".format(wanted, suffix) in taken:
            suffix += 1
        return "{0}-{1}".format(wanted, suffix)

    def _add_callable(
        self,
        step: Callable[[Any], Any],
        *,
        name: Optional[str],
        on_error: str,
        kind: str,
    ) -> "Pipeline":
        if not callable(step):
            raise TypeError(
                "{0}() needs a callable that takes the data and returns it, got {1}".format(
                    kind if kind in _ROLE_KINDS else "add", type(step).__name__
                )
            )
        if on_error not in ON_ERROR:
            raise ValueError(
                "on_error must be one of {0}, got {1!r}".format(", ".join(ON_ERROR), on_error)
            )
        entry = _Entry(
            name=self._unique(name, _callable_name(step, kind)),
            kind=kind,
            func=step,
            on_error=on_error,
            config={"callable": _callable_label(step)},
        )
        self._entries.append(entry)
        return self

    def add(
        self,
        step: Callable[[Any], Any],
        *,
        name: Optional[str] = None,
        on_error: str = "raise",
    ) -> "Pipeline":
        """Add a step: any callable that takes the data and returns it.

        `name` defaults to the callable's own name, or to "step" for a lambda.
        `on_error` is "raise" (let the error out, named and chained), "skip"
        (carry on with the data the previous step produced, and record it) or
        "stop" (end the run and report it in the result).
        """
        return self._add_callable(step, name=name, on_error=on_error, kind="step")

    def preprocess(
        self,
        func: Callable[[Any], Any],
        *,
        name: Optional[str] = None,
        on_error: str = "raise",
    ) -> "Pipeline":
        """Add a step labelled `preprocess`. Same mechanism as :meth:`add`."""
        return self._add_callable(func, name=name, on_error=on_error, kind="preprocess")

    def predict(
        self,
        func: Callable[[Any], Any],
        *,
        name: Optional[str] = None,
        on_error: str = "raise",
    ) -> "Pipeline":
        """Add a step labelled `predict`. Same mechanism as :meth:`add`."""
        return self._add_callable(func, name=name, on_error=on_error, kind="predict")

    def postprocess(
        self,
        func: Callable[[Any], Any],
        *,
        name: Optional[str] = None,
        on_error: str = "raise",
    ) -> "Pipeline":
        """Add a step labelled `postprocess`. Same mechanism as :meth:`add`."""
        return self._add_callable(func, name=name, on_error=on_error, kind="postprocess")

    def validate(
        self,
        check: Callable[[Any], Any],
        *,
        name: Optional[str] = None,
        severity: str = "error",
    ) -> "Pipeline":
        """Add a check at this point in the pipeline.

        `check` is called with the data and returns ``True`` / ``False``, a
        ``(bool, message)`` pair, or a boolean mask with one value per row.
        `severity` "error" stops the run before the next step and names the
        check in ``result.failures``; "warn" records it in ``result.warnings``
        and carries on. A check that raises is a failed check, not a crash.
        """
        if not callable(check):
            raise TypeError(
                "validate() needs a callable taking the data, got {0}".format(type(check).__name__)
            )
        if severity not in SEVERITIES:
            raise ValueError(
                "severity must be one of {0}, got {1!r}".format(", ".join(SEVERITIES), severity)
            )
        entry = _Entry(
            name=self._unique(name, _callable_name(check, "check")),
            kind="validate",
            func=check,
            severity=severity,
            config={"callable": _callable_label(check)},
        )
        self._entries.append(entry)
        return self

    def expect_schema(
        self,
        schema: Any,
        *,
        name: Optional[str] = None,
        severity: str = "error",
    ) -> "Pipeline":
        """Require these columns, and these dtypes, at this point.

        `schema` is ``{"age": "int", "city": "str"}``, a list of column names
        when only presence matters, or a DataFrame to copy the dtypes from.
        Dtypes match by family, so "int" accepts any integer width. Columns
        beyond the schema are allowed.
        """
        if severity not in SEVERITIES:
            raise ValueError(
                "severity must be one of {0}, got {1!r}".format(", ".join(SEVERITIES), severity)
            )
        normalized = normalize_schema(schema)
        entry = _Entry(
            name=self._unique(name, "schema"),
            kind="schema",
            severity=severity,
            config={"schema": {key: dtype_label(value) for key, value in normalized.items()}},
        )
        self._entries.append(entry)
        return self

    def expect_range(
        self,
        column: Optional[str],
        low: Optional[float] = None,
        high: Optional[float] = None,
        *,
        name: Optional[str] = None,
        severity: str = "error",
    ) -> "Pipeline":
        """Require the numbers in `column` to sit between `low` and `high`.

        Either bound may be left out for an open side. Missing values are not
        counted as out of range; values that are present but cannot be read as a
        number are counted and named in ``result.warnings``, and a column with
        no usable numbers at all records a note there too, instead of passing
        silently. A date or a duration is reported as a type problem, never
        compared as nanoseconds. Pass ``None`` as the column when the data
        itself is the series of numbers.
        """
        if severity not in SEVERITIES:
            raise ValueError(
                "severity must be one of {0}, got {1!r}".format(", ".join(SEVERITIES), severity)
            )
        low_value = _number(low, "low")
        high_value = _number(high, "high")
        if low_value is None and high_value is None:
            raise ValueError("expect_range() needs at least one of low= or high=")
        if low_value is not None and high_value is not None and low_value > high_value:
            raise ValueError(
                "expect_range() got low={0:.6g} above high={1:.6g}".format(low_value, high_value)
            )
        label = "range" if column is None else "range[{0}]".format(column)
        entry = _Entry(
            name=self._unique(name, label),
            kind="range",
            severity=severity,
            config={
                "column": None if column is None else str(column),
                "low": low_value,
                "high": high_value,
            },
        )
        self._entries.append(entry)
        return self

    # -------------------------------------------------------------- inspecting
    @property
    def step_names(self) -> List[str]:
        """The names of every step and check, in run order."""
        return [entry.name for entry in self._entries]

    @property
    def unbound(self) -> List[str]:
        """Steps loaded from a file that are still waiting for their callable."""
        return [e.name for e in self._entries if e.needs_callable and e.func is None]

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return "Pipeline(name={0!r}, steps={1})".format(self.name, self.step_names)

    def describe(self) -> str:
        """A plain-text listing of the steps, for the CLI and for humans."""
        lines = ["pipeline '{0}' with {1} step(s)".format(self.name, len(self._entries))]
        if not self._entries:
            lines.append("  (no steps yet)")
        width = max((len(e.name) for e in self._entries), default=1)
        for position, entry in enumerate(self._entries, start=1):
            detail = ""
            if entry.kind == "schema":
                detail = describe_schema(entry.config.get("schema") or {})
            elif entry.kind == "range":
                low, high = entry.config.get("low"), entry.config.get("high")
                bounds = "{0} to {1}".format("-inf" if low is None else low, "inf" if high is None else high)
                detail = "{0} in {1}".format(entry.config.get("column") or "the data", bounds)
            elif entry.config.get("callable"):
                detail = "callable: {0}".format(entry.config["callable"])
            if entry.func is None and entry.needs_callable:
                detail = (detail + "; " if detail else "") + "needs bind()"
            if entry.is_check:
                detail = "severity={0}".format(entry.severity) + ("; " + detail if detail else "")
            else:
                detail = "on_error={0}".format(entry.on_error) + ("; " + detail if detail else "")
            lines.append("  {0}. {1}  {2}  {3}".format(position, entry.name.ljust(width), entry.kind.ljust(10), detail))
        missing = self.unbound
        if missing:
            lines.append("  re-register with pipeline.bind(name, func): " + ", ".join(missing))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe description of the pipeline, without any callables."""
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "name": self.name,
            "note": _LOAD_NOTE,
            "steps": [entry.to_dict() for entry in self._entries],
        }

    # ----------------------------------------------------------- save and load
    def save(self, path: Union[str, Path]) -> Path:
        """Write the step names and configuration to `path` as JSON.

        The callables are never written: a pipeline is code plus configuration,
        and only the configuration is data. The file says so, and
        :meth:`load` hands back a pipeline whose steps must be re-registered
        with :meth:`bind` before it will run.
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        LOG.debug("saved pipeline '%s' to %s", self.name, target)
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """Read a pipeline saved by :meth:`save`.

        Schema and range checks come back ready to run. Steps and custom checks
        come back as names waiting for their callable: re-register each one with
        ``pipeline.bind(name, func)``. ``pipeline.unbound`` lists them, and
        running before they are bound raises a ``PipelineError`` that names them.
        """
        source = Path(path)
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
            raise ValueError(
                "{0} is not an ml-pipeline-kit pipeline file (no '{1}' marker)".format(source, _FORMAT)
            )
        pipe = cls(payload.get("name") or "pipeline")
        for raw in payload.get("steps") or []:
            if not isinstance(raw, dict):
                raise ValueError("{0} has a step that is not an object".format(source))
            kind = str(raw.get("kind") or "step")
            if kind not in _KINDS:
                raise ValueError(
                    "{0} has an unknown step kind {1!r}; expected one of {2}".format(
                        source, kind, ", ".join(_KINDS)
                    )
                )
            on_error = str(raw.get("on_error") or "raise")
            severity = str(raw.get("severity") or "error")
            if on_error not in ON_ERROR or severity not in SEVERITIES:
                raise ValueError("{0} has a step with an unknown on_error or severity".format(source))
            config = raw.get("config") or {}
            if kind == "schema" and not isinstance(config.get("schema"), dict):
                raise ValueError("{0} has a schema step without a schema".format(source))
            pipe._entries.append(
                _Entry(
                    name=pipe._unique(raw.get("name"), kind),
                    kind=kind,
                    func=None,
                    on_error=on_error,
                    severity=severity,
                    config=dict(config),
                )
            )
        return pipe

    def bind(self, name: str, func: Callable[[Any], Any]) -> "Pipeline":
        """Give a loaded step its callable back."""
        if not callable(func):
            raise TypeError("bind() needs a callable, got {0}".format(type(func).__name__))
        for entry in self._entries:
            if entry.name == name:
                if not entry.needs_callable:
                    raise ValueError(
                        "step '{0}' is a {1} check built from configuration and needs no callable".format(
                            name, entry.kind
                        )
                    )
                entry.func = func
                return self
        known = ", ".join(self.step_names) or "(no steps)"
        raise KeyError("no step named '{0}' in pipeline '{1}'; it has: {2}".format(name, self.name, known))

    # ------------------------------------------------------------------ running
    def _guard_ready(self) -> None:
        missing = self.unbound
        if missing:
            raise PipelineError(
                "pipeline '{0}' has steps with no callable: {1}. A saved pipeline records names and "
                "configuration only; re-register each one with pipeline.bind(name, func) before "
                "running.".format(self.name, ", ".join(missing))
            )

    def _check_now(self, entry: _Entry, data: Any) -> Tuple[bool, Optional[str]]:
        """Run one check and return ``(passed, message)``."""
        if entry.kind == "schema":
            problems = schema_problems(data, entry.config.get("schema") or {})
            return (not problems), ("; ".join(problems) if problems else None)
        if entry.kind == "range":
            problems, note = range_problems(
                data,
                entry.config.get("column"),
                entry.config.get("low"),
                entry.config.get("high"),
            )
            if problems:
                detail = "; ".join(problems)
                return False, (detail + "; " + note) if note else detail
            return True, note
        return run_check(entry.func, data)

    def run(self, data: Any, *, collect_timings: bool = True) -> Result:
        """Run every step over `data` and report what happened.

        The caller's value is never touched: pandas, numpy and the builtin
        containers are copied before the first step. Objects the caller stored
        inside a DataFrame cell are the one exception, because pandas does not
        copy those either. An empty pipeline returns the input unchanged with
        ``ok`` True and a warning saying so.

        A failed ``severity="error"`` check stops the run before the next step
        and is named in ``result.failures``; it does not raise. A step that
        raises is reported with its name, the rows that reached it and the
        original exception chained, either as a ``StepError`` (``on_error=
        "raise"``, the default) or inside the result (``"skip"``, ``"stop"``).
        """
        self._guard_ready()
        _guard_duplicate_columns(data, "run()")
        current = _defensive_copy(data)
        rows_in = _row_count(current)
        runs: List[StepRun] = []
        failures: List[str] = []
        warnings: List[str] = []
        stopped_at: Optional[str] = None

        if not self._entries:
            warnings.append(
                "pipeline '{0}' has no steps; the input was returned unchanged".format(self.name)
            )

        for entry in self._entries:
            rows_here = _row_count(current)
            start = perf_counter() if collect_timings else None

            if entry.is_check:
                passed, message = self._check_now(entry, current)
                duration = _elapsed(start)
                runs.append(
                    StepRun(
                        name=entry.name,
                        ok=passed,
                        duration_ms=duration,
                        rows_in=rows_here,
                        rows_out=rows_here,
                        error=None if passed else message,
                        kind=entry.kind,
                        severity=entry.severity,
                    )
                )
                if passed:
                    if message:
                        warnings.append("check '{0}': {1}".format(entry.name, message))
                    continue
                detail = message or "the check did not pass"
                if entry.severity == "warn":
                    warnings.append("check '{0}' warned: {1}".format(entry.name, detail))
                    continue
                failures.append("check '{0}' failed: {1}".format(entry.name, detail))
                stopped_at = entry.name
                LOG.debug("pipeline '%s' stopped at check '%s'", self.name, entry.name)
                break

            try:
                output = entry.func(current)
            except Exception as exc:  # noqa: BLE001 - re-raised named and chained below
                duration = _elapsed(start)
                rows_text = "an unknown number of rows" if rows_here is None else "{0} rows".format(rows_here)
                detail = "{0}: {1}".format(type(exc).__name__, exc)
                runs.append(
                    StepRun(
                        name=entry.name,
                        ok=False,
                        duration_ms=duration,
                        rows_in=rows_here,
                        rows_out=None,
                        error="{0} ({1} reached it)".format(detail, rows_text),
                        kind=entry.kind,
                        skipped=entry.on_error == "skip",
                    )
                )
                if entry.on_error == "raise":
                    raise StepError(
                        "pipeline '{0}' step '{1}' failed with {2} in: {3}".format(
                            self.name, entry.name, rows_text, detail
                        ),
                        step=entry.name,
                        rows_in=rows_here,
                    ) from exc
                if entry.on_error == "stop":
                    failures.append(
                        "step '{0}' failed with {1} in: {2}".format(entry.name, rows_text, detail)
                    )
                    stopped_at = entry.name
                    break
                failures.append(
                    "step '{0}' failed with {1} in and was skipped, the previous data carried on: "
                    "{2}".format(entry.name, rows_text, detail)
                )
                continue

            duration = _elapsed(start)
            current = output
            runs.append(
                StepRun(
                    name=entry.name,
                    ok=True,
                    duration_ms=duration,
                    rows_in=rows_here,
                    rows_out=_row_count(current),
                    kind=entry.kind,
                )
            )

        return Result(
            output=current,
            name=self.name,
            steps=runs,
            failures=failures,
            warnings=warnings,
            stopped_at=stopped_at,
            rows_in=rows_in,
            rows_out=_row_count(current),
            n_planned=len(self._entries),
            collect_timings=collect_timings,
        )

    def __call__(self, data: Any) -> Any:
        """Run the pipeline and return the output value alone.

        This is the form to use inside a service. It raises rather than hand
        back a half-finished value: a failed check raises ``ValidationError``, a
        step that ended the run raises ``StepError``. A step with
        ``on_error="skip"`` is a tolerated failure and still returns its output.
        """
        result = self.run(data, collect_timings=False)
        if result.stopped_at is None:
            return result.output
        last = result.steps[-1] if result.steps else None
        detail = (last.error if last is not None and last.error else None) or "the run stopped"
        if last is not None and last.is_check:
            raise ValidationError(
                "pipeline '{0}' stopped at check '{1}': {2}".format(self.name, result.stopped_at, detail),
                check=result.stopped_at,
            )
        raise StepError(
            "pipeline '{0}' stopped at step '{1}': {2}".format(self.name, result.stopped_at, detail),
            step=result.stopped_at,
            rows_in=None if last is None else last.rows_in,
        )


def run(
    data: Any,
    *steps: Callable[[Any], Any],
    name: str = "pipeline",
    collect_timings: bool = True,
) -> Result:
    """Run a few callables over `data` as a pipeline, in one line.

    ``run(frame, clean, score)`` is the same as building a
    :class:`Pipeline`, adding each callable with :meth:`Pipeline.add` and
    running it, and gives back the same :class:`Result`. Reach for the class
    when you want checks, names or error handling per step.
    """
    pipe = Pipeline(name)
    for step in steps:
        pipe.add(step)
    return pipe.run(data, collect_timings=collect_timings)
