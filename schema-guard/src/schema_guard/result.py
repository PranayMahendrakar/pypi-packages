"""Validation results: :class:`Problem`, :class:`ValidationResult` and :class:`SchemaError`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

#: Every problem kind that validation can report.
PROBLEM_KINDS = (
    "missing_column",
    "extra_column",
    "dtype_mismatch",
    "unexpected_null",
    "unknown_category",
    "out_of_range",
    "wrong_order",
)

#: Problem severities. Only ``"error"`` problems make a result fail.
SEVERITIES = ("error", "warning")


@dataclass
class Problem:
    """One thing that is wrong with a DataFrame relative to a schema.

    ``kind`` is one of :data:`PROBLEM_KINDS`, ``column`` is the affected column name
    (``None`` for frame-level problems such as ``wrong_order``), ``message`` is a short
    human sentence and ``detail`` holds JSON-safe specifics (counts, examples, expectations).
    ``severity`` is ``"error"`` or ``"warning"``; ``out_of_range`` is a warning unless
    validation ran with ``strict_ranges=True``.
    """

    kind: str
    column: Optional[Any]
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.kind not in PROBLEM_KINDS:
            raise ValueError(f"unknown problem kind {self.kind!r}; expected one of {PROBLEM_KINDS}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}; expected one of {SEVERITIES}")
        if self.detail is None:
            self.detail = {}

    @property
    def is_error(self) -> bool:
        """True when this problem makes validation fail."""
        return self.severity == "error"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict with ``kind``, ``column``, ``message``, ``detail`` and ``severity``."""
        return {
            "kind": self.kind,
            "column": self.column,
            "message": self.message,
            "detail": dict(self.detail),
            "severity": self.severity,
        }

    def __str__(self) -> str:
        where = f" {self.column!r}" if self.column is not None else ""
        return f"[{self.severity}] {self.kind}{where}: {self.message}"


class SchemaError(ValueError):
    """Raised when a DataFrame does not match a schema and the caller asked for an exception.

    ``problems`` holds every :class:`Problem` that was found, so callers can inspect or
    log them instead of parsing the message.
    """

    def __init__(self, problems: Iterable[Problem], message: Optional[str] = None) -> None:
        self.problems: List[Problem] = list(problems)
        if message is None:
            count = len(self.problems)
            noun = "problem" if count == 1 else "problems"
            lines = [f"DataFrame does not match the schema ({count} {noun}):"]
            lines.extend(f"  - {problem}" for problem in self.problems)
            message = "\n".join(lines)
        super().__init__(message)


@dataclass
class ValidationResult:
    """Outcome of :meth:`Schema.validate`.

    ``ok`` is True when no error-level problem was found (warnings such as
    ``out_of_range`` do not fail validation). ``problems`` lists everything found,
    errors and warnings alike, in a stable order: missing columns, extra columns,
    column order, then per-column content problems in schema order.
    """

    problems: List[Problem] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0

    @property
    def ok(self) -> bool:
        """True when there are no error-level problems."""
        return not any(problem.is_error for problem in self.problems)

    @property
    def errors(self) -> List[Problem]:
        """Problems that make validation fail."""
        return [problem for problem in self.problems if problem.is_error]

    @property
    def warnings(self) -> List[Problem]:
        """Problems that are reported but do not fail validation."""
        return [problem for problem in self.problems if not problem.is_error]

    def summary(self) -> str:
        """Human-readable multi-line summary (ASCII only, safe for any console)."""
        status = "OK" if self.ok else "FAILED"
        head = f"Schema validation: {status} - {self.n_rows} rows x {self.n_columns} columns"
        if not self.problems:
            return head + " (no problems)"
        n_err, n_warn = len(self.errors), len(self.warnings)
        head += f" ({n_err} error{'s' if n_err != 1 else ''}, {n_warn} warning{'s' if n_warn != 1 else ''})"
        lines = [head]
        lines.extend(f"  {problem}" for problem in self.problems)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: ``ok``, row/column counts, error/warning counts and ``problems``."""
        return {
            "ok": self.ok,
            "n_rows": int(self.n_rows),
            "n_columns": int(self.n_columns),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "problems": [problem.to_dict() for problem in self.problems],
        }

    def raise_if_invalid(self) -> "ValidationResult":
        """Raise :class:`SchemaError` listing every error when ``ok`` is False; else return self."""
        if not self.ok:
            raise SchemaError(self.errors)
        return self
