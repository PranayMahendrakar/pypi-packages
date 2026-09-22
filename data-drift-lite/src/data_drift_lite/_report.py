"""Result objects: ``ColumnDrift``, ``SchemaDrift`` and ``DriftReport``.

All three are plain dataclasses. ``to_dict()`` is JSON-safe (no numpy scalars,
no NaN/inf) and ``summary()`` is a short human-readable text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_NAME_WIDTH = 40


def json_safe(obj: Any) -> Any:
    """Recursively turn numpy scalars, tuples and non-finite floats into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_p(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"


def _clip(text: str, width: int = _NAME_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


@dataclass
class ColumnDrift:
    """Drift result for one column.

    ``statistic`` and ``p_value`` come from the KS test (numeric columns) or the
    chi-square test (categorical columns); ``psi`` is the Population Stability
    Index. Each is ``None`` when it could not be computed; ``notes`` says why.
    """

    kind: str
    statistic: Optional[float]
    p_value: Optional[float]
    psi: Optional[float]
    drifted: bool
    reference_stats: Dict[str, Any]
    current_stats: Dict[str, Any]
    name: str = ""
    test: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this column's result."""
        return json_safe(
            {
                "name": self.name,
                "kind": self.kind,
                "test": self.test,
                "statistic": self.statistic,
                "p_value": self.p_value,
                "psi": self.psi,
                "drifted": self.drifted,
                "reference_stats": self.reference_stats,
                "current_stats": self.current_stats,
                "notes": list(self.notes),
            }
        )


@dataclass
class SchemaDrift:
    """Structural differences between the reference and the current data."""

    missing_columns: List[Any] = field(default_factory=list)
    new_columns: List[Any] = field(default_factory=list)
    dtype_changed: Dict[Any, Tuple[str, str]] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        """True when a reference column is missing or changed its type family.

        New columns are reported but do not count: nothing the reference relied on
        has changed.
        """
        return bool(self.missing_columns or self.dtype_changed)

    @property
    def changed(self) -> bool:
        """True when anything at all differs, new columns included."""
        return bool(self.missing_columns or self.new_columns or self.dtype_changed)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the schema differences."""
        return json_safe(
            {
                "missing_columns": list(self.missing_columns),
                "new_columns": list(self.new_columns),
                "dtype_changed": {
                    name: {"reference": before, "current": after}
                    for name, (before, after) in self.dtype_changed.items()
                },
                "drifted": self.drifted,
            }
        )


@dataclass
class DriftReport:
    """Everything :func:`data_drift_lite.detect` found.

    ``columns`` maps each compared column to its :class:`ColumnDrift`, in reference
    order. Columns that are missing from the current data or changed type family are
    not compared; they live in ``schema``.
    """

    columns: Dict[Any, ColumnDrift]
    schema: SchemaDrift = field(default_factory=SchemaDrift)
    threshold: Optional[float] = 0.05
    psi_threshold: Optional[float] = 0.2
    reference_rows: int = 0
    current_rows: int = 0
    notes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- headline numbers

    @property
    def drifted_columns(self) -> List[Any]:
        """Names of the compared columns flagged as drifted, in reference order."""
        return [name for name, col in self.columns.items() if col.drifted]

    @property
    def drift_share(self) -> float:
        """Fraction of compared columns that drifted (0.0 when nothing was compared)."""
        return len(self.drifted_columns) / len(self.columns) if self.columns else 0.0

    @property
    def drifted(self) -> bool:
        """True when any column drifted, a column went missing, or a type family changed."""
        return bool(self.drifted_columns) or self.schema.drifted

    @property
    def missing_columns(self) -> List[Any]:
        """Reference columns absent from the current data."""
        return self.schema.missing_columns

    @property
    def new_columns(self) -> List[Any]:
        """Current columns absent from the reference."""
        return self.schema.new_columns

    @property
    def dtype_changed(self) -> Dict[Any, Tuple[str, str]]:
        """Columns whose type family changed: name -> (reference dtype, current dtype)."""
        return self.schema.dtype_changed

    # ---------------------------------------------------------------- output

    def _rule_text(self) -> str:
        parts = []
        if self.threshold is not None:
            parts.append(f"p < {self.threshold:g}")
        if self.psi_threshold is not None:
            parts.append(f"PSI > {self.psi_threshold:g}")
        return "drifted when " + " or ".join(parts) if parts else "drift rules disabled (both thresholds are None)"

    def _table_lines(self) -> List[str]:
        headers = ("column", "kind", "test", "statistic", "p-value", "PSI", "status")
        rows = [
            (
                _clip(str(name)),
                col.kind,
                col.test or "-",
                _fmt(col.statistic),
                _fmt_p(col.p_value),
                _fmt(col.psi),
                "DRIFTED" if col.drifted else "ok",
            )
            for name, col in self.columns.items()
        ]
        widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
        right = {3, 4, 5}

        def render(cells: Tuple[str, ...]) -> str:
            padded = [
                cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]) for i, cell in enumerate(cells)
            ]
            return "  " + "  ".join(padded).rstrip()

        return [render(headers)] + [render(row) for row in rows]

    def summary(self) -> str:
        """Human-readable summary: verdict, one line per column, schema changes, notes."""
        n_compared = len(self.columns)
        n_drifted = len(self.drifted_columns)
        if n_drifted:
            verdict = f"DRIFT DETECTED: {n_drifted} of {n_compared} columns drifted ({self.drift_share:.0%})"
        elif self.schema.drifted:
            verdict = f"DRIFT DETECTED: schema changed (0 of {n_compared} columns drifted)"
        elif n_compared:
            verdict = f"no drift detected ({n_compared} columns compared)"
        else:
            verdict = "no columns compared"
        lines = [
            f"data-drift-lite: {verdict}",
            f"reference rows: {self.reference_rows:,} | current rows: {self.current_rows:,} | {self._rule_text()}",
        ]
        if n_compared:
            lines.append("")
            lines.extend(self._table_lines())
        if self.schema.changed:
            lines.append("")
            lines.append("schema:")
            if self.schema.missing_columns:
                names = ", ".join(str(c) for c in self.schema.missing_columns)
                lines.append(f"  - missing in current: {names}")
            if self.schema.new_columns:
                names = ", ".join(str(c) for c in self.schema.new_columns)
                lines.append(f"  - new in current: {names}")
            for name, (before, after) in self.schema.dtype_changed.items():
                lines.append(f"  - dtype changed: {name} ({before} -> {after})")
        column_notes = [f"{name}: {note}" for name, col in self.columns.items() for note in col.notes]
        if self.notes or column_notes:
            lines.append("")
            lines.append("notes:")
            lines.extend(f"  - {note}" for note in list(self.notes) + column_notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict with the verdict, every column, the schema section and notes."""
        return json_safe(
            {
                "drifted": self.drifted,
                "drift_share": self.drift_share,
                "drifted_columns": self.drifted_columns,
                "threshold": self.threshold,
                "psi_threshold": self.psi_threshold,
                "reference_rows": self.reference_rows,
                "current_rows": self.current_rows,
                "columns": {name: col.to_dict() for name, col in self.columns.items()},
                "schema": self.schema.to_dict(),
                "notes": list(self.notes),
            }
        )

    def __str__(self) -> str:
        return self.summary()
