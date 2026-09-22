"""The object ``analyze()`` hands back, which is meant to explain itself."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._model import StepChange, Trend, _clean, _stamp, _when_text
from ._tariff import Tariff

LOG = logging.getLogger(__name__)


def _num(value: Any, digits: int = 3) -> str:
    """A compact number for human text, ASCII only."""
    number = _clean(value)
    if number is None:
        return "n/a"
    size = abs(number)
    if size >= 1000:
        return f"{number:,.0f}"
    if size >= 10:
        return f"{number:,.1f}"
    return f"{number:,.{digits}g}"


def _money(value: Any) -> str:
    """A cost for human text; ``'n/a'`` when nothing was priced."""
    number = _clean(value)
    return "n/a" if number is None else f"{number:,.2f}"


@dataclass(frozen=True)
class Anomaly:
    """One period that did not look like the others of its kind."""

    when: Any
    observed: float
    expected: float
    excess: float
    cost: Optional[float] = None
    kind: str = "spike"
    score: float = 0.0
    rate: Optional[float] = None

    def __str__(self) -> str:
        sign = "+" if self.excess >= 0 else "-"
        text = (
            f"{_when_text(self.when)}: {_num(self.observed)} against an expected "
            f"{_num(self.expected)} ({sign}{_num(abs(self.excess))}, "
            f"{_num(abs(self.score))} sigmas)"
        )
        if self.cost is not None:
            text += f", {_money(abs(self.cost))} {'extra' if self.excess >= 0 else 'saved'}"
        return text

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of one anomaly."""
        return {
            "when": _stamp(self.when),
            "kind": self.kind,
            "observed": _clean(self.observed),
            "expected": _clean(self.expected),
            "excess": _clean(self.excess),
            "score": _clean(self.score),
            "rate": _clean(self.rate),
            "cost": _clean(self.cost),
            "text": str(self),
        }


@dataclass
class EnergyReport:
    """What was used, what looks wrong, what changed, and what it costs.

    Every number is in the unit of the meter column. Cost fields are ``None``
    whenever no ``tariff`` was given, never zero.
    """

    by_period: pd.DataFrame
    anomalies: List[Anomaly] = field(default_factory=list)
    total: float = 0.0
    total_cost: Optional[float] = None
    baseline_load: float = 0.0
    baseline_method: str = "none"
    baseline_how: str = ""
    excess_units: float = 0.0
    excess_cost: Optional[float] = None
    trend: Trend = field(default_factory=Trend)
    steps: List[StepChange] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    label: str = "value"
    time_label: Optional[str] = None
    unit: str = "units"
    granularity: str = "period"
    has_time: bool = False
    n_readings: int = 0
    n_gaps: int = 0
    longest_gap: int = 0
    negative_readings: int = 0
    cumulative_meter: bool = False
    meter_resets: int = 0
    scale: Optional[float] = None
    sensitivity: float = 3.0
    tariff: Tariff = field(default_factory=Tariff)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def n_periods(self) -> int:
        """Length of the regular grid the readings were placed on."""
        return int(len(self.by_period))

    @property
    def n_anomalies(self) -> int:
        """How many periods were flagged."""
        return len(self.anomalies)

    @property
    def spikes(self) -> List[Anomaly]:
        """The flagged periods that used more than expected."""
        return [item for item in self.anomalies if item.excess > 0]

    @property
    def drops(self) -> List[Anomaly]:
        """The flagged periods that used less than expected."""
        return [item for item in self.anomalies if item.excess <= 0]

    @property
    def gap_share(self) -> float:
        """Share of the grid with no reading, in ``[0, 1]``."""
        return 0.0 if self.n_periods == 0 else self.n_gaps / self.n_periods

    @property
    def baseline_total(self) -> float:
        """What the always-on floor accounts for across the whole window."""
        return float(self.baseline_load * max(0, self.n_periods - self.n_gaps))

    @property
    def baseline_share(self) -> float:
        """Share of the total that never switches off, in ``[0, 1]``."""
        if self.total <= 0:
            return 0.0
        return float(min(1.0, self.baseline_total / self.total))

    @property
    def span(self) -> Optional[tuple]:
        """``(first, last)`` period of the window, or ``None`` when empty."""
        if self.n_periods == 0:
            return None
        return (self.by_period.index[0], self.by_period.index[-1])

    def __len__(self) -> int:
        return self.n_periods

    def __bool__(self) -> bool:
        return self.n_anomalies > 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EnergyReport(total={self.total:.4g} {self.unit}, periods={self.n_periods}, "
            f"anomalies={self.n_anomalies}, trend={self.trend.direction!r})"
        )

    # ------------------------------------------------------------------ output
    def top(self, limit: int = 5) -> List[Anomaly]:
        """The flagged periods, worst first."""
        ranked = sorted(self.anomalies, key=lambda item: abs(item.score), reverse=True)
        return ranked[: max(0, int(limit))]

    def to_frame(self) -> pd.DataFrame:
        """The per-period table; the same object as :attr:`by_period`."""
        return self.by_period

    def to_dict(self) -> Dict[str, Any]:
        """Everything worth keeping, JSON-safe."""
        window = self.span
        return {
            "meter": self.label,
            "unit": self.unit,
            "time_column": self.time_label,
            "granularity": self.granularity,
            "periods": self.n_periods,
            "readings": self.n_readings,
            "window": {
                "start": _stamp(window[0]) if window else None,
                "end": _stamp(window[1]) if window else None,
            },
            "total": _clean(self.total),
            "total_cost": _clean(self.total_cost),
            "baseline_load": _clean(self.baseline_load),
            "baseline_total": _clean(self.baseline_total),
            "baseline_share": round(self.baseline_share, 6),
            "baseline_how": self.baseline_how,
            "baseline_method": self.baseline_method,
            "scale": _clean(self.scale),
            "sensitivity": float(self.sensitivity),
            "n_anomalies": self.n_anomalies,
            "n_spikes": len(self.spikes),
            "n_drops": len(self.drops),
            "excess_units": _clean(self.excess_units),
            "excess_cost": _clean(self.excess_cost),
            "anomalies": [item.to_dict() for item in self.anomalies],
            "trend": self.trend.to_dict(),
            "steps": [item.to_dict() for item in self.steps],
            "gaps": {
                "periods": self.n_gaps,
                "share": round(self.gap_share, 6),
                "longest_run": self.longest_gap,
            },
            "cumulative_meter": bool(self.cumulative_meter),
            "meter_resets": int(self.meter_resets),
            "negative_readings": int(self.negative_readings),
            "tariff": self.tariff.to_dict(),
            "findings": list(self.findings),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "summary": self.summary(),
        }

    def summary(self, limit: int = 5) -> str:
        """A short plain-text report, ASCII only so any console can print it."""
        word = "period" if self.n_anomalies == 1 else "periods"
        share = 0.0 if self.n_periods == 0 else self.n_anomalies / self.n_periods
        lines = [
            f"energy-analyzer-ai: {self.n_anomalies} unusual {word} in "
            f"{self.n_periods:,} {self.granularity} periods ({share:.1%})"
        ]
        meter = f"  meter     : {self.label}"
        if self.time_label:
            meter += f" over {self.time_label}"
        window = self.span
        if window is not None and self.has_time:
            meter += f", {_when_text(window[0])} to {_when_text(window[1])}"
        lines.append(meter)
        total = f"  total     : {_num(self.total)} {self.unit}"
        if self.total_cost is not None:
            total += f", costing {_money(self.total_cost)}"
        if self.n_gaps:
            total += f" ({self.n_gaps:,} period(s) with no reading, left as gaps)"
        lines.append(total)
        if self.n_periods:
            lines.append(
                f"  always-on : {_num(self.baseline_load)} {self.unit} per "
                f"{self.granularity}, about {self.baseline_share:.0%} of everything used"
            )
        if self.scale is None:
            lines.append("  baseline  : flat, no usable variation, so nothing can stand out")
        else:
            lines.append(
                f"  baseline  : {self.baseline_method}, robust sigma {_num(self.scale)}, "
                f"threshold {self.sensitivity:g} sigmas"
            )
        lines.append(f"  trend     : {self.trend}")
        if self.anomalies:
            unusual = f"  unusual   : {_num(self.excess_units)} {self.unit} above normal"
            if self.excess_cost is not None:
                unusual += f", costing about {_money(self.excess_cost)}"
            lines.append(unusual)
            for item in self.top(limit):
                lines.append(f"    {item.kind:5s} {item}")
            if self.n_anomalies > max(0, int(limit)):
                lines.append(f"    ... and {self.n_anomalies - int(limit):,} more")
        if self.steps:
            lines.append("  steps     :")
            for item in self.steps:
                lines.append(f"    - {item}")
        if self.findings:
            lines.append("  findings:")
            for finding in self.findings:
                lines.append(f"    - {finding}")
        if self.notes:
            lines.append("  notes:")
            for note in self.notes:
                lines.append(f"    - {note}")
        if self.warnings:
            lines.append("  warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        return "\n".join(lines)
