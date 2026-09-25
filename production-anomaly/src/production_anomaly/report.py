"""Result objects: :class:`Stoppage`, :class:`Event` and :class:`ProductionReport`."""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

QUALITY_NOTE = (
    "Quality is not measured: true OEE is availability x performance x quality, and quality "
    "needs scrap or reject counts that this data does not have. oee_partial is availability x "
    "performance, so it is an upper bound on the line's real OEE."
)

CAUSE_NAMES = {
    "downtime": "downtime",
    "micro_stops": "micro-stops",
    "slow_running": "slow running",
    "speed_loss": "speed loss",
}

EVENT_KINDS = (
    "downtime",
    "micro_stop",
    "slow_running",
    "rate_drift",
    "spike",
    "counter_reset",
    "data_gap",
)


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars, timestamps and NaN into plain JSON-ready values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def when(ts: Optional[pd.Timestamp]) -> str:
    """A timestamp as local wall-clock text: ``2026-03-02 09:14`` (seconds only if needed)."""
    if ts is None or ts is pd.NaT:
        return "n/a"
    if ts.second or ts.microsecond or ts.nanosecond:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.strftime("%Y-%m-%d %H:%M")


def duration(minutes: Optional[float]) -> str:
    """``48 min``, ``2.5 h`` or ``1.3 days``."""
    if minutes is None or not math.isfinite(minutes):
        return "n/a"
    if minutes <= 0:
        return "0 min"
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    if minutes < 90:
        return f"{minutes:.0f} min" if abs(minutes - round(minutes)) < 0.05 else f"{minutes:.1f} min"
    hours = minutes / 60.0
    if hours < 72:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def number(value: Optional[float]) -> str:
    """Units as text: ``152,340`` for big numbers, ``12.5`` for small ones."""
    if value is None or not math.isfinite(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.1f}"


def percent(value: Optional[float]) -> str:
    """A fraction as ``91.2%``, or ``n/a``."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


@dataclass
class Stoppage:
    """A run of zero or near-zero output, merged into one stop.

    ``kind`` is ``"downtime"`` for a stop of at least ``min_stop_minutes`` (an availability
    loss) and ``"micro_stop"`` for a shorter one (counted as a performance loss, as OEE does).
    ``end`` is exclusive: the start of the first interval after the stop.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    minutes: float
    kind: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return json_safe(
            {"start": self.start, "end": self.end, "minutes": self.minutes, "kind": self.kind}
        )

    def __str__(self) -> str:
        return f"{when(self.start)} to {when(self.end)}  {duration(self.minutes)}  {self.kind}"


@dataclass
class Event:
    """Something the line did that is worth knowing about, explained in ``detail``.

    ``kind`` is one of ``downtime``, ``micro_stop``, ``slow_running``, ``rate_drift``,
    ``spike``, ``counter_reset`` or ``data_gap``. ``units_lost`` is the shortfall against
    the reference rate during the event (``None`` where it does not apply); losses of
    different events can overlap, so use ``ProductionReport.lost_units`` for the total.
    ``value`` is the reading that triggered a spike or a reset.
    """

    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    minutes: float
    severity: str
    detail: str
    units_lost: Optional[float] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return json_safe(
            {
                "kind": self.kind,
                "start": self.start,
                "end": self.end,
                "minutes": self.minutes,
                "severity": self.severity,
                "detail": self.detail,
                "units_lost": self.units_lost,
                "value": self.value,
            }
        )

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


@dataclass
class ProductionReport:
    """What happened on the line, with the numbers and the reasoning behind them.

    ``availability`` is run time over scheduled time, ``performance`` is output over what
    the reference rate would have made in that run time, and ``oee_partial`` is their
    product. Quality is never included (see ``quality_note``). Metrics are ``None`` when
    there was nothing to measure; ``notes`` then says why.
    """

    availability: Optional[float]
    performance: Optional[float]
    oee_partial: Optional[float]
    stoppages: List[Stoppage]
    events: List[Event]
    by_shift: pd.DataFrame
    lost_units: Optional[float]
    findings: List[str]
    notes: List[str] = field(default_factory=list)
    lost_by_cause: Dict[str, float] = field(default_factory=dict)
    units: float = 0.0
    typical_rate: Optional[float] = None
    reference_rate: Optional[float] = None
    reference: str = "none"
    target_rate: Optional[float] = None
    interval_minutes: Optional[float] = None
    scheduled_minutes: float = 0.0
    run_minutes: float = 0.0
    downtime_minutes: float = 0.0
    unknown_minutes: float = 0.0
    unscheduled_minutes: float = 0.0
    time_column: Optional[str] = None
    output_column: Optional[str] = None
    mode: str = "counts"
    timezone: Optional[str] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    schedule: str = ""
    shift_hours_given: bool = False
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    quality: Optional[float] = None
    quality_note: str = QUALITY_NOTE

    # ------------------------------------------------------------------ text
    def _count_kinds(self) -> str:
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return ", ".join(f"{counts[k]} {k}" for k in EVENT_KINDS if k in counts)

    def summary(self, max_items: int = 8) -> str:
        """Plain-text report: headline metrics, findings, stoppages, events and notes."""
        lines: List[str] = []
        what = "running counter" if self.mode == "counter" else "per-interval counts"
        column = self.output_column if self.output_column is not None else "n/a"
        every = (
            f", {duration(self.interval_minutes)} intervals" if self.interval_minutes else ""
        )
        lines.append(f"production report for '{column}' ({what}{every})")
        if self.start is not None and self.end is not None:
            zone = f" (timezone {self.timezone})" if self.timezone else ""
            lines.append(f"period: {when(self.start)} to {when(self.end)}{zone}")
        if self.schedule:
            line = f"schedule: {self.schedule}"
            lines.append(line)
        sched = f"scheduled {duration(self.scheduled_minutes)}"
        if self.unknown_minutes:
            sched += f", of which {duration(self.unknown_minutes)} had no usable reading"
        if self.shift_hours_given and self.unscheduled_minutes:
            sched += f"; {duration(self.unscheduled_minutes)} outside the shifts not counted"
        lines.append(sched)

        n_down = sum(1 for s in self.stoppages if s.kind == "downtime")
        if self.availability is None:
            down_text = "nothing measured"
        elif n_down:
            down_text = (
                f"{duration(self.downtime_minutes)} downtime in {n_down} "
                f"stoppage{'s' if n_down != 1 else ''}"
            )
        else:
            down_text = "no downtime"
        lines.append(f"availability  {percent(self.availability):>6}  ({down_text})")
        if self.availability is None:
            ref_text = "nothing measured"
        elif self.reference_rate is not None and self.interval_minutes:
            per_hour = self.reference_rate * 60.0 / self.interval_minutes
            ref = "target_rate" if self.reference == "target" else "the line's typical rate"
            ref_text = (
                f"vs {ref}: {number(self.reference_rate)} per interval = {number(per_hour)}/h"
            )
        else:
            ref_text = "no reference rate: the line never ran and no target_rate was given"
        lines.append(f"performance   {percent(self.performance):>6}  ({ref_text})")
        lines.append(
            f"oee_partial   {percent(self.oee_partial):>6}  (availability x performance; "
            "quality not measured - it needs scrap data)"
        )
        loss = f"units made {number(self.units)}"
        if self.lost_units is not None:
            parts = ", ".join(
                f"{CAUSE_NAMES.get(name, name)} {number(value)}"
                for name, value in self.lost_by_cause.items()
                if value > 0.5
            )
            loss += f"; estimated lost {number(self.lost_units)}"
            if parts:
                loss += f" ({parts})"
        lines.append(loss)

        lines.append("findings:")
        for text in self.findings:
            lines.append(f"  - {text}")

        if self.stoppages:
            total = sum(s.minutes for s in self.stoppages)
            lines.append(
                f"stoppages ({len(self.stoppages)}, {duration(total)} in total):"
            )
            shown = self.stoppages
            if len(shown) > max_items:
                longest = sorted(shown, key=lambda s: -s.minutes)[:max_items]
                shown = sorted(longest, key=lambda s: s.start)
            for stop in shown:
                lines.append(f"  {stop}")
            if len(self.stoppages) > len(shown):
                lines.append(
                    f"  ... and {len(self.stoppages) - len(shown)} shorter ones "
                    "(see report.stoppages)"
                )
        if self.events:
            lines.append(f"events: {self._count_kinds()} (see report.events)")
        if self.notes:
            lines.append("notes:")
            for text in self.notes:
                lines.append(f"  - {text}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"ProductionReport(availability={percent(self.availability)}, "
            f"performance={percent(self.performance)}, oee_partial={percent(self.oee_partial)}, "
            f"stoppages={len(self.stoppages)}, events={len(self.events)}, "
            f"lost_units={number(self.lost_units) if self.lost_units is not None else None})"
        )

    # ------------------------------------------------------------------ data
    def to_dict(self) -> Dict[str, Any]:
        """Everything except the per-interval table, as JSON-safe values."""
        by_shift = []
        if isinstance(self.by_shift, pd.DataFrame) and len(self.by_shift):
            for row in self.by_shift.to_dict(orient="records"):
                by_shift.append(json_safe(row))
        return json_safe(
            {
                "output_column": self.output_column,
                "time_column": self.time_column,
                "mode": self.mode,
                "interval_minutes": self.interval_minutes,
                "timezone": self.timezone,
                "start": self.start,
                "end": self.end,
                "schedule": self.schedule,
                "shift_hours_given": self.shift_hours_given,
                "availability": self.availability,
                "performance": self.performance,
                "oee_partial": self.oee_partial,
                "quality": self.quality,
                "quality_note": self.quality_note,
                "units": self.units,
                "lost_units": self.lost_units,
                "lost_by_cause": dict(self.lost_by_cause),
                "typical_rate": self.typical_rate,
                "reference_rate": self.reference_rate,
                "reference": self.reference,
                "target_rate": self.target_rate,
                "scheduled_minutes": self.scheduled_minutes,
                "run_minutes": self.run_minutes,
                "downtime_minutes": self.downtime_minutes,
                "unknown_minutes": self.unknown_minutes,
                "unscheduled_minutes": self.unscheduled_minutes,
                "stoppages": [s.to_dict() for s in self.stoppages],
                "events": [e.to_dict() for e in self.events],
                "by_shift": by_shift,
                "findings": list(self.findings),
                "notes": list(self.notes),
            }
        )
