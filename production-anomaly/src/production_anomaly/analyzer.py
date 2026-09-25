"""Find downtime, slow running, rate drift, micro-stops and spikes in production counts."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._io import TableLike, check_columns, load_table
from ._prepare import (
    NS_PER_DAY,
    NS_PER_MIN,
    Grid,
    aggregate_same_time,
    build_grid,
    coarsen_low_counts,
    epoch_ns,
    interval_seconds,
    is_integer_counts,
    label_text,
    looks_like_counter,
    resolve_output,
    resolve_time,
    timestamp,
    to_float,
)
from .report import (
    QUALITY_NOTE,
    Event,
    ProductionReport,
    Stoppage,
    duration,
    number,
    percent,
    when,
)
from .shifts import MINUTES_PER_DAY, Schedule, clock, parse_shift_hours

log = logging.getLogger(__name__)

BY_SHIFT_COLUMNS = [
    "date",
    "shift",
    "start",
    "end",
    "scheduled_min",
    "measured_min",
    "downtime_min",
    "run_min",
    "units",
    "rate_per_hour",
    "availability",
    "performance",
    "oee_partial",
    "stoppages",
    "micro_stops",
]
# Whole parts an interval must typically hold before the analysis trusts one interval on
# its own; below that, intervals are merged (see _prepare.coarsen_low_counts).
MIN_UNITS_PER_INTERVAL = 10.0
_KIND_ORDER = {
    "downtime": 0,
    "rate_drift": 1,
    "slow_running": 2,
    "micro_stop": 3,
    "spike": 4,
    "counter_reset": 5,
    "data_gap": 6,
}


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, end) index ranges where ``mask`` is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def _robust(values: np.ndarray) -> Tuple[Optional[float], float]:
    """(median, MAD-based standard deviation) of the finite values, or (None, 0)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, 0.0
    centre = float(np.median(values))
    return centre, float(1.4826 * np.median(np.abs(values - centre)))


def _positive(name: str, value: Any, allow_none: bool = True) -> Optional[float]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    number_ = float(value)
    if not math.isfinite(number_) or number_ <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return number_


@dataclass
class _Context:
    """What the preparation step learned about the input, carried into the report."""

    time_name: Optional[str]
    output_name: Optional[str]
    mode: str
    notes: List[str]


class ProductionAnalyzer:
    """Configurable analysis of a production line's output; :func:`analyze` uses defaults.

    Parameters beyond :func:`analyze`'s:

    - ``interval``: the length of one interval (``"1min"``, ``"5min"``, ``"1h"``, minutes as a
      number, or a timedelta). By default it is measured from the timestamps. Whole-part
      counts too low for it (under about 10 an interval) are merged into longer intervals,
      and a note says so.
    - ``counter``: ``"auto"`` (default) detects a running total that only goes up;
      ``True``/``False`` force it.
    - ``near_zero``: output at or below this fraction of the line's high (90th percentile)
      rate counts as stopped. Default 0.05.
    - ``slow_fraction``: running below this fraction of the reference rate is slow.
      Default 0.8.
    - ``min_stop_minutes``: stops at least this long are downtime; shorter ones are
      micro-stops. Default 5.
    - ``min_slow_minutes``: slow running must last this long (and at least two intervals)
      to be reported. Default 15.
    - ``spike_factor``: output above this multiple of the typical rate (and far outside its
      normal spread) is a spike. Default 1.5.
    - ``drift_threshold``: the decline across shifts, as a fraction, that counts as drift.
      Default 0.10.
    """

    def __init__(
        self,
        *,
        time: Any = None,
        output: Any = None,
        target_rate: Optional[float] = None,
        shift_hours: Any = None,
        interval: Any = None,
        counter: Union[str, bool] = "auto",
        near_zero: float = 0.05,
        slow_fraction: float = 0.8,
        min_stop_minutes: float = 5.0,
        min_slow_minutes: float = 15.0,
        spike_factor: float = 1.5,
        drift_threshold: float = 0.10,
    ) -> None:
        self.time = time
        self.output = output
        self.target_rate = _positive("target_rate", target_rate)
        self.shift_hours = shift_hours
        self.schedule: Schedule = parse_shift_hours(shift_hours)
        self.interval = interval
        self.interval_ns: Optional[int] = (
            None if interval is None else int(round(interval_seconds(interval) * 1e9))
        )
        if counter not in ("auto", True, False):
            raise ValueError(f"counter must be 'auto', True or False, got {counter!r}")
        self.counter = counter
        if not isinstance(near_zero, (int, float)) or not 0 <= float(near_zero) < 1:
            raise ValueError(f"near_zero must be a fraction in [0, 1), got {near_zero!r}")
        self.near_zero = float(near_zero)
        if not isinstance(slow_fraction, (int, float)) or not 0 < float(slow_fraction) < 1:
            raise ValueError(f"slow_fraction must be a fraction in (0, 1), got {slow_fraction!r}")
        self.slow_fraction = float(slow_fraction)
        self.min_stop_minutes = float(_positive("min_stop_minutes", min_stop_minutes, False))
        self.min_slow_minutes = float(_positive("min_slow_minutes", min_slow_minutes, False))
        spike = float(_positive("spike_factor", spike_factor, False))
        if spike <= 1:
            raise ValueError(f"spike_factor must be above 1, got {spike_factor!r}")
        self.spike_factor = spike
        if not isinstance(drift_threshold, (int, float)) or not 0 < float(drift_threshold) < 1:
            raise ValueError(
                f"drift_threshold must be a fraction in (0, 1), got {drift_threshold!r}"
            )
        self.drift_threshold = float(drift_threshold)

    # ------------------------------------------------------------------ entry
    def analyze(self, df: TableLike) -> ProductionReport:
        """Analyze ``df`` (a DataFrame or a .csv/.tsv/.parquet path). The input is never modified."""
        frame = load_table(df)
        check_columns(frame)
        notes: List[str] = []
        if frame.shape[0] == 0:
            names = self._names_if_possible(frame)
            return self._empty(_Context(names[0], names[1], "counts", notes), "the table has no rows")

        stamps, time_name = resolve_time(frame, self.time, notes)
        out_label = resolve_output(frame, self.output, time_name, notes)
        out_name = label_text(out_label)
        values = to_float(frame[out_label], out_name, notes)

        has_time = np.array(stamps.notna().to_numpy(dtype=bool), dtype=bool, copy=True)
        dropped = int((~has_time).sum())
        if dropped:
            notes.append(f"{dropped} row(s) had no readable timestamp and were left out.")
        stamps = stamps[has_time]
        values = values[has_time]
        if values.size == 0:
            return self._empty(
                _Context(time_name, out_name, "counts", notes),
                f"no readable timestamps in '{time_name}'",
            )

        ns, wall, tz = epoch_ns(stamps)
        order = np.argsort(ns, kind="stable")
        if np.any(np.diff(ns) < 0):
            notes.append("rows were not in time order; they were sorted by time.")
        ns, wall, values = ns[order], wall[order], values[order]
        # Rows that share a timestamp follow one rule whatever their values: counts are added
        # up (two machines, or two parts logged in the same second). Rows that repeat the
        # same value are counted too, and pointed out, since they may be one reading logged
        # twice; a doubled interval then also shows up as a spike.
        repeats = int(pd.DataFrame({"t": ns, "v": values}).duplicated().sum())

        if self.counter == "auto":
            counter = looks_like_counter(values)
        else:
            counter = bool(self.counter)
        mode = "counter" if counter else "counts"
        if counter:
            missing = int((~np.isfinite(values)).sum())
            notes.append(
                f"'{out_name}' behaves like a running counter (it only goes up, apart from "
                "resets), so units per interval are the differences between readings."
                + (" Missing readings were bridged." if missing else "")
            )
        integer_counts = is_integer_counts(values)
        ns, wall, values, merged = aggregate_same_time(ns, wall, values, counter)
        if merged:
            if counter:
                notes.append(
                    f"{merged} row(s) shared a timestamp with another row; the highest "
                    "reading was kept."
                )
            else:
                text = (
                    f"{merged} row(s) shared a timestamp with another row; their units were "
                    "added up."
                )
                if repeats:
                    text += (
                        f" {repeats} of them repeat the same value at the same time; if that "
                        "is one reading logged twice, drop the duplicates first "
                        "(df.drop_duplicates())."
                    )
                notes.append(text)

        grid = build_grid(ns, wall, values, tz, counter, self.interval_ns, notes)
        if integer_counts:
            boundaries = sorted(
                {s.start for s in self.schedule.shifts} | {s.end for s in self.schedule.shifts}
            )
            grid = coarsen_low_counts(
                grid, self.schedule.lookup(), boundaries, self._min_units(), notes
            )
        log.debug(
            "output=%r mode=%s interval=%.3g min intervals=%d resampled=%s",
            out_name, mode, grid.dt_ns / NS_PER_MIN, grid.units.size, grid.resampled,
        )
        if tz is not None:
            notes.append(
                f"timestamps carry the timezone {tz}; every time in this report is in that "
                "timezone, and shifts are read in its local clock."
            )
        context = _Context(time_name, out_name, mode, notes)
        if grid.units.size == 0:
            return self._empty(context, "a counter needs at least two readings to measure output")
        return self._measure(grid, context)

    def _min_units(self) -> float:
        """Units an interval must typically hold for one part more or less not to matter.

        One part short must stay above ``slow_fraction`` of the typical interval and one
        part over must stay below ``spike_factor`` times it, with room for a second part of
        jitter either way.
        """
        return max(
            MIN_UNITS_PER_INTERVAL,
            2.0 / (1.0 - self.slow_fraction),
            2.0 / (self.spike_factor - 1.0),
        )

    def _names_if_possible(self, frame: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        scratch: List[str] = []
        try:
            _, time_name = resolve_time(frame, self.time, scratch)
        except ValueError:
            time_name = None if self.time is None else str(self.time)
        try:
            output_name: Optional[str] = label_text(
                resolve_output(frame, self.output, time_name, scratch)
            )
        except ValueError:
            output_name = None if self.output is None else str(self.output)
        return time_name, output_name

    # ------------------------------------------------------------------ empty
    def _empty(self, context: _Context, reason: str) -> ProductionReport:
        notes = [QUALITY_NOTE] + context.notes
        return ProductionReport(
            availability=None,
            performance=None,
            oee_partial=None,
            stoppages=[],
            events=[],
            by_shift=pd.DataFrame(columns=BY_SHIFT_COLUMNS),
            lost_units=None,
            findings=[f"Nothing to measure: {reason}."],
            notes=notes,
            lost_by_cause={},
            units=0.0,
            target_rate=self.target_rate,
            time_column=context.time_name,
            output_column=context.output_name,
            mode=context.mode,
            schedule=self.schedule.describe(),
            shift_hours_given=self.schedule.given,
            intervals=pd.DataFrame(columns=["time", "units", "state", "shift", "shift_date"]),
        )

    # ------------------------------------------------------------------ core
    def _measure(self, grid: Grid, context: _Context) -> ProductionReport:
        schedule = self.schedule
        k = grid.units.size
        dt_min = grid.dt_ns / NS_PER_MIN
        u = np.array(grid.units, dtype=float, copy=True)
        tz = grid.tz

        minute_of_day = np.mod(np.floor_divide(grid.wall_ns, NS_PER_MIN), MINUTES_PER_DAY).astype(
            np.int64
        )
        day = np.floor_divide(grid.wall_ns, NS_PER_DAY).astype(np.int64)
        shift_index = schedule.lookup()[minute_of_day]
        sched = shift_index >= 0
        shift_start = np.array([s.start for s in schedule.shifts], dtype=np.int64)
        shift_day = day.copy()
        if sched.any():
            wraps_back = minute_of_day[sched] < shift_start[shift_index[sched]]
            shift_day[sched] = day[sched] - wraps_back.astype(np.int64)

        finite = np.isfinite(u)
        known = sched & finite

        # --- the line's own rates -------------------------------------------------
        positive = u[known & (u > 0)]
        high = float(np.quantile(positive, 0.9)) if positive.size else None
        near_zero = self.near_zero * high if high else 0.0
        down = known & (u <= near_zero)
        # The median and its robust spread set the detection limits; the report's typical
        # rate is the mean of the intervals that turn out to be normal running (below).
        median_rate, sigma = _robust(u[known & ~down])
        spike = np.zeros(k, dtype=bool)
        if median_rate is not None and median_rate > 0:
            spike_limit = max(self.spike_factor * median_rate, median_rate + 6.0 * sigma)
            spike = known & ~down & (u > spike_limit)
            if spike.any():
                median_rate, sigma = _robust(u[known & ~down & ~spike])
        if median_rate is not None and median_rate <= 0:
            median_rate = None

        # A counter reset or a short dropout inside a stop belongs to the stop.
        absorbed = np.zeros(k, dtype=bool)
        for a, b in _runs(sched & ~finite):
            if b - a <= 3 and a > 0 and b < k and down[a - 1] and down[b]:
                absorbed[a:b] = True
        down |= absorbed
        known |= absorbed

        counted = np.array(u, dtype=float, copy=True)
        counted[absorbed] = 0.0
        if median_rate is not None:
            counted[spike] = median_rate

        target_per_interval = (
            self.target_rate * dt_min / 60.0 if self.target_rate is not None else None
        )
        if median_rate is not None and target_per_interval is not None:
            slow_reference: Optional[float] = min(median_rate, target_per_interval)
        else:
            slow_reference = target_per_interval if target_per_interval is not None else median_rate

        # --- stoppages ------------------------------------------------------------
        stoppages: List[Stoppage] = []
        events: List[Event] = []
        downtime_mask = np.zeros(k, dtype=bool)
        micro_zero = np.zeros(k, dtype=bool)
        stop_runs: List[Tuple[int, int, str]] = []
        for a, b in _runs(down):
            minutes = (b - a) * dt_min
            kind = "downtime" if minutes >= self.min_stop_minutes - 1e-9 else "micro_stop"
            (downtime_mask if kind == "downtime" else micro_zero)[a:b] = True
            stop_runs.append((a, b, kind))
            stoppages.append(
                Stoppage(
                    start=grid.index[a],
                    end=timestamp(grid.ns[b - 1] + grid.dt_ns, tz),
                    minutes=float(minutes),
                    kind=kind,
                )
            )

        running = known & ~down & ~spike
        # --- slow running ---------------------------------------------------------
        slow_mask = np.zeros(k, dtype=bool)
        slow_runs: List[Tuple[int, int]] = []
        slow_limit = None
        if slow_reference is not None and slow_reference > 0:
            slow_limit = self.slow_fraction * slow_reference
            # Slow running must persist: the smoothing spans at least three intervals and a
            # spell at least two, so one noisy 15-minute or hourly interval is not a finding.
            per_slow = int(math.ceil(self.min_slow_minutes / dt_min - 1e-9))
            window = max(3, per_slow)
            min_intervals = max(2, per_slow)
            smooth = (
                pd.Series(np.where(running, counted, np.nan))
                .rolling(window, center=True, min_periods=max(1, (window + 1) // 2))
                .median()
                .to_numpy(dtype=float, copy=True)
            )
            # ...and it must sit further below the line's typical rate than the median of
            # `window` ordinary intervals strays by chance (three standard errors), so the
            # scatter of a noisy line, or of a few parts per interval, is not slow running.
            threshold = slow_limit
            if median_rate is not None:
                threshold = min(slow_limit, median_rate - 3.0 * 1.2533 * sigma / math.sqrt(window))
            candidate = running & (smooth < threshold)
            for a, b in _runs(candidate | (micro_zero & sched)):
                while a < b and not (running[a] and counted[a] < slow_limit):
                    a += 1
                while b > a and not (running[b - 1] and counted[b - 1] < slow_limit):
                    b -= 1
                if b - a >= min_intervals:
                    slow_mask[a:b] = running[a:b]
                    slow_runs.append((a, b))

        # --- micro-stops: short drops that are not part of a stop or a slow spell --
        dip = np.zeros(k, dtype=bool)
        if median_rate is not None:
            dip_limit = min(self.slow_fraction * median_rate, median_rate - 4.0 * sigma)
            dip = running & ~slow_mask & (counted < dip_limit)
        micro_mask = dip | micro_zero
        run_mask = known & ~downtime_mask

        # The typical rate is the average of normal running: not stopped, slow, dipping or
        # spiking. Measured this way a steady line scores 100% performance against itself.
        normal = running & ~slow_mask & ~dip
        if normal.any():
            typical: Optional[float] = float(counted[normal].mean())
        else:
            typical = median_rate
        if typical is not None:
            counted[spike] = typical
        if target_per_interval is not None:
            reference, reference_kind = target_per_interval, "target"
        elif typical is not None and typical > 0:
            reference, reference_kind = typical, "typical"
        else:
            reference, reference_kind = None, "none"

        # --- losses ---------------------------------------------------------------
        lost_by_cause: Dict[str, float] = {}
        lost_units: Optional[float] = None
        shortfall = np.zeros(k)
        if reference is not None:
            shortfall = np.where(known, reference - counted, 0.0)
            rest = run_mask & ~micro_mask & ~slow_mask
            lost_by_cause = {
                "downtime": max(0.0, float(shortfall[downtime_mask].sum())),
                "micro_stops": max(0.0, float(shortfall[micro_mask].sum())),
                "slow_running": max(0.0, float(shortfall[slow_mask].sum())),
                "speed_loss": max(0.0, float(shortfall[rest].sum())),
            }
            lost_units = float(sum(lost_by_cause.values()))

        # --- events ---------------------------------------------------------------
        def span(a: int, b: int) -> Tuple[pd.Timestamp, pd.Timestamp, float]:
            return grid.index[a], timestamp(grid.ns[b - 1] + grid.dt_ns, tz), (b - a) * dt_min

        def loss(a: int, b: int, mask: Optional[np.ndarray] = None) -> Optional[float]:
            if reference is None:
                return None
            part = shortfall[a:b] if mask is None else shortfall[a:b][mask[a:b]]
            return max(0.0, float(part.sum()))

        for a, b, kind in stop_runs:
            if kind != "downtime":
                continue
            start, end, minutes = span(a, b)
            edge = ""
            if a == 0:
                edge = " (already stopped when the data begins)"
            elif b == k:
                edge = " (still stopped when the data ends)"
            lost = loss(a, b)
            events.append(
                Event(
                    kind="downtime",
                    start=start,
                    end=end,
                    minutes=minutes,
                    severity="critical" if minutes >= 60 else "warning",
                    detail=(
                        f"stopped for {duration(minutes)}, {when(start)} to {when(end)}{edge}"
                        + (f"; about {number(lost)} units lost" if lost is not None else "")
                    ),
                    units_lost=lost,
                )
            )

        for a, b in _runs(micro_mask):
            start, end, minutes = span(a, b)
            lost = loss(a, b)
            made = float(counted[a:b].sum())
            if micro_zero[a:b].all():
                what = f"stopped for {duration(minutes)}"
            else:
                what = f"made {number(made)} in {duration(minutes)} where {number(typical * (b - a))} is typical"
            equiv = (lost / reference * dt_min) if (lost is not None and reference) else None
            events.append(
                Event(
                    kind="micro_stop",
                    start=start,
                    end=end,
                    minutes=minutes,
                    severity="info",
                    detail=(
                        f"short drop at {when(start)}: {what}"
                        + (
                            f"; about {number(lost)} units, {duration(equiv)} of full-speed running"
                            if lost is not None and equiv is not None
                            else ""
                        )
                    ),
                    units_lost=lost,
                )
            )

        for a, b in slow_runs:
            start, end, minutes = span(a, b)
            lost = loss(a, b, slow_mask)
            mask = slow_mask[a:b]
            average = float(counted[a:b][mask].mean()) if mask.any() else float("nan")
            share = average / slow_reference if slow_reference else float("nan")
            ref_name = "target_rate" if (
                target_per_interval is not None and slow_reference == target_per_interval
            ) else "the typical rate"
            events.append(
                Event(
                    kind="slow_running",
                    start=start,
                    end=end,
                    minutes=minutes,
                    severity="warning",
                    detail=(
                        f"ran slow from {when(start)} to {when(end)} ({duration(minutes)}): "
                        f"{number(average)} per interval, {percent(share)} of {ref_name}"
                        + (f"; about {number(lost)} units lost" if lost is not None else "")
                    ),
                    units_lost=lost,
                )
            )

        for a, b in _runs(spike):
            start, end, minutes = span(a, b)
            peak = float(np.max(u[a:b]))
            ratio = peak / typical if typical else float("nan")
            excess = float((u[a:b] - (typical or 0.0)).sum())
            events.append(
                Event(
                    kind="spike",
                    start=start,
                    end=end,
                    minutes=minutes,
                    severity="warning",
                    detail=(
                        f"{number(peak)} units in the interval starting {when(start)}, "
                        f"{ratio:.1f}x the typical {number(typical)}, which {_spike_cause(ratio)}; "
                        f"counted at the typical rate, leaving {number(excess)} excess units out"
                    ),
                    value=peak,
                )
            )

        for reset in grid.resets:
            start, end = timestamp(reset.start_ns, tz), timestamp(reset.end_ns, tz)
            if reset.before is None:
                text = (
                    f"a reading of {number(reset.after)} at {when(start)} is negative, which is "
                    "what a counter reset looks like when counts are taken as differences"
                )
            elif reset.after <= 0.5 * reset.before:
                text = (
                    f"the counter dropped from {number(reset.before)} to {number(reset.after)} "
                    f"between {when(start)} and {when(end)}"
                )
            else:
                text = (
                    f"the counter went backwards from {number(reset.before)} to "
                    f"{number(reset.after)} between {when(start)} and {when(end)}"
                )
            events.append(
                Event(
                    kind="counter_reset",
                    start=start,
                    end=end,
                    minutes=(reset.end_ns - reset.start_ns) / NS_PER_MIN,
                    severity="info",
                    detail=(
                        f"{text}; reported as a reset, not as a stoppage or a negative rate, "
                        "and the units made in that interval are unknown"
                    ),
                    value=reset.after,
                )
            )

        gap_mask = sched & ~known & ~grid.reset_bins
        gap_runs = _runs(gap_mask)
        for a, b in gap_runs:
            start, end, minutes = span(a, b)
            events.append(
                Event(
                    kind="data_gap",
                    start=start,
                    end=end,
                    minutes=minutes,
                    severity="info",
                    detail=(
                        f"no usable readings from {when(start)} to {when(end)} "
                        f"({duration(minutes)}) inside scheduled hours; counted neither as "
                        "downtime nor as running time"
                    ),
                )
            )

        # --- headline metrics -----------------------------------------------------
        n_known = int(known.sum())
        n_run = int(run_mask.sum())
        notes = [QUALITY_NOTE]
        availability = n_run / n_known if n_known else None
        performance: Optional[float] = None
        if n_run and reference:
            raw = float(counted[run_mask].sum()) / (reference * n_run)
            performance = min(1.0, raw)
            if raw > 1.005:
                why = (
                    "the line ran faster than target_rate, so target_rate is probably too low"
                    if reference_kind == "target"
                    else "the line's output is skewed above its typical rate"
                )
                notes.append(f"performance came to {percent(raw)} and was capped at 100%: {why}.")
        if availability is not None and performance is not None:
            oee: Optional[float] = availability * performance
        elif availability == 0:
            oee = 0.0
        else:
            oee = None
        units = float(counted[known].sum())

        # --- shifts and drift -----------------------------------------------------
        by_shift = self._by_shift(
            grid, sched, shift_index, shift_day, known, downtime_mask, run_mask, counted, reference,
            stop_runs, _runs(micro_mask),
        )
        drift_event, drift_note = self._drift(by_shift, dt_min, reference)
        if drift_event is not None:
            events.append(drift_event)

        events.sort(key=lambda e: (e.start, _KIND_ORDER.get(e.kind, 99)))
        stoppages.sort(key=lambda s: s.start)

        # --- notes ----------------------------------------------------------------
        interval_text = duration(dt_min)
        if reference_kind == "target":
            notes.append(
                f"performance is measured against target_rate: {number(self.target_rate)} "
                f"units/h, {number(reference)} per {interval_text} interval."
            )
        elif reference_kind == "typical":
            notes.append(
                "no target_rate given: performance is measured against the line's own typical "
                f"rate ({number(reference)} per {interval_text} interval, "
                f"{number(reference * 60 / dt_min)}/h: the average of its normal running "
                "intervals), so it shows speed lost against its usual pace, not against rated "
                "speed."
            )
        else:
            notes.append(
                "the line never ran and no target_rate was given, so performance and lost "
                "units cannot be measured."
            )
        if schedule.given:
            outside = float(np.nansum(u[~sched & finite]))
            notes.append(
                f"shift_hours given ({schedule.describe()}); time outside the shifts is "
                "unscheduled and is not counted as downtime."
            )
            if outside > 0:
                notes.append(
                    f"{number(outside)} units were made outside the scheduled hours; they are "
                    "not part of units, availability or performance."
                )
        else:
            notes.append(
                "shift_hours not given: every hour counts as scheduled, so a line that is "
                "simply not scheduled at night shows those hours as downtime. by_shift uses "
                "8-hour blocks from 06:00."
            )
        notes.extend(context.notes)
        if grid.gap_units > 0:
            notes.append(
                f"{number(grid.gap_units)} units were counted across data gaps longer than two "
                "intervals; when they were made is unknown, so they are left out."
            )
        if absorbed.any():
            notes.append(
                f"{int(absorbed.sum())} interval(s) without a usable reading inside stoppages "
                "were counted as part of the stop."
            )
        n_running = int(running.sum())
        if 0 < n_running < 10:
            notes.append(
                f"WARNING: only {n_running} running interval(s); the typical rate is measured "
                "from very little data."
            )
        if dt_min >= self.min_slow_minutes:
            notes.append(
                f"with {interval_text} intervals, stops shorter than one interval cannot be seen "
                "one by one; they show up as lower performance instead of as micro-stops."
            )
        if drift_note:
            notes.append(drift_note)
        if (
            target_per_interval is not None
            and typical is not None
            and typical > 0
            and not 0.2 <= typical / target_per_interval <= 5
        ):
            ratio = typical / target_per_interval
            notes.append(
                f"WARNING: target_rate={number(self.target_rate)} units/h is "
                f"{max(ratio, 1 / ratio):.0f}x {'below' if ratio > 1 else 'above'} the line's "
                f"typical {number(typical * 60 / dt_min)}/h. target_rate is in units per HOUR, "
                "not per interval."
            )

        findings = self._findings(
            events=events,
            stoppages=stoppages,
            lost_by_cause=lost_by_cause,
            units=units,
            dt_min=dt_min,
            scheduled_minutes=float(sched.sum() * dt_min),
            typical=typical,
            target_per_interval=target_per_interval,
            slow_limit=slow_limit,
            slow_reference=slow_reference,
            n_known=n_known,
            grid=grid,
            known=known,
            downtime_mask=downtime_mask,
            minute_of_day=minute_of_day,
            day=day,
            sched=sched,
            shift_day=shift_day,
        )

        state = np.full(k, "running", dtype=object)
        state[~sched] = "unscheduled"
        state[sched & ~known] = "unknown"
        state[sched & ~known & grid.reset_bins] = "reset"
        state[downtime_mask] = "downtime"
        state[micro_mask] = "micro_stop"
        state[slow_mask] = "slow"
        state[spike] = "spike"
        names = np.array([s.name for s in schedule.shifts] + [None], dtype=object)
        shift_names = names[np.where(sched, shift_index, len(schedule.shifts))]
        shift_dates = np.where(sched, shift_day, 0).astype("datetime64[D]").astype(object)
        shift_dates = np.where(sched, shift_dates, None)
        intervals = pd.DataFrame(
            {
                "time": grid.index,
                "units": u,
                "state": state,
                "shift": shift_names,
                "shift_date": shift_dates,
            }
        )

        return ProductionReport(
            availability=availability,
            performance=performance,
            oee_partial=oee,
            stoppages=stoppages,
            events=events,
            by_shift=by_shift,
            lost_units=lost_units,
            findings=findings,
            notes=notes,
            lost_by_cause=lost_by_cause,
            units=units,
            typical_rate=typical,
            reference_rate=reference,
            reference=reference_kind,
            target_rate=self.target_rate,
            interval_minutes=float(dt_min),
            scheduled_minutes=float(sched.sum() * dt_min),
            run_minutes=float(n_run * dt_min),
            downtime_minutes=float(downtime_mask.sum() * dt_min),
            unknown_minutes=float((sched & ~known).sum() * dt_min),
            unscheduled_minutes=float((~sched).sum() * dt_min),
            time_column=context.time_name,
            output_column=context.output_name,
            mode=context.mode,
            timezone=None if tz is None else str(tz),
            start=grid.index[0],
            end=timestamp(grid.ns[-1] + grid.dt_ns, tz),
            schedule=schedule.describe(),
            shift_hours_given=schedule.given,
            intervals=intervals,
        )

    # ------------------------------------------------------------------ shifts
    def _by_shift(
        self,
        grid: Grid,
        sched: np.ndarray,
        shift_index: np.ndarray,
        shift_day: np.ndarray,
        known: np.ndarray,
        downtime_mask: np.ndarray,
        run_mask: np.ndarray,
        counted: np.ndarray,
        reference: Optional[float],
        stop_runs: Sequence[Tuple[int, int, str]],
        micro_runs: Sequence[Tuple[int, int]],
    ) -> pd.DataFrame:
        k = sched.size
        code_of_bin = np.full(k, -1, dtype=np.int64)
        if not sched.any():
            return pd.DataFrame(columns=BY_SHIFT_COLUMNS)
        positions = np.flatnonzero(sched)
        keys = shift_day[positions] * 10_000 + shift_index[positions]
        codes, uniques = pd.factorize(pd.Series(keys), sort=False)
        codes = np.array(codes, dtype=np.int64, copy=True)
        groups = len(uniques)
        code_of_bin[positions] = codes
        dt_min = grid.dt_ns / NS_PER_MIN

        def total(mask_or_values: np.ndarray) -> np.ndarray:
            return np.bincount(codes, weights=mask_or_values[positions].astype(float), minlength=groups)

        sched_bins = np.bincount(codes, minlength=groups).astype(float)
        known_bins = total(known)
        down_bins = total(downtime_mask)
        run_bins = total(run_mask)
        run_units = total(np.where(run_mask, counted, 0.0))
        first = np.full(groups, k, dtype=np.int64)
        last = np.full(groups, -1, dtype=np.int64)
        np.minimum.at(first, codes, positions)
        np.maximum.at(last, codes, positions)
        stops = np.zeros(groups, dtype=np.int64)
        for a, _, kind in stop_runs:
            if kind == "downtime" and code_of_bin[a] >= 0:
                stops[code_of_bin[a]] += 1
        micros = np.zeros(groups, dtype=np.int64)
        for a, _ in micro_runs:
            if code_of_bin[a] >= 0:
                micros[code_of_bin[a]] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(run_bins > 0, run_units / (run_bins * dt_min / 60.0), np.nan)
            availability = np.where(known_bins > 0, run_bins / known_bins, np.nan)
            if reference:
                performance = np.where(
                    run_bins > 0, np.minimum(1.0, run_units / (reference * run_bins)), np.nan
                )
            else:
                performance = np.full(groups, np.nan)
        oee = np.where(availability == 0, 0.0, availability * performance)

        names = [s.name for s in self.schedule.shifts]
        group_shift = np.array(uniques, dtype=np.int64) % 10_000
        group_day = np.array(uniques, dtype=np.int64) // 10_000
        frame = pd.DataFrame(
            {
                "date": list(group_day.astype("datetime64[D]").astype(object)),
                "shift": [names[i] for i in group_shift],
                "start": [grid.index[i] for i in first],
                "end": [timestamp(grid.ns[i] + grid.dt_ns, grid.tz) for i in last],
                "scheduled_min": sched_bins * dt_min,
                "measured_min": known_bins * dt_min,
                "downtime_min": down_bins * dt_min,
                "run_min": run_bins * dt_min,
                "units": total(np.where(known, counted, 0.0)),
                "rate_per_hour": rate,
                "availability": availability,
                "performance": performance,
                "oee_partial": oee,
                "stoppages": stops,
                "micro_stops": micros,
            }
        )
        frame = frame.sort_values("start", kind="stable").reset_index(drop=True)
        return frame

    def _drift(
        self, by_shift: pd.DataFrame, dt_min: float, reference: Optional[float]
    ) -> Tuple[Optional[Event], Optional[str]]:
        if by_shift.empty:
            return None, None
        run_min = by_shift["run_min"].to_numpy(dtype=float)
        sched_min = by_shift["scheduled_min"].to_numpy(dtype=float)
        rates_all = by_shift["rate_per_hour"].to_numpy(dtype=float)
        eligible = (run_min >= np.maximum(2 * dt_min, 0.3 * sched_min)) & np.isfinite(rates_all)
        rates = rates_all[eligible]
        n = rates.size
        if n < 3:
            return None, (
                "rate drift is judged across shifts and needs at least 3 with enough running "
                f"time to compare; this data has {n}."
            )
        idx = np.arange(n, dtype=float)
        i, j = np.triu_indices(n, k=1)
        slope = float(np.median((rates[j] - rates[i]) / (j - i)))
        intercept = float(np.median(rates - slope * idx))
        fitted = intercept + slope * idx
        first, last = float(fitted[0]), float(fitted[-1])
        if first <= 0:
            return None, None
        change = (last - first) / first
        signs = np.sign(rates[j] - rates[i])
        tau = float(signs.sum() / signs.size)
        half = n // 2
        early, late = float(np.median(rates[:half])), float(np.median(rates[n - half:]))
        steady_decline = n >= 4 or bool(np.all(np.diff(rates) < 0))
        if not (
            change <= -self.drift_threshold
            and tau <= -0.5
            and late <= early * (1 - self.drift_threshold / 2)
            and steady_decline
        ):
            return None, None
        rows = by_shift[eligible]
        start = rows["start"].iloc[0]
        end = rows["end"].iloc[-1]
        run_hours = run_min[eligible] / 60.0
        lost = float(np.sum(np.maximum(0.0, first - fitted) * run_hours))
        down_pairs = int((signs < 0).sum())
        detail = (
            f"the running rate fell from about {number(first)}/h to {number(last)}/h "
            f"({change:+.0%}) across {n} shifts, {when(start)} to {when(end)}; {down_pairs} of "
            f"{signs.size} shift-to-shift comparisons point down, so this is a sustained "
            "decline, not one bad hour"
        )
        event = Event(
            kind="rate_drift",
            start=start,
            end=end,
            minutes=float((end - start).total_seconds() / 60.0),
            severity="warning",
            detail=detail + f"; about {number(lost)} units lost against the first shift's pace",
            units_lost=lost,
            value=change,
        )
        return event, None

    # ------------------------------------------------------------------ text
    def _findings(
        self,
        *,
        events: List[Event],
        stoppages: List[Stoppage],
        lost_by_cause: Dict[str, float],
        units: float,
        dt_min: float,
        scheduled_minutes: float,
        typical: Optional[float],
        target_per_interval: Optional[float],
        slow_limit: Optional[float],
        slow_reference: Optional[float],
        n_known: int,
        grid: Grid,
        known: np.ndarray,
        downtime_mask: np.ndarray,
        minute_of_day: np.ndarray,
        day: np.ndarray,
        sched: np.ndarray,
        shift_day: np.ndarray,
    ) -> List[str]:
        if n_known == 0:
            if not np.isfinite(grid.units).any():
                return ["Nothing to measure: the output column has no usable numbers."]
            windows = ", ".join(shift.window for shift in self.schedule.shifts)
            return [
                "Nothing to measure: no usable readings fall inside the scheduled hours "
                f"({windows})."
            ]
        ranked: List[Tuple[float, str]] = []
        by_kind: Dict[str, List[Event]] = {}
        for event in events:
            by_kind.setdefault(event.kind, []).append(event)

        downtime = [s for s in stoppages if s.kind == "downtime"]
        if downtime:
            total = sum(s.minutes for s in downtime)
            longest = max(downtime, key=lambda s: s.minutes)
            lost = lost_by_cause.get("downtime")
            text = (
                f"Downtime: stopped for {duration(total)} of {duration(scheduled_minutes)} "
                f"scheduled ({percent(total / scheduled_minutes if scheduled_minutes else None)}) "
                f"in {len(downtime)} stoppage{'s' if len(downtime) != 1 else ''}; the longest ran "
                f"{duration(longest.minutes)} from {when(longest.start)}."
            )
            if lost is not None:
                text += f" About {number(lost)} units lost."
            ranked.append((lost if lost is not None else total, text))

        drift = by_kind.get("rate_drift", [])
        if drift:
            event = drift[0]
            text = "Rate drift: " + event.detail[0].lower() + event.detail[1:] + "."
            text += " Look for wear, material or setting changes."
            ranked.append((event.units_lost or 0.0, text))

        slow = by_kind.get("slow_running", [])
        if slow:
            total = sum(e.minutes for e in slow)
            longest = max(slow, key=lambda e: e.minutes)
            ref_name = (
                "target_rate"
                if target_per_interval is not None and slow_reference == target_per_interval
                else "the line's typical rate"
            )
            lost = lost_by_cause.get("slow_running")
            text = (
                f"Slow running: {len(slow)} period{'s' if len(slow) != 1 else ''}, "
                f"{duration(total)} in total, below {self.slow_fraction:.0%} of {ref_name} "
                f"({number(slow_limit)} per interval); the longest ran from "
                f"{when(longest.start)} to {when(longest.end)}."
            )
            if lost is not None:
                text += f" About {number(lost)} units lost."
            ranked.append((lost if lost is not None else total, text))

        micro = by_kind.get("micro_stop", [])
        if micro:
            lost = lost_by_cause.get("micro_stops")
            hours = scheduled_minutes / 60.0
            per_hour = len(micro) / hours if hours else float("nan")
            longest = max(e.minutes for e in micro)
            text = (
                f"Micro-stops: {len(micro)} short drop{'s' if len(micro) != 1 else ''} "
                f"({per_hour:.1f} per scheduled hour), none longer than {duration(longest)}."
            )
            if lost is not None and units + lost > 0:
                equivalent = lost / self._rate_or_one(target_per_interval, typical) * dt_min
                text += (
                    f" Each looks harmless, but together they cost about {number(lost)} units "
                    f"({percent(lost / (units + lost))} of possible output), the same as "
                    f"{duration(equivalent)} of full-speed running."
                )
            ranked.append((lost if lost is not None else 0.0, text))

        if target_per_interval is not None and typical is not None and typical > 0:
            ratio = typical / target_per_interval
            per_hour_typical = typical * 60.0 / dt_min
            if ratio < 0.95:
                ranked.append(
                    (
                        lost_by_cause.get("speed_loss", 0.0),
                        f"Running speed: when it runs, the line typically makes "
                        f"{number(per_hour_typical)}/h, {1 - ratio:.0%} below target_rate "
                        f"{number(self.target_rate)}/h.",
                    )
                )
            elif ratio > 1.2:
                ranked.append(
                    (
                        0.0,
                        f"Running speed: the line typically makes {number(per_hour_typical)}/h, "
                        f"{ratio - 1:.0%} above target_rate {number(self.target_rate)}/h, so "
                        "performance is capped at 100%. Check that target_rate is in units per hour.",
                    )
                )

        ranked.sort(key=lambda item: -item[0])
        findings = [text for _, text in ranked]

        spikes = by_kind.get("spike", [])
        if spikes:
            worst = max(spikes, key=lambda e: e.value or 0.0)
            ratio = (worst.value or 0.0) / typical if typical else float("nan")
            findings.append(
                f"Spikes: {len(spikes)} interval{'s' if len(spikes) != 1 else ''} far above "
                f"normal; the largest, {number(worst.value)} units at {when(worst.start)}, is "
                f"{ratio:.1f}x the typical rate and {_spike_cause(ratio)}. Spikes are counted "
                "at the typical rate so they do not inflate the totals."
            )
        resets = by_kind.get("counter_reset", [])
        if len(resets) == 1:
            detail = resets[0].detail
            findings.append("Counter reset: " + detail[0].lower() + detail[1:] + ".")
        elif resets:
            findings.append(
                f"Counter resets: {len(resets)}, the first at {when(resets[0].start)}. Each is "
                "reported as a reset, not as a stoppage or a negative rate; the units made in "
                "those intervals are unknown."
            )
        days = _production_days(sched, known, downtime_mask, shift_day)
        gaps = by_kind.get("data_gap", [])
        empty = days["empty"] if days is not None else []
        if empty:
            findings.append(
                f"Days without data: {_plural(len(empty), 'whole scheduled day')} had no "
                f"readings at all ({_day_list(empty)}). If the line is not scheduled then, "
                "that is expected; they count neither as downtime nor as running time."
            )
            empty_set = set(empty)
            gaps = [e for e in gaps if _day_number(e.start) not in empty_set]
        if gaps:
            total = sum(e.minutes for e in gaps)
            findings.append(
                f"Data gaps: {len(gaps)} gap{'s' if len(gaps) != 1 else ''}, {duration(total)} "
                "in total, inside scheduled hours had no usable readings; they count neither "
                "as downtime nor as running time."
            )
        hint = self._schedule_hint(known, downtime_mask, minute_of_day, day)
        if hint:
            findings.append(hint)
        if days is not None and days["idle"]:
            idle = days["idle"]
            hours = days["idle_down_bins"] * dt_min
            every = _every_weekday(idle, days["full"])
            findings.append(
                f"Schedule: the line made nothing on {_plural(len(idle), 'whole scheduled day')} "
                f"({_day_list(idle)}){every}, {duration(hours)} of the downtime. If it is "
                "simply not scheduled on those days, leave their rows out of df: days "
                "without data count neither as downtime nor as running time."
            )
        if not ranked and not spikes:
            findings.insert(
                0,
                f"No downtime, slow running, drift, micro-stops or spikes in "
                f"{duration(scheduled_minutes)} of scheduled production.",
            )
        return findings

    @staticmethod
    def _rate_or_one(target: Optional[float], typical: Optional[float]) -> float:
        if target:
            return target
        if typical:
            return typical
        return 1.0

    def _schedule_hint(
        self,
        known: np.ndarray,
        downtime_mask: np.ndarray,
        minute_of_day: np.ndarray,
        day: np.ndarray,
    ) -> Optional[str]:
        """Without shift_hours, spot hours that are down every day: probably unscheduled."""
        if self.schedule.given or not downtime_mask.any():
            return None
        hour = minute_of_day[known] // 60
        cells = pd.DataFrame(
            {"cell": day[known] * 24 + hour, "hour": hour, "down": downtime_mask[known]}
        )
        per_cell = cells.groupby("cell").agg(hour=("hour", "first"), down=("down", "mean"))
        per_cell["stopped"] = per_cell["down"] >= 0.9
        per_hour = per_cell.groupby("hour").agg(days=("stopped", "size"), stopped=("stopped", "sum"))
        if int(np.unique(day[known]).size) < 2:
            return None
        recurring = per_hour[(per_hour["days"] >= 2) & (per_hour["stopped"] >= 0.8 * per_hour["days"])]
        hours = set(int(h) for h in recurring.index)
        if not hours or len(hours) >= 24:
            return None
        down_windows = _hour_windows(hours)
        run_windows = _hour_windows(set(range(24)) - hours)
        every = bool((recurring["stopped"] == recurring["days"]).all())
        days = int(recurring["days"].max())
        when_text = f"on every day in the data ({days} days)" if every else "on most days"
        windows = ", ".join(f"{clock(a * 60)}-{clock(b * 60)}" for a, b in down_windows)
        if len(run_windows) == 1:
            a, b = run_windows[0]
            suggestion = f"shift_hours=({a}, {b})"
        else:
            suggestion = "shift_hours=[" + ", ".join(f"({a}, {b})" for a, b in run_windows) + "]"
        return (
            f"Schedule: the line was stopped {windows} {when_text}. If it is simply not "
            f"scheduled then, pass {suggestion} so those hours count as unscheduled rather "
            "than as downtime."
        )


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _day_number(stamp: pd.Timestamp) -> int:
    """Days since 1970-01-01 of a timestamp's local date."""
    naive = stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
    return int(naive.value // NS_PER_DAY)


def _weekday(day: int) -> int:
    """Monday = 0 for a day counted from 1970-01-01, which was a Thursday."""
    return (int(day) + 3) % 7


def _day_list(days: Sequence[int], limit: int = 6) -> str:
    """'Sat 2026-04-11, Sun 2026-04-12' (and 'and 3 more' past ``limit``)."""
    texts = [
        f"{_WEEKDAYS[_weekday(d)][:3]} {np.datetime64(int(d), 'D')}" for d in list(days)[:limit]
    ]
    if len(days) > limit:
        texts.append(f"and {len(days) - limit} more")
    return ", ".join(texts)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _production_days(
    sched: np.ndarray, known: np.ndarray, downtime_mask: np.ndarray, shift_day: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Whole production days that were idle from start to end, or had no readings at all.

    A production day is the day its shift started on. Only days with (nearly) their full
    scheduled time inside the data count, so a partial first or last day is never listed.
    """
    if not sched.any():
        return None
    days, inverse = np.unique(shift_day[sched], return_inverse=True)
    n_sched = np.bincount(inverse).astype(float)
    n_known = np.bincount(inverse, weights=known[sched].astype(float))
    n_down = np.bincount(inverse, weights=downtime_mask[sched].astype(float))
    full = n_sched >= 0.9 * n_sched.max()
    measured = n_known >= 0.5 * n_sched
    idle = full & measured & (n_known > 0) & (n_down >= n_known - 1e-9)
    worked = (n_known > 0) & (n_down < n_known)
    empty = full & (n_known == 0)
    if not worked.any():
        return {"idle": [], "empty": [], "full": [], "idle_down_bins": 0.0}
    return {
        "idle": [int(d) for d in days[idle]],
        "empty": [int(d) for d in days[empty]],
        "full": [int(d) for d in days[full & measured]],
        "idle_down_bins": float(n_down[idle].sum()),
    }


def _every_weekday(idle: Sequence[int], full: Sequence[int]) -> str:
    """', every Saturday and Sunday in the data' when the idle days are exactly those weekdays."""
    weekdays = sorted({_weekday(d) for d in idle})
    if not weekdays:
        return ""
    for w in weekdays:
        of_that_day = [d for d in full if _weekday(d) == w]
        if len(of_that_day) < 2 or any(d not in set(idle) for d in of_that_day):
            return ""
    names = [_WEEKDAYS[w] for w in weekdays]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f", every {joined} in the data"


def _hour_windows(hours: set) -> List[Tuple[int, int]]:
    """Circular runs of whole hours: {22, 23, 0, 1} -> [(22, 2)]."""
    if not hours:
        return []
    flags = np.array([h in hours for h in range(24)], dtype=bool)
    if flags.all():
        return [(0, 24)]
    shift = int(np.flatnonzero(~flags)[0])
    rolled = np.roll(flags, -shift)
    out = []
    for a, b in _runs(rolled):
        out.append(((a + shift) % 24, (b + shift) % 24 or 24))
    out.sort(key=lambda w: ((w[0] - 6) % 24))
    return out


def _spike_cause(ratio: float) -> str:
    """What a spike of ``ratio`` times the typical rate most likely is, as a verb phrase."""
    if not math.isfinite(ratio):
        return "cannot be compared with a typical rate"
    if 1.7 <= ratio <= 2.3:
        return "looks like a double count (the same units counted twice)"
    if ratio >= 8:
        return "looks like a counter reset or rollover read as production"
    return "is more than the line makes in one interval; check the counter"


def analyze(
    df: TableLike,
    *,
    time: Any = None,
    output: Any = None,
    target_rate: Optional[float] = None,
    shift_hours: Any = None,
) -> ProductionReport:
    """Analyze a production line's output and explain what went wrong.

    ``df`` holds units made per interval (or a running counter) with a time column, as a
    DataFrame or a .csv/.tsv/.parquet path. ``time`` and ``output`` name the timestamp and
    units columns (found automatically when left out). ``target_rate`` is the rated speed in
    units per HOUR; without it the line is measured against its own typical rate.
    ``shift_hours`` says when the line is scheduled, e.g. ``(6, 22)`` or
    ``[(6, 14), (14, 22)]``; hours outside it are never counted as downtime.
    """
    return ProductionAnalyzer(
        time=time, output=output, target_rate=target_rate, shift_hours=shift_hours
    ).analyze(df)
