"""The scorer: four components, one 0-100 number, and a penalty that adds up."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._io import TableLike, check_columns, load_table
from ._util import as_float, clamp, column_map, describe_names, select_channels, resolve_time, sort_key
from .result import COMPONENTS, MachineScore, grade_for
from .rules import (
    Rule,
    Violation,
    breach_fraction,
    check_rule_channels,
    evaluate_rule,
    normalize_rules,
)

log = logging.getLogger(__name__)

#: Default weight of each component. They sum to 1 and are documented in the README.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "stability": 0.30,
    "compliance": 0.30,
    "anomaly": 0.20,
    "availability": 0.20,
}

#: A channel varying this many times more than its baseline scores 0 for stability.
STABILITY_CAP = 4.0
#: A steady level within this many baseline spreads of the healthy median is free.
LEVEL_FREE = 1.0
#: A steady level this many baseline spreads away from it scores 0 for stability.
LEVEL_CAP = 1000.0
#: Robust z above which a reading counts as an outlier.
ANOMALY_Z = 3.5
#: Share of outlying readings at which a channel scores 0 for anomaly.
ANOMALY_FULL = 0.10
#: Points a flatlined channel loses from availability, on top of its missing share.
FLATLINE_DEFICIT = 50.0
#: Rows a window needs before a trend is worth computing.
TREND_MIN_ROWS = 6
#: Points a batch must move against its history before HealthMonitor's trend reacts.
TREND_TOLERANCE = 2.0
#: Baseline spreads a half-window must move before score()'s trend stops being "stable".
TREND_DRIFT_TOLERANCE = 0.75
#: What a fully missing half-window counts as, in baseline spreads.
TREND_MISSING_WEIGHT = 3.0
#: What a fully breached rule counts as, in baseline spreads.
TREND_BREACH_WEIGHT = 5.0
#: What a move off a baseline that never varied counts as, in baseline spreads.
TREND_FLAT_STEP = 4.0
#: Default share of the data treated as the healthy baseline period.
DEFAULT_BASELINE_FRACTION = 0.2

#: numpy is told to keep quiet about overflow on absurd magnitudes: the score stays
#: inside 0-100 either way, and a library should not raise warnings its caller did
#: not ask for. A channel whose scale stops being finite is reported as a note.
_QUIET = dict(over="ignore", invalid="ignore", divide="ignore", under="ignore")

_MEASURED_BY_CHANNEL = ("stability", "anomaly", "availability")


@dataclass
class _ChannelBaseline:
    """What the healthy period looked like for one channel."""

    name: str
    n_valid: int
    median: float
    scale: float  # standard deviation
    mad_scale: float  # median absolute deviation, scaled to be comparable to a std
    tol: float
    varied: bool
    finite: bool = True  # False when the readings overflow and the moments are not real


def _baseline_stats(
    frame: pd.DataFrame,
    channels: Sequence[str],
    colmap: Mapping[str, Any],
    notes: Optional[List[str]] = None,
) -> Dict[str, _ChannelBaseline]:
    stats: Dict[str, _ChannelBaseline] = {}
    for name in channels:
        values = as_float(frame[colmap[name]]) if name in colmap else np.array([], dtype="float64")
        valid = values[np.isfinite(values)]
        with np.errstate(**_QUIET):
            if valid.size:
                median = float(np.median(valid))
                scale = float(np.std(valid)) if valid.size >= 2 else 0.0
                mad = float(np.median(np.abs(valid - median))) * 1.4826 if valid.size >= 2 else 0.0
            else:
                median, scale, mad = float("nan"), 0.0, 0.0
        finite = not valid.size or (
            math.isfinite(median) and math.isfinite(scale) and math.isfinite(mad)
        )
        if not finite:
            scale = scale if math.isfinite(scale) else 0.0
            mad = mad if math.isfinite(mad) else 0.0
            if notes is not None:
                notes.append(
                    f"channel {name!r} has readings too large to measure; its baseline spread "
                    "overflowed, so stability and anomaly are not scored for it"
                )
        tol = 1e-12 * max(1.0, abs(median) if math.isfinite(median) else 1.0)
        stats[name] = _ChannelBaseline(
            name=name,
            n_valid=int(valid.size),
            median=median,
            scale=scale,
            mad_scale=mad,
            tol=tol,
            varied=bool(finite and valid.size >= 2 and scale > tol),
            finite=finite,
        )
    return stats


def _spread_of(base: _ChannelBaseline) -> float:
    """The baseline's own spread: the robust MAD scale, or the plain std when it is flat."""
    return base.mad_scale if base.mad_scale > base.tol else base.scale


def _level_shift(base: _ChannelBaseline, valid: np.ndarray) -> Optional[float]:
    """How far the window's centre sits from the healthy median, in baseline spreads.

    ``None`` when there is nothing to compare. A baseline that never varied has no
    spread to measure in, so a move off it counts as :data:`TREND_FLAT_STEP`.
    """
    if valid.size == 0 or not base.finite or not math.isfinite(base.median):
        return None
    with np.errstate(**_QUIET):
        centre = float(np.median(valid))
    shift = abs(centre - base.median)
    if not math.isfinite(shift):
        return None
    spread = _spread_of(base)
    if spread <= base.tol:
        return 0.0 if shift <= base.tol else TREND_FLAT_STEP
    spreads = shift / spread
    return spreads if math.isfinite(spreads) else None


def _level_deficit(base: _ChannelBaseline, valid: np.ndarray) -> float:
    """0 while a channel sits at its healthy level, 100 once it is LEVEL_CAP spreads off it.

    This is what stops a channel that has stepped to a steady but wrong level from
    looking healthy: its variation is unchanged, so the spread ratio alone sees
    nothing, and every reading is an outlier, so the anomaly share saturates.
    """
    spreads = _level_shift(base, valid)
    if spreads is None or spreads <= LEVEL_FREE:
        return 0.0
    if spreads >= LEVEL_CAP:
        return 100.0
    return clamp(100.0 * math.log(spreads / LEVEL_FREE) / math.log(LEVEL_CAP / LEVEL_FREE))


def _stability_deficit(base: _ChannelBaseline, values: np.ndarray) -> Optional[float]:
    """How far a channel has moved from the way it behaved during the baseline.

    Two ways to move, and the worse of them is what stability costs: it swings more
    (0 at the baseline's own spread, 100 at STABILITY_CAP times it) or it has settled
    at a different level (0 within LEVEL_FREE baseline spreads of the healthy median,
    100 at LEVEL_CAP of them).
    """
    valid = values[np.isfinite(values)]
    if base.n_valid < 2 or valid.size < 2 or not base.finite:
        return None
    with np.errstate(**_QUIET):
        current = float(np.std(valid))
    if base.scale <= base.tol and current <= base.tol:
        ratio = 1.0
    elif base.scale <= base.tol:
        ratio = STABILITY_CAP
    else:
        ratio = current / base.scale
    if not math.isfinite(ratio) or ratio <= 1.0:
        spread_deficit = 0.0
    else:
        spread_deficit = clamp(100.0 * math.log(ratio) / math.log(STABILITY_CAP))
    return max(spread_deficit, _level_deficit(base, valid))


def _anomaly_deficit(base: _ChannelBaseline, values: np.ndarray) -> Optional[float]:
    """0 when no reading is an outlier against the baseline, 100 at ANOMALY_FULL of them."""
    valid = values[np.isfinite(values)]
    if base.n_valid < 3 or valid.size < 1 or not base.finite or not math.isfinite(base.median):
        return None
    scale = _spread_of(base)
    with np.errstate(**_QUIET):
        distance = np.abs(valid - base.median)
        if scale <= base.tol:
            outliers = distance > base.tol
        else:
            outliers = (distance / scale) > ANOMALY_Z
        share = float(np.count_nonzero(outliers)) / valid.size
    return clamp(100.0 * share / ANOMALY_FULL)


def _availability_deficit(base: _ChannelBaseline, values: np.ndarray) -> Optional[float]:
    """Missing readings, plus a flat penalty when a channel that used to move is stuck."""
    n_rows = int(values.size)
    if n_rows == 0:
        return None
    valid = values[np.isfinite(values)]
    missing = 1.0 - (valid.size / n_rows)
    deficit = 100.0 * missing
    with np.errstate(**_QUIET):
        flat = bool(valid.size >= 2 and base.varied and float(np.ptp(valid)) <= base.tol)
    if flat:
        deficit += FLATLINE_DEFICIT
    return clamp(deficit)


def resolve_weights(weights: Any) -> Tuple[Dict[str, float], List[str]]:
    """Merge user weights over the defaults, normalize to sum 1, and report what changed."""
    merged = dict(DEFAULT_WEIGHTS)
    notes: List[str] = []
    if weights is not None:
        if not isinstance(weights, Mapping):
            raise TypeError(
                "weights must be a mapping like {'stability': 0.5, 'compliance': 0.5}, "
                f"not {type(weights).__name__}"
            )
        unknown = [str(k) for k in weights if str(k) not in DEFAULT_WEIGHTS]
        if unknown:
            raise ValueError(
                f"unknown weight(s) {describe_names(unknown)}; "
                f"valid names: {', '.join(COMPONENTS)}"
            )
        for key, value in weights.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"weight {str(key)!r} must be a number, got {value!r}") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    f"weight {str(key)!r} must be a finite number >= 0, got {value!r}"
                )
            merged[str(key)] = number
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("weights are all zero; at least one component must count")
    if abs(total - 1.0) > 1e-9:
        merged = {k: v / total for k, v in merged.items()}
        if weights is not None:
            notes.append(
                "weights did not sum to 1 (sum was "
                f"{total:.4g}); normalized to "
                + ", ".join(f"{k}={merged[k]:.3f}" for k in COMPONENTS)
            )
    return merged, notes


def window_values(
    frame: pd.DataFrame, channels: Sequence[str]
) -> Dict[str, np.ndarray]:
    """One float array per channel; a channel absent from ``frame`` becomes all-NaN."""
    colmap = column_map(frame)
    out: Dict[str, np.ndarray] = {}
    for name in channels:
        if name in colmap:
            out[name] = as_float(frame[colmap[name]])
        else:
            out[name] = np.full(len(frame), np.nan, dtype="float64")
    return out


def _core(
    values: Mapping[str, np.ndarray],
    stats: Mapping[str, _ChannelBaseline],
    channels: Sequence[str],
    rules: Sequence[Rule],
    weights: Mapping[str, float],
    times: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Score one window against a fixed baseline. Returns every piece of the result."""
    notes: List[str] = []
    deficits: Dict[str, Optional[float]] = {}
    shares: Dict[str, Dict[str, float]] = {name: {} for name in COMPONENTS}
    violations: List[Violation] = []

    per_channel = {
        "stability": {name: _stability_deficit(stats[name], values[name]) for name in channels},
        "anomaly": {name: _anomaly_deficit(stats[name], values[name]) for name in channels},
        "availability": {
            name: _availability_deficit(stats[name], values[name]) for name in channels
        },
    }
    for component in _MEASURED_BY_CHANNEL:
        measured = {k: v for k, v in per_channel[component].items() if v is not None}
        if not measured:
            deficits[component] = None
            continue
        count = len(measured)
        deficits[component] = sum(measured.values()) / count
        shares[component] = {name: value / count for name, value in measured.items()}

    if rules:
        total_weight = 0.0
        weighted: Dict[str, float] = {}
        for rule in rules:
            deficit, rule_violations = evaluate_rule(rule, values[rule.channel], times)
            if deficit is None:
                continue
            total_weight += rule.weight
            weighted[rule.channel] = weighted.get(rule.channel, 0.0) + rule.weight * deficit
            violations.extend(rule_violations)
        if total_weight <= 0:
            deficits["compliance"] = None
        else:
            deficits["compliance"] = sum(weighted.values()) / total_weight
            shares["compliance"] = {k: v / total_weight for k, v in weighted.items()}
    else:
        deficits["compliance"] = None

    measurable = [name for name in COMPONENTS if deficits.get(name) is not None]
    effective = {name: 0.0 for name in COMPONENTS}
    if measurable:
        live_total = sum(weights[name] for name in measurable)
        if live_total <= 0:
            share = 1.0 / len(measurable)
            for name in measurable:
                effective[name] = share
            notes.append(
                "every component that could be measured had weight 0; "
                "they were given equal weight instead"
            )
        else:
            for name in measurable:
                effective[name] = weights[name] / live_total

    component_penalties = {
        name: effective[name] * float(deficits[name] or 0.0) for name in COMPONENTS
    }
    penalty = sum(component_penalties.values())
    contributors = {name: 0.0 for name in channels}
    for component in COMPONENTS:
        weight = effective[component]
        if weight <= 0:
            continue
        for name, share in shares[component].items():
            contributors[name] = contributors.get(name, 0.0) + weight * share

    components = {
        name: (100.0 - float(deficits[name]) if deficits[name] is not None else 100.0)
        for name in COMPONENTS
    }
    violations.sort(key=lambda v: (0 if v.severity == "critical" else 1, -v.fraction, v.channel))
    return {
        "value": clamp(100.0 - penalty),
        "components": {k: clamp(v) for k, v in components.items()},
        "weights": effective,
        "component_penalties": component_penalties,
        "contributors": contributors,
        "violations": violations,
        "notes": notes,
        "unmeasured": [name for name in COMPONENTS if deficits.get(name) is None],
    }


def _distress(
    values: Mapping[str, np.ndarray],
    stats: Mapping[str, _ChannelBaseline],
    channels: Sequence[str],
    rules: Sequence[Rule],
) -> Optional[float]:
    """How far half a window sits from the baseline, in baseline spreads.

    Deliberately not the score: the score is bounded and saturates, so two halves
    that are both thoroughly broken compare equal, and a half that straddles a step
    looks worse than the steady bad level that follows it. This number keeps rising
    with the level shift, the extra spread, the missing share and the breached
    fraction of each rule, so "worse than before" always reads as worse.
    """
    parts: List[float] = []
    for name in channels:
        base = stats[name]
        array = np.asarray(values[name], dtype="float64")
        if array.size == 0:
            continue
        valid = array[np.isfinite(array)]
        distress = TREND_MISSING_WEIGHT * (1.0 - valid.size / array.size)
        shift = _level_shift(base, valid)
        if shift is not None:
            distress += shift
        if valid.size >= 2 and base.n_valid >= 2 and base.scale > base.tol:
            with np.errstate(**_QUIET):
                ratio = float(np.std(valid)) / base.scale
            if math.isfinite(ratio):
                distress += max(0.0, ratio - 1.0)
        parts.append(distress if math.isfinite(distress) else 0.0)
    for rule in rules:
        fraction = breach_fraction(rule, np.asarray(values[rule.channel], dtype="float64"))
        if fraction is not None:
            parts.append(TREND_BREACH_WEIGHT * fraction)
    if not parts:
        return None
    return sum(parts) / len(parts)


def _trend(
    values: Mapping[str, np.ndarray],
    n_rows: int,
    stats: Mapping[str, _ChannelBaseline],
    channels: Sequence[str],
    rules: Sequence[Rule],
) -> Tuple[str, Optional[str]]:
    """Compare the two halves of the window; returns (trend, note)."""
    if n_rows < TREND_MIN_ROWS:
        return "stable", (
            f"trend needs at least {TREND_MIN_ROWS} rows in the window; reported as stable"
        )
    half = n_rows // 2
    older = {name: array[:half] for name, array in values.items()}
    newer = {name: array[half:] for name, array in values.items()}
    before = _distress(older, stats, channels, rules)
    after = _distress(newer, stats, channels, rules)
    if before is None or after is None:
        return "stable", (
            "trend needs readings in both halves of the window; reported as stable"
        )
    change = after - before
    if change > TREND_DRIFT_TOLERANCE:
        return "degrading", None
    if change < -TREND_DRIFT_TOLERANCE:
        return "improving", None
    return "stable", None


def build_score(
    values: Mapping[str, np.ndarray],
    n_rows: int,
    stats: Mapping[str, _ChannelBaseline],
    channels: Sequence[str],
    rules: Sequence[Rule],
    weights: Mapping[str, float],
    *,
    times: Optional[Sequence[Any]] = None,
    notes: Optional[Sequence[str]] = None,
    n_baseline_rows: int = 0,
) -> MachineScore:
    """Score one window and assemble the MachineScore, trend and notes included."""
    collected = list(notes or [])
    result = _core(values, stats, channels, rules, weights, times)
    collected.extend(result["notes"])
    for component in result["unmeasured"]:
        collected.append(_unmeasured_note(component, bool(rules)))
    trend, trend_note = _trend(values, n_rows, stats, channels, rules)
    if trend_note:
        collected.append(trend_note)
    value = result["value"]
    log.debug(
        "scored %d row(s) on %d channel(s): %.2f (%s)", n_rows, len(channels), value, trend
    )
    return MachineScore(
        value=value,
        grade=grade_for(value),
        components=result["components"],
        violations=result["violations"],
        contributors=result["contributors"],
        trend=trend,
        weights=result["weights"],
        component_penalties=result["component_penalties"],
        unmeasured=list(result["unmeasured"]),
        channels=list(channels),
        notes=collected,
        n_rows=n_rows,
        n_baseline_rows=n_baseline_rows,
        when=(times[-1] if times is not None and len(times) else None),
    )


def _split_baseline(
    n_rows: int, baseline: Any
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (baseline positions, window positions, notes) for the ``baseline`` argument."""
    notes: List[str] = []
    everything = np.arange(n_rows)
    if isinstance(baseline, (pd.Series, np.ndarray, list, tuple)) and not isinstance(baseline, str):
        series = pd.Series(list(baseline))
        if series.dtype == object:
            series = series.infer_objects()
        if series.dtype != bool or len(series) != n_rows:
            raise ValueError(
                "baseline given as a mask must be a boolean sequence with one value per row "
                f"({n_rows} expected, got {len(series)} of dtype {series.dtype})"
            )
        mask = series.to_numpy(dtype=bool)
        if not mask.any():
            raise ValueError("baseline mask selects no rows; it must mark the healthy period")
        base_pos = everything[mask]
        window_pos = everything[~mask]
        if window_pos.size == 0:
            window_pos = everything
            notes.append("baseline mask covers every row; the same rows are also the window")
        return base_pos, window_pos, notes

    if baseline is None:
        fraction = DEFAULT_BASELINE_FRACTION
        n_base = max(1, int(round(n_rows * fraction)))
    elif isinstance(baseline, bool):
        raise TypeError("baseline must be rows, a fraction or a row count, not a bool")
    elif isinstance(baseline, float):
        if not 0.0 < baseline < 1.0:
            raise ValueError(f"baseline as a fraction must be between 0 and 1, got {baseline}")
        n_base = max(1, int(round(n_rows * baseline)))
    elif isinstance(baseline, (int, np.integer)):
        if baseline < 1:
            raise ValueError(f"baseline as a row count must be at least 1, got {baseline}")
        n_base = int(baseline)
    else:  # pragma: no cover - guarded by the caller
        raise TypeError(f"unsupported baseline of type {type(baseline).__name__}")

    if n_base >= n_rows:
        notes.append(
            f"baseline wanted {n_base} of {n_rows} row(s); the whole table is both baseline "
            "and window, so the score compares the data with itself"
        )
        return everything, everything, notes
    return everything[:n_base], everything[n_base:], notes


class HealthScorer:
    """The class behind :func:`score`, for when you want to keep the settings around.

    All arguments match :func:`score`. Call :meth:`score` with as many tables as you
    like; each one is split into its own baseline and window. Use
    :class:`~machine_health.monitor.HealthMonitor` instead when the baseline must
    stay fixed across batches.
    """

    def __init__(
        self,
        *,
        time: Any = None,
        channels: Optional[Sequence[Any]] = None,
        rules: Any = None,
        weights: Optional[Mapping[str, float]] = None,
        baseline: Any = None,
    ) -> None:
        self.time = time
        self.channels = list(channels) if channels is not None else None
        self.rules = normalize_rules(rules)
        self.weights, self._weight_notes = resolve_weights(weights)
        self.baseline = baseline

    def score(self, df: TableLike) -> MachineScore:
        """Score one table. See :func:`machine_health.score`."""
        frame = load_table(df, "df")
        check_columns(frame, "df")
        if frame.shape[1] == 0:
            raise ValueError("df has no columns; machine-health needs at least one sensor channel")
        if len(frame) == 0:
            raise ValueError("df has no rows; machine-health needs at least one reading")

        notes = list(self._weight_notes)
        times, time_name, time_notes = resolve_time(frame, self.time)
        notes.extend(time_notes)
        if times is not None:
            order = np.argsort(sort_key(times), kind="stable")
            if not np.array_equal(order, np.arange(len(frame))):
                frame = frame.iloc[order]
                times = times.iloc[order]
                notes.append(f"rows were sorted by {time_name or 'time'} before scoring")

        channels, skipped, channel_notes = _resolve_channels(
            frame, self.channels, self.rules, time_name
        )
        notes.extend(channel_notes)
        if skipped:
            notes.append(
                f"ignored non-numeric column(s): {describe_names(skipped)}"
            )
        base_frame, window_pos, split_notes = self._baseline_frames(frame)
        notes.extend(split_notes)
        window = frame.iloc[window_pos]
        base_map = column_map(base_frame)
        absent = [name for name in channels if name not in base_map]
        if absent:
            notes.append(
                f"baseline table has no column for channel(s) {describe_names(absent)}; "
                "they are only scored for availability"
            )
        stats = _baseline_stats(base_frame, channels, base_map, notes)
        window_times = list(times.iloc[window_pos]) if times is not None else None
        return build_score(
            window_values(window, channels),
            len(window),
            stats,
            channels,
            self.rules,
            self.weights,
            times=window_times,
            notes=notes,
            n_baseline_rows=len(base_frame),
        )

    def _baseline_frames(
        self, frame: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        baseline = self.baseline
        if isinstance(baseline, (pd.DataFrame, str)) or hasattr(baseline, "__fspath__"):
            base_frame = load_table(baseline, "baseline")
            check_columns(base_frame, "baseline")
            if len(base_frame) == 0:
                raise ValueError("baseline has no rows; it must describe the healthy period")
            return base_frame, np.arange(len(frame)), ["baseline came from a separate table"]
        base_pos, window_pos, notes = _split_baseline(len(frame), baseline)
        return frame.iloc[base_pos], window_pos, notes


def _unmeasured_note(component: str, has_rules: bool) -> str:
    if component == "compliance":
        if not has_rules:
            return (
                "no rules given, so compliance was not scored; its weight went to the other "
                "components. Pass rules={'channel': {'max': ..}} to include your own limits"
            )
        return "no rule could be checked (no usable readings on those channels); compliance was not scored"
    reasons = {
        "stability": "needs at least 2 readings in both the baseline and the window",
        "anomaly": "needs at least 3 baseline readings",
        "availability": "needs at least one row in the window",
    }
    return f"{component} was not scored ({reasons[component]}); its weight went to the other components"


def _resolve_channels(
    frame: pd.DataFrame,
    channels: Optional[Sequence[Any]],
    rules: Sequence[Rule],
    time_name: Optional[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Pick the channels to score, pulling in any extra channel a rule names."""
    exclude = (time_name,) if time_name else ()
    available, skipped = select_channels(frame, None, exclude=exclude)
    notes: List[str] = []
    if channels is None:
        chosen = list(available)
    else:
        chosen, _ = select_channels(frame, channels, exclude=exclude)
        extra = [r.channel for r in rules if r.channel in available and r.channel not in chosen]
        for name in extra:
            if name not in chosen:
                chosen.append(name)
        if extra:
            notes.append(
                f"channel(s) {describe_names(sorted(set(extra)))} were added because a rule names them"
            )
        skipped = []
    check_rule_channels(rules, available)
    if not chosen:
        raise ValueError(
            "no numeric channel found in df; machine-health scores numeric sensor columns. "
            f"columns present: {describe_names(list(frame.columns))}"
        )
    return chosen, skipped, notes


def score(
    df: TableLike,
    *,
    time: Any = None,
    channels: Optional[Sequence[Any]] = None,
    rules: Any = None,
    weights: Optional[Mapping[str, float]] = None,
    baseline: Any = None,
) -> MachineScore:
    """Score one machine's telemetry as a single 0-100 health number.

    Args:
        df: a DataFrame or a path to a ``.csv`` / ``.tsv`` / ``.parquet`` file, one
            row per reading and one column per sensor channel.
        time: name of the timestamp column (or a sequence of timestamps). Rows are
            sorted by it and violations report when they first happened.
        channels: which columns to score. Defaults to every numeric column.
        rules: your own limits: ``{"temp": {"max": 80}}``, a :class:`Rule`, or a list
            of them. A rule naming a channel that is not present raises ValueError.
        weights: how much each component counts, e.g. ``{"compliance": 0.5}``.
            Weights that do not sum to 1 are normalized and a note is recorded.
        baseline: the healthy period. A DataFrame or file path, a fraction, a row
            count, or a boolean mask. Defaults to the first 20% of the rows.

    Returns:
        A :class:`~machine_health.result.MachineScore`: ``.value`` 0-100, ``.grade``,
        ``.components``, ``.violations``, ``.contributors``, ``.trend``,
        ``.summary()`` and ``.to_dict()``.

    Note:
        Without ``rules`` the score is statistical: it knows how the machine behaves
        now compared with its own baseline, not what its readings are supposed to be.
        A channel that has settled at a steady but wrong level is caught by stability,
        but a baseline that was never healthy will still look fine. ``.trend`` compares
        the two halves of this one window with each other; use
        :class:`~machine_health.monitor.HealthMonitor` to compare batch with batch.
    """
    return HealthScorer(
        time=time, channels=channels, rules=rules, weights=weights, baseline=baseline
    ).score(df)
