"""The three calls everybody uses, and the class underneath them."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import _model, _prep, _tariff
from ._io import load
from ._model import Baseline, Trend, _when_text
from ._prep import Prepared
from ._report import Anomaly, EnergyReport, _money, _num
from ._tariff import Tariff

LOG = logging.getLogger(__name__)

#: default threshold, in robust sigmas away from the same-phase baseline.
#: 4.0 rather than 3.0 because a meter series is long: at 3 sigmas a few hundred
#: periods of ordinary noise throw up a handful of flags that mean nothing.
SENSITIVITY = 4.0

#: share of comparable periods above which a baseline the caller supplied is
#: calling almost everything unusual, which tells the reader nothing about which
#: periods matter.
DEGENERATE_SHARE = 0.5
#: below this many comparable periods a share is not worth reading anything into.
DEGENERATE_MIN_PERIODS = 8


def _warn_if_baseline_is_degenerate(
    flagged: np.ndarray, comparable: np.ndarray, warnings: List[str]
) -> None:
    """Say so when the caller's own baseline flags nearly every period.

    A report in which almost everything is an anomaly conveys nothing about which
    periods matter, so the likeliest explanation is that the supplied baseline does
    not describe this meter. Falling back would quietly ignore what the caller
    asked for, so the report keeps their baseline and says what happened instead.
    """
    usable = int(np.count_nonzero(comparable))
    if usable < DEGENERATE_MIN_PERIODS:
        return
    share = int(np.count_nonzero(flagged)) / usable
    if share < DEGENERATE_SHARE:
        return
    warnings.append(
        f"{share:.0%} of periods were flagged against the baseline you supplied, so "
        "the baseline probably does not describe this meter; leave baseline=None to "
        "learn normal behaviour from the readings themselves"
    )


def _unit_for(label: Any) -> str:
    """Guess the unit from the column name, falling back to a neutral word."""
    lowered = str(label).lower()
    for needle, unit in (
        ("kwh", "kWh"),
        ("mwh", "MWh"),
        ("wh", "Wh"),
        ("kw", "kW"),
        ("m3", "m3"),
        ("therm", "therms"),
    ):
        if needle in lowered:
            return unit
    return "units"


class EnergyAnalyzer:
    """The analyser behind :func:`analyze`, for when you want the knobs.

    Parameters
    ----------
    tariff:
        A flat price per unit, or a ``{hour: rate}`` mapping for time-of-use
        pricing. ``None`` leaves every cost on the report as ``None``.
    baseline:
        ``None`` learns normal behaviour from the data itself. A number fixes the
        expected consumption per period. A DataFrame, Series or file path uses
        that reference period as "normal" instead.
    granularity:
        ``"auto"`` reads the grid size off the timestamps. Otherwise ``"hourly"``,
        ``"daily"``, ``"weekly"``, ``"15min"`` or any fixed pandas offset.
    sensitivity:
        Threshold in robust sigmas away from the same-phase baseline. Higher means
        fewer anomalies. The default 4.0 keeps ordinary noise out of a long series;
        drop it to 3.0 if you would rather see borderline periods.
    min_effect:
        A period must also differ from its baseline by at least this share of the
        expected value before it is worth reporting. The default 0.10 keeps a 7%
        wobble out of the findings; set it to 0.0 to report every threshold cross.
    night:
        The overnight window used for the always-on load, as ``(start, end)`` hours.
    cumulative:
        ``None`` detects a cumulative meter. ``True`` or ``False`` forces it.
    """

    def __init__(
        self,
        *,
        tariff: Any = None,
        baseline: Any = None,
        granularity: Any = "auto",
        sensitivity: float = SENSITIVITY,
        min_effect: float = 0.10,
        night: Tuple[int, int] = (0, 5),
        cumulative: Optional[bool] = None,
    ) -> None:
        self.tariff = tariff
        self.baseline = baseline
        self.granularity = granularity
        self.sensitivity = float(sensitivity)
        if not np.isfinite(self.sensitivity) or self.sensitivity <= 0:
            raise ValueError(f"sensitivity must be a positive number, got {sensitivity!r}")
        self.min_effect = float(min_effect)
        if not np.isfinite(self.min_effect) or self.min_effect < 0:
            raise ValueError(f"min_effect must be zero or more, got {min_effect!r}")
        self.night = (int(night[0]) % 24, int(night[1]) % 24)
        self.cumulative = cumulative

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EnergyAnalyzer(granularity={self.granularity!r}, "
            f"sensitivity={self.sensitivity:g})"
        )

    # ------------------------------------------------------------------ inputs
    def _prepare(
        self, df: Any, value: Optional[str], time: Optional[str], step: Any = None
    ) -> Prepared:
        """Load and grid one set of readings."""
        loaded = load(df, value=value, time=time)
        granularity = self.granularity if step is None else step
        return _prep.prepare(loaded, granularity=granularity, cumulative=self.cumulative)

    def _baseline_for(
        self, prepared: Prepared, warnings: List[str], notes: List[str]
    ) -> Baseline:
        """Fit the baseline, or build the one the caller asked for."""
        values = prepared.series.to_numpy(dtype=float)
        if self.baseline is None:
            return _model.fit(
                prepared.series,
                prepared.step,
                prepared.has_time,
                exclude=prepared.negative,
                warnings=warnings,
            )
        if isinstance(self.baseline, (int, float, np.integer, np.floating)) and not isinstance(
            self.baseline, bool
        ):
            constant = float(self.baseline)
            if not np.isfinite(constant):
                raise ValueError(f"baseline must be a finite number, got {self.baseline!r}")
            notes.append(f"the baseline was given as a fixed {_num(constant)} per period")
            learned = _model.fit(
                prepared.series,
                prepared.step,
                prepared.has_time,
                exclude=prepared.negative,
                warnings=[],
            )
            expected = np.full(values.size, constant, dtype=float)
            return Baseline(
                expected=expected,
                scale=_scale_of(values - expected, prepared.negative),
                method=f"fixed {_num(constant)} per {prepared.granularity}",
                level=constant,
                trend=learned.trend,
                phase_kind="flat",
                step_minutes=learned.step_minutes,
                phase_medians={0: constant},
                fallback=constant,
                slope_per_day=0.0,
                origin=prepared.series.index[0] if values.size else None,
                n_used=int(np.count_nonzero(np.isfinite(values))),
            )
        reference = self._prepare(self.baseline, None, None, step=prepared.step)
        if reference.n_periods == 0:
            raise ValueError("the baseline reference has no readings to learn from")
        learned = _model.fit(
            reference.series,
            reference.step,
            reference.has_time,
            exclude=reference.negative,
            warnings=warnings,
        )
        expected = learned.predict(prepared.series.index)
        notes.append(
            f"normal behaviour was learned from a reference period of "
            f"{reference.n_periods:,} {reference.granularity} period(s)"
        )
        return Baseline(
            expected=expected,
            scale=_scale_of(values - expected, prepared.negative) or learned.scale,
            method=f"reference period, {learned.method}",
            level=learned.level,
            trend=learned.trend,
            phase_kind=learned.phase_kind,
            step_minutes=learned.step_minutes,
            phase_medians=learned.phase_medians,
            fallback=learned.fallback,
            slope_per_day=learned.slope_per_day,
            origin=learned.origin,
            n_used=learned.n_used,
        )

    # ----------------------------------------------------------------- analyse
    def analyze(
        self, df: Any, *, value: Optional[str] = None, time: Optional[str] = None
    ) -> EnergyReport:
        """Analyse one meter and return an :class:`EnergyReport`."""
        prepared = self._prepare(df, value, time)
        notes = list(prepared.notes)
        warnings = list(prepared.warnings)
        tariff = _tariff.build(self.tariff, warnings)
        unit = _unit_for(prepared.label)

        if prepared.n_periods == 0:
            empty = pd.DataFrame(
                {name: pd.Series(dtype=float) for name in ("observed", "expected", "excess", "score")}
            )
            empty["is_anomaly"] = pd.Series(dtype=bool)
            empty["is_gap"] = pd.Series(dtype=bool)
            return EnergyReport(
                by_period=empty,
                findings=["No readings were given, so there is nothing to analyse."],
                label=prepared.label,
                time_label=prepared.time_label,
                unit=unit,
                granularity=prepared.granularity,
                has_time=prepared.has_time,
                sensitivity=self.sensitivity,
                tariff=tariff,
                notes=notes,
                warnings=warnings,
            )

        index = prepared.series.index
        values = prepared.values
        baseline = self._baseline_for(prepared, warnings, notes)
        expected = np.asarray(baseline.expected, dtype=float)
        excess = values - expected
        gaps = ~np.isfinite(values)
        negative = np.asarray(prepared.negative, dtype=bool)
        scale = baseline.scale

        if scale is None or not np.isfinite(scale) or scale <= 0:
            score = np.full(values.size, np.nan)
            flagged = np.zeros(values.size, dtype=bool)
        else:
            score = excess / scale
            level = abs(baseline.level) if np.isfinite(baseline.level) else 1.0
            floor = np.maximum(
                1e-9 * max(1.0, level), self.min_effect * np.abs(expected)
            )
            flagged = (
                np.isfinite(score)
                & (np.abs(score) >= self.sensitivity)
                & (np.abs(excess) >= floor)
                & ~gaps
                & ~negative
            )
        if self.baseline is not None:
            _warn_if_baseline_is_degenerate(flagged, ~gaps & ~negative, warnings)

        rates = tariff.rates(index, prepared.step, warnings)
        table: Dict[str, Any] = {
            "observed": values,
            "expected": expected,
            "excess": np.where(gaps, np.nan, excess),
            "score": score,
            "is_anomaly": flagged,
            "is_gap": gaps,
        }
        if negative.any():
            table["is_negative"] = negative
        if rates is not None:
            table["rate"] = rates
            table["cost"] = values * rates
            table["expected_cost"] = expected * rates
            table["excess_cost"] = np.where(gaps, np.nan, excess * rates)
        by_period = pd.DataFrame(table, index=index)
        by_period.index.name = index.name or ("time" if prepared.has_time else "reading")

        anomalies = [
            Anomaly(
                when=index[position],
                observed=float(values[position]),
                expected=float(expected[position]),
                excess=float(excess[position]),
                cost=None if rates is None else float(excess[position] * rates[position]),
                kind="spike" if excess[position] > 0 else "drop",
                score=float(score[position]),
                rate=None if rates is None else float(rates[position]),
            )
            for position in np.flatnonzero(flagged)
        ]

        usable = np.isfinite(values)
        total = float(np.sum(values[usable])) if usable.any() else 0.0
        total_cost = (
            None if rates is None else float(np.sum(values[usable] * rates[usable]))
        )
        spikes = [item for item in anomalies if item.excess > 0]
        excess_units = float(sum(item.excess for item in spikes))
        excess_cost = (
            None if rates is None else float(sum(item.cost or 0.0 for item in spikes))
        )
        standby, standby_how = _model.standby_load(
            prepared.series, prepared.step, prepared.has_time, negative, self.night
        )
        # Steps are looked for with the slow trend put back, so that one level
        # change is reported as one step instead of being sliced up by the ramp
        # the trend fit draws through it.
        level_only = excess + baseline.drift(index)
        residual = pd.Series(np.where(gaps | negative, np.nan, level_only), index=index)
        steps = _model.find_steps(residual, prepared.step, prepared.has_time, baseline.level)

        report = EnergyReport(
            by_period=by_period,
            anomalies=anomalies,
            total=total,
            total_cost=total_cost,
            baseline_load=float(standby),
            baseline_method=baseline.method,
            baseline_how=standby_how,
            excess_units=excess_units,
            excess_cost=excess_cost,
            trend=baseline.trend,
            steps=steps,
            label=prepared.label,
            time_label=prepared.time_label,
            unit=unit,
            granularity=prepared.granularity,
            has_time=prepared.has_time,
            n_readings=prepared.n_readings,
            n_gaps=prepared.n_gaps,
            longest_gap=prepared.longest_gap,
            negative_readings=int(np.count_nonzero(negative)),
            cumulative_meter=prepared.cumulative,
            meter_resets=prepared.meter_resets,
            scale=scale,
            sensitivity=self.sensitivity,
            tariff=tariff,
            notes=notes,
            warnings=warnings,
        )
        report.findings = _findings(report)
        return report

    # ------------------------------------------------------------ single facts
    def baseline_load(
        self, df: Any, *, value: Optional[str] = None, time: Optional[str] = None
    ) -> float:
        """The always-on floor on its own, per period of the analysis grid."""
        prepared = self._prepare(df, value, time)
        if prepared.n_periods == 0:
            return 0.0
        load_value, _ = _model.standby_load(
            prepared.series,
            prepared.step,
            prepared.has_time,
            np.asarray(prepared.negative, dtype=bool),
            self.night,
        )
        return float(load_value)

    def forecast(
        self,
        df: Any,
        periods: int = 24,
        *,
        value: Optional[str] = None,
        time: Optional[str] = None,
    ) -> pd.Series:
        """Project the next `periods` forward from the same-phase baseline."""
        try:
            count = int(periods)
        except (TypeError, ValueError):
            raise ValueError(f"periods must be a whole number, got {periods!r}") from None
        if count < 1:
            raise ValueError(f"periods must be at least 1, got {periods!r}")
        prepared = self._prepare(df, value, time)
        name = f"{prepared.label} forecast"
        if prepared.n_periods == 0:
            return pd.Series([], index=prepared.series.index[:0], name=name, dtype=float)
        warnings: List[str] = []
        notes: List[str] = []
        baseline = self._baseline_for(prepared, warnings, notes)
        for warning in warnings:
            LOG.info("forecast: %s", warning)
        index = prepared.series.index
        if prepared.has_time and prepared.step is not None:
            future = pd.date_range(
                start=index[-1] + prepared.step, periods=count, freq=prepared.step
            )
            future.name = index.name
        else:
            start = int(index[-1]) + 1
            future = pd.RangeIndex(start, start + count, name=index.name)
        projected = baseline.predict(future)
        observed = prepared.values
        usable = observed[np.isfinite(observed)]
        if usable.size and float(np.min(usable)) >= 0:
            projected = np.clip(projected, 0.0, None)
        return pd.Series(projected, index=future, name=name, dtype=float)


def _scale_of(excess: np.ndarray, exclude: np.ndarray) -> Optional[float]:
    """Robust sigma of a residual array, or ``None`` when there is no variation."""
    keep = np.isfinite(excess) & ~np.asarray(exclude, dtype=bool)
    if int(np.count_nonzero(keep)) < _model.MIN_PER_PHASE:
        return None
    gap = excess[keep]
    centre = float(np.median(gap))
    scale = float(np.median(np.abs(gap - centre))) * _model.MAD_TO_SIGMA
    if scale > 0 and np.isfinite(scale):
        return scale
    spread = float(np.std(gap))
    return spread if spread > 0 and np.isfinite(spread) else None


def _findings(report: EnergyReport) -> List[str]:
    """The plain-language read-out, in the order a person would want it."""
    out: List[str] = []
    unit = report.unit
    window = report.span
    where = ""
    if window is not None and report.has_time:
        where = f" from {_when_text(window[0])} to {_when_text(window[1])}"
    total = (
        f"Used {_num(report.total)} {unit} across {report.n_periods:,} "
        f"{report.granularity} periods{where}."
    )
    if report.total_cost is not None:
        total = total[:-1] + f", costing {_money(report.total_cost)}."
    out.append(total)

    if report.cumulative_meter:
        out.append(
            "The readings only ever increase, so they were read as a cumulative meter "
            "and analysed as the difference between consecutive readings, not as one "
            "huge value."
        )
    if report.meter_resets:
        out.append(
            f"{report.meter_resets} meter reset or rollover was found; those intervals "
            "are left as gaps rather than counted as consumption."
        )
    if report.n_periods:
        out.append(
            f"The always-on load is {_num(report.baseline_load)} {unit} per "
            f"{report.granularity}, about {report.baseline_share:.0%} of everything used "
            f"({report.baseline_how})."
        )
    if report.anomalies:
        spikes, drops = len(report.spikes), len(report.drops)
        parts = []
        if spikes:
            parts.append(f"{spikes} spike{'s' if spikes != 1 else ''}")
        if drops:
            parts.append(f"{drops} drop{'s' if drops != 1 else ''}")
        sentence = (
            f"{report.n_anomalies} period{'s' if report.n_anomalies != 1 else ''} did not "
            f"look like the same hour on other days: {' and '.join(parts)}."
        )
        out.append(sentence)
        if spikes:
            extra = f"The spikes used {_num(report.excess_units)} {unit} more than normal"
            if report.excess_cost is not None:
                extra += f", costing about {_money(report.excess_cost)}"
            out.append(extra + ".")
            worst = max(report.spikes, key=lambda item: item.excess)
            out.append(
                f"The biggest one was {_when_text(worst.when)}: "
                f"{_num(worst.observed)} {unit} against an expected "
                f"{_num(worst.expected)} {unit}."
            )
    elif report.scale is not None:
        out.append(
            f"Nothing stood out: every period is within {report.sensitivity:g} robust "
            "sigmas of its same-phase baseline."
        )
    if report.trend.direction == "flat":
        out.append("The baseline is steady; there is no clear trend across the window.")
    elif report.steps:
        out.append(
            f"Consumption is {report.trend}, but a straight line drawn through a window "
            "that contains a step change reads as a trend; the step below is the more "
            "likely explanation."
        )
    else:
        out.append(f"Consumption is {report.trend}.")
    for item in report.steps:
        out.append(f"The {item}.")
    if report.n_gaps:
        out.append(
            f"{report.n_gaps:,} of {report.n_periods:,} periods have no reading "
            f"({report.gap_share:.1%}, longest run {report.longest_gap}); they are "
            "reported as gaps and were not filled in."
        )
    if report.negative_readings:
        out.append(
            f"{report.negative_readings} period(s) have negative consumption, which "
            "usually means export to the grid or a meter reset; they are flagged and "
            "left out of the baseline."
        )
    if report.tariff.kind == "none":
        out.append(
            "No tariff was given, so every cost is left as None rather than zero. "
            "Pass tariff=0.28 or tariff={0: 0.12, 7: 0.31} to price it."
        )
    return out


def analyze(
    df: Any,
    *,
    value: Optional[str] = None,
    time: Optional[str] = None,
    tariff: Any = None,
    baseline: Any = None,
    granularity: Any = "auto",
) -> EnergyReport:
    """Find unusual consumption, explain what changed, and price it.

    Parameters
    ----------
    df:
        Meter readings: a DataFrame with a value column and a timestamp column
        (or a DatetimeIndex), a Series, a list of numbers, or a ``.csv`` /
        ``.parquet`` path.
    value, time:
        Column names. Both are guessed when omitted.
    tariff:
        A flat cost per unit, or ``{hour: rate}`` for time-of-use pricing.
        ``None`` leaves every cost field as ``None``.
    baseline:
        ``None`` learns normal behaviour from the data itself; a number fixes it;
        a DataFrame, Series or path uses that reference period as normal.
    granularity:
        ``"auto"``, or ``"hourly"`` / ``"daily"`` / ``"15min"`` / any fixed offset.

    Returns
    -------
    EnergyReport
        Totals, the always-on load, the anomalies, the trend, and findings in
        plain language.
    """
    return EnergyAnalyzer(
        tariff=tariff, baseline=baseline, granularity=granularity
    ).analyze(df, value=value, time=time)


def baseline_load(df: Any, **kw: Any) -> float:
    """The always-on floor on its own: the persistent overnight minimum.

    Takes the same keywords as :func:`analyze` plus ``night=(start, end)``.
    """
    value = kw.pop("value", None)
    time = kw.pop("time", None)
    kw.pop("tariff", None)
    return EnergyAnalyzer(**kw).baseline_load(df, value=value, time=time)


def forecast(df: Any, periods: int = 24, **kw: Any) -> pd.Series:
    """Project the next `periods` with a same-phase seasonal-naive projection.

    Each future period takes the median of its own phase - the same hour on the
    same weekday - carried forward along the trend. Takes the same keywords as
    :func:`analyze`.
    """
    value = kw.pop("value", None)
    time = kw.pop("time", None)
    kw.pop("tariff", None)
    return EnergyAnalyzer(**kw).forecast(df, periods, value=value, time=time)
