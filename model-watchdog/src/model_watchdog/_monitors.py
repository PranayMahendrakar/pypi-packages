"""The seven monitors, and the thresholds they compare against.

Every monitor takes the same three arguments - the window of recent records,
the reference profile, and the thresholds - and returns exactly one
:class:`~model_watchdog._report.Check`. A monitor that does not have the data
it needs returns an *inactive* check saying so; it never guesses and it never
fails for lack of input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from . import _stats
from ._records import timestamp_to_datetime, to_frame
from ._reference import ReferenceProfile
from ._report import Check, failed_check, inactive_check, ok_check


@dataclass
class Thresholds:
    """What counts as a problem. Sensible defaults; tune what you care about."""

    #: PSI above this means the prediction distribution moved.
    prediction_psi: float = 0.2
    #: PSI above this, for any single feature, means the input moved.
    feature_psi: float = 0.2
    #: Accuracy may fall this far below the reference before it is a failure.
    accuracy_drop: float = 0.05
    #: Error may rise this much above the reference, as a fraction.
    error_increase: float = 0.25
    #: Latency percentiles may reach this multiple of the reference.
    latency_multiplier: float = 2.0
    #: Share of missing feature values that is still acceptable.
    null_rate: float = 0.05
    #: Longest run of identical predictions tolerated in a full window.
    constant_run: int = 50
    #: Traffic may be this many times above or below the reference rate.
    volume_ratio: float = 3.0
    #: Below this many records, the data-hungry monitors stay inactive.
    min_records: int = 10
    #: Below this many records the two PSI monitors stay inactive. They need
    #: far more than ``min_records``: a quantile PSI needs enough points to
    #: fill its bins, and on a handful it reads as drift when nothing moved.
    min_drift_records: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the thresholds."""
        return {
            "prediction_psi": self.prediction_psi,
            "feature_psi": self.feature_psi,
            "accuracy_drop": self.accuracy_drop,
            "error_increase": self.error_increase,
            "latency_multiplier": self.latency_multiplier,
            "null_rate": self.null_rate,
            "constant_run": int(self.constant_run),
            "volume_ratio": self.volume_ratio,
            "min_records": int(self.min_records),
            "min_drift_records": int(self.min_drift_records),
        }


class Window:
    """The records a report is about, in a shape the monitors can use."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)
        frame = to_frame(self.records)
        self.frame = frame.data
        self.n = int(len(self.frame))
        self.predictions = self.frame["prediction"]
        self.actuals = self.frame["actual"]
        self.latency = self.frame["latency_ms"]
        self.timestamps = self.frame["ts"]
        self.features: Dict[str, pd.Series] = self._collect_features()

    def _collect_features(self) -> Dict[str, pd.Series]:
        columns: Dict[str, List[Any]] = {}
        for index, record in enumerate(self.records):
            for name, value in (record.get("features") or {}).items():
                key = str(name)
                if key not in columns:
                    columns[key] = [None] * self.n
                columns[key][index] = value
        return {name: pd.Series(values, dtype="object") for name, values in columns.items()}

    @property
    def feature_frame(self) -> pd.DataFrame:
        """The logged features as a DataFrame, empty when none were logged."""
        if not self.features:
            return pd.DataFrame()
        return pd.DataFrame(dict(self.features))

    def span(self) -> Any:
        """``(first, last)`` timestamp of the window, or ``(None, None)``."""
        stamps = self.timestamps.dropna()
        if stamps.empty:
            return None, None
        return timestamp_to_datetime(stamps.min()), timestamp_to_datetime(stamps.max())


def _present(series: pd.Series) -> pd.Series:
    return series[series.notna()] if series is not None else pd.Series([], dtype="object")


def _psi_min_records(reference: Any, current: Any, thresholds: Thresholds) -> int:
    """How many records a PSI comparison of these two sides needs to mean anything.

    A per-value comparison of a handful of labels is sound on very little data,
    so it only asks for ``min_records``. A quantile comparison of a continuous
    column is not: below ``min_drift_records`` its bins hold too few points and
    the number it produces is sampling noise, which is worse than silence.
    """
    plan = _stats.psi_plan(reference, current)
    if plan is not None and plan[0] == _stats.PSI_BY_QUANTILE:
        return max(int(thresholds.min_records), int(thresholds.min_drift_records))
    return int(thresholds.min_records)


def prediction_drift(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """PSI of the logged predictions against the reference predictions."""
    name = "prediction_drift"
    if reference.predictions is None or len(reference.predictions) == 0:
        return inactive_check(name, "no reference predictions; pass reference= to enable")
    current = _present(window.predictions)
    if current.empty:
        return inactive_check(name, "no predictions logged")
    needed = _psi_min_records(reference.predictions, current, thresholds)
    if window.n < needed:
        return inactive_check(
            name, "only %s; need %d" % (_plural(window.n, "record"), needed)
        )
    value = _stats.psi(reference.predictions, current)
    if value is None:
        return inactive_check(name, "predictions could not be compared with the reference")
    message = "PSI %.4f against %d reference predictions: %s" % (
        value,
        len(reference.predictions),
        _stats.psi_label(value),
    )
    builder = ok_check if value <= thresholds.prediction_psi else failed_check
    return builder(name, value, thresholds.prediction_psi, message, reference_n=len(reference.predictions))


def feature_drift(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """Per-feature PSI; the worst feature decides the check."""
    name = "feature_drift"
    if not reference.features:
        return inactive_check(name, "no reference features; pass reference= with feature columns")
    if not window.features:
        return inactive_check(name, "no features logged; pass features= to log()")
    shared = [key for key in reference.features if key in window.features]
    if not shared:
        return inactive_check(
            name,
            "no feature names in common (reference has %s, logs have %s)"
            % (", ".join(sorted(reference.features)[:5]), ", ".join(sorted(window.features)[:5])),
        )
    needed = max(
        _psi_min_records(reference.features[key], window.features[key], thresholds)
        for key in shared
    )
    if window.n < needed:
        return inactive_check(
            name, "only %s; need %d" % (_plural(window.n, "record"), needed)
        )
    scores: Dict[str, float] = {}
    for key in shared:
        value = _stats.psi(reference.features[key], window.features[key])
        if value is not None:
            scores[key] = value
    if not scores:
        return inactive_check(name, "no feature had values on both sides")
    worst = max(scores, key=lambda key: scores[key])
    value = scores[worst]
    drifted = sorted(key for key, score in scores.items() if score > thresholds.feature_psi)
    if drifted:
        message = "%d of %d features drifted (worst: %s, PSI %.4f)" % (
            len(drifted),
            len(scores),
            worst,
            value,
        )
        return failed_check(
            name, value, thresholds.feature_psi, message, per_feature=scores, drifted=drifted
        )
    message = "%d features checked, worst is %s at PSI %.4f: %s" % (
        len(scores),
        worst,
        value,
        _stats.psi_label(value),
    )
    return ok_check(name, value, thresholds.feature_psi, message, per_feature=scores, drifted=[])


def accuracy_drop(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """Accuracy (or error) of the window against the reference, using actuals."""
    name = "accuracy_drop"
    if reference.accuracy is None and reference.error is None:
        return inactive_check(name, "reference has no accuracy or error to compare against")
    if _present(window.actuals).empty:
        return inactive_check(name, "no actuals logged; call log(actual=...) when the label lands")
    scored = _stats.score(window.predictions, window.actuals)
    if scored is None:
        return inactive_check(name, "no prediction/actual pair to score")
    if scored["n"] < thresholds.min_records:
        return inactive_check(
            name,
            "only %s; need %d" % (_plural(scored["n"], "labelled record"), thresholds.min_records),
        )
    if reference.accuracy is not None and scored["accuracy"] is not None:
        value = float(scored["accuracy"])
        limit = max(0.0, float(reference.accuracy) - thresholds.accuracy_drop)
        message = "accuracy %.4f on %d labelled records vs reference %.4f (drop %.4f)" % (
            value,
            scored["n"],
            reference.accuracy,
            reference.accuracy - value,
        )
        builder = ok_check if value >= limit else failed_check
        return builder(
            name, value, limit, message, metric="accuracy", reference=reference.accuracy, n=scored["n"]
        )
    if reference.error is not None and scored["error"] is not None:
        value = float(scored["error"])
        limit = _error_limit(reference, thresholds)
        message = "mean absolute error %.4f on %d labelled records vs reference %.4f (limit %.4f)" % (
            value,
            scored["n"],
            reference.error,
            limit,
        )
        builder = ok_check if value <= limit else failed_check
        return builder(
            name, value, limit, message, metric="mae", reference=reference.error, n=scored["n"]
        )
    return inactive_check(
        name,
        "reference is a %s baseline but the logs score as %s"
        % (reference.task or "different", scored["task"]),
    )


def latency(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """p50 and p95 latency against the reference percentiles."""
    name = "latency"
    if reference.latency_p50 is None and reference.latency_p95 is None:
        return inactive_check(name, "no reference latency; add latency_ms to the reference")
    if _present(window.latency).empty:
        return inactive_check(name, "no latency logged; pass latency_ms= to log()")
    current_p50 = _stats.quantile(window.latency, 0.50)
    current_p95 = _stats.quantile(window.latency, 0.95)
    limit_p50 = _limit(reference.latency_p50, thresholds.latency_multiplier)
    limit_p95 = _limit(reference.latency_p95, thresholds.latency_multiplier)
    over_p95 = limit_p95 is not None and current_p95 is not None and current_p95 > limit_p95
    over_p50 = limit_p50 is not None and current_p50 is not None and current_p50 > limit_p50
    message = "p50 %s ms (reference %s), p95 %s ms (reference %s)" % (
        _ms(current_p50),
        _ms(reference.latency_p50),
        _ms(current_p95),
        _ms(reference.latency_p95),
    )
    details = {
        "p50": current_p50,
        "p95": current_p95,
        "reference_p50": reference.latency_p50,
        "reference_p95": reference.latency_p95,
        "limit_p50": limit_p50,
        "limit_p95": limit_p95,
    }
    if over_p95:
        return failed_check(name, current_p95, limit_p95, "p95 too slow: " + message, **details)
    if over_p50:
        return failed_check(name, current_p50, limit_p50, "p50 too slow: " + message, **details)
    value = current_p95 if current_p95 is not None else current_p50
    threshold = limit_p95 if limit_p95 is not None else limit_p50
    return ok_check(name, value, threshold, message, **details)


def null_rate(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """Share of feature values that arrived missing."""
    name = "null_rate"
    frame = window.feature_frame
    if frame.empty:
        return inactive_check(name, "no features logged; pass features= to log()")
    value = _stats.missing_share(frame)
    if value is None:
        return inactive_check(name, "no feature values to count")
    per_feature = {
        str(column): float(frame[column].isna().mean()) for column in frame.columns
    }
    worst = max(per_feature, key=lambda key: per_feature[key])
    message = "%.2f%% of feature values missing across %d feature%s (worst: %s at %.2f%%)" % (
        value * 100.0,
        frame.shape[1],
        "" if frame.shape[1] == 1 else "s",
        worst,
        per_feature[worst] * 100.0,
    )
    builder = ok_check if value <= thresholds.null_rate else failed_check
    return builder(name, value, thresholds.null_rate, message, per_feature=per_feature)


def constant_output(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """The classic silent failure: the same prediction, over and over.

    What counts as "over and over" depends on what the model normally outputs.
    A fraud or churn classifier that says 0 for 95% of requests repeats itself
    constantly by design - the longest run of zeros in a healthy 1000-request
    window is about 87 - so a fixed run limit would call every one of them
    stuck. When the reference says what the output mix looks like, the limit is
    the longest run that mix would produce by chance; the flat limit is the
    floor under it, and all there is to go on with no reference.
    """
    name = "constant_output"
    current = _present(window.predictions)
    if current.empty:
        return inactive_check(name, "no predictions logged")
    if window.n < thresholds.min_records:
        return inactive_check(
            name, "only %s; need %d" % (_plural(window.n, "record"), thresholds.min_records)
        )
    run, value_repeated = _stats.longest_run(current)
    flat = max(int(thresholds.min_records), min(int(thresholds.constant_run), window.n // 2))
    limit = float(flat)
    basis = "flat limit"
    share = None
    if reference.predictions is not None and len(reference.predictions) > 0:
        share = _stats.value_share(reference.predictions)
        expected = _stats.expected_longest_run(reference.predictions, len(current))
        if share is not None and share >= 1.0:
            return inactive_check(
                name,
                "the reference is one constant value, so a constant window is "
                "normal here; nothing to compare a run against",
                reference_share=share,
            )
        if expected is not None:
            if expected > limit:
                limit = float(expected)
                basis = "chance limit for a %.0f%% majority class" % (share * 100.0)
            if limit >= len(current):
                return inactive_check(
                    name,
                    "the reference repeats itself too much to judge this window: at a "
                    "%.0f%% majority class a run of %d is ordinary, so %s is not enough "
                    "to tell a stuck model from normal traffic"
                    % (share * 100.0, int(limit), _plural(len(current), "record")),
                    reference_share=share,
                    needed=int(limit) + 1,
                )
    message = "longest run of one value is %d of %d predictions (value: %s, %s %d)" % (
        run,
        len(current),
        value_repeated,
        basis,
        int(limit),
    )
    builder = ok_check if run < limit else failed_check
    if run >= limit:
        message = "model looks stuck: " + message
    return builder(
        name,
        float(run),
        float(limit),
        message,
        repeated_value=value_repeated,
        flat_limit=float(flat),
        reference_share=share,
    )


def volume(window: Window, reference: ReferenceProfile, thresholds: Thresholds) -> Check:
    """Traffic well above or below the rate the reference was collected at."""
    name = "volume"
    if reference.rate_per_hour is None or reference.rate_per_hour <= 0:
        return inactive_check(
            name, "no reference traffic rate; add a timestamp column to the reference"
        )
    if window.n < 2:
        return inactive_check(name, "need at least 2 records to measure a rate")
    current = _stats.rate_per_hour(window.timestamps)
    if current is None:
        return inactive_check(name, "the records span less than a minute; rate not meaningful")
    low = float(reference.rate_per_hour) / thresholds.volume_ratio
    high = float(reference.rate_per_hour) * thresholds.volume_ratio
    message = "%.1f records/hour against a reference of %.1f (normal band %.1f to %.1f)" % (
        current,
        reference.rate_per_hour,
        low,
        high,
    )
    details = {"low": low, "high": high, "reference_rate": reference.rate_per_hour}
    if current > high:
        return failed_check(name, current, high, "traffic spike: " + message, **details)
    if current < low:
        return failed_check(name, current, low, "traffic dropped: " + message, **details)
    return ok_check(name, current, high, message, **details)


def _plural(count: int, noun: str) -> str:
    """``3 records``, but ``1 record``."""
    return "%d %s%s" % (count, noun, "" if count == 1 else "s")


def _limit(reference_value: Optional[float], multiplier: float) -> Optional[float]:
    """Allowed ceiling for a latency percentile, with a 1 ms floor."""
    if reference_value is None:
        return None
    return max(float(reference_value) * float(multiplier), float(reference_value) + 1.0)


#: A perfect reference gets this much of the target's own magnitude as slack.
PERFECT_REFERENCE_TOLERANCE = 0.01

#: ...and this much when even the magnitude is zero, so exact stays exact.
_TINY_ERROR = 1e-12


def _error_limit(reference: ReferenceProfile, thresholds: Thresholds) -> float:
    """Allowed mean absolute error, with a floor for a flawless reference.

    ``reference.error * (1 + error_increase)`` is zero when the reference
    scored perfectly, and then any error at all - a prediction off by a
    thousandth - reads as a failure. The latency monitor guards the same case
    with its 1 ms floor; here there is no natural unit, so the floor is a small
    fraction of the magnitude of the values being predicted.
    """
    limit = float(reference.error or 0.0) * (1.0 + float(thresholds.error_increase))
    if limit > 0.0:
        return limit
    scale = _stats.typical_magnitude(reference.actuals)
    if scale is None:
        scale = _stats.typical_magnitude(reference.predictions)
    if scale:
        return PERFECT_REFERENCE_TOLERANCE * float(scale)
    return _TINY_ERROR


def _ms(value: Optional[float]) -> str:
    return "-" if value is None else format(float(value), ".1f")


#: Every monitor, in the order they appear in a report.
MONITORS: List[Callable[[Window, ReferenceProfile, Thresholds], Check]] = [
    prediction_drift,
    feature_drift,
    accuracy_drop,
    latency,
    null_rate,
    constant_output,
    volume,
]

#: The monitor names, in report order.
MONITOR_NAMES: List[str] = [
    "prediction_drift",
    "feature_drift",
    "accuracy_drop",
    "latency",
    "null_rate",
    "constant_output",
    "volume",
]


def run_all(
    window: Window, reference: ReferenceProfile, thresholds: Thresholds
) -> List[Check]:
    """Run every monitor. A monitor that explodes becomes an inactive check.

    The crash is recorded in the check's details as ``monitor_error``. One
    monitor tripping over an odd window is not a failure of the model, but a
    report in which *every* monitor crashed is not healthy either - it is
    blind - and :attr:`WatchReport.ok` reads that marker to say so.
    """
    checks: List[Check] = []
    for monitor in MONITORS:
        try:
            checks.append(monitor(window, reference, thresholds))
        except Exception as exc:  # noqa: BLE001 - one bad monitor must not hide the rest
            checks.append(
                inactive_check(
                    monitor.__name__,
                    "monitor raised %s: %s" % (type(exc).__name__, exc),
                    monitor_error="%s: %s" % (type(exc).__name__, exc),
                )
            )
    return checks
