"""The two things a caller touches: :func:`detect` and :class:`Detector`."""
from __future__ import annotations

import logging
import warnings as _pywarnings
from typing import Any, List, Optional

import numpy as np

from ._io import LoadedSeries, load
from ._methods import (
    METHODS,
    Baseline,
    applicable,
    fit as fit_method,
    infer_period,
    infer_period_from_values,
    resolve_method,
)
from ._result import AnomalyResult
from ._robust import confidence_from, sigmas

LOG = logging.getLogger(__name__)


def _validate(method: str, sensitivity: Any, seasonality: Any) -> None:
    """Reject option values that could only produce nonsense.

    Everything is checked here so that a wrong option is answered in this
    package's own words, rather than as whatever numpy or ``int()`` says several
    frames deeper.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {', '.join(METHODS)}")
    wrong_sensitivity = f"sensitivity must be a positive number of sigmas, got {sensitivity!r}"
    if isinstance(sensitivity, (str, bytes)) or sensitivity is None:
        raise ValueError(wrong_sensitivity)
    try:
        threshold = float(sensitivity)
    except (TypeError, ValueError):
        raise ValueError(wrong_sensitivity) from None
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(wrong_sensitivity)
    if seasonality is not None:
        wrong_season = (
            "seasonality must be an integer number of points per cycle, "
            f"got {seasonality!r}"
        )
        if isinstance(seasonality, (str, bytes)):
            raise ValueError(wrong_season)
        try:
            cycle = int(seasonality)
        except (TypeError, ValueError):
            raise ValueError(wrong_season) from None
        if cycle < 2:
            raise ValueError(
                f"seasonality must be at least 2 points per cycle, got {seasonality!r}"
            )


def _nan_median_rows(stack: List[np.ndarray]) -> np.ndarray:
    """Column-wise median across methods, quiet about all-missing columns."""
    matrix = np.vstack(stack)
    with _pywarnings.catch_warnings():
        _pywarnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(matrix, axis=0)


def _unique(messages: List[str]) -> List[str]:
    """The same warnings in the same order, but said only once each."""
    seen = set()
    out: List[str] = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            out.append(message)
    return out


def _tiny_result(
    loaded: LoadedSeries,
    method: str,
    sensitivity: float,
    warnings: List[str],
) -> AnomalyResult:
    """The honest answer for a series of zero or one point: nothing to compare."""
    n = loaded.n
    values = loaded.values.astype(float, copy=True)
    expected = np.where(np.isfinite(values), values, 0.0)
    if n == 0:
        warnings.append("the series is empty, so there is nothing to detect")
    else:
        warnings.append("a single point has nothing to be compared against; no anomalies reported")
    return AnomalyResult(
        values=values,
        expected=expected,
        scores=np.zeros(n, dtype=float),
        mask=np.zeros(n, dtype=bool),
        confidence=np.zeros(n, dtype=float),
        time=loaded.time,
        order=loaded.order,
        method=method,
        methods_used=[],
        sensitivity=float(sensitivity),
        scale=None,
        scale_kind="none",
        baseline_level=float(expected[0]) if n else None,
        seasonality=None,
        window=None,
        label=loaded.label,
        time_label=loaded.time_label,
        index=loaded.index,
        warnings=_unique(warnings),
    )


def _finalise(
    loaded: LoadedSeries,
    ordered_expected: np.ndarray,
    ordered_scores: np.ndarray,
    *,
    method: str,
    methods_used: List[str],
    sensitivity: float,
    scale: Optional[float],
    scale_kind: str,
    level: Optional[float],
    seasonality: Optional[int],
    window: Optional[int],
    warnings: List[str],
    ordered_mask: Optional[np.ndarray] = None,
) -> AnomalyResult:
    """Threshold the scores and put everything back into input order."""
    ordered_values = loaded.in_time_order
    confidence = confidence_from(ordered_scores, sensitivity)
    if ordered_mask is None:
        ordered_mask = np.isfinite(ordered_scores) & (ordered_scores > float(sensitivity))
    ordered_mask = ordered_mask & np.isfinite(ordered_values)
    return AnomalyResult(
        values=loaded.values.astype(float, copy=True),
        expected=loaded.restore(np.asarray(ordered_expected, dtype=float)),
        scores=loaded.restore(np.asarray(ordered_scores, dtype=float)),
        mask=loaded.restore(np.asarray(ordered_mask, dtype=bool)),
        confidence=loaded.restore(np.asarray(confidence, dtype=float)),
        time=loaded.time,
        order=loaded.order,
        method=method,
        methods_used=list(methods_used),
        sensitivity=float(sensitivity),
        scale=scale,
        scale_kind=scale_kind,
        baseline_level=level,
        seasonality=seasonality,
        window=window,
        label=loaded.label,
        time_label=loaded.time_label,
        index=loaded.index,
        warnings=_unique(warnings),
    )


def _vote(
    loaded: LoadedSeries,
    ordered: np.ndarray,
    period: Optional[int],
    *,
    sensitivity: float,
    warnings: List[str],
) -> AnomalyResult:
    """Run every applicable method and flag a point when a strict majority agree."""
    names = applicable(loaded.n, period)
    baselines = [
        fit_method(name, ordered, seasonality=period, sensitivity=sensitivity)
        for name in names
    ]
    votes = np.zeros(loaded.n, dtype=float)
    score_stack: List[np.ndarray] = []
    expected_stack: List[np.ndarray] = []
    scales: List[float] = []
    levels: List[float] = []
    for baseline in baselines:
        with np.errstate(over="ignore", invalid="ignore"):
            score = sigmas(ordered - baseline.expected, baseline.scale)
        votes += (np.isfinite(score) & (score > float(sensitivity))).astype(float)
        score_stack.append(score)
        expected_stack.append(np.asarray(baseline.expected, dtype=float))
        if baseline.scale is not None:
            scales.append(float(baseline.scale))
        levels.append(float(baseline.level))
        warnings.extend(baseline.warnings)
    scores = _nan_median_rows(score_stack)
    expected = _nan_median_rows(expected_stack)
    mask = votes * 2.0 > float(len(baselines))
    warnings.append(
        f"voted across {len(baselines)} method(s): {', '.join(names)}; "
        f"a point is an anomaly when more than half of them agree"
    )
    # a vote can only ever be as informative as its electorate: on a steadily
    # climbing series the two flat-baseline methods have nothing to say and
    # outnumber the two that do, so say out loud that something was overruled
    # rather than returning an empty answer as if nothing had been seen
    overruled = int(np.count_nonzero((votes > 0.0) & ~mask))
    if overruled:
        warnings.append(
            f"{overruled} point(s) were flagged by some of the {len(baselines)} method(s) "
            "but not by a strict majority, so they are not reported here; run one method "
            "on its own to see them"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        middle_scale = float(np.median(scales)) if scales else None
        middle_level = float(np.median(levels)) if levels else None
    return _finalise(
        loaded,
        expected,
        scores,
        method="all",
        methods_used=names,
        sensitivity=sensitivity,
        scale=middle_scale,
        scale_kind="vote" if scales else "none",
        level=middle_level,
        seasonality=period if "seasonal" in names else None,
        window=None,
        warnings=warnings,
        ordered_mask=mask,
    )


def detect(
    data: Any,
    *,
    value: Optional[str] = None,
    time: Optional[str] = None,
    method: str = "auto",
    sensitivity: float = 3.0,
    seasonality: Optional[int] = None,
) -> AnomalyResult:
    """Find the points in a time series that do not belong.

    Args:
        data: a pandas Series (optionally with a DatetimeIndex), a DataFrame, a
            list or array of numbers, or a path to a ``.csv`` / ``.parquet`` file.
        value: name of the column holding the measurement, when `data` is a table.
        time: name of the column holding the timestamps, when `data` is a table.
        method: ``"auto"`` (rolling for 30+ points, otherwise zscore), ``"zscore"``,
            ``"iqr"``, ``"rolling"``, ``"seasonal"``, ``"ewma"``, or ``"all"`` to run
            every applicable method and flag what a strict majority agree on.
        sensitivity: the threshold in robust sigmas. Higher means fewer anomalies.
        seasonality: points per cycle for ``"seasonal"``; inferred from a DatetimeIndex
            when omitted.

    Returns:
        An :class:`AnomalyResult`. Call ``.summary()`` to read it, ``.anomalies``
        for the positional indices, ``.to_frame()`` for a table.
    """
    _validate(method, sensitivity, seasonality)
    loaded = load(data, value, time)
    warnings: List[str] = list(loaded.warnings)
    n = loaded.n
    if n < 2:
        return _tiny_result(loaded, method, sensitivity, warnings)

    ordered = loaded.in_time_order
    period: Optional[int] = int(seasonality) if seasonality else None
    if period is None and method in ("seasonal", "all"):
        period = infer_period(loaded.time, n)
        if period:
            warnings.append(f"inferred a season of {period} points from the timestamps")
        else:
            period = infer_period_from_values(ordered)
            if period:
                warnings.append(
                    f"inferred a season of {period} points from the shape of the series"
                )
            elif method == "seasonal":
                warnings.append(
                    "no seasonality given and none could be inferred from the input"
                )

    resolved = resolve_method(method, n)
    if method == "auto":
        reason = "30 or more points" if n >= 30 else f"only {n} points"
        warnings.append(f"method 'auto' chose {resolved!r} because the series has {reason}")

    if resolved == "all":
        return _vote(
            loaded,
            ordered,
            period,
            sensitivity=sensitivity,
            warnings=warnings,
        )

    baseline = fit_method(resolved, ordered, seasonality=period, sensitivity=sensitivity)
    warnings.extend(baseline.warnings)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = sigmas(ordered - baseline.expected, baseline.scale)
    return _finalise(
        loaded,
        baseline.expected,
        scores,
        method=method,
        methods_used=[baseline.name],
        sensitivity=sensitivity,
        scale=baseline.scale,
        scale_kind=baseline.scale_kind,
        level=baseline.level,
        seasonality=baseline.seasonality,
        window=baseline.window,
        warnings=warnings,
    )


class Detector:
    """The same detection, with the baseline learned once and reused.

    Use it when the data arrives in batches: :meth:`fit` on a stretch of normal
    history, then :meth:`score` each new batch against that frozen baseline and
    the same threshold. :meth:`detect` is the one-shot form, identical to the
    module-level :func:`detect`.

    Example::

        detector = Detector(method="rolling", sensitivity=4.0).fit(history)
        result = detector.score(todays_readings)
    """

    def __init__(
        self,
        *,
        value: Optional[str] = None,
        time: Optional[str] = None,
        method: str = "auto",
        sensitivity: float = 3.0,
        seasonality: Optional[int] = None,
    ) -> None:
        _validate(method, sensitivity, seasonality)
        self.value = value
        self.time = time
        self.method = method
        self.sensitivity = float(sensitivity)
        self.seasonality = int(seasonality) if seasonality else None
        self._baseline: Optional[Baseline] = None
        self._warnings: List[str] = []
        self._n_reference = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted else "not fitted"
        return (
            f"Detector(method={self.method!r}, sensitivity={self.sensitivity}, {state})"
        )

    # ----------------------------------------------------------------- state
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run."""
        return self._baseline is not None

    @property
    def baseline_(self) -> Baseline:
        """The fitted baseline; raises when :meth:`fit` has not run."""
        if self._baseline is None:
            raise ValueError("this Detector is not fitted yet; call fit(reference) first")
        return self._baseline

    @property
    def method_(self) -> str:
        """The concrete method that was fitted, after any fallback."""
        return self.baseline_.name

    @property
    def level_(self) -> float:
        """The frozen baseline level new points are compared against."""
        return float(self.baseline_.level)

    @property
    def scale_(self) -> Optional[float]:
        """The robust sigma learned from the reference, or ``None`` if flat."""
        return self.baseline_.scale

    @property
    def scale_kind_(self) -> str:
        """Which spread estimate produced :attr:`scale_`."""
        return self.baseline_.scale_kind

    @property
    def n_reference_(self) -> int:
        """How many reference points the baseline was learned from."""
        return self._n_reference

    @property
    def warnings_(self) -> List[str]:
        """What had to be adjusted while fitting."""
        return list(self._warnings)

    # ----------------------------------------------------------------- verbs
    def detect(self, data: Any) -> AnomalyResult:
        """One-shot detection over `data`, using this detector's options."""
        return detect(
            data,
            value=self.value,
            time=self.time,
            method=self.method,
            sensitivity=self.sensitivity,
            seasonality=self.seasonality,
        )

    def fit(self, reference: Any) -> "Detector":
        """Learn the baseline and the spread from a stretch of reference data."""
        loaded = load(reference, self.value, self.time)
        if loaded.n == 0:
            raise ValueError("cannot fit on an empty reference series")
        warnings: List[str] = list(loaded.warnings)
        period = self.seasonality
        if period is None and self.method in ("seasonal", "all"):
            period = infer_period(loaded.time, loaded.n)
            if period is None:
                period = infer_period_from_values(loaded.in_time_order)
            if period:
                warnings.append(
                    f"inferred a season of {period} points from the reference data"
                )
        resolved = resolve_method(self.method, loaded.n)
        if resolved == "all":
            resolved = resolve_method("auto", loaded.n)
            warnings.append(
                "method 'all' votes across methods and only works on a whole series; "
                f"the streaming baseline was fitted with {resolved!r} instead"
            )
        elif self.method == "auto":
            warnings.append(f"method 'auto' chose {resolved!r} for {loaded.n} reference points")
        baseline = fit_method(
            resolved,
            loaded.in_time_order,
            seasonality=period,
            sensitivity=self.sensitivity,
        )
        warnings.extend(baseline.warnings)
        self._baseline = baseline
        self._warnings = warnings
        self._n_reference = loaded.n
        return self

    def score(self, new_values: Any) -> AnomalyResult:
        """Score a later batch against the baseline learned by :meth:`fit`."""
        baseline = self.baseline_
        loaded = load(new_values, self.value, self.time)
        warnings: List[str] = list(self._warnings) + list(loaded.warnings)
        n = loaded.n
        if n == 0:
            return _tiny_result(loaded, self.method, self.sensitivity, warnings)
        warnings.append(
            f"scored against a baseline frozen on {self._n_reference:,} reference point(s) "
            f"using {baseline.name}"
        )
        if baseline.seasonal is not None and baseline.seasonality:
            phase = (baseline.next_phase + np.arange(n)) % baseline.seasonality
            expected = float(baseline.level) + baseline.seasonal[phase]
        else:
            expected = np.full(n, float(baseline.level), dtype=float)
        ordered = loaded.in_time_order
        with np.errstate(over="ignore", invalid="ignore"):
            scores = sigmas(ordered - expected, baseline.scale)
        return _finalise(
            loaded,
            expected,
            scores,
            method=self.method,
            methods_used=[baseline.name],
            sensitivity=self.sensitivity,
            scale=baseline.scale,
            scale_kind=baseline.scale_kind,
            level=float(baseline.level),
            seasonality=baseline.seasonality,
            window=baseline.window,
            warnings=warnings,
        )
