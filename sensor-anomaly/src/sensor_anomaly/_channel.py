"""Per-channel detection: a robust z-score of the rolling residual.

Each channel is compared against its own recent level rather than against a global
mean, so a slow drift or a daily cycle does not drown the report in false alarms.
The score is the residual divided by a median-based spread, which a few genuine
spikes cannot inflate the way a standard deviation can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._faults import noise_sigma, robust_sigma, scan_faults, tolerance, FaultEvent


def to_float_array(series: pd.Series) -> Tuple[np.ndarray, bool]:
    """Return ``(float64 copy of the column, whether it held numbers)``.

    Missing values, infinities and anything unparseable all become NaN. The array
    is always a fresh copy, so nothing here can ever reach back into the caller's
    DataFrame.
    """
    numeric = True
    if pd.api.types.is_bool_dtype(series):
        values = series.astype("float64").to_numpy(copy=True)
    elif pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").astype("float64").to_numpy(copy=True)
    else:
        try:
            converted = pd.to_numeric(series, errors="coerce")
        except (TypeError, ValueError):
            return np.full(len(series), np.nan, dtype="float64"), False
        values = converted.astype("float64").to_numpy(copy=True)
        numeric = bool(np.isfinite(values).any())
    values[~np.isfinite(values)] = np.nan
    return values, numeric


def pick_window(n: int) -> int:
    """An odd rolling window: long enough to be stable, short enough to track drift."""
    window = int(max(5, min(101, n // 20)))
    if window % 2 == 0:
        window += 1
    return window


@dataclass
class ChannelScan:
    """Everything learned about one channel, before it becomes a report entry."""

    name: str
    values: np.ndarray
    valid: np.ndarray
    n_valid: int
    n_missing: int
    scores: np.ndarray
    scale: float
    window: int
    anomalies: np.ndarray
    score_max: Optional[float]
    faults: List[str] = field(default_factory=list)
    fault_events: List[FaultEvent] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    constant: bool = False
    numeric: bool = True
    #: Typical reading for this channel (median), used to attribute joint events.
    level: float = 0.0
    #: Robust spread of the raw readings, used to attribute joint events.
    spread: float = 0.0

    def deviation(self, start: int, end: int) -> float:
        """How far this channel strays from its own normal level over a span.

        In robust sigmas, so it is comparable between channels. Used to name the
        channels behind a cross-channel event.
        """
        if self.spread <= 0.0:
            return 0.0
        window = self.values[start : end + 1]
        window = window[np.isfinite(window)]
        if window.size == 0:
            return 0.0
        return float(np.max(np.abs(window - self.level)) / self.spread)

    @property
    def failed(self) -> bool:
        """True when the channel produced no usable readings."""
        return self.n_valid == 0

    @property
    def usable_for_joint(self) -> bool:
        """True when this channel can contribute to the cross-channel model."""
        return self.n_valid > 0 and not self.constant

    @property
    def rate(self) -> float:
        """Share of usable readings that were flagged, 0.0 to 1.0."""
        if self.n_valid == 0:
            return 0.0
        return float(self.anomalies.size) / float(self.n_valid)

    def fill_value(self) -> float:
        """The value used to impute this channel for the cross-channel model."""
        if self.n_valid == 0:
            return 0.0
        return float(np.median(self.values[self.valid]))

    def imputed(self) -> np.ndarray:
        """A copy of the channel with missing readings replaced by its median."""
        filled = self.values.copy()
        if self.n_missing:
            filled[~self.valid] = self.fill_value()
        return filled


def scan_channel(name: str, series: pd.Series, sensitivity: float) -> ChannelScan:
    """Score one channel and scan it for faults.

    Args:
        name: the column name, used only in messages.
        series: the raw column, any dtype.
        sensitivity: flag readings this many robust sigmas from the local level.

    Returns:
        A :class:`ChannelScan`. A channel with no usable data, or no variation at
        all, yields no anomalies rather than raising or flagging everything.
    """
    values, numeric = to_float_array(series)
    valid = np.isfinite(values)
    n_valid = int(valid.sum())
    n_missing = int(values.size - n_valid)
    scores = np.full(values.size, np.nan, dtype="float64")
    window = pick_window(values.size)

    notes: List[str] = []
    faults: List[str] = []
    constant = False
    scale = 0.0

    if n_valid == 0:
        faults = ["all-missing"] if numeric else ["no-numeric-data"]
        return ChannelScan(
            name=name,
            values=values,
            valid=valid,
            n_valid=0,
            n_missing=n_missing,
            scores=scores,
            scale=0.0,
            window=window,
            anomalies=np.empty(0, dtype=int),
            score_max=None,
            faults=faults,
            fault_events=[],
            notes=notes,
            constant=True,
            numeric=numeric,
        )

    finite = values[valid]
    constant = bool(float(np.max(finite) - np.min(finite)) <= tolerance(values))
    level = float(np.median(finite))
    spread = robust_sigma(values)

    if not constant and n_valid >= 3:
        column = pd.Series(values, dtype="float64")
        baseline = column.rolling(window, center=True, min_periods=1).median()
        residual = (column - baseline).to_numpy(dtype="float64")
        # Score against the channel's real noise level, not the narrower spread of
        # the residuals: the centred rolling median absorbs part of the noise, and
        # scoring against what is left makes sensitivity=3 behave like about 2.4.
        scale = max(robust_sigma(residual), noise_sigma(values))
        if scale > 0.0:
            centre = float(np.median(residual[np.isfinite(residual)]))
            with np.errstate(invalid="ignore"):
                scores = (residual - centre) / scale
            scores[~valid] = np.nan

    with np.errstate(invalid="ignore"):
        flagged = np.isfinite(scores) & (np.abs(scores) >= float(sensitivity))
    anomalies = np.flatnonzero(flagged)

    peak: Optional[float] = None
    if np.isfinite(scores).any():
        peak = float(np.nanmax(np.abs(scores)))
        if not np.isfinite(peak):
            peak = None

    faults, fault_events, fault_notes = scan_faults(values, n_valid, scale, sensitivity)
    notes.extend(fault_notes)
    if constant:
        notes.append(
            "never varied, so it cannot be scored statistically; reported as a fault instead"
        )
    elif scale <= 0.0:
        notes.append("too few distinct readings to score; only faults were checked")

    return ChannelScan(
        name=name,
        values=values,
        valid=valid,
        n_valid=n_valid,
        n_missing=n_missing,
        scores=scores,
        scale=scale,
        window=window,
        anomalies=anomalies,
        score_max=peak,
        faults=faults,
        fault_events=fault_events,
        notes=notes,
        constant=constant,
        numeric=numeric,
        level=level,
        spread=spread,
    )
