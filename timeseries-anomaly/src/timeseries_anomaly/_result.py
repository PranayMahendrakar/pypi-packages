"""The object ``detect()`` hands back, which is meant to explain itself."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _number(value: Any) -> Optional[float]:
    """A JSON-safe float: NaN and infinities become ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _stamp(value: Any) -> Any:
    """A JSON-safe timestamp: ISO text for dates, a float for numeric axes."""
    if value is None:
        return None
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - exotic objects
        pass
    return str(value)


def _short(value: Any) -> str:
    """Compact fixed text for a number, for the human summary."""
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"


@dataclass
class AnomalyResult:
    """Which points look wrong, what they should have been, and how sure we are.

    ``mask``, ``scores``, ``expected`` and ``values`` are all aligned to the input
    order, even when the series had to be sorted by timestamp first.
    """

    values: np.ndarray
    expected: np.ndarray
    scores: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    time: Optional[np.ndarray] = None
    order: Optional[np.ndarray] = None
    method: str = "auto"
    methods_used: List[str] = field(default_factory=list)
    sensitivity: float = 3.0
    scale: Optional[float] = None
    scale_kind: str = "none"
    baseline_level: Optional[float] = None
    seasonality: Optional[int] = None
    window: Optional[int] = None
    label: str = "value"
    time_label: Optional[str] = None
    index: Optional[pd.Index] = None
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def anomalies(self) -> List[int]:
        """Positional indices of the flagged points, in input order."""
        return [int(i) for i in np.flatnonzero(self.mask)]

    @property
    def method_used(self) -> str:
        """The concrete method that produced these scores, after any fallback.

        ``"auto"`` reports the method it picked, a method that had to fall back
        reports the one that actually ran, and a majority vote across several
        methods reports ``"all"``.
        """
        if not self.methods_used:
            return self.method
        if len(self.methods_used) == 1:
            return self.methods_used[0]
        return "all"

    @property
    def n_points(self) -> int:
        """How many points were looked at, missing values included."""
        return int(self.values.size)

    @property
    def n_valid(self) -> int:
        """How many points carried a usable number."""
        return int(np.count_nonzero(np.isfinite(self.values)))

    @property
    def n_missing(self) -> int:
        """How many points were missing and therefore never flagged."""
        return self.n_points - self.n_valid

    @property
    def n_anomalies(self) -> int:
        """How many points were flagged."""
        return int(np.count_nonzero(self.mask))

    @property
    def rate(self) -> float:
        """Share of all input points that were flagged, in ``[0, 1]``."""
        if self.n_points == 0:
            return 0.0
        return self.n_anomalies / self.n_points

    def __len__(self) -> int:
        return self.n_points

    def __bool__(self) -> bool:
        return self.n_anomalies > 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AnomalyResult(method={self.method!r}, n_points={self.n_points}, "
            f"n_anomalies={self.n_anomalies}, sensitivity={self.sensitivity})"
        )

    # ------------------------------------------------------------------ output
    def time_at(self, index: int) -> Any:
        """The timestamp of one point, or its position when there is no time axis."""
        if self.time is None:
            return int(index)
        return self.time[int(index)]

    def to_frame(self) -> pd.DataFrame:
        """A tidy table: ``time, value, expected, score, is_anomaly``.

        One row per input point, in the order the caller passed them, and with
        the caller's own index when the input carried one.
        """
        if self.time is None:
            stamps: Any = np.arange(self.n_points)
        else:
            stamps = self.time
        return pd.DataFrame(
            {
                "time": stamps,
                "value": self.values,
                "expected": self.expected,
                "score": self.scores,
                "is_anomaly": self.mask,
            },
            index=self.index,
        )

    def top(self, limit: int = 5) -> List[Dict[str, Any]]:
        """The worst points first, whether or not they cleared the threshold."""
        scores = np.where(np.isfinite(self.scores), self.scores, -np.inf)
        ranked = np.argsort(scores, kind="stable")[::-1]
        out: List[Dict[str, Any]] = []
        for position in ranked[: max(0, int(limit))]:
            index = int(position)
            if not np.isfinite(self.scores[index]):
                continue
            out.append(
                {
                    "index": index,
                    "time": _stamp(self.time_at(index)),
                    "value": _number(self.values[index]),
                    "expected": _number(self.expected[index]),
                    "score": _number(self.scores[index]),
                    "confidence": _number(self.confidence[index]),
                    "is_anomaly": bool(self.mask[index]),
                }
            )
        return out

    def plot_data(self) -> Dict[str, Any]:
        """Plain lists in time order, ready to hand to any plotting library.

        ``plot_data()["anomaly_index"]`` indexes into these time-ordered lists,
        which is not the same as :attr:`anomalies` when the input was unsorted.
        """
        order = self.order if self.order is not None else np.arange(self.n_points)
        order = np.asarray(order, dtype=int)
        mask = self.mask[order]
        if self.time is None:
            stamps = [int(i) for i in order]
        else:
            stamps = [_stamp(value) for value in np.asarray(self.time)[order]]
        return {
            "time": stamps,
            "value": [_number(v) for v in self.values[order]],
            "expected": [_number(v) for v in self.expected[order]],
            "score": [_number(v) for v in self.scores[order]],
            "is_anomaly": [bool(flag) for flag in mask],
            "anomaly_index": [int(i) for i in np.flatnonzero(mask)],
            "threshold": float(self.sensitivity),
            "label": self.label,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Everything worth keeping, JSON-safe."""
        return {
            "label": self.label,
            "time_label": self.time_label,
            "method": self.method,
            "method_used": self.method_used,
            "methods_used": list(self.methods_used),
            "sensitivity": float(self.sensitivity),
            "seasonality": int(self.seasonality) if self.seasonality else None,
            "window": int(self.window) if self.window else None,
            "scale": _number(self.scale),
            "scale_kind": self.scale_kind,
            "baseline_level": _number(self.baseline_level),
            "n_points": self.n_points,
            "n_valid": self.n_valid,
            "n_missing": self.n_missing,
            "n_anomalies": self.n_anomalies,
            "rate": round(self.rate, 6),
            "anomalies": [
                {
                    "index": index,
                    "time": _stamp(self.time_at(index)),
                    "value": _number(self.values[index]),
                    "expected": _number(self.expected[index]),
                    "score": _number(self.scores[index]),
                    "confidence": _number(self.confidence[index]),
                }
                for index in self.anomalies
            ],
            "warnings": list(self.warnings),
            "summary": self.summary(),
        }

    def summary(self, limit: int = 5) -> str:
        """A short plain-text report, ASCII only so any console can print it."""
        word = "anomaly" if self.n_anomalies == 1 else "anomalies"
        lines = [
            f"timeseries-anomaly: {self.n_anomalies} {word} in {self.n_points:,} points "
            f"({self.rate:.1%})"
        ]
        series = f"  series    : {self.label}"
        if self.time_label:
            series += f" over {self.time_label}"
        if self.n_missing:
            series += f" ({self.n_missing:,} missing value(s) ignored)"
        lines.append(series)
        method = f"  method    : {', '.join(self.methods_used) or self.method}"
        if (
            self.methods_used
            and self.method in ("auto", "all")
            and self.methods_used != [self.method]
        ):
            method += f" (asked for {self.method!r})"
        if self.seasonality:
            method += f", season of {self.seasonality} points"
        lines.append(method)
        if self.scale is None:
            lines.append("  baseline  : flat, no usable variation, so nothing can be an anomaly")
        else:
            level = "" if self.baseline_level is None else f"level {_short(self.baseline_level)}, "
            lines.append(
                f"  baseline  : {level}robust sigma {_short(self.scale)} ({self.scale_kind})"
            )
        threshold = f"  threshold : {self.sensitivity:g} robust sigmas from expected"
        lines.append(threshold)
        flagged = self.anomalies[: max(0, int(limit))]
        if flagged:
            lines.append("  flagged   :")
            for index in flagged:
                when = _stamp(self.time_at(index))
                where = f"index {index}" if self.time is None else f"{when} (index {index})"
                lines.append(
                    f"    {where}: value {_short(self.values[index])}, "
                    f"expected {_short(self.expected[index])}, "
                    f"{_short(self.scores[index])} sigmas"
                )
            if self.n_anomalies > len(flagged):
                lines.append(f"    ... and {self.n_anomalies - len(flagged):,} more")
        elif self.n_points:
            worst = self.top(1)
            if worst and worst[0]["score"] is not None:
                lines.append(
                    f"  closest   : index {worst[0]['index']} at "
                    f"{_short(worst[0]['score'])} sigmas, below the threshold"
                )
        if self.warnings:
            lines.append("  warnings:")
            for note in self.warnings:
                lines.append(f"    - {note}")
        return "\n".join(lines)
