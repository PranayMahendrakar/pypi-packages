"""Result objects. Every one carries a .summary() and a JSON-safe .to_dict()."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

TREND_WORDS = ("improving", "stable", "degrading")
RISK_WORDS = ("low", "medium", "high")
CONFIDENCE_WORDS = ("low", "medium", "high")


def _clean_float(value: Any) -> Optional[float]:
    """A plain float, or None when the number is missing or not finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _stamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    number = _clean_float(value)
    return None if number is None else "{0:g}".format(number)


def display_width(text: str) -> int:
    """How many terminal columns ``text`` occupies.

    East-asian wide and fullwidth characters take two columns each, so counting
    code points (what ``"{:<24}".format`` does) leaves CJK channel names
    over-indented and the column after them ragged.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad_display(text: str, width: int) -> str:
    """``text`` padded (or truncated) to ``width`` terminal columns."""
    if display_width(text) > width:
        clipped = ""
        for ch in text:
            if display_width(clipped + ch) > width - 3:
                break
            clipped += ch
        text = clipped + "..."
    return text + " " * max(0, width - display_width(text))


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(100.0, score)) / 100.0 * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


@dataclass
class HealthResult:
    """How far equipment has drifted from its healthy baseline, 0 to 100."""

    score: float
    series: pd.Series
    trend: str
    contributors: Dict[str, float]
    channels: List[str]
    window: int
    baseline_rows: int
    baseline_score: float
    #: the score :func:`predictive_maintenance.estimate_rul` treats as end of
    #: useful life unless a ``threshold`` is passed.
    threshold_hint: float = 30.0
    notes: List[str] = field(default_factory=list)

    @property
    def degrading(self) -> bool:
        """True when the recent trend is upward."""
        return self.trend == "degrading"

    def top_contributors(self, limit: int = 3) -> List[str]:
        """The channels driving most of the degradation, worst first."""
        ranked = sorted(self.contributors.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, share in ranked[:limit] if share > 0.0]

    def summary(self) -> str:
        """Human-readable text. Plain ASCII, safe for any console."""
        lines = [
            "predictive-maintenance: health score {0:.1f}/100 {1} ({2})".format(
                self.score, _bar(self.score), self.trend
            ),
            "baseline score {0:.1f} over {1} rows | window {2} rows | {3} channel(s): {4}".format(
                self.baseline_score,
                self.baseline_rows,
                self.window,
                len(self.channels),
                ", ".join(self.channels),
            ),
            "",
            "  channel                    share of degradation",
        ]
        ranked = sorted(self.contributors.items(), key=lambda kv: kv[1], reverse=True)
        for name, share in ranked:
            lines.append("  {0}   {1:>6.1%}".format(pad_display(name, 24), share))
        if not ranked:
            lines.append("  (no channels)")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend("  - " + note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole result."""
        return {
            "score": _clean_float(self.score),
            "trend": self.trend,
            "degrading": self.degrading,
            "baseline_score": _clean_float(self.baseline_score),
            "baseline_rows": int(self.baseline_rows),
            "threshold_hint": _clean_float(self.threshold_hint),
            "window": int(self.window),
            "channels": list(self.channels),
            "contributors": {k: _clean_float(v) for k, v in self.contributors.items()},
            "top_contributors": self.top_contributors(),
            "n_rows": int(self.series.size),
            "first": _stamp(self.series.index[0]) if self.series.size else None,
            "last": _stamp(self.series.index[-1]) if self.series.size else None,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


@dataclass
class RULResult:
    """Remaining useful life: when the degradation trend reaches the threshold."""

    remaining: Union[pd.Timedelta, float]
    confidence: str
    eta: Optional[Any]
    score: float
    threshold: float
    trend: str
    slope_per_step: float
    r_squared: float
    is_temporal: bool
    notes: List[str] = field(default_factory=list)

    @property
    def infinite(self) -> bool:
        """True when the trend never reaches the threshold."""
        if isinstance(self.remaining, pd.Timedelta):
            return False
        return not np.isfinite(float(self.remaining))

    def remaining_text(self) -> str:
        """The remaining life as a short phrase."""
        if self.infinite:
            return "infinite (not trending toward the threshold)"
        if isinstance(self.remaining, pd.Timedelta):
            return str(self.remaining)
        return "{0:.1f} rows".format(float(self.remaining))

    def summary(self) -> str:
        """Human-readable text. Plain ASCII, safe for any console."""
        lines = [
            "predictive-maintenance: remaining useful life {0} (confidence: {1})".format(
                self.remaining_text(), self.confidence
            ),
            "health score {0:.1f}/100 now, threshold {1:.1f}, trend {2}".format(
                self.score, self.threshold, self.trend
            ),
        ]
        if self.eta is not None:
            lines.append("expected to cross the threshold at: {0}".format(_stamp(self.eta)))
        else:
            lines.append("no crossing expected on the current trend")
        lines.append(
            "trend fit: {0:+.4g} points per {1}, r-squared {2:.2f}".format(
                self.slope_per_step, "second" if self.is_temporal else "row", self.r_squared
            )
        )
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend("  - " + note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict. Infinite life is reported as None plus a flag."""
        seconds = None
        rows = None
        if not self.infinite:
            if isinstance(self.remaining, pd.Timedelta):
                seconds = _clean_float(self.remaining.total_seconds())
            else:
                rows = _clean_float(self.remaining)
        return {
            "remaining_text": self.remaining_text(),
            "remaining_seconds": seconds,
            "remaining_rows": rows,
            "infinite": bool(self.infinite),
            "confidence": self.confidence,
            "eta": _stamp(self.eta),
            "score": _clean_float(self.score),
            "threshold": _clean_float(self.threshold),
            "trend": self.trend,
            "slope_per_step": _clean_float(self.slope_per_step),
            "r_squared": _clean_float(self.r_squared),
            "time_based": bool(self.is_temporal),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


@dataclass
class FailureRisk:
    """Per-row probability that a failure lands inside the model's horizon."""

    probability: np.ndarray
    risk: str
    horizon: int
    index: pd.Index
    medium_threshold: float = 0.25
    high_threshold: float = 0.60
    notes: List[str] = field(default_factory=list)

    @property
    def probability_now(self) -> float:
        """The probability for the most recent row."""
        return float(self.probability[-1]) if self.probability.size else 0.0

    @property
    def peak_probability(self) -> float:
        """The highest probability anywhere in the scored rows."""
        return float(self.probability.max()) if self.probability.size else 0.0

    @property
    def peak_at(self) -> Optional[Any]:
        """Where that peak sits on the time axis."""
        if not self.probability.size:
            return None
        return self.index[int(np.argmax(self.probability))]

    def as_series(self) -> pd.Series:
        """The probabilities as a pandas Series on the original time axis."""
        return pd.Series(self.probability, index=self.index, name="failure_probability")

    def summary(self) -> str:
        """Human-readable text. Plain ASCII, safe for any console."""
        flagged = int(np.sum(self.probability >= self.high_threshold))
        lines = [
            "predictive-maintenance: failure risk {0} ({1:.1%} within {2} rows)".format(
                self.risk.upper(), self.probability_now, self.horizon
            ),
            "scored {0} row(s) | peak {1:.1%} at {2} | {3} row(s) above {4:.0%}".format(
                self.probability.size,
                self.peak_probability,
                _stamp(self.peak_at) or "n/a",
                flagged,
                self.high_threshold,
            ),
            "bands: low < {0:.0%} <= medium < {1:.0%} <= high".format(
                self.medium_threshold, self.high_threshold
            ),
        ]
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend("  - " + note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict. The full probability array is included as a list."""
        return {
            "risk": self.risk,
            "probability_now": _clean_float(self.probability_now),
            "peak_probability": _clean_float(self.peak_probability),
            "peak_at": _stamp(self.peak_at),
            "horizon": int(self.horizon),
            "n_rows": int(self.probability.size),
            "probability": [_clean_float(p) for p in self.probability.tolist()],
            "medium_threshold": _clean_float(self.medium_threshold),
            "high_threshold": _clean_float(self.high_threshold),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()
