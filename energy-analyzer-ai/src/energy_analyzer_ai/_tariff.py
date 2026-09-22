"""Turning units into money, or leaving money out of it entirely.

``tariff=None`` is not a tariff of zero. Every cost on the report stays ``None``
so nobody can mistake "not priced" for "free".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class Tariff:
    """A flat price per unit, a price per hour of the day, or no price at all."""

    kind: str = "none"
    flat: Optional[float] = None
    hours: Optional[Dict[int, float]] = None
    mean_rate: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.kind != "none"

    def __str__(self) -> str:
        if self.kind == "none":
            return "no tariff given, so costs are not estimated"
        if self.kind == "flat":
            return f"flat rate of {self.flat:g} per unit"
        assert self.hours is not None
        low = min(self.hours.values())
        high = max(self.hours.values())
        return f"time-of-use rates from {low:g} to {high:g} per unit"

    def rates(
        self,
        index: Any,
        step: Optional[pd.Timedelta] = None,
        warnings: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """Price per unit for every period, or ``None`` when there is no tariff."""
        n = len(index)
        if self.kind == "none":
            return None
        if self.kind == "flat":
            return np.full(n, float(self.flat), dtype=float)
        assert self.hours is not None and self.mean_rate is not None
        usable_clock = isinstance(index, pd.DatetimeIndex) and (
            step is None or step < pd.Timedelta(days=1)
        )
        if not usable_clock:
            if warnings is not None:
                warnings.append(
                    "time-of-use rates need the hour of each reading; the average rate "
                    f"of {self.mean_rate:g} per unit was used instead"
                )
            return np.full(n, float(self.mean_rate), dtype=float)
        hours = pd.DatetimeIndex(index).hour.to_numpy()
        return np.array(
            [float(self.hours.get(int(hour), self.mean_rate)) for hour in hours], dtype=float
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the tariff."""
        return {
            "kind": self.kind,
            "flat": None if self.flat is None else float(self.flat),
            "hours": None if self.hours is None else {str(k): float(v) for k, v in sorted(self.hours.items())},
            "mean_rate": None if self.mean_rate is None else float(self.mean_rate),
            "text": str(self),
        }


def _as_hour(key: Any) -> int:
    """Read a dict key as an hour of the day."""
    if isinstance(key, bool):
        raise ValueError(f"tariff hour {key!r} is not a number from 0 to 23")
    try:
        hour = int(key)
        if float(key) != float(hour):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"tariff key {key!r} is not an hour; use whole numbers from 0 to 23, "
            "for example {0: 0.08, 7: 0.22}"
        ) from None
    if not 0 <= hour <= 23:
        raise ValueError(f"tariff hour {hour} is outside 0 to 23")
    return hour


def _as_rate(value: Any, where: str) -> float:
    """Read a tariff value as a price per unit."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"tariff rate for {where} is not a number: {value!r}") from None
    if not np.isfinite(rate):
        raise ValueError(f"tariff rate for {where} is not a finite number: {value!r}")
    return rate


def build(tariff: Any, warnings: Optional[List[str]] = None) -> Tariff:
    """Normalise whatever was passed as ``tariff=`` into a :class:`Tariff`.

    Accepts ``None``, a number (a flat price per unit), or a mapping of
    ``hour -> price`` for time-of-use pricing.
    """
    warnings = warnings if warnings is not None else []
    if tariff is None:
        return Tariff(kind="none")
    if isinstance(tariff, Tariff):
        return tariff
    if isinstance(tariff, pd.Series):
        tariff = tariff.to_dict()
    if isinstance(tariff, (dict, pd.DataFrame)):
        if isinstance(tariff, pd.DataFrame):
            raise TypeError(
                "tariff must be a number or a dict of hour -> rate, not a DataFrame; "
                "pass dict(zip(frame['hour'], frame['rate']))"
            )
        if not tariff:
            raise ValueError("tariff is an empty mapping; pass a number or hour -> rate pairs")
        hours = {_as_hour(key): _as_rate(value, f"hour {key}") for key, value in tariff.items()}
        mean_rate = float(np.mean(list(hours.values())))
        missing = [hour for hour in range(24) if hour not in hours]
        if missing:
            warnings.append(
                f"the tariff does not price {len(missing)} hour(s) of the day "
                f"({missing[0]:02d}:00 onwards); the average rate of {mean_rate:g} was used there"
            )
        if any(rate < 0 for rate in hours.values()):
            warnings.append("the tariff has negative rates; costs can come out negative")
        return Tariff(kind="time-of-use", hours=hours, mean_rate=mean_rate)
    if isinstance(tariff, bool):
        raise TypeError("tariff must be a number or a dict of hour -> rate, not a bool")
    if isinstance(tariff, (int, float, np.integer, np.floating)):
        rate = _as_rate(tariff, "the flat tariff")
        if rate < 0:
            warnings.append("the tariff is negative; costs will come out negative")
        return Tariff(kind="flat", flat=rate, mean_rate=rate)
    raise TypeError(
        f"tariff must be None, a number, or a dict of hour -> rate, not "
        f"{type(tariff).__name__}"
    )
