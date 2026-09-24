"""Small numeric helpers shared by every measure.

Two ideas live here. ``dbfs`` never returns ``-inf``, so digital silence is a
large negative number rather than something that poisons every later average.
``piecewise`` turns a measured quantity into a 0-100 score by naming the points
the scale passes through, which keeps every scoring curve readable as data
instead of hidden in arithmetic.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .thresholds import DB_FLOOR

__all__ = [
    "amplitude_from_dbfs",
    "clamp",
    "dbfs",
    "dbfs_array",
    "piecewise",
    "round_or_none",
]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """``value`` pulled inside ``[low, high]``."""
    return float(min(max(float(value), low), high))


def dbfs(amplitude: float) -> float:
    """Amplitude relative to full scale, in dB, with a floor instead of ``-inf``.

    Args:
        amplitude: a non-negative linear amplitude where 1.0 is full scale.

    Returns:
        ``20 * log10(amplitude)``, never below :data:`DB_FLOOR`.
    """
    value = float(amplitude)
    if not math.isfinite(value) or value <= 0.0:
        return DB_FLOOR
    return max(DB_FLOOR, 20.0 * math.log10(value))


def dbfs_array(amplitudes: np.ndarray) -> np.ndarray:
    """:func:`dbfs` over an array, vectorised and still free of ``-inf``."""
    values = np.asarray(amplitudes, dtype=np.float64)
    safe = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 20.0 * np.log10(safe)
    return np.maximum(np.where(np.isfinite(out), out, DB_FLOOR), DB_FLOOR)


def amplitude_from_dbfs(level_db: float) -> float:
    """The linear amplitude a dBFS level stands for."""
    return float(10.0 ** (float(level_db) / 20.0))


def piecewise(value: float, points: Sequence[Tuple[float, float]]) -> float:
    """Score ``value`` on a linear scale through named ``(value, score)`` points.

    The points must be sorted by value. Anything below the first point takes the
    first score, anything above the last takes the last, and everything between
    is interpolated. Writing a curve this way means the thresholds that matter
    are visible at the call site.

    Args:
        value: the measurement to score.
        points: at least two ``(value, score)`` pairs, ascending by value.

    Returns:
        A score clamped to 0-100.

    Raises:
        ValueError: fewer than two points were given.
    """
    if len(points) < 2:
        raise ValueError("piecewise needs at least two points")
    number = float(value)
    if not math.isfinite(number):
        return clamp(points[0][1])
    if number <= points[0][0]:
        return clamp(points[0][1])
    for (low_x, low_y), (high_x, high_y) in zip(points, points[1:]):
        if number <= high_x:
            if high_x == low_x:
                return clamp(high_y)
            fraction = (number - low_x) / (high_x - low_x)
            return clamp(low_y + fraction * (high_y - low_y))
    return clamp(points[-1][1])


def round_or_none(value, digits: int = 4):
    """Round a float for JSON, passing ``None`` and non-finite values through."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)
