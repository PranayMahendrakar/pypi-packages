"""Partial dependence curves, and the operating windows they point at.

A partial dependence curve answers "if every part had been made at this
temperature, how would the average outcome have moved?". The window is the
stretch of that curve that stays close to its best point.

A curve is only worth a window when it actually moves. Gradient boosting still
splits on a parameter that drives nothing, so such a curve wobbles by a hair;
scaling the keep band to that wobble would collapse the window onto one grid
point and invent a setting nobody should hold. Every spread is therefore judged
against an absolute yardstick passed in as ``floor``, never against the curve's
own noise.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

#: How many settings of a parameter to try when drawing its curve.
GRID_POINTS = 20

#: A grid point counts as part of the good window while it stays within this
#: fraction of the curve's full spread below the best point.
KEEP_FRACTION = 0.25

#: A curve must move at least this fraction of the spread of the model's own
#: predictions before the window means anything. Measured on real drivers and on
#: pure-noise columns, the two sit far apart: a driver's curve moves 0.7 to 3
#: times the prediction spread, a noise column's moves under 0.06 of it.
MIN_SPREAD_FRACTION = 0.20

#: A parameter carrying less than this share of the model's decisions gets no
#: window either, however its curve happens to wobble.
MIN_IMPORTANCE = 0.01

#: Below this a spread is floating-point dust, whatever the yardstick says.
FLAT_SPREAD = 1e-12

ScoreFn = Callable[[pd.DataFrame], np.ndarray]


def build_grid(values: pd.Series, n_points: int = GRID_POINTS) -> np.ndarray:
    """The settings to try for one numeric parameter, spread over its observed range."""
    numbers = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        return np.empty(0, dtype="float64")
    unique = np.unique(numbers)
    if unique.size <= n_points:
        return unique
    # quantiles rather than a linear sweep, so a long tail does not spend the
    # whole grid on settings the plant has never actually run
    quantiles = np.linspace(0.02, 0.98, n_points)
    return np.unique(np.quantile(numbers, quantiles))


def dependence_curve(
    score: ScoreFn, frame: pd.DataFrame, feature: str, grid: np.ndarray
) -> np.ndarray:
    """Average score across ``frame`` with ``feature`` pinned to each grid value."""
    curve = np.empty(grid.size, dtype="float64")
    work = frame.copy()
    original = work[feature].copy()
    for index, value in enumerate(grid):
        work[feature] = value
        curve[index] = float(np.mean(score(work)))
    work[feature] = original
    return curve


def curve_spread(curve: np.ndarray) -> float:
    """How far the curve moves from its worst point to its best."""
    if curve.size == 0:
        return 0.0
    finite = curve[np.isfinite(curve)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite) - np.min(finite))


def is_flat(curve: np.ndarray, floor: float = 0.0) -> bool:
    """True when the curve moves too little for a window to mean anything.

    ``floor`` is the absolute yardstick: the smallest movement that counts as
    real, normally a fraction of the spread of the model's own predictions.
    """
    try:
        limit = float(floor)
    except (TypeError, ValueError):
        limit = 0.0
    if not np.isfinite(limit) or limit < 0.0:
        limit = 0.0
    return curve_spread(curve) <= max(limit, FLAT_SPREAD)


def window(
    grid: np.ndarray,
    curve: np.ndarray,
    keep: float = KEEP_FRACTION,
    floor: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """The contiguous ``(low, high)`` stretch of ``grid`` around the curve's best point.

    A curve that is flat against ``floor`` gets the whole observed range rather
    than a window scaled to its own noise; callers are expected to say so.
    """
    if grid.size == 0:
        return None
    if grid.size == 1:
        return float(grid[0]), float(grid[0])
    if is_flat(curve, floor):
        # this parameter moves nothing worth acting on: the whole observed range
        # is equally fine, and pretending otherwise would invent a setpoint
        return float(grid[0]), float(grid[-1])
    best = float(np.max(curve))
    threshold = best - keep * curve_spread(curve)
    peak = int(np.argmax(curve))
    low = peak
    while low > 0 and curve[low - 1] >= threshold:
        low -= 1
    high = peak
    while high < grid.size - 1 and curve[high + 1] >= threshold:
        high += 1
    return float(grid[low]), float(grid[high])
