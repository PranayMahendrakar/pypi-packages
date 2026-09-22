"""Robust spread estimates that degrade gracefully instead of dividing by zero.

Every detector in this package turns a residual into a number of "robust sigmas".
The scale used for that conversion comes from here, and it is always either a
finite positive float or ``None``, which means "this series carries no usable
variation, so nothing can be called an anomaly".
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: median absolute deviation -> standard-deviation-equivalent for a normal sample
MAD_TO_SIGMA = 1.4826
#: interquartile range -> standard-deviation-equivalent for a normal sample
IQR_TO_SIGMA = 1.3490


def finite(values: np.ndarray) -> np.ndarray:
    """The finite (non-NaN, non-inf) entries of `values`."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def median(values: np.ndarray, default: float = 0.0) -> float:
    """Median of the finite entries; `default` when there are none."""
    kept = finite(values)
    if kept.size == 0:
        return default
    # an even-length median is the mean of the two middle readings, and on a
    # series near the top of the float range that sum leaves it. The scale
    # ladder below already answers a baseline it cannot use, so the arithmetic
    # is left to produce its infinity quietly rather than complaining past the
    # caller's own warning handler.
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.median(kept))


def _mad(values: np.ndarray) -> float:
    """Raw median absolute deviation of already-finite entries."""
    if values.size == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        centre = float(np.median(values))
        return float(np.median(np.abs(values - centre)))


def mad_sigma(values: np.ndarray) -> float:
    """Robust sigma from the median absolute deviation; 0.0 when undefined."""
    kept = finite(values)
    if kept.size == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scale = MAD_TO_SIGMA * _mad(kept)
    return scale if np.isfinite(scale) and scale > 0.0 else 0.0


def iqr_sigma(values: np.ndarray) -> float:
    """Robust sigma from the interquartile range; 0.0 when undefined."""
    kept = finite(values)
    if kept.size < 2:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        q1, q3 = np.percentile(kept, [25.0, 75.0])
        scale = float(q3 - q1) / IQR_TO_SIGMA
    return scale if np.isfinite(scale) and scale > 0.0 else 0.0


def std_sigma(values: np.ndarray) -> float:
    """Plain standard deviation of the finite entries; 0.0 when undefined."""
    kept = finite(values)
    if kept.size < 2:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scale = float(np.std(kept, ddof=0))
    return scale if np.isfinite(scale) and scale > 0.0 else 0.0


#: ratio of step-to-step scatter to overall spread expected of independent noise
SQRT_2 = 1.4142135623730951


def roughness(residuals: np.ndarray) -> float:
    """How much of a residual's spread is scatter rather than a slow drift.

    Independent noise moves by about ``sqrt(2)`` times its own spread from one
    point to the next, so this ratio sits near ``1.0``. A residual left over from
    a baseline that is systematically wrong - the warm-up curve of an exponential
    average trailing a ramp, say - drifts smoothly instead, and the ratio falls
    towards zero. That spread is the shape of the fit, not a sigma to divide by.

    Returns ``1.0`` (i.e. "looks like noise, no objection") when there is too
    little to judge.
    """
    kept = finite(residuals)
    if kept.size < 4:
        return 1.0
    spread = _mad(kept)
    if not np.isfinite(spread) or spread <= 0.0:
        return 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = _mad(np.diff(kept)) / (SQRT_2 * spread)
    return float(ratio) if np.isfinite(ratio) else 1.0


#: a spread this far below the data it describes is rounding error, not variation
RELATIVE_FLOOR = 1e-13


def noise_floor(values: np.ndarray) -> float:
    """The spread below which an estimate is rounding error, not variation.

    A robust estimate can collapse to a few times machine epsilon while the data
    it describes still varies: it happens whenever more than half the residuals
    land on exactly the same number, which a clean seasonal, quantised or
    perfectly-fitted signal does routinely. Dividing by that leftover noise turns
    every ordinary point into a trillion sigmas, so an estimate at or below this
    floor is rejected and the next rung of the ladder is tried instead.

    The yardstick is the data itself, not the residuals, because when the fit is
    near-perfect the residuals are made of nothing but rounding error and cannot
    measure themselves. Ten orders of magnitude below the signal leaves genuine
    but tiny variation well clear of the line.
    """
    kept = finite(values)
    if kept.size == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        largest = float(np.max(np.abs(kept)))
        if not np.isfinite(largest) or largest <= 0.0:
            return 0.0
        floor = largest * RELATIVE_FLOOR
    return floor if np.isfinite(floor) else 0.0


def choose_scale(
    residuals: np.ndarray,
    values: np.ndarray,
    *,
    prefer: str = "mad",
) -> Tuple[Optional[float], str]:
    """Pick the scale used to turn residuals into robust sigmas.

    The ladder is: the preferred robust estimate, then the standard deviation,
    then ``None``. Each rung must clear :func:`noise_floor`, so an estimate that
    has collapsed to rounding error counts as no estimate at all. ``None`` is
    returned for a constant (or near-empty) series and means no point can be
    flagged, which is how divide-by-zero is avoided.

    Returns ``(scale, kind)`` where `kind` names the estimate that was used, one
    of ``"mad"``, ``"iqr"``, ``"std"`` or ``"none"``.
    """
    floor = noise_floor(values)
    if prefer == "iqr":
        scale = iqr_sigma(residuals)
        if scale > floor:
            return scale, "iqr"
    scale = mad_sigma(residuals)
    if scale > floor:
        return scale, "mad"
    scale = std_sigma(residuals)
    if scale > floor:
        return scale, "std"
    return None, "none"


def sigmas(residuals: np.ndarray, scale: Optional[float]) -> np.ndarray:
    """Absolute residuals expressed in robust sigmas; all zeros when `scale` is None."""
    residuals = np.asarray(residuals, dtype=float)
    if scale is None or not np.isfinite(scale) or scale <= 0.0:
        out = np.zeros(residuals.shape, dtype=float)
        out[~np.isfinite(residuals)] = np.nan
        return out
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.abs(residuals) / float(scale)


def confidence_from(scores: np.ndarray, threshold: float) -> np.ndarray:
    """How far past the threshold a point sits, mapped into ``[0, 1)``.

    ``1 - threshold / score`` for scores above the threshold and 0 below it, so a
    point at twice the threshold has confidence 0.5 and one at four times has 0.75.
    """
    scores = np.asarray(scores, dtype=float)
    out = np.zeros(scores.shape, dtype=float)
    if threshold <= 0.0:
        return out
    above = np.isfinite(scores) & (scores > threshold)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out[above] = 1.0 - (threshold / scores[above])
    return np.clip(out, 0.0, 1.0)


def as_float_array(values: Sequence) -> np.ndarray:
    """A 1-D float array, raising a clear error for anything that is not numeric."""
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(
            f"expected a one-dimensional series of numbers, got an array with shape {array.shape}"
        )
    try:
        return array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "the series contains values that are not numbers; "
            "pass a numeric column or convert it first"
        ) from exc
