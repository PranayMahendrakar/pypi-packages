"""Rolling window statistics over one sensor channel.

Every statistic is computed on full windows with
:func:`numpy.lib.stride_tricks.sliding_window_view`, then back-filled so the
result lines up row for row with the input. Constant channels yield zeros
throughout rather than NaN, so a sensor that never moves contributes nothing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TINY = 1e-12
_FFT_CHUNK_CELLS = 2_000_000

STAT_NAMES = ("mean", "std", "min", "max", "slope", "energy")

# A channel held at a constant value through the baseline gives no noise estimate
# at all. Rather than divide by TINY - which turns last-digit float wobble into a
# catastrophic deviation - the scale is floored at a fraction of the level the
# channel is held at, so "one unit" means a real move relative to that level.
FLAT_LEVEL_FRACTION = 1e-3


def choose_window(n_rows: int, requested: "int | None" = None) -> int:
    """Pick a window length: the caller's, or about a tenth of the history."""
    if requested is not None:
        try:
            window = int(requested)
        except (TypeError, ValueError):
            raise ValueError(
                "window must be a whole number of rows, got {0!r}".format(requested)
            ) from None
        if window < 2:
            raise ValueError("window must be at least 2 rows, got {0}".format(requested))
    else:
        window = max(3, min(50, n_rows // 10))
    return int(max(2, min(window, max(n_rows, 2))))


def _spectral(views: np.ndarray, window: int) -> "tuple[np.ndarray, np.ndarray]":
    """High-frequency energy of each window, absolute and as a share of the total.

    The windows are mean-removed first, so the DC bin carries nothing and a flat
    channel comes out as exactly zero on both measures.
    """
    n_windows = views.shape[0]
    n_bins = window // 2 + 1
    cut = max(1, (n_bins + 1) // 2)
    energy = np.zeros(n_windows, dtype=float)
    share = np.zeros(n_windows, dtype=float)
    step = max(1, _FFT_CHUNK_CELLS // max(window, 1))
    for start in range(0, n_windows, step):
        chunk = np.asarray(views[start : start + step], dtype=float)
        chunk = chunk - chunk.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(chunk, axis=1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2) / float(window * window)
        total = power[:, 1:].sum(axis=1)
        high = power[:, cut:].sum(axis=1)
        energy[start : start + chunk.shape[0]] = high
        share[start : start + chunk.shape[0]] = np.where(total > TINY, high / (total + TINY), 0.0)
    return energy, share


def _backfill(values: np.ndarray, n_rows: int) -> np.ndarray:
    """Stretch a full-window result back over the leading partial windows."""
    pad = n_rows - values.size
    if pad <= 0:
        return values
    head = np.full(pad, values[0] if values.size else 0.0, dtype=float)
    return np.concatenate([head, values])


def window_stats(x: np.ndarray, window: int) -> Dict[str, np.ndarray]:
    """Rolling mean, std, min, max, slope, spectral energy and HF share.

    Each array has the same length as ``x``. ``slope`` is the least-squares
    gradient across the window in units per row; ``energy`` is the
    high-frequency band power; ``hf_share`` is that band as a fraction of the
    window's total alternating power.
    """
    values = np.asarray(x, dtype=float)
    n_rows = values.size
    names = STAT_NAMES + ("hf_share",)
    if n_rows == 0:
        return {name: np.zeros(0, dtype=float) for name in names}
    if n_rows == 1:
        flat = {name: np.zeros(1, dtype=float) for name in names}
        for name in ("mean", "min", "max"):
            flat[name] = values.astype(float).copy()
        return flat
    width = int(max(2, min(window, n_rows)))
    views = sliding_window_view(values, width)

    centred = np.arange(width, dtype=float) - (width - 1) / 2.0
    denominator = float(centred @ centred) or TINY
    energy, share = _spectral(views, width)

    stats = {
        "mean": views.mean(axis=1),
        "std": views.std(axis=1),
        "min": views.min(axis=1),
        "max": views.max(axis=1),
        "slope": np.asarray(views @ centred, dtype=float) / denominator,
        "energy": energy,
        "hf_share": share,
    }
    return {name: _backfill(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), n_rows)
            for name, arr in stats.items()}


def robust_scale(values: np.ndarray) -> float:
    """A positive spread estimate: standard deviation, floored away from zero."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return TINY
    spread = float(np.std(finite))
    if spread > TINY:
        return spread
    return TINY


def baseline_scale(baseline_values: np.ndarray, all_values: np.ndarray) -> "tuple[float, bool]":
    """The scale one deviation unit is measured in, and whether the baseline was flat.

    Normally this is the standard deviation of the channel during the healthy
    period. When that period never moves at all there is no noise to measure, so
    the scale falls back to a small fraction of the level the channel is held at
    (and, for a channel held at exactly zero, to the spread of the whole series).
    Without that floor a controller-held setpoint would divide by ~0 and read
    float dust as a destroyed bearing.
    """
    spread = robust_scale(baseline_values)
    if spread > TINY:
        return spread, False
    reference = np.asarray(baseline_values, dtype=float)
    reference = reference[np.isfinite(reference)]
    magnitude = float(np.median(np.abs(reference))) if reference.size else 0.0
    floored = FLAT_LEVEL_FRACTION * magnitude
    if floored > TINY:
        return floored, True
    whole = robust_scale(all_values)
    return (whole if whole > TINY else TINY), True


def linear_fit(x: np.ndarray, y: np.ndarray) -> "tuple[float, float, float]":
    """Least-squares ``slope, intercept, r_squared`` with no NaN escaping."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if xs.size < 2:
        return 0.0, float(ys[0]) if ys.size else 0.0, 0.0
    x_centred = xs - xs.mean()
    y_centred = ys - ys.mean()
    denominator = float(x_centred @ x_centred)
    if denominator <= TINY:
        return 0.0, float(ys.mean()), 0.0
    slope = float(x_centred @ y_centred) / denominator
    intercept = float(ys.mean() - slope * xs.mean())
    total = float(y_centred @ y_centred)
    if total <= TINY:
        return slope, intercept, 1.0
    residual = ys - (slope * xs + intercept)
    r_squared = 1.0 - float(residual @ residual) / total
    return slope, intercept, float(min(max(r_squared, 0.0), 1.0))
