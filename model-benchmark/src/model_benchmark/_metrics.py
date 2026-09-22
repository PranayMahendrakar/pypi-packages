"""The built-in metrics.

Pure numpy. Nothing here imports scikit-learn, torch or tensorflow, and nothing
downloads a model. A metric is ``callable(y_true, y_pred) -> float`` plus a flag
saying whether a bigger number is better, which is what ``report.best("score")``
needs in order to pick a winner.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

__all__ = ["METRICS", "accuracy", "f1", "mae", "r2", "resolve_metric", "rmse"]

_NAN = float("nan")


def _unwrap(values: Any) -> Any:
    """pandas objects become numpy arrays; everything else is left alone."""
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        return to_numpy()
    return values


def _as_labels(y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Two flat arrays that can be compared element by element, whatever the dtype."""
    true = np.asarray(_unwrap(y_true)).ravel()
    pred = np.asarray(_unwrap(y_pred)).ravel()
    if true.dtype.kind != pred.dtype.kind:
        true = true.astype(object)
        pred = pred.astype(object)
    return true, pred


def _as_floats(y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Two flat float arrays, for the regression metrics."""
    true = np.asarray(_unwrap(y_true), dtype=float).ravel()
    pred = np.asarray(_unwrap(y_pred), dtype=float).ravel()
    return true, pred


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Share of predictions that match the truth exactly."""
    true, pred = _as_labels(y_true, y_pred)
    if true.size == 0:
        return _NAN
    return float(np.mean(true == pred))


def _binary_f1(true: np.ndarray, pred: np.ndarray, positive: Any) -> float:
    true_pos = float(np.sum((true == positive) & (pred == positive)))
    false_pos = float(np.sum((true != positive) & (pred == positive)))
    false_neg = float(np.sum((true == positive) & (pred != positive)))
    denominator = 2.0 * true_pos + false_pos + false_neg
    if denominator == 0.0:
        return 0.0
    return 2.0 * true_pos / denominator


def f1(y_true: Any, y_pred: Any) -> float:
    """F1 score: binary for two labels, macro averaged for more.

    The positive class is the last label in sorted order, so ``1`` of
    ``{0, 1}`` and ``"yes"`` of ``{"no", "yes"}``. When only one label is
    present and it is predicted correctly this returns ``1.0``, where
    scikit-learn's ``f1_score`` reports ``0.0`` for a ``pos_label`` that never
    appears; a benchmark cares that the model got every row right.
    """
    true, pred = _as_labels(y_true, y_pred)
    if true.size == 0:
        return _NAN
    labels = np.unique(np.concatenate([true, pred]))
    if labels.size == 0:
        return _NAN
    if labels.size <= 2:
        return _binary_f1(true, pred, labels[-1])
    return float(np.mean([_binary_f1(true, pred, label) for label in labels]))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error. Lower is better."""
    true, pred = _as_floats(y_true, y_pred)
    if true.size == 0:
        return _NAN
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error. Lower is better."""
    true, pred = _as_floats(y_true, y_pred)
    if true.size == 0:
        return _NAN
    return float(np.mean(np.abs(true - pred)))


def r2(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination, with the constant-truth case handled."""
    true, pred = _as_floats(y_true, y_pred)
    if true.size == 0:
        return _NAN
    residual = float(np.sum((true - pred) ** 2))
    total = float(np.sum((true - true.mean()) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)


# name -> (function, higher_is_better)
METRICS: Dict[str, Tuple[Callable[[Any, Any], float], bool]] = {
    "accuracy": (accuracy, True),
    "f1": (f1, True),
    "rmse": (rmse, False),
    "mae": (mae, False),
    "r2": (r2, True),
}


def resolve_metric(metric: Any) -> Tuple[Any, Any, bool]:
    """Turn `metric` into (name, function, higher_is_better).

    `metric` may be None, one of the names in :data:`METRICS`, or any callable
    taking ``(y_true, y_pred)``. A custom callable is assumed to be better when
    bigger unless it carries a ``higher_is_better`` attribute saying otherwise.
    """
    if metric is None:
        return None, None, True
    if isinstance(metric, str):
        key = metric.strip().lower()
        if key not in METRICS:
            known = ", ".join(sorted(METRICS))
            raise ValueError(f"unknown metric {metric!r}; use one of {known}, or pass a callable")
        function, higher = METRICS[key]
        return key, function, higher
    if callable(metric):
        name = getattr(metric, "__name__", None) or metric.__class__.__name__
        higher = bool(getattr(metric, "higher_is_better", True))
        return str(name), metric, higher
    raise TypeError(
        f"metric must be None, a metric name or a callable(y_true, y_pred), got {type(metric).__name__}"
    )
