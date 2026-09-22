"""The two metric sets: pass/fail scores for classification, error scores for regression."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
REGRESSION_METRICS = ("r2", "mae", "rmse")


def _float(value: object) -> Optional[float]:
    """A plain JSON-safe float, or ``None`` when the score could not be computed."""
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else number


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: Optional[np.ndarray],
    classes: List[object],
    pos_label: object = None,
) -> Dict[str, Optional[float]]:
    """Accuracy, precision, recall, f1 and (where computable) ROC AUC.

    With two classes, precision/recall/f1 describe ``pos_label`` (the good
    outcome). With more than two they are macro-averaged so every class counts
    equally, which is what you want when the defect class is the rare one. They
    report 0 rather than warning when a class never appears in ``y_pred``.
    """
    if pos_label is None or not any(pos_label == c for c in classes):
        pos_label = classes[-1] if classes else None
    average = "binary" if len(classes) == 2 else "macro"
    kwargs: Dict[str, object] = {"zero_division": 0, "average": average}
    if average == "binary":
        kwargs["pos_label"] = pos_label
    scores: Dict[str, Optional[float]] = {
        "accuracy": _float(accuracy_score(y_true, y_pred)),
        "precision": _float(precision_score(y_true, y_pred, **kwargs)),
        "recall": _float(recall_score(y_true, y_pred, **kwargs)),
        "f1": _float(f1_score(y_true, y_pred, **kwargs)),
        "roc_auc": _roc_auc(y_true, proba, classes, pos_label),
    }
    return scores


def _roc_auc(
    y_true: np.ndarray,
    proba: Optional[np.ndarray],
    classes: List[object],
    pos_label: object = None,
) -> Optional[float]:
    """ROC AUC, or ``None`` when the held-out rows cannot support one."""
    if proba is None or proba.ndim != 2 or proba.shape[1] != len(classes):
        return None
    present = set(np.asarray(y_true).tolist())
    if len(present) < 2:
        # one class in the holdout: an AUC here would be undefined, not zero
        return None
    try:
        if len(classes) == 2:
            index = max(i for i, c in enumerate(classes) if c == pos_label)
            positive = np.asarray([1 if v == classes[index] else 0 for v in y_true])
            return _float(roc_auc_score(positive, proba[:, index]))
        if present != set(classes):
            return None
        return _float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
        )
    except (ValueError, IndexError):  # pragma: no cover - a score never crashes a fit
        return None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """R2, mean absolute error and root mean squared error."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": _float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "mae": _float(mean_absolute_error(y_true, y_pred)),
        "rmse": _float(np.sqrt(mse)),
    }


def format_metrics(metrics: Dict[str, Optional[float]]) -> str:
    """``"accuracy 0.950, precision 0.941, ..."`` with unavailable scores named as such."""
    parts = []
    for name, value in metrics.items():
        parts.append(f"{name} n/a" if value is None else f"{name} {value:.3f}")
    return ", ".join(parts)
