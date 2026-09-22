"""Supervised failure prediction from labelled failure events."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ._features import STAT_NAMES, choose_window, window_stats
from ._frame import Prepared, prepare
from ._io import TableLike
from ._results import FailureRisk, pad_display

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 10
MEDIUM_RISK = 0.25
HIGH_RISK = 0.60

# Gradient boosting costs roughly one pass per row per feature per tree, so a
# long history with many channels takes minutes rather than seconds. Past this
# many cells the caller is told, so a slow fit never looks like a hung one.
LARGE_FIT_CELLS = 1_000_000

_SORTED_LABELS_NOTE = (
    "the rows were not in time order, so the labels were carried through the sort "
    "with them and still mark the same readings"
)


def _as_labels(labels: Any, n_rows: int) -> np.ndarray:
    """Coerce the label argument to a boolean array of the right length."""
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError(
                "labels must be one column of failure flags, got a DataFrame with "
                "{0} columns".format(labels.shape[1])
            )
        labels = labels.iloc[:, 0]
    if isinstance(labels, pd.Series):
        values = labels.to_numpy()
    else:
        values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional, got shape {0}".format(values.shape))
    if values.size != n_rows:
        raise ValueError(
            "labels has {0} entries but the data has {1} rows; they must line up "
            "row for row".format(values.size, n_rows)
        )
    if values.dtype == object:
        values = pd.Series(values).fillna(False).to_numpy()
    try:
        flags = np.asarray(values).astype(bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "labels must be boolean or 0/1 failure flags; {0}".format(exc)
        ) from exc
    return flags


def _horizon_target(flags: np.ndarray, horizon: int) -> np.ndarray:
    """True where a failure is labelled at this row or within the next ``horizon``."""
    reversed_flags = pd.Series(flags[::-1].astype(float))
    rolled = reversed_flags.rolling(horizon + 1, min_periods=1).max()
    return np.asarray(rolled.to_numpy()[::-1] > 0.0, dtype=bool)


def _feature_matrix(
    data: Prepared, window: int
) -> Tuple[np.ndarray, List[str]]:
    """Windowed features for every channel, stacked column-wise."""
    columns: List[np.ndarray] = []
    names: List[str] = []
    for channel in data.channels:
        stats = window_stats(data.values[channel], window)
        for stat in STAT_NAMES:
            columns.append(stats[stat])
            names.append("{0}__{1}".format(channel, stat))
    matrix = np.column_stack(columns) if columns else np.zeros((data.n_rows, 0))
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def _risk_band(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "high"
    if probability >= MEDIUM_RISK:
        return "medium"
    return "low"


class MaintenanceModel:
    """Learn which sensor patterns come before a labelled failure.

    Give it sensor history plus a boolean Series marking the failures, and it
    learns to flag the rows that sit within ``horizon`` rows of the next one.

        model = MaintenanceModel(random_state=0).fit(df, failures, horizon=10)
        risk = model.predict(df)

    ``labels`` is read row for row against ``df`` as the caller passes it, and
    travels with its rows if the frame has to be sorted into time order.

    Boosting costs a pass per row, per feature, per tree, so a six-figure
    history takes minutes rather than seconds; ``fit`` records a note when it is
    about to. Lower ``n_estimators`` or ``max_depth``, or pass fewer channels,
    when that matters more than the last point of accuracy.

    Args:
        random_state: seed for the gradient boosting fit, so results repeat.
        window: rows per feature window; follows the data size when omitted.
        n_estimators: boosting rounds. The main cost knob.
        max_depth: depth of each tree.
        learning_rate: boosting learning rate.
    """

    def __init__(
        self,
        random_state: int = 0,
        *,
        window: Optional[int] = None,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ) -> None:
        self.random_state = int(random_state)
        self.window = window
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.horizon: int = DEFAULT_HORIZON
        self.channels: List[str] = []
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.fitted_window: int = 0
        self.notes: List[str] = []
        self._estimator: Optional[GradientBoostingClassifier] = None

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._estimator is not None

    def _require_fitted(self) -> GradientBoostingClassifier:
        if self._estimator is None:
            raise ValueError(
                "this MaintenanceModel is not fitted yet; call fit(df, labels) first"
            )
        return self._estimator

    def fit(
        self,
        df: TableLike,
        labels: Any,
        *,
        horizon: int = DEFAULT_HORIZON,
        time: Optional[str] = None,
        channels: Optional[Sequence[str]] = None,
    ) -> "MaintenanceModel":
        """Train on ``df`` using ``labels`` to mark the failure rows.

        Args:
            df: sensor history as a DataFrame or a path to .csv / .parquet.
            labels: boolean Series or array, True on the rows where a failure happened.
            horizon: how many rows ahead of a failure count as "about to fail".
            time: the timestamp column; auto-detected when omitted.
            channels: the sensor columns; every numeric column when omitted.

        Returns:
            self, so you can chain ``.fit(...).predict(...)``.
        """
        span = int(horizon)
        if span < 1:
            raise ValueError("horizon must be at least 1 row, got {0}".format(horizon))

        data = prepare(df, time=time, channels=channels, min_rows=3)
        # prepare() may have sorted the rows into time order. A label belongs to
        # its own reading, so it has to travel through exactly the same sort or
        # the model learns the opposite of the truth.
        flags = data.align(_as_labels(labels, data.n_rows)).astype(bool)

        if not bool(flags.any()):
            raise ValueError(
                "labels marks no failures at all, so there is nothing to learn from. "
                "Pass a boolean Series with True on the rows where the equipment failed, "
                "or use health_score()/estimate_rul(), which need no labels."
            )

        self.notes = list(data.notes)
        if data.was_sorted:
            self.notes.append(_SORTED_LABELS_NOTE)
        target = _horizon_target(flags, span)
        if bool(target.all()):
            raise ValueError(
                "every row counts as a failure at horizon={0} ({1} of {2} rows labelled), "
                "so there is no healthy behaviour to contrast against. Use a shorter "
                "horizon or label only the failure rows.".format(
                    span, int(flags.sum()), data.n_rows
                )
            )

        self.horizon = span
        self.channels = list(data.channels)
        self.fitted_window = choose_window(data.n_rows, self.window)
        matrix, names = _feature_matrix(data, self.fitted_window)
        self.feature_names = names

        cells = int(matrix.shape[0]) * int(matrix.shape[1]) * self.n_estimators
        if cells > LARGE_FIT_CELLS:
            self.notes.append(
                "fitting {0} rows x {1} features with {2} trees; on a history this "
                "size the fit takes minutes, not seconds. Lower n_estimators or "
                "max_depth, or pass fewer channels, if that is too slow".format(
                    matrix.shape[0], matrix.shape[1], self.n_estimators
                )
            )
            logger.info(
                "fitting %d rows x %d features with %d trees; this may take minutes",
                matrix.shape[0], matrix.shape[1], self.n_estimators,
            )

        estimator = GradientBoostingClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )
        estimator.fit(matrix, target.astype(int))
        self._estimator = estimator

        importances = np.asarray(estimator.feature_importances_, dtype=float)
        ranked = sorted(zip(names, importances), key=lambda kv: kv[1], reverse=True)
        self.feature_importance = {name: float(value) for name, value in ranked}

        positives = int(target.sum())
        self.notes.append(
            "trained on {0} rows, {1} of them within {2} rows of a failure".format(
                data.n_rows, positives, span
            )
        )
        if positives < 5:
            self.notes.append(
                "only {0} positive row(s) to learn from; the model will be crude".format(positives)
            )
            logger.warning("only %d positive rows available for training", positives)
        return self

    def _score(self, df: TableLike, time: Optional[str], label: str) -> Tuple[np.ndarray, Prepared]:
        estimator = self._require_fitted()
        data = prepare(df, time=time, channels=self.channels, label=label, min_rows=1)
        if data.n_rows < self.fitted_window:
            data.notes.append(
                "{0} has {1} rows, fewer than the {2}-row window the model was fitted "
                "with, so every row is scored from the same single window".format(
                    label, data.n_rows, self.fitted_window
                )
            )
        matrix, _ = _feature_matrix(data, self.fitted_window)
        proba = estimator.predict_proba(matrix)
        if proba.shape[1] == 1:  # pragma: no cover - guarded at fit time
            column = np.zeros(proba.shape[0]) if estimator.classes_[0] == 0 else np.ones(proba.shape[0])
        else:
            column = proba[:, 1]
        return np.asarray(column, dtype=float), data

    def predict(
        self, df: TableLike, *, time: Optional[str] = None
    ) -> FailureRisk:
        """Score each row of ``df`` for failure within the fitted horizon.

        Args:
            df: sensor history with the same channels the model was fitted on.
            time: the timestamp column; auto-detected when omitted.

        Returns:
            A :class:`FailureRisk` with ``.probability``, ``.risk`` and ``.summary()``.
        """
        probability, data = self._score(df, time, "df")
        notes = list(data.notes)
        return FailureRisk(
            probability=probability,
            risk=_risk_band(float(probability[-1]) if probability.size else 0.0),
            horizon=self.horizon,
            index=data.index,
            medium_threshold=MEDIUM_RISK,
            high_threshold=HIGH_RISK,
            notes=notes,
        )

    def evaluate(
        self, df: TableLike, labels: Any, *, time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Score the model against known labels on held-out history.

        Uses the same horizon the model was fitted with. Any metric that cannot
        be computed (for example ROC AUC when the labels hold a single class)
        comes back as ``None`` with the reason in ``notes``.

        Returns:
            A JSON-safe dict of metrics.
        """
        probability, data = self._score(df, time, "df")
        # Same sort, same alignment as fit(): metrics computed against labels
        # left behind in the caller's row order would be quietly meaningless.
        flags = data.align(_as_labels(labels, data.n_rows)).astype(bool)
        target = _horizon_target(flags, self.horizon)
        predicted = probability >= 0.5
        notes: List[str] = list(data.notes)
        if data.was_sorted:
            notes.append(_SORTED_LABELS_NOTE)

        both_classes = bool(target.any()) and not bool(target.all())
        if both_classes:
            roc = float(roc_auc_score(target.astype(int), probability))
            ap = float(average_precision_score(target.astype(int), probability))
        else:
            roc = None
            ap = None
            notes.append(
                "the labels hold a single class at horizon={0}, so ranking metrics "
                "(roc_auc, average_precision) cannot be computed".format(self.horizon)
            )

        precision, recall, f1, _ = precision_recall_fscore_support(
            target.astype(int), predicted.astype(int), average="binary", zero_division=0
        )
        return {
            "n_rows": int(data.n_rows),
            "n_positive": int(target.sum()),
            "positive_rate": float(target.mean()) if target.size else 0.0,
            "horizon": int(self.horizon),
            "roc_auc": roc,
            "average_precision": ap,
            "accuracy": float(accuracy_score(target.astype(int), predicted.astype(int))),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "decision_threshold": 0.5,
            "notes": notes,
        }

    def top_features(self, limit: int = 5) -> List[str]:
        """The most informative features, strongest first."""
        return [name for name, value in list(self.feature_importance.items())[:limit] if value > 0]

    def summary(self) -> str:
        """Human-readable description of the fitted model. Plain ASCII."""
        if not self.is_fitted:
            return "predictive-maintenance: MaintenanceModel (not fitted)"
        lines = [
            "predictive-maintenance: MaintenanceModel fitted on {0} channel(s), "
            "horizon {1} rows, window {2} rows".format(
                len(self.channels), self.horizon, self.fitted_window
            ),
            "",
            "  feature                              importance",
        ]
        for name, value in list(self.feature_importance.items())[:8]:
            lines.append("  {0}   {1:>8.1%}".format(pad_display(name, 34), value))
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend("  - " + note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe description of the fitted model."""
        return {
            "fitted": self.is_fitted,
            "horizon": int(self.horizon),
            "window": int(self.fitted_window),
            "channels": list(self.channels),
            "random_state": int(self.random_state),
            "feature_importance": {k: float(v) for k, v in self.feature_importance.items()},
            "top_features": self.top_features(),
            "notes": list(self.notes),
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        state = "fitted" if self.is_fitted else "unfitted"
        return "MaintenanceModel(random_state={0}, {1})".format(self.random_state, state)
