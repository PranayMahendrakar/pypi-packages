"""The objects a run hands back: a per-row explanation and a whole-run result.

Both are plain dataclasses with a ``summary()`` you can print and a
``to_dict()`` that is safe to hand to :func:`json.dumps`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._metrics import format_metrics


def jsonable(value: Any) -> Any:
    """``value`` as something :func:`json.dumps` accepts, recursing into containers."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def fmt(value: Any) -> str:
    """A number rounded for reading; anything else as text."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        return f"{number:.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def label(value: Any) -> str:
    """A class label as it should read in human text.

    Numbers drop the wrapper their repr would otherwise show, so a 0/1 pass/fail
    target reads as ``0``, ``1`` and not as ``np.int64(0)``. Text keeps its
    quotes, so a label with a comma or a trailing space is still unambiguous.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer, float, np.floating)):
        return fmt(value)
    if isinstance(value, str):
        return repr(value)
    converted = jsonable(value)
    if isinstance(converted, str):
        return repr(converted)
    return fmt(converted)


@dataclass
class Explanation:
    """Why one row got the prediction it did.

    ``contributions`` holds a signed effect per original feature: how far this
    row's value for that feature moved the score away from what a typical row
    scores. Positive pushed the outcome toward good, negative away from it.

    Each effect is measured one feature at a time against a background sample of
    the training rows, so the ranking and the directions are dependable; the
    effects are not forced to add up exactly to ``score - baseline``.
    """

    prediction: Any
    contributions: Dict[str, float]
    score: float
    baseline: float
    task: str
    target: str
    good_class: Any = None
    values: Dict[str, Any] = field(default_factory=dict)

    @property
    def ranked(self) -> List[Tuple[str, float]]:
        """``(feature, effect)`` pairs ordered by how much they moved the score."""
        return sorted(self.contributions.items(), key=lambda kv: (-abs(kv[1]), str(kv[0])))

    def top(self, n: int = 5) -> List[Tuple[str, float]]:
        """The ``n`` features that moved this prediction the most."""
        return self.ranked[: max(int(n), 0)]

    def summary(self, top: int = 5) -> str:
        """Human text: the prediction, then the features that drove it."""
        lines = [f"quality-predictor: {self.target} predicted as {fmt(self.prediction)}"]
        if self.task == "classification":
            lines.append(
                f"score {self.score:.3f} = P({self.target} = {label(self.good_class)}); "
                f"a typical row scores {self.baseline:.3f}"
            )
        else:
            lines.append(
                f"score {fmt(self.score)}; a typical row scores {fmt(self.baseline)}"
            )
        ranked = self.top(top)
        if not ranked:
            lines.append("No feature moved the score measurably for this row.")
            return "\n".join(lines)
        lines.append("")
        lines.append("WHAT DROVE IT (signed effect on the score)")
        width = max(len(str(name)) for name, _ in ranked)
        for name, effect in ranked:
            if effect > 0:
                direction = "toward good"
            elif effect < 0:
                direction = "away from good"
            else:
                direction = "neutral"
            value = self.values.get(name, "?")
            lines.append(
                (
                    f"  {str(name).ljust(width)}  {effect:+.4f}  {direction:<15}"
                    f" (value: {fmt(value)})"
                ).rstrip()
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole explanation."""
        return {
            "target": self.target,
            "task": self.task,
            "prediction": jsonable(self.prediction),
            "score": jsonable(self.score),
            "baseline": jsonable(self.baseline),
            "good_class": jsonable(self.good_class),
            "contributions": {str(k): jsonable(v) for k, v in self.contributions.items()},
            "ranked": [[str(k), jsonable(v)] for k, v in self.ranked],
            "values": {str(k): jsonable(v) for k, v in self.values.items()},
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


@dataclass
class QualityResult:
    """Everything one :func:`predict_quality` call worked out.

    ``model`` is the fitted :class:`~quality_predictor.QualityModel`, so anything
    the convenience function did not surface is one attribute away.
    """

    model: Any
    target: str
    task: str
    metrics: Dict[str, Optional[float]]
    feature_importance: pd.Series
    features: List[str]
    n_rows: int
    n_train: int
    n_test: int
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    classes: Optional[List[Any]] = None
    good_class: Any = None
    optimal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    explain_row: Optional[pd.DataFrame] = None

    @property
    def _uninformative(self) -> frozenset:
        """Parameters whose window is the whole observed range, for want of a signal."""
        names = getattr(self.model, "uninformative_parameters", None) or ()
        return frozenset(str(name) for name in names)

    @property
    def _good_class_guessed(self) -> bool:
        """True when no class name read as pass or fail and the fallback chose."""
        return bool(getattr(self.model, "good_class_guessed", False))

    @property
    def _dropped_without_outcome(self) -> int:
        """How many input rows had no outcome and so were left out of the model."""
        try:
            return int(getattr(self.model, "n_dropped_no_outcome", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    @property
    def top_factors(self) -> pd.Series:
        """The five parameters that matter most, importance descending."""
        return self.feature_importance.head(5)

    def explain(self, row: Any = None) -> Explanation:
        """Explain one row.

        With no argument this explains the first row that was predicted, or, when
        nothing new was predicted, a reference row kept from the training data.
        """
        if row is None:
            row = self.explain_row
        return self.model.explain(row)

    def summary(self, top: int = 8) -> str:
        """Human text: what was fitted, how honestly it scored, and what drives it."""
        headline = "pass/fail" if self.task == "classification" else "measured value"
        lines = [
            f"quality-predictor: {self.target} ({self.task}, {headline}) from "
            f"{len(self.features)} parameter(s)",
            f"rows: {self.n_rows} | trained on: {self.n_train} | held out: {self.n_test}",
            f"held-out scores: {format_metrics(self.metrics)}",
        ]
        if self.task == "classification" and self.classes is not None:
            lines.append(
                f"classes: {', '.join(label(c) for c in self.classes)}"
                f" | good outcome: {label(self.good_class)}"
                " (precision, recall and f1 describe this class)"
            )
        importance = self.feature_importance.head(max(int(top), 0))
        if len(importance):
            lines.append("")
            lines.append("WHAT DRIVES QUALITY (share of the model's decisions)")
            width = max(len(str(name)) for name in importance.index)
            for name, value in importance.items():
                bar = "#" * max(int(round(float(value) * 30)), 0)
                lines.append(
                    f"  {str(name).ljust(width)}  {float(value):6.1%}  {bar}".rstrip()
                )
        if self.optimal_ranges:
            lines.append("")
            heading = "SETTINGS MOST ASSOCIATED WITH GOOD OUTCOMES"
            if self._good_class_guessed:
                heading += (
                    " (reading {0} as the good outcome;"
                    " pass good_class= if that is the wrong way round)".format(
                        label(self.good_class)
                    )
                )
            lines.append(heading)
            flat = self._uninformative
            width = max(len(str(name)) for name in self.optimal_ranges)
            for name, bounds in self.optimal_ranges.items():
                low, high = bounds
                tail = ""
                if str(name) in flat:
                    tail = "   (whole observed range: moves {0} too little to call)".format(
                        self.target
                    )
                lines.append(
                    f"  {str(name).ljust(width)}  {fmt(low)} to {fmt(high)}{tail}"
                )
        if self.predictions is not None and len(self.predictions):
            preview = ", ".join(fmt(p) for p in list(self.predictions[:5]))
            more = ", ..." if len(self.predictions) > 5 else ""
            count = len(self.predictions)
            covers = ""
            dropped = self._dropped_without_outcome
            if dropped and count == self.n_rows + dropped:
                # the row count above only counts the modelled rows, so say where
                # the extra predictions came from rather than leave two numbers
                # that read like an arithmetic error
                covers = (
                    ", the {0} above plus the {1} with no recorded {2}".format(
                        self.n_rows, dropped, self.target
                    )
                )
            lines.append("")
            lines.append(f"PREDICTIONS ({count} row(s){covers}): {preview}{more}")
        if self.notes:
            lines.append("")
            lines.append("NOTES")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the result, without the fitted model object."""
        return {
            "target": self.target,
            "task": self.task,
            "n_rows": int(self.n_rows),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "features": [str(f) for f in self.features],
            "metrics": {str(k): jsonable(v) for k, v in self.metrics.items()},
            "feature_importance": jsonable(self.feature_importance),
            "classes": jsonable(self.classes),
            "good_class": jsonable(self.good_class),
            "optimal_ranges": {
                str(k): [jsonable(v[0]), jsonable(v[1])]
                for k, v in self.optimal_ranges.items()
            },
            "predictions": jsonable(self.predictions),
            "probabilities": jsonable(self.probabilities),
            "notes": list(self.notes),
        }

    def to_frame(self) -> pd.DataFrame:
        """The feature importances as a two-column ``feature``/``importance`` frame."""
        return pd.DataFrame(
            {
                "feature": [str(name) for name in self.feature_importance.index],
                "importance": [float(v) for v in self.feature_importance.to_numpy()],
            }
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()
