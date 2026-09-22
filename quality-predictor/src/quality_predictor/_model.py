"""The fitted model: preprocessing, gradient boosting, and the reasons behind a score."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._detect import TASKS, coerce_features, detect_task, pick_good_class, split_feature_types
from ._io import FrameLike, load_frame, require_unique_columns
from ._metrics import classification_metrics, regression_metrics
from ._ranges import (
    GRID_POINTS,
    MIN_IMPORTANCE,
    MIN_SPREAD_FRACTION,
    build_grid,
    dependence_curve,
    is_flat,
    window,
)
from ._result import Explanation, label

logger = logging.getLogger(__name__)

#: Below this many rows the held-out scores are noise, and the model says so.
SMALL_DATA_ROWS = 20

#: How many training rows to keep as the reference sample used by explanations
#: and by the parameter windows.
MAX_BACKGROUND = 200

#: How many of those rows to sweep when drawing the curve for one parameter.
MAX_RANGE_BACKGROUND = 100

#: Bumped only if the saved layout changes in a way older files cannot follow.
FORMAT_VERSION = 1

RowLike = Union[int, Dict[str, Any], pd.Series, pd.DataFrame, None]


def _one_hot_encoder() -> OneHotEncoder:
    """An encoder that ignores categories it never saw, on old and new scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class QualityModel:
    """Predict product quality from process parameters, and say what drives it.

        model = QualityModel().fit(df, "quality")
        model.metrics                  # honest held-out scores
        model.feature_importance       # what matters, by original column name
        model.optimal_ranges()         # the settings that go with good outcomes

    ``task="auto"`` reads the target and decides between classification
    (pass/fail, grades) and regression (a measured value). Missing numbers are
    filled with the column median and scaled; missing categories are filled with
    the most common value and one-hot encoded; a gradient boosting model sits on
    top. A test split is held out during :meth:`fit` so the scores are honest,
    then the model is refitted on every row so predictions use all the evidence.
    """

    def __init__(self, task: str = "auto", random_state: int = 0) -> None:
        if task not in TASKS:
            raise ValueError(
                "task must be one of {0}, got {1!r}".format(
                    ", ".join(repr(t) for t in TASKS), task
                )
            )
        self.requested_task = task
        self.task = task
        self.random_state = int(random_state)
        self.fitted = False
        self.target: Optional[str] = None
        self.features: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.classes_: Optional[List[Any]] = None
        self.good_class: Any = None
        self.higher_is_better = True
        self.metrics: Dict[str, Optional[float]] = {}
        self.notes: List[str] = []
        #: True when no class name read as pass or fail and the fallback chose,
        #: so the good-outcome windows may be the wrong way round.
        self.good_class_guessed = False
        #: Input rows that carried no outcome and so were left out of the model.
        self.n_dropped_no_outcome = 0
        #: Parameters whose window is the whole observed range, for want of a signal.
        self.uninformative_parameters: List[str] = []
        self.n_rows = 0
        self.n_train = 0
        self.n_test = 0
        self._pipeline: Optional[Pipeline] = None
        self._background: Optional[pd.DataFrame] = None
        self._importance: Optional[pd.Series] = None
        self._ranges: Optional[Dict[str, Tuple[float, float]]] = None
        self._good_index = 0

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        df: FrameLike,
        target: str,
        *,
        features: Optional[Sequence[str]] = None,
        test_size: float = 0.2,
        good_class: Any = None,
        higher_is_better: bool = True,
    ) -> "QualityModel":
        """Learn ``target`` from the other columns of ``df`` and return ``self``.

        Args:
            df: a DataFrame, or a path to a ``.csv``/``.tsv``/``.parquet`` file.
            target: the column holding the outcome (pass/fail, a grade, a measurement).
            features: the parameter columns to use; every other column by default.
            test_size: fraction of rows held out to score the model honestly.
            good_class: which class counts as the good outcome (classification only);
                read from the class names when omitted.
            higher_is_better: for a numeric target, whether a bigger number is better.
                Set ``False`` for targets such as a defect count.
        """
        frame = load_frame(df)
        require_unique_columns(frame)
        self.notes = []
        self._importance = None
        self._ranges = None
        self.good_class_guessed = False
        self.n_dropped_no_outcome = 0
        self.uninformative_parameters = []
        self.fitted = False

        if target not in frame.columns:
            known = ", ".join(repr(str(c)) for c in list(frame.columns)[:12])
            raise ValueError(
                "target column {0!r} is not in the data; columns are: {1}".format(target, known)
            )
        if len(frame) == 0:
            raise ValueError("the data has no rows, so there is nothing to learn from")

        chosen = self._resolve_features(frame, target, features)
        frame, chosen = self._drop_unusable(frame, target, chosen)

        y = self._prepare_target(frame[target])
        task = self._resolve_task(y, target)
        self.task = task
        self._check_target_variety(y, target, task)

        numeric, categorical = split_feature_types(frame, chosen)
        X = coerce_features(frame, numeric, categorical)

        self.target = str(target)
        self.features = [str(name) for name in chosen]
        self.numeric_features = [str(name) for name in numeric]
        self.categorical_features = [str(name) for name in categorical]
        self.higher_is_better = bool(higher_is_better) if task == "regression" else True
        self.n_rows = int(len(frame))
        if task == "regression" and not higher_is_better:
            self.notes.append("a lower {0} is treated as the better outcome".format(self.target))

        if self.n_rows < SMALL_DATA_ROWS:
            message = (
                "only {0} row(s): the model still fits, but the held-out scores are "
                "unreliable, so treat them as a hint rather than a measurement".format(
                    self.n_rows
                )
            )
            self.notes.append(message)
            logger.warning("quality-predictor: %s", message)

        y_values = y.to_numpy()
        split = self._split(X, y_values, task, test_size)
        self._fit_and_score(X, y_values, task, split)
        self._finalise(X, y_values, task, good_class)
        self.fitted = True
        return self

    # ------------------------------------------------------------- fit parts

    def _resolve_features(
        self, frame: pd.DataFrame, target: str, features: Optional[Sequence[str]]
    ) -> List[str]:
        """The feature columns to use, validated against the frame."""
        if features is None:
            chosen = [c for c in frame.columns if c != target]
        else:
            if isinstance(features, str):
                features = [features]
            chosen = list(features)
            missing = [str(c) for c in chosen if c not in frame.columns]
            if missing:
                raise ValueError(
                    "these feature columns are not in the data: " + ", ".join(missing)
                )
            if target in chosen:
                chosen = [c for c in chosen if c != target]
                self.notes.append(
                    "the target {0!r} was listed as a feature and was dropped".format(str(target))
                )
        if not chosen:
            raise ValueError(
                "no feature columns left once the target {0!r} is removed; the data "
                "needs at least one parameter column".format(str(target))
            )
        return chosen

    def _drop_unusable(
        self, frame: pd.DataFrame, target: str, chosen: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop rows with no outcome and columns with no values at all."""
        keep = frame[target].notna()
        dropped = int((~keep).sum())
        self.n_dropped_no_outcome = dropped
        if dropped:
            frame = frame.loc[keep]
            self.notes.append(
                "{0} row(s) had no {1} and were left out".format(dropped, str(target))
            )
        if len(frame) == 0:
            raise ValueError(
                "every row is missing {0!r}, so there is no outcome to learn".format(str(target))
            )
        empty = [c for c in chosen if frame[c].isna().all()]
        if empty:
            chosen = [c for c in chosen if c not in empty]
            self.notes.append(
                "column(s) with no values at all were skipped: "
                + ", ".join(str(c) for c in empty)
            )
        if not chosen:
            raise ValueError("every feature column is empty, so there is nothing to learn from")
        return frame, chosen

    def _prepare_target(self, column: pd.Series) -> pd.Series:
        """The target as a dtype scikit-learn can label cleanly."""
        if pd.api.types.is_bool_dtype(column):
            return column.astype(bool)
        if pd.api.types.is_numeric_dtype(column):
            return column
        # text, categories and mixed objects all become text, so a column holding
        # both 1 and "1" does not blow up on a mixed-type sort
        return pd.Series([str(v) for v in column], index=column.index, dtype=object)

    def _resolve_task(self, y: pd.Series, target: str) -> str:
        """Either the task that was asked for, or the one the target implies."""
        if self.requested_task == "auto":
            task, reason = detect_task(y)
            self.notes.append(reason)
            return task
        task = self.requested_task
        if task == "regression" and not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "task='regression' needs a numeric target, but {0!r} holds text; "
                "use task='classification' or task='auto'".format(str(target))
            )
        self.notes.append("task was set to {0!r} rather than detected".format(task))
        return task

    def _check_target_variety(self, y: pd.Series, target: str, task: str) -> None:
        """A target that never changes cannot be learned; say so plainly."""
        values = y.dropna()
        if int(values.nunique()) >= 2:
            return
        only = values.iloc[0] if len(values) else None
        # the noun follows the column itself, not the detected task: a numeric
        # column holding one value is a constant measurement, and calling that a
        # "class" purely because the detector saw fewer than two of them reads wrong
        numeric = bool(pd.api.types.is_numeric_dtype(y)) and not pd.api.types.is_bool_dtype(y)
        what = "value" if numeric else "class"
        raise ValueError(
            "target {0!r} has only one distinct {1} ({2}): there is nothing to tell "
            "apart, so the data needs rows with at least two different outcomes".format(
                str(target), what, label(only)
            )
        )

    def _split(
        self, X: pd.DataFrame, y: np.ndarray, task: str, test_size: float
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]]:
        """The train/test split, or ``None`` when the data is too small to hold one out."""
        try:
            fraction = float(test_size)
        except (TypeError, ValueError):
            raise ValueError(
                "test_size must be a fraction between 0 and 1, got {0!r}".format(test_size)
            ) from None
        if fraction >= 1 or fraction < 0:
            raise ValueError(
                "test_size must be a fraction between 0 and 1, got {0!r}".format(test_size)
            )
        if fraction == 0 or len(X) < 2:
            return None
        stratify: Optional[np.ndarray] = None
        if task == "classification":
            _, counts = np.unique(y, return_counts=True)
            if int(counts.min()) >= 2:
                stratify = y
        for attempt in (stratify, None):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=fraction,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=attempt,
                )
            except ValueError:
                continue
            if task == "classification" and len(np.unique(y_train)) < 2:
                continue
            return X_train, X_test, y_train, y_test
        return None

    def _fit_and_score(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: str,
        split: Optional[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]],
    ) -> None:
        """Fit on the training rows and score on the rows that were held back."""
        if split is None:
            X_train, X_test, y_train, y_test = X, X, y, y
            self.n_train = int(len(X))
            self.n_test = 0
            self.notes.append(
                "too few rows to hold any back, so the scores below are in-sample "
                "and will look better than the model really is"
            )
        else:
            X_train, X_test, y_train, y_test = split
            self.n_train = int(len(X_train))
            self.n_test = int(len(X_test))

        scorer = self._build_pipeline(task)
        scorer.fit(X_train, y_train)
        predictions = scorer.predict(X_test)
        if task == "classification":
            classes = list(scorer.named_steps["model"].classes_)
            good, _reason, _recognised = pick_good_class(classes)
            proba = scorer.predict_proba(X_test)
            self.metrics = classification_metrics(y_test, predictions, proba, classes, good)
        else:
            self.metrics = regression_metrics(
                np.asarray(y_test, dtype="float64"),
                np.asarray(predictions, dtype="float64"),
            )

    def _finalise(self, X: pd.DataFrame, y: np.ndarray, task: str, good_class: Any) -> None:
        """Refit on every row, then work out classes, reference rows and importance."""
        pipeline = self._build_pipeline(task)
        pipeline.fit(X, y)
        self._pipeline = pipeline
        if self.n_test:
            self.notes.append(
                "scores come from {0} held-out row(s); the model you now hold was "
                "refitted on all {1}".format(self.n_test, self.n_rows)
            )

        if task == "classification":
            classes = list(pipeline.named_steps["model"].classes_)
            self.classes_ = classes
            if good_class is None:
                chosen, reason, recognised = pick_good_class(classes)
                self.good_class = chosen
                self.good_class_guessed = not recognised
                self.notes.append(reason)
            else:
                matches = [c for c in classes if c == good_class]
                if not matches:
                    known = ", ".join(label(c) for c in classes)
                    raise ValueError(
                        "good_class={0} is not one of the classes in the target: {1}".format(
                            label(good_class), known
                        )
                    )
                self.good_class = matches[0]
                self.good_class_guessed = False
            self._good_index = max(i for i, c in enumerate(classes) if c == self.good_class)
        else:
            self.classes_ = None
            self.good_class = None
            self._good_index = 0

        if len(X) > MAX_BACKGROUND:
            self._background = X.sample(
                MAX_BACKGROUND, random_state=self.random_state
            ).reset_index(drop=True)
        else:
            self._background = X.reset_index(drop=True)
        self._importance = self._compute_importance()

    def _build_pipeline(self, task: str) -> Pipeline:
        """Impute, scale, one-hot encode, then boost."""
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical = Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")), ("encode", _one_hot_encoder())]
        )
        blocks: List[Any] = []
        if self.numeric_features:
            blocks.append(("num", numeric, list(self.numeric_features)))
        if self.categorical_features:
            blocks.append(("cat", categorical, list(self.categorical_features)))
        preprocess = ColumnTransformer(blocks, remainder="drop")
        if task == "classification":
            model: Any = GradientBoostingClassifier(random_state=self.random_state)
        else:
            model = GradientBoostingRegressor(random_state=self.random_state)
        return Pipeline([("preprocess", preprocess), ("model", model)])

    # ----------------------------------------------------------- importance

    def _block_widths(self) -> List[Tuple[str, int]]:
        """``(original column, how many model inputs it became)`` in model input order."""
        pipeline = self._pipeline
        if pipeline is None:  # pragma: no cover - guarded by _require_fitted
            return []
        preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
        named = dict((name, step) for name, step, _ in preprocess.transformers_)
        widths: List[Tuple[str, int]] = []
        if "num" in named and self.numeric_features:
            imputer = named["num"].named_steps["impute"]
            for name in imputer.get_feature_names_out(self.numeric_features):
                widths.append((str(name), 1))
        if "cat" in named and self.categorical_features:
            block = named["cat"]
            kept = list(
                block.named_steps["impute"].get_feature_names_out(self.categorical_features)
            )
            categories = list(block.named_steps["encode"].categories_)
            for name, levels in zip(kept, categories):
                widths.append((str(name), int(len(levels))))
        return widths

    def _compute_importance(self) -> pd.Series:
        """Importance per ORIGINAL column, with one-hot levels summed back together."""
        pipeline = self._pipeline
        raw = np.asarray(
            getattr(pipeline.named_steps["model"], "feature_importances_", []), dtype="float64"
        )
        widths = self._block_widths()
        totals = dict((str(name), 0.0) for name in self.features)
        cursor = 0
        for name, width in widths:
            chunk = raw[cursor : cursor + width]
            totals[str(name)] = float(np.sum(chunk)) if chunk.size else 0.0
            cursor += width
        modelled = set(str(name) for name, _ in widths)
        skipped = [name for name in self.features if name not in modelled]
        if skipped:
            self.notes.append(
                "column(s) dropped before modelling and scored 0: " + ", ".join(skipped)
            )
        total = float(sum(totals.values()))
        if total > 0:
            totals = dict((key, value / total) for key, value in totals.items())
        series = pd.Series(totals, dtype="float64")
        series = series.sort_values(ascending=False, kind="mergesort")
        series.index.name = "feature"
        series.name = "importance"
        return series

    @property
    def feature_importance(self) -> pd.Series:
        """How much each ORIGINAL column drives the prediction, biggest first.

        One-hot levels are summed back onto the column they came from, so a
        ``machine`` column with six values appears once, not six times.
        """
        self._require_fitted("feature_importance")
        return self._importance.copy()

    # -------------------------------------------------------------- predict

    def _require_fitted(self, what: str) -> None:
        """Raise a clear error when the model has not seen any data yet."""
        if not self.fitted or self._pipeline is None:
            raise ValueError(
                "QualityModel is not fitted yet, so {0} has nothing to work with: "
                "call fit(df, target) first".format(what)
            )

    def _prepare(self, df: FrameLike) -> pd.DataFrame:
        """New rows as the exact columns and dtypes the pipeline was fitted on."""
        if isinstance(df, pd.Series):
            frame = df.to_frame().T
        elif isinstance(df, dict):
            frame = pd.DataFrame([df])
        else:
            frame = load_frame(df)
        require_unique_columns(frame)
        missing = [str(c) for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(
                "these columns were used to fit the model but are not in the data: "
                + ", ".join(missing)
            )
        if len(frame) == 0:
            raise ValueError("the data has no rows, so there is nothing to predict")
        return coerce_features(frame, self.numeric_features, self.categorical_features)

    def predict(self, df: FrameLike) -> np.ndarray:
        """Predict the outcome for every row of ``df``."""
        self._require_fitted("predict()")
        return np.asarray(self._pipeline.predict(self._prepare(df)))

    def predict_proba(self, df: FrameLike) -> np.ndarray:
        """Class probabilities per row, in :attr:`classes_` order (classification only)."""
        self._require_fitted("predict_proba()")
        if self.task != "classification":
            raise ValueError(
                "predict_proba() is only available for classification, but this model "
                "predicts a number ({0!r} is a regression target); use predict()".format(
                    self.target
                )
            )
        return np.asarray(self._pipeline.predict_proba(self._prepare(df)))

    def _score(self, prepared: pd.DataFrame) -> np.ndarray:
        """The number explanations and parameter windows are built on.

        Classification: the probability of the good class. Regression: the
        predicted value itself.
        """
        if self.task == "classification":
            proba = np.asarray(self._pipeline.predict_proba(prepared), dtype="float64")
            return proba[:, self._good_index]
        return np.asarray(self._pipeline.predict(prepared), dtype="float64")

    @property
    def _direction(self) -> float:
        """``+1`` when a higher score is the better outcome, ``-1`` when a lower one is."""
        return 1.0 if (self.task == "classification" or self.higher_is_better) else -1.0

    # -------------------------------------------------------------- reports

    def evaluate(
        self, df: Optional[FrameLike] = None, target: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """Score the model. With no arguments, the held-out scores from :meth:`fit`.

        Pass a fresh DataFrame (and optionally a different target column name) to
        score rows the model has never seen.
        """
        self._require_fitted("evaluate()")
        if df is None:
            return dict(self.metrics)
        frame = load_frame(df)
        require_unique_columns(frame)
        column = self.target if target is None else target
        if column not in frame.columns:
            raise ValueError(
                "target column {0!r} is not in the data given to evaluate()".format(str(column))
            )
        frame = frame.loc[frame[column].notna()]
        if len(frame) == 0:
            raise ValueError("no rows with a known {0} to score".format(str(column)))
        truth = self._prepare_target(frame[column]).to_numpy()
        prepared = self._prepare(frame)
        predictions = self._pipeline.predict(prepared)
        if self.task == "classification":
            classes = list(self.classes_ or [])
            proba = np.asarray(self._pipeline.predict_proba(prepared))
            return classification_metrics(truth, predictions, proba, classes, self.good_class)
        return regression_metrics(
            np.asarray(truth, dtype="float64"), np.asarray(predictions, dtype="float64")
        )

    def _as_row(self, row: RowLike) -> pd.DataFrame:
        """One row, whatever shape it arrived in, as a one-row prepared frame."""
        background = self._background
        if row is None:
            return background.iloc[[0]].copy()
        if isinstance(row, (int, np.integer)) and not isinstance(row, bool):
            index = int(row)
            size = len(background)
            if index < -size or index >= size:
                raise ValueError(
                    "row {0} is out of range: {1} reference row(s) were kept from "
                    "training, so use 0..{2}, or pass the row itself".format(
                        index, size, size - 1
                    )
                )
            return background.iloc[[index]].copy()
        if isinstance(row, pd.DataFrame):
            if len(row) == 0:
                raise ValueError(
                    "explain() was given an empty frame, so there is no row to explain"
                )
            return self._prepare(row.iloc[[0]])
        return self._prepare(row)

    def explain(self, row: RowLike = None) -> Explanation:
        """Why one row gets the prediction it does.

        ``row`` may be a dict, a Series, a one-row DataFrame, or an integer
        position into the reference rows kept from training. Each feature's
        effect is measured by replaying the reference rows with that one value
        swapped in, so a positive number means this row's setting pushed the
        outcome toward good.
        """
        self._require_fitted("explain()")
        prepared = self._as_row(row)
        background = self._background
        baseline = float(np.mean(self._score(background)))
        score = float(self._score(prepared)[0])
        direction = self._direction
        values = dict((str(name), prepared.iloc[0][name]) for name in prepared.columns)
        contributions: Dict[str, float] = {}
        for name in self.features:
            if name not in prepared.columns:  # pragma: no cover - _prepare guarantees it
                continue
            work = background.copy()
            work[name] = prepared.iloc[0][name]
            moved = float(np.mean(self._score(work))) - baseline
            contributions[str(name)] = moved * direction
        prediction = self.predict(prepared)[0]
        return Explanation(
            prediction=prediction,
            contributions=contributions,
            score=score,
            baseline=baseline,
            task=self.task,
            target=str(self.target),
            good_class=self.good_class,
            values=values,
        )

    def _spread_floor(self, sweep: pd.DataFrame) -> float:
        """The smallest curve movement that counts as a real effect.

        Measured as a fraction of the spread of the model's own predictions over
        real rows, so every parameter is judged against one absolute yardstick
        instead of against its own wobble. Gradient boosting still splits
        occasionally on a parameter that drives nothing, and scaling the keep
        band to that wobble is what collapses such a parameter's window onto a
        single grid point and invents a setpoint nobody should hold.
        """
        try:
            scores = np.asarray(self._score(sweep), dtype="float64")
        except Exception:  # pragma: no cover - defensive
            return 0.0
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            return 0.0
        spread = float(np.max(finite) - np.min(finite))
        if not np.isfinite(spread) or spread <= 0.0:
            return 0.0
        return MIN_SPREAD_FRACTION * spread

    def optimal_ranges(self, n_points: int = GRID_POINTS) -> Dict[str, Tuple[float, float]]:
        """The setting window for each numeric parameter that goes with good outcomes.

        For every numeric parameter the model is replayed across the observed
        range of that parameter; the window is the stretch that stays near the
        best average outcome. Ordered by importance, most important first.

        A parameter that moves the outcome too little to act on gets the whole
        observed range rather than a window scaled to its own noise. Its name
        goes into :attr:`uninformative_parameters` and a note says so, so a
        pure-noise column is never printed as a setpoint to hold.
        """
        self._require_fitted("optimal_ranges()")
        if self._ranges is not None:
            return dict(self._ranges)
        background = self._background
        sweep = background
        if len(sweep) > MAX_RANGE_BACKGROUND:
            sweep = sweep.sample(
                MAX_RANGE_BACKGROUND, random_state=self.random_state
            ).reset_index(drop=True)
        direction = self._direction
        floor = self._spread_floor(sweep)
        ordered = [name for name in self._importance.index if name in self.numeric_features]
        ranges: Dict[str, Tuple[float, float]] = {}
        flat: List[str] = []
        for name in ordered:
            grid = build_grid(background[name], n_points=n_points)
            if grid.size == 0:
                continue
            curve = dependence_curve(self._score, sweep, name, grid)
            shifted = curve * direction
            # a parameter carrying this little of the model's decisions has no
            # window worth printing, however its curve happens to wobble
            weak = float(self._importance.get(name, 0.0)) < MIN_IMPORTANCE
            if weak or is_flat(shifted, floor):
                flat.append(str(name))
                ranges[str(name)] = (float(grid[0]), float(grid[-1]))
                continue
            bounds = window(grid, shifted, floor=floor)
            if bounds is not None:
                ranges[str(name)] = bounds
        self.uninformative_parameters = flat
        if flat:
            self.notes.append(
                "{0} move(s) {1} too little to recommend a window, so the whole "
                "observed range is shown for {2} rather than a setting to hold".format(
                    ", ".join(flat), self.target, "them" if len(flat) > 1 else "it"
                )
            )
        self._ranges = ranges
        return dict(ranges)

    # ----------------------------------------------------------- save, load

    def save(self, path: Union[str, "os.PathLike[str]"]) -> Path:
        """Write the fitted model to ``path`` and return that path."""
        self._require_fitted("save()")
        target = Path(path)
        if str(target.parent) not in ("", "."):
            target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            pickle.dump({"format": FORMAT_VERSION, "model": self}, handle, protocol=4)
        return target

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "QualityModel":
        """Read back a model written by :meth:`save`."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError("no such model file: {0}".format(source))
        with open(source, "rb") as handle:
            payload = pickle.load(handle)
        model = payload.get("model") if isinstance(payload, dict) else payload
        if not isinstance(model, cls):
            raise ValueError(
                "{0} does not hold a QualityModel; it holds {1}".format(
                    source, type(model).__name__
                )
            )
        return model

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe description of what was fitted, without the model object itself."""
        from ._result import jsonable

        return {
            "target": self.target,
            "task": self.task,
            "features": list(self.features),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "n_rows": int(self.n_rows),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "metrics": dict((str(k), jsonable(v)) for k, v in self.metrics.items()),
            "classes": jsonable(self.classes_),
            "good_class": jsonable(self.good_class),
            "feature_importance": (
                jsonable(self._importance) if self._importance is not None else {}
            ),
            "notes": list(self.notes),
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        if not self.fitted:
            return "QualityModel(task={0!r}, not fitted)".format(self.requested_task)
        return "QualityModel(target={0!r}, task={1!r}, features={2}, rows={3})".format(
            self.target, self.task, len(self.features), self.n_rows
        )
