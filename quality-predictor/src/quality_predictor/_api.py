"""The one-line entry point: fit a quality model and predict, in a single call."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._io import FrameLike, load_frame
from ._model import QualityModel
from ._result import QualityResult

MODEL_OPTIONS = ("task", "random_state")
FIT_OPTIONS = ("features", "test_size", "good_class", "higher_is_better")


def predict_quality(
    df: FrameLike,
    target: str,
    new_data: Optional[FrameLike] = None,
    **kw: Any,
) -> QualityResult:
    """Fit a quality model on ``df`` and predict, in one call.

        result = predict_quality(df, "quality")
        print(result.summary())

    Args:
        df: the history to learn from: a DataFrame, or a path to a
            ``.csv``/``.tsv``/``.parquet`` file.
        target: the outcome column (pass/fail, a grade, or a measured value).
        new_data: rows to predict. Without it the fitted rows are predicted, which
            is handy for a first look but optimistic, and the result says so.
        **kw: passed through to :class:`~quality_predictor.QualityModel` and its
            ``fit``: ``task``, ``random_state``, ``features``, ``test_size``,
            ``good_class``, ``higher_is_better``.

    Returns:
        A :class:`~quality_predictor.QualityResult` with the held-out metrics,
        the feature importances, the good-outcome parameter windows, the
        predictions, and a ``summary()`` that explains itself.
    """
    model_kw = dict((key, kw.pop(key)) for key in MODEL_OPTIONS if key in kw)
    fit_kw = dict((key, kw.pop(key)) for key in FIT_OPTIONS if key in kw)
    if kw:
        known = ", ".join(MODEL_OPTIONS + FIT_OPTIONS)
        raise TypeError(
            "predict_quality() got unexpected keyword argument(s): {0}. "
            "The options are: {1}".format(", ".join(sorted(kw)), known)
        )

    model = QualityModel(**model_kw).fit(df, target, **fit_kw)
    # work the windows out BEFORE copying the notes: doing so is what discovers a
    # parameter with no window worth recommending, and that warning has to travel
    # with the report rather than stay behind on the model
    ranges = model.optimal_ranges()
    notes = list(model.notes)

    if new_data is None:
        frame = load_frame(df)
        notes.append(
            "no new_data was given, so the predictions below are for the rows the "
            "model was fitted on; pass new_data= to score fresh parts"
        )
    else:
        frame = load_frame(new_data)

    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    explain_row = None
    if len(frame):
        predictions = model.predict(frame)
        if model.task == "classification":
            probabilities = model.predict_proba(frame)
        explain_row = frame.iloc[[0]]

    return QualityResult(
        model=model,
        target=str(model.target),
        task=model.task,
        metrics=dict(model.metrics),
        feature_importance=model.feature_importance,
        features=list(model.features),
        n_rows=model.n_rows,
        n_train=model.n_train,
        n_test=model.n_test,
        predictions=predictions,
        probabilities=probabilities,
        classes=list(model.classes_) if model.classes_ is not None else None,
        good_class=model.good_class,
        optimal_ranges=ranges,
        notes=notes,
        explain_row=explain_row,
    )
