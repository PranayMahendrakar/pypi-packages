"""The lightweight model that extends rule labels to the rest of the data.

Text: TF-IDF (word uni+bigrams) + LogisticRegression (C=10, balanced classes).
Tabular: TF-IDF for free-text columns, one-hot for genuine categories, median-imputed
and scaled numerics, + LogisticRegression.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from auto_label._data import TABULAR, Prepared

logger = logging.getLogger(__name__)

MISSING_TOKEN = "<NA>"

# A string column is read as free text (TF-IDF) instead of a category (one-hot) when its
# values average this many whitespace tokens, or when it holds this many near-unique values.
TEXT_MIN_MEAN_TOKENS = 2.0
TEXT_MIN_UNIQUE = 10
TEXT_MIN_UNIQUE_RATIO = 0.5

# A numeric column is read as a row id, and dropped, when it holds this many rows of
# near-unique whole numbers in order.
ID_MIN_ROWS = 5
ID_MIN_UNIQUE_RATIO = 0.95


def _classifier(random_state: int) -> LogisticRegression:
    # C=10: the scikit-learn default (C=1) over-regularises L2-normalised TF-IDF rows, so
    # even clear matches come out near 0.5 on small rule-labelled sets.
    return LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=random_state)


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )


def build_text_model(random_state: int = 0) -> Pipeline:
    """TF-IDF + LogisticRegression pipeline for a list of strings."""
    return Pipeline([("tfidf", _tfidf()), ("clf", _classifier(random_state))])


def looks_like_free_text(series: pd.Series) -> bool:
    """True when a string-like column reads as free text rather than as a category.

    One-hot encoding free text makes one category per distinct string: every value the
    model did not see while training becomes an all-zero row and the prediction collapses
    to the class prior. Such a column has to go through TF-IDF instead. A column the
    caller declared ``Categorical`` is always taken at their word.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return False
    values = [v for v in series.tolist() if isinstance(v, str) and v.strip()]
    if not values:
        return False
    present = int(series.notna().sum())
    if present and len(values) * 2 < present:
        return False  # mostly non-strings in an object column: treat them as categories
    mean_tokens = sum(len(v.split()) for v in values) / len(values)
    if mean_tokens >= TEXT_MIN_MEAN_TOKENS:
        return True
    distinct = len(set(values))
    return distinct >= TEXT_MIN_UNIQUE and distinct >= TEXT_MIN_UNIQUE_RATIO * len(values)


def looks_like_row_id(values: pd.Series) -> bool:
    """True when a numeric column reads as a row identifier rather than a measurement.

    Near-unique whole numbers in order describe where a row sits in the file, not what
    the row holds. Left in as a feature they are actively harmful: an export sorted by
    type (which most exports are) has its classes separated perfectly by the id, so the
    id outvotes every real feature and the wrong label comes back more confident than
    the right one. ``values`` is the already-coerced float column.
    """
    clean = values.dropna()
    n = len(clean)
    if n < ID_MIN_ROWS:
        return False
    if clean.nunique() < ID_MIN_UNIQUE_RATIO * n:
        return False
    if not (clean.is_monotonic_increasing or clean.is_monotonic_decreasing):
        return False
    arr = np.asarray(clean, dtype="float64")
    return bool(np.all(arr == np.floor(arr)))


def _to_seconds(series: pd.Series) -> pd.Series:
    """Datetime / timedelta column -> float seconds (NaT -> NaN)."""
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        return series.dt.total_seconds().astype("float64")
    out = []
    for v in series.tolist():
        if isinstance(v, (pd.Timestamp, _dt.datetime)) and not pd.isna(v):
            out.append(v.timestamp())
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="float64")


def tabular_features(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    """Split a DataFrame into model-ready columns.

    Returns ``(features, categorical_columns, numeric_columns, text_columns, dropped)``.
    Columns that are entirely missing are dropped, and so are numeric columns that read as
    row ids (see :func:`looks_like_row_id`); ``dropped`` names the latter so the caller can
    report them. Free-text columns are kept for TF-IDF, booleans and other non-numerics
    become categories (missing -> ``"<NA>"``), numerics become float with inf -> NaN.
    Feature names are the stringified column labels, made unique so a frame with columns
    such as ``1`` and ``"1"`` still works.
    """
    features = pd.DataFrame(index=frame.index)
    cat_cols: List[str] = []
    num_cols: List[str] = []
    text_cols: List[str] = []
    dropped: List[str] = []
    used: Set[str] = set()

    def unique_name(col: Any) -> str:
        base = str(col)
        name = base
        suffix = 2
        while name in used:
            name = f"{base}__{suffix}"
            suffix += 1
        used.add(name)
        return name

    for col in frame.columns:
        series = frame[col]
        if series.isna().all():
            continue
        name = unique_name(col)
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            features[name] = _to_seconds(series)
            num_cols.append(name)
        elif pd.api.types.is_bool_dtype(dtype):
            features[name] = series.astype(object).where(series.notna(), MISSING_TOKEN).astype(str)
            cat_cols.append(name)
        elif pd.api.types.is_numeric_dtype(dtype):
            values = pd.to_numeric(series, errors="coerce").astype("float64")
            values = values.replace([np.inf, -np.inf], np.nan)
            if looks_like_row_id(values):
                dropped.append(name)
                continue
            features[name] = values
            num_cols.append(name)
        elif looks_like_free_text(series):
            features[name] = series.astype(object).where(series.notna(), "").astype(str)
            text_cols.append(name)
        else:
            features[name] = series.astype(object).where(series.notna(), MISSING_TOKEN).astype(str)
            cat_cols.append(name)
    return features, cat_cols, num_cols, text_cols, dropped


def build_tabular_model(
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    text_cols: Sequence[str] = (),
    random_state: int = 0,
) -> Pipeline:
    """TF-IDF (free text) + one-hot (categories) + scaled numerics, then LogisticRegression."""
    transformers: List[Any] = []
    for i, col in enumerate(text_cols):
        # a bare column name (not a list): TfidfVectorizer needs a 1-D column, not a frame
        transformers.append((f"text{i}", _tfidf(), col))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)))
    if num_cols:
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("num", numeric, list(num_cols)))
    if not transformers:
        raise ValueError("no usable feature columns")
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", _classifier(random_state))])


def describe_tabular_split(cat_cols: Sequence[str], num_cols: Sequence[str], text_cols: Sequence[str]) -> str:
    """One line naming which columns the tabular model read as text, category and number."""
    shown = ", ".join(repr(c) for c in text_cols)
    rest = []
    if cat_cols:
        rest.append("one-hot " + ", ".join(repr(c) for c in cat_cols))
    if num_cols:
        rest.append("numeric " + ", ".join(repr(c) for c in num_cols))
    tail = "; " + "; ".join(rest) if rest else ""
    return f"model: column(s) {shown} read as free text (TF-IDF){tail}"


def describe_dropped_ids(dropped: Sequence[str]) -> str:
    """One line naming the columns that were ignored because they read as row ids."""
    return (
        "model: ignored column(s) "
        + ", ".join(repr(c) for c in dropped)
        + " as row identifiers (near-unique whole numbers in order); they describe where a "
        "row sits in the file, not what it holds, so they were kept out of the features"
    )


def drop_columns_empty_in_train(
    features: pd.DataFrame,
    num_cols: Sequence[str],
    text_cols: Sequence[str],
    train_idx: Sequence[int],
) -> Tuple[List[str], List[str], List[str]]:
    """Drop feature columns that hold nothing for the rule-labeled rows.

    :func:`tabular_features` only drops a column that is empty over the *whole* frame. A
    column filled in for just the newest rows is empty exactly where the model learns:
    the imputer silently skips it (leaking a scikit-learn UserWarning to the caller),
    StandardScaler is then handed zero features and raises, and the entire fit is
    abandoned. Dropping such a column instead lets the usable columns train the model.
    Returns ``(num_cols, text_cols, notes)``.
    """
    train = features.iloc[list(train_idx)]
    blank_num = [c for c in num_cols if train[c].isna().all()]
    blank_text = [c for c in text_cols if not (train[c].astype(str).str.strip() != "").any()]
    notes: List[str] = []
    if blank_num or blank_text:
        message = (
            "model: ignored column(s) "
            + ", ".join(repr(c) for c in blank_num + blank_text)
            + " because they are empty for every rule-labeled row, so the model has nothing "
            "to learn from them; the remaining columns were used"
        )
        logger.warning("%s", message)
        notes.append(message)
    return (
        [c for c in num_cols if c not in blank_num],
        [c for c in text_cols if c not in blank_text],
        notes,
    )


def fit_predict(
    prepared: Prepared,
    train_idx: Sequence[int],
    train_labels: Sequence[str],
    predict_idx: Sequence[int],
    random_state: int = 0,
) -> Tuple[List[str], np.ndarray, Pipeline, List[str]]:
    """Fit on the rule-labelled rows, return ``(classes, probabilities, pipeline, notes)``.

    ``probabilities`` has one row per entry of ``predict_idx`` and one column per class.
    ``notes`` records what the tabular feature split decided, and every column it dropped,
    so the caller can report it.
    Raises whatever scikit-learn raises (for example an empty TF-IDF vocabulary); the
    caller decides how to degrade.
    """
    train_idx = list(train_idx)
    predict_idx = list(predict_idx)
    y = np.asarray(list(train_labels), dtype=object)
    notes: List[str] = []

    if prepared.mode == TABULAR and prepared.frame is not None:
        features, cat_cols, num_cols, text_cols, dropped = tabular_features(prepared.frame)
        if dropped:
            message = describe_dropped_ids(dropped)
            logger.warning("%s", message)
            notes.append(message)
        num_cols, text_cols, blank_notes = drop_columns_empty_in_train(
            features, num_cols, text_cols, train_idx
        )
        notes.extend(blank_notes)
        if not (cat_cols or num_cols or text_cols):
            raise ValueError(
                "no column was usable: every column was either empty for the rule-labeled "
                "rows or read as a row id"
            )
        pipeline = build_tabular_model(cat_cols, num_cols, text_cols, random_state)
        if text_cols:
            notes.append(describe_tabular_split(cat_cols, num_cols, text_cols))
        x_train: Any = features.iloc[train_idx]
        x_pred: Any = features.iloc[predict_idx]
    else:
        pipeline = build_text_model(random_state)
        x_train = [prepared.texts[i] for i in train_idx]
        x_pred = [prepared.texts[i] for i in predict_idx]

    pipeline.fit(x_train, y)
    proba = pipeline.predict_proba(x_pred) if predict_idx else np.zeros((0, len(pipeline.classes_)))
    classes = [str(c) for c in pipeline.classes_]
    return classes, np.asarray(proba, dtype=float), pipeline, notes
