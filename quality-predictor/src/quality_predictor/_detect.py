"""Deciding what kind of problem this is, and what kind of column each feature is.

Every heuristic here reports what it decided, so nothing about a fitted model is
a mystery: :attr:`QualityModel.notes` carries the sentences these produce.
"""

from __future__ import annotations

import unicodedata
from typing import List, Tuple

import numpy as np
import pandas as pd

from ._result import label

TASKS = ("auto", "classification", "regression")

#: A numeric target with whole-number values and no more than this many distinct
#: values is read as a set of classes rather than a quantity to predict.
MAX_INT_CLASSES = 10

# Label words that mark the good side of a pass/fail outcome, and the bad side.
# Matching is exact on the whole label, never on a substring: the Japanese and
# Chinese words for "failed" contain the word for "passed", so a substring test
# would read them backwards.
_GOOD_WORDS = frozenset(
    {
        # English
        "pass", "passed", "passing", "ok", "okay", "good", "accept", "accepted",
        "yes", "true", "success", "in_spec", "in-spec", "inspec", "conforming",
        "a", "grade_a", "healthy", "within_spec", "within-spec",
        # German, Dutch, Scandinavian
        "gut", "in_ordnung", "io", "goed", "bra", "god",
        # Spanish, Portuguese, Italian, French, Romanian
        "bueno", "buena", "aprobado", "conforme", "bom", "boa", "aprovado",
        "buono", "buona", "bon", "bonne", "conforme", "reussi", "bun",
        # Nordic/Slavic/Turkish
        "dobry", "dobre", "godny", "iyi", "uygun",
        # CJK and Korean
        "合格", "良", "良品", "合格品",
        "パス", "양품", "합격",
    }
)
_BAD_WORDS = frozenset(
    {
        # English
        "fail", "failed", "failing", "failure", "defect", "defective", "reject",
        "rejected", "scrap", "bad", "no", "false", "ng", "nok", "out_of_spec",
        "out-of-spec", "nonconforming", "f", "grade_f", "faulty", "rework",
        # German, Dutch, Scandinavian
        "schlecht", "fehler", "ausschuss", "nicht_io", "nio", "slecht",
        "daarlig", "dalig",
        # Spanish, Portuguese, Italian, French, Romanian
        "malo", "mala", "rechazado", "defecto", "ruim", "reprovado", "rejeitado",
        "cattivo", "cattiva", "scarto", "mauvais", "mauvaise", "rebut", "echec",
        "rau", "defect",
        # Nordic/Slavic/Turkish
        "zly", "zle", "brak", "kotu", "uygunsuz", "hatali",
        # CJK and Korean
        "不合格", "不良", "不良品",
        "不合格品", "不良品",
        "불량", "불합격",
    }
)


def detect_task(y: pd.Series) -> Tuple[str, str]:
    """Return ``(task, reason)`` for target ``y``: ``"classification"`` or ``"regression"``."""
    values = y.dropna()
    n_unique = int(values.nunique())
    if pd.api.types.is_bool_dtype(y):
        return "classification", "the target is boolean, so this is classification"
    if isinstance(y.dtype, pd.CategoricalDtype):
        return "classification", "the target is a pandas category, so this is classification"
    if pd.api.types.is_numeric_dtype(y):
        if n_unique <= 2:
            return (
                "classification",
                f"the target is numeric with {n_unique} distinct value(s), "
                "so this is classification",
            )
        numbers = values.to_numpy(dtype="float64")
        whole = bool(np.all(np.isfinite(numbers))) and bool(
            np.all(np.equal(np.mod(numbers, 1), 0))
        )
        if whole and n_unique <= MAX_INT_CLASSES:
            return (
                "classification",
                f"the target holds whole numbers with only {n_unique} distinct values, "
                "so they are treated as classes; pass task='regression' to predict "
                "the number itself",
            )
        return (
            "regression",
            f"the target is numeric with {n_unique} distinct values, so this is regression",
        )
    return (
        "classification",
        f"the target is text with {n_unique} distinct value(s), so this is classification",
    )


def split_feature_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    """Split ``features`` into ``(numeric, categorical)`` by dtype, keeping frame order."""
    numeric: List[str] = []
    categorical: List[str] = []
    for name in features:
        column = df[name]
        if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
            numeric.append(name)
        elif pd.api.types.is_datetime64_any_dtype(column):
            # a timestamp is a quantity once it is a number of seconds, which is
            # far more useful to a tree than 10,000 one-hot columns would be
            numeric.append(name)
        else:
            categorical.append(name)
    return numeric, categorical


def _as_text(value: object) -> object:
    """A category label as text, or ``np.nan`` for anything missing.

    Missing categories must be real ``NaN`` and never ``None``: scikit-learn's
    imputer looks for NaN in object columns and would carry a ``None`` straight
    through into the encoder.
    """
    if value is None:
        return np.nan
    if isinstance(value, float) and value != value:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    return str(value)


def _seconds_since_epoch(column: pd.Series) -> pd.Series:
    """A timestamp column as float seconds, with ``NaT`` becoming ``NaN``."""
    stamps = pd.to_datetime(column, errors="coerce")
    tz = getattr(stamps.dt, "tz", None)
    if tz is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("float64")


def coerce_features(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    """Return the feature columns as the dtypes the pipeline expects.

    Booleans and timestamps become floats; everything categorical becomes text so
    that a column holding both ``1`` and ``"1"`` one-hot encodes as one value.
    """
    out = {}
    for name in numeric:
        column = df[name]
        if pd.api.types.is_datetime64_any_dtype(column):
            out[name] = _seconds_since_epoch(column)
        elif pd.api.types.is_bool_dtype(column):
            out[name] = column.astype("float64")
        else:
            out[name] = pd.to_numeric(column, errors="coerce").astype("float64")
    for name in categorical:
        out[name] = pd.Series(
            [_as_text(v) for v in df[name].tolist()], index=df.index, dtype="object"
        )
    frame = pd.DataFrame(out, index=df.index)
    return frame[numeric + categorical]


def _folded(value: object) -> str:
    """A class label folded for lookup: trimmed, lowercased, accents stripped.

    Stripping accents lets one spelling cover ``reussi`` and ``réussi``; the
    fold is only ever used to look a label up, never to display it.
    """
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def pick_good_class(classes: List[object]) -> Tuple[object, str, bool]:
    """Guess which class is the desirable outcome, and say why.

    Returns ``(class, reason, recognised)``. The guess reads the label text first
    (``"pass"`` beats ``"fail"``, in several languages); with nothing to read it
    falls back to the last class, which for ``0/1`` and ``False/True`` targets is
    the ``1``/``True`` side. ``recognised`` is False for that fallback, so callers
    can say out loud that the direction is a guess.
    """
    folded = {}
    for c in classes:
        folded.setdefault(_folded(c), c)
    good = [folded[k] for k in folded if k in _GOOD_WORDS]
    bad = [folded[k] for k in folded if k in _BAD_WORDS]
    if len(good) == 1:
        chosen = good[0]
        return chosen, f"class {label(chosen)} reads as the good outcome", True
    if len(bad) == 1 and len(classes) == 2:
        rest = [c for c in classes if not (c is bad[0] or c == bad[0])]
        if rest:
            chosen = rest[0]
            return (
                chosen,
                "class {0} reads as the bad outcome, so {1} is the good one".format(
                    label(bad[0]), label(chosen)
                ),
                True,
            )
    chosen = classes[-1]
    return (
        chosen,
        "no class name reads as pass or fail, so {0} (the last class) is treated "
        "as the good outcome, and the good-outcome windows assume it; pass "
        "good_class= to fit() to choose the other way round".format(label(chosen)),
        False,
    )
