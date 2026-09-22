"""Gaussian-copula synthesizer: ``generate(df)`` and the ``Synthesizer`` class."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ._columns import ColumnModel, build_column_model
from ._io import TableLike, load_table

logger = logging.getLogger(__name__)

DEFAULT_PRESERVE: Tuple[str, ...] = ("marginals", "correlations")
_PRESERVE_OPTIONS = frozenset(DEFAULT_PRESERVE)

RandomState = Union[None, int, np.random.Generator]


def _normalize_preserve(preserve: Union[None, str, Iterable[str]]) -> Tuple[str, ...]:
    if preserve is None:
        return ()
    if isinstance(preserve, str):
        preserve = (preserve,)
    items = tuple(preserve)
    for item in items:
        if item not in _PRESERVE_OPTIONS:
            raise ValueError(
                f"preserve may only contain {sorted(_PRESERVE_OPTIONS)}, got {item!r}"
            )
    return items


def _make_rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_frame(df: pd.DataFrame) -> None:
    if not df.columns.is_unique:
        duplicates = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        raise ValueError(f"Column names must be unique; duplicated: {duplicates}")


def nearest_correlation_matrix(
    matrix: np.ndarray, eps: float = 1e-6, max_iter: int = 100
) -> np.ndarray:
    """Closest symmetric, unit-diagonal, positive-definite matrix to ``matrix``.

    Clips negative eigenvalues and rescales the diagonal until the smallest
    eigenvalue is above ``eps``. A valid correlation matrix is returned unchanged.
    """
    a = np.array(matrix, dtype="float64", copy=True)
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 1.0)
    for _ in range(max_iter):
        w, v = np.linalg.eigh(a)
        if w.min() > eps:
            break
        w = np.clip(w, eps, None)
        a = (v * w) @ v.T
        d = np.sqrt(np.clip(np.diag(a), eps, None))
        a = a / np.outer(d, d)
        a = (a + a.T) / 2.0
        np.fill_diagonal(a, 1.0)
    return a


def _cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L @ L.T == corr (eigen fallback if Cholesky fails)."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(corr)
        return v * np.sqrt(np.clip(w, 0.0, None))


def _as_series(values: Any, n: int) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values, index=pd.RangeIndex(n))


def _cast_like(series: pd.Series, dtype: Any) -> pd.Series:
    try:
        return series.astype(dtype)
    except (TypeError, ValueError, OverflowError):
        logger.debug("Could not cast column back to %s; keeping %s", dtype, series.dtype)
        return series


class Synthesizer:
    """Gaussian-copula synthesizer for a pandas DataFrame.

    ``fit`` learns each column's marginal distribution (empirical quantiles for
    numeric, datetime and timedelta columns; category frequencies for everything
    else) and the correlation matrix of the columns' normal scores. ``sample``
    draws correlated normals, maps them back through every marginal and reinjects
    missing values at the rate seen in the input.

    Args:
        random_state: seed (or numpy Generator) that makes ``sample`` reproducible.
        preserve: which structure to keep, a subset of ``("marginals", "correlations")``.
            Without ``"correlations"`` columns are sampled independently; without
            ``"marginals"`` numeric columns use a fitted normal instead of their
            empirical distribution (smoother, less identifiable values).
        text_threshold: string columns whose share of unique values exceeds this
            are treated as free text and resampled from the originals (with a warning).
    """

    def __init__(
        self,
        random_state: RandomState = 0,
        *,
        preserve: Union[str, Iterable[str]] = DEFAULT_PRESERVE,
        text_threshold: float = 0.5,
    ) -> None:
        self.random_state = random_state
        self.preserve = _normalize_preserve(preserve)
        if not 0.0 <= float(text_threshold) <= 1.0:
            raise ValueError("text_threshold must be between 0 and 1")
        self.text_threshold = float(text_threshold)
        self._models: List[ColumnModel] = []
        self._columns: Optional[pd.Index] = None
        self._transform = np.zeros((0, 0))
        self.n_rows_: int = 0
        self.columns_: List[Any] = []
        self.column_kinds_: Dict[Any, str] = {}
        self.high_cardinality_columns_: List[Any] = []
        self.correlation_ = pd.DataFrame()

    # ------------------------------------------------------------------ fit
    def fit(self, data: TableLike) -> "Synthesizer":
        """Learn the marginals and correlations of ``data`` (a DataFrame or .csv/.parquet path)."""
        df = load_table(data)
        _validate_frame(df)
        smooth = "marginals" not in self.preserve
        models = [
            build_column_model(
                name, df.iloc[:, i], smooth=smooth, text_threshold=self.text_threshold
            )
            for i, name in enumerate(df.columns)
        ]
        modeled = [m for m in models if m.modeled]
        k = len(modeled)
        if k == 0:
            corr = np.zeros((0, 0))
        elif "correlations" in self.preserve and len(df) > 1:
            scores = pd.DataFrame({i: m.scores for i, m in enumerate(modeled)})
            corr = scores.corr().to_numpy(dtype="float64")
            corr = np.where(np.isfinite(corr), corr, 0.0)
            corr = nearest_correlation_matrix(corr)
        else:
            corr = np.eye(k)

        self._models = models
        self._columns = df.columns
        self._transform = _cholesky_factor(corr) if k else np.zeros((0, 0))
        self.n_rows_ = int(len(df))
        self.columns_ = list(df.columns)
        self.column_kinds_ = {m.name: m.kind for m in models}
        self.high_cardinality_columns_ = [m.name for m in models if m.kind == "text"]
        names = [m.name for m in modeled]
        self.correlation_ = pd.DataFrame(corr, index=names, columns=names)

        for model in models:
            if model.kind == "text":
                ratio = getattr(model, "unique_ratio", float("nan"))
                detail = (
                    "holds unhashable values"
                    if np.isnan(ratio)
                    else f"has {ratio:.0%} unique text values (more than {self.text_threshold:.0%})"
                )
                warnings.warn(
                    f"Column {model.name!r} {detail}; it is sampled with replacement from "
                    "the original values instead of being modelled. Pass a higher "
                    "text_threshold to Synthesizer to model it as a category.",
                    UserWarning,
                    stacklevel=2,
                )
        logger.info(
            "Fitted %d columns on %d rows (%d in the copula, %d resampled text, %d constant)",
            len(models),
            self.n_rows_,
            k,
            len(self.high_cardinality_columns_),
            sum(m.kind == "constant" for m in models),
        )
        return self

    # --------------------------------------------------------------- sample
    def sample(self, n: Optional[int] = None, *, random_state: RandomState = None) -> pd.DataFrame:
        """Draw ``n`` synthetic rows (default: as many as the fitted data).

        Repeated calls with the same seed return the same rows; pass ``random_state``
        to draw a different batch from the same fit.
        """
        self._check_fitted()
        n = self.n_rows_ if n is None else int(n)
        if n < 0:
            raise ValueError("n must be zero or positive")
        if n > 0 and self.n_rows_ == 0:
            raise ValueError(
                "The synthesizer was fitted on an empty DataFrame, so there is nothing to sample from"
            )
        rng = _make_rng(self.random_state if random_state is None else random_state)
        k = self._transform.shape[0]
        if k:
            z = rng.standard_normal((n, k)) @ self._transform.T
            u = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
        else:
            u = np.zeros((n, 0))

        out: Dict[int, pd.Series] = {}
        j = 0
        for position, model in enumerate(self._models):
            if model.modeled:
                values = model.inverse(u[:, j])
                j += 1
            else:
                values = model.resample(n, rng)
            column = _as_series(values, n)
            if model.missing_rate > 0:
                column = column.where(rng.random(n) >= model.missing_rate)
            out[position] = _cast_like(column, model.dtype)

        if out:
            frame = pd.DataFrame(out, index=pd.RangeIndex(n))
        else:
            frame = pd.DataFrame(index=pd.RangeIndex(n))
        frame.columns = self._columns
        return frame

    # ------------------------------------------------------------ reporting
    def summary(self) -> str:
        """Human-readable description of what was learned per column."""
        self._check_fitted()
        preserving = ", ".join(self.preserve) if self.preserve else "dtypes only"
        lines = [
            f"Synthesizer fitted on {self.n_rows_} rows x {len(self._models)} columns "
            f"(preserving: {preserving})"
        ]
        width = max([len(str(m.name)) for m in self._models] + [6])
        lines.append(f"{'column':<{width}}  {'kind':<11}  {'missing':>7}  detail")
        for m in self._models:
            lines.append(
                f"{str(m.name):<{width}}  {m.kind:<11}  {m.missing_rate:>7.1%}  {m.describe()}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe description of the fit."""
        self._check_fitted()
        return {
            "n_rows": self.n_rows_,
            "preserve": list(self.preserve),
            "columns": {
                str(m.name): {
                    "kind": m.kind,
                    "missing_rate": float(m.missing_rate),
                    "detail": m.describe(),
                }
                for m in self._models
            },
            "high_cardinality_columns": [str(c) for c in self.high_cardinality_columns_],
        }

    def _check_fitted(self) -> None:
        if self._columns is None:
            raise RuntimeError("Call fit() before sample()")


def generate(
    df: TableLike,
    n: Optional[int] = None,
    *,
    random_state: RandomState = 0,
    preserve: Union[str, Iterable[str]] = DEFAULT_PRESERVE,
) -> pd.DataFrame:
    """Return ``n`` synthetic rows that look like ``df`` (default ``n``: ``len(df)``).

    Output columns and dtypes match the input. See :class:`Synthesizer` for the
    meaning of ``preserve``.
    """
    return Synthesizer(random_state=random_state, preserve=preserve).fit(df).sample(n)
