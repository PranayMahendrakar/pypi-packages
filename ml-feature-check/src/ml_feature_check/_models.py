"""scikit-learn helpers: the single-feature stump and the small random forest."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def stump_score(
    x: np.ndarray, y: np.ndarray, *, classification: bool, depth: int = 1, random_state: int = 0
) -> Optional[Tuple[float, float]]:
    """2-fold cross-fitted score of a shallow tree that sees only feature ``x``.

    Returns ``(score, baseline)`` where score is accuracy (classification) or R^2
    (regression) on rows the tree did not train on, and baseline is the
    majority-class fraction or 0.0. ``None`` when there is too little to learn from.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = int(len(y))
    if n < 4:
        return None
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    half = n // 2
    folds = (perm[:half], perm[half:])
    correct = 0.0
    ss_res = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        for train, test in ((folds[0], folds[1]), (folds[1], folds[0])):
            if classification:
                model = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
            else:
                model = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
            model.fit(x[train], y[train])
            pred = model.predict(x[test])
            if classification:
                correct += float((pred == y[test]).sum())
            else:
                ss_res += float(((pred - y[test]) ** 2).sum())
    if classification:
        counts = np.bincount(np.asarray(y, dtype=np.int64))
        return correct / n, float(counts.max() / n)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return None
    return 1.0 - ss_res / ss_tot, 0.0


def forest_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    random_state: int = 0,
    n_estimators: int = 60,
    max_depth: int = 8,
    n_repeats: int = 10,
    n_shadow: int = 5,
    test_frac: float = 0.3,
) -> Dict[str, object]:
    """Fit a small random forest and measure holdout permutation importance.

    Shuffled copies of up to ``n_shadow`` real columns ("shadow" features) are
    added so the caller can compare every real column against pure noise. The
    result has ``score`` and ``baseline`` on the holdout rows, and, when the
    forest beats a constant prediction, ``importances`` (one per real column)
    and ``noise_ceiling`` (the best shadow importance). Otherwise
    ``importances`` is ``None``.
    """
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    rng = np.random.default_rng(random_state)
    n_shadow = int(min(n_shadow, k))
    shadow_src = np.unique(np.linspace(0, k - 1, n_shadow).round().astype(int)) if n_shadow else np.array([], int)
    shadows = [rng.permutation(X[:, i]) for i in shadow_src]
    Xs = np.column_stack([X] + shadows) if shadows else X
    perm = rng.permutation(n)
    n_test = max(int(round(n * test_frac)), 5)
    test, train = perm[:n_test], perm[n_test:]
    seed = int(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        if classification:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=2, n_jobs=1, random_state=seed
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=2, n_jobs=1, random_state=seed
            )
        model.fit(Xs[train], y[train])
        score = float(model.score(Xs[test], y[test]))
        if classification:
            baseline = float(np.bincount(np.asarray(y[test], dtype=np.int64)).max() / n_test)
        else:
            baseline = 0.0
        if not np.isfinite(score) or score <= baseline:
            return {"score": score, "baseline": baseline, "importances": None, "noise_ceiling": None}
        result = permutation_importance(
            model, Xs[test], y[test], n_repeats=n_repeats, random_state=seed, n_jobs=1
        )
    means = np.asarray(result.importances_mean, dtype=np.float64)
    ceiling = float(max(means[k:].max(), 0.0)) if len(shadows) else 0.0
    return {"score": score, "baseline": baseline, "importances": means[:k], "noise_ceiling": ceiling}
