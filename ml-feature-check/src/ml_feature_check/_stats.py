"""Numeric building blocks: correlations, Cramer's V and leave-one-out mapping scores.

Everything here is warning-free by construction: zero-variance inputs yield NaN
(or ``None``) explicitly instead of tripping numpy's divide/invalid warnings.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def pairwise_pearson(
    X: np.ndarray,
    *,
    min_periods: int = 10,
    chunk_rows: Optional[int] = None,
    return_counts: bool = False,
):
    """Exact pairwise-complete Pearson correlation matrix of the columns of ``X``.

    Rows are only used for a pair when both values are present, like
    ``DataFrame.corr()``, but the work is done with matrix products so it stays
    fast for hundreds of columns. Entries with fewer than ``min_periods``
    shared rows, or with zero variance on the shared rows, are NaN.

    With ``return_counts=True`` the shared-row count matrix is returned beside
    the correlations, so a caller can quote the evidence a coefficient rests on.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    n, k = X.shape
    if k == 0:
        empty = np.zeros((0, 0))
        return (empty, empty.copy()) if return_counts else empty
    step = chunk_rows or max(1, 2_000_000 // k)

    # pass 1: per-column means over present values (used only to centre, for stability)
    total = np.zeros(k)
    count = np.zeros(k)
    for start in range(0, n, step):
        block = X[start : start + step]
        present = np.isfinite(block)
        total += np.where(present, block, 0.0).sum(axis=0)
        count += present.sum(axis=0)
    mu = np.divide(total, count, out=np.zeros(k), where=count > 0)

    # pass 2: N (shared counts), S (sum x_i over rows shared with j),
    # Q (sum x_i^2 over rows shared with j) and P (sum x_i * x_j)
    N = np.zeros((k, k))
    S = np.zeros((k, k))
    Q = np.zeros((k, k))
    P = np.zeros((k, k))
    for start in range(0, n, step):
        block = X[start : start + step]
        present = np.isfinite(block)
        m = present.astype(np.float64)
        xc = np.where(present, block - mu, 0.0)
        N += m.T @ m
        S += xc.T @ m
        Q += (xc * xc).T @ m
        P += xc.T @ xc

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = P - S * S.T / N
        var = Q - S * S / N
        denom = np.sqrt(np.maximum(var * var.T, 0.0))
        corr = cov / denom
    valid = (N >= max(min_periods, 2)) & (denom > 0) & np.isfinite(corr)
    corr = np.where(valid, corr, np.nan)
    corr = np.clip(corr, -1.0, 1.0)
    if return_counts:
        return corr, N
    return corr


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson and Spearman correlation of two 1-d arrays without missing values."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    rx = pd.Series(x).rank().to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank().to_numpy(dtype=np.float64)
    return _pearson(x, y), _pearson(rx, ry)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = float((xc * xc).sum())
    syy = float((yc * yc).sum())
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    r = float((xc * yc).sum()) / float(np.sqrt(sxx * syy))
    if not np.isfinite(r):
        return float("nan")
    return float(np.clip(r, -1.0, 1.0))


def cramers_v(a: np.ndarray, b: np.ndarray, n_a: int, n_b: int, *, dense_limit: int = 2_000_000) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013) between two integer-coded arrays.

    ``a`` takes codes in ``[0, n_a)`` and ``b`` in ``[0, n_b)``; ``-1`` marks a
    missing value and is dropped pairwise. Returns NaN when either column has
    fewer than two categories on the shared rows, or when there are too few rows
    for the bias correction (roughly: more categories than rows).
    """
    ok = (a >= 0) & (b >= 0)
    a = a[ok].astype(np.int64)
    b = b[ok].astype(np.int64)
    n = int(a.size)
    if n < 2 or n_a < 2 or n_b < 2:
        return float("nan")
    if n_a * n_b <= dense_limit:
        table = np.bincount(a * n_b + b, minlength=n_a * n_b).reshape(n_a, n_b).astype(np.float64)
        rows = table.sum(axis=1)
        cols = table.sum(axis=0)
        table = table[rows > 0][:, cols > 0]
        rows = rows[rows > 0]
        cols = cols[cols > 0]
        r, c = table.shape
        if r < 2 or c < 2:
            return float("nan")
        expected = np.outer(rows, cols) / n
        chi2 = float(((table - expected) ** 2 / expected).sum())
    else:  # sparse: only observed cells, the unobserved ones contribute their expectation
        rows = np.bincount(a, minlength=n_a).astype(np.float64)
        cols = np.bincount(b, minlength=n_b).astype(np.float64)
        r = int((rows > 0).sum())
        c = int((cols > 0).sum())
        if r < 2 or c < 2:
            return float("nan")
        cell, observed = np.unique(a * n_b + b, return_counts=True)
        expected = rows[cell // n_b] * cols[cell % n_b] / n
        chi2 = float(((observed - expected) ** 2 / expected).sum() + n - expected.sum())
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    ccorr = c - (c - 1) ** 2 / (n - 1)
    denom = min(rcorr - 1.0, ccorr - 1.0)
    if denom <= 0.0:
        return float("nan")
    return float(min(1.0, np.sqrt(phi2corr / denom)))


def loo_mapping_accuracy(f: np.ndarray, y: np.ndarray, n_classes: int) -> Optional[Tuple[float, float, int, int]]:
    """Leave-one-out accuracy of predicting class ``y`` from feature code ``f``.

    Each row is predicted by the majority class of the *other* rows sharing its
    feature value (a row alone in its group gets the overall majority), so an
    identifier column cannot score well. Codes ``< 0`` are missing and dropped.
    Returns ``(accuracy, majority_fraction, n_rows, n_groups)`` or ``None``.
    """
    ok = (f >= 0) & (y >= 0)
    f = f[ok]
    y = y[ok].astype(np.int64)
    n = int(f.size)
    if n < 2:
        return None
    _, f = np.unique(f, return_inverse=True)
    f = f.astype(np.int64)
    g = int(f.max()) + 1
    n_classes = int(max(n_classes, int(y.max()) + 1))
    cell, cnt = np.unique(f * n_classes + y, return_counts=True)
    grp = cell // n_classes
    cls = cell % n_classes
    order = np.lexsort((-cnt, grp))  # group ascending, then count descending
    grp, cnt, cls = grp[order], cnt[order], cls[order]
    first = np.ones(len(grp), dtype=bool)
    first[1:] = grp[1:] != grp[:-1]
    pos1 = np.flatnonzero(first)  # one entry per group, in group order
    top1_cnt = cnt[pos1]
    top1_cls = cls[pos1]
    pos2 = pos1 + 1
    has2 = pos2 < len(grp)
    has2[has2] = grp[pos2[has2]] == grp[pos1[has2]]
    top2_cnt = np.zeros(g, dtype=np.int64)
    top2_cnt[has2] = cnt[pos2[has2]]
    top2_cls = np.zeros(g, dtype=np.int64)
    top2_cls[has2] = cls[pos2[has2]]

    t1c = top1_cnt[f]
    t2c = top2_cnt[f]
    is_top = y == top1_cls[f]
    others_top = t1c - 1
    class_counts = np.bincount(y, minlength=n_classes)
    majority_cls = int(np.argmax(class_counts))
    singleton = (t1c == 1) & (t2c == 0)
    # Only a row whose class leads its group can be predicted correctly once it is
    # removed, and it needs a strict lead over the runner-up: counting a post-removal
    # tie as correct would resolve every tie in the majority's favour and report an
    # accuracy above the true leave-one-out number. A tie goes to the lower class
    # code, which is how the group leaders were ordered in the first place.
    ahead = others_top > t2c
    tied_and_lower = (others_top == t2c) & (top1_cls[f] < top2_cls[f])
    correct = np.where(singleton, y == majority_cls, is_top & (ahead | tied_and_lower))
    return float(correct.mean()), float(class_counts.max() / n), n, g


def loo_mapping_r2(f: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, int, int]]:
    """Leave-one-out R^2 of predicting ``y`` by the mean of its feature group.

    Rows alone in their group are predicted by the overall mean, so identifier
    columns score ~0. Returns ``(r2, n_rows, n_groups)`` or ``None`` when the
    target has no variance on the shared rows.
    """
    ok = (f >= 0) & np.isfinite(y)
    f = f[ok]
    y = y[ok].astype(np.float64)
    n = int(f.size)
    if n < 2:
        return None
    _, f = np.unique(f, return_inverse=True)
    g = int(f.max()) + 1
    cnt = np.bincount(f, minlength=g).astype(np.float64)
    total = np.bincount(f, weights=y, minlength=g)
    others_cnt = cnt[f] - 1.0
    others_sum = total[f] - y
    mean = float(y.mean())
    pred = np.where(others_cnt > 0, others_sum / np.maximum(others_cnt, 1.0), mean)
    ss_tot = float(((y - mean) ** 2).sum())
    if ss_tot <= 0.0:
        return None
    ss_res = float(((y - pred) ** 2).sum())
    return float(1.0 - ss_res / ss_tot), n, g
