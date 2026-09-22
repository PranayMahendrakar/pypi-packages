"""Cross-channel detection: the faults that no single channel shows.

A pump whose flow drops while its current climbs may keep both readings inside
their normal ranges. Nothing is an outlier on its own; the *combination* is what
never happens. An IsolationForest over the standardized channels finds those rows.

IsolationForest cannot see a NaN, so channels are imputed with their own median
before fitting. That is a real modelling decision, not a detail, so the report
says how many readings were filled and in which channels.

Where the *cut* goes is a separate question from what does the scoring, and it is
the one that decides whether a healthy plant reads as healthy. See
:func:`cross_channel_cut`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ._channel import ChannelScan
from ._faults import robust_sigma

#: Below this many rows an IsolationForest has too little to isolate, so a
#: covariance-based distance is used instead.
MIN_FOREST_ROWS = 50

#: Trees in the forest. Large tables use fewer so a 100k-row table stays quick.
FOREST_TREES = 100
FOREST_TREES_LARGE = 64
LARGE_TABLE_ROWS = 20_000

#: Above this many channels the partial-residual features cost more than they are
#: worth (the work grows with the cube of the channel count), so they are skipped.
MAX_RESIDUAL_CHANNELS = 60

#: With contamination="auto" the cross-channel stage never flags more than this
#: share of rows. Pass a contamination number to ask for a different share.
JOINT_MAX_RATE = 0.01

#: Where a forest score sits on the severity scale: ``JOINT_SIGMA_FACTOR *
#: sensitivity`` robust sigmas of the score counts as severity 1.0. This is a
#: reporting scale only. It used to be the cut as well, and that was wrong: see
#: :func:`cross_channel_cut`.
JOINT_SIGMA_FACTOR = 2.0

#: Bisection bound for :func:`_gaussian_cut`; ``erfc`` underflows past this.
_MAX_SIGMA = 38.0
_SQRT2 = math.sqrt(2.0)


def _gaussian_cut(tail: float) -> float:
    """How many sigmas leave ``tail`` probability outside a two-sided Gaussian.

    Bisection on :func:`math.erfc`, which stays accurate far into the tail where
    the interesting thresholds live. Pure standard library, no SciPy.
    """
    if tail >= 1.0:
        return 0.0
    if tail <= 0.0:
        return _MAX_SIGMA
    low, high = 0.0, _MAX_SIGMA
    for _ in range(90):
        middle = 0.5 * (low + high)
        if math.erfc(middle / _SQRT2) > tail:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def cross_channel_cut(n_cells: int, sensitivity: float) -> float:
    """The cross-channel threshold in robust sigmas, corrected for table size.

    ``sensitivity`` means the same thing here as it does per channel - about
    ``erfc(sensitivity / sqrt(2))`` of clean readings trip it - but the question
    the cross-channel stage asks is different. Per channel the caller sees one
    rate per channel and expects a few flags. Across channels the caller sees one
    verdict about the whole table, so the level that matters is the chance the
    *whole clean table* trips at all. A table of 100,000 rows by 20 channels gives
    itself two million chances; uncorrected, it takes one somewhere every time.

    So the per-reading level is ``sensitivity``'s tail spread over every cell:
    a clean table has about an ``erfc(sensitivity / sqrt(2))`` chance of producing
    a single cross-channel flag, whatever its size or shape. Never below
    ``sensitivity`` itself, so the cross-channel stage is never looser than the
    per-channel one.

    Args:
        n_cells: rows times channels - how many chances the table gives itself.
        sensitivity: the caller's threshold in robust sigmas.

    Returns:
        The cut in robust sigmas. Around 5.0 for a small table, 6.1 for two
        million cells, growing only with the logarithm of the table.
    """
    family = math.erfc(float(sensitivity) / _SQRT2)
    return max(float(sensitivity), _gaussian_cut(family / max(int(n_cells), 1)))


def row_disagreement(block: np.ndarray) -> np.ndarray:
    """Per row, how far the worst channel strays from what the others predict.

    Columns of ``block`` are unit-variance by construction, which is what makes
    :func:`cross_channel_cut` comparable across tables. An ill-conditioned channel
    covariance can still leave one over-dispersed, so a column whose robust spread
    is above 1 is divided by it. A column below 1 is left alone: shrinking a cut
    is safe, inflating it is not.
    """
    if block.ndim != 2 or block.size == 0:
        return np.zeros(block.shape[0] if block.ndim == 2 else 0, dtype=float)
    scaled = np.abs(np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0))
    for index in range(scaled.shape[1]):
        spread = max(robust_sigma(block[:, index]), 1.0)
        if spread > 1.0:
            scaled[:, index] /= spread
    return scaled.max(axis=1)


@dataclass
class JointResult:
    """What the cross-channel stage found."""

    positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    severity: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    method: str = "none"
    channels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    """Zero mean, unit spread per channel, without importing a scaler for two lines."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(matrix)


def partial_residuals(standardized: np.ndarray) -> np.ndarray:
    """Each channel's residual against what the other channels predict for it.

    This is the feature that makes a broken correlation visible. An IsolationForest
    splits one feature at a time, so on raw channels it can only find a row that is
    extreme along some axis - and the interesting sensor faults are not. A pump
    whose flow drops while its current climbs leaves both readings inside their
    normal ranges; only the *relationship* broke.

    For a row ``z`` the residual of channel ``j`` given all the others is
    ``(z @ Theta)_j / sqrt(Theta_jj)``, where ``Theta`` is the inverse covariance.
    That is one matrix product for every channel at once, and it turns a broken
    relationship into exactly the kind of single-axis outlier the forest is good at.
    """
    n_cols = standardized.shape[1]
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
    ridge = max(1e-9, 1e-6 * float(np.trace(covariance)) / max(n_cols, 1))
    precision = np.linalg.pinv(covariance + ridge * np.eye(n_cols))
    spread = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    residuals = (standardized @ precision) / spread
    return np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)


def _severity_from_scores(scores: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Turn decision-function values into severities where 1.0 sits on the threshold."""
    if selected.size == 0:
        return np.empty(0, dtype=float)
    spread = robust_sigma(scores)
    if spread <= 0.0:
        spread = float(np.std(scores)) or 1.0
    return 1.0 + np.abs(scores[selected]) / spread


def _mahalanobis(
    matrix: np.ndarray, sensitivity: float, contamination: Any
) -> "tuple[np.ndarray, np.ndarray, List[str]]":
    """Distance from the joint centre, for tables too short to grow a forest."""
    notes: List[str] = []
    n_rows, n_cols = matrix.shape
    centre = np.median(matrix, axis=0)
    centred = matrix - centre
    covariance = np.cov(centred, rowvar=False)
    covariance = np.atleast_2d(covariance)
    trace = float(np.trace(covariance))
    ridge = max(1e-9, 1e-6 * (trace / max(n_cols, 1)))
    covariance = covariance + ridge * np.eye(n_cols)
    inverse = np.linalg.pinv(covariance)
    squared = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    distance = np.sqrt(np.clip(squared, 0.0, None))

    if isinstance(contamination, float):
        cutoff = float(np.quantile(distance, max(0.0, 1.0 - contamination)))
        selected = np.flatnonzero(distance > cutoff)
        scale = robust_sigma(distance) or 1.0
        severity = 1.0 + np.abs(distance[selected] - cutoff) / scale
        return selected, severity, notes

    spread = robust_sigma(distance)
    if spread <= 0.0:
        notes.append(
            "The cross-channel distances were all identical, so no row stood out."
        )
        return np.empty(0, dtype=int), np.empty(0, dtype=float), notes
    centre_distance = float(np.median(distance))
    z = (distance - centre_distance) / spread
    selected = np.flatnonzero(z >= float(sensitivity))
    severity = z[selected] / float(sensitivity)
    return selected, severity, notes


def _joint_severity(
    forest: Any,
    features: np.ndarray,
    disagreement: np.ndarray,
    selected: np.ndarray,
    cut_sigmas: float,
    sensitivity: float,
    result: JointResult,
) -> "Tuple[np.ndarray, bool]":
    """Severity for the rows the calibrated cut kept, and a note when it kept none.

    1.0 sits exactly on the threshold. A row is worse when it disagrees with the
    other channels by more, and worse again when the forest found it easier to
    isolate than the rest of the table, so both models show in the number.

    Returns the severities and whether it already explained an empty result, so
    the caller does not follow a specific reason with a vague one.
    """
    unusual = -forest.score_samples(features)
    spread = robust_sigma(unusual)
    forest_cut = JOINT_SIGMA_FACTOR * float(sensitivity)
    if spread > 0.0:
        z = (unusual - float(np.median(unusual))) / spread
    else:
        z = np.zeros_like(unusual)

    if selected.size == 0:
        ranked = int(np.count_nonzero(z >= forest_cut))
        if ranked:
            result.notes.append(
                "The cross-channel model ranked %s row%s at the top of its own "
                "score, but none of them strayed more than %.1f robust sigmas from "
                "what the other channels predict - the cut a table this size needs "
                "before one reading is more than chance - so none were flagged."
                % (format(ranked, ","), "" if ranked == 1 else "s", cut_sigmas)
            )
        return np.empty(0, dtype=float), bool(ranked)

    from_residual = disagreement[selected] / max(cut_sigmas, 1e-12)
    from_forest = np.clip(z[selected] / max(forest_cut, 1e-12), 0.0, None)
    return np.maximum(from_residual, from_forest), False


def detect_joint(
    scans: Sequence[ChannelScan],
    *,
    sensitivity: float,
    contamination: Any,
    random_state: Any,
) -> JointResult:
    """Run the cross-channel model over every channel that has usable variation.

    Args:
        scans: per-channel scans, already scored.
        sensitivity: threshold in robust sigmas, used by the short-table fallback.
        contamination: ``"auto"`` or the expected share of anomalous rows.
        random_state: seed, so the same table always gives the same answer.

    Returns:
        A :class:`JointResult`. Skipping is normal and is always explained in
        ``notes`` rather than failing.
    """
    usable = [scan for scan in scans if scan.usable_for_joint]
    names = [scan.name for scan in usable]
    result = JointResult(channels=names)

    if len(scans) < 2:
        result.notes.append(
            "Only one sensor channel was given, so the cross-channel stage was "
            "skipped; a fault visible only between channels cannot be found in a "
            "single channel."
        )
        return result
    if len(usable) < 2:
        dropped = [s.name for s in scans if not s.usable_for_joint]
        result.notes.append(
            "The cross-channel stage was skipped: fewer than two channels had "
            "usable variation (skipped %s)." % ", ".join(repr(d) for d in dropped[:8])
        )
        return result

    n_rows = int(usable[0].values.size)
    if n_rows < 3:
        result.notes.append(
            "The cross-channel stage was skipped: %d row%s is not enough to model "
            "how the channels move together." % (n_rows, "" if n_rows == 1 else "s")
        )
        return result

    filled = int(sum(scan.n_missing for scan in usable))
    if filled:
        affected = [scan.name for scan in usable if scan.n_missing]
        result.notes.append(
            "The cross-channel model cannot read a missing value, so %s reading%s "
            "in %d channel%s (%s) were filled with that channel's own median before "
            "fitting. No row was dropped."
            % (
                format(filled, ","),
                "" if filled == 1 else "s",
                len(affected),
                "" if len(affected) == 1 else "s",
                ", ".join(affected[:6]) + (", ..." if len(affected) > 6 else ""),
            )
        )

    matrix = np.column_stack([scan.imputed() for scan in usable])
    standardized = _standardize(matrix)
    if not np.isfinite(standardized).all():
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    if n_rows < MIN_FOREST_ROWS:
        selected, severity, notes = _mahalanobis(standardized, sensitivity, contamination)
        result.positions = selected.astype(int)
        result.severity = severity
        result.method = "mahalanobis"
        result.notes.extend(notes)
        result.notes.append(
            "Only %d rows, too few to grow a useful IsolationForest, so the "
            "cross-channel stage used a covariance distance instead." % n_rows
        )
        result.info = {"n_channels": len(names), "n_rows": n_rows}
        return result

    from sklearn.ensemble import IsolationForest

    features = standardized
    disagreement_block = standardized
    if len(names) <= MAX_RESIDUAL_CHANNELS:
        disagreement_block = partial_residuals(standardized)
        features = np.hstack([standardized, disagreement_block])
    else:
        result.notes.append(
            "%d channels is too many to model every channel against the others, so "
            "the cross-channel stage used the readings alone. A fault visible only "
            "as a broken correlation may be missed; pass a narrower channels=[...] "
            "to get that back." % len(names)
        )

    extra: Dict[str, Any] = {}
    trees = FOREST_TREES if n_rows <= LARGE_TABLE_ROWS else FOREST_TREES_LARGE
    forest = IsolationForest(
        n_estimators=trees,
        max_samples=min(256, n_rows),
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    forest.fit(features)

    if contamination == "auto":
        # Two separate decisions live here, and conflating them is what used to
        # flag 1% of a healthy plant.
        #
        # WHAT SCORES: the forest, over the readings and the partial residuals.
        #
        # WHERE THE CUT GOES: not the forest score's own robust z. That z is not
        # a calibrated quantity. Its denominator is the MAD of the path lengths,
        # which measures the bulk; redundant channels - the normal case in a
        # plant, where sensors are deliberately correlated - squeeze the bulk
        # without squeezing the tail, so the same z means a different thing on
        # every table. Measured on clean tables the top row's z runs about 5 with
        # independent channels and past 9 once the channel covariance is
        # ill-conditioned, so a fixed multiple of sensitivity flagged hundreds of
        # rows of perfectly good data and the JOINT_MAX_RATE ceiling turned into
        # a target. The cut goes on the partial residuals instead: unit-variance
        # by construction whatever the channels do, and corrected for how many
        # chances the table gives itself. See cross_channel_cut.
        cut_sigmas = cross_channel_cut(disagreement_block.size, sensitivity)
        disagreement = row_disagreement(disagreement_block)
        selected = np.flatnonzero(disagreement >= cut_sigmas)
        # A ceiling, never a target: it only ever raises the bar. A clean table
        # comes back empty long before it applies.
        ceiling = n_rows * JOINT_MAX_RATE
        if selected.size > ceiling:
            cut_sigmas = max(
                cut_sigmas,
                float(np.quantile(disagreement, max(0.0, 1.0 - JOINT_MAX_RATE))),
            )
            selected = np.flatnonzero(disagreement >= cut_sigmas)
            result.notes.append(
                "More than %.1f%% of rows looked unusual across channels, so only "
                "the most unusual %s were kept. Pass contamination=<fraction> if you "
                "expect a larger share to be bad."
                % (100.0 * JOINT_MAX_RATE, format(selected.size, ","))
            )
        severity, explained = _joint_severity(
            forest, features, disagreement, selected, cut_sigmas, sensitivity, result
        )
        extra["cut_sigmas"] = float(cut_sigmas)
    else:
        scores = forest.decision_function(features)
        selected = np.flatnonzero(scores < 0.0)
        severity = _severity_from_scores(scores, selected)
        explained = False

    result.positions = selected.astype(int)
    result.severity = severity
    result.method = "isolation-forest"
    result.info = {
        "n_channels": len(names),
        "n_rows": n_rows,
        "n_estimators": trees,
        "max_samples": int(min(256, n_rows)),
    }
    result.info.update(extra)
    if selected.size == 0 and not explained:
        result.notes.append(
            "The cross-channel model found no unusual combination of readings."
        )
    return result
