"""The one call that does everything: :func:`detect`, plus the :class:`Detector`
class for callers who want to keep the same settings across many tables.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._channel import ChannelScan, scan_channel
from ._faults import explain, runs_of
from ._io import TableLike, load_table, normalize_columns
from ._joint import detect_joint
from ._result import NOISE_FLOOR_SLACK, ChannelResult, Event, SensorReport, at_noise_floor

logger = logging.getLogger(__name__)

#: Column names that suggest a time axis rather than a sensor reading.
TIME_NAMES = frozenset({"time", "timestamp", "datetime", "date", "ts", "index"})

#: Keep the report readable even on a table that is anomalous end to end.
MAX_EVENTS = 1000

#: How much each kind of event matters, regardless of how big its number is.
#: A dead sensor outranks a loud spike: the spike may be a real reading, the dead
#: sensor never is. Severities are only comparable within a tier, so events sort
#: by tier first and severity second.
_KIND_TIER = {
    "stuck-at-zero": 3,
    "flatline": 3,
    "railed": 3,
    "step": 2,
    "missing": 2,
    "joint": 1,
    "spike": 0,
}

#: A channel must stray at least this far from its own level, in robust sigmas,
#: before a cross-channel event names it as a contributor.
JOINT_ATTRIBUTION_SIGMAS = 2.0

#: At most this many channels are named on one cross-channel event.
JOINT_ATTRIBUTION_MAX = 4


def _event_order(event: Event) -> tuple:
    """Sort key: what matters most first, then how far past threshold it went."""
    return (_KIND_TIER.get(event.kind, 0), event.severity, -event.start)


def _check_sensitivity(sensitivity: float) -> float:
    try:
        value = float(sensitivity)
    except (TypeError, ValueError):
        raise ValueError(
            "sensitivity must be a positive number of robust sigmas, not %r" % (sensitivity,)
        ) from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            "sensitivity must be a positive, finite number of robust sigmas, got %r"
            % (sensitivity,)
        )
    return value


def _check_contamination(contamination: Any) -> Any:
    if isinstance(contamination, str):
        if contamination != "auto":
            raise ValueError(
                'contamination must be "auto" or a fraction above 0 and at most 0.5, got %r'
                % (contamination,)
            )
        return "auto"
    try:
        value = float(contamination)
    except (TypeError, ValueError):
        raise ValueError(
            'contamination must be "auto" or a fraction above 0 and at most 0.5, got %r'
            % (contamination,)
        ) from None
    if not np.isfinite(value) or not (0.0 < value <= 0.5):
        raise ValueError(
            "contamination must be above 0 and at most 0.5, got %r" % (contamination,)
        )
    return value


#: Largest seed the cross-channel model accepts, so a bad one is caught here and
#: named as sensor-anomaly's own argument rather than surfacing from inside sklearn.
MAX_RANDOM_STATE = 2**32 - 1


def _check_random_state(random_state: Any) -> Any:
    """Accept only what the cross-channel model can actually seed with.

    Validated here, at the entry point, so the same value gives the same answer on
    a 30-row table as on a 300,000-row one - the short-table fallback never touches
    the seed, so an unchecked bad value used to sail straight through it.
    """
    if random_state is None or isinstance(random_state, np.random.RandomState):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        value = int(random_state)
        if 0 <= value <= MAX_RANDOM_STATE:
            return value
    hint = ""
    if isinstance(random_state, np.random.Generator):
        hint = (
            " A numpy Generator cannot be used as a seed here; pass an integer, or "
            "int(rng.integers(0, 2**32)) to draw one from it."
        )
    raise ValueError(
        "random_state must be None or a whole number from 0 to %d, got %r.%s"
        % (MAX_RANDOM_STATE, random_state, hint)
    )


def _as_name_list(channels: Any) -> Optional[List[str]]:
    if channels is None:
        return None
    if isinstance(channels, str):
        return [channels]
    if isinstance(channels, pd.Index):
        return [str(c) for c in channels]
    if isinstance(channels, Iterable):
        return [str(c) for c in channels]
    raise TypeError(
        "channels must be a column name or a list of column names, not %s"
        % type(channels).__name__
    )


def _available(frame: pd.DataFrame, limit: int = 12) -> str:
    names = [repr(c) for c in frame.columns[:limit]]
    if len(frame.columns) > limit:
        names.append("...")
    return ", ".join(names) if names else "(none)"


def _looks_like_time(frame: pd.DataFrame, name: str) -> bool:
    """True when this column is a time axis rather than a sensor reading."""
    column = frame[name]
    if pd.api.types.is_datetime64_any_dtype(column):
        return True
    if str(name).strip().lower() not in TIME_NAMES:
        return False
    if pd.api.types.is_numeric_dtype(column):
        return bool(column.is_monotonic_increasing)
    try:
        parsed = pd.to_datetime(column, errors="coerce")
    except (TypeError, ValueError):
        return False
    return bool(parsed.notna().mean() >= 0.8)


def _resolve_time(
    frame: pd.DataFrame, time: Optional[str], requested: Optional[List[str]]
) -> Tuple[Optional[str], Optional[pd.Series], List[str]]:
    """Find the time column, if there is one, and never steal a requested channel."""
    notes: List[str] = []
    if time is not None:
        name = str(time)
        if name not in frame.columns:
            raise ValueError(
                "time column %r is not in the table; available columns: %s"
                % (name, _available(frame))
            )
        column = frame[name]
        if not column.is_monotonic_increasing:
            notes.append(
                "The time column %r is not sorted. Rows were analysed in the order "
                "given, which is what row positions in this report refer to." % name
            )
        return name, column, notes

    blocked = set(requested or [])
    for name in frame.columns:
        if name in blocked:
            continue
        if _looks_like_time(frame, name):
            notes.append(
                "Column %r looks like the time axis, so it was used for event "
                "timestamps rather than treated as a sensor channel. Pass "
                "time=None, channels=[...] to override." % name
            )
            column = frame[name]
            if not column.is_monotonic_increasing:
                notes.append(
                    "That time column is not sorted; rows were analysed in the order given."
                )
            return name, column, notes
    return None, None, notes


def _resolve_channels(
    frame: pd.DataFrame, requested: Optional[List[str]], time_col: Optional[str]
) -> Tuple[List[str], List[str]]:
    """Decide which columns are sensor channels, explaining anything skipped."""
    notes: List[str] = []
    if requested is None and len(frame) == 0:
        # With no rows there is no dtype evidence either way - a CSV export with
        # only a header gives every column object dtype - so saying they "held no
        # numbers" would be a wrong diagnosis of a table whose real problem is
        # that it is empty. Keep the columns; detect() reports the emptiness.
        return [c for c in frame.columns if c != time_col], notes
    if requested is not None:
        missing = [name for name in requested if name not in frame.columns]
        if missing:
            raise ValueError(
                "channels not found in the table: %s; available columns: %s"
                % (", ".join(repr(m) for m in missing), _available(frame))
            )
        duplicates = sorted({n for n, k in Counter(requested).items() if k > 1})
        if duplicates:
            raise ValueError(
                "channels lists the same column twice: %s"
                % ", ".join(repr(d) for d in duplicates)
            )
        return list(requested), notes

    candidates = [c for c in frame.columns if c != time_col]
    numeric = [
        c
        for c in candidates
        if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
    ]
    skipped = [c for c in candidates if c not in numeric]

    if not numeric and candidates:
        # Fall back rather than return nothing: text columns sometimes hold numbers.
        rescued = []
        for name in candidates:
            try:
                parsed = pd.to_numeric(frame[name], errors="coerce")
            except (TypeError, ValueError):
                continue
            if len(parsed) and float(parsed.notna().mean()) >= 0.5:
                rescued.append(name)
        if rescued:
            notes.append(
                "No column had a numeric dtype, so %d text column%s that parse as "
                "numbers were used as channels: %s."
                % (
                    len(rescued),
                    "" if len(rescued) == 1 else "s",
                    ", ".join(repr(r) for r in rescued[:8]),
                )
            )
            return rescued, notes

    if skipped:
        notes.append(
            "%d column%s held no numbers and %s skipped: %s."
            % (
                len(skipped),
                "" if len(skipped) == 1 else "s",
                "was" if len(skipped) == 1 else "were",
                ", ".join(repr(s) for s in skipped[:8])
                + (", ..." if len(skipped) > 8 else ""),
            )
        )
    if not numeric:
        notes.append(
            "No usable sensor channels were found. Name them explicitly with "
            "channels=[...] if the readings are in text columns."
        )
    return numeric, notes


def _merge_spans(
    spans: List[Tuple[int, int, List[str], float]]
) -> List[Tuple[int, int, List[str], float]]:
    """Merge overlapping or touching spans, unioning their channels."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda item: (item[0], item[1]))
    merged: List[Tuple[int, int, List[str], float]] = []
    start, end, names, severity = spans[0][0], spans[0][1], list(spans[0][2]), spans[0][3]
    for next_start, next_end, next_names, next_severity in spans[1:]:
        if next_start <= end + 1:
            end = max(end, next_end)
            for name in next_names:
                if name not in names:
                    names.append(name)
            severity = max(severity, next_severity)
        else:
            merged.append((start, end, names, severity))
            start, end, names, severity = next_start, next_end, list(next_names), next_severity
    merged.append((start, end, names, severity))
    return merged


def _build_events(
    scans: Sequence[ChannelScan],
    joint_positions: np.ndarray,
    joint_severity: np.ndarray,
    n_rows: int,
    time_values: Optional[pd.Series],
    sensitivity: float,
) -> Tuple[List[Event], int]:
    """Turn flagged rows into merged, human-sized events, worst first."""
    by_kind: Dict[str, List[Tuple[int, int, List[str], float]]] = {}

    for scan in scans:
        if scan.anomalies.size:
            flagged = np.zeros(n_rows, dtype=bool)
            flagged[scan.anomalies] = True
            for start, end in runs_of(flagged):
                by_kind.setdefault("spike", []).append(
                    (start, end, [scan.name], _spike_severity(scan, start, end, sensitivity))
                )
        for fault in scan.fault_events:
            by_kind.setdefault(fault.kind, []).append(
                (int(fault.start), int(fault.end), [scan.name], float(fault.severity))
            )

    if joint_positions.size:
        flagged = np.zeros(n_rows, dtype=bool)
        flagged[joint_positions] = True
        severity_by_row = np.zeros(n_rows, dtype=float)
        severity_by_row[joint_positions] = joint_severity
        for start, end in runs_of(flagged):
            # Name the channels that actually drove this row away from normal,
            # not merely the ones that happened to be flagged on their own.
            contributions = sorted(
                (
                    (scan.deviation(start, end), scan.name)
                    for scan in scans
                    if scan.usable_for_joint
                ),
                reverse=True,
            )
            involved = [
                name
                for value, name in contributions[:JOINT_ATTRIBUTION_MAX]
                if value >= JOINT_ATTRIBUTION_SIGMAS
            ]
            peak = float(np.max(severity_by_row[start : end + 1]))
            by_kind.setdefault("joint", []).append((start, end, involved, peak))

    events: List[Event] = []
    for kind, spans in by_kind.items():
        for start, end, names, severity in _merge_spans(spans):
            events.append(
                Event(
                    start=int(start),
                    end=int(end),
                    channels=list(names),
                    kind=kind,
                    severity=float(severity),
                )
            )

    events.sort(key=_event_order, reverse=True)
    total = len(events)
    if total > MAX_EVENTS:
        events = events[:MAX_EVENTS]

    if time_values is not None and len(time_values):
        for event in events:
            try:
                event.start_time = time_values.iloc[event.start]
                event.end_time = time_values.iloc[event.end]
            except (IndexError, KeyError):  # pragma: no cover - defensive
                event.start_time = event.end_time = None
    return events, total


def _spike_severity(scan: ChannelScan, start: int, end: int, sensitivity: float) -> float:
    window = np.abs(scan.scores[start : end + 1])
    if not window.size or not np.isfinite(window).any():
        return 1.0
    return float(np.nanmax(window)) / sensitivity


def detect(
    df: TableLike,
    *,
    time: Optional[str] = None,
    channels: Optional[Sequence[str]] = None,
    sensitivity: float = 3.0,
    contamination: Any = "auto",
    random_state: Any = 0,
) -> SensorReport:
    """Find abnormal behaviour across many sensor channels in one pass.

    Three things happen, and all three land in the same report:

    1. **Per channel** - each channel is scored against its own rolling level with
       a robust z-score, so drift and cycles do not drown the result.
    2. **Across channels** - each channel is also compared with what the other
       channels predict for it, which catches rows where no single reading is odd
       but the *combination* never happens: the classic broken-correlation fault.
       The threshold is corrected for the size of the table, so a healthy plant of
       redundant sensors comes back with nothing flagged.
    3. **Sensor faults** - flatlines, sensors stuck at zero, channels railed at
       their measurement limits, bursts of missing samples and sudden step changes.
       None of these are statistical outliers, and all of them mean a broken sensor.

    Args:
        df: a wide DataFrame with one column per channel, or a path to a
            ``.csv`` / ``.tsv`` / ``.parquet`` file. Never modified.
        time: name of the timestamp column, if the table has one. Left as ``None``
            an obvious time column is detected and reported; it is used for event
            timestamps and is never treated as a sensor channel.
        channels: which columns are sensor channels. Left as ``None``, every
            numeric column is used and anything skipped is named in the notes.
        sensitivity: per-channel threshold in robust sigmas. Higher means fewer
            flags. 3.0 is a good default; 4.0 or 5.0 for noisy plant data.
        contamination: ``"auto"``, or the share of rows you expect to be
            anomalous, passed to the cross-channel model.
        random_state: seed for the cross-channel model, so the same table always
            gives the same answer.

    Returns:
        A :class:`~sensor_anomaly.SensorReport`. Call ``.summary()`` to read it,
        ``.to_frame()`` for a per-channel table, ``.to_dict()`` for JSON.

    Raises:
        ValueError: duplicate column names, an unknown ``time`` or ``channels``
            name, or an out-of-range ``sensitivity``, ``contamination`` or
            ``random_state``.

    Example:
        >>> import pandas as pd, sensor_anomaly
        >>> frame = pd.DataFrame({"a": [1.0, 1.1, 0.9, 9.0], "b": [2.0, 2.1, 1.9, 2.0]})
        >>> report = sensor_anomaly.detect(frame)
        >>> report.n_rows
        4
    """
    sensitivity = _check_sensitivity(sensitivity)
    contamination = _check_contamination(contamination)
    random_state = _check_random_state(random_state)
    requested = _as_name_list(channels)

    frame = load_table(df, "df")
    frame, notes = normalize_columns(frame)
    requested = [str(name) for name in requested] if requested is not None else None
    n_rows = int(len(frame))

    time_col, time_values, time_notes = _resolve_time(frame, time, requested)
    notes.extend(time_notes)
    channel_names, channel_notes = _resolve_channels(frame, requested, time_col)
    notes.extend(channel_notes)

    params: Dict[str, Any] = {
        "sensitivity": float(sensitivity),
        "contamination": contamination,
        "random_state": random_state,
        "n_channels": len(channel_names),
        "time_column": time_col,
    }

    if n_rows == 0:
        notes.append("The table has no rows, so there was nothing to score.")
        return SensorReport(
            channels={name: ChannelResult() for name in channel_names},
            joint=[],
            events=[],
            notes=notes,
            n_rows=0,
            method="none",
            params=params,
            time_column=time_col,
        )

    scans = [scan_channel(name, frame[name], sensitivity) for name in channel_names]
    params["window"] = int(scans[0].window) if scans else 0

    joint = detect_joint(
        scans,
        sensitivity=sensitivity,
        contamination=contamination,
        random_state=random_state,
    )
    notes.extend(joint.notes)
    params["joint_channels"] = len(joint.channels)

    events, total_events = _build_events(
        scans, joint.positions, joint.severity, n_rows, time_values, sensitivity
    )
    if total_events > MAX_EVENTS:
        notes.append(
            "%s events were found; the %d worst are listed. Raise sensitivity to "
            "see fewer." % (format(total_events, ","), MAX_EVENTS)
        )
    params["events_found"] = int(total_events)

    results: Dict[str, ChannelResult] = {}
    all_faults: List[str] = []
    for scan in scans:
        all_faults.extend(scan.faults)
        results[scan.name] = ChannelResult(
            anomalies=[int(i) for i in scan.anomalies],
            rate=float(scan.rate),
            score_max=scan.score_max,
            faults=list(scan.faults),
            n_valid=int(scan.n_valid),
            n_missing=int(scan.n_missing),
            notes=list(scan.notes),
        )

    # At sensitivity 3.0 a perfectly healthy channel still trips about 0.27% of
    # the time: that is what a 3-sigma test means. Say so, so a handful of flags
    # on clean data does not read as a finding. The same judgement drives
    # SensorReport.anomalous, so the boolean and this sentence cannot disagree.
    expected = float(math.erfc(sensitivity / math.sqrt(2.0)))
    params["expected_false_rate"] = expected
    joint_rows = [int(i) for i in joint.positions]
    if events and at_noise_floor(results, events, joint_rows, n_rows, expected):
        notes.append(
            "Nothing here stands out. At sensitivity %.1f about %.2f%% of perfectly "
            "normal readings are expected to be flagged by chance, and every channel "
            "is within %.0f times that rate, so these are very likely noise rather "
            "than faults. Raise sensitivity to quieten them."
            % (sensitivity, 100.0 * expected, NOISE_FLOOR_SLACK)
        )

    failed = [name for name, res in results.items() if res.failed]
    if failed:
        notes.append(
            "%d channel%s produced no usable readings and %s reported as a failed "
            "sensor rather than scored: %s."
            % (
                len(failed),
                "" if len(failed) == 1 else "s",
                "is" if len(failed) == 1 else "are",
                ", ".join(repr(f) for f in failed[:8])
                + (", ..." if len(failed) > 8 else ""),
            )
        )
    notes.extend(explain(sorted(set(all_faults))))

    report = SensorReport(
        channels=results,
        joint=joint_rows,
        events=events,
        notes=notes,
        n_rows=n_rows,
        method=joint.method,
        params=params,
        time_column=time_col,
    )
    logger.debug(
        "sensor-anomaly: %d channels, %d rows, %d events, method=%s",
        len(results),
        n_rows,
        len(events),
        joint.method,
    )
    return report


@dataclass
class Detector:
    """The same detection with settings you set once and reuse.

    Handy when many tables come off the same plant and must be judged the same way::

        detector = Detector(sensitivity=4.0, time="timestamp")
        for batch in batches:
            print(detector.detect(batch).summary())

    Attributes mirror the keyword arguments of :func:`detect`.
    """

    time: Optional[str] = None
    channels: Optional[Sequence[str]] = None
    sensitivity: float = 3.0
    contamination: Any = "auto"
    random_state: Any = 0

    def detect(self, df: TableLike) -> SensorReport:
        """Run detection on ``df`` with this detector's settings."""
        return detect(
            df,
            time=self.time,
            channels=self.channels,
            sensitivity=self.sensitivity,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    __call__ = detect
