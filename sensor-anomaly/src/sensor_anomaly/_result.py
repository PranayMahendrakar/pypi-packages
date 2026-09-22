"""Result objects: :class:`ChannelResult`, :class:`Event` and :class:`SensorReport`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

#: Fault codes that mean the channel produced no usable data at all.
FAILED_FAULTS = frozenset({"all-missing", "no-numeric-data"})

#: How a fault kind reads in the summary.
KIND_LABELS = {
    "joint": "cross-channel",
    "spike": "outlier",
    "step": "step change",
    "flatline": "flatlined",
    "stuck-at-zero": "stuck at zero",
    "railed": "railed",
    "missing": "missing data",
}

_STATUS_RANK = {"failed": 3, "faulty": 2, "anomalous": 1, "ok": 0}

#: How far above the rate that ``sensitivity`` itself predicts a channel may sit
#: and still count as quiet. A 3-sigma test flags about 0.27% of perfectly normal
#: readings; a channel that lands anywhere near that has found nothing.
NOISE_FLOOR_SLACK = 2.0

#: How many standard deviations above the expected false count a channel may sit
#: before its flags stop looking like chance. Counts are Poisson-like, so the worst
#: of several channels routinely lands two or three deviations high with nothing
#: wrong; judging the maximum against a flat multiple of the mean calls almost every
#: healthy multi-channel table anomalous. Flags also arrive in clusters, because one
#: noise excursion spans several readings, so the counts are overdispersed relative to
#: a plain Poisson. Measured over 40 clean six-channel tables, an allowance of three
#: deviations still called 20 percent of them anomalous and four called 8 percent;
#: five reaches zero while every planted spike, flatline, stuck-at-zero and step change
#: is still caught. The allowance is deliberately generous: this flag exists to stop
#: the report crying wolf, and the per-channel rates remain in the report either way.
NOISE_FLOOR_SIGMAS = 5.0


def at_noise_floor(
    channels: Mapping[str, "ChannelResult"],
    events: "List[Event]",
    joint: "List[int]",
    n_rows: int,
    expected_false_rate: Optional[float],
) -> bool:
    """True when every flag in this report is what ``sensitivity`` predicts by chance.

    A 3-sigma test on 3,000 clean rows flags about eight readings per channel. That
    is the test working, not a finding, and a report that calls it one cries wolf on
    every healthy table. This is the single place that judgement is made, so the
    ``anomalous`` flag, the summary headline and the reassurance note can never
    disagree with each other.

    Quiet means: nothing the statistics cannot explain. Any hardware fault, any
    cross-channel flag, any event that is not a plain outlier, or any channel
    flagging more than :data:`NOISE_FLOOR_SLACK` times the expected rate all mean
    the report has something real to say.
    """
    if not channels or expected_false_rate is None:
        return False
    if joint or any(res.faults for res in channels.values()):
        return False
    if any(event.kind != "spike" for event in events):
        return False
    rows = max(int(n_rows), 1)
    expected_count = float(expected_false_rate) * rows
    # The worst of k channels is a maximum, not an average: allow the Poisson spread
    # of the count itself, plus a one-flag floor so a short table is not condemned by
    # a single reading.
    allowed_count = max(
        NOISE_FLOOR_SLACK * expected_count,
        expected_count + NOISE_FLOOR_SIGMAS * math.sqrt(max(expected_count, 1.0)),
        1.0,
    )
    return max(res.rate for res in channels.values()) <= allowed_count / rows


def _clean_float(value: Any) -> Optional[float]:
    """A plain JSON-safe float, or ``None`` for NaN, inf and missing values."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy and pandas scalars/containers to JSON-safe Python."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _clean_float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _fmt_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    return str(value)


@dataclass
class ChannelResult:
    """What one sensor channel did.

    Attributes:
        anomalies: row positions flagged by the per-channel detector.
        rate: fraction of the channel's usable samples that were flagged, 0.0 to 1.0.
        score_max: highest robust z-score seen, or ``None`` for a channel with no
            usable variation (a flatlined or failed channel).
        faults: hardware-shaped problems such as ``"flatline"`` or ``"railed-at-max"``
            that a purely statistical detector would miss. Empty when the channel is healthy.
    """

    anomalies: List[int] = field(default_factory=list)
    rate: float = 0.0
    score_max: Optional[float] = None
    faults: List[str] = field(default_factory=list)
    #: Samples that were usable numbers.
    n_valid: int = 0
    #: Samples that were missing: NaN, +/-inf, or non-numeric.
    n_missing: int = 0
    #: One short sentence per fault, explaining it in plain words.
    notes: List[str] = field(default_factory=list)

    @property
    def n_anomalies(self) -> int:
        """How many rows the per-channel detector flagged."""
        return len(self.anomalies)

    @property
    def failed(self) -> bool:
        """True when the channel delivered no usable data at all."""
        return any(f in FAILED_FAULTS for f in self.faults)

    @property
    def ok(self) -> bool:
        """True when nothing was flagged and no fault was found."""
        return not self.faults and not self.anomalies

    @property
    def status(self) -> str:
        """One of ``"failed"``, ``"faulty"``, ``"anomalous"``, ``"ok"``."""
        if self.failed:
            return "failed"
        if self.faults:
            return "faulty"
        if self.anomalies:
            return "anomalous"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of this channel's result."""
        return {
            "status": self.status,
            "anomalies": [int(i) for i in self.anomalies],
            "n_anomalies": self.n_anomalies,
            "rate": _clean_float(self.rate),
            "score_max": _clean_float(self.score_max),
            "faults": list(self.faults),
            "n_valid": int(self.n_valid),
            "n_missing": int(self.n_missing),
            "notes": list(self.notes),
        }


@dataclass
class Event:
    """A contiguous run of flagged rows, merged across channels where they overlap.

    Attributes:
        start: first row position in the run.
        end: last row position in the run, inclusive.
        channels: channel names involved. Empty for a cross-channel event where no
            single channel was an outlier on its own, which is the classic
            broken-correlation fault.
        kind: ``"spike"``, ``"joint"``, ``"step"``, ``"flatline"``, ``"stuck-at-zero"``,
            ``"railed"`` or ``"missing"``.
        severity: how far past the threshold the event went. 1.0 sits exactly on the
            threshold; bigger is worse.
    """

    start: int
    end: int
    channels: List[str]
    kind: str
    severity: float
    #: Value of the time column at ``start``, when the data had one.
    start_time: Any = None
    #: Value of the time column at ``end``, when the data had one.
    end_time: Any = None

    @property
    def n_rows(self) -> int:
        """How many rows this event spans."""
        return int(self.end) - int(self.start) + 1

    @property
    def level(self) -> str:
        """``"high"``, ``"medium"`` or ``"low"``, derived from ``severity``."""
        if self.severity >= 3.0:
            return "high"
        if self.severity >= 1.5:
            return "medium"
        return "low"

    def describe(self) -> str:
        """One plain-ASCII line describing the event."""
        label = KIND_LABELS.get(self.kind, self.kind)
        where = "rows %d-%d" % (self.start, self.end)
        if self.start_time is not None:
            where += " (%s .. %s)" % (_fmt_time(self.start_time), _fmt_time(self.end_time))
        who = ", ".join(self.channels) if self.channels else "no single channel"
        return "%s  %s  [%s]  severity %.1f (%s)" % (
            where,
            label,
            who,
            self.severity,
            self.level,
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of this event."""
        return {
            "start": int(self.start),
            "end": int(self.end),
            "n_rows": self.n_rows,
            "channels": list(self.channels),
            "kind": self.kind,
            "severity": _clean_float(self.severity),
            "level": self.level,
            "start_time": _json_safe(self.start_time),
            "end_time": _json_safe(self.end_time),
        }


@dataclass
class SensorReport:
    """Everything found in one pass over a wide table of sensor channels.

    Attributes:
        channels: channel name mapped to its :class:`ChannelResult`.
        joint: row positions flagged by the cross-channel model.
        events: contiguous runs merged into :class:`Event` records, worst first.
        notes: plain-language remarks about what was skipped, assumed or fallen back to.
    """

    channels: Dict[str, ChannelResult] = field(default_factory=dict)
    joint: List[int] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    n_rows: int = 0
    #: Which cross-channel model actually ran: ``"isolation-forest"``,
    #: ``"mahalanobis"`` or ``"none"``.
    method: str = "none"
    #: The arguments detection actually used, after every "auto" was resolved.
    params: Dict[str, Any] = field(default_factory=dict)
    #: Name of the time column used for ``Event.start_time``, if there was one.
    time_column: Optional[str] = None

    @property
    def channel_names(self) -> List[str]:
        """Channel names, in input order."""
        return list(self.channels)

    @property
    def n_events(self) -> int:
        """How many merged events were found."""
        return len(self.events)

    @property
    def quiet(self) -> bool:
        """True when every flag here is what ``sensitivity`` predicts by chance."""
        return at_noise_floor(
            self.channels,
            self.events,
            self.joint,
            self.n_rows,
            self.params.get("expected_false_rate"),
        )

    @property
    def anomalous(self) -> bool:
        """True when something stands out beyond what ``sensitivity`` predicts.

        Safe to gate an alarm on: ``if report.anomalous: page_the_operator()``. A
        clean table at the default sensitivity still trips a handful of readings -
        that is what a 3-sigma test means - and this stays ``False`` for those, in
        agreement with the "Nothing here stands out" note the report prints.
        """
        if not self.events and all(c.ok for c in self.channels.values()):
            return False
        return not self.quiet

    @property
    def failed_channels(self) -> List[str]:
        """Channels that delivered no usable data."""
        return [name for name, res in self.channels.items() if res.failed]

    @property
    def worst_channels(self) -> List[str]:
        """Channel names worst first.

        Failed channels come first, then channels carrying a hardware fault, then
        channels ranked by anomaly rate and peak score. Healthy channels come last.
        """

        def key(name: str):
            res = self.channels[name]
            return (
                _STATUS_RANK[res.status],
                float(res.rate),
                float(res.score_max or 0.0),
            )

        return sorted(self.channels, key=key, reverse=True)

    def channels_with(self, fault: str) -> List[str]:
        """Channel names carrying ``fault``, for example ``"flatline"``."""
        return [name for name, res in self.channels.items() if fault in res.faults]

    def to_frame(self) -> pd.DataFrame:
        """One row per channel: status, anomalies, rate, peak score and faults."""
        columns = [
            "channel",
            "status",
            "anomalies",
            "rate",
            "score_max",
            "n_valid",
            "n_missing",
            "faults",
        ]
        rows = []
        for name in self.channel_names:
            res = self.channels[name]
            rows.append(
                {
                    "channel": name,
                    "status": res.status,
                    "anomalies": res.n_anomalies,
                    "rate": float(res.rate),
                    "score_max": (
                        float("nan") if res.score_max is None else float(res.score_max)
                    ),
                    "n_valid": int(res.n_valid),
                    "n_missing": int(res.n_missing),
                    "faults": ", ".join(res.faults),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def events_frame(self) -> pd.DataFrame:
        """One row per merged event: where it was, what kind, and how bad."""
        columns = [
            "start",
            "end",
            "n_rows",
            "kind",
            "channels",
            "severity",
            "level",
            "start_time",
            "end_time",
        ]
        rows = [
            {
                "start": ev.start,
                "end": ev.end,
                "n_rows": ev.n_rows,
                "kind": ev.kind,
                "channels": ", ".join(ev.channels),
                "severity": float(ev.severity),
                "level": ev.level,
                "start_time": ev.start_time,
                "end_time": ev.end_time,
            }
            for ev in self.events
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if self.time_column is None:
            frame = frame.drop(columns=["start_time", "end_time"])
        return frame

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of the whole report, ready for ``json.dumps``."""
        return {
            "n_rows": int(self.n_rows),
            "n_channels": len(self.channels),
            "anomalous": self.anomalous,
            "method": self.method,
            "time_column": self.time_column,
            "params": _json_safe(self.params),
            "channels": {name: res.to_dict() for name, res in self.channels.items()},
            "joint": [int(i) for i in self.joint],
            "n_joint": len(self.joint),
            "events": [ev.to_dict() for ev in self.events],
            "n_events": self.n_events,
            "worst_channels": self.worst_channels,
            "failed_channels": self.failed_channels,
            "notes": list(self.notes),
        }

    def summary(self, max_events: int = 10) -> str:
        """A human-readable report in plain ASCII, safe to print or pipe anywhere."""
        lines: List[str] = []
        n_ch = len(self.channels)
        plural = "" if n_ch == 1 else "s"
        flagged = [n for n, r in self.channels.items() if not r.ok]
        if n_ch == 0:
            lines.append("sensor-anomaly: no usable sensor channels found")
        elif self.n_rows == 0:
            # An empty export is a finding in itself, and a truer one than
            # anything said about the columns it still declares.
            lines.append(
                "sensor-anomaly: the table has no rows, so there was nothing to "
                "score (%d channel%s named)" % (n_ch, plural)
            )
        elif not self.anomalous:
            if self.events:
                lines.append(
                    "sensor-anomaly: OK - nothing stands out across %d channel%s in "
                    "%s rows; the %d flag%s found are at the noise floor"
                    % (
                        n_ch,
                        plural,
                        format(self.n_rows, ","),
                        self.n_events,
                        "" if self.n_events == 1 else "s",
                    )
                )
            else:
                lines.append(
                    "sensor-anomaly: OK - nothing flagged across %d channel%s in %s rows"
                    % (n_ch, plural, format(self.n_rows, ","))
                )
        else:
            lines.append(
                "sensor-anomaly: %d of %d channel%s flagged, %d event%s in %s rows"
                % (
                    len(flagged),
                    n_ch,
                    plural,
                    self.n_events,
                    "" if self.n_events == 1 else "s",
                    format(self.n_rows, ","),
                )
            )
        method_text = {
            "isolation-forest": "IsolationForest across channels",
            "mahalanobis": "Mahalanobis distance across channels",
            "none": "not run",
        }.get(self.method, self.method)
        lines.append(
            "per-channel: robust z-score on the rolling residual, sensitivity %.1f"
            " | cross-channel: %s"
            % (float(self.params.get("sensitivity", 3.0)), method_text)
        )
        if self.joint:
            pct = 100.0 * len(self.joint) / self.n_rows if self.n_rows else 0.0
            lines.append(
                "cross-channel rows flagged: %d (%.2f%%)" % (len(self.joint), pct)
            )

        if self.channels:
            lines.append("")
            width = max([len("channel")] + [len(str(c)) for c in self.channels])
            lines.append(
                "  %-*s  %-9s  %9s  %7s  %10s  %s"
                % (width, "channel", "status", "anomalies", "rate", "peak score", "faults")
            )
            for name in self.worst_channels:
                res = self.channels[name]
                score = "-" if res.score_max is None else "%.2f" % res.score_max
                lines.append(
                    "  %-*s  %-9s  %9d  %6.2f%%  %10s  %s"
                    % (
                        width,
                        name,
                        res.status,
                        res.n_anomalies,
                        100.0 * res.rate,
                        score,
                        ", ".join(res.faults) if res.faults else "-",
                    )
                )

        if self.events:
            shown = self.events[: max(0, int(max_events))]
            lines.append("")
            suffix = "" if len(shown) == self.n_events else ", showing %d" % len(shown)
            lines.append("events (%d, worst first%s):" % (self.n_events, suffix))
            for ev in shown:
                lines.append("  " + ev.describe())
            if len(shown) < self.n_events:
                lines.append("  ... %d more" % (self.n_events - len(shown)))

        if self.notes:
            lines.append("")
            lines.append("notes:")
            for note in self.notes:
                lines.append("  - " + note)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()

    def __repr__(self) -> str:
        return "SensorReport(channels=%d, rows=%d, events=%d, method=%r)" % (
            len(self.channels),
            self.n_rows,
            self.n_events,
            self.method,
        )
