"""The result objects: Check, WatchReport and Alert.

These are what a user actually reads, so they explain themselves: every check
carries the number it measured, the number it was compared against, and a
sentence saying what that means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from ._records import utc_now
from ._stats import jsonable

#: Status words used in ``summary()``. Plain ASCII on purpose.
STATUS_OK = "ok"
STATUS_FAIL = "FAIL"
STATUS_INACTIVE = "inactive"


def _number(value: Optional[float]) -> str:
    """Format a number for the summary table, or ``-`` when there is none."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "-"
    if abs(number) >= 1000 or number == int(number):
        return format(number, ",.0f")
    if abs(number) < 0.001:
        return format(number, ".2e")
    return format(number, ".4f")


def _stamp(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _short_stamp(value: Optional[datetime]) -> str:
    """Whole seconds only, so the summary header stays readable."""
    if value is None:
        return "-"
    return value.replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Check:
    """The outcome of one monitor.

    ``ok`` is ``True`` for a monitor that passed *and* for one that could not
    run: an inactive monitor is missing data, not a failure. ``active`` says
    which of the two it was.
    """

    name: str
    ok: bool
    value: Optional[float]
    threshold: Optional[float]
    message: str
    active: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """``"ok"``, ``"FAIL"`` or ``"inactive"``."""
        if not self.active:
            return STATUS_INACTIVE
        return STATUS_OK if self.ok else STATUS_FAIL

    @property
    def failed(self) -> bool:
        """True only for an active monitor that failed."""
        return self.active and not self.ok

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the check."""
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "active": bool(self.active),
            "status": self.status,
            "value": jsonable(self.value),
            "threshold": jsonable(self.threshold),
            "message": self.message,
            "details": {key: jsonable(value) for key, value in dict(self.details).items()},
        }

    def __str__(self) -> str:
        return "%s: %s (value %s, threshold %s) - %s" % (
            self.name,
            self.status,
            _number(self.value),
            _number(self.threshold),
            self.message,
        )


def ok_check(name: str, value: Optional[float], threshold: Optional[float], message: str, **details: Any) -> Check:
    """A monitor that ran and passed."""
    return Check(name, True, value, threshold, message, True, details)


def failed_check(name: str, value: Optional[float], threshold: Optional[float], message: str, **details: Any) -> Check:
    """A monitor that ran and tripped."""
    return Check(name, False, value, threshold, message, True, details)


def inactive_check(name: str, message: str, **details: Any) -> Check:
    """A monitor that could not run, with the reason why."""
    return Check(name, True, None, None, message, False, details)


@dataclass
class WatchReport:
    """Everything the monitors found, in one object.

    ``report.ok`` is the one thing most callers need; ``report.summary()`` is
    the thing to put in a log line or an alert.
    """

    name: str
    checks: List[Check] = field(default_factory=list)
    records: int = 0
    window: Optional[int] = None
    since: Optional[datetime] = None
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    reference: Optional[str] = None
    storage: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)

    @property
    def blind(self) -> bool:
        """True when every monitor crashed, so nothing was actually checked.

        One monitor tripping over an odd window is expected and stays an
        inactive check. All of them going down together is a broken set-up -
        a wrong ``thresholds=``, an unreadable log - and a cron job asking
        ``if not report.ok`` must not be told everything is fine while the
        watchdog is looking at nothing.
        """
        if not self.checks:
            return False
        return all(check.details.get("monitor_error") for check in self.checks)

    @property
    def ok(self) -> bool:
        """True when no active monitor failed and at least one could run."""
        if any(check.failed for check in self.checks):
            return False
        return not self.blind

    @property
    def failed(self) -> List[Check]:
        """The checks that failed, worst first is not implied - source order."""
        return [check for check in self.checks if check.failed]

    @property
    def active(self) -> List[Check]:
        """The checks that had enough data to run."""
        return [check for check in self.checks if check.active]

    @property
    def inactive(self) -> List[Check]:
        """The checks that could not run, each with a reason in its message."""
        return [check for check in self.checks if not check.active]

    def get(self, name: str) -> Optional[Check]:
        """One check by monitor name, or ``None``."""
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def __getitem__(self, name: str) -> Check:
        check = self.get(name)
        if check is None:
            raise KeyError(name)
        return check

    def __bool__(self) -> bool:
        return self.ok

    def headline(self) -> str:
        """One line: what happened, in words."""
        active = len(self.active)
        if self.records == 0:
            return "no records logged yet - nothing to check"
        if active == 0:
            if self.blind:
                return (
                    "ALERT: %d records logged and every monitor failed to run - "
                    "nothing was checked" % self.records
                )
            return "%d records logged, but no monitor could run yet" % self.records
        if self.ok:
            return "OK: %d of %d checks passed" % (active, active)
        return "ALERT: %d of %d active checks failed (%s)" % (
            len(self.failed),
            active,
            ", ".join(check.name for check in self.failed),
        )

    def summary(self) -> str:
        """Human-readable report. Plain ASCII, safe to print or pipe."""
        lines = ["model-watchdog: %s - %s" % (self.name, self.headline())]
        span = ""
        if self.first_ts is not None and self.last_ts is not None:
            span = " | %s to %s" % (_short_stamp(self.first_ts), _short_stamp(self.last_ts))
        scope = "window %s" % self.window if self.window else "all records"
        if self.since is not None:
            scope = "since %s" % _short_stamp(self.since)
        lines.append("records: %d (%s)%s" % (self.records, scope, span))
        if self.reference:
            lines.append("reference: %s" % self.reference)
        if self.checks:
            lines.append("")
            lines.extend(self._table())
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend("  - %s" % note for note in self.notes)
        return "\n".join(lines)

    def _table(self) -> List[str]:
        header = ("check", "status", "value", "threshold", "detail")
        rows = [
            (
                check.name,
                check.status,
                _number(check.value),
                _number(check.threshold),
                check.message,
            )
            for check in self.checks
        ]
        widths = [
            max(len(header[index]), max(len(row[index]) for row in rows))
            for index in range(4)
        ]
        out = []
        for row in (header,) + tuple(rows):
            cells = [row[index].ljust(widths[index]) for index in range(4)]
            out.append("  " + "  ".join(cells) + "  " + row[4])
        return [line.rstrip() for line in out]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole report."""
        return {
            "name": self.name,
            "ok": self.ok,
            "headline": self.headline(),
            "records": int(self.records),
            "window": self.window,
            "since": _stamp(self.since),
            "first_ts": _stamp(self.first_ts),
            "last_ts": _stamp(self.last_ts),
            "reference": self.reference,
            "storage": self.storage,
            "created_at": _stamp(self.created_at),
            "checks": [check.to_dict() for check in self.checks],
            "failed": [check.name for check in self.failed],
            "inactive": [check.name for check in self.inactive],
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class Alert:
    """What an ``alert=`` callable receives when a check fails."""

    name: str
    report: WatchReport
    failed: List[Check]
    message: str
    ts: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict, report included."""
        return {
            "name": self.name,
            "message": self.message,
            "ts": _stamp(self.ts),
            "failed": [check.to_dict() for check in self.failed],
            "report": self.report.to_dict(),
        }

    def __str__(self) -> str:
        return self.message
