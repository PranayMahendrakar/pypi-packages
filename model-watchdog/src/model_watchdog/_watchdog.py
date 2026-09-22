"""The Watchdog itself: log records, then ask it what it thinks.

    wd = Watchdog("checkout-model", reference=last_month)
    wd.log(features=row, prediction=p, latency_ms=ms)   # in the request handler
    report = wd.check()                                 # in a cron job

``log()`` and ``metrics()`` swallow their own errors, because monitoring must
never take down the application it is watching. ``check()`` and ``report()``
are the diagnostic path and do raise, so a broken set-up is visible.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._monitors import MONITOR_NAMES, Thresholds, Window, run_all
from ._records import build_record, to_frame, to_utc
from ._reference import ReferenceProfile, profile_or_note
from ._report import Alert, WatchReport
from ._storage import JsonlStorage

logger = logging.getLogger(__name__)

#: Where logs go when no ``storage`` is given.
DEFAULT_ROOT = Path(".model_watchdog")

_UNSAFE = re.compile(r"[^\w.-]+", re.UNICODE)

#: Longest folder name produced from a watchdog name, before the hash suffix.
MAX_NAME = 100


def safe_name(name: str) -> str:
    """A watchdog name turned into something safe to use as a folder name.

    ``name`` must already be a ``str``; :class:`Watchdog` rejects anything else.

    A name longer than :data:`MAX_NAME` keeps its readable head and gains a
    short digest of the whole thing. Plain truncation made two models whose
    names differ only after the hundredth character share one directory, and
    their predictions interleaved into a single log: each report was then
    computed over the other model's traffic.
    """
    cleaned = _UNSAFE.sub("_", name).strip("._")
    digest = hashlib.sha1(name.encode("utf-8", "replace")).hexdigest()[:8]
    if not cleaned:
        # Every name made only of separators would otherwise become "model".
        return "model" if not name else "model-%s" % digest
    if len(cleaned) <= MAX_NAME:
        return cleaned
    return "%s-%s" % (cleaned[: MAX_NAME - len(digest) - 1], digest)


class Watchdog:
    """Monitors one model.

    Args:
        name: what this model is called; also the default folder name.
        reference: what normal looks like - a DataFrame of past traffic, a path
            to a ``.csv``/``.parquet`` file, a dict of values, or a list of past
            predictions. Optional: without one, the monitors that need a
            baseline report themselves inactive instead of failing.
        storage: directory for the JSONL log. Defaults to
            ``./.model_watchdog/<name>``.
        alert: optional callable taking one :class:`Alert`, called by
            ``check()`` when a monitor fails. Its errors are swallowed.
    """

    def __init__(
        self,
        name: str,
        *,
        reference: Any = None,
        storage: Any = None,
        alert: Optional[Callable[[Alert], Any]] = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(
                "name must be a str, got %s; a watchdog name is also its folder "
                "name, and coercing one silently logs to a directory such as "
                "'.model_watchdog/None'" % type(name).__name__
            )
        self.name = name
        self.storage = Path(storage) if storage is not None else DEFAULT_ROOT / safe_name(name)
        self.alert = alert
        self.thresholds = Thresholds()
        self._store = JsonlStorage(self.storage)
        self._reference = profile_or_note(reference)

    # -- reference ---------------------------------------------------------

    @property
    def reference(self) -> ReferenceProfile:
        """The parsed reference profile. Assign anything ``reference=`` accepts."""
        return self._reference

    @reference.setter
    def reference(self, value: Any) -> None:
        self._reference = profile_or_note(value)

    # -- writing -----------------------------------------------------------

    def log(
        self,
        features: Any = None,
        prediction: Any = None,
        actual: Any = None,
        latency_ms: Any = None,
        **meta: Any,
    ) -> None:
        """Append one prediction to the log. Never raises into the caller.

        Args:
            features: the model input, as a dict, Series, one-row DataFrame or
                sequence. Optional.
            prediction: whatever the model returned - a number, a label, a
                probability.
            actual: the true label, when you already have it. You can also log
                it later as its own record.
            latency_ms: how long the prediction took, in milliseconds.
            **meta: anything else worth keeping (request id, model version,
                user segment). A ``ts=`` here sets the record timestamp, which
                is handy for backfilling.
        """
        try:
            timestamp = meta.pop("ts", None)
            record = build_record(
                features=features,
                prediction=prediction,
                actual=actual,
                latency_ms=latency_ms,
                meta=meta,
                ts=timestamp,
            )
            self._store.append(record)
        except Exception as exc:  # noqa: BLE001 - deliberate: never break the caller
            logger.warning(
                "model-watchdog: could not log a record for %r: %s", self.name, exc
            )

    # -- reading -----------------------------------------------------------

    def metrics(self) -> pd.DataFrame:
        """Everything logged so far as a DataFrame. Never raises.

        Columns: ``ts`` (timezone-aware UTC), ``prediction``, ``actual``,
        ``latency_ms``, then one column per logged feature, then one per meta
        key. Returns an empty frame when there is nothing to read.
        """
        try:
            frame = to_frame(self._store.read())
            data = frame.data
            data.attrs["feature_columns"] = list(frame.feature_columns)
            data.attrs["meta_columns"] = list(frame.meta_columns)
            return data
        except Exception as exc:  # noqa: BLE001 - deliberate: never break the caller
            logger.warning(
                "model-watchdog: could not read the log for %r: %s", self.name, exc
            )
            return to_frame([]).data

    def check(self, window: int = 1000, *, thresholds: Optional[Thresholds] = None) -> WatchReport:
        """Run every monitor over the last ``window`` records.

        Calls the ``alert`` callable once when at least one monitor failed.

        Raises:
            TypeError: ``window`` is not an integer, or ``thresholds`` is not a
                :class:`Thresholds`.
            ValueError: ``window`` is zero or negative.
        """
        window = _positive_int(window, "window")
        _check_thresholds(thresholds)
        records = self._store.read(limit=window)
        report = self._build(records, thresholds=thresholds, window=window)
        self._fire(report)
        return report

    def report(
        self, since: Any = None, *, thresholds: Optional[Thresholds] = None
    ) -> WatchReport:
        """Run every monitor over everything logged since ``since``.

        ``since`` accepts a datetime (naive is read as UTC), an ISO string, or
        ``None`` for the whole log. Unlike ``check()``, this never alerts.

        Raises:
            ValueError: ``since`` could not be read as a timestamp. It is not
                widened to the whole log, which is what a nightly job with a
                mistyped cutoff would quietly check instead of one day of it.
            TypeError: ``thresholds`` is not a :class:`Thresholds`.
        """
        _check_thresholds(thresholds)
        start = _since_or_raise(since)
        records = self._store.read()
        if start is not None:
            kept = []
            for record in records:
                stamp = _record_time(record)
                if stamp is not None and stamp >= start:
                    kept.append(record)
            records = kept
        return self._build(records, thresholds=thresholds, since=start)

    # -- internals ---------------------------------------------------------

    def _build(
        self,
        records: Sequence[Any],
        *,
        thresholds: Optional[Thresholds] = None,
        window: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> WatchReport:
        limits = thresholds or self.thresholds
        window_data = Window(records)
        checks = run_all(window_data, self._reference, limits)
        first, last = window_data.span()
        report = WatchReport(
            name=self.name,
            checks=checks,
            records=window_data.n,
            window=window,
            since=since,
            first_ts=first,
            last_ts=last,
            reference=self._reference.describe(),
            storage=str(self.storage),
            notes=list(self._reference.notes),
        )
        if window_data.n == 0:
            report.notes.append("nothing logged yet at %s" % self.storage)
        inactive = [check.name for check in report.inactive]
        if inactive:
            report.notes.append(
                "inactive monitors (not failures, just no data yet): %s" % ", ".join(inactive)
            )
        return report

    def _fire(self, report: WatchReport) -> None:
        if self.alert is None or report.ok:
            return
        alert = Alert(
            name=self.name,
            report=report,
            failed=list(report.failed),
            message="model-watchdog: %s - %s" % (self.name, report.headline()),
        )
        try:
            self.alert(alert)
        except Exception as exc:  # noqa: BLE001 - an alert sink must not break the check
            logger.warning(
                "model-watchdog: the alert callable raised for %r: %s", self.name, exc
            )

    # -- niceties ----------------------------------------------------------

    @property
    def monitors(self) -> List[str]:
        """The monitor names, in report order."""
        return list(MONITOR_NAMES)

    def __len__(self) -> int:
        try:
            return self._store.count()
        except Exception:  # noqa: BLE001 - counting must not break either
            return 0

    def __repr__(self) -> str:
        return "Watchdog(%r, storage=%r)" % (self.name, str(self.storage))


def _positive_int(value: Any, argument: str) -> int:
    """``value`` as a positive int, with an error that names the argument."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "%s must be a positive integer, got %s" % (argument, type(value).__name__)
        )
    if value <= 0:
        raise ValueError("%s must be a positive integer, got %d" % (argument, value))
    return int(value)


def _check_thresholds(thresholds: Any) -> None:
    """Reject anything that is not a :class:`Thresholds`.

    Without this a wrong type - a config string straight out of a YAML file -
    breaks all seven monitors identically, and ``run_all`` turns each identical
    crash into an inactive check. The report then reads healthy while nothing
    at all is being monitored.
    """
    if thresholds is not None and not isinstance(thresholds, Thresholds):
        raise TypeError(
            "thresholds must be a Thresholds, got %s; build one with "
            "model_watchdog.Thresholds(...)" % type(thresholds).__name__
        )


#: What ``report(since=)`` accepts. A bare number is deliberately not on the
#: list: pandas reads it as nanoseconds since the epoch, so a unix timestamp in
#: seconds silently becomes a moment in January 1970 and the window widens to
#: the whole log - the exact miss this check exists to prevent.
_SINCE_TYPES = (datetime, date, pd.Timestamp, np.datetime64, str)


def _since_or_raise(since: Any) -> Optional[datetime]:
    """``since`` as aware UTC, or ``ValueError`` - never a silently wider window."""
    if since is None:
        return None
    start = None
    if isinstance(since, _SINCE_TYPES):
        try:
            start = to_utc(since)
        except Exception:  # noqa: BLE001 - every unreadable value gets one message
            start = None
    if start is None:
        raise ValueError(
            "could not read since=%r; pass a datetime or an ISO timestamp such "
            "as '2026-09-22' or '2026-09-22T10:00:00+00:00'" % (since,)
        )
    return start


def _record_time(record: Any) -> Optional[datetime]:
    try:
        return to_utc(record.get("ts"))
    except Exception:  # noqa: BLE001 - a hand-edited line must not break a report
        return None


def check(
    name: str,
    *,
    reference: Any = None,
    storage: Any = None,
    window: int = 1000,
) -> WatchReport:
    """Check an existing log in one line.

        report = model_watchdog.check("checkout-model")

    Same arguments as :class:`Watchdog`, plus ``window``. Handy in a cron job
    or a notebook, where the process that wrote the log is long gone.
    """
    return Watchdog(name, reference=reference, storage=storage).check(window=window)

