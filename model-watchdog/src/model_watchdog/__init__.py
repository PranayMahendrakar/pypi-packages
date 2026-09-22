"""Lightweight production monitoring for any ML model.

    import model_watchdog

    wd = model_watchdog.Watchdog("checkout-model", reference=last_month_df)
    wd.log(features=row, prediction=p, latency_ms=ms)   # in the request path
    print(wd.check().summary())                         # in a cron job

``log()`` never raises into the caller, ``check()`` returns a ``WatchReport``
whose ``.ok``, ``.failed`` and ``.summary()`` explain what happened, and every
monitor that lacks the data it needs says so instead of failing.
"""

from ._monitors import MONITOR_NAMES, Thresholds
from ._records import utc_now
from ._reference import ReferenceProfile
from ._report import Alert, Check, WatchReport
from ._storage import JsonlStorage
from ._watchdog import DEFAULT_ROOT, Watchdog, check

__version__ = "0.1.0"

__all__ = [
    "Watchdog",
    "check",
    "Check",
    "WatchReport",
    "Alert",
    "Thresholds",
    "ReferenceProfile",
    "JsonlStorage",
    "MONITOR_NAMES",
    "DEFAULT_ROOT",
    "utc_now",
    "__version__",
]
