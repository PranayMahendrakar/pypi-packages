"""Command line interface: ``model-watchdog checkout-model [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from ._records import to_utc
from ._watchdog import DEFAULT_ROOT, Watchdog, safe_name


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``model-watchdog`` command."""
    parser = argparse.ArgumentParser(
        prog="model-watchdog",
        description=(
            "Check a logged model for drift, silent failure and slowdowns. "
            "NAME is the watchdog name, or the path to a storage directory."
        ),
    )
    parser.add_argument("name", help="watchdog name, or a path to its storage directory")
    parser.add_argument(
        "--storage",
        metavar="DIR",
        help="storage directory (default: ./%s/NAME)" % DEFAULT_ROOT.name,
    )
    parser.add_argument(
        "--reference",
        metavar="PATH",
        help="reference data (.csv or .parquet) describing normal traffic",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        metavar="N",
        help="check the last N records (default: 1000)",
    )
    parser.add_argument(
        "--since",
        metavar="TIMESTAMP",
        help="check everything logged since this ISO timestamp instead of a window",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="print the logged records as CSV instead of the report",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--output", metavar="PATH", help="also write the report as JSON to PATH")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="exit with status 1 when a monitor failed (handy in CI and cron jobs)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def _configure_streams() -> None:
    """Make stdout/stderr UTF-8 tolerant so non-Latin text never raises."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic stream
                pass


def _emit(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - only when reconfigure was unavailable
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def _as_json(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _resolve(name: str, storage: Optional[str]) -> tuple:
    """Work out the watchdog name and its storage directory."""
    if storage:
        return name, Path(storage)
    candidate = Path(name)
    looks_like_path = len(candidate.parts) > 1 or candidate.is_dir()
    if looks_like_path:
        return candidate.name or name, candidate
    return name, DEFAULT_ROOT / safe_name(name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit status."""
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    name, storage = _resolve(args.name, args.storage)
    since = None
    if args.since:
        # A typo here must not silently widen the window to the whole log.
        since = to_utc(args.since)
        if since is None:
            parser.error(
                "could not read --since %r; use an ISO timestamp such as "
                "2026-09-22 or 2026-09-22T10:00:00+00:00" % args.since
            )
    try:
        watchdog = Watchdog(name, reference=args.reference, storage=storage)
        if args.metrics:
            frame = watchdog.metrics()
            # Ask pandas for plain "\n" and let the text layer add the
            # platform ending once. Its default "\r\n" would be translated a
            # second time on Windows, leaving "\r\r\n" and a blank row in
            # anything stricter than csv.reader.
            csv_text = frame.to_csv(index=False, lineterminator="\n")
            _emit(csv_text.rstrip("\r\n"))
            return 0
        if since is not None:
            report = watchdog.report(since=since)
        else:
            report = watchdog.check(window=args.window)
    except (ValueError, TypeError, ImportError, OSError) as exc:
        parser.error(str(exc))
    payload = report.to_dict()
    _emit(_as_json(payload) if args.json else report.summary())
    if args.output:
        Path(args.output).write_text(_as_json(payload), encoding="utf-8")
    return 1 if (args.fail_on_alert and not report.ok) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
