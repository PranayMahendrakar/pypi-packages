"""Command line interface: ``timeseries-anomaly DATA [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from ._core import detect
from ._methods import METHODS
from ._result import AnomalyResult

_DESCRIPTION = """Find the points in a time series or IoT signal that do not belong.

DATA is a .csv or .parquet file. The measurement column and the timestamp column
are guessed when you do not name them."""

_EPILOG = """examples:
  timeseries-anomaly readings.csv
  timeseries-anomaly readings.csv --value temperature --time recorded_at
  timeseries-anomaly readings.csv --method seasonal --seasonality 24 --sensitivity 4
  timeseries-anomaly readings.csv --json > anomalies.json
  timeseries-anomaly readings.csv --output scored.csv"""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="timeseries-anomaly",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data", metavar="DATA", help="path to a .csv or .parquet file")
    parser.add_argument("--value", metavar="COL", help="column holding the measurement")
    parser.add_argument("--time", metavar="COL", help="column holding the timestamps")
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default="auto",
        help="how to build the baseline (default: auto)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=3.0,
        help="threshold in robust sigmas, higher means fewer anomalies (default: 3.0)",
    )
    parser.add_argument(
        "--seasonality",
        type=int,
        metavar="N",
        help="points per cycle for --method seasonal (inferred from the data when omitted)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="how many flagged points to list in the summary (default: 10)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON instead of the summary"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the scored table (.csv or .parquet) with every point and its score",
    )
    parser.add_argument("--version", action="version", version=f"timeseries-anomaly {__version__}")
    return parser


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed or unusual streams
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def write_output(path: str, result: AnomalyResult) -> int:
    """Write the scored table to `path`; returns the number of rows written."""
    frame = result.to_frame()
    if path.lower().endswith((".parquet", ".pq")):
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8")
    return int(len(frame))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = detect(
            args.data,
            value=args.value,
            time=args.time,
            method=args.method,
            sensitivity=args.sensitivity,
            seasonality=args.seasonality,
        )
        rows = write_output(args.output, result) if args.output else 0
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(result.summary(limit=args.limit))
            if args.output:
                print(f"  wrote     : {rows:,} scored rows to {args.output}")
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"timeseries-anomaly: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
