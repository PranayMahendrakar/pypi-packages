"""Command line entry point: ``production-anomaly line.csv --shift-hours 6-22``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .analyzer import ProductionAnalyzer


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be above zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="production-anomaly",
        description=(
            "Find downtime, slow running, rate drift, micro-stops and counter spikes in a "
            "production line's output, and report availability, performance and "
            "availability x performance (quality is not measured: it needs scrap data)."
        ),
        epilog=(
            "example: production-anomaly line.csv --time ts --column units "
            "--shift-hours 06:00-14:00,14:00-22:00 --target-rate 3600"
        ),
    )
    parser.add_argument("path", help="table of units per interval or a running counter (.csv, .tsv or .parquet)")
    parser.add_argument("--time", metavar="COL", help="timestamp column (default: found automatically)")
    parser.add_argument(
        "--column",
        metavar="COL",
        help="units-produced column, the analyze() output= argument (default: found automatically)",
    )
    parser.add_argument(
        "--target-rate",
        metavar="UNITS_PER_HOUR",
        type=_positive_float,
        help="rated speed in units per hour; without it the line is compared with its own typical rate",
    )
    parser.add_argument(
        "--shift-hours",
        metavar="SPEC",
        help="scheduled hours, like '6-22', '06:00-14:00,14:00-22:00' or 'A=6-14,B=14-22'; "
        "hours outside them are not downtime",
    )
    parser.add_argument(
        "--interval",
        metavar="DURATION",
        help="length of one interval, like '1min' or '1h' (default: measured from the timestamps)",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--output", metavar="FILE", help="write the JSON report to this file")
    parser.add_argument(
        "--by-shift",
        metavar="FILE",
        help="write the per-shift table to this CSV file",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _write_parent(path: Path) -> None:
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, analyze the table, print the report. Returns the exit status."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # a stream that cannot be reconfigured
                pass
    args = build_parser().parse_args(argv)
    try:
        analyzer = ProductionAnalyzer(
            time=args.time,
            output=args.column,
            target_rate=args.target_rate,
            shift_hours=args.shift_hours,
            interval=args.interval,
        )
        report = analyzer.analyze(args.path)
        if args.output:
            path = Path(args.output)
            _write_parent(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        if args.by_shift:
            path = Path(args.by_shift)
            _write_parent(path)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                report.by_shift.to_csv(handle, index=False)
    except (ValueError, TypeError, ImportError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
