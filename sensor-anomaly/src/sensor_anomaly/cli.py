"""Command line interface: ``sensor-anomaly DATA [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

from . import __version__
from ._core import detect
from ._result import SensorReport

_DESCRIPTION = """Spot abnormal behaviour across many sensor channels at once.

DATA is a wide .csv, .tsv or .parquet file with one column per sensor channel.
Every numeric column is treated as a channel unless you name them with
--channels, and an obvious timestamp column is detected and reported."""

_EPILOG = """examples:
  sensor-anomaly plant.csv
  sensor-anomaly plant.csv --time timestamp --sensitivity 4
  sensor-anomaly plant.csv --channels temp,flow,vibration
  sensor-anomaly plant.csv --json > findings.json
  sensor-anomaly plant.csv --output channels.csv"""


def _contamination(text: str) -> Any:
    """``auto`` or a fraction; argparse calls this for --contamination."""
    if text.strip().lower() == "auto":
        return "auto"
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'expected "auto" or a fraction above 0 and at most 0.5, got %r' % text
        ) from None


def _channel_list(text: str) -> list:
    """Split a comma-separated channel list, keeping names that contain spaces."""
    return [part.strip() for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="sensor-anomaly",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data", metavar="DATA", help="path to a .csv, .tsv or .parquet file")
    parser.add_argument("--time", metavar="COL", help="column holding the timestamps")
    parser.add_argument(
        "--channels",
        metavar="A,B,C",
        type=_channel_list,
        help="comma-separated sensor columns (default: every numeric column)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=3.0,
        help="per-channel threshold in robust sigmas; higher flags less (default: 3.0)",
    )
    parser.add_argument(
        "--contamination",
        type=_contamination,
        default="auto",
        metavar="VALUE",
        help='share of rows expected to be anomalous, or "auto" (default: auto)',
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        dest="random_state",
        metavar="N",
        help="seed for the cross-channel model (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="how many events to list in the summary (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the whole report as JSON instead of the summary",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the per-channel table (.csv/.tsv/.parquet), or the whole report (.json)",
    )
    parser.add_argument("--version", action="version", version="sensor-anomaly %s" % __version__)
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


def write_output(path: str, report: SensorReport) -> str:
    """Write the report to ``path``; returns a one-line description of what was written."""
    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        return "wrote the full report to %s" % path
    frame = report.to_frame()
    if lower.endswith((".parquet", ".pq")):
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            # pandas answers with five lines of engine internals naming an extra
            # this package does not declare. Point at the one that exists.
            raise ImportError(
                'writing .parquet needs pyarrow: pip install "sensor-anomaly[parquet]"'
            ) from exc
    elif lower.endswith(".tsv"):
        frame.to_csv(path, index=False, sep="\t", encoding="utf-8")
    else:
        frame.to_csv(path, index=False, encoding="utf-8")
    return "wrote %d channel rows to %s" % (len(frame), path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = detect(
            args.data,
            time=args.time,
            channels=args.channels,
            sensitivity=args.sensitivity,
            contamination=args.contamination,
            random_state=args.random_state,
        )
        written = write_output(args.output, report) if args.output else ""
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.summary(max_events=args.limit))
            if written:
                print("")
                print(written)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print("sensor-anomaly: error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
