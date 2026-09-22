"""Command line interface: ``energy-analyzer-ai DATA [options]``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

from . import __version__
from ._core import SENSITIVITY, EnergyAnalyzer
from ._report import EnergyReport

_DESCRIPTION = """Find unusual energy consumption, explain what changed, and estimate the cost.

DATA is a .csv or .parquet file of meter readings. The reading column and the
timestamp column are guessed when you do not name them. A cumulative meter is
detected and differenced, and periods with no reading are reported as gaps."""

_EPILOG = """examples:
  energy-analyzer-ai meter.csv
  energy-analyzer-ai meter.csv --value kwh --time recorded_at
  energy-analyzer-ai meter.csv --tariff 0.28 --granularity hourly
  energy-analyzer-ai meter.csv --tariff '{"0": 0.12, "7": 0.31, "19": 0.45}'
  energy-analyzer-ai meter.csv --tariff rates.json --json > report.json
  energy-analyzer-ai meter.csv --output by_period.csv --forecast 24"""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="energy-analyzer-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data", metavar="DATA", help="path to a .csv or .parquet file")
    parser.add_argument("--value", metavar="COL", help="column holding the meter reading")
    parser.add_argument("--time", metavar="COL", help="column holding the timestamps")
    parser.add_argument(
        "--tariff",
        metavar="RATE",
        help="a flat cost per unit (0.28), a JSON hour->rate map, or a path to a .json file",
    )
    parser.add_argument(
        "--granularity",
        metavar="G",
        default="auto",
        help="grid size: auto, hourly, daily, weekly, 15min, 1h (default: auto)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=SENSITIVITY,
        help=f"threshold in robust sigmas, higher means fewer anomalies (default: {SENSITIVITY:g})",
    )
    parser.add_argument(
        "--min-effect",
        type=float,
        default=0.10,
        dest="min_effect",
        help="ignore deviations smaller than this share of the expected value (default: 0.10)",
    )
    parser.add_argument(
        "--baseline",
        metavar="X",
        help="a fixed expected value per period, or a .csv/.parquet reference period",
    )
    parser.add_argument(
        "--cumulative",
        choices=("auto", "yes", "no"),
        default="auto",
        help="whether the readings are a cumulative meter (default: auto)",
    )
    parser.add_argument(
        "--forecast",
        type=int,
        metavar="N",
        help="also project the next N periods",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="how many unusual periods to list in the summary (default: 5)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON instead of the summary"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the per-period table (.csv or .parquet) with every period and its score",
    )
    parser.add_argument("--version", action="version", version=f"energy-analyzer-ai {__version__}")
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


def parse_tariff(text: Optional[str]) -> Any:
    """Read ``--tariff`` as a number, a JSON mapping, or a path to a .json file."""
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.lower().endswith(".json"):
        if not os.path.exists(candidate):
            raise ValueError(f"tariff file {candidate!r} does not exist")
        with open(candidate, "r", encoding="utf-8") as handle:
            return json.load(handle)
    try:
        return float(candidate)
    except ValueError:
        pass
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        raise ValueError(
            f"could not read --tariff {text!r}; pass a number like 0.28, a JSON map "
            'like \'{"0": 0.12, "7": 0.31}\', or a path to a .json file'
        ) from None


def parse_baseline(text: Optional[str]) -> Any:
    """Read ``--baseline`` as a number or as a path to a reference file."""
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return float(candidate)
    except ValueError:
        return candidate


def write_output(path: str, report: EnergyReport) -> int:
    """Write the per-period table to `path`; returns the number of rows written."""
    frame = report.by_period.reset_index()
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
        cumulative = {"auto": None, "yes": True, "no": False}[args.cumulative]
        analyzer = EnergyAnalyzer(
            tariff=parse_tariff(args.tariff),
            baseline=parse_baseline(args.baseline),
            granularity=args.granularity,
            sensitivity=args.sensitivity,
            min_effect=args.min_effect,
            cumulative=cumulative,
        )
        report = analyzer.analyze(args.data, value=args.value, time=args.time)
        rows = write_output(args.output, report) if args.output else 0
        projection = None
        if args.forecast:
            projection = analyzer.forecast(args.data, args.forecast, value=args.value, time=args.time)
        if args.json:
            payload = report.to_dict()
            if projection is not None:
                payload["forecast"] = [
                    {"when": str(when), "value": None if value != value else float(value)}
                    for when, value in projection.items()
                ]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(report.summary(limit=args.limit))
            if args.output:
                print(f"  wrote     : {rows:,} periods to {args.output}")
            if projection is not None and len(projection):
                total = float(projection.sum())
                print(
                    f"  forecast  : next {len(projection):,} {report.granularity} periods "
                    f"total {total:,.3g} {report.unit}"
                )
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"energy-analyzer-ai: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
