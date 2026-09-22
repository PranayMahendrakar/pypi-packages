"""Command line interface: ``predictive-maintenance health|rul``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._health import health_score
from ._rul import estimate_rul

COMMANDS = ("health", "rul")


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass


def _parse_channels(text: Optional[str]) -> Optional[List[str]]:
    """Turn ``--channels a,b,c`` into a list of column names."""
    if text is None:
        return None
    picked = [part.strip() for part in text.split(",") if part.strip()]
    if not picked:
        raise ValueError("--channels was given but named no columns")
    return picked


def _parse_baseline(text: Optional[str]) -> Any:
    """Turn ``--baseline`` into a row count (int) or a fraction (float)."""
    if text is None:
        return None
    raw = text.strip()
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        raise ValueError(
            "--baseline takes a row count (e.g. 40) or a fraction of the history "
            "(e.g. 0.25), got {0!r}".format(text)
        ) from None


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", help="path to a .csv, .tsv or .parquet file of sensor history")
    parser.add_argument(
        "--time", metavar="COLUMN", help="the timestamp column (auto-detected when omitted)"
    )
    parser.add_argument(
        "--channels",
        metavar="A,B,C",
        help="comma-separated sensor columns (every numeric column when omitted)",
    )
    parser.add_argument(
        "--baseline",
        metavar="SPEC",
        help="the healthy period: a row count (40) or a fraction (0.25); default 0.2",
    )
    parser.add_argument(
        "--window", type=int, metavar="N", help="rows per rolling window (about a tenth by default)"
    )
    parser.add_argument("--json", action="store_true", help="print to_dict() as JSON")
    parser.add_argument("--output", "-o", metavar="PATH", help="also write the result JSON to PATH")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``predictive-maintenance`` command."""
    parser = argparse.ArgumentParser(
        prog="predictive-maintenance",
        description=(
            "Score how close equipment is to failure from its sensor history, and "
            "estimate the remaining useful life."
        ),
        epilog=(
            "A bare data file runs 'health', so 'predictive-maintenance sensors.csv' "
            "prints the degradation score."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s {0}".format(__version__))
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    health = subparsers.add_parser(
        "health", help="score degradation against a healthy baseline (the default)"
    )
    _add_shared(health)

    rul = subparsers.add_parser("rul", help="estimate the remaining useful life")
    _add_shared(rul)
    rul.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        metavar="SCORE",
        help="the health score that counts as end of useful life (default 30)",
    )
    return parser


def _write(payload: Dict[str, Any], path: str) -> None:
    """Write the result as UTF-8 JSON, creating the parent folder if needed."""
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _emit(result: Any, args: argparse.Namespace) -> int:
    payload = result.to_dict()
    if args.output:
        _write(payload, args.output)
        print("result written to {0}".format(args.output), file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result.summary())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``predictive-maintenance`` command; returns the exit code."""
    _make_console_safe()
    args_list: List[str] = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in COMMANDS and not args_list[0].startswith("-"):
        args_list.insert(0, "health")
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 2
    args = parser.parse_args(args_list)
    try:
        shared = {
            "time": args.time,
            "channels": _parse_channels(args.channels),
            "baseline": _parse_baseline(args.baseline),
            "window": args.window,
        }
        if args.command == "rul":
            return _emit(estimate_rul(args.data, threshold=args.threshold, **shared), args)
        return _emit(health_score(args.data, **shared), args)
    except (OSError, ValueError, TypeError, ImportError, KeyError) as exc:
        print("predictive-maintenance: error: {0}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
