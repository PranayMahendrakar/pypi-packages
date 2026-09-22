"""Command line interface: ``data-drift-lite reference.csv current.csv [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from ._core import detect


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``data-drift-lite`` command."""
    parser = argparse.ArgumentParser(
        prog="data-drift-lite",
        description="Detect whether current data has drifted from reference data, column by column.",
    )
    parser.add_argument("reference", help="reference dataset (.csv or .parquet), e.g. the training data")
    parser.add_argument("current", help="current dataset (.csv or .parquet), e.g. a production batch")
    parser.add_argument(
        "--columns", nargs="+", metavar="COL", help="compare only these columns (default: every reference column)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        metavar="P",
        help="p-value below which a column counts as drifted (default: 0.05)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.2,
        metavar="PSI",
        help="PSI above which a column counts as drifted (default: 0.2)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100_000,
        metavar="N",
        help="cap each dataset at N random rows; 0 disables sampling (default: 100000)",
    )
    parser.add_argument(
        "--random-state", type=int, default=0, metavar="SEED", help="seed for the sampling (default: 0)"
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON instead of the summary")
    parser.add_argument("--output", metavar="PATH", help="also write the full report as JSON to PATH")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit with status 1 when drift is detected (handy in CI and cron jobs)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _configure_streams() -> None:
    """Make stdout/stderr UTF-8 tolerant so non-Latin column names never raise."""
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit status."""
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = detect(
            args.reference,
            args.current,
            columns=args.columns,
            threshold=args.threshold,
            psi_threshold=args.psi_threshold,
            sample=args.sample or None,
            random_state=args.random_state,
        )
    except (ValueError, TypeError, ImportError, OSError) as exc:
        parser.error(str(exc))
    payload = report.to_dict()
    _emit(_as_json(payload) if args.json else report.summary())
    if args.output:
        Path(args.output).write_text(_as_json(payload), encoding="utf-8")
    return 1 if (args.fail_on_drift and report.drifted) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
