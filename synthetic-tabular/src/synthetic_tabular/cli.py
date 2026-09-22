"""Command line interface: ``synthetic-tabular INPUT [options]``."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import List, Optional

from . import __version__
from ._io import load_table, write_table
from .evaluate import evaluate
from .synthesizer import Synthesizer


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser (exposed for documentation and tests)."""
    parser = argparse.ArgumentParser(
        prog="synthetic-tabular",
        description=(
            "Generate realistic synthetic tabular data that preserves distributions "
            "and correlations, then report how faithful it is."
        ),
        epilog=(
            "examples:\n"
            "  synthetic-tabular data.csv                          score a synthetic copy of data.csv\n"
            "  synthetic-tabular data.csv --n 1000 -o synth.csv    write 1000 synthetic rows\n"
            "  synthetic-tabular data.csv --evaluate synth.csv     score an existing synthetic file\n"
            "  synthetic-tabular data.csv --json                   machine-readable report"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="real data: a .csv, .tsv or .parquet file")
    parser.add_argument(
        "--n", type=int, default=None, metavar="ROWS",
        help="number of synthetic rows (default: same as the input)",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="write the synthetic rows to this .csv, .tsv or .parquet file",
    )
    parser.add_argument(
        "--random-state", type=int, default=0, metavar="SEED",
        help="seed for reproducible output (default: 0)",
    )
    parser.add_argument(
        "--no-correlations", action="store_true",
        help="sample columns independently (keep marginals only)",
    )
    parser.add_argument(
        "--evaluate", metavar="PATH",
        help="score an existing synthetic file against INPUT instead of generating",
    )
    parser.add_argument("--json", action="store_true", help="print the fidelity report as JSON")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="do not print the fidelity report"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # already-detached or in-memory streams
                pass
    args = build_parser().parse_args(argv)
    try:
        real = load_table(args.input)
        if args.evaluate:
            synthetic = load_table(args.evaluate)
        else:
            preserve = ("marginals",) if args.no_correlations else ("marginals", "correlations")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                synthesizer = Synthesizer(random_state=args.random_state, preserve=preserve)
                synthetic = synthesizer.fit(real).sample(args.n)
            for warning in caught:
                print(f"note: {warning.message}", file=sys.stderr)
            if args.output:
                write_table(synthetic, args.output)
        report = evaluate(real, synthetic)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(report.summary())
    if args.output and not args.json:
        print(f"Wrote {len(synthetic)} synthetic rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
