"""Command line interface: ``semantic-dedup INPUT [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from ._core import KEEP_MODES, METHODS, Deduper
from ._io import load_texts, write_texts

_DESCRIPTION = """Remove passages that repeat the same meaning, not just the same words.

INPUT is a .txt file (one text per line), a .csv/.tsv file (a text column is
picked automatically, or name it with --column) or a .jsonl file."""

_EPILOG = """examples:
  semantic-dedup notes.txt
  semantic-dedup faq.csv --column question --threshold 0.9
  semantic-dedup notes.txt --output clean.txt
  semantic-dedup notes.jsonl --json > report.json"""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can poke at it."""
    parser = argparse.ArgumentParser(
        prog="semantic-dedup",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", metavar="INPUT", help="a .txt, .csv/.tsv or .jsonl file")
    parser.add_argument(
        "--threshold", type=float, default=0.82,
        help="minimum similarity, 1.0 = exact matches only (default: 0.82)",
    )
    parser.add_argument(
        "--method", choices=list(METHODS), default="auto",
        help="similarity engine (default: auto, which is minhash above 1000 texts)",
    )
    parser.add_argument(
        "--keep", choices=list(KEEP_MODES), default="longest",
        help="which member of a duplicate group survives (default: longest)",
    )
    parser.add_argument(
        "--column", metavar="NAME",
        help="csv column or json key holding the text (default: detected)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="only report duplicates, do not choose anything to remove",
    )
    parser.add_argument(
        "--max-groups", type=int, default=5, dest="max_groups",
        help="how many duplicate groups to show in the summary (default: 5)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the full result as JSON instead of the summary",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="write the surviving texts here, one per line, UTF-8",
    )
    parser.add_argument("--version", action="version", version=f"semantic-dedup {__version__}")
    return parser


def _make_console_safe() -> None:
    """Never let a non-ASCII text crash the run under a pipe or a CI log."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, LookupError):  # pragma: no cover - odd streams
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        texts: List[str] = load_texts(args.input, args.column)
        deduper = Deduper(
            threshold=args.threshold,
            method=args.method,
            keep=args.keep,
            column=args.column,
        )
        result = deduper.run(texts, drop=not args.report)
        if args.output:
            write_texts(args.output, result.texts)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(result.summary(max_groups=args.max_groups))
            if args.output:
                print(f"  wrote {result.n_kept} texts to {args.output}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"semantic-dedup: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
