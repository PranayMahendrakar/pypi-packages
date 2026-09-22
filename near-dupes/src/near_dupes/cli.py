"""Command line interface: ``near-dupes INPUT [INPUT ...] [options]``."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from . import __version__
from ._core import DuplicateFinder, DuplicateResult
from ._images import IMAGE_SUFFIXES, looks_like_image_path
from ._records import is_table_path, load_table

_DESCRIPTION = """Find near-duplicate texts, table rows or images and optionally write the deduplicated set.

INPUT is a .csv/.parquet file (rows are compared), a text file (one item per
line), an image file, a directory of images, or several image files / globs."""

_EPILOG = """examples:
  near-dupes contacts.csv --key name email --output contacts_clean.csv
  near-dupes tickets.txt --threshold 0.9 --json
  near-dupes photos/ --threshold 0.9"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="near-dupes",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="file(s), directory or glob to scan")
    p.add_argument("--kind", choices=["auto", "text", "records", "images"], default="auto", help="how to read INPUT (default: auto)")
    p.add_argument("--threshold", type=float, default=0.85, help="minimum similarity, 1.0 = exact only (default: 0.85)")
    p.add_argument("--key", nargs="+", metavar="COL", help="column(s) to compare rows on (default: all)")
    p.add_argument("--n-gram", type=int, default=3, dest="n_gram", help="character shingle length (default: 3)")
    p.add_argument("--num-perm", type=int, default=128, dest="num_perm", help="MinHash permutations (default: 128)")
    p.add_argument("--random-state", type=int, default=0, dest="random_state", help="MinHash seed (default: 0)")
    p.add_argument("--json", action="store_true", help="print the full result as JSON instead of the summary")
    p.add_argument("--output", metavar="PATH", help="write the deduplicated items here (.csv/.parquet for tables, one item per line otherwise)")
    p.add_argument("--version", action="version", version=f"near-dupes {__version__}")
    return p


def _expand(inputs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in inputs:
        if any(ch in raw for ch in "*?["):
            matches = sorted(glob.glob(raw))
            if not matches:
                raise FileNotFoundError(f"no files match {raw!r}")
            out.extend(matches)
        else:
            out.append(raw)
    return out


def _images_in_dir(directory: str) -> List[str]:
    return [str(p) for p in sorted(Path(directory).iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def load_inputs(inputs: Sequence[str], kind: str = "auto") -> Tuple[Any, str]:
    """Turn CLI inputs into ``(items, kind)`` for ``DuplicateFinder``."""
    paths = _expand(inputs)
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p!r} does not exist")
    single = len(paths) == 1 and not os.path.isdir(paths[0])
    if kind == "records" or (kind == "auto" and single and is_table_path(paths[0])):
        if not single:
            raise ValueError("--kind records needs a single .csv or .parquet file")
        return load_table(paths[0]), "records"
    if kind == "text":
        lines: List[str] = []
        for p in paths:
            if os.path.isdir(p):
                raise ValueError(f"{p!r} is a directory; --kind text needs text files")
            lines.extend(_read_lines(p))
        return lines, "text"
    if kind == "auto" and single and not looks_like_image_path(paths[0]):
        return _read_lines(paths[0]), "text"
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(_images_in_dir(p))
        elif kind == "images" or looks_like_image_path(p):
            files.append(p)
        else:
            raise ValueError(f"{p!r} is not an image file; use --kind text for text files")
    return files, "images"


def write_output(path: str, items: Any, result: DuplicateResult) -> None:
    """Write ``result.dedupe(items)`` to ``path``."""
    kept = result.dedupe(items)
    if result.kind == "records":
        if path.lower().endswith((".parquet", ".pq")):
            kept.to_parquet(path, index=False)
        else:
            kept.to_csv(path, index=False)
        return
    with open(path, "w", encoding="utf-8") as fh:
        for item in kept:
            fh.write(f"{item}\n")


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        items, kind = load_inputs(args.inputs, args.kind)
        key: Any = None
        if args.key:
            key = args.key[0] if len(args.key) == 1 else list(args.key)
        finder = DuplicateFinder(
            kind=kind,
            threshold=args.threshold,
            key=key,
            n_gram=args.n_gram,
            num_perm=args.num_perm,
            random_state=args.random_state,
        )
        result = finder.find(items)
        if args.output:
            write_output(args.output, items, result)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(result.summary())
            if args.output:
                print(f"  wrote {len(result.keep_indices):,} items to {args.output}")
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"near-dupes: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
