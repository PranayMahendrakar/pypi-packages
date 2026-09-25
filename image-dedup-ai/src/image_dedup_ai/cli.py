"""Command line interface: ``image-dedup-ai PATH [PATH ...] [options]``."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._hashing import METHODS
from ._index import Index, ignored_note
from ._result import DedupeResult, Match

_DESCRIPTION = """Find duplicate and near-duplicate images in folders and report what one copy per
group would free. Perceptual hashing is a heuristic: resized and re-compressed
copies are found; rotated, flipped or heavily cropped copies are not.

PATH is a folder (scanned for image files, recursively by default) or an image
file. With --index FILE the hashes are kept, so the next run only reads new or
edited files. Nothing is ever deleted or modified."""

_EPILOG = """examples:
  image-dedup-ai D:/Photos
  image-dedup-ai D:/Photos --index photos.idx --threshold 0.95 --output dupes.csv
  image-dedup-ai --index photos.idx --near holiday.jpg -k 10
  image-dedup-ai D:/Photos --json > dupes.json"""


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser (exposed for testing and documentation)."""
    p = argparse.ArgumentParser(
        prog="image-dedup-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="*", metavar="PATH", help="folders or image files to scan")
    p.add_argument("--index", metavar="FILE", help="SQLite file that keeps the hashes between runs (default: memory only)")
    p.add_argument("--method", choices=list(METHODS), default="phash", help="hash to compare with (default: phash)")
    p.add_argument("--threshold", type=float, default=0.9, help="minimum similarity in (0, 1], 1.0 = identical hashes only (default: 0.9)")
    p.add_argument("--hash-size", type=int, default=8, dest="hash_size", help="hash side; bits = hash_size**2 (default: 8)")
    p.add_argument("--no-recursive", action="store_true", dest="no_recursive", help="do not descend into subfolders")
    p.add_argument("--workers", type=int, default=None, help="decoding threads (default: up to 8)")
    p.add_argument("--near", metavar="IMAGE", help="list the indexed images most similar to IMAGE instead of grouping")
    p.add_argument("-k", type=int, default=5, help="how many neighbours --near lists (default: 5)")
    p.add_argument("--json", action="store_true", help="print the result as JSON instead of the summary")
    p.add_argument("--output", metavar="FILE", help="also write the result: .csv (one row per image in a group) or .json")
    p.add_argument("--version", action="version", version=f"image-dedup-ai {__version__}")
    return p


def _make_console_safe() -> None:
    """Never crash on a console or pipe that cannot encode a file name."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed or unusual streams
                pass


def write_csv(path: str, result: DedupeResult) -> None:
    """One row per image in a duplicate group; ``action`` is ``keep`` or ``copy``."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "action", "path", "width", "height", "bytes", "similarity_to_keep", "identical_to_keep"])
        for number, group in enumerate(result.details, start=1):
            for m in group:
                writer.writerow(
                    [number, "keep" if m.kept else "copy", m.path, m.width, m.height, m.bytes, f"{m.similarity:.6f}", int(m.identical)]
                )


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _near_payload(matches: List[Match]) -> List[Dict[str, Any]]:
    return [{"path": m.path, "similarity": m.similarity} for m in matches]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.paths and not args.index:
        parser.error("give at least one PATH to scan, or --index FILE to use an existing index")
    try:
        for path in args.paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path!r} does not exist")
        with Index(args.index, hash_size=args.hash_size) as idx:
            scan: Dict[str, Any] = {}
            if args.paths:
                idx.add(list(args.paths), recursive=not args.no_recursive, workers=args.workers)
                scan = idx.last_add.to_dict()
            if args.near:
                matches = idx.near(args.near, k=args.k, method=args.method)
                payload: Any = _near_payload(matches)
                if args.output:
                    _write_json(args.output, payload)
                if args.json:
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                else:
                    if args.paths:
                        print(idx.last_add.summary())
                    print(f"{len(matches)} nearest of {idx.count():,} indexed images to {args.near}:")
                    for m in matches:
                        print(f"  {m.similarity:.3f}  {m.path}")
                return 0
            result = idx.find_duplicates(threshold=args.threshold, method=args.method)
            note = ignored_note(idx.last_add)
            if note:
                result.notes.insert(0, note)
            payload = result.to_dict()
            if scan:
                payload["scan"] = scan
            if args.output:
                if args.output.lower().endswith(".csv"):
                    write_csv(args.output, result)
                else:
                    _write_json(args.output, payload)
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                if args.paths:
                    print(idx.last_add.summary())
                print(result.summary())
                if args.output:
                    print(f"\nwrote {args.output}")
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(f"image-dedup-ai: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
