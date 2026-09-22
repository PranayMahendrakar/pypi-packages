"""``document-quality`` on the command line: a preflight gate for an OCR run.

The exit code is the point. ``0`` when every page assessed is ready, ``1`` when
any page is not, ``2`` when nothing could be read at all, so a shell script can
put this in front of a paid OCR run and stop before it spends anything::

    document-quality scans/ --dpi 300 --only-problems || exit 1

Everything else is argparse. Paths may be files or directories; a directory is
walked for the suffixes Pillow normally reads.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._assess import assess_batch
from ._images import IMAGE_SUFFIXES
from ._report import BatchReport
from ._thresholds import describe_thresholds

logger = logging.getLogger(__name__)

#: Every page assessed is ready to OCR.
EXIT_OK = 0
#: At least one page is not ready.
EXIT_NOT_READY = 1
#: Nothing could be read at all.
EXIT_UNREADABLE = 2


def make_utf8_tolerant() -> None:
    """Let the console print any page name without raising.

    A scan called ``facture_reçu.png`` must not crash the tool because the
    console happens to be code page 1252, and it must survive a pipe, a
    subprocess capture and a CI log just the same.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed/odd stream
                pass


def collect_paths(given: Sequence[str]) -> List[str]:
    """Expand files and directories into a sorted list of image paths.

    Directories are walked; a file named outright is taken whatever its suffix,
    so an image with an unusual extension is still assessable.

    Raises:
        FileNotFoundError: if one of the given paths does not exist.
    """
    found: List[str] = []
    for item in given:
        if os.path.isdir(item):
            for root, _, names in os.walk(item):
                for name in sorted(names):
                    if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                        found.append(os.path.join(root, name))
        elif os.path.exists(item):
            found.append(item)
        else:
            raise FileNotFoundError("{0!r} does not exist".format(item))
    return found


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="document-quality",
        description=(
            "Decide whether a scanned page is good enough to OCR before you "
            "spend money OCRing it."
        ),
        epilog=(
            "Exit code 0 when every page is ready, 1 when any page is not, "
            "2 when nothing could be read."
        ),
    )
    parser.add_argument(
        "paths", nargs="*", metavar="PATH",
        help="image files, or folders of them",
    )
    parser.add_argument(
        "--dpi", type=float, default=None,
        help="scan resolution, for files that do not carry their own",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the report as JSON instead of text",
    )
    parser.add_argument(
        "--output", metavar="FILE", default=None,
        help="write the JSON report to FILE as well",
    )
    parser.add_argument(
        "--only-problems", action="store_true",
        help="print only the pages that are not ready to OCR",
    )
    parser.add_argument(
        "--worst", type=int, default=5, metavar="N",
        help="how many problem pages the batch summary names (default 5)",
    )
    parser.add_argument(
        "--no-orientation", action="store_true",
        help="skip the quarter-turn check on pages you know are upright",
    )
    parser.add_argument(
        "--threshold", action="append", default=[], metavar="NAME=VALUE",
        help="override one threshold; repeatable",
    )
    parser.add_argument(
        "--list-thresholds", action="store_true",
        help="print every threshold, its value and what it means, then exit",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="print nothing; report the verdict through the exit code alone",
    )
    parser.add_argument(
        "--version", action="version", version="document-quality {0}".format(__version__)
    )
    return parser


def parse_overrides(given: Sequence[str]) -> Dict[str, float]:
    """Turn ``NAME=VALUE`` strings into a threshold override dict.

    Raises:
        ValueError: if an item has no ``=`` or its value is not a number.
    """
    overrides: Dict[str, float] = {}
    for item in given:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                "--threshold wants NAME=VALUE, got {0!r}".format(item)
            )
        try:
            overrides[name.strip()] = float(value)
        except ValueError:
            raise ValueError(
                "threshold {0} must be a number, got {1!r}".format(name.strip(), value)
            ) from None
    return overrides


def render(batch: BatchReport, options: argparse.Namespace) -> str:
    """The text this run prints, respecting ``--json`` and ``--only-problems``."""
    if options.json:
        return json.dumps(batch.to_dict(), indent=2, ensure_ascii=False)
    if len(batch.reports) == 1 and not options.only_problems:
        text = batch.reports[0].summary()
        if batch.failures:
            text += "\n" + batch.summary(worst=0).splitlines()[-1]
        return text
    if options.only_problems:
        problems = batch.not_ready
        if not problems:
            return "All {0} page(s) are ready to OCR.".format(len(batch.reports))
        parts = [item.summary() for item in problems]
        return "\n\n".join(parts)
    return batch.summary(worst=max(0, options.worst))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool. Returns the process exit code."""
    make_utf8_tolerant()
    parser = build_parser()
    options = parser.parse_args(argv)

    try:
        overrides = parse_overrides(options.threshold)
    except ValueError as error:
        parser.error(str(error))
        return EXIT_UNREADABLE          # pragma: no cover - parser.error exits

    if options.list_thresholds:
        print(describe_thresholds(overrides or None))
        return EXIT_OK

    if not options.paths:
        parser.print_help()
        return EXIT_OK

    try:
        paths = collect_paths(options.paths)
    except FileNotFoundError as error:
        print("document-quality: {0}".format(error), file=sys.stderr)
        return EXIT_UNREADABLE

    if not paths:
        print(
            "document-quality: no images found in {0}".format(
                ", ".join(options.paths)
            ),
            file=sys.stderr,
        )
        return EXIT_UNREADABLE

    batch = assess_batch(
        paths,
        dpi=options.dpi,
        thresholds=overrides or None,
        check_orientation=not options.no_orientation,
    )

    text = render(batch, options)
    if not options.quiet:
        print(text)
    if options.output:
        payload: Any = batch.to_dict()
        with open(options.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        if not options.quiet:
            print("Wrote {0}".format(options.output))

    if not batch.reports:
        return EXIT_UNREADABLE
    return EXIT_OK if not batch.not_ready else EXIT_NOT_READY


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
