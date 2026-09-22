"""Command line interface: ``image-quality-ai [PATH ...] [options]``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, NoReturn, Optional, Sequence

from . import __version__
from ._core import assess_batch
from ._loading import IMAGE_SUFFIXES
from ._report import BatchReport
from ._thresholds import Thresholds, describe_thresholds, resolve_thresholds

_DESCRIPTION = """Find the photos that are too blurry, too dark, too bright, too grainy,
too flat or too badly framed to be worth feeding to a model.

PATH is an image file or a directory of them. Directories are read in sorted
order; add --recursive to walk into subdirectories as well."""

_EPILOG = """examples:
  image-quality-ai photo.jpg
  image-quality-ai shots/ --recursive --workers 4
  image-quality-ai shots/ --json > report.json
  image-quality-ai shots/ --worst 20 --quiet
  image-quality-ai photo.jpg --threshold sharpness_blurry=40
  image-quality-ai shots/ --fail-under 60
  image-quality-ai --list-thresholds"""

#: Exit code when --fail-under is not met.
EXIT_BELOW_THRESHOLD = 2
#: Exit code for a usage or read error.
EXIT_ERROR = 1


class _Parser(argparse.ArgumentParser):
    """argparse, with usage errors exiting 1 instead of its default 2.

    Exit code 2 is reserved for ``--fail-under``, so a CI step can tell "an
    image scored too low" apart from "I typed the flag wrong". ``--help`` and
    ``--version`` still exit 0.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, "{0}: error: {1}\n".format(self.prog, message))


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``--help`` can be tested."""
    parser = _Parser(
        prog="image-quality-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", metavar="PATH",
                        help="image file(s) or directory(ies) to assess")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="walk into subdirectories of any directory given")
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help="threads to assess with (default: 1)")
    parser.add_argument("--worst", type=int, default=5, metavar="N",
                        help="how many of the worst images to list (default: 5)")
    parser.add_argument("--threshold", action="append", default=[], metavar="NAME=VALUE",
                        dest="overrides",
                        help="override one threshold; repeatable, e.g. sharpness_blurry=40")
    parser.add_argument("--list-thresholds", action="store_true", dest="list_thresholds",
                        help="print every threshold with its default and exit")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="one line per image instead of the full report")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON instead of the summary")
    parser.add_argument("--output", metavar="PATH",
                        help="write the output to this file as UTF-8")
    parser.add_argument("--fail-under", type=float, default=None, metavar="SCORE",
                        dest="fail_under",
                        help="exit with code 2 if any image scores below SCORE")
    parser.add_argument("--version", action="version",
                        version="image-quality-ai {0}".format(__version__))
    return parser


def parse_overrides(pairs: Sequence[str]) -> Dict[str, float]:
    """Turn ``["sharpness_blurry=40"]`` into ``{"sharpness_blurry": 40.0}``.

    Raises:
        ValueError: if a pair is malformed or names an unknown threshold.
    """
    overrides: Dict[str, float] = {}
    known = set(Thresholds.field_names())
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                "--threshold needs NAME=VALUE, got {0!r}".format(pair)
            )
        name, _, raw = pair.partition("=")
        name = name.strip()
        if name not in known:
            raise ValueError("unknown threshold {0!r}; see --list-thresholds".format(name))
        try:
            overrides[name] = float(raw.strip())
        except ValueError:
            raise ValueError(
                "threshold {0} needs a number, got {1!r}".format(name, raw.strip())
            ) from None
    return overrides


def collect_paths(paths: Sequence[str], recursive: bool) -> List[str]:
    """Expand directories into the image files inside them, in sorted order.

    Raises:
        FileNotFoundError: if one of the paths does not exist.
    """
    found: List[str] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError("no such file or directory: {0}".format(path))
        if os.path.isfile(path):
            found.append(path)
            continue
        if recursive:
            for root, directories, names in os.walk(path):
                directories.sort()
                for name in sorted(names):
                    if name.lower().endswith(IMAGE_SUFFIXES):
                        found.append(os.path.join(root, name))
        else:
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if os.path.isfile(full) and name.lower().endswith(IMAGE_SUFFIXES):
                    found.append(full)
    return found


def render(report: BatchReport, as_json: bool, quiet: bool, worst: int) -> str:
    """The text the CLI prints for one run.

    ``worst`` is honoured whatever its value: the batch report is asked for a
    worst-first list exactly that long, so ``--worst 2`` lists two and
    ``--worst 8`` lists eight, each once. ``--worst 0`` leaves the list out.
    Previously anything from 0 to 5 was silently ignored and anything above 5
    printed a second block repeating the first five rows.
    """
    if as_json:
        return report.to_json()
    if quiet:
        lines = [
            "{0:5.1f} {1} {2} {3}".format(
                item.score,
                item.grade,
                "usable    " if item.usable else "not usable",
                item.source,
            )
            for item in report.results
        ]
        for failure in report.failures:
            lines.append("    - !  unreadable {0}".format(failure["source"]))
        return "\n".join(lines) if lines else "No images found."
    if len(report.results) == 1 and not report.failures:
        return report.results[0].summary()
    return report._summary_text(max(0, int(worst)))


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a character from a filename."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):   # pragma: no cover - closed or odd streams
                pass


def _write(text: str, output: Optional[str]) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_thresholds:
        _write(describe_thresholds(resolve_thresholds(overrides or None)), args.output)
        return 0

    if not args.paths:
        parser.print_help()
        return EXIT_ERROR
    if args.workers < 1:
        parser.error("--workers must be 1 or more, got {0}".format(args.workers))
    if args.worst < 0:
        parser.error("--worst must be zero or more, got {0}".format(args.worst))

    try:
        files = collect_paths(args.paths, args.recursive)
    except OSError as exc:
        print("image-quality-ai: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    if not files:
        print(
            "image-quality-ai: no images found in {0}".format(", ".join(args.paths)),
            file=sys.stderr,
        )
        return EXIT_ERROR

    report = assess_batch(files, workers=args.workers, thresholds=overrides or None)

    try:
        _write(render(report, args.json, args.quiet, args.worst), args.output)
    except OSError as exc:
        print("image-quality-ai: cannot write output: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    if args.fail_under is not None:
        below = [item for item in report.results if item.score < args.fail_under]
        if below:
            print(
                "image-quality-ai: {0} image(s) scored below {1:g}".format(
                    len(below), args.fail_under
                ),
                file=sys.stderr,
            )
            return EXIT_BELOW_THRESHOLD
    if report.failures and not report.results:
        return EXIT_ERROR
    return 0


if __name__ == "__main__":              # pragma: no cover - module entry
    sys.exit(main())
