"""Command line front end: ``vision-anomaly --good good/ suspect/``.

Prints the same report the library returns, and exits non-zero when something
was flagged, so it drops straight into a build step, a cron job or a shell
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import List, Optional, Sequence

from . import __version__
from ._detector import DEFAULT_SENSITIVITY, Detector
from ._features import DEFAULT_GRID, describe_features
from ._loading import ANALYSIS_SIZE, iter_images
from ._result import DEFAULT_REASONS, DEFAULT_REGIONS, BatchReport

OK = 0
"""Exit code when nothing was flagged."""
FOUND = 1
"""Exit code when at least one image was called anomalous."""
BAD_USAGE = 2
"""Exit code when the arguments or the inputs could not be used at all."""


def _make_utf8_safe() -> None:
    """Stop a non-ASCII file name from raising UnicodeEncodeError on any console.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout. A folder of parts named in Japanese, or one image called
    ``piece-cote-gauche.png``, would then crash the tool rather than report on
    it - and the crash lands at print time, long after the work was done.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):   # pragma: no cover - exotic streams
                pass


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="vision-anomaly",
        description=(
            "Spot unusual images against a set of normal ones. Show it images "
            "you are happy with, then ask it about anything else. No labelled "
            "defects, no training and no model download."
        ),
        epilog=(
            "exit codes: 0 nothing flagged, 1 at least one image flagged, "
            "2 the arguments or inputs could not be used.\n"
            "examples:\n"
            "  vision-anomaly --good known_good/ today/\n"
            "  vision-anomaly --good known_good/ --save-profile belt.json\n"
            "  vision-anomaly --profile belt.json today/ --json > report.json\n"
            "  vision-anomaly --good known_good/ today/ --sensitivity 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="images or directories to check against normal",
    )
    parser.add_argument(
        "--good",
        "-g",
        action="append",
        default=[],
        metavar="PATH",
        help="a known-good image or directory to learn normal from; repeat the "
        "option to give several. One value each, so the images to check stay "
        "readable as the trailing arguments",
    )
    parser.add_argument(
        "--profile",
        "-p",
        metavar="PATH",
        help="load a profile saved earlier instead of fitting one now",
    )
    parser.add_argument(
        "--save-profile",
        metavar="PATH",
        help="write the fitted profile to PATH as JSON",
    )
    parser.add_argument(
        "--sensitivity",
        "-s",
        type=float,
        default=DEFAULT_SENSITIVITY,
        metavar="SIGMAS",
        help="robust sigmas outside normal before an image is flagged "
        "(default: {0:g})".format(DEFAULT_SENSITIVITY),
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=DEFAULT_GRID,
        metavar="N",
        help="cells on a side of the grid regions are reported on "
        "(default: {0})".format(DEFAULT_GRID),
    )
    parser.add_argument(
        "--analysis-size",
        type=int,
        default=ANALYSIS_SIZE,
        metavar="PIXELS",
        help="edge of the square every image is resampled onto "
        "(default: {0})".format(ANALYSIS_SIZE),
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=DEFAULT_REGIONS,
        metavar="N",
        help="how many departing grid cells to list (default: {0})".format(
            DEFAULT_REGIONS
        ),
    )
    parser.add_argument(
        "--reasons",
        type=int,
        default=DEFAULT_REASONS,
        metavar="N",
        help="how many feature families to name (default: {0})".format(
            DEFAULT_REASONS
        ),
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="walk directories rather than reading only their top level",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        metavar="N",
        help="how many flagged images to describe in full (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON instead of text",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="one line per image: score, verdict and name",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="write the report to PATH as well as printing it",
    )
    parser.add_argument(
        "--describe-features",
        action="store_true",
        help="print what the images are measured on, and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="vision-anomaly {0}".format(__version__),
    )
    return parser


def _collect(targets: Sequence[str], recursive: bool, what: str) -> List[str]:
    """Expand files and directories into a flat, sorted list of image paths."""
    found: List[str] = []
    for target in targets:
        if not os.path.exists(target):
            raise ValueError("no such {0}: {1}".format(what, target))
        matched = iter_images(target, recursive=recursive)
        if not matched:
            raise ValueError("no image files found in {0}".format(target))
        for path in matched:
            if path not in found:
                found.append(path)
    return found


def _quiet_lines(report: BatchReport) -> List[str]:
    """One line per image, widest name padded so the scores line up."""
    width = max((len(result.source) for result in report), default=0)
    return [
        "{0:>8.2f}  {1:<9} {2}".format(
            result.score, result.verdict, result.source.ljust(width)
        ).rstrip()
        for result in report
    ]


def _render(report: BatchReport, args: argparse.Namespace) -> str:
    """The whole report as the text that will be printed and maybe written."""
    if args.json:
        return report.to_json()
    if args.quiet:
        return "\n".join(_quiet_lines(report))
    return report.summary(limit=args.worst)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``vision-anomaly`` command.

    Args:
        argv: arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit code: :data:`OK`, :data:`FOUND` or :data:`BAD_USAGE`.
    """
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.describe_features:
        print(describe_features())
        return OK

    if not args.good and not args.profile:
        parser.error(
            "give either --good with known-good images, or --profile with a "
            "profile saved earlier"
        )
    if args.good and args.profile:
        parser.error("--good and --profile do the same job; give one or the other")
    if not args.images and not args.save_profile:
        parser.error(
            "give images to check, or --save-profile to only fit a profile"
        )

    try:
        detector = _prepare(args)
    except ValueError as exc:
        print("vision-anomaly: {0}".format(exc), file=sys.stderr)
        return BAD_USAGE

    if args.save_profile:
        try:
            written = detector.save(args.save_profile)
        except OSError as exc:
            print(
                "vision-anomaly: cannot write profile {0}: {1}".format(
                    args.save_profile, exc.strerror or exc
                ),
                file=sys.stderr,
            )
            return BAD_USAGE
        print("profile written to {0}".format(written), file=sys.stderr)
        if not args.images:
            return OK

    try:
        images = _collect(args.images, args.recursive, "image or directory")
        report = detector.predict_batch(images)
    except ValueError as exc:
        print("vision-anomaly: {0}".format(exc), file=sys.stderr)
        return BAD_USAGE

    text = _render(report, args)
    print(text)
    if args.output:
        try:
            _write(args.output, text)
        except OSError as exc:
            print(
                "vision-anomaly: cannot write {0}: {1}".format(
                    args.output, exc.strerror or exc
                ),
                file=sys.stderr,
            )
            return BAD_USAGE
    return FOUND if report.n_anomalous else OK


def _prepare(args: argparse.Namespace) -> Detector:
    """Load or fit the detector the run needs."""
    options = {
        "sensitivity": args.sensitivity,
        "max_regions": args.regions,
        "max_reasons": args.reasons,
    }
    if args.profile:
        return Detector.load(args.profile, **options)
    good = _collect(args.good, args.recursive, "known-good image or directory")
    detector = Detector(
        grid=args.grid, analysis_size=args.analysis_size, **options
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        detector.fit(good)
    for warning in caught:
        print("vision-anomaly: {0}".format(warning.message), file=sys.stderr)
    return detector


def _write(path: str, text: str) -> None:
    """Write the report, always UTF-8, so non-ASCII names survive the trip."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


if __name__ == "__main__":              # pragma: no cover - exercised as a module
    sys.exit(main())
