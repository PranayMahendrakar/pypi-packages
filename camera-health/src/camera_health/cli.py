"""Command line front end: ``camera-health frame.png``.

Prints the same summary the library returns, and exits non-zero when the camera
is not usable, so it drops straight into cron, a health probe or a CI step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._frames import list_images
from .checks import check, check_stream
from .thresholds import DEFAULT_THRESHOLDS, Thresholds

OK = 0
"""Exit code when every frame is usable."""
FAULTY = 1
"""Exit code when a frame is not ok - a critical fault was found."""
BAD_USAGE = 2
"""Exit code when the input could not be read at all."""


def _make_utf8_safe() -> None:
    """Stop a non-ASCII file name from raising UnicodeEncodeError on any console.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout. A camera folder full of names like ``camera-est-2`` in Cyrillic
    or Japanese would then crash the tool rather than report on it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def _parse_override(text: str) -> Dict[str, Any]:
    """Turn ``--set dark_mean=20`` into ``{"dark_mean": 20.0}``.

    Raises:
        argparse.ArgumentTypeError: the text is not ``name=value``, the name is
            not a threshold, or the value is not a number.
    """
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "--set wants name=value, for example dark_mean=20; got {!r}".format(text)
        )
    name, _, raw = text.partition("=")
    name = name.strip()
    known = set(DEFAULT_THRESHOLDS.to_dict())
    if name not in known:
        raise argparse.ArgumentTypeError(
            "unknown threshold {!r}; --show-thresholds lists all {} names".format(
                name, len(known)
            )
        )
    try:
        value: Any = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "threshold {} wants a number; got {!r}".format(name, raw)
        ) from None
    if isinstance(getattr(DEFAULT_THRESHOLDS, name), int):
        value = int(value)
    return {name: value}


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="camera-health",
        description=(
            "Tell whether a camera is actually seeing anything: obstruction, "
            "defocus, darkness, overexposure, a frozen feed, tampering, colour "
            "cast and heavy noise, judged from the frames themselves."
        ),
        epilog=(
            "exit codes: 0 every frame usable, 1 a critical fault was found, "
            "2 the input could not be read. "
            "examples: camera-health frame.png -- "
            "camera-health frames/ --reference day1.png -- "
            "camera-health a.png b.png --json --output report.json"
        ),
    )
    parser.add_argument(
        "frames",
        nargs="*",
        help=(
            "one image to check on its own, several images to check as a run, or "
            "one directory of images (read in natural filename order)"
        ),
    )
    parser.add_argument(
        "--reference",
        metavar="PATH",
        help="an image of how this view is supposed to look; enables the tampering check",
    )
    parser.add_argument(
        "--previous",
        metavar="PATH",
        help="the frame just before a single frame; enables the frozen-feed check",
    )
    parser.add_argument(
        "--freeze-frames",
        type=int,
        default=5,
        metavar="N",
        help="identical frames in a row before the feed is called frozen (default: 5)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=30,
        metavar="N",
        help="frames the drift comparison looks back over (default: 30)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        type=_parse_override,
        metavar="NAME=VALUE",
        help="override one threshold, repeatable (for example --set dark_mean=20)",
    )
    parser.add_argument(
        "--show-thresholds",
        action="store_true",
        help="print every threshold and its value as JSON, then exit",
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the report to this file (UTF-8) as well as printing it",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print nothing; only set the exit code"
    )
    parser.add_argument(
        "--version", action="version", version="camera-health {}".format(__version__)
    )
    return parser


def _render(result: Any, as_json: bool) -> str:
    """The text to print and to save, in whichever form was asked for."""
    if as_json:
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=False)
    return result.summary()


def _expand(paths: Sequence[str]) -> List[str]:
    """One directory becomes its images; anything else is passed through."""
    if len(paths) == 1 and os.path.isdir(paths[0]):
        return list_images(paths[0])
    return list(paths)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool.

    Args:
        argv: arguments to parse; None reads ``sys.argv``.

    Returns:
        0 when every frame is usable, 1 when a critical fault was found, 2 when
        the input could not be read.
    """
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    thresholds: Thresholds = DEFAULT_THRESHOLDS
    for override in args.overrides:
        thresholds = thresholds.replace(**override)

    if args.show_thresholds:
        if not args.quiet:
            print(json.dumps(thresholds.to_dict(), indent=2, ensure_ascii=False))
        return OK

    if not args.frames:
        parser.print_usage(sys.stderr)
        print(
            "camera-health: give at least one image, or a directory of images "
            "(--show-thresholds needs no input)",
            file=sys.stderr,
        )
        return BAD_USAGE

    freeze_frames = max(2, int(args.freeze_frames))
    try:
        paths = _expand(args.frames)
        if len(paths) == 1:
            result: Any = check(
                paths[0],
                reference=args.reference,
                previous=args.previous,
                thresholds=thresholds,
                freeze_frames=freeze_frames,
            )
        else:
            if args.previous:
                print(
                    "camera-health: --previous applies to a single frame; in a run, the "
                    "previous frame is the one before it in the run",
                    file=sys.stderr,
                )
                return BAD_USAGE
            result = check_stream(
                paths,
                reference=args.reference,
                freeze_frames=freeze_frames,
                history=max(2, int(args.history)),
                thresholds=thresholds,
            )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print("camera-health: {}".format(exc), file=sys.stderr)
        return BAD_USAGE

    text = _render(result, args.json)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print("camera-health: cannot write {}: {}".format(args.output, exc), file=sys.stderr)
            return BAD_USAGE
    if not args.quiet:
        print(text)
    return OK if result.ok else FAULTY


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
