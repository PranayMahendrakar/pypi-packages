"""Command line front end: ``video-event-detector frames/ --fps 10``.

Prints the same summary the library returns. Exit status: 0 when nothing was
found, 1 when at least one event was reported, 2 when the input could not be
read - so it drops straight into a script or a scheduled job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .api import detect_events

NOTHING_FOUND = 0
"""Exit code when the frames were read and no event was reported."""
EVENTS_FOUND = 1
"""Exit code when at least one event was reported."""
BAD_INPUT = 2
"""Exit code when the input could not be read at all."""


def _make_utf8_safe() -> None:
    """Stop non-ASCII paths from raising UnicodeEncodeError on any console.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout; a frame folder named in Cyrillic or Japanese would then crash
    the tool instead of being reported on.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def _number(kind: type, name: str):
    def parse(text: str) -> Any:
        try:
            return kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError("{} wants a number; got {!r}".format(name, text)) from None

    return parse


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="video-event-detector",
        description=(
            "Look for sudden motion, falls, crowding and abandoned objects in a "
            "sequence of still frames from a fixed camera. These are motion "
            "heuristics: every event is a hypothesis to review, not a verdict. "
            "Video files are not decoded; extract frames first, for example: "
            "ffmpeg -i clip.mp4 -vf fps=10 frames/%05d.png"
        ),
        epilog=(
            "exit status: 0 nothing found, 1 at least one event, 2 the input could not "
            "be read. Example: video-event-detector frames/ --fps 10 --dwell 30"
        ),
    )
    parser.add_argument(
        "frames",
        nargs="+",
        help="a directory of numbered image files, or image files in order",
    )
    parser.add_argument(
        "--fps",
        type=_number(float, "--fps"),
        default=None,
        help="frames per second; times are then in seconds (default: frame numbers)",
    )
    parser.add_argument(
        "--sensitivity",
        type=_number(float, "--sensitivity"),
        default=0.5,
        help="0-1, higher finds weaker motion and raises more false alarms (default 0.5)",
    )
    parser.add_argument(
        "--background-frames",
        type=_number(int, "--background-frames"),
        default=30,
        help="frames used to learn the empty scene before events can be reported (default 30)",
    )
    parser.add_argument(
        "--dwell",
        type=_number(float, "--dwell"),
        default=None,
        help="how long a new object must stay still to count as abandoned, in seconds "
        "with --fps, else frames (default: as many frames as --background-frames)",
    )
    parser.add_argument(
        "--hold",
        type=_number(float, "--hold"),
        default=None,
        help="how long crowding or a collapsed shape must last, same units as --dwell",
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument(
        "--output", metavar="PATH", default=None, help="also write the JSON result to PATH"
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def _source(paths: List[str]) -> Any:
    if len(paths) == 1:
        return paths[0]
    for path in paths:
        if os.path.isdir(path):
            raise ValueError(
                "{} is a directory; pass one directory of frames, or image files, not both".format(path)
            )
    return list(paths)


def _fail(message: str, args: argparse.Namespace) -> int:
    if args.json:
        payload: Dict[str, Any] = {"error": message, "input": list(args.frames)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("video-event-detector: error: {}".format(message), file=sys.stderr)
    return BAD_INPUT


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``video-event-detector`` command."""
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = detect_events(
            _source(list(args.frames)),
            fps=args.fps,
            sensitivity=args.sensitivity,
            background_frames=args.background_frames,
            dwell=args.dwell,
            hold=args.hold,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        return _fail(str(exc), args)

    result = report.to_dict()
    if args.output:
        try:
            folder = os.path.dirname(os.path.abspath(args.output))
            os.makedirs(folder, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            return _fail("cannot write {}: {}".format(args.output, exc), args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        if args.output:
            print("wrote {}".format(args.output))
    return EVENTS_FOUND if report.found else NOTHING_FOUND


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
