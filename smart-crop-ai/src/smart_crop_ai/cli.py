"""Command line interface: ``smart-crop-ai IMAGE [IMAGE ...] [options]``.

By default this is a dry run: it prints what it would crop and how confident it
is, and writes nothing. Add ``--output`` or ``--out-dir`` to actually save.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, NoReturn, Optional, Sequence

from . import __version__
from ._core import DEFAULT_PADDING, DEFAULT_STRATEGY, STRATEGIES, crop
from ._images import IMAGE_SUFFIXES, resize_exact
from ._result import CropResult

_DESCRIPTION = """Crop to the interesting part of an image instead of the middle of it.

IMAGE is an image file or a directory of them. Nothing is written unless you ask
for it with --output or --out-dir, so you can look at the confidence first."""

_EPILOG = """examples:
  smart-crop-ai photo.jpg --ratio 16:9
  smart-crop-ai photo.jpg --width 800 --height 600 --output hero.jpg
  smart-crop-ai shots/ --ratio 1:1 --out-dir square/ --strategy edges
  smart-crop-ai shots/ --recursive --quiet
  smart-crop-ai photo.jpg --ratio 1:1 --json
  smart-crop-ai shots/ --ratio 4:3 --out-dir out/ --min-confidence 0.1

exit codes:
  0  all good
  1  a usage or read error
  2  an image came back under --min-confidence"""

#: Exit code when --min-confidence is not met.
EXIT_LOW_CONFIDENCE = 2
#: Exit code for a usage or read error.
EXIT_ERROR = 1


class _Parser(argparse.ArgumentParser):
    """argparse, with usage errors exiting 1 instead of its default 2.

    Exit code 2 is reserved for ``--min-confidence``, so a CI step can tell "the
    crop was a guess" apart from "I typed the flag wrong". ``--help`` and
    ``--version`` still exit 0.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, "{0}: error: {1}\n".format(self.prog, message))


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``--help`` can be tested."""
    parser = _Parser(
        prog="smart-crop-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", metavar="IMAGE",
        help="image file(s) or directory(ies) to crop",
    )
    parser.add_argument("--width", type=int, default=None, metavar="N",
                        help="target width in pixels")
    parser.add_argument("--height", type=int, default=None, metavar="N",
                        help="target height in pixels")
    parser.add_argument("--ratio", default=None, metavar="W:H",
                        help="aspect ratio instead of a size, e.g. 16:9 or 1.0")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=list(STRATEGIES),
                        help="how to find the subject (default: {0})".format(DEFAULT_STRATEGY))
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING, metavar="F",
                        help="breathing room around the subject, 0 to 0.45 "
                             "(default: {0})".format(DEFAULT_PADDING))
    parser.add_argument("--thumb", default=None, metavar="WxH",
                        help="after cropping, resize to exactly WxH (or one number "
                             "for a square)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="walk into subdirectories of any directory given")
    parser.add_argument("--output", metavar="PATH",
                        help="write the crop here; one input image only")
    parser.add_argument("--out-dir", metavar="DIR", dest="out_dir",
                        help="write every crop into DIR, keeping each file name")
    parser.add_argument("--suffix", default="", metavar="TEXT",
                        help="add TEXT to each written file name, e.g. -crop")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="one line per image instead of the full report")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON instead of the summary")
    parser.add_argument("--report", metavar="PATH",
                        help="write the printed report to this file as UTF-8")
    parser.add_argument("--min-confidence", type=float, default=None, metavar="F",
                        dest="min_confidence",
                        help="exit with code 2 if any crop scores below F")
    parser.add_argument("--version", action="version",
                        version="smart-crop-ai {0}".format(__version__))
    return parser


def parse_thumb(text: Optional[str]) -> Optional[Sequence[int]]:
    """Turn ``"200x150"`` or ``"200"`` into a size, or ``None``.

    Raises:
        ValueError: if the text is not a size.
    """
    if text is None:
        return None
    cleaned = str(text).strip().lower().replace(",", "x").replace(":", "x")
    parts = [part for part in cleaned.split("x") if part != ""]
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        raise ValueError("--thumb needs WxH or one number, got {0!r}".format(text)) from None
    if len(numbers) == 1:
        numbers = [numbers[0], numbers[0]]
    if len(numbers) != 2 or min(numbers) < 1:
        raise ValueError("--thumb needs WxH or one number, got {0!r}".format(text))
    return (numbers[0], numbers[1])


#: Options whose value is free text and may legitimately start with a dash.
_DASH_VALUE_OPTIONS = ("--suffix",)


def glue_dash_values(
    argv: Sequence[str], parser: argparse.ArgumentParser
) -> List[str]:
    """Let ``--suffix -crop`` mean what it looks like it means.

    argparse reads any token starting with ``-`` as another option, so the
    ``--suffix -crop`` in the README dies with "expected one argument" even
    though ``--suffix=-crop`` works. A leading dash is the normal shape of a
    filename suffix, so the two are joined here rather than made the user's
    problem. A token that really is a known option is left alone, so
    ``--suffix --quiet`` still reports a missing value.
    """
    known = {
        option
        for action in parser._actions            # the only public-ish route in
        for option in action.option_strings
    }
    glued: List[str] = []
    index = 0
    tokens = list(argv)
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if (
            token in _DASH_VALUE_OPTIONS
            and following is not None
            and following.startswith("-")
            and following not in known
            and following.split("=", 1)[0] not in known
        ):
            glued.append("{0}={1}".format(token, following))
            index += 2
            continue
        glued.append(token)
        index += 1
    return glued


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
                    if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                        found.append(os.path.join(root, name))
        else:
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                    found.append(full)
    return found


def destination_for(source: str, out_dir: str, suffix: str) -> str:
    """Where a crop of ``source`` goes inside ``out_dir``."""
    stem, extension = os.path.splitext(os.path.basename(source))
    return os.path.join(out_dir, "{0}{1}{2}".format(stem, suffix, extension))


def render(results: Sequence[CropResult], as_json: bool, quiet: bool) -> str:
    """The text the CLI prints for one run."""
    if as_json:
        payload: Any = [result.to_dict() for result in results]
        if len(payload) == 1:
            payload = payload[0]
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if quiet:
        return "\n".join(
            "{0:.2f} {1:<10} {2:>5}x{3:<5} at ({4},{5})  {6}".format(
                result.confidence,
                result.strategy_used,
                result.size[0],
                result.size[1],
                result.offset[0],
                result.offset[1],
                result.source,
            )
            for result in results
        )
    return "\n\n".join(result.summary() for result in results)


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
        directory = os.path.dirname(output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(glue_dash_values(tokens, parser))

    if not args.paths:
        parser.print_help()
        return EXIT_ERROR

    try:
        thumb = parse_thumb(args.thumb)
    except ValueError as exc:
        parser.error(str(exc))

    if args.ratio is None and args.width is None and args.height is None:
        parser.error(
            "give --width and --height, or --ratio; there is nothing to crop to"
        )
    if args.ratio is not None and (args.width is not None or args.height is not None):
        parser.error("give --width/--height or --ratio, not both")

    try:
        files = collect_paths(args.paths, args.recursive)
    except OSError as exc:
        print("smart-crop-ai: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR
    if not files:
        print(
            "smart-crop-ai: no images found in {0}".format(", ".join(args.paths)),
            file=sys.stderr,
        )
        return EXIT_ERROR
    if args.output and len(files) > 1:
        print(
            "smart-crop-ai: --output takes one image, got {0}; use --out-dir "
            "instead".format(len(files)),
            file=sys.stderr,
        )
        return EXIT_ERROR

    options: Dict[str, Any] = {
        "strategy": args.strategy,
        "padding": args.padding,
    }
    if args.ratio is not None:
        options["ratio"] = args.ratio
    else:
        options["width"] = args.width
        options["height"] = args.height

    results: List[CropResult] = []
    failures = 0
    for path in files:
        try:
            result = crop(path, **options)
        except (OSError, ValueError, TypeError) as exc:
            print("smart-crop-ai: {0}: {1}".format(path, exc), file=sys.stderr)
            failures += 1
            continue
        result.source = path
        destination = args.output or (
            destination_for(path, args.out_dir, args.suffix) if args.out_dir else None
        )
        if destination:
            if thumb:
                result.image = resize_exact(result.image, thumb)
                result.notes.append(
                    "Resized to {0} x {1} after cropping.".format(thumb[0], thumb[1])
                )
            try:
                result.save(destination)
            except OSError as exc:
                print(
                    "smart-crop-ai: cannot write {0}: {1}".format(destination, exc),
                    file=sys.stderr,
                )
                failures += 1
                continue
        elif thumb:
            result.notes.append(
                "--thumb {0} x {1} was not applied: nothing was written, so there was "
                "nothing to resize. Add --output or --out-dir.".format(
                    thumb[0], thumb[1]
                )
            )
        results.append(result)

    if not results:
        return EXIT_ERROR

    try:
        _write(render(results, args.json, args.quiet), args.report)
    except OSError as exc:
        print("smart-crop-ai: cannot write report: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    if failures:
        return EXIT_ERROR
    if args.min_confidence is not None:
        weak = [r for r in results if r.confidence < args.min_confidence]
        if weak:
            print(
                "smart-crop-ai: {0} of {1} crop(s) scored below {2:g}".format(
                    len(weak), len(results), args.min_confidence
                ),
                file=sys.stderr,
            )
            return EXIT_LOW_CONFIDENCE
    return 0


if __name__ == "__main__":              # pragma: no cover - module entry
    sys.exit(main())
