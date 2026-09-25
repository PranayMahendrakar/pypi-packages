"""Command line interface: ``ocr-cleaner IMAGE [IMAGE ...] [options]``.

By default this is a dry run: it reads each page, prints what it would do to it
and why, and writes nothing. Add ``--output`` or ``--out-dir`` to save the
cleaned pages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Set

from . import __version__
from ._core import DEFAULT_THRESHOLD, THRESHOLD_MODES, clean
from ._images import IMAGE_SUFFIXES
from ._result import CleanResult

_DESCRIPTION = """Prepare a scanned page so OCR reads it better: deskew, denoise, threshold.

IMAGE is an image file or a directory of them. Nothing is written unless you ask
for it with --output or --out-dir, so you can read the report first."""

_EPILOG = """examples:
  ocr-cleaner scan.png
  ocr-cleaner scan.png --output clean.png
  ocr-cleaner scans/ --out-dir cleaned/ --suffix -clean
  ocr-cleaner scan.tif --dpi 200 --upscale-to-dpi 300 --output big.tif
  ocr-cleaner scan.png --threshold otsu --no-denoise
  ocr-cleaner scans/ --recursive --quiet
  ocr-cleaner scan.png --json

exit codes:
  0  all good
  1  a usage or read error
  2  a page was blank or was not a document, and --require-document was given"""

#: Exit code when --require-document is not met.
EXIT_NOT_A_DOCUMENT = 2
#: Exit code for a usage or read error.
EXIT_ERROR = 1


class _Parser(argparse.ArgumentParser):
    """argparse, with usage errors exiting 1 instead of its default 2.

    Exit code 2 is reserved for ``--require-document``, so a batch job can tell
    "that page was blank" apart from "I typed the flag wrong". ``--help`` and
    ``--version`` still exit 0.

    It also keeps :attr:`known_options`, the set of option strings it has been
    given. :func:`glue_dash_values` needs to know whether ``-clean`` is one of
    this program's options or a value that happens to start with a dash, and
    argparse offers no documented way to ask. Recording them as they are added
    costs one line, cannot drift out of step with the options themselves, and
    leans on nothing but ``add_argument``'s documented return value.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        #: Every option string this parser knows, ``-q`` and ``--quiet`` alike.
        #: Filled in before ``super().__init__``, which adds ``-h`` itself and
        #: so goes through the override below.
        self.known_options: Set[str] = set()
        super().__init__(*args, **kwargs)

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        action = super().add_argument(*args, **kwargs)
        self.known_options.update(action.option_strings)
        return action

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, "{0}: error: {1}\n".format(self.prog, message))


def build_parser() -> _Parser:
    """The argument parser, exposed so ``--help`` can be tested."""
    parser = _Parser(
        prog="ocr-cleaner",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", metavar="IMAGE",
        help="scanned page(s) or directory(ies) of them",
    )
    parser.add_argument("--threshold", default=DEFAULT_THRESHOLD, choices=list(THRESHOLD_MODES),
                        help="how to binarise the page (default: {0})".format(DEFAULT_THRESHOLD))
    parser.add_argument("--no-deskew", action="store_false", dest="deskew",
                        help="do not straighten the page")
    parser.add_argument("--no-denoise", action="store_false", dest="denoise",
                        help="do not median-filter the page")
    parser.add_argument("--no-border", action="store_false", dest="border",
                        help="do not trim scanner edges and black margins")
    parser.add_argument("--dpi", type=float, default=None, metavar="N",
                        help="the resolution of the input, if you know it")
    parser.add_argument("--upscale-to-dpi", type=float, default=None, metavar="N",
                        dest="upscale_to_dpi",
                        help="enlarge to this resolution; needs --dpi as well")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="walk into subdirectories of any directory given")
    parser.add_argument("--output", metavar="PATH",
                        help="write the cleaned page here; one input image only")
    parser.add_argument("--out-dir", metavar="DIR", dest="out_dir",
                        help="write every cleaned page into DIR, keeping each file name")
    parser.add_argument("--suffix", default="", metavar="TEXT",
                        help="add TEXT to each written file name, e.g. -clean")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="one line per page instead of the full report")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON instead of the summary")
    parser.add_argument("--report", metavar="PATH",
                        help="write the printed report to this file as UTF-8")
    parser.add_argument("--require-document", action="store_true", dest="require_document",
                        help="exit with code 2 if any page is blank or is not a document")
    parser.add_argument("--version", action="version",
                        version="ocr-cleaner {0}".format(__version__))
    return parser


#: Options whose value is free text and may legitimately start with a dash.
_DASH_VALUE_OPTIONS = ("--suffix",)


def glue_dash_values(argv: Sequence[str], parser: _Parser) -> List[str]:
    """Let ``--suffix -clean`` mean what it looks like it means.

    argparse reads any token starting with ``-`` as another option, so the
    ``--suffix -clean`` in the README would die with "expected one argument"
    even though ``--suffix=-clean`` works. A leading dash is the normal shape of
    a filename suffix, so the two are joined here rather than made the user's
    problem. A token that really is a known option is left alone, so
    ``--suffix --quiet`` still reports a missing value.
    """
    known = parser.known_options
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
    """Where the cleaned version of ``source`` goes inside ``out_dir``."""
    stem, extension = os.path.splitext(os.path.basename(source))
    return os.path.join(out_dir, "{0}{1}{2}".format(stem, suffix, extension))


def render(results: Sequence[CleanResult], as_json: bool, quiet: bool) -> str:
    """The text the CLI prints for one run."""
    if as_json:
        payload: Any = [result.to_dict() for result in results]
        if len(payload) == 1:
            payload = payload[0]
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if quiet:
        return "\n".join(
            "{0:+6.2f} deg  {1:<10} {2:>5}x{3:<5}  {4:<34} {5}".format(
                result.estimated_skew_degrees,
                result.page_kind,
                result.output_size[0],
                result.output_size[1],
                ",".join(name for name in result.applied if name != "grayscale") or "-",
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
    if args.upscale_to_dpi is not None and args.dpi is None:
        parser.error(
            "--upscale-to-dpi needs --dpi as well; nothing can be scaled to a "
            "resolution without knowing the one it has now"
        )
    for name, value in (("--dpi", args.dpi), ("--upscale-to-dpi", args.upscale_to_dpi)):
        if value is not None and value <= 0:
            parser.error("{0} must be a positive number, got {1:g}".format(name, value))

    try:
        files = collect_paths(args.paths, args.recursive)
    except OSError as exc:
        print("ocr-cleaner: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR
    if not files:
        print(
            "ocr-cleaner: no images found in {0}".format(", ".join(args.paths)),
            file=sys.stderr,
        )
        return EXIT_ERROR
    if args.output and len(files) > 1:
        print(
            "ocr-cleaner: --output takes one image, got {0}; use --out-dir "
            "instead".format(len(files)),
            file=sys.stderr,
        )
        return EXIT_ERROR

    options: Dict[str, Any] = {
        "deskew": args.deskew,
        "denoise": args.denoise,
        "border": args.border,
        "threshold": args.threshold,
        "dpi": args.dpi,
        "upscale_to_dpi": args.upscale_to_dpi,
    }

    results: List[CleanResult] = []
    failures = 0
    for path in files:
        try:
            result = clean(path, **options)
        except (OSError, ValueError, TypeError) as exc:
            print("ocr-cleaner: {0}: {1}".format(path, exc), file=sys.stderr)
            failures += 1
            continue
        destination = args.output or (
            destination_for(path, args.out_dir, args.suffix) if args.out_dir else None
        )
        if destination:
            try:
                result.save(destination)
            except (OSError, ValueError) as exc:
                print(
                    "ocr-cleaner: cannot write {0}: {1}".format(destination, exc),
                    file=sys.stderr,
                )
                failures += 1
                continue
        results.append(result)

    if not results:
        return EXIT_ERROR

    try:
        _write(render(results, args.json, args.quiet), args.report)
    except OSError as exc:
        print("ocr-cleaner: cannot write report: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    if failures:
        return EXIT_ERROR
    if args.require_document:
        odd = [result for result in results if result.page_kind != "document"]
        if odd:
            print(
                "ocr-cleaner: {0} of {1} page(s) were not documents: {2}".format(
                    len(odd), len(results),
                    ", ".join(sorted({result.page_kind for result in odd})),
                ),
                file=sys.stderr,
            )
            return EXIT_NOT_A_DOCUMENT
    return 0


if __name__ == "__main__":              # pragma: no cover - module entry
    sys.exit(main())
