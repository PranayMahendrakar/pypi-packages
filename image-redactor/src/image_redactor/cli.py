"""Command line interface: ``image-redactor SRC [-o OUT] [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from ._core import METHOD_HELP, METHODS, RedactResult, redact

_DESCRIPTION = """Blur or mask regions of an image and report what was hidden.

With no --region the built-in heuristic detectors run. They are weak: they miss
faces and plates and they fire on things that are neither. Mark the regions
yourself, or call the library with detector=your_model, for anything that
matters."""

_EPILOG = "methods:\n" + "\n".join(
    f"  {name:<9} {help_text}" for name, help_text in METHOD_HELP.items()
) + """

examples:
  image-redactor photo.jpg -o safe.jpg --region 40,20,100,90
  image-redactor photo.jpg -o safe.png --method blackout --region 10,10,60,60
  image-redactor photo.jpg --detect-only --json"""

REGION_HINT = (
    "--region takes four numbers LEFT,TOP,RIGHT,BOTTOM, for example --region 40,20,100,90"
)


def parse_region(raw: str) -> List[float]:
    """Parse one ``LEFT,TOP,RIGHT,BOTTOM`` string from the command line."""
    parts = [piece.strip() for piece in raw.replace(" ", ",").split(",") if piece.strip()]
    if len(parts) != 4:
        raise ValueError(f"{REGION_HINT}; got {raw!r}")
    try:
        return [float(piece) for piece in parts]
    except ValueError:
        raise ValueError(f"{REGION_HINT}; {raw!r} is not four numbers") from None


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="image-redactor",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", metavar="SRC", help="image to redact")
    parser.add_argument(
        "-o", "--output", metavar="PATH", help="write the redacted image here"
    )
    parser.add_argument(
        "--region",
        metavar="L,T,R,B",
        action="append",
        default=[],
        help="a box to hide; repeat for more than one",
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default="blur",
        help="how to hide each region (default: blur)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.9,
        help="0 to 1, how hard blur and pixelate hit (default: 0.9)",
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=0.08,
        help="grow each box by this fraction first (default: 0.08)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="report the regions that would be hidden and write nothing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the full result as JSON instead of the summary",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print nothing on success"
    )
    parser.add_argument(
        "--version", action="version", version=f"image-redactor {__version__}"
    )
    return parser


def render_summary(result: RedactResult, detect_only: bool) -> str:
    """The result summary, plus a note when nothing was written."""
    lines = [result.summary()]
    if detect_only:
        lines.append("  (--detect-only: nothing was written)")
    return "\n".join(lines)


def _make_console_safe() -> None:
    """Never crash on a console or pipe that cannot encode a character from the data."""
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
        regions = [parse_region(raw) for raw in args.region] or None
        result = redact(
            args.src,
            regions=regions,
            method=args.method,
            strength=args.strength,
            expand=args.expand,
        )
        if args.output and not args.detect_only:
            result.save(args.output)
        elif not args.output and not args.detect_only:
            raise ValueError(
                "nothing would be written; pass -o PATH to save the redacted image, "
                "or --detect-only to just report the regions"
            )
        if args.quiet:
            return 0
        if args.as_json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(render_summary(result, args.detect_only))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"image-redactor: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
