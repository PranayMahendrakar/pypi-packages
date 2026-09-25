"""Command line interface: ``object-counter-ai IMAGE [IMAGE ...] [options]``."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence, Tuple

from . import __version__
from ._core import count
from ._counter import Counter
from ._images import looks_like_image_path

_DESCRIPTION = """Count objects in images or across video frames.

Without --detector, the built-in classical counter counts high-contrast blobs
against a plain background: parts on a conveyor, cells on a slide, bolts on a
tray. It counts blobs, not objects, and is useless for things like people in a
street. For those, pass your own detector with --detector."""

_EPILOG = """examples:
  object-counter-ai tray.png
  object-counter-ai tray.png --min-area 50 --expect 12
  object-counter-ai slide.tif --region 100,100,900,700 --json
  object-counter-ai shots/ --quiet
  object-counter-ai frames/ --line 0,240,640,240
  object-counter-ai street.jpg --detector my_models:detect_people

--region is a box "left,top,right,bottom" or a polygon "x,y x,y x,y ...".
--line is "x1,y1,x2,y2"; it treats the images as frames of one stream in name
order and counts each tracked object once when it crosses the line.
--detector MODULE:FUNCTION names a function detector(image) -> list of boxes
(left, top, right, bottom) or (box, label, score); the current folder is
importable.

exit codes:
  0  all good
  1  a usage or read error, or the detector failed on an image
  2  a count differed from --expect"""

EXIT_ERROR = 1
EXIT_MISMATCH = 2


class _Parser(argparse.ArgumentParser):
    """argparse, with usage errors exiting 1 so that 2 can mean --expect failed."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``--help`` can be tested."""
    parser = _Parser(
        prog="object-counter-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", metavar="IMAGE",
                        help="image file(s), or folder(s) of images")
    parser.add_argument("--min-area", type=float, default=None, metavar="PX",
                        help="ignore blobs (or boxes) smaller than this many pixels")
    parser.add_argument("--max-area", type=float, default=None, metavar="PX",
                        help="ignore blobs (or boxes) larger than this many pixels")
    parser.add_argument("--region", default=None, metavar="SPEC",
                        help='count only inside a box "l,t,r,b" or a polygon "x,y x,y x,y"')
    parser.add_argument("--detector", default=None, metavar="MODULE:FUNC",
                        help="use your detector function instead of the classical counter")
    parser.add_argument("--line", action="append", default=[], metavar="X1,Y1,X2,Y2",
                        help="count objects crossing this line across the images as frames "
                             "(repeatable)")
    parser.add_argument("--frames", action="store_true",
                        help="treat the images as frames of one stream even without --line")
    parser.add_argument("--max-distance", type=float, default=None, metavar="PX",
                        help="with --line/--frames: furthest an object moves between frames")
    parser.add_argument("--expect", type=int, default=None, metavar="N",
                        help="exit 2 if any image's count is not N")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="also write the JSON result to PATH (UTF-8)")
    parser.add_argument("--quiet", action="store_true",
                        help="print one 'count<TAB>path' line per image")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _numbers(text: str) -> List[float]:
    parts = text.replace(";", " ").replace(",", " ").split()
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"could not read numbers from {text!r}") from None


def parse_region_spec(text: str) -> Any:
    """``"l,t,r,b"`` is a box; ``"x,y x,y x,y ..."`` is a polygon."""
    groups = [g for g in text.replace(";", " ").split() if g]
    if len(groups) >= 3:
        points = []
        for group in groups:
            pair = _numbers(group)
            if len(pair) != 2:
                raise ValueError(f"polygon corners must be x,y pairs, got {group!r}")
            points.append((pair[0], pair[1]))
        return points
    values = _numbers(text)
    if len(values) != 4:
        raise ValueError(f'--region needs "left,top,right,bottom" or "x,y x,y x,y ...", got {text!r}')
    return tuple(values)


def parse_line_spec(text: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """``"x1,y1,x2,y2"`` into two points."""
    values = _numbers(text)
    if len(values) != 4:
        raise ValueError(f'--line needs "x1,y1,x2,y2", got {text!r}')
    return (values[0], values[1]), (values[2], values[3])


def load_detector(spec: str) -> Callable[[Any], Any]:
    """Import ``module:function`` (the current folder is importable)."""
    if ":" not in spec:
        raise ValueError(f"--detector needs MODULE:FUNCTION, got {spec!r}")
    module_name, _, attr = spec.partition(":")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"could not import {module_name!r}: {exc}") from None
    for part in attr.split("."):
        if not hasattr(target, part):
            raise ValueError(f"{module_name!r} has no attribute {attr!r}")
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(f"{spec!r} is not callable")
    return target


def expand_paths(paths: Sequence[str]) -> List[str]:
    """Files as given; folders become their image files in name order."""
    out: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            names = sorted(n for n in os.listdir(path) if looks_like_image_path(n))
            out.extend(os.path.join(path, n) for n in names)
        else:
            out.append(path)
    return out


def _write_json(path: str, payload: Any) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``object-counter-ai`` command."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.paths:
        parser.print_help()
        return EXIT_ERROR

    try:
        region = parse_region_spec(args.region) if args.region else None
        lines = [parse_line_spec(spec) for spec in args.line]
        detector = load_detector(args.detector) if args.detector else None
    except ValueError as exc:
        parser.error(str(exc))

    paths = expand_paths(args.paths)
    if not paths:
        sys.stderr.write("object-counter-ai: no image files found\n")
        return EXIT_ERROR

    if lines or args.frames:
        return _run_stream(args, paths, region, lines, detector)
    return _run_images(args, paths, region, detector)


def _run_images(args: argparse.Namespace, paths: List[str], region: Any,
                detector: Optional[Callable[[Any], Any]]) -> int:
    status = 0
    records: List[Dict[str, Any]] = []
    blocks: List[str] = []
    total = 0
    for path in paths:
        try:
            result = count(path, detector=detector, min_area=args.min_area,
                           max_area=args.max_area, region=region)
        except (OSError, ValueError, TypeError) as exc:
            sys.stderr.write(f"object-counter-ai: {path}: {exc}\n")
            records.append({"path": path, "error": str(exc)})
            status = EXIT_ERROR
            continue
        if not result.ok:
            status = EXIT_ERROR
        total += result.count
        record = result.to_dict()
        record["path"] = path
        records.append(record)
        if args.quiet:
            blocks.append(f"{result.count}\t{path}")
        else:
            blocks.append(f"{path}\n{result.summary()}")
        if args.expect is not None and result.ok and result.count != args.expect:
            if status == 0:
                status = EXIT_MISMATCH
            sys.stderr.write(
                f"object-counter-ai: {path}: counted {result.count}, expected {args.expect}\n")
    payload: Any = records[0] if len(records) == 1 else {
        "images": records, "total": total}
    if args.json:
        _emit(json.dumps(payload, ensure_ascii=False, indent=2))
    elif blocks:
        _emit(("\n" if args.quiet else "\n\n").join(blocks))
        if len(paths) > 1 and not args.quiet:
            _emit(f"\ntotal: {total} across {len(paths)} images")
    if args.output:
        _write_json(args.output, payload)
    return status


def _run_stream(args: argparse.Namespace, paths: List[str], region: Any,
                lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                detector: Optional[Callable[[Any], Any]]) -> int:
    try:
        counter = Counter(detector, min_area=args.min_area, max_area=args.max_area,
                          region=region, max_distance=args.max_distance)
        for a, b in lines:
            counter.line(a, b)
    except ValueError as exc:
        sys.stderr.write(f"object-counter-ai: {exc}\n")
        return EXIT_ERROR
    status = 0
    for path in paths:
        try:
            result = counter.update(path)
        except (OSError, ValueError, TypeError) as exc:
            sys.stderr.write(f"object-counter-ai: {path}: {exc}\n")
            return EXIT_ERROR
        if not result.ok:
            status = EXIT_ERROR
            sys.stderr.write(f"object-counter-ai: {path}: {result.error}\n")
        if args.quiet:
            _emit(f"{result.count}\t{path}")
    payload = counter.to_dict()
    payload["paths"] = paths
    if args.json:
        _emit(json.dumps(payload, ensure_ascii=False, indent=2))
    elif not args.quiet:
        _emit(counter.summary())
    if args.output:
        _write_json(args.output, payload)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
