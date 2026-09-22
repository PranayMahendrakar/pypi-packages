"""Command line interface: ``offline-ml [MODELS] [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._hardware import detect
from ._recommend import PREFERENCES, recommend

_DESCRIPTION = """Detect this machine and pick a model configuration that fits.

With no models given, print what this machine is. Give it models and it picks
one and says why. MODELS is a .json file holding a list of model objects, or an
object with a "models" key; each model needs at least "name" and "size_gb"."""

_EPILOG = """examples:
  offline-ml
  offline-ml --json
  offline-ml --model tinyllama-1.1b-q4:0.7 --model mistral-7b-q4:4.1
  offline-ml models.json --prefer quality --task chat
  offline-ml models.json --json > pick.json

sizes are in GiB (1024**3 bytes)."""


def _list_length(text: str) -> int:
    """argparse type for --limit: it is a count, so it cannot be negative."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a whole number, got {text!r}"
        ) from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="offline-ml",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        metavar="MODELS",
        nargs="?",
        help="path to a .json file listing the models to choose from",
    )
    parser.add_argument(
        "--model",
        metavar="NAME:SIZE_GB",
        action="append",
        default=[],
        dest="inline_models",
        help="add one model on the command line; repeatable",
    )
    parser.add_argument("--task", metavar="TASK", help="only consider models for this task")
    parser.add_argument(
        "--prefer",
        choices=list(PREFERENCES),
        default="balanced",
        help="what to optimise for (default: balanced)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.2,
        help="spare room to leave, as a fraction (default: 0.2)",
    )
    parser.add_argument(
        "--limit",
        type=_list_length,
        default=5,
        help="how many alternatives and rejections to list (default: 5)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON instead of the summary"
    )
    parser.add_argument("--output", metavar="PATH", help="also write the JSON result to PATH")
    parser.add_argument("--version", action="version", version=f"offline-ml {__version__}")
    return parser


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a model name."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed or unusual streams
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def parse_inline_model(text: str) -> Dict[str, Any]:
    """Turn ``"mistral-7b-q4:4.1"`` into a model dict."""
    name, sep, raw_size = str(text).rpartition(":")
    if not sep or not name.strip():
        raise ValueError(
            f"--model expects NAME:SIZE_GB, got {text!r} "
            "(for example --model mistral-7b-q4:4.1)"
        )
    try:
        size = float(raw_size)
    except ValueError:
        raise ValueError(
            f"--model {text!r}: {raw_size!r} is not a size in GiB"
        ) from None
    return {"name": name.strip(), "size_gb": size}


def load_models(path: str) -> List[Dict[str, Any]]:
    """Read a .json file of model objects."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("models", data)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a JSON list of model objects, or an object with "
            'a "models" list'
        )
    return data


def _collect_models(args: argparse.Namespace) -> List[Any]:
    """Every model the user asked about, from the file and from --model."""
    models: List[Any] = []
    if args.models:
        models.extend(load_models(args.models))
    models.extend(parse_inline_model(text) for text in args.inline_models)
    return models


def _asked_about_models(args: argparse.Namespace) -> bool:
    """True when the user named a model source, even if it turns out to be empty.

    An empty source is a mistake worth reporting, not a reason to quietly fall
    back to printing the machine.
    """
    return bool(args.models) or bool(args.inline_models)


def write_output(path: str, payload: Dict[str, Any]) -> None:
    """Write the JSON result to `path`, as UTF-8."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        picking = _asked_about_models(args)
        if picking:
            # An empty list here raises a clear ValueError rather than
            # silently reporting the machine instead.
            result = recommend(
                _collect_models(args),
                task=args.task,
                prefer=args.prefer,
                headroom=args.headroom,
            )
        else:
            result = detect()
        payload = result.to_dict()
        if args.output:
            write_output(args.output, payload)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(result.summary(limit=args.limit) if picking else result.summary())
            if args.output:
                print(f"  wrote     : {args.output}")
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"offline-ml: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
