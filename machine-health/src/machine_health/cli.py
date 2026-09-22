"""Command line entry point: ``machine-health telemetry.csv --rule "temp:max=80"``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .rules import Rule
from .scoring import DEFAULT_WEIGHTS, score

_LIMIT_KEYS = ("min", "max", "warn_min", "warn_max", "weight")


def _parse_rule(text: str) -> Rule:
    """'temp:max=80,warn_max=75' -> Rule(channel='temp', max=80.0, warn_max=75.0)."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"rule {text!r} must look like 'channel:max=80' or 'vib:min=0.1,max=2.5'"
        )
    channel, _, spec = text.partition(":")
    channel = channel.strip()
    if not channel:
        raise argparse.ArgumentTypeError(f"rule {text!r} has no channel name before the ':'")
    limits: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        key = key.strip().lower()
        if not sep or key not in _LIMIT_KEYS:
            raise argparse.ArgumentTypeError(
                f"rule {text!r}: expected one of {', '.join(_LIMIT_KEYS)} as 'key=number', "
                f"got {part!r}"
            )
        try:
            limits[key] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"rule {text!r}: {key} must be a number, got {value!r}"
            ) from None
    if not limits:
        raise argparse.ArgumentTypeError(f"rule {text!r} sets no limit")
    try:
        return Rule(channel=channel, **limits)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_weight(text: str) -> Dict[str, float]:
    """'compliance=0.5' -> {'compliance': 0.5}."""
    key, sep, value = text.partition("=")
    key = key.strip().lower()
    if not sep or key not in DEFAULT_WEIGHTS:
        raise argparse.ArgumentTypeError(
            f"weight {text!r} must look like 'compliance=0.5'; "
            f"names: {', '.join(DEFAULT_WEIGHTS)}"
        )
    try:
        return {key: float(value)}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"weight {text!r}: {value!r} is not a number"
        ) from None


def _parse_baseline(text: str) -> Any:
    """A fraction, a row count, or a path to a table of healthy rows."""
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _rules_from_file(path: str) -> List[Rule]:
    """Read rules from a JSON file: a mapping, or a list of rule objects."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"--rules: no such file: {path}") from None
    except IsADirectoryError:
        raise ValueError(f"--rules: {path} is a directory, not a JSON file") from None
    except UnicodeDecodeError:
        raise ValueError(f"--rules: {path} is not UTF-8 text") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"--rules: {path} is not valid JSON: {exc}") from None
    except OSError as exc:
        raise OSError(f"--rules: cannot read {path}: {exc.strerror or exc}") from None
    from .rules import normalize_rules

    return normalize_rules(data)


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="machine-health",
        description=(
            "Score a machine's telemetry as one 0-100 health number built from "
            "stability, compliance with your limits, anomalies and availability."
        ),
        epilog=(
            "example: machine-health telemetry.csv --time ts --rule \"temp:max=80\" "
            "--rule \"vibration:max=2.5,warn_max=2.0\""
        ),
    )
    parser.add_argument("path", help="telemetry table (.csv, .tsv or .parquet)")
    parser.add_argument("--time", metavar="COL", help="timestamp column to order rows by")
    parser.add_argument(
        "--channels", metavar="COL", nargs="+", help="columns to score (default: every numeric one)"
    )
    parser.add_argument(
        "--rule",
        metavar="SPEC",
        action="append",
        type=_parse_rule,
        default=[],
        help="a limit, like 'temp:max=80' or 'vib:min=0.1,max=2.5,weight=2'; repeatable",
    )
    parser.add_argument("--rules", metavar="FILE", help="JSON file of rules")
    parser.add_argument(
        "--weight",
        metavar="NAME=VALUE",
        action="append",
        type=_parse_weight,
        default=[],
        help="component weight, like 'compliance=0.5'; repeatable, normalized to sum 1",
    )
    parser.add_argument(
        "--baseline",
        metavar="SPEC",
        type=_parse_baseline,
        help="healthy period: a fraction (0.2), a row count (500) or a path to a table",
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--output", metavar="FILE", help="write the JSON result to this file")
    parser.add_argument(
        "--fail-under",
        metavar="SCORE",
        type=float,
        help="exit with status 1 when the score is below this",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, score the table, print the result. Returns the exit status."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # a stream that cannot be reconfigured
                pass
    args = build_parser().parse_args(argv)

    weights: Dict[str, float] = {}
    for item in args.weight:
        weights.update(item)
    try:
        rules: List[Rule] = list(args.rule)
        if args.rules:
            rules.extend(_rules_from_file(args.rules))
        result = score(
            args.path,
            time=args.time,
            channels=args.channels,
            rules=rules or None,
            weights=weights or None,
            baseline=args.baseline,
        )
        if args.output:
            path = Path(args.output)
            if path.parent and str(path.parent) not in ("", "."):
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
    except (ValueError, TypeError, ImportError, FileNotFoundError, OSError) as exc:
        # ValueError also covers json.JSONDecodeError from a malformed --rules file.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.summary())
    if args.fail_under is not None and result.value < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
