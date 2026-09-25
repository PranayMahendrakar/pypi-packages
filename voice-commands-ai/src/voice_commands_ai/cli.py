"""Command line interface: ``voice-commands-ai -c PATTERN [-c PATTERN ...] TEXT ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from ._commands import Commands
from ._numbers import parse_number

_DESCRIPTION = """Match transcribed voice commands against patterns and print what was asked.

This tool maps TEXT to commands; it does not recognise speech. Feed it the output
of any speech-to-text engine, one utterance per argument or per line of stdin.

Patterns are phrases with slots: {name} or {name:word} is one word,
{name:number} a spoken or written number, {name:a|b|c} one of the choices,
{name:text} free text, and [words] are optional."""

_EPILOG = """examples:
  voice-commands-ai -c "set temperature to {value:number} degrees" "um set the temperature to twenty one degrees please"
  voice-commands-ai -c "thermostat = set temperature to {value:number} degrees" -c "switch = turn {state:on|off} the {device:text}" "turn the kitchen light off"
  voice-commands-ai -f commands.txt transcripts.txt --json > matches.json
  some-transcriber | voice-commands-ai -f commands.json --fail-on-no-match
  voice-commands-ai --number "a hundred and five" "two point five" "minus three"

commands file: a .json file holding {"name": "pattern", ...}, a list of patterns, or
a list of {"pattern": ..., "name": ..., "examples": [...]}; or a text file with one
"name = pattern" (or just "pattern") per line, where lines starting with # are comments.

exit status: 0 on success, 1 with --fail-on-no-match when an utterance matched
nothing, 2 when the commands or input could not be read."""


class _UsageError(Exception):
    """A problem with the command line or its files, reported with exit status 2."""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="voice-commands-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="*",
        metavar="TEXT",
        help="utterances to match; a path to an existing .txt file reads one utterance "
        "per line; with no TEXT, lines are read from stdin",
    )
    parser.add_argument(
        "-c", "--command",
        action="append",
        default=[],
        metavar="PATTERN",
        help='a command pattern, optionally named: "name = pattern" (repeatable)',
    )
    parser.add_argument(
        "-f", "--commands-file",
        action="append",
        default=[],
        dest="commands_file",
        metavar="PATH",
        help="read command patterns from a .json or text file (repeatable)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        dest="min_confidence",
        metavar="X",
        help="confidence floor between 0 and 1; below it nothing matches (default: 0.6)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also list how every command scored, including those below the floor",
    )
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="also write the results to PATH as JSON",
    )
    parser.add_argument(
        "--fail-on-no-match",
        action="store_true",
        dest="fail_on_no_match",
        help="exit with status 1 when any utterance matched no command",
    )
    parser.add_argument(
        "--number",
        action="store_true",
        help="read each TEXT as a spoken number instead, and print its value",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def _split_named(line: str) -> Tuple[Optional[str], str]:
    """``"name = pattern"`` -> (name, pattern); a bare pattern -> (None, pattern)."""
    head, sep, tail = line.partition("=")
    if sep and head.strip() and not any(ch in head for ch in "{}[]|") and tail.strip():
        return head.strip(), tail.strip()
    return None, line.strip()


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except OSError as exc:
        raise _UsageError("could not read %s: %s" % (path, exc)) from None
    except UnicodeDecodeError as exc:
        raise _UsageError("%s is not UTF-8 text: %s" % (path, exc)) from None


def _load_commands_file(path: str) -> List[Dict[str, Any]]:
    """Command specs ({"pattern", "name", "examples"}) from a .json or text file."""
    content = _read_text_file(path)
    specs: List[Dict[str, Any]] = []
    if path.lower().endswith(".json"):
        try:
            data = json.loads(content)
        except ValueError as exc:
            raise _UsageError("%s is not valid JSON: %s" % (path, exc)) from None
        if isinstance(data, dict):
            for name, pattern in data.items():
                specs.append({"name": name, "pattern": pattern})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    specs.append({"pattern": item})
                elif isinstance(item, dict) and isinstance(item.get("pattern"), str):
                    specs.append({
                        "pattern": item["pattern"],
                        "name": item.get("name"),
                        "examples": item.get("examples"),
                    })
                else:
                    raise _UsageError(
                        "%s: each entry must be a pattern string or an object with a "
                        '"pattern" key, got %r' % (path, item)
                    )
        else:
            raise _UsageError("%s must hold a JSON object or list of commands" % path)
        return specs
    for number, raw in enumerate(content.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, pattern = _split_named(line)
        specs.append({"name": name, "pattern": pattern, "line": number})
    return specs


def _build_commands(args: argparse.Namespace) -> Commands:
    try:
        commands = Commands(min_confidence=args.min_confidence)
    except (TypeError, ValueError) as exc:
        raise _UsageError(str(exc)) from None
    specs: List[Dict[str, Any]] = []
    for path in args.commands_file:
        for spec in _load_commands_file(path):
            spec["source"] = path
            specs.append(spec)
    for raw in args.command:
        name, pattern = _split_named(raw)
        specs.append({"name": name, "pattern": pattern, "source": "--command"})
    if not specs:
        raise _UsageError("give at least one command with -c PATTERN or -f FILE (or use --number)")
    for spec in specs:
        try:
            commands.add(spec["pattern"], name=spec.get("name"), examples=spec.get("examples"))
        except (TypeError, ValueError) as exc:
            where = spec["source"]
            if "line" in spec:
                where = "%s line %d" % (where, spec["line"])
            raise _UsageError("%s: %s" % (where, exc)) from None
    return commands


def _utterances(args: argparse.Namespace) -> List[str]:
    lines: List[str] = []
    if args.text:
        for item in args.text:
            if item.lower().endswith(".txt") and os.path.isfile(item):
                lines.extend(_read_text_file(item).splitlines())
            else:
                lines.append(item)
    else:
        lines = sys.stdin.read().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _no_match_text(text: str, ranked: list, floor: float) -> str:
    if not ranked:
        return 'No command matched "%s" (nothing to match in it)' % text
    closest = ranked[0]
    return 'No command matched "%s" (closest: \'%s\' at %.2f, floor %.2f)' % (
        text, closest.name, closest.confidence, floor
    )


def _run_numbers(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    results = []
    lines = []
    for text in _utterances(args):
        value = parse_number(text)
        shown: Any = value
        if value is not None and value.is_integer():
            shown = int(value)
        results.append({"text": text, "value": shown})
        lines.append("%s = %s" % (text, "not a number" if value is None else shown))
    return results, lines


def _run_matches(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    commands = _build_commands(args)
    results: List[Dict[str, Any]] = []
    blocks: List[str] = []
    missed = False
    for text in _utterances(args):
        found = commands.match(text)
        ranked = commands.rank(text) if (args.all or found is None) else []
        entry: Dict[str, Any] = {
            "text": text,
            "matched": found is not None,
            "match": found.to_dict() if found is not None else None,
        }
        if found is None:
            missed = True
            entry["closest"] = (
                {"name": ranked[0].name, "confidence": round(ranked[0].confidence, 4)}
                if ranked else None
            )
            block = _no_match_text(text, ranked, commands.min_confidence)
        else:
            block = found.summary()
        if args.all:
            entry["ranking"] = [
                {"name": m.name, "confidence": round(m.confidence, 4)} for m in ranked
            ]
            block += "\nAll commands: " + ", ".join(
                "'%s' %.2f" % (m.name, m.confidence) for m in ranked
            )
        results.append(entry)
        blocks.append(block)
    return results, blocks, missed


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``voice-commands-ai`` command."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - a closed or detached stdin
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    missed = False
    try:
        if args.number:
            results, blocks = _run_numbers(args)
            separator = "\n"
        else:
            results, blocks, missed = _run_matches(args)
            separator = "\n\n"
    except _UsageError as exc:
        print("voice-commands-ai: error: %s" % exc, file=sys.stderr)
        return 2

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError as exc:
            print("voice-commands-ai: error: could not write %s: %s" % (args.output, exc),
                  file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif blocks:
        print(separator.join(blocks))
        if args.output:
            print("\nResults written to %s" % args.output)
    else:
        print("voice-commands-ai: no input text given", file=sys.stderr)
    if args.fail_on_no_match and missed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
