"""Command line interface: ``llm-router-lite complexity|route|models``.

The CLI never calls a model. It scores prompts and shows which model *would*
be picked, which is enough to sanity-check a line-up before wiring handlers in.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .complexity import complexity
from .router import PREFERENCES, Router

COMMANDS = ("complexity", "route", "models")

#: The line-up used when you do not pass --models or --model. These are
#: examples, not real services: nothing here is ever contacted.
EXAMPLE_MODELS = (
    {
        "name": "local-small",
        "cost": 0.0,
        "quality": 0.35,
        "latency_ms": 120.0,
        "max_tokens": 4096,
        "tags": ["local", "offline"],
    },
    {
        "name": "cloud-mid",
        "cost": 0.0006,
        "quality": 0.70,
        "latency_ms": 700.0,
        "max_tokens": 32000,
        "tags": ["cloud"],
    },
    {
        "name": "cloud-large",
        "cost": 0.01,
        "quality": 0.96,
        "latency_ms": 2200.0,
        "max_tokens": 128000,
        "tags": ["cloud"],
    },
)

_SPEC_HELP = "NAME[:COST[:QUALITY[:LATENCY_MS[:TAG,TAG]]]]"


def _make_console_utf8_safe() -> None:
    """Never let a non-ASCII prompt kill the process under a pipe or in CI."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def _unused_handler(prompt: str, **kw: Any) -> str:
    """Stand-in handler. The CLI routes but never calls."""
    raise RuntimeError(
        "the llm-router-lite CLI never calls a model; use the Python API with "
        "your own handlers to actually complete a prompt"
    )


def _float_or_fail(text: str, label: str, spec: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            "{0} in --model {1!r} must be a number, got {2!r}".format(label, spec, text)
        )


def parse_model_spec(spec: str) -> Dict[str, Any]:
    """Turn ``name:cost:quality:latency_ms:tag,tag`` into model options."""
    parts = spec.split(":")
    name = parts[0].strip()
    if not name:
        raise ValueError("--model {0!r} has no name; expected {1}".format(spec, _SPEC_HELP))
    if len(parts) > 5:
        raise ValueError(
            "--model {0!r} has too many fields; expected {1}".format(spec, _SPEC_HELP)
        )
    options: Dict[str, Any] = {"name": name, "handler": _unused_handler}
    if len(parts) > 1 and parts[1].strip():
        options["cost"] = _float_or_fail(parts[1].strip(), "cost", spec)
    if len(parts) > 2 and parts[2].strip():
        options["quality"] = _float_or_fail(parts[2].strip(), "quality", spec)
    if len(parts) > 3 and parts[3].strip():
        options["latency_ms"] = _float_or_fail(parts[3].strip(), "latency_ms", spec)
    if len(parts) > 4 and parts[4].strip():
        options["tags"] = [tag.strip() for tag in parts[4].split(",") if tag.strip()]
    return options


def load_model_file(path: str) -> List[Dict[str, Any]]:
    """Read a JSON file of model options: a list, or an object of name -> options."""
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("{0} is not valid JSON: {1}".format(path, exc))
    if isinstance(data, dict):
        items = []
        for name, options in data.items():
            if not isinstance(options, dict):
                raise ValueError(
                    "{0}: model {1!r} must map to an object of options".format(path, name)
                )
            merged = dict(options)
            merged["name"] = name
            items.append(merged)
    elif isinstance(data, list):
        items = []
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError(
                    "{0}: every model must be an object, got {1}".format(
                        path, type(entry).__name__
                    )
                )
            items.append(dict(entry))
    else:
        raise ValueError(
            "{0}: expected a list of models or an object of name -> options".format(path)
        )
    for entry in items:
        entry.setdefault("handler", _unused_handler)
        entry.pop("description", None)
    return items


def build_router(args: argparse.Namespace) -> Router:
    """The line-up for this run: --models file, --model specs, or the examples."""
    specs: List[Dict[str, Any]] = []
    if getattr(args, "models", None):
        specs.extend(load_model_file(args.models))
    for spec in getattr(args, "model", None) or []:
        specs.append(parse_model_spec(spec))
    if not specs:
        specs = [dict(entry, handler=_unused_handler) for entry in EXAMPLE_MODELS]
    return Router(specs)


def _read_stdin_text() -> str:
    """Read stdin as UTF-8 whatever the console codepage happens to be.

    On Windows ``sys.stdin`` under a pipe decodes with the ANSI codepage, which
    turns a piped Japanese or emoji prompt into mojibake. Reading the raw bytes
    and decoding them here keeps a piped prompt intact, and ``errors="replace"``
    means input that genuinely is not UTF-8 still scores instead of crashing.
    """
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        return buffer.read().decode("utf-8", "replace")
    return sys.stdin.read()  # a text stream stood in, e.g. StringIO in a test


def read_prompt(raw: Optional[str]) -> str:
    """The prompt from the command line, or from stdin when it is '-' or absent."""
    if raw is not None and raw != "-":
        return raw
    if raw == "-" or not sys.stdin.isatty():
        data = _read_stdin_text()
        return data.rstrip("\n")
    raise ValueError("no prompt given; pass one as an argument or pipe it on stdin")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``llm-router-lite`` command."""
    parser = argparse.ArgumentParser(
        prog="llm-router-lite",
        description=(
            "Score how hard a prompt is and show which model would answer it. "
            "This tool never contacts a model; it only compares the numbers you give it."
        ),
        epilog=(
            "With no command, 'complexity' runs. A bare prompt that is itself a "
            "command word - complexity, route or models - is read as that "
            "command; to score one of those three words as a prompt, say so: "
            "llm-router-lite complexity \"models\". Without --models or --model "
            "an example line-up (local-small, cloud-mid, cloud-large) is used."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s {0}".format(__version__)
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    def _shared(sub: argparse.ArgumentParser, *, with_models: bool) -> None:
        sub.add_argument(
            "prompt",
            nargs="?",
            help="the prompt; '-' or nothing reads stdin",
        )
        sub.add_argument("--json", action="store_true", help="print JSON instead of text")
        sub.add_argument(
            "--output", metavar="PATH", help="write the output to a UTF-8 file as well"
        )
        if with_models:
            sub.add_argument(
                "--models", metavar="PATH", help="JSON file describing the line-up"
            )
            sub.add_argument(
                "--model",
                metavar=_SPEC_HELP,
                action="append",
                help="one model, repeatable, e.g. --model big:0.01:0.95:900:cloud",
            )

    complexity_parser = subparsers.add_parser(
        "complexity", help="score how hard a prompt is"
    )
    _shared(complexity_parser, with_models=False)

    route_parser = subparsers.add_parser(
        "route", help="show which model would answer the prompt"
    )
    _shared(route_parser, with_models=True)
    route_parser.add_argument(
        "--prefer",
        choices=PREFERENCES,
        default="balanced",
        help="what to optimise for (default: balanced)",
    )
    route_parser.add_argument(
        "--require",
        metavar="TAG",
        action="append",
        help="a tag the model must carry, repeatable",
    )
    route_parser.add_argument(
        "--max-cost",
        metavar="AMOUNT",
        type=float,
        dest="max_cost",
        help="ceiling on the estimated cost of one call",
    )

    models_parser = subparsers.add_parser("models", help="list the line-up")
    models_parser.add_argument(
        "--json", action="store_true", help="print JSON instead of text"
    )
    models_parser.add_argument(
        "--output", metavar="PATH", help="write the output to a UTF-8 file as well"
    )
    models_parser.add_argument(
        "--models", metavar="PATH", help="JSON file describing the line-up"
    )
    models_parser.add_argument(
        "--model", metavar=_SPEC_HELP, action="append", help="one model, repeatable"
    )

    return parser


def with_default_command(argv: Sequence[str]) -> List[str]:
    """Let ``llm-router-lite "some prompt"`` mean ``complexity "some prompt"``.

    A command word wins over a bare prompt, so ``llm-router-lite models`` lists
    the line-up rather than scoring the word "models". The three command words
    are therefore the one set of prompts the shorthand cannot express; write
    ``llm-router-lite complexity "models"`` to score one of them. Every other
    prompt, one word or many, goes to ``complexity`` as documented.
    """
    args = list(argv)
    if not args:
        return ["complexity"]
    if args[0] in COMMANDS or args[0] in ("-h", "--help", "--version"):
        return args
    return ["complexity"] + args


def _signal_lines(payload: Dict[str, Any]) -> List[str]:
    return [
        "  {0:<13}{1:.2f}".format(name, value)
        for name, value in sorted(payload.items())
    ]


def _run_complexity(args: argparse.Namespace) -> str:
    prompt = read_prompt(getattr(args, "prompt", None))
    result = complexity(prompt)
    if args.json:
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    lines = [result.summary(), "", "signals:"]
    lines.extend(_signal_lines(result.signals))
    lines.append("")
    lines.append(
        "{0} characters, about {1} tokens".format(
            result.characters, result.estimated_tokens
        )
    )
    return "\n".join(lines)


def _run_route(args: argparse.Namespace) -> str:
    prompt = read_prompt(getattr(args, "prompt", None))
    router = build_router(args)
    decision = router.route(
        prompt,
        prefer=args.prefer,
        require=args.require,
        max_cost=args.max_cost,
    )
    if args.json:
        return json.dumps(decision.to_dict(), indent=2, ensure_ascii=False)
    lines = [
        decision.summary(),
        "",
        "considered:",
    ]
    for candidate in decision.considered:
        lines.append(
            "  {0:<16} score {1:.3f}  cost {2:<10g} quality {3:.2f}".format(
                candidate.name,
                candidate.score,
                candidate.estimated_cost,
                candidate.quality,
            )
        )
    if decision.demoted:
        lines.append("")
        lines.append("ranked behind, still a fallback:")
        for name, why in sorted(decision.demoted.items()):
            lines.append("  {0:<16} {1}".format(name, why))
    if decision.skipped:
        lines.append("")
        lines.append("set aside, out of the running:")
        for name, why in sorted(decision.skipped.items()):
            lines.append("  {0:<16} {1}".format(name, why))
    lines.append("")
    lines.append(
        "no model was called: routing only compares the numbers you registered"
    )
    return "\n".join(lines)


def _run_models(args: argparse.Namespace) -> str:
    router = build_router(args)
    if args.json:
        return json.dumps(
            [model.to_dict() for model in router.models], indent=2, ensure_ascii=False
        )
    lines = ["{0} model(s):".format(len(router))]
    for model in router.models:
        lines.append("  " + model.describe())
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``llm-router-lite`` command."""
    _make_console_utf8_safe()
    parser = build_parser()
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(with_default_command(raw))
    command = args.command or "complexity"
    try:
        if command == "complexity":
            text = _run_complexity(args)
        elif command == "route":
            text = _run_route(args)
        else:
            text = _run_models(args)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = exc.args[0] if (isinstance(exc, KeyError) and exc.args) else exc
        sys.stderr.write("error: {0}\n".format(message))
        return 1
    sys.stdout.write(text + "\n")
    output = getattr(args, "output", None)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
