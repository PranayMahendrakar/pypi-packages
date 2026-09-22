"""Command line interface: ``document-memory STORE [COMMAND] ...``."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from typing import Any, Dict, Optional, Sequence

from . import __version__
from ._memory import Memory

_DESCRIPTION = """Add to, search and read a persistent document memory.

STORE is the SQLite file to use; it is created on demand.  With no command the
store describes itself."""

_EPILOG = """examples:
  document-memory notes.db add "Solar panels turn light into power." --source guide.md
  document-memory notes.db search "solar" -k 3
  document-memory notes.db search "solar" --where '{"project": "roof"}' --json
  document-memory notes.db context "solar" --budget 120
  document-memory notes.db remember user "how do panels do in winter?"
  document-memory notes.db history -n 10
  echo "panels on the roof" | document-memory notes.db add -

--json, --output and --namespace are accepted before or after the command."""

_JSON_HINT = "{option} must be a JSON object, for example '{{\"project\": \"roof\"}}'"


def make_utf8(*streams: Any) -> None:
    """Make the given streams carry non-ASCII text under pipes and CI logs.

    Windows consoles and captured pipes often default to a legacy code page, on
    which printing a single accented or CJK character raises
    ``UnicodeEncodeError`` and reading one back gives mojibake.  Reconfiguring
    to UTF-8 with ``errors="replace"`` means text is never lost to an encoding,
    whatever the command is piped into or fed from.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def parse_json_object(raw: Optional[str], option: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON-object option; both failure modes name the option in the message."""
    if not raw:
        return None
    hint = _JSON_HINT.format(option=option)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{hint} ({exc})") from None
    if not isinstance(value, dict):
        raise ValueError(hint)
    return value


def read_text(value: str) -> str:
    """The argument itself, or standard input when it is ``-``."""
    if value == "-":
        data = sys.stdin.read()
        if not data.strip():
            raise ValueError("nothing arrived on standard input")
        return data.strip()
    return value


GLOBAL_DEFAULTS = {"namespace": "default", "as_json": False, "output": None}
"""Defaults for the flags that may appear on either side of the command name."""


def _common_options() -> argparse.ArgumentParser:
    """The flags every level accepts, so they work before *and* after the command.

    ``default=SUPPRESS`` is what makes both positions work: a flag that was not
    given adds no attribute at all, so parsing the sub-command cannot overwrite
    a value the top level already set.  The defaults are filled in afterwards by
    :func:`parse_args` rather than by ``set_defaults``, which would write a real
    default back onto these very actions and reintroduce the clobber.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--namespace",
        default=argparse.SUPPRESS,
        help="compartment inside the file (default: default)",
    )
    common.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        default=argparse.SUPPRESS,
        help="print the result as JSON instead of the summary",
    )
    common.add_argument(
        "--output",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="write the output here as well, encoded UTF-8",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so tests and docs can read it."""
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="document-memory",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    parser.add_argument("store", metavar="STORE", help="path to the memory file")
    parser.add_argument(
        "--version", action="version", version=f"document-memory {__version__}"
    )

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    add = commands.add_parser("add", parents=[common], help="store a document")
    add.add_argument("text", metavar="TEXT", help="the text to store, or - for stdin")
    add.add_argument("--id", help="give it this id (replaces any memory with that id)")
    add.add_argument("--source", help="where it came from, e.g. a filename")
    add.add_argument("--metadata", metavar="JSON", help="JSON object stored alongside")
    add.add_argument("--timestamp", help="ISO 8601 date or datetime (default: now)")

    search = commands.add_parser(
        "search", parents=[common], help="rank memories against a query"
    )
    search.add_argument("query", metavar="QUERY", help="what to look for")
    search.add_argument("-k", type=int, default=5, help="how many hits (default: 5)")
    search.add_argument("--where", metavar="JSON", help="metadata filter as a JSON object")
    search.add_argument("--since", help="only memories at or after this ISO date")
    search.add_argument(
        "--text", action="store_true", dest="as_text", help="print only the matched texts"
    )

    context = commands.add_parser(
        "context", parents=[common], help="best matches packed for a prompt"
    )
    context.add_argument("query", metavar="QUERY", help="what the prompt is about")
    context.add_argument(
        "--budget", type=int, default=2000, help="word budget (default: 2000)"
    )

    recent = commands.add_parser("recent", parents=[common], help="the newest memories")
    recent.add_argument("-n", type=int, default=10, help="how many (default: 10)")
    recent.add_argument("--source", help="only this source")

    remember = commands.add_parser(
        "remember", parents=[common], help="store a conversation turn"
    )
    remember.add_argument("role", metavar="ROLE", help="who spoke, e.g. user")
    remember.add_argument("content", metavar="TEXT", help="what was said, or - for stdin")

    history = commands.add_parser(
        "history", parents=[common], help="the last conversation turns"
    )
    history.add_argument("-n", type=int, default=20, help="how many (default: 20)")

    get = commands.add_parser("get", parents=[common], help="show one memory by id")
    get.add_argument("id", metavar="ID")

    delete = commands.add_parser(
        "delete", parents=[common], help="remove one memory by id"
    )
    delete.add_argument("id", metavar="ID")

    commands.add_parser(
        "count", parents=[common], help="how many memories the namespace holds"
    )

    clear = commands.add_parser(
        "clear", parents=[common], help="delete every memory in the namespace"
    )
    clear.add_argument("--yes", action="store_true", help="required: confirms the deletion")

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse ``argv`` and fill in the defaults the SUPPRESS trick leaves out."""
    args = build_parser().parse_args(argv)
    for dest, default in GLOBAL_DEFAULTS.items():
        if not hasattr(args, dest):
            setattr(args, dest, default)
    return args


def run(args: argparse.Namespace, memory: Memory) -> "_Output":
    """Carry out one command and return what should be printed."""
    command = args.command

    if command is None:
        return _Output(
            memory.summary(), {"store": memory.summary(), "count": memory.count()}
        )

    if command == "add":
        identifier = memory.add(
            read_text(args.text),
            id=args.id,
            metadata=parse_json_object(args.metadata, "--metadata"),
            source=args.source,
            timestamp=args.timestamp,
        )
        record = memory.get(identifier)
        return _Output(f"stored {identifier}\n{record.summary()}", record.to_dict())

    if command == "search":
        hits = memory.search(
            args.query,
            k=args.k,
            where=parse_json_object(args.where, "--where"),
            since=args.since,
        )
        if getattr(args, "as_text", False):
            return _Output("\n\n".join(hit.text for hit in hits), hits.to_dict())
        return _Output(hits.summary(), hits.to_dict())

    if command == "context":
        text = memory.context(args.query, budget=args.budget)
        return _Output(text, {"query": args.query, "budget": args.budget, "context": text})

    if command == "recent":
        records = memory.recent(args.n, source=args.source)
        return _Output(records.summary(), records.to_dict())

    if command == "remember":
        identifier = memory.remember(args.role, read_text(args.content))
        record = memory.get(identifier)
        return _Output(f"stored {identifier}\n{record.summary()}", record.to_dict())

    if command == "history":
        turns = memory.history(args.n)
        return _Output(turns.summary(), turns.to_dict())

    if command == "get":
        record = memory.get(args.id)
        if record is None:
            raise ValueError(
                f"no memory with id {args.id!r} in namespace {memory.namespace!r}"
            )
        return _Output(record.summary(), record.to_dict())

    if command == "delete":
        gone = memory.delete(args.id)
        word = "deleted" if gone else "no memory with id"
        return _Output(f"{word} {args.id}", {"id": args.id, "deleted": gone})

    if command == "count":
        total = memory.count()
        return _Output(str(total), {"namespace": memory.namespace, "count": total})

    if command == "clear":
        if not args.yes:
            raise ValueError(
                f"clear deletes every memory in namespace {memory.namespace!r}; "
                "pass --yes to confirm"
            )
        removed = memory.clear()
        return _Output(
            f"cleared {removed} memories from namespace {memory.namespace!r}",
            {"namespace": memory.namespace, "cleared": removed},
        )

    raise ValueError(f"unknown command {command!r}")  # pragma: no cover - argparse guards


class _Output:
    """What a command produced: human text and the JSON-safe equivalent."""

    def __init__(self, text: str, payload: Any) -> None:
        self.text = text
        self.payload = payload

    def render(self, as_json: bool) -> str:
        """The text to print, as JSON when ``as_json`` is set."""
        if as_json:
            return json.dumps(self.payload, indent=2, ensure_ascii=False)
        return self.text


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``document-memory`` command."""
    make_utf8(sys.stdin, sys.stdout, sys.stderr)
    args = parse_args(argv)

    memory: Optional[Memory] = None
    try:
        memory = Memory(args.store, namespace=args.namespace)
        output = run(args, memory)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"document-memory: {exc}", file=sys.stderr)
        return 2
    finally:
        if memory is not None:
            memory.close()

    rendered = output.render(args.as_json)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"wrote {args.output}")
    if rendered:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
