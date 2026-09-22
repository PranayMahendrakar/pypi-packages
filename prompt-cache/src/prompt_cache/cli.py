"""Command line interface: ``prompt-cache stats|list|get|set|clear|prune``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .cache import Cache

COMMANDS = ("stats", "list", "get", "set", "clear", "prune")

#: Widest prompt preview printed in the table, in characters.
PREVIEW_WIDTH = 56


def _make_console_utf8_safe() -> None:
    """Never let a non-ASCII prompt kill the process under a pipe or in CI."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--path",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="cache folder or .db file (default: ~/.prompt_cache)",
    )
    parser.add_argument(
        "--namespace",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="which logical cache to work on (default: default)",
    )
    parser.add_argument(
        "--ttl",
        metavar="SECONDS",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds an entry stays valid; 0 disables caching",
    )


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``prompt-cache`` command."""
    parser = argparse.ArgumentParser(
        prog="prompt-cache",
        description="Inspect and maintain the on-disk cache of LLM answers.",
        epilog="With no command, 'stats' runs: 'prompt-cache' prints what the cache holds.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s {0}".format(__version__))
    _common(parser)
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    stats = subparsers.add_parser("stats", help="print what the cache holds (the default)")
    _common(stats)
    stats.add_argument("--json", action="store_true", help="print the statistics as JSON")
    stats.add_argument("--output", "-o", metavar="PATH", help="also write the JSON to PATH")

    listing = subparsers.add_parser("list", help="list cached entries, most recently used first")
    _common(listing)
    listing.add_argument("--limit", type=int, default=20, metavar="N", help="how many to show (default 20)")
    listing.add_argument("--json", action="store_true", help="print the entries as JSON")
    listing.add_argument("--output", "-o", metavar="PATH", help="also write the JSON to PATH")

    get = subparsers.add_parser("get", help="print one cached answer (exit 1 when there is none)")
    _common(get)
    get.add_argument("prompt", help="the prompt to look up")
    get.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="a parameter that was part of the key, e.g. --param model=gpt-4o (repeatable)",
    )
    get.add_argument("--json", action="store_true", help="print the answer as JSON")

    setter = subparsers.add_parser("set", help="store one answer by hand")
    _common(setter)
    setter.add_argument("prompt", help="the prompt to store under")
    setter.add_argument("value", help="the answer to store")
    setter.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="a parameter that becomes part of the key (repeatable)",
    )

    clear = subparsers.add_parser("clear", help="delete every entry in the namespace")
    _common(clear)
    clear.add_argument("--all", action="store_true", help="delete every namespace in the file")

    prune = subparsers.add_parser("prune", help="delete entries whose time to live has run out")
    _common(prune)

    return parser


def _parse_params(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Turn ``--param model=gpt-4o --param temperature=0.7`` into a dict.

    Values are read as JSON when they look like JSON, otherwise kept as text,
    so ``0.7`` is a number and ``gpt-4o`` is a string.
    """
    params: Dict[str, Any] = {}
    for item in items or ():
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError("--param needs KEY=VALUE, got {0!r}".format(item))
        try:
            params[name] = json.loads(raw)
        except ValueError:
            params[name] = raw
    return params


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=repr, sort_keys=False)


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")


def _shorten(text: str, width: int = PREVIEW_WIDTH) -> str:
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 3] + "..."


def _cache_from(args: argparse.Namespace) -> Cache:
    return Cache(
        getattr(args, "path", None),
        namespace=getattr(args, "namespace", None),
        ttl=getattr(args, "ttl", None),
    )


def _cmd_stats(args: argparse.Namespace, cache: Cache) -> int:
    stats = cache.stats()
    if getattr(args, "json", False):
        data = stats.to_dict()
        data["path"] = str(cache.path)
        data["namespace"] = cache.namespace
        text = _dump_json(data)
        print(text)
    else:
        text = _dump_json(dict(stats.to_dict(), path=str(cache.path), namespace=cache.namespace))
        print(stats.summary())
        print("  namespace: {0}".format(cache.namespace))
        print("  file: {0}".format(cache.path))
    if getattr(args, "output", None):
        _write(args.output, text)
    return 0


def _cmd_list(args: argparse.Namespace, cache: Cache) -> int:
    rows: List[Dict[str, Any]] = cache.entries(limit=args.limit)
    text = _dump_json(rows)
    if args.json:
        print(text)
    elif not rows:
        print("prompt-cache: namespace {0!r} is empty ({1})".format(cache.namespace, cache.path))
    else:
        print(
            "prompt-cache: {0} entr{1} in namespace {2!r}".format(
                len(rows), "y" if len(rows) == 1 else "ies", cache.namespace
            )
        )
        print("  {0:>5}  {1:>8}  {2:<7}  {3}".format("hits", "bytes", "format", "prompt"))
        for row in rows:
            print(
                "  {0:>5}  {1:>8}  {2:<7}  {3}".format(
                    row["hits"], row["size"], row["format"], _shorten(row["preview"])
                )
            )
    if getattr(args, "output", None):
        _write(args.output, text)
    return 0


def _cmd_get(args: argparse.Namespace, cache: Cache) -> int:
    params = _parse_params(args.param)
    found, value = cache.lookup(args.prompt, **params)
    if not found:
        sys.stderr.write("prompt-cache: nothing cached for that prompt\n")
        if args.json:
            print(_dump_json({"found": False, "value": None}))
        return 1
    if args.json:
        print(_dump_json({"found": True, "value": value}))
    elif isinstance(value, str):
        print(value)
    else:
        print(_dump_json(value))
    return 0


def _cmd_set(args: argparse.Namespace, cache: Cache) -> int:
    params = _parse_params(args.param)
    cache.set(args.prompt, args.value, **params)
    if not cache.enabled:
        print("prompt-cache: ttl is 0, so nothing was stored")
        return 0
    print("prompt-cache: stored under key {0}".format(cache.key(args.prompt, **params)[:16]))
    return 0


def _cmd_clear(args: argparse.Namespace, cache: Cache) -> int:
    removed = cache.clear("*" if args.all else None)
    where = "the whole file" if args.all else "namespace {0!r}".format(cache.namespace)
    print("prompt-cache: removed {0} entr{1} from {2}".format(removed, "y" if removed == 1 else "ies", where))
    return 0


def _cmd_prune(args: argparse.Namespace, cache: Cache) -> int:
    removed = cache.prune()
    print(
        "prompt-cache: removed {0} expired entr{1} from namespace {2!r}".format(
            removed, "y" if removed == 1 else "ies", cache.namespace
        )
    )
    return 0


_HANDLERS = {
    "stats": _cmd_stats,
    "list": _cmd_list,
    "get": _cmd_get,
    "set": _cmd_set,
    "clear": _cmd_clear,
    "prune": _cmd_prune,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``prompt-cache`` command. Returns the exit code."""
    _make_console_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "stats"
    if command == "stats" and not hasattr(args, "json"):
        args.json = False
        args.output = None
    try:
        cache = _cache_from(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error exits
    try:
        try:
            return _HANDLERS[command](args, cache)
        finally:
            cache.close()
    except (OSError, ValueError, TypeError) as exc:
        sys.stderr.write("prompt-cache: error: {0}\n".format(exc))
        return 1


__all__ = ["COMMANDS", "build_parser", "main"]
