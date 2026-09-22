"""Command line interface: ``rag-chunker INPUT [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from ._core import METHOD_HELP, METHODS, ChunkResult, Chunker

_DESCRIPTION = """Split a document into retrieval-sized chunks at meaning boundaries.

INPUT is a .txt, .md or .html file, or "-" to read text from standard input."""

_EPILOG = "methods:\n" + "\n".join(
    f"  {name:<11} {help_text}" for name, help_text in METHOD_HELP.items()
) + """

examples:
  rag-chunker guide.md --size 200 --overlap 30
  rag-chunker guide.md --method structural --json > chunks.json
  cat notes.txt | rag-chunker - --method recursive --text"""

SEPARATOR = "----- chunk %d -----"

METADATA_HINT = (
    "--metadata must be a JSON object, for example '{\"doc\": \"guide\"}'"
)


def parse_metadata(raw: Optional[str]) -> Optional[dict]:
    """Parse ``--metadata``; both failure modes name the option in the message."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{METADATA_HINT} ({exc})") from None
    if not isinstance(value, dict):
        raise ValueError(METADATA_HINT)
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="rag-chunker",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", metavar="INPUT", help='document to chunk, or "-" for stdin')
    parser.add_argument("--size", type=int, default=512, help="words per chunk (default: 512)")
    parser.add_argument("--overlap", type=int, default=64, help="words repeated from the previous chunk (default: 64)")
    parser.add_argument("--method", choices=list(METHODS), default="recursive", help="how to find the boundaries (default: recursive)")
    parser.add_argument("--source", choices=["auto", "text", "markdown", "html"], default="auto", help="how to read INPUT (default: auto)")
    parser.add_argument("--sensitivity", type=float, default=30.0, help="semantic only: percentile of similarity valleys treated as topic shifts (default: 30)")
    parser.add_argument("--metadata", metavar="JSON", help='JSON object copied onto every chunk, e.g. {"doc": "guide"}')
    parser.add_argument("--json", action="store_true", dest="as_json", help="print the full result as JSON instead of the summary")
    parser.add_argument("--text", action="store_true", dest="as_text", help="print the chunk texts separated by a marker line")
    parser.add_argument("--show", type=int, default=3, metavar="N", help="preview the first N chunks under the summary (default: 3, 0 for none)")
    parser.add_argument("--output", metavar="PATH", help="write results here (.json, .jsonl, otherwise the chunk texts)")
    parser.add_argument("--version", action="version", version=f"rag-chunker {__version__}")
    return parser


def read_input(target: str):
    """Return the document text (for ``-``) or the path to read."""
    if target == "-":
        return sys.stdin.buffer.read().decode("utf-8", errors="replace")
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError(f"{target!r} does not exist")
    return path


def render_text(result: ChunkResult) -> str:
    """All chunk texts, separated by a marker line."""
    parts: List[str] = []
    for chunk in result.chunks:
        parts.append(SEPARATOR % chunk.index)
        parts.append(chunk.text)
    return "\n".join(parts)


def _preview(text: str, width: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def render_summary(result: ChunkResult, show: int) -> str:
    """The summary plus a short preview of the first ``show`` chunks."""
    lines = [result.summary()]
    for chunk in result.chunks[: max(0, show)]:
        trail = " > ".join(chunk.heading_path)
        label = f"  [{chunk.index}] {chunk.tokens} {result.unit}"
        if trail:
            label += f" under {trail}"
        lines.append(label)
        lines.append(f"      {_preview(chunk.text)}")
    if show and result.n_chunks > show:
        lines.append(f"  ... {result.n_chunks - show} more chunk(s)")
    return "\n".join(lines)


def write_output(path: str, result: ChunkResult) -> None:
    """Write the result to ``path``; the suffix picks the format."""
    lowered = path.lower()
    if lowered.endswith(".json"):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return
    if lowered.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as handle:
            for chunk in result.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_text(result))
        handle.write("\n")


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
        metadata = parse_metadata(args.metadata)
        chunker = Chunker(
            size=args.size,
            overlap=args.overlap,
            method=args.method,
            metadata=metadata,
            sensitivity=args.sensitivity,
            source=args.source,
        )
        result = chunker.chunk(read_input(args.input))
        if args.output:
            write_output(args.output, result)
        if args.as_json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        elif args.as_text:
            print(render_text(result))
        else:
            print(render_summary(result, args.show))
            if args.output:
                print(f"  wrote {result.n_chunks} chunk(s) to {args.output}")
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"rag-chunker: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
