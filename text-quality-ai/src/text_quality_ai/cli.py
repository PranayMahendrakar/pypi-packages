"""Command line interface: ``text-quality-ai [FILE ...] [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence, Tuple

from . import __version__
from ._core import TextScorer, available_targets, compare
from ._report import QualityReport

_DESCRIPTION = """Score text for readability, repetition, structure and clarity, and say what to fix.

FILE is a plain text file; "-" or no FILE at all reads standard input. Several
files are scored together, with one line per file at the end."""

_EPILOG = """examples:
  text-quality-ai article.md
  text-quality-ai *.md --target marketing
  text-quality-ai --text "Your sentence here."
  cat draft.txt | text-quality-ai --json > report.json
  text-quality-ai draft.txt --compare final.txt
  text-quality-ai article.md --fail-under 70"""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``--help`` can be tested."""
    parser = argparse.ArgumentParser(
        prog="text-quality-ai",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", metavar="FILE",
                        help='text file(s) to score; "-" or nothing reads standard input')
    parser.add_argument("--text", metavar="TEXT", help="score this string instead of reading a file")
    parser.add_argument("--target", choices=available_targets(), default="general",
                        help="audience to aim at (default: general)")
    parser.add_argument("--compare", metavar="FILE", dest="compare_with",
                        help="also score this file and print how the two differ")
    parser.add_argument("--long-sentence-words", type=int, default=30, dest="long_sentence_words",
                        metavar="N", help="a sentence longer than N words counts as long (default: 30)")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON instead of the summary")
    parser.add_argument("--output", metavar="PATH", help="write the output to this file as UTF-8")
    parser.add_argument("--fail-under", type=float, default=None, metavar="SCORE",
                        dest="fail_under", help="exit with code 2 if the overall score is below SCORE")
    parser.add_argument("--version", action="version", version=f"text-quality-ai {__version__}")
    return parser


def read_text_file(path: str) -> str:
    """Read one input. ``-`` means standard input."""
    if path == "-":
        data = sys.stdin.buffer.read()
        return data.decode("utf-8", errors="replace")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def load_inputs(inputs: Sequence[str], text: Optional[str]) -> Tuple[List[str], List[str]]:
    """Return ``(texts, labels)`` for the CLI arguments given."""
    if text is not None:
        if inputs:
            raise ValueError("give either --text or file arguments, not both")
        return [text], ["--text"]
    if not inputs:
        return [read_text_file("-")], ["<stdin>"]
    return [read_text_file(path) for path in inputs], list(inputs)


def _per_document_lines(report: QualityReport, labels: Sequence[str]) -> List[str]:
    if report.documents is None or len(report.documents) < 2:
        return []
    lines = ["", "  Per document:"]
    for label, document in zip(labels, report.documents):
        lines.append(f"    {document.score:5.1f} {document.grade}  {label}")
    return lines


def render(report: QualityReport, labels: Sequence[str], as_json: bool) -> str:
    """The text the CLI prints for one run."""
    if as_json:
        payload = report.to_dict()
        payload["inputs"] = list(labels)
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return "\n".join([report.summary()] + _per_document_lines(report, labels))


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a character from the text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):       # pragma: no cover - closed or odd streams
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        texts, labels = load_inputs(args.inputs, args.text)
        scorer = TextScorer(args.target, long_sentence_words=args.long_sentence_words)
        report = scorer.score(texts if len(texts) > 1 else texts[0])

        if args.compare_with:
            other = read_text_file(args.compare_with)
            # The same threshold has to reach both sides, or the report block
            # and the compare block would disagree about the same file.
            deltas = compare(texts if len(texts) > 1 else texts[0], other,
                             target=args.target,
                             long_sentence_words=args.long_sentence_words)
            if args.json:
                payload = {"report": report.to_dict(), "inputs": list(labels),
                           "compare_with": args.compare_with, "compare": deltas}
                output = json.dumps(payload, indent=2, ensure_ascii=False)
            else:
                lines = [render(report, labels, False), "",
                         f"  Compared with {args.compare_with}: {deltas['summary']}"]
                for name, delta in deltas["components"].items():
                    lines.append(f"    {name:<12} {delta:+6.1f}")
                output = "\n".join(lines)
        else:
            output = render(report, labels, args.json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(output + "\n")
            print(f"wrote {args.output}")
        else:
            print(output)

        if args.fail_under is not None and report.score < args.fail_under:
            print(f"text-quality-ai: score {report.score:.1f} is below {args.fail_under:.1f}",
                  file=sys.stderr)
            return 2
    except (ValueError, TypeError, KeyError, OSError, UnicodeError) as exc:
        print(f"text-quality-ai: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
