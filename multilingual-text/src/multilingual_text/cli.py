"""Command line interface: ``multilingual-text TEXT [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence

from . import __version__
from ._detect import detect, is_rtl, script_of
from ._normalize import CASE_MODES, FORMS, normalize
from ._translit import TARGETS, transliterate

_DESCRIPTION = """Detect the language of some text, clean it up, or romanise it.

TEXT is the text itself, or "-" to read standard input.  Use --file to read a
UTF-8 file instead."""

_EPILOG = """examples:
  multilingual-text "El perro es muy grande"
  multilingual-text "Privet" --mode all --json
  multilingual-text --file notes.txt --mode normalize --case fold --diacritics
  echo "Nou mepwe" | multilingual-text - --top 3
  multilingual-text "Привет, мир" --mode transliterate --to ascii"""

MODES = ("detect", "normalize", "transliterate", "all")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="multilingual-text",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        metavar="TEXT",
        help='the text to work on, or "-" to read standard input',
    )
    parser.add_argument(
        "--file", metavar="PATH", help="read the text from this UTF-8 file instead"
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="detect",
        help="what to do with the text (default: detect)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        metavar="N",
        help="how many languages to rank (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the result as JSON instead of a summary",
    )
    parser.add_argument(
        "--output", metavar="PATH", help="also write the result to this file"
    )

    group = parser.add_argument_group("normalize options")
    group.add_argument(
        "--form",
        choices=list(FORMS) + ["none"],
        default="NFC",
        help="unicode normalisation form (default: NFC)",
    )
    group.add_argument(
        "--case", choices=list(CASE_MODES), help="change the case of every letter"
    )
    group.add_argument(
        "--no-quotes",
        action="store_false",
        dest="quotes",
        help="keep smart quotes, dashes and ellipses as they are",
    )
    group.add_argument(
        "--no-whitespace",
        action="store_false",
        dest="whitespace",
        help="keep exotic spaces and runs of whitespace as they are",
    )
    group.add_argument(
        "--punctuation", action="store_true", help="delete punctuation"
    )
    group.add_argument(
        "--digits", action="store_true", help="convert every digit to ASCII"
    )
    group.add_argument(
        "--diacritics", action="store_true", help="strip accents from Latin-like text"
    )

    group = parser.add_argument_group("transliterate options")
    group.add_argument(
        "--to",
        choices=list(TARGETS),
        default="latin",
        help="latin keeps the scholarly diacritics, ascii folds them (default: latin)",
    )

    parser.add_argument(
        "--version", action="version", version="multilingual-text %s" % __version__
    )
    return parser


def read_input(text: Optional[str], path: Optional[str]) -> str:
    """The text to work on, from the argument, a file, or standard input."""
    if path:
        if text not in (None, "-"):
            raise ValueError("give either TEXT or --file, not both")
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    if text is None:
        raise ValueError('nothing to work on: pass TEXT, --file PATH, or "-" for stdin')
    if text == "-":
        return sys.stdin.buffer.read().decode("utf-8", errors="replace")
    return text


def _normalize_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "form": None if args.form == "none" else args.form,
        "case": args.case,
        "whitespace": args.whitespace,
        "punctuation": args.punctuation,
        "digits": args.digits,
        "quotes": args.quotes,
        "diacritics": args.diacritics,
    }


def run(text: str, args: argparse.Namespace) -> Dict[str, object]:
    """Do the work and return a JSON-safe dict of everything that was asked for."""
    payload: Dict[str, object] = {}
    if args.mode in ("detect", "all"):
        result = detect(text, top=max(1, args.top))
        payload["detect"] = result.to_dict()
        payload["detect"]["summary"] = result.summary()
    if args.mode in ("normalize", "all"):
        payload["normalize"] = normalize(text, **_normalize_kwargs(args))
    if args.mode in ("transliterate", "all"):
        romanised = transliterate(text, to=args.to)
        payload["transliterate"] = romanised.to_dict()
    if args.mode == "all":
        payload["script"] = script_of(text)
        payload["rtl"] = is_rtl(text)
    return payload


def _one_block(text: object) -> str:
    """Trim the trailing newline a file always has, so the layout holds.

    Only the printed form is trimmed.  What ``--json`` and ``--output`` carry
    is the exact text the library returned.
    """
    return str(text).rstrip("\n")


def render(payload: Dict[str, object], mode: str) -> str:
    """Human-readable text for everything in ``payload``."""
    lines: List[str] = []
    if "detect" in payload:
        detection = payload["detect"]
        lines.append(str(detection["summary"]))
        if mode == "all":
            lines.append(
                "  scripts: "
                + ", ".join(
                    "%s %.0f%%" % (name, share * 100)
                    for name, share in dict(detection["scripts"]).items()
                )
            )
    if "normalize" in payload:
        lines.append("normalized: %s" % _one_block(payload["normalize"]))
    if "transliterate" in payload:
        romanised = payload["transliterate"]
        lines.append("transliterated: %s" % _one_block(romanised["text"]))
        lines.append("  %s" % romanised["note"])
    if mode == "all":
        lines.append(
            "script: %s; right-to-left: %s" % (payload["script"], payload["rtl"])
        )
    return "\n".join(lines)


def write_output(path: str, payload: Dict[str, object], mode: str, as_json: bool) -> None:
    """Write the result to ``path``; ``.json`` gets JSON, anything else text."""
    with open(path, "w", encoding="utf-8") as handle:
        if as_json or path.lower().endswith(".json"):
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        else:
            handle.write(render(payload, mode))
            handle.write("\n")


def _make_console_safe() -> None:
    """Never crash on a console or pipe that cannot encode a character."""
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
        text = read_input(args.text, args.file)
        payload = run(text, args)
        if args.output:
            write_output(args.output, payload, args.mode, args.as_json)
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(render(payload, args.mode))
            if args.output:
                print("wrote %s" % args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print("multilingual-text: error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
