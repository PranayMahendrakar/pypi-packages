"""Command line: ``offline-stt-router [choose|engines|machine|transcribe]``.

``offline-stt-router`` alone says what to use on this machine for English.
A bare audio path (``offline-stt-router talk.wav``) transcribes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._plan import PREFERENCES
from ._run import TranscriptionFailed
from .router import Router

PROG = "offline-stt-router"
COMMANDS = ("choose", "engines", "machine", "transcribe")


def _make_console_utf8_safe() -> None:
    """Never let a non-ASCII path, language name or transcript kill the process."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``offline-stt-router`` command."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Find which speech-to-text engines are installed locally and pick the right "
            "one for this machine and language. Probing imports no engine, runs no "
            "binary and makes no network call; transcribing uses local models only."
        ),
        epilog=(
            "With no command, 'choose' runs. A first argument that is not a command or an "
            "option is taken as an audio file to transcribe. Extra model folders can also "
            "be given in the OFFLINE_STT_MODELS environment variable."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s {0}".format(__version__))
    sub = parser.add_subparsers(dest="command", metavar="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true", help="print JSON instead of text")
        p.add_argument("--output", metavar="PATH", help="also write the output to this UTF-8 file")
        p.add_argument("--models-dir", metavar="DIR", action="append", default=[], dest="models_dir",
                       help="another folder to search for models (repeatable)")

    def choosing(p: argparse.ArgumentParser) -> None:
        p.add_argument("--language", "-l", default="en",
                       help="language code or name, e.g. en, hi, 'Hindi'; 'auto' to detect (default: en)")
        p.add_argument("--prefer", choices=PREFERENCES, default="balanced",
                       help="what to optimise for (default: balanced)")
        p.add_argument("--max-ram-gb", type=float, default=None, dest="max_ram_gb", metavar="GB",
                       help="cap on the RAM a model may use")

    choose = sub.add_parser("choose", help="say which engine and model to use here")
    choosing(choose)
    common(choose)

    engines = sub.add_parser("engines", help="list every engine, installed or not")
    common(engines)

    machine = sub.add_parser("machine", help="show RAM, CPU and GPU as the router sees them")
    machine.add_argument("--json", action="store_true", help="print JSON instead of text")
    machine.add_argument("--output", metavar="PATH", help="also write the output to this UTF-8 file")

    transcribe = sub.add_parser("transcribe", help="transcribe an audio file with local models only")
    transcribe.add_argument("audio", help="path to an audio file (WAV is read directly)")
    choosing(transcribe)
    transcribe.add_argument("--engine", default=None, help="insist on one engine")
    transcribe.add_argument("--model", default=None, help="insist on one model (a name or a path)")
    transcribe.add_argument("--no-fallback", action="store_true", dest="no_fallback",
                            help="stop after the first engine fails instead of trying the alternatives")
    common(transcribe)
    return parser


def with_default_command(argv: Sequence[str]) -> List[str]:
    """``offline-stt-router`` -> choose; ``offline-stt-router talk.wav`` -> transcribe."""
    args = list(argv)
    if not args:
        return ["choose"]
    first = args[0]
    if first in COMMANDS or first in ("-h", "--help", "--version"):
        return args
    if first.startswith("-"):
        return ["choose"] + args
    return ["transcribe"] + args


def _emit(text: str, output: Optional[str]) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _run(args: argparse.Namespace) -> str:
    command = args.command or "choose"
    router = Router(model_dirs=getattr(args, "models_dir", []) or [])
    if command == "machine":
        found = router.machine()
        return _dump(found.to_dict()) if args.json else found.summary()
    if command == "engines":
        engines = router.available()
        if args.json:
            return _dump([e.to_dict() for e in engines])
        ready = sum(1 for e in engines if e.ready)
        lines = ["{0} of {1} engines ready.".format(ready, len(engines)), ""]
        for engine in engines:
            lines.append(engine.summary())
            lines.append("")
        return "\n".join(lines).rstrip()
    if command == "transcribe":
        result = router.transcribe_detailed(
            args.audio,
            language=args.language,
            prefer=args.prefer,
            max_ram_gb=args.max_ram_gb,
            engine=args.engine,
            model=args.model,
            fallback=not args.no_fallback,
        )
        if args.json:
            return _dump(result.to_dict())
        return result.text
    choice = router.choose(language=args.language, prefer=args.prefer, max_ram_gb=args.max_ram_gb)
    return _dump(choice.to_dict()) if args.json else choice.summary()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``offline-stt-router`` command."""
    _make_console_utf8_safe()
    parser = build_parser()
    raw = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(with_default_command(raw))
    wants_json = bool(getattr(args, "json", False))
    try:
        text = _run(args)
    except (TranscriptionFailed, ValueError, TypeError, OSError, KeyError) as exc:
        message = exc.args[0] if (isinstance(exc, KeyError) and exc.args) else str(exc)
        if wants_json:
            payload: Dict[str, Any] = {"ok": False, "error": message}
            choice = getattr(exc, "choice", None)
            if choice is not None:
                payload["choice"] = choice.to_dict()
            attempts = getattr(exc, "attempts", None)
            if attempts:
                payload["attempts"] = [a.to_dict() for a in attempts]
            _emit(_dump(payload), getattr(args, "output", None))
        else:
            sys.stderr.write("{0}: error: {1}\n".format(PROG, message))
        return 1
    _emit(text, getattr(args, "output", None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
