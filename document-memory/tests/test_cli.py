"""The ``document-memory`` command line.

Two things get most of the attention here: that every sub-command reaches the
right ``Memory`` method, and that non-ASCII text survives the trip out through a
pipe.  A Windows console or a captured pipe often defaults to a legacy code
page, on which printing one accented or CJK character raises
``UnicodeEncodeError``; ``cli.make_utf8`` is what prevents that, and
:func:`test_non_ascii_output_survives_a_legacy_code_page` drives a real
subprocess to prove it.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from document_memory import Memory, __version__
from document_memory.cli import build_parser, main, parse_args, parse_json_object

FRENCH = "Les panneaux solaires transforment la lumiere en electricite."
ACCENTED = "Les panneaux solaires transforment la lumière en électricité."
JAPANESE = "太陽光パネルは光を電気に変えます。"


def run(capsys, *argv):
    """Call ``main`` with ``argv`` and return ``(exit code, stdout, stderr)``."""
    code = main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------- the basics


def test_help_and_version_work():
    """``--help`` and ``--version`` exit 0 through argparse's SystemExit."""
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as exit_info:
            parse_args([flag])
        assert exit_info.value.code == 0
    assert build_parser().prog == "document-memory"


def test_help_text_lists_every_command():
    text = build_parser().format_help()
    for command in (
        "add",
        "search",
        "context",
        "recent",
        "remember",
        "history",
        "get",
        "delete",
        "count",
        "clear",
    ):
        assert command in text


def test_version_matches_the_package(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--version"])
    assert __version__ in capsys.readouterr().out


def test_with_no_command_the_store_describes_itself(capsys, tmp_path):
    store = tmp_path / "notes.db"
    with Memory(store) as memory:
        memory.add("solar panels turn light into power")
    code, out, _err = run(capsys, store)
    assert code == 0
    assert "1 memory" in out
    assert "BM25" in out


# --------------------------------------------------------------- the commands


def test_add_then_search_then_get(capsys, tmp_path):
    store = tmp_path / "notes.db"

    code, out, _ = run(capsys, store, "add", "solar panels turn light into power",
                       "--source", "guide.md", "--id", "solar")
    assert code == 0
    assert "stored solar" in out

    code, out, _ = run(capsys, store, "search", "solar")
    assert code == 0
    assert "1 hit for 'solar'" in out
    assert "guide.md" in out

    code, out, _ = run(capsys, store, "get", "solar")
    assert code == 0
    assert "solar panels" in out


def test_search_json_is_valid_json_carrying_the_hits(capsys, tmp_path):
    store = tmp_path / "notes.db"
    run(capsys, store, "add", "solar panels turn light into power", "--id", "solar")
    code, out, _ = run(capsys, store, "search", "solar", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "solar"
    assert payload["items"][0]["terms"] == ["solar"]


def test_search_text_prints_only_the_matched_texts(capsys, tmp_path):
    store = tmp_path / "notes.db"
    run(capsys, store, "add", "solar panels turn light into power")
    code, out, _ = run(capsys, store, "search", "solar", "--text")
    assert out.strip() == "solar panels turn light into power"


def test_add_reads_stdin_when_the_text_is_a_dash(capsys, tmp_path, monkeypatch):
    import io

    store = tmp_path / "notes.db"
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped in from the shell\n"))
    code, out, _ = run(capsys, store, "add", "-")
    assert code == 0
    assert "piped in from the shell" in out


def test_empty_stdin_is_a_clear_error(capsys, tmp_path, monkeypatch):
    import io

    store = tmp_path / "notes.db"
    monkeypatch.setattr(sys, "stdin", io.StringIO("   \n"))
    code, _out, err = run(capsys, store, "add", "-")
    assert code == 2
    assert "standard input" in err


def test_metadata_and_where_filter_through_the_cli(capsys, tmp_path):
    store = tmp_path / "notes.db"
    run(capsys, store, "add", "solar panels on the roof", "--metadata",
        '{"project": "roof"}')
    run(capsys, store, "add", "solar panels in the field", "--metadata",
        '{"project": "field"}')

    code, out, _ = run(capsys, store, "search", "solar", "--where",
                       '{"project": "roof"}', "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["count"] == 1
    assert payload["items"][0]["metadata"]["project"] == "roof"


def test_context_respects_the_budget(capsys, tmp_path):
    store = tmp_path / "notes.db"
    for n in range(10):
        run(capsys, store, "add", f"solar panels number {n} turn light into power")
    code, out, _ = run(capsys, store, "context", "solar", "--budget", "40")
    assert code == 0
    assert len(out.split()) <= 40


def test_remember_and_history(capsys, tmp_path):
    store = tmp_path / "notes.db"
    run(capsys, store, "remember", "user", "how do panels do in winter?")
    run(capsys, store, "remember", "assistant", "less light, so less power.")
    code, out, _ = run(capsys, store, "history", "-n", "10")
    assert code == 0
    assert out.index("how do panels") < out.index("less light"), "oldest first"


def test_recent_count_delete_and_clear(capsys, tmp_path):
    store = tmp_path / "notes.db"
    run(capsys, store, "add", "first", "--id", "a", "--source", "one.md")
    run(capsys, store, "add", "second", "--id", "b", "--source", "two.md")

    code, out, _ = run(capsys, store, "count")
    assert out.strip() == "2"

    code, out, _ = run(capsys, store, "recent", "--source", "one.md")
    assert "1 memory" in out and "one.md" in out

    code, out, _ = run(capsys, store, "delete", "a")
    assert "deleted a" in out

    code, out, _ = run(capsys, store, "delete", "a")
    assert "no memory with id a" in out

    code, out, _ = run(capsys, store, "clear", "--yes")
    assert "cleared 1" in out


def test_clear_without_yes_refuses(capsys, tmp_path):
    store = tmp_path / "notes.db"
    run(capsys, store, "add", "something")
    code, _out, err = run(capsys, store, "clear")
    assert code == 2
    assert "--yes" in err
    with Memory(store) as memory:
        assert memory.count() == 1


def test_namespace_works_before_and_after_the_command(capsys, tmp_path):
    """Both positions must reach the same store; SUPPRESS is what allows it."""
    store = tmp_path / "notes.db"
    run(capsys, store, "--namespace", "ada", "add", "ada's note")
    run(capsys, store, "add", "alan's note", "--namespace", "alan")

    code, out, _ = run(capsys, store, "--namespace", "ada", "count")
    assert out.strip() == "1"
    code, out, _ = run(capsys, store, "count", "--namespace", "alan")
    assert out.strip() == "1"
    code, out, _ = run(capsys, store, "count")
    assert out.strip() == "0"


def test_output_writes_utf8_and_says_where(capsys, tmp_path):
    store = tmp_path / "notes.db"
    target = tmp_path / "result.json"
    run(capsys, store, "add", ACCENTED, "--source", "guide.md")
    code, out, _ = run(capsys, store, "search", "panneaux", "--json",
                       "--output", str(target))
    assert code == 0
    assert f"wrote {target}" in out
    written = target.read_text(encoding="utf-8")
    assert ACCENTED in written
    assert json.loads(written)["count"] == 1


# --------------------------------------------------------------- bad arguments


def test_a_bad_json_option_names_the_option(capsys, tmp_path):
    store = tmp_path / "notes.db"
    code, _out, err = run(capsys, store, "search", "solar", "--where", "not json")
    assert code == 2
    assert "--where" in err

    code, _out, err = run(capsys, store, "add", "x", "--metadata", "[1, 2]")
    assert code == 2
    assert "--metadata" in err


def test_parse_json_object_accepts_none_and_rejects_non_objects():
    assert parse_json_object(None, "--where") is None
    assert parse_json_object("", "--where") is None
    assert parse_json_object('{"a": 1}', "--where") == {"a": 1}
    with pytest.raises(ValueError, match="--where"):
        parse_json_object("3", "--where")


def test_get_of_an_unknown_id_exits_two(capsys, tmp_path):
    store = tmp_path / "notes.db"
    code, _out, err = run(capsys, store, "get", "nope")
    assert code == 2
    assert "nope" in err


def test_a_bad_timestamp_is_reported_not_raised(capsys, tmp_path):
    store = tmp_path / "notes.db"
    code, _out, err = run(capsys, store, "add", "x", "--timestamp", "last tuesday")
    assert code == 2
    assert "ISO 8601" in err


# ------------------------------------------------------- the encoding guarantee


_SCRIPT = """
import sys
from document_memory.cli import main
sys.exit(main(sys.argv[1:]))
"""


def _child(tmp_path, *argv, encoding="cp1252"):
    """Run the CLI in a subprocess whose stdout claims a legacy code page."""
    import os

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = encoding
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, *[str(arg) for arg in argv]],
        capture_output=True,
        env=environment,
        cwd=str(tmp_path),
        timeout=120,
    )


def test_non_ascii_output_survives_a_legacy_code_page(tmp_path):
    """The convention this package exists under: no UnicodeEncodeError, ever.

    ``PYTHONIOENCODING=cp1252`` reproduces a Windows console or a captured pipe
    that cannot represent Japanese.  Without ``make_utf8`` this raises
    ``UnicodeEncodeError`` and exits non-zero; with it the bytes come out UTF-8.
    """
    store = tmp_path / "notes.db"
    added = _child(tmp_path, store, "add", JAPANESE, "--source", "案内.md")
    assert added.returncode == 0, added.stderr.decode("utf-8", "replace")

    # Every one of these puts the stored Japanese back on stdout.
    for argv in (
        (store, "search", "電気"),
        (store, "search", "電気", "--json"),
        (store, "search", "電気", "--text"),
        (store, "recent"),
        (store, "context", "電気", "--budget", "50"),
    ):
        done = _child(tmp_path, *argv)
        message = done.stderr.decode("utf-8", "replace")
        assert "UnicodeEncodeError" not in message, message
        assert done.returncode == 0, message
        assert JAPANESE in done.stdout.decode("utf-8", "replace")

    # The bare store summary carries no Japanese of its own, but the path it
    # prints might, so it must still not crash.
    done = _child(tmp_path, store)
    assert "UnicodeEncodeError" not in done.stderr.decode("utf-8", "replace")
    assert done.returncode == 0


def test_accented_text_survives_a_legacy_code_page(tmp_path):
    store = tmp_path / "notes.db"
    added = _child(tmp_path, store, "add", ACCENTED, "--source", "guide-français.md")
    assert added.returncode == 0, added.stderr.decode("utf-8", "replace")

    done = _child(tmp_path, store, "search", "lumière", "--json")
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    text = done.stdout.decode("utf-8", "replace")
    assert ACCENTED in text
    assert "guide-français.md" in text
    assert json.loads(text)["count"] == 1


def test_summary_text_stays_plain_ascii_punctuation(tmp_path):
    """No arrows, bullets or box characters in the human-readable output."""
    with Memory() as memory:
        memory.add(FRENCH, source="guide.md", metadata={"project": "roof"})
        memory.remember("user", "and in winter?")
        pieces = [
            memory.summary(),
            memory.search("solaires").summary(),
            memory.recent().summary(),
            memory.history().summary(),
            memory.search("solaires")[0].why(),
        ]
    forbidden = set("→←↔·•►▶□■▪◦–—“”‘’…")
    for piece in pieces:
        assert not (set(piece) & forbidden), f"non-ASCII punctuation in: {piece!r}"
