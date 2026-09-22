"""The command line interface, including output that is piped rather than shown."""
import json
import subprocess
import sys

import pytest

from rag_chunker.cli import build_parser, main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "rag-chunker" in out
    assert "semantic" in out and "structural" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_parser_defaults_match_the_library():
    """Derived from the library signature, never hard-coded: the CLI once defaulted to
    semantic while the library defaulted to recursive, so the same document chunked
    differently depending on how you called it, and a hard-coded expectation here was
    what let the two drift apart."""
    import inspect

    import rag_chunker

    params = inspect.signature(rag_chunker.chunk).parameters
    args = build_parser().parse_args(["doc.md"])
    assert args.size == params["size"].default
    assert args.overlap == params["overlap"].default
    assert args.method == params["method"].default


def test_summary_run_on_a_markdown_file(md_file, capsys):
    code = main([str(md_file), "--size", "40", "--overlap", "8"])
    out = capsys.readouterr().out
    assert code == 0
    assert "rag-chunker:" in out
    assert "reproduce the source exactly: yes" in out
    assert "handbook.md" in out


def test_json_output_is_parseable(md_file, capsys):
    code = main([str(md_file), "--size", "40", "--overlap", "8", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_chunks"] >= 1
    assert payload["reproduces_source"] is True


def test_text_output_shows_every_chunk(md_file, capsys):
    code = main([str(md_file), "--size", "40", "--overlap", "8", "--text"])
    assert code == 0
    out = capsys.readouterr().out
    assert "----- chunk 0 -----" in out
    assert "----- chunk 1 -----" in out


def test_metadata_flag(md_file, capsys):
    code = main([str(md_file), "--size", "40", "--overlap", "8",
                 "--metadata", '{"doc": "handbook"}', "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks"][0]["metadata"] == {"doc": "handbook"}


def test_bad_metadata_is_an_error_not_a_traceback(md_file, capsys):
    code = main([str(md_file), "--metadata", "[1, 2]"])
    assert code == 1
    assert "rag-chunker: error:" in capsys.readouterr().err


def test_missing_file_is_an_error_not_a_traceback(tmp_path, capsys):
    code = main([str(tmp_path / "absent.md")])
    assert code == 1
    assert "does not exist" in capsys.readouterr().err


def test_bad_size_is_an_error_not_a_traceback(md_file, capsys):
    code = main([str(md_file), "--size", "0"])
    assert code == 1
    assert "size must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".txt"])
def test_output_file(md_file, tmp_path, suffix, capsys):
    out_path = tmp_path / ("chunks" + suffix)
    code = main([str(md_file), "--size", "40", "--overlap", "8", "--output", str(out_path)])
    assert code == 0
    assert "wrote" in capsys.readouterr().out
    written = out_path.read_text(encoding="utf-8")
    assert written.strip()
    if suffix == ".json":
        assert json.loads(written)["n_chunks"] >= 1
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in written.splitlines() if line.strip()]
        assert rows[0]["index"] == 0


def test_stdin(monkeypatch, capsys):
    import io

    payload = io.BytesIO("One two three. Four five six. Seven eight.".encode("utf-8"))

    class FakeStdin:
        buffer = payload

    monkeypatch.setattr(sys, "stdin", FakeStdin)
    code = main(["-", "--size", "6", "--overlap", "1"])
    assert code == 0
    assert "rag-chunker:" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# encoding safety: the output is piped, not shown on a console
# --------------------------------------------------------------------------- #
NON_ASCII = (
    "# Rapport d'etude\n\n"
    "El informe senala una subida. La reunion se celebro en Malaga.\n\n"
    "日本語の文章です。これは二番目の文。\n\n"
    "Emoji: \U0001f9ea \U0001f4da and a dash — plus quotes “ok”.\n"
)


def _run_piped(args, tmp_path):
    """Run the CLI as a subprocess with stdout captured (a pipe, not a console)."""
    doc = tmp_path / "unicode.md"
    doc.write_text(NON_ASCII, encoding="utf-8")
    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        # A console encoding that cannot represent the document at all.
        "PYTHONIOENCODING": "cp1252",
    }
    return subprocess.run(
        [sys.executable, "-m", "rag_chunker", str(doc)] + args,
        capture_output=True,
        env=env,
    )


@pytest.mark.parametrize("args", [[], ["--json"], ["--text"]])
def test_piped_non_ascii_output_never_raises_unicodeencodeerror(args, tmp_path):
    proc = _run_piped(args + ["--size", "12", "--overlap", "2"], tmp_path)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in proc.stderr
    text = proc.stdout.decode("utf-8", errors="replace")
    assert "日本語" in text or "\\u65e5" in text
