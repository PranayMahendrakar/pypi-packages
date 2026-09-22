"""The command line interface, including the encoding safety net."""
import io
import json
import subprocess
import sys

import pytest

from multilingual_text import cli

SPANISH = "El rápido zorro marrón salta sobre el perro perezoso"
RUSSIAN = "Привет, мир! Это тест."


def run(argv, capsys):
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_help_builds_and_mentions_the_modes():
    text = cli.build_parser().format_help()
    assert "multilingual-text" in text
    assert "--mode" in text
    assert "transliterate" in text


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


def test_detect_is_the_default_mode(capsys):
    code, out, _ = run([SPANISH], capsys)
    assert code == 0
    assert "es (Spanish)" in out
    assert out.startswith("multilingual-text:")


def test_detect_output_is_plain_ascii_even_for_non_latin_input(capsys):
    code, out, _ = run(["नमस्ते दुनिया यह एक परीक्षण है"], capsys)
    assert code == 0
    assert out.isascii()
    assert "hi (Hindi)" in out


def test_json_output_round_trips(capsys):
    code, out, _ = run([SPANISH, "--json", "--top", "3"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["detect"]["language"] == "es"
    assert len(payload["detect"]["alternatives"]) == 2


def test_normalize_mode(capsys):
    code, out, _ = run(["  “x”  —  y  ", "--mode", "normalize"], capsys)
    assert code == 0
    assert 'normalized: "x" - y' in out


def test_normalize_mode_options(capsys):
    code, out, _ = run(
        ["Crème, Brûlée!", "--mode", "normalize", "--case", "fold",
         "--diacritics", "--punctuation"],
        capsys,
    )
    assert code == 0
    assert "normalized: creme brulee" in out


def test_transliterate_mode(capsys):
    code, out, _ = run([RUSSIAN, "--mode", "transliterate"], capsys)
    assert code == 0
    assert "Privet, mir!" in out
    assert "Cyrillic romanised" in out


def test_transliterate_mode_ascii(capsys):
    code, out, _ = run(["भाषा", "--mode", "transliterate", "--to", "ascii"], capsys)
    assert code == 0
    assert "transliterated: bhasa" in out


def test_transliterate_says_when_it_cannot_help(capsys):
    code, out, _ = run(["مرحبا", "--mode", "transliterate"], capsys)
    assert code == 0
    assert "not covered" in out


def test_all_mode_reports_everything(capsys):
    code, out, _ = run([RUSSIAN, "--mode", "all"], capsys)
    assert code == 0
    assert "ru (Russian)" in out
    assert "normalized:" in out
    assert "transliterated:" in out
    assert "script: Cyrillic" in out
    assert "right-to-left: False" in out


def test_stdin_input(capsys, monkeypatch):
    data = io.BytesIO(SPANISH.encode("utf-8"))
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(data, encoding="utf-8"))
    code, out, _ = run(["-"], capsys)
    assert code == 0
    assert "es (Spanish)" in out


def test_file_input(tmp_path, capsys):
    path = tmp_path / "sample.txt"
    path.write_text(RUSSIAN, encoding="utf-8")
    code, out, _ = run(["--file", str(path)], capsys)
    assert code == 0
    assert "ru (Russian)" in out


def test_missing_input_is_a_clear_error(capsys):
    code, _, err = run([], capsys)
    assert code == 1
    assert "nothing to work on" in err


def test_both_text_and_file_is_a_clear_error(tmp_path, capsys):
    path = tmp_path / "sample.txt"
    path.write_text("hi", encoding="utf-8")
    code, _, err = run(["some text", "--file", str(path)], capsys)
    assert code == 1
    assert "not both" in err


def test_missing_file_is_a_clear_error(capsys):
    code, _, err = run(["--file", "definitely-not-here.txt"], capsys)
    assert code == 1
    assert "multilingual-text: error:" in err


def test_output_file_json(tmp_path, capsys):
    path = tmp_path / "out.json"
    code, out, _ = run([SPANISH, "--output", str(path)], capsys)
    assert code == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["detect"]["language"] == "es"
    assert "wrote" in out


def test_output_file_text(tmp_path, capsys):
    path = tmp_path / "out.txt"
    code, _, _ = run([RUSSIAN, "--mode", "all", "--output", str(path)], capsys)
    assert code == 0
    written = path.read_text(encoding="utf-8")
    assert "ru (Russian)" in written
    assert "Privet" in written


def test_subprocess_with_non_ascii_output_does_not_crash_when_piped():
    """A pipe on a legacy codepage must not raise UnicodeEncodeError."""
    completed = subprocess.run(
        [sys.executable, "-m", "multilingual_text", "Привет, мир!", "--mode", "all"],
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    text = completed.stdout.decode("utf-8", "replace")
    assert "ru (Russian)" in text
    assert "Privet, mir!" in text


def test_subprocess_help_runs():
    completed = subprocess.run(
        [sys.executable, "-m", "multilingual_text", "--help"], capture_output=True
    )
    assert completed.returncode == 0
    assert b"--mode" in completed.stdout
