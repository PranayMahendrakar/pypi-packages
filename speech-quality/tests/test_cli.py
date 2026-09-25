"""The command line: exit codes, output shapes, and non-ASCII under a pipe."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from conftest import RATE, silence, speech_like, write_wav
from speech_quality.cli import BAD_USAGE, NOT_USABLE, USABLE, build_parser, main


def test_help_renders(capsys):
    """--help is a contract: it must not raise and must name the tool."""
    with pytest.raises(SystemExit) as caught:
        main(["--help"])

    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "speech-quality" in out
    assert "exit codes" in out


def test_version_prints_the_version(capsys):
    """--version answers without touching any audio."""
    with pytest.raises(SystemExit):
        main(["--version"])

    assert "0.1.0" in capsys.readouterr().out


def test_no_arguments_is_a_usage_error(capsys):
    """Naming no file is a mistake worth an exit code, not an empty report."""
    code = main([])

    assert code == BAD_USAGE
    assert "at least one" in capsys.readouterr().err


def test_a_clean_take_exits_zero(wav_dir, capsys):
    """The everyday success path prints the report and exits 0."""
    path = write_wav(wav_dir / "clean.wav", speech_like(seconds=1.0))

    code = main([path])

    assert code == USABLE
    assert "usable" in capsys.readouterr().out


def test_a_dead_take_exits_one(wav_dir, capsys):
    """A file of silence is read fine and reported as not usable."""
    path = write_wav(wav_dir / "dead.wav", silence(seconds=1.0))

    code = main([path])

    assert code == NOT_USABLE
    assert "NOT usable" in capsys.readouterr().out


def test_a_missing_file_exits_two(wav_dir, capsys):
    """Nothing to read is a different failure from something bad to read."""
    code = main([str(wav_dir / "gone.wav")])

    assert code == BAD_USAGE
    assert "no such file" in capsys.readouterr().err


def test_a_non_wav_file_exits_two_with_advice(tmp_path, capsys):
    """The CLI passes the library's explanation straight through."""
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"ID3")

    code = main([str(path)])

    assert code == BAD_USAGE
    assert "WAV" in capsys.readouterr().err


def test_json_output_parses(wav_dir, capsys):
    """--json is machine-readable, which is the whole point of it."""
    path = write_wav(wav_dir / "clean.wav", speech_like(seconds=1.0))

    main([path, "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["grade"]
    assert "level" in payload["metrics"]


def test_several_files_report_as_a_batch(wav_dir, capsys):
    """More than one file switches to the batch summary and lists the worst."""
    good = write_wav(wav_dir / "good.wav", speech_like(seconds=1.0))
    dead = write_wav(wav_dir / "dead.wav", silence(seconds=1.0))

    code = main([good, dead, "--worst", "1"])

    out = capsys.readouterr().out
    assert code == NOT_USABLE
    assert "2 recording(s)" in out
    assert "dead.wav" in out


def test_a_batch_where_nothing_opens_exits_two(wav_dir, capsys):
    """Two unreadable files is a usage failure, not a quality verdict."""
    code = main([str(wav_dir / "a.wav"), str(wav_dir / "b.wav")])

    assert code == BAD_USAGE
    assert "could not be read (2)" in capsys.readouterr().out


def test_output_file_is_written_as_utf8(wav_dir, tmp_path, capsys):
    """--output writes UTF-8 whatever the console encoding is."""
    path = write_wav(wav_dir / "録音.wav", speech_like(seconds=0.5))
    target = tmp_path / "report.txt"

    main([path, "--output", str(target), "--quiet"])

    assert capsys.readouterr().out == ""
    assert "録音.wav" in target.read_text(encoding="utf-8")


def test_output_to_an_impossible_path_exits_two(wav_dir, tmp_path, capsys):
    """A write that cannot happen is reported rather than swallowed."""
    path = write_wav(wav_dir / "clean.wav", speech_like(seconds=0.5))

    code = main([path, "--output", str(tmp_path / "no-such-dir" / "r.txt")])

    assert code == BAD_USAGE
    assert "cannot write" in capsys.readouterr().err


def test_threshold_flags_change_the_verdict(wav_dir, capsys):
    """Demanding 40 dB of signal-to-noise turns a pass into a fail."""
    path = write_wav(wav_dir / "clean.wav", speech_like(seconds=1.0))

    assert main([path, "--quiet"]) == USABLE
    assert main([path, "--min-snr", "40", "--usable-score", "95", "--quiet"]) == NOT_USABLE


def test_parser_exposes_every_threshold_flag():
    """The documented flags are really there, spelled the way the README spells them."""
    parser = build_parser()

    args = parser.parse_args(["x.wav", "--target-dbfs", "-18", "--min-bandwidth", "4000"])

    assert args.target_rms_dbfs == pytest.approx(-18.0)
    assert args.min_bandwidth_hz == pytest.approx(4000.0)


def test_non_ascii_output_survives_a_pipe(wav_dir):
    """The real encoding test: a subprocess with a captured, non-UTF-8-friendly stdout.

    This is the failure that only appears once output is piped, which is exactly
    when nobody is watching. The file name is Japanese, Cyrillic and an emoji.
    """
    name = "録音-звук-\U0001f3a4.wav"
    path = write_wav(wav_dir / name, speech_like(seconds=0.5))

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "ascii"
    finished = subprocess.run(
        [sys.executable, "-m", "speech_quality.cli", path],
        capture_output=True,
        env=environment,
    )

    assert finished.returncode in (USABLE, NOT_USABLE), finished.stderr.decode(
        "utf-8", "replace"
    )
    assert b"Traceback" not in finished.stderr
    assert b"UnicodeEncodeError" not in finished.stderr
    assert b"usable" in finished.stdout


def test_the_module_runs_as_a_script():
    """python -m speech_quality.cli is the same entry point the console script uses."""
    finished = subprocess.run(
        [sys.executable, "-m", "speech_quality.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert finished.returncode == 0
    assert "speech-quality" in finished.stdout
