"""The audio-anomaly command: help, summaries, JSON, profiles, exit codes, encodings."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from audio_anomaly import cli
from _signals import add_knock, hum, write_pcm


@pytest.fixture()
def wavs(tmp_path):
    healthy = write_pcm(tmp_path / "healthy.wav", hum(seconds=3.0, seed=1))
    knocked = write_pcm(tmp_path / "Pumpe_Prüfung_測試.wav", add_knock(hum(seed=2)))
    return healthy, knocked


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--reference" in out and "--save-profile" in out and "exit status" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_summary_for_a_recording(wavs, capsys):
    assert cli.main([str(wavs[1])]) == 0
    out = capsys.readouterr().out
    assert "1 anomaly" in out and "burst" in out
    assert "Pumpe_Prüfung_測試.wav" in out


def test_json_output_is_valid_and_keeps_unicode(wavs, capsys):
    assert cli.main([str(wavs[1]), "--json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["counts"] == {"burst": 1}
    assert "測試" in out  # ensure_ascii=False


def test_output_file_is_utf8_json(wavs, tmp_path, capsys):
    target = tmp_path / "rapport_é.json"
    assert cli.main([str(wavs[1]), "--output", str(target)]) == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["source"].endswith("Pumpe_Prüfung_測試.wav")
    assert "Full report written" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", [".npy", ".csv"])
def test_save_profile_then_use_it_as_reference(wavs, tmp_path, capsys, suffix):
    target = tmp_path / ("pump" + suffix)
    assert cli.main([str(wavs[0]), "--save-profile", str(target)]) == 0
    assert "Saved the typical spectrum" in capsys.readouterr().out
    assert target.exists()
    assert cli.main([str(wavs[1]), "--reference", str(target), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "reference" and data["counts"] == {"burst": 1}
    assert data["reference"] == str(target)


def test_wav_reference_and_sensitivity(wavs, capsys):
    assert cli.main([str(wavs[1]), "--reference", str(wavs[0]), "--sensitivity", "500"]) == 0
    assert "no anomalies" in capsys.readouterr().out


def test_fail_on_anomaly_exit_code(wavs, capsys):
    assert cli.main([str(wavs[1]), "--fail-on-anomaly"]) == 1
    assert cli.main([str(wavs[0]), "--fail-on-anomaly"]) == 0
    capsys.readouterr()


def test_unreadable_input_exits_2_with_a_message(tmp_path, capsys):
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio", encoding="utf-8")
    assert cli.main([str(bad)]) == 2
    assert "audio-anomaly: error:" in capsys.readouterr().err
    assert cli.main([str(tmp_path / "missing.wav")]) == 2
    assert cli.main([str(bad), "--frame-ms", "-3"]) == 2


def test_piped_output_with_non_ascii_never_crashes(wavs):
    # A pipe with a legacy console encoding is where UnicodeEncodeError used to bite.
    env = dict(os.environ, PYTHONIOENCODING="cp1252" if os.name == "nt" else "ascii")
    for extra in ([], ["--json"]):
        result = subprocess.run(
            [sys.executable, "-m", "audio_anomaly.cli", str(wavs[1])] + extra,
            capture_output=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        text = result.stdout.decode("utf-8")
        assert "Pumpe_Prüfung_測試.wav" in text


def test_profile_csv_has_a_header(wavs, tmp_path, capsys):
    target = tmp_path / "p.csv"
    cli.main([str(wavs[0]), "--save-profile", str(target)])
    capsys.readouterr()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "frequency_hz,level_db,spread_db"
    table = np.loadtxt(target, delimiter=",", skiprows=1)
    assert table.shape == (28, 3)


def test_a_damaged_file_exits_2_even_with_fail_on_anomaly(tmp_path, capsys):
    # Exit status 1 means "anomalies found"; a damaged file must never look like that.
    bad = tmp_path / "corrupt.wav"
    bad.write_bytes(b"RIFF\x10\x00\x00\x00WAVEjunkjunkjunk")
    assert cli.main([str(bad), "--fail-on-anomaly"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("audio-anomaly: error:") and "damaged" in err
    result = subprocess.run(
        [sys.executable, "-m", "audio_anomaly.cli", str(bad), "--fail-on-anomaly"],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert b"Traceback" not in result.stderr
    assert b"audio-anomaly: error:" in result.stderr


def test_an_unwritable_output_path_exits_2(wavs, tmp_path, capsys):
    target = tmp_path / "no_such_folder" / "findings.json"
    assert cli.main([str(wavs[1]), "--output", str(target), "--fail-on-anomaly"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("audio-anomaly: error:") and "could not write" in err


def test_an_unexpected_failure_still_exits_2(wavs, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("something inside broke")

    monkeypatch.setattr(cli, "detect", broken)
    assert cli.main([str(wavs[1]), "--fail-on-anomaly"]) == 2
    err = capsys.readouterr().err
    assert "audio-anomaly: error:" in err and "RuntimeError" in err
