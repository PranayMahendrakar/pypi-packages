"""The command line: help, summary, JSON, output file, exit codes, encodings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import synth
from video_event_detector.cli import BAD_INPUT, EVENTS_FOUND, NOTHING_FOUND, main


def save_frames(folder: Path, frames):
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / "{:04d}.png".format(index))
    return folder


@pytest.fixture(scope="module")
def sudden_dir(tmp_path_factory):
    # 10 frames to learn the scene, then a large block bursts in at frame 30.
    return save_frames(tmp_path_factory.mktemp("cli") / "кадры_カメラ", synth.sudden_sequence(count=65)[20:])


@pytest.fixture(scope="module")
def still_dir(tmp_path_factory):
    return save_frames(tmp_path_factory.mktemp("cli") / "still", synth.noisy(synth.texture(), 45))


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "ffmpeg" in out and "hypothesis" in out


def test_summary_and_exit_code_when_events_found(sudden_dir, capsys):
    code = main([str(sudden_dir), "--fps", "10", "--background-frames", "10"])
    assert code == EVENTS_FOUND
    out = capsys.readouterr().out
    assert "sudden_motion" in out and "3.0" in out


def test_exit_code_zero_on_a_quiet_scene(still_dir, capsys):
    assert main([str(still_dir)]) == NOTHING_FOUND
    assert "0 found" in capsys.readouterr().out


def test_json_and_output_file(sudden_dir, tmp_path, capsys):
    target = tmp_path / "out" / "événements.json"
    code = main([str(sudden_dir), "--json", "--output", str(target), "--background-frames", "10"])
    assert code == EVENTS_FOUND
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(target.read_text(encoding="utf-8"))
    assert printed == written
    assert printed["by_kind"]["sudden_motion"] == 1
    assert "кадры_カメラ" in target.read_text(encoding="utf-8")  # not \u-escaped


def test_unreadable_input_exits_two_cleanly(tmp_path, capsys):
    table = tmp_path / "u.csv"
    table.write_text("name,city\nZoë,Kraków\n李雷,北京\n", encoding="utf-8")
    assert main([str(table)]) == BAD_INPUT
    captured = capsys.readouterr()
    assert "video-event-detector: error:" in captured.err and "Traceback" not in captured.err
    assert main([str(table), "--json"]) == BAD_INPUT
    payload = json.loads(capsys.readouterr().out)
    assert "error" in payload and payload["input"] == [str(table)]
    assert main([str(tmp_path / "missing")]) == BAD_INPUT


def test_bad_option_value_is_an_argparse_error(still_dir, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([str(still_dir), "--fps", "fast"])
    assert exit_info.value.code == 2
    assert "--fps wants a number" in capsys.readouterr().err


def test_piped_non_ascii_output_does_not_crash(sudden_dir):
    # A pipe with a legacy encoding is where UnicodeEncodeError used to appear.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)
    for extra in ([], ["--json"]):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from video_event_detector.cli import main; sys.exit(main())",
             str(sudden_dir), "--background-frames", "10"] + extra,
            capture_output=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == EVENTS_FOUND, result.stderr.decode("utf-8", "replace")
        out = result.stdout.decode("utf-8")
        assert "кадры_カメラ" in out
        assert b"Traceback" not in result.stderr
