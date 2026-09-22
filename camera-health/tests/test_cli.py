"""The command line front end, including the encodings it has to survive."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from camera_health.cli import BAD_USAGE, FAULTY, OK, build_parser, main
from conftest import live_frame, obstructed, save_png, scene


def test_help_renders(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    assert "camera-health" in text
    assert "--reference" in text
    assert "exit codes" in text


def test_version_renders():
    with pytest.raises(SystemExit):
        main(["--version"])


def test_a_healthy_frame_exits_zero(tmp_path, base, capsys):
    path = save_png(tmp_path / "ok.png", base)

    assert main([path]) == OK
    assert "camera health: ok" in capsys.readouterr().out


def test_a_failed_frame_exits_one(tmp_path, base, capsys):
    path = save_png(tmp_path / "dark.png", np.zeros_like(base))

    assert main([path]) == FAULTY
    assert "NOT OK" in capsys.readouterr().out


def test_json_output_is_parseable(tmp_path, base, capsys):
    path = save_png(tmp_path / "ok.png", base)

    main([path, "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["size"] == {"width": 320, "height": 240}


def test_output_is_written_as_utf8(tmp_path, base):
    path = save_png(tmp_path / "ok.png", base)
    target = tmp_path / "report.json"

    main([path, "--json", "--output", str(target), "--quiet"])

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is True


def test_quiet_prints_nothing(tmp_path, base, capsys):
    path = save_png(tmp_path / "ok.png", base)

    assert main([path, "--quiet"]) == OK
    assert capsys.readouterr().out == ""


def test_a_run_of_frames_uses_the_stream_report(tmp_path, base, capsys):
    paths = [
        save_png(tmp_path / "f{}.png".format(tick), live_frame(base, tick)) for tick in range(3)
    ]

    assert main(paths) == OK
    assert "camera health over 3 frame(s)" in capsys.readouterr().out


def test_a_directory_is_read_as_a_run(tmp_path, base, capsys):
    for tick in range(3):
        save_png(tmp_path / "f{}.png".format(tick), live_frame(base, tick))

    assert main([str(tmp_path)]) == OK
    assert "over 3 frame(s)" in capsys.readouterr().out


def test_reference_and_previous_are_wired_in(tmp_path, base, other_scene, capsys):
    frame = save_png(tmp_path / "now.png", other_scene)
    reference = save_png(tmp_path / "ref.png", base)

    assert main([frame, "--reference", reference]) == FAULTY
    assert "tampering" in capsys.readouterr().out

    previous = save_png(tmp_path / "prev.png", other_scene)
    assert main([frame, "--previous", previous, "--freeze-frames", "2"]) == FAULTY
    assert "frozen" in capsys.readouterr().out


def test_previous_with_a_run_is_refused(tmp_path, base, capsys):
    paths = [save_png(tmp_path / "f{}.png".format(i), live_frame(base, i)) for i in range(2)]

    assert main(paths + ["--previous", paths[0]]) == BAD_USAGE
    assert "applies to a single frame" in capsys.readouterr().err


def test_show_thresholds_prints_json(capsys):
    assert main(["--show-thresholds"]) == OK

    payload = json.loads(capsys.readouterr().out)
    assert payload["dark_mean"] == 35.0
    assert payload["tile_grid"] == 16


def test_set_overrides_a_threshold(tmp_path, base, capsys):
    dim = save_png(tmp_path / "dim.png", (base.astype(np.float32) * 0.22).astype(np.uint8))

    main([dim])
    assert "darkness" in capsys.readouterr().out

    main([dim, "--set", "dark_mean=5"])
    assert "darkness" not in capsys.readouterr().out


def test_a_bad_set_is_rejected_by_the_parser():
    parser = build_parser()
    for bad in ("dark_mean", "nonsense=1", "dark_mean=abc"):
        with pytest.raises(SystemExit):
            parser.parse_args(["x.png", "--set", bad])


def test_no_input_is_a_usage_error(capsys):
    assert main([]) == BAD_USAGE
    assert "give at least one image" in capsys.readouterr().err


def test_a_missing_file_is_reported_not_traced(tmp_path, capsys):
    assert main([str(tmp_path / "gone.png")]) == BAD_USAGE
    assert "camera-health:" in capsys.readouterr().err


def test_an_unwritable_output_path_is_reported(tmp_path, base, capsys):
    path = save_png(tmp_path / "ok.png", base)

    assert main([path, "--output", str(tmp_path / "no-such-dir" / "r.json")]) == BAD_USAGE
    assert "cannot write" in capsys.readouterr().err


def test_non_ascii_output_survives_a_pipe_on_any_console(tmp_path, base):
    """The encoding case: a Cyrillic file name printed into a captured pipe.

    Run as a real subprocess with an ASCII stdout encoding forced, because that
    is the shape a Windows console, a CI log or `... | tee` actually has.

    The environment is inherited and only PYTHONIOENCODING is overridden: a
    scrubbed environment would stop the interpreter booting on Windows, which
    would test nothing about this package.
    """
    name = "камера-廊下-2.png"
    path = save_png(tmp_path / name, obstructed(base, share=0.6))

    finished = subprocess.run(
        [sys.executable, "-m", "camera_health.cli", path, "--json"],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "ascii"},
    )

    assert finished.returncode == FAULTY, finished.stderr.decode("utf-8", "replace")
    payload = json.loads(finished.stdout.decode("utf-8"))
    assert payload["source"] == name
    assert payload["faults"][0]["kind"] == "obstruction"


def test_the_installed_console_script_runs():
    finished = subprocess.run(
        [sys.executable, "-m", "camera_health.cli", "--help"], capture_output=True
    )

    assert finished.returncode == 0
    assert b"camera-health" in finished.stdout
