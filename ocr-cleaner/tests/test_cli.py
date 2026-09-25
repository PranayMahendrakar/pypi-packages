"""The command line: what it prints, what it writes, and what it exits with."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from ocr_cleaner import cli

from conftest import blank_page, rotate, text_page

#: A filename that no Latin-1 console can encode, which is the point of it.
AWKWARD_NAME = "scan-ページ-éç-क.png"


@pytest.fixture
def scans(tmp_path):
    """A directory holding two scans, one of them awkwardly named."""
    folder = tmp_path / "scans"
    folder.mkdir()
    Image.fromarray(rotate(text_page(), -2.0), mode="L").save(folder / "a.png")
    Image.fromarray(text_page(), mode="L").save(folder / AWKWARD_NAME)
    return folder


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "ocr-cleaner" in capsys.readouterr().out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_no_arguments_prints_help_and_fails(capsys):
    assert cli.main([]) == cli.EXIT_ERROR
    assert "usage" in capsys.readouterr().out


def test_one_page_prints_a_report(capsys, page_file):
    assert cli.main([page_file]) == 0
    out = capsys.readouterr().out
    assert "ocr-cleaner:" in out
    assert "steps" in out


def test_nothing_is_written_without_being_asked(tmp_path, page_file):
    before = sorted(os.listdir(tmp_path))
    assert cli.main([page_file]) == 0
    assert sorted(os.listdir(tmp_path)) == before


def test_output_writes_one_page(tmp_path, capsys, page_file):
    destination = tmp_path / "clean.png"
    assert cli.main([page_file, "--output", str(destination)]) == 0
    assert destination.exists()
    assert "written to" in capsys.readouterr().out


def test_out_dir_and_suffix(tmp_path, capsys, scans):
    out = tmp_path / "cleaned"
    assert cli.main([str(scans), "--out-dir", str(out), "--suffix", "-clean"]) == 0
    written = sorted(p.name for p in out.iterdir())
    assert "a-clean.png" in written
    assert any("-clean.png" in name for name in written)
    assert len(written) == 2


def test_a_dash_leading_suffix_is_not_read_as_an_option(scans, tmp_path):
    parser = cli.build_parser()
    glued = cli.glue_dash_values([str(scans), "--suffix", "-clean"], parser)
    assert glued[-1] == "--suffix=-clean"


def test_a_real_option_after_suffix_is_left_alone():
    parser = cli.build_parser()
    glued = cli.glue_dash_values(["--suffix", "--quiet"], parser)
    assert glued == ["--suffix", "--quiet"]


def test_output_refuses_more_than_one_page(capsys, scans):
    assert cli.main([str(scans), "--output", "one.png"]) == cli.EXIT_ERROR
    assert "--out-dir" in capsys.readouterr().err


def test_json_output_parses(capsys, page_file):
    assert cli.main([page_file, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["page_kind"] == "document"
    assert len(payload["steps"]) == 6


def test_json_for_several_pages_is_a_list(capsys, scans):
    assert cli.main([str(scans), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 2


def test_quiet_is_one_line_per_page(capsys, scans):
    assert cli.main([str(scans), "--quiet"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert "deg" in lines[0]


def test_recursive_finds_nested_pages(tmp_path, capsys, page_file):
    nested = tmp_path / "top" / "inner"
    nested.mkdir(parents=True)
    Image.fromarray(text_page(), mode="L").save(nested / "deep.png")
    assert cli.main([str(tmp_path / "top"), "--recursive", "--quiet"]) == 0
    assert "deep.png" in capsys.readouterr().out


def test_missing_path_fails(capsys, tmp_path):
    assert cli.main([str(tmp_path / "nope.png")]) == cli.EXIT_ERROR
    assert "no such file" in capsys.readouterr().err


def test_an_empty_directory_fails(capsys, tmp_path):
    (tmp_path / "empty").mkdir()
    assert cli.main([str(tmp_path / "empty")]) == cli.EXIT_ERROR
    assert "no images found" in capsys.readouterr().err


def test_upscale_needs_dpi(capsys, page_file):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([page_file, "--upscale-to-dpi", "300"])
    assert exit_info.value.code == cli.EXIT_ERROR
    assert "needs --dpi" in capsys.readouterr().err


def test_a_negative_dpi_is_refused(capsys, page_file):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([page_file, "--dpi", "-5"])
    assert exit_info.value.code == cli.EXIT_ERROR
    assert "positive" in capsys.readouterr().err


def test_an_unknown_threshold_is_refused(capsys, page_file):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([page_file, "--threshold", "sauvola"])
    assert exit_info.value.code == cli.EXIT_ERROR


def test_require_document_flags_a_blank_page(tmp_path, capsys):
    blank = tmp_path / "blank.png"
    Image.fromarray(blank_page(), mode="L").save(blank)
    assert cli.main([str(blank), "--require-document"]) == cli.EXIT_NOT_A_DOCUMENT
    assert "not documents" in capsys.readouterr().err


def test_require_document_passes_a_real_page(page_file):
    assert cli.main([page_file, "--require-document"]) == 0


def test_the_steps_can_be_turned_off_from_the_command_line(capsys, page_file):
    assert cli.main([page_file, "--no-deskew", "--no-denoise", "--no-border",
                     "--threshold", "none", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] == []
    assert payload["binary"] is False


def test_report_is_written_as_utf8(tmp_path, capsys, scans):
    report = tmp_path / "report.txt"
    assert cli.main([str(scans), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "ページ" in text


def test_a_non_image_file_is_reported_not_crashed(tmp_path, capsys):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not a PNG")
    assert cli.main([str(broken)]) == cli.EXIT_ERROR
    assert "ocr-cleaner:" in capsys.readouterr().err


def test_console_safety_helper_is_idempotent():
    cli._make_console_safe()
    cli._make_console_safe()


def test_a_non_ascii_page_name_survives_a_pipe(tmp_path):
    """The encoding promise: piping output with non-ASCII in it must not crash.

    Run in a real subprocess with a redirected stdout and a Latin-1 preferred
    encoding, because that is the arrangement that raises UnicodeEncodeError
    when the CLI has not made its streams tolerant.
    """
    awkward = tmp_path / AWKWARD_NAME
    Image.fromarray(text_page(), mode="L").save(awkward)
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "latin-1"
    environment.pop("PYTHONUTF8", None)
    environment["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    finished = subprocess.run(
        [sys.executable, "-m", "ocr_cleaner.cli", str(awkward), "--quiet"],
        capture_output=True, env=environment,
    )
    assert finished.returncode == 0, finished.stderr.decode("utf-8", "replace")
    assert b"Traceback" not in finished.stderr
    assert b"UnicodeEncodeError" not in finished.stderr
    assert finished.stdout.strip()


def test_the_installed_console_script_runs():
    finished = subprocess.run(
        [sys.executable, "-m", "ocr_cleaner.cli", "--help"], capture_output=True
    )
    assert finished.returncode == 0
    assert b"ocr-cleaner" in finished.stdout


def test_destination_for_builds_the_name():
    assert cli.destination_for(os.path.join("a", "b.png"), "out", "-clean") == os.path.join(
        "out", "b-clean.png"
    )


def test_render_handles_an_empty_run():
    assert cli.render([], as_json=False, quiet=True) == ""
