"""assess_batch, and the command line - including a pipe that cannot print Unicode."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import _synthetic as S
import document_quality
from document_quality import cli


@pytest.fixture(scope="module")
def batch(page, blank, photo, tmp_path_factory):
    folder = tmp_path_factory.mktemp("batch")
    broken = folder / "not_really.png"
    broken.write_text("this is not an image", encoding="utf-8")
    return document_quality.assess_batch(
        [page, S.rotate(page, 3.0), blank, photo, str(folder / "missing.png"), str(broken)],
        dpi=300,
    )


def test_batch_sorts_pages_and_records_unreadable_ones(batch):
    assert len(batch) == 4
    assert len(batch.failures) == 2
    assert all(item["error"] for item in batch.failures)
    assert len(batch.ready) == 1
    not_ready = batch.not_ready
    assert len(not_ready) == 3
    assert [item.score for item in not_ready] == sorted(item.score for item in not_ready)
    assert [item.kind for item in batch.blank] == ["blank"]
    assert [item.kind for item in batch.photographs] == ["photograph"]
    assert batch.kind_counts() == {"document": 2, "blank": 1, "photograph": 1}
    assert batch.issue_counts()["skew"] == 1


def test_batch_summary_and_dict(batch):
    text = batch.summary()
    assert text.splitlines()[0] == "4 page(s) assessed: 1 ready to OCR, 3 not."
    assert "2 file(s) could not be read." in text
    assert "deskew by 3.0 degrees clockwise" in text
    data = json.loads(json.dumps(batch.to_dict(), allow_nan=False))
    assert data["pages"] == 4 and data["ready"] == 1 and data["not_ready"] == 3
    rows = batch.rows()
    assert len(rows) == 4 and set(rows[0]) >= {"source", "kind", "ocr_ready", "fix"}
    assert json.loads(batch.to_json())["kind_counts"]["blank"] == 1
    assert batch[0].kind == "document"
    assert [item.kind for item in batch] == ["document", "document", "blank", "photograph"]


def test_batch_edge_cases(page):
    empty = document_quality.assess_batch([])
    assert empty.summary() == "No pages were assessed."
    assert empty.not_ready == [] and empty.mean_score() is None
    with pytest.raises(TypeError):
        document_quality.assess_batch(page)
    with pytest.raises(TypeError):
        document_quality.assess_batch("scan.png")
    single = document_quality.assess_batch(iter([page]))
    assert len(single) == 1


@pytest.fixture(scope="module")
def files(page, tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli")
    good = folder / "good.png"
    skewed = folder / "crooked_reçu_文書.png"
    Image.fromarray(page).save(good, dpi=(300, 300))
    Image.fromarray(S.rotate(page, 3.0)).save(skewed, dpi=(300, 300))
    empty = tmp_path_factory.mktemp("empty")
    return {"good": str(good), "skewed": str(skewed), "folder": str(folder),
            "empty": str(empty)}


def test_cli_ready_page_exits_zero(files, capsys):
    assert cli.main([files["good"]]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "ready to OCR" in out.splitlines()[0]


def test_cli_folder_with_a_problem_exits_one(files, capsys):
    assert cli.main([files["folder"], "--no-orientation"]) == cli.EXIT_NOT_READY
    out = capsys.readouterr().out
    assert out.startswith("2 page(s) assessed: 1 ready to OCR, 1 not.")
    assert "文書" in out


def test_cli_only_problems_and_json(files, capsys):
    assert cli.main([files["folder"], "--only-problems"]) == cli.EXIT_NOT_READY
    assert "deskew by 3.0 degrees clockwise" in capsys.readouterr().out
    assert cli.main([files["good"], "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["pages"] == 1 and data["reports"][0]["ocr_ready"] is True


def test_cli_output_file_is_utf8_json(files, tmp_path, capsys):
    target = tmp_path / "report.json"
    code = cli.main([files["skewed"], "--output", str(target), "--quiet"])
    assert code == cli.EXIT_NOT_READY
    assert capsys.readouterr().out == ""
    raw = target.read_bytes().decode("utf-8")
    assert "文書" in raw                     # written as text, not \\u escapes
    assert json.loads(raw)["reports"][0]["issues"][0]["kind"] == "skew"


def test_cli_errors_and_listings(files, capsys):
    assert cli.main([os.path.join(files["folder"], "nope.png")]) == cli.EXIT_UNREADABLE
    assert cli.main([files["empty"]]) == cli.EXIT_UNREADABLE
    capsys.readouterr()
    assert cli.main(["--list-thresholds"]) == cli.EXIT_OK
    assert "target_dpi" in capsys.readouterr().out
    assert cli.main([]) == cli.EXIT_OK
    assert "usage: document-quality" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    with pytest.raises(SystemExit):
        cli.main([files["good"], "--threshold", "target_dpi"])
    assert cli.main([files["good"], "--threshold", "target_dpi=600",
                     "--dpi", "300"]) in (cli.EXIT_OK, cli.EXIT_NOT_READY)
    assert cli.parse_overrides(["limit_dpi = 150"]) == {"limit_dpi": 150.0}


def test_cli_survives_a_console_that_cannot_encode_the_file_name(files):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"          # cannot encode the CJK file name
    for extra in ([], ["--json"]):
        done = subprocess.run(
            [sys.executable, "-m", "document_quality", files["skewed"],
             "--no-orientation"] + extra,
            capture_output=True, env=env, timeout=120,
        )
        assert done.returncode == cli.EXIT_NOT_READY, done.stderr.decode("utf-8", "replace")
        assert b"Traceback" not in done.stderr
        out = done.stdout.decode("utf-8")
        assert "文書" in out


def test_cli_json_with_output_file_keeps_stdout_one_json_document(files, tmp_path, capsys):
    target = tmp_path / "out.json"
    code = cli.main([files["good"], "--json", "--output", str(target)])
    assert code == cli.EXIT_OK
    captured = capsys.readouterr()
    assert json.loads(captured.out)["pages"] == 1          # nothing after the JSON
    assert "Wrote" in captured.err
    assert json.loads(target.read_text(encoding="utf-8"))["pages"] == 1


@pytest.mark.parametrize("argv", [
    ["{good}", "--threshold", "bogus=1"],
    ["--list-thresholds", "--threshold", "bogus=1"],
    ["{good}", "--threshold", "target_dpi=high"],
    ["{good}", "--dpi", "0"],
    ["{good}", "--dpi", "-300"],
    ["{good}", "--dpi", "abc"],
])
def test_cli_bad_options_are_usage_errors(files, capsys, argv):
    # Exit 1 means "a page is not ready"; a bad option must never look like it.
    with pytest.raises(SystemExit) as stop:
        cli.main([item.format(**files) for item in argv])
    assert stop.value.code == 2
    assert "error:" in capsys.readouterr().err
