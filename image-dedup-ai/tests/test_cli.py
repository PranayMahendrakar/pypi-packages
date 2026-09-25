"""The command line: summary, JSON, CSV, --near, errors, and UTF-8 safety in pipes."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys

import pytest

from image_dedup_ai import __version__
from image_dedup_ai.cli import main
from conftest import make_photo


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "Fotos été 写真"
    root.mkdir()
    make_photo(1).save(root / "plage.png")
    shutil.copyfile(root / "plage.png", root / "plage copie ☀.png")
    make_photo(1).resize((160, 120)).save(root / "plage petite.jpg", quality=60)
    make_photo(2).save(root / "montagne.png")
    (root / "cassé.jpg").write_bytes(b"\x00\x01 nope")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    return root


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "image-dedup-ai" in out and "heuristic" in out and "--index" in out and "--near" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_summary(folder, capsys):
    assert main([str(folder)]) == 0
    out = capsys.readouterr().out
    assert "indexed 4 images" in out and "1 duplicate group covering 3 images" in out
    assert "plage copie ☀.png" in out and "cassé.jpg" in out
    assert "1 file without an image extension was ignored" in out


def test_json_and_outputs(folder, tmp_path, capsys):
    db = tmp_path / "idx.sqlite"
    csv_path = tmp_path / "dupes.csv"
    assert main([str(folder), "--index", str(db), "--json", "--output", str(csv_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_groups"] == 1 and payload["scan"]["hashed"] == 4 and len(payload["skipped"]) == 1
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["action"] for r in rows] == ["keep", "copy", "copy"]
    assert rows[0]["path"].endswith("plage.png") and rows[1]["identical_to_keep"] == "1"
    json_path = tmp_path / "dupes.json"
    assert main([str(folder), "--index", str(db), "--output", str(json_path)]) == 0
    out = capsys.readouterr().out
    assert "0 hashed, 4 unchanged" in out and "wrote" in out
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["scan"]["unchanged"] == 4
    assert "☀" in json_path.read_text(encoding="utf-8"), "JSON is written with ensure_ascii=False"


def test_near_and_index_only(folder, tmp_path, capsys):
    db = tmp_path / "idx.sqlite"
    assert main([str(folder), "--index", str(db)]) == 0
    capsys.readouterr()
    assert main(["--index", str(db), "--near", str(folder / "plage.png"), "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "2 nearest of 4 indexed images" in out and "1.000" in out
    assert main(["--index", str(db), "--near", str(folder / "plage.png"), "--json"]) == 0
    matches = json.loads(capsys.readouterr().out)
    assert len(matches) == 3 and matches[0]["similarity"] == 1.0
    assert main(["--index", str(db), "--method", "dhash", "--threshold", "0.95"]) == 0


def test_errors(tmp_path, capsys):
    assert main([str(tmp_path / "does-not-exist")]) == 1
    assert "does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert main([str(tmp_path), "--threshold", "2"]) == 1
    assert "threshold" in capsys.readouterr().err
    assert main([str(tmp_path), "--hash-size", "2"]) == 1


@pytest.mark.parametrize("encoding", ["ascii", "cp1252"])
def test_piped_output_with_non_ascii_names_never_crashes(folder, encoding):
    env = dict(os.environ, PYTHONIOENCODING=encoding)
    env.pop("PYTHONUTF8", None)
    for extra in ([], ["--json"]):
        proc = subprocess.run(
            [sys.executable, "-m", "image_dedup_ai", str(folder), *extra],
            capture_output=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        text = proc.stdout.decode("utf-8")
        assert "plage copie ☀.png" in text and "写真" in text
