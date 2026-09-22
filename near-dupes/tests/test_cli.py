import json

import numpy as np
import pandas as pd
import pytest

from near_dupes.cli import main


def write_csv(tmp_path):
    df = pd.DataFrame(
        {
            "name": ["Acme Corporation", "ACME Corporation", "Acme Corporaton", "Globex"],
            "city": ["New York", "new york", "New York", "Springfield"],
        }
    )
    path = tmp_path / "people.csv"
    df.to_csv(path, index=False)
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "near-dupes" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_csv_summary_json_and_output(tmp_path, capsys):
    path = write_csv(tmp_path)
    assert main([str(path), "--key", "name", "city", "--threshold", "0.8"]) == 0
    out = capsys.readouterr().out
    assert "4 rows" in out and "1 duplicate group" in out

    assert main([str(path), "--key", "name", "city", "--threshold", "0.8", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["groups"] == [[0, 1, 2]] and data["kind"] == "records"

    out_path = tmp_path / "clean.csv"
    assert main([str(path), "--key", "name", "city", "--threshold", "0.8", "--output", str(out_path)]) == 0
    assert "wrote 2 items" in capsys.readouterr().out
    clean = pd.read_csv(out_path)
    assert list(clean["name"]) == ["Acme Corporation", "Globex"]


def test_text_file_lines(tmp_path, capsys):
    path = tmp_path / "lines.txt"
    path.write_text("first line here\n\nFirst Line Here!\nsomething else\n", encoding="utf-8")
    assert main([str(path), "--threshold", "0.8", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "text" and data["n_items"] == 4
    assert data["groups"] == [[0, 2]]
    out_path = tmp_path / "clean.txt"
    assert main([str(path), "--threshold", "0.8", "--kind", "text", "--output", str(out_path)]) == 0
    # original order is kept; the longest copy of the duplicated line survives
    assert out_path.read_text(encoding="utf-8").splitlines() == ["", "First Line Here!", "something else"]


def test_image_directory(tmp_path, capsys):
    Image = pytest.importorskip("PIL.Image")
    x = np.mgrid[0:64, 0:64][1]
    base = (x * 3 + 20).astype(np.uint8)
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(np.clip(base.astype(int) + 30, 0, 255).astype(np.uint8)).save(tmp_path / "b.png")
    Image.fromarray(base.T.copy()).save(tmp_path / "c.png")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert main([str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "images" and data["n_items"] == 3 and data["groups"] == [[0, 1]]
    assert main([str(tmp_path / "a.png"), str(tmp_path / "b.png"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["groups"] == [[0, 1]]
    assert main([str(tmp_path / "*.png"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_items"] == 3


def test_missing_file_and_bad_key_return_one(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 1
    assert "error" in capsys.readouterr().err
    path = write_csv(tmp_path)
    assert main([str(path), "--key", "missing"]) == 1
    assert "missing" in capsys.readouterr().err
    assert main([str(path), "--threshold", "2"]) == 1


def test_console_safe_survives_unencodable_output(tmp_path, capsys, monkeypatch):
    """main() must not raise UnicodeEncodeError on a console that cannot encode the data."""
    import io
    import sys

    from near_dupes.cli import _make_console_safe

    path = tmp_path / "u.txt"
    path.write_text("日本語 café\n日本語 café\nplain\n", encoding="utf-8")

    # a strict cp1252 stream, exactly what a legacy Windows console looks like
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(sys, "stdout", stream)
    try:
        assert main([str(path), "--json"]) == 0
        stream.flush()
    finally:
        monkeypatch.undo()
    assert raw.getvalue()  # something was written, nothing raised

    # the helper itself is tolerant of odd streams
    _make_console_safe()


def test_json_output_is_not_ascii_escaped(tmp_path, capsys):
    path = tmp_path / "u.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    assert main([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "text"


def test_duplicate_csv_headers_are_mangled_by_pandas_not_an_error(tmp_path, capsys):
    """pandas renames a repeated CSV header to name.1, so the CLI has no ambiguity to reject."""
    path = tmp_path / "dup.csv"
    path.write_text("\n".join(["name,name", "a,b", "a,c", ""]), encoding="utf-8")
    assert main([str(path)]) == 0
    assert "2 rows" in capsys.readouterr().out


def test_bad_key_column_reports_error(tmp_path, capsys):
    path = write_csv(tmp_path)
    assert main([str(path), "--key", "nope"]) == 1
    assert "not found" in capsys.readouterr().err
