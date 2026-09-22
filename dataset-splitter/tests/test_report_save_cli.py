"""SplitReport serialisation, Split.save, and the command line."""
import json

import numpy as np
import pandas as pd
import pytest

from dataset_splitter import Split, SplitReport, load_table, split
from dataset_splitter.cli import main


def frame(n=120):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": rng.integers(0, 30, size=n),
            "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y": rng.choice(["yes", "no"], size=n),
            "x": rng.normal(size=n),
        }
    )


def test_report_to_dict_is_json_safe_and_complete():
    r = split(frame(), target="y", group="customer_id").report()
    assert isinstance(r, SplitReport)
    d = r.to_dict()
    json.dumps(d)
    for key in (
        "ok", "rows", "sizes", "fractions", "strategy", "class_balance", "class_counts",
        "group_overlap", "duplicate_leakage", "time_ordering", "partition", "warnings",
    ):
        assert key in d
    assert d["ok"] is True and d["rows"] == 120
    assert abs(sum(d["fractions"].values()) - 1) < 1e-9


def test_summary_lines():
    text = split(frame(), target="y", time="ts").report().summary()
    assert text.startswith("dataset-splitter report: 120 rows")
    for word in ("chronological", "duplicates", "partition", "class balance", "ok: True"):
        assert word in text


def test_report_is_cached():
    s = split(frame())
    assert s.report() is s.report()


def test_save_csv_round_trip(tmp_path):
    df = frame().set_index(pd.Index([f"k{i}" for i in range(120)]))
    s = split(df, target="y")
    paths = s.save(tmp_path / "out")
    assert set(paths) == {"train", "val", "test", "report"}
    for name, part in s.frames().items():
        back = pd.read_csv(paths[name], index_col="index")
        assert len(back) == len(part) and back.index.tolist() == part.index.tolist()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["ok"] is True
    plain = split(frame()).save(tmp_path / "plain")
    assert "index" not in pd.read_csv(plain["train"]).columns


def test_save_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    s = split(frame(), val_size=0)
    paths = s.save(tmp_path / "pq", format="parquet")
    assert set(paths) == {"train", "test", "report"}
    assert len(load_table(paths["test"])) == len(s.test)
    with pytest.raises(ValueError):
        s.save(tmp_path, format="xlsx")


def test_hand_built_split_can_report():
    df = frame()
    s = Split(train=df.iloc[:100], val=None, test=df.iloc[100:], indices={}, strategy={"method": "manual"})
    r = s.report()
    assert r.sizes == {"train": 100, "test": 20} and r.partition["exact"]


def test_cli_summary_json_and_output(tmp_path, capsys):
    path = tmp_path / "data.csv"
    frame().to_csv(path, index=False)
    assert main([str(path), "--target", "y", "--group", "customer_id"]) == 0
    out = capsys.readouterr().out
    assert "ok: True" in out and "'customer_id'" in out

    assert main([str(path), "--time", "ts", "--json", "--val-size", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"]["method"] == "chronological" and set(payload["sizes"]) == {"train", "test"}

    out_dir = tmp_path / "splits"
    assert main([str(path), "--group", "auto", "--output", str(out_dir), "--test-size", "30"]) == 0
    assert (out_dir / "train.csv").exists() and (out_dir / "report.json").exists()
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["strategy"]["group"] == ["customer_id"] and report["ok"] is True
    assert abs(len(pd.read_csv(out_dir / "test.csv")) - 30) <= 8  # whole groups: close, not exact

    exact_dir = tmp_path / "exact"
    assert main([str(path), "--output", str(exact_dir), "--test-size", "30", "--val-size", "10"]) == 0
    assert len(pd.read_csv(exact_dir / "test.csv")) == 30
    assert len(pd.read_csv(exact_dir / "val.csv")) == 10


def test_cli_exit_codes(tmp_path, capsys):
    path = tmp_path / "dups.csv"
    base = frame(30)
    pd.concat([base, base]).to_csv(path, index=False)
    assert main([str(path), "--no-dedupe"]) == 1  # leakage detected
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert main([str(path), "--target", "nope"]) == 2
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--group" in capsys.readouterr().out


def test_cli_handles_non_ascii_under_a_byte_pipe(tmp_path, capsys, monkeypatch):
    """main() makes stdout UTF-8 tolerant, so unicode labels never raise UnicodeEncodeError."""
    import io
    import sys

    path = tmp_path / "unicode.csv"
    n = 60
    pd.DataFrame(
        {
            "customer_id": [i // 3 for i in range(n)],
            "y": ["café", "東京", "naïve"] * (n // 3),
            "x": range(n),
        }
    ).to_csv(path, index=False, encoding="utf-8")

    for as_json in (False, True):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict", newline="")
        monkeypatch.setattr(sys, "stdout", stream)
        argv = [str(path), "--target", "y", "--group", "customer_id"]
        assert main(argv + (["--json"] if as_json else [])) == 0
        stream.flush()
        text = raw.getvalue().decode("utf-8")
        assert "café" in text and "東京" in text  # written through, not escaped or crashed


def test_saved_report_json_keeps_unicode_readable(tmp_path):
    n = 60
    df = pd.DataFrame({"y": ["café", "東京"] * (n // 2), "x": range(n)})
    paths = split(df, target="y").save(tmp_path / "out")
    text = paths["report"].read_text(encoding="utf-8")
    assert "café" in text and "u00e9" not in text
    assert json.loads(text)["ok"] is True
