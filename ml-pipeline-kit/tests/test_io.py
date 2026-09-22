"""Reading files, and the dtype matching the schema check is built on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline_kit import dtype_matches
from ml_pipeline_kit._io import read_table


def test_reads_a_utf8_csv(tmp_path):
    path = tmp_path / "cafés.csv"
    pd.DataFrame({"ville": ["Zürich"], "n": [1]}).to_csv(path, index=False, encoding="utf-8")
    frame, notes = read_table(path)
    assert list(frame.columns) == ["ville", "n"]
    assert frame.loc[0, "ville"] == "Zürich"
    assert notes == []


def test_reads_a_latin1_csv_and_says_so(tmp_path):
    path = tmp_path / "old.csv"
    path.write_bytes("ville,n\nZ\xfcrich,1\n".encode("latin-1"))
    frame, notes = read_table(path)
    assert frame.loc[0, "ville"] == "Zürich"
    assert notes and "latin-1" in notes[0]


def test_reads_a_tsv_and_a_parquet(tmp_path):
    tsv = tmp_path / "rows.tsv"
    tsv.write_text("a\tb\n1\t2\n", encoding="utf-8")
    frame, _ = read_table(tsv)
    assert list(frame.columns) == ["a", "b"]

    pyarrow = pytest.importorskip("pyarrow")
    assert pyarrow is not None
    parquet = tmp_path / "rows.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(parquet)
    frame, _ = read_table(parquet)
    assert list(frame["a"]) == [1, 2]


def test_missing_file_and_unknown_extension(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "ghost.csv")
    other = tmp_path / "notes.docx"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="expected .csv"):
        read_table(other)


def test_duplicate_headers_are_named_before_pandas_renames_them(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text('a,"a","b,c"\n1,2,3\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names: a"):
        read_table(path)


def test_dtype_matches_by_family_and_exactly():
    assert dtype_matches(pd.Series([1, 2], dtype="int32"), "int")
    assert dtype_matches(pd.Series([1, 2], dtype="int32"), "int64")
    assert dtype_matches(pd.Series([1.0]), float)
    assert dtype_matches(pd.Series([1.0]), "number")
    assert dtype_matches(pd.Series(["a"]), str)
    assert dtype_matches(pd.Series(["a"], dtype="category"), "category")
    assert dtype_matches(pd.Series([True]), "bool")
    assert dtype_matches(pd.to_datetime(pd.Series(["2026-01-01"])), "datetime")
    assert dtype_matches(pd.Series([1]), None)
    assert dtype_matches(pd.Series([1]), "any")
    assert not dtype_matches(pd.Series([1]), "float")
    assert not dtype_matches(pd.Series([1]), "not a dtype at all")
    assert dtype_matches(pd.Series(np.array([1], dtype="uint8")), "int")
