"""Regression tests for the faults an independent review found in 0.1.0.

Each test names the thing that used to be wrong, so a later change that quietly
brings it back fails here instead of in somebody's plant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sensor_anomaly
from sensor_anomaly import detect
from sensor_anomaly._joint import cross_channel_cut


def correlated_plant(n: int = 6000, k: int = 20, seed: int = 3) -> pd.DataFrame:
    """A healthy plant whose channels all track one process variable.

    This is what real instrumentation looks like - sensors are deliberately
    redundant - and it is the shape that used to break the cross-channel stage.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "s%02d" % i: 10 * (i + 1) + (i % 4 + 1) * base + rng.normal(0, 0.3, n)
            for i in range(k)
        }
    )


# --------------------------------------------------------------------------
# major: near-collinear channels used to put ~1% of a clean table in .joint
# --------------------------------------------------------------------------


def test_a_clean_plant_of_redundant_channels_trips_nothing_across_channels():
    """Correlated channels are the normal case, not an edge case.

    The cut used to be a fixed multiple of the forest score's own robust spread.
    That spread is measured on the bulk of the score, and channels that track each
    other squeeze the bulk without squeezing the tail, so a perfectly healthy plant
    came back with 1% of its rows flagged - the JOINT_MAX_RATE ceiling acting as a
    target. At 20 channels this table used to yield 60 cross-channel flags.
    """
    df = correlated_plant()
    standardized = (df - df.mean()) / df.std()
    condition = np.linalg.cond(np.cov(standardized.to_numpy(), rowvar=False))
    assert condition > 500, "this fixture is meant to be near-collinear"

    report = detect(df)
    assert report.joint == [], "a clean table must not trip the cross-channel stage"
    assert not report.anomalous
    assert not any("unusual across channels" in note for note in report.notes)


@pytest.mark.parametrize("k", [3, 10, 20])
def test_clean_tables_stay_quiet_at_every_channel_count(k):
    """It was the channel covariance, not the channel count - check both anyway."""
    assert detect(correlated_plant(n=4000, k=k)).joint == []
    rng = np.random.default_rng(5)
    independent = pd.DataFrame(
        {"c%02d" % i: rng.normal(i, 1.0, 4000) for i in range(k)}
    )
    assert detect(independent).joint == []


def test_the_full_size_table_from_the_review_comes_back_empty():
    """The reviewer's own repro, at the reviewer's own size: 50,000 rows, 20 channels.

    Size matters here in a way it does not for most bugs. The cut is corrected for
    how many chances the table gives itself (:func:`cross_channel_cut`), so a table
    an order of magnitude larger than the other fixtures is a genuinely different
    test of it, not a slower copy. This exact frame used to return 500 rows in
    ``.joint`` - precisely 1.00% of it, the JOINT_MAX_RATE ceiling acting as a
    target - and the report announced that about a healthy plant.
    """
    rng = np.random.default_rng(3)
    n = 50_000
    base = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "s%02d" % i: 10 * (i + 1) + (i % 4 + 1) * base + rng.normal(0, 0.3, n)
            for i in range(20)
        }
    )
    report = detect(df)

    assert report.joint == []
    assert report.n_events == 0 or {e.kind for e in report.events} <= {"spike"}
    assert not report.anomalous
    assert not any("unusual across channels" in note for note in report.notes)
    assert "OK" in report.summary().splitlines()[0]


def test_a_real_fault_is_what_the_cross_channel_stage_returns():
    """With a genuine fault present, .joint must be about the fault.

    One channel dies to exactly zero while every other channel keeps tracking the
    base. Precision here used to be 0.9%: the flags were almost all clean rows.
    """
    df = correlated_plant()
    df.loc[3000:3080, "s03"] = 0.0

    report = detect(df)
    flagged = np.array(report.joint, dtype=int)
    assert flagged.size > 0, "the cross-channel stage found nothing"
    inside = int(((flagged >= 3000) & (flagged <= 3080)).sum())
    assert inside / flagged.size >= 0.9, "cross-channel flags must be the faulty rows"
    assert inside / 81 >= 0.5, "and most of the faulty rows should be found"
    assert report.worst_channels[0] == "s03"
    assert any(
        event.kind == "joint" and event.start <= 3080 and event.end >= 3000
        for event in report.events
    )


def test_the_cross_channel_cut_is_corrected_for_how_big_the_table_is():
    """One reading in a clean table of any size, not one reading in every table."""
    small = cross_channel_cut(6_000, 3.0)
    large = cross_channel_cut(2_000_000, 3.0)
    assert 4.5 < small < large < 7.0
    assert cross_channel_cut(2_000_000, 5.0) > cross_channel_cut(2_000_000, 3.0)
    # Never looser than the per-channel test, however small the table.
    assert cross_channel_cut(1, 4.0) == 4.0


# --------------------------------------------------------------------------
# major: the headline and `anomalous` used to contradict the report's own note
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [2000, 2001, 2002, 2003, 2004])
def test_clean_data_is_not_anomalous_and_the_headline_agrees(seed):
    """`anomalous` has to be usable as `if report.anomalous: page_the_operator()`.

    At the default sensitivity a 3-sigma test flags about 0.27% of perfectly normal
    readings. The report always said so in its notes, while the first line
    announced "5 of 5 channels flagged" and `anomalous` was True.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.normal(0, 1, 3000) for name in "abcde"})
    report = detect(df)

    assert report.anomalous is False
    assert report.quiet is True
    headline = report.summary().splitlines()[0]
    assert "OK" in headline and "channels flagged" not in headline
    assert any("Nothing here stands out" in note for note in report.notes)
    assert report.to_dict()["anomalous"] is False
    # It stayed quiet for the right reason: only noise-floor outliers, no faults.
    assert {event.kind for event in report.events} <= {"spike"}


def test_a_real_fault_still_reads_as_anomalous():
    """The quiet headline must not swallow a finding."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"temp": rng.normal(70, 1, 500), "flow": rng.normal(12, 0.5, 500)}
    )
    df.loc[300:340, "flow"] = 0.0
    report = detect(df)

    assert report.anomalous is True
    assert report.quiet is False
    assert "OK" not in report.summary().splitlines()[0]

    spike = pd.DataFrame({name: rng.normal(0, 1, 2000) for name in "abc"})
    spike.loc[900, "b"] = 60.0
    loud = detect(spike)
    assert loud.anomalous is True, "one genuine outlier is above the noise floor"


def test_the_reassurance_note_states_the_test_it_actually_ran():
    """It used to claim every channel was at or below a rate half of them exceeded."""
    rng = np.random.default_rng(77)
    df = pd.DataFrame({name: rng.normal(0, 1, 5000) for name in "abcdefgh"})
    report = detect(df)

    expected = report.params["expected_false_rate"]
    worst = max(res.rate for res in report.channels.values())
    assert worst > expected, "this fixture is meant to sit above the bare rate"
    assert worst <= 2.0 * expected

    note = next(n for n in report.notes if "Nothing here stands out" in n)
    assert "within 2 times that rate" in note
    assert "at or below that" not in note


# --------------------------------------------------------------------------
# minor: random_state was the one public argument nobody checked
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows", [30, 200])
@pytest.mark.parametrize(
    "bad", ["x", -1, 2.5, [1, 2], 2**32, np.random.default_rng(0)]
)
def test_a_bad_random_state_is_our_error_whatever_the_table_size(n_rows, bad):
    """Same value, same outcome - the short-table path used to accept anything."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"a": rng.normal(0, 1, n_rows), "b": rng.normal(0, 1, n_rows)}
    )
    with pytest.raises(ValueError, match="random_state") as exc:
        detect(df, random_state=bad)
    message = str(exc.value)
    assert "IsolationForest" not in message, "never leak the model choice"
    assert "sklearn" not in message


def test_a_numpy_generator_is_refused_with_a_way_forward():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    with pytest.raises(ValueError, match="Generator"):
        detect(df, random_state=np.random.default_rng(0))


def test_the_random_states_that_are_meant_to_work_still_do():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    for good in (0, 7, None, np.random.RandomState(3), np.uint32(11)):
        assert detect(df, random_state=good).n_rows == 200
    assert detect(df, random_state=4).to_dict() == detect(df, random_state=4).to_dict()


# --------------------------------------------------------------------------
# minor: a duplicated CSV header was renamed behind the caller's back
# --------------------------------------------------------------------------


def test_a_duplicate_header_in_a_file_fails_the_way_a_dataframe_does(tmp_path):
    """pandas turns the second 'a' into 'a.1' before any check of ours can see it."""
    path = tmp_path / "dup.csv"
    path.write_text("a,a\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names"):
        detect(str(path))

    tsv = tmp_path / "dup.tsv"
    tsv.write_text("a\ta\n1\t2\n3\t4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names"):
        detect(str(tsv))


def test_blank_headers_are_not_mistaken_for_duplicates(tmp_path):
    """pandas names empty headers 'Unnamed: N', which are distinct. Do not refuse."""
    path = tmp_path / "blanks.csv"
    path.write_text("a,b,,\n1,2,3,4\n5,6,7,8\n", encoding="utf-8")
    assert detect(str(path)).n_rows == 2

    indexed = tmp_path / "indexed.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(indexed, encoding="utf-8")
    assert detect(str(indexed)).n_rows == 2


# --------------------------------------------------------------------------
# minor: an empty export was diagnosed as a table full of text
# --------------------------------------------------------------------------


def test_a_header_only_file_is_reported_as_empty_not_as_unreadable(tmp_path):
    """The columns are still channels; the finding is that there are no rows."""
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n", encoding="utf-8")
    report = detect(str(path))

    assert list(report.channels) == ["a", "b"]
    assert report.notes == ["The table has no rows, so there was nothing to score."]
    assert "no rows" in report.summary().splitlines()[0]
    assert not any("held no numbers" in note for note in report.notes)
    assert not any("channels=[...]" in note for note in report.notes)


def test_a_file_that_really_is_all_text_still_says_so(tmp_path):
    """The old diagnosis was not wrong in general, only on an empty table."""
    path = tmp_path / "text.csv"
    path.write_text("a,b\nyes,no\nyes,no\n", encoding="utf-8")
    report = detect(str(path))
    assert report.channels == {}
    assert any("held no numbers" in note for note in report.notes)


# --------------------------------------------------------------------------
# minor: --output x.parquet leaked five lines of pandas engine internals
# --------------------------------------------------------------------------


def test_writing_parquet_without_pyarrow_points_at_the_extra(tmp_path, monkeypatch, capsys):
    from sensor_anomaly import cli

    def no_engine(self, *args, **kwargs):
        raise ImportError(
            "Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'."
        )

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine, raising=True)

    source = tmp_path / "plant.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)}
    ).to_csv(source, index=False, encoding="utf-8")

    assert cli.main([str(source), "--output", str(tmp_path / "out.parquet")]) == 1
    error = capsys.readouterr().err
    assert "sensor-anomaly: error:" in error
    assert 'pip install "sensor-anomaly[parquet]"' in error
    assert "fastparquet" not in error
    assert len(error.strip().splitlines()) == 1


def test_the_read_path_message_is_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "plant.parquet"
    path.write_bytes(b"not really parquet")

    def no_engine(*args, **kwargs):
        raise ImportError("Unable to find a usable engine")

    monkeypatch.setattr(pd, "read_parquet", no_engine, raising=True)
    with pytest.raises(ImportError, match="sensor-anomaly\\[parquet\\]"):
        sensor_anomaly.detect(str(path))
