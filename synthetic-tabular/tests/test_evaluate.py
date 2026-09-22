import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from synthetic_tabular import FidelityReport, evaluate, generate
from synthetic_tabular.evaluate import ks_statistic, total_variation_distance


def test_identical_tables_score_100(rich_df):
    report = evaluate(rich_df, rich_df.copy())
    assert isinstance(report, FidelityReport)
    assert report.score == 100.0 and report.marginal_score == 100.0
    assert report.correlation_mad == 0.0 and report.correlation_score == 100.0
    assert all(c.value == 0.0 for c in report.columns.values())
    assert report.n_real == report.n_synthetic == len(rich_df)


def test_column_metrics_by_kind(rich_df):
    report = evaluate(rich_df, generate(rich_df, random_state=0))
    assert report.columns["age"].metric == "ks" and report.columns["age"].kind == "numeric"
    assert report.columns["joined"].metric == "ks"
    assert report.columns["city"].metric == "tvd" and report.columns["city"].kind == "categorical"
    assert report.columns["active"].metric == "tvd"
    assert report.score > 85


def test_shuffling_columns_keeps_marginals_but_breaks_correlations(rich_df):
    rng = np.random.default_rng(0)
    real = rich_df[["age", "income", "score"]]  # age/income strongly correlated
    shuffled = real.apply(lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 30))).to_numpy())
    report = evaluate(real, shuffled)
    assert report.marginal_score == 100.0
    assert report.correlation_mad > 0.2
    assert report.score < 100.0
    # the whole frame: most pairs were uncorrelated already, so the mean difference is smaller but positive
    assert evaluate(rich_df, rich_df.sample(frac=1, random_state=1)).correlation_mad == pytest.approx(0.0)


def test_disjoint_numeric_column_scores_zero():
    real = pd.DataFrame({"x": np.linspace(0, 1, 50)})
    synth = pd.DataFrame({"x": np.linspace(10, 11, 50)})
    report = evaluate(real, synth)
    assert report.columns["x"].value == 1.0 and report.columns["x"].score == 0.0
    assert report.correlation_score is None and report.score == 0.0


def test_total_variation_distance_known_value():
    real = pd.Series(["a", "a", "b", "b"])
    synth = pd.Series(["a", "a", "a", "a"])
    assert total_variation_distance(real, synth) == pytest.approx(0.5)
    assert evaluate(pd.DataFrame({"c": real}), pd.DataFrame({"c": synth})).columns["c"].value == pytest.approx(0.5)


def test_ks_statistic_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(0.4, 1.2, size=200)
    assert ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic)
    assert ks_statistic(np.array([]), np.array([])) == 0.0
    assert ks_statistic(a, np.array([])) == 1.0


def test_to_dict_is_json_safe_and_summary_lists_columns(rich_df):
    report = evaluate(rich_df, generate(rich_df, random_state=0))
    payload = json.loads(json.dumps(report.to_dict()))
    assert set(payload) >= {"score", "marginal_score", "correlation_mad", "columns", "n_real", "n_synthetic"}
    assert payload["columns"]["city"]["metric"] == "tvd"
    text = report.summary()
    assert "Fidelity score" in text and "correlations" in text
    for column in rich_df.columns:
        assert column in text


def test_missing_and_no_common_columns(rich_df):
    report = evaluate(rich_df, rich_df[["age", "city"]])
    assert report.missing_columns == ["income", "score", "active", "joined"]
    assert "missing from the synthetic" in report.summary()
    with pytest.raises(ValueError, match="share no columns"):
        evaluate(rich_df, pd.DataFrame({"other": [1, 2]}))


def test_single_column_has_no_correlation_score():
    real = pd.DataFrame({"x": np.arange(100)})
    report = evaluate(real, generate(real, random_state=0))
    assert report.correlation_score is None and report.correlation_mad is None
    assert report.score == report.marginal_score
    assert "n/a" in report.summary()


def test_missing_rates_and_empty_columns():
    real = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan], "e": [np.nan] * 4})
    report = evaluate(real, real)
    assert report.columns["x"].missing_rate_real == 0.5
    assert report.columns["e"].value == 0.0


def test_paths_and_dtype_mismatch_are_tolerated(tmp_path, rich_df):
    real_path = tmp_path / "real.csv"
    synth_path = tmp_path / "synth.csv"
    rich_df.head(200).to_csv(real_path, index=False)
    generate(rich_df.head(200), random_state=0).to_csv(synth_path, index=False)
    report = evaluate(real_path, str(synth_path))
    assert report.score > 60
    # real datetime vs synthetic strings (as loaded from csv)
    report2 = evaluate(rich_df.head(200), pd.read_csv(synth_path))
    assert report2.columns["joined"].metric == "ks" and report2.columns["joined"].value < 0.5


def test_unhashable_categorical_values_are_compared_as_text():
    real = pd.DataFrame({"tags": [["a"], ["b"], ["a"], ["c"]], "x": [1, 2, 3, 4]})
    report = evaluate(real, real)
    assert report.columns["tags"].value == 0.0
