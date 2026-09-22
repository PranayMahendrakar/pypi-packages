"""One test per finding kind, plus the quickstart path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import all_kinds, kinds
from ml_feature_check import KINDS, FeatureChecker, check


def test_quickstart(quickstart_frame):
    report = check(quickstart_frame, target="churned")

    assert report.n_rows == 60
    assert report.n_rows_checked == 60
    assert report.target == "churned"
    assert "churned" not in report.features
    assert report.n_features == 6

    assert "id_like" in kinds(report, "row_id")
    assert "duplicate_of" in kinds(report, "age_in_years")
    assert "constant" in kinds(report, "country")
    assert "date_as_string" in kinds(report, "signed_up")
    assert "leakage_suspect" in kinds(report, "churn_flag")

    assert set(report.drop_recommended) == {"row_id", "age_in_years", "country", "churn_flag"}
    assert report.keep == ["age", "signed_up"]

    clean = report.apply(quickstart_frame)
    assert list(clean.columns) == ["age", "signed_up", "churned"]
    assert list(quickstart_frame.columns) == [
        "row_id",
        "age",
        "age_in_years",
        "country",
        "signed_up",
        "churn_flag",
        "churned",
    ]  # the input frame is untouched


def test_constant_and_all_nan():
    df = pd.DataFrame(
        {
            "one_value": ["x"] * 30,
            "all_nan": [np.nan] * 30,
            "one_value_and_nan": [1.0] * 15 + [np.nan] * 15,
            "ok": list(range(30)),
        }
    )
    report = check(df)

    assert kinds(report, "one_value") == {"constant"}
    assert kinds(report, "all_nan") == {"constant"}
    # a constant column is not also reported as high_missing
    assert kinds(report, "one_value_and_nan") == {"constant"}
    assert "every one of the 30 values is missing" in report.features["all_nan"][0].message
    assert report.features["one_value"][0].severity == "high"


def test_near_constant():
    df = pd.DataFrame({"flag": [1] * 995 + [0] * 5, "other": list(range(1000))})
    report = check(df)

    assert "near_constant" in kinds(report, "flag")
    finding = [f for f in report.features["flag"] if f.kind == "near_constant"][0]
    assert finding.severity == "medium"
    assert finding.detail["top_frac"] == pytest.approx(0.995)
    assert "flag" in report.drop_recommended


def test_high_missing():
    values = [1.0, 2.0, 3.0] + [np.nan] * 27
    df = pd.DataFrame({"sparse": values, "gone": [1.0] + [np.nan] * 29, "ok": list(range(30))})
    report = check(df)

    assert "high_missing" in kinds(report, "sparse")
    sparse = [f for f in report.features["sparse"] if f.kind == "high_missing"][0]
    assert sparse.severity == "medium"
    assert sparse.detail["missing_frac"] == pytest.approx(0.9)
    # a single distinct value is "constant", not "high_missing"
    assert kinds(report, "gone") == {"constant"}


def test_high_missing_severity_is_high_above_95_percent():
    df = pd.DataFrame({"rare": [1.0, 2.0] + [np.nan] * 98, "ok": list(range(100))})
    report = check(df)
    rare = [f for f in report.features["rare"] if f.kind == "high_missing"][0]
    assert rare.severity == "high"


def test_id_like_text_and_integer():
    df = pd.DataFrame(
        {
            "code": [f"c{i:04d}" for i in range(50)],
            "counter": list(range(1000, 1050)),
            "grade": ["a", "b", "c", "d", "e"] * 10,
        }
    )
    report = check(df)

    assert "id_like" in kinds(report, "code")
    assert "id_like" in kinds(report, "counter")
    assert "id_like" not in kinds(report, "grade")
    counter = [f for f in report.features["counter"] if f.kind == "id_like"][0]
    assert counter.severity == "high"
    assert "consecutively" in counter.message
    assert counter.detail["unique_ratio"] == pytest.approx(1.0)


def test_id_like_datetime_is_medium():
    df = pd.DataFrame(
        {
            "seen_at": pd.date_range("2024-01-01", periods=40, freq="h"),
            "value": [1.5, 2.5] * 20,
        }
    )
    report = check(df)
    seen = [f for f in report.features["seen_at"] if f.kind == "id_like"][0]
    assert seen.severity == "medium"
    assert "timestamps" in seen.message


def test_continuous_float_is_not_id_like(rng):
    df = pd.DataFrame({"measurement": rng.normal(size=200), "group": ["a", "b"] * 100})
    report = check(df)
    assert "id_like" not in kinds(report, "measurement")


def test_id_like_needs_enough_rows():
    df = pd.DataFrame({"code": ["a", "b", "c", "d", "e"], "n": [1, 1, 2, 2, 3]})
    report = check(df)  # only 5 rows, below min_id_rows
    assert "id_like" not in kinds(report, "code")


def test_duplicate_of():
    df = pd.DataFrame(
        {
            "price": [1.5, 2.5, 3.5, np.nan] * 10,
            "cost": [1.5, 2.5, 3.5, np.nan] * 10,
            "label": ["x", "y", "z", None] * 10,
            "tag": ["x", "y", "z", None] * 10,
        }
    )
    report = check(df)

    assert kinds(report, "price") == set()  # the first of the pair is kept
    assert "duplicate_of" in kinds(report, "cost")
    assert "duplicate_of" in kinds(report, "tag")
    dup = [f for f in report.features["cost"] if f.kind == "duplicate_of"][0]
    assert dup.detail["other"] == "price"
    assert dup.severity == "high"
    # and the duplicate is not also reported as correlated with its twin
    assert "highly_correlated_with" not in kinds(report, "cost")


def test_highly_correlated_numeric(rng):
    base = rng.normal(size=300)
    df = pd.DataFrame(
        {
            "height_cm": base * 10 + 170,
            "height_in": base * 10 / 2.54 + 66.9,
            "weight": rng.normal(size=300),
        }
    )
    report = check(df)

    assert "highly_correlated_with" in kinds(report, "height_in")
    finding = [f for f in report.features["height_in"] if f.kind == "highly_correlated_with"][0]
    assert finding.detail["other"] == "height_cm"
    assert finding.detail["method"] == "pearson"
    assert abs(finding.detail["corr"]) > 0.95
    assert kinds(report, "weight") == set()


def test_highly_correlated_categorical_cramers_v():
    df = pd.DataFrame(
        {
            "city": ["paris", "berlin", "rome"] * 40,
            "country": ["france", "germany", "italy"] * 40,
            "colour": ["red", "green", "blue", "red"] * 30,
        }
    )
    report = check(df)

    finding = [f for f in report.features["country"] if f.kind == "highly_correlated_with"][0]
    assert finding.detail["method"] == "cramers_v"
    assert finding.detail["corr"] == pytest.approx(1.0)
    assert finding.detail["other"] == "city"


def test_leakage_numeric_correlation():
    target = np.linspace(0.0, 1.0, 80)
    df = pd.DataFrame(
        {
            "shadow_price": 3.2 * target + 0.7,
            "noise": np.tile([0.3, -0.2, 0.9, 0.1], 20),
            "price": target,
        }
    )
    report = check(df, target="price")

    leak = [f for f in report.features["shadow_price"] if f.kind == "leakage_suspect"][0]
    assert leak.severity == "high"
    assert leak.detail["pearson"] == pytest.approx(1.0)
    assert "leakage_suspect" not in kinds(report, "noise")


def test_leakage_stump_on_classification(quickstart_frame):
    report = check(quickstart_frame, target="churned")
    leak = [f for f in report.features["churn_flag"] if f.kind == "leakage_suspect"][0]
    assert leak.detail["metric"] == "accuracy"
    assert leak.detail["score"] == pytest.approx(1.0)
    assert leak.detail["baseline"] == pytest.approx(0.5)


def test_leakage_one_to_one_category():
    df = pd.DataFrame(
        {
            "plan_name": ["free", "paid"] * 40,
            "region": ["north", "south", "east", "west", "central"] * 16,
            "is_paying": [0, 1] * 40,
        }
    )
    report = check(df, target="is_paying")

    leak = [f for f in report.features["plan_name"] if f.kind == "leakage_suspect"][0]
    assert "one-to-one" in leak.message
    assert leak.detail["score"] == pytest.approx(1.0)
    assert "leakage_suspect" not in kinds(report, "region")


def test_an_identifier_is_not_mistaken_for_leakage():
    """A unique key would score 1.0 under a plain groupby; leave-one-out says otherwise."""
    df = pd.DataFrame(
        {
            "uuid_col": [f"u-{i}" for i in range(80)],
            "signal": [0.1, 0.9] * 40,
            "label": [0, 1] * 40,
        }
    )
    report = check(df, target="label")
    assert "leakage_suspect" not in kinds(report, "uuid_col")


def test_date_as_string():
    df = pd.DataFrame(
        {
            "signed_up": ["2024-01-05", "2024-02-11", "2024-03-18", "2024-04-02"] * 10,
            "note": ["hello", "there", "friend", "again"] * 10,
        }
    )
    report = check(df)

    date = [f for f in report.features["signed_up"] if f.kind == "date_as_string"][0]
    assert date.severity == "low"
    assert "pandas.to_datetime" in date.message
    assert "date_as_string" not in kinds(report, "note")
    assert "signed_up" not in report.drop_recommended  # low severity never forces a drop


@pytest.mark.parametrize(
    "name, token",
    [
        ("user_id", "id"),
        ("userId", "id"),
        ("customerid", "id"),
        ("session_uuid", "uuid"),
        ("Index", "index"),
        ("primary key", "key"),
        ("event timestamp", "timestamp"),
        ("row_number", "row"),
        ("Unnamed: 0", "unnamed"),
    ],
)
def test_suspicious_name(name, token):
    df = pd.DataFrame({name: ["a", "b", "c", "d"] * 10, "value": [1, 2, 3, 4] * 10})
    report = check(df)
    finding = [f for f in report.features[name] if f.kind == "suspicious_name"][0]
    assert finding.detail["token"] == token
    assert finding.severity == "low"


def test_ordinary_names_are_not_suspicious():
    df = pd.DataFrame({"humidity": [1.0, 2.0, 3.0, 4.0] * 10, "city": ["a", "b"] * 20})
    report = check(df)
    assert "suspicious_name" not in kinds(report, "humidity")
    assert "suspicious_name" not in kinds(report, "city")


def test_zero_importance(rng):
    n = 300
    label = np.tile([0, 1], n // 2)
    df = pd.DataFrame(
        {
            "signal": label * 4.0 + rng.normal(scale=1.0, size=n),
            "junk": rng.normal(size=n),
            "label": label,
        }
    )
    report = check(df, target="label", random_state=0)

    assert "zero_importance" in kinds(report, "junk")
    assert "zero_importance" not in kinds(report, "signal")
    finding = [f for f in report.features["junk"] if f.kind == "zero_importance"][0]
    assert finding.severity == "low"
    assert "junk" not in report.drop_recommended
    assert "junk" in report.review


def test_every_documented_kind_can_fire(quickstart_frame, rng):
    """Between a handful of frames, all ten kinds are reachable."""
    seen = set()
    seen |= all_kinds(check(quickstart_frame, target="churned"))
    seen |= all_kinds(check(pd.DataFrame({"a": [1] * 995 + [0] * 5, "b": list(range(1000))})))
    seen |= all_kinds(check(pd.DataFrame({"m": [1.0, 2.0] + [np.nan] * 28, "n": list(range(30))})))
    base = rng.normal(size=200)
    seen |= all_kinds(check(pd.DataFrame({"x": base, "y": base * 3 + 1})))
    n = 300
    label = np.tile([0, 1], n // 2)
    seen |= all_kinds(
        check(
            pd.DataFrame(
                {
                    "signal": label * 4.0 + rng.normal(size=n),
                    "junk": rng.normal(size=n),
                    "t": label,
                }
            ),
            target="t",
        )
    )
    assert seen == set(KINDS)


def test_thresholds_are_honoured(rng):
    base = rng.normal(size=200)
    df = pd.DataFrame({"a": base, "b": base + rng.normal(scale=0.35, size=200)})

    assert "highly_correlated_with" not in kinds(check(df), "b")
    assert "highly_correlated_with" in kinds(check(df, corr_threshold=0.85), "b")

    missing = pd.DataFrame({"m": [1.0, 2.0, 3.0] + [np.nan] * 17, "k": list(range(20))})
    assert "high_missing" in kinds(check(missing), "m")
    assert "high_missing" not in kinds(check(missing, missing_threshold=0.9), "m")

    ids = pd.DataFrame({"code": [f"c{i}" for i in range(40)], "v": [1.0, 2.0] * 20})
    assert "id_like" in kinds(check(ids), "code")
    almost = pd.DataFrame({"code": [f"c{i}" for i in range(39)] + ["c0"], "v": [1.0, 2.0] * 20})
    assert "id_like" not in kinds(check(almost, cardinality_threshold=0.99), "code")


def test_checker_class_matches_the_function(quickstart_frame):
    from_class = FeatureChecker(corr_threshold=0.9).check(quickstart_frame, "churned")
    from_func = check(quickstart_frame, target="churned", corr_threshold=0.9)
    assert from_class.to_dict() == from_func.to_dict()


def test_results_are_deterministic(rng):
    n = 400
    label = np.tile([0, 1], n // 2)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "t": label})
    first = check(df, target="t", random_state=7).to_dict()
    second = check(df, target="t", random_state=7).to_dict()
    assert first == second


# --------------------------------------------------------------------- QA round 1


def test_redundant_group_keeps_a_column_the_report_is_not_dropping():
    """A correlated pair must not lose its clean half to a doomed one (QA round 1)."""
    rng = np.random.default_rng(0)
    good = rng.normal(size=200) * 2 + 5
    sparse = good.copy()
    sparse[:160] = np.nan  # 80% missing, a high_missing drop on its own

    report = check(pd.DataFrame({"mostly_missing": sparse, "complete": good}))

    assert report.drop_recommended == ["mostly_missing"]
    assert report.keep == ["complete"]
    assert "highly_correlated_with" in kinds(report, "mostly_missing")
    assert "highly_correlated_with" not in kinds(report, "complete")
    # the survivor the message points at is a column the report keeps
    pointed_at = [
        f.detail["other"]
        for _, f in report.iter_findings()
        if f.kind == "highly_correlated_with"
    ]
    assert pointed_at == ["complete"]


def test_redundant_group_verdict_does_not_depend_on_column_order():
    rng = np.random.default_rng(0)
    good = rng.normal(size=200) * 2 + 5
    sparse = good.copy()
    sparse[:160] = np.nan
    left = pd.DataFrame({"mostly_missing": sparse, "complete": good})
    right = left[["complete", "mostly_missing"]]

    assert check(left).keep == check(right).keep == ["complete"]
    assert check(left).drop_recommended == check(right).drop_recommended == ["mostly_missing"]


def test_an_id_column_never_survives_a_correlated_real_feature():
    rng = np.random.default_rng(0)
    cid = np.arange(1000, 1200)
    df = pd.DataFrame({"customer_id": cid, "revenue": cid * 1.5 + rng.normal(scale=0.01, size=200)})

    for frame in (df, df[["revenue", "customer_id"]]):
        report = check(frame)
        assert report.drop_recommended == ["customer_id"]
        assert report.keep == ["revenue"]


def test_categorical_redundant_group_keeps_the_clean_column():
    """Cramer's V groups obey the same rule as the numeric ones (QA round 1)."""
    base = ["a", "b", "c", "d"] * 50
    sparse = [v if i % 5 == 0 else None for i, v in enumerate(base)]  # 80% missing

    df = pd.DataFrame({"segment_sparse": sparse, "segment": base})
    report = check(df)
    assert report.keep == ["segment"]
    assert report.drop_recommended == ["segment_sparse"]
    assert "highly_correlated_with" in kinds(report, "segment_sparse")
    assert "highly_correlated_with" not in kinds(report, "segment")

    flipped = check(df[["segment", "segment_sparse"]])
    assert flipped.keep == ["segment"]


def test_random_state_rejection_uses_the_house_wording():
    with pytest.raises(ValueError, match=r"random_state must be an integer, got 'seed'"):
        check(pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), random_state="seed")


def test_labels_that_share_a_text_form_are_rejected_at_the_entry_point():
    df = pd.DataFrame({1: [1.0] * 30, "1": ["a"] * 30, "x": list(range(30))})
    with pytest.raises(ValueError, match="share the text form"):
        check(df)


def test_to_dict_keys_every_checked_column_exactly_once(quickstart_frame):
    payload = check(quickstart_frame, target="churned").to_dict()
    assert len(payload["features"]) == payload["n_features"]
    assert len(set(payload["drop_recommended"])) == len(payload["drop_recommended"])


# --------------------------------------------------------------------- QA round 2


@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(lambda y: y, id="int"),
        pytest.param(lambda y: y.astype(bool), id="bool"),
        pytest.param(lambda y: np.array([str(v) for v in y]), id="str"),
        pytest.param(lambda y: pd.array(y.astype(bool), dtype="boolean"), id="nullable_boolean"),
    ],
)
def test_a_bool_label_is_classification_like_an_int_label(encode):
    """The same label encoded four ways must reach the same verdict (QA round 2).

    A bool column has a meaningful float view, so it used to fall through to the
    regression branch: leakage was scored with R2 instead of accuracy-against-
    majority, and a 99.5%-pure mapping scored under the 0.99 R2 threshold and
    produced no finding at all - the leaking column was silently kept.
    """
    n = 400
    group = np.repeat(np.arange(40), 10)
    y = (group % 2).astype(np.int64).copy()
    y[3] = 1 - y[3]  # 99.5% pure: a near-perfect leak, not a perfect one
    y[7] = 1 - y[7]
    df = pd.DataFrame(
        {
            "plan": np.array([f"plan_{g}" for g in group]),
            "filler": np.tile([1.0, 2.0, 3.0, 4.0], n // 4),
            "y": encode(y),
        }
    )

    report = check(df, target="y")

    assert "target 'y' was treated as classification" in report.notes
    assert "leakage_suspect" in kinds(report, "plan")
    assert report.drop_recommended == ["plan"]
    assert report.keep == ["filler"]
    leak = next(f for f in report.features["plan"] if f.kind == "leakage_suspect")
    assert leak.detail["metric"] == "accuracy"
    assert leak.detail["baseline"] < 0.99  # scored against the majority class


def test_a_wide_frame_caps_the_importance_check_and_says_what_it_cost():
    """zero_importance must not silently burn a minute on a wide frame (QA round 2).

    The permutation pass costs roughly features x rows x repeats predictions, so
    300 features over 20,000 rows ran for 68 seconds with nothing in the report
    to say why. The feature count is now capped like the correlation check
    already was, the row count scales down with it, and both are noted.
    """
    rng = np.random.default_rng(0)
    n_cols, n_rows = 220, 400
    X = rng.normal(size=(n_rows, n_cols))
    y = (X[:, 0] * 2 + X[:, 1] + rng.normal(scale=0.3, size=n_rows) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_cols)])
    df["y"] = y

    report = check(df, target="y")

    assert any("first 200 of 220 eligible columns" in note for note in report.notes)
    cost = [note for note in report.notes if "permuted" in note]
    assert cost and "200 feature(s)" in cost[0]
    assert "sample=" in cost[0]  # the workaround is named, not left to be guessed
    # nothing outside the cap can carry a zero_importance verdict
    assert len(report.columns_with("zero_importance")) <= 200


def test_a_narrow_frame_is_not_told_the_importance_check_was_expensive():
    """The cost note belongs to wide frames only, not to every run with a target."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"f{i}": rng.normal(size=200) for i in range(6)})
    df["y"] = (df["f0"] * 2 + rng.normal(scale=0.3, size=200) > 0).astype(int)

    report = check(df, target="y")

    assert not any("permuted" in note for note in report.notes)
    assert not any("eligible columns" in note for note in report.notes)


def test_a_redundant_tie_is_broken_by_frame_order_as_the_readme_says():
    """Two equally good copies tie, and the tiebreak is frame order (QA round 2).

    README.md documents the survivor rule as "no drop-level finding of its own,
    then the fewest missing values", with frame order as the explicit fallback.
    Both halves are pinned here: the missing-value rule holds in either order,
    and a genuine tie follows the frame.
    """
    rs = np.random.RandomState(7)
    a = rs.normal(size=50)
    tie = pd.DataFrame({"alpha": a, "beta": a * 2 + 1, "pad": rs.rand(50)})

    # a tie on both criteria: the frame decides, and the report stays correct
    assert check(tie).drop_recommended == ["beta"]
    assert check(tie[["beta", "alpha", "pad"]]).drop_recommended == ["alpha"]

    # not a tie: the emptier column loses whichever way the frame lists it
    sparse = a.copy()
    sparse[:45] = np.nan
    broken = pd.DataFrame({"sparse": sparse, "complete": a, "pad": rs.rand(50)})
    assert check(broken).keep == check(broken[["complete", "sparse", "pad"]]).keep


def test_a_correlation_on_a_handful_of_shared_rows_is_not_a_drop():
    """A drop must rest on more than five overlapping rows (QA round 2)."""
    n = 40
    rng = np.random.default_rng(0)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    a[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    a[20:] = rng.normal(size=20)
    b[:5] = [2.0, 4.0, 6.0, 8.0, 10.1]  # correlates +1.000 on those five rows
    b[8:20] = rng.normal(size=12)

    report = check(pd.DataFrame({"a": a, "b": b, "c": np.linspace(0, 1, n)}))

    assert "highly_correlated_with" not in kinds(report, "b")
    assert "b" not in report.drop_recommended


def test_a_partial_overlap_correlation_reports_the_rows_it_rests_on():
    """When the overlap is partial the message and detail say how partial."""
    rng = np.random.default_rng(1)
    n = 200
    base = rng.normal(size=n)
    twin = base * 3 + 0.5
    twin[:60] = np.nan  # 140 shared rows, well over the floor

    report = check(pd.DataFrame({"base": base, "twin": twin}))

    finding = next(f for f in report.features["twin"] if f.kind == "highly_correlated_with")
    assert finding.detail["n_shared"] == 140
    assert "140 shared non-missing rows" in finding.message
    assert finding.detail["min_shared"] >= 20
