import json

import numpy as np
import pandas as pd

import data_drift_lite as ddl
from data_drift_lite import ColumnDrift, DriftReport, SchemaDrift


def _frames():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame(
        {
            "age": rng.normal(40, 10, 300),
            "plan": rng.choice(["basic", "pro"], 300),
            "score": rng.normal(0, 1, 300),
        }
    )
    current = pd.DataFrame(
        {
            "age": rng.normal(60, 10, 300),
            "plan": rng.choice(["basic", "pro"], 300),
            "score": rng.normal(0, 1, 300),
        }
    )
    return reference, current


def test_summary_has_verdict_table_and_thresholds():
    reference, current = _frames()
    text = ddl.detect(reference, current).summary()
    lines = text.splitlines()
    assert lines[0].startswith("data-drift-lite: DRIFT DETECTED: 1 of 3 columns drifted (33%)")
    assert "reference rows: 300 | current rows: 300 | drifted when p < 0.05 or PSI > 0.2" == lines[1]
    assert any(line.strip().startswith("age") and line.rstrip().endswith("DRIFTED") for line in lines)
    assert any(line.strip().startswith("score") and line.rstrip().endswith("ok") for line in lines)
    assert "column" in lines[3] and "PSI" in lines[3] and "status" in lines[3]


def test_summary_for_no_drift_and_disabled_rules():
    reference, current = _frames()
    report = ddl.detect(reference, current, columns=["score"])
    assert report.summary().startswith("data-drift-lite: no drift detected (1 columns compared)")
    off = ddl.detect(reference, current, threshold=None, psi_threshold=None)
    assert "drift rules disabled" in off.summary()
    only_psi = ddl.detect(reference, current, threshold=None)
    assert "drifted when PSI > 0.2" in only_psi.summary()


def test_str_is_summary():
    reference, current = _frames()
    report = ddl.detect(reference, current)
    assert str(report) == report.summary()


def test_to_dict_shape_and_json_safety():
    reference, current = _frames()
    payload = ddl.detect(reference, current).to_dict()
    assert set(payload) == {
        "drifted",
        "drift_share",
        "drifted_columns",
        "threshold",
        "psi_threshold",
        "reference_rows",
        "current_rows",
        "columns",
        "schema",
        "notes",
    }
    assert payload["drifted"] is True and payload["drifted_columns"] == ["age"]
    age = payload["columns"]["age"]
    assert set(age) == {
        "name",
        "kind",
        "test",
        "statistic",
        "p_value",
        "psi",
        "drifted",
        "reference_stats",
        "current_stats",
        "notes",
    }
    assert isinstance(age["statistic"], float) and isinstance(age["drifted"], bool)
    assert payload["schema"] == {"missing_columns": [], "new_columns": [], "dtype_changed": {}, "drifted": False}
    json.dumps(payload, allow_nan=False)


def test_drift_share_math():
    reference, current = _frames()
    report = ddl.detect(reference, current)
    assert report.drift_share == len(report.drifted_columns) / len(report.columns)
    assert report.drift_share == 1 / 3


def test_column_drift_positional_fields_match_spec():
    col = ColumnDrift("numeric", 0.5, 0.01, 0.3, True, {"mean": 1.0}, {"mean": 2.0})
    assert (col.kind, col.statistic, col.p_value, col.psi, col.drifted) == ("numeric", 0.5, 0.01, 0.3, True)
    assert col.reference_stats == {"mean": 1.0} and col.current_stats == {"mean": 2.0}
    assert col.name == "" and col.notes == []
    assert col.to_dict()["psi"] == 0.3


def test_report_can_be_built_by_hand():
    col = ColumnDrift("numeric", None, None, float("nan"), False, {}, {}, name="x")
    report = DriftReport(columns={"x": col}, schema=SchemaDrift(missing_columns=["y"]))
    assert report.drifted is True  # schema drift
    assert report.drifted_columns == [] and report.drift_share == 0.0
    payload = report.to_dict()
    assert payload["columns"]["x"]["psi"] is None  # NaN becomes null, not an invalid JSON token
    json.dumps(payload, allow_nan=False)
    assert "DRIFT DETECTED: schema changed" in report.summary()


def test_public_api_surface():
    assert ddl.__version__ == "0.1.0"
    assert set(ddl.__all__) >= {"detect", "DriftMonitor", "DriftReport", "ColumnDrift", "SchemaDrift"}
