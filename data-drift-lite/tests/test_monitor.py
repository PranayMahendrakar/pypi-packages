import numpy as np
import pandas as pd
import pytest

import data_drift_lite as ddl


def _reference(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "c": rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1]),
            "flag": rng.random(n) < 0.5,
        }
    )


def _batch(n=200, seed=1, shift=0.0, p=(0.6, 0.3, 0.1)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(shift, 1, n),
            "c": rng.choice(["a", "b", "c"], n, p=list(p)),
            "flag": rng.random(n) < 0.5,
        }
    )


def test_monitor_matches_detect():
    reference, batch = _reference(), _batch(shift=2.0)
    monitor = ddl.DriftMonitor(reference, psi_threshold=0.1)
    assert monitor.check(batch).to_dict() == ddl.detect(reference, batch, psi_threshold=0.1).to_dict()


def test_monitor_is_reusable_across_batches():
    monitor = ddl.DriftMonitor(_reference())
    calm = monitor.check(_batch(seed=1))
    shifted = monitor.check(_batch(seed=2, shift=2.0))
    recoded = monitor.check(_batch(seed=3, p=(0.1, 0.3, 0.6)))
    assert not calm.drifted
    assert shifted.drifted_columns == ["x"]
    assert recoded.drifted_columns == ["c"]
    # the reference profile is untouched by earlier checks
    assert monitor.check(_batch(seed=1)).to_dict() == calm.to_dict()


def test_monitor_exposes_columns_and_rows():
    monitor = ddl.DriftMonitor(_reference(n=1000), sample=250, random_state=1, columns=["x", "flag"])
    assert monitor.columns == ["x", "flag"]
    assert monitor.reference_rows == 250
    report = monitor.check(_batch())
    assert list(report.columns) == ["x", "flag"]
    assert report.reference_rows == 250 and report.current_rows == 200


def test_monitor_reports_schema_changes_per_batch():
    monitor = ddl.DriftMonitor(_reference())
    batch = _batch().drop(columns=["flag"]).assign(extra=1.0)
    report = monitor.check(batch)
    assert report.missing_columns == ["flag"] and report.new_columns == ["extra"]
    assert report.drifted
    retyped = _batch()
    retyped["x"] = retyped["x"].astype(str)
    assert monitor.check(retyped).dtype_changed == {"x": ("float64", "object")}


def test_monitor_validates_options_up_front():
    with pytest.raises(ValueError):
        ddl.DriftMonitor(_reference(), threshold=-0.1)
    with pytest.raises(ValueError):
        ddl.DriftMonitor(_reference(), columns=["nope"])
