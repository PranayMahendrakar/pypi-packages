


def test_few_large_groups_still_fill_every_requested_split():
    """Regression: 3 groups of 200 rows left val empty because no unit's midpoint
    landed inside the narrow validation band."""
    import numpy as np
    import pandas as pd

    import dataset_splitter as ds

    n = 600
    df = pd.DataFrame(
        {
            "store_id": np.repeat([1, 2, 3], n // 3),
            "x": np.random.RandomState(0).randn(n),
            "y": np.random.RandomState(1).randint(0, 2, n),
        }
    )
    split = ds.split(df, target="y", group="auto", test_size=0.2, val_size=0.1, random_state=0)
    assert len(split.train) > 0
    assert split.val is not None and len(split.val) > 0, "validation split must not be empty"
    assert len(split.test) > 0
    report = split.report()
    assert report.ok, report.warnings
    # the group guarantee still holds: no store spans two splits
    assert report.group_overlap["overlapping_groups"] == 0
    # and nothing was lost or duplicated
    assert len(split.train) + len(split.val) + len(split.test) == n


def test_group_split_with_only_two_units_keeps_train_non_empty():
    """With two units there is not enough to fill three splits; train must survive."""
    import numpy as np
    import pandas as pd

    import dataset_splitter as ds

    n = 400
    df = pd.DataFrame(
        {
            "site_id": np.repeat([1, 2], n // 2),
            "x": np.random.RandomState(0).randn(n),
            "y": np.random.RandomState(1).randint(0, 2, n),
        }
    )
    split = ds.split(df, target="y", group="site_id", test_size=0.2, val_size=0.1, random_state=0)
    assert len(split.train) > 0
    # test outranks val when units are scarce: a model with no test set cannot be evaluated
    assert len(split.test) > 0
    assert split.report().group_overlap["overlapping_groups"] == 0
    assert len(split.train) + (0 if split.val is None else len(split.val)) + len(split.test) == n
