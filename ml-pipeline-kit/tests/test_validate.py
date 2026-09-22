

def test_a_mask_where_every_row_fails_does_not_pass():
    """Regression: a list mask fell through to bool(raw), and a non-empty list is
    always truthy, so a check where every single row failed reported a pass."""
    import numpy as np
    import pandas as pd

    import ml_pipeline_kit as mpk

    df = pd.DataFrame({"x": [1, 2, 3]})
    for mask in ([False, False, False], (False, False, False), np.array([False] * 3)):
        result = mpk.Pipeline("t").validate(lambda d, m=mask: m, name="all_fail").run(df)
        assert result.ok is False, f"{mask!r} was accepted as a pass"
        assert result.failures

    for mask in ([True, True, True], (True, True, True)):
        result = mpk.Pipeline("t").validate(lambda d, m=mask: m, name="all_pass").run(df)
        assert result.ok is True


def test_two_row_mask_is_not_mistaken_for_a_message_pair():
    import pandas as pd

    import ml_pipeline_kit as mpk

    df = pd.DataFrame({"x": [1, 2]})
    result = mpk.Pipeline("t").validate(lambda d: [False, False], name="both_fail").run(df)
    assert result.ok is False
    result = mpk.Pipeline("t").validate(lambda d: [True, False], name="one_fails").run(df)
    assert result.ok is False


def test_failing_check_with_a_non_string_message_still_fails():
    """(False, 404) matched no branch and landed on bool(raw); a non-empty tuple is
    truthy, so the failure was silently swallowed."""
    import pandas as pd

    import ml_pipeline_kit as mpk

    df = pd.DataFrame({"x": [1, 2, 3]})
    result = mpk.Pipeline("t").validate(lambda d: (False, 404), name="coded").run(df)
    assert result.ok is False
    assert any("404" in str(f) for f in result.failures)
    result = mpk.Pipeline("t").validate(lambda d: (True, 200), name="coded_ok").run(df)
    assert result.ok is True
