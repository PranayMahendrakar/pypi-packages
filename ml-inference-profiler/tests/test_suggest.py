

# --- regression: stage classification and over-eager advice -----------------------------

def test_decode_is_not_assumed_to_be_preprocessing():
    """"decode" used to be a preprocessing word, so an output-decoding pipeline was
    told its data path was the problem - the exact opposite diagnosis."""
    from ml_inference_profiler.suggest import classify

    assert classify("decode_tokens") == "postprocess"
    assert classify("token decode") == "postprocess"
    assert classify("detokenize") == "postprocess"
    # image decoding really is input work and must stay that way
    assert classify("jpeg_decode") == "preprocess"
    assert classify("image decode") == "preprocess"
    # bare "decode" says nothing either way, so the heuristic stays out of it
    assert classify("decode") == "other"


def test_output_heavy_pipeline_is_not_blamed_on_preprocessing():
    import time

    import ml_inference_profiler as mp

    steps = [
        ("load", lambda d: (time.sleep(0.005), d)[1]),
        ("model", lambda d: (time.sleep(0.010), d)[1]),
        ("decode_tokens", lambda d: (time.sleep(0.060), d)[1]),
    ]
    report = mp.profile_pipeline(steps, [1], repeats=3, warmup=1)
    assert not [s for s in report.suggestions if "preprocessing is" in s.lower()]
    assert "decode_tokens" in report.bottleneck.label


def test_a_balanced_pipeline_gets_no_preprocessing_complaint():
    """A 40/35/25 split is healthy; calling it out taught users to ignore the advice."""
    import time

    import ml_inference_profiler as mp

    steps = [
        ("preprocess", lambda d: (time.sleep(0.020), d)[1]),
        ("model", lambda d: (time.sleep(0.018), d)[1]),
        ("postprocess", lambda d: (time.sleep(0.012), d)[1]),
    ]
    report = mp.profile_pipeline(steps, [1], repeats=3, warmup=1)
    assert not [s for s in report.suggestions if "preprocessing is" in s.lower()]


def test_identical_healthy_runs_give_identical_advice():
    """The p95 rule used to fire intermittently, so the same clean pipeline got
    different advice from one run to the next."""
    import time

    import ml_inference_profiler as mp

    def run():
        steps = [
            ("preprocess", lambda d: (time.sleep(0.004), d)[1]),
            ("model", lambda d: (time.sleep(0.010), d)[1]),
            ("postprocess", lambda d: (time.sleep(0.003), d)[1]),
        ]
        report = mp.profile_pipeline(steps, [1], repeats=6, warmup=1)
        return tuple(s.split(":")[0] for s in report.suggestions)

    advice = {run() for _ in range(6)}
    assert len(advice) == 1, f"advice varied between identical runs: {advice}"
