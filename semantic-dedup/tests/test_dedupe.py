

def test_auto_is_not_an_order_of_magnitude_slower_than_minhash():
    """Regression: auto stayed exhaustive until 5,000 texts, so the default took 15.2s on
    2,700 passages that minhash finished in 1.6s with exactly the same answer."""
    import random
    import time

    import semantic_dedup as sd

    rng = random.Random(0)
    words = ["system", "report", "bearing", "revenue", "meeting", "customer", "install",
             "network", "sensor", "quarter", "budget", "service", "update", "request"]
    texts = [" ".join(rng.choice(words) for _ in range(rng.randint(12, 30))) for _ in range(2000)]
    texts += texts[:150]  # 150 genuine duplicates planted

    start = time.perf_counter()
    result = sd.dedupe(texts)
    elapsed = time.perf_counter() - start

    assert result.n_removed == 150, "the planted duplicates must still all be found"
    # generous ceiling so a slow machine does not make this flaky; the point is that it
    # is not the ~15s the exhaustive path took
    assert elapsed < 8.0, f"default dedupe took {elapsed:.1f}s on 2,150 texts"


def test_a_csv_field_over_the_python_limit_is_read_not_crashed():
    """Regression: a passage longer than 131,072 characters raised a raw _csv.Error."""
    import csv
    import tempfile
    from pathlib import Path

    import semantic_dedup as sd

    path = Path(tempfile.mkdtemp()) / "big.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text"])
        writer.writerow(["x" * 200_000])
        writer.writerow(["y" * 200_000])
        writer.writerow(["x" * 200_000])

    result = sd.dedupe(str(path))
    assert result.n_removed == 1, "the two identical long passages should collapse to one"
