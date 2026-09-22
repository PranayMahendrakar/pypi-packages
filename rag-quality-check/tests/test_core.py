

def test_long_ids_are_not_mistaken_for_passage_prose():
    """Regression: anything over 40 characters was assumed to be prose "which no
    identifier does", but source URLs and sha256 chunk ids both run past that. Passing
    either in 'retrieved' scored the answer against a bare id and reported a
    hallucination that never happened."""
    from rag_quality_check._text import looks_like_passage

    for identifier in (
        "https://docs.example.com/handbook/maintenance/bearings/section-4-3-lubrication",
        "a3f9c21e8b7d4f5e6071829304a5b6c7d8e9f0a1b2c3d4e5f60718293a4b5c6d7",
        "corpus/2024/bearings/chunk-000412.json",
        "doc-42",
        "p_reset",
    ):
        assert not looks_like_passage(identifier), identifier

    for prose in (
        "The drive end bearing should be lubricated every 500 hours of operation.",
        "Lubrication schedule: every 500 hours, or sooner in dusty conditions.",
    ):
        assert looks_like_passage(prose), prose


def test_retrieving_only_ids_does_not_report_a_hallucination():
    import rag_quality_check as rq

    url = "https://docs.example.com/handbook/maintenance/bearings/section-4-3-lubrication"
    report = rq.evaluate([{
        "query": "how often should the drive end bearing be lubricated",
        "retrieved": [url],
        "relevant": [url],
        "answer": "The drive end bearing should be lubricated every 500 hours.",
    }])
    assert not [f for f in report.failures if "not grounded" in f]
    # retrieval metrics still work on ids alone
    assert report.metrics["hit_rate"] == 1.0


def test_a_genuinely_unsupported_answer_is_still_caught():
    import rag_quality_check as rq

    report = rq.evaluate([{
        "query": "lubrication interval",
        "retrieved": ["The drive end bearing must be lubricated every 500 hours of operation."],
        "answer": "The bearing was replaced by an external contractor for four thousand dollars.",
    }])
    assert [f for f in report.failures if "not grounded" in f]
