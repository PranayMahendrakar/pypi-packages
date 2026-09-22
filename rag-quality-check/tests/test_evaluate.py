"""evaluate(), evaluate_case() and every edge case the README promises."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

import rag_quality_check
from rag_quality_check import ANSWER_METRICS, RETRIEVAL_METRICS, evaluate, evaluate_case

QUICKSTART = [
    {"query": "reset my password", "retrieved": ["p_reset", "p_billing"], "relevant": ["p_reset"]},
    {"query": "cancel my plan", "retrieved": ["p_faq", "p_billing"], "relevant": ["p_cancel"]},
]


def all_metric_values(report):
    values = list(report.metrics.values())
    for case in report.per_case:
        values.extend(case.metrics.values())
        values.append(case.score)
    return values


# --------------------------------------------------------------------------
# the quickstart path
# --------------------------------------------------------------------------

def test_quickstart_numbers_are_exactly_the_documented_formulas():
    report = evaluate(QUICKSTART, k=2)
    assert report.n_cases == 2
    assert report.metrics["precision_at_k"] == pytest.approx(0.25)
    assert report.metrics["recall_at_k"] == pytest.approx(0.5)
    assert report.metrics["mrr"] == pytest.approx(0.5)
    assert report.metrics["ndcg_at_k"] == pytest.approx(0.5)
    assert report.metrics["hit_rate"] == pytest.approx(0.5)
    assert report.support["recall_at_k"] == 2
    assert report.estimated is False
    assert report.similarity == "lexical overlap"


def test_quickstart_summary_is_printable_and_names_the_broken_case():
    text = evaluate(QUICKSTART, k=2).summary()
    assert "rag-quality-check: 2 case(s), top-2" in text
    assert "precision_at_k" in text and "hit_rate" in text
    assert "cancel my plan" in text
    assert "failures (1)" in text
    assert text == str(evaluate(QUICKSTART, k=2))


def test_quickstart_report_round_trips_through_json():
    payload = evaluate(QUICKSTART, k=2).to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["n_cases"] == 2
    assert payload["per_case"][0]["metrics"]["hit_rate"] == 1.0


def test_failures_read_as_plain_language():
    report = evaluate(QUICKSTART, k=2)
    assert report.ok is False
    assert len(report.failures) == 1
    assert "nothing relevant in the top 2" in report.failures[0]
    assert report.per_case[0].ok is True


def test_weakest_puts_the_worst_case_first_and_is_stable():
    report = evaluate(QUICKSTART, k=2)
    worst = report.weakest(2)
    assert [case.index for case in worst] == [1, 0]
    assert report.weakest(0) == []
    assert len(report.weakest(99)) == 2
    assert [c.index for c in report.weakest(5)] == [c.index for c in report.weakest(5)]


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------

def test_empty_retrieved_list_scores_zeros_instead_of_dividing_by_zero():
    report = evaluate([{"query": "anything", "retrieved": [], "relevant": ["a"]}], k=5)
    case = report.per_case[0]
    assert case.n_retrieved == 0 and case.n_unique == 0
    assert case.metrics["precision_at_k"] == 0.0
    assert case.metrics["recall_at_k"] == 0.0
    assert case.metrics["mrr"] == 0.0
    assert case.metrics["ndcg_at_k"] == 0.0
    assert case.metrics["hit_rate"] == 0.0
    assert any("retrieved nothing" in failure for failure in report.failures)


def test_empty_retrieved_with_no_labels_also_survives():
    report = evaluate([{"query": "anything", "retrieved": []}], k=3)
    case = report.per_case[0]
    assert case.metrics["precision_at_k"] == 0.0
    assert "ndcg_at_k" in case.excluded
    assert case.estimated is True


def test_k_larger_than_the_number_retrieved_is_handled_and_noted():
    report = evaluate(
        [{"query": "solar", "retrieved": ["solar panels"], "relevant": ["solar panels"]}],
        k=5,
    )
    case = report.per_case[0]
    # all 1 retrieved items are relevant, but k is 5, so precision is 1/5
    assert case.metrics["precision_at_k"] == pytest.approx(0.2)
    assert case.metrics["recall_at_k"] == pytest.approx(1.0)
    note = " ".join(case.notes)
    assert "asked for top-5" in note and "only 1 passage(s) came back" in note
    assert "still divides by 5" in note


def test_a_case_with_no_relevant_ids_is_excluded_from_recall_and_says_so():
    report = evaluate(
        [
            {"query": "a", "retrieved": ["x"], "relevant": []},
            {"query": "b", "retrieved": ["y"], "relevant": ["y"]},
        ],
        k=1,
    )
    first, second = report.per_case
    assert "recall_at_k" not in first.metrics
    assert "recall_at_k" in first.excluded
    assert "ndcg_at_k" in first.excluded
    # a query labelled as having no right answer cannot score on retrieval at
    # all: every metric there could only ever be 0
    assert first.metrics == {}
    assert second.metrics["recall_at_k"] == 1.0
    # the mean covers only the case that could produce one
    assert report.metrics["recall_at_k"] == pytest.approx(1.0)
    assert report.support["recall_at_k"] == 1
    notes = " ".join(report.notes)
    assert "no relevant ids at all" in notes
    assert "recall_at_k is excluded for 1 of 2 case(s)" in notes


def test_duplicate_passages_are_counted_once_and_noted():
    report = evaluate(
        [
            {
                "query": "solar",
                "retrieved": ["p1", "p1", "p2", "p1"],
                "relevant": ["p1"],
            }
        ],
        k=2,
    )
    case = report.per_case[0]
    assert case.n_retrieved == 4
    assert case.n_unique == 2
    # counted once: 1 relevant of k=2, not 3 of 2
    assert case.metrics["precision_at_k"] == pytest.approx(0.5)
    assert any("2 duplicate passage(s)" in note for note in case.notes)


def test_duplicates_are_detected_by_id_for_dict_items():
    case = evaluate_case(
        "solar",
        [{"id": "p1", "text": "one wording"}, {"id": "p1", "text": "another wording"}],
        relevant=["p1"],
        k=1,
    )
    assert case.n_unique == 1
    assert case.metrics["precision_at_k"] == 1.0


def test_unicode_queries_passages_and_answers_are_measured_not_dropped():
    report = evaluate(
        [
            {
                "query": "パスワードを再設定",
                "retrieved": ["パスワードを再設定するには設定を開きます"],
                "answer": "設定を開きます。",
            },
            {
                "query": "café ouvert",
                "retrieved": ["le café est ouvert le matin"],
                "answer": "Le café est ouvert.",
            },
            {
                "query": "पासवर्ड रीसेट",
                "retrieved": ["पासवर्ड रीसेट करने के लिए सेटिंग्स खोलें"],
                "answer": "सेटिंग्स खोलें।",
            },
        ],
        k=1,
    )
    assert report.n_cases == 3
    for case in report.per_case:
        assert case.metrics["hit_rate"] == 1.0
        assert case.metrics["groundedness"] == pytest.approx(1.0)
    assert "パスワード" in report.summary()
    json.dumps(report.to_dict(), ensure_ascii=False)


def test_every_metric_everywhere_stays_inside_zero_to_one():
    report = evaluate(
        [
            {"query": "a b c", "retrieved": ["a b c", "d", "a b c"], "relevant": {"a b c": 7.5}},
            {"query": "", "retrieved": [], "relevant": []},
            {"query": "x", "retrieved": ["x"], "answer": "x y z", "ground_truth": "x"},
            {"query": "long query about solar power", "retrieved": ["solar"], "relevant": ["solar", "wind"]},
        ],
        k=3,
    )
    for value in all_metric_values(report):
        assert 0.0 <= value <= 1.0, value
        assert not math.isnan(value)


def test_one_thousand_cases_evaluate_in_a_few_seconds():
    cases = [
        {
            "query": "question number {0} about topic {1}".format(i, i % 50),
            "retrieved": ["doc-{0}".format((i + j) % 400) for j in range(5)],
            "relevant": ["doc-{0}".format(i % 400)],
            "answer": "The answer to question number {0} is topic {1}.".format(i, i % 50),
        }
        for i in range(1000)
    ]
    start = time.perf_counter()
    report = evaluate(cases, k=5)
    elapsed = time.perf_counter() - start
    assert report.n_cases == 1000
    assert len(report.per_case) == 1000
    assert elapsed < 5.0, "1000 cases took {0:.2f}s".format(elapsed)


# --------------------------------------------------------------------------
# estimated relevance
# --------------------------------------------------------------------------

def test_without_relevant_ids_the_numbers_are_labelled_estimated_everywhere():
    report = evaluate(
        [
            {
                "query": "how do I reset my password",
                "retrieved": ["to reset your password open settings", "billing is monthly"],
            }
        ],
        k=2,
    )
    case = report.per_case[0]
    assert report.estimated is True
    assert report.fully_estimated is True
    assert case.estimated is True
    assert case.metrics["hit_rate"] == 1.0
    assert "recall_at_k" in case.excluded
    text = report.summary()
    assert "ESTIMATED" in text
    assert "estimated, not" in text
    assert "(estimated relevance)" in case.summary()
    assert report.to_dict()["estimated"] is True


def test_a_mix_of_labelled_and_unlabelled_cases_reports_the_split():
    report = evaluate(
        [
            {"query": "a", "retrieved": ["a"], "relevant": ["a"]},
            {"query": "b", "retrieved": ["b"]},
        ],
        k=1,
    )
    assert report.estimated is True
    assert report.fully_estimated is False
    assert report.n_estimated == 1
    assert "1 of 2 cases" in report.summary()


def test_estimated_failure_lines_say_they_are_estimated():
    report = evaluate([{"query": "solar power", "retrieved": ["unrelated text"]}], k=1)
    assert any("(estimated)" in failure for failure in report.failures)


# --------------------------------------------------------------------------
# answer metrics
# --------------------------------------------------------------------------

def test_answer_metrics_appear_only_when_an_answer_is_given():
    with_answer = evaluate_case(
        "how do I reset my password",
        ["To reset your password, open Settings and choose Reset."],
        answer="Open Settings and choose Reset.",
        k=1,
    )
    assert with_answer.metrics["groundedness"] == pytest.approx(1.0)
    assert with_answer.metrics["citation_coverage"] == pytest.approx(1.0)
    # pinned, not a range: the query keeps 3 content words (reset, my,
    # password) and the answer repeats exactly one of them
    assert with_answer.metrics["answer_relevance"] == pytest.approx(1 / 3)
    assert "answer_correctness" not in with_answer.metrics
    # and a correct, fully grounded answer raises nothing
    assert with_answer.failures == []

    without = evaluate_case("q", ["p"], k=1)
    for name in ANSWER_METRICS:
        assert name not in without.metrics
        assert name in without.excluded


def test_an_ungrounded_answer_is_a_named_failure():
    case = evaluate_case(
        "how do I reset my password",
        ["Billing runs on the first of every month."],
        answer="Press the big red lever twice. Then wait for the fax.",
        k=1,
    )
    assert case.metrics["groundedness"] == pytest.approx(0.0)
    assert any("answer not grounded" in failure for failure in case.failures)


def test_citation_coverage_falls_when_the_context_is_padded():
    case = evaluate_case(
        "solar panel output",
        [
            "Solar panel output peaks at noon.",
            "Unrelated note about invoices.",
            "Another unrelated note about staffing.",
            "A fourth passage about parking.",
        ],
        answer="Solar panel output peaks at noon.",
        k=4,
    )
    assert case.metrics["groundedness"] == pytest.approx(1.0)
    assert case.metrics["citation_coverage"] == pytest.approx(0.25)


def test_ground_truth_adds_answer_correctness_and_flags_a_wrong_answer():
    good = evaluate_case(
        "capital of France",
        ["The capital of France is Paris."],
        answer="The capital of France is Paris.",
        ground_truth="The capital of France is Paris.",
        k=1,
    )
    assert good.metrics["answer_correctness"] == pytest.approx(1.0)

    bad = evaluate_case(
        "capital of France",
        ["The capital of France is Paris."],
        answer="The capital of France is Paris.",
        ground_truth="Lyon is a city in the Rhone valley known for silk weaving.",
        k=1,
    )
    assert bad.metrics["answer_correctness"] < 0.5
    assert any("differs from ground_truth" in failure for failure in bad.failures)


def test_a_blank_answer_is_excluded_and_noted_rather_than_scored_zero():
    case = evaluate_case("q", ["p"], answer="   ", k=1)
    assert "groundedness" not in case.metrics
    assert "groundedness" in case.excluded
    assert any("'answer' is blank" in note for note in case.notes)


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------

def test_embed_replaces_lexical_overlap_and_is_reported_as_such():
    vectors = {
        "reset my password": [1.0, 0.0],
        "changing your login credentials": [1.0, 0.0],
        "monthly invoices and billing": [0.0, 1.0],
    }
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.array([vectors.get(text, [0.5, 0.5]) for text in texts])

    report = evaluate(
        [
            {
                "query": "reset my password",
                "retrieved": ["changing your login credentials", "monthly invoices and billing"],
            }
        ],
        k=2,
        embed=embed,
    )
    case = report.per_case[0]
    assert report.similarity == "embeddings"
    # lexical overlap would score the first passage 0; the embedding scores it 1
    assert case.metrics["hit_rate"] == 1.0
    assert case.metrics["mrr"] == 1.0
    assert len(calls) == 1, "the whole run must be one batched embed call"
    assert "similarity by embeddings" in report.summary()


def test_embed_reaches_evaluate_case_too():
    case = evaluate_case(
        "alpha",
        ["beta"],
        answer="beta",
        k=1,
        embed=lambda texts: np.ones((len(texts), 3)),
    )
    assert case.metrics["hit_rate"] == 1.0
    assert case.metrics["groundedness"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# input shapes and validation
# --------------------------------------------------------------------------

def test_retrieved_accepts_strings_dicts_and_plain_ids():
    case = evaluate_case(
        "q",
        ["plain string", {"id": 7, "text": "from a dict"}, 42],
        relevant=[7, 42],
        k=3,
    )
    assert case.n_unique == 3
    assert case.metrics["precision_at_k"] == pytest.approx(2 / 3)


def test_graded_relevance_mapping_is_used_for_ndcg():
    ranked_well = evaluate_case("q", ["a", "b"], relevant={"a": 3.0, "b": 1.0}, k=2)
    ranked_badly = evaluate_case("q", ["b", "a"], relevant={"a": 3.0, "b": 1.0}, k=2)
    assert ranked_well.metrics["ndcg_at_k"] == pytest.approx(1.0)
    assert ranked_badly.metrics["ndcg_at_k"] < 1.0


def test_negative_and_nan_gains_are_clipped_to_zero():
    case = evaluate_case("q", ["a", "b"], relevant={"a": -2.0, "b": float("nan")}, k=2)
    assert case.n_relevant == 0
    assert "recall_at_k" in case.excluded


def test_an_empty_case_list_is_an_empty_report_not_an_error():
    report = evaluate([])
    assert report.n_cases == 0
    assert report.metrics == {}
    assert report.per_case == []
    assert report.ok is True
    assert "no cases were given" in " ".join(report.notes)
    assert "no metric could be computed" in report.summary()


def test_a_blank_query_is_noted_rather_than_raising():
    report = evaluate([{"query": "", "retrieved": ["something"], "relevant": ["something"]}], k=1)
    assert report.per_case[0].metrics["hit_rate"] == 1.0
    assert any("the query is empty" in note for note in report.notes)


def test_a_missing_retrieved_key_is_treated_as_nothing_retrieved():
    report = evaluate([{"query": "q"}], k=1)
    assert report.per_case[0].n_retrieved == 0
    assert any("retrieved nothing" in failure for failure in report.failures)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 0}, "at least 1"),
        ({"k": -3}, "at least 1"),
        ({"k": 2.5}, "must be an int"),
        ({"k": True}, "must be an int"),
        ({"threshold": 1.5}, "must be in 0..1"),
        ({"threshold": -0.1}, "must be in 0..1"),
        ({"threshold": "high"}, "number in 0..1"),
    ],
)
def test_bad_parameters_raise_a_message_that_names_the_parameter(kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluate(QUICKSTART, **kwargs)


def test_a_single_case_dict_is_rejected_with_the_fix_in_the_message():
    with pytest.raises(ValueError, match=r"wrap it: evaluate\(\[case\]\)"):
        evaluate({"query": "q", "retrieved": []})


def test_obvious_shape_mistakes_name_what_was_wrong():
    with pytest.raises(ValueError, match="list of case dicts"):
        evaluate("cases.json")
    with pytest.raises(ValueError, match="must be a dict"):
        evaluate([["query", "retrieved"]])
    with pytest.raises(ValueError, match="not a single string"):
        evaluate([{"query": "q", "retrieved": "one passage"}])
    with pytest.raises(ValueError, match="must be a list"):
        evaluate([{"query": "q", "retrieved": 5}])
    with pytest.raises(ValueError, match="not a single string"):
        evaluate([{"query": "q", "retrieved": ["a"], "relevant": "a"}])
    with pytest.raises(ValueError, match="list of ids"):
        evaluate([{"query": "q", "retrieved": ["a"], "relevant": 7}])
    with pytest.raises(ValueError, match="not a number"):
        evaluate([{"query": "q", "retrieved": ["a"], "relevant": {"a": "high"}}])
    with pytest.raises(ValueError, match="must be hashable"):
        evaluate([{"query": "q", "retrieved": [{"id": ["a"], "text": "t"}]}])
    with pytest.raises(ValueError, match="strings, mappings or ids"):
        evaluate([{"query": "q", "retrieved": [["a", "b"]]}])


# --------------------------------------------------------------------------
# result objects
# --------------------------------------------------------------------------

def test_case_result_to_dict_is_json_safe_and_complete():
    case = evaluate_case("q", ["p"], relevant=["p"], answer="p", k=1)
    payload = case.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["query"] == "q"
    for key in ("index", "k", "threshold", "n_retrieved", "n_unique", "score",
                "metrics", "excluded", "notes", "failures", "estimated"):
        assert key in payload


def test_case_score_is_the_mean_of_what_was_measurable():
    case = evaluate_case("q", ["p"], relevant=["p"], k=1)
    assert case.score == pytest.approx(
        sum(case.metrics.values()) / len(case.metrics)
    )
    empty = evaluate([])
    assert empty.weakest() == []


def test_to_frame_gives_one_row_per_case_with_nan_for_excluded_metrics():
    pd = pytest.importorskip("pandas")
    report = evaluate(
        [
            {"query": "a", "retrieved": ["a"], "relevant": ["a"]},
            {"query": "b", "retrieved": ["b"], "relevant": []},
        ],
        k=1,
    )
    frame = report.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert list(frame["index"]) == [0, 1]
    assert frame["recall_at_k"].isna().tolist() == [False, True]
    # the frame mean agrees with the reported mean instead of quietly disagreeing
    assert frame["recall_at_k"].mean() == pytest.approx(report.metrics["recall_at_k"])


def test_to_frame_of_an_empty_report_has_the_columns_and_no_rows():
    pytest.importorskip("pandas")
    frame = evaluate([]).to_frame()
    assert len(frame) == 0
    assert "query" in frame.columns


# --------------------------------------------------------------------------
# regressions: numbers that used to be wrong, or right numbers that used to
# be accused of being failures
# --------------------------------------------------------------------------

IDS_WITH_ANSWERS = [
    {"query": "reset my password", "retrieved": ["p_reset", "p_billing"],
     "relevant": ["p_reset"],
     "answer": "To reset your password, open Settings and choose Reset password."},
    {"query": "cancel my plan", "retrieved": ["p_cancel", "p_faq"],
     "relevant": ["p_cancel"],
     "answer": "Open Billing and choose Cancel plan."},
    {"query": "what is the refund window", "retrieved": ["p_refund"],
     "relevant": ["p_refund"],
     "answer": "Refunds are accepted within 30 days."},
]


def test_ids_without_passage_text_exclude_the_answer_metrics_instead_of_crying_hallucination():
    report = evaluate(IDS_WITH_ANSWERS, k=2)
    # retrieval is perfect and must stay perfect
    assert report.metrics["hit_rate"] == pytest.approx(1.0)
    assert report.metrics["mrr"] == pytest.approx(1.0)
    # three correct answers, and not one of them called a hallucination
    assert report.failures == []
    for case in report.per_case:
        assert "groundedness" not in case.metrics
        assert "citation_coverage" not in case.metrics
        assert "groundedness" in case.excluded
        assert "citation_coverage" in case.excluded
        assert any("no passage text" in note or "not passage text" in note
                   for note in case.notes)
    assert "groundedness" not in report.metrics
    assert any("only ids" in note for note in report.notes)


def test_passages_that_do_carry_text_are_still_measured_beside_bare_ids():
    case = evaluate_case(
        "how do I reset my password",
        ["p_billing", {"id": "p_reset",
                       "text": "To reset your password, open Settings and choose Reset."}],
        relevant=["p_reset"],
        answer="Open Settings and choose Reset.",
        k=2,
    )
    assert case.metrics["groundedness"] == pytest.approx(1.0)
    # the id-only item cannot be cited, so it is out of the denominator too
    assert case.metrics["citation_coverage"] == pytest.approx(1.0)
    assert any("1 of 2 retrieved item(s) carry an id" in note for note in case.notes)


def test_an_empty_retrieved_list_still_scores_the_answer_metrics_zero():
    # nothing retrieved is a real 0, not an unknown: there is no context at all
    case = evaluate_case("q", [], answer="Some answer.", k=1)
    assert case.metrics["groundedness"] == pytest.approx(0.0)
    assert case.metrics["citation_coverage"] == pytest.approx(0.0)
    assert any("answer not grounded" in failure for failure in case.failures)


def test_a_correct_grounded_answer_raises_no_failure_at_all():
    # the README's own API example
    case = evaluate_case(
        "how do I reset my password",
        ["To reset your password, open Settings and choose Reset."],
        answer="Open Settings and choose Reset.",
        k=1,
    )
    assert case.metrics["groundedness"] == pytest.approx(1.0)
    assert case.metrics["citation_coverage"] == pytest.approx(1.0)
    # the query keeps 3 content words (reset, my, password) and the answer
    # repeats one: a correct answer does not have to echo the question
    assert case.metrics["answer_relevance"] == pytest.approx(1 / 3)
    assert case.failures == []
    assert case.ok is True


def test_a_right_answer_that_does_not_echo_the_query_is_noted_not_failed():
    case = evaluate_case(
        "how much does the pro plan cost",
        [{"id": "p1", "text": "The Pro plan is billed at 29 dollars per seat each month."}],
        relevant=["p1"],
        answer="It is 29 dollars per seat per month.",
        k=1,
    )
    assert case.metrics["groundedness"] == pytest.approx(1.0)
    assert case.metrics["answer_relevance"] == pytest.approx(0.0)
    assert case.failures == []
    assert any("answer_relevance is 0.00" in note for note in case.notes)


def test_answer_relevance_still_fails_a_case_when_embeddings_back_it():
    vectors = {
        "where is my data stored": [1.0, 0.0],
        "data lives in the Frankfurt region": [1.0, 0.0],
        "Pineapple belongs on nothing.": [0.0, 1.0],
    }
    report = evaluate(
        [{"query": "where is my data stored",
          "retrieved": ["data lives in the Frankfurt region"],
          "answer": "Pineapple belongs on nothing."}],
        k=1,
        embed=lambda texts: np.array([vectors.get(text, [0.0, 1.0]) for text in texts]),
    )
    assert report.similarity == "embeddings"
    assert report.per_case[0].metrics["answer_relevance"] == pytest.approx(0.0)
    assert any("does not address the query" in failure for failure in report.failures)


def test_a_one_word_sentence_does_not_halve_groundedness():
    case = evaluate_case(
        "can I use it offline",
        [{"id": "p1",
          "text": "The desktop client keeps working without a network connection and syncs later."}],
        relevant=["p1"],
        answer="Yes. The desktop client keeps working without a network connection and syncs later.",
        k=1,
    )
    # 9 supported content words out of 10: "yes" is one word, not half the answer
    assert case.metrics["groundedness"] == pytest.approx(0.9)
    assert case.failures == []


def test_groundedness_falls_back_to_equal_weights_when_no_sentence_has_words():
    case = evaluate_case(
        "punctuation",
        ["some real passage text about nothing"],
        answer="!!! ???",
        k=1,
    )
    assert case.metrics["groundedness"] == pytest.approx(0.0)


def test_an_unanswerable_query_is_excluded_rather_than_reported_as_a_failure():
    case = evaluate_case("unanswerable query", ["a", "b"], relevant=[], k=2)
    assert case.metrics == {}
    assert case.failures == []
    for name in RETRIEVAL_METRICS:
        assert name in case.excluded
    assert any("no relevant ids at all" in note for note in case.notes)


def test_an_unanswerable_query_that_retrieved_nothing_is_not_a_failure():
    case = evaluate_case("who is the king of france", [], relevant=[], k=2)
    assert case.failures == []
    assert case.ok is True
    assert any("wanted outcome" in note for note in case.notes)


def test_unanswerable_cases_do_not_drag_the_aggregate_down():
    report = evaluate(
        [
            {"query": "refund window", "retrieved": ["p_refund"], "relevant": ["p_refund"]},
            {"query": "who is the king of france", "retrieved": ["p_faq"], "relevant": []},
        ],
        k=1,
    )
    assert report.metrics["hit_rate"] == pytest.approx(1.0)
    assert report.metrics["precision_at_k"] == pytest.approx(1.0)
    assert report.support["hit_rate"] == 1
    assert report.failures == []


def test_weakest_skips_a_case_where_nothing_could_be_measured():
    report = evaluate(
        [
            {"query": "refund window", "retrieved": ["p_refund"], "relevant": ["p_refund"]},
            {"query": "who is the king of france", "retrieved": ["p_faq"], "relevant": []},
        ],
        k=1,
    )
    # the unanswerable case scores 0.0 only because it has no metric at all;
    # it is not the weakest case, it is the unmeasured one
    assert [case.index for case in report.weakest(5)] == [0]
    assert "who is the king of france" not in report.summary().split("weakest cases:")[-1]


def test_an_id_space_mismatch_gets_its_own_run_level_note():
    report = evaluate(
        [
            {"query": "q{0}".format(i),
             "retrieved": [{"id": "doc_{0}".format(i), "text": "body {0}".format(i)}],
             "relevant": ["DOC-{0}".format(i)]}
            for i in range(5)
        ],
        k=1,
    )
    assert report.metrics["hit_rate"] == pytest.approx(0.0)
    assert any("same id form" in note for note in report.notes)


def test_no_id_mismatch_note_when_at_least_one_case_lines_up():
    report = evaluate(QUICKSTART, k=2)
    assert not any("same id form" in note for note in report.notes)


def test_the_public_surface_is_importable_and_versioned():
    assert rag_quality_check.__version__ == "0.1.0"
    for name in rag_quality_check.__all__:
        assert hasattr(rag_quality_check, name), name
    assert set(RETRIEVAL_METRICS) == {
        "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate"
    }
    assert "groundedness" in ANSWER_METRICS
    for name in RETRIEVAL_METRICS + ANSWER_METRICS:
        assert name in rag_quality_check.METRIC_HELP
