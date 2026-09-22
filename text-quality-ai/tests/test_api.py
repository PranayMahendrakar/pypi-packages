"""The public API: score, the report object and the scorer class."""
import json

import pytest

import text_quality_ai
from text_quality_ai import Issue, QualityReport, TextScorer, available_targets, score

GOOD = (
    "Good writing is easy to read. It says one thing at a time, and it says it plainly. "
    "Short sentences help. Longer ones are fine too, as long as they carry their weight "
    "and never wander off. However, a paragraph that never pauses will lose the reader. "
    "So give them a break."
)
BAD = (
    "It should be noted that the implementation of the solution was undertaken by the team. "
    "The evaluation of the results was subsequently performed by the team. "
    "The documentation of the findings was then completed by the team."
)


def test_score_returns_a_report_with_everything_promised():
    report = score(GOOD)
    assert isinstance(report, QualityReport)
    assert 0.0 <= report.score <= 100.0
    assert report.grade in {"A", "B", "C", "D", "F"}
    assert set(report.components) == {"readability", "repetition", "structure", "clarity", "vocabulary"}
    assert all(0.0 <= v <= 100.0 for v in report.components.values())
    for key in ("words", "sentences", "paragraphs", "syllables", "reading_time_seconds"):
        assert key in report.stats
    assert report.suggestions
    assert report.documents is None


def test_every_measure_explains_itself():
    report = score(BAD)
    for name in report.components:
        explanation = report.explain(name)
        assert isinstance(explanation, str) and len(explanation) > 20
        assert report.measures[name]["score"] == report.components[name]
    with pytest.raises(KeyError):
        report.explain("nonsense")


def test_good_text_beats_bad_text():
    assert score(GOOD).score > score(BAD).score
    assert score(BAD).grade in {"C", "D", "F"}
    assert score(GOOD).grade in {"A", "B"}


def test_issues_carry_examples_and_counts():
    report = score(BAD)
    kinds = {issue.kind for issue in report.issues}
    assert "passive_voice" in kinds
    for issue in report.issues:
        assert isinstance(issue, Issue)
        assert issue.severity in {"low", "medium", "high"}
        assert isinstance(issue.examples, list)
        assert issue.count >= 0
        assert issue.message
    passive = next(i for i in report.issues if i.kind == "passive_voice")
    assert passive.count == 3
    assert passive.examples


def test_suggestions_are_ordered_by_how_much_they_help():
    report = score(BAD)
    assert len(report.suggestions) == len(report.issues)
    assert "passive" in report.suggestions[0].lower()


def test_no_issues_still_gives_a_suggestion():
    report = score(GOOD)
    assert report.suggestions


def test_to_dict_is_json_safe_and_round_trips():
    report = score(BAD)
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text)["grade"] == report.grade
    assert payload["components"] == {k: round(v, 1) for k, v in report.components.items()}
    assert json.loads(report.to_json())["score"] == report.score


def test_summary_is_plain_ascii_punctuation():
    text = score(BAD).summary()
    assert "Text quality:" in text
    assert "What to fix first:" in text
    for forbidden in ("→", "•", "─", "—", "“"):
        assert forbidden not in text


def test_targets_shift_the_ideal_grade():
    plain = "The cat sat on the mat. The dog ran to the park. We had a good day."
    simple = score(plain, target="simple")
    academic = score(plain, target="academic")
    assert simple.components["readability"] > academic.components["readability"]
    assert set(available_targets()) == {"general", "academic", "marketing", "technical", "simple"}
    for target in available_targets():
        assert 0.0 <= score(plain, target=target).score <= 100.0


def test_unknown_target_is_rejected_clearly():
    with pytest.raises(ValueError) as excinfo:
        score("hello", target="casual")
    assert "casual" in str(excinfo.value) and "general" in str(excinfo.value)


def test_list_input_gives_one_report_per_document():
    report = score([GOOD, BAD])
    assert report.documents is not None and len(report.documents) == 2
    assert report.stats["documents"] == 2
    assert report.stats["words"] == sum(d.stats["words"] for d in report.documents)
    assert 0.0 <= report.score <= 100.0
    assert report.documents[0].score > report.documents[1].score
    assert any("document(s)" in issue.message for issue in report.issues)
    assert "documents" in report.to_dict()


def test_empty_list_is_a_zero_word_report():
    report = score([])
    assert report.documents == []
    assert report.stats["words"] == 0
    assert report.score == 0.0


def test_list_rejects_non_strings():
    with pytest.raises(TypeError) as excinfo:
        score(["fine", 7])
    assert "item 1" in str(excinfo.value)


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        score(7)


def test_scorer_class_is_reusable_and_configurable():
    scorer = TextScorer("technical", long_sentence_words=12)
    first = scorer.score_one(BAD)
    second = scorer.score_one(BAD)
    assert first.score == second.score
    assert first.measures["clarity"]["long_sentence_words"] == 12
    assert first.measures["clarity"]["long_sentences"] >= 1
    with pytest.raises(ValueError):
        TextScorer("general", long_sentence_words=2)


def test_weakest_names_the_lowest_component():
    report = score(BAD)
    assert report.weakest == min(report.components, key=lambda k: report.components[k])


def test_version_is_exposed():
    assert text_quality_ai.__version__ == "0.1.0"
