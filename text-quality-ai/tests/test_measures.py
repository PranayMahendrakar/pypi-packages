"""The individual measures and the helper functions around them."""
import pytest

from text_quality_ai import compare, readability, repetition, score
from text_quality_ai._measures import band, clip, moving_average_ttr

PASSIVE = ("The report was written by the committee. The findings were reviewed by the board. "
           "The decision was taken by the chair. The minutes were circulated by the clerk.")
ACTIVE = ("The committee wrote the report. The board reviewed the findings. "
          "The chair took the decision. The clerk circulated the minutes.")


# -- readability ------------------------------------------------------------

def test_readability_reports_every_promised_number():
    result = readability("The cat sat on the mat. A dog ran past the gate. Birds sang all day.")
    for key in ("flesch_reading_ease", "flesch_kincaid_grade", "smog_index",
                "avg_sentence_length", "avg_word_length", "score", "explanation"):
        assert key in result
    assert 0.0 <= result["score"] <= 100.0
    assert result["avg_sentence_length"] == pytest.approx(5.3, abs=0.1)


def test_readability_tracks_difficulty():
    easy = readability("The cat sat. The dog ran. We went home.")
    hard = readability(
        "The epistemological ramifications of the aforementioned methodological "
        "considerations necessitate a comprehensive reconsideration of the "
        "theoretical underpinnings that characterise contemporary scholarship."
    )
    assert hard["flesch_kincaid_grade"] > easy["flesch_kincaid_grade"]
    assert hard["flesch_reading_ease"] < easy["flesch_reading_ease"]
    assert hard["polysyllable_words"] > easy["polysyllable_words"]


def test_readability_target_changes_the_score_not_the_numbers():
    text = "The epistemological ramifications require a comprehensive reconsideration of scholarship."
    general = readability(text, target="general")
    academic = readability(text, target="academic")
    assert general["flesch_kincaid_grade"] == academic["flesch_kincaid_grade"]
    assert academic["score"] > general["score"]
    assert academic["target_grade"] == 14.0


def test_readability_rejects_a_bad_target():
    with pytest.raises(ValueError):
        readability("hello", target="chatty")


# -- repetition -------------------------------------------------------------

def test_repetition_finds_repeated_words():
    result = repetition("Widgets are great. Widgets are cheap. Widgets are everywhere. Buy widgets now.")
    words = {entry["word"]: entry["count"] for entry in result["repeated_words"]}
    assert words["widgets"] == 4
    assert result["score"] < 100.0
    assert "widgets" in result["explanation"]


def test_repetition_finds_repeated_openers():
    text = ("The team shipped it. The board liked it. The users found it. "
            "The press covered it. The rivals copied it.")
    result = repetition(text)
    openers = {entry["word"]: entry["count"] for entry in result["repeated_openers"]}
    assert openers["the"] == 5
    assert result["opener_repetition_rate"] > 0.5


def test_repetition_finds_bigrams_and_trigrams():
    result = repetition("We ship fast and we ship well. We ship fast every day. We ship fast always.")
    trigrams = {entry["phrase"] for entry in result["repeated_trigrams"]}
    assert "we ship fast" in trigrams
    bigrams = {entry["phrase"] for entry in result["repeated_bigrams"]}
    assert "we ship" in bigrams


def test_repetition_ignores_pure_stopword_phrases():
    result = repetition("It is in the box. It is in the bag. It is in the car. It is in the van.")
    for entry in result["repeated_bigrams"] + result["repeated_trigrams"]:
        assert entry["phrase"] not in {"is in", "in the", "it is in", "is in the"}


def test_varied_text_has_no_repetition_penalty():
    result = repetition(
        "Rain fell across the valley. Sheep wandered between hedgerows. "
        "A kestrel hung above the ridge, waiting. Evening came quickly."
    )
    assert result["score"] == 100.0
    assert "Little repetition" in result["explanation"]


# -- structure --------------------------------------------------------------

def test_structure_flags_uniform_sentence_length():
    uniform = score("One two three four. Five six seven eight. Nine ten more words. Here are four words.")
    assert uniform.measures["structure"]["sentence_length_variation"] < 0.2
    assert any(issue.kind == "uniform_sentence_length" for issue in uniform.issues)


def test_structure_counts_transitions():
    report = score(
        "Sales rose. However, costs rose faster. For example, freight doubled. "
        "Therefore margin fell. Meanwhile the team grew. In addition, rent went up."
    )
    assert report.measures["structure"]["transition_sentences"] >= 4
    assert report.measures["structure"]["transition_share"] > 0.5


def test_structure_flags_a_very_long_paragraph():
    long_paragraph = ("The quarterly figures moved again this month and the reasons vary widely. " * 40)
    report = score(long_paragraph)
    assert report.measures["structure"]["longest_paragraph_words"] > 200
    assert any(issue.kind == "long_paragraphs" for issue in report.issues)


def test_structure_balances_paragraphs():
    report = score("First part here, short and tidy.\n\nSecond part here, also short and tidy.")
    assert report.measures["structure"]["paragraphs"] == 2
    assert 0.0 <= report.measures["structure"]["paragraph_share_of_longest"] <= 1.0


# -- clarity ----------------------------------------------------------------

def test_clarity_finds_passive_voice():
    passive = score(PASSIVE)
    active = score(ACTIVE)
    assert passive.measures["clarity"]["passive_sentences"] == 4
    assert active.measures["clarity"]["passive_sentences"] == 0
    assert active.components["clarity"] > passive.components["clarity"]
    assert passive.measures["clarity"]["passive_examples"]


def test_clarity_finds_fillers_hedges_and_nominalisations():
    report = score(
        "It should be noted that this is really very simply a basically complete waste. "
        "Perhaps the implementation of the configuration might possibly be an improvement. "
        "The determination of the specification was maybe a consideration."
    )
    clarity = report.measures["clarity"]
    assert clarity["filler_count"] >= 4
    assert clarity["hedge_count"] >= 3
    assert clarity["nominalisation_count"] >= 4
    kinds = {issue.kind for issue in report.issues}
    assert {"filler_words", "hedging", "nominalisations"} <= kinds


def test_clarity_flags_long_sentences_with_examples():
    long_sentence = " ".join(["word"] * 45) + ". Short one here."
    report = score(long_sentence)
    clarity = report.measures["clarity"]
    assert clarity["long_sentences"] == 1
    assert clarity["long_sentence_examples"]
    assert clarity["long_sentence_examples"][0].endswith("...")
    assert any(issue.kind == "long_sentences" for issue in report.issues)


# -- vocabulary -------------------------------------------------------------

def test_vocabulary_reports_type_token_ratio():
    report = score("alpha beta gamma delta epsilon zeta eta theta iota kappa.")
    vocabulary = report.measures["vocabulary"]
    assert vocabulary["type_token_ratio"] == 1.0
    assert vocabulary["unique_words"] == 10


def test_vocabulary_flags_a_narrow_range():
    report = score(("the cat sat on the mat and the cat sat on the mat again " * 8))
    assert report.measures["vocabulary"]["moving_average_ttr"] < 0.45
    assert any(issue.kind == "narrow_vocabulary" for issue in report.issues)


def test_vocabulary_flags_long_words_against_the_target():
    text = ("Organisational transformation initiatives necessitate comprehensive "
            "communication strategies throughout implementation programmes, "
            "particularly where accountability considerations predominate. "
            "Subsequent optimisation requires additional stakeholder participation "
            "alongside systematic documentation of operational dependencies.")
    report = score(text, target="simple")
    assert report.measures["vocabulary"]["long_word_share"] > 0.4
    assert any(issue.kind == "long_words" for issue in report.issues)


def test_moving_average_ttr_is_length_independent():
    short = ["a", "b", "c", "d"]
    assert moving_average_ttr(short) == 1.0
    repeated = ["a", "b"] * 200
    assert moving_average_ttr(repeated) == pytest.approx(0.02, abs=0.005)
    assert moving_average_ttr([]) == 0.0


# -- compare ----------------------------------------------------------------

def test_compare_reports_deltas_in_both_directions():
    forward = compare(PASSIVE, ACTIVE)
    backward = compare(ACTIVE, PASSIVE)
    assert forward["score"] > 0 and forward["better"] == "b"
    assert backward["score"] < 0 and backward["better"] == "a"
    assert forward["score"] == pytest.approx(-backward["score"], abs=0.1)
    assert forward["components"]["clarity"] > 0
    assert set(forward["stats"]) == {"words", "sentences", "paragraphs", "syllables", "reading_time_seconds"}
    assert forward["a"]["grade"] and forward["b"]["grade"]
    assert "higher" in forward["summary"]


def test_compare_of_identical_text_is_a_tie():
    result = compare(ACTIVE, ACTIVE)
    assert result["score"] == 0.0
    assert result["better"] == "tie"
    assert all(value == 0.0 for value in result["components"].values())


def test_compare_accepts_lists():
    result = compare([ACTIVE, ACTIVE], [PASSIVE, PASSIVE])
    assert result["score"] < 0


# -- helpers ----------------------------------------------------------------

def test_band_and_clip_behave():
    assert band(5.0, 4.0, 6.0, 10.0) == 100.0
    assert band(9.0, 4.0, 6.0, 10.0) == pytest.approx(70.0)
    assert band(100.0, 4.0, 6.0, 10.0) == 0.0
    assert clip(150.0) == 100.0 and clip(-3.0) == 0.0
    assert clip(float("nan")) == 0.0


# --- regression: the nominalisation check used to fire on ordinary prose ----------------

_CLEAN = (
    "The distance between the two cities is considerable. Her kindness impressed everyone. "
    "The quality of the equipment matters. His patience was remarkable. "
    "The balance of the system is delicate. Tourism drives the local economy. "
    "The silence in the audience was total and the science was sound."
)
_BUREAUCRATIC = (
    "The implementation of the utilisation assessment required consideration of the "
    "documentation, and compliance with the specification remained a requirement."
)


def _nominalisation_issues(text):
    from text_quality_ai import score

    return [i for i in score(text).issues if "nominal" in (i.kind + i.message).lower()]


def test_ordinary_nouns_are_not_called_nominalisations():
    """distance, kindness, quality, patience, balance, tourism, silence, audience and
    science are not built out of verbs. -ness and -ity build nouns from ADJECTIVES and
    -ism from nouns, so matching those suffixes could only ever fire on clean writing."""
    assert not _nominalisation_issues(_CLEAN), "clean prose must not be flagged"


def test_real_nominalisations_are_still_caught():
    issues = _nominalisation_issues(_BUREAUCRATIC)
    assert issues, "genuinely bureaucratic prose must still be flagged"
    found = " ".join(" ".join(i.examples) for i in issues)
    for word in ("implementation", "utilisation", "assessment", "consideration"):
        assert word in found, f"{word} should have been reported"


def test_ance_and_ence_need_a_verb_behind_them():
    from text_quality_ai._measures import _nominalisations

    # verb-derived: these really are nouns made from verbs
    for word in ("performance", "acceptance", "dependence", "maintenance", "compliance"):
        assert _nominalisations([word]), f"{word} is verb-derived and should count"
    # not verb-derived: no verb "dist", "pati", "sci" or "sent" lies behind these
    for word in ("distance", "patience", "science", "sentence", "audience", "experience"):
        assert not _nominalisations([word]), f"{word} is not built from a verb"
