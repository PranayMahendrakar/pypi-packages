"""Edge cases: empty text, one word, no punctuation, unicode, size and bounds."""
import random
import time

import pytest

import text_quality_ai
from text_quality_ai import available_targets, compare, readability, repetition, score

ACCENTED = "Le cafe etait tres chaud. Naive resume facade. Uber grosser strasse."
ACCENTED_REAL = "Le café était très chaud. Naïve résumé façade. Über größer straße."
DEVANAGARI = "यह एक परीक्षण है। " \
             "हिंदी में लिखा गया " \
             "वाक्य। नमस्ते दुनिया।"
CHINESE = "这是一个测试。中文没有空格。" \
          "我们需要按字符计算。"
JAPANESE = "日本語のテキストです。これはテストです。"
KOREAN = "이것은 테스트입니다. 한국어 문장입니다."
EMOJI = "Great work \U0001F44D that was fun \U0001F389"

SAMPLES = [
    "", "   ", "\n\n\t  \n", "Word", "hello world",
    "no ending punctuation here at all",
    "...", "!!! ??? ---", "1234 5678 9012.",
    ACCENTED, ACCENTED_REAL, DEVANAGARI, CHINESE, JAPANESE, KOREAN, EMOJI,
    "Mixed 世界 scripts नमस्ते and café here!",
    "word " * 300,
    "A. B. C. D. E.",
    "Dr. Smith went to Washington. He met Mr. Jones at 3.5 p.m.",
]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n", "!!! ??? ---"])
def test_empty_and_wordless_text_gives_a_zero_word_report(text):
    report = score(text)
    assert report.stats["words"] == 0
    assert report.stats["sentences"] == 0
    assert report.stats["paragraphs"] == 0
    assert report.stats["syllables"] == 0
    assert report.stats["reading_time_seconds"] == 0.0
    assert report.score == 0.0 and report.grade == "F"
    assert all(value == 0.0 for value in report.components.values())
    assert [issue.kind for issue in report.issues] == ["empty_text"]
    assert report.suggestions
    assert report.summary()
    assert report.to_dict()["stats"]["words"] == 0


def test_empty_text_in_the_helper_functions():
    assert readability("")["flesch_kincaid_grade"] == 0.0
    assert repetition("")["repeated_words"] == []
    assert compare("", "")["score"] == 0.0


def test_a_single_word():
    report = score("Word")
    assert report.stats["words"] == 1
    assert report.stats["sentences"] == 1
    assert 0.0 <= report.score <= 100.0
    assert any(issue.kind == "very_short_text" for issue in report.issues)


def test_no_sentence_ending_punctuation_is_one_sentence():
    report = score("this is a run of words with no ending punctuation whatsoever")
    assert report.stats["sentences"] == 1
    assert report.stats["words"] == 11


def test_abbreviations_and_decimals_do_not_end_a_sentence():
    report = score("Dr. Smith paid 3.5 dollars at 4 p.m. He then left.")
    assert report.stats["sentences"] == 2


@pytest.mark.parametrize("text", [ACCENTED_REAL, DEVANAGARI, CHINESE, JAPANESE, KOREAN, EMOJI])
def test_unicode_text_counts_sensibly_and_never_crashes(text):
    report = score(text)
    assert report.stats["words"] > 0
    assert report.stats["syllables"] > 0
    assert 0.0 <= report.score <= 100.0
    assert report.summary()


def test_accented_latin_is_counted_like_latin():
    plain = score(ACCENTED)
    accented = score(ACCENTED_REAL)
    assert plain.stats["words"] == accented.stats["words"] == 11
    assert accented.stats["word_counting"] == "space"


def test_devanagari_words_stay_whole():
    report = score(DEVANAGARI)
    assert report.stats["word_counting"] == "space"
    assert report.stats["words"] == 11
    assert report.stats["sentences"] == 3
    assert report.stats["syllables"] >= report.stats["words"]


@pytest.mark.parametrize("text", [CHINESE, JAPANESE, KOREAN])
def test_space_less_scripts_fall_back_to_characters_and_say_so(text):
    report = score(text)
    assert report.stats["word_counting"] == "character"
    assert report.notes and "no spaces" in report.notes[0]
    # the whole run is not reported as one enormous word
    assert report.stats["words"] > len(text.split())
    assert "reference only" in report.explain("readability")


def test_syllable_counter_never_raises_on_odd_input():
    from text_quality_ai._text import syllables

    for word in ["", "a", "rhythm", "你", "नमस्ते",
                 "привет", "αβγ", "\U0001F44D", "x" * 200]:
        value = syllables(word)
        assert isinstance(value, int) and value >= (0 if word == "" else 1)


@pytest.mark.parametrize("text", SAMPLES)
def test_scores_stay_inside_zero_to_one_hundred(text):
    for target in available_targets():
        report = score(text, target=target)
        assert 0.0 <= report.score <= 100.0
        assert all(0.0 <= value <= 100.0 for value in report.components.values())
        assert report.grade in {"A", "B", "C", "D", "F"}


def test_a_200kb_document_scores_in_under_a_second():
    random.seed(0)
    vocabulary = ["analysis", "system", "result", "report", "method", "value", "process",
                  "change", "impact", "study", "the", "of", "and", "in", "to", "with",
                  "for", "from", "that", "this", "however", "therefore"]
    parts = []
    size = 0
    while size < 205_000:
        sentence = " ".join(random.choice(vocabulary) for _ in range(random.randint(8, 28)))
        parts.append(sentence.capitalize() + ".")
        size += len(sentence) + 2
    document = " ".join(parts)
    assert len(document.encode("utf-8")) > 200_000

    start = time.perf_counter()
    report = score(document)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"200 KB took {elapsed:.2f}s"
    assert report.stats["words"] > 10_000
    assert 0.0 <= report.score <= 100.0


def test_a_large_space_less_document_is_also_fast():
    document = "汉字测试。" * 8000
    start = time.perf_counter()
    report = score(document)
    assert time.perf_counter() - start < 1.0
    assert report.stats["word_counting"] == "character"


def test_whitespace_only_paragraphs_are_not_counted():
    report = score("First paragraph here.\n\n   \n\nSecond paragraph here.")
    assert report.stats["paragraphs"] == 2


def test_repeated_phrases_do_not_cross_sentence_boundaries():
    result = repetition("The team shipped it. The team shipped it. The team shipped it.")
    phrases = {entry["phrase"] for entry in result["repeated_trigrams"]}
    assert "the team shipped" in phrases
    assert "it the team" not in phrases


def test_scoring_is_deterministic():
    first = score(ACCENTED_REAL).to_dict()
    second = text_quality_ai.score(ACCENTED_REAL).to_dict()
    assert first == second


def test_a_short_quotation_does_not_switch_the_whole_document_to_characters():
    report = score(
        "The team shipped the release on Friday afternoon after a long week of testing. "
        "The sign on the door simply read 你好世界. Everyone went home happy and tired."
    )
    assert report.stats["word_counting"] == "space"
    assert not report.notes
