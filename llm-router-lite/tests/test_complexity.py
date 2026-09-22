"""The prompt scorer: bands, signals, determinism and odd input."""

import pytest

from llm_router_lite import (
    MODERATE_MAX,
    SIGNAL_WEIGHTS,
    SIMPLE_MAX,
    Complexity,
    band_for,
    complexity,
    estimate_tokens,
)
from llm_router_lite.complexity import count_words, non_latin_share

HARD_PROMPT = (
    "Explain step by step why this design scales, compare it with a "
    "queue-based approach, and then recommend one."
)

CHECKLIST_PROMPT = """Do all of the following:
- read the CSV
- clean the null rows
- compare the two designs
- explain the trade-offs
- write a summary"""


def test_weights_sum_to_one():
    assert round(sum(SIGNAL_WEIGHTS.values()), 10) == 1.0


def test_empty_prompt_is_simple_not_an_error():
    result = complexity("")
    assert result.score == 0.0
    assert result.band == "simple"
    assert result.words == 0
    assert result.characters == 0
    assert result.estimated_tokens == 0
    assert result.reasons == ()


def test_whitespace_only_prompt_is_simple():
    result = complexity("   \n\t  ")
    assert result.band == "simple"
    assert result.score < SIMPLE_MAX


def test_easy_prompt_is_simple():
    result = complexity("What is 2 + 2?")
    assert result.band == "simple"
    assert 0.0 < result.score < SIMPLE_MAX


def test_reasoning_prompt_is_hard():
    result = complexity(HARD_PROMPT)
    assert result.band == "hard"
    assert result.score >= MODERATE_MAX
    assert result.signals["reasoning"] == 1.0


def test_multi_instruction_prompt_is_hard():
    result = complexity(CHECKLIST_PROMPT)
    assert result.band == "hard"
    assert result.signals["instructions"] > 0.0


def test_all_three_bands_are_reachable():
    bands = {
        complexity("").band,
        complexity("Why does my code fail?").band,
        complexity(CHECKLIST_PROMPT).band,
    }
    assert bands == {"simple", "moderate", "hard"}


def test_code_markers_are_seen():
    result = complexity("Fix this:\n```\ndef f():\n    print(x)\n```")
    assert result.signals["code"] > 0.0


def test_score_stays_in_range_for_a_maximal_prompt():
    result = complexity((HARD_PROMPT + "\n" + CHECKLIST_PROMPT + "\n") * 20)
    assert 0.0 <= result.score <= 1.0
    assert result.band == "hard"


def test_scoring_is_deterministic():
    first = complexity(HARD_PROMPT)
    assert all(complexity(HARD_PROMPT) == first for _ in range(5))


def test_signals_are_the_documented_five():
    assert set(complexity("anything").signals) == set(SIGNAL_WEIGHTS)


def test_unicode_prompt_scores_without_crashing():
    result = complexity("Explique pourquoi le systeme tombe en panne, 説明してください, and then fix it 🚀")
    assert 0.0 <= result.score <= 1.0
    assert result.words > 0
    assert result.summary()


def test_band_for_boundaries():
    assert band_for(0.0) == "simple"
    assert band_for(SIMPLE_MAX - 0.001) == "simple"
    assert band_for(SIMPLE_MAX) == "moderate"
    assert band_for(MODERATE_MAX - 0.001) == "moderate"
    assert band_for(MODERATE_MAX) == "hard"
    assert band_for(1.0) == "hard"


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_summary_and_to_dict_are_usable():
    result = complexity(HARD_PROMPT)
    assert isinstance(result, Complexity)
    assert result.band in result.summary()
    payload = result.to_dict()
    assert payload["band"] == result.band
    assert set(payload["signals"]) == set(SIGNAL_WEIGHTS)
    assert str(result) == result.summary()


def test_empty_prompt_summary_says_so():
    assert "empty" in complexity("").summary()


def test_summary_is_plain_ascii():
    for prompt in ("", "What is 2 + 2?", HARD_PROMPT, CHECKLIST_PROMPT):
        complexity(prompt).summary().encode("ascii")


def test_non_string_prompt_raises_typeerror():
    with pytest.raises(TypeError) as excinfo:
        complexity(123)
    assert "int" in str(excinfo.value)


# ----------------------------------------------------------------- scripts
#
# Regression tests for the reviewer's major finding: the word regex was ASCII
# only and every marker list was English, so a prompt in any non-Latin script
# scored exactly 0.0, banded "simple", and summary() called a 78-character
# prompt empty.

SAME_REQUEST = {
    "english": (
        "Explain step by step why this distributed architecture scales, compare "
        "it with a queue-based design, analyze the trade-offs, prove the limits, "
        "and then recommend one."
    ),
    "japanese": (
        "この分散アーキテクチャがなぜスケールするのかを段階的に説明し、"
        "キューベースの設計と比較し、トレードオフを分析し、限界を証明し、"
        "そして一つを推奨してください。"
    ),
    "chinese": (
        "请逐步解释这个分布式架构为什么能扩展，"
        "与基于队列的设计进行比较，分析权衡，证明极限，然后推荐一个。"
    ),
    "korean": (
        "이 분산 아키텍처가 왜 확장되는지 단계별로 설명하고, "
        "큐 기반 설계와 비교하고, 분석한 다음 하나를 추천해 주세요."
    ),
    "russian": (
        "Объясни пошагово, почему эта распределённая архитектура масштабируется, "
        "сравни её с очередной схемой, проанализируй компромиссы, докажи пределы, "
        "затем порекомендуй один."
    ),
    "arabic": (
        "اشرح خطوة بخطوة لماذا تتوسع هذه البنية الموزعة، "
        "وقارنها بتصميم يعتمد على الطوابير، وحلل المقارنة، ثم أوص بواحد."
    ),
    "hindi": (
        "चरण दर चरण समझाइए कि यह वितरित आर्किटेक्चर क्यों स्केल करता है, "
        "कतार आधारित डिज़ाइन से तुलना कीजिए, और विश्लेषण कीजिए।"
    ),
    "greek": (
        "Εξήγησε βήμα βήμα γιατί κλιμακώνεται αυτή η κατανεμημένη αρχιτεκτονική, "
        "σύγκρινέ την με σχεδίαση βασισμένη σε ουρά, ανάλυσε τους συμβιβασμούς, "
        "απόδειξε τα όρια και μετά πρότεινε ένα."
    ),
    "hebrew": (
        "הסבר שלב אחר שלב למה הארכיטקטורה המבוזרת הזאת מתרחבת, "
        "השווה אותה לעיצוב מבוסס תור, ונתח את הפשרות, "
        "הוכח את הגבולות ולבסוף המלץ על אחד."
    ),
}


def test_the_same_hard_request_bands_hard_in_every_script_we_read():
    for language, prompt in SAME_REQUEST.items():
        result = complexity(prompt)
        assert result.band == "hard", (language, result.score)
        assert result.score >= MODERATE_MAX, (language, result.score)


def test_no_readable_script_scores_a_flat_zero():
    for language, prompt in SAME_REQUEST.items():
        result = complexity(prompt)
        assert result.score > 0.0, language
        assert result.words > 0, language
        assert result.signals["reasoning"] > 0.0, language


def test_words_are_counted_in_a_script_written_without_spaces():
    # Japanese has no spaces, so a naive split gives one "word" or none at all.
    result = complexity(SAME_REQUEST["japanese"])
    assert result.characters == 78
    assert result.words > 20


def test_words_are_counted_in_a_spaced_non_latin_script():
    assert complexity("почему это масштабируется").words == 3
    assert complexity("لماذا يتوسع هذا").words == 3


def test_summary_never_calls_a_non_empty_prompt_empty():
    for prompt in list(SAME_REQUEST.values()) + ["こんにちは", "🚀🚀🚀", "..."]:
        assert "the prompt is empty" not in complexity(prompt).summary()
    assert "the prompt is empty" in complexity("").summary()


def test_a_script_without_markers_warns_instead_of_scoring_zero_in_silence():
    thai = "อธิบายทีละขั้นตอนว่าทำไมสถาปัตยกรรมนี้จึงขยายตัวได้และเปรียบเทียบกับคิว"
    result = complexity(thai)
    assert result.warnings
    assert "script" in result.warnings[0]
    assert result.warnings[0] in result.summary()
    assert result.warnings[0] == result.to_dict()["warnings"][0]
    result.warnings[0].encode("ascii")


def test_text_that_matches_nothing_says_so_rather_than_passing_as_scored():
    result = complexity("🚀🚀🚀")
    assert result.score == 0.0
    assert result.warnings
    assert "0.00" in result.warnings[0]


def test_an_ordinary_prompt_carries_no_warning():
    for prompt in ("", "   ", "What is 2 + 2?", HARD_PROMPT, CHECKLIST_PROMPT):
        assert complexity(prompt).warnings == ()
    for language, prompt in SAME_REQUEST.items():
        assert complexity(prompt).warnings == (), language


def test_a_short_greeting_is_simple_in_any_script_and_needs_no_warning():
    for greeting in ("hello", "こんにちは", "привет", "مرحبا"):
        result = complexity(greeting)
        assert result.band == "simple"
        assert result.warnings == ()


def test_scoring_stays_deterministic_across_scripts():
    for prompt in SAME_REQUEST.values():
        first = complexity(prompt)
        assert all(complexity(prompt) == first for _ in range(3))


def test_non_latin_share_is_reported_honestly():
    assert non_latin_share("") == 0.0
    assert non_latin_share("plain ascii") == 0.0
    assert non_latin_share("説明してください") == 1.0
    assert 0.0 < non_latin_share("Explique 説明") < 1.0


def test_count_words_handles_mixed_scripts():
    assert count_words("") == 0
    assert count_words("two words") == 2
    assert count_words("hello 説明してください") == 1 + 4


def test_several_instructions_in_one_sentence_are_counted_separately():
    """Regression: instructions were counted per SENTENCE, so a prompt packing four
    requests behind one full stop scored as a single instruction and the signal - a
    fifth of the total weight - stayed at zero for exactly the prompts it exists for."""
    from llm_router_lite.complexity import _count_instructions

    prompt = ("Refactor this service so the retry logic is testable, then write "
              "property-based tests for the backoff, explain the trade-offs against a "
              "token bucket, and outline how you would migrate existing callers.")
    assert _count_instructions(prompt, prompt.lower()) >= 4


def test_a_multi_step_engineering_prompt_reaches_the_hard_band():
    from llm_router_lite import complexity

    prompt = ("Refactor this service so the retry logic is testable, then write "
              "property-based tests for the backoff, explain the trade-offs against a "
              "token bucket, and outline how you would migrate existing callers without "
              "downtime. Consider thread safety throughout.")
    assert complexity(prompt).band == "hard"


def test_a_trivial_prompt_stays_simple():
    from llm_router_lite import complexity

    for prompt in ("What is 2+2?", "Hello", "Summarise this in one line."):
        assert complexity(prompt).band == "simple", prompt
