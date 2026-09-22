"""Tokenizing, sentence splitting and the two similarity backends."""
from __future__ import annotations

import numpy as np
import pytest

from rag_quality_check import Similarity, cosine, coverage, split_sentences, tokenize
from rag_quality_check._text import looks_like_passage


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("How do I Reset the Password") == frozenset({"reset", "password"})


def test_tokenize_keeps_everything_when_a_query_is_all_stopwords():
    assert tokenize("what is it") == frozenset({"what", "is", "it"})


def test_tokenize_of_nothing_is_empty():
    assert tokenize("") == frozenset()
    assert tokenize("   ") == frozenset()
    assert tokenize("!!! ???") == frozenset()


def test_tokenize_splits_cjk_per_character():
    assert tokenize("東京都") == frozenset({"東", "京", "都"})


def test_tokenize_handles_accents_and_devanagari():
    assert "café" in tokenize("Le café est chaud")
    assert tokenize("पासवर्ड रीसेट") == frozenset({"पासवर्ड", "रीसेट"})


def test_tokenize_keeps_combining_marks_with_their_word():
    # \w does not match combining marks, so a naive \w+ cuts each of these
    # apart at its vowel signs. Every one of them must stay a single token.
    assert tokenize("رحلة") == frozenset({"رحلة"})
    assert tokenize("كَلِمَة") == frozenset({"كَلِمَة"})
    assert tokenize("รหัสผ่าน") == frozenset({"รหัสผ่าน"})
    assert tokenize("পাসওয়ার্ড") == frozenset({"পাসওয়ার্ড"})
    assert tokenize("שָׁלוֹם") == frozenset({"שָׁלוֹם"})


def test_a_word_and_its_marks_are_one_token_not_several():
    # The regression this guards: "पासवर्ड" once tokenized as प / सवर / ड.
    assert len(tokenize("पासवर्ड")) == 1


def test_cjk_per_character_split_keeps_marks_on_their_kana():
    # Decomposed か + U+3099 COMBINING VOICED SOUND MARK is one kana, not a
    # kana plus a stray mark. The length assertion keeps this from quietly
    # becoming the precomposed U+304C and testing nothing.
    decomposed = "が"
    assert len(decomposed) == 2
    assert tokenize(decomposed) == frozenset({decomposed})
    assert tokenize(decomposed + "ん") == frozenset({decomposed, "ん"})


def test_split_sentences_across_scripts():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("これは一だ。これは二だ。") == ["これは一だ。", "これは二だ。"]
    assert split_sentences("a\nb") == ["a", "b"]


def test_cjk_sentences_split_without_a_following_space():
    # Chinese and Japanese are not written with spaces after the full stop;
    # requiring one made a whole paragraph a single sentence, which silently
    # flattened groundedness.
    assert len(split_sentences("第一句。第二句。第三句。")) == 3
    assert len(split_sentences("本当？はい！そう。")) == 3


def test_a_decimal_point_does_not_end_a_latin_sentence():
    assert split_sentences("Pi is 3.14 today") == ["Pi is 3.14 today"]


def test_devanagari_and_urdu_sentence_enders_split_on_whitespace():
    assert split_sentences("यह एक है। यह दो है।") == ["यह एक है।", "यह दो है।"]


def test_split_sentences_always_gives_one_for_unpunctuated_text():
    assert split_sentences("no punctuation here") == ["no punctuation here"]
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_coverage_is_asymmetric_and_bounded():
    left = frozenset({"a", "b"})
    right = frozenset({"a", "b", "c", "d"})
    assert coverage(left, right) == pytest.approx(1.0)
    assert coverage(right, left) == pytest.approx(0.5)
    assert coverage(frozenset(), right) == 0.0


def test_cosine_is_clipped_into_zero_to_one():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == 0.0
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_lexical_similarity_reports_its_kind():
    sim = Similarity()
    assert sim.kind == "lexical overlap"
    assert sim.uses_embeddings is False
    assert sim.score("reset password", "to reset your password open settings") == 1.0
    assert sim.score("", "anything") == 0.0
    assert sim.best("reset password", []) == 0.0


def test_embedding_similarity_batches_and_caches():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.array([[float(len(text)), 1.0] for text in texts])

    sim = Similarity(embed)
    assert sim.kind == "embeddings"
    assert sim.uses_embeddings is True
    sim.warm(["alpha", "beta", "alpha"])
    assert calls == [["alpha", "beta"]]
    sim.score("alpha", "beta")
    assert len(calls) == 1, "already-embedded texts must not be embedded again"


def test_embed_must_be_callable_and_return_one_vector_per_text():
    with pytest.raises(ValueError, match="callable"):
        Similarity(embed="not a function")
    sim = Similarity(lambda texts: np.zeros((1, 3)))
    with pytest.raises(ValueError, match="one vector per input text"):
        sim.warm(["a", "b"])


def test_embed_of_a_single_text_may_return_a_flat_vector():
    sim = Similarity(lambda texts: np.array([1.0, 2.0, 3.0]))
    sim.warm(["only"])
    assert sim.score("only", "only") == pytest.approx(1.0)


# --------------------------------------------------------------------------
# telling a bare id apart from passage prose
# --------------------------------------------------------------------------

def test_bare_ids_are_not_passage_text():
    for ident in ("p_reset", "doc-42", "7f3c9b", "p1", "beta", "42",
                  "a1b2c3d4-e5f6-4711-8899-aabbccddeeff", "", "   "):
        assert looks_like_passage(ident) is False, ident


def test_prose_is_passage_text_in_every_script():
    assert looks_like_passage("To reset your password, open Settings.") is True
    assert looks_like_passage("le café est ouvert") is True
    assert looks_like_passage("पासवर्ड रीसेट करने के लिए") is True
    # written without spaces, and text all the same
    assert looks_like_passage("パスワードを再設定するには設定を開きます") is True
    assert looks_like_passage("設定を開きます") is True
    assert looks_like_passage("รหัสผ่านของคุณ") is True
    # nothing this long is an identifier
    assert looks_like_passage("x" * 41) is True
