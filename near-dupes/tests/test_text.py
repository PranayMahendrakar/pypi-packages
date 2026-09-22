import math

import numpy as np
import pytest

import near_dupes
from near_dupes import DuplicateFinder, DuplicateResult, find_duplicates, text_similarity
from near_dupes import _text


def edges(result):
    return {(i, j) for i, j, _ in result.pairs}


def test_identical_strings_form_one_group():
    r = find_duplicates(["same text here", "same text here", "same text here"])
    assert r.groups == [[0, 1, 2]]
    assert edges(r) == {(0, 1), (0, 2)}
    assert all(s == 1.0 for _, _, s in r.pairs)
    assert r.keep_indices == [0]
    assert r.n_duplicates == 2
    assert r.kind == "text"


def test_near_duplicate_is_found_with_exact_jaccard_score():
    a = "The quick brown fox jumps over the lazy dog."
    b = "the quick brown fox jumps over the lazy dog"
    r = find_duplicates([a, b, "Something else entirely."])
    assert r.groups == [[0, 1]]
    ((i, j, score),) = r.pairs
    assert (i, j) == (0, 1)
    assert score == pytest.approx(text_similarity(a, b))
    assert 0.85 <= score < 1.0
    assert r.representatives == [0]  # longest copy is kept
    assert r.dedupe([a, b, "Something else entirely."]) == [a, "Something else entirely."]


def test_case_whitespace_and_unicode_form_are_normalized():
    r = find_duplicates(["Hello   World", "hello world", "HELLO\tWORLD\n"])
    assert r.groups == [[0, 1, 2]]
    assert r.pairs[0][2] == 1.0


def test_normalize_false_compares_raw_strings():
    r = DuplicateFinder(normalize=False, threshold=1.0).find(["Hello", "hello", "Hello"])
    assert r.groups == [[0, 2]]


def test_threshold_one_means_exact_only():
    items = ["abc def ghi", "abc def ghi", "abc def ghj", "abc def ghi "]
    r = find_duplicates(items, threshold=1.0)
    assert r.groups == [[0, 1, 3]]  # trailing whitespace is normalized away
    assert all(s == 1.0 for _, _, s in r.pairs)
    low = find_duplicates(items, threshold=0.6)
    assert low.groups == [[0, 1, 2, 3]]


def test_strings_shorter_than_ngram():
    r = find_duplicates(["ab", "ab", "ac", "a", "", "x"], n_gram=3)
    assert r.groups == [[0, 1]]
    assert r.keep_indices == [0, 2, 3, 4, 5]
    assert find_duplicates(["a", "b"]).groups == []


def test_blank_items_never_group():
    r = find_duplicates(["", "  ", None, float("nan"), "", "real text", "real text"])
    assert r.groups == [[5, 6]]
    assert r.keep_indices == [0, 1, 2, 3, 4, 5]


def test_empty_input():
    r = find_duplicates([])
    assert isinstance(r, DuplicateResult)
    assert r.groups == [] and r.pairs == [] and r.keep_indices == []
    assert r.n_items == 0 and r.n_duplicates == 0
    assert r.dedupe([]) == []
    assert "nothing to compare" in r.summary()
    assert near_dupes.dedupe([]) == []


def test_single_item():
    r = find_duplicates(["only one"])
    assert r.groups == [] and r.keep_indices == [0]
    assert r.dedupe(["only one"]) == ["only one"]


def test_unicode_text():
    items = ["café au lait", "Café au lait", "日本語のテキストです", "日本語のテキストです。", "Ünïcödé", "unrelated"]
    r = find_duplicates(items, threshold=0.8)
    assert r.groups == [[0, 1], [2, 3]]
    assert r.dedupe(items) == ["café au lait", "日本語のテキストです。", "Ünïcödé", "unrelated"]


def test_indices_refer_to_original_positions(planted):
    filler, _ = planted(20, every=10_000)
    items = list(filler[:20])  # 20 distinct random strings, no planted pairs
    items[5] = "a near duplicate sentence appears here twice!"  # longest copy: kept
    items[17] = "a near duplicate sentence appears here twice"
    r = find_duplicates(items)
    assert r.groups == [[5, 17]]
    assert edges(r) == {(5, 17)}
    assert r.representatives == [5]
    assert 17 not in r.keep_indices and 5 in r.keep_indices
    assert r.drop_indices == [17]
    kept = r.dedupe(items)
    assert len(kept) == 19 and items[17] not in kept


def test_keep_longest_text_per_group():
    items = ["short copy of text", "short copy of text!!", "short copy of text"]
    r = find_duplicates(items, threshold=0.8)
    assert r.groups == [[0, 1, 2]]
    assert r.representatives == [1]
    assert r.dedupe(items) == ["short copy of text!!"]


def test_non_string_items_are_stringified():
    r = find_duplicates([12345, "12345", 12345.0, None])
    assert r.groups == [[0, 1]]
    assert r.n_items == 4


def test_deterministic_results_and_seeds(planted):
    items, _ = planted(200)
    a = DuplicateFinder(all_pairs_max=0).find(items)
    b = DuplicateFinder(all_pairs_max=0).find(items)
    assert a.pairs == b.pairs and a.groups == b.groups
    h0, h1, h0b = _text.MinHasher(128, 0), _text.MinHasher(128, 1), _text.MinHasher(128, 0)
    assert np.array_equal(h0.a, h0b.a) and np.array_equal(h0.b, h0b.b)
    assert not np.array_equal(h0.a, h1.a)
    sets = [np.array([1, 5, 9], dtype=np.int32), np.array([1, 5, 9, 11], dtype=np.int32)]
    assert np.array_equal(h0.signatures(sets), h0b.signatures(sets))


def test_lsh_path_finds_planted_pairs_like_all_pairs(planted):
    items, expected = planted(300)
    exhaustive = DuplicateFinder(all_pairs_max=10_000).find(items)
    lsh = DuplicateFinder(all_pairs_max=0).find(items)
    assert edges(exhaustive) == set(expected)
    assert edges(lsh) == set(expected)
    assert lsh.pairs == exhaustive.pairs
    for i, j, s in lsh.pairs:
        assert s == pytest.approx(text_similarity(items[i], items[j]))


def test_large_input_uses_lsh_and_finds_planted_pairs(planted):
    items, expected = planted(2400)  # 2640 items > 2000 -> LSH banding
    r = find_duplicates(items)
    assert edges(r) == set(expected)
    assert r.n_duplicates == len(expected)
    assert len(r.dedupe(items)) == len(items) - len(expected)


def test_shingle_hashes_properties():
    ids = _text.shingle_hashes("abcabc", 3)
    assert ids.dtype == np.uint64
    assert ids.size == 3 and np.all(ids[1:] > ids[:-1])  # abc, bca, cab; sorted unique
    assert _text.shingle_hashes("ab", 3).size == 1
    assert _text.shingle_hashes("", 3).size == 0
    assert np.array_equal(_text.shingle_hashes("hello", 3), _text.shingle_hashes("hello", 3))
    assert _text.jaccard(_text.shingle_hashes("abcd", 3), _text.shingle_hashes("abcd", 3)) == 1.0
    assert _text.jaccard(_text.shingle_hashes("abcd", 3), _text.shingle_hashes("xyz", 3)) == 0.0


def test_text_similarity_values():
    assert text_similarity("abc", "abc") == 1.0
    assert text_similarity("abc", "xyz") == 0.0
    assert text_similarity("", "") == 1.0
    assert text_similarity("abcd", "abcde") == pytest.approx(2 / 3)


def test_minhash_estimate_tracks_jaccard():
    rng = np.random.default_rng(0)
    a = np.sort(rng.choice(100_000, 2000, replace=False)).astype(np.int32)
    b = np.concatenate([a[:1000], np.arange(200_000, 201_000, dtype=np.int32)])
    b = np.sort(b)
    true = _text.jaccard(a.astype(np.uint64), b.astype(np.uint64))
    sig = _text.MinHasher(256, 0).signatures([a, b])
    est = float((sig[0] == sig[1]).mean())
    assert abs(est - true) < 0.1


def test_optimal_bands_are_valid_and_keep_recall():
    for t in (0.5, 0.7, 0.85, 0.95, 0.99):
        b, r = _text.optimal_bands(128, t)
        assert b >= 1 and r >= 1 and b * r <= 128
        assert _text.candidate_probability(t, b, r) >= 0.98
        assert _text.candidate_probability(min(1.0, t + 0.05), b, r) >= 0.99
    b_hi, r_hi = _text.optimal_bands(128, 0.95)
    b_lo, r_lo = _text.optimal_bands(128, 0.5)
    assert r_hi > r_lo


def test_pairs_sharing_key_helper():
    keys = np.array([3, 1, 3, 2, 1, 3])
    pairs = _text.pairs_sharing_key(keys)
    assert sorted(map(tuple, pairs.tolist())) == [(0, 2), (0, 5), (1, 4), (2, 5)]
    assert _text.pairs_sharing_key(np.array([1, 2, 3])).shape == (0, 2)
    assert _text.pairs_sharing_key(np.array([7])).shape == (0, 2)


def test_signature_batching_handles_large_sets():
    big = np.arange(0, 50_000, dtype=np.int32)
    small = np.arange(0, 50_000, 7, dtype=np.int32)
    hasher = _text.MinHasher(64, 3)
    together = hasher.signatures([big, small, big[:10]])
    alone = hasher.signatures([big])
    assert np.array_equal(together[0], alone[0])
    assert np.array_equal(together[1], hasher.signatures([small])[0])
    assert (together[0] == together[1]).mean() > 0.05


def test_invalid_parameters():
    with pytest.raises(ValueError):
        find_duplicates(["a"], threshold=0)
    with pytest.raises(ValueError):
        find_duplicates(["a"], threshold=1.5)
    with pytest.raises(ValueError):
        find_duplicates(["a"], n_gram=0)
    with pytest.raises(ValueError):
        find_duplicates(["a"], kind="bogus")
    with pytest.raises(ValueError):
        DuplicateFinder(num_perm=1)
    with pytest.raises(TypeError):
        find_duplicates(42)


def test_single_string_is_rejected_not_split_into_characters():
    with pytest.raises(TypeError, match="wrap it in a list"):
        find_duplicates("one lonely string")
    with pytest.raises(TypeError):
        near_dupes.dedupe("one lonely string")
    with pytest.raises(TypeError):
        find_duplicates(["a", "b"]).dedupe("ab")
    assert find_duplicates(["one lonely string"]).n_items == 1


def _brute_force_edges(items, threshold, n_gram=3):
    return {
        (i, j)
        for i in range(len(items))
        for j in range(i + 1, len(items))
        if text_similarity(items[i], items[j], n_gram=n_gram) + 1e-9 >= threshold
    }


def test_small_inputs_match_brute_force_exactly():
    rng = np.random.default_rng(7)
    alphabet = np.array(list("abcdefghij "))
    bases = ["".join(rng.choice(alphabet, 30)) for _ in range(10)]
    items = list(bases)
    for base in bases:  # variants with 1..8 characters changed: similarities spread over (0, 1)
        for k in range(1, 9):
            chars = list(base)
            for p in rng.choice(30, size=k, replace=False):
                chars[int(p)] = str(rng.choice(alphabet))
            items.append("".join(chars))
    items = [x for x in dict.fromkeys(_text.normalize_text(x) for x in items) if x]  # distinct, non-blank
    for threshold in (0.3, 0.5, 0.85):
        r = find_duplicates(items, threshold=threshold)
        assert edges(r) == _brute_force_edges(items, threshold)
        for i, j, s in r.pairs:
            assert s == pytest.approx(text_similarity(items[i], items[j]))


def test_all_pairs_budget_fallback_matches_exact_pass(planted):
    items, expected = planted(150)
    keys = [_text.normalize_text(x) for x in items]
    exact = _text.find_text_pairs(keys, threshold=0.85, n_gram=3)
    screened = _text.find_text_pairs(keys, threshold=0.85, n_gram=3, exact_work_budget=0)
    assert exact == screened
    assert {(i, j) for i, j, _ in exact} == set(expected)
    assert _text.all_pairs(np.array([10, 10, 3], dtype=np.int64), 0.85).tolist() == [[0, 1]]
    assert _text.all_pairs(np.array([10], dtype=np.int64), 0.85).shape == (0, 2)
