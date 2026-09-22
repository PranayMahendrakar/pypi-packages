import random
from typing import List, Tuple

import pytest


def planted_texts(n_base: int, every: int = 10, seed: int = 0, length: int = 80) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Random lowercase strings plus one near-duplicate (one character changed) for every ``every``-th base string.

    Returns ``(items, expected_pairs)`` where each expected pair is ``(base_index, variant_index)``.
    """
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    base = ["".join(rng.choice(alphabet) for _ in range(length)) for _ in range(n_base)]
    items = list(base)
    expected = []
    for k in range(0, n_base, every):
        s = base[k]
        p = length // 2
        variant = s[:p] + ("z" if s[p] != "z" else "q") + s[p + 1 :]
        expected.append((k, len(items)))
        items.append(variant)
    return items, expected


@pytest.fixture
def planted():
    return planted_texts
