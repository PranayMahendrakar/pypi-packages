

# --- regression: the headline semantic claim ------------------------------------------

_MAINT = [
    "The bearing temperature climbed through the shift.",
    "Maintenance logged the reading.",
    "Vibration on the drive end rose.",
    "The technician scheduled a check.",
    "Downtime was avoided.",
]
_FIN = [
    "Quarterly revenue beat the forecast.",
    "Renewals drove the gain.",
    "Marketing credited the campaign.",
    "The board raised the budget.",
    "Hiring starts next quarter.",
]


def test_an_embedding_model_finds_the_real_topic_boundary():
    """Word overlap cannot see that two passages are about different things when
    neither repeats its own vocabulary. A real embedding model can, and this is the
    hook that lets one be used."""
    import numpy as np

    from rag_chunker._semantic import analyse_shifts

    sentences = _MAINT + _FIN

    def embed(texts):
        return np.array([[1.0, 0.0] if t in _MAINT else [0.0, 1.0] for t in texts])

    shifts, _sims, reason = analyse_shifts(sentences, embed=embed)
    assert reason is None
    assert shifts == {5}, f"expected the split at sentence 5, got {sorted(shifts)}"


def test_embed_must_return_one_vector_per_sentence():
    import numpy as np
    import pytest

    from rag_chunker._semantic import analyse_shifts

    with pytest.raises(ValueError, match="one vector per sentence"):
        analyse_shifts(_MAINT + _FIN, embed=lambda texts: np.zeros((3, 4)))


def test_the_default_method_is_predictable_and_loses_nothing():
    """The default used to be `semantic`, which cannot deliver what its name promises
    on ordinary prose. The default is now recursive, which always reproduces the input."""
    import rag_chunker as rc

    text = " ".join(_MAINT + _FIN)
    result = rc.chunk(text, size=40, overlap=0)
    assert result.method == "recursive"
    assert "".join(c.text for c in result.chunks).split() == text.split()


def test_chunk_accepts_embed_at_the_top_level():
    import numpy as np

    import rag_chunker as rc

    def embed(texts):
        return np.array([[1.0, 0.0] if t in _MAINT else [0.0, 1.0] for t in texts])

    result = rc.chunk(" ".join(_MAINT + _FIN), size=40, overlap=0, method="semantic", embed=embed)
    assert result.n_chunks >= 1
    assert "".join(c.text for c in result.chunks).split() == " ".join(_MAINT + _FIN).split()
