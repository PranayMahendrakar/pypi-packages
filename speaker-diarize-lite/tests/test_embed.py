"""The embed hook: a caller-supplied model describes the windows instead."""

from __future__ import annotations

import numpy as np
import pytest

from _voices import SR
from speaker_diarize_lite import Diarizer, diarize


def pitch_embedding(window, sample_rate):
    """A stand-in 'speaker model': log pitch from autocorrelation, as a 2-D vector."""
    x = window[: 2048] - window[: 2048].mean()
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    lo, hi = int(sample_rate / 400), int(sample_rate / 70)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return np.array([np.log(sample_rate / lag), 1.0])


def test_embed_is_called_per_window_with_samples_and_rate(two_voices):
    signal, _ = two_voices
    calls = []

    def spy(window, sample_rate):
        calls.append((window.dtype, window.ndim, window.size, sample_rate, window.flags.writeable))
        window[:] = 0.0  # a careless model must not be able to damage the recording
        return np.ones(4)

    result = diarize(signal, sample_rate=SR, embed=spy)
    assert len(calls) == result.windows > 0
    assert all(d == np.float64 and nd == 1 and rate == SR for d, nd, _, rate, _ in calls)
    assert all(size >= int(0.5 * SR) for _, _, size, _, _ in calls)
    assert result.method == "embed"
    # Identical vectors for every window: one speaker, not a split.
    assert result.num_speakers == 1
    # The spy zeroed its copies, not the caller's signal.
    assert np.abs(signal).max() > 0.1


def test_embed_separates_the_voices(two_voices):
    signal, truth = two_voices
    result = diarize(signal, sample_rate=SR, embed=pitch_embedding)
    assert result.method == "embed"
    assert result.num_speakers == 2
    labels = {who: result.speaker_at((a + b) / 2) for a, b, who in truth}
    assert labels["man"] != labels["woman"]
    assert "supplied embedding" in result.summary()
    assert result.settings["embed"] == "pitch_embedding"


def test_embed_separates_voices_the_spectral_features_merge():
    from _voices import conversation

    turns = [("man", 3.0, 0.5), ("man_twin", 3.0, 0.5), ("man", 3.0, 0.5), ("man_twin", 3.0, 0.5)]
    signal, truth = conversation(turns, seed=4)
    marked = signal.copy()
    for start, end, who in truth:
        if who == "man_twin":
            # A tag only the "model" knows about: a small DC shift, which the
            # hand-built features remove frame by frame and so never see.
            marked[int(start * SR):int(end * SR)] += 0.05

    spectral = diarize(marked, sample_rate=SR)
    assert spectral.num_speakers == 1

    def model(window, sample_rate):
        return np.array([1.0, 0.0]) if window.mean() > 0 else np.array([0.0, 1.0])

    result = diarize(marked, sample_rate=SR, embed=model)
    assert result.num_speakers == 2
    labels = {who: result.speaker_at((a + b) / 2) for a, b, who in truth}
    assert labels["man"] != labels["man_twin"]


def test_one_dimensional_embedding_uses_euclidean_distance(two_voices):
    signal, _ = two_voices
    result = diarize(signal, sample_rate=SR, embed=lambda w, r: pitch_embedding(w, r)[:1])
    assert result.num_speakers == 2
    assert any("one dimension" in note for note in result.notes)


def test_embed_accepts_a_1_by_d_batch_shape(two_voices):
    signal, _ = two_voices
    result = diarize(signal, sample_rate=SR, embed=lambda w, r: pitch_embedding(w, r)[None, :])
    assert result.num_speakers == 2


@pytest.mark.parametrize(
    "bad, message",
    [
        (lambda w, r: np.zeros((2, 3)), "1-D"),
        (lambda w, r: np.array([]), "1-D"),
        (lambda w, r: np.array([np.nan, 1.0]), "NaN"),
        (lambda w, r: "text", "vector of numbers"),
    ],
)
def test_bad_embeddings_raise_value_error(two_voices, bad, message):
    signal, _ = two_voices
    with pytest.raises(ValueError, match=message):
        diarize(signal, sample_rate=SR, embed=bad)


def test_embeddings_of_changing_length_raise(two_voices):
    signal, _ = two_voices
    sizes = iter(range(1, 10_000))
    with pytest.raises(ValueError, match="same length"):
        diarize(signal, sample_rate=SR, embed=lambda w, r: np.ones(next(sizes)))


def test_errors_from_the_model_propagate(two_voices):
    signal, _ = two_voices

    def broken(window, sample_rate):
        raise RuntimeError("model not loaded")

    with pytest.raises(RuntimeError, match="model not loaded"):
        diarize(signal, sample_rate=SR, embed=broken)


def test_explicit_metric_choice(two_voices):
    signal, _ = two_voices
    result = Diarizer(embed=pitch_embedding, metric="euclidean").diarize(signal, sample_rate=SR)
    assert result.num_speakers == 2
    spectral_cosine = Diarizer(metric="cosine").diarize(signal, sample_rate=SR)
    assert spectral_cosine.num_speakers == 2
