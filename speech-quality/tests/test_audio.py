"""Getting samples in: every bit depth, every shape, and every way it can go wrong."""

from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import RATE, silence, sine, speech_like, write_wav
from speech_quality import Audio, load_audio, read_wav
from speech_quality.audio import DEFAULT_SAMPLE_RATE


@pytest.mark.parametrize("width,bits", [(1, 8), (2, 16), (3, 24), (4, 32)])
def test_every_bit_depth_round_trips(wav_dir, width, bits):
    """8, 16, 24 and 32 bit PCM all decode back to the signal that was written."""
    original = sine(seconds=0.5, freq=500.0, amplitude=0.5)
    path = write_wav(wav_dir / "depth{}.wav".format(bits), original, width=width)

    loaded = read_wav(path)

    assert loaded.sample_rate == RATE
    assert loaded.channels == 1
    assert loaded.n_samples == original.size
    assert any("{}-bit".format(bits) in note for note in loaded.notes)
    # 8-bit has only 256 levels, so it gets a looser tolerance than the rest.
    tolerance = 0.01 if width == 1 else 0.0005
    assert np.max(np.abs(loaded.samples - original)) < tolerance


def test_sixteen_bit_full_scale_is_full_scale(wav_dir):
    """A 16-bit file written at the rails reads back at the rails, not near them."""
    path = write_wav(wav_dir / "rails.wav", np.array([1.0, -1.0, 0.0, 0.5]), width=2)

    samples = read_wav(path).samples

    assert samples[0] == pytest.approx(1.0, abs=1e-4)
    assert samples[1] == pytest.approx(-1.0, abs=1e-4)
    assert samples[2] == 0.0


def test_twenty_four_bit_sign_is_decoded_from_raw_bytes(wav_dir):
    """24-bit arrives as loose bytes, so negative samples must be reassembled by hand."""
    original = np.array([0.0, 0.75, -0.75, 0.25, -0.25])
    path = write_wav(wav_dir / "signed24.wav", original, width=3)

    samples = read_wav(path).samples

    assert np.max(np.abs(samples - original)) < 1e-6
    assert samples[2] < 0.0


@pytest.mark.parametrize("rate", [8000, 11025, 16000, 22050, 44100, 48000])
def test_every_usual_sample_rate_is_read(wav_dir, rate):
    """8 kHz through 48 kHz all load with their own rate and the right duration."""
    signal = speech_like(seconds=0.5, rate=rate)
    path = write_wav(wav_dir / "rate{}.wav".format(rate), signal, rate=rate)

    loaded = read_wav(path)

    assert loaded.sample_rate == rate
    assert loaded.duration == pytest.approx(0.5, abs=0.01)


def test_stereo_is_mixed_to_mono_with_a_note(wav_dir):
    """Two channels become one, and the report says so rather than hiding it."""
    left = sine(seconds=0.5, freq=300.0, amplitude=0.4)
    right = sine(seconds=0.5, freq=300.0, amplitude=0.2)
    stereo = np.stack([left, right], axis=1)
    path = write_wav(wav_dir / "stereo.wav", stereo, width=2, channels=2)

    loaded = read_wav(path)

    assert loaded.channels == 2
    assert loaded.samples.ndim == 1
    assert loaded.n_samples == left.size
    assert any("mixed down to mono" in note for note in loaded.notes)
    assert np.max(np.abs(loaded.samples - (left + right) / 2.0)) < 0.001


def test_stereo_array_is_mixed_to_mono_with_a_note():
    """The same mixdown happens for an array handed in directly."""
    stereo = np.stack([np.full(100, 0.4), np.full(100, 0.2)], axis=1)

    loaded = load_audio(stereo, sample_rate=RATE)

    assert loaded.channels == 2
    assert loaded.samples.shape == (100,)
    assert np.allclose(loaded.samples, 0.3)
    assert any("mixed down to mono" in note for note in loaded.notes)


def test_missing_path_raises_file_not_found(wav_dir):
    """A path that is not there is a FileNotFoundError, not a ValueError."""
    with pytest.raises(FileNotFoundError):
        read_wav(wav_dir / "nothing-here.wav")


def test_non_wav_path_names_what_is_supported(tmp_path):
    """An mp3 gets a clear ValueError that says what to do about it."""
    path = tmp_path / "podcast.mp3"
    path.write_bytes(b"ID3 not really an mp3")

    with pytest.raises(ValueError) as caught:
        read_wav(path)

    message = str(caught.value)
    assert "WAV" in message
    assert ".mp3" in message
    assert "podcast.mp3" in message


def test_a_wav_suffix_on_something_that_is_not_a_wav_is_a_clear_error(tmp_path):
    """Named .wav but full of nonsense: still a ValueError a human can read."""
    path = tmp_path / "broken.wav"
    path.write_bytes(b"this is not a RIFF header at all")

    with pytest.raises(ValueError) as caught:
        read_wav(path)

    assert "PCM WAV" in str(caught.value)


def test_directory_is_not_a_recording(tmp_path):
    """Handing over a folder says so instead of failing somewhere deeper."""
    folder = tmp_path / "takes.wav"
    folder.mkdir()

    with pytest.raises(ValueError) as caught:
        read_wav(folder)

    assert "directory" in str(caught.value)


def test_a_directory_without_a_wav_suffix_is_still_named_as_a_directory(tmp_path):
    """"." is a folder, not a file with a missing suffix, and the message says which."""
    with pytest.raises(ValueError) as caught:
        read_wav(str(tmp_path))

    assert "directory" in str(caught.value)
    assert "no suffix" not in str(caught.value)


def test_a_truncated_file_says_the_data_ran_short(wav_dir):
    """A valid header over a short body must not read as a fine 5 ms recording."""
    path = write_wav(wav_dir / "cut.wav", speech_like(seconds=4.0))
    whole = open(path, "rb").read()
    with open(path, "wb") as handle:
        handle.write(whole[:200])

    audio = read_wav(path)

    assert audio.duration < 0.01
    assert any("truncated" in note for note in audio.notes)
    assert any("declares 4.000 s" in note for note in audio.notes)


def test_a_whole_file_is_not_called_truncated(wav_dir):
    """The note only appears when frames are actually missing."""
    path = write_wav(wav_dir / "whole.wav", speech_like(seconds=0.5))

    audio = read_wav(path)

    assert not any("truncated" in note for note in audio.notes)


def test_caller_array_is_never_modified():
    """Loading copies; the caller's samples come out exactly as they went in."""
    original = speech_like(seconds=0.5)
    original[10] = np.nan  # forces the non-finite repair path, which writes
    before = original.copy()

    loaded = load_audio(original, sample_rate=RATE)

    assert np.array_equal(before, original, equal_nan=True)
    assert np.isfinite(loaded.samples).all()
    assert any("not finite" in note for note in loaded.notes)


def test_samples_and_rate_pair_is_accepted():
    """The (samples, sample_rate) form carries its own rate."""
    loaded = load_audio((sine(seconds=0.2, rate=8000), 8000))

    assert loaded.sample_rate == 8000
    assert not loaded.assumed_sample_rate


def test_two_sample_signal_is_not_mistaken_for_a_pair():
    """A tuple of two numbers is two samples, not samples plus a rate."""
    loaded = load_audio(np.array([0.5, -0.5]), sample_rate=RATE)

    assert loaded.n_samples == 2


def test_missing_rate_is_assumed_and_said_so():
    """No rate means 16 kHz is assumed, and the note tells the caller."""
    loaded = load_audio(sine(seconds=0.1))

    assert loaded.sample_rate == DEFAULT_SAMPLE_RATE
    assert loaded.assumed_sample_rate
    assert any("no sample rate was given" in note for note in loaded.notes)


def test_conflicting_rates_are_refused():
    """Two different rates in one call is a mistake worth raising."""
    with pytest.raises(ValueError) as caught:
        load_audio((sine(seconds=0.1), 8000), sample_rate=16000)

    assert "two different sample rates" in str(caught.value)


def test_integer_samples_are_scaled():
    """int16 input is scaled to -1.0 to 1.0 with a note saying how."""
    values = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    loaded = load_audio(values, sample_rate=RATE)

    assert loaded.samples[1] == pytest.approx(0.5)
    assert loaded.samples[2] == pytest.approx(-0.5)
    assert any("signed 16-bit" in note for note in loaded.notes)


def test_unsigned_integer_samples_are_centred():
    """uint8 input is centred on 128 the way an 8-bit WAV is."""
    values = np.array([0, 128, 255], dtype=np.uint8)

    loaded = load_audio(values, sample_rate=RATE)

    assert loaded.samples[1] == pytest.approx(0.0, abs=0.01)
    assert loaded.samples[0] < 0.0
    assert loaded.samples[2] > 0.0


def test_empty_recording_is_refused():
    """No samples is not a quiet recording, it is not a recording."""
    with pytest.raises(ValueError) as caught:
        load_audio(np.zeros(0), sample_rate=RATE)

    assert "no samples" in str(caught.value)


def test_bad_sample_rate_is_refused():
    """A rate of zero would divide by zero later, so it is refused up front."""
    with pytest.raises(ValueError) as caught:
        load_audio(sine(seconds=0.1), sample_rate=0)

    assert "positive number" in str(caught.value)


def test_text_samples_are_refused():
    """An array of strings is a TypeError naming what is accepted."""
    with pytest.raises(TypeError) as caught:
        load_audio(np.array(["a", "b"]), sample_rate=RATE)

    assert ".wav path" in str(caught.value)


def test_raw_bytes_are_refused():
    """Bytes carry no sample format, so guessing one would be worse than raising."""
    with pytest.raises(TypeError) as caught:
        load_audio(b"\x00\x01\x02\x03", sample_rate=RATE)

    assert "sample format" in str(caught.value)


def test_unicode_filename_is_read_and_labelled(wav_dir):
    """A file named in Japanese and Cyrillic loads and keeps its name."""
    path = write_wav(wav_dir / "録音-звук.wav", sine(seconds=0.2))

    loaded = read_wav(path)

    assert loaded.source == os.path.basename(path)
    assert "録音" in loaded.source


def test_digital_silence_loads_and_knows_it():
    """All zeros is a normal input that reports itself, not an error."""
    loaded = load_audio(silence(seconds=0.5), sample_rate=RATE)

    assert loaded.is_digital_silence
    assert loaded.duration == pytest.approx(0.5)


def test_describe_and_to_dict_are_readable():
    """The loaded audio can describe itself for a headline and for JSON."""
    loaded = load_audio(np.stack([sine(0.2), sine(0.2)], axis=1), sample_rate=RATE)

    assert "2 channels mixed to mono" in loaded.describe()
    payload = loaded.to_dict()
    assert payload["channels"] == 2
    assert payload["sample_rate"] == RATE
    assert isinstance(payload["notes"], list)


def test_an_already_loaded_audio_passes_straight_through():
    """Loading an Audio gives that same Audio back rather than copying it again."""
    loaded = load_audio(sine(seconds=0.1), sample_rate=RATE)

    assert load_audio(loaded) is loaded
    assert isinstance(loaded, Audio)
