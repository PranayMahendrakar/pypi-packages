

def test_quiet_room_tone_with_no_speech_is_reported_as_silent():
    """Regression: room tone at -59 dBFS barely cleared SILENCE_DB (-60), fell into the
    'flat' branch, and had nearly every frame marked active - a completely empty
    recording came back as one speaker talking for the whole clip."""
    import numpy as np

    import speaker_diarize_lite as sd

    rng = np.random.default_rng(0)
    room_tone = rng.normal(0, 0.001, 16000 * 8)
    result = sd.diarize(room_tone, sample_rate=16000)
    assert result.segments == []
    assert result.speakers == []


def test_an_audible_steady_tone_still_counts_as_active():
    """The fix for room tone must not swallow a genuinely audible flat signal."""
    import numpy as np

    import speaker_diarize_lite as sd

    t = np.arange(16000 * 4) / 16000
    tone = 0.05 * np.sin(2 * np.pi * 300 * t)
    result = sd.diarize(tone, sample_rate=16000)
    assert result.segments, "an audible steady tone should still be found as activity"
