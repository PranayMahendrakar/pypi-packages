# audio-clean-ai

Remove steady background noise - hum, hiss, fans, air conditioning - from speech recordings, without a model.
It is **spectral gating, a signal-processing heuristic, not a neural denoiser.**

It learns what the noise sounds like from the quietest stretch of the recording, then turns
down every time-frequency bin that sits close to that noise. That works well on noise that
stays the same from second to second, and does little for noise that moves the way speech
does: music, babble, a second speaker or traffic going by are not removed. Nothing is
downloaded, nothing is sent anywhere, and the only dependency is numpy.

## Install

```
pip install audio-clean-ai
```

Needs numpy and nothing else. No model download, no compiler, no system audio library.

## Quickstart

```python
import numpy as np
from audio_clean_ai import clean

t = np.arange(32000) / 16000                                         # 2 s at 16 kHz
noisy = 0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.5)                # a tone from 0.5 s on
noisy += 0.03 * np.random.default_rng(0).standard_normal(t.size)     # plus steady hiss
print(clean(noisy, sample_rate=16000).summary())
```

```
audio clean: noise down 15.5 dB (estimated), SNR 15.7 dB before, 30.0 dB after  [<array>, 2.000 s at 16000 Hz, mono]
  noise floor -30.5 dBFS, noise learned from the quietest stretch 0.01-0.43 s (reliable); noise is strongest in the 4000 Hz octave
  strength 0.80: noise-only bins cut by up to 24 dB, the 300-3400 Hz speech band by at most 15 dB
```

With a file it is one line each way:

```python
result = clean("interview.wav")        # learn the noise, turn it down
result.save("interview-clean.wav")     # same channels, rate and bit depth
```

## What it does

- **Learns the noise from the recording itself.** It finds the quietest 0.4 s of sound,
  takes the per-frequency median of its power (so a breath or a click in that stretch
  does not inflate it), and treats that as the noise. Leading digital silence, such as a
  zero-filled lead-in, is skipped rather than learned as "no noise at all".
- **Turns down only what sits near the noise.** Every 32 ms frame is split into
  frequency bins with `numpy.fft`. A bin within about 2 dB of the noise profile is noise
  and is cut; a bin 10 dB or more above it is signal and passes untouched; in between the
  gain rises smoothly.
- **Does not warble.** The gate is smoothed across time (about 40 ms) and frequency
  (about 100 Hz) before it is applied. Unsmoothed gating flickers bins on and off, which is
  the watery "musical noise" that makes cheap noise reduction sound worse than the noise.
- **Never gates speech away entirely.** With `preserve_speech=True` (the default) the
  300-3400 Hz band is never cut by more than 15 dB, so a quiet consonant that happens to
  sit near the noise level is turned down, not deleted.
- **Says when it is guessing.** If the quietest stretch is barely quieter than the
  loudest parts (less than 6 dB), it is probably not a pause, so the profile is flagged as
  unreliable, a warning says so, and the cut is held to 6 dB instead of guessing hard.
  Give it a `noise_profile` learned from a noise-only clip to clean such recordings fully.
- **Leaves silence silent and never clips.** Digital silence comes back as exact zeros.
  No bin is ever boosted. If the rebuilt waveform would pass full scale, the whole output
  is turned down instead of clipped (a fraction of a dB in practice, more if a float input
  was itself louder than -1 to 1), measured against the input format's own rails: 8-bit
  WAV stops at +127/128, not +1, and each rail is counted separately.
- **Keeps the shape of your audio.** The output has exactly as many samples as the input,
  in the same layout (1-D, samples x channels, or channels x samples). Stereo and
  multi-channel audio are cleaned one channel at a time, each with its own profile.
- **Takes the audio however you have it.** A `.wav` path (8/16/24/32-bit PCM and 32/64-bit
  float, including WAVE_FORMAT_EXTENSIBLE), a numpy array of floats or integers, or a
  `(samples, sample_rate)` pair, at 8 kHz to 48 kHz. Your array is never modified, and a
  read-only array is fine.
- **Deterministic.** The same input gives the same output every time.

What the numbers mean: `noise_reduction_db`, `snr_before` and `snr_after` are
**estimates** made from the noise profile, since there is no clean reference to measure
against. On a synthetic test - a voice-like harmonic signal in white noise at 0 dB SNR -
the true SNR measured against the known clean signal rose by about 8 dB at 8 kHz, 11 dB at
16 kHz, 12 dB at 22.05 kHz and 14 dB at 44.1 and 48 kHz, with the voice harmonics kept
within 1 dB. Higher rates gain more because more of their spectrum is noise alone. Mains
hum in the same test dropped by 20 dB or more in the pauses. Real recordings vary.

Where it does not help: music, a television, a crowd, a second voice, wind gusts, a door
slam, reverberation, and clipping are not steady noise, and one profile cannot describe
them. Steady noise that lands right on the voice is only partly removed while the voice is
speaking: a hum harmonic within a few tens of Hz of a voice harmonic cannot be told apart
from it in a 32 ms frame, so it drops in the pauses but survives under the words. A recording that is steady noise from end to end (no quiet stretch that is quieter
than the rest) is flagged as unreliable and cleaned only gently, because a steady tone you
want to keep looks exactly like a hum you want gone. A recording that opens with a muted
or near-silent lead-in (quiet, but not digital zero) learns that lead-in as its noise and
removes little; trim it, or pass `noise_profile`.

## API

```python
clean(audio, *, sample_rate=None, strength=0.8, noise_profile=None,
      preserve_speech=True) -> CleanResult
```

| Argument | Meaning |
| --- | --- |
| `audio` | a `.wav` path, a numpy array (1-D, or 2-D either way round), or `(samples, sample_rate)` |
| `sample_rate` | required for a bare array; must agree if given with a file or pair |
| `strength` | 0 to 1, how hard to cut; 0 returns the audio unchanged, 0.8 cuts noise by up to 24 dB, 1 by up to 30 dB |
| `noise_profile` | a profile from `noise_profile()` or a previous result, used instead of learning one |
| `preserve_speech` | keep the 300-3400 Hz band within 15 dB of where it was |

```python
noise_profile(audio, *, sample_rate=None, frame_ms=32) -> ndarray
```

Learns a profile from a clip that holds only the noise - a few seconds of the room with
nobody talking. Returns per-bin noise power, 1-D for one channel or `(channels, bins)`
for more; `10 * log10(profile.sum())` is the noise level in dBFS. The clip must be at the
same sample rate as the recordings you clean with it.

```python
profile = noise_profile("room-tone.wav")
result = clean("interview.wav", noise_profile=profile)
```

`CleanResult` carries:

| Member | Meaning |
| --- | --- |
| `.audio` | cleaned samples, float64 in -1 to 1, same shape and layout as the input |
| `.sample_rate` | frames per second |
| `.noise_reduction_db` | estimated drop in the noise level, averaged over the recording |
| `.noise_profile` | the per-bin noise power the gate used; `.frequencies` gives each bin's Hz |
| `.snr_before`, `.snr_after` | estimated signal-to-noise ratio in dB, or `None` when no signal stands above the noise |
| `.profile_reliable` | `False` when the noise could not be learned with confidence and the cut was held back |
| `.channels` | one `ChannelReport` per channel: where its profile came from, its noise floor, its octave-band noise levels |
| `.clipping_before`, `.clipping_after` | samples at the positive and negative rails, counted separately |
| `.output_gain_db` | 0, or the small negative gain used to stay inside full scale |
| `.warnings`, `.notes` | what deserves a look, and what was decided for you |
| `.summary()` | the human-readable report above |
| `.to_dict()` | everything except the samples, as a JSON-safe dict |
| `.save(path, bits=None)` | write a PCM `.wav` with the `wave` module; keeps the input file's bit depth, else 16-bit |

For more control, `SpectralGate` is the class underneath, with every threshold exposed:

```python
from audio_clean_ai import SpectralGate

gate = SpectralGate(strength=0.9, max_cut_db=36, time_smoothing_ms=60)
result = gate.clean("interview.wav")
```

## CLI

```
audio-clean-ai noisy.wav                              # dry run: print what would be removed
audio-clean-ai noisy.wav --output clean.wav           # clean and write the result
audio-clean-ai noisy.wav -o clean.wav --strength 0.6  # cut less
audio-clean-ai noisy.wav -o clean.wav --noise-clip room.wav   # learn the noise from a clip
audio-clean-ai noisy.wav --json --report report.json  # full report as JSON, also to a file
audio-clean-ai noisy.wav -o clean.wav --no-preserve-speech --bits 24
```

Exit code is 0 on success and 2 when the input could not be read or the output could not
be written.

## License

MIT
