# speaker-diarize-lite

Work out who spoke when in a recording, and get back turns you can read, count and export.
The built-in method is a lightweight heuristic, not a trained model: it describes the voice
with hand-built spectral features and clusters them, which separates clearly different voices
(a man and a woman, say) and struggles with similar ones. Real diarization needs speaker
embeddings from a trained model; pass one in with `embed=` and this package does the rest.

## Install

```
pip install speaker-diarize-lite
```

The only dependency is numpy. Nothing is downloaded and nothing touches the network.

## Quickstart

```python
import numpy as np
from speaker_diarize_lite import diarize
sr = 16000; t = np.arange(3 * sr) / sr
voice = lambda f0: sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 30)) * (0.6 + 0.4 * np.sin(6 * np.pi * t))
audio = np.concatenate([voice(110), np.zeros(sr // 2), voice(230), np.zeros(sr // 2), voice(110)]) * 0.05
print(diarize(audio, sample_rate=sr).summary())
```

The result explains itself:

```
speaker-diarize-lite: 2 speakers (estimated from the recording) in 10.0 s of audio (9.0 s of speech)
recording: audio
method: hand-built spectral features (log mel-band energies and deltas), 28 windows
separation: closest pair of voices 88.7 spreads apart (distinct at 6.0 or more)
speaking time:
  SPEAKER_00      6.0 s   66.6%  2 segments
  SPEAKER_01      3.0 s   33.4%  1 segment
segments:
      0.01 -     3.01  SPEAKER_00  confidence 0.99
      3.50 -     6.51  SPEAKER_01  confidence 0.97
      7.00 -     9.99  SPEAKER_00  confidence 0.99
notes:
  - energy threshold -36.4 dBFS (quiet floor -41.2, peak -27.5)
```

With a file it is one line: `diarize("meeting.wav")`.

## What it does

- **Finds speech with an energy detector.** The threshold adapts to the recording, between its
  quiet floor and its loudest 100 ms. It finds the loud parts, not speech as such: music, a
  door or steady noise louder than the talkers counts as "speech", and talk quieter than the
  background is missed.
- **Describes short windows of speech.** One-second windows every 0.25 s, each described by the
  average of its log mel-band energies (40 bands, decorrelated with a DCT as in the MFCC recipe,
  with the overall level removed so a louder take of the same voice looks the same).
- **Clusters them and estimates how many voices there are.** Average-linkage clustering and
  k-means written in numpy. Two groups count as different voices only if they sit at least 6
  pooled spreads apart; one voice split in two lands at about 2.5 to 5. Voices heard for under
  3 seconds are held to a stricter standard, and windows that straddle a change of speaker
  are not mistaken for a third voice. The result reports the separation it found, so you can
  see how clear-cut the decision was.
- **Places each change of speaker at the most likely 10 ms frame**, using per-voice Gaussian
  models of the frame spectra and their deltas, within half a window of the first estimate.
- **Handles the awkward inputs.** Stereo and multichannel audio is mixed to mono (with a
  fallback when the channels cancel); 8/16/24/32-bit PCM and 32/64-bit float WAV; clipping is
  measured against each rail on its own, since 8-bit's positive rail stops at 127/128; silence
  gives no segments; a recording shorter than `min_segment_s` gives an empty result, not an
  error. The same input always gives the same output.

What it does not do, so you can decide whether it fits:

- **Similar voices are merged.** Two men with close pitch and vocal tract length look like one
  voice to these features. The result says so, and `embed=` is the fix.
- **One speaker at a time.** Overlapping speech goes to whichever voice dominates.
- **Rapid back-and-forth blurs.** Turns shorter than about two seconds with no pause between
  them mix the voices inside each window; the count can come out too low.
- **No resampling or decoding.** WAV is read directly; for other formats decode them yourself and
  pass `(samples, sample_rate)`.

How well it works was measured on synthetic voices built in the test suite (a harmonic source
shaped by moving formants, at clearly different pitches and formant scales). On those, a man
and a woman are told apart and every change of turn is placed within 0.25 s, one voice stays
one voice, and near-identical voices are merged. Real recordings are harder than synthetic
ones; treat the default method as a quick first pass and use a trained embedding model when
the answer matters.

## API

```python
diarize(audio, *, sample_rate=None, num_speakers=None, min_segment_s=0.5, embed=None) -> Diarization
```

- `audio`: a `.wav` path, a numpy array (1-D, or 2-D with channels either way round), or a
  `(samples, sample_rate)` tuple. Integer arrays are scaled from their type's range. Your
  array is never modified, and read-only arrays work.
- `sample_rate`: required with a bare array; must agree with the file if given with a path.
- `num_speakers`: how many to find. `None` estimates it and the result's `estimated_speakers`
  is True. A number larger than the recording supports is honoured, and a warning says the
  extra speakers are probably one voice split into parts.
- `min_segment_s`: shortest turn reported. Shorter changes of speaker are absorbed into the
  neighbouring turn; shorter bursts of sound are ignored.
- `embed`: optional `embed(window_samples, sample_rate) -> vector`. It is called once per window
  (about four calls per second of speech) with a 1-D float64 copy of the samples, and must
  return a 1-D vector of the same length every time. Vectors are compared by cosine geometry.
  Change points then stay at window resolution (0.25 s), because the frame-level refinement
  uses the hand-built features.

```python
def embed(window, sample_rate):
    return my_speaker_model(window, sample_rate)   # any trained model, returns a 1-D vector

result = diarize("meeting.wav", embed=embed)
```

`Diarizer(**settings).diarize(audio, sample_rate=None)` is the same with every knob exposed:
`num_speakers`, `min_segment_s`, `embed`, `max_speakers=8`, `window_s=1.0`, `hop_s=0.25`,
`min_speaker_s=0.75`, `separation_threshold=6.0`, `silence_db=-60.0`, `refine_boundaries=True`,
`metric="auto"` (or `"euclidean"`, `"cosine"`) and `random_state=0`.

**`Diarization`** (the result)

- `segments`: list of `Segment(start_s, end_s, speaker, confidence)`, sorted and never
  overlapping. Speakers are `SPEAKER_00`, `SPEAKER_01`, ... in order of first appearance.
  `confidence` (0 to 1) is how much closer a turn sits to its own voice than to the next most
  similar one; 1.0 when only one voice was found. It ranks turns; it is not a probability.
- `speakers`, `num_speakers`, `speaking_time` (seconds per speaker), `estimated_speakers`.
- `evidence_speakers`: how many voices the recording itself supports, whatever you asked for.
- `separation` and `separation_threshold`: how far apart the closest two voices are, in pooled
  spreads (for a single voice, how far the best split into two got).
- `duration_s`, `speech_s`, `windows`, `method` (`"spectral"` or `"embed"`), `warnings`,
  `notes`, `settings`.
- `summary()`: the report above, plain ASCII punctuation. `to_dict()`: JSON-safe dict.
- `to_rttm(file_id=None)`: RTTM, the plain-text format diarization tools exchange, one
  `SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>` line per turn.
- `speaker_at(time_s)`: who is talking at that moment, or None.

`read_wav(path)` and `write_wav(path, samples, sample_rate, bits=16)` are the WAV reader and
writer the package uses, exposed because they are handy.

## CLI

```
speaker-diarize-lite meeting.wav                   # the summary
speaker-diarize-lite meeting.wav --speakers 2      # you know how many
speaker-diarize-lite meeting.wav --rttm > meeting.rttm
speaker-diarize-lite meeting.wav --json
speaker-diarize-lite meeting.wav --output meeting.json   # or .rttm
```

Other options: `--max-speakers N`, `--min-segment SECONDS`, `--file-id NAME` (the recording name
in RTTM). Output is UTF-8 whatever the console, so file names in any script print safely.
Exit status is 0 on success and 2 when the recording cannot be read or the output cannot be
written.

## License

MIT. Copyright (c) 2026 Pranay Mahendrakar.
