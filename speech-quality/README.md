# speech-quality

Measure whether a voice recording is clean enough to transcribe or publish, and
say in plain words what is wrong with it when it is not.

## Install

```bash
pip install speech-quality
```

The only dependency is numpy. WAV files are read with the standard library
`wave` module, so there is no audio library to install, no codec to find and
nothing is ever downloaded.

## Quickstart

```python
import numpy as np
from speech_quality import assess

t = np.arange(3 * 16000) / 16000.0                          # three seconds at 16 kHz
hiss = np.random.default_rng(0).standard_normal(t.size + 4)
voice = np.convolve(hiss, np.ones(5) / 5, "valid")          # roll off the top
voice *= 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t))   # syllable-rate envelope
print(assess(voice, sample_rate=16000).summary())
```

```
recording: usable (score 96.9 / 100, grade A)
  3.00 s, 16000 Hz, mono
measures:
  ok   level       98.7  -20.27 dBFS    level is healthy at -20.3 dBFS RMS, peaks at -3.6 dBFS, 3.6 dB of headroom left
  ok   clipping   100.0  0.0%           no samples reach full scale, so nothing is clipped
  ok   noise       85.7  25.12 dB       the voice stands 25.1 dB above a -41.7 dBFS noise floor, clean enough to transcribe
  ok   silence     99.5  3.2%           3% silence overall, 0.00 s at the start and 0.00 s at the end, which is normal for speech
  ok   speech     100.0  96.8%          97% of the recording carries speech, centred on 1300 Hz with the level swinging 9.0 dB between syllables
  ok   dynamics   100.0  16.52 dB       a 16.5 dB crest factor over the parts that carry sound, the natural rise and fall of unprocessed speech
  ok   bandwidth  100.0  8000 Hz        energy reaches 8000 Hz, 100% of the way to the 8000 Hz ceiling, wide enough for clear consonants
issues: none
```

On a real file it is one line:

```python
report = assess("interview.wav")
if not report.usable:
    print(report.issues[0])
```

## What it checks

Seven measures, each reporting the raw number it found, a 0-100 score, whether
that passes, and a sentence you can act on.

- **level** - RMS and peak in dBFS, and how far the level sits from the -20 dBFS
  a healthy speech recording holds. Catches material recorded too quiet to
  survive noise reduction and material recorded too hot to survive anything.
- **clipping** - the share of samples pinned at or near full scale, how many
  separate clipped runs there are and how long the worst one lasts. Clipping is
  destroyed waveform; no processing brings it back.
- **noise** - the noise floor taken from the quietest decile of frames, and the
  signal-to-noise ratio between that and the loudest decile. No voice-activity
  decision is involved, so none can go wrong.
- **silence** - leading silence, trailing silence and the total silent share, so
  dead air is trimmed before anyone pays to transcribe it.
- **speech** - the share of the recording carrying speech-like energy, judged on
  band energy, spectral centroid and whether the level actually swings at
  syllable rate. A steady tone passes every spectral test and is still not
  speech.
- **dynamics** - crest factor, and whether the life has been compressed or
  limited out of the recording.
- **bandwidth** - the highest frequency still carrying real energy. This is what
  exposes telephone-band audio, and audio upsampled from a narrower original:
  the file says 48 kHz, the sound stops at 3.4 kHz, and every consonant that
  lived above that is already gone.

Four of those faults are damage rather than a chore. Clipped samples are a
destroyed waveform, noise is already mixed in under the voice, a recording that
does not behave like speech has nothing in it to transcribe, and a band that
stops at 3.4 kHz has lost its consonants. Nothing you do later brings any of
them back, so **any one of clipping, noise, speech or bandwidth failing makes
the recording unusable**, however well the other measures average out;
`report.blocking` names the ones that did and the CLI exits `1`. A level 6 dB
off, dead air at the ends and heavy compression are all things a later pass
fixes, so they pull the score down without condemning the take.

Input can be a path to a `.wav` file (PCM 8, 16, 24 or 32 bit, 8 kHz to 48 kHz
and beyond), a numpy array of samples, or a `(samples, sample_rate)` pair.
Multi-channel audio is mixed down to mono and the mixdown is recorded in
`report.notes`. Your array is never written to. The same input always gives the
same answer.

**These are signal measurements, not a perceptual model.** speech-quality is not
PESQ, POLQA or ViSQOL and does not estimate a mean opinion score. It tells you
that a recording clips, hisses, sits 8 dB too quiet or stops dead at 3.4 kHz. It
does not tell you how a listener would rate it, and it has no opinion at all
about accent, delivery or content.

## API

```python
from speech_quality import assess, assess_batch, signal_to_noise, estimate_noise_floor
```

`assess(audio, *, sample_rate=None, thresholds=None, source=None) -> AudioReport`
Assess one recording.

`assess_batch(items, *, sample_rate=None, thresholds=None) -> BatchReport`
Assess many, keeping going when one cannot be read. An item may be a
`(label, recording)` pair, which is how arrays get names.

`signal_to_noise(audio, **kw) -> float` the ratio in dB.
`estimate_noise_floor(audio, **kw) -> float` the floor in dBFS.

**AudioReport**

| attribute | meaning |
| --- | --- |
| `.score` | 0-100 overall, the weighted average of the measures |
| `.grade` | `"A"` (best) through `"F"` |
| `.usable` | good enough to transcribe or publish: the score clears `usable_score`, enough of the measures could be taken, and none of the four decisive ones failed |
| `.blocking` | the decisive measures that failed, worst fault first; empty when none did |
| `.metrics` | `dict[name -> Metric]`, one per measure |
| `.issues` | what is wrong, worst first, one sentence each |
| `.notes` | decisions taken while loading, such as a stereo mixdown |
| `.coverage` | share of the measures that had anything to work with |
| `.duration`, `.sample_rate`, `.channels`, `.digital_silence` | what was read |
| `.summary()` | the whole report as plain text |
| `.to_dict()` | the whole report as JSON-safe data |

**Metric** carries `.value` (the raw number), `.unit`, `.score` (0-100), `.ok`,
`.measured` and `.message`. `.measured` is `False` when the recording could not
support that measurement - a spectrum of four samples, a noise floor of a signal
whose level never drops - in which case the score is a neutral placeholder and
not a judgement, and `.ok` is `None` rather than `True`, because nothing was
measured and nothing passed. A recording most of the measures could not touch is
never called usable, however well the placeholders average out.

**BatchReport** carries `.results`, `.failures`, `.usable`, `.unusable`,
`.mean_score`, `.worst(n=5)`, `.summary()` and `.to_dict()`.

**Thresholds** holds every limit the verdict uses, and any of them can be moved:

```python
from speech_quality import assess

report = assess("podcast.wav", thresholds={"min_snr_db": 25, "usable_score": 70})
```

## CLI

```bash
speech-quality interview.wav                       # the report as text
speech-quality takes/*.wav --worst 3               # a batch, worst offenders first
speech-quality interview.wav --json                # the report as JSON
speech-quality interview.wav --output report.txt   # written as UTF-8
speech-quality interview.wav --min-snr 20 --usable-score 70
speech-quality --help
```

Exit codes: `0` every recording is usable, `1` at least one is not, `2` nothing
could be read. That makes it a filter:

```bash
for f in takes/*.wav; do speech-quality "$f" --quiet || echo "redo: $f"; done
```

## License

MIT. Copyright (c) 2026 Pranay Mahendrakar.
