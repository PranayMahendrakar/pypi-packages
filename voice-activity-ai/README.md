# voice-activity-ai

Find where someone is actually speaking in a recording, and where it is just noise.

Point it at a `.wav` file or a numpy array and it hands back the speech segments with
timestamps, how much of the recording is speech, and the audio with the dead air at the
ends taken off.

## Install

```
pip install voice-activity-ai
```

Needs numpy and nothing else. No model download, no compiler, no system audio library.

## Quickstart

```python
import numpy as np
from voice_activity_ai import detect

seconds = np.arange(48000) / 16000                                   # 3 s at 16 kHz
audio = 0.02 * np.random.default_rng(0).standard_normal(48000)       # room noise
audio[16000:32000] += 0.3 * np.sin(2 * np.pi * 140 * seconds[16000:32000])  # talking
print(detect(audio, sample_rate=16000).summary())
```

```
voice activity: 34.0% speech in 1 segment  [<array>, 3.000s at 16000 Hz]
  speech 1.02s, silence 1.98s; noise floor -34.3 dBFS, threshold 0.40
    1. 0:01.0 - 0:02.0  (1.02s, confidence 0.98)
```

## What it does

- **Scores every frame three ways at once.** Loudness alone is a bad voice detector: it
  calls a slammed door speech and a whisper silence. Each frame is measured for
  short-time energy, zero-crossing rate and spectral flatness, and all three go into the
  verdict.
- **Adapts to the recording's own noise floor.** The threshold is set from the quietest
  frames of the file in front of it, not from an absolute level, so a quiet phone
  recording and a loud studio one are judged the same way.
- **Does not mistake noise for talking.** Flatness is measured after each frame's broad
  spectral tilt is divided out, so a fan, an air conditioner or traffic through a window
  reads as noise instead of as wall-to-wall speech.
- **Smooths the result the way a listener would.** Short gaps are bridged before short
  runs are dropped, so one breath does not split a sentence in two and one loud
  transient does not become a segment.
- **Takes the audio however you have it.** A `.wav` path (8/16/24/32-bit PCM and IEEE
  float, including WAVE_FORMAT_EXTENSIBLE), a numpy array, or a `(samples, sample_rate)`
  pair. Stereo is mixed to mono, integers are scaled by their dtype, 8 kHz to 48 kHz and
  beyond all work. Your array is never modified.
- **Deterministic.** The same input gives the same answer every time; there is no
  randomness anywhere in it.

This is an energy and spectral heuristic, not a neural voice activity detector. It is
good at what heuristics are good at - trimming dead air, splitting a recording into
utterances, and telling you how much of a file is someone talking. It does not know
language, it cannot tell one speaker from another, and on music or a busy cafe it will
report sound that is structured rather than speech specifically. If you need a neural
VAD, use one; this is the dependency-free tool you reach for first.

## API

```python
detect(audio, *, sample_rate=None, frame_ms=30, sensitivity=0.5,
       min_speech_ms=200, min_silence_ms=300) -> VoiceActivity
```

| Argument | Meaning |
| --- | --- |
| `audio` | a `.wav` path, a numpy array, or `(samples, sample_rate)` |
| `sample_rate` | required for a bare array; must agree if given with a file or pair |
| `frame_ms` | analysis frame length, and so the resolution of every reported boundary |
| `sensitivity` | 0 to 1; higher calls more of the recording speech |
| `min_speech_ms` | speech runs shorter than this are dropped |
| `min_silence_ms` | gaps inside speech shorter than this are bridged |

`VoiceActivity` carries:

| Member | Meaning |
| --- | --- |
| `.segments` | `list[Segment(start_s, end_s, duration_s, confidence)]`, in time order |
| `.n_segments` | how many segments were found |
| `.speech_ratio` | share of the recording that is speech, 0 to 1 |
| `.total_speech_s` / `.total_silence_s` | seconds of each |
| `.mask` | numpy bool array, one entry per analysed frame |
| `.frame_times` | start time in seconds of each frame, same length as `.mask` |
| `.scores` | raw frame score in 0 to 1, before smoothing |
| `.threshold`, `.noise_floor_dbfs` | what the decision was actually made with |
| `.trim()` | ndarray with leading and trailing silence removed |
| `.summary()` | the human-readable report above |
| `.to_dict()` | the whole result as a JSON-safe dict |

Two shortcuts for the one-line cases:

```python
speech_ratio(audio, **kw) -> float     # how much of it is talking
trim_silence(audio, **kw) -> ndarray   # the same audio, dead air at the ends removed
```

A recording shorter than one frame gives an empty result rather than raising, and a
recording with nothing above the threshold gives no segments rather than one long one.

## CLI

```
voice-activity-ai recording.wav                  # print the summary
voice-activity-ai recording.wav --json           # print to_dict() as JSON
voice-activity-ai recording.wav --output r.json  # also write it to a file
voice-activity-ai recording.wav --trim clean.wav # write the trimmed audio
voice-activity-ai recording.wav --sensitivity 0.7 --min-speech-ms 120
```

Exit code is 0 when speech was found, 1 when none was, and 2 when the input could not be
read - so it drops straight into a shell script or a batch pipeline.

## Limits

Worth knowing before you trust a boundary:

- **This is an energy and spectral heuristic, not a neural detector.** It is built for
  trimming, segmenting and measuring how much of a recording carries speech. It has no
  model, downloads nothing, and will not tell speech from a television in the next room.
- **Boundaries are accurate to roughly a frame, and less so at low pitch.** Speech is
  amplitude modulated at around 4 Hz, so a phrase ends in a quiet trough that gets
  trimmed. Measured on synthetic bursts: a 200 Hz voice ends within about 40 ms of the
  truth, a 90 Hz one within about 120 ms. Raise `min_silence_ms` if you would rather
  keep the tail than cut it.
- **`min_silence_ms` below about 110 ms will split inside a phrase**, because the
  modulation troughs within normal speech are that long. That is the parameter working,
  not failing: it is the shortest gap you are willing to call a gap.

## License

MIT
