# audio-anomaly

Find the knock, the new whine, the dropout or the clipping in a machine or
environment recording without any labelled faults, and without flagging the
healthy hum that makes up the rest of it.

## Install

```bash
pip install audio-anomaly
```

## Quickstart

```python
import numpy as np, audio_anomaly

t = np.arange(4 * 16000) / 16000                                   # 4 s of machine hum at 16 kHz
hum = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.01 * np.random.default_rng(0).standard_normal(t.size)
hum[32000:32480] += np.random.default_rng(1).standard_normal(480) * np.exp(-np.arange(480) / 60)  # a knock at 2 s
print(audio_anomaly.detect(hum, sample_rate=16000).summary())
```

```text
audio-anomaly: 1 anomaly in the audio (4.00 s at 16000 Hz)
Normal means: the recording's own typical sound (no reference given)
Flagged 0.2% of the recording (1 burst); highest frame score 14.9, threshold 3.0.

Anomalies, in time order:
  1. burst at 2.000-2.010 s, score 14.9
     Broadband burst (an impact, knock or click) at 2.000 s lasting 0.010 s: 29.8 dB above the surrounding sound in 27 of 28 bands.

What the kinds mean:
  burst       a short broadband sound: an impact, knock, click or rattle

Notes:
  - at 16000 Hz the frequency bands reach up to 7.09 kHz; sound between that and the 8.00 kHz Nyquist limit is not monitored, so record at a higher sample rate to watch higher frequencies
```

Take the knock out and the same call answers `no anomalies`, with the highest
frame score it saw (1.6) against the threshold (3.0), so you can see how
much headroom a healthy recording had.

With a known-good recording of the same machine, every frame is judged against
that instead of against the recording itself:

```python
report = audio_anomaly.detect("pump_today.wav", reference="pump_healthy.wav")
```

## What it does

Each recording is cut into half-overlapping frames (50 ms by default), and each
frame into up to 32 log-spaced frequency bands from 30 Hz to 12 kHz. Every band
level is the power *in* that band in dBFS, so a tone reads the same at any
sample rate. The top band has to end below 95% of the Nyquist frequency, so at
lower sample rates the bands stop sooner: with 50 ms frames they reach 3.67 kHz
at 8 kHz, 7.09 kHz at 16 kHz and 9.23 kHz at 22.05 kHz, and full 12 kHz from
32 kHz up. Nothing above that is monitored, and the report says so in a note.
Then it looks for:

- **`burst`** - a short broadband sound: an impact, knock, click or rattle. Each
  frame is compared with the sound just before *and* just after it, so a knock
  stands out wherever it happens, and the first frames of a change that then
  persists are not mistaken for one. Burst edges are refined to the millisecond.
  Given a reference recording of a machine that makes impacts of its own (a
  press, a reciprocating pump), their typical strength and how much it varies
  are learned from it band by band (it needs at least three), and a burst is
  then reported only when it clearly exceeds them. A saved profile holds only
  the spectrum, so for such machines pass the reference recording itself.
- **`tonal`** - narrowband energy that was not there before: a bearing whine, a
  squeal, a new hum. The band levels are smoothed over 0.6 s, which a knock
  cannot move, and compared with normal. A change has to hold at or above the
  threshold for 0.3 s, and then lasts while it stays above three quarters of
  it, so a band hovering at the threshold does not flicker in and out. A tone
  is a band standing out from the rest, so it is judged above any rise the
  whole spectrum shares: a gain change or a harder-working machine is a level
  change, never a tone. The report names the frequency, always inside the band
  it quotes.
- **`level_rise`** - the sound got louder across much of the spectrum and stayed
  louder: a machine running rough. It must hold, like a tone.
- **`level_drop`** - the overall level fell well below normal, again held with
  the same hysteresis.
- **`dropout`** - the signal cut out: digital silence in the middle of the
  sound, or runs of exact zeros as short as 5 ms that no frame average would
  notice.
- **`clipping`** - samples flattened against the full-scale rail, measured per
  channel before the mix to mono and against the format's own rail: +127/128 for
  8-bit WAV, +32767/32768 for 16-bit, 1.0 for float. (A fixed 0.995 cut-off would
  never see 8-bit clipping at all, since 127/128 = 0.992.)

**What "normal" is.** Given a `reference`, it is the reference's typical
spectrum, band by band. Without one, it is learned from the recording itself:
the *quieter three quarters* of each band. That is why a whine that appears
halfway through is still caught with no reference: the normal level of its band
is set by the half without it. A change that covers more than about three
quarters of a recording is, by then, its typical sound - give a reference for a
machine that may already be faulty.

**Why healthy machinery stays quiet.** Every score is a robust z-score (median
and MAD, never mean and standard deviation) *and* must be backed by at least
2 dB of real change per unit, so `sensitivity=3.0` means "3 sigma and 6 dB". A
perfectly steady hum barely wobbles, so its sigma is tiny; without the dB floor a
0.3 dB flutter would read as a ten-sigma event. Spreads are measured in ways the
anomalies themselves cannot inflate, and widened by their own uncertainty when a
recording is short. Across 700 synthetic healthy machines (8-44.1 kHz, 1-10 s,
50-900 Hz hum, three noise levels, slow amplitude wobble, pink noise, 25-100 ms
frames), each judged on its own and against a second healthy recording of the
same machine, the highest score without a reference was 2.4; against a
reference, one run in 1,400 crossed the threshold (3.2, in the lowest band of a
pink-noise recording). Against a reference, a whole-spectrum gain change of up
to 5 dB raises nothing but a note, and a larger one is a single level change
that the note explains.

**The awkward cases.** Pure silence reports no anomalies (and says it is
silent) rather than flagging everything. A recording shorter than one frame
returns an empty report. Leading and trailing silence is noted, not called a
dropout. Stereo is mixed to mono. NaN samples are read as silence and counted. A
reference of a different length or sample rate is fine: only its typical
spectrum is used, mapped onto this recording's bands, and bands it cannot reach
are judged against the recording itself - the notes say which. A recording 3 dB
or more louder or quieter than its reference across the whole spectrum gets a
note, since that is usually the microphone, its gain or its position rather
than the machine. A WAV file that was cut short is analysed as far as it goes,
with a warning giving the length its header promised; one whose RIFF size field
is wrong is read chunk by chunk; a damaged one raises a `ValueError` naming the
file. Your array is never modified, and the same input always gives the same
report.

Only numpy is required: the FFT is `numpy.fft`, and WAV files are read with the
standard library (`wave`, plus a small reader for the float files `wave`
refuses).

## API

### `detect(audio, *, sample_rate=None, reference=None, sensitivity=3.0, frame_ms=50) -> AudioAnomalyReport`

| argument | meaning |
| --- | --- |
| `audio` | a `.wav` path, a numpy array (`(n,)` or `(n, channels)`), or `(samples, sample_rate)` |
| `sample_rate` | samples per second; required for a bare array |
| `reference` | a known-good recording in any of those forms, or a profile from `spectral_profile` |
| `sensitivity` | score a frame must reach to be anomalous; higher flags less. Try 4-5 for noisy environments |
| `frame_ms` | analysis frame length in milliseconds; frames overlap by half |

### `AudioAnomalyReport`

| member | what it gives you |
| --- | --- |
| `.anomalies` | `list[Event]` in time order |
| `.scores` | numpy array, one score per frame; at or above `sensitivity` is anomalous |
| `.frame_times` | the centre of each frame in seconds, aligned with `.scores` |
| `.anomaly_ratio` | share of the recording's duration covered by anomalies, 0-1 |
| `.loudest(n=5)` | the `n` highest-scoring events, strongest first |
| `.counts`, `.of_kind(kind)`, `.has_anomalies`, `.max_score` | at a glance |
| `.summary()` | the human report, plain ASCII punctuation, safe to pipe anywhere |
| `.to_dict()` | JSON-safe dict of everything |
| `.notes`, `.warnings` | what was assumed, converted or fallen back to |
| `.mode`, `.compared_against` | `"self"` or `"reference"`, and what normal meant, in words |

`Event(start_s, end_s, kind, score, message)` is a frozen dataclass; `message`
is one plain sentence, for example *"New tone near 3.10 kHz from 1.98 s to the
end: 24.1 dB above normal in the 2.81 kHz-3.21 kHz band."*

### `Monitor(reference, *, sample_rate=None, sensitivity=3.0, frame_ms=50).check(audio) -> AudioAnomalyReport`

The reference is analysed once; each `check` compares a new recording with it.

```python
monitor = audio_anomaly.Monitor("pump_healthy.wav")
for path in todays_recordings:
    report = monitor.check(path)
    if report.has_anomalies:
        print(path, report.loudest(1)[0])
```

### `spectral_profile(audio, *, sample_rate=None, frame_ms=50) -> numpy.ndarray`

The typical spectrum of a recording: an `(n_bands, 3)` array with columns
`frequency_hz`, `level_db` (the average spectrum, dBFS) and `spread_db` (how
much each band wanders). Save it and pass it back as `reference` later; a
profile made with a different `frame_ms` is recognised and its own frame length
is used, with a note.

```python
np.save("pump.npy", audio_anomaly.spectral_profile("pump_healthy.wav"))
report = audio_anomaly.detect("pump_today.wav", reference=np.load("pump.npy"))
```

## CLI

```bash
audio-anomaly pump.wav
audio-anomaly pump_today.wav --reference pump_healthy.wav
audio-anomaly pump_healthy.wav --save-profile pump.npy
audio-anomaly pump_today.wav --reference pump.npy --sensitivity 4
audio-anomaly pump.wav --json > findings.json
audio-anomaly pump.wav --output findings.json --fail-on-anomaly
```

`--save-profile` writes `.npy`, or `.csv` for any other extension, and
`--reference` reads either back. `--fail-on-anomaly` exits with status 1 when
anything was found, for scripts and scheduled checks. Any failure - an
unreadable or damaged recording, an `--output` path that cannot be written -
prints `audio-anomaly: error: ...` and exits with status 2, never 1, so a
scheduled check cannot mistake a failure for a finding. Run
`audio-anomaly --help` for the rest.

## Limits

- **Level drops that recur at a stable period are read as beating, not independent
  faults.** Two close frequencies (two motors just out of sync, say) produce a level
  that dips to near-zero on a completely regular cycle, and each dip looks identical
  to a real drop from a local, before/after comparison - even against a clean
  reference of the same machine, since that comparison is inherently local. Three or
  more drops at a near-constant interval and depth are reported once, as a note
  describing the period, rather than as repeated alarms.
- **This means a genuinely periodic mechanical fault - a loose part striking on every
  rotation, say - can look the same as beating and gets the same treatment: reported
  once as a note, not as repeated anomalies.** Nothing in a level trace alone can
  tell physical beating apart from a real fault that happens to recur at a stable
  interval; that needs either a healthy reference recorded at the *same* speed and
  load, or knowledge of the machine's own RPM. The note is never suppressed, so a
  recurring pattern is always visible in the report - it is just not escalated to N
  separate alarms.

## License

MIT
