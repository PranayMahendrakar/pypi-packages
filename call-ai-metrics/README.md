# call-ai-metrics

Measure talk time, interruptions, silence and pace in a two-party call.

Give it a stereo call recording with one party per channel, a pair of mono arrays, or
diarized segments `[(start_s, end_s, speaker), ...]` from any tool, and it tells you who
talked how much, who talked over whom, how long the silences were and how quickly each
side replied. Every figure is computed from "who was speaking when", so **every figure
is only as good as the segmentation it is given**: on audio, speech is found by a
per-channel energy heuristic, not a neural model; on segments, it is whatever your
diarizer said. The report says which one it used, every time.

## Install

```
pip install call-ai-metrics
```

Needs numpy and nothing else. No model download, no network access, no system audio
library. `CallReport.to_frame()` additionally needs pandas
(`pip install call-ai-metrics[pandas]`).

## Quickstart

```python
from call_ai_metrics import analyze

segments = [(0.0, 6.5, "agent"), (6.9, 9.0, "customer"), (8.7, 20.0, "agent"),
            (12.0, 12.4, "customer"), (21.5, 24.0, "customer"), (23.0, 41.0, "agent")]
print(analyze(segments).summary())
```

```
call report: 2 parties, 41.0s from first to last speech (41.0s recorded)
source: <segments> - 6 diarized segments supplied by the caller

  party        talk  share  turns  avg turn  longest  interrupts  reply gap   wpm
  agent       35.8s    88%      3     11.9s    18.0s           1       0.0s     -
  customer     5.0s    12%      2      2.3s     2.5s           0       0.9s     -

  overall: silence 5% (longest 1.5s at 0:20), overlap 4%, 5 turns, typical reply after 0.2s, 1 interruption

flags:
  - agent did most of the talking: 88% of the talk time.

Every figure is only as good as the speech segmentation behind it (6 diarized segments supplied by the caller).
```

The agent's 1.0 s talk-over at 23.0 s is an interruption. The customer's "mm-hm" at
12.0 s and the agent's reply that starts 0.3 s early at 8.7 s are not: both stay inside
the 0.5 s grace window.

A stereo recording, where each channel is one party, is one line too:

```python
report = analyze("call.wav", speakers=["agent", "customer"])
report["customer"].talk_share, report.response_latency_s, report.flags
```

## What it does

- **Talk time and share per party.** The union of each party's speech, so overlapping
  segments from one speaker are merged and never counted twice.
- **Interruptions, with a grace window.** Starting to talk while the other party is
  still speaking, and then overlapping them for longer than `grace_s` (0.5 s by
  default). A backchannel ("mm-hm", "right") or a reply that starts a beat before the
  other finishes stays under the grace window, is counted as a backchannel, and is not
  an interruption. Each interruption is listed with its time, who, over whom, and for
  how long.
- **Turns and monologues.** A turn is a stretch holding the floor; it ends when the
  other party takes the floor, or when its owner pauses for longer than
  `max_turn_pause_s`, so a hold does not become one long monologue. Speech made
  entirely inside the other party's turn does not take the floor.
- **Silence, overlap and response latency.** Silence and overlap are shares of the time
  from the first speech to the last, so ring time and the tail after hang-up do not
  count. Response latency is the median gap before one party replies to the other,
  overall and per party.
- **Pace.** Words per minute of transcribed speech, when the segments carry text
  (`(start_s, end_s, speaker, text)`). Chinese and Japanese written without spaces count
  one word per character, a rough approximation; when the transcript separates their
  words with spaces, each spaced run counts as one word.
- **Plain-language flags.** One party doing most of the talking, a one-sided call, a
  long monologue, repeated interruptions, heavy talk-over, long or frequent silence,
  slow replies, very fast or very slow speech. A balanced call gets no flags.
- **Crosstalk detection for stereo files.** When one party leaks into the other's
  channel (recorder crosstalk, handset or line echo), the leaked copy rises and falls
  with the source channel. That is measured, reported with its level and direction,
  and a frame only counts as the quieter channel's own speech if it stands clearly
  above the leak predicted for it. Two channels carrying the same audio (a mono call
  saved as stereo) get a single flag saying the parties cannot be told apart, instead
  of a report full of talk-over that never happened.
- **Takes the input you have.** A stereo `.wav` (8/16/24/32-bit PCM, float,
  WAVE_FORMAT_EXTENSIBLE, and the G.711 mu-law and A-law that call recorders write), a
  pair of mono arrays or a 2-D array, or segments as tuples, dicts, objects, a pandas
  DataFrame, a pyannote-style annotation, or a `.csv`/`.tsv`/`.json` file. That is how
  it works with speaker-diarize-lite or any other diarizer.
- **Honest edges.** A one-sided call reports the silent party with zero talk; an empty
  call returns a report of zeros; segments out of order are sorted; everything is
  deterministic; the caller's data is never modified. Every decision made about the
  input is recorded in `report.notes`.

What it does not do: it does not transcribe, recognise speakers, or separate voices
from a mono recording (diarize it first and pass the segments). On audio, the speech
detector is an energy threshold set from each channel's own noise floor: it is good at
the clean one-party-per-channel layout call recorders produce, and it will count a
cough, a keyboard or loud background noise on a channel as that party speaking. During
crosstalk, quiet speech that overlaps the other party's loud speech can be missed.
Metrics that depend on overlap (interruptions, backchannels) are the most sensitive to
segmentation quality; treat them as indicators, not as a verdict on anyone.

## API

```python
analyze(source, *, sample_rate=None, speakers=None) -> CallReport
```

| Argument | Meaning |
| --- | --- |
| `source` | a stereo `.wav` path; a pair of mono arrays or a 2-D samples x channels array; or segments `[(start_s, end_s, speaker[, text]), ...]` as tuples, dicts, objects, a DataFrame, or a `.csv`/`.tsv`/`.json` path |
| `sample_rate` | samples per second, for array input only |
| `speakers` | party names: one per channel (or `{channel_index: name}`) for audio; for segments, a list declaring the parties (absent ones get zero talk) or a dict renaming labels, e.g. `{0: "agent", 1: "customer"}` |

`CallAnalyzer(grace_s=0.5, max_turn_pause_s=3.0, min_speech_s=0.12, bridge_gap_s=0.3,
dominance_share=0.7, long_monologue_s=60.0, long_silence_s=10.0).analyze(...)` is the
same with every setting exposed; `min_speech_s` and `bridge_gap_s` only apply to audio.

`CallReport` carries:

| Member | Meaning |
| --- | --- |
| `.speakers` | `{name: SpeakerStats}` in party order; `report["agent"]` and `report[0]` work too |
| `.silence_share`, `.longest_silence_s`, `.longest_silence_at_s` | silence between the first and last speech |
| `.overlap_share` | share of that time with two or more parties talking |
| `.turn_count`, `.interruption_count` | across the call |
| `.response_latency_s` | median gap before a reply, or `None` if nobody replied to anyone |
| `.duration_s`, `.speech_start_s`, `.speech_end_s`, `.speech_span_s` | the time frame |
| `.flags` | plain-language observations |
| `.notes` | what was done to the input (sorted, merged, crosstalk found, ...) |
| `.method`, `.method_detail`, `.source` | where the speech timeline came from |
| `.bleed` | `BleedCheck(detected, same_audio, paths, same_audio_pairs)` for audio, else `None` |
| `.detection` | per-channel noise floor, peak and threshold in dBFS, for audio |
| `.segments`, `.turns`, `.interruption_events` | the timeline, the turns and each interruption |
| `.summary()` | the human-readable report above |
| `.explain()` | how each figure was measured, with the settings used |
| `.to_dict()` | the whole report as a JSON-safe dict |
| `.to_frame()` | one row per party as a pandas DataFrame, overall figures in `.attrs` |

`SpeakerStats` fields: `speaker`, `talk_time_s`, `talk_share`, `turns`, `mean_turn_s`,
`longest_monologue_s`, `interruptions` (made), `interrupted`, `backchannels`,
`response_latency_s`, `replies`, `words`, `words_per_minute` (the last two `None`
without a transcript).

Bad input raises `ValueError` with a message that says what to do, for example a mono
recording ("run a diarizer on it first ... and pass its segments") or a segment that
ends before it starts.

## CLI

```
call-ai-metrics call.wav                                # print the summary
call-ai-metrics call.wav --speakers agent customer      # name the channels
call-ai-metrics segments.csv --json                     # to_dict() as JSON
call-ai-metrics segments.json --output report.json      # also write it to a file
call-ai-metrics segments.csv --grace 0.8 --explain      # wider grace, show the method
```

Segment files are `.csv`/`.tsv` with `start,end,speaker[,text]` columns (a header row is
optional) or `.json` (a list of segments, or an object with a `segments` list). Exit
code is 0 when the call was measured, 1 when it held no speech, and 2 when the input
could not be read. Output is UTF-8 whatever the console, so non-Latin names survive a
pipe.

## License

MIT
