# meeting-intelligence

Turns a meeting **transcript** into decisions, action items and a summary. It reads text, never
audio: there is no recorder, no decoder and no speech model in this package, so getting words out
of a recording is a separate job done before `meeting-intelligence` is called. Export captions from
your meeting tool (Zoom, Teams and Meet all produce a `.vtt` file), or use a dedicated
speech-to-text package - the speech and audio packages in this family, or any tool that writes
`.srt`/`.vtt`/plain text - and hand the result to `analyse()`.

Nothing is downloaded, no model is loaded, and every decision and action can be traced back to the
published cue phrase that produced it.

## Install

```
pip install meeting-intelligence
```

## Quickstart

```python
import meeting_intelligence as mi

report = mi.analyse("""
Alice: We decided to go with Postgres for the event store.
Bob: I'll write the migration by Friday. Can you review it, Alice?
Alice: Sure. What do we do about the old rows?
""")
print(report.summary())
print(report.action_items[0].owner, "->", report.action_items[0].due)
```

That is the whole API for most people. Pass plain text, a list of
`{"speaker", "text", "start"}` turns, a path to a `.txt`/`.vtt`/`.srt`/`.md` file, or a WebVTT or
SubRip string - the parser works out which it got and reports it as `report.source_format`.

## What it does

- **Decisions** come from a published cue list, `mi.DECISION_CUES`: *we decided*, *the decision is*,
  *let's go with*, *agreed to*, *the plan is*, *settled on*, *signed off on* and a dozen more. Each
  `Decision` carries the `cue` that matched and a `confidence`. A cue inside a negation ("we have
  **not** decided", "still deciding") or inside a question is never reported as settled.
- **Action items** come from `mi.ACTION_RULES`: *I'll take*, *I will*, *let me*, *X will*, *X owns*,
  *can you*, *Name, can you*, *assigned to X*, *please*, *action item:*, *follow up*, *we need to*,
  *needs to be*, *make sure*, *todo*. Each `Action` is `Action(text, owner, due, confidence)` and
  carries the `cue` that matched. *I'll* on its own is not a commitment: **I'll be honest**, *I'll
  say*, *I'll admit*, *I'll second that*, *I'll bet* and *I'll leave it there* are remarks, not
  tasks, and are not reported as action items.
- **Owners are people the report can name.** *I* and *I'll* become the speaker of the turn. *can
  you* becomes the person addressed: a name written into the sentence wins, whether it leads
  (*Alice, can you review it?*) or trails (*Can you review it, Alice?*), and otherwise it is the
  other person on a two-person call or the next person to speak. A named owner is resolved against
  the roster - the speakers found in the transcript plus anything passed to `speakers=` - so a bare
  first name matches a fuller roster name. **An owner that cannot be resolved is left `None` rather
  than guessed**: a capitalised word that matches nobody is not a person, so *Hey*, *Sorry*,
  *Folks* and *Hopefully Monday* never appear as owners. Pass `speakers=[...]` to name somebody who
  is mentioned but never speaks.
- **Due dates** come from `mi.DUE_PATTERNS`: *by Friday*, *end of week*, *EOD*, *by 2026-04-01*,
  *within two weeks*, *ASAP*. The leading "by"/"before"/"due" is trimmed, so `due` reads as a date.
- **Questions** are matched to whether a later turn answered them. A different speaker replying
  within six turns counts as the answer when the reply shares a content word with the question, or
  when a *who* question is answered by naming somebody ("Assigned to Dana."). A bare affirmation
  ("yes", "no", "it is", "we can") counts only in the next two turns, where a direct answer
  actually lands, and a sign-off ("That is everything. Thanks all.") never counts - an unanswered
  question stays open rather than being paired with an unrelated later turn.
  `report.open_questions` is what nobody answered. Tag questions and filler ("Right?", "Okay?")
  are not counted as questions.
- **Topics** are segments of the discussion. Each turn becomes a hashed bag-of-words vector, the
  cosine similarity across every turn boundary is measured with a window on each side, and the
  discussion is cut at the *valleys*. The label is drawn from the words that are distinctive to that
  segment relative to the rest of the meeting - meeting scaffolding, greetings, dates and the cue
  verbs themselves (*decided*, *agreed*, *assigned*) are kept out so the label names the subject,
  and an acronym keeps its capitals (`PR`, not `Pr`). `numpy` does the arithmetic.
- **Participation** is per speaker: turns, words, share of the talking, longest monologue, questions
  asked and interruptions. Interruptions are counted from timing overlap when the source is timed,
  and from cut-off markers (`--`, `...`) when it is not; which rule was available is reported as
  `report.interruptions_basis`, so the number is never ambiguous.
- **A heuristic that finds nothing says so.** A transcript too short to split into topics, a
  transcript with no speaker labels, and a source with no way to detect interruptions each add a
  line to `report.warnings` instead of silently returning something degenerate.
- **An empty transcript returns an empty report**, not an exception.
- Deterministic: the word hash is `zlib.crc32`, so the same transcript gives the same report in
  every process and on every machine.
- Unicode throughout - names, text, and CJK sentence enders (`。`, `！`, `？`).

## Timestamps and formats

| Source | Detected as | Timings |
| --- | --- | --- |
| `WEBVTT` text or `.vtt` file | `vtt` | `00:01:02.500`, hours optional, milliseconds kept |
| Numbered cues with `00:00:04,000` | `srt` | comma or dot before the milliseconds |
| `Alice: ...` lines, optionally `[00:01:02]` | `labelled-text` | from the stamp, when present |
| Prose with no labels | `plain-text` | none; paragraphs become turns |
| `[{"speaker": ..., "text": ..., "start": ...}]` | `turns` | `start` as seconds or `"00:01:02"` |
| `""` or `[]` | `empty` | - |

Consecutive cues from the same speaker are merged into one real turn, so a caption track that
breaks a sentence across four cues still counts as one turn and one monologue.

## Using your own LLM (optional)

```python
report = mi.analyse(transcript, llm=my_model)   # callable(prompt) -> str
```

The `llm` is used **only** to reword the summary paragraph and to tidy the phrasing of action items
that were already extracted. It never finds new decisions, owners or dates, it is never required,
and the package is fully useful without it. If your callable raises, returns nothing, or returns
something unusable, the extractive result is kept and the reason is appended to `report.warnings`.
`report.llm_used` says whether any of its output was used, and `Action.source_text` keeps the
original sentence whenever the wording was changed.

## API

```python
mi.analyse(transcript, *, speakers=None, llm=None) -> MeetingReport
mi.parse_transcript(source) -> TurnList        # list[Turn], with .source_format
mi.detect_format(source) -> str                # name the shape without keeping the parse
```

`mi.analyze` is the same function under the US spelling.

`MeetingReport` fields: `summary_text`, `decisions`, `action_items`, `questions`, `topics`,
`participation`, `duration`, plus `turns`, `speakers`, `source_format`, `n_turns`, `n_words`,
`interruptions_basis`, `llm_used` and `warnings`.

`MeetingReport` methods and properties: `summary()` (plain text for a terminal), `to_dict()`
(JSON-safe), `to_markdown()` (minutes to paste), `open_questions`, `answered_questions`,
`owned_actions`, `unowned_actions`, `duration_text`, `is_empty` and `actions_for(owner)`.

Record types, all with `to_dict()`: `Decision(text, speaker, cue, confidence, ...)`,
`Action(text, owner, due, confidence, ...)`, `Question(text, speaker, answered, answer_text, ...)`,
`Topic(label, keywords, start_turn, end_turn, ...)`, `Participation(speaker, turns, words, share,
longest_monologue, interruptions, ...)` and `Turn(index, speaker, text, start, end)`.

The pieces are exported individually too, for callers who want one of them on its own:
`extract_decisions`, `extract_actions`, `extract_questions`, `segment_topics`,
`measure_participation`, `find_due`, `unanswered` and `format_duration` - along with the cue lists
`DECISION_CUES`, `ACTION_RULES` and `DUE_PATTERNS`.

## CLI

```
meeting-intelligence standup.vtt                          # the text summary
meeting-intelligence notes.txt --speakers "Alice,Bob Chen"
meeting-intelligence notes.txt --json > report.json
meeting-intelligence notes.txt --markdown --output minutes.md
meeting-intelligence notes.txt --actions                  # just the task lines
cat notes.txt | meeting-intelligence -                    # transcript on stdin
```

`meeting-intelligence --help` lists every option. Output is UTF-8 even through a pipe, so a
transcript full of non-ASCII names prints without a `UnicodeEncodeError`.

## License

MIT
