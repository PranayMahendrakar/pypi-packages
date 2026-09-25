# voice-commands-ai

This package maps TEXT to actions and does not recognise speech itself: give it the transcript from any speech-to-text engine and it turns "um could you set the tempreture to twenty one degrees please" into `thermostat(value=21)`. The matching is a heuristic (edit distance, filler-word lists, a spoken-number parser and a weighted alignment score), not a language model, and it needs no model, no network and no dependencies beyond the standard library.

## Install

```bash
pip install voice-commands-ai
```

## Quickstart

```python
from voice_commands_ai import Commands

cmds = Commands()
cmds.add("set temperature to {value:number} degrees", lambda value: f"Heating to {value} C", name="thermostat")
cmds.add("turn {state:on|off} the {device:text}", name="switch")
print(cmds.run("um could you set the tempreture to twenty one degrees please"))
print(cmds.match("turn the kitchen lights off").summary())
```

```text
Heating to 21 C
Matched command 'switch' with confidence 0.93 (floor 0.60)
Heard:   "turn the kitchen lights off"
Pattern: turn {state:on|off} the {device:text}
Slots:
  state = 'off'
  device = 'kitchen lights'
Scores: 95% of the pattern found, 100% of the words explained
Out of order: {state}
```

Text that matches nothing returns `None` instead of the closest poor guess:
`cmds.match("what time is it")` is `None`, because the best candidate scored far
below the confidence floor (0.6 by default).

### Plugging in a transcriber

`listen` takes any function that turns your audio into text, calls it, and
matches the result. The transcriber is yours: a cloud speech API, a local
engine, or a test stub like the one below.

```python
def transcribe(audio):
    return "please turn on the porch light"   # replace with your speech-to-text call

match = cmds.listen(transcribe, audio=b"...raw audio bytes...")
if match:
    print(match.name, match.slots)             # switch {'state': 'on', 'device': 'porch light'}
    cmds.run(match)                            # calls the switch handler, if it has one
```

If the transcriber returns several guesses (a list of strings, best first), the
one that matches a command best is used.

## What it does

- **Patterns with typed slots.** `{name}` or `{name:word}` is one word,
  `{name:number}` a number, `{name:on|off}` one of the listed choices (a choice
  can be several words: `{room:living room|kitchen}`), `{name:text}` free text up
  to the next fixed word or to the end, and `[words]` are optional words.
- **Filler words cost nothing.** Hesitations ("um", "uh", "erm") are dropped;
  politeness and framing ("please", "could you", "i'd like to", "go ahead and",
  "for me", "just", "okay") can be skipped for free, and "please" is trimmed off
  the end of a free-text slot. Add your own with `Commands(fillers=[...])`.
- **Spoken numbers.** "twenty one" is 21, "a hundred and five" 105, "two point
  five" 2.5, "minus three" -3, plus "twelve hundred", "1.5 million", "two and a
  half", "three quarters", "twenty first", "nineteen eighty four" and digits read
  one by one ("four five six" is 456). Whole numbers come back as `int`.
- **Small transcription errors.** A heard word can stand in for a pattern word
  by edit distance (one typo in 4-5 letters, two in 6-8, three in longer
  words, none in words of three letters or fewer), as a plural, with a doubled
  letter dropped ("of" for "off"), as a rough English sound-alike ("whether"
  for "weather"), or split in two or run together. Every such substitution is
  listed in `Match.corrections`, and a word read fuzzily earns less confidence
  than one heard exactly.
- **Word order within reason.** A fixed word, number or choice said somewhere
  else ("turn the lights off" for "turn {state:on|off} the {device}") is still
  found, at a small cost, and listed in `Match.reordered`. Free-text and
  one-word slots must stay in order.
- **A confidence floor.** Confidence is `coverage^1.5 * precision^0.5`:
  coverage is the weighted share of the pattern that was found (fuzzy words earn
  their similarity squared, and a word contradicted by a different word in its
  place loses extra), precision the weighted share of the words the pattern
  explains, with chatter before or after the command counting half. A pattern
  none of whose fixed words were heard exactly is scaled by 0.85. Below
  `min_confidence` (default 0.6) `match` returns `None`.
- **The more specific pattern wins.** When two commands both clear the floor
  within 0.05 of each other, the one with more fixed words and fewer open slots
  wins ("play {song:text} by {artist:text}" beats "play {song:text}" for "play
  hello by adele"), and the other is reported in `Match.alternatives`.
- **Handlers fail loudly.** If a handler raises, `run` raises `CommandError`
  naming the command, with the original exception as its cause.

Limits worth knowing: a `{name:text}` slot accepts anything, so "call
{contact:text}" will happily match "call of duty is a great game"; put fixed
words around text slots where you can. Sound-alike matching, fillers and spoken
numbers are English. Words are split on spaces and punctuation, so for languages
written without spaces (Chinese, Japanese, Thai) the transcriber has to separate
the words. The filler lists and weights are defaults chosen on hand-written test
transcripts, not learned from data; check `Match.summary()` or `Commands.rank()`
when a result surprises you.

## API

`Commands(*, min_confidence=0.6, fillers=())` holds the commands.

- `.add(pattern, handler=None, *, name=None, examples=None)` registers a command
  and returns the `Commands`, so calls chain. `name` defaults to the handler's
  function name, else the pattern. A handler that cannot accept the pattern's
  slots as keyword arguments is refused here, not at run time. Each of
  `examples` must match its own pattern, which catches pattern typos early.
- `.match(text) -> Match | None` gives the best command, or `None` for empty
  text, text of only filler, or text that matches nothing above the floor.
- `.run(text, *, default=None)` matches and calls the handler with the slots
  as keyword arguments, returning its result. It also accepts a `Match` (from
  `listen`). A command registered without a handler returns its `Match`;
  nothing matching returns `default`. A handler that raises becomes a
  `CommandError` with `.name`, `.match` and `.original`.
- `.listen(transcribe, audio) -> Match | None` calls `transcribe(audio)`, which
  must return a string, a list of alternative strings, or `None` for silence,
  and matches the text. Errors raised by the transcriber propagate unchanged.
- `.rank(text) -> list[Match]` scores every command, best first, including
  those below the floor. Use it to see why something did or did not match.
- `len(cmds)`, `"name" in cmds`, and iterating gives the registered `Command`s.

`Match` fields: `name`, `slots` (typed values), `confidence` (0-1), `pattern`,
`alternatives` (other commands above the floor), and the evidence: `text`,
`slot_words` (the words each slot came from), `corrections`, `ignored`,
`extra`, `missing`, `reordered`, `replaced`, `coverage`, `precision`,
`specificity`, `threshold` and `chosen_because`. `.summary()` explains it in
plain text; `.to_dict()` gives the same as JSON-safe data.

`parse_number(text) -> float | None` reads one spoken English number, or returns
`None` when the whole text is not a number (`"twenty one degrees"` is not).
Zero is `0.0`, so compare the result with `is None`.

## CLI

```bash
voice-commands-ai -c "set temperature to {value:number} degrees" "um set the temperature to twenty one degrees please"
voice-commands-ai -c "thermostat = set temperature to {value:number} degrees" \
                  -c "switch = turn {state:on|off} the {device:text}" "turn the kitchen light off"
voice-commands-ai -f commands.txt transcripts.txt --json > matches.json
some-transcriber | voice-commands-ai -f commands.json --fail-on-no-match
voice-commands-ai --number "a hundred and five" "two point five" "minus three"
```

Utterances come from the arguments, from `.txt` files named as arguments (one
per line) or from stdin. A commands file is JSON (`{"name": "pattern"}`, a list
of patterns, or a list of `{"pattern", "name", "examples"}` objects) or text with
one `name = pattern` per line. `--all` shows how every command scored,
`--min-confidence` moves the floor, `--output PATH` writes JSON. Exit status is
0 on success, 1 with `--fail-on-no-match` when an utterance matched nothing,
and 2 when the commands or input could not be read.

## License

MIT
