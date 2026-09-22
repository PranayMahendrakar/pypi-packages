# multilingual-text

Detect what language a string is in, clean it up, and romanise it - three jobs that
normally mean three libraries and three different result shapes, behind one small API
with no dependencies at all.

## Install

```
pip install multilingual-text
```

Python 3.9 or newer. The standard library is the only requirement: no model download,
no `langdetect`, no network access, ever.

## Quickstart

```python
import multilingual_text as mlt

d = mlt.detect("El rápido zorro marrón salta sobre el perro perezoso", top=3)
print(d.summary())
print(mlt.normalize("  “Smart” quotes — and\u00a0hard spaces  "))
print(mlt.transliterate("नमस्ते दुनिया", to="ascii"), "|", mlt.transliterate("Привет, мир!"))
print(mlt.script_of("مرحبا"), "| rtl:", mlt.is_rtl("مرحبا"), "| empty:", mlt.detect("").language)
```

```
multilingual-text: es (Spanish), confidence 0.95, script Latin; reliable; next: pt 0.02, it 0.01
"Smart" quotes - and hard spaces
namaste duniya | Privet, mir!
Arabic | rtl: True | empty: und
```

## What it does

- **`detect` covers 30 languages** - en, es, fr, de, it, pt, nl, pl, sv, tr, vi, id, ru,
  uk, el, ar, he, hi, mr, bn, ta, te, gu, pa, kn, ml, zh, ja, ko, th - by profiling the
  character scripts first and then scoring the common words of every language that
  script allows. Scripts that map to a single language (Thai, Hebrew, Tamil) are decided
  on the spot; Latin and Cyrillic are decided on words and on which accented letters the
  orthography does and does not use.
- **Every result says how much to believe it.** `.confidence` is a probability that
  already accounts for how short the text is, and `.reliable` is the one flag to branch
  on. Three letters of Spanish come back as `es` with `reliable=False`, not as a
  confident answer.
- **Nothing raises on odd input.** An empty string, a lone character, a line of digits,
  punctuation on its own, a bare URL or e-mail address - all come back as an ordinary
  result with `language="und"` and `confidence=0.0`. A URL is not written in any
  language, so URLs and e-mail addresses are removed before scoring; one *inside* a
  sentence simply drops out and the rest of the sentence is detected normally.
- **Mixed-script text reports the dominant script and flags the mixture**, with the full
  breakdown in `.scripts`. Japanese is not counted as "mixed" merely for using kanji and
  kana together.
- **`normalize` is idempotent.** `normalize(normalize(x)) == normalize(x)` for every
  combination of its seven options - which is what makes it safe in front of a cache
  key, a dedup hash or a search index.
- **`transliterate` covers Devanagari, Cyrillic and Greek properly**, and says so when
  it cannot help. See [Transliteration](#transliteration) below.
- **Deterministic and offline.** The tables are plain data compiled into the package.
  The same input gives the same answer in every process and on every machine.

## How accurate is it, honestly

Short strings are hard, and this package does not pretend otherwise.

| text length | what to expect |
| --- | --- |
| a full sentence (25+ letters) | reliable for all 30 languages; `.reliable` is `True` |
| 10-25 letters | usually right, `.confidence` drops, `.reliable` often `False` |
| 1-9 letters | a guess. `.reliable` is always `False`. A bare `"A"` is reported as `en` with a confidence near 0.06, because that is what one Latin letter is worth |

Non-Latin scripts that only one supported language uses (Thai, Hebrew, Greek, Tamil,
Telugu, Gujarati, Gurmukhi, Kannada, Malayalam, Bengali, Hangul) are decided by the
script alone and are effectively always right - but note that this also means Persian
and Urdu come back as `ar`, and Serbian and Bulgarian come back as `ru` or `uk`, because
those languages are not in the list. Check `.script` when that distinction matters.

The two pairs that share a script are separated on words and letters: Russian against
Ukrainian (`і ї є ґ` against `ы э ъ ё`) and Hindi against Marathi (`आहे`, `आणि`, `ळ`
against `है`, `और`). Both are solid on a sentence and coin-flips on a single word.

## API

### `detect(text, *, top=1) -> Detection`

Ranks `top` candidates in total. `top=1` fills `.language` only; `top=3` also puts the
two runners-up in `.alternatives`.

### `Detection`

A frozen dataclass.

| attribute | meaning |
| --- | --- |
| `.language` | ISO 639-1 code, or `"und"` when there is nothing to go on |
| `.name` | English name of the language, for printing |
| `.confidence` | `0.0`-`1.0`, already scaled down for short text |
| `.script` | dominant writing system, e.g. `"Latin"`, `"Devanagari"`, `"Han"` |
| `.alternatives` | runners-up as `[(code, confidence), ...]` |
| `.reliable` | `True` only when the text is long enough and the winner is clear |
| `.mixed` | `True` when two or more writing systems are really in play |
| `.scripts` | every script found, as fractions of the text, largest first |
| `.characters`, `.words` | how much evidence there was |
| `.ranked` | `.language` followed by `.alternatives`, as one list |
| `.rtl` | `True` when the dominant script is right-to-left |
| `.summary()` | one line of plain ASCII, safe for any console |
| `.to_dict()` | JSON-safe dict of everything above |

### `normalize(text, *, form="NFC", case=None, whitespace=True, punctuation=False, digits=False, quotes=True, diacritics=False) -> str`

| option | default | before | after |
| --- | --- | --- | --- |
| `form` | `"NFC"` | `"école"` (decomposed) | `"école"` (composed). Also `"NFD"`, `"NFKC"`, `"NFKD"`, or `None` to leave the encoding alone |
| `case` | `None` | `"Straße"` | `"strasse"` with `case="fold"`, `"straße"` with `"lower"`, `"STRASSE"` with `"upper"`, `"Straße"` with `"title"` |
| `whitespace` | `True` | `"  a\u00a0 b  "` | `"a b"` - no-break and exotic spaces become ordinary ones, zero-width characters go, runs collapse, ends are stripped |
| `punctuation` | `False` | `"Hello, world!"` | `"Hello world"`. It joins hyphenated words: `"co-operate"` becomes `"cooperate"` |
| `digits` | `False` | `"٤٢ और ४२"` | `"42 और 42"` - every Unicode decimal digit becomes ASCII, Arabic-Indic and Devanagari included |
| `quotes` | `True` | `"“a” – b…"` | `'"a" - b...'` - smart quotes, guillemets, dashes, ellipses and primes folded to ASCII |
| `diacritics` | `False` | `"Crème Brûlée"` | `"Creme Brulee"`. Letters Unicode cannot decompose (`ø ł đ æ œ ß`) are mapped by hand. Indic vowel signs and Arabic vowel points are **not** touched: `"नमस्ते"` stays `"नमस्ते"` |

`normalize(normalize(x)) == normalize(x)` holds for every combination of these. The
test checks the full matrix of options against a corpus of inputs, not one string -
including the characters where this is genuinely hard to get right: the Latin letters
whose accent sits on a base Unicode refuses to decompose (`Ǣ ǣ Ǽ ǽ Ǿ ǿ`) and the Greek
ano teleia, whose canonical form is itself foldable.

### `transliterate(text, *, to="latin") -> Transliteration`

Covered: **Devanagari** (IAST), **Cyrillic** (BGN/PCGN for Russian; the Ukrainian
national system when the text contains `і ї є ґ`, so `"Київ"` gives `"Kyiv"`) and
**Greek** (ISO 843 (modern Greek: beta is "v", phi is "f"), accents folded first; the ano teleia `·` becomes `;` and the Greek
question mark `;` becomes `?`, while an ordinary typed ASCII `;` is left alone).

**Not covered, and returned completely unchanged: every other script** - Arabic,
Hebrew, Han, kana, Hangul, Thai, Tamil, Telugu, Gujarati, Gurmukhi, Kannada, Malayalam,
Bengali and the rest. That is deliberate. A half-right romanisation is worse than none,
because it looks finished.

Nothing is raised for an unsupported script. The result is a `str` subclass that carries
the reason:

```python
out = mlt.transliterate("مرحبا")
out == "مرحبا"      # True - byte for byte
out.supported        # False
out.untouched        # ('Arabic',)
out.note             # 'Arabic not covered and returned unchanged'
```

`to="latin"` keeps the scholarly diacritics (`bhāṣā`, `Athēna`); `to="ascii"` folds them
away (`bhasa`, `Athena`). Capitalisation is carried across, so `"ШКОЛА"` gives
`"SHKOLA"` and `"Привет"` gives `"Privet"`.

### `script_of(text) -> str`

The dominant Unicode script name: `"Latin"`, `"Cyrillic"`, `"Devanagari"`, `"Han"`,
`"Hiragana"`, and so on. Digits or punctuation on their own are `"Common"`; an empty
string is `"Unknown"`.

### `is_rtl(text) -> bool`

`True` when most of the letters are right-to-left (Arabic, Hebrew, Syriac, Thaana,
N'Ko). Mixed text goes with the majority, so `is_rtl("مرحبا hi")` is `True` and
`is_rtl("Hello مرحبا")` is `False`.

### `detect_batch(texts) -> list[Detection]`

One `Detection` per input, in order. Passing a single string instead of a list raises
`TypeError` with an explanation, because that mistake is easy to make and silent
otherwise.

## CLI

```
multilingual-text --help
multilingual-text "El perro es muy grande"
multilingual-text "Привет, мир" --mode transliterate --to ascii
multilingual-text "  “smart” quotes  " --mode normalize --case fold --diacritics
multilingual-text "私は日本語を勉強しています" --mode all --json
echo "bonjour tout le monde" | multilingual-text - --top 3
multilingual-text --file notes.txt --mode detect --output report.json
```

`--mode` is one of `detect` (the default), `normalize`, `transliterate` or `all`.
`--json` prints the same thing `to_dict()` would give you. Output is written as UTF-8
whatever the console codepage is, so piping non-Latin text into a file or a log never
raises `UnicodeEncodeError`.

## Which scripts transliterate

- **Cyrillic** - Russian, Ukrainian (national system, so Kyiv rather than Kyyiv),
  Bulgarian, Serbian and Macedonian, including the letters only the last two use.
- **Greek** - ISO 843, which romanises *modern* Greek: beta becomes "v" and phi "f",
  the way the letters are spoken today. A classical scheme would give "b" and "ph".
- **Devanagari** - Hindi and Marathi.

Anything else is returned unchanged, and the result reports `supported == False` so you
can tell. That flag is the point: a half-romanised string that claims success is worse
than an honest refusal, because nothing downstream can detect it.

## License

MIT. Copyright (c) 2026 Pranay Mahendrakar.
