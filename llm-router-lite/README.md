# llm-router-lite

Send each prompt to the cheapest model that can handle it, and fall back to the next one when a call fails.

## Install

```bash
pip install llm-router-lite
```

No dependencies. Standard library only, and no model is ever downloaded.

**This package is not an API client.** It never talks to OpenAI, Anthropic, Ollama or anything else, and it holds no keys. You give it a handler per model - a plain function that takes a prompt and returns a string - and it decides which of *your* functions to call. There is no network code in this package at all.

## Quickstart

```python
from llm_router_lite import Router

router = Router()
router.add("small", lambda p, **kw: "small says: " + p, cost=0.0002, quality=0.4)
router.add("large", lambda p, **kw: "large says: " + p, cost=0.01, quality=0.95)

hard = "Explain step by step why this design scales, compare it with a queue-based approach, and then recommend one."
print(router.route("What is 2 + 2?").reason)
print(router.complete(hard).summary())
```

```
'small' for a simple prompt (complexity 0.11); balanced routing leans on cost for a simple prompt; it is the cheapest of the 2 that qualify; fallbacks, in order: large.
large answered in 0.0 ms for about 0.00283
  large: ok in 0.0 ms
```

The easy prompt goes to the cheap model, the hard one to the good model, and nothing in the line-up had to be described twice.

Swap the two lambdas for your real calls and nothing else changes.

## What it does

- **Scores the prompt first.** `complexity(prompt)` gives a 0-1 score and a band of `simple`, `moderate` or `hard`, from five documented signals: length, question depth, reasoning words, code markers and the number of distinct instructions. Same prompt, same score, every time, in any script it can read - and it says so in `warnings` when it cannot read one.
- **Routes on cost, speed and quality.** `prefer="balanced"` (the default) reads the band: an easy prompt leans on cost, a hard one leans on quality. `"cheap"`, `"fast"` and `"quality"` say it plainly instead. Cost and latency are scored as ratios against the best in the field - the cheapest scores 1.0, one costing ten times more scores 0.1 - so registering a third, pricier model cannot change which of the other two wins, and `prefer="cheap"` picks the cheapest thing that qualifies however many models are registered.
- **Keeps a prompt off a model that cannot handle it.** Under the default `prefer="balanced"`, a hard prompt is not handed to a model rated below 0.60 quality while a better one is registered, and a moderate one wants 0.35. The floor demotes rather than discards: the weak model still stands behind as a fallback, listed in `route.demoted`, because a weak answer beats no answer when the good model is down. If nothing clears the floor at all, the best available model is used anyway and the route says so - it never returns nothing.
- **Respects hard constraints.** `require=("local",)` keeps a prompt off the network. `max_cost=0.002` refuses to route above a price. A model with `max_tokens` set is skipped when the prompt would not fit.
- **Falls back in order.** `complete()` tries the pick, then each alternative. A handler that raises is caught and recorded, never propagated raw. If every model fails you get one `AllModelsFailed` naming the last error, with every attempt attached.
- **Explains itself.** `route.reason` is a sentence, not a score dump: which model, for what kind of prompt, why it won, what was set aside and what stands behind it.
- **Counts what happened.** `router.stats()` gives calls, failures, total cost and mean latency per model.

## How the score is built

| signal | weight | what it measures |
| --- | --- | --- |
| length | 0.15 | words, reaching 1.0 at 60 words |
| questions | 0.15 | question marks plus question words, 1.0 at 3 |
| reasoning | 0.35 | words such as "explain", "compare", "trade-off", 1.0 at 3 |
| code | 0.15 | markers such as a fenced block, `def `, a traceback, 1.0 at 2 |
| instructions | 0.20 | distinct instructions - list items, imperative sentences, "and then" - 1.0 at 4 |

Reasoning carries the most weight on purpose: a short prompt asking for a derivation is harder than a long one pasting a log, so length is the smallest term.

Score below 0.22 is `simple`, below 0.45 is `moderate`, and 0.45 or above is `hard`. An empty prompt scores 0.0 and routes as simple.

### Which languages the score reads

Words are counted in every script. A script that separates words with spaces is counted word by word; Chinese, Japanese and Korean are written without spaces, so a run of them counts one word per two characters instead of collapsing to one.

The question, reasoning and "and then" markers are word lists, so each one only fires in a language it is written for. They cover **English, Chinese, Japanese, Korean, Russian, Arabic, Hindi, Greek and Hebrew**: the same hard request scores within a few hundredths of a point and bands `hard` in all of them.

A prompt with real substance written mostly in some other non-Latin script is still scored on its length and structure, and the result then carries a plain sentence in `complexity(prompt).warnings` - repeated in `route.reason` and in the log - saying the language markers could not be read and the score is lower than the prompt may deserve. The scorer never quietly returns 0.0 for text it could not read.

Cost is estimated, not measured: tokens are counted at 4 characters each, an answer of up to 256 tokens is assumed, and `estimated_cost = cost * tokens / 1000`. `cost` is whatever unit you registered - dollars, cents, credits. The result is rounded to 12 significant digits, not to a fixed number of decimal places, so the float noise of the multiplication is tidied off while prices as small as `1e-12` keep their order.

## API

```python
from llm_router_lite import Router, route, complexity

Router(models=None)
Router.add(name, handler, *, cost=0.0, latency_ms=None, quality=0.5, max_tokens=None, tags=())
Router.route(prompt, *, prefer="balanced", require=None, max_cost=None) -> Route
Router.complete(prompt, **kw) -> Completion
Router.stats() -> Stats
route(prompt, models, **kw) -> Route
complexity(prompt) -> Complexity
```

**`Router(models=None)`** - `models` may be a mapping of `name -> handler`, a mapping of `name -> options dict`, or a sequence of `Model` objects, option dicts or `(name, handler)` pairs. `Router()` starts empty.

**`.add(name, handler, *, cost=0.0, latency_ms=None, quality=0.5, max_tokens=None, tags=())`** - register one model and return the router, so `.add(...).add(...)` chains.

- `handler` - `callable(prompt, **kw) -> str`. Nothing is called until you call `complete()`; `route()` only compares numbers.
- `cost` - price of 1000 tokens, prompt plus answer, in any unit.
- `latency_ms` - typical round trip. Left out, a model is assumed to be as fast as the median of the models that do declare one.
- `quality` - your own 0-1 opinion of the answers.
- `max_tokens` - the largest prompt-plus-answer it can hold. A prompt that does not fit skips the model.
- `tags` - free labels such as `"local"` or `"vision"`, matched case-insensitively by `require`.

**`.route(prompt, *, prefer="balanced", require=None, max_cost=None) -> Route`** - decide, call nothing.

**`Route`** - `.model` (the name), `.reason` (plain language), `.estimated_cost`, `.alternatives` (fallbacks in order), `.complexity`, `.prefer`, `.estimated_tokens`, `.considered` (every candidate with its score), `.skipped` (name -> why it is out of the running), `.demoted` (name -> why it ranks behind but is still a fallback), `.cost_of(name)`, `.summary()`, `.to_dict()`.

**`.complete(prompt, **kw) -> Completion`** - route, call, fall back. `prefer`, `require`, `max_cost` and `route` are reserved keywords; every other keyword is passed to the handler.

**`Completion`** - `.text`, `.model`, `.attempts` (a list of `Attempt(model, ok, error, ms)`), `.cost`, `.route`, `.ms`, `.fallbacks_used`, `.summary()`, `.to_dict()`. `str(completion)` is the text.

**`.stats() -> Stats`** - `stats["small"]` gives a `ModelStats(name, calls, failures, total_cost, mean_latency_ms)` with `.successes` and `.failure_rate`; `stats.calls`, `.failures` and `.total_cost` are the totals. `total_cost` counts the calls that succeeded; `mean_latency_ms` covers every call, failures included. `.reset_stats()` zeroes them.

**`complexity(prompt) -> Complexity`** - `.score`, `.band`, `.words`, `.characters`, `.estimated_tokens`, `.signals` (the five sub-scores), `.reasons`, `.warnings` (empty unless the scorer could not read the prompt's script, or nothing matched at all), `.summary()`, `.to_dict()`.

**`route(prompt, models, **kw)`** - build a router and route in one line.

Errors are plain built-ins wherever one fits:

- `ValueError` when no models are registered, when `require` matches nothing, when the prompt fits in no model, and when nothing meets `max_cost` - that last message names the cheapest model and what it would cost.
- `TypeError` when a prompt is not a string or a handler is not callable.
- `AllModelsFailed` (a `RuntimeError`) when every model in the chain fails. It carries `.attempts`, `.route`, `.last_model` and `.last_error`, chains the original exception as `__cause__`, and has `.summary()` and `.to_dict()`.

## CLI

```
llm-router-lite "Why is the sky blue?"                    # how hard is this prompt
llm-router-lite complexity - < prompt.txt                 # ... from a file or a pipe
llm-router-lite route "Explain why this fails" --prefer cheap
llm-router-lite route "Summarise this" --require local --max-cost 0.002
llm-router-lite route "..." --model tiny:0:0.3:80:local --model big:0.01:0.95:900:cloud
llm-router-lite route "..." --models lineup.json --json --output route.json
llm-router-lite models                                    # the line-up being used
```

Without `--models` or `--model` an example line-up (`local-small`, `cloud-mid`, `cloud-large`) stands in, so the command does something useful immediately. A `--model` spec is `NAME[:COST[:QUALITY[:LATENCY_MS[:TAG,TAG]]]]`; `--models` takes a JSON list of option objects, or an object of `name -> options`. `--json` prints `to_dict()`, `--output` also writes it as UTF-8.

`route` prints the pick, every candidate with its score, the models **ranked behind** it (demoted but still fallbacks) and the ones **set aside** (out of the running) - the same two blocks `--json` carries as `demoted` and `skipped`.

One wrinkle worth knowing: a command word wins over a bare prompt, so `llm-router-lite models` lists the line-up rather than scoring the word "models". `complexity`, `route` and `models` are therefore the only three prompts the shorthand cannot express - write `llm-router-lite complexity "models"` to score one of them. Every other prompt, one word or many, goes to `complexity`.

The CLI never calls a model - it routes and reports. Output is UTF-8 whatever the console encoding is, so piping a prompt full of Japanese or emoji through another program is safe.

## License

MIT
