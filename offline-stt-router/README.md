# offline-stt-router

Find which speech-to-text engines are installed on this machine, and pick the right engine, model and device for its hardware and your language. The pick is a **heuristic**: engines and models are ranked with rough published figures for memory, speed and accuracy, scaled to this machine's RAM, CPU cores and GPU. They are estimates for ranking, not a benchmark of your computer.

## Install

```bash
pip install offline-stt-router
```

The only dependency is `psutil`. No speech engine comes with it and no model is ever downloaded: it finds what you already have (faster-whisper, openai-whisper, whisper.cpp, Vosk, or your own) and tells you what to add when nothing fits.

## Quickstart

```python
import offline_stt_router as stt

for engine in stt.available():
    print(engine.describe())
print(stt.choose(language="en", prefer="balanced").summary())
```

On a fresh laptop with nothing installed, which is the normal case, you get a clear answer rather than an exception:

```
faster-whisper - not installed
openai-whisper - not installed
whisper.cpp - not installed
vosk - not installed
No speech-to-text engine is ready on this machine for English.
Why: None of the engines this package knows (faster-whisper, openai-whisper, whisper.cpp, vosk) is installed.
To get one (this package never downloads anything itself):
  - pip install faster-whisper, then fetch the large-v3-turbo model: python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo')" (large-v3-turbo on this 8-core CPU, about 3.5x faster than real time, about 1.9 GB of RAM)
  - build whisper.cpp (https://github.com/ggml-org/whisper.cpp) and put whisper-cli on PATH or set WHISPER_CPP_BIN (macOS: brew install whisper-cpp), then download ggml-large-v3-turbo.bin from https://huggingface.co/ggerganov/whisper.cpp into ~\.cache\whisper.cpp (or any folder named in WHISPER_CPP_MODELS) (large-v3-turbo on this 8-core CPU, about 2.5x faster than real time, about 1.7 GB of RAM)
Checked: faster-whisper (not installed); openai-whisper (not installed); whisper.cpp (not installed); vosk (not installed).
Machine: Windows (AMD64), 8 cores / 16 threads, 15.7 GB RAM (9.8 GB available), no usable GPU found
```

On a workstation with faster-whisper and three models in the Hugging Face cache:

```
Use faster-whisper with the large-v3 model on the NVIDIA GPU (float16) for English.
Why: The large-v3 model is the most accurate option on disk that should still run at least 2x faster than real time on the NVIDIA RTX A6000 (about 14x faster than real time). It needs about 1.2 GB of RAM; 90.7 GB is available right now (of 127.7 GB).
Estimated: about 14x faster than real time, about 1.2 GB of RAM and 4.5 GB of GPU memory.
Model: ~\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\snapshots\edaa852ec7e145841d8ffdb056a99866b5f0a478
Alternatives:
  1. faster-whisper medium (NVIDIA GPU, float16) - about 27x faster than real time, about 1.2 GB RAM, less accurate
  2. faster-whisper tiny (NVIDIA GPU, float16) - about 308x faster than real time, about 0.3 GB RAM, less accurate
Checked: faster-whisper (ready); openai-whisper (not installed); whisper.cpp (not installed); vosk (installed, no model).
Machine: Windows (AMD64), 24 cores / 32 threads, 127.7 GB RAM (90.7 GB available), GPU: NVIDIA RTX A6000 (45.0 GB VRAM, 40.5 GB free)
```

The speed figures are the router's estimates, not measurements. On that machine the same call transcribed a synthesized sentence exactly, in 0.6 seconds once the model was loaded.

Once something is installed, transcribing is one more line, from local models only:

```python
text = stt.transcribe("meeting.wav", language="en")
```

## What it does

- **Probes without importing.** faster-whisper, openai-whisper and Vosk are found with `importlib.util.find_spec` and their package metadata; whisper.cpp is found as a binary on PATH (`whisper-cli`, `whisper-cpp`, or `WHISPER_CPP_BIN`) and is never run while probing. Torch, whisper and vosk are not imported at import time, or by `available()` or `choose()`. They are imported inside a `try` only when you actually transcribe.
- **Reports broken installs instead of crashing.** An engine whose dependency is missing (faster-whisper without `ctranslate2`, openai-whisper without `torch`), whose native library is gone (Vosk's `libvosk`), whose metadata outlived its files, or whose custom probe raises comes back as `status="failing"` with the reason and the repair command in `error`.
- **Finds models on disk, and checks them.** It looks in the Hugging Face cache (faster-whisper), `~/.cache/whisper` (openai-whisper), the usual whisper.cpp folders, the Vosk folders, and any folder in `OFFLINE_STT_MODELS`. whisper.cpp models are identified from their file header (true size, quantization, English-only or not), and a file shorter than its header describes is reported as an interrupted download. For `.pt` checkpoints, a file with no zip directory at its end is reported the same way. A Hugging Face snapshot with no `model.bin` is reported as unfinished, and a Transformers checkpoint that faster-whisper cannot load without conversion is named as such.
- **Weighs the machine.** It reads RAM (total and available) and cores with psutil. NVIDIA GPUs and their free memory come from the driver's own NVML library through `ctypes`, with no CUDA toolkit or PyTorch needed. Apple Silicon is detected as a Metal GPU sharing system RAM. AMD and Intel GPUs are not detected.
- **Chooses by a stated rule.** `prefer="fast"` picks the quickest option that is good enough for the language. `"balanced"` (the default) picks the most accurate option expected to run at least 2x faster than real time here, or the fastest if nothing does. `"accurate"` picks the most accurate option that fits in memory, whatever the speed.
- **Rejects what will not fit, and says why.** A model that needs more RAM than is available (or more than `max_ram_gb`) is set aside with the numbers, for example `needs about 3.6 GB of RAM on the CPU; only 2.1 GB is available right now`. A model too big for the GPU falls back to the CPU with a note.
- **Knows about languages.** English-only models (`.en`, distil-whisper) are never offered for other languages, and are preferred for English at small sizes. Vosk models are matched by the language in their folder name. For languages Whisper saw less of in training, models below `base` or `small` are not counted as good enough. This grouping follows the per-language training hours in the Whisper paper and is itself a heuristic. `language=None` asks for auto-detection and keeps only multilingual Whisper models.
- **Explains itself.** Every `Choice` has a plain-language `reason`, ranked `alternatives`, the `rejected` options with reasons, and `missing`: what to install or fetch for a better result, including "you already have these models on disk" when an engine is missing but its downloads are not.
- **Transcribes locally, never downloads.** `transcribe()` gives each engine a local path (a model folder, a `.pt` file or a ggml file), never a model name, so no engine can start a download. WAV files are decoded here, so Vosk, whisper.cpp and openai-whisper read any WAV without ffmpeg. If the first choice fails (CUDA libraries missing, a corrupt model), the next alternative is tried and every attempt is recorded.
- **Takes your own engines.** `register(name, probe, transcribe)` adds any engine; it is weighed with the same rules.

### The figures behind the ranking

| model | RAM on CPU, faster-whisper int8 | GPU memory, float16 | relative compute (small = 1) | accuracy rank |
| --- | --- | --- | --- | --- |
| tiny | 0.35 GB | 0.5 GB | 0.25 | 0.35 |
| base | 0.45 GB | 0.7 GB | 0.45 | 0.45 |
| small | 0.9 GB | 1.2 GB | 1.0 | 0.62 |
| medium | 2.0 GB | 2.6 GB | 2.8 | 0.74 |
| large-v3-turbo | 1.9 GB | 2.5 GB | 2.2 | 0.86 |
| large-v3 | 3.6 GB | 4.5 GB | 5.5 | 0.88 |

The accuracy rank is a relative 0-1 ordering loosely following published word error rates, not a WER. The speed estimate is the compute figure times a per-engine base rate (small on an 8-thread CPU: faster-whisper 0.13 s per second of audio, whisper.cpp 0.18, openai-whisper 0.55; on a GPU about ten times less), scaled by `(8 / cores) ** 0.7`. openai-whisper and whisper.cpp have their own RAM tables, and whisper.cpp's RAM is estimated from the model file size, which also covers quantized files. Vosk models are ranked below Whisper small for accuracy, run at about 0.08 s (small) to 0.35 s (big) per second of audio on one core, and are sized from their folder.

## API

```python
import offline_stt_router as stt

stt.available() -> list[Engine]
stt.choose(*, language="en", prefer="balanced", max_ram_gb=None) -> Choice
stt.register(name, probe, transcribe=None) -> None
stt.unregister(name) -> bool
stt.transcribe(audio, **kw) -> str
stt.machine() -> Machine
stt.Router(*, machine=None, model_dirs=(), search_paths=None, builtins=True)
```

**`available()`** returns one `Engine` per known engine, installed or not. **`Engine`** has `.name`, `.installed`, `.version`, `.languages` (what its models on disk cover), `.needs_gpu`, `.can_use_gpu`, `.models_on_disk` (a list of `LocalModel` with `.name`, `.family`, `.path`, `.size_gb`, `.languages`, `.quantization`), `.notes`, `.status` (`"ready"`, `"no-model"`, `"failing"`, `"not-installed"`), `.error`, `.location`, `.install_hint`, `.describe()`, `.summary()` and `.to_dict()`.

**`choose(*, language="en", prefer="balanced", max_ram_gb=None)`** never raises because nothing is installed. `language` takes a code (`"hi"`), a tag (`"pt-BR"`), an English name (`"Hindi"`), a native name (`"हिन्दी"`), or `None` for auto-detection. An unrecognised name raises `ValueError`, and so does a `prefer` other than `"fast"`, `"balanced"` or `"accurate"`. `max_ram_gb` caps the RAM a model may use; it never raises the limit above what is free.

**`Choice`** has `.engine`, `.model`, `.device` (`"cpu"`, `"cuda"`, `"metal"`, or `None` when nothing fits), `.compute_type`, `.model_path`, `.reason`, `.alternatives` (a list of `Option`), `.rejected` (a list of `Option`, each with a `.reason`), `.missing` (what to install or fetch, as sentences), `.notes`, `.estimated_ram_gb`, `.estimated_vram_gb`, `.realtime_factor` (estimated seconds of compute per second of audio), `.speed`, `.ok`, `.machine`, `.engines`, `.summary()` and `.to_dict()`. `bool(choice)` is `choice.ok`.

**`register(name, probe, transcribe=None)`**: `probe()` returns `True`/`False`, an `Engine`, or a dict of Engine fields. Useful keys are `version`, `languages`, `models_on_disk`, `quality` (0-1), `realtime_factor`, `ram_gb`, `vram_gb` and `needs_gpu`. A probe that raises `ModuleNotFoundError` means "not installed"; any other exception means installed-but-failing. `transcribe(audio, *, language, model, model_path, device, sample_rate, **kw)` returns the text, and it receives only the keywords it declares. Registering a built-in name replaces that built-in.

**`transcribe(audio, **kw)`**: `audio` is a file path, the bytes of an audio file, or float samples (a list or numpy array) together with `sample_rate=`. Keywords: `language`, `prefer` and `max_ram_gb` as for `choose()`; `engine=` or `model=` (a name or a path) to insist on one; `fallback=False` to stop at the first failure. Anything else goes to the engine, for example `beam_size=5` for faster-whisper. It raises `TranscriptionFailed` (a `RuntimeError` with `.choice` and `.attempts`) when nothing installed can do the job, and its message says what to install. `Router.transcribe_detailed()` returns a `Transcript` with `.text`, `.engine`, `.model`, `.device`, `.seconds` and `.attempts`. The last model used stays loaded for the next call; `Router.unload()` frees it.

**`Router`** is the same API with the knobs exposed. `machine=Machine(ram_total_gb=4, cpu_physical=2)` asks "what would it pick on a small laptop?", `model_dirs=[...]` adds model folders, and `builtins=False` considers only your registered engines.

Environment variables: `OFFLINE_STT_MODELS` (extra model folders, separated like PATH), `WHISPER_CPP_BIN`, `WHISPER_CPP_MODELS`, `VOSK_MODEL_PATH` and the usual `HF_HOME` / `HF_HUB_CACHE`. Set `OFFLINE_STT_ROUTER_NO_GPU=1` to skip GPU detection.

## CLI

```
offline-stt-router                                  # what to use here, for English
offline-stt-router choose --language hi --prefer accurate --max-ram-gb 4
offline-stt-router choose --language auto --json --output choice.json
offline-stt-router engines                          # every engine, its models and notes
offline-stt-router machine --json                   # RAM, cores and GPUs as seen
offline-stt-router transcribe talk.wav --language en
offline-stt-router talk.wav                         # the same, shorter
offline-stt-router transcribe talk.wav --engine vosk --models-dir D:\models
```

`--json` prints `to_dict()`. On an error it prints `{"ok": false, "error": ...}` and exits with status 1. `--output` also writes the result as UTF-8. Output is UTF-8 whatever the console's code page, so non-Latin transcripts, paths and language names survive pipes and CI logs. A first argument that is neither a command nor an option is read as an audio file to transcribe.

## License

MIT
