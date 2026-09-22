# offline-ml

Find out what machine you are actually on, and pick a model that will fit and run on it - before you download 40 GiB and find out the hard way.

## Install

```bash
pip install offline-ml
```

One dependency, `psutil`. No GPU library, no model downloads, no network calls, ever.

## Quickstart

```python
import offline_ml

print(offline_ml.detect().summary())

models = [
    {"name": "tinyllama-1.1b-q4", "size_gb": 0.7, "quality": 3, "speed": 9},
    {"name": "mistral-7b-q4", "size_gb": 4.1, "quality": 7, "speed": 6},
    {"name": "llama-70b-q4", "size_gb": 39.0, "quality": 10, "speed": 2},
]
print(offline_ml.recommend(models).summary())
```

```text
offline-ml: windows, 32 cores, 127.7 GiB RAM, NVIDIA RTX A6000 (45.0 GiB)
  cpu       : 32 logical cores at 3000 MHz
  ram       : 127.7 GiB total, 70.8 GiB available now
  disk      : 139.5 GiB free on C:\Users\you
  gpu       : NVIDIA RTX A6000, 45.0 GiB VRAM (cuda)
  python    : 3.10.11
  device    : cuda
offline-ml: run 'mistral-7b-q4' on cuda
  reason    : 'mistral-7b-q4' is the best balance of quality and speed among
              the 3 models that fit. It needs about 4.9 GiB of RAM out of
              127.7 GiB, 70.8 GiB of it free right now. Its 4.9 GiB of VRAM
              fits the 45.0 GiB on NVIDIA RTX A6000, so it will run on the
              GPU (cuda).
  needs     : 4.1 GiB on disk, 4.9 GiB of RAM, 4.9 GiB of VRAM to use a GPU
  machine   : 32 cores, 127.7 GiB RAM, NVIDIA RTX A6000 (45.0 GiB)
  asked for : prefer=balanced, headroom=0.2
  alternatives:
    tinyllama-1.1b-q4 (0.7 GiB)
    llama-70b-q4 (39.0 GiB)
```

On a laptop with no GPU at all - the normal case, and not an error - the same
two calls print this instead:

```text
offline-ml: linux, 8 cores, 15.8 GiB RAM, no GPU
  ...
  gpu       : none found - models will run on the CPU
  device    : cpu
offline-ml: run 'mistral-7b-q4' on cpu
  reason    : 'mistral-7b-q4' is the best balance of quality and speed among
              the 2 models that fit. It needs about 4.9 GiB of RAM out of
              15.8 GiB, 6.2 GiB of it free right now. No GPU was found, so
              it will run on the CPU.
  ...
  rejected:
    llama-70b-q4: needs about 46.8 GiB of RAM, this machine has only 15.8 GiB in total
```

## What it does

- **Reads the machine, not a config file.** Cores, clock, installed and free RAM,
  free disk where model caches live, the OS, the Python version, and the GPUs.
- **Finds GPUs without needing a GPU library.** It asks `nvidia-smi`, then `torch`
  but only if torch already happens to be installed, then platform APIs: the
  Windows registry, Linux `/proc` and `/sys`, the macOS machine type. `torch` is
  never imported when you import this package, and never installed by it.
- **Treats "no GPU" as normal.** Missing `nvidia-smi`, a driver that will not
  answer, a timeout, a locked-down container: all of it comes back as
  `has_gpu == False` and `device == "cpu"`. No exception, no warning, no printed
  noise. Details go to `logging.getLogger("offline_ml")` at DEBUG for anyone
  curious.
- **Rejects what will not fit, and says why in plain language.** A model bigger
  than the installed RAM is rejected with the numbers, never recommended and
  never silently truncated to something else.
- **Tells you when it will be tight.** A model that fits the installed RAM but
  not what is free right now is still recommended, with the reason saying so.
- **Falls back from GPU to CPU on its own.** Too big for the VRAM you have, or
  the card's memory could not be read, means `device == "cpu"` and a sentence
  explaining it - not a crash at load time.
- **Works everywhere.** Windows, Linux, macOS (Intel and Apple Silicon), with no
  network and no optional packages installed.

### Units

Every `*_gb` number in this package is a **binary gigabyte (GiB, 1024\*\*3 bytes)** -
inputs and outputs alike. Frequencies are in MHz. `headroom=0.2` means "treat a
4 GiB model as needing 4.8 GiB".

## API

### `detect() -> Hardware`

```python
machine = offline_ml.detect()
```

| Member | What it gives you |
|--------|-------------------|
| `.cpu_count` | logical cores |
| `.cpu_freq_mhz` | clock in MHz, or `None` where the OS does not report one |
| `.ram_total_gb` / `.ram_available_gb` | installed RAM, and what is free right now |
| `.gpus` | `list[GPU(name, vram_gb, backend)]`; empty when there is no GPU |
| `.has_gpu` | True when one of them has a usable compute backend |
| `.gpu` / `.gpu_backend` / `.vram_gb` | the best GPU, its backend, its VRAM |
| `.platform` | `"windows"`, `"linux"` or `"macos"` |
| `.python_version` | e.g. `"3.10.11"` |
| `.disk_free_gb` / `.disk_path` | free space where model caches land (your home directory); `None` when it could not be read |
| `.device` | `"cuda"`, `"mps"` or `"cpu"` |
| `.summary()` / `.short()` | the report above, and a one-line version |
| `.to_dict()` | JSON-safe dict of all of it |

`GPU.backend` is `"cuda"`, `"rocm"`, `"mps"`, or `"none"` for a display adapter
with nothing usable behind it. A display-only adapter is still listed in `.gpus`
but does not make `has_gpu` True. `GPU.vram_gb` is `None` when the device was
found but its memory could not be read; on Apple Silicon it is the unified
memory, which the GPU really can use.

On a machine with more than one card, `.gpu` and `.vram_gb` are the **largest
usable accelerator**, not the first one on the PCI bus - a small display card in
a lower slot does not shadow the compute card beside it, and `recommend()`
places models on the big one. Every card is still listed in `.gpus`, in the
order the driver reported them.

`.disk_free_gb` is `None` when free space could not be read at all - a sandbox,
an unreadable mount point, an exotic filesystem. That is missing information,
not a full disk: `recommend()` skips the disk check in that case and says so in
its reason, rather than rejecting everything.

### `recommend(models, *, task=None, prefer="balanced", headroom=0.2) -> Recommendation`

- **models** - a non-empty list of `ModelSpec` or of dicts with the same keys.
  Each needs at least `name` and `size_gb`.
- **task** - only consider models for this task. A model that declares no task is
  considered for every task.
- **prefer** - `"balanced"` (default), `"quality"`, `"speed"` or `"smallest"`.
  `quality` and `speed` come from the model's own fields when you set them; when
  you do not, size stands in - bigger is treated as better, smaller as faster.
  `"balanced"` weighs quality a little ahead of speed, favours a model that fits
  the GPU, and marks down one that would not fit alongside what is running now.
  `"quality"` means quality: it will pick a model that has to run on the CPU over
  a smaller one that fits the GPU, and the reason says so. Ties break towards the
  smaller model, then by name, so the answer never changes between runs.
- **headroom** - spare room to leave, as a fraction of the model size. A model's
  own `min_ram_gb` / `min_vram_gb` wins when it is larger.
- **hardware** - optional; pass a `Hardware` to plan for a machine you are not on.

```python
ModelSpec(name, size_gb, min_ram_gb=None, min_vram_gb=None,
          task=None, quality=None, speed=None)
```

`size_gb` must be greater than 0 and `min_ram_gb` / `min_vram_gb` cannot be
negative, since they are amounts of memory. `quality` and `speed` are ratings on
whatever numeric scale you like, higher being better, so negative values are
accepted - a centred scale such as `-5..+5` works as well as `0..10`. Only the
order matters; the scale is normalised across the models you pass in.

`Recommendation`:

| Member | What it gives you |
|--------|-------------------|
| `.model` | the `ModelSpec` to run, or `None` when nothing fits |
| `.device` | `"cuda"`, `"mps"` or `"cpu"` - where to load it |
| `.reason` | plain language, safe to show a user as-is |
| `.fits` | False when nothing fits; check it before using `.model` |
| `.alternatives` | the other models that fit, best first |
| `.rejected` | `{name: why it will not fit}` |
| `.requirements` | `{name: {"disk_gb", "ram_gb", "vram_gb"}}`, headroom included |
| `.hardware` | the machine this was decided against |
| `.summary()` / `.to_dict()` | the report above, and a JSON-safe dict |

### `fits(spec) -> bool` and `best_device() -> str`

```python
offline_ml.fits(7.5)                                  # just a size in GiB
offline_ml.fits({"name": "mistral-7b-q4", "size_gb": 4.1})
offline_ml.best_device()                              # "cuda" | "mps" | "cpu"
```

`fits()` answers for the installed RAM and free disk; `recommend()` is the one
that also tells you whether it will be tight alongside what is running now. An
AMD ROCm card reports `"cuda"`, because that is the device string PyTorch uses
for ROCm builds.

Bad input raises `ValueError` with the problem named: an empty model list, two
models sharing a name, a non-positive `size_gb`, an unknown key in a model dict,
or a `prefer` / `headroom` that is not valid.

## CLI

```bash
offline-ml                                              # what is this machine?
offline-ml --json                                       # the same, as JSON
offline-ml --model tinyllama:0.7 --model mistral-7b:4.1 # pick between them
offline-ml models.json --prefer quality --task chat
offline-ml models.json --json > pick.json
offline-ml models.json --output pick.json
```

`MODELS` is a `.json` file holding a list of model objects, or an object with a
`"models"` list. `--limit` is a count of how many alternatives and rejections to
print, so it cannot be negative. `--help` lists every option. Output is UTF-8 everywhere, so
piping a summary that contains a non-ASCII model name is safe.

## License

MIT
