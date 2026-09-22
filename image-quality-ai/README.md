# image-quality-ai

Find the photos that are too blurry, too dark, too bright, too grainy, too flat or too badly
framed to be worth feeding to a model, before they cost you a training run or an inference bill.

## Install

```
pip install image-quality-ai
```

## Quickstart

```python
import numpy as np, image_quality_ai

photo = np.random.default_rng(0).integers(0, 60, (240, 320, 3), dtype=np.uint8)  # a dark, grainy snap
report = image_quality_ai.assess(photo)
print(report.summary())
print("usable:", report.usable, "| worst problem:", report.issues[0])
```

`assess` also takes a file path or a `PIL.Image`, so the real version of that is
`image_quality_ai.assess("photo.jpg")`.

## What it checks

Five measures, each reported with the raw number it computed **and** a 0-100 score, so every
verdict can be traced back to a quantity you can check yourself:

- **sharpness** - variance of the Laplacian. Every frame is put on the same fixed analysis grid
  first, area-averaged coming down and pixel-replicated going up, so a 12-megapixel photo, a 512px
  copy of it and a 224px thumbnail of it all report the same number instead of the small ones
  reading several times sharper. This is what `is_blurry` compares against.
- **exposure** - mean luminance, plus the share of pixels crushed to black and blown out to white.
  Clipping is penalised separately, so a photo that averages out fine but has lost its highlights
  is still called out.
- **contrast** - luminance standard deviation and the 5th-to-95th percentile spread, which together
  separate a genuinely flat frame from one with a few dark corners.
- **noise** - the residual left after a 3x3 median filter, measured at the image's own resolution
  on tiles sampled across the frame. Grain lives in the pixels the camera produced, so this is the
  one measure that is deliberately *not* taken from a shrunken copy.
- **framing** - how far the largest high-detail region sits from the centre, and how much of the
  frame it fills. When no area of detail stands out there is no subject to place, and the measure
  steps aside rather than inventing a verdict.

The overall score is a weighted average of the measures that applied. Sharpness and exposure are
pass/fail gates: a frame that fails either carries no usable detail whatever its other numbers look
like, so the overall score is held below the usable mark and `usable` and `score` can never
disagree.

Handled without special-casing by the caller: 1x1 images, fully black and fully white frames,
greyscale, RGB, RGBA (composited over white), 16-bit and palette images, and EXIF orientation,
which is applied before anything is measured. A corrupt or truncated file raises a `ValueError`
naming the path instead of a Pillow traceback; a missing path raises `FileNotFoundError`. Scores
are deterministic, and a 4000x3000 photo is assessed in well under a second.

Pure numpy and Pillow. No OpenCV, no model downloads, nothing touches the network.

## API

**`assess(image, *, thresholds=None) -> QualityReport`**
Measure one image. `image` is a path, a `PIL.Image.Image`, or a numpy array shaped HxW, HxWx3 or
HxWx4. `thresholds` is `None`, a `Thresholds`, or a dict of overrides.

**`assess_batch(images, *, workers=1, thresholds=None) -> BatchReport`**
Measure many. Unlike `assess`, a batch does not stop for one bad file: unreadable images land in
`report.failures` and the rest are still assessed. `workers` adds threads, which helps when the
images come from disk.

**`is_blurry(image, threshold=None) -> bool`**
The one-question version. `threshold` is a Laplacian variance, defaulting to
`Thresholds.sharpness_blurry` (100.0).

**`ImageAssessor(thresholds=None)`**
The same three calls with the thresholds fixed once: `.assess()`, `.assess_batch()`, `.is_blurry()`.

**`QualityReport`** - `.score` (0-100), `.grade` ("A" to "F"), `.usable` (bool),
`.metrics` (dict of name -> `Metric`), `.issues` (plain language, worst first), `.failed_gates`,
`.image` (how the file was read), `.summary()`, `.to_dict()`, `.to_json()`, `.scores()`,
`.explain("noise")`.

**`Metric`** - `.value` (the raw number), `.score` (0-100), `.ok`, `.message` (one sentence naming
the numbers and the boundary used), `.details`, `.applies`.

**`BatchReport`** - `.results`, `.failures`, `.usable`, `.rejected`, `.worst(n=5)`, `.mean_score()`,
`.grade_counts()`, `.rows()`, `.to_frame()` (needs pandas, which this package does not install),
`.summary()`, `.to_dict()`, `.to_json()`. Iterates and indexes over `.results`.

### Thresholds

Every boundary is a documented constant on `Thresholds`, not a literal buried in a function, so a
disagreement with the defaults is a keyword argument rather than a fork:

```python
report = image_quality_ai.assess(photo, thresholds={"sharpness_blurry": 40.0})
print(image_quality_ai.describe_thresholds())   # the whole table, with defaults
```

The defaults suit ordinary photographs going into a vision model. Scanned documents, microscopy,
thermal frames and astrophotography all have a different notion of "normal", so expect to move
them.

## CLI

```
image-quality-ai photo.jpg                        # the full report for one image
image-quality-ai shots/ --recursive --workers 4   # a whole tree, worst listed first
image-quality-ai shots/ --quiet                   # one line per image
image-quality-ai shots/ --json > report.json      # to_dict() as JSON, UTF-8
image-quality-ai shots/ --fail-under 60           # exit code 2 if anything scores below 60
image-quality-ai photo.jpg --threshold sharpness_blurry=40
image-quality-ai --list-thresholds
```

Exit codes: 0 all good, 1 usage or read error, 2 something scored below `--fail-under`.

## Limits

Worth knowing before you trust a number:

- **Sharpness is measured at a fixed analysis size**, so scores are comparable between
  images, but resizing the same photo can still move it across a threshold. Treat the
  grade as a signal, not a constant of the file.
- **A still frame cannot always tell motion blur from a one-directional subject.**
  Crisp vertical stripes, a page of text and a fence against plain sky all leave one
  axis nearly featureless, exactly as a sideways smear does. Those are flagged in the
  sharpness message rather than failed outright, so read the message before acting.
- **Thresholds are defaults, not physics.** Every one is a documented constant you can
  override with `thresholds=`, and the right values depend on your camera and subject.

## License

MIT
