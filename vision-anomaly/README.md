# vision-anomaly

Spot unusual images against a set of normal ones, without a single labelled
defect - show it images you are happy with, then ask it about anything else.

No training, no model download, no OpenCV, no torch. Just numpy and Pillow.

## Install

```
pip install vision-anomaly
```

## Quickstart

```python
import numpy as np
from vision_anomaly import detect_anomalies

rng = np.random.default_rng(0)
good = [rng.normal(120, 4, (96, 96, 3)).clip(0, 255).astype(np.uint8) for _ in range(8)]
odd = good[0].copy()
odd[20:60, 20:60] = [240, 30, 30]          # a red blotch that does not belong

print(detect_anomalies(good, [good[0], odd]).summary())
```

```
2 images checked against 8 known-good, threshold 3.00: 1 anomalous
the known-good images themselves score 0.00 typical, 0.02 at worst

<array>: ANOMALOUS (score 64.69, threshold 3.00, confidence 1.00)
compared against 8 known-good images, which themselves score 0.00 typical and 0.02 at worst
what moved:
  colour (64.7 sigmas): the mix of colours in the frame; furthest is colour.chroma.r1c1, 403.0 sigmas above normal at row 1, column 1
  contrast (9.6 sigmas): local contrast within each part of the frame; furthest is contrast.r2c1, 22.6 sigmas above normal at row 2, column 1
  orientation (2.2 sigmas): the directions the edges run in; furthest is orientation.r1c2[0-45deg], 10.3 sigmas above normal at row 1, column 2
where it moved most:
  row 1, column 1      126.5 sigmas  (analysis-grid box x=64 y=64 64x64)
  row 1, column 2      63.2 sigmas  (analysis-grid box x=128 y=64 64x64)
  row 2, column 1      63.1 sigmas  (analysis-grid box x=64 y=128 64x64)
```

An image is a path, a PIL image or a numpy array. Nothing you pass in is ever
written to.

Any integer or float dtype works. The 0.0-1.0 scale an array is put on is read
from the values it holds, never from the container they arrived in, so `uint8`,
`uint16` and `int32` carrying the same pixels give one score. A profile
remembers the scale it was fitted on and reads later images the same way; if a
known-good set disagrees with itself about its own bit depth, `fit` says so
rather than quietly building a profile that flags nothing.

## What it does

`fit` reduces your known-good images to a **profile**: the median and a robust
spread of a few hundred hand-built numbers. `score` turns any other image into
the same numbers and asks how far outside that spread it lands, in robust
sigmas. Nothing is learned, so there is nothing to train and nothing to
download.

The numbers come in six families:

| family | what it measures |
|---|---|
| `colour` | per-channel histograms of red, green and blue over the frame, plus the mean chroma of each grid cell |
| `edges` | per cell, the mean gradient magnitude and the share of pixels sitting on an edge |
| `orientation` | per cell, a gradient-orientation histogram, plus a finer one over the whole frame |
| `brightness` | per cell, mean luminance |
| `contrast` | per cell, the spread of luminance |
| `texture` | per cell, the mean absolute Laplacian - fine detail, which separates a rough surface from a smooth one at the same brightness |

216 numbers in all, at the default 4x4 grid. `describe_features()` prints the
table with the counts.

Four decisions are worth knowing about, because they are what makes the scores
usable rather than merely computable:

- **Median and MAD, not mean and standard deviation.** A folder of good images
  nearly always has one in it that should not be there, and a single bad image
  moves a mean and inflates a standard deviation enough to hide every real
  anomaly behind it.
- **Histograms are interpolated, not hard-binned.** A pixel a hair below a bin
  edge would otherwise land wholly in one bin, and one grey level brighter in
  the next, so two photographs of the same thing would differ by a whole bin.
- **Every feature has a floor on its spread** - 5% of its own level, and an
  absolute floor below that. A dozen images never show all the variation a
  camera, a lamp and a hand placing a part will produce.
- **Only the distance past 3 sigmas counts.** Summing up the wobble that every
  clean image has would give them all a floor of about one sigma and leave no
  room to say "this one is identical". It is why a copy of a fitted image scores
  near zero - usually `0.0`, and never far from it. `.fit_worst` is the exact
  figure for your own set: it is the highest any of your known-good images
  scores against the profile they built.

The score is the worst of the six family scores, and each family's score is the
root mean square of how far its features lie outside the band. Averaging the six
instead would divide a real colour fault by the five families that are fine.

## API

```python
from vision_anomaly import Detector, detect_anomalies
```

**`Detector(sensitivity=3.0, *, grid=4, analysis_size=256, max_regions=3, max_reasons=3)`**

| method | does |
|---|---|
| `.fit(images)` | learn normal; returns the detector, so `Detector().fit(good)` is one expression |
| `.score(image)` | how far the image sits from normal, in robust sigmas. `0.0` means it landed entirely inside normal |
| `.predict(image)` | an `AnomalyResult` |
| `.predict_batch(images)` | a `BatchReport` |
| `.save(path)` / `Detector.load(path)` | the fitted profile as JSON. Not a pickle: loading it executes nothing |
| `.summary()` | what this detector was fitted on |

`images` can be a list of paths, PIL images or arrays, a stacked `NxHxWx3`
array, or the path of a directory.

**`detect_anomalies(good_images, test_images, sensitivity=3.0, **kw) -> BatchReport`**
does the fit and the check in one call.

**`AnomalyResult`**

| attribute | is |
|---|---|
| `.anomalous` | `True` when `score >= sensitivity` |
| `.score` | the distance, in robust sigmas |
| `.confidence` | 0.0 to 1.0, how decisively the score sits on its side of the threshold. `0.5` means it landed on the line - and that is confidence in the *verdict*, so a clean image is a confident `normal` |
| `.regions` | the grid cells that departed most, worst first, as `Box` objects - so you can see **where** |
| `.reasons` | plain sentences naming the families that moved and the single feature that moved furthest |
| `.group_scores` | every family's score, including the quiet ones |
| `.fit_typical`, `.fit_worst` | what the known-good images score against their own profile. The honest yardstick: nothing below `fit_worst` was stranger than an image you called normal |
| `.notes`, `.warnings` | how the image was read, and anything that makes the verdict less trustworthy |
| `.summary()`, `.to_dict()` | human text, and JSON-safe data |

**`Box`** carries `.row`, `.col`, `.x`, `.y`, `.width`, `.height` and `.score`.
The coordinates are pixels on the square analysis grid, so use
`box.scaled(image.width, image.height)` to draw it on your own picture.

**`BatchReport`** holds `.results`, `.anomalous` (worst first), `.normal`,
`.n_anomalous`, `.scores`, `.worst`, `.summary(limit=5)`, `.to_dict()` and
`.to_json()`. It iterates and indexes, so `for result in report` works.

### Edge cases, and what they do

| you do | it does |
|---|---|
| fit on fewer than 5 images | warns that the profile is weak, and still works |
| fit on 1 image | `ValueError` - one image says nothing about how much variation is normal |
| mix image sizes | resamples everything onto one square grid, and says so in `.notes` |
| pass greyscale, RGB or RGBA | all fine; grey is carried in three channels, alpha is composited over white |
| point at a corrupt file | `ValueError` naming the path and what Pillow objected to |
| `score()` before `fit()` | `RuntimeError` telling you to fit or load first |
| score a copy of a fitted image | near zero - `0.0` for most, never above `.fit_worst` |
| pass the same pixels as `uint8`, `uint16` or `int32` | one score; the scale is read from the values, not the container |
| mix 8-bit and 16-bit images in one fit set | warns that the set disagrees about its own depth, and re-reads it all on one scale |
| run it twice | identical numbers; there is no randomness anywhere |

## CLI

```
vision-anomaly --good known_good/ today/            # check a folder against another
vision-anomaly --good known_good/ --save-profile belt.json    # fit once, keep the profile
vision-anomaly --profile belt.json today/ --json    # reuse it, as JSON
vision-anomaly --good known_good/ today/ -s 5 -q    # less sensitive, one line per image
vision-anomaly --good known_good/ today/ -r -o report.txt
vision-anomaly --describe-features
```

Exit codes: 0 nothing flagged, 1 at least one image flagged, 2 the arguments or
inputs could not be used. Output is UTF-8 whatever the console claims, so
non-ASCII file names do not crash it.

## Limits

Read this before trusting a number.

- **It compares hand-built statistics, not learned features.** It catches gross
  departures - the wrong colour, the wrong texture, a missing part, a different
  scene, a lens change - and it will **miss a subtle defect that a trained model
  would see**. A hairline crack across a part occupies a few hundred pixels out
  of sixty thousand, moves no histogram and barely moves one grid cell's texture.
  If that is the defect you care about, this package is the wrong tool and no
  amount of turning `sensitivity` down will fix it - you will only buy false
  alarms.
- **The profile only knows what you showed it.** If every good image you fitted
  has the part in the left half and a real good image has it in the right half,
  that is an anomaly as far as the profile is concerned, and correctly so. Fit
  on a set that spans the variation you are willing to accept.
- **Regions are grid cells, not outlines.** At the default 4x4 they say "the
  middle right of the picture". Raise `grid` for a finer answer, at the cost of
  tolerating less camera movement.
- **`sensitivity` is a threshold on a robust distance, not a probability.** 3.0
  is a starting point. Check `.fit_worst` first: if your own known-good images
  score 2.5 against their own profile, a threshold of 3.0 is already tight.
- **Images are stretched onto a square.** Layout is compared as proportions, so
  a 16:9 frame and a 4:3 frame of the same scene line up cell for cell. If your
  set is all one shape, nothing is lost.

## License

MIT
