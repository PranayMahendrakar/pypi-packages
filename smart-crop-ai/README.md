# smart-crop-ai

Crop to the interesting part of an image instead of the middle of it — and get a
confidence number that tells you when the crop is barely better than a guess.

## Install

```
pip install smart-crop-ai
```

## Quickstart

```python
import numpy as np, smart_crop_ai
from PIL import Image

wall = np.full((400, 800, 3), 200, dtype=np.uint8)                       # a plain wall
subject = np.random.default_rng(0).integers(0, 255, (140, 140, 3))       # something worth keeping
wall[140:280, 560:700] = subject.astype(np.uint8)                        # off to the right
result = smart_crop_ai.crop(Image.fromarray(wall), ratio=(1, 1))
print(result.summary())
```

```
smart-crop-ai: <image>
  source      800 x 400 pixels
  asked for   400 x 400 pixels
  crop        400 x 400 at (328, 0), box (328, 0, 728, 400)
  strategy    saliency (asked for: auto)
  confidence  0.69  strong - a centre crop of the same size would have missed 69% of the detail this window keeps
  keeps       50% of the source area, off centre
```

`result.image` is the cropped `PIL.Image`. A centre crop would have sliced the
subject in half; this one contains it, and says by how much it won.

## What it does

- **Finds the subject.** Every window of the size you asked for is scored
  against an energy map and the best one wins. No model, no download, no OpenCV.
- **Tells you when it did not.** `confidence` is the share of the chosen
  window's detail that a plain centre crop would have missed. At 0, cropping
  here bought you nothing, and `notes` says why.
- **Refuses to bluff.** A blank wall, a gradient, a sky — anything with a flat
  energy map falls back to the centre and records that it did, instead of
  maximising sensor noise and calling it a subject.
- **Says which of the two it was.** Confidence 0 has two very different causes,
  and `notes` never confuses them: either the image holds no detail anywhere, or
  it holds plenty but your crop is so large that every placement contains the
  same thing. The second one tells you to ask for a smaller crop.
- **Never upscales behind your back.** Ask for a crop bigger than the source and
  you get the whole image, confidence 0, and a note saying so.
- **Takes any image Pillow can open.** Greyscale, RGBA, palette, 16-bit and CMYK
  all work. EXIF orientation is applied before anything is measured, so the box
  refers to the upright image.
- **Leaves your image alone.** The input is never modified, and the same input
  always produces the same box.

## API

```python
crop(image, width=None, height=None, *, ratio=None, strategy="auto", padding=0.05) -> CropResult
```

Give `width` and `height`, or a `ratio` such as `(16, 9)`, `1.0` or `"16:9"` —
not both, which raises `ValueError`. A single `width` or `height` keeps the
source's aspect ratio. `image` is a `PIL.Image`, a path, or a numpy array.

| strategy | what it measures | when to reach for it |
| --- | --- | --- |
| `"auto"` | saliency, falling back to the centre when the map is nearly flat | the default; use this |
| `"saliency"` | gradient magnitude plus local colour variance | general photographs |
| `"entropy"` | local Shannon entropy | texture, foliage, crowds |
| `"edges"` | Sobel edge density | product shots on plain backgrounds |
| `"center"` | nothing | a baseline to compare against |

`padding` (0 to 0.45) is breathing room: that fraction of each edge is left out
of the score, so the subject is framed inside the crop rather than pressed
against its border.

```python
crop_to_file(src, dst, **kw) -> CropResult   # crop and save; **kw goes to crop()
thumbnail(image, size, **kw) -> PIL.Image    # crop to the subject, then resize to exactly size
```

`thumbnail` crops at the aspect ratio of `size` and then scales, so the subject
survives the shrink instead of landing wherever it lands.

**CropResult**

| attribute | what it holds |
| --- | --- |
| `.image` | the cropped `PIL.Image.Image` |
| `.box` | `(left, top, right, bottom)` in source pixels, always inside the image |
| `.strategy_used` | what actually chose the box: a strategy name, `"center"` on a flat image, or `"whole_image"` |
| `.confidence` | 0 to 1 — how much the chosen window beat a centre crop |
| `.confidence_label` | `"strong"`, `"moderate"`, `"weak"` or `"none"` |
| `.notes` | anything you should know, in plain language |
| `.scores` | the raw window and energy numbers behind the decision |
| `.size`, `.offset`, `.covers`, `.moved` | convenience views of the box |
| `.summary()` | the human-readable report above |
| `.to_dict()` / `.to_json()` | JSON-safe, pixels left out |
| `.save(path)` | write the crop; alpha is flattened onto white for JPEG |

## CLI

Nothing is written unless you ask, so you can look at the confidence first.

```
smart-crop-ai photo.jpg --ratio 16:9
smart-crop-ai photo.jpg --width 800 --height 600 --output hero.jpg
smart-crop-ai shots/ --ratio 1:1 --out-dir square/ --strategy edges --suffix -crop
smart-crop-ai shots/ --recursive --ratio 4:3 --quiet
smart-crop-ai photo.jpg --ratio 1:1 --json
```

`--thumb WxH` resizes after cropping, `--report PATH` writes the printed report
as UTF-8, and `--min-confidence F` exits with code 2 when any crop scores below
`F` — useful in a pipeline that would rather fail than ship a guess. Exit codes:
0 all good, 1 a usage or read error, 2 under `--min-confidence`.

## License

MIT
