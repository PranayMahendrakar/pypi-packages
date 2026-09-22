# image-redactor

Blur or mask faces, plates and regions you mark, and get back a result that tells you exactly
what was hidden. **The two built-in detectors are weak pixel heuristics, not models** - a
skin-tone blob finder for faces and a bright-rectangle finder for plates. They miss faces and
plates routinely and they fire on hands, walls, signage and paper. They exist so you can wire a
redaction pipeline up and test it, and so you can hide obvious cases. **Do not use them as a
privacy control and do not rely on them for compliance.** For anything that matters, mark the
regions yourself or pass a real model with `detector=` - this package deliberately depends on no
detection library, so the model is your choice.

## Install

```
pip install image-redactor
```

## Quickstart

```python
import numpy as np, image_redactor

photo = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
result = image_redactor.redact(photo, regions=[(40, 20, 100, 90)], method="pixelate")
print(result.summary())
print(result.count, "region(s) hidden; pixels changed:", result.changed)
assert np.array_equal(photo, np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8))
```

The last line is the promise: `photo` is untouched. The redacted copy is `result.image`, and it
is the same kind of object you passed in - numpy array in, numpy array out; `PIL.Image` in,
`PIL.Image` out; a path in gives you a `PIL.Image`.

## What it does

**Plug in a real detector.** This is the point of the package. Your detector is handed a
`PIL.Image.Image` and returns `[(left, top, right, bottom), ...]` - that is the whole contract,
so any face or plate model works and none of them becomes a dependency here.

```python
result = image_redactor.redact("street.jpg", detector=my_face_model.predict, method="blur")
result.save("street-safe.jpg")
```

- **Four methods.** `blur` smooths the region away, `pixelate` replaces each block with its
  average colour, `fill` paints a flat colour (mid grey), `blackout` paints solid black.
- **`fill`, `blackout` and `pixelate` are irreversible.** The original pixels are gone;
  `result.irreversible` says so. **`blur` is not.** A gaussian blur is a linear filter and can
  be attacked by deconvolution, so treat a blurred face as obscured, not destroyed. If the
  image is going somewhere you do not control, use `blackout`, `fill` or `pixelate`.
- **`expand` grows every box before redacting**, by default 8 percent of its own width and
  height, because a box drawn tight around a face leaves the jaw line, hairline and ears at the
  edge and those identify people.
- **Boxes are clipped, not rejected.** A box reaching past the edge is trimmed to the image and
  a note lands in `result.warnings`. A box entirely outside is dropped and counted. A box with
  `right < left` or `bottom < top` raises a `ValueError` naming both numbers - that is a bug in
  your coordinates, not something to silently paper over.
- **Nothing to redact is not an error.** With no regions and no detections you get the image
  back unchanged, `result.count == 0`, and a `summary()` that says so in plain words.
- **One bad detector does not sink the rest.** A detector that raises is caught, recorded in
  `result.detections` and `result.errors` with its exception text, and every other detector
  still runs. So does a detector that returns something that is not a list of boxes.
- **Your image is never modified in place.** Every path copies first.
- **Greyscale, RGB and RGBA all work.** The alpha channel is carried through untouched, so a
  redacted PNG keeps its transparency.
- **Overlapping boxes are tidied.** Exact duplicates and boxes fully inside another are dropped,
  so two detectors firing on the same face count once.
- No network, no model download, no OpenCV, no torch. `numpy` and `Pillow` are the only
  dependencies.

## How good are the built-in detectors, honestly

Not good. Here is what they actually do.

`detect_faces` marks pixels whose colour passes two classic skin-tone rules, groups the
connected ones, and keeps blobs that are roughly as tall as wide and solidly filled. It has no
idea what a face is. It will miss faces in shadow, in profile, behind glasses or a mask, and
faces whose skin tone falls outside rules written decades ago against a narrow sample - which is
a real and well-documented bias, not a hypothetical. It returns `[]` for greyscale images
because there is no colour to judge. It will happily box a hand, a forearm, a wooden door or a
patch of sand.

`detect_plates` marks bright pixels, groups them, and keeps blobs that are wide, rectangular and
solid. It reads no characters and verifies nothing. A white sign, a window, a headlight or a
sheet of paper looks exactly like a plate to it.

Both run only when you pass neither `regions` nor `detector`, and when they do run they add the
caveat to `result.warnings` so it shows up in the summary and the JSON. Treat a run that finds
nothing as "this heuristic found nothing", never as "this image is clean".

## API

```python
image_redactor.redact(image, *, regions=None, detector=None,
                      method="blur", strength=0.9, expand=0.08) -> RedactResult
image_redactor.redact_file(src, dst, **kw) -> RedactResult
image_redactor.detect_faces(image) -> list[box]
image_redactor.detect_plates(image) -> list[box]
```

`image` is a path (`str` or `Path`), a `PIL.Image.Image`, or a numpy array of shape `(h, w)`,
`(h, w, 1)`, `(h, w, 3)` or `(h, w, 4)`. A path that does not exist raises `FileNotFoundError`
naming it.

`regions` is `[(left, top, right, bottom), ...]`; a single box may be passed unwrapped as
`(10, 20, 90, 100)`.

`detector` is any `callable(image) -> [(left, top, right, bottom), ...]`, or a list of them.
Detectors receive a `PIL.Image.Image` in mode `L`, `RGB` or `RGBA` matching the input;
`numpy.asarray(img)` is one line away if your model wants an array.

`strength` is 0 to 1 and only affects `blur` and `pixelate` - `fill` and `blackout` always
replace every pixel, whatever you pass. `strength=0` is the *lightest* setting, not "do
nothing": you still get a one-pixel blur or the smallest block.

`expand` is 0 or more; each edge moves outward by that fraction of the box's own width (left and
right) or height (top and bottom), so `expand=0.08` turns a 100-pixel-wide box into 116.

```python
image_redactor.redact(photo, regions=[(10, 10, 60, 60)], method="blackout")
image_redactor.redact(photo, detector=[my_faces, my_plates])       # both, failures isolated
image_redactor.redact(photo, regions=boxes, expand=0.25)           # a much looser margin
image_redactor.redact_file("in.png", "out/clean.png", method="fill")
```

`Redactor(method="blur", strength=0.9, expand=0.08, detector=None, fill_color=(128, 128, 128))`
is the class underneath when you want to keep settings across many images. It has
`.redact(image, regions=None, detector=None)`, and `fill_color` is the only knob the function
does not expose.

`RedactResult`

- `.image` - the redacted image, the same kind of object you passed in
- `.boxes` - the regions actually redacted, after expanding, clipping and de-duplication
- `.count` - how many, the same as `len(result.boxes)`
- `.detector_used` - the detector names that ran, or `None` when your regions were used as given
- `.detections` - one `DetectorReport` per detector, with `.name`, `.boxes` and `.error`
- `.changed` - whether any pixel actually differs from the input
- `.irreversible` - whether the method threw the original pixels away
- `.method`, `.strength`, `.expand`, `.width`, `.height`, `.size`, `.mode`, `.source`
- `.warnings`, `.errors` - plain strings, including the heuristic caveat when it applies
- `.save(path)` - write the image; creates missing folders, returns the path
- `.summary()` - the human-readable report
- `.to_dict()` / `.to_json()` - JSON-safe, everything except the pixels

`DetectorReport` is a frozen dataclass with `.name`, `.boxes`, `.error`, `.ok` and `.to_dict()`.

## CLI

```
image-redactor SRC [-o PATH] [--region L,T,R,B ...] [--method blur]
               [--strength 0.9] [--expand 0.08] [--detect-only] [--json]
               [--quiet] [--version]
```

- `image-redactor photo.jpg -o safe.jpg --region 40,20,100,90` hides one box and prints the summary.
- `--region` repeats: pass it once per box.
- `image-redactor photo.jpg --detect-only --json` runs the built-in heuristics and prints
  `to_dict()` as JSON without writing anything.
- With no `--region` the built-in heuristics run, and the caveat is printed with the result.

Output is written as UTF-8 even when piped or redirected, so a file name with non-ASCII
characters never raises `UnicodeEncodeError`.

## License

MIT
