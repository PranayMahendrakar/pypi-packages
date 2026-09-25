# ocr-cleaner

An OCR engine reads what you hand it. Hand it a page that is two degrees off
horizontal, speckled, and darker at one edge than the other, and it will read
that. This straightens the page, takes the grain off it and binarises it
against the lighting it actually has - and tells you which of those it did,
which it skipped, and why.

## Install

```
pip install ocr-cleaner
```

## Quickstart

```python
import numpy as np, ocr_cleaner

page = np.full((1100, 850), 246, dtype=np.uint8)      # a sheet of paper
for top in range(120, 1000, 34):                      # rows of text on it
    page[top:top + 11, 90:760] = 50

result = ocr_cleaner.clean(page)
print(result.summary())
```

```
ocr-cleaner: <array 850x1100>
  page        document, 26 line-shaped bands of text, 79% of the page bare paper
  size        850 x 1100 in, 850 x 1100 out
  text        lines about 11 px tall
  skew        +0.00 degrees measured, not corrected
  result      black and white
  steps
    grayscale skipped already 8-bit greyscale
    deskew    skipped the page is already straight at +0.00 degrees, under the 0.05 degree
                     floor worth an interpolation pass
    border    skipped no scanner edge or black margin found; every side of the page is
                     already paper
    denoise   skipped the page is already clean: the paper measures 0.0 grey levels of grain
                     and 0.00% specks, under the 1.5 level floor - filtering it would only
                     soften the text
    threshold applied local mean over a 33 x 33 window (3x the 11 px text height), ink is 24
                     levels below its surroundings
    upscale   skipped no upscale_to_dpi was asked for, so the page keeps its own resolution
```

`result.image` is the cleaned page. `result.steps` is the list above, and it is
the point: a page that comes back looking much as it went in tells you which
steps stood down rather than leaving you to guess.

For a file in and a file out, one line:

```python
result = ocr_cleaner.clean_file("scan.tif", "clean.png", dpi=200, upscale_to_dpi=300)
```

## What it does

Six steps, in this order, each one reported in `result.steps` whether it ran
or not:

- **grayscale** - luminance, from any Pillow mode: colour, palette, 16-bit,
  1-bit, or transparency composited onto white. An EXIF orientation tag is
  honoured first, so a page photographed sideways is not reported as ninety
  degrees of skew.
- **deskew** - the angle of the text, to a fraction of a degree, then a rotation
  back to horizontal with the new corners filled in the page's own paper colour
  rather than white. A page already straight is left alone and says so.
- **border** - the black band a scanner leaves down the side of a page smaller
  than its glass. A band is solid black across its whole width, where even
  heavy display type leaves paper between the words, so the two are told apart
  by counting rather than guessing.
- **denoise** - a median filter, sized so its window stays well inside the width
  of a stroke: 3 x 3 for ordinary body text, 5 x 5 only for text large enough to
  survive it. Grain is measured on the paper, where paper is supposed to be
  flat, so a crisp page is not softened for nothing.
- **threshold** - `"adaptive"` compares every pixel to the mean of its own
  neighbourhood, with a window taken from the height of the text. That is the
  default because a page lit unevenly is the normal case, and one global cut has
  to choose between losing the text at the dark end and flooding the bright one.
  `"otsu"` is that single cut, for a page lit evenly end to end. `"none"` leaves
  the page in greyscale.
- **upscale** - to `upscale_to_dpi`, and only when `dpi` says what the page is
  now. OCR engines do better at 300 dpi, and enlarging blind makes a page worse
  rather than better, so with only one of the two numbers this step does nothing
  and says so.

And three pages that are not put through any of it:

- **A blank sheet** is reported blank and handed back untouched. Thresholding
  blank paper turns its grain into a field of speckles that an OCR engine reads
  as text, which is worse than doing nothing.
- **A photograph** is reported as a photograph. It has no paper level, no lines
  and no skew, and binarising one destroys it.
- **A page that is already clean** passes through with only the threshold
  applied, and the other four steps each say what they measured and why they
  stood down.

Pure numpy and Pillow. No OpenCV, no OCR engine, no model download, no network,
and the same page always gives the same result. A 4000 x 3000 scan cleans in a
couple of seconds, and your image is never modified.

This is the companion to `document-quality`, which decides whether a page is
worth OCRing. This one improves it.

## API

| Call | What you get |
| --- | --- |
| `clean(image, *, deskew=True, denoise=True, threshold="adaptive", border=True, upscale_to_dpi=None, dpi=None)` | a `CleanResult` |
| `clean_file(src, dst, **kw)` | the same, written to `dst` |
| `estimate_skew(image)` | degrees counter-clockwise off horizontal, as a float |

`image` is a path, a `PIL.Image.Image`, or a numpy array shaped `(h, w)`,
`(h, w, 1)`, `(h, w, 3)` or `(h, w, 4)`. Greyscale and colour are both fine.

**`CleanResult`**

| Attribute | Meaning |
| --- | --- |
| `.image` | the cleaned page, a `PIL.Image.Image` in mode `L` |
| `.steps` | `list[Step]`, one per stage, in order |
| `.skew_corrected_degrees` | how far the page was actually turned |
| `.estimated_text_height_px` | line height in source pixels, ascender top to descender foot, or `None` |
| `.page_kind` | `"document"`, `"blank"` or `"photograph"` |
| `.estimated_skew_degrees` | what was measured, corrected or not |
| `.applied` / `.skipped` | step names, as two lists |
| `.is_blank` / `.is_photograph` / `.binary` | the one-word answers |
| `.summary()` | the report above, as plain ASCII text |
| `.to_dict()` / `.to_json()` | the same, JSON-safe, minus the pixels |
| `.save(path)` | write `.image`, creating parent directories |

**`Step`** has `.name`, `.applied` and `.detail` - and `.detail` is never empty,
including when `.applied` is `False`.

`estimate_skew` is positive counter-clockwise, matching `PIL.Image.rotate`, so
`image.rotate(-ocr_cleaner.estimate_skew(image))` straightens a page by hand.

## CLI

```
ocr-cleaner scan.png                                  # report, writes nothing
ocr-cleaner scan.png --output clean.png
ocr-cleaner scans/ --out-dir cleaned/ --suffix -clean
ocr-cleaner scan.tif --dpi 200 --upscale-to-dpi 300 --output big.tif
ocr-cleaner scan.png --threshold otsu --no-denoise
ocr-cleaner scans/ --recursive --quiet
ocr-cleaner scan.png --json
```

Nothing is written unless you ask with `--output` or `--out-dir`.
`--require-document` exits 2 when a page turns out to be blank or not a
document, which is the flag a batch job wants. `ocr-cleaner --help` lists the
rest.

## License

MIT
