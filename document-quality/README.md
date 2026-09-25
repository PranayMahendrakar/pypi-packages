# document-quality

OCR is billed per page, and a bad scan costs exactly as much as a good one before
anybody notices it was unreadable. This reads the page first and tells you whether
it is worth sending, and what to fix if it is not.

## Install

```
pip install document-quality
```

## Quickstart

```python
import numpy as np, document_quality
from PIL import Image

page = np.full((1100, 850), 245, dtype=np.uint8)          # a sheet of paper
for top in range(100, 1000, 40):                          # rows of word-shaped ink
    for left in range(80, 740, 66):
        page[top:top + 22, left:left + 50] = 30
scan = Image.fromarray(page).rotate(2.3, fillcolor=245)   # fed in crooked

report = document_quality.assess(scan, dpi=300)
print(report.summary())
```

```
<image 850x1100>: not ready to OCR - score 91 of 100, held back by skew
Document page, 850 x 1100 px, 300 dpi (from argument), 2.8 x 3.7 in.
Text lines about 24 px tall, skew +2.29 degrees.
Note: The letters give no clear sign of which way up the page reads, so it was taken to be upright; an upside-down page cannot be ruled out.

What to do, worst first:
  [failure] The page is turned 2.29 degrees counter-clockwise of horizontal.
      fix: deskew by 2.3 degrees clockwise

Measures:
  resolution    300.000  100  Scanned at 300 dpi (argument), ...
  text_size      23.641   91  Text lines stand about 24 px tall across 23 row(s); ...
  skew            2.288   40  Text runs 2.29 degrees counter-clockwise of horizontal; ...
  ...
```

The first three lines are the answer. (The note is there because this page's
"words" are solid bars: with no ascenders or descenders to read, the report
says it could not tell upright from upside down rather than guess.) `report.ocr_ready` is the yes or no,
`report.score` is 0-100, and every entry in `report.issues` carries the concrete
remedy in `.fix` - "rescan at 300 dpi", "deskew by 2.3 degrees clockwise",
"increase lighting on the left edge", "crop the black border before OCR".

For a file on disk it is the same call: `document_quality.assess("scan.png")`.

## What it checks

- **Effective resolution** - dots per inch and the sheet size that implies. Left out
  of the report entirely when no dpi is known, rather than guessed from pixel count.
- **Skew** - how far the text runs off horizontal, typically to within a tenth of
  a degree across plus or minus 15 degrees, in either direction.
- **Ink-to-paper contrast** - how far the darkest ink sits from clean paper.
- **Sharpness** - how many pixels ink takes to become paper, scaled by the size of
  the text so a 600 dpi scan is not marked down for spreading the same edge wider.
- **Uneven lighting** - measured as a gradient: the paper level is estimated
  everywhere on the page and the report says how far it falls and towards where,
  in words: "the left edge", "the top-right corner". Everything else - including
  the text lines, their height and which way up they read - is measured against
  that local paper level, so a shadow is reported as a shadow and not as low
  contrast, show-through, a scanner border or an upside-down page.
- **Scanner borders** - only genuinely black, empty bands running in from the
  edge of the image and ending in the sharp edge of the sheet count; a deep
  shadow, which fades back to paper and still has text on it, never does. Borders
  are cropped off before anything else is measured and reported with the pixels
  to crop from each side. The black corners around a sheet scanned crooked on a
  dark lid are left out of every measure too.
- **Show-through** - pale, soft marks bleeding through from the reverse side.
- **Black and white clipping** - strokes crushed to solid black, faint content
  erased by the white point.
- **Text line height in pixels** - the number that actually decides whether an OCR
  engine has enough pixels per character.
- **How much of the page looks like text** - the quantity behind the blank and
  photograph verdicts.
- **Orientation** - a page lying on its side or upside down is caught, and a
  sideways page is measured as the upright page it will be once rotated.
  Upright is told from upside down by where each line's ink sits: Latin type
  carries far more of it in the zone above its x-height than in the zone below
  its baseline. When the letters give no clear sign - a soft scan, text in
  capitals - the report says so in a note instead of guessing.

It also answers the two questions that come before all of those: is this sheet
blank, and is this a document page at all. A blank page is reported as blank, not
as eight failures about text it does not have - dust specks and a crooked scan on
a black lid included - and a faint page that still has rows of text is not
mistaken for a blank one. A photograph, or a sheet holding only a solid shape, is
reported as not a document page, not as a badly scanned one. A document page with
no rows of text on it at all is never called ready.

Greyscale and colour scans are both read, 8-bit, 16-bit or float, each on its own
fixed scale, so a faded 16-bit scan is as faded as the 8-bit one. A 4000 x 3000
scan takes well under a second. Your image is never modified. Pure numpy and
Pillow: no OpenCV, no OCR engine, no model download, no network.

## API

| Call | What you get |
| --- | --- |
| `assess(image, *, dpi=None)` | a `PageReport` for one page |
| `assess_batch(images)` | a `BatchReport` for many, unreadable files recorded not raised |
| `estimate_skew(image)` | degrees counter-clockwise off horizontal, as a float; `0.0` with no text lines |
| `detect_orientation(image)` | `0`, `90`, `180` or `270` - the counter-clockwise turn that sets it upright |
| `describe_thresholds()` | every boundary, its value and what it means |

`image` is a path, a `PIL.Image.Image`, or a numpy array shaped `(h, w)`,
`(h, w, 1)`, `(h, w, 3)` or `(h, w, 4)`. `assess` also takes `thresholds=`,
`source=` (a name for the page) and `check_orientation=False` for pages known to
be upright.

**`PageReport`**

| Attribute | Meaning |
| --- | --- |
| `.ocr_ready` | `True` when this page is worth sending to an OCR engine |
| `.score` | overall quality for OCR, 0-100 |
| `.kind` | `"document"`, `"blank"` or `"photograph"` |
| `.skew_degrees` | counter-clockwise off horizontal; `image.rotate(-skew_degrees)` straightens it |
| `.estimated_text_height_px` | inked line height, or `None` when no rows of text were found |
| `.issues` | `list[Issue(kind, severity, message, fix)]`, worst first |
| `.fixes` | just the remedies, worst first, de-duplicated |
| `.measures` | every measure by name, with its value, score and message |
| `.dpi`, `.dpi_source` | the resolution used and where it came from, or `None` |
| `.summary()`, `.to_dict()`, `.to_json()` | the whole report as text, as a dict, as JSON |

`Issue.kind` is one of `resolution`, `dpi_tag`, `text_size`, `skew`,
`orientation`, `contrast`, `sharpness`, `lighting`, `show_through`, `clipping`,
`border`, `blank` or `not_a_document`, and `Issue.severity` is `failure`,
`warning` or `info`. Any failure keeps a page from being OCR-ready.

**`BatchReport`**: `.not_ready` (worst first), `.ready`, `.blank`, `.photographs`,
`.failures`, `.rows()`, `.issue_counts()`, `.summary()`, `.to_dict()`, `.to_json()`.

**Thresholds.** Every boundary is a field you can override, for pages that are not
300 dpi office scans:

```python
report = document_quality.assess(page, thresholds={"target_dpi": 600})
print(document_quality.describe_thresholds())
```

## CLI

```
document-quality scan.png                 # the summary for one page
document-quality scans/ --dpi 300         # every image in a folder
document-quality scans/ --json            # to_dict() as JSON
document-quality scan.png --output report.json
document-quality scans/ --only-problems   # just the pages needing work
document-quality --list-thresholds        # every boundary and what it means
```

The exit code is `0` when every page assessed is ready to OCR, `1` when any page is
not, and `2` when no page could be read at all - so it drops straight into a shell
script guarding an OCR run. `python -m document_quality` works the same way.

## License

MIT
