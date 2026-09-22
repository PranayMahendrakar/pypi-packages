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
import numpy as np
import document_quality

page = np.full((1100, 850), 248, dtype=np.uint8)   # a sheet of paper
for top in range(120, 1000, 34):                   # rows of text on it
    page[top:top + 11, 90:760] = 45

report = document_quality.assess(page, dpi=300)
print(report.summary())
```

```
<array 1100x850>: ready to OCR - score 88 of 100
Document page, 850 x 1100 px, 300 dpi (from argument), 2.8 x 3.7 in.
Text lines about 11 px tall, skew +0.00 degrees.
...
```

`report.ocr_ready` is the one-word answer, `report.score` is 0-100, and every entry
in `report.issues` carries the concrete remedy in `.fix` - "rescan at 300 dpi",
"deskew by 2.30 degrees", "increase lighting on the left edge".

## What it checks

- **Effective resolution** - dots per inch and the sheet size that implies. Left out
  of the report entirely when no dpi is known, rather than guessed from pixel count.
- **Skew** - how far the text runs off horizontal, to within about a tenth of a degree.
- **Ink-to-paper contrast** - how far the darkest ink sits from clean paper.
- **Sharpness** - how many pixels ink takes to become paper, scaled by the size of
  the text so a 600 dpi scan is not marked down for spreading the same edge wider.
- **Uneven lighting** - how much the paper level swings across the page, and where
  the dark part is, in words: "the left edge", "the top-right corner".
- **Show-through** - pale, soft marks bleeding through from the reverse side.
- **Black and white clipping** - strokes crushed to solid black, faint content
  erased by the white point.
- **Text line height in pixels** - the number that actually decides whether an OCR
  engine has enough pixels per character.
- **How much of the page looks like text** - the quantity behind the blank and
  photograph verdicts.

It also answers the two questions that come before all of those: is this sheet
blank, and is this a document page at all. A blank page is reported as blank, not
as eight failures about text it does not have. A photograph is reported as not a
document page, not as a badly scanned one.

Pure numpy and Pillow. No OpenCV, no OCR engine, no model download, no network.

## API

| Call | What you get |
| --- | --- |
| `assess(image, *, dpi=None)` | a `PageReport` for one page |
| `assess_batch(images)` | a `BatchReport` for many, unreadable files recorded not raised |
| `estimate_skew(image)` | degrees counter-clockwise off horizontal, as a float |
| `detect_orientation(image)` | `0`, `90`, `180` or `270` - the turn that sets it upright |
| `describe_thresholds()` | every boundary, its value and what it means |

`image` is a path, a `PIL.Image.Image`, or a numpy array shaped `(h, w)`,
`(h, w, 1)`, `(h, w, 3)` or `(h, w, 4)`. Greyscale and colour are both fine, and
your image is never modified.

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
| `.measures` | every measure by name, with its value, score and the boundary used |
| `.dpi`, `.dpi_source` | the resolution used and where it came from, or `None` |
| `.summary()`, `.to_dict()`, `.to_json()` | the whole report as text, as a dict, as JSON |

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
script guarding an OCR run.

## License

MIT
