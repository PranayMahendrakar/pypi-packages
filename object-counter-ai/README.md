# object-counter-ai

Count things in an image, or across the frames of a video, and get a result that
explains how it reached the number. For real objects, plug in a detector you already
have (any function that returns boxes); this package does the counting, clipping,
region filtering and line-crossing tracking around it, and never depends on a model
itself. With no detector, a built-in **classical** counter runs instead. It is a
**heuristic**: it thresholds the picture against its background, finds connected
regions and filters them by area. It counts **blobs, not objects**. That makes it
genuinely useful for high-contrast countable things (parts on a conveyor, cells on
a slide, bolts on a tray) and useless for people in a street.

## Install

```
pip install object-counter-ai
```

## Quickstart

```python
import numpy as np, object_counter_ai

tray = np.full((120, 200), 230, dtype=np.uint8)            # a light tray
for x in (30, 80, 130, 170):                                 # four dark parts
    tray[40:70, x - 12:x + 12] = 40
print(object_counter_ai.count(tray).summary())
```

```
object-counter-ai: 4 blobs counted (classical blob counter)
  image       200 x 120 pixels, grey
  by label    blob 4
  sizes       720 px each
  confidence  1.00 (a heuristic score, not a probability)
  background  light, things darker; even
  threshold   96 levels from the background (noise 0.0, floor 10)
  caveat      it counts high-contrast blobs, not objects; things touching
              with no narrow neck between them count as one
```

Bring your own detector for anything that is not a high-contrast blob:

```python
def my_detector(image):                  # wrap any model: YOLO, torchvision, a web API
    return [((12, 30, 48, 72), "bolt", 0.91), ((60, 28, 101, 75), "nut", 0.84)]

result = object_counter_ai.count(tray, detector=my_detector)
print(result.count, result.by_label, result.confidence)   # 2 {'bolt': 1, 'nut': 1} 0.875
```

Count things crossing a line in a stream, each one once:

```python
counter = object_counter_ai.Counter().line((0, 60), (200, 60))
for y in range(10, 120, 10):                        # one part sliding down the belt
    belt = np.full((120, 200), 230, dtype=np.uint8)
    belt[max(0, y - 8):y + 8, 90:110] = 40
    counter.update(belt)
print(counter.totals)                               # {'line 1': 1}
```

## What it does

- **Counts your detector's boxes.** Any callable `detector(image)` works. It can
  return plain boxes, `(box, label, score)` tuples, dicts, a torchvision-style
  `{"boxes", "labels", "scores"}` dict or an `(n, 4/5/6)` numpy array. Boxes
  outside the image are clipped to it, and boxes left with no area are dropped. The
  result says how many of each.
- **Or counts blobs without any model.** The classical counter estimates the
  background, which may be unevenly lit (a smooth fitted surface absorbs gradients and
  vignetting). It thresholds each pixel's distance from that background, using
  Otsu's method with a floor set by the measured noise, labels connected regions and
  applies `min_area`/`max_area`. It works on grey, colour-only and transparent
  images.
- **Splits touching blobs where a narrow neck joins them.** Two discs pressed
  together are cut apart when the neck is narrower than roughly 70% of the smaller
  one's width. The limit is real and tested: things that overlap heavily, or touch
  along a whole edge, have no neck and count as one. Bolts, washers and brackets
  are not cut up.
- **Counts inside a region.** Pass a box, a polygon or a boolean mask. A detection
  counts when its box centre is inside. The classical counter measures the background
  inside the region, so a dark table outside the tray does not confuse it.
- **Counts line crossings once per object.** `Counter` follows objects frame to
  frame by nearest-centroid matching. An object that lingers on the line, wobbles
  across it or comes back is still counted once.
- **Survives a failing detector.** If your detector raises or returns something
  unreadable, the error is caught and reported on that frame's result (`ok` is
  False). The frame stays in the history and tracking carries on.
- **Explains itself.** `confidence`, `notes` and `details` say what lowered
  confidence: noise, edge-cut blobs, split blobs, very mixed sizes, a crowded picture.
  For the classical counter, confidence is a heuristic score, not a probability.
- **Takes any image.** numpy arrays (grey, grey+alpha, RGB, RGBA, 8/16-bit, float,
  bool, channels-first), `PIL.Image` in any mode, or a file path. An empty image
  counts 0 instead of raising.
- **Leaves your image alone and gives the same answer every time.** The input is
  never modified, even by a detector that writes into what it is given, and nothing
  is random.

### When the classical counter is the wrong tool

It knows nothing about what an object is. Do not use it for:

- people, vehicles, animals or anything on a cluttered background: use a detector;
- things touching along a broad edge, or stacked on top of each other: they merge;
- pictures packed so densely that the objects also cover most of the border: the
  background is taken to be the most common colour or, when that fails, whatever
  fills the border, and a full frame leaves neither;
- scenes with very faint and very strong objects together: one global threshold
  can miss the faint ones;
- lighting more complicated than a smooth gradient, such as hard shadows.

## API

```python
count(image, *, detector=None, min_area=None, max_area=None, region=None) -> CountResult
```

- `image`: numpy array, `PIL.Image` or file path.
- `detector`: a callable `detector(image) -> boxes`. It receives a copy of what you
  passed (a file path is opened for it as an upright `PIL.Image`). Boxes are
  `(left, top, right, bottom)` in pixels, or `(box, label, score)`.
- `min_area`, `max_area`: in pixels. Blob area for the classical counter (the
  default `min_area` ignores specks: 0.002% of the area, at least 9 px), box area
  for a detector.
- `region`: a box `(left, top, right, bottom)`, a polygon `[(x, y), ...]` or a
  boolean mask the size of the image.

`CountResult`

| field | meaning |
| --- | --- |
| `count` | how many were counted |
| `boxes` | `(left, top, right, bottom)` per item, right and bottom exclusive |
| `labels`, `scores`, `areas`, `centroids` | per item, aligned with `boxes` (`"blob"` and no score for the classical counter) |
| `by_label` | `{label: count}`, largest first |
| `confidence` | 0 to 1; mean detector score (None without scores), or the classical heuristic |
| `method` | `"detector"` or `"classical"` |
| `ok`, `error` | False and a message when the detector failed; the count is then not real |
| `notes`, `details` | why the number is what it is: threshold, background, specks, splits, clipping |
| `frame`, `track_ids`, `crossings` | filled in by `Counter.update` |
| `summary()`, `to_dict()` | readable text; JSON-safe dict |

```python
Counter(detector=None, *, min_area=None, max_area=None, region=None,
        max_distance=None, max_missed=3, line_margin=2.0)
```

- `.update(frame) -> CountResult`: count one frame and follow its objects.
- `.line(a, b, name=None) -> Counter`: add a counting line from point `a` to point
  `b`. Crossing from the left of a->b to its right (as seen on screen) is
  `forward`, so for a line drawn left to right, moving down the image is forward.
- `.totals`: `{line name: objects crossed}`. `.lines` adds the ends, `forward`,
  `backward`, `by_label` and ignored re-crossings.
- `.history`: every frame's `CountResult`, failed ones included. `.frames`,
  `.failed_frames`, `.tracks_started` and `.peak` are also available.
- `.summary()`, `.to_dict()`, `.reset()`.
- `max_distance`: the furthest (in pixels) an object may move between frames and still
  be the same object. The default is 1.5 times the median box diagonal. Set it when
  things move faster than that.
- `max_missed`: how many frames a track survives unseen.
- `line_margin`: how far past the line a centre must be before it counts as on the
  other side.

Tracking is deliberately simple. Objects that swap places between frames, or move
further than `max_distance`, can be miscounted.

## CLI

```
object-counter-ai tray.png                          # print the summary
object-counter-ai tray.png --min-area 50 --expect 12   # exit 2 if the count is not 12
object-counter-ai slide.tif --region 100,100,900,700 --json
object-counter-ai shots/ --quiet                    # one "count<TAB>path" line per image
object-counter-ai frames/ --line 0,240,640,240      # images as frames, count crossings
object-counter-ai street.jpg --detector my_models:detect_people
object-counter-ai tray.png --output counts.json     # write the JSON result (UTF-8)
```

`--region` takes a box `l,t,r,b` or a polygon `"x,y x,y x,y ..."`. `--detector
MODULE:FUNCTION` imports your function; the current folder is importable. Exit codes:
0 for success, 1 for a usage or read error or a detector failure, and 2 when a count
differs from `--expect`.

## License

MIT
