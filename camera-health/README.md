# camera-health

Tell whether a camera is actually seeing anything - obstructed, defocused, dark,
blown out, frozen or tampered with - from its own frames, with no model download
and no OpenCV.

## Install

```
pip install camera-health
```

## Quickstart

```python
import numpy as np
from camera_health import check

y, x = np.indices((240, 320))
view = (120 + 60 * np.sin(x / 4) * np.cos(y / 5)).astype(np.uint8)   # a sharp, well-lit view
blocked = view.copy()
blocked[30:210, 40:280] = 90                                          # something over the lens

print(check(blocked, previous=view).summary())
```

```
camera health: NOT OK (score 55.9 / 100)  [frame 320x240]
faults (1):
  [critical] obstruction  confidence 0.98  at least 47% of the view has no detail at all, in one blob over the centre of the view (and it is darker than the rest); the rest of the frame is sharp, so something is in front of the lens
not checkable (3):
  tampering    no reference frame was given, so nothing says what this view should look like
  colour_cast  the frame is greyscale, so it cannot carry a colour cast
  drift        no reference frame was given, so drift away from it cannot be seen
measured: brightness 102.9, contrast 24.6, focus 0.270, noise 0.00, blown pixels 0.0%, blocked area 47.3%, motion vs previous 19.736, pixels moving 55.8%
```

A frame is a numpy array, a PIL image, or a path to an image file. Nothing is
ever written back to it.

## What it checks

- **obstruction** - a large low-detail blob over part of a view that is sharp
  elsewhere: a hand, a sticker, a spider web, a lens cap. A view that is flat
  *everywhere*, at a normal brightness, is reported as a covered or painted-over
  lens instead.
- **defocus** - fine detail measured against the contrast the scene has, so
  dimming the lights or watching a plain wall does not read as a soft lens.
- **darkness** and **overexposure** - mean level, near-black share, clipped
  highlights and the usable dynamic range.
- **frozen** - the feed handing back the same image over and over.
- **tampering** - the scene no longer matches a reference view, compared on a
  brightness-normalised thumbnail so turning the lights off is not tampering.
- **drift** - the match to the reference falling away across a run of frames,
  which is a camera being nudged rather than moved.
- **colour cast** - grey-world channel balance, for an IR filter stuck in or out.
- **noise** - a robust per-pixel sensor noise estimate.

### A frozen feed is not a static scene

This is the distinction the package exists for. A camera watching an empty
corridor produces frames that look identical to the eye, and calling that
"frozen" is the failure mode that makes camera monitoring useless.

They are not actually identical. A live sensor has read noise: between two
exposures of a motionless room nearly every pixel moves by a grey level or two.
A frozen feed is a repeated buffer, so *no* pixel moves.

So the mean change is the wrong thing to threshold - it is near zero for both.
`camera-health` thresholds the **share of pixels that moved at all** alongside
it:

| | mean change | pixels moving | verdict |
|---|---|---|---|
| empty corridor, live sensor | ~1.2 grey levels | ~75% | healthy, noted as static |
| repeated buffer | 0.00 | 0% | frozen |

A single repeat is only ever a warning, because one dropped frame on a still
scene looks exactly like the start of a freeze. It takes `freeze_frames`
repeats in a row - the default is 5 - before the feed is called frozen, and only
`Monitor` and `check_stream` can count that far, because only they remember
earlier frames.

## API

### `check(frame, *, reference=None, previous=None, thresholds=None, freeze_frames=2, index=None) -> FrameHealth`

Check one frame. `reference` is how the view is meant to look; `previous` is the
frame just before it. Leave either out and the checks that need it report
**not checkable** rather than passing - a single frame is never silently
declared free of a fault it could not test for.

`FrameHealth` carries:

| attribute | meaning |
|---|---|
| `.ok` | `True` when nothing critical is wrong; warnings lower the score but leave it `True` |
| `.score` | 0-100, every fault costing points by severity and confidence |
| `.faults` | `list[Fault(kind, severity, confidence, message)]`, worst first |
| `.metrics` | every number measured, fault or not |
| `.checks` | each kind mapped to `"pass"`, `"fault"` or `"not_checkable"` |
| `.not_checkable` | the checks that could not run, and why |
| `.notes` | decisions taken for you, such as resizing a mismatched reference |
| `.summary()` / `.to_dict()` | human text / JSON-safe mapping |
| `.has(kind)`, `.fault(kind)`, `.was_checked(kind)` | ask about one kind |

### `Monitor(reference=None, *, freeze_frames=5, history=30, thresholds=None)`

Feed frames in as they arrive.

- `.update(frame) -> FrameHealth` - keeps the rolling history, so frozen and
  drift become decidable
- `.state` - `"healthy"`, `"degraded"` or `"failed"`
- `.alerts` - `list[Alert(when_index, kind, message)]`, one per fault as it
  starts, plus a `"recovered"` alert when the camera comes back
- `.uptime` - share of frames that were ok
- `.summary()` / `.to_dict()`

With no `reference`, the first frame is adopted as one and a note records it.

```python
from camera_health import Monitor

monitor = Monitor(freeze_frames=5)
for frame in frames:
    health = monitor.update(frame)
print(monitor.state, monitor.uptime, monitor.summary())
```

### `check_stream(frames, *, reference=None, freeze_frames=5, history=30, thresholds=None) -> StreamReport`

Run a whole sequence - a list of frames, or a path to a directory of images read
in natural filename order. `StreamReport` has `.by_frame`, `.faults` (one entry
per kind, with how many frames it hit), `.first_failure`, `.uptime`,
`.summary()` and `.to_dict()`.

### `Thresholds`

Every limit is a field on `Thresholds`, and any call takes either one or a dict
of overrides: `check(frame, thresholds={"dark_mean": 20})`.

## CLI

```
camera-health frame.png                          # one frame
camera-health frames/ --reference day1.png       # a directory, in natural order
camera-health a.png b.png --json --output r.json
camera-health --show-thresholds
```

`--freeze-frames N`, `--history N`, `--set name=value` (repeatable), `--json`,
`--output PATH`, `--quiet`. Exit codes: `0` every frame usable, `1` a critical
fault was found, `2` the input could not be read.

## Limits

Two things this cannot do, stated plainly because a monitor that overstates itself gets
switched off:

- **A covered lens and a plain wall look identical to a single frame.** Both are a flat,
  featureless field. The fault is still raised, because a flat view really can be a covered
  lens, but the message describes what was measured rather than asserting a cause. Comparing
  against a reference does not settle it either: two photographs of the same blank wall differ
  only by sensor noise, so their signatures do not correlate.
- **A frozen feed and a genuinely still scene are close but separable.** A camera watching an
  empty corridor legitimately produces near-identical frames, so a feed counts as frozen only
  when consecutive frames are identical or differ only at the sensor-noise floor, and only
  after `freeze_frames` of them in a row. That distinction is tested in both directions.

## License

MIT
