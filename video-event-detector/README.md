# video-event-detector

Flag sudden movement, possible falls, crowding and objects left behind in a
sequence of frames from a fixed camera, with no model download and no OpenCV.
These are **motion heuristics on a running background model, not activity
recognition**: the package knows nothing about people or bags, only about
regions of the picture that move, change shape or stop moving. Every event it
reports is a hypothesis for a person to review, not a verdict.

## Install

```
pip install video-event-detector
```

It depends on numpy and Pillow only.

**It does not decode video files.** There is no video codec without a heavy
dependency, so it reads still frames: a list of numpy arrays or PIL images, or
a directory of numbered image files. Extract frames from a clip first, for
example with ffmpeg:

```
ffmpeg -i clip.mp4 -vf fps=10 frames/%05d.png
```

## Quickstart

```python
import numpy as np
from video_event_detector import detect_events

frames = [np.full((120, 160), 90, np.uint8) for _ in range(80)]   # 8 s of an empty, still view
for frame in frames[40:]:
    frame[60:84, 70:100] = 200                                   # a box is put down at 4.0 s and left
print(detect_events(frames, fps=10, dwell=2).summary())
```

```
video events: 1 found in 80 frames, 0.0-7.9 s at 10 fps
  abandoned      4.0-7.9 s        confidence 0.95  at x=70 y=60 w=30 h=24
      a new region of about 720 px (3.8% of the view) appeared at 4.0 s and has not moved since 4.1 s (dwell 2.0 s); a person standing still, a parked vehicle or a moved piece of furniture look the same, so review it
looked for, not seen: sudden_motion, fall, crowding
background: learned from the first 30 frames (0.0-2.9 s); events can start after 2.9 s
these are motion heuristics, not activity recognition: every event is a hypothesis for a person to review, not a verdict
```

For a folder of frames it is the same call: `detect_events("frames/", fps=10)`.

## What it does

The first `background_frames` frames (30 by default) are used to learn the
empty scene as their per-pixel median; nothing can be reported before that.
From then on every frame is compared with the background, and with the frame
before it. Foreground regions with changing pixels are *moving*; foreground
regions that have stopped changing are *still*. Four rules read those regions:

- **sudden_motion** - the share of the view that is moving jumps by at least
  8% of the view, and at least doubles, with most of the jump inside three
  frames. A door swinging open or a vehicle driving in qualifies as much as a
  person running in.
- **fall** - a tracked moving region, about as tall as it is wide or taller, loses
  enough of its height (to 55% or less) within a short window while its width
  holds and its lowest point does not rise, and then *stays* low for `hold`.
  A brief dip that recovers is not reported.
- **crowding** - the share of the view in motion rises past 20% and stays
  there for `hold`. It measures how much of the picture moves, not how many
  people are in it: one large vehicle close to the camera counts too.
- **abandoned** - a new region appears and then does not move for `dwell`.
  An object *taken away* leaves a hole in the background that looks similar;
  it is told apart by where the edges are (in the background, not in the
  frame) and is not reported.

`sensitivity` (0-1, default 0.5) scales every threshold together: higher
finds weaker, smaller and shorter events, and raises more false alarms.

### Event labels are hypotheses, especially falls

A "fall" here is a shape change in a moving region and nothing more. Sitting
down, bending over, lying down on purpose and a person partly hidden behind
furniture can all look the same, and a real fall that happens away from the
camera, behind something, in a crowd, or too slowly can be missed. Both
mistakes matter, so use this to point a person at the moments worth looking
at, never as the only safeguard. The same goes for the other labels: a person
standing still for the dwell time is an "abandoned" region, and a camera
aimed at trees in wind will see motion everywhere.

### What it guards against

- **Camera shake.** When most of the frame differs from the background, a
  phase correlation checks whether the whole picture simply shifted. If undoing
  the shift removes most of the difference, the frame is counted as camera
  motion and is not used for any rule, so a brief shake does not raise a crowd
  of events. If the camera settles in a new position, the background is moved
  with it and the rules restart.
- **Lighting.** A robust gain-and-offset fit matches each frame's brightness to
  the background, and the background follows slow changes where nothing is in
  front of it, so a sunrise or a dimmer does not read as motion.
- **Mixed frame sizes.** Frames that differ in size from the first are resized
  to match, and the report says how many were.
- **Short clips.** A sequence shorter than the background window still runs;
  the report says the background is still learning and suggests a smaller
  `background_frames`.
- **Your data.** Frames are never modified, and the same input always gives
  the same result.

### Limits

It expects a fixed camera and a mostly static scene. Water, foliage, screens
and flickering lights are motion to it. Small or distant figures (under about
8% of the frame height) are not checked for falls, and falls are not looked
for while two regions are merging or splitting, because people crossing change
box heights in exactly the same way. Confidence values are strength-of-evidence
scores between 0 and 1, not calibrated probabilities.

## API

```python
detect_events(frames, *, fps=None, **kw) -> EventReport
```

`frames` is a list (or any iterable) of numpy arrays or PIL images, a numpy
stack shaped `(frames, height, width[, channels])`, a list of image paths, or a
directory of numbered image files (read in natural order, so `2.png` comes
before `10.png`). With `fps`, every time is in seconds; without it, times are
frame numbers. `**kw` goes to `Detector`.

```python
Detector(*, sensitivity=0.5, background_frames=30, dwell=None, hold=None)
    .update(frame, *, timestamp=None) -> list[Event]   # events confirmed on this frame
    .events -> list[Event]
    .report() -> EventReport
    .summary() -> str
    .to_dict() -> dict
```

Feed `update` one frame at a time, in order, for live use. Give a `timestamp`
in seconds for every frame or for none. `dwell` (how long a new region must
stay still to be `abandoned`) and `hold` (how long crowding or a collapsed
shape must last) are on the same clock; left as None they are
`background_frames` frames and `max(3, background_frames // 6)` frames. An
event is returned once, when it is confirmed; while it lasts, the detector
keeps moving `end` forward on that same object.

```python
Event(kind, start, end, confidence, region, message)
```

`kind` is `sudden_motion`, `fall`, `crowding` or `abandoned`; `region` is
`(x, y, width, height)` in the pixels of the first frame; `message` says in
plain words what was measured. `.to_dict()` is JSON-safe.

`EventReport` has `.events`, `.by_kind` (every kind present, possibly empty),
`.timeline` (a plain list of `(time, kind)` pairs in time order), `.summary()`
and `.to_dict()`, plus `frames`, `analysed_frames`, `camera_motion_frames`,
`background`, `notes`, `settings` and `stats`.

## CLI

```
video-event-detector frames/ --fps 10
video-event-detector frames/ --fps 10 --dwell 30 --json
video-event-detector frames/ --fps 10 --output events.json
video-event-detector frame_001.png frame_002.png frame_003.png
```

It prints the summary, or the full result as JSON with `--json`, and `--output`
also writes the JSON to a file (UTF-8). Options: `--sensitivity`,
`--background-frames`, `--dwell`, `--hold`. Exit status is 0 when nothing was
found, 1 when at least one event was reported and 2 when the input could not be
read.

## License

MIT
