"""Input shapes, the public API surface, the result objects and bad input."""

from __future__ import annotations

import dataclasses
import inspect
import io
import json
import os
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import synth
import video_event_detector as ved
from video_event_detector import EVENT_KINDS, Detector, Event, EventReport, detect_events

README = Path(__file__).resolve().parents[1] / "README.md"


def kinds(report):
    return [event.kind for event in report.events]


def short_left():
    """55 frames: 8 to learn the scene, then a bag is put down at 2.5 s and left."""
    return synth.abandoned_sequence(count=80)[25:]


SHORT = {"fps": 10, "dwell": 2, "background_frames": 8}


def short_burst():
    """45 frames: 10 to learn the scene, then a large block bursts in at frame 30."""
    return synth.sudden_sequence(count=65)[20:]


def save_frames(folder: Path, frames, name="frame_{}.png"):
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / name.format(index))
    return folder


# -- the quickstart and the API surface -------------------------------------------
def test_readme_quickstart_runs_and_finds_the_box():
    text = README.read_text(encoding="utf-8")
    match = re.search(r"##\s*Quickstart\s*\n+```(?:python)?\n(.*?)```", text, re.S)
    assert match, "README has no Quickstart python block"
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(match.group(1), "quickstart", "exec"), {"__name__": "__quickstart__"})
    printed = out.getvalue()
    assert "abandoned" in printed and "4.0-7.9 s" in printed
    assert "x=70 y=60 w=30 h=24" in printed
    # The output shown in the README is what the code prints.
    assert printed.strip().splitlines()[1] in text


def test_public_api_matches_the_documented_shape():
    assert ved.__version__ == "0.1.0"
    params = inspect.signature(Detector).parameters
    assert params["sensitivity"].default == 0.5
    assert params["background_frames"].default == 30
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    update = inspect.signature(Detector.update).parameters
    assert update["timestamp"].kind is inspect.Parameter.KEYWORD_ONLY
    detect = inspect.signature(detect_events).parameters
    assert detect["fps"].kind is inspect.Parameter.KEYWORD_ONLY and detect["fps"].default is None
    assert [f.name for f in dataclasses.fields(Event)] == [
        "kind", "start", "end", "confidence", "region", "message"
    ]
    assert EVENT_KINDS == ("sudden_motion", "fall", "crowding", "abandoned")
    for name in ("events", "by_kind", "timeline", "summary", "to_dict"):
        assert hasattr(EventReport, name) or name in {f.name for f in dataclasses.fields(EventReport)}


def test_report_objects_explain_themselves():
    report = detect_events(synth.abandoned_sequence(), fps=10, dwell=2)
    assert set(report.by_kind) == set(EVENT_KINDS)
    assert report.by_kind["abandoned"] == report.events
    assert report.timeline == [(report.events[0].start, "abandoned")]
    data = report.to_dict()
    text = json.dumps(data, allow_nan=False)
    assert json.loads(text)["events"][0]["kind"] == "abandoned"
    assert data["by_kind"] == {"sudden_motion": 0, "fall": 0, "crowding": 0, "abandoned": 1}
    assert data["time_unit"] == "seconds" and data["fps"] == 10.0
    assert data["settings"]["dwell"] == {"clock": 2.0}
    summary = report.summary()
    assert summary.isascii()
    assert "hypothesis" in summary
    event = report.events[0]
    assert event.to_dict()["region"] == {"x": 67, "y": 78, "width": 10, "height": 10}
    assert event.duration == pytest.approx(event.end - event.start)
    assert "abandoned" in event.describe("seconds")


def test_detector_summary_and_to_dict_while_streaming():
    detector = Detector()
    assert "no frames" in detector.summary()
    for frame in synth.fall_sequence()[0]:
        detector.update(frame)
    assert detector.report().time_unit == "frames"
    assert [e.kind for e in detector.events] == ["fall"]
    assert "frames" in detector.summary()
    json.dumps(detector.to_dict(), allow_nan=False)


# -- input shapes --------------------------------------------------------------------
def test_directory_of_numbered_frames_is_read_in_natural_order(tmp_path):
    frames = short_left()
    folder = save_frames(tmp_path / "frames", frames)
    (folder / "notes.txt").write_text("not a frame", encoding="utf-8")
    from_disk = detect_events(str(folder), **SHORT)
    in_memory = detect_events(frames, **SHORT)
    assert kinds(from_disk) == kinds(in_memory) == ["abandoned"]
    assert from_disk.events[0].start == in_memory.events[0].start
    assert from_disk.source == str(folder)
    assert detect_events(folder, **SHORT).frames == len(frames)  # a Path works too


def test_unicode_paths(tmp_path):
    frames = short_burst()
    folder = save_frames(tmp_path / "кадры_カメラ_é", frames, name="снимок_{}.png")
    report = detect_events(str(folder), background_frames=10)
    assert kinds(report) == ["sudden_motion"]
    assert "кадры_カメラ_é" in report.summary()
    assert "кадры_カメラ_é" in json.dumps(report.to_dict(), ensure_ascii=False)


def test_pil_images_rgb_uint16_float_and_stacks_agree():
    frames = short_left()
    expected = kinds(detect_events(frames, **SHORT))
    assert expected == ["abandoned"]
    variants = {
        "pil": [Image.fromarray(f) for f in frames],
        "rgb": [np.repeat(f[:, :, None], 3, axis=2) for f in frames],
        "rgba": [np.dstack([f, f, f, np.full_like(f, 255)]) for f in frames],
        "uint16": [f.astype(np.uint16) * 257 for f in frames],
        "float01": [f.astype(np.float64) / 255.0 for f in frames],
        "stack": np.stack(frames),
        "generator": (f for f in frames),
    }
    for name, value in variants.items():
        report = detect_events(value, **SHORT)
        assert kinds(report) == expected, name
    assert any("16-bit" in n for n in detect_events(variants["uint16"][:3]).notes)


def test_mixed_dtypes_in_one_sequence():
    frames = synth.noisy(synth.texture(seed=6), 60, seed=6)
    mixed = [f.astype(np.float32) / 255.0 if i % 3 == 0 else f for i, f in enumerate(frames)]
    mixed[0] = frames[0]  # start on uint8; the float scale is set by the first float frame
    assert detect_events(mixed).events == []


def test_list_of_image_paths(tmp_path):
    frames = short_burst()
    folder = save_frames(tmp_path / "f", frames)
    paths = [str(folder / "frame_{}.png".format(i)) for i in range(len(frames))]
    assert kinds(detect_events(paths, background_frames=10)) == ["sudden_motion"]


def test_empty_and_single_frame_sequences():
    empty = detect_events([])
    assert empty.frames == 0 and empty.events == []
    assert empty.background["state"] == "no frames"
    assert "no frames" in empty.summary()
    json.dumps(empty.to_dict(), allow_nan=False)
    single = detect_events([synth.noisy(synth.flat(), 1)[0]])
    assert single.frames == 1 and single.background["state"] == "learning"
    assert "STILL LEARNING" in single.summary()


def test_nan_frames_are_skipped_or_patched_not_read_as_black():
    frames = [f.astype(np.float64) for f in synth.noisy(synth.texture(seed=7), 70, seed=7)]
    frames[45] = np.full_like(frames[45], np.nan)         # a dead frame
    frames[50] = frames[50].copy()
    frames[50][10:40, 10:60] = np.nan                      # a partly corrupt frame
    frames[52] = frames[52].copy()
    frames[52][0:5, :] = np.inf
    report = detect_events(frames)
    assert report.events == []
    assert any("skipped" in note for note in report.notes)
    assert any("treated as unchanged" in note for note in report.notes)


def test_timestamps_give_seconds_and_are_validated():
    detector = Detector(dwell=2.0)
    for index, frame in enumerate(synth.abandoned_sequence()):
        detector.update(frame, timestamp=100.0 + index * 0.1)
    report = detector.report()
    assert report.time_unit == "seconds"
    assert kinds(report) == ["abandoned"]
    assert 104.6 <= report.events[0].start <= 105.05

    detector = Detector()
    frame = synth.noisy(synth.flat(), 1)[0]
    detector.update(frame, timestamp=1.0)
    with pytest.raises(ValueError, match="backwards"):
        detector.update(frame, timestamp=0.5)
    with pytest.raises(ValueError, match="every frame or for none"):
        detector.update(frame)
    assert detector.report().frames == 1  # a refused frame changes nothing

    counting = Detector()
    counting.update(frame)
    with pytest.raises(ValueError, match="every frame or for none"):
        counting.update(frame, timestamp=3.0)


# -- bad input ---------------------------------------------------------------------------
def test_video_files_are_refused_with_the_ffmpeg_hint(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(64))
    with pytest.raises(ValueError, match="ffmpeg"):
        detect_events(str(clip))
    folder = tmp_path / "only_video"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"not frames")
    with pytest.raises(ValueError, match="ffmpeg"):
        detect_events(str(folder))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensitivity": 1.5},
        {"sensitivity": "high"},
        {"background_frames": 0},
        {"background_frames": 2.5},
        {"dwell": -1},
        {"hold": float("nan")},
    ],
)
def test_bad_settings_are_clear_value_errors(kwargs):
    with pytest.raises(ValueError):
        Detector(**kwargs)


def test_other_bad_input():
    frame = synth.noisy(synth.flat(), 1)[0]
    with pytest.raises(ValueError, match="fps"):
        detect_events([frame], fps=0)
    with pytest.raises(FileNotFoundError):
        detect_events(os.path.join("no", "such", "folder"))
    with pytest.raises(ValueError, match="at least 8x8"):
        detect_events([np.zeros((4, 4), np.uint8)])
    with pytest.raises(ValueError, match="shape"):
        detect_events([np.zeros((5, 20, 20, 2, 2), np.uint8)])
    with pytest.raises(ValueError, match="None"):
        detect_events([None])
    with pytest.raises(ValueError, match="single PIL image"):
        detect_events(Image.fromarray(frame))
    with pytest.raises(ValueError, match="list of numpy arrays"):
        detect_events(42)
    with pytest.raises(TypeError):
        detect_events([frame], no_such_option=1)
