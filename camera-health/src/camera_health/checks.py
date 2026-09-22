"""The checks themselves: measure a frame, decide what is wrong with it.

The order matters. Exposure is judged first because a frame that is black or
blown out carries no evidence about focus, blockage, colour or noise - in that
case those checks are skipped and said to be skipped, rather than guessed at.
That is also why a completely black frame comes back as darkness and never as an
obstruction: there is no detail anywhere to compare a dead region against.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as _metrics
from ._frames import Frame, as_frame, iter_frames
from .faults import (
    COLOUR_CAST,
    CRITICAL,
    DARKNESS,
    DEFOCUS,
    DRIFT,
    FAULT_KINDS,
    FROZEN,
    INFO,
    NOISE,
    OBSTRUCTION,
    OVEREXPOSURE,
    TAMPERING,
    WARNING,
    Fault,
    sort_faults,
)
from .result import FAULT, NOT_CHECKABLE, PASS, FrameHealth, StreamReport, summarise_faults
from .thresholds import Thresholds, resolve

log = logging.getLogger(__name__)

_PENALTY_ATTR = {
    CRITICAL: "penalty_critical",
    WARNING: "penalty_warning",
    INFO: "penalty_info",
}


class _Report:
    """Scratch space while the checks run: faults, notes and skipped checks."""

    def __init__(self) -> None:
        self.faults: List[Fault] = []
        self.notes: List[str] = []
        self.checks: Dict[str, str] = {kind: PASS for kind in FAULT_KINDS}
        self.not_checkable: Dict[str, str] = {}

    def add(self, kind: str, severity: str, confidence: float, message: str) -> None:
        """Record a fault and mark its check as failed."""
        self.faults.append(Fault(kind, severity, float(np.clip(confidence, 0.0, 1.0)), message))
        self.checks[kind] = FAULT

    def skip(self, kind: str, reason: str) -> None:
        """Mark a check as impossible on this frame, with the reason."""
        self.checks[kind] = NOT_CHECKABLE
        self.not_checkable[kind] = reason

    def note(self, text: str) -> None:
        """Record a decision taken on the caller's behalf."""
        if text not in self.notes:
            self.notes.append(text)


def _confidence(value: float, start: float, full: float) -> float:
    """Ramp a measurement into a 0.4-0.98 confidence between two thresholds."""
    if full == start:
        return 0.9
    position = (value - start) / (full - start)
    return float(np.clip(0.45 + 0.53 * position, 0.4, 0.98))


def _check_exposure(report: _Report, values: Dict[str, float], limits: Thresholds) -> Tuple[bool, bool]:
    """Darkness and overexposure. Returns whether each one blocks other checks."""
    brightness = values["brightness"]
    dark_blocking = False
    bright_blocking = False

    if brightness <= limits.dark_mean or values["percentile_99"] < limits.dark_bright_pixel:
        critical = (
            brightness <= limits.dark_mean_critical
            or values["dark_fraction"] >= 0.98
            or values["percentile_99"] < limits.dark_bright_pixel
        )
        severity = CRITICAL if critical else WARNING
        dark_blocking = critical
        if values["dark_fraction"] >= 0.995 and values["dynamic_range"] < 2.0:
            message = (
                "the frame is entirely black (brightness {:.1f}); the sensor is dead, "
                "capped or getting no light at all".format(brightness)
            )
            confidence = 0.98
        else:
            message = (
                "the frame is too dark to use: brightness {:.1f}, {:.0%} of pixels are "
                "near black, brightest 1% only reaches {:.0f}".format(
                    brightness, values["dark_fraction"], values["percentile_99"]
                )
            )
            confidence = _confidence(
                limits.dark_mean - brightness, 0.0, limits.dark_mean - limits.dark_mean_critical
            )
        report.add(DARKNESS, severity, confidence, message)

    if brightness >= limits.bright_mean or values["clipped_high"] >= limits.clipped_fraction:
        critical = (
            brightness >= limits.bright_mean_critical
            or values["clipped_high"] >= limits.clipped_fraction_critical
        )
        severity = CRITICAL if critical else WARNING
        bright_blocking = critical
        report.add(
            OVEREXPOSURE,
            severity,
            _confidence(
                values["clipped_high"], limits.clipped_fraction, limits.clipped_fraction_critical
            ),
            "the frame is blown out: brightness {:.1f} and {:.0%} of pixels are clipped "
            "white; the lens is pointed at a light or the exposure is stuck".format(
                brightness, values["clipped_high"]
            ),
        )
    return dark_blocking, bright_blocking


def _check_obstruction(
    report: _Report, frame: Frame, values: Dict[str, float], limits: Thresholds
) -> None:
    """A large low-detail blob over part of a view that has detail elsewhere.

    A frame with no detail ANYWHERE is reported too, but the message says what was
    measured rather than asserting a cause: a covered lens and a camera pointed at a
    plain painted wall look identical to a single frame, and no amount of pixel
    arithmetic separates them. Comparing against a reference does not help either,
    because two photographs of the same blank wall differ only by sensor noise, so
    their signatures do not correlate. The reader is told what was seen and decides.
    """
    rows, cols = frame.shape
    grid = _metrics.tile_grid(rows, cols, limits.tile_grid)
    tiles = _metrics.tile_detail(frame.gray, values["noise"], grid) if grid else None
    if tiles is None:
        report.skip(
            OBSTRUCTION,
            "the working image is only {}x{} pixels, too small to split into tiles".format(
                cols, rows
            ),
        )
        return

    detail = tiles["detail"]
    typical = float(np.percentile(detail, 75.0))
    values["tile_detail_p75"] = typical
    values["tile_detail_raw_p75"] = float(np.percentile(tiles["raw"], 75.0))
    values["tile_noise_floor"] = float(tiles["noise_floor"])

    if (
        values["scene_contrast"] >= limits.min_content_contrast
        and values["tile_detail_raw_p75"]
        <= tiles["noise_floor"] * limits.obstruction_noise_margin
    ):
        # The view has real structure, but grain accounts for every bit of fine
        # detail in it. Subtracting that floor drives most tiles to zero, and
        # which tiles land there is then decided by the grain rather than by the
        # scene - random tiles clump together and read as a blockage. A covered
        # lens is excluded from this branch by its contrast, which is near zero,
        # so it still gets reported below.
        report.skip(
            OBSTRUCTION,
            "sensor noise of about {:.1f} grey levels accounts for all the fine detail in "
            "the frame, so a blocked region cannot be told from a noisy one".format(
                values["noise"]
            ),
        )
        return

    if typical < limits.live_tile_detail:
        # Nothing anywhere in the view has detail. Either the whole frame is a
        # flat field - a covered, fogged or painted-over lens, which is the most
        # complete obstruction there is - or the scene has broad shapes but no
        # fine detail, which is defocus and is left to that check to report.
        if values["scene_contrast"] < limits.min_content_contrast:
            values["obstruction_area"] = 1.0
            report.add(
                OBSTRUCTION,
                CRITICAL,
                0.9,
                "the whole view is one flat, featureless field (contrast {:.1f}, no detail "
                "in any part of the frame) at a normal brightness of {:.0f}; the lens is "
                "covered, fogged or painted over, or the camera is facing a blank "
                "surface".format(values["scene_contrast"], values["brightness"]),
            )
            return
        values["obstruction_area"] = 0.0
        report.note(
            "no part of the view carries fine detail (typical tile detail {:.2f}), so a "
            "blocked region cannot be told apart from a soft or blank scene".format(typical)
        )
        return

    dead = detail <= max(limits.dead_tile_floor, limits.dead_tile_ratio * typical)
    size, cells = _metrics.largest_blob(dead)
    area = size / float(detail.size)
    values["obstruction_area"] = area
    if area < limits.obstruction_area or area > limits.obstruction_area_max:
        return

    blob_brightness = float(np.mean([tiles["brightness"][row, col] for row, col in cells]))
    live = ~dead
    rest_brightness = float(tiles["brightness"][live].mean()) if live.any() else blob_brightness
    darker = blob_brightness < rest_brightness - 20.0
    severity = CRITICAL if area >= limits.obstruction_area_critical else WARNING
    where = _metrics.blob_position(cells, detail.shape[0])
    report.add(
        OBSTRUCTION,
        severity,
        _confidence(area, limits.obstruction_area, limits.obstruction_area_critical),
        "at least {:.0%} of the view has no detail at all, in one blob over {} ({}); the "
        "rest of the frame is sharp, so something is in front of the lens".format(
            area,
            where,
            "and it is darker than the rest" if darker else "same brightness as the rest",
        ),
    )


def _check_focus(report: _Report, values: Dict[str, float], limits: Thresholds) -> None:
    """Whole-frame softness, judged relative to how much contrast the scene has."""
    ratio = values["focus"]
    contrast = values["scene_contrast"]
    if contrast < limits.min_content_contrast:
        report.skip(
            DEFOCUS,
            "the view has almost no contrast (std {:.1f}), so there is nothing whose "
            "sharpness could be measured".format(contrast),
        )
        return
    if values.get("focus_noise_limited", 0.0) and values["noise"] >= limits.noise_sigma:
        # Sensor noise of this strength would by itself produce every bit of the
        # edge energy in the frame, so a sharp lens and a soft one measure the
        # same. Saying "not checkable" is the only honest answer.
        report.skip(
            DEFOCUS,
            "sensor noise of about {:.1f} grey levels accounts for all the fine detail in "
            "the frame, so a sharp lens cannot be told from a soft one here".format(
                values["noise"]
            ),
        )
        return
    if ratio >= limits.focus_ratio:
        return
    severity = CRITICAL if ratio < limits.focus_ratio_critical else WARNING
    report.add(
        DEFOCUS,
        severity,
        _confidence(limits.focus_ratio - ratio, 0.0, limits.focus_ratio - limits.focus_ratio_critical),
        "the view carries almost no fine detail (focus ratio {:.3f} against a limit of "
        "{:.3f}) while its contrast is a healthy {:.1f}; the lens is out of focus, or "
        "the scene itself is genuinely featureless".format(
            ratio, limits.focus_ratio, contrast
        ),
    )


def _check_noise(report: _Report, values: Dict[str, float], limits: Thresholds) -> None:
    """Sensor noise heavy enough to swamp the picture."""
    sigma = values["noise"]
    if sigma < limits.noise_sigma:
        return
    severity = CRITICAL if sigma >= limits.noise_sigma_critical else WARNING
    report.add(
        NOISE,
        severity,
        _confidence(sigma, limits.noise_sigma, limits.noise_sigma_critical),
        "heavy sensor noise: about {:.1f} grey levels per pixel against a limit of "
        "{:.1f}; the gain is too high, the sensor is failing, or the scene is too "
        "dark for it".format(sigma, limits.noise_sigma),
    )


def _check_colour(
    report: _Report, frame: Frame, values: Dict[str, float], limits: Thresholds
) -> None:
    """Grey-world colour balance, for colour frames that have colour in them."""
    colour = _metrics.colour_cast(frame.rgb)
    if colour is None:
        report.skip(COLOUR_CAST, "the frame is greyscale, so it cannot carry a colour cast")
        return
    values["colour_cast"] = colour["cast"]
    values["channel_red"] = colour["red"]
    values["channel_green"] = colour["green"]
    values["channel_blue"] = colour["blue"]
    if colour["spread"] < limits.mono_channel_spread:
        report.skip(
            COLOUR_CAST,
            "the three channels are identical ({:.2f} grey levels apart); this is a mono "
            "feed carried in a colour frame".format(colour["spread"]),
        )
        return
    if colour["cast"] < limits.colour_cast:
        return
    name = _metrics.CHANNEL_NAMES[int(colour["channel"])]
    direction = "too much" if colour["direction"] > 0 else "too little"
    severity = CRITICAL if colour["cast"] >= limits.colour_cast_critical else WARNING
    report.add(
        COLOUR_CAST,
        severity,
        # A genuinely one-coloured scene looks the same to a grey-world test, so
        # this never claims more than moderate confidence.
        min(0.8, _confidence(colour["cast"], limits.colour_cast, limits.colour_cast_critical)),
        "colour balance is off by {:.0%}: {} {} (R {:.0f}, G {:.0f}, B {:.0f}); an IR "
        "filter stuck in or out, a failing sensor, or a scene that really is that "
        "colour".format(
            colour["cast"], direction, name, colour["red"], colour["green"], colour["blue"]
        ),
    )


def _check_freeze(
    report: _Report,
    frame: Frame,
    previous: Optional[Frame],
    values: Dict[str, float],
    limits: Thresholds,
    repeats: int,
    freeze_frames: int,
) -> int:
    """Tell a repeated buffer from a still room. Returns the new repeat count.

    A live sensor always jitters: read noise moves nearly every pixel by a grey
    level or two between frames, even when the scene is a locked-off view of an
    empty corridor. A frozen feed hands back the same buffer, so almost no pixel
    moves at all. ``changed_fraction`` is what separates the two; ``mean_diff``
    alone cannot, because both are close to zero.
    """
    if previous is None:
        report.skip(FROZEN, "no previous frame was given, so a repeat cannot be seen")
        return 0

    if previous.shape != frame.shape:
        report.note(
            "the previous frame was {}x{} and the current one is {}x{}; the older frame "
            "was resized before comparing".format(
                previous.width, previous.height, frame.width, frame.height
            )
        )
    difference = _metrics.frame_difference(frame.gray, previous.gray)
    values["mean_diff"] = difference["mean_diff"]
    values["max_diff"] = difference["max_diff"]
    values["changed_fraction"] = difference["changed_fraction"]

    repeated = (
        difference["mean_diff"] <= limits.freeze_mean_diff
        and difference["changed_fraction"] <= limits.freeze_changed_fraction
    )
    if not repeated:
        if (
            difference["mean_diff"] <= limits.static_mean_diff
            and difference["changed_fraction"] >= limits.static_changed_fraction
        ):
            report.note(
                "the scene is static but the sensor is live: the picture moved by only "
                "{:.2f} grey levels, yet {:.0%} of pixels changed, which is sensor "
                "noise and not a repeated frame".format(
                    difference["mean_diff"], difference["changed_fraction"]
                )
            )
        return 0

    run = repeats + 1
    # ``run`` counts repeat events; run + 1 frames have now been the same image.
    same_frames = run + 1
    identical = difference["identical"] >= 1.0
    how = "bit for bit identical" if identical else (
        "identical to within {:.3f} grey levels".format(difference["mean_diff"])
    )
    values["identical_frames"] = float(same_frames)
    if same_frames >= freeze_frames:
        report.add(
            FROZEN,
            CRITICAL,
            0.97,
            "the feed has served the same image {} times in a row ({}, only {:.2%} of "
            "pixels moving); a live sensor never repeats a frame exactly, so the camera "
            "or the decoder has stopped".format(
                same_frames, how, difference["changed_fraction"]
            ),
        )
    else:
        report.add(
            FROZEN,
            WARNING,
            float(np.clip(0.40 + 0.40 * (same_frames / float(max(freeze_frames, 2))), 0.4, 0.9)),
            "this frame is {} to the previous one ({:.2%} of pixels moving); that is {} "
            "identical frames of the {} needed to call the feed frozen, and a single "
            "dropped frame on a still scene looks exactly like this".format(
                how, difference["changed_fraction"], same_frames, freeze_frames
            ),
        )
    return run


def _check_tampering(
    report: _Report,
    frame: Frame,
    reference: Optional[Frame],
    values: Dict[str, float],
    limits: Thresholds,
) -> Optional[float]:
    """Has the scene itself changed - moved, covered, or pointed somewhere else?

    The comparison runs on a brightness-normalised thumbnail, so turning the
    lights off does not read as tampering; only a change in the layout does.
    """
    if reference is None:
        report.skip(TAMPERING, "no reference frame was given, so nothing says what this view should look like")
        return None

    if reference.shape != frame.shape:
        report.note(
            "the reference frame was {}x{} and this one is {}x{}; both were reduced to a "
            "{}x{} signature before comparing".format(
                reference.width,
                reference.height,
                frame.width,
                frame.height,
                limits.signature_size,
                limits.signature_size,
            )
        )
    current, contrast = _metrics.signature(frame.gray, limits.signature_size)
    wanted, reference_contrast = _metrics.signature(reference.gray, limits.signature_size)
    if reference_contrast < limits.min_signature_contrast:
        report.skip(
            TAMPERING,
            "the reference frame is almost featureless (contrast {:.2f}), so it carries no "
            "layout to compare against".format(reference_contrast),
        )
        return None
    if contrast < limits.min_signature_contrast:
        report.note(
            "this frame is almost featureless (contrast {:.2f}); the scene comparison is "
            "reported but is weak evidence on its own".format(contrast)
        )

    correlation = _metrics.scene_correlation(current, wanted)
    values["scene_correlation"] = correlation
    if correlation >= limits.tamper_correlation:
        return correlation
    severity = CRITICAL if correlation < limits.tamper_correlation_critical else WARNING
    report.add(
        TAMPERING,
        severity,
        _confidence(
            limits.tamper_correlation - correlation,
            0.0,
            limits.tamper_correlation - limits.tamper_correlation_critical,
        ),
        "the view no longer matches the reference (layout match {:.2f} against a limit "
        "of {:.2f}); the camera has been moved, turned, or the scene has been "
        "rearranged".format(correlation, limits.tamper_correlation),
    )
    return correlation


def score_from(faults: Sequence[Fault], limits: Thresholds) -> float:
    """100 minus what every fault costs, weighted by severity and confidence."""
    lost = 0.0
    for fault in faults:
        weight = getattr(limits, _PENALTY_ATTR[fault.severity])
        lost += weight * float(fault.confidence)
    return float(round(max(0.0, 100.0 - lost), 2))


def _assess(
    frame: Frame,
    *,
    reference: Optional[Frame],
    previous: Optional[Frame],
    limits: Thresholds,
    repeats: int,
    freeze_frames: int,
    index: Optional[int],
    drift_window: int = 0,
    extra_notes: Sequence[str] = (),
) -> Tuple[FrameHealth, int, Optional[float]]:
    """Run every check on one prepared frame. Used by check(), Monitor and streams."""
    report = _Report()
    for note in extra_notes:
        report.note(note)

    rows, cols = frame.shape
    values = _metrics.measure(frame)
    if rows < 3 or cols < 3:
        report.note(
            "the frame is only {}x{} pixels; detail based checks need at least 3x3 and "
            "were skipped".format(frame.width, frame.height)
        )

    dark_blocking, bright_blocking = _check_exposure(report, values, limits)
    blocked_by = None
    if dark_blocking:
        blocked_by = "the frame is too dark for this check to mean anything"
    elif bright_blocking:
        blocked_by = "the frame is blown out, so this check cannot mean anything"

    if blocked_by is not None:
        for kind in (OBSTRUCTION, DEFOCUS, COLOUR_CAST, NOISE):
            report.skip(kind, blocked_by)
    elif rows < 3 or cols < 3:
        for kind in (OBSTRUCTION, DEFOCUS, NOISE):
            report.skip(kind, "the frame is smaller than the 3x3 window these checks need")
        _check_colour(report, frame, values, limits)
    else:
        _check_obstruction(report, frame, values, limits)
        _check_focus(report, values, limits)
        _check_noise(report, values, limits)
        _check_colour(report, frame, values, limits)

    run = _check_freeze(report, frame, previous, values, limits, repeats, freeze_frames)
    correlation = _check_tampering(report, frame, reference, values, limits)
    if correlation is None:
        report.skip(
            DRIFT,
            "no reference frame was given, so drift away from it cannot be seen"
            if reference is None
            else report.not_checkable.get(
                TAMPERING, "this view could not be compared with the reference"
            ),
        )
    elif drift_window + 1 < limits.drift_min_frames:
        # Drift is a trend, not a reading. One frame, or a handful, cannot show it.
        report.skip(
            DRIFT,
            "drift is a fall in the match to the reference across at least {} frames; "
            "{} frame(s) have been compared so far".format(
                limits.drift_min_frames, drift_window + 1
            ),
        )

    faults = sort_faults(report.faults)
    health = FrameHealth(
        ok=not any(fault.severity == CRITICAL for fault in faults),
        score=score_from(faults, limits),
        faults=faults,
        metrics=values,
        checks=report.checks,
        not_checkable=report.not_checkable,
        notes=report.notes,
        size=frame.size,
        source=frame.source,
        index=index,
    )
    return health, run, correlation


def check(
    frame: Any,
    *,
    reference: Any = None,
    previous: Any = None,
    thresholds: Any = None,
    freeze_frames: int = 2,
    index: Optional[int] = None,
) -> FrameHealth:
    """Check one camera frame and say what is wrong with it.

    Args:
        frame: a numpy array (HxW, HxWx3 or HxWx4), a PIL image, or a path to an
            image file.
        reference: a frame of how this view is supposed to look. Without one,
            tampering and drift are reported as not checkable, not as passing.
        previous: the frame immediately before this one. Without one, a frozen
            feed is reported as not checkable.
        thresholds: a Thresholds, or a dict of overrides such as
            ``{"dark_mean": 20}``. None uses the defaults.
        freeze_frames: how many identical frames in a row make a frozen feed.
            ``check`` sees only this frame and the previous one, so the default
            of 2 means one exact repeat is enough. Raise it, and a repeat is
            reported as a warning instead - but only ``Monitor`` can then reach
            the threshold, because only ``Monitor`` remembers earlier frames.
        index: position in a stream, recorded on the result.

    Returns:
        A FrameHealth carrying ``ok``, ``score``, ``faults``, the numbers behind
        them, and the checks that could not run.

    Raises:
        ValueError: a frame could not be read, or thresholds were malformed.
        FileNotFoundError: a path was given and there is no file there.
    """
    limits = resolve(thresholds)
    if int(freeze_frames) < 2:
        raise ValueError(
            "freeze_frames counts identical frames in a row, so it must be 2 or more; "
            "got {}".format(freeze_frames)
        )
    current = as_frame(frame, analysis_pixels=limits.analysis_pixels)
    before = (
        as_frame(previous, analysis_pixels=limits.analysis_pixels) if previous is not None else None
    )
    wanted = (
        as_frame(reference, analysis_pixels=limits.analysis_pixels)
        if reference is not None
        else None
    )
    health, _, _ = _assess(
        current,
        reference=wanted,
        previous=before,
        limits=limits,
        repeats=0,
        freeze_frames=int(freeze_frames),
        index=index,
    )
    return health


def check_stream(
    frames: Any,
    *,
    reference: Any = None,
    freeze_frames: int = 5,
    history: int = 30,
    thresholds: Any = None,
) -> StreamReport:
    """Check a run of frames in order and report on the run as a whole.

    Args:
        frames: an iterable of frames, or a path to a directory of image files
            (read in natural filename order, so frame2.png precedes frame10.png).
        reference: how the view is supposed to look. When this is None the first
            frame is adopted as the reference and a note says so.
        freeze_frames: identical frames in a row before the feed is called frozen.
        history: how many frames the drift comparison looks back over.
        thresholds: a Thresholds or a dict of overrides.

    Returns:
        A StreamReport with ``by_frame``, ``faults``, ``first_failure`` and
        ``summary()``.

    Raises:
        ValueError: a frame could not be read, or the directory holds no images.
    """
    from .monitor import Monitor

    limits = resolve(thresholds)
    sequence = iter_frames(frames)
    monitor = Monitor(
        reference, freeze_frames=freeze_frames, history=history, thresholds=limits
    )
    by_frame = [monitor.update(item) for item in sequence]

    if not by_frame:
        return StreamReport(
            uptime=0.0,
            notes=[
                "no frames were given, so nothing was checked; the run is reported as "
                "not ok because no frame has been proven healthy"
            ]
        )

    faults, counts, first = summarise_faults(by_frame)
    first_failure = next(
        (health.index for health in by_frame if not health.ok and health.index is not None), None
    )
    good = sum(1 for health in by_frame if health.ok)
    return StreamReport(
        by_frame=by_frame,
        faults=faults,
        first_failure=first_failure,
        fault_counts=counts,
        first_seen=first,
        frames_checked=len(by_frame),
        uptime=good / len(by_frame),
        notes=list(monitor.notes),
    )
