"""The pipeline: find speech, describe short windows, cluster them, turn clusters into turns."""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._audio import Loaded, load_audio
from ._cluster import (
    SEPARATION_THRESHOLD,
    cluster_windows,
    project,
    standardise,
    unit_rows,
)
from ._features import Frames, analyse, cepstra, frame_vectors, window_descriptors
from ._result import Diarization, Segment
from ._vad import SILENCE_DB, Activity, detect_activity

logger = logging.getLogger(__name__)

EmbedFn = Callable[[np.ndarray, int], Any]

_METRICS = ("auto", "euclidean", "cosine")
_VAR_FLOOR = 0.01
_SIMILAR_VOICES_NOTE = (
    "one voice found. Hand-built spectral features separate clearly different voices "
    "(say a man and a woman) but can merge similar ones; if you expect more speakers, "
    "pass num_speakers or embed= a speaker-embedding model"
)


def _check_int(value: Any, name: str, minimum: int, allow_none: bool = False) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and float(value).is_integer():
            value = int(value)
        else:
            raise ValueError("{} must be a whole number, not {!r}".format(name, value))
    if int(value) < minimum:
        raise ValueError("{} must be at least {}, not {}".format(name, minimum, value))
    return int(value)


def _check_float(value: Any, name: str, minimum: float, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError("{} must be a finite number, not {!r}".format(name, value))
    number = float(value)
    if number < minimum or (strict and number == minimum):
        raise ValueError("{} must be {} {}, not {}".format(name, "more than" if strict else "at least",
                                                          minimum, value))
    return number


def _callable_name(fn: Optional[Callable[..., Any]]) -> Optional[str]:
    if fn is None:
        return None
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__


class Diarizer:
    """Configurable speaker diarization. :func:`diarize` is this with the defaults.

    Args:
        num_speakers: how many speakers to find. None estimates it from the
            clustering itself and the result says so.
        min_segment_s: shortest turn reported, in seconds. Shorter changes of
            speaker are absorbed into the neighbouring turn and bursts of sound
            shorter than this are ignored. A speaker's only turn is kept even
            if shorter, so no speaker vanishes.
        embed: optional ``embed(window_samples, sample_rate) -> vector`` used
            instead of the hand-built features, so a real speaker-embedding
            model can do the describing. Called once per window with a 1-D
            float64 copy of the samples.
        max_speakers: most speakers to consider when estimating.
        window_s: length of each analysis window in seconds.
        hop_s: step between window starts in seconds.
        min_speaker_s: least window coverage a group needs to count as a
            speaker when estimating, so a cough is not a new voice.
        separation_threshold: how far apart (in pooled spreads) two groups of
            windows must be to count as different voices. Lower finds more
            speakers and splits more single voices by mistake.
        silence_db: a recording whose loudest 100 ms is quieter than this
            (dBFS) is treated as silent.
        refine_boundaries: move each change of speaker to the best frame
            within half a window (hand-built features only).
        metric: ``"auto"``, ``"euclidean"`` or ``"cosine"``. Auto uses Euclidean
            distance on standardised hand-built features and cosine geometry
            for multi-dimensional embeddings.
        random_state: seed for the k-means summarising step used on long
            recordings; the result is identical for the same seed.
    """

    def __init__(
        self,
        *,
        num_speakers: Optional[int] = None,
        min_segment_s: float = 0.5,
        embed: Optional[EmbedFn] = None,
        max_speakers: int = 8,
        window_s: float = 1.0,
        hop_s: float = 0.25,
        min_speaker_s: float = 0.75,
        separation_threshold: float = SEPARATION_THRESHOLD,
        silence_db: float = SILENCE_DB,
        refine_boundaries: bool = True,
        metric: str = "auto",
        random_state: int = 0,
    ) -> None:
        self.num_speakers = _check_int(num_speakers, "num_speakers", 1, allow_none=True)
        self.min_segment_s = _check_float(min_segment_s, "min_segment_s", 0.0)
        if embed is not None and not callable(embed):
            raise ValueError("embed must be a callable embed(window_samples, sample_rate) -> vector")
        self.embed = embed
        self.max_speakers = int(_check_int(max_speakers, "max_speakers", 1) or 1)
        self.window_s = _check_float(window_s, "window_s", 0.1)
        self.hop_s = _check_float(hop_s, "hop_s", 0.0, strict=True)
        if self.hop_s < 0.01:
            raise ValueError("hop_s must be at least 0.01 s, not {}".format(hop_s))
        if self.hop_s > self.window_s:
            raise ValueError("hop_s ({}) must not exceed window_s ({})".format(hop_s, window_s))
        self.min_speaker_s = _check_float(min_speaker_s, "min_speaker_s", 0.0)
        self.separation_threshold = _check_float(separation_threshold, "separation_threshold", 0.0, strict=True)
        self.silence_db = _check_float(silence_db, "silence_db", -200.0)
        self.refine_boundaries = bool(refine_boundaries)
        if metric not in _METRICS:
            raise ValueError("metric must be one of {}, not {!r}".format(", ".join(_METRICS), metric))
        self.metric = metric
        self.random_state = int(_check_int(random_state, "random_state", 0) or 0)

    # ------------------------------------------------------------------ helpers
    def settings(self) -> Dict[str, Any]:
        """The parameters in effect, JSON-safe."""
        return {
            "num_speakers": self.num_speakers,
            "min_segment_s": self.min_segment_s,
            "embed": _callable_name(self.embed),
            "max_speakers": self.max_speakers,
            "window_s": self.window_s,
            "hop_s": self.hop_s,
            "min_speaker_s": self.min_speaker_s,
            "separation_threshold": self.separation_threshold,
            "silence_db": self.silence_db,
            "refine_boundaries": self.refine_boundaries,
            "metric": self.metric,
            "random_state": self.random_state,
        }

    def _empty(self, loaded: Loaded, duration: float, warnings: List[str], notes: List[str],
               speech_s: float = 0.0) -> Diarization:
        return Diarization(
            segments=[],
            estimated_speakers=self.num_speakers is None,
            duration_s=duration,
            speech_s=speech_s,
            sample_rate=loaded.sample_rate,
            method="spectral" if self.embed is None else "embed",
            requested_speakers=self.num_speakers,
            evidence_speakers=0,
            separation=None,
            separation_threshold=self.separation_threshold,
            windows=0,
            file_id=loaded.file_id,
            warnings=list(loaded.warnings) + warnings,
            notes=list(loaded.notes) + notes,
            settings=self.settings(),
        )

    @staticmethod
    def _cut_windows(regions: Sequence[Tuple[int, int]], size: int,
                     step: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Windows of ``size`` frames every ``step`` frames inside each region.

        A region shorter than one window becomes a single shorter window; the
        last window of a longer region is aligned to its end. Returns the
        ``(first_frame, end_frame)`` windows and the region index of each.
        """
        out: List[Tuple[int, int]] = []
        owner: List[int] = []
        for number, (start, end) in enumerate(regions):
            if end - start <= size:
                out.append((start, end))
                owner.append(number)
                continue
            first = start
            while first + size <= end:
                out.append((first, first + size))
                owner.append(number)
                first += step
            if out[-1][1] < end:
                out.append((end - size, end))
                owner.append(number)
        return out, owner

    def _embed_windows(self, loaded: Loaded, frames: Frames, windows: Sequence[Tuple[int, int]]) -> np.ndarray:
        assert self.embed is not None
        vectors: List[np.ndarray] = []
        size: Optional[int] = None
        for index, (first, last) in enumerate(windows):
            start, end = frames.sample_span(first, last)
            chunk = np.array(loaded.samples[start:end], dtype=np.float64, copy=True)
            raw = self.embed(chunk, loaded.sample_rate)
            try:
                vector = np.array(raw, dtype=np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "embed returned {!r} for window {}; it must return a 1-D vector of numbers".format(
                        type(raw).__name__, index)
                ) from exc
            if vector.ndim == 2 and vector.shape[0] == 1:
                vector = vector[0]
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(
                    "embed must return a 1-D vector; window {} gave shape {}".format(index, vector.shape)
                )
            if size is None:
                size = vector.size
            elif vector.size != size:
                raise ValueError(
                    "embed returned {} values for window {} but {} for earlier windows; every "
                    "vector must be the same length".format(vector.size, index, size)
                )
            if not np.isfinite(vector).all():
                raise ValueError("embed returned NaN or infinity for window {}".format(index))
            vectors.append(vector)
        return np.vstack(vectors)

    def _points(self, loaded: Loaded, frames: Frames, ceps: np.ndarray, activity: Activity,
                windows: Sequence[Tuple[int, int]], notes: List[str]) -> np.ndarray:
        if self.embed is None:
            raw = window_descriptors(ceps, activity.active, windows)
            metric = "euclidean" if self.metric == "auto" else self.metric
        else:
            raw = self._embed_windows(loaded, frames, windows)
            if self.metric == "auto":
                metric = "cosine" if raw.shape[1] >= 2 else "euclidean"
                if raw.shape[1] == 1:
                    notes.append("the embedding has one dimension, so Euclidean distance was used "
                                 "instead of cosine")
            else:
                metric = self.metric
        if metric == "cosine":
            prepared = unit_rows(raw)
        else:
            prepared = standardise(raw)
        return project(prepared)

    @staticmethod
    def _margins(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """Per window and speaker: how much closer the window is to that speaker than to the rest."""
        n = points.shape[0]
        if k <= 1:
            return np.ones((n, max(k, 1)))
        centres = np.vstack([points[labels == c].mean(axis=0) for c in range(k)])
        dist = np.sqrt(np.maximum(
            (points ** 2).sum(1)[:, None] + (centres ** 2).sum(1)[None, :] - 2 * points @ centres.T, 0.0))
        margins = np.zeros((n, k))
        for c in range(k):
            others = np.delete(dist, c, axis=1).min(axis=1)
            margins[:, c] = (others - dist[:, c]) / np.maximum(others + dist[:, c], 1e-12)
        return np.clip(margins, 0.0, 1.0)

    @staticmethod
    def _smooth(region_runs: List[List[List[int]]], min_len: int) -> List[List[List[int]]]:
        """Absorb turns shorter than ``min_len`` frames into a neighbour, never a speaker's last one."""
        counts: Dict[int, int] = {}
        for region in region_runs:
            for _, _, label in region:
                counts[label] = counts.get(label, 0) + 1
        out: List[List[List[int]]] = []
        for region in region_runs:
            current = [list(r) for r in region]
            while len(current) > 1:
                short = [i for i, r in enumerate(current) if r[1] - r[0] < min_len and counts[r[2]] > 1]
                if not short:
                    break
                i = min(short, key=lambda j: (current[j][1] - current[j][0], j))
                left = current[i - 1] if i > 0 else None
                right = current[i + 1] if i + 1 < len(current) else None
                if left is None:
                    target = right
                elif right is None:
                    target = left
                else:
                    target = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
                assert target is not None
                counts[current[i][2]] -= 1
                current[i][2] = target[2]
                counts[target[2]] += 1
                merged: List[List[int]] = []
                for run in current:
                    if merged and merged[-1][2] == run[2]:
                        merged[-1][1] = run[1]
                        counts[run[2]] -= 1
                    else:
                        merged.append(run)
                current = merged
            out.append(current)
        return out

    @staticmethod
    def _refine(region_runs: List[List[List[int]]], ceps: np.ndarray, activity: Activity,
                frame_label: np.ndarray, half: int, min_len: int) -> None:
        """Move each change of speaker to the most likely frame within ``half`` frames."""
        normed = frame_vectors(ceps)
        models: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for label in np.unique(frame_label[frame_label >= 0]):
            chosen = normed[(frame_label == label) & activity.active]
            if chosen.shape[0] < 5:
                continue
            spread = chosen.var(axis=0)
            floor = _VAR_FLOOR * float(spread.mean()) + 1e-12
            models[int(label)] = (chosen.mean(axis=0), np.maximum(spread, floor))

        def loglik(label: int, lo: int, hi: int) -> np.ndarray:
            mean, var = models[label]
            x = normed[lo:hi]
            return -0.5 * (((x - mean) ** 2) / var + np.log(var)).sum(axis=1)

        for region in region_runs:
            for index in range(len(region) - 1):
                a, b = region[index], region[index + 1]
                if a[2] not in models or b[2] not in models or a[2] == b[2]:
                    continue
                lo = max(a[1] - half, a[0] + min(min_len, a[1] - a[0]))
                hi = min(a[1] + half, b[1] - min(min_len, b[1] - b[0]))
                if hi - lo < 2:
                    continue
                gain = loglik(a[2], lo, hi) - loglik(b[2], lo, hi)
                gain[~activity.active[lo:hi]] = 0.0
                profile = np.concatenate([[0.0], np.cumsum(gain)])
                split = lo + int(np.argmax(profile))
                a[1] = split
                b[0] = split

    # --------------------------------------------------------------- the work
    def diarize(self, audio: Any, sample_rate: Optional[int] = None) -> Diarization:
        """Work out who spoke when in ``audio``.

        Args:
            audio: a ``.wav`` path, a numpy array (1-D mono, or 2-D with channels,
                which are mixed to mono), or a ``(samples, sample_rate)`` tuple.
            sample_rate: samples per second; required with a bare array.

        Returns:
            A :class:`Diarization`.
        """
        loaded = load_audio(audio, sample_rate)
        rate = loaded.sample_rate
        samples = loaded.samples
        duration = samples.shape[0] / float(rate)
        warnings: List[str] = []
        notes: List[str] = []

        if samples.shape[0] == 0:
            return self._empty(loaded, 0.0, ["the recording holds no samples"], notes)
        if duration < self.min_segment_s:
            return self._empty(loaded, duration, [
                "the recording is {:.3f} s long, shorter than min_segment_s={:g}, so no turn "
                "could be reported".format(duration, self.min_segment_s)], notes)
        frames = analyse(samples, rate)
        if frames.n_frames == 0:
            return self._empty(loaded, duration, [
                "the recording is {:.3f} s long, shorter than one 25 ms analysis frame".format(duration)
            ], notes)
        hop_s = frames.hop_s
        activity = detect_activity(frames.level_db, hop_s, silence_db=self.silence_db)
        if activity.silent:
            return self._empty(loaded, duration, [
                "no speech found: the loudest 100 ms is {:.1f} dBFS, below the {:.0f} dBFS "
                "silence level".format(activity.peak_db, self.silence_db)], notes)
        if activity.flat:
            warnings.append(
                "no clear speech: the level barely changes ({:.1f} dB between quiet floor and peak), so the whole "
                "recording above {:.0f} dBFS was treated as speech; an energy detector cannot tell "
                "steady noise from talking".format(activity.peak_db - activity.floor_db, self.silence_db)
            )
        notes.append("energy threshold {:.1f} dBFS (quiet floor {:.1f}, peak {:.1f})".format(
            activity.threshold_db, activity.floor_db, activity.peak_db))

        min_len = int(round(self.min_segment_s / hop_s))
        regions = [(s, e) for s, e in activity.regions if e - s >= max(min_len, 3)]
        dropped = [(s, e) for s, e in activity.regions if e - s < max(min_len, 3)]
        if dropped:
            notes.append("ignored {} burst(s) of sound shorter than min_segment_s ({:.2f} s in total)".format(
                len(dropped), sum(e - s for s, e in dropped) * hop_s))
        if not regions:
            longest = max((e - s for s, e in activity.regions), default=0) * hop_s
            return self._empty(loaded, duration, [
                "no speech found that lasts min_segment_s={:g} s; the longest burst of sound is "
                "{:.2f} s".format(self.min_segment_s, longest)], notes)
        speech_s = sum(e - s for s, e in regions) * hop_s

        size = max(3, int(round(self.window_s / hop_s)))
        step = max(1, int(round(self.hop_s / hop_s)))
        windows, window_region = self._cut_windows(regions, size, step)
        ceps = cepstra(frames.logmel)
        points = self._points(loaded, frames, ceps, activity, windows, notes)
        min_windows = max(2, int(math.ceil(self.min_speaker_s / self.hop_s - 1e-9)))
        clustering = cluster_windows(
            points,
            num_speakers=self.num_speakers,
            max_speakers=self.max_speakers,
            min_windows=min_windows,
            threshold=self.separation_threshold,
            random_state=self.random_state,
            region_ids=np.array(window_region, dtype=np.int64),
            max_blend_run=max(1, size // step - 1),
            spans=np.array([(frames.frame_time(a), frames.frame_time(b)) for a, b in windows]),
        )
        k = clustering.k
        margins = self._margins(clustering.points, clustering.labels, k)

        # Every frame of a region takes the label of the window whose centre is nearest.
        frame_label = np.full(frames.n_frames, -1, dtype=np.int64)
        frame_window = np.full(frames.n_frames, -1, dtype=np.int64)
        window_index = 0
        region_runs: List[List[List[int]]] = []
        for start, end in regions:
            members: List[int] = []
            while window_index < len(windows) and windows[window_index][0] < end:
                members.append(window_index)
                window_index += 1
            centres = np.array([(windows[w][0] + windows[w][1]) / 2.0 for w in members])
            positions = np.arange(start, end) + 0.5
            if len(members) > 1:
                nearest = np.searchsorted((centres[:-1] + centres[1:]) / 2.0, positions, side="right")
            else:
                nearest = np.zeros(end - start, dtype=np.int64)
            owner = np.array(members, dtype=np.int64)[nearest]
            frame_window[start:end] = owner
            frame_label[start:end] = clustering.labels[owner]
            region_runs.append([[start + s, start + e, int(frame_label[start + s])]
                                for s, e in _label_runs(frame_label[start:end])])

        region_runs = self._smooth(region_runs, max(min_len, 1))
        for region in region_runs:
            for s, e, label in region:
                frame_label[s:e] = label
        if self.embed is None and self.refine_boundaries and k >= 2:
            self._refine(region_runs, ceps, activity, frame_label, size // 2, max(min_len, 1))
            for region in region_runs:
                for s, e, label in region:
                    frame_label[s:e] = label

        raw_segments: List[Tuple[float, float, int, float]] = []
        for region in region_runs:
            for s, e, label in region:
                if e <= s:
                    continue
                live = np.flatnonzero(activity.active[s:e])
                first, last = (s + int(live[0]), s + int(live[-1]) + 1) if live.size else (s, e)
                owners = frame_window[first:last]
                confidence = float(margins[owners, label].mean()) if k > 1 else 1.0
                start_s = max(0.0, frames.frame_time(first))
                end_s = min(duration, frames.frame_time(last))
                if end_s > start_s:
                    raw_segments.append((start_s, end_s, label, confidence))
        raw_segments.sort(key=lambda item: (item[0], item[1]))

        order: Dict[int, str] = {}
        for _, _, label, _ in raw_segments:
            if label not in order:
                order[label] = "SPEAKER_{:02d}".format(len(order))
        segments = [Segment(round(a, 3), round(b, 3), order[label], round(conf, 3))
                    for a, b, label, conf in raw_segments if round(b, 3) > round(a, 3)]

        found = len(order)
        requested = self.num_speakers
        threshold = self.separation_threshold
        if requested is not None:
            if found < requested:
                warnings.append(
                    "asked for {} speakers but only {} could be separated: there are {} analysis "
                    "windows of speech ({:.1f} s)".format(requested, found, len(windows), speech_s))
            if clustering.evidence < requested:
                warnings.append(
                    "num_speakers={} was honoured, but the recording itself supports only {}: the "
                    "extra speaker(s) are most likely one voice split into parts (closest pair {} "
                    "spreads apart; distinct voices need {:.1f})".format(
                        requested, clustering.evidence, _fmt_sep(clustering.separation), threshold))
            elif clustering.evidence > requested:
                warnings.append(
                    "num_speakers={} was honoured, but the recording itself suggests {} distinct "
                    "voices, so some were merged".format(requested, clustering.evidence))
        elif found == 1 and self.embed is None:
            notes.append(_SIMILAR_VOICES_NOTE)
        if requested is None and len(windows) < 2 * min_windows:
            notes.append("only {} analysis window(s) of speech: too little to tell voices apart "
                         "reliably".format(len(windows)))

        logger.debug("diarized %.1f s: %d speaker(s), %d segment(s), %d window(s)",
                     duration, found, len(segments), len(windows))
        return Diarization(
            segments=segments,
            estimated_speakers=requested is None,
            duration_s=duration,
            speech_s=speech_s,
            sample_rate=rate,
            method="spectral" if self.embed is None else "embed",
            requested_speakers=requested,
            evidence_speakers=clustering.evidence,
            separation=clustering.separation if len(windows) >= 2 else None,
            separation_threshold=threshold,
            windows=len(windows),
            file_id=loaded.file_id,
            warnings=list(loaded.warnings) + warnings,
            notes=list(loaded.notes) + notes,
            settings=self.settings(),
        )


def _label_runs(labels: np.ndarray) -> List[Tuple[int, int]]:
    """``(start, end)`` of runs of equal values in an integer array."""
    if labels.size == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate([[0], change, [labels.size]])
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.size - 1)]


def _fmt_sep(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "infinitely"
    return "{:.1f}".format(value)


def diarize(
    audio: Any,
    *,
    sample_rate: Optional[int] = None,
    num_speakers: Optional[int] = None,
    min_segment_s: float = 0.5,
    embed: Optional[EmbedFn] = None,
) -> Diarization:
    """Work out who spoke when.

    Args:
        audio: a ``.wav`` path, a numpy array, or a ``(samples, sample_rate)`` tuple.
            Stereo and multichannel audio is mixed to mono.
        sample_rate: samples per second; required when ``audio`` is a bare array.
        num_speakers: how many speakers to find; None estimates it from the
            clustering and the result's ``estimated_speakers`` is True.
        min_segment_s: shortest turn reported, in seconds.
        embed: optional ``embed(window_samples, sample_rate) -> vector`` to use a
            real speaker-embedding model instead of the hand-built features.

    Returns:
        A :class:`Diarization` with ``segments``, ``speakers``,
        ``speaking_time``, ``summary()``, ``to_rttm()`` and ``to_dict()``.
    """
    return Diarizer(num_speakers=num_speakers, min_segment_s=min_segment_s, embed=embed).diarize(
        audio, sample_rate=sample_rate
    )


__all__ = ["Diarizer", "diarize"]
