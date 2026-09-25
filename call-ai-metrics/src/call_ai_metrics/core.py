"""The entry points: :func:`analyze` for the one-line case, :class:`CallAnalyzer` for control."""

from __future__ import annotations

import logging
import numbers
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ._audio import CallAudio, load_channels, load_wav, split_channels
from ._detect import Detection, detect
from ._metrics import Measurement, measure
from ._report import (
    BleedCheck,
    CallReport,
    Crosstalk,
    Interruption,
    Segment,
    SpeakerStats,
    Turn,
    clock_text,
    duration_text,
    method_detail,
)
from ._segments import (
    Timeline,
    build_timeline,
    rows_from_annotation,
    rows_from_csv,
    rows_from_frame,
    rows_from_json,
)

logger = logging.getLogger(__name__)

_SEGMENT_FILES = (".csv", ".tsv", ".txt")
_JSON_FILES = (".json",)
_TABLE_FILES = (".parquet", ".pq", ".feather", ".xlsx", ".xls")

HEAVY_OVERLAP_SHARE = 0.15
HIGH_SILENCE_SHARE = 0.30
SLOW_REPLY_S = 2.0
MIN_REPLIES_TO_FLAG = 3
INTERRUPTIONS_TO_FLAG = 3
FAST_WPM = 190.0
SLOW_WPM = 100.0
MIN_PACE_SPEECH_S = 15.0


def _default_names(count: int) -> List[str]:
    letters = "ABCDEFGHIJKLMNOP"
    return [letters[index] if index < len(letters) else "P{}".format(index + 1) for index in range(count)]


def _channel_names(count: int, speakers: Any) -> List[str]:
    names = _default_names(count)
    if speakers is None:
        return names
    if isinstance(speakers, (str, bytes)):
        raise ValueError(
            "speakers={!r} is a single string; pass one name per channel, e.g. "
            "['agent', 'customer']".format(speakers)
        )
    if isinstance(speakers, Mapping):
        renamed = list(names)
        for key, value in speakers.items():
            if isinstance(key, numbers.Integral) and not isinstance(key, bool) and 0 <= int(key) < count:
                renamed[int(key)] = str(value)
            elif str(key) in names:
                renamed[names.index(str(key))] = str(value)
            else:
                raise ValueError(
                    "speakers has the key {!r}, but the recording has channels 0 to {} "
                    "(named {})".format(key, count - 1, ", ".join(names))
                )
        names = renamed
    else:
        given = [str(name) for name in speakers]
        if len(given) != count:
            raise ValueError(
                "speakers names {} parties but the recording has {} channels; give "
                "one name per channel, in channel order".format(len(given), count)
            )
        names = given
    if len(set(names)) != len(names):
        raise ValueError("speakers gives two channels the same name: {}".format(names))
    return names


def _is_frame(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "iloc") and hasattr(value, "to_dict")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _looks_like_channel(value: Any, rate_given: bool) -> bool:
    """True for a 1-D run of samples, as opposed to one diarized segment.

    A segment has three or four fields, so anything longer is a channel. With a
    sample rate given, any 1-D array is taken as a channel, however short.
    """
    if isinstance(value, np.ndarray) or (
        hasattr(value, "__array__") and not isinstance(value, (list, tuple))
    ):
        array = np.asarray(value)
        return array.ndim == 1 and (array.size > 4 or rate_given)
    if isinstance(value, (list, tuple)) and len(value) > 4:
        return all(_is_number(item) for item in value[:8])
    return False


class _Loaded:
    __slots__ = ("timeline", "method", "source", "count", "audio", "detection")

    def __init__(
        self,
        timeline: Timeline,
        method: str,
        source: str,
        count: int,
        audio: Optional[CallAudio] = None,
        detection: Optional[Detection] = None,
    ) -> None:
        self.timeline = timeline
        self.method = method
        self.source = source
        self.count = count
        self.audio = audio
        self.detection = detection


class CallAnalyzer:
    """Measure talk time, turns, interruptions, silence and pace in a call.

    Args:
        grace_s: Overlap allowed before starting to talk over someone counts as
            an interruption. Backchannels ("mm-hm") and replies that start a
            beat early stay under it.
        max_turn_pause_s: A pause longer than this ends a turn even if the same
            party speaks next, so a hold does not become one long monologue.
        min_speech_s: Audio only - a burst shorter than this is not speech.
        bridge_gap_s: Audio only - a pause shorter than this does not split
            speech in two.
        dominance_share: Flag a party with at least this share of talk time.
        long_monologue_s: Flag a turn at least this long.
        long_silence_s: Flag a silence at least this long.
    """

    def __init__(
        self,
        *,
        grace_s: float = 0.5,
        max_turn_pause_s: float = 3.0,
        min_speech_s: float = 0.12,
        bridge_gap_s: float = 0.3,
        dominance_share: float = 0.7,
        long_monologue_s: float = 60.0,
        long_silence_s: float = 10.0,
    ) -> None:
        checks = {
            "grace_s": grace_s,
            "max_turn_pause_s": max_turn_pause_s,
            "min_speech_s": min_speech_s,
            "bridge_gap_s": bridge_gap_s,
            "long_monologue_s": long_monologue_s,
            "long_silence_s": long_silence_s,
        }
        for name, value in checks.items():
            if not _is_number(value) or not np.isfinite(value) or value < 0:
                raise ValueError("{} must be a number of seconds >= 0; got {!r}".format(name, value))
        if not _is_number(dominance_share) or not 0.5 <= dominance_share <= 1.0:
            raise ValueError(
                "dominance_share must be between 0.5 and 1; got {!r}".format(dominance_share)
            )
        self.grace_s = float(grace_s)
        self.max_turn_pause_s = float(max_turn_pause_s)
        self.min_speech_s = float(min_speech_s)
        self.bridge_gap_s = float(bridge_gap_s)
        self.dominance_share = float(dominance_share)
        self.long_monologue_s = float(long_monologue_s)
        self.long_silence_s = float(long_silence_s)

    def __repr__(self) -> str:
        return (
            "CallAnalyzer(grace_s={}, max_turn_pause_s={}, min_speech_s={}, "
            "bridge_gap_s={}, dominance_share={}, long_monologue_s={}, "
            "long_silence_s={})".format(
                self.grace_s,
                self.max_turn_pause_s,
                self.min_speech_s,
                self.bridge_gap_s,
                self.dominance_share,
                self.long_monologue_s,
                self.long_silence_s,
            )
        )

    # ------------------------------------------------------------------ input

    def _from_audio(self, audio: CallAudio, speakers: Any) -> _Loaded:
        names = _channel_names(audio.n_channels, speakers)
        found = detect(audio, min_speech_s=self.min_speech_s, bridge_gap_s=self.bridge_gap_s)
        notes = list(audio.notes)
        for index, name in enumerate(names):
            level = found.levels[index]
            if level.threshold_db is None and audio.n_samples:
                notes.append(
                    "no speech found on channel {} ({}): its loudest moment is {:.0f} "
                    "dBFS over a noise floor of {:.0f} dBFS".format(
                        index + 1, name, level.peak_db, level.noise_db
                    )
                )
        for estimate in found.bleed:
            if not estimate.affected:
                notes.append(
                    "faint crosstalk from {} into {} ({:.0f} dB) sits below the speech "
                    "threshold and did not affect the figures".format(
                        names[estimate.source], names[estimate.target], estimate.level_db
                    )
                )
        timeline = Timeline(
            speakers=names,
            intervals={name: list(found.intervals[index]) for index, name in enumerate(names)},
            duration_s=audio.duration_s,
            words={},
            transcribed_s={},
            notes=notes,
            given=0,
        )
        return _Loaded(timeline, audio.kind, audio.source, audio.n_channels, audio, found)

    def _from_rows(self, rows: Sequence[Any], speakers: Any, source: str, notes: List[str]) -> _Loaded:
        timeline = build_timeline(rows, speakers, notes=notes)
        return _Loaded(timeline, "segments", source, timeline.given)

    def _load(self, source: Any, sample_rate: Optional[int], speakers: Any) -> _Loaded:
        notes: List[str] = []

        if isinstance(source, (str, bytes, os.PathLike)):
            path = os.fspath(source)
            if isinstance(path, bytes):
                path = path.decode("utf-8", "replace")
            if not os.path.isfile(path):
                raise FileNotFoundError("no such file: {}".format(path))
            extension = os.path.splitext(path)[1].lower()
            name = os.path.basename(path) or path
            if extension in _TABLE_FILES:
                raise ValueError(
                    "{}: read it with pandas (e.g. pandas.read_parquet) and pass the "
                    "DataFrame; this package reads .wav, .csv, .tsv and .json "
                    "directly".format(name)
                )
            if extension in _SEGMENT_FILES or extension in _JSON_FILES:
                if sample_rate is not None:
                    notes.append("sample_rate is only used for audio and was ignored")
                rows = rows_from_json(path) if extension in _JSON_FILES else rows_from_csv(path)
                return self._from_rows(rows, speakers, name, notes)
            return self._from_audio(load_wav(path, sample_rate), speakers)

        if hasattr(source, "itertracks"):
            return self._from_rows(rows_from_annotation(source), speakers, "<annotation>", notes)
        if _is_frame(source):
            return self._from_rows(rows_from_frame(source), speakers, "<DataFrame>", notes)
        if isinstance(source, np.ndarray):
            if source.dtype.kind in "biuf":
                channels = split_channels(source)
                return self._from_audio(
                    load_channels(channels, sample_rate, "<array>", "audio-arrays"), speakers
                )
            items = list(source)
        elif isinstance(source, Mapping):
            if "segments" not in source:
                raise ValueError(
                    "a dict source needs a 'segments' list; got keys {}".format(
                        sorted(str(key) for key in source.keys())
                    )
                )
            items = list(source["segments"])
        elif hasattr(source, "segments") and not isinstance(source, (list, tuple)):
            items = list(source.segments)
        elif isinstance(source, (list, tuple)) or hasattr(source, "__iter__"):
            items = list(source)
        else:
            raise ValueError(
                "cannot read a {} as a call; pass a stereo .wav path, a pair of mono "
                "arrays, or a list of (start_s, end_s, speaker) segments".format(
                    type(source).__name__
                )
            )

        if (
            len(items) == 2
            and isinstance(items[0], np.ndarray)
            and items[0].ndim == 2
            and isinstance(items[1], numbers.Integral)
        ):
            rate = int(items[1])
            if sample_rate is not None and int(sample_rate) != rate:
                raise ValueError(
                    "sample_rate={} was passed but the (samples, sample_rate) pair says "
                    "{}; drop one of them".format(int(sample_rate), rate)
                )
            return self._from_audio(
                load_channels(split_channels(items[0]), rate, "<array>", "audio-arrays"), speakers
            )
        if len(items) >= 2 and all(
            _looks_like_channel(item, sample_rate is not None) for item in items
        ):
            return self._from_audio(
                load_channels(items, sample_rate, "<arrays>", "audio-arrays"), speakers
            )
        if sample_rate is not None:
            notes.append("sample_rate is only used for audio and was ignored")
        return self._from_rows(items, speakers, "<segments>", notes)

    # ---------------------------------------------------------------- output

    def _bleed(self, loaded: _Loaded) -> Optional[BleedCheck]:
        found = loaded.detection
        if found is None:
            return None
        names = loaded.timeline.speakers
        paths = tuple(
            Crosstalk(
                source=names[item.source],
                target=names[item.target],
                level_db=round(item.level_db, 2),
                correlation=round(item.correlation, 4),
                lag_ms=round(item.lag_s * 1000.0, 1),
                affected=item.affected,
            )
            for item in found.bleed
        )
        pairs = tuple((names[a], names[b]) for a, b in found.same_audio)
        return BleedCheck(
            detected=bool(paths or pairs),
            same_audio=bool(pairs),
            paths=paths,
            same_audio_pairs=pairs,
        )

    def _flags(self, report: CallReport) -> List[str]:
        flags: List[str] = []
        stats = list(report.speakers.values())
        if report.bleed is not None:
            for first, second in report.bleed.same_audio_pairs:
                flags.append(
                    "Channels {} and {} carry the same audio (one recording saved "
                    "twice), so the parties cannot be told apart and every figure here "
                    "is unreliable; diarize the call and pass its segments "
                    "instead.".format(first, second)
                )
            for path in report.bleed.paths:
                if path.affected:
                    flags.append(
                        "Channel {} picks up {} at about {:.0f} dB (crosstalk). It was "
                        "detected and separated by level, but treat {}'s talk time and "
                        "the overlap figure with some care.".format(
                            path.target, path.source, path.level_db, path.target
                        )
                    )
        if report.bleed is not None and report.bleed.same_audio and len(stats) == 2:
            # Every other observation would be about one voice counted twice.
            return flags
        talkers = [item for item in stats if item.talk_time_s > 0]
        if not talkers:
            flags.append("No speech was found, so there is nothing to measure.")
            return flags
        for item in stats:
            if item.talk_time_s <= 0:
                flags.append("{} never spoke; the call is one-sided.".format(item.speaker))
        if len(talkers) >= 2:
            for item in talkers:
                if item.talk_share >= self.dominance_share:
                    flags.append(
                        "{} did most of the talking: {:.0f}% of the talk time.".format(
                            item.speaker, 100.0 * item.talk_share
                        )
                    )
        for item in stats:
            if item.longest_monologue_s >= self.long_monologue_s and len(talkers) >= 2:
                flags.append(
                    "{} spoke for {} at a stretch without the other side taking a "
                    "turn.".format(item.speaker, duration_text(item.longest_monologue_s))
                )
        span_minutes = report.speech_span_s / 60.0
        for item in stats:
            if item.interruptions >= INTERRUPTIONS_TO_FLAG and item.interruptions >= span_minutes / 2.0:
                flags.append(
                    "{} interrupted {} times, talking over the other side for more than "
                    "{:.1f}s each time.".format(item.speaker, item.interruptions, self.grace_s)
                )
        if report.overlap_share >= HEAVY_OVERLAP_SHARE:
            flags.append(
                "The parties talked over each other for {:.0f}% of the "
                "conversation.".format(100.0 * report.overlap_share)
            )
        if report.silence_share >= HIGH_SILENCE_SHARE:
            flags.append(
                "{:.0f}% of the conversation was silence.".format(100.0 * report.silence_share)
            )
        if report.longest_silence_s >= self.long_silence_s:
            flags.append(
                "The longest silence lasted {}, starting at {}.".format(
                    duration_text(report.longest_silence_s), clock_text(report.longest_silence_at_s)
                )
            )
        for item in stats:
            if (
                item.response_latency_s is not None
                and item.replies >= MIN_REPLIES_TO_FLAG
                and item.response_latency_s >= SLOW_REPLY_S
            ):
                flags.append(
                    "{} typically took {:.1f}s to reply.".format(item.speaker, item.response_latency_s)
                )
        return flags

    def _pace_flags(self, measured: Measurement) -> List[str]:
        flags: List[str] = []
        for name, party in measured.parties.items():
            if party.wpm is None or party.transcribed < MIN_PACE_SPEECH_S:
                continue
            if party.wpm >= FAST_WPM:
                flags.append("{} spoke fast: {:.0f} words per minute.".format(name, party.wpm))
            elif party.wpm <= SLOW_WPM:
                flags.append("{} spoke slowly: {:.0f} words per minute.".format(name, party.wpm))
        return flags

    def analyze(
        self,
        source: Any,
        *,
        sample_rate: Optional[int] = None,
        speakers: Any = None,
    ) -> CallReport:
        """Measure a call. See :func:`analyze` for the accepted inputs."""
        loaded = self._load(source, sample_rate, speakers)
        timeline = loaded.timeline
        measured = measure(
            timeline, grace_s=self.grace_s, max_turn_pause_s=self.max_turn_pause_s
        )

        def r6(value: float) -> float:
            return round(float(value), 6)

        stats: Dict[str, SpeakerStats] = {}
        for name in timeline.speakers:
            party = measured.parties[name]
            stats[name] = SpeakerStats(
                speaker=name,
                talk_time_s=r6(party.talk_time),
                talk_share=r6(party.talk_share),
                turns=party.turns,
                mean_turn_s=r6(party.mean_turn),
                longest_monologue_s=r6(party.longest_turn),
                interruptions=party.interruptions,
                interrupted=party.interrupted,
                backchannels=party.backchannels,
                response_latency_s=None if party.latency is None else r6(party.latency),
                replies=party.replies,
                words=party.words,
                words_per_minute=None if party.wpm is None else r6(party.wpm),
            )

        detection: Optional[Dict[str, Dict[str, Optional[float]]]] = None
        if loaded.detection is not None:
            detection = {}
            for index, name in enumerate(timeline.speakers):
                level = loaded.detection.levels[index]
                detection[name] = {
                    "noise_floor_dbfs": round(level.noise_db, 2),
                    "peak_dbfs": round(level.peak_db, 2),
                    "threshold_dbfs": None
                    if level.threshold_db is None
                    else round(level.threshold_db, 2),
                }

        report = CallReport(
            speakers=stats,
            duration_s=r6(max(timeline.duration_s, measured.speech_end)),
            speech_start_s=r6(measured.speech_start),
            speech_end_s=r6(measured.speech_end),
            silence_share=r6(measured.silence_share),
            longest_silence_s=r6(measured.longest_silence),
            longest_silence_at_s=r6(measured.longest_silence_at),
            overlap_share=r6(measured.overlap_share),
            turn_count=len(measured.turns),
            response_latency_s=None if measured.latency is None else r6(measured.latency),
            interruption_count=len(measured.interruptions),
            notes=list(timeline.notes),
            source=loaded.source,
            method=loaded.method,
            method_detail=method_detail(loaded.method, loaded.count),
            detection=detection,
            segments=tuple(
                Segment(r6(start), r6(end), name) for start, end, name in measured.segments
            ),
            turns=tuple(Turn(turn.speaker, r6(turn.start), r6(turn.end)) for turn in measured.turns),
            interruption_events=tuple(
                Interruption(r6(event.time), event.by, event.of, r6(event.overlap))
                for event in measured.interruptions
            ),
            settings={
                "grace_s": self.grace_s,
                "max_turn_pause_s": self.max_turn_pause_s,
                "min_speech_s": self.min_speech_s,
                "bridge_gap_s": self.bridge_gap_s,
                "dominance_share": self.dominance_share,
                "long_monologue_s": self.long_monologue_s,
                "long_silence_s": self.long_silence_s,
            },
        )
        report.bleed = self._bleed(loaded)
        report.flags = self._flags(report) + self._pace_flags(measured)
        if len(timeline.speakers) > 2:
            report.notes.append(
                "{} parties: interruptions, replies and turns are counted between "
                "any two of them".format(len(timeline.speakers))
            )
        logger.debug("analyzed %s: %r", loaded.source, report)
        return report


def analyze(source: Any, *, sample_rate: Optional[int] = None, speakers: Any = None) -> CallReport:
    """Measure talk time, interruptions, silence and pace in a two-party call.

    Args:
        source: One of

            * a path to a stereo ``.wav`` file where each channel is one party
              (the usual call-recording layout); PCM, float, mu-law and A-law;
            * a pair of mono sample arrays, one per party (needs ``sample_rate``),
              or a 2-D samples-by-channels array;
            * diarized segments from any tool: ``[(start_s, end_s, speaker), ...]``
              or ``(start_s, end_s, speaker, text)`` to get words per minute, as
              tuples, dicts, objects, a pandas DataFrame, or a ``.csv``/``.json``
              file.
        sample_rate: Samples per second, for array input only.
        speakers: Names for the parties. For audio, one per channel in order,
            or ``{channel_index: name}``. For segments, a list declaring the
            parties (a party with no segments is reported with zero talk), or a
            dict renaming labels, e.g. ``{0: "agent", 1: "customer"}``.

    Returns:
        A :class:`CallReport`. Nothing is ever written to the input.

    Raises:
        ValueError: The input cannot be read as a call (a mono recording, a
            segment that ends before it starts, and so on). The message says why.
        FileNotFoundError: A path was given and nothing is there.
    """
    return CallAnalyzer().analyze(source, sample_rate=sample_rate, speakers=speakers)
