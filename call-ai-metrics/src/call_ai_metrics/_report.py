"""The result objects: plain dataclasses with a summary, a dict and a DataFrame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

_METHODS = {
    "stereo-wav": "a {n}-channel recording, one party per channel, speech found by a "
    "per-channel energy threshold",
    "audio-arrays": "{n} audio channels, one party per channel, speech found by a "
    "per-channel energy threshold",
    "segments": "{n} diarized segments supplied by the caller",
}


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def duration_text(seconds: float) -> str:
    """``41.2s`` under a minute, ``1:32`` under an hour, ``1:02:03`` beyond."""
    if seconds < 59.95:
        return "{:.1f}s".format(seconds)
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, secs)
    return "{}:{:02d}".format(minutes, secs)


def clock_text(seconds: float) -> str:
    """A position in the call as ``m:ss`` (or ``h:mm:ss``)."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, secs)
    return "{}:{:02d}".format(minutes, secs)


def _percent(share: float) -> str:
    return "{:.0f}%".format(100.0 * share)


@dataclass(frozen=True)
class Segment:
    """A stretch of speech by one party, as used for every measurement."""

    start_s: float
    end_s: float
    speaker: str

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.end_s - self.start_s

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {"start_s": _round(self.start_s), "end_s": _round(self.end_s), "speaker": self.speaker}


@dataclass(frozen=True)
class Turn:
    """One party holding the floor, from their first word to their last."""

    speaker: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length of the turn in seconds, pauses inside it included."""
        return self.end_s - self.start_s

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {"speaker": self.speaker, "start_s": _round(self.start_s), "end_s": _round(self.end_s)}


@dataclass(frozen=True)
class Interruption:
    """``by`` started talking at ``time_s`` over ``of``, for ``overlap_s`` seconds."""

    time_s: float
    by: str
    of: str
    overlap_s: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "time_s": _round(self.time_s),
            "by": self.by,
            "of": self.of,
            "overlap_s": _round(self.overlap_s),
        }


@dataclass(frozen=True)
class Crosstalk:
    """Channel ``target`` carries a quieter copy of channel ``source``."""

    source: str
    target: str
    level_db: float
    correlation: float
    lag_ms: float
    affected: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "source": self.source,
            "target": self.target,
            "level_db": _round(self.level_db, 1),
            "correlation": _round(self.correlation),
            "lag_ms": _round(self.lag_ms, 1),
            "affected": self.affected,
        }


@dataclass(frozen=True)
class BleedCheck:
    """Whether the channels of a recording leak into each other.

    ``detected`` is True when any channel carries a copy of another. ``paths``
    says which way, how loud (relative to the source) and whether the leak was
    loud enough to have been mistaken for speech; such frames were separated
    by level before anything was measured. ``same_audio`` means two channels
    are one recording twice, and cannot be separated at all.
    """

    detected: bool
    same_audio: bool
    paths: Tuple[Crosstalk, ...] = ()
    same_audio_pairs: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "detected": self.detected,
            "same_audio": self.same_audio,
            "paths": [path.to_dict() for path in self.paths],
            "same_audio_pairs": [list(pair) for pair in self.same_audio_pairs],
        }


@dataclass(frozen=True)
class SpeakerStats:
    """Everything measured about one party of the call.

    Attributes:
        speaker: The party's name.
        talk_time_s: Seconds of speech; overlapping segments are not double counted.
        talk_share: This party's part of all talk time, 0 to 1.
        turns: How many times this party took the floor.
        mean_turn_s: Average turn length.
        longest_monologue_s: Longest turn, the longest stretch holding the floor.
        interruptions: Times this party started talking over someone else for
            longer than the grace window.
        interrupted: Times someone else did that to this party.
        backchannels: Short sounds ("mm-hm") made inside the other's speech,
            within the grace window, which are not interruptions.
        response_latency_s: Median gap before this party replied, or None if
            they never replied to anyone.
        replies: How many replies that median is taken over.
        words: Words in this party's transcript, or None without one.
        words_per_minute: Words per minute of transcribed speech, or None.
    """

    speaker: str
    talk_time_s: float
    talk_share: float
    turns: int
    mean_turn_s: float
    longest_monologue_s: float
    interruptions: int
    interrupted: int
    backchannels: int
    response_latency_s: Optional[float]
    replies: int
    words: Optional[int]
    words_per_minute: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "speaker": self.speaker,
            "talk_time_s": _round(self.talk_time_s),
            "talk_share": _round(self.talk_share, 4),
            "turns": self.turns,
            "mean_turn_s": _round(self.mean_turn_s),
            "longest_monologue_s": _round(self.longest_monologue_s),
            "interruptions": self.interruptions,
            "interrupted": self.interrupted,
            "backchannels": self.backchannels,
            "response_latency_s": _round(self.response_latency_s),
            "replies": self.replies,
            "words": self.words,
            "words_per_minute": _round(self.words_per_minute, 1),
        }


@dataclass
class CallReport:
    """What happened on a call: who talked, how much, over whom, and how fast.

    Every figure is computed from a speech timeline - who was speaking when -
    and is only as good as that timeline. ``method`` and ``notes`` say where the
    timeline came from and what was done to it; ``flags`` are plain-language
    observations worth a look.
    """

    speakers: Dict[str, SpeakerStats]
    duration_s: float
    speech_start_s: float
    speech_end_s: float
    silence_share: float
    longest_silence_s: float
    longest_silence_at_s: float
    overlap_share: float
    turn_count: int
    response_latency_s: Optional[float]
    interruption_count: int
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source: str = "<segments>"
    method: str = "segments"
    method_detail: str = ""
    bleed: Optional[BleedCheck] = None
    detection: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    segments: Tuple[Segment, ...] = ()
    turns: Tuple[Turn, ...] = ()
    interruption_events: Tuple[Interruption, ...] = ()
    settings: Dict[str, float] = field(default_factory=dict)

    @property
    def speech_span_s(self) -> float:
        """Seconds from the first speech to the last; silence shares use this."""
        return max(0.0, self.speech_end_s - self.speech_start_s)

    @property
    def talk_time_s(self) -> float:
        """Total talk time of all parties (overlap counts once per party)."""
        return sum(stats.talk_time_s for stats in self.speakers.values())

    def __getitem__(self, key: Union[str, int]) -> SpeakerStats:
        if isinstance(key, int):
            return list(self.speakers.values())[key]
        return self.speakers[key]

    def __iter__(self) -> Iterator[SpeakerStats]:
        return iter(self.speakers.values())

    def __len__(self) -> int:
        return len(self.speakers)

    def __repr__(self) -> str:
        shares = " / ".join(
            "{} {}".format(stats.speaker, _percent(stats.talk_share))
            for stats in self.speakers.values()
        )
        return "CallReport({} parties, {} of speech span, {}, {} interruption{}, {} flag{})".format(
            len(self.speakers),
            duration_text(self.speech_span_s),
            shares,
            self.interruption_count,
            "" if self.interruption_count == 1 else "s",
            len(self.flags),
            "" if len(self.flags) == 1 else "s",
        )

    def _overall_dict(self) -> Dict[str, Any]:
        return {
            "silence_share": _round(self.silence_share, 4),
            "longest_silence_s": _round(self.longest_silence_s),
            "longest_silence_at_s": _round(self.longest_silence_at_s),
            "overlap_share": _round(self.overlap_share, 4),
            "turn_count": self.turn_count,
            "response_latency_s": _round(self.response_latency_s),
            "interruption_count": self.interruption_count,
            "talk_time_s": _round(self.talk_time_s),
        }

    def to_dict(self) -> Dict[str, Any]:
        """The whole report as a JSON-safe dict."""
        return {
            "source": self.source,
            "method": self.method,
            "method_detail": self.method_detail,
            "duration_s": _round(self.duration_s),
            "speech_start_s": _round(self.speech_start_s),
            "speech_end_s": _round(self.speech_end_s),
            "speech_span_s": _round(self.speech_span_s),
            "overall": self._overall_dict(),
            "speakers": {name: stats.to_dict() for name, stats in self.speakers.items()},
            "flags": list(self.flags),
            "notes": list(self.notes),
            "bleed": self.bleed.to_dict() if self.bleed is not None else None,
            "detection": {
                name: {key: _round(value, 1) for key, value in levels.items()}
                for name, levels in self.detection.items()
            }
            if self.detection is not None
            else None,
            "settings": dict(self.settings),
            "turns": [turn.to_dict() for turn in self.turns],
            "interruptions": [event.to_dict() for event in self.interruption_events],
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def to_frame(self) -> Any:
        """One row per party as a pandas DataFrame; overall figures in ``.attrs``.

        Needs pandas (``pip install call-ai-metrics[pandas]``); nothing else in
        this package does.
        """
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - depends on the environment
            raise ImportError(
                "CallReport.to_frame() needs pandas; install it with "
                "pip install pandas (or pip install call-ai-metrics[pandas])"
            ) from None
        rows = [stats.to_dict() for stats in self.speakers.values()]
        columns = list(SpeakerStats.__dataclass_fields__)
        frame = pd.DataFrame(rows, columns=columns).set_index("speaker")
        frame.attrs["overall"] = self._overall_dict()
        frame.attrs["flags"] = list(self.flags)
        return frame

    def summary(self) -> str:
        """A human-readable report, plain ASCII apart from the party names."""
        lines: List[str] = []
        lines.append(
            "call report: {} parties, {} from first to last speech ({} recorded)".format(
                len(self.speakers),
                duration_text(self.speech_span_s),
                duration_text(self.duration_s),
            )
        )
        lines.append("source: {} - {}".format(self.source, self.method_detail))
        width = max([5] + [len(name) for name in self.speakers])
        header = "  {:<{w}}  {:>7}  {:>5}  {:>5}  {:>8}  {:>7}  {:>10}  {:>9}  {:>4}".format(
            "party", "talk", "share", "turns", "avg turn", "longest", "interrupts", "reply gap", "wpm", w=width
        )
        lines.append("")
        lines.append(header)
        for name, stats in self.speakers.items():
            lines.append(
                "  {:<{w}}  {:>7}  {:>5}  {:>5}  {:>8}  {:>7}  {:>10}  {:>9}  {:>4}".format(
                    name,
                    duration_text(stats.talk_time_s),
                    _percent(stats.talk_share),
                    stats.turns,
                    duration_text(stats.mean_turn_s),
                    duration_text(stats.longest_monologue_s),
                    stats.interruptions,
                    "-"
                    if stats.response_latency_s is None
                    else "{:.1f}s".format(stats.response_latency_s),
                    "-" if stats.words_per_minute is None else "{:.0f}".format(stats.words_per_minute),
                    w=width,
                )
            )
        reply = (
            "no replies to time"
            if self.response_latency_s is None
            else "typical reply after {:.1f}s".format(self.response_latency_s)
        )
        silence = "silence {}".format(_percent(self.silence_share))
        if self.longest_silence_s > 0:
            silence += " (longest {} at {})".format(
                duration_text(self.longest_silence_s), clock_text(self.longest_silence_at_s)
            )
        lines.append("")
        lines.append(
            "  overall: {}, overlap {}, {} turn{}, {}, {} interruption{}".format(
                silence,
                _percent(self.overlap_share),
                self.turn_count,
                "" if self.turn_count == 1 else "s",
                reply,
                self.interruption_count,
                "" if self.interruption_count == 1 else "s",
            )
        )
        lines.append("")
        if self.flags:
            lines.append("flags:")
            lines.extend("  - {}".format(flag) for flag in self.flags)
        else:
            lines.append("flags: none")
        if self.notes:
            lines.append("notes:")
            lines.extend("  - {}".format(note) for note in self.notes)
        lines.append("")
        lines.append(
            "Every figure is only as good as the speech segmentation behind it "
            "({}).".format(self.method_detail)
        )
        return "\n".join(lines)

    def explain(self) -> str:
        """How each figure was measured, with the settings this report used."""
        grace = self.settings.get("grace_s", 0.0)
        pause = self.settings.get("max_turn_pause_s", 0.0)
        lines = [
            "How this report was measured",
            "  timeline: {}. Every figure below is computed from it, so it is only "
            "as good as that segmentation.".format(self.method_detail),
            "  talk time: the union of each party's speech; overlapping segments "
            "from one party count once. talk share is a party's part of all talk time.",
            "  interruption: a party starts talking while another is still "
            "speaking and they overlap for more than {:.2f}s (the grace window). "
            "Shorter overlap, such as 'mm-hm' or a reply that starts a beat early, "
            "is not an interruption.".format(grace),
            "  backchannel: a short sound made entirely inside the other party's "
            "speech, within the grace window.",
            "  turn: a stretch holding the floor. It ends when another party takes "
            "the floor or its owner pauses longer than {:.1f}s; speech made entirely "
            "inside someone else's turn does not take the floor. longest monologue "
            "is the longest turn.".format(pause),
            "  response latency: the silence between one party's turn ending and "
            "another's starting (0 if they overlapped); the median is reported.",
            "  silence and overlap: shares of the time from the first speech to the "
            "last, so ring time and the tail after hang-up do not count.",
            "  words per minute: transcript words divided by the minutes of speech "
            "that carried a transcript; only when text is given with the segments.",
        ]
        if self.method != "segments":
            lines.append(
                "  speech detection: each channel's threshold is set from its own "
                "noise floor and loudest speech; this is an energy heuristic, not a "
                "neural voice detector. Crosstalk between channels is detected from "
                "how closely one channel's loudness tracks the other's."
            )
        return "\n".join(lines)


def method_detail(method: str, count: int) -> str:
    """One phrase saying where the speech timeline came from."""
    return _METHODS.get(method, method).format(n=count)
