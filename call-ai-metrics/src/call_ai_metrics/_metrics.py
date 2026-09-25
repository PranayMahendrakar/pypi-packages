"""Talk time, turns, interruptions, silence, overlap and response latency.

Every number here is computed from a :class:`~call_ai_metrics._segments.Timeline`,
which is nothing but "who was speaking when". Where that timeline came from - a
stereo recording, a diarizer, a hand-made list - does not matter to this module,
which is exactly why every figure is only as good as the timeline it is given.

Definitions, all in seconds:

* talk time: the union of a party's speech, so overlapping segments never count
  twice.
* interruption: a party starts to speak while another is still speaking, and the
  two then talk over each other for longer than the grace window. Shorter
  overlap - "mm-hm", or a reply that starts a beat before the other finishes -
  is not an interruption.
* backchannel: a short vocalisation made entirely inside someone else's speech,
  overlapping it for no longer than the grace window.
* turn: a stretch where one party holds the floor. It ends when someone else
  takes the floor, or when its owner pauses longer than ``max_turn_pause_s``.
  Speech made entirely inside someone else's turn (backchannels, talking over
  them without taking over) does not take the floor.
* response latency: the silence between the end of one party's turn and the
  start of the next party's turn, 0 when the reply overlapped. The report gives
  the median, so one long hold does not swamp a call of quick replies.
* silence and overlap: measured between the first and the last speech of the
  call, so ring time before pick-up and the tail after hang-up do not count.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Sequence, Tuple

from ._segments import EPS, Timeline, merge_intervals

JOIN_GAP_S = 0.25
"""Pauses shorter than this do not stop someone "still speaking"."""

EDGE_TOLERANCE_S = 0.1
"""Speech ending this soon after the other party stops still counts as inside."""

Span = Tuple[float, float]


class TurnSpan:
    """One party holding the floor."""

    __slots__ = ("speaker", "start", "end")

    def __init__(self, speaker: str, start: float, end: float) -> None:
        self.speaker = speaker
        self.start = start
        self.end = end


class InterruptionEvent:
    """One party starting to talk over another for longer than the grace window."""

    __slots__ = ("time", "by", "of", "overlap")

    def __init__(self, time: float, by: str, of: str, overlap: float) -> None:
        self.time = time
        self.by = by
        self.of = of
        self.overlap = overlap


class PartyNumbers:
    """Everything measured about one party."""

    __slots__ = (
        "talk_time",
        "talk_share",
        "turns",
        "mean_turn",
        "longest_turn",
        "interruptions",
        "interrupted",
        "backchannels",
        "latency",
        "replies",
        "words",
        "wpm",
        "transcribed",
    )

    def __init__(self) -> None:
        self.talk_time = 0.0
        self.talk_share = 0.0
        self.turns = 0
        self.mean_turn = 0.0
        self.longest_turn = 0.0
        self.interruptions = 0
        self.interrupted = 0
        self.backchannels = 0
        self.latency: Optional[float] = None
        self.replies = 0
        self.words: Optional[int] = None
        self.wpm: Optional[float] = None
        self.transcribed = 0.0


class Measurement:
    """The whole call, measured."""

    __slots__ = (
        "parties",
        "speech_start",
        "speech_end",
        "silence",
        "silence_share",
        "longest_silence",
        "longest_silence_at",
        "overlap",
        "overlap_share",
        "latency",
        "turns",
        "interruptions",
        "segments",
    )

    def __init__(self) -> None:
        self.parties: Dict[str, PartyNumbers] = {}
        self.speech_start = 0.0
        self.speech_end = 0.0
        self.silence = 0.0
        self.silence_share = 0.0
        self.longest_silence = 0.0
        self.longest_silence_at = 0.0
        self.overlap = 0.0
        self.overlap_share = 0.0
        self.latency: Optional[float] = None
        self.turns: List[TurnSpan] = []
        self.interruptions: List[InterruptionEvent] = []
        self.segments: List[Tuple[float, float, str]] = []


def _bridge(spans: Sequence[Span], gap: float) -> List[Span]:
    joined: List[Span] = []
    for start, end in spans:
        if joined and start - joined[-1][1] < gap:
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def _containing(spans: Sequence[Span], starts: Sequence[float], moment: float) -> Optional[Span]:
    """The span that was already under way at ``moment`` and has not ended."""
    index = bisect.bisect_right(starts, moment) - 1
    if index < 0:
        return None
    start, end = spans[index]
    if start < moment - EPS and moment < end - EPS:
        return spans[index]
    return None


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def measure(timeline: Timeline, *, grace_s: float, max_turn_pause_s: float) -> Measurement:
    """Measure a timeline. See the module docstring for every definition."""
    result = Measurement()
    names = list(timeline.speakers)
    rank = {name: index for index, name in enumerate(names)}
    spans = {name: list(timeline.intervals.get(name, [])) for name in names}
    for name in names:
        result.parties[name] = PartyNumbers()

    segments = sorted(
        (start, end, rank[name], name) for name in names for start, end in spans[name]
    )
    result.segments = [(start, end, name) for start, end, _, name in segments]

    # Talk time and share.
    total_talk = 0.0
    for name in names:
        talk = sum(end - start for start, end in spans[name])
        result.parties[name].talk_time = talk
        total_talk += talk
    for name in names:
        party = result.parties[name]
        party.talk_share = party.talk_time / total_talk if total_talk > EPS else 0.0

    if not segments:
        return result

    # Silence and overlap, swept between the first and the last speech.
    events: List[Tuple[float, int]] = []
    for name in names:
        for start, end in spans[name]:
            events.append((start, 1))
            events.append((end, -1))
    events.sort()
    result.speech_start = events[0][0]
    result.speech_end = events[-1][0]
    active = 0
    previous = events[0][0]
    for moment, delta in events:
        width = moment - previous
        if width > EPS:
            if active == 0:
                result.silence += width
                if width > result.longest_silence:
                    result.longest_silence = width
                    result.longest_silence_at = previous
            elif active >= 2:
                result.overlap += width
        active += delta
        previous = moment
    span = result.speech_end - result.speech_start
    if span > EPS:
        result.silence_share = result.silence / span
        result.overlap_share = result.overlap / span

    # Who was speaking when each segment started: interruptions and backchannels.
    own_bridged = {name: _bridge(spans[name], JOIN_GAP_S) for name in names}
    own_starts = {name: [s for s, _ in own_bridged[name]] for name in names}
    others: Dict[str, List[Span]] = {}
    for name in names:
        merged, _ = merge_intervals(
            span for other in names if other != name for span in spans[other]
        )
        others[name] = _bridge(merged, JOIN_GAP_S)
    other_starts = {name: [s for s, _ in others[name]] for name in names}

    floor: List[Tuple[float, float, str]] = []
    for start, end, _, name in segments:
        under_way = _containing(others[name], other_starts[name], start)
        if under_way is None:
            floor.append((start, end, name))
            continue
        overlap = min(end, under_way[1]) - start
        speaking: List[Tuple[float, str]] = []
        for other in names:
            if other == name:
                continue
            found = _containing(own_bridged[other], own_starts[other], start)
            if found is not None:
                speaking.append((found[0], other))
        victim = min(speaking)[1] if speaking else next(o for o in names if o != name)
        interrupting = overlap > grace_s + EPS
        if interrupting:
            result.interruptions.append(InterruptionEvent(start, name, victim, overlap))
            result.parties[name].interruptions += 1
            result.parties[victim].interrupted += 1
        if end <= under_way[1] + EDGE_TOLERANCE_S:
            if not interrupting:
                result.parties[name].backchannels += 1
        else:
            floor.append((start, end, name))

    # Turns: who holds the floor, and for how long.
    turns: List[TurnSpan] = []
    for start, end, name in floor:
        if (
            turns
            and turns[-1].speaker == name
            and start - turns[-1].end <= max_turn_pause_s + EPS
        ):
            turns[-1].end = max(turns[-1].end, end)
        else:
            turns.append(TurnSpan(name, start, end))
    result.turns = turns

    durations: Dict[str, List[float]] = {name: [] for name in names}
    for turn in turns:
        durations[turn.speaker].append(turn.end - turn.start)
    for name in names:
        party = result.parties[name]
        party.turns = len(durations[name])
        if durations[name]:
            party.mean_turn = sum(durations[name]) / len(durations[name])
            party.longest_turn = max(durations[name])

    # Response latency: the gap before each change of speaker.
    replies: Dict[str, List[float]] = {name: [] for name in names}
    every: List[float] = []
    for before, after in zip(turns, turns[1:]):
        if before.speaker != after.speaker:
            gap = max(0.0, after.start - before.end)
            replies[after.speaker].append(gap)
            every.append(gap)
    result.latency = _median(every)
    for name in names:
        party = result.parties[name]
        party.replies = len(replies[name])
        party.latency = _median(replies[name])

    # Pace, only where a transcript was given.
    for name in names:
        if name in timeline.words:
            party = result.parties[name]
            party.words = timeline.words[name]
            party.transcribed = timeline.transcribed_s.get(name, 0.0)
            if party.transcribed > EPS:
                party.wpm = party.words / (party.transcribed / 60.0)
    return result
