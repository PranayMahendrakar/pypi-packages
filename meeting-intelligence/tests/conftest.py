"""Shared transcripts for the test suite."""
from __future__ import annotations

import pytest

MEETING = """
Alice: Morning everyone. We decided to go with Postgres for the event store.
Bob: I'll write the migration by Friday.
Alice: Can you also update the runbook, Bob?
Bob: Sure, I can do that. What about the old rows?
Carla: We agreed to archive anything older than a year.
Alice: Who owns the archive job?
Carla: Assigned to Bob. He has the cron access.
Bob: Fine. Action item: drop the legacy table after the archive lands.
Alice: The plan is to ship the whole thing next quarter.
Carla: Do we need a second reviewer for the migration?
"""

UNLABELLED = (
    "We decided to go with Postgres for the event store. "
    "Someone will write the migration by Friday. "
    "We need to archive the old rows before that lands. "
    "What do we do about the reporting views?"
)

VTT = """WEBVTT

00:00:00.000 --> 00:00:04.500
<v Alice>We decided to go with Postgres.

00:00:04.500 --> 00:00:09.250
<v Bob>I'll write the migration by Friday.

01:02:03.125 --> 01:02:08.000
<v Alice>Can you review it, Bob?

01:02:08.000 --> 01:02:12.000
<v Bob>Yes, I can review the migration.
"""

SRT = """1
00:00:00,000 --> 00:00:04,500
Alice: We decided to go with Postgres.

2
00:00:04,500 --> 00:00:09,250
Bob: I'll write the migration by Friday.

3
01:02:03,125 --> 01:02:08,000
Alice: Can you review it, Bob?
"""

UNICODE = """
Zoë Müller: We decided to go with Postgres, naturlich.
李雷: I'll write the migration by Friday. 我们需要归档旧数据。
Björn Ødegård: Can you review it, Zoë?
Zoë Müller: Ja, sure. What about the café dashboard?
"""


@pytest.fixture
def meeting() -> str:
    """A labelled multi-speaker transcript with decisions, actions and questions."""
    return MEETING


@pytest.fixture
def raising_llm():
    """An ``llm`` callable that always blows up."""

    def _llm(prompt: str) -> str:
        raise RuntimeError("model unavailable")

    return _llm
