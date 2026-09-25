"""voice-commands-ai: turn spoken or transcribed commands into structured Python actions.

This package maps TEXT to actions; it does not recognise speech. Plug in any
transcriber and it turns the transcript into a command with typed slots::

    from voice_commands_ai import Commands

    cmds = Commands()
    cmds.add("set temperature to {value:number} degrees", name="thermostat")
    cmds.match("um could you set the temperature to twenty one degrees please").slots
    # {'value': 21}

The matcher is a heuristic: filler words are skipped, spoken numbers are read,
small transcription errors are tolerated by edit distance, and a confidence
floor turns poor guesses into ``None``. :func:`parse_number` reads spoken
English numbers on their own.
"""

from ._commands import Command, CommandError, Commands
from ._match import Match
from ._numbers import parse_number

__version__ = "0.1.0"

__all__ = [
    "Commands",
    "Match",
    "Command",
    "CommandError",
    "parse_number",
    "__version__",
]
