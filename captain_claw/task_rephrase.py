"""Whether an agent rewrites a complex request before working on it.

Task rephrase is an extra LLM call that restructures a long, free-form prompt
(URLs, column lists, file names) into a tidy brief before the agent starts.
Captain began with models that needed that help. Today's models read the
prompt as written, and the rewrite costs a call and can drift from the words
the person chose, so it is off unless the agent's owner turns it on.

The switch is ``~/.captain-claw/task_rephrase.txt`` ("on" or "off"). Flight
Deck's agent settings write it, and the agent reads it on every turn, so a
change applies from its next message without a restart. With no file, the
config's ``scale.task_rephrase_enabled`` decides (off by default).
"""

from __future__ import annotations

from pathlib import Path

FILENAME = "task_rephrase.txt"
SETTING_PATH = Path("~/.captain-claw").expanduser() / FILENAME

_ON = {"on", "true", "1", "yes"}
_OFF = {"off", "false", "0", "no"}


def parse(text: object) -> bool | None:
    """"on"/"off" (and the usual spellings) as True/False; anything else None."""
    value = str(text or "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    return None


def render(enabled: bool) -> str:
    """What the switch file holds for *enabled*."""
    return "on" if enabled else "off"


def enabled() -> bool:
    """Is task rephrase on for this agent right now? The switch file, else the config."""
    try:
        switch = parse(SETTING_PATH.read_text(encoding="utf-8"))
    except OSError:
        switch = None
    if switch is not None:
        return switch
    from captain_claw.config import get_config

    return bool(get_config().scale.task_rephrase_enabled)


def clear() -> None:
    """Drop the switch file so the config decides again (the agent's own
    settings page changed ``scale.task_rephrase_enabled``)."""
    try:
        SETTING_PATH.unlink()
    except OSError:
        pass
