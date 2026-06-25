"""Local interactive-terminal facility.

A standalone PTY daemon (``daemon.py``) holds long-lived pseudo-terminal
sessions on the Mac and survives agent restarts.  The agent reaches it
through the ``terminal`` tool (``captain_claw/tools/terminal.py``) over
localhost — so the same terminal is drivable from both the Flight Deck
web chat and WhatsApp, with no new Flight Deck transport.
"""

DEFAULT_PORT = 23190
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}"
TOKEN_HEADER = "X-Claw-Token"

__all__ = ["DEFAULT_PORT", "DEFAULT_URL", "TOKEN_HEADER"]
