"""Resolve the Flight Deck home directory — per instance.

Several state stores (code-apps under ``apps/``, their ``app_data/`` /
``app_files/`` / ``app_manifests/``, and the ``agent_secret`` file) used
to live only under ``CAPTAIN_CLAW_FD_HOME`` or the shared
``~/.captain-claw-fd`` default. On a host running several Flight Decks
that meant one deck's app data / secret leaked into another. Every other
FD store already isolates by ``FD_DATA_DIR``; this helper lets those
stores do the same, with one precedence defined in one place.

Precedence (first match wins):

1. ``CAPTAIN_CLAW_FD_HOME`` — an explicit FD home override, always wins.
2. ``FD_DATA_DIR`` — the general per-instance data dir the DB, plans,
   events, scheduler and MCP config already sit under. State lands flat
   inside it (``<data_dir>/apps``, ``<data_dir>/agent_secret``, …), the
   same way :mod:`mcp_storage` writes ``<data_dir>/mcp_servers.json``.
3. the caller's ``legacy`` fallback — the shared default used when no
   isolation env is set (single-instance installs, unchanged behaviour).

Only #3 is shared across instances, so a deck that isolates via
``FD_DATA_DIR`` (or ``CAPTAIN_CLAW_FD_HOME``) starts with its own empty
tree instead of inheriting another deck's.
"""

from __future__ import annotations

import os
from pathlib import Path


def fd_home(legacy: Path) -> Path:
    """Return the base dir for per-instance FD state (see module docstring).

    ``legacy`` is the shared fallback to use when neither isolation env is
    set; callers pass their own so the pre-existing default is preserved
    exactly (``app_*`` anchor to ``$HOME``; ``agent_secret`` anchors to the
    passwd home so a sandboxed-``HOME`` agent still converges).
    """
    home = os.environ.get("CAPTAIN_CLAW_FD_HOME", "").strip()
    if home:
        return Path(home).expanduser().resolve()
    data_dir = os.environ.get("FD_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    return legacy
