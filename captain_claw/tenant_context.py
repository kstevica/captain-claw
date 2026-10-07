"""Owner profile ("tenant context") that Flight Deck hands to this agent.

Flight Deck composes the owner's profile — who they are, their company and
their standing preferences, merged with the deck-wide defaults — into two
markdown files in the agent's config home:

  ``~/.captain-claw/tenant_context.md``          full block
  ``~/.captain-claw/tenant_context.compact.md``  short block (≤1000 chars)

The agent inserts the text verbatim into its system prompt (FD owns the
wording; there is no template rendering here, so braces need no escaping).
FD rewrites the files on every profile save and deletes them when the profile
is empty, so reads resolve ``Path.home()`` at call time and are mtime-cached:
a change applies on the agent's next turn with no restart.
"""

from __future__ import annotations

import os
from pathlib import Path

FULL_FILENAME = "tenant_context.md"
COMPACT_FILENAME = "tenant_context.compact.md"
CACHE_SPLIT_MARKER = "<!-- CACHE_SPLIT -->"

# FD caps the composed full block at roughly 10k chars; this only guards the
# system prompt against a corrupt or hand-edited file.
_MAX_CHARS = 16_000

# path → ((mtime_ns, size, inode), text). FD writes via tmp + os.replace, so
# the inode changes on every save even when mtime granularity is coarse.
_cache: dict[str, tuple[tuple[int, int, int], str]] = {}


def _read_cached(path: Path) -> str:
    """Return the stripped text of *path*, or "" when it is missing/unreadable."""
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        _cache.pop(key, None)
        return ""
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    hit = _cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        text = path.read_text(encoding="utf-8").strip()[:_MAX_CHARS]
    except (OSError, UnicodeDecodeError):
        return ""
    _cache[key] = (stamp, text)
    return text


def load_tenant_context(compact: bool) -> str:
    """Return the owner-profile block for the system prompt ("" when none).

    Reads the compact file when *compact* is true, else the full one, and
    falls back to the other file when the chosen one is missing or empty.
    """
    try:
        base = Path.home() / ".captain-claw"
    except RuntimeError:  # no resolvable home directory
        return ""
    order = (COMPACT_FILENAME, FULL_FILENAME) if compact else (FULL_FILENAME, COMPACT_FILENAME)
    for name in order:
        text = _read_cached(base / name)
        if text:
            return text
    return ""


# Shared context (PR B context packs): what the agent's owner and members chose
# to share with everyone who uses the agent — composed by Flight Deck into the
# same config home, read by every instance (owner lanes and member instances)
# and inserted after the owner/member block on instances that may use packs
# (pack_access.packs_allowed).
SHARED_FULL_FILENAME = "shared_context.md"
SHARED_COMPACT_FILENAME = "shared_context.compact.md"


def load_shared_context(compact: bool) -> str:
    """Return the shared-context block for the system prompt ("" when none).

    Same home lookup, mtime cache, size guard and fallback order as
    :func:`load_tenant_context`: the compact file when *compact* is true, else
    the full one, falling back to the other when the chosen one is missing or
    empty. Flight Deck rewrites the files on every pack change and deletes them
    when nothing is shared, so a change applies on the next prompt build.
    """
    try:
        base = Path.home() / ".captain-claw"
    except RuntimeError:  # no resolvable home directory
        return ""
    order = ((SHARED_COMPACT_FILENAME, SHARED_FULL_FILENAME) if compact
             else (SHARED_FULL_FILENAME, SHARED_COMPACT_FILENAME))
    for name in order:
        text = _read_cached(base / name)
        if text:
            return text
    return ""


# Who the agent is shared with (PR D): Flight Deck composes a short roster
# block for the OWNER's instances (never inserted on a member instance) and
# deletes the file when the agent has no live member. Its presence also lists
# the ``shared_agent_usage`` tool (tools/registry.py).
SHARED_MEMBERS_FILENAME = "shared_members.md"


def load_shared_members() -> str:
    """Return the shared-members block for the owner's prompt ("" when none).

    Same home lookup, mtime cache and size guard as the other loaders."""
    try:
        base = Path.home() / ".captain-claw"
    except RuntimeError:  # no resolvable home directory
        return ""
    return _read_cached(base / SHARED_MEMBERS_FILENAME)


def use_compact_tenant_context(*, micro: bool, nano: bool) -> bool:
    """Compact block for the micro/nano templates and for orchestrated
    workers (Council/Basna/Vatra teammates, bound via ``CLAW_VFS_PROJECT``)."""
    return bool(micro or nano or os.environ.get("CLAW_VFS_PROJECT", "").strip())


def insert_tenant_block(prompt: str, block: str) -> str:
    """Insert *block* just before the first ``<!-- CACHE_SPLIT -->`` marker,
    so it sits in the cached static part of the prompt; append it when the
    template has no marker (nano)."""
    block = (block or "").strip()
    if not block:
        return prompt
    idx = prompt.find(CACHE_SPLIT_MARKER)
    head = (prompt if idx < 0 else prompt[:idx]).rstrip()
    lead = f"{head}\n\n" if head else ""
    if idx < 0:
        return f"{lead}{block}"
    return f"{lead}{block}\n\n{prompt[idx:]}"
