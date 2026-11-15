"""Secrets a person typed for an agent: a password or a one-time code.

An agent asks with ``request_secret`` (a ``secret`` question in
``person_asks``). The person types into a masked box; the answer the agent
gets is a reference (``sec_…``) and the length, never the value. The value
stays here, in memory only:

- bound to the person who typed it and the agent that asked;
- for ``TTL_S`` (10 minutes) and at most ``MAX_USES`` (3) releases — a login
  that fails on a typo can try again, a stale code can't linger;
- released only to that agent, for that person (``POST /fd/asks/agent/secret``),
  right when its browser types it into a field.

Nothing writes it to disk, the bell, the Line, logs or the tool-call register.
A Flight Deck restart forgets every secret.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

TTL_S = 600.0
MAX_USES = 3
MAX_LEN = 500
REF_PREFIX = "sec_"
MAX_KEPT = 1000


@dataclass
class _Kept:
    value: str
    user_id: str
    agent: str
    expires: float
    uses: int = 0


_SECRETS: dict[str, _Kept] = {}


def reset_for_tests() -> None:
    _SECRETS.clear()


def _sweep(now: float) -> None:
    for ref in [r for r, k in _SECRETS.items() if k.expires <= now or k.uses >= MAX_USES]:
        _SECRETS.pop(ref, None)


def keep(value: str, user_id: str, agent: str) -> str:
    """Hold *value* for *agent* acting for *user_id*; the reference to it."""
    now = time.time()
    _sweep(now)
    if len(_SECRETS) >= MAX_KEPT:  # never grows without bound: the oldest goes
        oldest = min(_SECRETS, key=lambda r: _SECRETS[r].expires)
        _SECRETS.pop(oldest, None)
    ref = REF_PREFIX + secrets.token_hex(8)
    _SECRETS[ref] = _Kept(value=value, user_id=user_id, agent=agent, expires=now + TTL_S)
    return ref


def release(ref: str, user_id: str, agent: str) -> str:
    """The value, for the agent that asked and the person who typed it.
    Raises KeyError for anyone else, or once it's used up or too old."""
    now = time.time()
    _sweep(now)
    kept = _SECRETS.get(str(ref or ""))
    if kept is None or kept.user_id != user_id or kept.agent != agent:
        raise KeyError(ref)
    kept.uses += 1
    value = kept.value
    if kept.uses >= MAX_USES:
        _SECRETS.pop(ref, None)
    return value


def forget(ref: str) -> None:
    _SECRETS.pop(str(ref or ""), None)
