"""Telling the person when their background model stops working.

The deck does a lot of small work on the person's own models with nobody
watching: File library and Data descriptions, an agent's skill cards, task
triage, profile research, custom tool reviews, First Mate, email auto-replies.
When the model they run on breaks — a key refused, a model gone, an account
out of credit, a server that stopped answering — those features just stop:
files keep no description, the cards fall back to plain ones, triage stops,
and nobody says why. Days can go by before someone notices.

Each background call says how it went (:func:`ok` / :func:`failed`), per
person and per model (provider, model, address). The rules:

* **a setup problem, at once** — the key refused, the model not there, no
  credit left: it won't fix itself, so the first failure tells the person;
* **anything else, after** :data:`RUNTIME_LIMIT` **in a row** — a timeout, an
  error on the provider's side, a server that didn't answer, an empty reply —
  so one bad moment of the network doesn't bother anyone;
* **any call that gets through clears it.** A call that was cancelled counts
  for nothing either way.

Telling means one item on the person's Line, under Needs you (not to their
phones): the model, why in plain words (never the provider's own error text,
which can carry an address or part of a key), and what waits on it. The item
is marked answered by itself once the model works again; a later breakage is a
new item. All of it lives in memory: a restart starts clean, and the first
calls after it say again what's wrong.

Only calls that name their feature are counted: a call the person waits on
shows its error to them already.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

RUNTIME_LIMIT = 3
REF_TYPE = "side_call_health"
CONFIG = "setup"
RUNTIME = "runtime"

# What a background call is for, in the person's words (the Line item lists them).
DATA_DESCRIPTIONS = "Data library descriptions"
FILE_DESCRIPTIONS = "File library descriptions"
SKILL_CARDS = "agents' skill cards"
TASK_TRIAGE = "task triage"
PROFILE_RESEARCH = "profile research"
TOOL_REVIEWS = "custom tool reviews"
FIRST_MATE = "First Mate"
EMAIL_AUTOREPLY = "email auto-replies"

_AUTH = re.compile(r"authenticat|unauthori[sz]ed|invalid.{0,20}(?:api.?)?key|incorrect.{0,20}key|api.?key.{0,30}"
                   r"(?:not valid|invalid|missing|required)|missing authentication|no auth credentials|"
                   r"permission.?denied|forbidden|access denied", re.I)
_CREDIT = re.compile(r"insufficient.?(?:quota|credit|balance|funds)|credit balance|out of credits?|billing|"
                     r"payment required|exceeded your current quota|quota exceeded|spend limit", re.I)
_GONE = re.compile(r"model.{0,40}(?:not found|does not exist|doesn't exist|not available|unknown|"
                   r"not supported)|no such model|model_not_found|unknown model|invalid model", re.I)
_SLOW = re.compile(r"timed? ?out|timeout", re.I)
_NO_ANSWER = re.compile(r"connect(?:ion)? (?:error|refused|reset)|cannot connect|could not connect|"
                        r"name or service not known|nodename nor servname|unreachable|getaddrinfo|"
                        r"remote.?protocol|server disconnected", re.I)


def _statuses(exc: BaseException) -> list[int]:
    from captain_claw.exceptions import _chain

    out = []
    for e in _chain(exc):
        for attr in ("status_code", "status"):
            value = getattr(e, attr, None)
            if isinstance(value, int):
                out.append(value)
        code = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(code, int):
            out.append(code)
    return out


def classify(exc: BaseException | None) -> tuple[str, str]:
    """``(kind, why)`` for a failed call — *kind* :data:`CONFIG` (it won't
    fix itself) or :data:`RUNTIME` — and *why* in plain words. ``None`` is a
    call that came back empty."""
    if exc is None:
        return RUNTIME, "it answered with nothing (a thinking model can use up all the room it's given)"
    from captain_claw.exceptions import _chain, is_busy

    codes = set(_statuses(exc))
    text = " ".join(f"{type(e).__name__}: {e}" for e in _chain(exc))
    if 402 in codes or _CREDIT.search(text):
        return CONFIG, "the provider says the account is out of credit"
    if codes & {401, 403} or _AUTH.search(text):
        return CONFIG, "the provider refused the key (it's wrong, expired or missing)"
    if 404 in codes or _GONE.search(text):
        return CONFIG, "the provider doesn't have that model (renamed or retired?)"
    if is_busy(exc):
        return RUNTIME, "the provider keeps turning calls away (busy or rate-limited)"
    if any(isinstance(e, (asyncio.TimeoutError, TimeoutError)) for e in _chain(exc)) or _SLOW.search(text):
        return RUNTIME, "it takes too long to answer"
    if any(c >= 500 for c in codes):
        return RUNTIME, "the provider answers with errors"
    if _NO_ANSWER.search(text) or any(isinstance(e, ConnectionError) for e in _chain(exc)):
        return RUNTIME, "its server doesn't answer"
    return RUNTIME, "its calls keep failing"


@dataclass
class _State:
    owner: str
    provider: str
    model: str
    failures: int = 0                    # in a row
    kind: str = ""
    why: str = ""
    since: float = 0.0                   # when this run of failures began
    last: float = 0.0
    features: dict[str, float] = field(default_factory=dict)   # what waits on it → when it last failed
    told: str = ""                       # the Line item's ref_id, once the person was told


_STATES: dict[tuple[str, str, str, str], _State] = {}


def _owner(owner: str | None) -> str:
    """The Line's own id for a person (auth off: ``local``)."""
    return str(owner or "").strip() or "local"


def _key(owner: str, provider: str, model: str, base_url: str) -> tuple[str, str, str, str]:
    return (_owner(owner), str(provider or ""), str(model or ""), str(base_url or "").rstrip("/"))


def _ref(key: tuple[str, str, str, str], since: float) -> str:
    """The Line item's ref: the model, when this run of failures began, and a
    random part (two runs can begin within one second)."""
    digest = hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:16]
    return f"{digest}:{int(since)}:{secrets.token_hex(4)}"


def _model_words(st: _State) -> str:
    return f"{st.model} ({st.provider})" if st.provider else st.model or "its model"


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


async def _tell(st: _State) -> None:
    from captain_claw.flight_deck import line

    waiting = sorted(st.features, key=lambda f: -st.features[f])
    body = (f"{st.why[0].upper()}{st.why[1:]}. Waiting on it: {_list(waiting)}. Fix the model or its key in "
            "Console → Models (or ask whoever set up this deck's models). This clears by itself once a call "
            "gets through.")
    await line.post(st.owner, "deck", f"Background work can't use {_model_words(st)}", body,
                    bell_type="warning", source="side_calls", needs_you=True, ref_type=REF_TYPE,
                    ref_id=st.told, to_phones=False)


async def _clear(st: _State) -> None:
    if not st.told:
        return
    try:
        from captain_claw.flight_deck import line
        from captain_claw.flight_deck.auth import get_db

        await get_db().resolve_line_items(line._norm(st.owner), REF_TYPE, [st.told])  # st.owner is _owner()'d
    except Exception as exc:  # noqa: BLE001 — the person reads it as answered anyway
        log.debug("side calls: notice not resolved", error=type(exc).__name__)


async def failed(owner: str, feature: str, provider: str, model: str, base_url: str = "",
                 error: BaseException | None = None) -> None:
    """A background call for *feature* failed on this model (*error* None: it
    came back empty). Never raises; a cancellation isn't a failure."""
    if not feature or isinstance(error, asyncio.CancelledError):
        return
    try:
        kind, why = classify(error)
        key = _key(owner, provider, model, base_url)
        now = time.time()
        st = _STATES.get(key)
        if st is None:
            st = _STATES[key] = _State(owner=key[0], provider=key[1], model=key[2])
        st.failures += 1
        st.since = st.since or now
        st.last = now
        st.kind, st.why = kind, why
        st.features[feature] = now
        if not st.told and (kind == CONFIG or st.failures >= RUNTIME_LIMIT):
            st.told = _ref(key, st.since)   # before the write: a failure meanwhile doesn't tell twice
            await _tell(st)
            log.warning("side calls: background model not working", provider=provider, model=model, kind=kind,
                        failures=st.failures)
    except Exception as exc:  # noqa: BLE001 — telling must never break the call it watches
        log.debug("side calls: failure not noted", error=type(exc).__name__)


async def ok(owner: str, feature: str, provider: str, model: str, base_url: str = "") -> None:
    """A background call got through on this model: what was wrong with it is
    over (and its Line item answered). Never raises."""
    if not feature:
        return
    try:
        st = _STATES.pop(_key(owner, provider, model, base_url), None)
        if st is not None:
            await _clear(st)
    except Exception as exc:  # noqa: BLE001
        log.debug("side calls: success not noted", error=type(exc).__name__)


def problems(owner: str) -> list[dict[str, Any]]:
    """*owner*'s background models that aren't working now (the person was
    told about each), newest first."""
    out = []
    for st in _STATES.values():
        if st.owner != _owner(owner) or not st.told:
            continue
        out.append({"provider": st.provider, "model": st.model, "kind": st.kind, "why": st.why,
                    "failures": st.failures, "since": st.since, "last": st.last,
                    "waiting": sorted(st.features, key=lambda f: -st.features[f])})
    return sorted(out, key=lambda p: -p["last"])


def reset_for_tests() -> None:
    _STATES.clear()
