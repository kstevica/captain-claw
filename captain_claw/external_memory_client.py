"""The agent's side of the external memory, which lives in Flight Deck.

The person connects external memories (Membase, Hindsight, Qdrant, ArcadeDB —
several at once) in Flight Deck → Connections → External memory; every agent
of theirs shares them. An agent never
holds the credentials: it asks ``/fd/external-memory/agent/*`` and Flight Deck
answers with the owner's connection.

* :func:`refresh` — once per turn (rate-limited): is a memory connected, and
  what does it offer? The answer (the *manifest*: provider, actions with their
  parameters, rules, which kinds to mirror) is cached here, so the tool
  registry can decide synchronously whether to show ``external_memory`` and the
  tool can build its schema from the provider's own actions.
* :func:`run` — one action through Flight Deck.
* :func:`recall` — what the memory holds on the message about to be answered
  (automatic recall, before every reply unless the person's reading priority
  is ``on_demand``); never raises, never waits past what Flight Deck allows.
* :func:`capture_every` / :func:`remember_cue` — how often facts are picked out
  of the chat while a memory is connected (more often than Captain's own pace,
  and right after "remember that…"), so the memory fills as the person talks.
* :func:`mirror` — copy one item Captain keeps (an insight, an intention, a
  reflection, an intuition) into the memory. Fire-and-forget: it never waits,
  never raises, and does nothing when no memory is connected, that kind is
  turned off, or the turn is a shared-agent member's (their words must not
  land in the owner's memory). Flight Deck queues it and writes it.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

MEMBER_REFUSED = (
    "The external memory belongs to this agent's owner — it isn't used on a shared-agent member's turn."
)
NOT_UNDER_FD = (
    "The external memory lives in Flight Deck — this agent isn't running under one, so it has none."
)

_MANIFEST: dict[str, Any] = {}
_MANIFEST_AT = 0.0
_CACHE_MAX_AGE = 120.0
# Flight Deck is asked at most every _REFRESH_MIN_GAP while connected; "not
# connected" is re-checked every time (a memory the user just connected must
# show on the next turn — it is a local lookup in Flight Deck).
_REFRESH_MIN_GAP = 20.0
_LAST_CHECK_AT = 0.0
_RUN_TIMEOUT = 150.0
_MIRROR_TIMEOUT = 15.0

_pending: set[asyncio.Task[Any]] = set()


class ExternalMemoryError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# ── Flight Deck plumbing (the same credentials Jev / Google use) ────


def _fd_base() -> str:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    return GoogleOAuthManager._flight_deck_base()


def under_flight_deck() -> bool:
    return bool(_fd_base())


def _member_call() -> bool:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    return GoogleOAuthManager._member_call()


def _fd_headers() -> dict[str, str]:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    # Owner-only: never the member grant — /fd/external-memory/* isn't grant-aware.
    return GoogleOAuthManager._flight_deck_headers(as_speaker=False)


def _multi_tenant_run() -> bool:
    """A public "computer" run keeps per-session (per-visitor) memory: never
    copied into the owner's external memory."""
    try:
        from captain_claw.config import get_config

        return get_config().web.public_run == "computer"
    except Exception:
        return False


def _detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:300]
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return str(detail)[:300]
    return ""


# ── the manifest ────────────────────────────────────────────────────


def _set_manifest(manifest: dict[str, Any]) -> None:
    global _MANIFEST, _MANIFEST_AT
    _MANIFEST = manifest if isinstance(manifest, dict) else {}
    _MANIFEST_AT = time.time()


def manifest() -> dict[str, Any]:
    """The last manifest Flight Deck gave (``{}`` when never asked)."""
    return dict(_MANIFEST)


def is_connected_cached(max_age: float | None = None) -> bool:
    age = time.time() - _MANIFEST_AT
    fresh = _MANIFEST_AT > 0 and age <= (_CACHE_MAX_AGE if max_age is None else max_age)
    return fresh and bool(_MANIFEST.get("connected"))


def usable() -> bool:
    """Whether the tool should be offered right now (sync, for the registry)."""
    return under_flight_deck() and is_connected_cached()


async def refresh(force: bool = False) -> bool:
    """Ask Flight Deck whether the owner has an external memory, and what it
    offers (while connected, at most every ~20 s). A member's turn reports not
    connected and leaves the owner's manifest alone; an unreachable Flight Deck
    keeps the previous answer."""
    global _LAST_CHECK_AT
    if not under_flight_deck() or _member_call():
        return False
    if not force and _MANIFEST.get("connected") and time.time() - _LAST_CHECK_AT < _REFRESH_MIN_GAP:
        _set_manifest(_MANIFEST)  # re-stamp: a long turn can't age it out
        return True
    _LAST_CHECK_AT = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_fd_base()}/fd/external-memory/agent/status", headers=_fd_headers())
    except Exception as exc:
        log.debug("Flight Deck external memory status check failed: %s", exc)
        return is_connected_cached()
    if resp.status_code != 200:
        # 404: a Flight Deck without external memory; 401/403: this agent can't
        # use its owner's connections. Either way there's nothing to use.
        _set_manifest({"connected": False})
        return False
    try:
        data = resp.json()
    except ValueError:
        data = {}
    _set_manifest(data if isinstance(data, dict) else {})
    return bool(_MANIFEST.get("connected"))


def _owner_turn() -> bool:
    return under_flight_deck() and not _member_call() and not _multi_tenant_run()


def priority() -> str:
    """The person's reading priority: ``first`` | ``alongside`` |
    ``on_demand`` — "" when no memory is connected (or not this turn's to use)."""
    if not _owner_turn() or not is_connected_cached():
        return ""
    value = str(_MANIFEST.get("priority") or "alongside")
    return value if value in ("first", "alongside", "on_demand") else "alongside"


def capture_every() -> int:
    """While a memory is connected: pick facts out of the chat every this many
    of the person's messages (0 = Captain's own pace)."""
    if not priority():
        return 0
    try:
        return max(0, int(_MANIFEST.get("capture_every") or 0))
    except (TypeError, ValueError):
        return 0


# A message that hands over something to keep — the gate for picking facts
# out of the chat right away rather than at the next interval. English and
# Croatian (the owners' languages so far). Only a gate: the extraction model
# decides what, if anything, is worth keeping.
_REMEMBER_CUES = re.compile(
    r"\b("
    r"remember|don'?t forget|do not forget|keep in mind|note that|make a note|for the record|fyi|"
    r"from now on|going forward|"
    r"my (wife|husband|partner|son|daughter|kids?|children|mother|mom|mum|father|dad|brother|sister|"
    r"boss|colleague|birthday|address|phone|number|email|name|company|team|doctor|dentist)|"
    r"i (live|work|moved|prefer|like|love|hate|am allergic|have an?|was born|started|quit|joined)|"
    r"we (moved|switched|decided|agreed|signed|hired|launched|closed)|"
    r"zapamti|upamti|zabilježi|zabiljezi|ne zaboravi|imaj na umu|za zapisnik|od sada|ubuduće|ubuduce|"
    r"moj[aeiu]? (žena|zena|supruga?|muž|muz|sin|kći|kćer|kcer|djeca|mama|majka|tata|otac|brat|sestra|"
    r"šef|sef|kolega|kolegica|rođendan|rodjendan|adresa|broj|mail|ime|firma|tvrtka|doktor|zubar)|"
    r"živim|zivim|radim u|radim kao|preselio|preselila|preferiram|volim|ne volim|alergičan|alergicna|"
    r"odlučili smo|odlucili smo|dogovorili smo|potpisali smo|zaposlili smo"
    r")\b",
    re.IGNORECASE,
)


def remember_cue(said: str) -> bool:
    """Whether *said* reads like it hands over something worth keeping."""
    text = " ".join(str(said or "").split())
    if len(text) < 6 or len(text) > 4000 or text.startswith("/"):
        return False
    return bool(_REMEMBER_CUES.search(text.replace("’", "'")))


async def recall(query: str) -> dict[str, Any]:
    """What the person's memories hold on *query*: ``{priority, label, groups,
    hits}`` (``groups``: one per memory asked, with its reading priority).
    Never raises; ``hits`` is empty when nothing bears on it, the priority is
    ``on_demand``, the memory is slow or failing, or this isn't the owner's turn."""
    prio = priority()
    out: dict[str, Any] = {"priority": prio, "label": str(_MANIFEST.get("label") or ""), "hits": []}
    query = str(query or "").strip()
    if not prio or prio == "on_demand" or not query:
        return out
    try:
        wait = float(_MANIFEST.get("recall_timeout") or 4.0) + 2.0
    except (TypeError, ValueError):
        wait = 6.0
    try:
        async with httpx.AsyncClient(timeout=wait) as client:
            resp = await client.post(f"{_fd_base()}/fd/external-memory/agent/recall",
                                     json={"query": query[:4000]}, headers=_fd_headers())
        data = resp.json() if resp.status_code == 200 else {}
    except Exception as exc:
        log.debug("external memory: recall failed: %s", type(exc).__name__)
        return out
    if not isinstance(data, dict):
        return out
    hits = [h for h in data.get("hits") or [] if isinstance(h, dict) and str(h.get("text") or "").strip()]
    # One group per memory asked (name, its reading priority, its hits) —
    # where the agent puts each depends on that memory's priority.
    groups = [g for g in data.get("groups") or [] if isinstance(g, dict)]
    out.update(hits=hits, groups=groups, label=str(data.get("label") or out["label"]),
               priority=str(data.get("priority") or prio), error=str(data.get("error") or ""))
    return out


# ── actions ─────────────────────────────────────────────────────────


async def run(action: str, args: dict[str, Any], *, timeout: float = _RUN_TIMEOUT) -> dict[str, Any]:
    """One action through Flight Deck: ``{text, data}``. Raises
    :class:`ExternalMemoryError` with Flight Deck's (or the provider's) words."""
    if _member_call():
        raise ExternalMemoryError(0, MEMBER_REFUSED)
    if not under_flight_deck():
        raise ExternalMemoryError(0, NOT_UNDER_FD)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{_fd_base()}/fd/external-memory/agent/run",
                json={"action": action, "args": args},
                headers=_fd_headers(),
            )
    except httpx.TimeoutException:
        raise ExternalMemoryError(0, "Flight Deck didn't answer in time — the memory may be slow; try again shortly") from None
    except Exception as exc:
        raise ExternalMemoryError(0, f"Could not reach Flight Deck for the external memory: {type(exc).__name__}") from None
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            raise ExternalMemoryError(200, "Flight Deck returned a non-JSON answer") from None
        return data if isinstance(data, dict) else {}
    detail = _detail(resp)
    if resp.status_code == 404 and detail in ("", "Not Found"):
        detail = "this Flight Deck has no external memory support yet — it needs an update and a restart"
    if resp.status_code in (401, 403, 404):
        _set_manifest({"connected": False})
    raise ExternalMemoryError(resp.status_code, detail or f"Flight Deck refused (HTTP {resp.status_code})")


# ── mirroring ───────────────────────────────────────────────────────


async def mirror_now(item: dict[str, Any]) -> dict[str, Any]:
    """Queue one item in Flight Deck now (awaited): ``{queued, reason}``."""
    if not under_flight_deck() or _member_call():
        return {"queued": False, "reason": "unavailable"}
    await refresh()
    if not _MANIFEST.get("connected"):
        return {"queued": False, "reason": "not_connected"}
    if item.get("kind") not in (_MANIFEST.get("mirror") or []):
        return {"queued": False, "reason": "kind_off"}
    async with httpx.AsyncClient(timeout=_MIRROR_TIMEOUT) as client:
        resp = await client.post(
            f"{_fd_base()}/fd/external-memory/agent/mirror", json=item, headers=_fd_headers(),
        )
    if resp.status_code != 200:
        return {"queued": False, "reason": f"http_{resp.status_code}", "detail": _detail(resp)}
    data = resp.json()
    return data if isinstance(data, dict) else {}


async def _mirror_quietly(item: dict[str, Any]) -> None:
    try:
        out = await mirror_now(item)
        if out.get("queued"):
            log.debug("external memory: queued", kind=item.get("kind"), source_id=item.get("source_id"))
    except Exception as exc:
        log.debug("external memory: mirror failed: %s", type(exc).__name__)


def mirror(
    kind: str, source_id: str, content: str, *, title: str = "", tags: Any = None,
    created_at: str = "", meta: dict[str, Any] | None = None,
) -> None:
    """Copy one item into the owner's external memory, in the background.
    Returns at once; never raises."""
    try:
        if not under_flight_deck() or _member_call() or _multi_tenant_run():
            return
        content = str(content or "").strip()
        source_id = str(source_id or "").strip()
        if not content or not source_id:
            return
        if isinstance(tags, str):
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        elif isinstance(tags, (list, tuple, set)):
            tag_list = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tag_list = []
        item = {
            "kind": kind, "source_id": source_id, "content": content, "title": str(title or "")[:300],
            "tags": tag_list[:20], "created_at": created_at or "",
            "meta": {k: v for k, v in (meta or {}).items() if v not in (None, "")},
        }
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(_mirror_quietly(item))
            _pending.add(task)
            task.add_done_callback(_pending.discard)
        else:
            threading.Thread(target=lambda: asyncio.run(_mirror_quietly(item)), daemon=True).start()
    except Exception as exc:
        log.debug("external memory: mirror not scheduled: %s", type(exc).__name__)
