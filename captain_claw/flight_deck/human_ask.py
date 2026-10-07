"""Bat — the human-in-the-loop ask registry (durable, pause-and-ask).

The owner chose "pause and ask" for anything Bat genuinely can't do on its own
(a 2FA/verification code, a CAPTCHA, a payment over cap, a credential, or plan
approval). This is that mechanism, built durable-first so it survives an FD
restart: an ask is a row in ``bat_asks`` and the run stands down (status
``awaiting_plan`` / ``awaiting_human``); answering the ask re-kicks the run.
There is no blocking in-process wait to lose on restart.

Secrets (2FA codes, passwords, CAPTCHA answers) must never be written to a
durable store, a notification body, or chat history. So a ``secret`` ask:
  * stores only a redacted placeholder in the row and in any notification;
  * keeps the raw answer ONLY in an in-process holder, consumed once by the
    resuming step;
  * is never fanned out to WhatsApp/chat (UI answer only) — the notifier is
    told it is secret.
If the FD restarts before the step consumes it, the secret is simply lost and
Bat re-asks — acceptable for short-lived codes, and far safer than persisting.

The notifier (how an ask reaches the human: bell, WhatsApp, the calling agent's
chat) is an injected seam so this module stays FD-light and unit-testable.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

# Called when an ask is raised, to fan it out. (ask_dict) -> None. The ask_dict
# the notifier receives already has its question redacted when secret.
Notifier = Callable[[dict], Awaitable[None]]

_NOTIFY: Notifier | None = None
# raw secret answers, keyed by ask_id, consumed once by the resuming step
_SECRETS: dict[str, str] = {}

_REDACTED = "«provided privately»"


def set_notifier(fn: Notifier | None) -> None:
    global _NOTIFY
    _NOTIFY = fn


def new_ask_id() -> str:
    return f"ask_{uuid.uuid4().hex[:12]}"


def redact(text: str, limit: int = 300) -> str:
    """For logs / non-secret surfaces: collapse whitespace and cap length. (Not
    used for secret VALUES — those never reach a log in the first place.)"""
    return " ".join(str(text or "").split())[:limit]


async def raise_ask(
    store,
    *,
    run_id: str,
    owner: str,
    kind: str,
    question: str,
    options: list | None = None,
    step_key: str = "",
    secret: bool = False,
    expires_at: float = 0.0,
) -> str:
    """Persist an ask and fan it out. Does NOT block — the caller sets the run's
    status to awaiting_* and stands down; answering re-kicks the run."""
    ask_id = new_ask_id()
    await store.create_ask(
        ask_id=ask_id, run_id=run_id, owner_id=owner, kind=kind,
        question=question, options=options or [], step_key=step_key,
        secret=secret, expires_at=expires_at,
    )
    if _NOTIFY is not None:
        payload = {
            "id": ask_id, "run_id": run_id, "owner_id": owner, "kind": kind,
            "question": (_REDACTED if secret else question), "options": options or [],
            "secret": secret,
        }
        try:
            await _NOTIFY(payload)
        except Exception as e:  # noqa: BLE001 — fan-out must never break the run
            log.warning("human_ask notify failed", ask_id=ask_id, error=str(e))
    return ask_id


async def answer(store, ask_id: str, text: str, *, via: str = "") -> dict:
    """Record an answer (compare-and-set, idempotent). For a secret ask the raw
    value is held only in memory and the row keeps a redacted placeholder.
    Returns {ok, run_id, kind} so the caller can re-kick the run."""
    ask = await store.get_ask(ask_id)
    if not ask or ask["status"] != "open":
        return {"ok": False, "reason": "ask is not open"}
    if ask["secret"]:
        _SECRETS[ask_id] = str(text)
        stored = _REDACTED
    else:
        stored = str(text)
    ok = await store.answer_ask(ask_id, stored, via=via)
    if not ok:
        _SECRETS.pop(ask_id, None)  # lost the race; don't keep the secret around
        return {"ok": False, "reason": "already answered"}
    return {"ok": True, "run_id": ask["run_id"], "kind": ask["kind"]}


def take_secret(ask_id: str) -> str | None:
    """Pop the in-memory raw secret answer (consumed once by the resuming step)."""
    return _SECRETS.pop(ask_id, None)


async def resolve_answer_text(store, ask: dict) -> str:
    """The answer a resuming step should use: the raw secret if still in memory,
    else the stored (redacted-for-secret) value."""
    if ask.get("secret"):
        return _SECRETS.pop(ask["id"], "") or ""
    return ask.get("answer", "") or ""


async def answer_for_owner(store, owner: str, text: str, *, via: str = "") -> dict:
    """Resolve a channel reply (WhatsApp/chat) that doesn't name an ask id: only
    when the owner has exactly ONE open ask, and never for a secret ask (secrets
    are UI-only). Ambiguity returns a reason instead of guessing."""
    open_asks = await store.open_asks_for_owner(owner)
    if not open_asks:
        return {"ok": False, "reason": "no open ask"}
    if len(open_asks) > 1:
        return {"ok": False, "reason": "multiple open asks — answer by id in the UI"}
    ask = open_asks[0]
    if ask["secret"]:
        return {"ok": False, "reason": "this answer must be entered in the UI, not a chat channel"}
    return await answer(store, ask["id"], text, via=via)
