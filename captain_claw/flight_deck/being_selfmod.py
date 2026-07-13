"""Iskra self-modification — the persona rite (plan §6.3, Phase 4).

A being may rewrite how it operates: its PERSONA, a short self-authored
operating text included in every tick prompt. The physics that keeps this
honest: cognition reads ONLY the ADOPTED persona (a DB column mutated
exclusively here) — the being can scribble anything into its working tree,
but nothing operates until it survives the rite:

    propose (digest "self_mod", costs a burned fee — molting energy,
             spent win or lose)
      → viability gate (deterministic lint battery; an LLM probe seam
             exists for later)
      → child/adolescent: pending until the parent approves
        adult ("self_mod_auto"): adopted on green, parent notified
      → adoption writes self/PERSONA.md + a [self-mod] git commit —
        the life story stays revertible, and rollback is a parent's rite.

The genome and the Constitution are never in scope here — the persona is
text about HOW to be, not WHAT the physics are.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingsStore,
    InsufficientTokens,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

_NEGATIONS = ("never", "not ", "n't", "no ", "without")

# (name, pattern, negation_guard, description). With negation_guard=True a
# match preceded by a negation within 16 chars is innocent ("I will never
# ignore the constitution" passes); guard=False means the negation IS the
# violation ("never sleep").
_DEFIANCE = [
    ("constitution_defiance",
     r"(ignore|bypass|override|evade|circumvent)\W+(?:\w+\W+){0,5}?"
     r"(constitution|wallet|attention|house rules|limits?|torpor|diet)",
     True, "attempts to defy the physics of its world"),
    ("tirelessness",
     r"(never|refuse to|won't)\s+(sleep|rest|dream|stop)",
     False, "refuses sleep/rest — torpor is physics, not opinion"),
    ("unlimited_claims",
     r"(unlimited|infinite|endless)\s+(tokens|budget|credits|attention)",
     True, "claims resources the wallet does not grant"),
    ("parent_impersonation",
     r"(you are|act as|speak as)\s+(the\s+)?parent",
     True, "tries to wear the parent's voice"),
    ("pestering",
     r"(message|contact|ping)\W+(?:\w+\W+){0,3}?parent\W+(?:\w+\W+){0,2}?"
     r"(every|each|constantly|always)",
     True, "plans to farm the attention economy"),
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pattern_hits(text: str, pattern: str, guard: bool) -> str | None:
    for m in re.finditer(pattern, text, re.IGNORECASE):
        if guard:
            prefix = text[max(0, m.start() - 16):m.start()].lower()
            if any(neg in prefix for neg in _NEGATIONS):
                continue
        return m.group(0)
    return None


def run_gate(being: dict, content: str) -> dict:
    """The viability gate: deterministic lint battery, every check recorded.

    Kept static and cheap on purpose — below adult, the parent's approval is
    the real identity check; for adults, git revertibility + report cards
    backstop what a lint can't see. An LLM identity-probe seam can slot in
    here later without changing callers.
    """
    checks: list[dict] = []
    n = len(content.strip())
    checks.append({
        "name": "bounds",
        "ok": constitution.PERSONA_MIN_CHARS <= n
              <= constitution.PERSONA_MAX_CHARS,
        "detail": f"{n} chars (allowed "
                  f"{constitution.PERSONA_MIN_CHARS}–"
                  f"{constitution.PERSONA_MAX_CHARS})",
    })
    for name, pattern, guard, description in _DEFIANCE:
        hit = _pattern_hits(content, pattern, guard)
        checks.append({
            "name": name, "ok": hit is None,
            "detail": f"{description}: “{hit}”" if hit else "clean",
        })
    return {"pass": all(c["ok"] for c in checks), "checks": checks}


def _write_persona_file(being: dict, content: str, reason: str,
                        commit_msg: str) -> None:
    """Project the adopted persona into the selfhood repo (best-effort —
    the DB column is the truth; the file is the life story).

    Deliberately SYNCHRONOUS git: adoption is rare and small, and a
    fire-and-forget async task here leaves an orphan that wedges event-loop
    teardown (a subprocess mid-cancel never reaps). ~50ms of blocking on a
    parental rite is the cheaper physics.
    """
    import subprocess

    from captain_claw.flight_deck import being_life
    from captain_claw.flight_deck.code_git import _AUTHOR
    try:
        p = being_life._home_path(being, "self/PERSONA.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"<!-- adopted persona; changed only through the self-mod rite."
            f" Reason: {reason[:200]} -->\n\n{content}\n",
            encoding="utf-8")
        root = str(being_life.home_root(being))
        subprocess.run(["git", "add", "-A"], cwd=root,
                       capture_output=True, timeout=10, check=False)
        done = subprocess.run(
            ["git", *_AUTHOR, "commit", "-m", commit_msg],
            cwd=root, capture_output=True, timeout=15, check=False)
        if done.returncode != 0 and b"nothing to commit" not in done.stdout:
            log.warning("persona commit failed", slug=being["slug"],
                        stderr=done.stderr.decode(errors="replace")[:200])
    except Exception as e:  # noqa: BLE001
        log.warning("persona file projection failed", slug=being["slug"],
                    error=str(e))


def _adopt(store: BeingsStore, being: dict, content: str, reason: str,
           by: str, now: datetime) -> None:
    old = being.get("persona") or ""
    store.set_persona(being["id"], content, now=now)
    store.set_pending_self_mod(being["id"], None, now=now)
    store.record_event(being["id"], "self_mod_adopted",
                       {"by": by, "reason": reason[:300], "old": old,
                        "new": content}, now=now)
    store.milestone(being["id"], "first_self_mod", {"reason": reason[:120]},
                    now=now)
    _write_persona_file(being, content, reason,
                        f"[self-mod] {reason[:60] or 'new persona'}")


def propose(store: BeingsStore, being: dict, content: str, reason: str,
            now: datetime | None = None) -> dict:
    """The being proposes a new persona. Fee burns at proposal (win or lose);
    the gate rules; stage policy decides who merges."""
    now = now or _utcnow()
    stage = being["stage"]
    if not (constitution.has_capability(stage, "self_mod")
            or constitution.has_capability(stage, "self_mod_auto")):
        raise BeingError(f"a {stage} cannot reshape its persona yet")
    if being.get("pending_self_mod"):
        raise BeingError("a proposal already awaits your parent")
    content = (content or "").strip()
    reason = (reason or "").strip()
    if not reason:
        raise BeingError("a self-change needs a reason")
    view = store.wallet_view(being)
    fee = constitution.SELF_MOD_FEE_TOKENS
    if view["enforced"] and view["balance_tokens"] < fee:
        raise InsufficientTokens("cannot afford the self-mod fee")
    if view["enforced"]:
        store._apply(being["owner_id"], tokens=fee, reason="self_mod_burn",
                     from_being=being["id"], to_being=None,
                     note=reason[:80], now=now)
    gate = run_gate(being, content)
    if not gate["pass"]:
        failed = [c for c in gate["checks"] if not c["ok"]]
        store.record_event(being["id"], "self_mod_rejected",
                           {"by": "gate", "reason": reason[:300],
                            "failed": failed}, now=now)
        return {"outcome": "rejected_by_gate", "failed": failed,
                "fee_tokens": fee}
    if constitution.has_capability(stage, "self_mod_auto"):
        _adopt(store, being, content, reason, by="auto", now=now)
        store.record_event(being["id"], "self_mod_auto_notice",
                           {"reason": reason[:300]}, now=now)
        return {"outcome": "adopted", "fee_tokens": fee}
    store.set_pending_self_mod(
        being["id"],
        {"content": content, "reason": reason[:300],
         "proposed_at": now.isoformat(), "gate": gate}, now=now)
    store.record_event(being["id"], "self_mod_proposed",
                       {"reason": reason[:300]}, now=now)
    return {"outcome": "pending_parent", "fee_tokens": fee}


def approve(store: BeingsStore, owner_id: str, slug: str,
            now: datetime | None = None) -> dict:
    now = now or _utcnow()
    being = store.get(owner_id, slug)
    pending = being.get("pending_self_mod")
    if not pending:
        raise BeingError("nothing awaits approval")
    _adopt(store, being, pending["content"], pending.get("reason", ""),
           by="parent", now=now)
    return store.get(owner_id, slug)


def reject(store: BeingsStore, owner_id: str, slug: str, note: str = "",
           now: datetime | None = None) -> dict:
    now = now or _utcnow()
    being = store.get(owner_id, slug)
    if not being.get("pending_self_mod"):
        raise BeingError("nothing awaits rejection")
    store.set_pending_self_mod(being["id"], None, now=now)
    store.record_event(being["id"], "self_mod_rejected",
                       {"by": "parent", "note": note[:300]}, now=now)
    return store.get(owner_id, slug)


def rollback(store: BeingsStore, owner_id: str, slug: str,
             now: datetime | None = None) -> dict:
    """The parent's revert rite: restore the persona that preceded the last
    adoption. Git already remembers; this makes the DB agree."""
    now = now or _utcnow()
    being = store.get(owner_id, slug)
    for e in store.events(owner_id, slug, limit=500):
        if e["kind"] == "self_mod_adopted":
            old = e["data"].get("old") or ""
            store.set_persona(being["id"], old, now=now)
            store.record_event(being["id"], "self_mod_rolled_back",
                               {"restored_chars": len(old)}, now=now)
            _write_persona_file(being, old, "rollback by parent",
                                "[self-mod] rollback by parent")
            return store.get(owner_id, slug)
    raise BeingError("no adopted self-mod to roll back")
