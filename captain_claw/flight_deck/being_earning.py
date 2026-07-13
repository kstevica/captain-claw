"""Iskra earning — the quest board and ventures (plan §5.1, deferred from §7).

Two income channels beyond the chore (which is parent→one child, fixed fee):

- QUESTS: open bounties on a family board. Any adolescent+ being may claim
  one (first-come, race-safe), deliver, and be judged; approve pays, reject
  returns it to the board. origin traces provenance (parent | autonomy), so
  an approved Autonomous-Work proposal can legitimately carry a bounty.

- VENTURES: the being-INITIATED recurring service — the sanctioned outlet
  for earning creativity. The being proposes ("weekly digest of your RSS
  backlog, 0.5M/week"); the parent prices+approves; thereafter it comes due
  every cadence, the being delivers, the parent accepts, and pay recurs.

As everywhere in Iskra, the only mint is the parent's judged payment
(reason='fee'), savings-ceiling clipped — conservation holds. This module is
orchestration only (percepts + digest routing + a board view); the store
(beings.py) owns the tables and the ledger discipline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck.beings import BeingError, BeingsStore
from captain_claw.logging import get_logger

log = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Percepts (wired into the tick's senses) ──────────────────────────────

def earning_percepts(store: BeingsStore, being: dict) -> list[str]:
    lines: list[str] = []
    stage = being["stage"]
    if constitution.has_capability(stage, "jobs"):
        for q in store.quests_claimed_by(being["owner_id"], being["slug"]):
            if q["state"] == "claimed":
                lines.append(
                    f"YOUR CLAIMED QUEST [{q['id'][:8]}]: {q['title']} — "
                    f"deliver when ready ({q['fee_tokens']} tokens).")
        for q in store.open_quests(being["owner_id"], limit=3):
            lines.append(
                f"QUEST ON THE BOARD [{q['id'][:8]}]: {q['title']} — "
                f"{q['fee_tokens']} tokens. {q['spec'][:140]} "
                "Claim it only if you can truly deliver.")
    if constitution.has_capability(stage, "ventures"):
        for v in store.due_ventures_for(being["owner_id"], being["slug"]):
            lines.append(
                f"YOUR VENTURE '{v['title']}' [{v['id'][:8]}] is DUE — "
                f"deliver this cycle ({v['price_tokens']} tokens).")
    return lines


# ── Digest routing ───────────────────────────────────────────────────────

def handle_earning_digest(store: BeingsStore, being: dict, digest: dict,
                          now: datetime | None = None) -> None:
    """Route the digest's optional earning fields. Never raises: a refused
    act becomes an ``earning_refused`` event — physics denying, not crashing."""
    now = now or _utcnow()

    def _refuse(what: str, err: Exception) -> None:
        store.record_event(being["id"], "earning_refused",
                           {"what": what, "reason": str(err)}, now=now)

    claim = digest.get("claim_quest")
    if isinstance(claim, dict) and claim.get("quest_id"):
        try:
            store.claim_quest(being["owner_id"], being["slug"],
                              str(claim["quest_id"]), now=now)
        except BeingError as e:
            _refuse("claim_quest", e)

    qd = digest.get("quest_deliver")
    if isinstance(qd, dict) and qd.get("quest_id"):
        try:
            store.deliver_quest(being["owner_id"], being["slug"],
                                str(qd["quest_id"]),
                                str(qd.get("result") or ""), now=now)
        except BeingError as e:
            _refuse("quest_deliver", e)

    pv = digest.get("propose_venture")
    if isinstance(pv, dict) and pv.get("title"):
        try:
            store.propose_venture(
                being["owner_id"], being["slug"], str(pv["title"]),
                str(pv.get("description") or ""),
                int(pv.get("price_tokens") or 0),
                int(pv.get("cadence_days") or 7), now=now)
        except (BeingError, ValueError, TypeError) as e:
            _refuse("propose_venture", e)

    vd = digest.get("venture_deliver")
    if isinstance(vd, dict) and vd.get("venture_id"):
        try:
            store.deliver_venture(being["owner_id"], being["slug"],
                                  str(vd["venture_id"]),
                                  str(vd.get("result") or ""), now=now)
        except BeingError as e:
            _refuse("venture_deliver", e)


# ── Prompt fragment (the fields offered to an eligible being) ────────────

def earning_prompt_fields(being: dict) -> list[str]:
    stage = being["stage"]
    fields: list[str] = []
    if constitution.has_capability(stage, "jobs"):
        fields.append(
            '"claim_quest": {"quest_id": "<id from a board percept>"}  — '
            "claim an open bounty you can truly finish")
        fields.append(
            '"quest_deliver": {"quest_id": "<your claimed quest>", '
            '"result": "what you did"}')
    if constitution.has_capability(stage, "ventures"):
        fields.append(
            '"propose_venture": {"title": "...", "description": "the '
            'recurring service you will run", "price_tokens": 500000, '
            '"cadence_days": 7}  — invent an income stream; your parent '
            "prices and approves it")
        fields.append(
            '"venture_deliver": {"venture_id": "<a DUE venture>", '
            '"result": "this cycle\'s work"}')
    return fields


# ── Board view (the parent's observer summary) ───────────────────────────

def board_summary(store: BeingsStore, owner_id: str) -> dict:
    names = store.names_by_id(owner_id)

    def _claimant(q: dict) -> str:
        return names.get(q.get("claimed_by") or "", "") if q.get("claimed_by") \
            else ""

    quests = [
        {**q, "claimant": _claimant(q)}
        for q in store.all_quests(owner_id, limit=50)
        if q["state"] != "expired"
    ]
    ventures = [
        {**v, "being": names.get(v["being_id"], "?")}
        for v in store.list_ventures(owner_id)
        if v["state"] != "ended"
    ]
    return {"quests": quests, "ventures": ventures}
