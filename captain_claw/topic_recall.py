"""Topic recall — one card about an earlier conversation thread, chosen from
the turn's own words, inside the per-turn context block (context engine P3).

Abstains by default. A topic is recalled only when the word ranking and the
meaning ranking agree on it, or one of them picks it by a clear margin, and
never when the live conversation already holds its latest messages. No LLM
call: the query is the turn's content words, so the block stays frozen for
the turn. ``conversation_topics.recall`` = off | shadow | on — shadow decides
and records the decision in the context trace, but sends nothing.

Topics pinned with the ``topics`` tool ride the block for a few turns
whatever the mode: tool results are dropped from later turns, so a pin is
how a thread stays in view.
"""

from __future__ import annotations

from typing import Any

from captain_claw import msg_origin

PIN_KEY = "pinned_topics"
MAX_PIN_TURNS = 20
MAX_PINS = 3
_RECALL_EXCERPTS = 3
_PIN_EXCERPTS = 6
_SUMMARY_CHARS = 300

RECALL_LEAD = (
    "Earlier topic that may relate to this message (recalled from past conversations "
    "— NOT the current request; use it only if it fits)"
)
PIN_LEAD = "Pinned topic (kept in view with the topics tool until unpinned or expired)"


# ── Deciding ───────────────────────────────────────────────────────────


def decide(
    mgr: Any,
    text: str,
    *,
    embedder: Any = None,
    live_ids: set[str] | frozenset[str] = frozenset(),
    speaker: str | None = None,
    min_cosine: float = 0.5,
    agree_min_cosine: float = 0.35,
    cosine_margin: float = 0.15,
    bm25_margin: float = 1.3,
) -> dict[str, Any]:
    """Which topic, if any, to recall for *text*.

    Rules, in order: ``agree`` — the word and meaning rankings put the same
    topic first, it is reasonably close in meaning, and something besides a
    single shared word sets it apart (2+ matched words, or a lead in either
    ranking); ``words`` — 2+ matched words and a bm25 lead over the next
    topic; ``meaning`` — close in meaning and well ahead of the next topic.
    Ties never count as a lead.

    *speaker* None = the agent's owner; a member's id limits recall to topics
    holding that member's messages; ``""`` for a member (unverified) recalls
    nothing. Returns ``{"topic": id|None, "rule", "reason", "terms",
    "candidates"}`` — candidates as compact dicts for the trace."""
    decision: dict[str, Any] = {"topic": None, "rule": "", "reason": "", "terms": [], "candidates": []}
    if speaker == "":
        decision["reason"] = "unverified member"
        return decision
    ids = mgr.topics_with_speaker(speaker) if speaker else None
    terms, fts, vec = mgr.rank_legs(text, embedder=embedder, ids=ids)
    decision["terms"] = terms
    seen: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(fts[:3], 1):
        seen.setdefault(row["id"], _brief(row)).update(fts_rank=rank, bm25=round(float(row["bm25"]), 3))
    for rank, row in enumerate(vec[:3], 1):
        seen.setdefault(row["id"], _brief(row)).update(vec_rank=rank, cosine=row["cosine"])
    decision["candidates"] = list(seen.values())
    if not fts and not vec:
        decision["reason"] = "no match"
        return decision

    f1 = fts[0] if fts else None
    f2 = fts[1] if len(fts) > 1 else None
    v1 = vec[0] if vec else None
    words = len(f1.get("matched_terms") or []) if f1 else 0
    fts_lead = bool(f1) and (f2 is None or abs(f1["bm25"]) >= bm25_margin * abs(f2["bm25"]))
    next_cos = v1.get("next_cosine") if v1 else None
    cos_lead = bool(v1) and (next_cos is None or v1["cosine"] - next_cos >= cosine_margin)
    chosen, rule = None, ""
    if (f1 and v1 and f1["id"] == v1["id"] and v1["cosine"] >= agree_min_cosine
            and (words >= 2 or fts_lead or cos_lead)):
        chosen, rule = f1, "agree"
    elif f1 and words >= 2 and fts_lead:
        chosen, rule = f1, "words"
    elif v1 and v1["cosine"] >= min_cosine and cos_lead:
        chosen, rule = v1, "meaning"
    if chosen is None:
        decision["reason"] = "no clear match"
        return decision

    recent = mgr.get_topic(chosen["id"], max_excerpts=3,
                           speaker=speaker if speaker else "") or {}
    if any(str(m.get("msg_id") or "") in live_ids for m in recent.get("messages", [])):
        decision["reason"] = "already in this conversation"
        decision["rule"] = rule
        return decision
    decision.update(topic=chosen["id"], rule=rule, reason="recalled")
    return decision


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    out = {"id": row["id"], "label": row.get("label", "")}
    if row.get("bm25") is not None:
        out["bm25"] = round(float(row["bm25"]), 3)
    if row.get("cosine") is not None:
        out["cosine"] = row["cosine"]
    out["matched_terms"] = list(row.get("matched_terms") or [])
    return out


# ── Rendering ──────────────────────────────────────────────────────────


def render_card(topic: dict[str, Any], *, pinned: bool = False, budget_tokens: int = 0,
                own_excerpts: bool = False) -> str:
    """The card for one topic. Larger tiers get longer excerpts.
    *own_excerpts* (a member's view): dated by their own latest message, not
    by when anyone last touched the topic."""
    # Sized by the room history has: roomy tiers get longer excerpts, small
    # ones fewer and shorter (budget 0 = unknown, the middle size).
    if budget_tokens >= 64000:
        excerpt_chars, limit = 360, (_PIN_EXCERPTS if pinned else _RECALL_EXCERPTS)
    elif budget_tokens >= 16000 or budget_tokens <= 0:
        excerpt_chars, limit = 200, (_PIN_EXCERPTS if pinned else _RECALL_EXCERPTS)
    else:
        excerpt_chars, limit = 120, (2 if pinned else 1)
    messages = topic.get("messages") or []
    if own_excerpts:
        dated = max((str(m.get("ts") or "") for m in messages), default="")
    else:
        dated = str(topic.get("last_seen") or "")
    lines = [
        f"{PIN_LEAD if pinned else RECALL_LEAD}:",
        f"\"{topic.get('label', '')}\" — last discussed {dated[:10] or 'earlier'}"
        f" · id {topic.get('id', '')}",
    ]
    summary = str(topic.get("summary") or "").strip()
    if summary:
        if len(summary) > _SUMMARY_CHARS:
            summary = summary[: _SUMMARY_CHARS - 1].rstrip() + "…"
        lines.append(f"Summary: {summary}")
    for m in messages[-limit:]:
        text = " ".join(str(m.get("excerpt") or "").split())
        if len(text) > excerpt_chars:
            text = text[: excerpt_chars - 1].rstrip() + "…"
        who = "user" if m.get("role") == "user" else "you"
        lines.append(f"- [{str(m.get('ts') or '')[:10]} {who}] {text}")
    lines.append(f"More: topics action=get topic={topic.get('id', '')}")
    return "\n".join(lines)


# ── Pins (session metadata) ────────────────────────────────────────────


def pins_shown(budget_tokens: int) -> int:
    """How many pinned cards a turn carries: all on roomy tiers, the newest
    one when history has little room."""
    return MAX_PINS if budget_tokens <= 0 or budget_tokens >= 16000 else 1


def current_opener_id(session: Any) -> str:
    """The message id of the session's latest message a person typed — pins
    are spent on people's turns, not on cron or autonomy turns."""
    for msg in reversed(list(getattr(session, "messages", None) or [])):
        if msg_origin.is_human_input(msg):
            return str(msg.get("message_id") or "")
    return ""


def pin(session: Any, topic_id: str, turns: int) -> int:
    """Keep *topic_id* in the context block for the next *turns* turns.
    Returns the turns granted. The oldest pin goes when more than
    ``MAX_PINS`` are held."""
    meta = session.metadata if isinstance(getattr(session, "metadata", None), dict) else None
    if meta is None:
        session.metadata = meta = {}
    pins = dict(meta.get(PIN_KEY) or {})
    pins.pop(topic_id, None)
    granted = max(1, min(MAX_PIN_TURNS, int(turns)))
    pins[topic_id] = {"turns": granted, "opener": current_opener_id(session)}
    while len(pins) > MAX_PINS:
        pins.pop(next(iter(pins)))
    meta[PIN_KEY] = pins
    return granted


def unpin(session: Any, topic_id: str | None = None) -> list[str]:
    """Drop one pin (or all with None). Returns the ids removed."""
    meta = getattr(session, "metadata", None)
    if not isinstance(meta, dict) or not meta.get(PIN_KEY):
        return []
    pins = dict(meta[PIN_KEY])
    removed = list(pins) if topic_id is None else [t for t in pins if t == topic_id]
    for tid in removed:
        pins.pop(tid, None)
    meta[PIN_KEY] = pins
    return removed


def active_pins(session: Any) -> list[str]:
    """Pinned topic ids for the current turn; a new turn spends one of each
    pin's turns, and a pin with none left is dropped."""
    meta = getattr(session, "metadata", None)
    if not isinstance(meta, dict) or not meta.get(PIN_KEY):
        return []
    opener = current_opener_id(session)
    pins = dict(meta[PIN_KEY])
    for tid, entry in list(pins.items()):
        if not isinstance(entry, dict):
            pins.pop(tid)
            continue
        if entry.get("opener") != opener:
            if int(entry.get("turns") or 0) <= 0:
                pins.pop(tid)
                continue
            pins[tid] = {"turns": int(entry["turns"]) - 1, "opener": opener}
    meta[PIN_KEY] = pins
    return list(pins)
