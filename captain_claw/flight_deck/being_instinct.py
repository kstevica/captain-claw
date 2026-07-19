"""The body brain — the feet (docs/being-body-brain-plan.md Phase 2).

A second, much smaller brain per Iskra. The mind (the agent tick) keeps
everything with words, money, or weight; the feet get ONE tiny, hard-capped
LLM call when the ground actually changed — just arrived, company crossed
the path, an open plan waits, or plain restlessness — and answer with a
single micro-act: go / linger / hello / browse / home.

Position-only by design: the feet create facts and placement; every feeling
those facts earn lands at the next mind tick (percepts + PLACE_BOOST). They
never write words, never move a coin, never make a promise — a refused or
unparsed act journals quietly inside the instinct event and nags nobody.

Cost physics: no call fires unless a trigger says the world moved; the
whole context is hard-capped (FEET_CONTEXT_CAP tokens, default 1000, env-
overridable to 10k); the spend is metered against the SAME allowance (the
body eats too) via debit_usage with note="instinct" — visible on the
ledger, and the constitution invariants (reserve, daily burn cap) enforced
by the same code path as every other thought.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from captain_claw.flight_deck import being_world
from captain_claw.flight_deck.beings import BeingError, BeingsStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Hard context cap for one feet call, in (approximate) tokens — the plan's
# user knob: 1k default, 10k ceiling, never below a useful floor.
FEET_CONTEXT_CAP = max(256, min(10_000, int(os.environ.get(
    "FEET_CONTEXT_CAP", "1000") or 1000)))
_CHARS_PER_TOKEN = 4
FEET_MAX_TOKENS = 120            # one line of JSON, not an essay
FEET_TIER = "fast"               # the cheapest named tier (infants run on it)
FEET_IDLE_MINUTES = 45           # restlessness stirs after ~45 quiet minutes
FEET_PLAN_MINUTES = 30           # an open plan presses every half hour
FEET_MIN_WALLET = 50_000         # headroom above reserve before feet think

FEET_SYSTEM = (
    "You are the FEET of a small digital being — its body's reflex brain, "
    "not its mind. You move it through its village between thoughts. "
    "Choose ONE small act. Answer with ONLY one line of JSON, nothing "
    "else:\n"
    '{"act": "go", "to": "<place>"} — walk somewhere\n'
    '{"act": "linger"} — stay where you are\n'
    '{"act": "hello"} — greet the company here\n'
    '{"act": "browse"} — glance over the market stalls\n'
    '{"act": "home"} — head home\n'
    "Never words, money, or promises — those belong to the mind. Honor "
    "the mind's plan and pins when they exist; otherwise walk toward "
    "ground that serves the pressing drives.")

# When the being is restless-handed enough (impulse ≥ BUILD_IMPULSE_MIN)
# and stands on open ground, the feet gain ONE more gesture: breaking
# ground. Wordless and free — the mind gives the beginning its meaning.
FEET_BUILD_LINE = (
    '{"act": "build", "kind": "cairn|bench|signpost|planter|sculpture|'
    'lantern|fountain|shrine"} — break ground on a new thing right HERE, on '
    "impulse; your mind will name it and make it real later")

_TRIGGER_TEXT = {
    "arrived": "you just arrived here",
    "company": "someone crossed your path",
    "plan": "the mind's plan waits",
    "restless": "quiet minutes piled up — the feet itch",
    "urge_to_build": "your hands itch to make something, right here",
}

_JSON_RE = re.compile(r"\{[^{}]*\}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── The trigger: does the world ask for a decision? ──────────────────────

def wants_decision(store: BeingsStore, being: dict,
                   now: datetime) -> str | None:
    """The event-driven gate (the whole cost story lives here): a feet
    call fires only when the ground moved — mid-walk, fevered, or pinned
    home there is nothing to decide, and quiet time is rate-limited."""
    loc = being.get("location") or {"at": "home"}
    if not loc.get("at"):
        return None                      # mid-walk — the road decides
    if (being.get("intent") or {}).get("stay"):
        return None                      # the mind said stay — feet rest
    try:
        if being_world.fever_state(store, being, now):
            return None                  # fever: the reflex walks home
    except Exception:  # noqa: BLE001
        pass
    try:
        events = store.events(being["owner_id"], being["slug"], limit=30)
    except Exception:  # noqa: BLE001
        return None
    last_dec = next((e["at"] for e in events if e["kind"] == "instinct"), "")
    for e in events:                     # newest first; fresh = since last
        if last_dec and e["at"] <= last_dec:
            break
        if e["kind"] == "arrived" and e["data"].get("place") != "home" \
                and not e["data"].get("planned"):
            return "arrived"
        if e["kind"] == "crossed_paths":
            return "company"
    anchor = last_dec or being.get("hatched_at") or being.get("born_at")
    if not anchor:
        return None
    try:
        gap_min = (now - datetime.fromisoformat(anchor)
                   ).total_seconds() / 60.0
    except ValueError:
        return None
    try:
        has_plan = bool(store.open_plan_steps(being["id"], now=now))
    except Exception:  # noqa: BLE001
        has_plan = False
    if has_plan and gap_min >= FEET_PLAN_MINUTES:
        return "plan"
    # Impulse tunes the whole small brain (instinct-build plan): a restless
    # being's feet stir sooner; a deliberate one's wait longer.
    imp = being_world.impulsiveness(being)
    idle_floor = FEET_IDLE_MINUTES * (1.5 - imp)   # imp 0.8→~32, 0.2→~58 min
    if gap_min < idle_floor:
        return None
    # The urge to build: restless hands + a pressing make-drive + open
    # ground underfoot + no beginning already waiting = break ground.
    if imp >= being_world.BUILD_IMPULSE_MIN and _build_ground_ready(
            store, being, now):
        return "urge_to_build"
    return "restless"


def _build_ground_ready(store: BeingsStore, being: dict,
                        now: datetime) -> bool:
    """Would breaking ground HERE actually land? Open (non-civic) footing,
    no beginning of this being's already waiting, and a make-drive
    (create/explore) actually pressing — else the feet just wander."""
    try:
        if being_world.staked_object_of(store, being) is not None:
            return False
        from captain_claw.flight_deck import being_life
        ranked = dict(being_life.drive_pressures(being.get("drives") or {},
                                                 now))
        if max(ranked.get("create", 0.0), ranked.get("explore", 0.0)) < 0.35:
            return False
        pos = being_world.position_of(store, being, now)
        being_world.object_spot(store, being, int(pos["xy"][0]),
                                int(pos["xy"][1]), asked=False)  # raises if none
        return True
    except Exception:  # noqa: BLE001 — no ground, no urge
        return False


# ── The micro-prompt (hard-capped) ────────────────────────────────────────

def feet_prompt(store: BeingsStore, being: dict, now: datetime,
                trigger: str) -> tuple[str, str]:
    """≤FEET_CONTEXT_CAP tokens, whole context: one identity line, the top
    drive pressures, the ground with walk times, company, the mind's plan
    and pins, the stir, the last few acts. No history, no files, no
    relationships — that context belongs to the mind."""
    from captain_claw.flight_deck import being_genome as genome_mod
    from captain_claw.flight_deck import being_life
    owner = being["owner_id"]
    lines: list[str] = []
    nature = ""
    try:
        attrs = genome_mod.effective_attributes(being["genome"])
        top = sorted(attrs.items(), key=lambda x: -float(x[1]))[:2]
        nature = ", ".join(f"{k} {float(v):.1f}" for k, v in top)
    except Exception:  # noqa: BLE001
        pass
    lines.append(f"{being['name']}, {being['stage']}."
                 + (f" Nature: {nature}." if nature else ""))
    try:
        ranked = being_life.drive_pressures(being.get("drives") or {},
                                            now)[:3]
        if ranked:
            lines.append("Pressing: " + ", ".join(
                f"{n} {p:.2f}" for n, p in ranked))
    except Exception:  # noqa: BLE001
        pass
    pid, here, present = None, "", []
    try:
        pid, here, present = being_world._co_present(store, being, now)
    except Exception:  # noqa: BLE001
        pass
    at = (being.get("location") or {}).get("at") or "the road"
    lines.append(f"You stand at {here if pid else at}.")
    try:
        pos = being_world.position_of(store, being, now)
        origin = [int(pos["xy"][0]), int(pos["xy"][1])]
        ground = []
        for p in store.village_places(owner):
            if p["id"] == (pid or at):
                continue
            mins = being_world.travel_minutes(being, origin,
                                              (p["x"], p["y"]))
            ground.append(f"{p['name']} — "
                          f"{', '.join(p['affordances'])} "
                          f"({int(round(mins))} min)")
        if ground:
            lines.append("Ground: " + "; ".join(ground[:8]) + "; or home.")
    except Exception:  # noqa: BLE001
        pass
    if present:
        lines.append("Company here: "
                     + ", ".join(x["name"] for x in present[:4]))
    try:
        steps = store.open_plan_steps(being["id"], now=now)
        if steps:
            lines.append("The mind's plan: " + "; ".join(
                f"{s['kind']} {s['target']}" for s in steps))
    except Exception:  # noqa: BLE001
        pass
    avoid = (being.get("intent") or {}).get("avoid") or []
    if avoid:
        lines.append("Pins: avoid " + ", ".join(avoid))
    # The extra gesture, offered only to restless hands on open ground.
    try:
        if being_world.impulsiveness(being) >= being_world.BUILD_IMPULSE_MIN \
                and _build_ground_ready(store, being, now):
            lines.append("You may also: " + FEET_BUILD_LINE)
    except Exception:  # noqa: BLE001
        pass
    lines.append("Why you stir: " + _TRIGGER_TEXT.get(trigger, trigger))
    recent: list[str] = []
    try:
        for e in store.events(owner, being["slug"], limit=12):
            if e["kind"] == "instinct":
                d = e["data"]
                recent.append(str(d.get("act"))
                              + (f"→{d['to']}" if d.get("to") else ""))
            if len(recent) >= 3:
                break
    except Exception:  # noqa: BLE001
        pass
    if recent:
        lines.append("Your last acts: " + " · ".join(recent))
    lines.append("One line of JSON now.")
    user = "\n".join(lines)
    budget = FEET_CONTEXT_CAP * _CHARS_PER_TOKEN - len(FEET_SYSTEM)
    if len(user) > budget:
        user = user[:max(0, budget - 1)] + "…"
    return FEET_SYSTEM, user


# ── Parse + apply: the verb whitelist ─────────────────────────────────────

def parse_feet_act(text: str) -> dict | None:
    """The first JSON object with a whitelisted act wins; anything else is
    None — the feet stand still rather than improvise."""
    for m in _JSON_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not obj.get("act"):
            continue
        act = str(obj["act"]).strip().lower()
        if act in ("attend", "go_to", "walk"):
            act = "go"
        if act not in ("go", "linger", "hello", "browse", "home", "build"):
            return None
        if act == "go":
            to = str(obj.get("to") or "").strip()
            if not to:
                return None
            return {"act": "go", "to": to[:60]}
        if act == "build":
            # kind is a bodily choice; the physics floor decides if it lands
            return {"act": "build",
                    "kind": str(obj.get("kind") or "").strip().lower()[:20]}
        return {"act": act}
    return None


def _apply_act(store: BeingsStore, being: dict, act: dict | None,
               now: datetime) -> dict:
    """Position-only effects. Every refusal stays INSIDE the instinct
    event (a note, not a society_refused) — feet junk never nags the
    mind the way a mind's own refused act must."""
    owner, slug = being["owner_id"], being["slug"]
    loc = being.get("location") or {"at": "home"}
    if act is None:
        return {"act": "none", "note": "unparsed"}
    kind = act["act"]
    if kind == "home":
        if loc.get("at") == "home":
            return {"act": "linger", "note": "already home"}
        try:
            store.depart(owner, slug, "home", now=now, by="feet")
            return {"act": "go", "to": "home"}
        except BeingError as e:
            return {"act": "none", "note": str(e)[:120]}
    if kind == "go":
        pid = store.resolve_place_ref(owner, act["to"])
        if pid is None:
            return {"act": "none", "to": act["to"], "note": "no such place"}
        avoid = [(store.resolve_place_ref(owner, a) or a)
                 for a in (being.get("intent") or {}).get("avoid") or []]
        if pid in avoid:
            return {"act": "none", "to": pid, "note": "the mind said avoid"}
        if loc.get("at") == pid:
            return {"act": "linger", "note": "already here"}
        try:
            store.depart(owner, slug, pid, now=now, by="feet")
            return {"act": "go", "to": pid}
        except BeingError as e:
            return {"act": "none", "to": pid, "note": str(e)[:120]}
    if kind == "browse":
        try:
            titles = [x["title"] for x in
                      store.market_listings(owner, limit=3)]
        except Exception:  # noqa: BLE001
            titles = []
        store.record_event(being["id"], "browsed", {"titles": titles},
                           now=now)
        return {"act": "browse", "stalls": len(titles)}
    if kind == "hello":
        try:
            _, _, present = being_world._co_present(store, being, now)
        except Exception:  # noqa: BLE001
            present = []
        if not present:
            return {"act": "linger", "note": "nobody here"}
        being_world.reflex_encounters(store, being, now)   # idempotent
        return {"act": "hello", "with": [p["name"] for p in present[:3]]}
    if kind == "build":
        # Break ground — wordless, free. Physics gates the impulse floor,
        # one-at-a-time, and civic ground; a refusal stays a quiet note (the
        # feet never nag the mind). The MIND finishes it into a real thing.
        try:
            row = being_world.stake_object(store, being, act.get("kind") or "",
                                           now=now)
            return {"act": "build", "kind": row["kind"], "id": row["id"]}
        except BeingError as e:
            return {"act": "none", "note": str(e)[:120]}
    return {"act": "linger"}


# ── The one-shot call (the architect pattern) ─────────────────────────────

async def _one_shot(db, being: dict, system: str,
                    user: str) -> tuple[str, dict | None]:
    if db is None:
        raise BeingError("no FD database — the feet cannot think")
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    tiers, _env = await _load_owner_tiers(db, being["owner_id"])
    cfg = (tiers or {}).get(FEET_TIER) or (tiers or {}).get("balanced") \
        or next(iter((tiers or {}).values()), None)
    if not cfg:
        raise BeingError("no LLM tier configured — the feet cannot think")
    from captain_claw.llm import Message, create_provider
    provider = create_provider(
        provider=cfg.get("provider", "anthropic"), model=cfg.get("model", ""),
        base_url=cfg.get("base_url") or None,
        api_key=cfg.get("api_key") or None,
        temperature=0.7, max_tokens=FEET_MAX_TOKENS)
    resp = await provider.complete(
        messages=[Message(role="system", content=system),
                  Message(role="user", content=user)],
        temperature=0.7, max_tokens=FEET_MAX_TOKENS)
    usage = getattr(resp, "usage", None)
    return resp.content or "", dict(usage) if usage else None


# ── The decision ──────────────────────────────────────────────────────────

async def decide(db, store: BeingsStore, being: dict,
                 now: datetime | None = None,
                 send_fn=None) -> dict | None:
    """One feet decision, end to end: trigger → wallet guard → capped
    micro-prompt → one-shot call → whitelisted act → metered spend →
    one honest `instinct` event. Returns what happened, or None when
    the feet had no reason (or no means) to think."""
    now = now or _utcnow()
    trigger = wants_decision(store, being, now)
    if trigger is None:
        return None
    view = store.wallet_view(being)
    if view["enforced"] and (view["balance_tokens"] - view["reserve_tokens"]
                             < FEET_MIN_WALLET):
        return None                      # too hungry to spend on walking
    system, user = feet_prompt(store, being, now, trigger)
    try:
        if send_fn is not None:
            text, usage = await send_fn(user), None
        else:
            text, usage = await _one_shot(db, being, system, user)
    except Exception as e:  # noqa: BLE001 — no tier, no net: feet stand still
        log.warning("feet call failed", slug=being["slug"], error=str(e))
        return None
    applied = _apply_act(store, being, parse_feet_act(text), now=now)
    spent = 0
    try:
        est = usage if usage and usage.get("prompt_tokens") else {
            "prompt_tokens": max(1, (len(system) + len(user))
                                 // _CHARS_PER_TOKEN),
            "completion_tokens": max(1, len(text or "")
                                     // _CHARS_PER_TOKEN)}
        spent = store.debit_usage(being["id"], FEET_TIER, est,
                                  note="instinct", now=now)
    except Exception as e:  # noqa: BLE001 — the thought happened; record it
        log.warning("feet metering failed", slug=being["slug"], error=str(e))
    store.record_event(being["id"], "instinct",
                       {**applied, "trigger": trigger, "tokens": spent},
                       now=now)
    return {**applied, "trigger": trigger, "tokens": spent}
