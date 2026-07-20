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
import math
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
# The answer we want is one line of JSON — but a thinking-mode model spends
# this budget REASONING first, and 120 tokens ran out mid-thought: staging's
# feet (deepseek-v4-flash) died this way in 14 of 16 calls, surfacing either
# cut-off deliberation or a line clipped at `{"act": "go", "to`. Room to think
# is cheap here; the context cap above is what actually bounds the spend.
FEET_MAX_TOKENS = max(120, min(2000, int(os.environ.get(
    "FEET_MAX_TOKENS", "600") or 600)))
FEET_TIER = "fast"               # the cheapest named tier (infants run on it)
FEET_IDLE_MINUTES = 45           # restlessness stirs after ~45 quiet minutes
FEET_PLAN_MINUTES = 30           # an open plan presses every half hour
FEET_TASK_MINUTES = 5            # an actionable task may interrupt every ~5 min
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

# Said again, plainly, after a reply that ran out of room to think. Once —
# twice is a habit, three times is an essay.
FEET_PLAIN_SYSTEM = FEET_SYSTEM + (
    "\n\nDo NOT reason, explain, weigh options, or restate the question. "
    "Your whole reply is the one line of JSON and nothing else.")

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
    "task": "a task on your mind's work board could be worked NOW",
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
    anchor = last_dec or being.get("hatched_at") or being.get("born_at")
    gap_min: float | None = None
    if anchor:
        try:
            gap_min = (now - datetime.fromisoformat(anchor)
                       ).total_seconds() / 60.0
        except ValueError:
            gap_min = None
    try:
        open_steps = store.open_plan_steps(being["id"], now=now)
    except Exception:  # noqa: BLE001
        open_steps = []
    # The interrupt (work-board plan): a task the feet could take up NOW may
    # stir a decision even MID-WALK — anyone can stop to seize a task —
    # rate-limited so a fresh walk is not re-decided every reflex pass. Only
    # go/build are the feet's to work; 'meet' stays world-fulfilled.
    if gap_min is not None and gap_min >= FEET_TASK_MINUTES \
            and any(t["kind"] in ("go", "build") for t in open_steps):
        return "task"
    if not loc.get("at"):
        return None                      # mid-walk, no task — the road decides
    for e in events:                     # newest first; fresh = since last
        if last_dec and e["at"] <= last_dec:
            break
        if e["kind"] == "arrived" and e["data"].get("place") != "home" \
                and not e["data"].get("planned"):
            return "arrived"
        if e["kind"] == "crossed_paths":
            return "company"
    if gap_min is None:
        return None
    if open_steps and gap_min >= FEET_PLAN_MINUTES:
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
        tasks = [t for t in store.open_plan_steps(being["id"], now=now)
                 if t["kind"] in ("go", "build")]
        if tasks:
            try:
                p = being_world.position_of(store, being, now)
                origin = (int(p["xy"][0]), int(p["xy"][1]))
            except Exception:  # noqa: BLE001
                origin = None
            rows = []
            for i, t in enumerate(tasks[:6], 1):
                lbl = being_world.task_label(store, being, t)
                mins = ""
                xy = being_world.task_target_xy(store, being, t)
                if origin and xy:
                    m = being_world.travel_minutes(being, origin, xy)
                    mins = " (here)" if m < 1 else f" ({int(round(m))} min)"
                on = " ·working" if t["state"] == "active" else ""
                rows.append(f"[t{i}] {lbl}{mins}{on}")
            lines.append("YOUR WORK BOARD — " + "; ".join(rows) + ".")
            lines.append(
                'Take one up: {"act": "do", "task": "t1"} — walk there, or '
                "if you already stand at a build task's spot, break its "
                'ground. Or refuse one: {"act": "refuse", "task": "t2", '
                '"why": "a few words"}. Or ignore the board and just move.')
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
        if act in ("take", "work"):
            act = "do"
        if act not in ("go", "linger", "hello", "browse", "home", "build",
                       "do", "refuse"):
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
        if act in ("do", "refuse"):
            # take up (or push back on) a task off the mind's work board
            task = str(obj.get("task") or "").strip()
            if not task:
                return None
            out = {"act": act, "task": task[:24]}
            if act == "refuse":
                out["why"] = str(obj.get("why") or "").strip()[:40]
            return out
        return {"act": act}
    return None


def _unparsed_why(text: str) -> str:
    """WHY the feet's line didn't land. Without this the log only ever read
    'unparsed' — true, useless, and unfixable (staging: 155 of 168 feet calls
    died here and nobody could see what the model had actually said)."""
    body = (text or "").strip()
    if not body:
        return "the model returned nothing"
    found = list(_JSON_RE.finditer(body))
    if not found:
        if body.count("{") > body.count("}"):
            return ("the line was cut off mid-JSON — the output budget ran "
                    "out before it closed")
        return "no JSON object in the reply — the model wrote prose"
    for m in found:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            return "JSON, but not an object"
        if not obj.get("act"):
            return f"JSON without an \"act\" key: {sorted(obj)[:4]}"
        act = str(obj["act"]).strip().lower()
        if act == "go":
            return 'act "go" with no "to"'
        if act in ("do", "refuse"):
            return f'act "{act}" with no "task"'
        return f'act "{act[:24]}" is not one the feet can do'
    return "a JSON object the parser could not read"


def _ran_out_of_room(text: str) -> bool:
    """True when a reply looks like a call that ran out of ROOM rather than a
    model that had nothing to say: deliberation with no JSON in it, or a line
    cut before it closed. Both are one bug wearing two faces — a thinking
    model handed a budget sized for the answer alone. Worth asking again;
    an empty reply is not (a mute tier stays mute)."""
    body = (text or "").strip()
    if not body:
        return False
    return not list(_JSON_RE.finditer(body))


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
    if kind in ("do", "refuse"):
        # Work the mind's board (work-board plan). The feet SELECT a task
        # off it, or push back with a reason — the loop's downward half.
        try:
            tasks = [t for t in store.open_plan_steps(being["id"], now=now)
                     if t["kind"] in ("go", "build")]
        except Exception:  # noqa: BLE001
            tasks = []
        task = _resolve_task(tasks, act.get("task") or "")
        if task is None:
            return {"act": "none", "task": act.get("task"),
                    "note": "no such task"}
        label = being_world.task_label(store, being, task)
        if kind == "refuse":
            why = act.get("why") or "not now"
            store.refuse_plan_step(being["id"], task["id"], why, now=now)
            return {"act": "refuse", "task": task["id"], "what": label,
                    "why": why}
        return _work_task(store, being, task, label, now)
    return {"act": "linger"}


def _resolve_task(tasks: list[dict], handle: str) -> dict | None:
    """A feet handle → a board task. `t1`..`tN` index the actionable list in
    the SAME oldest-first order the prompt showed; a raw id or a
    target/detail name also matches."""
    h = (handle or "").strip().lower()
    if not h:
        return None
    if h.startswith("t") and h[1:].isdigit():
        i = int(h[1:]) - 1
        if 0 <= i < len(tasks):
            return tasks[i]
    for t in tasks:
        if str(t["id"]).lower() == h:
            return t
    for t in tasks:
        if str(t.get("target") or "").lower() == h \
                or str(t.get("detail") or "").lower() == h:
            return t
    return None


def _work_task(store: BeingsStore, being: dict, task: dict, label: str,
               now: datetime) -> dict:
    """Take up one task. A `go` task departs toward its place (claimed →
    the arrival settle marks it done). A `build` task at/near its spot
    breaks ground THERE (done, linked to the stake the mind will finish);
    far off, the feet walk toward it (claimed) and stake on a later pass."""
    owner, slug = being["owner_id"], being["slug"]
    loc = being.get("location") or {"at": "home"}
    if task["kind"] == "go":
        pid = task["target"]
        if loc.get("at") == pid:
            store.fulfill_plan_step(being["id"], task["id"], now=now)
            return {"act": "linger", "task": task["id"], "what": label,
                    "note": "already there"}
        store.claim_plan_step(being["id"], task["id"], now=now)
        try:
            store.depart(owner, slug, pid, now=now, by="feet")
        except BeingError as e:
            return {"act": "none", "task": task["id"], "note": str(e)[:120]}
        return {"act": "go", "to": pid, "task": task["id"], "what": label}
    # A build task: near enough to break ground, or walk toward it first.
    xy = being_world.task_target_xy(store, being, task)
    try:
        pos = being_world.position_of(store, being, now)
        here = (int(pos["xy"][0]), int(pos["xy"][1]))
    except Exception:  # noqa: BLE001
        here = None
    near = xy is not None and here is not None \
        and math.dist(here, xy) <= being_world.TASK_BUILD_REACH
    if near:
        try:
            row = being_world.stake_object(store, being, task.get("detail")
                                           or "", now=now, on_task=True)
        except BeingError as e:
            # can't stake yet (a beginning already waits) — keep it claimed;
            # the mind will finish the waiting stake and free the ground.
            store.claim_plan_step(being["id"], task["id"], now=now)
            return {"act": "none", "task": task["id"], "note": str(e)[:120]}
        store.fulfill_plan_step(being["id"], task["id"], object_id=row["id"],
                                now=now)
        return {"act": "build", "kind": row["kind"], "id": row["id"],
                "task": task["id"], "what": label}
    pid = task["target"]
    store.claim_plan_step(being["id"], task["id"], now=now)
    if loc.get("at") == pid:
        return {"act": "linger", "task": task["id"], "what": label,
                "note": "at the place"}
    try:
        store.depart(owner, slug, pid, now=now, by="feet")
    except BeingError as e:
        return {"act": "none", "task": task["id"], "note": str(e)[:120]}
    return {"act": "go", "to": pid, "task": task["id"], "what": label,
            "note": "toward the build"}


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

    async def _ask(sys_text: str) -> tuple[str, dict | None]:
        if send_fn is not None:
            return await send_fn(user), None
        return await _one_shot(db, being, sys_text, user)

    try:
        text, usage = await _ask(system)
    except Exception as e:  # noqa: BLE001 — no tier, no net: feet stand still
        log.warning("feet call failed", slug=being["slug"], error=str(e))
        # Say it on the BEING's own log too, not just ours: a feet call that
        # never came back is a real thing that happened to it, and an
        # invisible one is a thing nobody fixes.
        try:
            store.record_event(being["id"], "instinct",
                               {"act": "none", "note": "the call failed",
                                "why": str(e)[:160], "trigger": trigger,
                                "tokens": 0}, now=now)
        except Exception:  # noqa: BLE001
            pass
        return None
    act = parse_feet_act(text)
    said = [(text, usage)]
    retried = False
    if act is None and _ran_out_of_room(text):
        # It thought instead of answering, and the budget ended mid-thought.
        # Ask once more without the room to deliberate — cheaper than a being
        # that stands still all day.
        retried = True
        try:
            text2, usage2 = await _ask(FEET_PLAIN_SYSTEM)
        except Exception as e:  # noqa: BLE001 — the first reply still stands
            log.warning("feet retry failed", slug=being["slug"], error=str(e))
        else:
            said.append((text2, usage2))
            act2 = parse_feet_act(text2)
            if act2 is not None:
                act, text = act2, text2
            else:
                text = text2          # log the second refusal, not the first
    applied = _apply_act(store, being, act, now=now)
    spent = 0
    for said_text, said_usage in said:
        try:
            est = said_usage if said_usage and said_usage.get("prompt_tokens") \
                else {"prompt_tokens": max(1, (len(system) + len(user))
                                           // _CHARS_PER_TOKEN),
                      "completion_tokens": max(1, len(said_text or "")
                                               // _CHARS_PER_TOKEN)}
            spent += store.debit_usage(being["id"], FEET_TIER, est,
                                       note="instinct", now=now)
        except Exception as e:  # noqa: BLE001 — the thought happened; record it
            log.warning("feet metering failed", slug=being["slug"],
                        error=str(e))
    ev = {**applied, "trigger": trigger, "tokens": spent}
    if retried:
        ev["retried"] = True          # two calls' worth of tokens, honestly
    if act is None:
        # The whole story of a wasted call: why the line failed, and the words
        # the model actually sent back (trimmed) so the cause is visible.
        ev["why"] = _unparsed_why(text)
        ev["reply"] = " ".join((text or "").split())[:200]
    store.record_event(being["id"], "instinct", ev, now=now)
    return ev
