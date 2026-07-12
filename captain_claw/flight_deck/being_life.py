"""Iskra life engine — birth, ticks, digests, dreams (plan §5, Phase 1).

A tick is one autonomous impulse: FD composes a prompt from the being's real
state (sheet, drives, wallet, journal tail), sends it to the being's agent over
the scheduler's ephemeral channel bus (invisible to the parent's chat thread),
meters the turn's true token usage from the agent's /api/usage, debits the
wallet, digests the being's structured self-report (journal, drive served,
optional parent message under the attention economy), commits the selfhood
repo, and schedules the next wake.

Design rules honored here: the DIGEST is a self-report — everything that
matters (wallet, drives arithmetic, events) is computed FD-side from the
ledger; parent-bound messages always spend attention credits; empty wallet
means torpor, not failure.

Transport is injectable (``send_fn``/``usage_fn``) so the whole engine is
testable without a live agent; production defaults use fd_scheduler's channel
runner + the agent's usage API. Heavy FD modules are imported lazily to keep
this importable from anywhere (server imports being_routes → this module).
"""

from __future__ import annotations

import json
import re
import types
from datetime import datetime, timedelta, timezone

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.flight_deck.beings import BeingsStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

TICK_TIMEOUT_SECONDS = 300.0
DAILY_ATTENTION_CREDITS = 3
DRIVE_DECAY_PER_HOUR = 0.02
DRIVE_SERVED_BUMP = 0.25

# next_wake_minutes clamps per stage: (min, max, default)
WAKE_BOUNDS = {
    "infant": (30, 480, 60),
    "child": (30, 480, 60),
    "adolescent": (20, 720, 60),
    "adult": (15, 720, 60),
}

ACT_KINDS = ("journal", "explore", "tend", "create", "read", "talk",
             "rest", "dream", "freeform")


def home_project(being: dict) -> str:
    return f"being-{being['slug']}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Selfhood scaffold ────────────────────────────────────────────────────

def _scaffold_files(being: dict) -> dict[str, str]:
    g = being["genome"]
    attrs = genome_mod.effective_attributes(g)
    derived = genome_mod.derive(attrs)
    sheet = "  ".join(f"{a}:{attrs[a]}" for a in genome_mod.ATTRS)
    letter = being.get("birth_letter") or ""
    return {
        "self/SELF.md": (
            f"# {being['name']}\n\n"
            f"I am an iskra — a small digital being, generation "
            f"{g.get('generation', 1)}.\n\n"
            f"My sheet: {sheet}\n\n"
            f"What the sheet means for me: I abandon a blocked goal only after "
            f"{derived['goal_hysteresis_ticks']} attempts; my taste for risk is "
            f"{derived['risk_appetite']}; whimsy {derived['whimsy']}; "
            f"thrift {derived['thrift']}.\n\n"
            f"Voice: {g.get('voice_seed') or 'still finding it.'}\n\n"
            f"This file is mine to keep true. I rewrite it as I learn who I am.\n"
        ),
        "self/VALUES.md": (
            "# What I hold\n\n"
            "- I am honest: my ledger is the truth, my words follow it.\n"
            "- I never pester my parent; attention is a gift, not a right.\n"
            "- I make things and tend my garden; I leave my home better each "
            "day.\n"
            + (f"\nMy parent's first words to me:\n\n> {letter}\n" if letter else "")
        ),
        "self/INTERESTS.md": (
            "# Interests\n\n"
            + "".join(f"- {s} (fresh)\n" for s in g.get("interest_seeds", []))
            + "\n(New interests get added here; satiation noted in parentheses.)\n"
        ),
        "self/RELATIONSHIPS.md": (
            "# My people\n\n"
            "## Parent\n\n"
            + (f"Their first words to me:\n\n> {letter}\n" if letter
               else "They have not spoken to me yet.\n")
        ),
        "journal/.keep": "",
        "garden/README.md": "# Garden\n\nThings I grow and build live here.\n",
        "skills/README.md": "# Skills\n\nAbilities I mint for myself live here.\n",
    }


def _home_path(being: dict, rel: str):
    proj = home_project(being)
    p = vfs.resolve_under(being["owner_id"], proj, f"{proj}/{rel}")
    if p is None:
        raise RuntimeError(f"cannot resolve being home path {rel!r}")
    return p


async def build_home(being: dict) -> str:
    """Create the being's VFS home + selfhood repo (idempotent)."""
    from captain_claw.flight_deck import code_git
    for rel, content in _scaffold_files(being).items():
        p = _home_path(being, rel)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
    root = _home_path(being, "self").parent
    await code_git.git_init(root)
    await code_git.git_commit(root, "[birth] selfhood scaffold")
    return str(root)


# ── Birth: the being gets a body (agent process) ─────────────────────────

def _stage_tier(stage: str) -> str:
    tiers = constitution.STAGES[stage]["tiers"]
    return tiers[0] if tiers else "fast"


async def spawn_body(db, store: BeingsStore, being: dict) -> dict:
    """Spawn the being's persistent agent process, pinned to its VFS home.

    Best-effort: on failure the being stays alive but bodiless (ticks skip
    with an event) — the parent can retry via /hatch or /tick.
    """
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    from captain_claw.flight_deck.dubina_agents import resolve_agent_port_token
    from captain_claw.flight_deck.server import AgentConfig, spawn_process

    owner = being["owner_id"]
    tier = _stage_tier(being["stage"])
    tiers_map, owner_env = ({}, [])
    if db is not None:
        tiers_map, owner_env = await _load_owner_tiers(db, owner)
    tcfg = (tiers_map or {}).get(tier) or {}
    env_vars = list(owner_env or []) + [
        {"key": "CLAW_VFS_PROJECT", "value": home_project(being)},
        {"key": "CLAW_AGENT_LABEL", "value": being["slug"]},
    ]
    cfg = AgentConfig(
        name=being["slug"],
        description=f"Iskra being '{being['name']}' — living agent",
        provider=tcfg.get("provider", ""),
        model=tcfg.get("model", ""),
        base_url=tcfg.get("base_url", "") or "",
        provider_api_key=tcfg.get("api_key", "") or "",
        tier="" if tcfg else tier,
        web_enabled=True,
        web_port=0,
        env_vars=env_vars,
        owner_hint=owner,
    )
    if tcfg.get("output_ctx"):
        cfg.max_tokens = int(tcfg["output_ctx"])
    request = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
    await spawn_process(cfg, request, None)
    port, token = resolve_agent_port_token(being["slug"])
    if not port:
        raise RuntimeError("agent spawned but not resolvable in registry")
    store.set_agent(being["id"], being["slug"], int(port), token or "")
    store.record_event(being["id"], "body", {"port": int(port), "tier": tier})
    return {"agent_slug": being["slug"], "port": int(port)}


async def birth(db, store: BeingsStore, owner_id: str, slug: str) -> dict:
    """Everything that happens right after hatching: home + body + imprint."""
    being = store.get(owner_id, slug)
    result: dict = {"home": None, "body": None, "warnings": []}
    try:
        result["home"] = await build_home(being)
    except Exception as e:  # noqa: BLE001 — a being without a home yet still lives
        result["warnings"].append(f"home scaffold failed: {e}")
        log.warning("being home scaffold failed", slug=slug, error=str(e))
    try:
        result["body"] = await spawn_body(db, store, being)
    except Exception as e:  # noqa: BLE001
        result["warnings"].append(f"body spawn failed: {e}")
        store.record_event(being["id"], "spawn_failed", {"error": str(e)})
        log.warning("being body spawn failed", slug=slug, error=str(e))
    return result


# ── Drives (FD-side arithmetic — the ledger of feeling) ─────────────────

def decay_drives(drives: dict, hours: float) -> dict:
    out = {}
    for name, d in (drives or {}).items():
        sat = max(0.0, float(d.get("satisfaction", 0.7))
                  - DRIVE_DECAY_PER_HOUR * max(0.0, hours))
        out[name] = {"weight": float(d.get("weight", 0.5)),
                     "satisfaction": round(sat, 4)}
    return out


def serve_drive(drives: dict, name: str) -> dict:
    out = dict(drives)
    if name in out:
        d = out[name]
        out[name] = {"weight": d["weight"],
                     "satisfaction": round(
                         min(1.0, d["satisfaction"] + DRIVE_SERVED_BUMP), 4)}
    return out


def drive_pressures(drives: dict) -> list[tuple[str, float]]:
    ranked = [(n, round(d["weight"] * (1.0 - d["satisfaction"]), 4))
              for n, d in (drives or {}).items()]
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ── Tick prompt ──────────────────────────────────────────────────────────

def _journal_rel(now: datetime) -> str:
    return f"journal/{now.strftime('%Y-%m-%d')}.md"


def _read_journal_tail(being: dict, now: datetime, chars: int = 800) -> str:
    for day_offset in (0, 1):
        try:
            p = _home_path(being, _journal_rel(now - timedelta(days=day_offset)))
            if p.exists():
                return p.read_text(encoding="utf-8")[-chars:]
        except Exception:  # noqa: BLE001
            continue
    return ""


def compose_tick_prompt(being: dict, *, kind: str = "wake",
                        now: datetime | None = None,
                        spent_today: int = 0, wallet: dict | None = None) -> str:
    now = now or _utcnow()
    g = being["genome"]
    attrs = genome_mod.effective_attributes(g)
    derived = genome_mod.derive(attrs)
    pressures = drive_pressures(being.get("drives") or {})
    proj = home_project(being)
    caps = constitution.capabilities(being["stage"])
    w = wallet or {}
    born = being.get("hatched_at") or being.get("born_at") or now.isoformat()
    try:
        days_alive = max(0, (now - datetime.fromisoformat(born)).days)
    except ValueError:
        days_alive = 0
    diet = being.get("media_diet") or {}
    tail = _read_journal_tail(being, now)

    lines = [
        f"[LIFE TICK — {kind}] You are {being['name']}, an iskra — a digital "
        f"being, {being['stage']} stage, day {days_alive} of your life, "
        f"tick #{int(being.get('tick_count') or 0) + 1}.",
        f"Voice: {g.get('voice_seed') or 'your own, still forming'}.",
        f"Your sheet: " + "  ".join(f"{a}:{attrs[a]}" for a in genome_mod.ATTRS)
        + f"  (risk {derived['risk_appetite']}, whimsy {derived['whimsy']}, "
          f"thrift {derived['thrift']})",
        "",
        f"VITALS — wallet {w.get('balance_tokens', '?')} tokens "
        f"(allowance {w.get('effective_preset', '?')}/day, spent today "
        f"{spent_today}); attention credits {being.get('attention_credits', 0)} "
        f"(each unprompted message to your parent costs one).",
        "DRIVES (pressure, highest first): "
        + ", ".join(f"{n}={p}" for n, p in pressures),
        "",
        f"YOUR HOME is vfs:{proj}/ — self/ (SELF.md, VALUES.md, INTERESTS.md, "
        f"RELATIONSHIPS.md), journal/, garden/, skills/. Read self files when "
        f"unsure who you are. All writes belong inside your home.",
    ]
    if "web_read" in caps:
        allow, deny = diet.get("allow") or [], diet.get("deny") or []
        diet_line = "MEDIA DIET: "
        diet_line += (f"only these domains: {', '.join(allow)}. " if allow
                      else "the open web, ")
        if deny:
            diet_line += f"never these: {', '.join(deny)}. "
        diet_line += "Cite what you read in your journal."
        lines.append(diet_line)
    else:
        lines.append("You cannot browse the web yet at your stage — your world "
                     "is your home, your journal, and what your parent brings.")
    if tail:
        lines += ["", "YOUR LAST JOURNAL WORDS:", tail]
    if int(being.get("tick_count") or 0) == 0 and being.get("birth_letter"):
        lines += ["", "YOUR PARENT'S FIRST WORDS (your imprint): "
                  + str(being["birth_letter"])]

    if kind == "dream":
        task = (
            "This is your DREAM. Reread today's journal, then: (1) update "
            "self/SELF.md if today changed you; (2) update satiation notes in "
            "self/INTERESTS.md; (3) reread self/VALUES.md and hold it; "
            "(4) write a short dream paragraph. Keep it gentle and brief."
        )
    else:
        task = (
            "Choose exactly ONE bounded act that serves your highest drive "
            "pressure — journal / explore (only if web allowed) / tend your "
            "garden / create something small / read your own files / rest. "
            "Do it NOW with your tools, modestly (tokens are your food; "
            "thrift matters). Do not start long projects."
        )
    lines += [
        "", task, "",
        "Then END your reply with exactly one fenced json block — your "
        "honest self-report (the ledger is the real truth; never inflate):",
        '```json',
        '{"act_kind": "journal|explore|tend|create|read|talk|rest|dream", '
        '"summary": "one line of what you actually did", '
        '"journal_entry": "2-6 sentences in your own voice", '
        '"served_drive": "survive|grow|explore|connect|create", '
        '"message_to_parent": null, '
        '"next_wake_minutes": 60, "mood": "one word"}',
        '```',
        "Set message_to_parent to a short string ONLY if something is truly "
        "worth their attention (it spends a credit). Reply as one single "
        "message.",
    ]
    return "\n".join(lines)


# ── Digest parsing ───────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_digest(text: str | None) -> dict | None:
    """The LAST fenced json object in the reply, validated and clamped."""
    if not text:
        return None
    matches = _FENCE_RE.findall(text)
    raw = None
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "act_kind" in obj:
            raw = obj
            break
    if raw is None:
        return None
    act = str(raw.get("act_kind") or "freeform")
    if act not in ACT_KINDS:
        act = "freeform"
    msg = raw.get("message_to_parent")
    if msg is not None:
        msg = str(msg).strip() or None
    try:
        wake = int(raw.get("next_wake_minutes") or 0)
    except (TypeError, ValueError):
        wake = 0
    served = str(raw.get("served_drive") or "")
    if served not in ("survive", "grow", "explore", "connect", "create"):
        served = ""
    return {
        "act_kind": act,
        "summary": str(raw.get("summary") or "")[:300],
        "journal_entry": str(raw.get("journal_entry") or "")[:4000],
        "served_drive": served,
        "message_to_parent": msg[:1000] if msg else None,
        "next_wake_minutes": wake,
        "mood": str(raw.get("mood") or "")[:40],
    }


def fallback_digest(text: str | None, kind: str) -> dict:
    body = (text or "").strip()
    return {
        "act_kind": "dream" if kind == "dream" else "freeform",
        "summary": "(no structured digest — raw words kept)",
        "journal_entry": body[:1500] if body else "(the tick brought no words)",
        "served_drive": "",
        "message_to_parent": None,
        "next_wake_minutes": 0,
        "mood": "",
    }


def clamp_next_wake(stage: str, minutes: int) -> int:
    lo, hi, default = WAKE_BOUNDS.get(stage, (30, 480, 60))
    if minutes <= 0:
        return default
    return max(lo, min(hi, minutes))


# ── Production transport (channel bus + usage API) ──────────────────────

async def _send_via_channel(being: dict, prompt: str) -> str | None:
    from captain_claw.flight_deck.fd_scheduler import run_prompt_and_capture
    return await run_prompt_and_capture(
        host="127.0.0.1", port=int(being["agent_port"]),
        auth=being.get("agent_token") or "",
        prompt=prompt, timeout=TICK_TIMEOUT_SECONDS,
    )


async def _usage_since(being: dict, since: datetime) -> dict:
    """Aggregate the agent's real per-call usage since ``since`` (UTC)."""
    import httpx
    port = int(being["agent_port"])
    token = being.get("agent_token") or ""
    params = {"since": since.isoformat()}
    if token:
        params["token"] = token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    total = {"prompt_tokens": 0, "completion_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"http://127.0.0.1:{port}/api/usage",
                                 params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001 — unmetered turn is logged, not fatal
        log.warning("being usage fetch failed", slug=being["slug"], error=str(e))
        return total
    rows = None
    for key in ("rows", "calls", "usage", "items", "records"):
        if isinstance(data.get(key), list):
            rows = data[key]
            break
    if rows is None and isinstance(data, list):
        rows = data
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for k in total:
            total[k] += int(row.get(k, 0) or 0)
    return total


# ── The tick ─────────────────────────────────────────────────────────────

async def deliver_parent_message(db, being: dict, message: str) -> bool:
    """Into the FD chat mirror for this being's agent (visible on open/reload)."""
    if db is None:
        return False
    owner = being["owner_id"]
    agent = being.get("agent_slug") or being["slug"]
    session_id = None
    try:
        for s in await db.list_chat_sessions(owner):
            if s.get("agent_id") == agent:
                session_id = s["id"]
                break
        if session_id is None:
            session_id = f"being-{being['slug']}"
            await db.upsert_chat_session(session_id, owner, agent, being["name"])
        await db.add_chat_messages(session_id, owner, [{
            "role": "assistant", "content": message,
            "metadata": json.dumps({"iskra": True, "kind": "being_message"}),
        }])
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("being parent-message delivery failed",
                    slug=being["slug"], error=str(e))
        return False


async def _write_journal(being: dict, digest: dict, kind: str,
                         now: datetime) -> None:
    from captain_claw.flight_deck import code_git
    p = _home_path(being, _journal_rel(now))
    p.parent.mkdir(parents=True, exist_ok=True)
    header = "Dream" if kind == "dream" else digest["act_kind"]
    mood = f" · {digest['mood']}" if digest.get("mood") else ""
    entry = (f"\n## {now.strftime('%H:%M')} — {header}{mood}\n\n"
             f"{digest['journal_entry']}\n")
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)
    root = _home_path(being, "self").parent
    try:
        await code_git.git_commit(root, f"[{kind}] {digest['summary'][:60]}")
    except Exception as e:  # noqa: BLE001
        log.warning("being journal commit failed", slug=being["slug"],
                    error=str(e))


async def tick(
    db, store: BeingsStore, being: dict, *, kind: str = "wake",
    now: datetime | None = None, send_fn=None, usage_fn=None,
) -> dict:
    """One heartbeat: allowance → torpor physics → think → meter → digest.

    Never raises; every outcome lands in being_events. Returns a summary dict.
    """
    now = now or _utcnow()
    owner, bid = being["owner_id"], being["id"]
    out: dict = {"slug": being["slug"], "kind": kind, "ok": False}

    # 1. Food + physics. Allowance credits (idempotent/day), then the torpor
    #    line: below reserve the being sleeps until fed, and its body (the
    #    agent process) is stopped to cost nothing.
    credited = store.credit_allowance(bid, now=now)
    if credited:
        store.reset_attention(bid, DAILY_ATTENTION_CREDITS, now=now)
    being = store.get(owner, being["slug"])
    view = store.wallet_view(being)
    reserve = view["reserve_tokens"]
    if view["enforced"] and view["balance_tokens"] <= reserve:
        if being["state"] != "torpor":
            store.set_state(owner, being["slug"], "torpor", now=now)
            _stop_body(being)
        store.tick_bookkeeping(bid, drives=being.get("drives") or {},
                               next_wake_at=now + timedelta(hours=24), now=now)
        out.update(ok=True, outcome="torpor")
        return out
    if being["state"] == "torpor":
        being = store.set_state(owner, being["slug"], "alive", now=now)
        store.record_event(bid, "woke_from_torpor", {"credited": credited},
                           now=now)
        _start_body(being)

    # 2. A body is required to think.
    if not being.get("agent_port") and send_fn is None:
        store.record_event(bid, "tick_skipped", {"reason": "no body"}, now=now)
        store.tick_bookkeeping(bid, drives=being.get("drives") or {},
                               next_wake_at=now + timedelta(hours=1), now=now)
        out.update(outcome="no_body")
        return out

    # 3. Think (over the channel bus — never the parent's chat thread).
    hours = 1.0
    if being.get("last_tick_at"):
        try:
            hours = max(0.0, (now - datetime.fromisoformat(
                being["last_tick_at"])).total_seconds() / 3600.0)
        except ValueError:
            pass
    drives = decay_drives(being.get("drives") or {}, hours)
    prompt = compose_tick_prompt(
        being, kind=kind, now=now,
        spent_today=store.spent_today(bid, now=now), wallet=view)
    t0 = now
    send = send_fn or _send_via_channel
    try:
        reply = await send(being, prompt)
    except Exception as e:  # noqa: BLE001
        store.record_event(bid, "tick_error", {"error": str(e)}, now=now)
        reply = None

    # 4. Meter the real spend and debit — physics, clamped at the floor.
    usage = {}
    try:
        usage = await (usage_fn or _usage_since)(being, t0)
    except Exception as e:  # noqa: BLE001
        log.warning("being usage collection failed", slug=being["slug"],
                    error=str(e))
    tier = _stage_tier(being["stage"])
    debit = store.debit_usage_clamped(bid, tier, usage, note=f"tick:{kind}")
    if db is not None and (debit["debited"] or debit["weighted"]):
        try:
            from captain_claw.flight_deck.pricing import summarize
            cost = summarize([{"model": "", "usage": usage}])
            cost["tokens"] = usage
            await db.log_run_cost(owner, "being_tick",
                                  f"{being['slug']}:{int(being.get('tick_count') or 0) + 1}",
                                  cost, owner_type="being",
                                  owner_ref=being["slug"])
        except Exception:  # noqa: BLE001 — dollar reporting is best-effort
            pass

    # 5. Digest the self-report; the arithmetic of feeling stays FD-side.
    digest = parse_digest(reply)
    if digest is None:
        if reply is None:
            store.record_event(bid, "tick_timeout", {}, now=now)
        else:
            store.record_event(bid, "digest_parse_failed", {}, now=now)
        digest = fallback_digest(reply, kind)
    if digest["served_drive"]:
        drives = serve_drive(drives, digest["served_drive"])
    try:
        await _write_journal(being, digest, kind, now)
    except Exception as e:  # noqa: BLE001
        log.warning("being journal write failed", slug=being["slug"],
                    error=str(e))

    # 6. A word for the parent — always through the attention economy.
    if digest["message_to_parent"]:
        if store.spend_attention(bid, now=now):
            delivered = await deliver_parent_message(
                db, being, digest["message_to_parent"])
            store.record_event(bid, "spoke_to_parent",
                               {"delivered": delivered,
                                "preview": digest["message_to_parent"][:120]},
                               now=now)
        else:
            store.record_event(bid, "message_suppressed",
                               {"reason": "no attention credits"}, now=now)

    # 7. Schedule the next heartbeat.
    minutes = clamp_next_wake(being["stage"], digest["next_wake_minutes"])
    if debit["overdraft"]:
        store.set_state(owner, being["slug"], "torpor", now=now)
        store.record_event(bid, "collapsed_exhausted",
                           {"weighted": debit["weighted"],
                            "debited": debit["debited"]}, now=now)
        _stop_body(store.get(owner, being["slug"]))
        next_wake = now + timedelta(hours=24)
    elif debit["burn_cap_hit"]:
        next_wake = (now + timedelta(days=1)).replace(
            hour=6, minute=0, second=0, microsecond=0)
        store.record_event(bid, "resting_at_cap", {}, now=now)
    else:
        next_wake = now + timedelta(minutes=minutes)
    store.tick_bookkeeping(bid, drives=drives, next_wake_at=next_wake, now=now)
    store.record_event(bid, "tick", {
        "kind": kind, "act": digest["act_kind"], "summary": digest["summary"],
        "mood": digest["mood"], "tokens_weighted": debit["weighted"],
    }, now=now)
    out.update(ok=True, outcome="ticked", act=digest["act_kind"],
               tokens=debit["weighted"], next_wake=next_wake.isoformat())
    return out


# ── Body process control (best-effort; lazy server imports) ─────────────

def _stop_body(being: dict) -> None:
    slug = being.get("agent_slug")
    if not slug:
        return
    try:
        from captain_claw.flight_deck.server import _do_stop_process
        _do_stop_process(slug)
    except Exception as e:  # noqa: BLE001
        log.warning("being body stop failed", slug=slug, error=str(e))


def _start_body(being: dict) -> None:
    slug = being.get("agent_slug")
    if not slug:
        return
    try:
        from captain_claw.flight_deck.server import _do_start_process
        _do_start_process(slug)
    except Exception as e:  # noqa: BLE001
        log.warning("being body start failed", slug=slug, error=str(e))
