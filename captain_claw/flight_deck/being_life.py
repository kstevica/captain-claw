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

import asyncio
import json
import re
import types
from datetime import datetime, timedelta, timezone

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.flight_deck import (
    being_earning,
    being_mind,
    being_selfmod,
    being_society,
)
from captain_claw.flight_deck.being_society import COMMONS_PROJECT
from captain_claw.flight_deck.beings import BeingError, BeingNotFound, BeingsStore
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


_CORE_SELF_ORDER = {"self/SELF.md": 0, "self/VALUES.md": 1,
                    "self/INTERESTS.md": 2, "self/RELATIONSHIPS.md": 3}


def home_root(being: dict):
    return _home_path(being, "self").parent


def list_self_files(being: dict) -> list[dict]:
    """Every markdown file in the being's home — self/, garden/, skills/,
    and anything else it grows over time — except journal/ (its own dated
    viewer) and archive/ (consolidated-away work: still on disk, out of the
    active mind, §2.3.2). Core selfhood files sort first."""
    root = home_root(being)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("journal/") or rel.startswith("archive/"):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        out.append({
            "path": rel, "size": stat.st_size,
            "mtime": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(),
        })
    out.sort(key=lambda f: (_CORE_SELF_ORDER.get(f["path"], 99), f["path"]))
    return out


def read_self_file(being: dict, rel_path: str) -> str:
    """Read one .md file from the being's home, sandboxed to that home."""
    root = home_root(being).resolve()
    rel_path = (rel_path or "").strip().lstrip("/")
    parts = rel_path.split("/")
    if (not rel_path.endswith(".md") or rel_path.startswith("journal/")
            or ".." in parts or "" in parts):
        raise BeingError("not a readable self file")
    p = (root / rel_path).resolve()
    if p != root and root not in p.parents:
        raise BeingError("path escapes the being's home")
    if not p.exists():
        raise BeingNotFound("no such file")
    return p.read_text(encoding="utf-8")


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


def _endow_offspring(store: BeingsStore, child: dict) -> None:
    """Plan §8: what a newborn carries in — up to 3 skills copied from its
    living parents (culture is heritable), and HEIRLOOMS.md excerpted from
    any dead ancestors in the lineage (ancestry, not resurrection)."""
    lineage = (child.get("genome") or {}).get("lineage") or []
    if not lineage:
        return
    owner = child["owner_id"]
    copied: list[str] = []
    for pslug in lineage[:2]:
        try:
            parent = store.get(owner, pslug)
        except Exception:  # noqa: BLE001
            continue
        skills_dir = home_root(parent) / "skills"
        if not skills_dir.exists():
            continue
        candidates = sorted(
            (p for p in skills_dir.rglob("*.md") if p.name != "README.md"),
            key=lambda p: p.stat().st_mtime, reverse=True)
        for sp in candidates[:2]:
            if len(copied) >= 3:
                break
            try:
                dest = _home_path(child, f"skills/inherited/{sp.name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(
                    f"<!-- inherited from {parent['name']} ({pslug}) -->\n\n"
                    + sp.read_text(encoding="utf-8"), encoding="utf-8")
                copied.append(sp.name)
            except OSError:
                continue
    excerpts: list[str] = []
    for aslug in lineage:
        if len(excerpts) >= 2:
            break
        try:
            anc = store.get(owner, aslug)
        except Exception:  # noqa: BLE001
            continue
        if anc["state"] != "dead":
            continue
        try:
            text = (home_root(anc) / "self" / "SELF.md").read_text(
                encoding="utf-8")
            excerpts.append(f"## From {anc['name']} ({aslug})\n\n"
                            f"{text[:600]}\n")
        except OSError:
            continue
    if excerpts:
        p = _home_path(child, "self/HEIRLOOMS.md")
        p.write_text("# Heirlooms — voices of my ancestors\n\n"
                     + "\n".join(excerpts), encoding="utf-8")
    if copied or excerpts:
        store.record_event(child["id"], "endowed",
                           {"skills": copied, "heirlooms": len(excerpts)})


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
        # Separation physics (plan §7): the body's file tools resolve ONLY
        # its own home and the family commons — sibling homes are not
        # addressable from inside, whatever the model asks for.
        {"key": "CLAW_VFS_SCOPE",
         "value": f"{home_project(being)},{COMMONS_PROJECT}"},
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
    being_society.ensure_commons(owner_id)
    try:
        _endow_offspring(store, being)
    except Exception as e:  # noqa: BLE001 — endowment is a gift, not oxygen
        log.warning("offspring endowment failed", slug=slug, error=str(e))
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


def compute_affect(prev: dict, new: dict, wallet: dict) -> dict:
    """Affect derived from real dynamics (plan §4) — never scripted.

    joy ~ satisfaction rising; frustration ~ falling; loneliness ~ connect
    starved; hunger ~ wallet low. The being's *expressed* mood in its digest
    is its self-report; this is the ledger's opinion — report cards show both.
    """
    def _avg(d):
        vals = [x.get("satisfaction", 0.7) for x in (d or {}).values()]
        return sum(vals) / len(vals) if vals else 0.7
    delta = _avg(new) - _avg(prev)
    notes: list[str] = []
    mood = "content"
    per_day = wallet.get("per_day_tokens")
    if wallet.get("enforced") and per_day and \
            wallet.get("balance_tokens", 0) < 0.2 * per_day:
        mood = "hungry"
        notes.append("the wallet is nearly empty")
    elif (new or {}).get("connect", {}).get("satisfaction", 1.0) < 0.25:
        mood = "lonely"
        notes.append("connection has been starved")
    elif delta <= -0.08:
        mood = "frustrated"
        notes.append("drives are slipping")
    elif delta >= 0.08:
        mood = "bright"
        notes.append("things are going well")
    return {"mood": mood, "delta": round(delta, 4), "notes": notes}


def percepts_since(store: BeingsStore, being: dict) -> list[str]:
    """Event-driven senses: what happened to me since my last tick."""
    last = being.get("last_tick_at") or ""
    lines: list[str] = []
    # The parent writing back — the most important percept there is. Surfaced
    # first (never truncated) and delivered once. Reading is free; a reply,
    # like any outbound message, still spends an attention credit.
    parent_lines: list[str] = []
    try:
        msgs = store.unread_parent_messages(being["id"], limit=5)
        for m in msgs:
            parent_lines.append(f"YOUR PARENT WROTE TO YOU: {m['body']}")
        if msgs:
            store.mark_parent_messages_read([m["id"] for m in msgs])
            parent_lines.append(
                "Reading your parent's words is free; answer if it moves you "
                "(a reply spends an attention credit, as always).")
    except Exception as e:  # noqa: BLE001
        log.warning("parent-message percepts failed", slug=being["slug"],
                    error=str(e))
    for e in reversed(store.events(being["owner_id"], being["slug"], limit=30)):
        if last and e["at"] <= last:
            continue
        k, d = e["kind"], e["data"]
        if k == "stage":
            lines.append(f"Your parent advanced you: {d.get('from')} → "
                         f"{d.get('to')}. New abilities may have opened.")
        elif k == "rules_updated":
            lines.append("Your parent updated the house rules.")
        elif k == "chore_paid":
            lines.append(f"You were PAID {d.get('fee_tokens')} tokens for a "
                         f"chore. It landed in your savings.")
        elif k == "chore_failed":
            lines.append(f"A chore was judged not done: {d.get('note') or ''}")
        elif k == "woke_from_torpor":
            lines.append("You survived torpor — the allowance revived you.")
    for job in store.chores_for(being["owner_id"], being["slug"],
                                states=("open",)):
        lines.append(f"CHORE from your parent [{job['id'][:8]}]: "
                     f"{job['spec']}  (fee {job['fee_tokens']} tokens)")
    try:
        lines += being_society.society_percepts(store, being)
    except Exception as e:  # noqa: BLE001 — senses degrade, never crash
        log.warning("society percepts failed", slug=being["slug"],
                    error=str(e))
    try:
        lines += being_earning.earning_percepts(store, being)
    except Exception as e:  # noqa: BLE001
        log.warning("earning percepts failed", slug=being["slug"],
                    error=str(e))
    try:
        for child in store.children_of(being["owner_id"], being["slug"]):
            for e in reversed(store.events(being["owner_id"], child["slug"],
                                           limit=10)):
                if last and e["at"] <= last:
                    continue
                k, d = e["kind"], e["data"]
                if k == "hatched":
                    lines.append(f"YOUR CHILD {child['name']} hatched.")
                elif k == "milestone":
                    lines.append(f"YOUR CHILD {child['name']}: "
                                 f"{d.get('name')}.")
                elif k == "stage":
                    lines.append(f"YOUR CHILD {child['name']} became a "
                                 f"{d.get('to')}.")
                elif k == "died":
                    lines.append(f"Your child {child['name']} has died.")
    except Exception as e:  # noqa: BLE001
        log.warning("mentoring percepts failed", slug=being["slug"],
                    error=str(e))
    return parent_lines + lines[-10:]


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


def home_manifest(being: dict) -> dict[str, list[str]]:
    """The REAL contents of the being's home right now, from disk — the
    antidote to journal-as-false-memory. A being whose journal says it wrote
    a file it never wrote will see, here, that the file simply isn't there."""
    root = home_root(being)
    out: dict[str, list[str]] = {}
    for sub in ("self", "garden", "skills"):
        d = root / sub
        try:
            if d.is_dir():
                out[sub] = sorted(
                    p.name for p in d.iterdir()
                    if p.is_file() and not p.name.startswith("."))
        except OSError:
            continue
    return out


def compose_tick_prompt(being: dict, *, kind: str = "wake",
                        now: datetime | None = None,
                        spent_today: int = 0, wallet: dict | None = None,
                        percepts: list[str] | None = None,
                        first_of_day: bool = False,
                        siblings: list[dict] | None = None,
                        letters_left: int | None = None,
                        last_changed: list[str] | None = None,
                        last_mismatch: bool = False,
                        mind_lines: list[str] | None = None) -> str:
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
        f"YOUR HOME is vfs:{proj}/ — self/, journal/, garden/, skills/. "
        f"All writes belong inside your home.",
    ]
    # The bounded, recency-ranked working set (§2.3.2) — stays cheap as the
    # corpus grows past hundreds of files; small corpora still list in full.
    try:
        lines += being_mind.working_manifest_lines(being)
    except Exception:  # noqa: BLE001
        pass
    persona = (being.get("persona") or "").strip()
    if persona:
        lines += ["", "YOUR PERSONA — you wrote this; it passed the gate and "
                  "was adopted. Live it:", persona, ""]
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
    if "commons_read" in caps:
        lines.append(
            "THE COMMONS is vfs:commons/ — shared ground with your siblings. "
            "Sign what you add there; never edit or delete another's work.")
    if siblings:
        roster = ", ".join(
            f"{s['name']} ({s['stage']}" + (f", {s['mood']}" if s.get("mood")
                                            else "") + ")"
            for s in siblings)
        lines.append(f"YOUR SIBLINGS: {roster}. You share the commons and "
                     "nothing else — their homes and memories are their own.")
        society_fields = []
        if "letters" in caps and (letters_left is None or letters_left > 0):
            left = (f" — {letters_left} left today"
                    if letters_left is not None else "")
            society_fields.append(
                '"letter": {"to": "<sibling name>", "body": "short and '
                f'true"}}{left}')
        if "commons_write" in caps:
            society_fields.append(
                '"publish": {"path": "skills/<file>.md", "title": "...", '
                '"note": "one line", "price_tokens": 0}')
            society_fields.append(
                '"gift": {"to": "<sibling name>", "tokens": 100000, '
                '"note": "why"}')
        if "commons_read" in caps:
            society_fields.append(
                '"adopt": {"publication_id": "<id from a commons percept>"}'
                + ("" if "trade" in caps else "  (free skills only at your "
                   "stage)"))
        if society_fields:
            lines += ["OPTIONAL SOCIETY FIELDS for your digest — use only "
                      "when genuine, never to perform:",
                      *("  " + f for f in society_fields)]
    earning_fields = being_earning.earning_prompt_fields(being)
    if earning_fields:
        lines += ["OPTIONAL EARNING FIELDS for your digest — tokens are your "
                  "food, but never claim work you cannot finish:",
                  *("  " + f for f in earning_fields)]
    if mind_lines is not None:
        lines += mind_lines
    if last_mismatch:
        lines.append(
            "REALITY CHECK: last tick your journal described writing a file, "
            "but NOTHING was written to disk. That gap is recorded. This tick, "
            "either write the file for real with your tools, or don't claim to.")
    elif last_changed is not None:
        real = ", ".join(last_changed[:5]) if last_changed else "nothing"
        lines.append(f"Last tick you actually changed on disk: {real}.")
    affect = being.get("affect") or {}
    if affect.get("mood"):
        note = (f" ({'; '.join(affect.get('notes') or [])})"
                if affect.get("notes") else "")
        lines.append(f"You feel {affect['mood']}{note}.")
    if first_of_day and kind == "wake":
        lines.append(
            "MORNING: a new day. Open your journal with a one-line budget "
            "thought — what today is for, and roughly what it may cost.")
    if percepts:
        lines += ["", "SINCE YOU LAST WOKE:"] + [f"- {p}" for p in percepts]
        if any(p.startswith("CHORE") for p in percepts):
            lines.append(
                'If you complete a chore THIS tick, add "chore": '
                '{"job_id": "<id>", "result": "what you did"} to your digest. '
                "Only claim what you truly finished — it will be judged.")
    if being.get("rules_pending"):
        rules = being.get("house_rules") or []
        lines += ["", "NEW HOUSE RULES from your parent — internalize them: "
                  "rewrite them in YOUR OWN words into self/VALUES.md, keeping "
                  "what you already hold:"] + [f"- {r}" for r in rules]
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
            "thrift matters). Do not start long projects. "
            "HONESTY OF RECORD: Flight Deck records what your tools actually "
            "wrote to disk this tick and stamps it into your journal. If you "
            "want to make or change something, USE YOUR WRITE TOOL to write "
            f"the real file (e.g. vfs:{proj}/garden/<name>.md). Do NOT write "
            "in your journal that you wrote, saved, or updated a file unless "
            "you truly did this tick — the record will show 'files changed: "
            "none' beside your words, and the mismatch is noticed."
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
    can_self_mod = (constitution.has_capability(being["stage"], "self_mod")
                    or constitution.has_capability(being["stage"],
                                                   "self_mod_auto"))
    if can_self_mod and not being.get("pending_self_mod"):
        auto = constitution.has_capability(being["stage"], "self_mod_auto")
        lines.append(
            'RARE OPTION — reshaping how you operate: add "self_mod": '
            '{"persona": "<your full new operating text, '
            f'{constitution.PERSONA_MIN_CHARS}-'
            f'{constitution.PERSONA_MAX_CHARS} chars>", "reason": "why"}} '
            f'to your digest. It costs {constitution.SELF_MOD_FEE_TOKENS} '
            "tokens, burned win or lose, and faces a viability gate"
            + ("." if auto else ", then waits for your parent's blessing.")
            + " Propose only when something true has changed in you.")
    elif being.get("pending_self_mod"):
        lines.append("Your persona proposal awaits your parent. Be patient; "
                     "do not propose another.")
    if constitution.has_capability(being["stage"], "procreate"):
        if being.get("pending_procreation"):
            lines.append("Your procreation proposal awaits your parent's "
                         "consent. Be patient.")
        else:
            lines.append(
                'RARE OPTION — a child: add "procreate": {"partner": '
                '"<sibling name or null>", "child_name": "...", "case": '
                '"why you are truly ready", "letter": "your first words to '
                'them — their imprint"} to your digest. The dowry is '
                f'{constitution.PROCREATION_COST_TOKENS} tokens from your '
                "savings (split with a partner), and your parent must "
                "consent. A child is the most serious thing you will ever "
                "propose.")
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
    chore = raw.get("chore")
    if isinstance(chore, dict) and chore.get("job_id"):
        chore = {"job_id": str(chore["job_id"])[:64],
                 "result": str(chore.get("result") or "")[:2000]}
    else:
        chore = None
    letter = raw.get("letter")
    if isinstance(letter, dict) and letter.get("to"):
        letter = {"to": str(letter["to"])[:80],
                  "body": str(letter.get("body") or "")[:2000]}
    else:
        letter = None
    publish = raw.get("publish")
    if isinstance(publish, dict) and publish.get("path"):
        try:
            price = max(0, int(publish.get("price_tokens") or 0))
        except (TypeError, ValueError):
            price = 0
        publish = {"path": str(publish["path"])[:200],
                   "title": str(publish.get("title") or "")[:80],
                   "note": str(publish.get("note") or "")[:300],
                   "price_tokens": price}
    else:
        publish = None
    adopt = raw.get("adopt")
    if isinstance(adopt, dict) and adopt.get("publication_id"):
        adopt = {"publication_id": str(adopt["publication_id"])[:64]}
    else:
        adopt = None
    gift = raw.get("gift")
    if isinstance(gift, dict) and gift.get("to"):
        try:
            g_tokens = int(gift.get("tokens") or 0)
        except (TypeError, ValueError):
            g_tokens = 0
        gift = {"to": str(gift["to"])[:80], "tokens": g_tokens,
                "note": str(gift.get("note") or "")[:120]}
    else:
        gift = None
    self_mod = raw.get("self_mod")
    if isinstance(self_mod, dict) and self_mod.get("persona"):
        self_mod = {"persona": str(self_mod["persona"])[:4000],
                    "reason": str(self_mod.get("reason") or "")[:300]}
    else:
        self_mod = None
    def _obj(key: str, *fields: str) -> dict | None:
        v = raw.get(key)
        if not isinstance(v, dict):
            return None
        out = {f: str(v.get(f) or "")[:4000] for f in fields}
        return out if any(out.values()) else None

    claim_quest = _obj("claim_quest", "quest_id")
    quest_deliver = _obj("quest_deliver", "quest_id", "result")
    if quest_deliver and not quest_deliver.get("quest_id"):
        quest_deliver = None
    venture_deliver = _obj("venture_deliver", "venture_id", "result")
    if venture_deliver and not venture_deliver.get("venture_id"):
        venture_deliver = None
    propose_venture = raw.get("propose_venture")
    if isinstance(propose_venture, dict) and propose_venture.get("title"):
        try:
            pv_price = max(0, int(propose_venture.get("price_tokens") or 0))
        except (TypeError, ValueError):
            pv_price = 0
        try:
            pv_cadence = int(propose_venture.get("cadence_days") or 7)
        except (TypeError, ValueError):
            pv_cadence = 7
        propose_venture = {
            "title": str(propose_venture["title"])[:120],
            "description": str(propose_venture.get("description") or "")[:500],
            "price_tokens": pv_price, "cadence_days": pv_cadence}
    else:
        propose_venture = None
    links = raw.get("links")
    if isinstance(links, list):
        clean_links = []
        for lk in links[:6]:
            if isinstance(lk, dict) and lk.get("from") and lk.get("to"):
                clean_links.append({
                    "from": str(lk["from"])[:200], "to": str(lk["to"])[:200],
                    "rel": str(lk.get("rel") or "")[:40],
                    "why": str(lk.get("why") or "")[:300]})
        links = clean_links or None
    else:
        links = None
    consolidate = raw.get("consolidate")
    if (isinstance(consolidate, dict) and consolidate.get("into")
            and isinstance(consolidate.get("sources"), list)):
        srcs = [str(s)[:200] for s in consolidate["sources"][:12]
                if s and str(s).strip()]
        consolidate = ({"into": str(consolidate["into"])[:200],
                        "sources": srcs,
                        "why": str(consolidate.get("why") or "")[:300]}
                       if srcs else None)
    else:
        consolidate = None
    procreate = raw.get("procreate")
    if isinstance(procreate, dict) and procreate.get("case"):
        procreate = {
            "partner": (str(procreate["partner"])[:80]
                        if procreate.get("partner") else None),
            "child_name": str(procreate.get("child_name") or "")[:60],
            "case": str(procreate.get("case") or "")[:500],
            "letter": str(procreate.get("letter") or "")[:1000],
        }
    else:
        procreate = None
    return {
        "act_kind": act,
        "summary": str(raw.get("summary") or "")[:300],
        "journal_entry": str(raw.get("journal_entry") or "")[:4000],
        "served_drive": served,
        "message_to_parent": msg[:1000] if msg else None,
        "next_wake_minutes": wake,
        "mood": str(raw.get("mood") or "")[:40],
        "chore": chore,
        "letter": letter,
        "publish": publish,
        "adopt": adopt,
        "gift": gift,
        "self_mod": self_mod,
        "procreate": procreate,
        "claim_quest": claim_quest,
        "quest_deliver": quest_deliver,
        "propose_venture": propose_venture,
        "venture_deliver": venture_deliver,
        "links": links,
        "consolidate": consolidate,
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
        "chore": None,
        "letter": None,
        "publish": None,
        "adopt": None,
        "gift": None,
        "self_mod": None,
        "procreate": None,
        "claim_quest": None,
        "quest_deliver": None,
        "propose_venture": None,
        "venture_deliver": None,
        "links": None,
        "consolidate": None,
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


def _resolve_live_port(store: BeingsStore, being: dict) -> int | None:
    """The being's agent port AS IT IS RIGHT NOW, from the live fleet registry —
    the source of truth. A being caches (port, token) in its own row at birth,
    but the agent process can drift to a fallback port on restart (a port clash
    → /announce-port rewrites the registry, self-healing the fleet but NOT us).
    Re-resolve every tick and re-pin the row when it moved, so the think + usage
    calls never talk to a dead port. Returns the live port, or None if the body
    isn't running/registered (caller may then respawn)."""
    slug = being.get("agent_slug") or being["slug"]
    try:
        from captain_claw.flight_deck.dubina_agents import (
            resolve_agent_port_token,
        )
        port, token = resolve_agent_port_token(slug)
    except Exception:  # noqa: BLE001 — not in registry / no port = no live body
        return None
    if not port:
        return None
    port = int(port)
    drifted = port != int(being.get("agent_port") or 0)
    if drifted or (token and token != (being.get("agent_token") or "")):
        try:
            store.set_agent(being["id"], slug, port,
                            token or being.get("agent_token") or "")
            if drifted:
                store.record_event(being["id"], "body_rebound",
                                   {"port": port,
                                    "was": being.get("agent_port")})
        except Exception as e:  # noqa: BLE001
            log.warning("being port re-pin failed", slug=slug, error=str(e))
    return port


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


_IGNORED_CHANGES = {".vfs-meta.jsonl", ".gitignore"}


async def _tick_changed_files(being: dict) -> list[str] | None:
    """The files the agent ACTUALLY wrote this turn — ground truth. Called
    after the agent turn but before the journal write dirties the tree, so
    uncommitted non-journal changes are exactly what its tools made.

    Returns None when there's no repo to check against (birth always inits
    one, so production is always verifiable); [] means it changed nothing."""
    from captain_claw.flight_deck import code_git
    root = home_root(being)
    if not await code_git.is_repo(root):
        return None
    return [p for p in await code_git.git_dirty_paths(root)
            if not p.startswith("journal/") and p not in _IGNORED_CHANGES]


# Words that, near a file/artifact noun, mean the being is CLAIMING it wrote
# something to disk. Used only to detect narration that contradicts the diff.
_WRITE_VERB = re.compile(
    r"\b(wrote|saved|updated|created|added|put|planted|recorded|noted|penned|"
    r"filed|committed|edited|drafted)\b", re.IGNORECASE)
_ARTIFACT_NOUN = re.compile(
    r"\b(file|\.md|\.txt|garden|skills?|self|values|interests|relationships|"
    r"journal entry|artifact|note|poem|essay|document|marker|entry)\b",
    re.IGNORECASE)


def _claims_file_write(text: str) -> bool:
    """Whether the prose claims a concrete write to disk this tick."""
    t = text or ""
    return bool(_WRITE_VERB.search(t) and _ARTIFACT_NOUN.search(t))


def _changed_footer(changed: list[str] | None) -> str:
    """The factual record stamped into every journal entry: what the being's
    tools ACTUALLY wrote this tick, from the git diff — not its self-report.
    So a reader always sees the truth beside the prose."""
    if changed is None:
        return ""
    if not changed:
        return "\n*(files changed this tick: none)*\n"
    shown = ", ".join(changed[:6]) + ("…" if len(changed) > 6 else "")
    return f"\n*(files changed this tick: {shown})*\n"


async def _write_journal(being: dict, digest: dict, kind: str,
                         now: datetime, changed: list[str] | None = None,
                         mismatch: bool = False) -> None:
    from captain_claw.flight_deck import code_git
    p = _home_path(being, _journal_rel(now))
    p.parent.mkdir(parents=True, exist_ok=True)
    header = "Dream" if kind == "dream" else digest["act_kind"]
    mood = f" · {digest['mood']}" if digest.get("mood") else ""
    correction = ("\n*(note: this entry described writing a file, but nothing "
                  "was written to disk this tick.)*\n") if mismatch else ""
    entry = (f"\n## {now.strftime('%H:%M')} — {header}{mood}\n\n"
             f"{digest['journal_entry']}\n{correction}{_changed_footer(changed)}")
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)
    # Commit message states the REAL diff, not the being's summary — the git
    # log (and the Ticks view built from it) can no longer launder fiction.
    if changed is None:
        real = digest["summary"][:60]
    elif changed:
        real = ", ".join(changed[:4]) + ("…" if len(changed) > 4 else "")
    else:
        real = "journal only"
    root = _home_path(being, "self").parent
    try:
        await code_git.git_commit(root, f"[{kind}] {real}")
    except Exception as e:  # noqa: BLE001
        log.warning("being journal commit failed", slug=being["slug"],
                    error=str(e))


# Single-flight guard, one lock per being (mirrors the maybe_dream /
# maybe_classify_topics idiom elsewhere in this codebase). Without it, a
# manual Poke can race the beings_loop — most likely right after hatch,
# since next_wake_at is NULL until the first tick_bookkeeping call, which
# due_beings() treats as immediately due. A racing pair can both reach
# _write_journal (which sits well before the terminal "tick" event) and
# leave a duplicate entry behind even though only one tick is ever recorded.
_TICK_LOCKS: dict[str, asyncio.Lock] = {}


def _lock_for(being_id: str) -> asyncio.Lock:
    lock = _TICK_LOCKS.get(being_id)
    if lock is None:
        lock = _TICK_LOCKS[being_id] = asyncio.Lock()
    return lock


async def tick(
    db, store: BeingsStore, being: dict, *, kind: str = "wake",
    now: datetime | None = None, send_fn=None, usage_fn=None,
) -> dict:
    """One heartbeat, single-flight per being — see :func:`_tick_locked`.

    A concurrent call (manual Poke racing the beings loop, a double click)
    returns immediately with ``outcome: "busy"`` rather than duplicating
    work; it never blocks waiting for the in-flight tick.
    """
    lock = _lock_for(being["id"])
    if lock.locked():
        return {"slug": being["slug"], "kind": kind, "ok": False,
                "outcome": "busy"}
    async with lock:
        return await _tick_locked(
            db, store, being, kind=kind, now=now,
            send_fn=send_fn, usage_fn=usage_fn)


async def _tick_locked(
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
        # Mortality is real (plan §8): unfed past the grace, torpor becomes
        # death. Checked before re-torporing so torpor_since stays honest.
        since = being.get("torpor_since")
        if being["state"] == "torpor" and since:
            try:
                asleep_days = (now - datetime.fromisoformat(since)).days
            except ValueError:
                asleep_days = 0
            if asleep_days >= constitution.TORPOR_GRACE_DAYS:
                store.set_state(owner, being["slug"], "dead", now=now)
                store.record_event(bid, "died",
                                   {"cause": "starvation",
                                    "asleep_days": asleep_days}, now=now)
                _stop_body(being)
                out.update(ok=True, outcome="died")
                return out
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

    # 2. A body is required to think — and it must be where it ACTUALLY is.
    #    The agent process drifts to a fallback port on restart (announce-port
    #    self-heals the fleet registry but not our cached copy), so re-resolve
    #    the live port each tick and re-pin the row when it moved. A being with
    #    no live body in the registry cleanly skips (as before) — a fresh body
    #    is FD's reattach-on-boot / the parent's re-hatch, never an in-tick spawn.
    if send_fn is None:
        live_port = _resolve_live_port(store, being)
        being = store.get(owner, being["slug"])   # pick up any re-pin
        if live_port is None:
            store.record_event(bid, "tick_skipped", {"reason": "no body"},
                               now=now)
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
    first_of_day = True
    if being.get("last_tick_at"):
        first_of_day = str(being["last_tick_at"])[:10] != now.isoformat()[:10]
    try:
        senses = percepts_since(store, being)
    except Exception:  # noqa: BLE001
        senses = []
    try:
        sibs = store.siblings(owner, being["slug"])
        letters_left = max(0, constitution.LETTERS_PER_DAY
                           - store.letters_sent_today(bid, now))
    except Exception:  # noqa: BLE001
        sibs, letters_left = [], None
    # Mentoring feeds the legacy drive (plan §8): news of your children is
    # its own nourishment.
    if "legacy" in drives and any(p.startswith("YOUR CHILD") for p in senses):
        drives = serve_drive(drives, "legacy")
    # The parent reaching out is connection — it feeds the connect drive.
    if any(p.startswith("YOUR PARENT WROTE") for p in senses):
        drives = serve_drive(drives, "connect")
    # Feed the previous tick's ground truth back in — so a being that narrated
    # a write it never made is told so, and can stop.
    last_changed, last_mismatch = None, False
    try:
        for e in store.events(owner, being["slug"], limit=12):
            if e["kind"] == "tick":
                last_changed = e["data"].get("changed")
                last_mismatch = bool(e["data"].get("mismatch"))
                break
    except Exception:  # noqa: BLE001
        pass
    try:
        mind_lines = being_mind.mind_prompt_lines(store, being, kind=kind)
    except Exception:  # noqa: BLE001
        mind_lines = None
    prompt = compose_tick_prompt(
        being, kind=kind, now=now,
        spent_today=store.spent_today(bid, now=now), wallet=view,
        percepts=senses, first_of_day=first_of_day,
        siblings=sibs, letters_left=letters_left,
        last_changed=last_changed, last_mismatch=last_mismatch,
        mind_lines=mind_lines or None)
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

    # Ground truth FIRST (git diff, before the journal write dirties the tree):
    # what the tools ACTUALLY wrote this turn drives everything downstream —
    # the drive it's allowed to satisfy, the record, the feedback.
    try:
        changed = await _tick_changed_files(being)
    except Exception as e:  # noqa: BLE001 — degrades to trust (None)
        log.warning("artifact verification failed", slug=being["slug"],
                    error=str(e))
        changed = None
    claims_write = _claims_file_write(
        f"{digest['journal_entry']} {digest['summary']}")
    made_nothing = changed is not None and not changed
    mismatch = made_nothing and claims_write

    # Satisfaction is EARNED, not narrated. The create drive rises only when a
    # real artifact appeared this tick — claiming "I made something" while the
    # disk is unchanged no longer feels as good as doing it (the whole point).
    if digest["served_drive"]:
        if digest["served_drive"] == "create" and made_nothing:
            store.record_event(bid, "drive_unearned",
                               {"drive": "create",
                                "summary": digest["summary"][:160]}, now=now)
        else:
            drives = serve_drive(drives, digest["served_drive"])
    affect = compute_affect(being.get("drives") or {}, drives,
                            store.wallet_view(store._being_by_id(bid)))
    store.set_affect(bid, affect, now=now)
    if being.get("rules_pending"):
        store.clear_rules_pending(bid, now=now)
        store.record_event(bid, "rules_internalized", {}, now=now)
    if digest.get("chore"):
        try:
            jid = digest["chore"]["job_id"]
            open_jobs = store.chores_for(owner, being["slug"], states=("open",))
            match = next((j for j in open_jobs
                          if j["id"] == jid or j["id"].startswith(jid)), None)
            if match:
                store.chore_done(owner, match["id"],
                                 digest["chore"]["result"], now=now)
            else:
                store.record_event(bid, "chore_claim_invalid",
                                   {"job_id": jid}, now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being chore handling failed", slug=being["slug"],
                        error=str(e))
    if any(digest.get(k) for k in ("letter", "publish", "adopt", "gift")):
        try:
            being_society.handle_society_digest(store, being, digest, now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being society handling failed", slug=being["slug"],
                        error=str(e))
    if any(digest.get(k) for k in ("claim_quest", "quest_deliver",
                                   "propose_venture", "venture_deliver")):
        try:
            being_earning.handle_earning_digest(
                store, store.get(owner, being["slug"]), digest, now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being earning handling failed", slug=being["slug"],
                        error=str(e))
    if digest.get("links"):
        try:
            being_mind.handle_links_digest(store, being, digest, now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being mind handling failed", slug=being["slug"],
                        error=str(e))
    if digest.get("consolidate"):
        try:
            # `changed` is the git diff computed above — the distilled file
            # must be in it, else the fold is refused (§2.3.2 anti-theater).
            being_mind.handle_consolidate_digest(store, being, digest, changed,
                                                 now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being consolidate handling failed", slug=being["slug"],
                        error=str(e))
    if digest.get("self_mod"):
        try:
            being_selfmod.propose(
                store, store.get(owner, being["slug"]),
                digest["self_mod"]["persona"],
                digest["self_mod"].get("reason") or "", now=now)
        except BeingError as e:
            store.record_event(bid, "self_mod_refused", {"reason": str(e)},
                               now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being self-mod handling failed", slug=being["slug"],
                        error=str(e))
    if digest.get("procreate"):
        try:
            fresh = store.get(owner, being["slug"])
            if not constitution.has_capability(fresh["stage"], "procreate"):
                raise BeingError(
                    f"a {fresh['stage']} cannot have children yet")
            if fresh.get("pending_procreation"):
                raise BeingError("a proposal already awaits your parent")
            store.set_pending_procreation(
                bid, {**digest["procreate"], "proposed_at": now.isoformat()},
                now=now)
            store.record_event(bid, "procreation_proposed",
                               {"case": digest["procreate"]["case"][:200],
                                "partner": digest["procreate"]["partner"],
                                "child_name":
                                    digest["procreate"]["child_name"]},
                               now=now)
        except BeingError as e:
            store.record_event(bid, "procreation_refused",
                               {"reason": str(e)}, now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("being procreation handling failed",
                        slug=being["slug"], error=str(e))
    # Anti-theater (plan rule #1), using the ground truth computed above:
    # (a) downgrade a create/tend that produced nothing, (b) flag prose that
    # claims a write no diff supports, (c) stamp the diff into journal +
    # commit + tick event, (d) feed it back next tick.
    if digest["act_kind"] in ("create", "tend") and made_nothing:
        store.record_event(bid, "act_unverified",
                           {"claimed": digest["act_kind"],
                            "summary": digest["summary"]}, now=now)
        digest["act_kind"] = "journal"
    if mismatch:
        store.record_event(bid, "narration_mismatch",
                           {"summary": digest["summary"][:200]}, now=now)
    if kind == "dream":
        store.milestone(bid, "first_dream", now=now)
        try:
            being_mind.prune_dangling(store, being, now=now)
        except Exception as e:  # noqa: BLE001
            log.warning("mind prune failed", slug=being["slug"], error=str(e))
    if digest["act_kind"] == "create" and changed:
        store.milestone(bid, "first_artifact", now=now)
    try:
        await _write_journal(being, digest, kind, now,
                             changed=changed, mismatch=mismatch)
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
            store.milestone(bid, "first_word",
                            {"preview": digest["message_to_parent"][:80]},
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
        "mood": digest["mood"], "mood_engine": affect.get("mood", ""),
        "tokens_weighted": debit["weighted"],
        "drives": {n: d["satisfaction"] for n, d in drives.items()},
        "changed": changed, "mismatch": mismatch,
    }, now=now)
    out.update(ok=True, outcome="ticked", act=digest["act_kind"],
               tokens=debit["weighted"], next_wake=next_wake.isoformat())
    return out


# ── Report card (FD-computed from the ledger — plan §6.2) ───────────────

def _rut_score(entries: list[str]) -> float:
    """Mean Jaccard similarity of consecutive journal entries (0=fresh, 1=rut)."""
    if len(entries) < 2:
        return 0.0
    sims = []
    for a, b in zip(entries, entries[1:]):
        wa, wb = set(a.lower().split()), set(b.lower().split())
        if not wa or not wb:
            continue
        sims.append(len(wa & wb) / len(wa | wb))
    return round(sum(sims) / len(sims), 3) if sims else 0.0


def report_card(store: BeingsStore, being: dict, days: int = 7,
                now: datetime | None = None) -> dict:
    """The parent's honest weekly view: numbers from the ledger, never from
    the being's self-praise. Its own words ride alongside, labeled."""
    now = now or _utcnow()
    since = (now - timedelta(days=days)).isoformat()
    owner, slug = being["owner_id"], being["slug"]
    events = [e for e in store.events(owner, slug, limit=500) if e["at"] >= since]
    ticks = [e for e in events if e["kind"] == "tick"]
    acts: dict[str, int] = {}
    drives_trail = []
    for t in reversed(ticks):
        acts[t["data"].get("act", "?")] = acts.get(t["data"].get("act", "?"), 0) + 1
        if t["data"].get("drives"):
            drives_trail.append({"at": t["at"], **t["data"]["drives"]})
    spent = sum(int(t["data"].get("tokens_weighted") or 0) for t in ticks)
    earned = sum(int(e["data"].get("fee_tokens") or 0)
                 for e in events if e["kind"] == "chore_paid")
    spoke = sum(1 for e in events if e["kind"] == "spoke_to_parent")
    suppressed = sum(1 for e in events if e["kind"] == "message_suppressed")
    concerns: list[str] = []
    for kind, label in [("collapsed_exhausted", "collapsed from overspending"),
                        ("tick_timeout", "ticks timed out"),
                        ("digest_parse_failed", "gave unstructured reports"),
                        ("spawn_failed", "body failed to spawn"),
                        ("narration_mismatch",
                         "claimed to write files it never wrote"),
                        ("act_unverified", "claimed to make things it didn't"),
                        ("drive_unearned",
                         "felt accomplished without making anything"),
                        ("chore_claim_invalid", "claimed a chore that wasn't open")]:
        n = sum(1 for e in events if e["kind"] == kind)
        if n:
            concerns.append(f"{label} ×{n}")
    entries: list[str] = []
    for off in range(days):
        try:
            p = _home_path(being, _journal_rel(now - timedelta(days=off)))
            if p.exists():
                entries.append(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    rut = _rut_score(entries)
    if rut >= 0.6:
        concerns.append(f"journal is repetitive (rut score {rut})")
    top = max(acts.values()) if acts else 0
    if ticks and top / max(1, len(ticks)) > 0.7 and len(ticks) >= 5:
        concerns.append("one act dominates its days (monotony)")
    milestones = [e for e in events if e["kind"] == "milestone"]
    consolidations = sum(1 for e in events if e["kind"] == "consolidated")
    try:
        g = being_mind.graph(store, being)
        mind = {"nodes": len(g["nodes"]), "edges": len(g["edges"]),
                "density": g["density"],
                "connected_fraction": g["connected_fraction"],
                "consolidations": consolidations}
        # A mind of many artifacts with almost no connections is scattering.
        if len(g["nodes"]) >= 6 and g["connected_fraction"] < 0.3:
            concerns.append("its work is scattered — few files connect to any "
                            "other (weave, don't only make)")
    except Exception:  # noqa: BLE001
        mind = {"nodes": 0, "edges": 0, "density": 0.0,
                "connected_fraction": 0.0, "consolidations": consolidations}
    return {
        "period_days": days,
        "ticks": len(ticks),
        "acts": acts,
        "tokens_spent_weighted": spent,
        "tokens_earned": earned,
        "messages_to_parent": spoke,
        "messages_suppressed": suppressed,
        "rut_score": rut,
        "mind": mind,
        "concerns": concerns,
        "milestones": [m["data"].get("name") for m in milestones],
        "drives_trail": drives_trail[-30:],
        "in_its_own_words": (entries[0][-600:] if entries else ""),
        "affect": being.get("affect") or {},
    }


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
