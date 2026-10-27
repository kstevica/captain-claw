"""Agent skills — the "things I can do" cards a kiosk chat offers.

Each card is a task the person starts with one click ("Plan my week",
"Label my emails"): a title, a one-line description and the prompt the click
sends to the agent — and what, if anything, the agent needs from the person
first (``input``: a question, answered with text, a file or a picture, which
the chat asks in a dialog before it sends). Flight Deck writes them per agent from what the agent is
and what it has done:

- what it is for: its archetype, description, standing instructions, tools,
  and which of the owner's accounts (Google, Microsoft) it can reach;
- its experience: insights, latest reflection, playbooks;
- its schedules: the deck's scheduled tasks for it and its own cron jobs;
- its conversation topics and open intentions.

The owner's fast tier writes them (the File library's model). The 6 cards
offered up front are written on their own and cached in the agent's folder
(``kiosk_skills.json``, gone with the agent); they are rewritten when the
agent's make-up changes (name, description, archetype, instructions, tools,
accounts, scheduled tasks) or they are ``_TTL`` old, which is how new topics
and experience reach them. Until then — and without a model, or with the
day's allowance spent — cards are built straight from the schedules,
intentions and topics.

The "More" list (up to 16, written 8 at a time) is written only when the
person asks for it — the More dialog asks when it opens and has none — and
kept, with when it was written, until they ask again: never on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.file_library import (
    _auth_enabled,
    _quota_spent,
    _resolve_tier,
    call_model,
    reply_json,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-skills", tags=["agent-skills"])

# The icons the SPA draws (services/agentSkills.ts maps each to a glyph).
ICONS = (
    "calendar", "mail", "user", "search", "file", "chart", "clock", "sparkles",
    "list", "globe", "pen", "bell", "folder", "database", "message", "book",
    "target", "code", "image", "wallet",
)
# Where a card comes from.
SOURCES = ("archetype", "experience", "schedule", "topic", "intention")
# What a card needs from the person before the agent can start, and in what form.
NEEDS = ("none", "optional", "required")
ACCEPTS = ("text", "file", "image")

_TOP = 6                  # the cards offered up front (the composer's list)
_MORE = 16                # and behind its "More", written only when asked for
_MAX_SKILLS = _TOP + _MORE
_VERSION = 3              # in the fingerprint: a new card shape rewrites the cards up front
_TTL = 12 * 3600          # a model-written set is rewritten after this
_BASIC_TTL = 3600         # a set built without the model, before trying again
_MORE_BATCH = 8           # "More" cards per call: a small local model writes 8 in time
_MAX_TOKENS = 4000        # reasoning models think before the JSON
_CALL_TIMEOUT = 300.0     # each background call (a provider's own may be 120s)
_RETRY_NUDGE = "Reply now with the JSON object only: no thinking aloud, no prose."
_AGENT_TIMEOUT = 5.0      # each read from the running agent
_CACHE_FILE = "kiosk_skills.json"

# One write per agent at a time — and one of its "More" list (key "<slug>:more").
_RUNNING: dict[str, asyncio.Task] = {}


def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


# ── What the agent is ─────────────────────────────────────────────────


def _tools(slug: str) -> list[str]:
    """The tools the agent's config turns on (beyond the always-on set)."""
    import yaml
    try:
        data = yaml.safe_load((_data_dir() / slug / "config.yaml").read_text()) or {}
    except Exception:
        return []
    tools = data.get("tools") if isinstance(data, dict) else None
    enabled = tools.get("enabled") if isinstance(tools, dict) else None
    return [str(t) for t in enabled][:40] if isinstance(enabled, list) else []


def _own_skills(slug: str) -> list[dict]:
    """The skills in the agent's own folder (an Advanced setup writes one per
    request it handles): name and when it applies, from each SKILL.md."""
    from captain_claw.skills import _parse_frontmatter
    root = _data_dir() / slug / "data" / "home-config-parent" / ".captain-claw" / "skills"
    out: list[dict] = []
    for md in sorted(root.glob("*/SKILL.md"))[:20] if root.is_dir() else []:
        try:
            meta = _parse_frontmatter(md.read_text(encoding="utf-8")[:4000])
        except Exception:
            continue
        name = str(meta.get("name") or md.parent.name)
        out.append({"name": name, "when": _clip(meta.get("description"), 200)})
    return out


async def _archetype(entry: dict, owner_id: str) -> dict | None:
    """The archetype the agent was made from: the id spawn recorded, else
    the one whose role became its description (an archetype spawn without a
    description of its own takes the role)."""
    from captain_claw.flight_deck.archetypes import merged_archetypes
    try:
        from captain_claw.flight_deck.auth import get_db
        archetypes = await merged_archetypes(get_db(), owner_id or None)
    except Exception:
        return None
    aid = str(entry.get("archetype") or "")
    desc = str(entry.get("description") or "").strip().casefold()
    for a in archetypes:
        if aid and a.get("id") == aid:
            return a
    if desc:
        for a in archetypes:
            if str(a.get("role") or "").strip().casefold() == desc:
                return a
    return None


async def _instructions(owner_id: str, slug: str) -> str:
    from captain_claw.flight_deck.server import _owner_agent_instructions
    return (await _owner_agent_instructions(owner_id, "process")).get(slug, "")


def _scheduled(slug: str, owner_id: str) -> list[dict]:
    """The deck's scheduled tasks for this agent (prompt jobs, not flows)."""
    from captain_claw.flight_deck.fd_scheduler import get_store
    try:
        # A deck without sign-in has one tenant: its jobs carry no owner.
        jobs = get_store().list((owner_id or None) if _auth_enabled() else None)
    except Exception:
        return []
    return [
        {"name": j.get("name") or "", "schedule": j.get("schedule") or "",
         "prompt": j.get("prompt") or "", "enabled": bool(j.get("enabled"))}
        for j in jobs
        if j.get("agent_slug") == slug and (j.get("prompt") or "").strip() and not j.get("flow_id")
    ]


async def _accounts(owner_id: str) -> dict[str, bool]:
    """Which of the owner's accounts the agent can reach (no network)."""
    from captain_claw.flight_deck.google_oauth_routes import is_google_connected
    from captain_claw.flight_deck.microsoft_oauth_routes import is_microsoft_connected
    return {"google": await is_google_connected(owner_id),
            "microsoft": await is_microsoft_connected(owner_id)}


async def _make_up(slug: str, entry: dict, owner_id: str) -> dict:
    """What the agent is — the part of its sources that, when it changes,
    makes the cards out of date at once."""
    arch = await _archetype(entry, owner_id)
    skills = _own_skills(slug)
    return {
        "accounts": await _accounts(owner_id),
        "name": str(entry.get("name") or slug),
        "description": str(entry.get("description") or ""),
        "archetype": {
            "id": arch.get("id"), "role": arch.get("role"),
            "description": arch.get("description"), "keywords": arch.get("keywords") or [],
            "tools": arch.get("tools") or [],
        } if arch else None,
        "instructions": await _instructions(owner_id, slug),
        "tools": _tools(slug),
        # Only when there are some: a new key would rewrite every agent's cards.
        **({"skills": skills} if skills else {}),
        "scheduled": _scheduled(slug, owner_id),
    }


def fingerprint(make_up: dict) -> str:
    return hashlib.sha1(json.dumps({**make_up, "v": _VERSION}, sort_keys=True, default=str).encode()).hexdigest()


# ── What the agent has done (read from it while it runs) ──────────────


async def _agent_reads(entry: dict) -> dict:
    """Cron jobs, open intentions, topics, insights, the latest reflection and
    playbooks from the running agent; each one empty when it can't be read."""
    import httpx

    port = int(entry.get("web_port") or 0)
    if not port:
        return {}
    params = {"token": entry["web_auth"]} if entry.get("web_auth") else {}
    reads = {
        "cron": ("/api/cron/jobs", {}),
        "intentions": ("/api/intentions", {}),
        "topics": ("/api/topics", {"limit": 12, "hidden": 0}),
        "insights": ("/api/insights", {"limit": 15}),
        "reflection": ("/api/reflections/latest", {}),
        "playbooks": ("/api/playbooks", {}),
    }
    async with httpx.AsyncClient(timeout=_AGENT_TIMEOUT) as client:
        async def one(path: str, query: dict) -> Any:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}{path}", params={**query, **params})
                return resp.json() if resp.status_code == 200 else None
            except Exception:
                return None
        results = await asyncio.gather(*(one(p, q) for p, q in reads.values()))
    raw = dict(zip(reads, results))

    def listed(value: Any, key: str) -> list:
        if isinstance(value, dict):
            value = value.get(key)
        return value if isinstance(value, list) else []

    return {
        "cron": [
            {"text": str((j.get("payload") or {}).get("text") or ""),
             "schedule": j.get("schedule_text") or ""}
            for j in listed(raw["cron"], "jobs")
            if isinstance(j, dict) and j.get("enabled") and (j.get("payload") or {}).get("text")
        ][:6],
        "intentions": [
            {"title": i.get("title") or "", "why": i.get("why") or i.get("body") or ""}
            for i in listed(raw["intentions"], "intentions") if isinstance(i, dict) and i.get("title")
        ][:6],
        "topics": [
            {"title": t.get("label") or "", "summary": t.get("summary") or ""}
            for t in listed(raw["topics"], "topics") if isinstance(t, dict) and t.get("label")
        ][:10],
        "insights": [
            str(i.get("content") or "") for i in listed(raw["insights"], "items")
            if isinstance(i, dict) and i.get("content")
        ][:15],
        "reflection": str((raw["reflection"] or {}).get("summary") or "") if isinstance(raw["reflection"], dict) else "",
        "playbooks": [
            {"name": p.get("name") or "", "when": p.get("trigger_description") or ""}
            for p in listed(raw["playbooks"], "items") if isinstance(p, dict) and p.get("name")
        ][:8],
    }


# ── Cards without the model ───────────────────────────────────────────


_NO_INPUT = {"needs": "none", "ask": "", "accepts": []}


def _input(raw: Any) -> dict:
    """What a card needs from the person, cleaned: ``none`` unless the model
    said optional / required (with a question; any form when it named none)."""
    d = raw if isinstance(raw, dict) else {}
    needs = str(d.get("needs") or "").strip().lower()
    if needs not in NEEDS or needs == "none":
        return dict(_NO_INPUT)
    accepts = d.get("accepts") if isinstance(d.get("accepts"), list) else []
    kinds = [k for k in ACCEPTS if k in {str(a).strip().lower() for a in accepts}]
    return {"needs": needs, "ask": _clip(d.get("ask"), 160), "accepts": kinds or list(ACCEPTS)}


def _card(title: str, description: str, prompt: str, icon: str, source: str, needs: Any = None) -> dict:
    return {"title": _clip(title, 48), "description": _clip(description, 120),
            "prompt": str(prompt).strip()[:1200], "icon": icon, "source": source, "input": _input(needs)}


def basic_skills(make_up: dict, done: dict | None = None) -> list[dict]:
    """Cards straight from the agent's schedules, intentions and topics, and
    one that asks what it can do."""
    done = done or {}
    cards: list[dict] = []
    for t in make_up.get("scheduled") or []:
        cards.append(_card(t["name"] or _clip(t["prompt"], 40), f"Run it now — normally {t['schedule']}",
                           t["prompt"], "clock", "schedule"))
    for c in done.get("cron") or []:
        cards.append(_card(_clip(c["text"], 40), f"Run it now — normally {c['schedule']}" if c["schedule"] else "Run it now",
                           c["text"], "clock", "schedule"))
    for i in done.get("intentions") or []:
        cards.append(_card(i["title"], i["why"] or "Something I mean to do for you",
                           f"Let's work on this now: {i['title']}", "target", "intention"))
    for t in (done.get("topics") or [])[:2]:
        cards.append(_card(f"Pick up {t['title']}", t["summary"] or "Where we left off",
                           f"Let's continue with \"{t['title']}\" — where did we leave off, and what's next?",
                           "message", "topic"))
    cards.append(_card(
        "What can you do?", "A short tour of what I'm set up for, with examples",
        "What can you do for me? Give me a short list of concrete things you can do, with an example request for each.",
        "sparkles", "archetype"))
    return _dedupe(cards, _TOP)


def _dedupe(cards: list[dict], limit: int = _MAX_SKILLS, skip: list[dict] = ()) -> list[dict]:
    """The cards with a title and a prompt, each title once and none of
    *skip*'s, at most *limit*."""
    seen = {c["title"].casefold() for c in skip}
    out: list[dict] = []
    for c in cards:
        key = c["title"].casefold()
        if c["title"] and c["prompt"] and key not in seen:
            seen.add(key)
            out.append(c)
    return out[:limit]


# ── Cards from the model ──────────────────────────────────────────────


_SYSTEM = """You design the "things I can do" cards a chat app shows above its message box.
Each card is a task the person starts with one click: the click sends the card's prompt to
their AI agent exactly as if they had typed it.
Reply with ONLY a JSON object, no prose, no code fence:
{"skills": [{"title": "...", "description": "...", "prompt": "...", "icon": "...", "source": "...",
             "input": {"needs": "...", "ask": "...", "accepts": ["..."]}}]}

%(cards)s
- title: 2-5 words in the person's voice, imperative ("Plan my week", "Label my emails"), at most 40 characters.
- description: one short sentence on what they get, at most 90 characters.
- prompt: the request itself, written as the person, specific and complete enough for the agent to act without asking back, at most 400 characters. It is sent as it is: no placeholders ("[name]", "<topic>") and nothing left for the person to fill in.
- icon: exactly one of: %(icons)s
- source: where the card comes from, exactly one of: archetype (what the agent is for), experience (what it has learned about the person), schedule (a scheduled task or cron job: run it now), topic (an ongoing conversation to pick up), intention (something the agent means to do).
- input: what the agent needs from the person before it can start.
  needs: "none" when it can act on its own (it reads the inbox, the calendar, the web, its files and memory);
  "required" when it cannot start without something only the person has (a document, a picture, a link,
  the text to check, a name or a date); "optional" when it can start but a detail would sharpen the result.
  ask: with "optional" or "required", one short question to the person, at most 120 characters; "" with "none".
  accepts: which of "text", "file", "image" would answer it ([] with "none").
  Most cards need "none". Ask only for the material the task itself works on that the agent can't find on
  its own; never ask whether an account is connected or how something is set up (the app knows that).
  With "optional" or "required", write the prompt so what they add reads after it ("Fact-check the claims in what I add below.").
- Only tasks this agent can do with its role, instructions and tools. Prefer the person's real topics, schedules and intentions over generic ideas, and mix sources when there are several.
- Email, calendar and file-drive tasks need the person's account connected (see "Connected accounts"). For an account that isn't, offer none of its tasks; when the agent is made for that work, one card may help the person connect it instead.
- Never a card that sends email, posts, pays or deletes by itself; drafting and preparing are fine.
- Write in the language the person uses with the agent (see the topics and instructions); English if unsure."""

# What a pass asks for: the cards offered up front, then the "More" list.
_TOP_CARDS = """- %d cards, the most useful first: the everyday tasks that matter most to this person."""
_MORE_CARDS = """- %d MORE cards, for its "More" list: the rest of the agent's range (other tools, rarer
  tasks, other topics). The person already has the cards under "Already offered": every new card
  is different from those and from each other."""


def _prompt(make_up: dict, done: dict) -> str:
    parts = [f"Agent: {make_up['name']}"]
    if make_up.get("description"):
        parts.append(f"Description: {make_up['description']}")
    arch = make_up.get("archetype")
    if arch:
        parts.append(f"Archetype: {arch.get('role')} — {arch.get('description')}"
                     f" (keywords: {', '.join(arch.get('keywords') or [])})")
    tools = sorted(set((make_up.get("tools") or []) + ((arch or {}).get("tools") or [])))
    if tools:
        parts.append(f"Tools: {', '.join(tools)}")
    accounts = make_up.get("accounts") or {}
    parts.append("Connected accounts: "
                 f"Google (Gmail, Calendar, Drive): {'yes' if accounts.get('google') else 'no'}; "
                 f"Microsoft 365 (Outlook, Calendar, OneDrive): {'yes' if accounts.get('microsoft') else 'no'}")
    if make_up.get("instructions"):
        parts.append(f"Standing instructions:\n{make_up['instructions'][:3000]}")
    if make_up.get("skills"):
        parts.append("Requests it has a procedure for (its skills):\n" + "\n".join(
            f"- {s['name']}: {s['when']}" for s in make_up["skills"]))
    if make_up.get("scheduled"):
        parts.append("Scheduled tasks:\n" + "\n".join(
            f"- {t['name']} ({t['schedule']}): {_clip(t['prompt'], 300)}" for t in make_up["scheduled"]))
    if done.get("cron"):
        parts.append("Cron jobs:\n" + "\n".join(
            f"- ({c['schedule']}) {_clip(c['text'], 300)}" for c in done["cron"]))
    if done.get("intentions"):
        parts.append("Open intentions:\n" + "\n".join(
            f"- {i['title']}: {_clip(i['why'], 200)}" for i in done["intentions"]))
    if done.get("topics"):
        parts.append("Recent conversation topics:\n" + "\n".join(
            f"- {t['title']}: {_clip(t['summary'], 240)}" for t in done["topics"]))
    if done.get("insights"):
        parts.append("What it has learned:\n" + "\n".join(f"- {_clip(i, 200)}" for i in done["insights"]))
    if done.get("reflection"):
        parts.append(f"Latest reflection:\n{_clip(done['reflection'], 1500)}")
    if done.get("playbooks"):
        parts.append("Playbooks:\n" + "\n".join(
            f"- {p['name']}: {_clip(p['when'], 160)}" for p in done["playbooks"]))
    return "\n\n".join(parts)


def _sources(make_up: dict, done: dict) -> set[str]:
    """The sources the agent has something in."""
    have = {"archetype"}
    if done.get("insights") or done.get("reflection") or done.get("playbooks"):
        have.add("experience")
    if make_up.get("scheduled") or done.get("cron"):
        have.add("schedule")
    if done.get("topics"):
        have.add("topic")
    if done.get("intentions"):
        have.add("intention")
    return have


def _salvage(content: str) -> list[dict]:
    """The whole cards in a reply that is not valid JSON — one cut off at
    the output limit mid-list, say."""
    found: list[dict] = []
    for m in re.finditer(r"\{[^{}]*\}", content or ""):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("title") and obj.get("prompt"):
            found.append(obj)
    return found


def parse_skills(content: str, sources: set[str] | None = None) -> list[dict]:
    """The cards in a model's reply, cleaned; raises when there are none. A
    card naming a source the agent has nothing in counts as its archetype's."""
    allowed = set(SOURCES) if sources is None else sources
    try:
        raw = reply_json(content).get("skills")
    except ValueError:
        raw = _salvage(content)
    cards: list[dict] = []
    for s in raw if isinstance(raw, list) else []:
        if not isinstance(s, dict):
            continue
        icon = str(s.get("icon") or "").strip().lower()
        source = str(s.get("source") or "").strip().lower()
        cards.append(_card(
            str(s.get("title") or ""), str(s.get("description") or ""), str(s.get("prompt") or ""),
            icon if icon in ICONS else "sparkles", source if source in allowed else "archetype",
            s.get("input")))
    cards = _dedupe(cards)
    if not cards:
        raise ValueError("the model wrote no cards")
    return cards


# ── The cache ─────────────────────────────────────────────────────────


def _cache_path(slug: str) -> Path:
    return _data_dir() / slug / _CACHE_FILE


def _migrate(data: dict) -> dict:
    """A set written before the "More" list was asked for apart (every card
    in ``skills``): its extra cards become the More list."""
    if "more" not in data and data.get("source") == "model" and len(data.get("skills") or []) > _TOP:
        return {**data, "skills": data["skills"][:_TOP], "more": data["skills"][_TOP:],
                "more_generated_at": data.get("generated_at")}
    return data


def read_cache(slug: str) -> dict | None:
    try:
        data = json.loads(_cache_path(slug).read_text())
    except (OSError, ValueError):
        return None
    return _migrate(data) if isinstance(data, dict) else None


def _write_cache(slug: str, payload: dict) -> None:
    path = _cache_path(slug)
    if not path.parent.is_dir():
        return  # the agent was removed meanwhile
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)


def _update_cache(slug: str, **fields: Any) -> dict:
    """Write *fields* over the cached set, keeping the rest: the cards up
    front and the More list are written apart, maybe at the same time (no
    await between the read and the write)."""
    data = {**(read_cache(slug) or {}), **fields}
    _write_cache(slug, data)
    return data


def is_fresh(cache: dict | None, fp: str, now: float | None = None) -> bool:
    """Are the cards up front current? (The More list never goes stale on
    its own: the person asks for a new one.)"""
    if not cache or cache.get("fingerprint") != fp:
        return False
    # A set built without the model, or a model set kept over a failed
    # rewrite, is tried again sooner.
    ttl = _TTL if cache.get("source") == "model" and not cache.get("error") else _BASIC_TTL
    return (now or time.time()) - float(cache.get("generated_at") or 0) < ttl


def _system(offered: list[dict] | None) -> str:
    cards = _TOP_CARDS % _TOP if offered is None else _MORE_CARDS % _MORE_BATCH
    return _SYSTEM % {"icons": ", ".join(ICONS), "cards": cards}


async def _model_skills(tier: dict, make_up: dict, done: dict, owner_id: str,
                        offered: list[dict] | None = None) -> list[dict]:
    """The model's cards up front — or, given the ones *offered*, the "More"
    list. One more try, told to skip straight to the JSON, when a reply has
    none (a small reasoning model can think its budget away)."""
    system, user = _system(offered), _prompt(make_up, done)
    if offered is not None:
        user += "\n\nAlready offered:\n" + "\n".join(f"- {c['title']}: {c['description']}" for c in offered)
    sources = _sources(make_up, done)
    limit = _TOP if offered is None else _MORE_BATCH
    try:
        content, _tokens = await call_model(tier, system, user, owner_id, max_tokens=_MAX_TOKENS,
                                            request_timeout=_CALL_TIMEOUT)
        return parse_skills(content, sources)[:limit]
    except ValueError:
        content, _tokens = await call_model(tier, system, f"{user}\n\n{_RETRY_NUDGE}", owner_id,
                                            max_tokens=_MAX_TOKENS, request_timeout=_CALL_TIMEOUT)
        return parse_skills(content, sources)[:limit]


async def generate(slug: str, entry: dict, owner_id: str) -> dict:
    """Write the agent's cards up front and cache them; the More list stays as
    it is. Without a tier, with the allowance spent or after bad replies, the
    last model-written cards stay (tried again sooner); with none, the cards
    built without the model."""
    from captain_claw.flight_deck.server import _process_is_alive

    make_up = await _make_up(slug, entry, owner_id)
    done = await _agent_reads(entry) if _process_is_alive(slug) else {}
    prev = read_cache(slug)
    fields: dict = {
        "fingerprint": fingerprint(make_up), "generated_at": time.time(),
        "skills": basic_skills(make_up, done), "source": "basic", "model": "", "error": "",
    }
    tier = await _resolve_tier(owner_id)
    error = ""
    if not tier:
        error = "no model tier set"
    elif await _quota_spent(owner_id):
        error = "daily allowance spent"
    else:
        try:
            fields.update(skills=await _model_skills(tier, make_up, done, owner_id), source="model",
                          model=str(tier.get("model") or ""))
        except Exception as exc:
            log.info("agent skills: model write failed", agent=slug, error=str(exc))
            error = str(exc)[:300]
    if error:
        fields["error"] = error
        if prev and prev.get("source") == "model":
            fields.update(skills=prev["skills"], source="model", model=prev.get("model") or "")
    return _update_cache(slug, **fields)


async def generate_more(slug: str, entry: dict, owner_id: str) -> dict:
    """Write the More list, asked for by the person: up to 16 cards, 8 at a
    time, each batch told what is already offered and cached as it lands
    (the new list replaces the old one with its first batch). Without a
    tier, with the allowance spent or when nothing new comes, the earlier
    list stays."""
    from captain_claw.flight_deck.server import _process_is_alive

    make_up = await _make_up(slug, entry, owner_id)
    done = await _agent_reads(entry) if _process_is_alive(slug) else {}
    top = (read_cache(slug) or {}).get("skills") or basic_skills(make_up, done)
    tier = await _resolve_tier(owner_id)
    error = ""
    if not tier:
        error = "no model tier set"
    elif await _quota_spent(owner_id):
        error = "daily allowance spent"
    more: list[dict] = []
    while not error and len(more) < _MORE:
        try:
            batch = await _model_skills(tier, make_up, done, owner_id, offered=top + more)
        except Exception as exc:
            log.info("agent skills: more cards failed", agent=slug, error=str(exc))
            error = str(exc)[:300]
            break
        grown = _dedupe(more + batch, _MORE, skip=top)
        if len(grown) == len(more):
            break  # nothing new: the agent's range is covered
        more = grown
        _update_cache(slug, more=more, more_generated_at=time.time(), more_error="")
    return _update_cache(slug, more_error=error)


def _running(key: str) -> bool:
    task = _RUNNING.get(key)
    return bool(task and not task.done())


def _start(slug: str, entry: dict, owner_id: str, more: bool = False) -> None:
    """Write the cards up front — or, with *more*, the More list — in the
    background, once at a time."""
    key = f"{slug}:more" if more else slug
    if _running(key):
        return

    async def run() -> None:
        try:
            await (generate_more if more else generate)(slug, entry, owner_id)
        except Exception:
            log.warning("agent skills: write failed", agent=slug, more=more, exc_info=True)
        finally:
            _RUNNING.pop(key, None)

    _RUNNING[key] = asyncio.create_task(run())


def _view(cache: dict | None, make_up: dict, slug: str) -> dict:
    cache = cache or {}
    top = cache.get("skills") or basic_skills(make_up)
    return {
        "skills": top,
        "source": cache.get("source") or "basic",
        "model": cache.get("model") or "",
        "generated_at": cache.get("generated_at"),
        "generating": _running(slug),
        # Asked for apart; a card that became one up front shows there only.
        "more": _dedupe(cache.get("more") or [], _MORE, skip=top),
        "more_generated_at": cache.get("more_generated_at"),
        "more_error": cache.get("more_error") or "",
        "generating_more": _running(f"{slug}:more"),
    }


# ── Routes ────────────────────────────────────────────────────────────


def _owned(slug: str, user: dict) -> tuple[dict, str]:
    """The agent's registry entry and owner — 404 unless it is the caller's."""
    from captain_claw.flight_deck.server import _verify_process_owner
    uid = str(user.get("id") or "")
    entry = _verify_process_owner(slug, uid)
    return entry, str(entry.get("owner") or uid)


@router.get("/{slug}")
async def agent_skills(slug: str, user: dict = Depends(get_current_user)) -> dict:
    """The agent's cards. A stale or missing set is rewritten in the
    background (``generating``); meanwhile the last set — or the cards
    built from its schedules — is returned."""
    entry, owner_id = _owned(slug, user)
    make_up = await _make_up(slug, entry, owner_id)
    cache = read_cache(slug)
    if not is_fresh(cache, fingerprint(make_up)):
        _start(slug, entry, owner_id)
    return _view(cache, make_up, slug)


@router.post("/{slug}/refresh")
async def refresh_agent_skills(slug: str, user: dict = Depends(get_current_user)) -> dict:
    """Write the agent's cards up front again now (in the background)."""
    entry, owner_id = _owned(slug, user)
    make_up = await _make_up(slug, entry, owner_id)
    _start(slug, entry, owner_id)
    return _view(read_cache(slug), make_up, slug)


@router.post("/{slug}/more")
async def more_agent_skills(slug: str, user: dict = Depends(get_current_user)) -> dict:
    """Write the agent's More list now (in the background) — only ever on the
    person's request."""
    entry, owner_id = _owned(slug, user)
    make_up = await _make_up(slug, entry, owner_id)
    _start(slug, entry, owner_id, more=True)
    return _view(read_cache(slug), make_up, slug)
