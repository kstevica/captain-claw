"""The agent that is the owner's First Mate.

See docs/first-mate-plan.md §5 (Phase 5, piece 3). The owner points First Mate
at one of their agents, or has one made (an agent called "First Mate" from the
concierge archetype, told what it is for). That agent:

* is where the owner talks to First Mate: its chat, with the ``first_mate``
  tool every agent has;
* does First Mate's thinking on its own model and key: drafting charters now.
  While it runs, the call goes through its own REST; asleep or switched off,
  through the model its config names (it is never woken for this); gone or
  with no usable model, through the owner's tier set.

Making one is the owner's own act (a button): an agent is a process with its
own cost, so turning First Mate on doesn't start one by itself.
"""

from __future__ import annotations

import logging
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck import side_call_health
from captain_claw.flight_deck.first_mate_store import FirstMateError

log = logging.getLogger(__name__)

NAME = "First Mate"
ARCHETYPE = "concierge@balanced"
DESCRIPTION = "Looks after your standing jobs (factories) on First Mate"
INSTRUCTIONS = (
    "You are First Mate: you look after your owner's standing jobs (\"factories\") on this deck, through the "
    "first_mate tool.\n"
    "- When they ask how things are going, use first_mate list, status or report, and answer in a few lines.\n"
    "- When they want a new standing job, use first_mate draft with their own words, every limit included, and "
    "tell them it waits for their approval on the Line.\n"
    "- Run, pause or resume a factory when they ask.\n"
    "- You never approve what a factory would do, carry it out, or change its trust: they do that on First Mate "
    "or the Line. Say so when asked.\n"
    "Keep answers short and concrete."
)
# A thinking model spends most of its answer's budget thinking first: on the
# Pi, about 3,000 tokens before a word of the draft, so 3,000 left it empty.
MAX_TOKENS = 12_000


GROUP = "First Mate"


async def label_agent(slug: str, *, name: str = "", description: str = "") -> None:
    """Show an agent First Mate made under *name*, in the owner's First Mate
    agent group. Only its registry entry changes: its slug stays (a Bat worker's
    ``bat-<run>`` slug is how sends and the tool-call register are tied to its
    run). Under the spawn lock, so it never races another spawn's write."""
    from captain_claw.flight_deck import server as srv
    async with srv._get_spawn_lock():
        reg = srv._load_process_registry()
        entry = reg.get(slug)
        if not isinstance(entry, dict):
            return
        if name:
            entry["name"] = name[:80]
        if description:
            entry["description"] = description[:200]
        entry["group"] = GROUP
        srv._save_process_registry(reg)


async def label_worker(slug: str, run: dict[str, Any], step_key: str) -> None:
    """A factory run's Bat worker: named for its factory and step, in the
    First Mate group. Nothing for a Bat run that isn't a factory's. Never raises."""
    fid = str(((run or {}).get("config") or {}).get("factory_id") or "")
    if not fid or not slug:
        return
    try:
        f = fms.get_store().get(fid)
        if f is None:
            return
        step = step_key.replace("-", " ").strip()
        what = "improving" if (run.get("config") or {}).get("source") == "captain_swarm" else ""
        name = " · ".join(x for x in (f["title"], what, step) if x)
        await label_agent(slug, name=name, description=f"First Mate · {f['title']}")
    except Exception:  # noqa: BLE001 — a worker without its label still works
        log.debug("first mate: couldn't label a worker", exc_info=True)


def mine(owner_id: str) -> list[dict[str, str]]:
    """The agents the owner may pick: their own, by name."""
    from captain_claw.flight_deck import line
    return line.user_agents(owner_id)


def current(store: fms.FirstMateStore, owner_id: str) -> dict[str, Any]:
    """The owner's First Mate agent as the page shows it: ``{slug, name,
    exists, running}``; ``slug`` "" when none is picked."""
    slug = store.prefs(owner_id).get("agent") or ""
    if not slug:
        return {"slug": "", "name": "", "exists": False, "running": False}
    agents = {a["slug"]: a["name"] for a in mine(owner_id)}
    running = False
    if slug in agents:
        try:
            from captain_claw.flight_deck import server as srv
            running = bool(srv._process_is_alive(slug))
        except Exception:  # noqa: BLE001
            running = False
    return {"slug": slug, "name": agents.get(slug, slug), "exists": slug in agents, "running": running}


def check_pick(owner_id: str, slug: str) -> None:
    """A First Mate agent must be one of the owner's own."""
    if slug and slug not in {a["slug"] for a in mine(owner_id)}:
        raise FirstMateError("that agent isn't one of yours")


async def create(store: fms.FirstMateStore, user: dict[str, Any]) -> dict[str, Any]:
    """Make the owner an agent called First Mate and point First Mate at it."""
    from captain_claw.flight_deck import agent_forge
    owner = fms.norm_user(str(user.get("id") or ""))
    res, used = await agent_forge._spawn_agent(user, name=NAME, description=DESCRIPTION, archetype=ARCHETYPE,
                                               instructions=INSTRUCTIONS, env=[])
    if not getattr(res, "ok", False):
        raise FirstMateError(getattr(res, "message", "") or "the agent didn't start")
    store.set_prefs(owner, {"agent": res.slug})
    try:
        await label_agent(res.slug)  # in the owner's First Mate group, beside its factories' workers
    except Exception:  # noqa: BLE001
        log.debug("first mate: couldn't group its agent", exc_info=True)
    return {"slug": res.slug, "name": used}


async def ask(store: fms.FirstMateStore, owner_id: str, system: str, user: str, *,
              max_tokens: int = MAX_TOKENS) -> str | None:
    """One call on the owner's First Mate agent's model, or None when there is
    no such agent (the caller uses the owner's tier set). An agent that stops
    answering hands over to its fallback, the owner's tier."""
    slug = store.prefs(owner_id).get("agent") or ""
    if not slug:
        return None
    from captain_claw.flight_deck import agent_preset, file_library, taskboard_triage
    reader = await taskboard_triage.reader_for({"agent_slug": slug, "agent_owner_id": fms.norm_user(owner_id)},
                                               fms.norm_user(owner_id))
    if reader is None:
        return None
    if reader.get("agent"):
        try:
            reply = await agent_preset._agent_complete(reader, system, user, fms.norm_user(owner_id), max_tokens)
            return reply.text or ""
        except agent_preset._AgentGoneError:
            reader = reader.get("fallback")
            if not reader:
                return None
    from captain_claw.flight_deck.bat_routes import call_timeout
    reply = await file_library.model_reply(reader, system, user, fms.norm_user(owner_id), max_tokens=max_tokens,
                                           request_timeout=call_timeout(), feature=side_call_health.FIRST_MATE)
    return reply.text or ""
