"""An expert agent beside a list funnel's run.

See docs/funnel-expert-agent-plan.md. The owner picks one of their agents for a
factory (``runner_config.expert`` → the run's ``config.funnel_expert``). A run then:

* wakes or starts it (``agent_idle.ready_for_task``): a run never goes without its
  expert because the agent slept;
* gives it a turn as itself, in a session opened for this run and no other, to add
  what it knows and to settle, as the owner would, what the charter leaves open
  (:func:`notes`);
* asks it short questions on its own model, with its instructions and lessons in
  the prompt (:func:`ask`): the new searches for a group that is short.

Everything it said is kept in ``shared/funnel/expert.jsonl``, and the summary says
what it did (:func:`summary_lines`). When the agent can't be reached the run goes on
without it and says so.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

STATE = "shared/funnel/expert.json"     # who it is, its session for this run, whether it could be reached
LOG = "shared/funnel/expert.jsonl"      # every time it was asked: what for, what it said
NOTES = "shared/funnel/expert-notes.md"  # what it added before the run, read wherever the brief is
NOTES_CHARS = 2500
KNOWS_CHARS = 4000        # of its instructions and lessons, in a short call's prompt
TURN_TIMEOUT_S = 1200.0   # a full turn: it may read and think as itself
ASK_TOKENS = 2000
KIND = "colleague"        # the turn's kind on the agent: its own session, nothing mirrored into its chat

NOTES_PROMPT = """You are asked as the expert on this, and as its owner: a run is about to build the list
below with code and short model calls, and it will work from what you say here. Do not build the list yourself,
search for its items, write files or send anything. Answer from what you know, in plain text, at most 350 words,
in two parts under exactly these two headings:

NOTES
What you know that the goal doesn't say and that helps find the right items and tell them from the wrong ones:
where items of this kind are listed, the words the field uses for them, the signs of a right one, the wrong ones
a search will keep finding and how to tell them, what to avoid.

DECISIONS
Whatever the goal leaves open that the run would otherwise have to guess: one line each, the question and your
decision, made as the owner would. Never change what the goal itself says. Nothing open: write "None".

# The goal
{goal}
{brief}"""

SEARCH_SYSTEM = """You write web searches for a list that is still short, as someone who knows the field.
Reply ONLY with JSON: {"queries": ["...", ...]} (10-15 new ones, none already used). Write searches that find
pages listing many of the right items and few of the wrong ones: use the field's own words for them, and steer
away from what was dropped. Keep your thinking short."""


COLUMNS_SYSTEM = """You write your columns for items a list is keeping, as the expert who knows the owner's
offer and the field. Reply ONLY with JSON: {"<id>": {"<column>": "<value>"}}. Each value names what of the
owner's fits this item, by its name in their material, and the fact read about this item that shows it. Choose
what fits this item best, from everything the material offers, not what was said for the others. One sentence
unless the column says otherwise; a column with a fixed set of values takes one of them exactly. Use only facts
shown under the item; never invent one. Keep your thinking short."""


def slug_for(run: dict) -> str:
    return str((run.get("config") or {}).get("funnel_expert") or "").strip()


def _read(folder: Path) -> dict[str, Any]:
    try:
        got = json.loads((folder / STATE).read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(folder: Path, state: dict[str, Any]) -> None:
    (folder / STATE).parent.mkdir(parents=True, exist_ok=True)
    (folder / STATE).write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


def _record(folder: Path, what: str, said: str, **more: Any) -> None:
    try:
        (folder / LOG).parent.mkdir(parents=True, exist_ok=True)
        with (folder / LOG).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "said": said[:6000], **more},
                                ensure_ascii=False) + "\n")
    except OSError:
        log.debug("funnel expert: couldn't record", exc_info=True)


def notes_text(folder: Path) -> str:
    try:
        return (folder / NOTES).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


async def _note(run: dict, text: str) -> None:
    from captain_claw.flight_deck import bat_loop
    store = bat_loop.get_store()
    if store is not None:
        await store.append_event(run["id"], "note", text[:600])


def _mine(owner: str, slug: str) -> str:
    """The agent's name when it is one of the owner's own, else ""."""
    from captain_claw.flight_deck import first_mate_agent
    return next((a["name"] for a in first_mate_agent.mine(owner) if a["slug"] == slug), "")


async def start(run: dict, folder: Path) -> dict[str, Any]:
    """The run's expert, reachable: woken when it slept, started when it was off (for
    the run; :func:`release` lets it go again). ``{slug, name, ok, why}``, kept in the
    run's folder: asked once per run, and a restart finds it again. ``{}`` when the
    factory has no expert."""
    slug = slug_for(run)
    if not slug:
        return {}
    state = _read(folder)
    owner = str(run.get("owner_id") or "")
    if state.get("slug") != slug:
        state = {"slug": slug, "name": _mine(owner, slug), "session": "", "task": f"exp-{run['id'][-24:]}"}
    if not state["name"]:
        state.update(ok=False, why="it isn't one of the owner's agents any more")
    else:
        from captain_claw.flight_deck import agent_idle
        agent_idle.touch(slug)
        try:
            ok = await agent_idle.ready_for_task(slug)
        except Exception as e:  # noqa: BLE001 — the run goes on without it
            log.warning("funnel expert: couldn't start it", slug=slug, error=str(e))
            ok = False
        state.update(ok=bool(ok), why="" if ok else "it couldn't be woken or started")
    if "told" not in state:  # said once in the run's timeline
        state["told"] = True
        await _note(run, f"Expert: {state['name'] or slug} works beside this run, in a session of its own."
                    if state["ok"] else f"Expert: {state['name'] or slug} isn't available ({state['why']}). "
                    "The run goes on without it.")
    await asyncio.to_thread(_write, folder, state)
    return state


async def turn(run: dict, folder: Path, state: dict[str, Any], text: str, what: str) -> str:
    """One turn of the agent as itself, in this run's own session on it (opened by
    the first turn, used by every later one). "" when it didn't answer."""
    from captain_claw.flight_deck import agent_idle, fd_scheduler
    slug = state["slug"]
    host, port, auth = fd_scheduler.resolve_agent_by_slug(slug, "")
    if not port:
        return ""
    automation = {"kind": KIND, "job_text": "", "mail_write": "deny",
                  "task": {"id": state["task"], "name": f"Expert for {str(run.get('title') or 'a list run')[:60]}",
                           "session_id": state.get("session") or "", "mirror": False}}
    try:
        from captain_claw.flight_deck.agent_handshake import identity_frame
        identity = await identity_frame(str(run.get("owner_id") or ""), slug)
    except Exception:  # noqa: BLE001 — it answers without its identity frame
        identity = None
    meta: dict[str, Any] = {}
    agent_idle.touch(slug)
    try:
        reply = await fd_scheduler.run_prompt_and_capture(host=host, port=port, auth=auth, prompt=text,
                                                          timeout=TURN_TIMEOUT_S, automation=automation,
                                                          meta=meta, identity=identity)
    except Exception as e:  # noqa: BLE001
        log.warning("funnel expert: turn failed", slug=slug, error=str(e))
        reply = None
    agent_idle.touch(slug)
    session = (meta.get("task_session") or {}).get("id") if isinstance(meta.get("task_session"), dict) else ""
    if session and session != state.get("session"):
        state["session"] = str(session)
        await asyncio.to_thread(_write, folder, state)
    await asyncio.to_thread(_record, folder, what, reply or "", turn=True)
    return (reply or "").strip()


def _knows(owner: str, slug: str, instructions: str) -> str:
    """What the agent brings to a short call: its instructions and its lessons."""
    from captain_claw.flight_deck import agent_exam, agent_preset
    parts = []
    text = agent_preset.strip_block(instructions or "").strip()
    if text:
        parts.append("Your instructions from your owner:\n" + text)
    try:
        lessons = agent_exam._lesson_texts(slug)
    except Exception:  # noqa: BLE001
        lessons = []
    if lessons:
        parts.append("What you have learnt:\n" + "\n".join(f"- {x}" for x in lessons))
    return "\n\n".join(parts)[:KNOWS_CHARS]


async def ask(run: dict, folder: Path, state: dict[str, Any], system: str, user: str, what: str,
              max_tokens: int = ASK_TOKENS) -> str | None:
    """A short question on the agent's own model, with its instructions and lessons
    ahead of the question. None when it can't be asked: the caller uses the run's own
    model."""
    if not state.get("ok"):
        return None
    from captain_claw.flight_deck import agent_idle, agent_preset, agent_skills
    from captain_claw.flight_deck import server as srv
    slug, owner = state["slug"], str(run.get("owner_id") or "")
    entry = srv._load_process_registry().get(slug) or {}
    own = agent_preset.agent_reader(slug, entry)
    if own is None:
        return None
    try:
        knows = _knows(owner, slug, await agent_skills._instructions(owner, slug))
    except Exception:  # noqa: BLE001
        knows = ""
    full = (f"You are {state.get('name') or slug}.\n{knows}\n\n" if knows else "") + system
    reader = {**own, "agent": slug}
    for again in (False, True):
        try:
            reply = await agent_preset._agent_complete(reader, full, user, owner, max_tokens)
            await asyncio.to_thread(_record, folder, what, reply.text or "", tokens=reply.tokens)
            return reply.text or ""
        except agent_preset._AgentGoneError:
            if again or not await agent_idle.ready_for_task(slug):
                break
        except Exception as e:  # noqa: BLE001
            log.warning("funnel expert: short call failed", slug=slug, error=str(e))
            break
    await asyncio.to_thread(_record, folder, what, "", failed=True)
    return None


async def notes(run: dict, folder: Path, brief: str) -> str:
    """Before the run: what the expert knows, and what it decides as the owner where
    the charter leaves something open. Asked once per run; kept beside the brief."""
    if not slug_for(run) or (folder / NOTES).exists():
        return notes_text(folder)
    state = await start(run, folder)
    if not state.get("ok"):
        return ""
    goal = str(run.get("task") or "")[:4000]
    said = await turn(run, folder, state, NOTES_PROMPT.format(
        goal=goal, brief=f"\n# What the owner's documents and links say\n{brief}\n" if brief else ""), "notes")
    said = re.sub(r"<think>.*?</think>", "", said, flags=re.S).strip()[:NOTES_CHARS]
    body = f"From {state['name']}, the run's expert, who also decides here as its owner would:\n\n{said}\n" if said else ""
    (folder / NOTES).parent.mkdir(parents=True, exist_ok=True)
    (folder / NOTES).write_text(body, encoding="utf-8")  # empty: asked, no answer; not asked again
    await _note(run, f"Expert: {state['name']} added its notes and decisions. The searches and every judging call "
                "work from them." if said else f"Expert: {state['name']} gave no answer; the run goes on without "
                "its notes.")
    return body.strip()


def decisions(folder: Path) -> list[str]:
    """What the expert decided as the owner, one line each, from its notes."""
    text = notes_text(folder)
    m = re.search(r"^\s*\W*DECISIONS\W*\s*$(.*)", text, re.S | re.M | re.I)
    if not m:
        return []
    lines = [re.sub(r"^[-*\d.)\s]+", "", ln).strip() for ln in m.group(1).splitlines()]
    return [ln for ln in lines if ln and ln.lower().rstrip(".") != "none"][:20]


async def searches(run: dict, folder: Path, user: str) -> Any:
    """New searches for a group that is short, written by the expert; None when it
    can't be asked or gave nothing readable."""
    state = _read(folder)
    if not slug_for(run) or not state.get("ok"):
        return None
    from captain_claw.flight_deck import funnel_extract as fx
    said = await ask(run, folder, state, SEARCH_SYSTEM, user, "searches")
    return fx.parse_json(said) if said else None


def available(run: dict, folder: Path) -> bool:
    """Whether this run has an expert it could reach when it started."""
    return bool(slug_for(run)) and bool(_read(folder).get("ok"))


async def columns(run: dict, folder: Path, user: str) -> Any:
    """The expert's own columns for the rows a judging call keeps; None when it can't
    be asked or gave nothing readable (the judging call's values stand)."""
    if not available(run, folder):
        return None
    from captain_claw.flight_deck import funnel_extract as fx
    said = await ask(run, folder, _read(folder), COLUMNS_SYSTEM, user, "columns")
    return fx.parse_json(said) if said else None


async def release(run: dict) -> None:
    """The run is over: an agent that was started only for it goes off again."""
    slug = slug_for(run)
    if not slug:
        return
    try:
        from captain_claw.flight_deck import agent_idle
        agent_idle.release_after_task_soon(slug)
    except Exception:  # noqa: BLE001
        log.debug("funnel expert: couldn't release it", exc_info=True)


def summary_lines(folder: Path, items: list[dict[str, str]], written: list[str]) -> list[str]:
    """The summary's part on the expert: who, whether it could be reached, what it
    decided for the owner, and what its searches found."""
    state = _read(folder)
    if not state.get("slug"):
        return []
    name = state.get("name") or state["slug"]
    if not state.get("ok"):
        return ["", "## The expert", "", f"{name} was to work beside this run but wasn't available "
                f"({state.get('why') or 'not reachable'}). The run went on without it."]
    lines = ["", "## The expert", "", f"{name} worked beside this run, in a session of its own on that agent."]
    if notes_text(folder):
        lines.append(f"Its notes were read by the setup, the new searches and every judging call (`{NOTES}`).")
    made = decisions(folder)
    if made:
        lines += ["", "Decided for you, where the charter left something open:", ""] + [f"- {d}" for d in made]
    if written:
        mine = {f"search: {q}" for q in written}
        found = [r for r in items if r.get("source") in mine]
        kept = sum(1 for r in found if r.get("status") in ("kept", "loose"))
        lines += ["", f"It wrote {len(written)} of the run's new searches. They found {len(found)} items, "
                  f"{kept} of them on the list."]
    wrote = 0
    try:
        for ln in (folder / LOG).read_text(encoding="utf-8").splitlines():
            x = json.loads(ln)
            wrote += 1 if x.get("what") == "columns" and not x.get("failed") else 0
    except (OSError, ValueError):
        pass
    if wrote:
        lines += ["", f"It wrote the owner's columns (why an item fits, what to offer it) for the rows being kept: "
                  f"asked {wrote} time{'' if wrote == 1 else 's'}."]
    lines.append(f"Everything it was asked and said: `{LOG}`.")
    return lines
