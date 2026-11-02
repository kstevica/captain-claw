"""Forge agents — say what you need, attach files, and Captain plans the team.

  POST   /fd/agent-forge/plans                          a request and its files; planning starts
  GET    /fd/agent-forge/plans                          the caller's plans
  GET    /fd/agent-forge/plans/{id}                     one plan, wherever it has got to
  POST   /fd/agent-forge/plans/{id}/replan              plan again, with answers and a note
  POST   /fd/agent-forge/plans/{id}/approve             create the approved agents, then recommend
  POST   /fd/agent-forge/plans/{id}/recommend           recommend again
  POST   /fd/agent-forge/plans/{id}/schedules           make the scheduled tasks the person kept
  POST   /fd/agent-forge/plans/{id}/apps/{app}/build    hand an app to Crew
  POST   /fd/agent-forge/plans/{id}/apps/{app}/connect  connect agents to the running app
  DELETE /fd/agent-forge/plans/{id}                     forget the plan; what it made stays

Planning is Vatra's: the file examiner opens what was attached, the Lead chooses
the agents (from its own brief, instructions/forge/lead.md, because these are
standing agents and not one run's workers), and the Long Horizon Planner writes
each one's job and may ask questions. A plan is a row in ``basna_sessions``, as a
run is: the plan, the files and the progress feed are the same ones, and it
stands at ``awaiting_plan`` until the person approves it. Only then is anything
created.

After that nothing happens by itself either. One model call recommends scheduled
tasks and apps; the person keeps what they want and makes it from the dialog.
Apps go to Crew with a fixed contract (instructions/forge/app_contract.md): agents
reach them over an HTTP API and a WebSocket. Connecting uses what agents already
have: an outbound agent link for the socket (Flight Deck dials the app, the key
sealed) and ``direct_api`` calls for the API, so the key never reaches a model.
The key is never kept in the plan.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import types
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from captain_claw.flight_deck import crew_access, crew_harness, crew_routes, crew_runner, crew_store, fd_scheduler
from captain_claw.flight_deck import vatra_routes as vr
from captain_claw.flight_deck.agent_handshake import hand_identity
from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.basna_routes import _MAX_FILE_BYTES, _safe_name
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-forge", tags=["agent-forge"])

MODE = "forge"
PLANNING, GATE, CREATING, RECOMMENDING, READY, FAILED = (
    "planning", "awaiting_plan", "creating", "recommending", "ready", "error")
_WORKING = {PLANNING, CREATING, RECOMMENDING}

MAX_TEAM = 8
MAX_FILES = 10
MAX_REQUEST_CHARS = 20_000
MAX_SCHEDULES = 12
MAX_APPS = 5
_MAX_INSTRUCTIONS = 64_000
_DISPATCH_TIMEOUT = 600.0
_APP_CALLS_MAX = 40
_BRIEF_CHARS = 12_000

_TEAM_OPEN, _TEAM_CLOSE = "<!-- forge-team -->", "<!-- /forge-team -->"
_SLUG_CHARS = re.compile(r"[^a-z0-9-]+")
_PART = re.compile(r"<!-- part:(\w+) -->")

_locks: dict[str, asyncio.Lock] = {}


# ── Helpers ───────────────────────────────────────────────────────────


def _lock(sid: str) -> asyncio.Lock:
    return _locks.setdefault(sid, asyncio.Lock())


def _stub(uid: str) -> Any:
    """The request a background spawn needs: only who it's for."""
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=uid))


def _slugify(text: str, fallback: str = "team") -> str:
    slug = _SLUG_CHARS.sub("-", str(text or "").lower()).strip("-")
    return slug[:48].strip("-") or fallback


def _route_of(sess: dict) -> dict:
    try:
        route = json.loads(sess.get("route") or "{}")
    except (json.JSONDecodeError, TypeError):
        route = {}
    return route if isinstance(route, dict) else {}


def _forge_of(sess: dict) -> dict:
    return _route_of(sess).get("forge") or {}


def _is_plan(sess: dict | None) -> bool:
    if not sess:
        return False
    try:
        return json.loads(sess.get("config") or "{}").get("mode") == MODE
    except (json.JSONDecodeError, TypeError):
        return False


async def _plan_of(plan_id: str, user: dict) -> dict:
    sess = await get_db().get_basna_session(plan_id, user["id"])
    if not _is_plan(sess):
        raise HTTPException(404, "That plan isn't here.")
    return sess


async def _save(sid: str, uid: str, *, status: str | None = None, route: dict | None = None,
                **forge: Any) -> dict:
    """Change a plan's status, route and forge state as one, under its lock: the
    background step and the person's clicks both write this row."""
    db = get_db()
    async with _lock(sid):
        sess = await db.get_basna_session(sid, uid)
        if not sess:
            return {}
        current = _route_of(sess)
        merged = {**route, "forge": current.get("forge") or {}} if route is not None else current
        merged.setdefault("forge", {}).update(forge)
        fields: dict[str, Any] = {"route": json.dumps(merged)}
        if status:
            fields["status"] = status
        if route is not None and route.get("domain"):
            fields["domain"] = route["domain"]
        await db.update_basna_session(sid, uid, **fields)
        return merged


async def _bell(uid: str, kind: str, title: str, body: str, sid: str) -> None:
    try:
        from captain_claw.flight_deck.autonomy import _norm_user

        await get_db().add_notification(_norm_user(uid), kind, title[:200], body[:2000],
                                        ref_type="agent_forge", ref_id=sid)
    except Exception:
        log.debug("agent forge: bell notification failed", exc_info=True)


async def _refuse_when_spent(uid: str) -> None:
    from captain_claw.flight_deck import file_library

    if await file_library._quota_spent(uid):
        raise HTTPException(429, "Today's allowance is used up, so Captain can't plan this now. It resets at midnight UTC.")


def _room(user: dict) -> int:
    """How many more agents this person's plan lets them have, at most a full team."""
    from captain_claw.flight_deck import rate_limiter
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    from captain_claw.flight_deck.server import _load_process_registry

    if rate_limiter.STANDALONE_MODE or not _fd_auth_enabled():
        return MAX_TEAM
    limit = int(rate_limiter.get_user_limits(user).get("max_agents") or 0)
    owned = sum(1 for e in _load_process_registry().values() if e.get("owner") == user["id"])
    return max(0, min(limit - owned, MAX_TEAM))


async def _count_tokens(sid: str, uid: str, seen: int) -> int:
    """Tokens this plan's own model calls spent since *seen*, toward the owner's day.
    (A planning agent's turns count themselves.) Returns where the count got to."""
    entries = vr._RUN_USAGE.get(sid) or []
    total = sum(int((e.get("usage") or {}).get("total_tokens", 0) or 0) for e in entries[seen:])
    if total:
        from captain_claw.flight_deck import usage_quota

        try:
            await usage_quota.record_tokens(get_db(), uid, total, byok=False)
        except Exception:
            log.debug("agent forge: token report failed", exc_info=True)
    return len(entries)


# ── Prompts ───────────────────────────────────────────────────────────


def _lead_task(request: str, brief: str, note: str, clarifications: str, previous: list[dict],
               file_names: list[str]) -> str:
    parts = [request.strip()]
    if brief.strip():
        parts.append("## What the attached files say (from an agent that opened them)\n" + brief.strip())
    if file_names:
        parts.append("## Files they attached\n" + "\n".join(f"- {n}" for n in file_names))
    if previous:
        lines = [f"- {a.get('name')}: {str(a.get('mandate') or '')[:300]}" for a in previous]
        parts.append("## The team proposed last time (keep what still fits)\n" + "\n".join(lines))
    if note.strip():
        parts.append("## What they want changed\n" + note.strip())
    return "\n\n".join(parts) + clarifications


def _planner_task(request: str, brief: str, note: str) -> str:
    """The Long Horizon Planner's frame: it was written for a run's phases, and these
    agents stay. Its `intent` is where that gets said."""
    parts = [
        "Someone needs a team of STANDING agents. Each piece below becomes a permanent agent "
        "they talk to and that runs scheduled tasks, not a step in one run. Write `mandate` as "
        "that agent's standing job: what it owns, how it does the work, and when it stops and "
        "asks a person. Write `produces` as what it regularly delivers. `consumes_from` is whose "
        "work it builds on, and `hand_off_notes` is how teammates get its work. Ignore anything "
        "below about phases or running in order: these agents don't run in order. Ask questions "
        "only when the answer would change who is on the team or what they own.",
        "## What they asked for\n" + request.strip(),
    ]
    if brief.strip():
        parts.append("## What the attached files say\n" + brief.strip())
    if note.strip():
        parts.append("## What they want changed\n" + note.strip())
    return "\n\n".join(parts)


def _team_block(team: str, overview: str, shared: str, me: dict, everyone: list[dict]) -> str:
    """What an agent knows about its place in the team, on top of its archetype's way
    of working. Marked so a later change can find it."""
    names = {a["id"]: a["name"] for a in everyone}
    lines = [_TEAM_OPEN, f"## Your job on this team\n{me['mandate'].strip()}"]
    if me.get("produces", "").strip():
        lines.append(f"**What you deliver:** {me['produces'].strip()}")
    uses = [names[i] for i in me.get("consumes_from") or [] if i in names]
    if uses:
        lines.append(f"**You build on the work of:** {', '.join(uses)}")
    if me.get("hand_off_notes", "").strip():
        lines.append(f"**How teammates get your work:** {me['hand_off_notes'].strip()}")
    others = [a for a in everyone if a["id"] != me["id"]]
    if others:
        roster = "\n".join(f"- {a['name']}: {' '.join(a['mandate'].split())[:220]}" for a in others)
        lines.append(f"### Your teammates\n{roster}\n\nAsk one with `consult_peer`, using the name above as `agent_name`.")
    if overview.strip():
        lines.append(f"### How the team works\n{overview.strip()}")
    lines.append(
        f"### What the team shares\nThe team's folder is `vfs:{team}/`. Files you make for teammates go there, and "
        f"what the person gave when this team was set up is in `vfs:{team}/inputs/`. The `datastore` tool is the team's "
        "one shared store: every teammate reads and writes the same tables, so keep structured data there, under a "
        "stable key, and check what is already there before starting.")
    if shared.strip():
        lines.append(f"### What the team agreed\n{shared.strip()}")
    lines.append(_TEAM_CLOSE)
    return "\n\n".join(lines)


def _compose(base: str, block: str) -> str:
    """Archetype + team, within the instruction limit: its way of working gives way
    first, because the job is why it exists."""
    room = _MAX_INSTRUCTIONS - len(block) - 2
    return f"{base[:max(0, room)].rstrip()}\n\n{block}".strip()[:_MAX_INSTRUCTIONS]


def _contract(connect: str) -> str:
    path = vr._INSTRUCTIONS_DIR / "forge" / "app_contract.md"
    pieces = _PART.split(path.read_text())
    parts = dict(zip(pieces[1::2], pieces[2::2], strict=False))
    wanted = ["intro"]
    if connect in ("both", "api"):
        wanted.append("api")
    if connect in ("both", "wss"):
        wanted.append("wss")
    wanted.append("handover")
    return "\n".join(parts.get(w, "").strip() for w in wanted if parts.get(w))


def app_brief(app: dict, team: str, names: dict[str, str], has_inputs: bool) -> str:
    """What the coding agent is told: the model's brief, then the part no model writes."""
    lines = [f"# Build {app['name']}", app.get("summary", ""), f"**Why it's software:** {app.get('why', '')}",
             "## What to build", app.get("brief", "")]
    use = [f"- {names.get(s, s)}: {app.get('agent_use', {}).get(s, 'uses this app')}" for s in app.get("agents") or []]
    if use:
        lines += ["## The agents that use it", "\n".join(use)]
    if app.get("endpoints") and app.get("connect") != "wss":
        lines += ["## What agents call", "\n".join(
            f"- `{e['method']} {e['path']}`: {e.get('purpose', '')}" for e in app["endpoints"])]
    if app.get("events") and app.get("connect") != "api":
        lines += ["## What the app sends agents over its socket", "\n".join(
            f"- `{e['name']}`: {e.get('purpose', '')}" for e in app["events"])]
    if has_inputs:
        lines += ["## Background", f"What the person gave when this was planned is in the attached `{team}` folder, under `inputs/`. Read it first."]
    lines.append(_contract(str(app.get("connect") or "both")))
    return "\n\n".join(x for x in lines if str(x).strip())


def _app_instructions(app: dict, name: str, slug: str, *, api_calls: int, linked: bool) -> tuple[str, str, str]:
    """(open marker, close marker, the block that tells an agent how it reaches the app)."""
    op, cl = f"<!-- forge-app:{app['slug']} -->", f"<!-- /forge-app:{app['slug']} -->"
    lines = [op, f"## The {name} app", str(app.get("summary") or "")]
    use = str((app.get("agent_use") or {}).get(slug) or "").strip()
    if use:
        lines.append(f"**You use it to:** {use}")
    if api_calls:
        lines.append(f"Call its API with the `direct_api` tool (action `call`). Its calls are registered under names starting "
                     f"`{name}:`; action `list` shows them, and the key goes with each call without you handling it.")
    if linked:
        lines.append(f"It can also hand you work directly: those turns come from the system \"{name}\". To send it something "
                     "of your own accord, use the `linked_system` tool.")
    lines.append(cl)
    return op, cl, "\n\n".join(x for x in lines if x)


def _swap_block(text: str, op: str, cl: str, block: str) -> str:
    start, end = text.find(op), text.find(cl)
    if start >= 0 and end > start:
        return f"{text[:start]}{block}{text[end + len(cl):]}"
    return f"{text.rstrip()}\n\n{block}".strip()


# ── Recommendations ───────────────────────────────────────────────────


def _delivery(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        chosen = {k for k in ("chat", "email", "telegram") if raw.get(k)}
    elif isinstance(raw, list):
        chosen = {str(x).strip().lower() for x in raw}
    else:
        chosen = set()
    if not chosen & {"chat", "email", "telegram"}:
        chosen = {"chat"}
    return {"chat": "chat" in chosen, "email": "email" in chosen, "telegram": "telegram" in chosen, "email_to": ""}


def _schedule_problem(schedule: str) -> str:
    try:
        fd_scheduler.validate_schedule(schedule, "")
        return ""
    except fd_scheduler.ScheduleError as exc:
        return str(exc)


def coerce_recommendations(raw: Any, agents: list[dict]) -> dict[str, Any]:
    """What the model said, held to what's real: agents that exist, schedules the
    scheduler can read (one it can't is kept, marked, for the person to fix)."""
    raw = raw if isinstance(raw, dict) else {}
    slugs = [a["slug"] for a in agents]
    known = set(slugs)
    schedules: list[dict] = []
    for i, s in enumerate(raw.get("schedules") or []):
        if len(schedules) >= MAX_SCHEDULES or not isinstance(s, dict):
            break
        prompt = str(s.get("prompt") or "").strip()
        name = " ".join(str(s.get("name") or "").split())[:120]
        if not prompt or not name:
            continue
        schedule = " ".join(str(s.get("schedule") or "").split())
        agent = str(s.get("agent") or "").strip()
        schedules.append({
            "id": f"s{i + 1}", "name": name, "agent": agent if agent in known else (slugs[0] if slugs else ""),
            "schedule": schedule, "schedule_problem": _schedule_problem(schedule) if schedule else "Say when it runs.",
            "prompt": prompt, "why": str(s.get("why") or "").strip(), "usage": str(s.get("usage") or "").strip(),
            "delivery": _delivery(s.get("delivery")),
        })
    apps: list[dict] = []
    taken: set[str] = set()
    for i, a in enumerate(raw.get("apps") or []):
        if len(apps) >= MAX_APPS or not isinstance(a, dict):
            break
        name = " ".join(str(a.get("name") or "").split())[:80]
        brief = str(a.get("brief") or "").strip()
        if not name or not brief:
            continue
        slug = _slugify(a.get("slug") or name, "app")
        while slug in taken:
            slug = f"{slug}-{i + 1}"
        taken.add(slug)
        users = [x for x in (str(u).strip() for u in a.get("agents") or []) if x in known] or list(slugs)
        use = a.get("agent_use") if isinstance(a.get("agent_use"), dict) else {}
        endpoints = [{"method": str(e.get("method") or "GET").strip().upper(),
                      "path": str(e.get("path") or "").strip(), "purpose": str(e.get("purpose") or "").strip()}
                     for e in a.get("endpoints") or [] if isinstance(e, dict) and str(e.get("path") or "").startswith("/")]
        events = [{"name": str(e.get("name") or "").strip(), "purpose": str(e.get("purpose") or "").strip()}
                  for e in a.get("events") or [] if isinstance(e, dict) and str(e.get("name") or "").strip()]
        connect = str(a.get("connect") or "").strip().lower()
        if connect not in ("both", "api", "wss"):
            connect = "both"
        apps.append({
            "id": f"a{i + 1}", "name": name, "slug": slug, "summary": str(a.get("summary") or "").strip(),
            "why": str(a.get("why") or "").strip(), "brief": brief, "agents": users,
            "agent_use": {k: str(v).strip() for k, v in use.items() if k in known and str(v).strip()},
            "endpoints": endpoints, "events": events, "connect": connect,
        })
    return {"usage": str(raw.get("usage") or "").strip(), "schedules": schedules, "apps": apps}


def _json_object(text: str) -> Any:
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            raw = raw[start:end + 1]
    return json.loads(raw)


def _recommend_prompt(request: str, brief: str, overview: str, agents: list[dict], crew_line: str) -> str:
    team = "\n".join(
        f"- slug `{a['slug']}`: {a['name']} ({a.get('role') or a.get('archetype')})\n"
        f"  Job: {' '.join(str(a.get('mandate') or '').split())[:900]}\n"
        f"  Delivers: {' '.join(str(a.get('produces') or '').split())[:300]}"
        for a in agents)
    parts = [f"## What they asked for\n{request.strip()}"]
    if brief.strip():
        parts.append(f"## What their files say\n{brief.strip()[:_BRIEF_CHARS]}")
    if overview.strip():
        parts.append(f"## How the team works\n{overview.strip()}")
    parts.append(f"## The agents they approved\n{team}")
    parts.append(f"## Crew on this deck\n{crew_line}")
    return "\n\n".join(parts)


async def _crew_line(user: dict) -> str:
    row = await crew_runner._user_of(user["id"])
    grant = crew_access.grant_of(row)
    if not crew_access.offered():
        return "Crew is turned off on this deck. Recommend an app only if it's truly needed, and they will need someone to enable Crew."
    if not grant.allowed:
        return "Crew isn't approved for this account yet, so an admin would have to approve it first."
    labels = [crew_harness.HARNESSES[h].label for h in grant.harnesses if h in crew_harness.HARNESSES]
    return f"They can use Crew, with: {', '.join(labels) or 'no coding CLI yet'}."


# ── Background steps ──────────────────────────────────────────────────


async def _run_plan(sid: str, user: dict, *, note: str = "", answers: list[dict] | None = None) -> None:
    """Vatra's planning for standing agents. Ends at the approval gate, or at FAILED
    with the reason; an examiner or planner that fails doesn't stop it."""
    uid = user["id"]
    db = get_db()
    vr._run_sid.set(sid)
    vr._RUN_USAGE[sid] = []
    unbounded = vr._run_extend_unbounded.set(True)
    stub = _stub(uid)
    try:
        sess = await db.get_basna_session(sid, uid)
        if not sess:
            return
        request = sess.get("intent") or ""
        forge = _forge_of(sess)
        previous = forge.get("draft") or []
        files = [f for f in vr._parse_files(sess) if f.get("kind") != "generated"]
        names = [f["name"] for f in files]
        tiers, env = await vr._load_owner_tiers(db, uid)
        creds = vr._resolve_creds(vr._load_registry(), tiers, "", "reason")
        vr._progress_start(sid)

        brief = str(forge.get("brief") or "")
        if files and not brief:
            vr._phase(sid, "Reading your files")
            vr._progress(sid, "brief", f"Opening {len(files)} file{'s' if len(files) != 1 else ''}…")
            try:
                brief = await vr._examine_files_brief(
                    stub, user, name=f"vatra-{sid[:8]}-forge-brief", intent=request, input_files=files,
                    src_dir=vr._session_files_dir(sid), tiers=tiers or None, api_key="", env_vars=env or None,
                    timeout=_DISPATCH_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 — the plan can go on from their words
                log.warning("Agent forge: file examiner failed", plan=sid, error=str(exc))
            if not brief:
                vr._progress(sid, "note", "Couldn't read the files; planning from what you wrote.", ok=False)
            await _save(sid, uid, brief=brief)

        vr._phase(sid, "Lead · which agents you need")
        vr._progress(sid, "route", "Lead deciding which agents you need…")
        clarifications = vr._format_clarifications(answers)
        room = _room(user) or 1
        await _save(sid, uid, room=room)
        route = await vr._build_plan(
            db, uid, _lead_task(request, brief, note, clarifications, previous, names), room, creds,
            shared_datastore=True, timeout=vr._decompose_timeout(), lead_prompt="forge/lead.md")
        counted = await _count_tokens(sid, uid, 0)
        vr._progress(sid, "route", f"{len(route['subtasks'])} agent{'s' if len(route['subtasks']) != 1 else ''} · {route['domain']}")

        vr._phase(sid, "Long Horizon Planner · each agent's job")
        archetypes = await merged_archetypes(db, uid)
        by_id = {a["id"]: a for a in archetypes}
        try:
            plan = await vr._run_group0_planner(
                stub, user, sid, intent=_planner_task(request, brief, note),
                shared_context=route.get("shared_context", ""), file_names=names, subtasks=route["subtasks"],
                arch_by_id=by_id, tiers=tiers or None, api_key="", env_vars=env or None,
                timeout=_DISPATCH_TIMEOUT, clarifications=clarifications)
        except HTTPException as exc:
            vr._progress(sid, "note", f"Planner unavailable ({exc.detail}); using the Lead's plan.", ok=False)
            plan = vr._passthrough_group0_plan(route["subtasks"], by_id)
        await _count_tokens(sid, uid, counted)

        route["group0_plan"] = plan
        await _save(sid, uid, status=GATE, route=route, error="", team=_slugify(route.get("domain") or ""),
                    draft=[{"name": s.get("title", ""), "mandate": s.get("brief", "")} for s in route["subtasks"]])
        vr._emit_awaiting_plan(sid, plan)
        await _bell(uid, "agent_forge", "Your team plan is ready",
                    f"{len(route['subtasks'])} agents planned. Open it to change anything and approve.", sid)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a plan always ends somewhere the person can see
        message = exc.detail if isinstance(exc, HTTPException) else vr._lead_error_msg(exc)
        log.warning("Agent forge planning failed", plan=sid, error=str(message))
        vr._progress(sid, "route", f"Planning stopped: {str(message)[:300]}", ok=False)
        vr._progress_done(sid)
        await _save(sid, uid, status=FAILED, error=str(message)[:1000])
        await _bell(uid, "agent_forge_error", "Planning your team stopped", str(message), sid)
    finally:
        # A planning agent that never got going holds an entry, and a place in the plan's agent count.
        vr._teardown([f"vatra-{sid[:8]}-forge-brief", f"vatra-{sid[:8]}-planner"])
        vr._run_extend_unbounded.reset(unbounded)
        vr._RUN_USAGE.pop(sid, None)


async def _spawn_agent(user: dict, *, name: str, description: str, archetype: str, instructions: str,
                       env: list[dict]) -> Any:
    """A name is a folder, and another account's agent can already hold it: try once
    more with a short suffix rather than make someone guess, as the New agent dialog does."""
    from captain_claw.flight_deck.server import AgentConfig, spawn_process

    stub = _stub(user["id"])
    for attempt in (name, f"{name} {secrets.token_hex(2)}"):
        config = AgentConfig(name=attempt, description=description, archetype=archetype,
                             fleet_instructions=instructions, env_vars=env,
                             botport_enabled=False, web_enabled=True, web_port=0)
        try:
            return await spawn_process(config, stub, user), attempt
        except HTTPException as exc:
            if not re.search(r"already (exists|running)", str(exc.detail), re.I):
                raise
    raise HTTPException(409, f"An agent named '{name}' already exists.")


def _free_names(wanted: list[str]) -> list[str]:
    from captain_claw.flight_deck.server import _load_process_registry, _slug

    taken = set(_load_process_registry())
    out: list[str] = []
    for base in wanted:
        name, n = base, 2
        while _slug(name) in taken:
            name, n = f"{base} {n}", n + 1
        taken.add(_slug(name))
        out.append(name)
    return out


async def _team_folder(uid: str, team: str, sid: str) -> str:
    from captain_claw.flight_deck.vfs_routes import _user_root

    root = _user_root(uid)
    name = team or f"forge-{sid[:8]}"
    if (root / name).exists():
        name = f"{name}-{sid[:4]}"
    (root / name).mkdir(parents=True, exist_ok=True)
    return name


async def _run_create(sid: str, user: dict) -> None:
    """Make the approved agents, one at a time; one that fails doesn't stop the rest.
    Safe to run again: what was made stays made."""
    uid = user["id"]
    db = get_db()
    try:
        sess = await db.get_basna_session(sid, uid)
        if not sess:
            return
        route = _route_of(sess)
        forge = route.get("forge") or {}
        approved: list[dict] = forge.get("approved") or []
        made: dict[str, dict] = {a["id"]: a for a in forge.get("created") or []}
        vr._progress_start(sid)
        vr._phase(sid, "Creating your agents")
        folder = forge.get("folder") or await _team_folder(uid, str(forge.get("team") or ""), sid)
        files = [f for f in vr._parse_files(sess) if f.get("kind") != "generated"]
        if files and not forge.get("folder"):
            vr._save_inputs_to_vfs(uid, f"{folder}/inputs", vr._session_files_dir(sid), files)
        await _save(sid, uid, folder=folder)

        pending = [a for a in approved if not (made.get(a["id"]) or {}).get("ok")]
        by_id = {a["id"]: a for a in await merged_archetypes(db, uid)}
        for agent, final in zip(pending, _free_names([a["name"] for a in pending]), strict=True):
            agent["name"] = final
        for agent in approved:  # teammates are told each other's real names
            if (made.get(agent["id"]) or {}).get("ok"):
                agent["name"] = made[agent["id"]]["name"]
        env = [{"key": "CLAW_DATASTORE_VFS", "value": folder}]
        for agent in pending:
            arch = by_id.get(agent["archetype"]) or {}
            block = _team_block(folder, str(forge.get("overview") or ""), str(route.get("shared_context") or ""),
                                agent, approved)
            composed = _compose(str(arch.get("fleet_instructions") or ""), block)
            entry = {"id": agent["id"], "name": agent["name"], "archetype": agent["archetype"],
                     "role": arch.get("role") or agent["archetype"], "mandate": agent["mandate"],
                     "produces": agent.get("produces", ""), "slug": "", "ok": False, "error": "", "instructions": composed}
            vr._progress(sid, "agent", f"Creating {agent['name']}…")
            try:
                res, used = await _spawn_agent(
                    user, name=agent["name"], description=str(arch.get("role") or arch.get("description") or "")[:200],
                    archetype=agent["archetype"], instructions=composed, env=env)
                if not res.ok:
                    raise HTTPException(502, res.message or "It didn't start.")
                entry.update(slug=res.slug, name=used, ok=True)
                vr._progress(sid, "agent", f"{used} is starting", ok=True)
            except Exception as exc:  # noqa: BLE001 — the rest of the team can still be made
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                entry["error"] = str(detail)[:400]
                vr._progress(sid, "agent", f"{agent['name']} couldn't be created: {entry['error']}", ok=False)
            made[agent["id"]] = entry
            await _save(sid, uid, created=list(made.values()))
        if not any(a.get("ok") for a in made.values()):
            first = next((a["error"] for a in made.values() if a.get("error")), "No agent could be created.")
            raise HTTPException(502, first)
        # Nobody may open these for days, and teammates and schedules need them to know their jobs now.
        for entry in made.values():
            if entry.get("ok"):
                _tell_later(uid, entry["slug"], str(entry.get("instructions") or ""))
        await _save(sid, uid, status=RECOMMENDING)
        await _run_recommend(sid, user)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        log.warning("Agent forge create failed", plan=sid, error=str(message))
        vr._progress_done(sid)
        await _save(sid, uid, status=FAILED, error=str(message)[:1000])
        await _bell(uid, "agent_forge_error", "Creating your team stopped", str(message), sid)


async def _run_recommend(sid: str, user: dict) -> None:
    """One model call. If it fails the agents still stand: the plan is READY with the
    reason, and Try again is there."""
    uid = user["id"]
    db = get_db()
    vr._run_sid.set(sid)
    vr._RUN_USAGE[sid] = []
    error = ""
    recs: dict[str, Any] = {"usage": "", "schedules": [], "apps": []}
    try:
        sess = await db.get_basna_session(sid, uid)
        if not sess:
            return
        forge = _forge_of(sess)
        agents = [a for a in forge.get("created") or [] if a.get("ok")]
        vr._phase(sid, "Recommending scheduled tasks and apps")
        vr._progress(sid, "recommend", "Working out what should run by itself and what needs building…")
        tiers, _env = await vr._load_owner_tiers(db, uid)
        creds = vr._resolve_creds(vr._load_registry(), tiers, "", "reason")
        system = (vr._INSTRUCTIONS_DIR / "forge" / "recommend.md").read_text()
        prompt = _recommend_prompt(sess.get("intent") or "", str(forge.get("brief") or ""),
                                   str(forge.get("overview") or ""), agents, await _crew_line(user))
        limit = vr._decompose_timeout()
        from captain_claw.llm import Message

        prov, max_tokens = vr._provider_call(creds, temperature=0.3, default_max=8192, cap=vr._LEAD_OUTPUT_CAP,
                                             request_timeout=limit)
        resp = await asyncio.wait_for(prov.complete(messages=[
            Message(role="system", content=system), Message(role="user", content=prompt),
        ], temperature=0.3, max_tokens=max_tokens), limit)
        await _count_tokens(sid, uid, 0)
        try:
            recs = coerce_recommendations(_json_object(resp.content), agents)
        except (ValueError, TypeError) as exc:
            cut = str(getattr(resp, "finish_reason", "") or "").lower() in ("length", "max_tokens")
            error = ("The recommendations were cut off at the model's output limit." if cut
                     else "The recommendations came back unreadable.")
            log.warning("Agent forge: bad recommendations", plan=sid, error=str(exc))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        error = str(exc.detail if isinstance(exc, HTTPException) else vr._lead_error_msg(exc))[:500]
        log.warning("Agent forge recommend failed", plan=sid, error=error)
    finally:
        vr._RUN_USAGE.pop(sid, None)
    await _save(sid, uid, status=READY, recommendations=recs, recommend_error=error, error="")
    vr._progress_done(sid)
    await _bell(uid, "agent_forge", "Your team is ready",
                ("Your agents are created, but recommending what to schedule or build didn't work. Open it to try again."
                 if error else "Your agents are created. Open it to choose scheduled tasks and apps."), sid)


def _start(sid: str, coro: Any) -> None:
    vr._track_run_task(sid, asyncio.create_task(coro))


_telling: set[asyncio.Task] = set()


def _tell_later(uid: str, slug: str, instructions: str) -> None:
    task = asyncio.create_task(hand_identity(uid, slug, instructions=instructions))
    _telling.add(task)
    task.add_done_callback(_telling.discard)


# ── Reading a plan ────────────────────────────────────────────────────


def _crew_task(uid: str, task_id: str) -> dict | None:
    if not task_id:
        return None
    task = crew_store.get_store().get(uid, task_id)
    if not task:
        return {"id": task_id, "status": "gone", "note": "This task isn't on the Crew page any more."}
    return {"id": task["id"], "status": task["status"], "note": task.get("note") or "",
            "branch": task.get("branch") or "", "project_name": crew_runner.view(task)["project_name"]}


def _created_view(created: list[dict]) -> list[dict]:
    """With sign-in off the browser keeps instructions, so it needs the text; with
    accounts Flight Deck has it and the agent list brings it."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if not _fd_auth_enabled():
        return created
    return [{k: v for k, v in a.items() if k != "instructions"} for a in created]


def _view(sess: dict, uid: str) -> dict[str, Any]:
    route = _route_of(sess)
    forge = route.get("forge") or {}
    plan = route.get("group0_plan") or {}
    entries = {a.get("subtask_id"): a for a in plan.get("agents") or []}
    agents = []
    for s in route.get("subtasks") or []:
        e = entries.get(s["id"]) or {}
        agents.append({
            "id": s["id"], "name": s.get("title") or s.get("owner_archetype_id"), "archetype": s.get("owner_archetype_id"),
            "mandate": e.get("mandate") or s.get("brief") or "", "produces": e.get("produces") or "",
            "consumes_from": e.get("consumes_from") or s.get("depends_on") or [],
            "hand_off_notes": e.get("hand_off_notes") or "",
        })
    recs = dict(forge.get("recommendations") or {})
    if recs.get("apps"):
        recs["apps"] = [{**a, "task": _crew_task(uid, str(a.get("task_id") or ""))} for a in recs["apps"]]
    status = str(sess.get("status") or "")
    return {
        "id": sess["id"], "title": sess.get("title") or "", "request": sess.get("intent") or "", "status": status,
        "stalled": status in _WORKING and vr._live_run_task(sess["id"]) is None,
        "error": forge.get("error") or "", "recommend_error": forge.get("recommend_error") or "",
        "files": [{"name": f.get("name"), "size": f.get("size")} for f in vr._parse_files(sess)],
        "team": forge.get("folder") or forge.get("team") or "", "rationale": route.get("rationale") or "",
        "room": int(forge.get("room") or MAX_TEAM), "max_team": MAX_TEAM,
        "overview": forge.get("overview") or plan.get("overview") or "", "agents": agents,
        "questions": plan.get("questions") or [], "created": _created_view(forge.get("created") or []),
        "recommendations": recs or None, "updated_at": sess.get("updated_at"), "created_at": sess.get("created_at"),
    }


@router.get("/plans")
async def list_plans(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    rows = [s for s in await get_db().list_basna_sessions(user["id"]) if _is_plan(s)]
    return {"plans": [{"id": s["id"], "title": s.get("title") or "", "status": s.get("status") or "",
                       "updated_at": s.get("updated_at"), "created": sum(1 for a in _forge_of(s).get("created") or [] if a.get("ok"))}
                      for s in rows[:30]]}


@router.get("/plans/{plan_id}")
async def get_plan(plan_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return _view(await _plan_of(plan_id, user), user["id"])


# ── Asking, changing, approving ───────────────────────────────────────


@router.post("/plans", status_code=201)
async def start_plan(request: str = Form(""), files: list[UploadFile] = File(default=[]),
                     user: dict = Depends(get_current_user)) -> dict[str, Any]:
    text = request.strip()
    if not text:
        raise HTTPException(400, "Say what you need the agents for.")
    if len(text) > MAX_REQUEST_CHARS:
        raise HTTPException(413, f"That's more than {MAX_REQUEST_CHARS:,} characters. Attach the long part as a file.")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"Attach at most {MAX_FILES} files.")
    if _room(user) <= 0:
        raise HTTPException(403, "You're at your plan's agent limit. Remove an agent first, so there's room for a team.")
    await _refuse_when_spent(user["id"])
    db = get_db()
    # Read them all first, so a file over the limit refuses the request before there's a plan.
    held: list[tuple[str, str, bytes]] = []
    for uf in files:
        content = await uf.read()
        if len(content) > _MAX_FILE_BYTES:
            raise HTTPException(413, f"{uf.filename} is over 25 MB.")
        held.append((_safe_name(uf.filename or "upload"), uf.content_type or "application/octet-stream", content))
    sess = await db.create_basna_session(user["id"], text, title=text.splitlines()[0][:60],
                                         config=json.dumps({"mode": MODE, "source": "kiosk"}))
    sid = sess["id"]
    folder = vr._session_files_dir(sid)
    listed = {}
    for name, mime, content in held:
        (folder / name).write_bytes(content)
        listed[name] = {"name": name, "mime": mime, "size": len(content), "kind": "input"}
    await db.update_basna_session(sid, user["id"], status=PLANNING, files=json.dumps(list(listed.values())))
    _start(sid, _run_plan(sid, user))
    return {"id": sid, "status": PLANNING}


class ReplanBody(BaseModel):
    note: str = Field(default="", max_length=8000)
    questions: list[dict] = []


@router.post("/plans/{plan_id}/replan")
async def replan(plan_id: str, body: ReplanBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    sess = await _plan_of(plan_id, user)
    status = str(sess.get("status") or "")
    if status == PLANNING and vr._live_run_task(plan_id):
        raise HTTPException(409, "It's still planning.")
    if status not in (PLANNING, GATE, FAILED) or _forge_of(sess).get("created"):
        raise HTTPException(409, "This team has been created already. Forge a new one to plan again.")
    await _refuse_when_spent(user["id"])
    answers = vr._coerce_group0_questions(body.questions, keep_answers=True)
    await get_db().update_basna_session(plan_id, user["id"], status=PLANNING)
    _start(plan_id, _run_plan(plan_id, user, note=body.note, answers=answers))
    return {"id": plan_id, "status": PLANNING}


class DraftAgent(BaseModel):
    id: str
    name: str
    archetype: str
    mandate: str = ""
    produces: str = ""
    consumes_from: list[str] = []
    hand_off_notes: str = ""


class ApproveBody(BaseModel):
    agents: list[DraftAgent] = []
    overview: str = ""


@router.post("/plans/{plan_id}/approve")
async def approve(plan_id: str, body: ApproveBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """The team as the person left it: edits kept, removed agents gone. Agents already
    made stay made, so this is also how a creation cut short by a restart goes on."""
    sess = await _plan_of(plan_id, user)
    status = str(sess.get("status") or "")
    forge = _forge_of(sess)
    if vr._live_run_task(plan_id):
        raise HTTPException(409, "It's still working on this.")
    if status not in (GATE, CREATING, FAILED) or (status == FAILED and not forge.get("approved")):
        raise HTTPException(409, "This plan isn't waiting for approval.")
    if status == GATE:
        if not body.agents:
            raise HTTPException(400, "Keep at least one agent.")
        known = {a["id"] for a in await merged_archetypes(get_db(), user["id"])}
        real = {s["id"] for s in _route_of(sess).get("subtasks") or []}
        kept = {a.id for a in body.agents}
        approved = []
        for a in body.agents:
            name = " ".join(a.name.split())[:60]
            if a.id not in real:
                raise HTTPException(400, "That agent isn't in this plan.")
            if not name:
                raise HTTPException(400, "Every agent needs a name.")
            if a.archetype not in known:
                raise HTTPException(400, f"There's no archetype '{a.archetype}'.")
            if not a.mandate.strip():
                raise HTTPException(400, f"Say what {name} is for.")
            approved.append({"id": a.id, "name": name, "title": name, "archetype": a.archetype,
                             "mandate": a.mandate.strip(), "produces": a.produces.strip(),
                             "consumes_from": [c for c in a.consumes_from if c in kept and c != a.id],
                             "hand_off_notes": a.hand_off_notes.strip()})
        await _save(plan_id, user["id"], status=CREATING, approved=approved, overview=body.overview.strip(), error="")
    else:
        await _save(plan_id, user["id"], status=CREATING, error="")
    _start(plan_id, _run_create(plan_id, user))
    return {"id": plan_id, "status": CREATING}


@router.post("/plans/{plan_id}/recommend")
async def recommend_again(plan_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    sess = await _plan_of(plan_id, user)
    if vr._live_run_task(plan_id):
        raise HTTPException(409, "It's still working on this.")
    if not any(a.get("ok") for a in _forge_of(sess).get("created") or []):
        raise HTTPException(409, "Create the team first.")
    done = (_forge_of(sess).get("recommendations") or {})
    if any(s.get("job_id") for s in done.get("schedules") or []) or any(a.get("task_id") for a in done.get("apps") or []):
        raise HTTPException(409, "You've already acted on these, so they stay as they are.")
    await _refuse_when_spent(user["id"])
    vr._progress_start(plan_id)
    await _save(plan_id, user["id"], status=RECOMMENDING, recommend_error="")
    _start(plan_id, _run_recommend(plan_id, user))
    return {"id": plan_id, "status": RECOMMENDING}


@router.delete("/plans/{plan_id}")
async def forget_plan(plan_id: str, user: dict = Depends(get_current_user)) -> dict[str, bool]:
    await _plan_of(plan_id, user)
    if vr._live_run_task(plan_id):
        raise HTTPException(409, "It's still working on this.")
    import shutil

    await get_db().delete_basna_session(plan_id, user["id"])
    shutil.rmtree(vr._session_files_dir(plan_id), ignore_errors=True)
    _locks.pop(plan_id, None)
    return {"ok": True}


# ── Making what was recommended ───────────────────────────────────────


class ScheduleChoice(BaseModel):
    id: str
    name: str
    agent: str
    schedule: str
    prompt: str
    delivery: dict = {}


class SchedulesBody(BaseModel):
    items: list[ScheduleChoice]
    tz: str = ""


@router.post("/plans/{plan_id}/schedules")
async def make_schedules(plan_id: str, body: SchedulesBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Each one the person kept, as they left it. One that's wrong says why and doesn't
    stop the rest; one already made isn't made twice."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    uid = user["id"]
    sess = await _plan_of(plan_id, user)
    forge = _forge_of(sess)
    recs = forge.get("recommendations") or {}
    slugs = {a["slug"] for a in forge.get("created") or [] if a.get("ok")}
    by_id = {s["id"]: s for s in recs.get("schedules") or []}
    store = fd_scheduler.get_store()
    results = []
    for item in body.items:
        rec = by_id.get(item.id)
        if rec is None:
            results.append({"id": item.id, "ok": False, "error": "That isn't one of the recommendations."})
            continue
        if rec.get("job_id"):
            results.append({"id": item.id, "ok": True, "job_id": rec["job_id"]})
            continue
        if item.agent not in slugs:
            results.append({"id": item.id, "ok": False, "error": "Choose one of this team's agents."})
            continue
        try:
            job = store.create(owner_id=uid if _fd_auth_enabled() else "", prompt_author="human", name=item.name.strip(),
                               schedule=item.schedule.strip(), tz=body.tz.strip(), agent_slug=item.agent,
                               prompt=item.prompt.strip(), delivery_kind="agent", delivery=_delivery(item.delivery), enabled=True)
        except ValueError as exc:  # ScheduleError is one
            results.append({"id": item.id, "ok": False, "error": str(exc)})
            continue
        rec.update(job_id=job["id"], name=item.name.strip(), agent=item.agent, schedule=item.schedule.strip(),
                   prompt=item.prompt.strip(), delivery=_delivery(item.delivery), schedule_problem="")
        results.append({"id": item.id, "ok": True, "job_id": job["id"]})
    await _save(plan_id, uid, recommendations=recs)
    return {"results": results}


class BuildBody(BaseModel):
    harness: str
    brief: str = ""
    model: str = ""
    effort: str = ""
    model_source: dict | None = None
    deliver: str = "branch"
    connect: str = ""


def _app_of(recs: dict, app_id: str) -> dict:
    for app in recs.get("apps") or []:
        if app.get("id") == app_id:
            return app
    raise HTTPException(404, "That app isn't in this plan.")


@router.post("/plans/{plan_id}/apps/{app_id}/build")
async def build_app(plan_id: str, app_id: str, body: BuildBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Crew's ordinary task, under Crew's own approval, with the contract in its brief.
    The first agent that uses the app is the one called back when it's done."""
    uid = user["id"]
    sess = await _plan_of(plan_id, user)
    forge = _forge_of(sess)
    recs = forge.get("recommendations") or {}
    app = _app_of(recs, app_id)
    row, grant = await crew_routes._approved(user)
    if app.get("task_id"):
        task = crew_store.get_store().get(uid, app["task_id"])
        if task and task["status"] not in ("failed", "stopped", "closed"):
            return {"task": _crew_task(uid, app["task_id"])}
    if body.brief.strip():
        app["brief"] = body.brief.strip()
    if body.connect in ("both", "api", "wss"):
        app["connect"] = body.connect
    names = {a["slug"]: a["name"] for a in forge.get("created") or [] if a.get("ok")}
    folder = str(forge.get("folder") or "")
    has_inputs = bool(vr._parse_files(sess)) and bool(folder)
    fields: dict[str, Any] = {
        "harness": body.harness, "brief": app_brief(app, folder, names, has_inputs),
        "title": f"Build {app['name']}", "kind": "ship", "deliver": body.deliver or "branch", "project": app["slug"],
        "model": body.model, "effort": body.effort, "model_source": body.model_source,
        "agent_slug": next((s for s in app.get("agents") or [] if s in names), ""),
        "source_host": "localhost", "origin": {"platform": "web", "user_id": "", "chat_id": 0},
        "vfs_folders": [{"project": folder}] if has_inputs else [],
    }
    task = await crew_runner.dispatch(row, grant, fields)
    app["task_id"] = task["id"]
    await _save(plan_id, uid, recommendations=recs)
    return {"task": _crew_task(uid, task["id"])}


class ConnectBody(BaseModel):
    url: str
    key: str
    agents: list[str] = []


def _ws_address(base: str, path: str) -> str:
    parts = urlsplit(base)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, f"{parts.path.rstrip('/')}/{path.lstrip('/')}", "", ""))


async def _read_manifest(base: str, key: str) -> dict:
    import httpx

    from captain_claw.flight_deck.agent_link_http import address_problem

    problem = await address_problem(base)
    if problem:
        raise HTTPException(400, f"Address: {problem}")
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False, trust_env=False) as client:
            resp = await client.get(f"{base}/api/agent/manifest", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise HTTPException(400, f"Couldn't reach the app at that address: {type(exc).__name__}.") from None
    if resp.status_code in (401, 403):
        raise HTTPException(400, "The app refused that key.")
    if resp.status_code != 200:
        raise HTTPException(400, f"The app has no agent manifest at that address (it answered {resp.status_code}).")
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(400, "The app's manifest isn't JSON.") from None
    if not isinstance(data, dict):
        raise HTTPException(400, "The app's manifest isn't what it should be.")
    return data


async def _register_calls(slug: str, app: dict, base: str, key: str, endpoints: list[dict]) -> int:
    """The app's API as ``direct_api`` calls on one agent, the key stored with each.
    Whatever was registered for this app before goes first, so connecting again doesn't
    leave old ones behind."""
    import httpx

    from captain_claw.flight_deck import agent_idle
    from captain_claw.flight_deck.server import _load_process_registry

    await agent_idle.wake(slug)
    entry = _load_process_registry().get(slug) or {}
    port, token = int(entry.get("web_port") or 0), str(entry.get("web_auth") or "")
    if not port:
        raise HTTPException(409, "It isn't running.")
    params = {"token": token} if token else {}
    agent = f"http://127.0.0.1:{port}/api/direct-api-calls"
    made = 0
    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        listed = await client.get(agent, params=params)
        items = listed.json() if listed.status_code == 200 else []
        items = items.get("items", []) if isinstance(items, dict) else items
        for old in items:
            if isinstance(old, dict) and old.get("app_name") == app["slug"]:
                await client.delete(f"{agent}/{old.get('id')}", params=params)
        for ep in endpoints[:_APP_CALLS_MAX]:
            method = str(ep.get("method") or "GET").upper()
            path = str(ep.get("path") or "")
            if method == "DELETE" or not path.startswith("/"):
                continue
            resp = await client.post(agent, params=params, json={
                "name": f"{app['name']}: {method} {path}", "url": f"{base}{path}", "method": method,
                "description": str(ep.get("purpose") or ""),
                "input_payload": json.dumps(ep.get("input") or {}), "result_payload": json.dumps(ep.get("output") or {}),
                "auth_type": "bearer", "auth_token": key, "auth_source": "manual", "app_name": app["slug"],
            })
            if resp.status_code < 300:
                made += 1
    return made


@router.post("/plans/{plan_id}/apps/{app_id}/connect")
async def connect_app(plan_id: str, app_id: str, body: ConnectBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """The running app, and the agents that use it. An http(s) address is asked for its
    manifest, which gives the API's calls and the socket's path; a ws(s) address is a
    socket and nothing else. The key goes into a sealed link and the agents' stored
    calls, not into this plan."""
    from captain_claw.flight_deck import agent_link
    from captain_claw.flight_deck.server import _owner_agent_instructions, _set_owner_agent_instructions

    uid = user["id"]
    sess = await _plan_of(plan_id, user)
    forge = _forge_of(sess)
    recs = forge.get("recommendations") or {}
    app = _app_of(recs, app_id)
    url, key = body.url.strip().rstrip("/"), body.key.strip()
    if not url or not key:
        raise HTTPException(400, "Give the app's address and its key.")
    names = {a["slug"]: a["name"] for a in forge.get("created") or [] if a.get("ok")}
    chosen = [s for s in (body.agents or app.get("agents") or []) if s in names]
    if not chosen:
        raise HTTPException(400, "Choose an agent of this team to connect.")
    scheme = urlsplit(url).scheme
    if scheme in ("ws", "wss"):
        socket, base, endpoints = url, "", []
    elif scheme in ("http", "https"):
        manifest = await _read_manifest(url, key)
        base = url
        endpoints = [e for e in manifest.get("endpoints") or [] if isinstance(e, dict)]
        socket = _ws_address(url, str(manifest["ws"])) if manifest.get("ws") else ""
    else:
        raise HTTPException(400, "Use an https:// or wss:// address.")
    if not socket and not endpoints:
        raise HTTPException(400, "The app's manifest lists no endpoints and no socket, so there's nothing to connect to.")

    earlier = (app.get("connection") or {}).get("links") or {}
    links: dict[str, str] = {}
    results = []
    instructions: dict[str, str] = {}
    stored = await _owner_agent_instructions(uid, "process")
    made_text = {a["slug"]: str(a.get("instructions") or "") for a in forge.get("created") or [] if a.get("ok")}
    for slug in chosen:
        result: dict[str, Any] = {"slug": slug, "name": names[slug], "linked": False, "api_calls": 0, "error": ""}
        try:
            if socket:
                fields = agent_link.LinkBody(
                    name=f"{app['name']} · {names[slug]}"[:80], agent_slug=slug, agent_may_send=True, direction="outbound",
                    remote_url=socket, remote_key=key,
                    instructions=f"This is {app['name']}, built for you. {(app.get('agent_use') or {}).get(slug, '')}".strip())
                if earlier.get(slug):
                    await agent_link.change_link(earlier[slug], fields, user)
                    links[slug] = earlier[slug]
                else:
                    links[slug] = (await agent_link.create_link(fields, user))["id"]
                result["linked"] = True
            if endpoints:
                result["api_calls"] = await _register_calls(slug, app, base, key, endpoints)
            op, cl, block = _app_instructions(app, app["name"], slug, api_calls=result["api_calls"], linked=result["linked"])
            text = _swap_block(stored.get(slug) or made_text.get(slug, ""), op, cl, block)[:_MAX_INSTRUCTIONS]
            await _set_owner_agent_instructions(uid, "process", slug, text, strict=True)
            instructions[slug] = text
            for made in forge.get("created") or []:  # with sign-in off this is the only copy Flight Deck has
                if made.get("slug") == slug:
                    made["instructions"] = text
            _tell_later(uid, slug, text)
        except HTTPException as exc:
            result["error"] = str(exc.detail)[:300]
        except Exception as exc:  # noqa: BLE001 — the other agents can still connect
            log.warning("Agent forge connect failed", agent=slug, error=str(exc))
            result["error"] = "Connecting this agent failed."
        results.append(result)
    app["connection"] = {"url": url, "socket": socket, "agents": [r["slug"] for r in results if not r["error"]],
                         "links": {**earlier, **links}, "api_calls": sum(r["api_calls"] for r in results), "at": time.time()}
    await _save(plan_id, uid, recommendations=recs, created=forge.get("created") or [])
    return {"results": results, "instructions": instructions, "connection": app["connection"]}
