"""Council of Agents REST endpoints for Flight Deck."""

from __future__ import annotations

import asyncio
import json
import shutil
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.agent_stuck import STUCK_MARKERS
from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

# Reuse Basna's shared archetype spine — catalog rendering, the deterministic
# keyword fallback, the registry/tier resolvers, and the instructions dir. The
# Council assembler differs only in its router prompt (favour diverse deliberation
# voices over a minimal single-answer team) and in how the spawned agents are
# handed back to the frontend-driven deliberation loop.
from captain_claw.flight_deck.basna_routes import (
    _INSTRUCTIONS_DIR,
    _build_catalog,
    _keyword_match,
    _load_registry,
)

log = get_logger(__name__)

router = APIRouter(prefix="/fd/council", tags=["council"])

# A Council panelist must never be able to launch a nested orchestration run.
_WORKER_MARKER = "CLAW_COUNCIL_WORKER"


@router.get("/stuck-markers")
async def stuck_markers(user: dict = Depends(get_current_user)):
    """Substrings that identify an agent's canned 'give-up' reply.

    Single source: captain_claw/agent_stuck.py. The Council UI fetches these so
    the stuck-detection strings aren't duplicated in the frontend.
    """
    return {"markers": STUCK_MARKERS}


# ── Request models ───────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str
    topic: str
    session_type: str = "brainstorm"
    verbosity: str = "message"
    max_rounds: int = Field(default=5, ge=1, le=20)
    moderator_mode: str = "round-robin"
    moderator_agent: str = ""
    agents: str = "[]"
    config: str = "{}"


class AssembleRequest(BaseModel):
    """Auto-assemble a council panel from archetypes modeled to the topic."""
    topic: str
    session_type: str = "brainstorm"
    max_agents: int = Field(default=4, ge=2, le=8)
    # Optional fixed panel — when set, the assembler uses exactly these archetypes.
    archetype_ids: list[str] | None = None
    # Library tier config: tier-name -> {provider, model, api_key, base_url, ...}.
    # Spawned panelists resolve their model/key from here; missing tiers fall back
    # to the registry tier defaults + the provider env var.
    tiers: dict | None = None
    # Library "Additional API Keys" passed to every panelist. [{key, value}].
    env_vars: list[dict] | None = None
    # Fallback key when a tier omits one (empty -> provider env var).
    api_key: str = ""
    # Router-call overrides (defaults to the registry's fast tier).
    provider: str = ""
    model: str = ""
    base_url: str = ""
    max_tokens: int = Field(default=2048, ge=256, le=8192)


class TeardownRequest(BaseModel):
    """Stop + remove ephemeral panelists spawned by /assemble."""
    slugs: list[str]


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    topic: str | None = None
    status: str | None = None
    current_round: int | None = None
    moderator_mode: str | None = None
    moderator_agent: str | None = None
    agents: str | None = None
    pinned_ids: str | None = None
    config: str | None = None


class AddMessagesRequest(BaseModel):
    messages: list[dict]


class UpdateMessageRequest(BaseModel):
    content: str | None = None
    action: str | None = None
    suitability: float | None = None
    target_agent_id: str | None = None
    metadata: str | None = None


class AddVotesRequest(BaseModel):
    votes: list[dict]


class UpsertArtifactRequest(BaseModel):
    kind: str
    agent_id: str = ""
    agent_name: str = ""
    content: str = ""


# ── Auto-assemble: route → spawn archetype panelists ─────────────

def _council_fleet(arch: dict, role: str, topic: str, session_type: str, why: str) -> str:
    """Compose a panelist's identity = its archetype persona + a council seat
    tailored to THIS topic. The frontend pushes this into the agent on connect
    (as `self.fleet_instructions`), so each spawned voice arrives already framed
    for the deliberation rather than as a blank general agent."""
    parts: list[str] = []
    base = (arch.get("fleet_instructions") or "").strip()
    if base:
        parts.append(base)
    parts.append(
        f"## Your seat on this council\n"
        f"You are the **{role}** on a council deliberating this topic:\n\n> {topic}\n")
    if why:
        parts.append(f"Your specific charge in this {session_type} session: {why}\n")
    parts.append(
        "Bring your distinct perspective. Engage directly with the other panelists — "
        "build on, challenge, or refine their points rather than restating your own — "
        "and keep every contribution concrete and grounded.")
    return "\n".join(parts)


async def _spawn_panelist(request: Request, user: dict, *, name: str, description: str,
                          cognitive_mode: str, tools: list[str], tier: str,
                          tiers: dict | None, api_key: str,
                          env_vars: list[dict] | None) -> dict:
    """Spawn one ephemeral panelist and resolve its web port. Mirrors the
    Basna/Vatra worker spawn: strips the run-starting orchestration tools and
    stamps the no-recursion marker so a panelist can never launch another run.
    Returns {ok, slug, port, auth, message}."""
    from captain_claw.flight_deck.server import (
        AgentConfig, spawn_process, _load_process_registry,
    )
    lt = (tiers or {}).get(tier) or {}
    provider = lt.get("provider") or ""
    model = lt.get("model") or ""
    key = lt.get("api_key") or api_key or ""
    base_url = lt.get("base_url") or ""
    max_tokens = int(lt.get("output_ctx") or 0) or 32768
    max_context = int(lt.get("input_ctx") or 0)
    worker_tools = [t for t in (tools or AgentConfig().tools) if t not in ("basna", "vatra")]
    base = dict(
        name=name, description=description,
        cognitive_mode=cognitive_mode or "neutra", tools=worker_tools,
        env_vars=(env_vars or []) + [{"key": _WORKER_MARKER, "value": "1"}],
        web_enabled=True, web_port=0,
    )
    if model:
        cfg = AgentConfig(**base, tier="", provider=provider or "", model=model,
                          provider_api_key=key, base_url=base_url,
                          max_tokens=max_tokens, max_context=max_context)
    else:
        cfg = AgentConfig(**base, tier=tier, provider_api_key=key)
    res = await spawn_process(cfg, request, user)
    reg = _load_process_registry()
    entry = reg.get(res.slug) or {}
    port = entry.get("web_port")
    if not res.ok or not port:
        return {"ok": False, "slug": res.slug, "port": 0, "auth": "",
                "message": res.message or "no port"}
    return {"ok": True, "slug": res.slug, "port": port,
            "auth": entry.get("web_auth", ""), "message": ""}


def _teardown(slugs: list[str]) -> None:
    """Fully remove ephemeral panelists so they don't pile up in the fleet."""
    from captain_claw.flight_deck.server import (
        _do_stop_process, _load_process_registry, _save_process_registry,
        _processes, DATA_DIR,
    )
    for slug in slugs:
        try:
            _do_stop_process(slug)
        except Exception as e:
            log.warning("Council teardown stop failed", slug=slug, error=str(e))
    if not slugs:
        return
    reg = _load_process_registry()
    for slug in slugs:
        reg.pop(slug, None)
        _processes.pop(slug, None)
        try:
            shutil.rmtree(DATA_DIR / slug, ignore_errors=True)
        except Exception:
            pass
    _save_process_registry(reg)


async def _route_panel(body: AssembleRequest, arch_by_id: dict,
                       archetypes: list[dict], reliability: dict, max_agents: int) -> dict:
    """Select a diverse panel for the topic via a fast-tier LLM; on any failure
    fall back to deterministic keyword matching so assembly always proceeds.
    Returns {selected:[{archetype_id, tier, why, role}], domain, rationale,
    title, source}."""
    registry = _load_registry()
    fast = registry.get("tiers", {}).get("fast", {})
    provider = body.provider or fast.get("provider", "anthropic")
    model = body.model or fast.get("model", "")
    base_url = body.base_url or fast.get("base_url", "")

    system_file = _INSTRUCTIONS_DIR / "council" / "router.md"
    if not system_file.is_file():
        raise HTTPException(500, "Council router prompt not found")
    system_prompt = system_file.read_text() + "\n\n" + _build_catalog(archetypes, reliability)

    forced_ids = [a for a in (body.archetype_ids or []) if a in arch_by_id]
    if forced_ids:
        forced_list = "\n".join(f"- {a}: {arch_by_id[a].get('role', '')}" for a in forced_ids)
        user_prompt = (
            f"Topic: {body.topic.strip()}\nSession type: {body.session_type}\n\n"
            f"The panel is FIXED by the user — you MUST use EXACTLY these archetypes, ALL of "
            f"them and NO others, in `selected`:\n{forced_list}\n\n"
            f"For each, write a `why` instructing it specifically for THIS topic. Still choose a domain.")
    else:
        user_prompt = (
            f"Topic: {body.topic.strip()}\nSession type: {body.session_type}\n\n"
            f"max_agents: {max_agents}. Assemble the panel whose complementary, sometimes "
            f"opposing perspectives will make for the richest deliberation.")

    raw: dict | None = None
    source = "llm"
    try:
        from captain_claw.llm import create_provider, Message
        prov = create_provider(provider=provider, model=model,
                               api_key=body.api_key or None, base_url=base_url or None,
                               temperature=0.3, max_tokens=body.max_tokens)
        resp = await prov.complete(messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ], temperature=0.3, max_tokens=body.max_tokens)
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        raw = json.loads(content)
    except Exception as e:
        log.warning("Council assembler LLM failed; using keyword fallback", error=str(e))
        raw = None
        source = "fallback"

    selected: list[dict] = []
    seen: set[str] = set()

    def _add(aid: str, tier: str | None, why: str) -> None:
        if aid not in arch_by_id or aid in seen:
            return
        seen.add(aid)
        selected.append({
            "archetype_id": aid,
            "tier": (tier or arch_by_id[aid].get("tier", "balanced")),
            "why": why.strip(),
            "role": arch_by_id[aid].get("role", aid),
        })

    if isinstance(raw, dict) and raw.get("selected"):
        for s in raw["selected"]:
            if len(selected) >= max_agents:
                break
            _add(str(s.get("archetype_id") or "").strip(),
                 (s.get("tier") or None), str(s.get("why") or ""))

    if forced_ids:
        # Honor the fixed panel exactly — keep the LLM's task `why`/tier where present.
        by_sel = {x["archetype_id"]: x for x in selected}
        selected = [by_sel.get(aid) or {
            "archetype_id": aid, "tier": arch_by_id[aid].get("tier", "balanced"),
            "why": "", "role": arch_by_id[aid].get("role", aid),
        } for aid in forced_ids]

    if not selected:
        picked = _keyword_match(body.topic, archetypes, max_agents) or archetypes[:max_agents]
        for a in picked:
            _add(a["id"], a.get("tier"), "")
        source = "fallback"

    # A council needs at least two voices — backfill from keyword/registry order.
    if len(selected) < 2:
        for a in _keyword_match(body.topic, archetypes, max_agents) + archetypes:
            if len(selected) >= 2:
                break
            _add(a["id"], a.get("tier"), "")

    domain = str((raw.get("domain") if isinstance(raw, dict) else "") or "general").lower()
    # A user-fixed panel is honored in full even if it exceeds max_agents (the
    # slider is irrelevant when the user hand-picks specialists); otherwise clamp.
    cap = max(max_agents, len(forced_ids)) if forced_ids else max_agents
    return {
        "selected": selected[:cap],
        "domain": domain,
        "rationale": str((raw.get("rationale") if isinstance(raw, dict) else "") or ""),
        "title": str((raw.get("title") if isinstance(raw, dict) else "") or ""),
        "source": source,
    }


@router.post("/assemble")
async def assemble_council(
    body: AssembleRequest, request: Request, user: dict = Depends(get_current_user),
):
    """Route a diverse archetype panel for the topic and spawn each as an
    ephemeral agent, returning council-ready agent defs (with task-tailored
    fleet instructions). The frontend then creates the session over these
    panelists and runs the normal deliberation loop; on session delete it calls
    /teardown to remove them."""
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(400, "topic is required")

    db = get_db()
    archetypes = await merged_archetypes(db, user["id"])
    arch_by_id = {a["id"]: a for a in archetypes}
    if not arch_by_id:
        raise HTTPException(500, "No archetypes available to assemble a council")

    rel_rows = await db.get_archetype_reliability(user["id"])
    reliability: dict[str, list[dict]] = {}
    for r in rel_rows:
        reliability.setdefault(r["archetype_id"], []).append(r)

    max_agents = max(2, min(body.max_agents, 8))
    plan = await _route_panel(body, arch_by_id, archetypes, reliability, max_agents)
    selected = plan["selected"]
    if len(selected) < 2:
        raise HTTPException(422, "Could not assemble at least two distinct panelists for this topic.")

    run_tag = format(int(time.time()), "x")[-6:]

    async def _spawn(sel: dict):
        arch = arch_by_id[sel["archetype_id"]]
        role = sel["role"] or arch.get("role") or arch["id"]
        res = await _spawn_panelist(
            request, user,
            name=f"council-{run_tag}-{arch['id']}",
            description=f"Council panelist · {role}",
            cognitive_mode=arch.get("cognitive_mode", "neutra"),
            tools=arch.get("tools") or [],
            tier=sel["tier"], tiers=body.tiers,
            api_key=body.api_key,
            # Auto-bind the panel to a shared VFS project (vfs:<project>/...).
            env_vars=(body.env_vars or []) + [
                {"key": "CLAW_VFS_PROJECT", "value": f"council-{run_tag}"},
            ],
        )
        return sel, arch, res

    spawn_out = await asyncio.gather(*[_spawn(s) for s in selected], return_exceptions=True)

    agents: list[dict] = []
    name_counts: dict[str, int] = {}
    for item in spawn_out:
        if isinstance(item, Exception):
            log.warning("Council spawn failed", error=str(item))
            continue
        sel, arch, res = item
        if not res.get("ok"):
            log.warning("Council spawn unusable", slug=res.get("slug"), message=res.get("message"))
            continue
        role = sel["role"]
        # De-duplicate display names (two of the same role get a numeric suffix).
        n = name_counts.get(role, 0) + 1
        name_counts[role] = n
        disp = role if n == 1 else f"{role} {n}"
        agents.append({
            "id": res["slug"], "slug": res["slug"], "name": disp,
            "host": "localhost", "port": res["port"], "auth": res["auth"],
            "archetype_id": arch["id"], "role": role, "why": sel.get("why", ""),
            "cognitive_mode": arch.get("cognitive_mode", "neutra"),
            "fleet_instructions": _council_fleet(arch, role, topic, body.session_type, sel.get("why", "")),
            "is_moderator": False,
        })

    if len(agents) < 2:
        # A council needs at least two voices — don't leave a single orphan running.
        _teardown([a["slug"] for a in agents])
        raise HTTPException(502, "Could not spawn enough panelists for a council (need at least 2).")

    return {
        "domain": plan["domain"], "rationale": plan["rationale"],
        "title": plan["title"], "source": plan["source"], "agents": agents,
    }


@router.post("/teardown")
async def teardown_council(body: TeardownRequest, user: dict = Depends(get_current_user)):
    """Stop + remove ephemeral panelists spawned for an auto-assembled session.
    Idempotent and best-effort — unknown slugs are simply skipped."""
    slugs = [s for s in (body.slugs or []) if isinstance(s, str) and s.strip()]
    _teardown(slugs)
    return {"ok": True, "removed": len(slugs)}


# ── Session endpoints ────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.list_council_sessions(user["id"])


@router.post("/sessions")
async def create_session(body: CreateSessionRequest, user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.create_council_session(
        user_id=user["id"], title=body.title, topic=body.topic,
        session_type=body.session_type, verbosity=body.verbosity,
        max_rounds=body.max_rounds, moderator_mode=body.moderator_mode,
        moderator_agent=body.moderator_agent, agents=body.agents,
        config=body.config,
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    sess = await db.get_council_session(session_id, user["id"])
    if not sess:
        raise HTTPException(status_code=404, detail="Council session not found")
    return sess


@router.put("/sessions/{session_id}")
async def update_session(
    session_id: str, body: UpdateSessionRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    ok = await db.update_council_session(session_id, user["id"], **fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    deleted = await db.delete_council_session(session_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(session_id: str, user: dict = Depends(get_current_user)):
    """Stop a council deliberation — marks it cancelled so no further rounds run.
    Council uses already-running agents (nothing ephemeral to kill), so a status
    flag is the stop. The arbiter's stop_run and the UI Stop button both hit this."""
    db = get_db()
    ok = await db.update_council_session(session_id, user["id"], status="cancelled")
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "status": "cancelled"}


# ── Message endpoints ────────────────────────────────────────────

@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str, round: int | None = None, limit: int = 500,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_council_messages(session_id, user["id"], round_num=round, limit=limit)


@router.post("/sessions/{session_id}/messages")
async def add_messages(
    session_id: str, body: AddMessagesRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ids = await db.add_council_messages(session_id, user["id"], body.messages)
    if not ids:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "ids": ids}


@router.put("/sessions/{session_id}/messages/{message_id}")
async def update_message(
    session_id: str, message_id: int, body: UpdateMessageRequest,
    user: dict = Depends(get_current_user),
):
    """Patch a message — used to checkpoint/finalize a streaming agent turn."""
    db = get_db()
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    ok = await db.update_council_message(session_id, user["id"], message_id, fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Message or session not found")
    return {"ok": True}


@router.delete("/sessions/{session_id}/messages")
async def delete_messages(
    session_id: str, round: int,
    user: dict = Depends(get_current_user),
):
    """Delete all messages for a round so it can be re-run (restart round)."""
    db = get_db()
    removed = await db.delete_council_messages(session_id, user["id"], round)
    if removed < 0:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "removed": removed}


@router.put("/sessions/{session_id}/messages/{message_id}/pin")
async def toggle_pin(
    session_id: str, message_id: int,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ok = await db.toggle_council_pin(session_id, user["id"], message_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}


# ── Vote endpoints ───────────────────────────────────────────────

@router.post("/sessions/{session_id}/votes")
async def add_votes(
    session_id: str, body: AddVotesRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ids = await db.add_council_votes(session_id, user["id"], body.votes)
    if not ids:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "ids": ids}


@router.get("/sessions/{session_id}/votes")
async def get_votes(
    session_id: str, round: int | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_council_votes(session_id, user["id"], round_num=round)


# ── Artifact endpoints ──────────────────────────────────────────

@router.get("/sessions/{session_id}/artifacts")
async def get_artifacts(
    session_id: str, kind: str | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_council_artifacts(session_id, user["id"], kind=kind)


@router.post("/sessions/{session_id}/artifacts")
async def upsert_artifact(
    session_id: str, body: UpsertArtifactRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    art_id = await db.upsert_council_artifact(
        session_id, user["id"],
        kind=body.kind, agent_id=body.agent_id,
        agent_name=body.agent_name, content=body.content,
    )
    if not art_id:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "id": art_id}


@router.delete("/sessions/{session_id}/artifacts")
async def delete_artifacts(
    session_id: str, kind: str | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ok = await db.delete_council_artifacts(session_id, user["id"], kind=kind)
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}
