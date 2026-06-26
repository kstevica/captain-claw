"""Vatra — the collaborative sibling of Basna.

Where Basna spawns independent agents that each answer the whole task and then
*merges* their uncorrelated outputs, Vatra runs a **collaborating team**: a Lead
decomposes the task into complementary subtasks, specialists each produce one
piece in parallel, and a dedicated **reporter** assembles the pieces into one
coherent deliverable. There is no merge and no reliability weighting of a single
answer — the contributions are interdependent by design.

Phase 1 (this module): Lead decompose → parallel subtasks → reporter assembles.
No delegation between agents yet (that's Phase 2's blackboard/ask protocol), and
no learning loop yet (Phase 3). The shared spine — dispatch, progress, notify,
creds, the archetype catalog — is imported from ``basna_routes``; only the
*coordinator* is new here. Spawn/teardown are mirrored locally for this first cut
so Basna's working path is untouched; unifying them into one helper is a
fast-follow once Phase 1 validates.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import types
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

# Reuse Basna's standalone spine — these are pure/side-effect-isolated helpers and
# the shared run-tracking + notify plumbing. Only the coordinator below is new.
from captain_claw.flight_deck.basna_routes import (
    AgentStartReq,
    ExecuteRequest,
    _AgentReq,
    _AGENT_RUN_WINDOW_SECONDS,
    _MAX_AGENT_RUNS_PER_OWNER,
    _MAX_AGENT_RUNS_PER_WINDOW,
    _INSTRUCTIONS_DIR,
    _PROGRESS,
    _active_agent_runs,
    _agent_run_starts,
    _agent_run_tasks,
    _basna_agent_tasks,
    _build_catalog,
    _dispatch_one,
    _guess_mime,
    _is_texty,
    _keyword_match,
    _llm_judge,
    _load_owner_tiers,
    _load_registry,
    _norm_text,
    _notify_source_agent,
    _parse_files,
    _progress,
    _progress_done,
    _progress_start,
    _provider_call,
    _resolve_owner,
    _run_workers,
    _session_files_dir,
    _tier_creds,
)

log = get_logger(__name__)

router = APIRouter(prefix="/fd/vatra", tags=["vatra"])

# A Vatra worker, like a Basna worker, must never be able to start another run.
_WORKER_MARKER = "CLAW_VATRA_WORKER"
# Default reporter archetype (a strong general writer); overridable via config.
_DEFAULT_REPORTER = "editor-writer"
# Inline this many chars of the slices into the reporter prompt; the full set is
# always also written to its workspace as a file it can read on demand.
_SLICES_INLINE_CHARS = 12_000
_MAX_TEXT_FALLBACK_BYTES = 256 * 1024
_DECOMPOSE_TIMEOUT = 120  # seconds for the Lead LLM call

# ── Phase 2 delegation budget (the termination guarantees) ───────────
_MAX_ASKS = 12          # total asks a single run may ever create (hard ceiling)
_MAX_ASK_DEPTH = 2      # an answer that itself asks increments depth; caps cascades
_MAX_HELPERS = 3        # concurrent helpers the coordinator may run at once
_COORD_POLL_S = 1.5     # how often the coordinator polls the blackboard
_INBOX_POLL_S = 1.0     # inbox long-poll granularity


def _vatra_env(sid: str, subtask: str, owner: str, depth: int) -> list[dict]:
    """Run-context env injected into a worker so the `vatra` tool knows where it is."""
    return [
        {"key": "CLAW_VATRA_SESSION", "value": sid},
        {"key": "CLAW_VATRA_SUBTASK", "value": subtask},
        {"key": "CLAW_VATRA_OWNER", "value": owner},
        {"key": "CLAW_VATRA_DEPTH", "value": str(depth)},
    ]


def _track_worker(sid: str, slug: str, *, add: bool) -> None:
    """Best-effort registry of a run's live worker slugs (so an external stop can
    kill them). Concurrency-safe enough for our single-process FD."""
    lst = _run_workers.setdefault(sid, [])
    if add:
        lst.append(slug)
    else:
        try:
            lst.remove(slug)
        except ValueError:
            pass


# ── Lead: decompose the task into owned subtasks ─────────────────────

def _normalize_plan(raw: dict, arch_by_id: dict, max_agents: int) -> dict:
    """Validate + clamp the Lead's decomposition. Drops subtasks whose owner
    isn't a real archetype; caps the count at max_agents; guarantees unique ids."""
    domain = str(raw.get("domain") or "general").strip().lower() or "general"
    subtasks: list[dict] = []
    seen_ids: set[str] = set()
    for i, s in enumerate(raw.get("subtasks") or []):
        if len(subtasks) >= max_agents:
            break
        owner = str(s.get("owner_archetype_id") or "").strip()
        brief = str(s.get("brief") or "").strip()
        if owner not in arch_by_id or not brief:
            continue
        sid = str(s.get("id") or "").strip() or f"s{i + 1}"
        while sid in seen_ids:
            sid = f"{sid}x"
        seen_ids.add(sid)
        subtasks.append({
            "id": sid,
            "title": (str(s.get("title") or "").strip() or owner)[:80],
            "owner_archetype_id": owner,
            "brief": brief,
        })
    return {"domain": domain, "rationale": str(raw.get("rationale") or "").strip(),
            "subtasks": subtasks}


async def _llm_decompose(intent: str, archetypes: list[dict], reliability: dict,
                         creds: dict, max_agents: int) -> dict:
    """Ask the Lead to split the task into complementary, owner-assigned subtasks.

    Returns a normalized plan. On any LLM/parse failure raises — the caller turns
    that into a clean run failure (Phase 1 has no deterministic fallback planner).
    """
    from captain_claw.llm import Message
    system_file = _INSTRUCTIONS_DIR / "vatra" / "lead.md"
    if not system_file.is_file():
        raise HTTPException(500, "Vatra lead prompt not found")
    arch_by_id = {a["id"]: a for a in archetypes}
    system_prompt = system_file.read_text() + "\n\n## Catalog\n" + _build_catalog(archetypes, reliability)
    user_prompt = (
        f"Task: {intent}\n\n"
        f"max_agents: {max_agents}. Decompose into the smallest set of complementary, "
        f"owner-assigned subtasks that together cover this task."
    )
    prov, mt = _provider_call(creds, temperature=0.2, default_max=4096, cap=8192)
    resp = await prov.complete(messages=[
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ], temperature=0.2, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    raw = json.loads(content)
    plan = _normalize_plan(raw, arch_by_id, max_agents)
    if not plan["subtasks"]:
        raise HTTPException(422, "Lead produced no usable subtasks")
    return plan


def _resolve_creds(registry: dict, tiers: dict | None, api_key: str, tier: str) -> dict:
    """Resolve LLM creds for a tier from the Library tiers, else the registry."""
    lt = (tiers or {}).get(tier)
    if lt and lt.get("model"):
        return {"provider": lt.get("provider", "anthropic"), "model": lt.get("model", ""),
                "base_url": lt.get("base_url") or None,
                "api_key": lt.get("api_key") or api_key or None,
                "output_ctx": int(lt.get("output_ctx") or 0)}
    return _tier_creds(registry, tier, api_key or "")


async def _build_plan(db, user_id: str, intent: str, max_agents: int, creds: dict) -> dict:
    """Run the Lead and shape the result into a persistable Vatra route:
    {mode, domain, rationale, subtasks, selected}. `selected` mirrors Basna's
    shape so the read-tool and list UI render the owners."""
    archetypes = await merged_archetypes(db, user_id)
    arch_by_id = {a["id"]: a for a in archetypes}
    rel_rows = await db.get_archetype_reliability(user_id)
    reliability: dict[str, list[dict]] = {}
    for r in rel_rows:
        reliability.setdefault(r["archetype_id"], []).append(r)
    plan = await asyncio.wait_for(
        _llm_decompose(intent, archetypes, reliability, creds, max_agents), _DECOMPOSE_TIMEOUT)
    selected = [{"archetype_id": s["owner_archetype_id"],
                 "role": arch_by_id[s["owner_archetype_id"]].get("role", ""),
                 "why": s["title"]} for s in plan["subtasks"]]
    return {"mode": "vatra", "domain": plan["domain"], "rationale": plan["rationale"],
            "subtasks": plan["subtasks"], "selected": selected}


# ── Spawn / teardown (mirrors Basna's; stamped CLAW_VATRA_WORKER) ─────

async def _spawn_worker(request: Request, user: dict, *, name: str, description: str,
                        cognitive_mode: str, tools: list[str], tier: str,
                        tiers: dict | None, api_key: str, env_vars: list[dict] | None,
                        extra_env: list[dict] | None = None,
                        ) -> dict:
    """Spawn one ephemeral agent and resolve its web port. Returns
    {ok, slug, port, auth, message}. Strips the run-starting `basna` tool and
    stamps the no-recursion marker so a Vatra worker can never launch another run;
    the `vatra` ask/inbox tool is registered unconditionally and stays available.
    `extra_env` carries the run context (session/subtask/owner/depth)."""
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
    worker_tools = [t for t in (tools or AgentConfig().tools) if t != "basna"]
    base = dict(
        name=name, description=description,
        cognitive_mode=cognitive_mode or "neutra", tools=worker_tools,
        env_vars=(env_vars or []) + (extra_env or []) + [{"key": _WORKER_MARKER, "value": "1"}],
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
    """Fully remove ephemeral agents so they don't pile up in the fleet."""
    from captain_claw.flight_deck.server import (
        _do_stop_process, _load_process_registry, _save_process_registry,
        _processes, DATA_DIR,
    )
    for slug in slugs:
        try:
            _do_stop_process(slug)
        except Exception as e:
            log.warning("Vatra teardown stop failed", slug=slug, error=str(e))
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


def _capture_generated(slug: str, exclude: set[str], dest_dir: Path,
                       agent_role: str, seen: set[str]) -> tuple[list[dict], str]:
    """Copy files an agent generated into the session dir (before teardown) and
    return (file metadata, concatenated text of its texty artifacts)."""
    from captain_claw.flight_deck.server import DATA_DIR
    ws = DATA_DIR / slug / "data" / "workspace"
    files: list[dict] = []
    texts: list[str] = []
    if not ws.is_dir():
        return files, ""
    for p in sorted(ws.rglob("*")):
        if not p.is_file() or p.name in exclude:
            continue
        out_name = p.name if p.name not in seen else f"{agent_role}__{p.name}"
        seen.add(out_name)
        try:
            shutil.copy2(p, dest_dir / out_name)
        except OSError as e:
            log.warning("Vatra generated-file capture failed", file=p.name, error=str(e))
            continue
        files.append({"name": out_name, "mime": _guess_mime(out_name),
                      "size": p.stat().st_size, "kind": "generated", "agent": agent_role})
        if _is_texty(out_name, _guess_mime(out_name)) and p.stat().st_size <= _MAX_TEXT_FALLBACK_BYTES:
            try:
                texts.append(p.read_text(errors="replace"))
            except OSError:
                pass
    return files, "\n\n".join(t.strip() for t in texts if t.strip()).strip()


# ── Orchestrator ─────────────────────────────────────────────────────

async def execute_vatra(body: ExecuteRequest, request: Request, user: dict) -> dict:
    """Run one Vatra session: Lead decompose → parallel subtasks → reporter assemble.

    Mirrors Basna's spawn/dispatch/teardown but replaces the weighted merge with a
    Lead-planned decomposition and a dedicated reporter that writes the final
    artifact. Phase 1: no inter-agent delegation, no learning loop.
    """
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    intent = (sess.get("intent") or "").strip()
    if not intent:
        raise HTTPException(400, "session has no intent")

    archetypes = await merged_archetypes(db, user["id"])
    arch_by_id = {a["id"]: a for a in archetypes}
    seeds = {a["id"]: float(a.get("reliability_seed", 0.7)) for a in archetypes}
    registry = _load_registry()

    def _creds(tier: str) -> dict:
        return _resolve_creds(registry, body.tiers, body.api_key, tier)

    try:
        cfg = json.loads(sess.get("config") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    max_agents = int(cfg.get("max_agents") or 6)

    session_files = _parse_files(sess)
    input_files = [f for f in session_files if f.get("kind") != "generated"]
    input_names = {f["name"] for f in input_files}

    sid = body.session_id
    sid8 = sid[:8]
    run_tag = format(int(time.time()), "x")[-6:]
    _progress_start(sid)
    await db.update_basna_session(sid, user["id"], status="running")

    # 1) Plan. Reuse a route prepared by the UI's /route step if present; otherwise
    # decompose now (the one-shot /start and agent paths). Splitting decompose from
    # spawn is what gives the UI a Basna-style prepare → review → run flow.
    try:
        existing = json.loads(sess.get("route") or "{}")
    except json.JSONDecodeError:
        existing = {}
    if existing.get("mode") == "vatra" and existing.get("subtasks"):
        route = existing
        _progress(sid, "route",
                  f"Using prepared plan · {len(route['subtasks'])} piece(s) · {route.get('domain', '')}")
    else:
        _progress(sid, "route", "Lead decomposing the task…")
        try:
            route = await _build_plan(db, user["id"], intent, max_agents, _creds("fast"))
        except HTTPException:
            await db.update_basna_session(sid, user["id"], status="error")
            raise
        except Exception as e:
            await db.update_basna_session(sid, user["id"], status="error")
            _progress(sid, "route", f"Lead decomposition failed: {str(e)[:200]}", ok=False)
            raise HTTPException(502, f"Vatra Lead failed: {e}")
        await db.update_basna_session(
            sid, user["id"], domain=route["domain"], route=json.dumps(route),
            config=json.dumps({**cfg, "mode": "vatra"}))
        _progress(sid, "route", f"{len(route['subtasks'])} subtask(s) · {route['domain']}")
    domain = route["domain"]
    subtasks = route["subtasks"]

    spawned: list[dict] = []   # {subtask, slug, port, auth}
    results: list[dict] = []   # {id, owner, role, output, ok, latency_ms, actions}
    generated_files: list[dict] = []
    seen_gen: set[str] = set()
    dest_dir = _session_files_dir(sid)

    try:
        # 2) Spawn one owner per subtask.
        _progress(sid, "spawn", f"Spawning {len(subtasks)} specialist(s)…")
        tiers = body.tiers or None

        async def _spawn_owner(st: dict) -> dict | None:
            arch = arch_by_id[st["owner_archetype_id"]]
            tier = arch.get("tier", "balanced")
            # Name by SUBTASK id, not just archetype — two pieces can share an
            # owner archetype (e.g. two researcher slices), and a per-archetype name
            # would collide so only one agent spawns.
            sp = await _spawn_worker(
                request, user,
                name=f"vatra-{sid8}-{run_tag}-{st['id']}-{arch['id']}",
                description=f"Vatra subtask · {arch.get('role', '')}",
                cognitive_mode=arch.get("cognitive_mode", "neutra"),
                tools=arch.get("tools") or [], tier=tier, tiers=tiers,
                api_key=body.api_key, env_vars=body.env_vars,
                extra_env=_vatra_env(sid, st["id"], arch["id"], 0),
            )
            if not sp["ok"]:
                _progress(sid, "spawn", f"{arch.get('role') or arch['id']}: unusable — {sp['message']}", ok=False)
                return None
            # Materialize input files into the owner's workspace.
            if input_files:
                from captain_claw.flight_deck.server import DATA_DIR
                ws = DATA_DIR / sp["slug"] / "data" / "workspace"
                for f in input_files:
                    try:
                        shutil.copy2(dest_dir / f["name"], ws / f["name"])
                    except OSError as e:
                        log.warning("Vatra input copy failed", file=f["name"], error=str(e))
            return {"subtask": st, "arch": arch, **sp}

        spawn_out = await asyncio.gather(
            *[_spawn_owner(st) for st in subtasks], return_exceptions=True)
        for item in spawn_out:
            if isinstance(item, Exception):
                log.warning("Vatra spawn failed", error=str(item))
                _progress(sid, "spawn", f"spawn failed: {str(item)[:160]}", ok=False)
                continue
            if item:
                spawned.append(item)
        _run_workers[sid] = [sp["slug"] for sp in spawned]
        _progress(sid, "spawn", f"Spawned {len(spawned)}/{len(subtasks)}; dispatching…")

        # 2b) Start the delegation coordinator — it watches the blackboard and
        # fulfils asks with helpers WHILE the owners work (non-blocking). It drains
        # and exits once stop_event is set and no open asks / helpers remain.
        stop_event = asyncio.Event()
        coordinator = asyncio.create_task(_coordinate_asks(
            request, user, sid, sid8, run_tag, intent, domain,
            archetypes=archetypes, arch_by_id=arch_by_id, tiers=tiers,
            api_key=body.api_key, env_vars=body.env_vars,
            dispatch_timeout=body.dispatch_timeout, stop_event=stop_event))

        # 3) Dispatch each owner its self-contained brief, in parallel.
        # Count how many pieces each archetype owns so we can disambiguate the live
        # panel label only when an archetype owns more than one piece.
        owner_counts: dict[str, int] = {}
        for sp in spawned:
            owner_counts[sp["arch"]["id"]] = owner_counts.get(sp["arch"]["id"], 0) + 1

        async def _dispatch_owner(sp: dict) -> dict:
            arch, st = sp["arch"], sp["subtask"]
            role = arch.get("role") or arch["id"]
            # Distinct live-panel label per piece when an archetype owns several, so
            # two researcher slices show as two cards, not one merged card.
            label = f"{role} · {st['title']}" if owner_counts.get(arch["id"], 1) > 1 else role
            fleet = arch.get("fleet_instructions", "")

            def _on_action(act: dict) -> None:
                detail = act.get("detail", "")
                if act["tool"] == "narration":
                    _progress(sid, "narration", f"{label}: {detail}", agent=label,
                              tool="narration", detail=detail)
                else:
                    suffix = f": {detail}" if detail else ""
                    _progress(sid, "action", f"{label} → {act['tool']}{suffix}",
                              agent=label, tool=act["tool"], detail=detail)

            def _on_usage(pt: int, ct: int, tt: int) -> None:
                _progress(sid, "usage", f"{label} · {pt:,}→{ct:,} tok",
                          agent=label, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

            from captain_claw.flight_deck.server import DATA_DIR
            ws = DATA_DIR / sp["slug"] / "data" / "workspace"
            img = [str(ws / f["name"]) for f in input_files if str(f.get("mime", "")).startswith("image/")]
            doc = [str(ws / f["name"]) for f in input_files if not str(f.get("mime", "")).startswith("image/")]
            prompt = _build_subtask_prompt(role, intent, st, [f["name"] for f in input_files])
            d = await _dispatch_one(
                sp["port"], sp["auth"], prompt, body.dispatch_timeout,
                on_action=_on_action, fleet_instructions=fleet, agent_name=label,
                file_paths=doc, image_paths=img, on_usage=_on_usage)
            mark = "✓" if d["ok"] else "✗"
            extra = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
            _progress(sid, "dispatch",
                      f"{label} {mark} · {len(d['actions'])} action(s) ({d['latency_ms'] / 1000:.1f}s){extra}",
                      ok=d["ok"], agent=label)
            return d

        dispatched = await asyncio.gather(*[_dispatch_owner(sp) for sp in spawned])
        for sp, d in zip(spawned, dispatched):
            results.append({
                "id": sp["subtask"]["id"], "owner": sp["arch"]["id"],
                "role": sp["arch"].get("role", ""), "title": sp["subtask"]["title"],
                "output": d["output"], "ok": d["ok"],
                "latency_ms": d["latency_ms"], "actions": d.get("actions", []),
            })

        # 3b) Owners are done — tell the coordinator to drain remaining asks and stop.
        stop_event.set()
        try:
            n_asks = await coordinator
        except Exception as e:
            log.warning("Vatra coordinator error", error=str(e))
            n_asks = 0
        if n_asks:
            _progress(sid, "ask", f"Coordinator resolved {n_asks} ask(s)")

        # 4) Capture owner-generated files + backfill empty replies from artifacts.
        for i, sp in enumerate(spawned):
            role = sp["arch"].get("role") or sp["arch"]["id"]
            files, text = _capture_generated(sp["slug"], input_names, dest_dir, role, seen_gen)
            generated_files.extend(files)
            if not (results[i].get("output") or "").strip() and text:
                results[i]["output"] = text
                results[i]["produced_file"] = True
    finally:
        _teardown([sp["slug"] for sp in spawned])
        _run_workers.pop(sid, None)

    usable = [r for r in results if (r.get("ok") or r.get("produced_file")) and (r.get("output") or "").strip()]
    if not usable:
        await db.update_basna_session(sid, user["id"], status="error")
        _progress(sid, "done", "No subtask produced usable output", ok=False)
        _progress_done(sid)
        raise HTTPException(502, "Vatra: no subtask produced usable output")

    # 5) Reporter assembles the slices (+ any answered asks) into one deliverable.
    answered = await db.list_vatra_asks(sid, status="answered")
    truth, reporter_files = await _run_reporter(
        request, user, sid, sid8, run_tag, intent, usable, cfg, arch_by_id,
        tiers=body.tiers, api_key=body.api_key, env_vars=body.env_vars,
        dispatch_timeout=body.dispatch_timeout, input_names=input_names,
        dest_dir=dest_dir, seen_gen=seen_gen, answered_asks=answered,
    )
    generated_files.extend(reporter_files)
    confidence = round(len(usable) / max(1, len(results)), 3)

    # 6) Persist one run per owner (success backfilled by the learning step below).
    run_ids = await db.add_basna_runs(sid, user["id"], [{
        "archetype_id": r["owner"], "role": r["role"], "tier": "",
        "weight_at_run": 0.0, "output": r["output"],
        "actions": json.dumps(r.get("actions", [])),
        "latency_ms": r["latency_ms"], "success": None,
    } for r in results])

    # 7) Learn: score owners (slice used + sound), ask-answerers, and the Lead +
    # reporter (holistic), folding outcomes into per-archetype reliability so the
    # next route's prior_weight improves. Owners learn under their real archetype
    # id (shared with Basna); Lead/reporter learn as separate pseudo-archetypes.
    _progress(sid, "learn", "Scoring contributions…")
    try:
        learned = await _learn(
            db, user, sid, domain, intent, subtasks, results, usable, answered, truth,
            run_ids, seeds, judge_creds=_creds("fast"), holistic_creds=_creds("reason"))
        _progress(sid, "learn", f"Scored {len(learned)} contribution(s)")
    except Exception as e:
        log.warning("Vatra learning failed", error=str(e))
        learned = []
        _progress(sid, "learn", "Scoring skipped (judge unavailable)", ok=False)

    files_by_name = {f["name"]: f for f in session_files}
    for g in generated_files:
        files_by_name[g["name"]] = g
    await db.update_basna_session(
        sid, user["id"], status="done", truth=truth, confidence=confidence,
        files=json.dumps(list(files_by_name.values())),
    )
    _progress(sid, "done", f"Done · {len(usable)}/{len(results)} subtask(s) assembled")
    _progress_done(sid)
    await db.update_basna_session(
        sid, user["id"],
        progress=json.dumps((_PROGRESS.get(sid) or {}).get("events", [])),
    )
    return {"session_id": sid, "domain": domain, "mode": "vatra",
            "truth": truth, "confidence": confidence,
            "subtasks": [{"id": r["id"], "owner": r["owner"], "role": r["role"],
                          "ok": r["ok"], "latency_ms": r["latency_ms"]} for r in results],
            "learned": learned, "spawned": len(spawned), "dispatched": len(results)}


# ── Learning: score owners / answerers / lead / reporter ─────────────

_LEAD_PSEUDO = "vatra-lead"
_REPORTER_PSEUDO = "vatra-reporter"


async def _llm_judge_holistic(intent: str, subtasks: list[dict], truth: str,
                              creds: dict) -> dict | None:
    """One reason-tier verdict on the two whole-run roles: was the Lead's
    decomposition good (complementary, well-scoped, covers the task), and is the
    reporter's assembled artifact coherent and complete? Returns
    {"lead": bool, "reporter": bool} or None if unparseable."""
    from captain_claw.llm import Message
    plan = "\n".join(f"- {s.get('title', '')} (owner: {s.get('owner_archetype_id', '')})"
                     for s in subtasks)
    prov, mt = _provider_call(creds, temperature=0.0, default_max=512, cap=1024)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "You are grading a collaborative run. A Lead split a task into subtasks "
            "(the plan), specialists each built a piece, and a reporter assembled them "
            "into the final deliverable. Judge two things independently:\n"
            "- lead: was the DECOMPOSITION good — complementary, well-scoped pieces that "
            "together cover the task (not overlapping, not missing obvious parts)?\n"
            "- reporter: is the FINAL DELIVERABLE coherent and complete for the task — one "
            "integrated whole, not stapled fragments?\n"
            'Reply ONLY with JSON: {"lead": true/false, "reporter": true/false}.')),
        Message(role="user", content=(
            f"TASK:\n{intent[:2000]}\n\nPLAN (subtasks):\n{plan}\n\n"
            f"FINAL DELIVERABLE:\n{truth[:6000]}")),
    ], temperature=0.0, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    raw = json.loads(content)
    if not isinstance(raw, dict):
        return None
    return {"lead": bool(raw.get("lead")), "reporter": bool(raw.get("reporter"))}


async def _learn(db, user: dict, sid: str, domain: str, intent: str,
                 subtasks: list[dict], results: list[dict], usable: list[dict],
                 answered: list[dict], truth: str, run_ids: list, seeds: dict, *,
                 judge_creds: dict, holistic_creds: dict) -> list[dict]:
    """Score every contribution against the assembled deliverable and fold the
    outcomes into per-archetype reliability. Resilient: a judge failure leaves the
    affected contributions unscored rather than guessed."""
    learned: list[dict] = []

    async def _record(archetype_id: str, success: bool) -> None:
        rel = await db.record_archetype_outcome(
            user["id"], archetype_id, domain, success, seeds.get(archetype_id, 0.7))
        learned.append({"archetype_id": archetype_id, "success": success,
                        "weight": rel["weight"]})

    # 1) Owners — usable slices judged against the deliverable; empty owners fail.
    verdicts: list[bool] = []
    if usable and truth:
        try:
            verdicts = await _llm_judge(usable, truth, judge_creds)
        except Exception as e:
            log.warning("Vatra owner judge failed; leaving owners unscored", error=str(e))
            verdicts = []
    usable_succ = {r["id"]: bool(v) for r, v in zip(usable, verdicts)}
    for i, r in enumerate(results):
        is_usable = bool((r.get("ok") or r.get("produced_file")) and (r.get("output") or "").strip())
        if not is_usable:
            succ = False
        elif r["id"] in usable_succ:
            succ = usable_succ[r["id"]]
        else:
            continue  # judge couldn't decide → don't guess
        rid = run_ids[i] if i < len(run_ids) else None
        if rid is not None:
            await db.score_basna_run(rid, user["id"], succ)
        await _record(r["owner"], succ)

    # 2) Ask-answerers — was each answer sound/used in the deliverable?
    ans_good = [{"output": a.get("answer", ""), "archetype_id": a.get("answered_by", "")}
                for a in answered
                if (a.get("answer") or "").strip() and a.get("answered_by")]
    if ans_good and truth:
        try:
            av = await _llm_judge(ans_good, truth, judge_creds)
            for g, v in zip(ans_good, av):
                await _record(g["archetype_id"], bool(v))
        except Exception as e:
            log.warning("Vatra answerer judge failed; leaving answerers unscored", error=str(e))

    # 3) Lead + reporter — holistic, as separate pseudo-archetypes.
    if truth:
        try:
            h = await _llm_judge_holistic(intent, subtasks, truth, holistic_creds)
            if h:
                await _record(_LEAD_PSEUDO, h["lead"])
                await _record(_REPORTER_PSEUDO, h["reporter"])
        except Exception as e:
            log.warning("Vatra holistic judge failed; lead/reporter unscored", error=str(e))

    return learned


def _build_subtask_prompt(role: str, intent: str, st: dict, file_names: list[str]) -> str:
    """Frame one subtask for its owner. Phase 1: owners work in parallel and blind,
    so the brief must be self-contained."""
    files_block = ""
    if file_names:
        listed = "\n".join(f"- {n}" for n in file_names)
        files_block = ("\n\n## Attached files (in your working directory)\n"
                       f"{listed}\nUse your read / extract tools to work with them.\n")
    return (
        f"You are the {role}, one specialist on a collaborating team. You own ONE part "
        f"of a larger deliverable; other specialists are producing the other parts in "
        f"parallel. Produce only your part, in full — another author (the reporter) will "
        f"assemble all parts into the final deliverable.\n\n"
        f"## Overall task (for context)\n{intent}\n\n"
        f"## Your part — {st['title']}\n{st['brief']}{files_block}\n\n"
        f"## Working with your team\n"
        f"If you need something OUTSIDE your part that another specialist should produce, "
        f"do NOT wait or do it yourself half-heartedly: use the `vatra` tool (action='ask') "
        f"to post a focused request and KEEP WORKING on your part. After you've made progress, "
        f"call `vatra` (action='inbox') to collect any answers. Never block on an ask — if no "
        f"answer arrives in time, finish your part as best you can; the reporter folds in "
        f"whatever the team delivers. Use this only for genuine cross-slice needs, not for work "
        f"that is your own.\n\n"
        f"Return only your finished part — no preamble, no meta-commentary about the team."
    )


async def _run_reporter(request: Request, user: dict, sid: str, sid8: str, run_tag: str,
                        intent: str, usable: list[dict], cfg: dict, arch_by_id: dict, *,
                        tiers, api_key, env_vars, dispatch_timeout, input_names,
                        dest_dir: Path, seen_gen: set[str],
                        answered_asks: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Spawn a dedicated reporter, feed it the slices (plus any answered cross-agent
    asks), and capture the assembled deliverable. Falls back to a labeled
    concatenation if the reporter fails."""
    from captain_claw.flight_deck.server import DATA_DIR

    reporter_id = str(cfg.get("reporter_archetype") or _DEFAULT_REPORTER)
    arch = arch_by_id.get(reporter_id) or arch_by_id.get(_DEFAULT_REPORTER)
    slices_full = "\n\n".join(
        f"### Piece: {r['title']} — by {r['role']}\n{r['output'].strip()}" for r in usable)
    # Answered asks are extra material the team surfaced via delegation — give the
    # reporter the same view the asker had, so nothing the team produced is lost.
    asks_block = ""
    for a in (answered_asks or []):
        ans = (a.get("answer") or "").strip()
        if ans:
            asks_block += f"\n\n### Resolved team request: {a.get('text', '')[:160]}\n{ans}"
    if asks_block:
        slices_full += "\n\n## Answers to cross-agent requests" + asks_block
    fallback = "\n\n".join(f"## {r['title']} ({r['role']})\n{r['output'].strip()}" for r in usable)
    if asks_block:
        fallback += "\n\n## Answers to cross-agent requests" + asks_block
    if not arch:
        _progress(sid, "report", "No reporter archetype available; using raw assembly", ok=False)
        return fallback, []

    role = arch.get("role") or arch["id"]
    _progress(sid, "report", f"Reporter ({role}) assembling {len(usable)} piece(s)…")
    sp = await _spawn_worker(
        request, user, name=f"vatra-{sid8}-{run_tag}-reporter",
        description=f"Vatra reporter · {role}",
        cognitive_mode=arch.get("cognitive_mode", "neutra"),
        tools=arch.get("tools") or [], tier=arch.get("tier", "reason"),
        tiers=tiers, api_key=api_key, env_vars=env_vars,
    )
    if not sp["ok"]:
        _progress(sid, "report", f"Reporter spawn failed — using raw assembly ({sp['message']})", ok=False)
        return fallback, []
    _track_worker(sid, sp["slug"], add=True)
    try:
        # Write the full slices into the reporter's workspace; inline a preview.
        ws = DATA_DIR / sp["slug"] / "data" / "workspace"
        try:
            (ws / "vatra-slices.md").write_text(slices_full)
        except OSError as e:
            log.warning("Vatra slices file write failed", error=str(e))
        big = len(slices_full) > _SLICES_INLINE_CHARS
        inline = slices_full[:_SLICES_INLINE_CHARS] + ("\n\n…(full text in vatra-slices.md)" if big else "")
        template = (_INSTRUCTIONS_DIR / "vatra" / "reporter.md").read_text()
        prompt = template.replace("{intent}", intent).replace("{slices}", inline)

        def _on_action(act: dict) -> None:
            detail = act.get("detail", "")
            if act["tool"] == "narration":
                _progress(sid, "narration", f"{role}: {detail}", agent=role, tool="narration", detail=detail)
            else:
                suffix = f": {detail}" if detail else ""
                _progress(sid, "action", f"{role} → {act['tool']}{suffix}", agent=role, tool=act["tool"], detail=detail)

        def _on_usage(pt: int, ct: int, tt: int) -> None:
            _progress(sid, "usage", f"{role} · {pt:,}→{ct:,} tok",
                      agent=role, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

        d = await _dispatch_one(sp["port"], sp["auth"], prompt, dispatch_timeout,
                                on_action=_on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                                agent_name=role, on_usage=_on_usage)
        out = (d.get("output") or "").strip()
        files, text = _capture_generated(sp["slug"], input_names | {"vatra-slices.md"},
                                         dest_dir, role, seen_gen)
        if not out and text:
            out = text
        mark = "✓" if d["ok"] and out else "✗"
        _progress(sid, "report", f"Reporter {mark} ({d['latency_ms'] / 1000:.1f}s)", ok=bool(out))
        return (out or fallback), files
    finally:
        _teardown([sp["slug"]])
        _track_worker(sid, sp["slug"], add=False)


# ── Coordinator: route blackboard asks to helpers (non-blocking) ─────

async def _coordinate_asks(request: Request, user: dict, sid: str, sid8: str,
                           run_tag: str, intent: str, domain: str, *,
                           archetypes: list[dict], arch_by_id: dict, tiers,
                           api_key, env_vars, dispatch_timeout,
                           stop_event: asyncio.Event) -> int:
    """Watch the blackboard and fulfil open asks with fresh helpers, concurrently
    with the owners' work. Runs until ``stop_event`` is set AND nothing is open or
    in flight. Returns the number of asks answered. Budget/depth/cycle guards are
    enforced at ask-creation time (see ``agent_ask``); this loop only routes.
    """
    db = get_db()
    sem = asyncio.Semaphore(_MAX_HELPERS)
    inflight: set = set()
    answered = 0

    async def _fulfill(ask: dict) -> None:
        nonlocal answered
        async with sem:
            ok = await _fulfill_ask(
                request, user, sid, sid8, run_tag, intent, ask,
                archetypes=archetypes, arch_by_id=arch_by_id, tiers=tiers,
                api_key=api_key, env_vars=env_vars, dispatch_timeout=dispatch_timeout)
            if ok:
                answered += 1

    while True:
        try:
            open_asks = await db.list_vatra_asks(sid, status="open")
        except Exception as e:
            log.warning("Vatra coordinator poll failed", error=str(e))
            open_asks = []
        for ask in open_asks:
            if await db.claim_vatra_ask(ask["id"]):
                t = asyncio.create_task(_fulfill(ask))
                inflight.add(t)
                t.add_done_callback(inflight.discard)
        if stop_event.is_set() and not open_asks and not inflight:
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_COORD_POLL_S)
        except asyncio.TimeoutError:
            pass
    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)
    return answered


async def _fulfill_ask(request: Request, user: dict, sid: str, sid8: str, run_tag: str,
                       intent: str, ask: dict, *, archetypes: list[dict],
                       arch_by_id: dict, tiers, api_key, env_vars,
                       dispatch_timeout) -> bool:
    """Spawn one fresh helper to answer a single ask, write the answer back, tear
    it down. Returns True if an answer was recorded."""
    db = get_db()
    text = (ask.get("text") or "").strip()
    depth = int(ask.get("depth") or 0) + 1
    # Route the ask to the best-matching specialist (deterministic keyword pick).
    picks = _keyword_match(text, archetypes, 1)
    arch = picks[0] if picks else (arch_by_id.get("concierge") or next(iter(arch_by_id.values())))
    role = arch.get("role") or arch["id"]
    _progress(sid, "ask", f"Helper ({role}) answering ask #{ask['id']}…", agent=role)
    sp = await _spawn_worker(
        request, user, name=f"vatra-{sid8}-{run_tag}-help{ask['id']}-{arch['id']}",
        description=f"Vatra helper · {role}",
        cognitive_mode=arch.get("cognitive_mode", "neutra"),
        tools=arch.get("tools") or [], tier=arch.get("tier", "balanced"),
        tiers=tiers, api_key=api_key, env_vars=env_vars,
        extra_env=_vatra_env(sid, f"help:{ask['id']}", arch["id"], depth),
    )
    if not sp["ok"]:
        await db.drop_vatra_ask(ask["id"], note=f"helper spawn failed: {sp['message']}")
        _progress(sid, "ask", f"Ask #{ask['id']} dropped — helper spawn failed", ok=False)
        return False
    _track_worker(sid, sp["slug"], add=True)
    helper_label = f"{role} (helper #{ask['id']})"

    def _on_usage(pt: int, ct: int, tt: int) -> None:
        _progress(sid, "usage", f"{helper_label} · {pt:,}→{ct:,} tok",
                  agent=helper_label, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

    try:
        prompt = _build_helper_prompt(role, intent, text)
        d = await _dispatch_one(sp["port"], sp["auth"], prompt, dispatch_timeout,
                                fleet_instructions=arch.get("fleet_instructions", ""),
                                agent_name=helper_label, on_usage=_on_usage)
        out = (d.get("output") or "").strip()
        if out:
            await db.answer_vatra_ask(ask["id"], out, arch["id"])
            _progress(sid, "ask", f"Ask #{ask['id']} answered by {role}", agent=role)
            return True
        await db.drop_vatra_ask(ask["id"], note="helper produced no output")
        _progress(sid, "ask", f"Ask #{ask['id']} dropped — no helper output", ok=False)
        return False
    except Exception as e:
        await db.drop_vatra_ask(ask["id"], note=f"helper error: {e}")
        log.warning("Vatra helper failed", ask_id=ask["id"], error=str(e))
        return False
    finally:
        _teardown([sp["slug"]])
        _track_worker(sid, sp["slug"], add=False)


def _build_helper_prompt(role: str, intent: str, ask_text: str) -> str:
    return (
        f"You are the {role}, helping a teammate on a collaborating team. A specialist "
        f"working on part of a larger task has asked you for something specific. Answer it "
        f"directly, completely, and concisely — they will use your answer in their own work.\n\n"
        f"## The overall task (for context)\n{intent}\n\n"
        f"## What your teammate needs\n{ask_text}\n\n"
        f"Return only the answer to their request — no preamble, no meta-commentary."
    )


# ── Agent-facing blackboard endpoints (port-identified, owner-scoped) ─

class _VatraAskReq(_AgentReq):
    subtask_id: str = ""
    owner: str = ""
    depth: int = 0
    text: str = ""


class _VatraInboxReq(_AgentReq):
    owner: str = ""
    wait: int = 0


def _norm_ask(text: str) -> frozenset:
    return frozenset(_norm_text(text))


@router.post("/agent/ask")
async def agent_ask(body: _VatraAskReq):
    """A specialist posts a cross-slice request. Enforces the delegation budget
    (max asks), the depth cap, and a cycle/dedup guard — all deterministic floors
    under the termination guarantee. Returns immediately (non-blocking)."""
    owner_id = _resolve_owner(body)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner_id)
    if not sess:
        raise HTTPException(404, "session not found")
    if int(body.depth) >= _MAX_ASK_DEPTH:
        return {"status": "rejected", "reason": f"max delegation depth ({_MAX_ASK_DEPTH}) reached"}
    existing = await db.list_vatra_asks(body.session_id)
    # Cycle/dedup guard: an essentially identical ask already exists → reuse it.
    norm = _norm_ask(text)
    for e in existing:
        if _norm_ask(e.get("text", "")) == norm:
            return {"status": "ok", "ask_id": e["id"], "dedup": True}
    if len(existing) >= _MAX_ASKS:
        return {"status": "rejected", "reason": f"delegation budget ({_MAX_ASKS} asks) reached"}
    ask = await db.create_vatra_ask(
        body.session_id, body.owner, body.subtask_id, text, depth=int(body.depth))
    return {"status": "ok", "ask_id": ask["id"]}


@router.post("/agent/inbox")
async def agent_inbox(body: _VatraInboxReq):
    """Collect a specialist's answered asks. Optionally long-polls up to `wait`
    seconds so the asker can pick up an answer in the same turn."""
    owner_id = _resolve_owner(body)
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner_id)
    if not sess:
        raise HTTPException(404, "session not found")
    wait = max(0, min(30, int(body.wait or 0)))
    deadline = time.monotonic() + wait
    while True:
        mine = await db.list_vatra_asks(body.session_id, from_owner=body.owner)
        answered = [a for a in mine if a.get("status") == "answered"]
        pending = [a for a in mine if a.get("status") in ("open", "claimed")]
        if answered or not pending or time.monotonic() >= deadline:
            return {"answered": [{"text": a.get("text", ""), "answer": a.get("answer", ""),
                                  "answered_by": a.get("answered_by", "")} for a in answered],
                    "pending": len(pending)}
        await asyncio.sleep(_INBOX_POLL_S)


# ── UI manual start (user-scoped, background) ────────────────────────

class VatraStartRequest(BaseModel):
    intent: str
    title: str = ""
    max_agents: int = Field(default=6, ge=1, le=10)
    # Per-tier model config from the Library (same shape Basna's execute uses).
    tiers: dict | None = None
    env_vars: list | None = None
    api_key: str = ""


class VatraExecuteRequest(BaseModel):
    session_id: str
    tiers: dict | None = None
    env_vars: list | None = None
    api_key: str = ""


@router.post("/route")
async def route_vatra(body: VatraStartRequest, user: dict = Depends(get_current_user)):
    """Vatra 'prepare' step: the Lead decomposes the task and the plan is persisted,
    but nothing is spawned. Mirrors Basna's Route → review → Run. Returns the plan;
    the UI shows it, then calls /execute to spawn the team."""
    intent = (body.intent or "").strip()
    if not intent:
        raise HTTPException(400, "intent is required")
    db = get_db()
    title = (body.title or intent[:60]).strip()
    sess = await db.create_basna_session(
        user["id"], intent, title=title,
        config=json.dumps({"mode": "vatra", "source": "ui", "max_agents": body.max_agents}))
    sid = sess["id"]
    registry = _load_registry()
    creds = _resolve_creds(registry, body.tiers, body.api_key, "fast")
    try:
        route = await _build_plan(db, user["id"], intent, body.max_agents, creds)
    except HTTPException:
        await db.delete_basna_session(sid, user["id"])
        raise
    except Exception as e:
        await db.delete_basna_session(sid, user["id"])
        raise HTTPException(502, f"Vatra Lead failed: {e}")
    await db.update_basna_session(
        sid, user["id"], domain=route["domain"], route=json.dumps(route), status="routed")
    return {"session_id": sid, "title": title, **route}


@router.post("/execute")
async def execute_vatra_ui(body: VatraExecuteRequest, request: Request,
                           user: dict = Depends(get_current_user)):
    """Spawn + run a prepared Vatra session in the background (its plan was made by
    /route). Returns immediately; the UI polls progress. execute_vatra reuses the
    persisted plan, so the Lead does not run again."""
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    exec_req = ExecuteRequest(
        session_id=body.session_id, tiers=body.tiers or None,
        env_vars=body.env_vars or None, api_key=body.api_key or "")
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(execute_vatra(exec_req, stub, user))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": body.session_id, "status": "running"}


@router.post("/start")
async def start_vatra(body: VatraStartRequest, request: Request,
                      user: dict = Depends(get_current_user)):
    """Launch a Vatra run from the UI. Creates a collaborative session and runs it
    in the background (the Lead decomposes inside execute_vatra — there's no
    separate route step). Returns immediately; the UI polls progress like any other
    running session. The agent path (`/agent/start`) is the other entry."""
    intent = (body.intent or "").strip()
    if not intent:
        raise HTTPException(400, "intent is required")
    db = get_db()
    title = (body.title or intent[:60]).strip()
    sess = await db.create_basna_session(
        user["id"], intent, title=title,
        config=json.dumps({"mode": "vatra", "source": "ui", "max_agents": body.max_agents}))
    sid = sess["id"]
    exec_req = ExecuteRequest(
        session_id=sid, tiers=body.tiers or None,
        env_vars=body.env_vars or None, api_key=body.api_key or "")
    # Background task with a stub request carrying the owner (spawn_process reads
    # request.state.user_id) — the real request object isn't safe to use post-response.
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(execute_vatra(exec_req, stub, user))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": sid, "title": title, "status": "running"}


# ── UI read endpoint: the blackboard for one session (user-scoped) ───

@router.get("/sessions/{session_id}/asks")
async def list_session_asks(session_id: str, user: dict = Depends(get_current_user)):
    """The Vatra blackboard for a session — every ask with its status, answer, and
    who answered it. Drives the ask-ledger + delegation-graph UI."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    return {"asks": await db.list_vatra_asks(session_id)}


@router.post("/agent/blackboard")
async def agent_blackboard(body: _AgentReq):
    """Read a Vatra session's persisted blackboard (every ask + answer) on behalf of
    the calling agent's owner. Lets a solo agent inspect how a past collaborative
    run delegated work, via the `basna` tool's `blackboard` action."""
    owner = _resolve_owner(body)
    if not body.session_id:
        raise HTTPException(400, "session_id is required")
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner)
    if not sess:
        raise HTTPException(404, "session not found")
    asks = await db.list_vatra_asks(body.session_id)
    return {"session_id": body.session_id, "count": len(asks), "asks": asks}


# ── Fire-and-forget entry (mirrors Basna's agent/start) ──────────────

async def _run_and_notify(user: dict, session_id: str, title: str, exec_req,
                          source_host: str, source_port: int, origin: dict) -> None:
    owner = user["id"]
    ok = False
    try:
        stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
        result = await execute_vatra(exec_req, stub, user)
        ok = True
        truth = (result.get("truth") or "").strip()
        n = len(result.get("subtasks") or [])
        summary = f"{n} piece(s) assembled.\n\n{truth[:1800]}{'…' if len(truth) > 1800 else ''}"
    except Exception as exc:
        log.warning("Agent-started Vatra failed", session_id=session_id, error=str(exc))
        summary = f"The run could not complete: {exc}"
    finally:
        runs = _active_agent_runs.get(owner)
        if runs:
            runs.discard(session_id)
            if not runs:
                _active_agent_runs.pop(owner, None)
    kind = str(origin.get("kind") or "").strip().lower()
    address = str(origin.get("address") or "").strip()
    if kind and kind != "web" and address:
        try:
            from captain_claw.flight_deck.delivery_routes import deliver_to_origin
            delivered, _ = await deliver_to_origin(
                {"kind": kind, "address": address}, f"🔥 Vatra — {title}\n\n{summary}")
            if delivered:
                return
        except Exception as exc:
            log.warning("Vatra origin delivery error; relaying via source agent", error=str(exc))
    await _notify_source_agent(
        source_host=source_host, source_port=source_port, origin=origin,
        title=title, session_id=session_id, ok=ok, summary=summary)


@router.post("/agent/start")
async def agent_start(body: AgentStartReq):
    """Start a Vatra run on behalf of the calling agent's owner (fire-and-forget).

    Reuses Basna's per-owner concurrency + run-rate guards (the shared dicts), then
    creates a session, runs the collaborative team in the background, and reports
    completion back to the agent. (Recursion is blocked worker-side in the basna
    tool, which refuses `start` when a CLAW_*_WORKER marker is set — the markers
    live in worker processes, not here in the FD process.)
    """
    owner = _resolve_owner(body)
    task = (body.task or "").strip()
    if not task:
        raise HTTPException(400, "task is required")

    active = _active_agent_runs.setdefault(owner, set())
    if len(active) >= _MAX_AGENT_RUNS_PER_OWNER:
        return {"status": "rejected",
                "reason": f"You already have {len(active)} run(s) in progress "
                          f"(limit {_MAX_AGENT_RUNS_PER_OWNER}). Wait for one to finish."}
    now_mono = time.monotonic()
    starts = _agent_run_starts.setdefault(owner, [])
    starts[:] = [s for s in starts if now_mono - s < _AGENT_RUN_WINDOW_SECONDS]
    if len(starts) >= _MAX_AGENT_RUNS_PER_WINDOW:
        return {"status": "rejected",
                "reason": f"Run-rate limit hit ({_MAX_AGENT_RUNS_PER_WINDOW} runs / "
                          f"{int(_AGENT_RUN_WINDOW_SECONDS / 60)} min) — cooling down."}
    starts.append(now_mono)

    db = get_db()
    user = await db.get_user_by_id(owner) or {"id": owner}
    tiers, env_vars = await _load_owner_tiers(db, owner)

    title = (body.title or task[:60]).strip()
    sess = await db.create_basna_session(
        user["id"], task, title=title,
        config=json.dumps({"mode": "vatra", "source": "agent",
                           "origin_platform": body.origin_platform,
                           "max_agents": body.max_agents}),
    )
    session_id = sess["id"]

    active.add(session_id)
    exec_req = ExecuteRequest(session_id=session_id, tiers=tiers or None, env_vars=env_vars or None)
    origin = {"platform": body.origin_platform, "user_id": body.origin_user_id,
              "chat_id": body.origin_chat_id, "kind": body.origin_kind,
              "address": body.origin_address}
    t = asyncio.create_task(_run_and_notify(
        user, session_id, title, exec_req, body.source_host, body.source_port, origin))
    _basna_agent_tasks.add(t)
    _agent_run_tasks[session_id] = t

    def _on_done(_t: Any, _sid: str = session_id) -> None:
        _basna_agent_tasks.discard(_t)
        _agent_run_tasks.pop(_sid, None)

    t.add_done_callback(_on_done)
    return {"status": "running", "session_id": session_id, "title": title}
