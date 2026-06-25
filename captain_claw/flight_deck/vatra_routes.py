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

from fastapi import APIRouter, HTTPException, Request

from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_db
from captain_claw.logging import get_logger

# Reuse Basna's standalone spine — these are pure/side-effect-isolated helpers and
# the shared run-tracking + notify plumbing. Only the coordinator below is new.
from captain_claw.flight_deck.basna_routes import (
    AgentStartReq,
    ExecuteRequest,
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
    _load_owner_tiers,
    _load_registry,
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


# ── Spawn / teardown (mirrors Basna's; stamped CLAW_VATRA_WORKER) ─────

async def _spawn_worker(request: Request, user: dict, *, name: str, description: str,
                        cognitive_mode: str, tools: list[str], tier: str,
                        tiers: dict | None, api_key: str, env_vars: list[dict] | None,
                        ) -> dict:
    """Spawn one ephemeral agent and resolve its web port. Returns
    {ok, slug, port, auth, message}. Strips run-starting tools and stamps the
    no-recursion marker so a Vatra worker can never launch another run."""
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
    rel_rows = await db.get_archetype_reliability(user["id"])
    reliability: dict[str, list[dict]] = {}
    for r in rel_rows:
        reliability.setdefault(r["archetype_id"], []).append(r)

    def _creds(tier: str) -> dict:
        lt = (body.tiers or {}).get(tier)
        if lt and lt.get("model"):
            return {"provider": lt.get("provider", "anthropic"), "model": lt.get("model", ""),
                    "base_url": lt.get("base_url") or None,
                    "api_key": lt.get("api_key") or body.api_key or None,
                    "output_ctx": int(lt.get("output_ctx") or 0)}
        return _tier_creds(registry, tier, body.api_key)

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

    # 1) Lead decomposes the task into owner-assigned subtasks.
    _progress(sid, "route", "Lead decomposing the task…")
    try:
        plan = await asyncio.wait_for(
            _llm_decompose(intent, archetypes, reliability, _creds("fast"), max_agents),
            _DECOMPOSE_TIMEOUT)
    except HTTPException:
        raise
    except Exception as e:
        await db.update_basna_session(sid, user["id"], status="error")
        _progress(sid, "route", f"Lead decomposition failed: {str(e)[:200]}", ok=False)
        raise HTTPException(502, f"Vatra Lead failed: {e}")
    domain = plan["domain"]
    subtasks = plan["subtasks"]
    # Persist the plan as the session route (mirrors Basna's `selected` so the
    # existing UI / basna read-tool render owners).
    selected = [{"archetype_id": s["owner_archetype_id"],
                 "role": arch_by_id[s["owner_archetype_id"]].get("role", ""),
                 "why": s["title"]} for s in subtasks]
    route = {"mode": "vatra", "domain": domain, "rationale": plan["rationale"],
             "subtasks": subtasks, "selected": selected}
    await db.update_basna_session(
        sid, user["id"], domain=domain, route=json.dumps(route),
        config=json.dumps({**cfg, "mode": "vatra"}),
    )
    _progress(sid, "route", f"{len(subtasks)} subtask(s) · {domain}")

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
            sp = await _spawn_worker(
                request, user,
                name=f"vatra-{sid8}-{run_tag}-{arch['id']}",
                description=f"Vatra subtask · {arch.get('role', '')}",
                cognitive_mode=arch.get("cognitive_mode", "neutra"),
                tools=arch.get("tools") or [], tier=tier, tiers=tiers,
                api_key=body.api_key, env_vars=body.env_vars,
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

        # 3) Dispatch each owner its self-contained brief, in parallel.
        async def _dispatch_owner(sp: dict) -> dict:
            arch, st = sp["arch"], sp["subtask"]
            role = arch.get("role") or arch["id"]
            fleet = arch.get("fleet_instructions", "")

            def _on_action(act: dict) -> None:
                detail = act.get("detail", "")
                if act["tool"] == "narration":
                    _progress(sid, "narration", f"{role}: {detail}", agent=role,
                              tool="narration", detail=detail)
                else:
                    suffix = f": {detail}" if detail else ""
                    _progress(sid, "action", f"{role} → {act['tool']}{suffix}",
                              agent=role, tool=act["tool"], detail=detail)

            from captain_claw.flight_deck.server import DATA_DIR
            ws = DATA_DIR / sp["slug"] / "data" / "workspace"
            img = [str(ws / f["name"]) for f in input_files if str(f.get("mime", "")).startswith("image/")]
            doc = [str(ws / f["name"]) for f in input_files if not str(f.get("mime", "")).startswith("image/")]
            prompt = _build_subtask_prompt(role, intent, st, [f["name"] for f in input_files])
            d = await _dispatch_one(
                sp["port"], sp["auth"], prompt, body.dispatch_timeout,
                on_action=_on_action, fleet_instructions=fleet, agent_name=role,
                file_paths=doc, image_paths=img)
            mark = "✓" if d["ok"] else "✗"
            extra = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
            _progress(sid, "dispatch",
                      f"{role} {mark} · {len(d['actions'])} action(s) ({d['latency_ms'] / 1000:.1f}s){extra}",
                      ok=d["ok"], agent=role)
            return d

        dispatched = await asyncio.gather(*[_dispatch_owner(sp) for sp in spawned])
        for sp, d in zip(spawned, dispatched):
            results.append({
                "id": sp["subtask"]["id"], "owner": sp["arch"]["id"],
                "role": sp["arch"].get("role", ""), "title": sp["subtask"]["title"],
                "output": d["output"], "ok": d["ok"],
                "latency_ms": d["latency_ms"], "actions": d.get("actions", []),
            })

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

    # 5) Reporter assembles the slices into one deliverable.
    truth, reporter_files = await _run_reporter(
        request, user, sid, sid8, run_tag, intent, usable, cfg, arch_by_id,
        tiers=body.tiers, api_key=body.api_key, env_vars=body.env_vars,
        dispatch_timeout=body.dispatch_timeout, input_names=input_names,
        dest_dir=dest_dir, seen_gen=seen_gen,
    )
    generated_files.extend(reporter_files)
    confidence = round(len(usable) / max(1, len(results)), 3)

    # 6) Persist runs (success=None — scoring is Phase 3) so the UI/read-tool work.
    await db.add_basna_runs(sid, user["id"], [{
        "archetype_id": r["owner"], "role": r["role"], "tier": "",
        "weight_at_run": 0.0, "output": r["output"],
        "actions": json.dumps(r.get("actions", [])),
        "latency_ms": r["latency_ms"], "success": None,
    } for r in results])

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
            "spawned": len(spawned), "dispatched": len(results)}


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
        f"Return only your finished part — no preamble, no meta-commentary about the team."
    )


async def _run_reporter(request: Request, user: dict, sid: str, sid8: str, run_tag: str,
                        intent: str, usable: list[dict], cfg: dict, arch_by_id: dict, *,
                        tiers, api_key, env_vars, dispatch_timeout, input_names,
                        dest_dir: Path, seen_gen: set[str]) -> tuple[str, list[dict]]:
    """Spawn a dedicated reporter, feed it the slices, and capture the assembled
    deliverable. Falls back to a labeled concatenation if the reporter fails."""
    from captain_claw.flight_deck.server import DATA_DIR

    reporter_id = str(cfg.get("reporter_archetype") or _DEFAULT_REPORTER)
    arch = arch_by_id.get(reporter_id) or arch_by_id.get(_DEFAULT_REPORTER)
    fallback = "\n\n".join(f"## {r['title']} ({r['role']})\n{r['output'].strip()}" for r in usable)
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
    _run_workers[sid] = [sp["slug"]]
    slices_full = "\n\n".join(
        f"### Piece: {r['title']} — by {r['role']}\n{r['output'].strip()}" for r in usable)
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

        d = await _dispatch_one(sp["port"], sp["auth"], prompt, dispatch_timeout,
                                on_action=_on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                                agent_name=role)
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
        _run_workers.pop(sid, None)


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
    completion back to the agent.
    """
    if str(__import__("os").environ.get(_WORKER_MARKER, "")).strip().lower() in ("1", "true", "yes"):
        raise HTTPException(400, "Vatra runs cannot be started from inside a run")
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
