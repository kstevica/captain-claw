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
import re
import shutil
import time
import types
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db

# Reuse Basna's standalone spine — these are pure/side-effect-isolated helpers and
# the shared run-tracking + notify plumbing. Only the coordinator below is new.
from captain_claw.flight_deck.basna_routes import (
    _AGENT_RUN_WINDOW_SECONDS,
    _INSTRUCTIONS_DIR,
    _MAX_AGENT_RUNS_PER_OWNER,
    _MAX_AGENT_RUNS_PER_WINDOW,
    _PROGRESS,
    AgentStartReq,
    ExecuteRequest,
    _active_agent_runs,
    _agent_run_starts,
    _agent_run_tasks,
    _AgentReq,
    _basna_agent_tasks,
    _build_catalog,
    _closer_on_event,
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
    _make_gate,
    _resolve_owner,
    _round_filename_rule,
    _RUN_USAGE,
    _run_gate,
    _run_sid,
    _run_workers,
    _session_files_dir,
    _tier_creds,
    _vfs_manifest,
)
from captain_claw.flight_deck import facts_ledger
from captain_claw.flight_deck import quality_findings
from captain_claw.flight_deck import research_brief
from captain_claw.flight_deck import research_consistency
from captain_claw.flight_deck import research_contract
from captain_claw.flight_deck import research_map
from captain_claw.flight_deck import research_rubric
from captain_claw.flight_deck import vatra_groups
from captain_claw.flight_deck.horizon_worker import HorizonConfig, run_horizon_closer
from captain_claw.flight_deck.quality_profile import (
    ACTED_CORRECTIVE,
    ESCALATE_CORRECTIVE,
    ESCALATE_DIRECTIVE,
    FACTS_LEDGER_DIRECTIVE,
    JUDGMENT_LEDGER_DIRECTIVE,
    REPORTER_FACTS_DIRECTIVE,
    REPORTER_HONESTY_DIRECTIVE,
    SOURCE_CORPUS_DIRECTIVE,
    UNVERIFIED_GUARD_DIRECTIVE,
    QualityProfile,
    TokenBudget,
    build_quality_metrics,
    escalate_reason,
    output_mode_directive,
    worker_produced_nothing,
)
from captain_claw.logging import get_logger
from captain_claw.vfs import resolve_under as _vfs_resolve_under

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
# The Lead's plan is the single heaviest planning call (a full decomposition +
# shared contract, up to a large token cap). Generous so a slow LOCAL model can
# finish it — cloud models return in a few seconds regardless.
_DECOMPOSE_TIMEOUT = 300  # seconds for the Lead LLM call


def _lead_error_msg(e: Exception) -> str:
    """A readable reason for a Lead-decompose failure. A bare TimeoutError
    stringifies to '' (the mysterious 'Vatra Lead failed:' with no detail), so
    name it and make it actionable."""
    if isinstance(e, TimeoutError):  # asyncio.TimeoutError is TimeoutError on 3.11+
        return (f"the Lead timed out after {_DECOMPOSE_TIMEOUT}s — the planning model is too "
                "slow for this. Try a faster Router tier, fewer Max agents, or a shorter task.")
    return str(e).strip() or type(e).__name__

# ── Phase 2 delegation budget (the termination guarantees) ───────────
_MAX_ASKS = 12          # total asks a single run may ever create (hard ceiling)
_MAX_ASK_DEPTH = 2      # an answer that itself asks increments depth; caps cascades
_MAX_HELPERS = 3        # concurrent helpers the coordinator may run at once
_COORD_POLL_S = 1.5     # how often the coordinator polls the blackboard
_INBOX_POLL_S = 1.0     # inbox long-poll granularity
_WAIT_POLL_S = 1.0      # `wait` long-poll granularity
_MAX_WAIT_S = 120       # max a SINGLE `vatra wait` may block; < dispatch timeout so it can't hang a run
_WAIT_MIN_S = 30        # floor for a single wait, so a short poll still gives the dep time to land
_WAIT_TOTAL_BUDGET_S = 300   # total seconds ONE owner may spend waiting across retries in a run
_WAIT_MAX_ATTEMPTS = 6       # hard cap on wait CALLS per owner — a loop backstop, belt-and-suspenders
_WAIT_CONTENT_CAP = 20_000  # chars of a ready file handed back to the waiter
# Per-run, per-owner wait ledger (the "stack"): session_id -> owner -> {waited, attempts}.
# Enforces the total budget so retries can never become an infinite loop; cleared on teardown.
_wait_ledger: dict[str, dict[str, dict[str, float]]] = {}
# Live grouped-run schedule per session — published by the run loop so worker-
# facing endpoints (wait) can answer "has that teammate even started?".
# sid → {"current": ordinal, "done": set[subtask_id], "owners": [...]}
_group_schedule: dict[str, dict] = {}


def _phase(sid: str, label: str, **extra) -> None:
    """Emit one high-level phase banner (Planning / Intro / Main / Synthesizing …)
    so the live UI can always show which stage of the run is active, separate from
    the noisy per-action detail events."""
    _progress(sid, "phase", label, **extra)


# Per-run override of the shared VFS folder, keyed by session id. A continuation
# round registers the ROOT run's folder here so EVERY resolution site below
# (worker env, prompt directives, reporter, ask helpers, agent_wait) lands on the
# same accumulated data — without threading a project param through each one.
# Populated at the top of execute_vatra, cleared in its teardown.
_run_vfs_project: dict[str, str] = {}

# Per-run flag: when set, every worker is bound to the run's folder-scoped shared
# datastore (vfs:<project>/.datastore). Populated at the top of execute_vatra,
# cleared in teardown — mirrors _run_vfs_project.
_run_shared_datastore: dict[str, bool] = {}


def _vfs_project(sid: str) -> str:
    """The single shared VFS project folder for this run.

    Source of truth for both the injected CLAW_VFS_PROJECT default and the
    folder pinned into every worker prompt, so all agents write to ONE place.
    A continuation round inherits the root run's folder via _run_vfs_project;
    a fresh run derives the folder from its own session id.
    """
    return _run_vfs_project.get(sid) or f"vatra-{sid[:8]}"


def _augment_tools(tools: list[str], research_dir: Path | None,
                   facts: bool = False) -> list[str]:
    """Add the run-scoped tools this run armed: `researchmap` when the Research
    Map is on, `facts` when the shared facts ledger is on."""
    tools = list(tools or [])
    if research_dir is not None and "researchmap" not in tools:
        tools = tools + ["researchmap"]
    if facts and "facts" not in tools:
        tools = tools + ["facts"]
    return tools


def _vfs_directive(project: str) -> str:
    """Mandatory instruction pinning every worker to one shared VFS folder.

    Without this, co-spawned agents each invent their own project name
    (game-suite, the bare session id, …) and the pieces never co-locate.
    """
    if not project:
        return ""
    return (
        "\n\n## Shared VFS project — MANDATORY\n"
        "Write EVERY file you produce to the shared cross-agent filesystem under this "
        "EXACT project folder:\n"
        f"  vfs:{project}/<filename>\n"
        f"Use `vfs:{project}/` verbatim. Do NOT invent a folder, do NOT derive one from the "
        "task or the session id, and do NOT create a new project — if you do, your files "
        "won't sit next to your teammates' and nothing will link up. Every teammate writes "
        f"to the SAME vfs:{project}/ folder, and you read theirs from there too "
        f"(e.g. read vfs:{project}/style.css)."
    )


def _datastore_directive(project: str, enabled: bool) -> str:
    """When the run's shared datastore is on, tell workers to collaborate through
    ONE relational store (the `datastore` tool) instead of improvising with JSON
    files or one-off scripts. Without this they don't know a shared store exists."""
    if not enabled or not project:
        return ""
    return (
        "\n\n## Shared team datastore — USE IT FOR STRUCTURED DATA\n"
        "This team shares ONE relational datastore (the `datastore` tool). Every teammate "
        "reads and writes the SAME tables — the single source of truth for structured data, "
        "and the TRACK a later run continues from (this run may itself be continuing earlier work).\n"
        "- BEFORE producing anything: `datastore(action=\"list_tables\")` and query the relevant "
        "tables to see what earlier runs already did. Do ONLY the outstanding work — don't redo "
        "rows that are already present and marked done.\n"
        "- Put records / rows / lists / results in the datastore, NOT ad-hoc JSON files or scripts.\n"
        "- Make writes IDEMPOTENT: create the table with a stable unique key + a `status` column "
        "— `datastore(action=\"create_table\", table=\"…\", columns=[…], unique=[\"<id>\"])` — then "
        "`datastore(action=\"upsert\", table=\"…\", rows=[…])` so re-running an item UPDATES its row "
        "instead of duplicating it. Set each row's status (e.g. \"done\") so the next run can resume.\n"
        "- Read teammates' data: `datastore(action=\"list_tables\")` and `datastore(action=\"query\", table=\"…\")`.\n"
        f"- Ignore vfs:{project}/.history/ and vfs:{project}/.datastore/ — those are system folders "
        "(backups + the raw DB); never read or write them directly.\n"
        f"Prose and reports still go to files under vfs:{project}/; structured data goes in the datastore."
    )


def _reference_directive(folders: list[str]) -> str:
    """Tell agents about READ-ONLY reference VFS folders (prior runs' folders + any
    the user added) to consult BEFORE web-searching. '' if none."""
    fs = list(dict.fromkeys(f.strip() for f in (folders or []) if f and f.strip()))
    if not fs:
        return ""
    listed = "\n".join(f"  - vfs:{f}/" for f in fs)
    return (
        "\n\n## Reference folders (READ-ONLY) — check these BEFORE a web search\n"
        "These VFS folders may already hold relevant material from earlier work. When you "
        "need background/context, or would otherwise run a web search, look HERE FIRST (if "
        "they exist):\n"
        f"{listed}\n"
        "- Use `glob vfs:<folder>/**/*` to see what's there, then `read` the relevant files.\n"
        "- Each folder may ALSO have a relational datastore from that run — READ it with "
        "`datastore(action=\"list_tables\", project=\"<folder>\")` and "
        "`datastore(action=\"query\", table=\"…\", project=\"<folder>\")`. Reuse that data "
        "instead of recomputing it.\n"
        "- Only web-search for what these folders don't cover. Treat them as READ-ONLY — "
        "never write into them."
    )


def _group_instr_block(st: dict, arch: dict, group_instructions: dict) -> str:
    """Per-group extra instructions the user attached in the team-plan editor,
    injected into every owner that runs in that group. '' if none for its group."""
    if not group_instructions:
        return ""
    letter = vatra_groups.group_label(vatra_groups.effective_group(st, arch))
    instr = str((group_instructions or {}).get(letter) or "").strip()
    if not instr:
        return ""
    return f"\n\n## Group {letter} — additional instructions for this phase\n{instr}"


# The archetype that runs as the permanent Group 0 pre-phase — it drafts the
# per-agent coordination plan the whole team executes against.
_GROUP0_PLANNER_ID = "long-horizon-planner"


def _plan_slice_block(st: dict, group0_by_subtask: dict, arch_by_id: dict) -> str:
    """This owner's slice of the Group 0 coordination plan: its mandate, the artifact
    it produces, which named teammates it consumes from, and hand-off notes. Returns
    '' when there is no plan or no entry for this subtask — so resume/legacy runs (no
    ``group0_plan``) emit byte-identical prompts."""
    e = (group0_by_subtask or {}).get(st["id"])
    if not e:
        return ""
    lines = ["\n\n## Your coordination plan (Group 0)"]
    if e.get("mandate"):
        lines.append(f"Your mandate: {e['mandate']}")
    if e.get("produces"):
        lines.append(f"You produce: {e['produces']}")
    cons = [c for c in (e.get("consumes_from") or []) if c]
    if cons:
        parts = []
        for cid in cons:
            src = (group0_by_subtask or {}).get(cid) or {}
            arch = arch_by_id.get(str(src.get("agent_id") or "")) or {}
            role = arch.get("role") or src.get("agent_id") or cid
            produced = str(src.get("produces") or "").strip()
            parts.append(f"{role} — {produced}" if produced else str(role))
        lines.append("You consume from these teammates (read their output before "
                     "you start, via the `vatra` tool): " + "; ".join(parts))
    if e.get("hand_off_notes"):
        lines.append(f"Hand-off notes for downstream teammates: {e['hand_off_notes']}")
    return "\n".join(lines)


def _vatra_env(sid: str, subtask: str, owner: str, depth: int) -> list[dict]:
    """Run-context env injected into a worker so the `vatra` tool knows where it is."""
    project = _vfs_project(sid)
    env = [
        {"key": "CLAW_VATRA_SESSION", "value": sid},
        {"key": "CLAW_VATRA_SUBTASK", "value": subtask},
        {"key": "CLAW_VATRA_OWNER", "value": owner},
        {"key": "CLAW_VATRA_DEPTH", "value": str(depth)},
        # Auto-bind every worker in this run to one shared VFS project so
        # they keep a common file context (vfs:<project>/...).
        {"key": "CLAW_VFS_PROJECT", "value": project},
    ]
    # Opt-in: bind workers to the run's folder-scoped shared datastore too.
    if _run_shared_datastore.get(sid):
        env.append({"key": "CLAW_DATASTORE_VFS", "value": project})
    return env


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


# Per-run set of agent labels the user has asked to skip (cancel their current turn
# and move on). Keyed by session id; the label matches the live-panel card.
_skip_agents: dict[str, set[str]] = {}


def _is_skipped(sid: str, label: str) -> bool:
    return label in _skip_agents.get(sid, ())


async def _dispatch_skippable(sid: str, label: str, factory) -> dict:
    """Run an agent dispatch; if the user marks this agent (label) to skip, cancel
    its turn and return a skipped result instead of waiting for it. `factory` is a
    no-arg callable returning the `_dispatch_one` coroutine. Token spend is recorded
    inside `_dispatch_one` (via the run contextvar), so no explicit tracking here."""
    task = asyncio.create_task(factory())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if task in done:
                return task.result()
            if _is_skipped(sid, label):
                task.cancel()
                try:
                    await task
                except BaseException:  # CancelledError + any teardown error
                    pass
                _skip_agents.get(sid, set()).discard(label)
                _progress(sid, "dispatch", f"{label} — skipped by user", ok=False, agent=label)
                return {"ok": False, "output": "", "actions": [], "latency_ms": 0, "skipped": True}
    finally:
        if sid in _skip_agents and not _skip_agents[sid]:
            _skip_agents.pop(sid, None)


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
        deps = s.get("depends_on") or []
        subtasks.append({
            "id": sid,
            "title": (str(s.get("title") or "").strip() or owner)[:80],
            "owner_archetype_id": owner,
            "brief": brief,
            "depends_on": [str(d).strip() for d in deps if isinstance(d, (str, int))],
            # Optional Lead-assigned execution group ('A'..'D'); clamped to the
            # archetype's preset floor at run time (grouped mode only).
            "group": s.get("group"),
        })
    # Keep only dependency refs that point at a real sibling (drop self + danglers).
    valid_ids = {s["id"] for s in subtasks}
    for s in subtasks:
        s["depends_on"] = [d for d in s["depends_on"] if d in valid_ids and d != s["id"]]
    # Pin final execution groups, repairing dependency inversions (a piece whose
    # output another piece consumes must never be scheduled after it — a recorded
    # dependency out-ranks archetype floors AND Lead pushes). Grouped mode reads
    # the pins; ungrouped runs ignore them. Notes surface in the run log.
    group_repairs = vatra_groups.resolve_groups(subtasks, arch_by_id)
    for note in group_repairs:
        log.info("Vatra group repair", note=note)
    return {"domain": domain, "rationale": str(raw.get("rationale") or "").strip(),
            "shared_context": str(raw.get("shared_context") or "").strip(),
            "subtasks": subtasks,
            "group_repairs": group_repairs}


async def _llm_decompose(intent: str, archetypes: list[dict], reliability: dict,
                         creds: dict, max_agents: int,
                         force_ids: list[str] | None = None,
                         shared_datastore: bool = False,
                         state_manifest: str = "",
                         prior_knowledge: str = "") -> dict:
    """Ask the Lead to split the task into complementary, owner-assigned subtasks.

    Returns a normalized plan. On any LLM/parse failure raises — the caller turns
    that into a clean run failure (Phase 1 has no deterministic fallback planner).
    `force_ids` fixes the team: the Lead must give each a subtask.
    """
    from captain_claw.llm import Message
    system_file = _INSTRUCTIONS_DIR / "vatra" / "lead.md"
    if not system_file.is_file():
        raise HTTPException(500, "Vatra lead prompt not found")
    arch_by_id = {a["id"]: a for a in archetypes}
    system_prompt = system_file.read_text() + "\n\n## Catalog\n" + _build_catalog(archetypes, reliability)
    if shared_datastore:
        # The team shares one relational datastore this run — plan for it, or the
        # Lead defaults "datastore" to a JSON file and the whole team follows suit.
        system_prompt += (
            "\n\n## Shared team datastore is ENABLED for this run\n"
            "The team shares ONE relational datastore (the `datastore` tool). It is also the "
            "TRACK a later run resumes from — and this run may itself be CONTINUING earlier work "
            "already sitting in the folder's datastore.\n"
            "When the task involves saving / storing / persisting structured data:\n"
            "- Plan persistence into the shared datastore (create_table + upsert) — NOT an ad-hoc "
            "JSON or CSV file. Do NOT describe 'the datastore' as a local JSON file.\n"
            "- In `shared_context`, pin the SCHEMA: table name(s), columns, a STABLE UNIQUE KEY, and "
            "a `status` column. Owners UPSERT on that key (idempotent) and set status, so a re-run "
            "updates rows instead of duplicating them and can skip already-done work.\n"
            "- Tell owners to CHECK the datastore first (list_tables / query) and do only the "
            "OUTSTANDING work — that is how runs continue instead of restarting."
        )
    user_prompt = (
        f"Task: {intent}\n\n"
        f"max_agents: {max_agents}. Decompose into the smallest set of complementary, "
        f"owner-assigned subtasks that together cover this task."
    )
    if force_ids:
        forced_list = "\n".join(f"- {a}: {arch_by_id[a].get('role', '')}" for a in force_ids if a in arch_by_id)
        user_prompt += (
            "\n\nThe team is FIXED by the user — you MUST create at least one subtask owned by "
            "EACH of these archetypes, and use only these as owners:\n" + forced_list +
            "\nGive each a meaningful, complementary piece derived from the task; split the work "
            "so every one of them has a real, non-overlapping part.")
    if state_manifest:
        user_prompt += state_manifest
    if prior_knowledge:
        user_prompt += prior_knowledge
    # The plan now carries shared_context + per-piece briefs + depends_on, so it can
    # be long — a tight cap truncates the JSON and the whole route fails. Give it room.
    prov, mt = _provider_call(creds, temperature=0.2, default_max=8192, cap=16384)
    resp = await prov.complete(messages=[
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ], temperature=0.2, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    truncated = str(getattr(resp, "finish_reason", "") or "").lower() in (
        "length", "max_tokens", "max_output_tokens", "max_completion_tokens")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        log.warning("Vatra Lead JSON parse failed",
                    chars=len(content), truncated=truncated, tail=content[-200:])
        if truncated:
            raise HTTPException(502, (
                "The Lead's plan was cut off at the output-token cap before the JSON "
                "finished. Try fewer max agents, a simpler task, or a planning tier with a "
                "higher output limit."))
        raise HTTPException(502, f"The Lead returned invalid JSON: {e}")
    plan = _normalize_plan(raw, arch_by_id, max_agents)
    if not plan["subtasks"]:
        raise HTTPException(422, "The Lead produced no usable subtasks — try rephrasing the task.")
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


async def _build_plan(db, user_id: str, intent: str, max_agents: int, creds: dict,
                      force_ids: list[str] | None = None,
                      shared_datastore: bool = False, vfs_project: str = "",
                      prior_knowledge: str = "") -> dict:
    """Run the Lead and shape the result into a persistable Vatra route:
    {mode, domain, rationale, subtasks, selected}. `selected` mirrors Basna's
    shape so the read-tool and list UI render the owners. `force_ids` fixes the
    team — every one of them is guaranteed a subtask."""
    archetypes = await merged_archetypes(db, user_id)
    arch_by_id = {a["id"]: a for a in archetypes}
    rel_rows = await db.get_archetype_reliability(user_id)
    reliability: dict[str, list[dict]] = {}
    for r in rel_rows:
        reliability.setdefault(r["archetype_id"], []).append(r)
    forced = [a for a in (force_ids or []) if a in arch_by_id]
    # A fixed team must all fit, even if larger than the requested max.
    cap = max(max_agents, len(forced)) if forced else max_agents
    # Seed the Lead with what the target folder already holds, so it plans to
    # CONTINUE prior work rather than restart (only when a shared datastore + an
    # existing folder are in play).
    state_manifest = ""
    if shared_datastore and vfs_project:
        try:
            from captain_claw.flight_deck.vfs_routes import folder_state_manifest
            state_manifest = await folder_state_manifest(user_id, vfs_project)
        except Exception as e:  # noqa: BLE001 — best-effort seeding
            log.debug("Vatra plan state manifest failed", error=str(e))
    plan = await asyncio.wait_for(
        _llm_decompose(intent, archetypes, reliability, creds, cap, force_ids=forced or None,
                       shared_datastore=shared_datastore, state_manifest=state_manifest,
                       prior_knowledge=prior_knowledge),
        _DECOMPOSE_TIMEOUT)
    subtasks = plan["subtasks"]
    # Guarantee every fixed-team archetype actually got a piece — if the Lead missed
    # one, add a task-derived subtask for it so "all selected are used" holds.
    if forced:
        covered = {s["owner_archetype_id"] for s in subtasks}
        n = len(subtasks)
        for aid in forced:
            if aid not in covered:
                n += 1
                role = arch_by_id[aid].get("role", aid)
                subtasks.append({
                    "id": f"s{n}", "title": f"{role} contribution",
                    "owner_archetype_id": aid,
                    "brief": (f"As the {role}, contribute your part to this task from your "
                              f"specialty's perspective: {intent[:400]}"),
                    "depends_on": [],
                })
    selected = [{"archetype_id": s["owner_archetype_id"],
                 "role": arch_by_id[s["owner_archetype_id"]].get("role", ""),
                 "why": s["title"]} for s in subtasks]
    # Fold prior-run knowledge into shared_context so EVERY worker (not just the
    # Lead) sees it verbatim, alongside the conventions the Lead produced.
    shared_context = plan.get("shared_context", "")
    if prior_knowledge:
        shared_context = (prior_knowledge.strip() + "\n\n" + shared_context).strip()
    return {"mode": "vatra", "domain": plan["domain"], "rationale": plan["rationale"],
            "shared_context": shared_context,
            "subtasks": subtasks, "selected": selected}


# ── Spawn / teardown (mirrors Basna's; stamped CLAW_VATRA_WORKER) ─────

# Subtasks whose shape suits a 2-4B model under the mrav 8k cap: mechanical
# extraction / digestion / formatting, not open-ended reasoning. Matched
# against subtask title+desc+role when the `micro_workers` lever is on.
_MICRO_SUITED_RE = re.compile(
    r"\b(extract|digest|summar|format|reformat|convert|collect|compile|"
    r"catalog|list|tabulat|scan|gather|dedup|normali[sz])", re.IGNORECASE)


def _micro_suited(text: str) -> bool:
    """True when a subtask's wording marks it as micro-runtime material."""
    return bool(_MICRO_SUITED_RE.search(text or ""))


async def _spawn_worker(request: Request, user: dict, *, name: str, description: str,
                        cognitive_mode: str, tools: list[str], tier: str,
                        tiers: dict | None, api_key: str, env_vars: list[dict] | None,
                        extra_env: list[dict] | None = None,
                        corpus: bool = False,
                        micro: bool = False,
                        ) -> dict:
    """Spawn one ephemeral agent and resolve its web port. Returns
    {ok, slug, port, auth, message}. Strips the run-starting `basna` tool and
    stamps the no-recursion marker so a Vatra worker can never launch another run;
    the `vatra` ask/inbox tool is registered unconditionally and stays available.
    `extra_env` carries the run context (session/subtask/owner/depth)."""
    from captain_claw.flight_deck.server import (
        AgentConfig,
        _load_process_registry,
        spawn_process,
    )
    lt = (tiers or {}).get(tier) or {}
    provider = lt.get("provider") or ""
    model = lt.get("model") or ""
    key = lt.get("api_key") or api_key or ""
    base_url = lt.get("base_url") or ""
    max_tokens = int(lt.get("output_ctx") or 0) or 32768
    max_context = int(lt.get("input_ctx") or 0)

    # S3 (mrav Phase 4): a micro-suited subtask spawns the same worker
    # process, same transport, but runs the mrav micro loop on the owner's
    # `micro` tier. An explicit tier "micro" ALWAYS means mrav — a classic
    # agent on a 2-4B model with the full tool-schema prompt is exactly the
    # failure mode mrav exists to avoid. Default off → byte-identical.
    runtime = ""
    if micro or tier == "micro":
        runtime = "mrav"
        mt = (tiers or {}).get("micro") or {}
        if str(mt.get("model") or "").strip():
            provider = mt.get("provider") or provider
            model = mt["model"]
            key = mt.get("api_key") or key
            base_url = mt.get("base_url") or ""
            max_tokens = int(mt.get("output_ctx") or 0) or 1024
            max_context = int(mt.get("input_ctx") or 0) or 8192

    worker_tools = [t for t in (tools or AgentConfig().tools) if t != "basna"]
    base = dict(
        name=name, description=description,
        cognitive_mode=cognitive_mode or "neutra", tools=worker_tools,
        runtime=runtime,
        env_vars=(env_vars or []) + (extra_env or []) + [{"key": _WORKER_MARKER, "value": "1"}]
        + ([{"key": "CLAW_SOURCE_CORPUS", "value": "1"}] if corpus else []),  # R10
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
        DATA_DIR,
        _do_stop_process,
        _load_process_registry,
        _processes,
        _save_process_registry,
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


def _save_inputs_to_vfs(user_id: str, vfs_project: str, src_dir: Path,
                        input_files: list[dict]) -> int:
    """Copy the user's attached input files into the run's shared VFS folder so they
    live in the corpus (browsable, and indexed by the research map) — not only in each
    worker's throwaway workspace. Best-effort; returns the count copied."""
    n = 0
    try:
        from captain_claw.flight_deck.vfs_routes import _user_root
        dst = _user_root(user_id) / vfs_project
        dst.mkdir(parents=True, exist_ok=True)
        for f in input_files:
            src = src_dir / f["name"]
            if src.is_file():
                try:
                    shutil.copy2(src, dst / f["name"])
                    n += 1
                except OSError as e:
                    log.warning("Vatra input→VFS copy failed", file=f["name"], error=str(e))
    except Exception as e:  # noqa: BLE001 — never block a run on the corpus copy
        log.warning("Vatra input→VFS setup failed", error=str(e))
    return n


async def _examine_files_brief(
    request: Request, user: dict, *, name: str, intent: str,
    input_files: list[dict], src_dir: Path, tiers: dict | None, api_key: str,
    env_vars: list[dict] | None, timeout: float,
) -> str:
    """Spawn a short-lived agent that OPENS the attached files and restates the task as
    a file-aware brief. Best-effort — returns "" on any failure so the caller falls back
    to the plan/text brief. Reuses the worker spawn + `_dispatch_one` plumbing, so the
    examiner's tokens are cost-accounted like any other agent (via the run contextvar).
    Emits only progress-log lines (no `agent=` events), so it never forms a phantom card."""
    sp = await _spawn_worker(
        request, user, name=name, description="Examine attached files",
        cognitive_mode="neutra", tools=["read", "ls"], tier="reason", tiers=tiers,
        api_key=api_key, env_vars=env_vars)
    if not sp["ok"]:
        return ""
    try:
        from captain_claw.flight_deck.server import DATA_DIR
        ws = DATA_DIR / sp["slug"] / "data" / "workspace"
        for f in input_files:
            try:
                shutil.copy2(src_dir / f["name"], ws / f["name"])
            except OSError:
                pass
        img = [str(ws / f["name"]) for f in input_files if str(f.get("mime", "")).startswith("image/")]
        doc = [str(ws / f["name"]) for f in input_files if not str(f.get("mime", "")).startswith("image/")]
        prompt = research_brief.derive_brief_with_files_prompt(
            intent, [f["name"] for f in input_files])
        d = await _dispatch_one(sp["port"], sp["auth"], prompt, timeout,
                                agent_name="File examiner", file_paths=doc, image_paths=img)
        return research_brief.parse_brief(d.get("output") or "") if d.get("ok") else ""
    finally:
        _teardown([sp["slug"]])


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


# ── Group 0 pre-phase: plan → gate ───────────────────────────────────

async def _ensure_route(db, user_id: str, sess: dict, sid: str, *, intent: str,
                        max_agents: int, creds: dict, cfg: dict,
                        shared_datastore: bool, vfs_project: str) -> dict:
    """Reuse a route prepared by the UI's /route step if present; otherwise decompose
    now and persist it. Shared by the Group 0 pre-phase and ``execute_vatra`` so both
    paths decompose identically. Emits the same route progress lines. Raises
    HTTPException (after marking the session errored) if the Lead fails."""
    try:
        existing = json.loads(sess.get("route") or "{}")
    except json.JSONDecodeError:
        existing = {}
    if existing.get("mode") == "vatra" and existing.get("subtasks"):
        _progress(sid, "route",
                  f"Using prepared plan · {len(existing['subtasks'])} piece(s) · "
                  f"{existing.get('domain', '')}")
        return existing
    _progress(sid, "route", "Lead decomposing the task…")
    try:
        # A plan-step child can fix the team via config.force_ids.
        _force = [str(a) for a in (cfg.get("force_ids") or []) if str(a).strip()]
        route = await _build_plan(db, user_id, intent, max_agents, creds,
                                  force_ids=_force or None, shared_datastore=shared_datastore,
                                  vfs_project=vfs_project)
    except HTTPException:
        await db.update_basna_session(sid, user_id, status="error")
        raise
    except Exception as e:
        await db.update_basna_session(sid, user_id, status="error")
        _msg = _lead_error_msg(e)
        _progress(sid, "route", f"Lead decomposition failed: {_msg[:200]}", ok=False)
        raise HTTPException(502, f"Vatra Lead failed: {_msg}")
    await db.update_basna_session(
        sid, user_id, domain=route["domain"], route=json.dumps(route),
        config=json.dumps({**cfg, "mode": "vatra"}))
    _progress(sid, "route", f"{len(route['subtasks'])} subtask(s) · {route['domain']}")
    return route


def _emit_awaiting_plan(sid: str, plan: dict) -> None:
    """Signal the poll-based UI that a coordination plan is ready for review. The
    plan rides on the event so the client can seed its editor without a second fetch;
    the authoritative copy is also persisted in ``route['group0_plan']``."""
    _phase(sid, "Awaiting plan approval")
    _progress(sid, "awaiting_plan",
              "Plan ready for your review — edit if needed, then Execute.", plan=plan)
    _progress_done(sid)


async def _run_group0_planner(request: Request, user: dict, sid: str, *, intent: str,
                              shared_context: str, file_names: list[str],
                              subtasks: list[dict], arch_by_id: dict, tiers: dict | None,
                              api_key: str, env_vars: list[dict] | None,
                              timeout: float, clarifications: str = "") -> dict:
    """Spawn the Long Horizon Planner, dispatch it to draft the per-agent coordination
    plan, and return the parsed structured plan. Best-effort: any spawn/dispatch/parse
    failure yields the pass-through plan so a dead planner never blocks the run. Emits
    ``agent='Long Horizon Planner'`` events so it renders as a live pre-phase card."""
    label = "Long Horizon Planner"
    _progress(sid, "group0", "Long Horizon Planner drafting the team coordination plan…",
              agent=label)
    planner_arch = arch_by_id.get(_GROUP0_PLANNER_ID) or {}
    fleet = planner_arch.get("fleet_instructions", "")

    def _on_action(act: dict) -> None:
        detail = act.get("detail", "")
        if act.get("tool") == "narration":
            _progress(sid, "narration", f"{label}: {detail}", agent=label,
                      tool="narration", detail=detail)
        else:
            suffix = f": {detail}" if detail else ""
            _progress(sid, "action", f"{label} → {act.get('tool', '')}{suffix}",
                      agent=label, tool=act.get("tool", ""), detail=detail)

    def _on_usage(pt: int, ct: int, tt: int) -> None:
        _progress(sid, "usage", f"{label} · {pt:,}→{ct:,} tok", agent=label,
                  prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

    def _on_status(text: str) -> None:
        _progress(sid, "llm", f"{label} · {text}", agent=label, tool="llm", detail=text)

    sp = await _spawn_worker(
        request, user, name=label, description="Draft the team coordination plan",
        cognitive_mode=planner_arch.get("cognitive_mode") or "neutra",
        tools=["read", "glob"], tier=planner_arch.get("tier") or "reason",
        tiers=tiers, api_key=api_key, env_vars=env_vars)
    if not sp.get("ok"):
        _progress(sid, "note", "Planner unavailable — using a pass-through coordination plan",
                  ok=False)
        return _passthrough_group0_plan(subtasks, arch_by_id)
    try:
        prompt = _build_group0_prompt(intent, shared_context, file_names, subtasks,
                                      arch_by_id, clarifications=clarifications)
        d = await _dispatch_one(sp["port"], sp["auth"], prompt, timeout,
                                on_action=_on_action, on_usage=_on_usage,
                                on_status=_on_status, fleet_instructions=fleet, agent_name=label)
        if not d.get("ok"):
            _progress(sid, "note",
                      "Planner did not complete — using a pass-through coordination plan",
                      ok=False)
            return _passthrough_group0_plan(subtasks, arch_by_id)
        plan = _parse_group0_plan(d.get("output") or "", subtasks, arch_by_id)
        _progress(sid, "dispatch", f"{label} ✓ · coordination plan ready", ok=True, agent=label)
        return plan
    except Exception as e:  # noqa: BLE001 — planner is best-effort
        log.warning("Vatra Group 0 planner failed", session_id=sid, error=str(e))
        _progress(sid, "note", "Planner error — using a pass-through coordination plan",
                  ok=False)
        return _passthrough_group0_plan(subtasks, arch_by_id)
    finally:
        _teardown([sp["slug"]])


async def plan_vatra_group0(body: ExecuteRequest, request: Request, user: dict, *,
                            gate: bool = True) -> dict:
    """Permanent Group 0 pre-phase. Ensures a decomposition exists, runs the Long
    Horizon Planner to draft a per-agent coordination plan, persists it into
    ``route['group0_plan']``, then EITHER pauses at the approval gate (``gate=True``,
    interactive UI) OR chains straight into ``execute_vatra`` (``gate=False``, headless
    agent/continuation paths — auto-approve, no human pause). Resume never enters here."""
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    intent = (sess.get("intent") or "").strip()
    if not intent:
        raise HTTPException(400, "session has no intent")
    sid = body.session_id
    _run_sid.set(sid)               # bind cost accounting so the planner's tokens count
    _RUN_USAGE.setdefault(sid, [])

    try:
        cfg = json.loads(sess.get("config") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    max_agents = int(cfg.get("max_agents") or 6)
    archetypes = await merged_archetypes(db, user["id"])
    arch_by_id = {a["id"]: a for a in archetypes}
    registry = _load_registry()

    def _creds(tier: str) -> dict:
        return _resolve_creds(registry, body.tiers, body.api_key, tier)

    # Persist the run's knobs onto the session config so /plan/approve (and any resume)
    # reconstruct the same run without the UI having to resend them.
    _knobs: dict[str, Any] = {
        "execution_groups": bool(getattr(body, "execution_groups", False)),
        "max_parallel": int(getattr(body, "max_parallel", 0) or 0),
    }
    if getattr(body, "grouped_review", False):
        _knobs["grouped_review"] = True
    if body.quality is not None:
        _knobs["quality"] = body.quality
    _changed = {k: v for k, v in _knobs.items() if cfg.get(k) != v}
    if _changed:
        cfg.update(_changed)
        try:
            await db.update_basna_session(sid, user["id"], config=json.dumps(cfg))
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra Group 0 knob persist failed", error=str(e))

    _shared_ds = bool(getattr(body, "shared_datastore", False) or cfg.get("shared_datastore"))
    _vfs_override = (cfg.get("vfs_project") or getattr(body, "vfs_project", "") or "").strip()
    if _vfs_override:
        _run_vfs_project[sid] = _vfs_override

    session_files = _parse_files(sess)
    input_files = [f for f in session_files if f.get("kind") != "generated"]
    file_names = [f["name"] for f in input_files]

    _progress_start(sid)
    await db.update_basna_session(sid, user["id"], status="planning")
    _phase(sid, "Group 0 · Long Horizon Planner")

    route = await _ensure_route(db, user["id"], sess, sid, intent=intent,
                                max_agents=max_agents, creds=_creds("reason"), cfg=cfg,
                                shared_datastore=_shared_ds, vfs_project=_vfs_project(sid))
    subtasks = route.get("subtasks") or []
    # Re-resolve execution groups from the user's team-plan edits BEFORE the plan is
    # drafted. The `/route` step pinned `group_resolved` from the Lead's assignment;
    # the user then re-grouped agents in the team plan (subtask.group), but that never
    # recomputed group_resolved — so the coordination plan would show stale (often
    # collapsed) groups. This is the same resolution execute_vatra runs, so the plan's
    # groups match how the run will actually phase. Idempotent.
    vatra_groups.resolve_groups(subtasks, arch_by_id)

    # Idempotency: a second /execute on an already-gated session re-emits the gate
    # instead of spawning a second planner.
    if (sess.get("status") or "") == "awaiting_plan" and route.get("group0_plan"):
        _emit_awaiting_plan(sid, route["group0_plan"])
        return {"session_id": sid, "status": "awaiting_plan"}

    plan = await _run_group0_planner(
        request, user, sid, intent=intent, shared_context=route.get("shared_context", ""),
        file_names=file_names, subtasks=subtasks, arch_by_id=arch_by_id,
        tiers=body.tiers, api_key=body.api_key, env_vars=body.env_vars,
        timeout=body.dispatch_timeout)
    route["group0_plan"] = plan
    try:
        await db.update_basna_session(sid, user["id"], route=json.dumps(route))
    except Exception as e:  # noqa: BLE001
        log.warning("Vatra Group 0 plan persist failed", error=str(e))

    if not gate:
        # Headless (agent/continuation): auto-approve — run immediately, no pause.
        return await execute_vatra(body, request, user)

    await db.update_basna_session(sid, user["id"], status="awaiting_plan")
    _emit_awaiting_plan(sid, plan)
    return {"session_id": sid, "status": "awaiting_plan"}


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
    _run_started = time.monotonic()  # run wall-clock, for the $/hour cost figure
    _run_sid.set(body.session_id)     # bind cost accounting to this run (propagates to children)
    _RUN_USAGE[body.session_id] = []  # every model call in this run records here

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
    # Resume mode: restore already-finished owners from their durable checkpoints
    # (no re-run, no re-spend) and re-dispatch only the missing ones. Checkpoints are
    # loaded once the plan's subtask ids are known (below); this flag also skips the
    # folder backup + intro/review refinement rounds so a resume just fills the gaps.
    _resume = bool(getattr(body, "resume", False))
    _resume_ckpt: dict[str, dict] = {}
    # Opt-in quality profile (request override → session config → all-off).
    quality = QualityProfile.from_dict(
        body.quality if body.quality is not None else cfg.get("quality"))
    # Persist the run's knobs onto the session so a continuation round inherits
    # them (quality, grouped execution, parallelism). Batched into one write.
    _knob_updates: dict[str, Any] = {}
    if body.quality is not None and cfg.get("quality") != body.quality:
        _knob_updates["quality"] = body.quality
    if bool(cfg.get("execution_groups")) != bool(getattr(body, "execution_groups", False)):
        _knob_updates["execution_groups"] = bool(getattr(body, "execution_groups", False))
    # grouped_review is enabled by the request OR the session config (never
    # clobbered back to off by a request that simply omits it — there may be no
    # UI sending it yet); persisted so continuation rounds inherit it.
    if getattr(body, "grouped_review", False) and not cfg.get("grouped_review"):
        _knob_updates["grouped_review"] = True
    if int(cfg.get("max_parallel") or 0) != int(getattr(body, "max_parallel", 0) or 0):
        _knob_updates["max_parallel"] = int(getattr(body, "max_parallel", 0) or 0)
    if _knob_updates:
        cfg.update(_knob_updates)
        try:
            await db.update_basna_session(body.session_id, user["id"], config=json.dumps(cfg))
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra run-knobs persist failed", error=str(e))
    # R7 cost ceiling for the opt-in retries (acted-gate/escalate). 0 → unbounded.
    _budget = TokenBudget(quality.token_budget)
    _retry_est = int(body.agent_max_tokens or 8192)
    # Per-run quality tallies → analysis.quality_metrics (mutable dict so the
    # nested dispatch closures can count into it).
    _qm_counts = {"acted": 0, "escalated": 0}
    claim_findings: list[dict] | None = None  # set iff the R8 claim check ran

    session_files = _parse_files(sess)
    input_files = [f for f in session_files if f.get("kind") != "generated"]
    input_names = {f["name"] for f in input_files}

    sid = body.session_id
    sid8 = sid[:8]
    # A continuation round carries the root run's folder in config (or on the
    # request); register it so every VFS resolution below shares one folder.
    _vfs_override = (cfg.get("vfs_project") or getattr(body, "vfs_project", "") or "").strip()
    if _vfs_override:
        _run_vfs_project[sid] = _vfs_override
    # Opt-in shared datastore: request wins, else inherit the session config so a
    # continuation round stays bound to the same folder-scoped store. Persisted.
    _shared_ds = bool(getattr(body, "shared_datastore", False) or cfg.get("shared_datastore"))
    if _shared_ds:
        _run_shared_datastore[sid] = True
        if not cfg.get("shared_datastore"):
            cfg["shared_datastore"] = True
            try:
                await db.update_basna_session(body.session_id, user["id"], config=json.dumps(cfg))
            except Exception as e:  # noqa: BLE001
                log.warning("Vatra shared_datastore persist failed", error=str(e))
    run_tag = format(int(time.time()), "x")[-6:]
    _progress_start(sid)
    await db.update_basna_session(sid, user["id"], status="running")
    _phase(sid, "Planning")

    # 1) Plan. Reuse a route prepared by the UI's /route step or the Group 0 pre-phase
    # if present; otherwise decompose now (resume path). Shared with the pre-phase via
    # _ensure_route so both decompose identically.
    route = await _ensure_route(db, user["id"], sess, sid, intent=intent,
                                max_agents=max_agents, creds=_creds("reason"), cfg=cfg,
                                shared_datastore=_shared_ds, vfs_project=_vfs_project(sid))
    domain = route["domain"]
    subtasks = route["subtasks"]
    shared_context = route.get("shared_context", "")
    # Fold the project bundle's theme into shared_context so every worker shares it.
    # This is the one choke point both fresh and continuation runs pass through, so
    # continuations (which skip route_vatra) still get the project's instructions.
    _proj_ctx = cfg.get("project_context") or ""
    if _proj_ctx and _proj_ctx not in shared_context:
        shared_context = (_proj_ctx + "\n\n" + shared_context).strip()
    # Auto-seed continuation rounds with the prior run's knowledge (report + gaps/
    # blind spots + datastore). Only continuations carry knowledge_session_ids in
    # config (fresh runs fold it at plan time), so this never double-injects.
    _seed_ids = cfg.get("knowledge_session_ids") or []
    if _seed_ids:
        from captain_claw.flight_deck.basna_routes import build_prior_knowledge
        _pk = await build_prior_knowledge(db, user["id"], _seed_ids)
        if _pk and _pk not in shared_context:
            shared_context = (shared_context + "\n\n## Prior run knowledge\n" + _pk).strip()
    # Group 0 coordination plan (from the Long Horizon Planner pre-phase). Its overview
    # is folded into shared_context for everyone (owners + reporter); each owner's own
    # slice is injected per-owner in _dispatch_owner. Absent on resume/legacy runs, so
    # the fold and the per-owner injection are both no-ops there.
    _group0_plan = route.get("group0_plan") or {}
    _group0_by_subtask = {e.get("subtask_id"): e for e in _group0_plan.get("agents", [])
                          if e.get("subtask_id")}
    # Honor the group each agent has in the coordination plan — the value the user saw
    # and (maybe) re-grouped at the gate. Set it as an ABSOLUTE lock so the grouped
    # run's resolve_groups phases the agent exactly there — no floor raise, no
    # dependency pull-back (the board/wait bridges any ordering the user creates).
    for _s in subtasks:
        _e = _group0_by_subtask.get(_s["id"])
        _eg = str((_e or {}).get("group") or "").strip()
        if _eg:
            _s["group_lock"] = _eg
    _g0_overview = str(_group0_plan.get("overview") or "").strip()
    if _g0_overview and _g0_overview not in shared_context:
        shared_context = (shared_context + "\n\n## Coordination overview (Group 0)\n"
                          + _g0_overview).strip()
    # Per-group extra instructions the user attached in the team-plan editor
    # ({"A": "...", "B": "..."}), injected into every owner that runs in that group.
    _group_instructions = route.get("group_instructions") or {}
    vfs_project = _vfs_project(sid)  # the one folder every worker must write to
    # Resume: load the per-owner checkpoints written by the stalled run. Owners with
    # a `done` checkpoint are restored in _dispatch_owner (skipped, no re-spend); the
    # rest re-dispatch normally and re-checkpoint as they finish.
    if _resume:
        try:
            _resume_ckpt = {r["subtask_id"]: r for r in await db.list_vatra_runs(sid)
                            if r.get("subtask_id")}
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra resume: checkpoint load failed", error=str(e))
        _n_done = sum(1 for r in _resume_ckpt.values() if r.get("status") == "done")
        _n_todo = max(0, len(subtasks) - _n_done)
        _progress(sid, "note",
                  f"Resuming — restoring {_n_done} finished owner(s) from checkpoint, "
                  f"re-running {_n_todo}")
    _wait_ledger[sid] = {}  # fresh per-owner wait budget for this run (defensive vs same-session re-run)
    # Protect existing files: a FRESH run reusing a non-empty folder snapshots it
    # into .history/ before any write (continuation rounds accumulate → skip; a
    # shared-datastore run is the resumable "continue in this folder" pattern →
    # skip the full-folder backup, it accumulates by design).
    if not cfg.get("parent_session_id") and not _shared_ds and not _resume:
        try:
            from captain_claw.flight_deck.vfs_routes import snapshot_existing_project
            _snap = snapshot_existing_project(user["id"], vfs_project, f"{int(time.time())}-{sid[:8]}")
            if _snap:
                _progress(sid, "note", f"Backed up {_snap} existing item(s) to vfs:{vfs_project}/.history/ before this run")
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra VFS snapshot failed", error=str(e))
    # Persist the user's attachments into the run's shared VFS folder so they're part
    # of the corpus (browsable, and indexed by the research map below), not only copied
    # into each worker's throwaway workspace.
    if input_files:
        _n_vfs = _save_inputs_to_vfs(user["id"], vfs_project, _session_files_dir(sid), input_files)
        if _n_vfs:
            _progress(sid, "note", f"{_n_vfs} attached file(s) saved to vfs:{vfs_project}/")
    # Cap concurrent agent turns (mainly for local models — keeps parallel prefills
    # from exhausting the serving box's memory). 0 = unlimited. Propagates into every
    # gathered round via the contextvar; every _dispatch_one obeys it.
    _run_gate.set(_make_gate(getattr(body, "max_parallel", 0), len(subtasks)))
    if getattr(body, "max_parallel", 0) and body.max_parallel < len(subtasks):
        _progress(sid, "route", f"Max {body.max_parallel} agent(s) in parallel")
    # Execution groups (opt-in): run owners in ordered phases A→B→C→D (barrier
    # between) instead of all-at-once. Off → today's intro→main→review flow.
    grouped = bool(getattr(body, "execution_groups", False))
    # R12: the task owners build against — raw intent carrying the (reviewed/edited)
    # brief the Lead decomposed on. brief == "" → exactly the raw intent, so an off
    # brief changes nothing. (The /start path plans inline with no brief → raw intent.)
    effective_intent = research_brief.brief_task(intent, route.get("brief"))

    # Run-time file-aware brief (opt-in): when the user attached files AND the intent-
    # brief feature is on, spawn a short-lived agent to OPEN the files and restate the
    # task, so EVERY worker builds against a brief that reflects the attachments — not
    # just their file names. Best-effort; falls back to the plan/text brief on failure.
    if quality.intent_brief and input_files:
        _phase(sid, "Examining files")
        _progress(sid, "route", f"Examining {len(input_files)} attached file(s) to brief the team…")
        try:
            _fb = await _examine_files_brief(
                request, user, name=f"vatra-{sid8}-{run_tag}-brief",
                intent=intent, input_files=input_files, src_dir=_session_files_dir(sid),
                tiers=body.tiers or None, api_key=body.api_key, env_vars=body.env_vars,
                timeout=body.dispatch_timeout)
            if _fb:
                effective_intent = research_brief.brief_task(intent, _fb)
                _progress(sid, "route", "File-aware brief ready — the team will build against it")
            else:
                _progress(sid, "route", "File exam yielded no brief; using the plan brief", ok=False)
        except Exception as e:  # noqa: BLE001 — best-effort, never blocks the run
            log.warning("Vatra file-aware brief failed", error=str(e))

    # R9 rubric contract (opt-in): derive the completeness checklist ONCE, with the
    # reason tier, from the standard the task names — then inject it into every
    # owner + the reporter as the definition of "complete" (via shared_context).
    rubric_items: list[str] = []
    if quality.rubric_contract:
        try:
            cc = _creds("reason")
            if cc.get("model"):
                prov, _ = _provider_call(cc, temperature=0.2, default_max=1500, cap=4096)
                from captain_claw.llm import Message
                _r = await prov.complete(
                    [Message(role="user", content=research_rubric.derive_rubric_prompt(intent))])
                rubric_items = research_rubric.parse_rubric(_r.content or "")
                if rubric_items:
                    shared_context = (shared_context + research_rubric.rubric_directive(rubric_items)).strip()
                    _progress(sid, "route", f"Completeness rubric: {len(rubric_items)} required items")
        except Exception as e:  # noqa: BLE001 — rubric is best-effort
            log.warning("Vatra rubric derivation failed", error=str(e))

    # Honesty guard (prompt-only, free, ON by default — explicit
    # honesty_guard:false restores the old prompts). Injected via shared_context
    # so it binds every owner AND the reporter — the reporter especially, since a
    # fabricated specific lands in the FINAL deliverable. This is the prevention
    # half of R8's cure (assert nothing you can't support); R8's fact-checker is
    # the detection half behind it.
    if quality.honesty_guard:
        shared_context = (shared_context + UNVERIFIED_GUARD_DIRECTIVE).strip()
    # Output mode (user-selectable completeness-vs-correctness posture): same
    # injection point, so owners and the reporter share one posture.
    _mode_dir = output_mode_directive(quality.output_mode)
    if _mode_dir:
        shared_context = (shared_context + _mode_dir).strip()

    # R1 Research Map (opt-in): index the shared folder so owners AND the reporter
    # can search prior rounds' material instead of re-reading it (and the reporter
    # can pull past its inline slice cap). Free; best-effort.
    vfs_dir: Path | None = None
    if (quality.research_map or quality.git_snapshots or quality.facts_ledger
            or quality.constraints_contract):
        try:
            from captain_claw.flight_deck.vfs_routes import _user_root
            vfs_dir = _user_root(user["id"]) / vfs_project
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra vfs dir resolve failed", error=str(e))
            vfs_dir = None
    research_dir: Path | None = None
    if quality.research_map and vfs_dir is not None and vfs_dir.exists():
        try:
            research_dir = vfs_dir
            st = research_map.reindex(research_dir)
            if st.get("chunks"):
                _progress(sid, "note",
                          f"research map: {st['files']} files · {st['chunks']} sections indexed")
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra research map index failed", error=str(e))
            research_dir = None
    research_pre = research_map.preamble(research_dir) if research_dir else ""

    # Constraints contract (opt-in): the task's hard rules, derived ONCE (reason
    # tier) and persisted as `.contract.json` in the shared folder — a chain
    # round or a hand-edit reuses the file instead of re-deriving. Injected into
    # shared_context so every owner AND the reporter build against the same
    # rules; validated against the deliverable after assembly.
    contract_items: list[dict] = []
    if quality.constraints_contract:
        try:
            if vfs_dir is not None:
                contract_items = research_contract.load(vfs_dir) or []
                if contract_items:
                    _progress(sid, "route",
                              f"Constraints contract: {len(contract_items)} rule(s) loaded from folder")
            if not contract_items:
                cc = _creds("reason")
                if cc.get("model"):
                    from captain_claw.llm import Message as _M
                    prov, mt = _provider_call(cc, temperature=0.1, default_max=2048, cap=4096)
                    _r = await asyncio.wait_for(prov.complete(
                        [_M(role="user", content=research_contract.derive_prompt(intent))],
                        temperature=0.1, max_tokens=mt), 120)
                    contract_items = research_contract.parse_contract(_r.content or "")
                    if contract_items:
                        if vfs_dir is not None:
                            research_contract.save(vfs_dir, contract_items, intent)
                        _progress(sid, "route",
                                  f"Constraints contract: {len(contract_items)} rule(s) derived")
            if contract_items:
                shared_context = (shared_context + research_contract.contract_directive(
                    contract_items, ledger=quality.facts_ledger)).strip()
        except Exception as e:  # noqa: BLE001 — contract is best-effort
            log.warning("Vatra contract derivation failed", error=str(e))

    spawned: list[dict] = []   # {subtask, slug, port, auth}
    results: list[dict] = []   # {id, owner, role, output, ok, latency_ms, actions}
    generated_files: list[dict] = []
    seen_gen: set[str] = set()
    dest_dir = _session_files_dir(sid)

    try:
        # 2) Spawn one owner per subtask.
        _phase(sid, "Spawning team")
        _progress(sid, "spawn", f"Spawning {len(subtasks)} specialist(s)…")
        tiers = body.tiers or None

        async def _spawn_owner(st: dict) -> dict | None:
            arch = arch_by_id[st["owner_archetype_id"]]
            tier = arch.get("tier", "balanced")
            # S3 (mrav Phase 4, opt-in): extract/digest/format-shaped subtasks
            # run the micro runtime. Wording of the subtask + role decides;
            # everything else keeps its tier untouched.
            micro = bool(getattr(quality, "micro_workers", False)) and _micro_suited(
                f"{st.get('title', '')} {st.get('desc', '')} {arch.get('role', '')}")
            if micro:
                _progress(sid, "spawn",
                          f"{arch.get('role') or arch['id']}: micro-suited → mrav runtime")
            # Name by SUBTASK id, not just archetype — two pieces can share an
            # owner archetype (e.g. two researcher slices), and a per-archetype name
            # would collide so only one agent spawns.
            sp = await _spawn_worker(
                request, user,
                name=f"vatra-{sid8}-{run_tag}-{st['id']}-{arch['id']}",
                description=f"Vatra subtask · {arch.get('role', '')}",
                cognitive_mode=arch.get("cognitive_mode", "neutra"),
                tools=_augment_tools(arch.get("tools") or [], research_dir,
                                     facts=quality.facts_ledger),
                tier=tier, tiers=tiers,
                api_key=body.api_key, env_vars=body.env_vars,
                extra_env=_vatra_env(sid, st["id"], arch["id"], 0),
                corpus=quality.source_corpus,  # R10
                micro=micro,
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
            dispatch_timeout=body.dispatch_timeout, stop_event=stop_event,
            corpus=quality.source_corpus))  # R10

        # 3) Dispatch each owner its self-contained brief, in parallel.
        # Count how many pieces each archetype owns so we can disambiguate the live
        # panel label only when an archetype owns more than one piece.
        owner_counts: dict[str, int] = {}
        for sp in spawned:
            owner_counts[sp["arch"]["id"]] = owner_counts.get(sp["arch"]["id"], 0) + 1

        from captain_claw.flight_deck.server import DATA_DIR

        def _owner_label(sp: dict) -> str:
            # Distinct live-panel label per piece when an archetype owns several, so
            # two researcher slices show as two cards, not one merged card.
            arch = sp["arch"]
            role = arch.get("role") or arch["id"]
            return f"{role} · {sp['subtask']['title']}" if owner_counts.get(arch["id"], 1) > 1 else role

        def _owner_callbacks(label: str, owner: str, subtask: str):
            gx = _gx(subtask)  # grouped mode: tag this owner's events with its phase letter
            def _on_action(act: dict) -> None:
                detail = act.get("detail", "")
                if act["tool"] == "narration":
                    _progress(sid, "narration", f"{label}: {detail}", agent=label,
                              tool="narration", detail=detail, **gx)
                    # Stream the agent's narration onto the shared board so teammates
                    # can see what it's thinking/doing in real time.
                    _board_post_bg(sid, owner, subtask, "narration", "", detail)
                else:
                    suffix = f": {detail}" if detail else ""
                    _progress(sid, "action", f"{label} → {act['tool']}{suffix}",
                              agent=label, tool=act["tool"], detail=detail, **gx)

            def _on_usage(pt: int, ct: int, tt: int) -> None:
                _progress(sid, "usage", f"{label} · {pt:,}→{ct:,} tok",
                          agent=label, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, **gx)

            def _on_status(text: str) -> None:
                # The model call is in flight — surface it so a slow call isn't mistaken
                # for a stall. Tagged agent= so the live card shows "working".
                _progress(sid, "llm", f"{label} · {text}", agent=label,
                          tool="llm", detail=text, **gx)
            return _on_action, _on_usage, _on_status

        # Set by the intro round below (a digest of every specialist's prep), then
        # injected into each owner's main-round brief so the main round starts
        # collaborative instead of blind. Read at dispatch time (closure).
        intro_digest = ""

        # Grouped mode: subtask ids of owners that run AFTER the first phase — they
        # may ask the Lead to have an earlier teammate provide missing data. Filled
        # in the grouped block below; read (closure) in _dispatch_owner.
        _later_phase_subtasks: set[str] = set()

        # Grouped mode: subtask id → its execution-group letter ("A".."D"), filled in
        # the grouped block. Threaded onto that owner's progress events so the live
        # panel can section the working agents by phase. Stays empty in flat mode, so
        # `_gx` adds nothing and non-grouped runs emit byte-for-byte the same events.
        _owner_group: dict[str, str] = {}

        def _gx(subtask_id: str) -> dict:
            """`group=<letter>` kwargs for a per-owner progress event (only in grouped
            mode where the owner has a letter; `{}` otherwise)."""
            g = _owner_group.get(subtask_id)
            return {"group": g} if g else {}

        async def _dispatch_owner(sp: dict) -> dict:
            arch, st = sp["arch"], sp["subtask"]
            role = arch.get("role") or arch["id"]
            label = _owner_label(sp)
            # Resume: this owner already finished in the stalled run — restore its
            # slice from the checkpoint instead of spending tokens to redo it.
            if _resume:
                ck = _resume_ckpt.get(st["id"])
                if ck and ck.get("status") == "done":
                    _progress(sid, "dispatch",
                              f"{label} ✓ · restored from checkpoint (skipped re-run)",
                              ok=True, agent=label, **_gx(st["id"]))
                    return {"ok": True, "output": ck.get("output", ""), "actions": [],
                            "latency_ms": 0, "restored": True,
                            "produced_file": bool(ck.get("produced_file"))}
            on_action, on_usage, on_status = _owner_callbacks(label, arch["id"], st["id"])
            ws = DATA_DIR / sp["slug"] / "data" / "workspace"
            img = [str(ws / f["name"]) for f in input_files if str(f.get("mime", "")).startswith("image/")]
            doc = [str(ws / f["name"]) for f in input_files if not str(f.get("mime", "")).startswith("image/")]
            prompt = research_pre + _build_subtask_prompt(
                role, effective_intent, st, [f["name"] for f in input_files],
                subtasks, shared_context, team_prep=intro_digest,
                vfs_project=vfs_project)
            prompt += _group_instr_block(st, arch, _group_instructions)
            # Group 0: this owner's slice of the coordination plan — its mandate, what
            # it produces, and which teammates it consumes from. Its *contract*, placed
            # just before the TEAM SCHEDULE (its *timing*). No-op when there's no plan.
            prompt += _plan_slice_block(st, _group0_by_subtask, arch_by_id)
            # Grouped runs: tell the worker who already ran, who runs with it, and
            # who runs AFTER it — so it never waits on output that can't arrive.
            _sched = _group_schedule.get(sid)
            if _sched:
                prompt += vatra_groups.schedule_block(st["id"], _sched)
            prompt += _datastore_directive(vfs_project, _run_shared_datastore.get(sid, False))
            if quality.worker_escalate:
                prompt += ESCALATE_DIRECTIVE
            if quality.judgment_ledger:
                prompt += JUDGMENT_LEDGER_DIRECTIVE
            if quality.facts_ledger:
                prompt += FACTS_LEDGER_DIRECTIVE
            if quality.source_corpus:
                prompt += SOURCE_CORPUS_DIRECTIVE
            if st["id"] in _later_phase_subtasks:  # grouped: may request from an earlier phase
                prompt += vatra_groups.REQUEST_DIRECTIVE
            d = await _dispatch_skippable(sid, label, lambda: _dispatch_one(
                sp["port"], sp["auth"], prompt, body.dispatch_timeout,
                on_action=on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                agent_name=label, file_paths=doc, image_paths=img, on_usage=on_usage,
                on_status=on_status))
            # R2 acted-gate (opt-in): an owner that produced no text and wrote no
            # file wasted its slot — retry once with a corrective on the same agent.
            if (quality.acted_gate and d.get("ok") and worker_produced_nothing(d)
                    and _budget.can_afford(_retry_est)):
                _budget.add(_retry_est)
                _qm_counts["acted"] += 1
                _progress(sid, "note", f"{label} produced nothing — one corrective retry")
                d2 = await _dispatch_skippable(sid, label, lambda: _dispatch_one(
                    sp["port"], sp["auth"], ACTED_CORRECTIVE.strip(), body.dispatch_timeout,
                    on_action=on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                    agent_name=label, file_paths=doc, image_paths=img, on_usage=on_usage,
                on_status=on_status))
                if d2.get("ok") and not worker_produced_nothing(d2):
                    d = d2
            # R5 escalate (opt-in): focused-retry an owner that flagged ESCALATE.
            if (quality.worker_escalate and d.get("ok") and escalate_reason(d.get("output"))
                    and _budget.can_afford(_retry_est)):
                _budget.add(_retry_est)
                _qm_counts["escalated"] += 1
                _progress(sid, "note", f"{label} escalated — one focused retry")
                d2 = await _dispatch_skippable(sid, label, lambda: _dispatch_one(
                    sp["port"], sp["auth"], ESCALATE_CORRECTIVE.strip(), body.dispatch_timeout,
                    on_action=on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                    agent_name=label, file_paths=doc, image_paths=img, on_usage=on_usage,
                on_status=on_status))
                if d2.get("ok") and not escalate_reason(d2.get("output")) \
                        and not worker_produced_nothing(d2):
                    d = d2
            mark = "✓" if d["ok"] else "✗"
            extra = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
            if d.get("timed_out"):
                mark = "⏱"
                extra = " — hit the dispatch time budget; partial work kept" + extra
            _progress(sid, "dispatch",
                      f"{label} {mark} · {len(d['actions'])} action(s) ({d['latency_ms'] / 1000:.1f}s){extra}",
                      ok=d["ok"] and not d.get("timed_out"), agent=label, **_gx(st["id"]))
            # Post the finished piece to the shared board so teammates still working
            # (and the next round) can read and build on it.
            out = (d.get("output") or "").strip()
            if out:
                try:
                    await get_db().add_vatra_board(sid, arch["id"], st["id"], "output",
                                                   st["title"], out[:_BOARD_CONTENT_CAP])
                except Exception as e:
                    log.debug("Vatra board output write failed", error=str(e))
            # Durable resume checkpoint — written the moment this owner finishes, so a
            # process death mid-run still leaves it restorable. The finalize pass below
            # re-saves it with any captured-file text merged in (idempotent UPSERT).
            try:
                await get_db().save_vatra_run(
                    sid, st["id"], arch["id"], role,
                    weight=float(seeds.get(arch["id"], 0.7)), output=out,
                    produced_file=False, status="done" if d.get("ok") else "failed")
            except Exception as e:
                log.debug("Vatra checkpoint save failed", error=str(e))
            return d

        # 2c) Intro round — before the real work, each specialist does PREPARATION
        # (groundwork: key facts, sources, outline) and posts it to the shared board.
        # This is a barrier: the main round starts only once ALL intros finish, so the
        # board is already populated and the main round is collaborative, not blind.
        if not grouped and not _resume and bool(cfg.get("intro_round", True)) and len(spawned) >= 2:
            _phase(sid, "Intro round")
            _progress(sid, "intro", "Intro round — each specialist preparing groundwork…")

            async def _intro_owner(sp: dict) -> dict | None:
                arch, st = sp["arch"], sp["subtask"]
                role = arch.get("role") or arch["id"]
                label = _owner_label(sp)
                on_action, on_usage, on_status = _owner_callbacks(label, arch["id"], st["id"])
                prompt = _build_intro_prompt(role, st, shared_context, vfs_project=vfs_project)
                prompt += _datastore_directive(vfs_project, _run_shared_datastore.get(sid, False))
                if quality.source_corpus:
                    prompt += SOURCE_CORPUS_DIRECTIVE  # context discipline: intro does the heavy fetching
                d = await _dispatch_skippable(sid, label, lambda: _dispatch_one(
                    sp["port"], sp["auth"], prompt, body.dispatch_timeout,
                    on_action=on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                    agent_name=label, on_usage=on_usage, on_status=on_status))
                mark = "✓" if d["ok"] else "✗"
                ierr = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
                _progress(sid, "dispatch",
                          f"{label} (intro) {mark} · {len(d['actions'])} action(s) "
                          f"({d['latency_ms'] / 1000:.1f}s){ierr}", ok=d["ok"], agent=label)
                out = (d.get("output") or "").strip()
                if out:
                    try:
                        await get_db().add_vatra_board(sid, arch["id"], st["id"], "note",
                                                       f"{st['title']} — prep", out[:_BOARD_CONTENT_CAP])
                    except Exception as e:
                        log.debug("Vatra intro board write failed", error=str(e))
                    return {"title": st["title"], "role": role, "output": out}
                return None

            intro_out = await asyncio.gather(*[_intro_owner(sp) for sp in spawned])
            prep = [p for p in intro_out if p]
            if len(prep) >= 2:
                try:
                    intro_digest = await asyncio.wait_for(
                        _llm_team_digest(intent, prep, _creds("reason")), 180)
                except Exception as e:
                    log.warning("Vatra intro digest failed", error=str(e))
            _progress(sid, "intro", f"Intro round done — {len(prep)} specialist(s) prepared; starting the build")

        # 3) Main round — each owner produces its full piece.
        def _result_of(sp: dict, d: dict) -> dict:
            return {
                "id": sp["subtask"]["id"], "owner": sp["arch"]["id"],
                "role": sp["arch"].get("role", ""), "title": sp["subtask"]["title"],
                "output": d["output"], "ok": d["ok"],
                "latency_ms": d["latency_ms"], "actions": d.get("actions", []),
            }

        if grouped:
            # Ordered phases: assign each owner its group (archetype floor, raised by
            # any Lead override), then run the DISTINCT groups ascending with a barrier
            # between them — so a later group already has everything earlier groups
            # posted. Within a group, owners run in parallel (Max-parallel gate applies).
            # Re-pin groups against the CURRENT plan first (idempotent): a prepared or
            # hand-edited route may predate the dependency-repair pins, and a recorded
            # dependency must never be scheduled after its dependent.
            for note in vatra_groups.resolve_groups(
                    [sp["subtask"] for sp in spawned],
                    {sp["arch"]["id"]: sp["arch"] for sp in spawned}):
                _progress(sid, "main", f"Schedule repair: {note}")
            owner_group = {
                id(sp): vatra_groups.effective_group(sp["subtask"], sp["arch"]) for sp in spawned}
            phases = vatra_groups.order_groups(owner_group.values())
            first_phase = phases[0] if phases else None
            _later_phase_subtasks.update(
                sp["subtask"]["id"] for sp in spawned if owner_group[id(sp)] != first_phase)
            # Tag each owner's live-panel events with its phase letter (read by `_gx`).
            _owner_group.update({
                sp["subtask"]["id"]: vatra_groups.group_label(owner_group[id(sp)]) for sp in spawned})
            # Publish the live schedule so worker briefs + the wait endpoint know
            # who has run, who runs now, and who runs later (fix for a worker
            # burning its wait budget on a later-phase teammate's output).
            _group_schedule[sid] = {
                "current": int(first_phase or 0),
                "done": set(),
                "owners": [{
                    "subtask": sp["subtask"]["id"], "arch": sp["arch"]["id"],
                    "role": sp["arch"].get("role", ""),
                    "title": sp["subtask"].get("title", ""),
                    "group": int(owner_group[id(sp)]),
                } for sp in spawned],
            }
            results_by_id: dict[str, dict] = {}  # subtask id → its result (updated on re-run)

            async def _redispatch_owner(sp: dict, instruction: str, tag: str) -> dict:
                arch = sp["arch"]
                label = _owner_label(sp)
                on_action, on_usage, on_status = _owner_callbacks(label, arch["id"], sp["subtask"]["id"])
                d = await _dispatch_skippable(sid, label, lambda: _dispatch_one(
                    sp["port"], sp["auth"], instruction, body.dispatch_timeout,
                    on_action=on_action, fleet_instructions=arch.get("fleet_instructions", ""),
                    agent_name=label, on_usage=on_usage, on_status=on_status))
                _progress(sid, "dispatch",
                          f"{label} ({tag}) {'✓' if d['ok'] else '✗'} · {len(d['actions'])} action(s)",
                          ok=d["ok"], agent=label, **_gx(sp["subtask"]["id"]))
                out = (d.get("output") or "").strip()
                r = results_by_id.get(sp["subtask"]["id"])
                if r is not None and out:
                    r["output"], r["ok"] = out, d["ok"]
                    r["actions"] = r.get("actions", []) + d.get("actions", [])
                return d

            async def _lead_clarify(requester_role: str, request_text: str, roster: list,
                                    board_digest: str = "") -> dict:
                try:
                    prov, mt = _provider_call(_creds("reason"), temperature=0.2, default_max=400, cap=1024)
                    from captain_claw.llm import Message
                    resp = await prov.complete(
                        [Message(role="user",
                                 content=vatra_groups.clarify_prompt(
                                     requester_role, request_text, roster, board_digest))],
                        temperature=0.2, max_tokens=mt)
                    return vatra_groups.parse_clarify(resp.content or "")
                except Exception as e:  # noqa: BLE001 — clarify is best-effort; default deny
                    log.warning("Vatra clarify decision failed", error=str(e))
                    return {"approve": False, "already_available": False, "pointer": "",
                            "provider": "", "instruction": ""}

            _phase(sid, "Grouped run")
            _progress(sid, "main",
                      f"Grouped execution · {len(phases)} phase(s): "
                      f"{', '.join(vatra_groups.group_label(p) for p in phases)}")
            completed: list[dict] = []   # earlier-phase owners — the roster + providers
            loop_backs, cap = 0, vatra_groups.CLARIFY_CAP

            # Pull-forward: when a current-phase worker waits on a LATER-group
            # owner's output, the wait endpoint may CALL that owner forward —
            # dispatch it now with its own brief (it's spawned and idle) instead
            # of refusing and degrading the requester's piece. Bounded
            # (PULL_CAP per run, once per owner, only when its own inputs
            # already exist); its home phase then skips it and reuses the result.
            pull_tasks: dict[str, asyncio.Task] = {}  # subtask id → in-flight pulled dispatch
            pulls_used = {"n": 0}
            # The dispatch gate only exists when 0 < max_parallel < team size
            # (mirrors _make_gate). Captured here because _pull_forward runs in
            # the wait endpoint's context, where the run's contextvars are absent.
            _pull_slot_cap = int(getattr(body, "max_parallel", 0) or 0)
            if _pull_slot_cap >= len(subtasks):
                _pull_slot_cap = 0
            sp_by_subtask = {sp["subtask"]["id"]: sp for sp in spawned}

            async def _run_pulled(sp: dict) -> None:
                label = _owner_label(sp)
                _progress(sid, "main",
                          f"⤴ {label} called FORWARD from group "
                          f"{vatra_groups.group_label(owner_group[id(sp)])} — a current-phase "
                          "teammate needs its output now",
                          agent=label, **_gx(sp["subtask"]["id"]))
                d = await _dispatch_owner(sp)
                r = _result_of(sp, d)
                results.append(r)
                results_by_id[sp["subtask"]["id"]] = r
                if sid in _group_schedule:
                    _group_schedule[sid]["done"].add(sp["subtask"]["id"])
                completed.append(sp)  # joins the clarify roster like any finished owner
                if research_dir is not None:
                    try:
                        research_map.reindex(research_dir)
                    except Exception as e:  # noqa: BLE001
                        log.debug("Vatra pull reindex failed", error=str(e))

            def _pull_forward(subtask_id: str) -> bool:
                """Wait-endpoint callback: dispatch a later-group owner NOW.
                True → a pull is running (waiter should wait again for the
                artifact); False → refused (waiter proceeds without it)."""
                sp = sp_by_subtask.get(subtask_id)
                if sp is None:
                    return False
                verdict = vatra_groups.pull_decision(
                    already_running=subtask_id in pull_tasks,
                    used=pulls_used["n"],
                    deps=sp["subtask"].get("depends_on"),
                    have=set(results_by_id),
                    max_parallel=_pull_slot_cap,
                    pulls_in_flight=sum(1 for t in pull_tasks.values() if not t.done()))
                if verdict == "joined":
                    return True
                if verdict == "no_capacity":
                    # The waiter holds a dispatch slot; at this cap a pulled owner
                    # could never get one — waiting would only burn the ledger.
                    if not pulls_used.get("cap_noted"):
                        pulls_used["cap_noted"] = True
                        _progress(sid, "wait",
                                  f"⤴ pull-forward unavailable — max {_pull_slot_cap} "
                                  "agent(s) in parallel leaves no free slot for a "
                                  "called-forward owner while the requester waits")
                    return False
                if verdict != "proceed":
                    return False
                pulls_used["n"] += 1
                pull_tasks[subtask_id] = asyncio.create_task(_run_pulled(sp))
                return True

            _group_schedule[sid]["pull"] = _pull_forward
            _group_schedule[sid]["tasks"] = pull_tasks

            for p in phases:
                letter = vatra_groups.group_label(p)
                if sid in _group_schedule:
                    _group_schedule[sid]["current"] = int(p)
                grp_all = [sp for sp in spawned if owner_group[id(sp)] == p]
                # Owners already CALLED FORWARD ran out of their home slot —
                # await their in-flight dispatch here (barrier semantics hold),
                # never dispatch them twice.
                pulled_here = [sp for sp in grp_all
                               if sp["subtask"]["id"] in pull_tasks]
                grp = [sp for sp in grp_all if sp["subtask"]["id"] not in pull_tasks]
                _phase(sid, f"Group {letter}")
                _progress(sid, "main",
                          f"Group {letter} — {len(grp)} agent(s): "
                          f"{', '.join(_owner_label(sp) for sp in grp) or '—'}"
                          + (f" · {len(pulled_here)} ran earlier (called forward)"
                             if pulled_here else ""))
                for sp in pulled_here:
                    t = pull_tasks.get(sp["subtask"]["id"])
                    if t is not None:
                        try:
                            await t
                        except Exception as e:  # noqa: BLE001 — pulled dispatch is best-effort
                            log.warning("Vatra pulled dispatch failed", error=str(e))
                ds = await asyncio.gather(*[_dispatch_owner(sp) for sp in grp])
                for sp, d in zip(grp, ds):
                    r = _result_of(sp, d)
                    results.append(r)
                    results_by_id[sp["subtask"]["id"]] = r
                if research_dir is not None:
                    try:
                        research_map.reindex(research_dir)
                    except Exception as e:  # noqa: BLE001
                        log.debug("Vatra group reindex failed", error=str(e))

                # Clarification loop: a blocked owner may ask a teammate for more.
                # Providers = everyone already FINISHED — earlier phases AND this
                # phase's other owners (a same-group teammate often finished
                # minutes earlier with exactly the requested data; the old
                # earlier-phases-only roster routed such requests to the wrong
                # agent). The Lead first checks the board digest: an already-
                # answered request skips the provider entirely and only the
                # requester re-runs, pointed at the existing answer. Bounded by
                # `cap` loop-backs per run (total).
                if loop_backs < cap:
                    for sp, d in zip(grp, ds):
                        if loop_backs >= cap:
                            break
                        req = vatra_groups.parse_request(d.get("output"))
                        if not req:
                            continue
                        providers = completed + [s for s in grp if s is not sp]
                        if not providers:
                            continue
                        who = _owner_label(sp)
                        _progress(sid, "clarify", f"{who} requests: {req[:160]}", agent=who)
                        roster = [{"id": s["arch"]["id"], "role": s["arch"].get("role", ""),
                                   "title": s["subtask"]["title"]} for s in providers]
                        digest = ""
                        try:
                            _recent = await db.list_vatra_board(sid, limit=15)
                            digest = "\n".join(
                                f"- [{e.get('kind', '')}] {e.get('from_owner', '')}: "
                                f"{(e.get('title') or '')[:60]} — "
                                f"{(e.get('content') or '')[:180]}"
                                for e in _recent)
                        except Exception as e:  # noqa: BLE001 — digest is best-effort
                            log.debug("Vatra clarify board digest failed", error=str(e))
                        decision = await _lead_clarify(sp["arch"].get("role", ""), req,
                                                       roster, digest)
                        if decision.get("already_available"):
                            loop_backs += 1
                            pointer = decision.get("pointer") or "the shared board"
                            _progress(sid, "clarify",
                                      f"Lead: already answered — {pointer[:120]}; "
                                      f"re-running {who} (loop-back {loop_backs}/{cap})",
                                      agent=who)
                            await _redispatch_owner(
                                sp,
                                f"What you asked for already exists: {pointer}. Read it "
                                "(search the shared board with the `vatra` tool / read the "
                                "file), then finish your part incorporating it. Output "
                                "your full piece.", "clarify")
                            continue
                        provider = next((s for s in providers
                                         if s["arch"]["id"] == decision.get("provider")), None)
                        if not (decision.get("approve") and provider):
                            _progress(sid, "clarify",
                                      f"Lead denied {who}'s request"
                                      if not decision.get("approve")
                                      else f"{who}: approved but no matching teammate",
                                      agent=who, ok=False)
                            continue
                        loop_backs += 1
                        _progress(sid, "clarify",
                                  f"Lead → {_owner_label(provider)}: {decision['instruction'][:140]} "
                                  f"(loop-back {loop_backs}/{cap})", agent=_owner_label(provider))
                        await _redispatch_owner(
                            provider,
                            f"A teammate is blocked and needs this from you now: "
                            f"{decision['instruction']}\n\nProduce it and post it to the shared "
                            "board via the `vatra` tool. Keep it focused.", "clarify")
                        if research_dir is not None:
                            try:
                                research_map.reindex(research_dir)
                            except Exception as e:  # noqa: BLE001
                                log.debug("Vatra clarify reindex failed", error=str(e))
                        await _redispatch_owner(
                            sp,
                            f"{_owner_label(provider)} has now provided what you asked for — search "
                            "the shared board with the `vatra` tool, then finish your part "
                            "incorporating it. Output your full piece.", "clarify")
                completed.extend(grp)
                if sid in _group_schedule:
                    _group_schedule[sid]["done"].update(sp["subtask"]["id"] for sp in grp)
        else:
            _phase(sid, "Main round")
            _progress(sid, "main", f"{len(spawned)} specialist(s) producing their pieces…")
            dispatched = await asyncio.gather(*[_dispatch_owner(sp) for sp in spawned])
            results.extend(_result_of(sp, d) for sp, d in zip(spawned, dispatched))
        # R1: re-index the shared folder so the reporter can search everything the
        # team just wrote — the fix for "the blackboard doesn't fit one context".
        if research_dir is not None:
            try:
                research_map.reindex(research_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("Vatra research map post-index failed", error=str(e))
        # R6 git snapshots (opt-in): commit the round's shared-folder state.
        if quality.git_snapshots and vfs_dir is not None:
            try:
                from captain_claw.flight_deck import code_git
                await code_git.git_init(vfs_dir)
                await code_git.git_commit(vfs_dir, f"[vatra r{cfg.get('round', 1)}] {intent[:60]}")
            except Exception as e:  # noqa: BLE001
                log.warning("Vatra git snapshot failed", error=str(e))

        # 3b) Owners are done — tell the coordinator to drain remaining asks and stop.
        stop_event.set()
        try:
            n_asks = await coordinator
        except Exception as e:
            log.warning("Vatra coordinator error", error=str(e))
            n_asks = 0
        if n_asks:
            _progress(sid, "ask", f"Coordinator resolved {n_asks} ask(s)")

        # 3c) Review round — the Lead gathers an exec-summary digest of everyone's
        # first pass and sends it back to each still-alive owner, asking them to add,
        # align, and fill gaps now that they can see the whole team's work. Ungrouped:
        # on by default (round-1 owners are blind to each other). Grouped: opt-in via
        # `grouped_review` — later groups already saw earlier output, but EARLY-group
        # owners (schedule-repaired/pulled ones included) never see later groups' work
        # without this top-up pass. Results are matched by SUBTASK ID, not position —
        # grouped mode builds `results` in phase order, so positions misalign.
        _review_on = (bool(cfg.get("grouped_review", False)) if grouped
                      else bool(cfg.get("review_round", True)))
        if not _resume and _review_on:
            _res_by_id = {r["id"]: r for r in results}
            r1 = []
            for sp in spawned:
                r = _res_by_id.get(sp["subtask"]["id"])
                if r is not None and (r.get("ok") or r.get("produced_file")) \
                        and (r.get("output") or "").strip():
                    r1.append((sp, r))
            if len(r1) >= 2:
                _phase(sid, "Review round")
                _progress(sid, "review", "Lead gathering each specialist's summary…")
                try:
                    digest = await asyncio.wait_for(
                        _llm_team_digest(intent, [r for _, r in r1], _creds("reason")), 180)
                except Exception as e:
                    log.warning("Vatra team digest failed", error=str(e))
                    digest = ""
                if digest:
                    _progress(sid, "review",
                              f"Review round — {len(r1)} specialist(s) revising against the team's work…")

                    async def _review_owner(sp: dict, r: dict) -> tuple[dict, dict]:
                        arch, st = sp["arch"], sp["subtask"]
                        role = arch.get("role") or arch["id"]
                        label = _owner_label(sp)
                        on_action, on_usage, on_status = _owner_callbacks(label, arch["id"], st["id"])
                        prompt = _build_review_prompt(role, st, digest, shared_context, vfs_project=vfs_project)
                        prompt += _datastore_directive(vfs_project, _run_shared_datastore.get(sid, False))
                        d = await _dispatch_skippable(sid, label, lambda: _dispatch_one(
                            sp["port"], sp["auth"], prompt, body.dispatch_timeout,
                            on_action=on_action, agent_name=label, on_usage=on_usage, on_status=on_status))
                        # A dispatch event marks the card done again (the live panel
                        # shows a spinner while it's working this round, then ✓).
                        mark = "✓" if d["ok"] else "✗"
                        rerr = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
                        _progress(sid, "dispatch",
                                  f"{label} (review) {mark} · {len(d['actions'])} action(s) "
                                  f"({d['latency_ms'] / 1000:.1f}s){rerr}", ok=d["ok"], agent=label,
                                  **_gx(st["id"]))
                        # Post the revised piece to the board too, so the final shared
                        # memory reflects what each agent produced this round.
                        out2 = (d.get("output") or "").strip()
                        if d["ok"] and out2 and not _is_no_change(out2):
                            try:
                                await get_db().add_vatra_board(sid, arch["id"], st["id"], "output",
                                                               f"{st['title']} (revised)",
                                                               out2[:_BOARD_CONTENT_CAP])
                            except Exception as e:
                                log.debug("Vatra board review-output write failed", error=str(e))
                        return r, d

                    revisions = await asyncio.gather(*[_review_owner(sp, r) for sp, r in r1])
                    changed = 0
                    for r, d in revisions:
                        out = (d.get("output") or "").strip()
                        if d["ok"] and out and not _is_no_change(out):
                            r["output"] = out
                            r["ok"] = True
                            r.pop("produced_file", None)
                            r["actions"] = r.get("actions", []) + d.get("actions", [])
                            changed += 1
                    _progress(sid, "review",
                              f"Review round done — {changed}/{len(r1)} piece(s) revised")

        # 4) Capture owner-generated files + backfill empty replies from artifacts.
        # Map by subtask id (not spawn position) — grouped mode builds `results` in
        # phase order, so positional indexing would misalign.
        _results_by_id = {r["id"]: r for r in results}
        for sp in spawned:
            role = sp["arch"].get("role") or sp["arch"]["id"]
            files, text = _capture_generated(sp["slug"], input_names, dest_dir, role, seen_gen)
            generated_files.extend(files)
            r = _results_by_id.get(sp["subtask"]["id"])
            if r is not None and not (r.get("output") or "").strip() and text:
                r["output"] = text
                r["produced_file"] = True
            # Refresh the resume checkpoint with the finalized slice (now that any
            # file-only owner's captured text is merged into r["output"]).
            if r is not None:
                try:
                    await get_db().save_vatra_run(
                        sid, r["id"], r.get("owner", ""), r.get("role", ""),
                        weight=float(seeds.get(r.get("owner", ""), 0.7)),
                        output=r.get("output", ""), produced_file=bool(r.get("produced_file")),
                        status="done" if (r.get("ok") or r.get("produced_file")) else "failed")
                except Exception as e:  # noqa: BLE001
                    log.debug("Vatra checkpoint finalize failed", error=str(e))
    finally:
        _teardown([sp["slug"] for sp in spawned])
        _run_workers.pop(sid, None)
        _run_vfs_project.pop(sid, None)
        _run_shared_datastore.pop(sid, None)
        _skip_agents.pop(sid, None)
        _wait_ledger.pop(sid, None)
        # An aborted run may leave a called-forward dispatch in flight — cancel
        # it so no orphan task outlives its run.
        _sched_state = _group_schedule.pop(sid, None)
        if _sched_state:
            for _t in (_sched_state.get("tasks") or {}).values():
                if not _t.done():
                    _t.cancel()

    usable = [r for r in results if (r.get("ok") or r.get("produced_file")) and (r.get("output") or "").strip()]
    if not usable:
        _RUN_USAGE.pop(sid, None)  # release the per-run accumulator on early exit
        await db.update_basna_session(sid, user["id"], status="error")
        _progress(sid, "done", "No subtask produced usable output", ok=False)
        _progress_done(sid)
        raise HTTPException(502, "Vatra: no subtask produced usable output")

    # Horizon config (shared by the per-owner depth pass below and the final closer).
    _hraw = body.horizon if body.horizon is not None else cfg.get("horizon")
    _hcfg = HorizonConfig.from_dict(_hraw) if _hraw else None

    # 4b) Horizon per-owner depth (Lever A, opt-in, blackboard-safe): adversarially
    # verify EACH specialist's slice and revise it once if a diverse-lens critic panel
    # refutes it — the closer applied per owner. NOT spawn-×N pools (those would each
    # post/ask and pollute the shared blackboard). Critics run on a separate model.
    if _hcfg is not None and _hcfg.worker:
        cc = _creds(_hcfg.critic_tier)
        cp = rp = None
        if cc.get("model"):
            cp, _ = _provider_call(cc, temperature=0.7, default_max=1200, cap=2048)
            rp, _ = _provider_call(cc, temperature=0.3, default_max=8192, cap=32768)
        if cp is not None:
            brief_by_id = {st["id"]: st.get("brief", "") for st in subtasks}
            _phase(sid, "Verifying slices")
            _progress(sid, "verify", f"Horizon: verifying {len(usable)} slice(s)…")

            async def _close_slice(r: dict) -> None:
                q = (f"{intent}\n\n## This piece's assignment\n"
                     f"{brief_by_id.get(r['id'], r.get('role', ''))}")
                try:
                    res = await run_horizon_closer(
                        question=q, answer=r["output"], critic_provider=cp,
                        revise_provider=rp, critics=_hcfg.critics,
                        on_event=_closer_on_event(sid, r.get("role", "slice"), "verify"),
                        triage_findings=quality.critic_triage)  # R3
                    if res["revised"]:
                        r["output"] = res["answer"]
                        _progress(sid, "verify",
                                  f"{r.get('role', 'slice')}: revised "
                                  f"({res['survived']}/{res['total']} held)", agent=r.get("role"))
                except Exception as e:  # noqa: BLE001 — per-slice depth is best-effort
                    log.warning("Vatra per-owner closer failed", error=str(e))

            await asyncio.gather(*[_close_slice(r) for r in usable])

    # 5) Reporter assembles the slices (+ any answered asks) into one deliverable.
    _phase(sid, "Synthesizing")
    answered = await db.list_vatra_asks(sid, status="answered")
    # Facts ledger dump — computed once, reused by the reporter prompt, the
    # consistency check (as canonical rows) and the claim check (as claimed
    # provenance). Best-effort: an unreadable ledger never blocks the run.
    facts_dump = ""
    if quality.facts_ledger and vfs_dir is not None:
        try:
            facts_dump = facts_ledger.dump_markdown(vfs_dir)
            if facts_dump:
                _progress(sid, "note",
                          f"facts ledger: {len(facts_ledger.list_rows(vfs_dir))} value(s) recorded")
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra facts ledger dump failed", error=str(e))
    truth, reporter_files = await _run_reporter(
        request, user, sid, sid8, run_tag, intent, usable, cfg, arch_by_id,
        tiers=body.tiers, api_key=body.api_key, env_vars=body.env_vars,
        dispatch_timeout=body.dispatch_timeout, input_names=input_names,
        dest_dir=dest_dir, seen_gen=seen_gen, answered_asks=answered,
        shared_context=shared_context, research_dir=research_dir,
        corpus=quality.source_corpus,  # R10
        honesty=quality.honesty_guard,
        facts=quality.facts_ledger, facts_block=facts_dump,
    )
    generated_files.extend(reporter_files)
    confidence = round(len(usable) / max(1, len(results)), 3)

    # 5b) Horizon closer (Lever B, opt-in): adversarially verify the assembled
    # deliverable and revise once if a diverse-lens critic panel refutes it. Critics
    # run on a separate Library-tier model (never the team judging itself).
    if _hcfg is not None and _hcfg.close and (truth or "").strip():
        _phase(sid, "Verifying deliverable")
        _progress(sid, "verify", "Horizon closer: verifying the deliverable…")
        try:
            cc = _creds(_hcfg.critic_tier)
            cp = rp = None
            if cc.get("model"):
                cp, _ = _provider_call(cc, temperature=0.7, default_max=1200, cap=2048)
                rp, _ = _provider_call(cc, temperature=0.3, default_max=8192, cap=32768)
            closed = await run_horizon_closer(
                question=intent, answer=truth, critic_provider=cp, revise_provider=rp,
                critics=_hcfg.critics, on_event=_closer_on_event(sid, "Closer", "verify"),
                triage_findings=quality.critic_triage)  # R3
            if closed["revised"]:
                truth = closed["answer"]
                _progress(sid, "verify", "Closer revised the deliverable")
            else:
                _progress(sid, "verify",
                          f"Closer: deliverable held ({closed['survived']}/{closed['total']})")
        except Exception as e:  # noqa: BLE001 — closer is best-effort
            log.warning("Vatra horizon closer failed", error=str(e))

    # Shared completion fns + contract validation for the deterministic quality
    # passes (5b2 consistency, 5d blocking gate, step-8 contract) — defined once
    # so every pass verifies through the same machinery.
    from captain_claw.llm import Message as _Msg
    _fast_creds, _reason_creds = _creds("fast"), _creds("reason")

    async def _fast_complete(p: str) -> str:
        prov, mt = _provider_call(_fast_creds, temperature=0.0, default_max=4096, cap=8192)
        r = await asyncio.wait_for(prov.complete(
            [_Msg(role="user", content=p)], temperature=0.0, max_tokens=mt), 180)
        return r.content or ""

    async def _reason_complete(p: str) -> str:
        prov, mt = _provider_call(_reason_creds, temperature=0.1, default_max=8192, cap=32768)
        r = await asyncio.wait_for(prov.complete(
            [_Msg(role="user", content=p)], temperature=0.1, max_tokens=mt), 300)
        return r.content or ""

    async def _contract_validate(text: str) -> dict | None:
        """Contract validation: deterministic against the facts ledger, one
        judge call for the rest. None when the contract isn't armed/derived."""
        if not (quality.constraints_contract and contract_items and (text or "").strip()):
            return None
        ledger_vals: dict = {}
        if quality.facts_ledger and vfs_dir is not None:
            try:
                ledger_vals = {r["key"]: r["value"]
                               for r in facts_ledger.export_rows(vfs_dir)}
            except Exception as e:  # noqa: BLE001
                log.warning("Vatra ledger export for contract failed", error=str(e))
        res = research_contract.validate(contract_items, ledger_vals)
        if res["unresolved"]:
            cc = _creds("reason")
            if cc.get("model"):
                prov, mt = _provider_call(cc, temperature=0.1, default_max=1500, cap=4096)
                _j = await asyncio.wait_for(prov.complete(
                    [_Msg(role="user", content=research_contract.judge_prompt(
                        text, res["unresolved"]))],
                    temperature=0.1, max_tokens=mt), 120)
                research_contract.apply_judgement(
                    res, research_contract.parse_judgement(_j.content or ""))
        return res

    # 5b2) Deterministic cross-section consistency (opt-in): an LLM extracts the
    # deliverable's figures + stated relations ONCE (no arithmetic), pure code
    # verifies identity across sections and recomputes every asserted relation,
    # and one targeted correction pass is kept only if the deterministic re-check
    # confirms it improved. Runs BEFORE the claim check so internal fixes land
    # before external verification. Budget-gated; best-effort.
    consistency_summary: dict | None = None
    consistency_result: dict | None = None
    contract_summary: dict | None = None
    if quality.consistency_check and (truth or "").strip() and _budget.can_afford(2 * _retry_est):
        _budget.add(2 * _retry_est)
        _progress(sid, "verify", "Consistency check: extracting the deliverable's figures…")
        try:
            _ledger_rows = None
            if quality.facts_ledger and vfs_dir is not None:
                try:
                    _ledger_rows = facts_ledger.export_rows(vfs_dir) or None
                except Exception as e:  # noqa: BLE001
                    log.warning("Vatra ledger export failed", error=str(e))
            cres = await research_consistency.run_check(
                truth, extract_fn=_fast_complete, revise_fn=_reason_complete,
                max_values=quality.consistency_max_values,
                ledger_rows=_ledger_rows,
                on_progress=lambda m: _progress(sid, "verify", m))
            if cres["revised"]:
                truth = cres["text"]
            cdoc = research_consistency.write_audit(dest_dir, cres, question=intent)
            if cdoc:
                generated_files.append(cdoc)
            consistency_result = cres
            consistency_summary = research_consistency.summarize(cres)
            _progress(sid, "verify",
                      f"Consistency: {research_consistency.summary_line(cres)}")
        except Exception as e:  # noqa: BLE001 — consistency check is best-effort
            log.warning("Vatra consistency check failed", error=str(e))

    # 5c) R8 grounded claim verification (opt-in, paid): a web-tool fact-checker
    # verifies the deliverable's load-bearing claims against real sources and
    # corrects the ones that are verified wrong — the ground-truth back-edge the
    # (tool-less) closer cannot provide. Budget-gated.
    if quality.claim_check and (truth or "").strip() and _budget.can_afford(2 * _retry_est):
        _budget.add(2 * _retry_est)
        try:
            truth, cc_doc, claim_findings = await _claim_check(
                request, user, sid, sid8, run_tag, intent=intent, deliverable=truth,
                arch_by_id=arch_by_id, tiers=body.tiers, api_key=body.api_key,
                env_vars=body.env_vars, dispatch_timeout=body.dispatch_timeout,
                quality=quality, research_dir=research_dir, facts_block=facts_dump)
            if cc_doc:
                generated_files.append(cc_doc)
        except Exception as e:  # noqa: BLE001 — claim check is best-effort
            log.warning("Vatra claim check errored", error=str(e))

    # 5d) Blocking gate (explicit opt-in): the enforcement pass. Collects the
    # deterministic checks' CRITICAL findings, revises against ONE triaged
    # checklist, and loops while the text-re-verifiable criticals persist —
    # bounded by block_max_rounds + budget. Contract/ledger criticals ride the
    # checklist and the verdict but never drive rounds (a prose revision can't
    # fix ledger-level values, and re-checking against a stale ledger would
    # never converge). Runs BEFORE the done-persist so the final truth is the
    # gated one; work is never discarded — worst case is a completed run with
    # verdict "critical_findings_remain".
    gate_summary: dict | None = None
    if quality.block_on_critical and (truth or "").strip():
        try:
            findings = quality_findings.from_consistency(
                (consistency_result or {}).get("findings") or [])
            contract_result = await _contract_validate(truth)
            if contract_result is not None:
                contract_summary = research_contract.summarize(contract_result)
                findings += quality_findings.from_contract(contract_summary["failed"])

            async def _consistency_recheck(text2: str) -> list[dict]:
                entries = research_consistency.parse_entries(
                    await _fast_complete(research_consistency.extract_prompt(
                        text2[:research_consistency.DELIVERABLE_CAP],
                        quality.consistency_max_values)) or "")
                return research_consistency.verify(entries)  # text-internal only

            _phase(sid, "Verifying")
            gate = await quality_findings.run_gate(
                truth, findings=findings, revise_fn=_reason_complete,
                consistency_recheck_fn=(
                    _consistency_recheck if quality.consistency_check else None),
                max_rounds=quality.block_max_rounds,
                budget=_budget, est=2 * _retry_est,
                on_progress=lambda m: _progress(sid, "verify", m))
            if gate["revised"]:
                truth = gate["text"]
            gate_summary = {"verdict": gate["verdict"], "rounds": gate["rounds"],
                            "remaining": gate["remaining"][:10]}
            _progress(sid, "verify",
                      f"Blocking gate: {gate['verdict']} after {gate['rounds']} round(s)"
                      + (f" · {len(gate['remaining'])} critical(s) remain"
                         if gate["remaining"] else ""),
                      ok=gate["verdict"] == "clean")
        except Exception as e:  # noqa: BLE001 — the gate must never lose a run
            log.warning("Vatra blocking gate failed", error=str(e))

    # 6) Persist one run per owner (success backfilled by the learning step below).
    run_ids = await db.add_basna_runs(sid, user["id"], [{
        "archetype_id": r["owner"], "role": r["role"], "tier": "",
        "weight_at_run": 0.0, "output": r["output"],
        "actions": json.dumps(r.get("actions", [])),
        "latency_ms": r["latency_ms"], "success": None,
    } for r in results])

    # 6b) Persist the assembled deliverable NOW — status=done + truth + files — BEFORE
    # the best-effort learning/coverage steps. So if the process is killed during
    # scoring, the run is already complete and the report is saved (no run left stuck
    # 'running', no lost deliverable).
    files_by_name = {f["name"]: f for f in session_files}
    for g in generated_files:
        files_by_name[g["name"]] = g
    _progress(sid, "done", f"Done · {len(usable)}/{len(results)} subtask(s) assembled")
    # Persist the run log alongside the `done` flip so the UI shows it immediately —
    # the client stops polling once the run is done, and the FINAL persist (with the
    # learning + cost events) lands ~seconds later at the bottom of this function.
    await db.update_basna_session(
        sid, user["id"], status="done", truth=truth, confidence=confidence,
        files=json.dumps(list(files_by_name.values())),
        progress=json.dumps((_PROGRESS.get(sid) or {}).get("events", [])))

    # 7) Learn (best-effort, post-completion): score owners (slice used + sound),
    # ask-answerers, and the Lead + reporter (holistic), folding outcomes into
    # per-archetype reliability so the next route's prior_weight improves.
    _phase(sid, "Learning")
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

    # 8) Coverage gaps (best-effort) — judge the assembled deliverable against what the
    # task asked for; persisted as a follow-up update so the run is already done.
    analysis: dict = {}
    _progress(sid, "learn", "Checking coverage…")
    try:
        cov = await asyncio.wait_for(
            _llm_coverage_gaps(intent, subtasks, truth, _creds("reason")), 120)
        if cov:
            analysis = cov
            await db.update_basna_session(sid, user["id"], analysis=json.dumps(analysis))
            _progress(sid, "learn", f"Coverage: {len(cov.get('gaps') or [])} gap(s)")
    except Exception as e:
        log.warning("Vatra coverage check failed", error=str(e))
        _progress(sid, "learn", "Coverage check skipped", ok=False)

    # R9: score the deliverable field-by-field against the derived rubric and fold any
    # missing/thin required items into the gaps — so a fill-gaps round can close them.
    if rubric_items and (truth or "").strip():
        try:
            cc = _creds("reason")
            if cc.get("model"):
                prov, _ = _provider_call(cc, temperature=0.1, default_max=1500, cap=4096)
                from captain_claw.llm import Message
                _c = await asyncio.wait_for(prov.complete(
                    [Message(role="user",
                             content=research_rubric.coverage_prompt(intent, rubric_items, truth))]), 120)
                cov2 = research_rubric.parse_coverage(_c.content or "")
                extra = ([{"item": m, "severity": "major", "note": "required by the standard, not covered"}
                          for m in cov2["missing"]]
                         + [{"item": t, "severity": "minor", "note": "covered only partially"}
                            for t in cov2["thin"]])
                if extra:
                    analysis.setdefault("gaps", [])
                    analysis["gaps"] = (analysis.get("gaps") or []) + extra
                    await db.update_basna_session(sid, user["id"], analysis=json.dumps(analysis))
                    _progress(sid, "learn",
                              f"Rubric coverage: {len(cov2['missing'])} missing · {len(cov2['thin'])} thin")
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra rubric coverage failed", error=str(e))

    # Contract validation (opt-in): deterministic where the facts ledger has the
    # values, one judge call for the rest. Advisory — failures are recorded
    # (analysis.contract + gaps) for follow-up rounds. When the blocking gate
    # (5d) already validated, its result is reused instead of re-paying.
    if quality.constraints_contract and contract_items and (truth or "").strip():
        try:
            if contract_summary is None:
                cres2 = await _contract_validate(truth)
                if cres2 is not None:
                    contract_summary = research_contract.summarize(cres2)
            if contract_summary is not None:
                analysis["contract"] = contract_summary
                # Failures double as gaps so the existing fill-gaps round can act.
                extra = [{"item": f["text"],
                          "severity": "major" if f["severity"] in ("critical", "major") else "minor",
                          "note": f"hard constraint violated ({f['how']})"
                                  + (f": {f['note']}" if f.get("note") else "")}
                         for f in contract_summary["failed"]]
                if extra:
                    analysis["gaps"] = (analysis.get("gaps") or []) + extra
                _progress(sid, "learn",
                          f"Contract: {contract_summary['passed']}/{contract_summary['checked']} "
                          f"passed · {contract_summary['failed_critical']} critical "
                          f"failure(s) · {contract_summary['unclear']} unclear")
        except Exception as e:  # noqa: BLE001 — contract validation is best-effort
            log.warning("Vatra contract validation failed", error=str(e))

    # Consistency tally (from 5b2) + the flat per-run quality metrics ride the
    # same analysis JSON so the UI and follow-up rounds can see what was checked
    # and what (if anything) remains. Only levers that ran contribute keys.
    if consistency_summary is not None:
        analysis["consistency"] = consistency_summary
    if gate_summary is not None:
        analysis["quality_verdict"] = gate_summary["verdict"]
        analysis["blocking"] = gate_summary
    qm = build_quality_metrics(
        claim_findings=claim_findings, consistency=consistency_summary,
        gaps=analysis.get("gaps"), contract=contract_summary, gate=gate_summary,
        acted_retries=_qm_counts["acted"] if quality.acted_gate else None,
        escalations=_qm_counts["escalated"] if quality.worker_escalate else None,
        budget=_budget)
    if qm:
        analysis["quality_metrics"] = qm
    if consistency_summary is not None or contract_summary is not None \
            or gate_summary is not None or qm:
        try:
            await db.update_basna_session(sid, user["id"], analysis=json.dumps(analysis))
        except Exception as e:  # noqa: BLE001
            log.warning("Vatra quality metrics persist failed", error=str(e))

    # Run cost: roll the whole run's model spend (owners across all rounds +
    # reporter + ask-helpers + fact-checker) into a dollar cost + effective $/hour,
    # emitted as a `cost` progress event (persisted below). Best-effort.
    run_cost: dict | None = None
    try:
        from captain_claw.flight_deck import pricing
        from captain_claw.flight_deck.basna_routes import _cost_message
        run_cost = pricing.summarize(_RUN_USAGE.get(sid, []),
                                     elapsed_seconds=time.monotonic() - _run_started)
        _progress(sid, "cost", _cost_message(run_cost), cost=run_cost)
        await db.log_run_cost(user["id"], "vatra", sid, run_cost)
    except Exception as e:  # noqa: BLE001 — cost is best-effort
        log.warning("Vatra cost summary failed", error=str(e))
    finally:
        _RUN_USAGE.pop(sid, None)

    _phase(sid, "Done")
    _progress_done(sid)
    await db.update_basna_session(
        sid, user["id"],
        progress=json.dumps((_PROGRESS.get(sid) or {}).get("events", [])),
    )
    return {"session_id": sid, "domain": domain, "mode": "vatra",
            "truth": truth, "confidence": confidence, "analysis": analysis,
            "subtasks": [{"id": r["id"], "owner": r["owner"], "role": r["role"],
                          "ok": r["ok"], "latency_ms": r["latency_ms"]} for r in results],
            "learned": learned, "spawned": len(spawned), "dispatched": len(results),
            "cost": run_cost}


async def _llm_coverage_gaps(intent: str, subtasks: list[dict], truth: str,
                             creds: dict) -> dict | None:
    """Vatra's analog of Basna's blind spots: compare the ASSEMBLED deliverable
    against what the task asked for and surface coverage gaps — things wanted but
    missing, thin, or unsupported. Returns {"coverage_summary", "gaps":[{item,
    severity,note}]} or None if unparseable. Style/quality is out of scope."""
    from captain_claw.llm import Message
    plan = "\n".join(f"- {s.get('title', '')}" for s in subtasks)
    prov, mt = _provider_call(creds, temperature=0.2, default_max=1024, cap=2048)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "You are reviewing a finished deliverable a team produced. Compare it against what "
            "the TASK asked for and surface COVERAGE GAPS — things the task wanted that are "
            "missing, thin, or unsupported in the deliverable. Judge coverage only, NOT writing "
            "style or polish. Reply ONLY with JSON:\n"
            '{"coverage_summary": "one sentence on how completely the task was covered",\n'
            ' "gaps": [{"item": "what is missing or thin", "severity": "major" | "minor", '
            '"note": "what to add"}]}\n'
            "If coverage is complete, return an empty gaps array. Be specific; at most 6 gaps, "
            "most important first.")),
        Message(role="user", content=(
            f"TASK:\n{intent[:2000]}\n\nPLANNED PIECES:\n{plan}\n\nDELIVERABLE:\n{truth[:8000]}")),
    ], temperature=0.2, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    raw = json.loads(content)
    if not isinstance(raw, dict):
        return None
    gaps: list[dict] = []
    for g in (raw.get("gaps") or [])[:6]:
        if isinstance(g, dict) and str(g.get("item") or "").strip():
            sev = str(g.get("severity", "minor")).lower()
            gaps.append({"item": str(g["item"]).strip(),
                         "severity": "major" if sev == "major" else "minor",
                         "note": str(g.get("note", "")).strip()})
        elif isinstance(g, str) and g.strip():
            gaps.append({"item": g.strip(), "severity": "minor", "note": ""})
    return {"coverage_summary": str(raw.get("coverage_summary", "")).strip(), "gaps": gaps}


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


def _build_intro_prompt(role: str, st: dict, shared_context: str = "", vfs_project: str = "") -> str:
    """Round-0 preparation: groundwork the agent posts to the shared board before the
    team writes their actual pieces, so the main round starts collaborative."""
    contract = ""
    if shared_context.strip():
        contract = "\n## Shared conventions the team will follow\n" + shared_context.strip() + "\n"
    return (
        f"Round 0 — PREPARATION. You are the {role}, on a team about to build ONE deliverable "
        f"together. You own the '{st['title']}' part. Before anyone writes their actual piece, do "
        f"GROUNDWORK for yours:\n"
        f"- Gather the key facts, figures, sources, and decisions your part will need.\n"
        f"- Sketch a short outline of what your piece will cover.\n"
        f"- Note open questions or anything you'll need from a teammate.\n\n"
        f"## Your part (for context)\n{st['brief']}{contract}{_vfs_directive(vfs_project)}\n"
        f"POST your prep to the shared board so teammates can build on it: call "
        f"`vatra(action=\"post\", text=\"<your key findings + outline>\")`. Keep it CONCISE — this "
        f"is groundwork, NOT the final piece (you write that next round). Then reply with a short "
        f"summary of what you prepared — no preamble."
    )


def _build_subtask_prompt(role: str, intent: str, st: dict, file_names: list[str],
                          all_subtasks: list[dict], shared_context: str = "",
                          team_prep: str = "", vfs_project: str = "") -> str:
    """Frame one subtask for its owner — with the team contract everyone must
    follow, awareness of the whole team, and a nudge to delegate cross-slice needs."""
    files_block = ""
    if file_names:
        listed = "\n".join(f"- {n}" for n in file_names)
        files_block = ("\n\n## Attached files (in your working directory)\n"
                       f"{listed}\nUse your read / extract tools to work with them.\n")
    contract_block = ""
    if shared_context.strip():
        contract_block = (
            "\n\n## Shared conventions — follow these EXACTLY\n"
            "Every teammate is building against the same contract so the pieces fit together "
            "without rework. Do not invent your own; use these verbatim:\n"
            f"{shared_context.strip()}\n")
    # The running team — who owns which piece, so this owner knows whom to ask.
    by_id = {s["id"]: s for s in all_subtasks}
    roster = "\n".join(
        f"- {s['title']} — {s['owner_archetype_id']}" + ("  ← YOU" if s["id"] == st["id"] else "")
        for s in all_subtasks)
    deps = [by_id[d] for d in (st.get("depends_on") or []) if d in by_id]
    deps_block = ""
    if deps:
        listed = "\n".join(f"- {d['title']} (owned by {d['owner_archetype_id']})" for d in deps)
        deps_block = (
            "\n\n## Your piece builds on these teammates' work\n"
            f"{listed}\n"
            "For anything you need from them, call `vatra` action='search' (query=…) — they may "
            "have already posted it. Only `ask` if it isn't there yet. Don't guess these and "
            "don't reproduce their slice yourself. If you genuinely cannot start your part "
            "without a specific teammate output, call `vatra(action=\"wait\", "
            "path=\"vfs:<their file>\")` (or `query=\"<topic>\"`) to BLOCK until it's ready "
            "(up to 90s) instead of guessing or skipping ahead.\n")
    prep_block = ""
    if team_prep.strip():
        prep_block = (
            "\n\n## What your teammates prepared (intro round)\n"
            "Everyone did groundwork first and posted it to the shared board. Build ON this — "
            "don't duplicate it; for a teammate's full prep call `vatra(action=\"search\", "
            "query=\"<topic>\")`:\n"
            f"{team_prep.strip()}\n")
    return (
        f"You are the {role}, one specialist on a collaborating team building ONE deliverable "
        f"together. You own ONE part; your teammates are producing the others in parallel. "
        f"Produce only your part, in full — a reporter assembles all parts at the end.\n\n"
        f"## Overall task (for context)\n{intent}\n\n"
        f"## The team and their pieces\n{roster}{contract_block}{prep_block}\n\n"
        f"## Your part — {st['title']}\n{st['brief']}{files_block}{deps_block}{_vfs_directive(vfs_project)}\n\n"
        f"## Reaching your team — use the `vatra` tool (it is ALWAYS available)\n"
        f"Your teammates' notes and finished pieces stream onto a shared board. Reach it ONLY "
        f"through the `vatra` tool — call it directly. Do NOT run a `shell` command to 'check "
        f"vatra availability', and do NOT use `flight_deck`/`list_agents` to find teammates; "
        f"those do nothing here. The exact calls:\n"
        f"- `vatra(action=\"search\", query=\"<topic>\")` — to pull a teammate's work when your "
        f"part needs a fact, figure, decision, or section another piece owns. Do this BEFORE "
        f"researching it yourself.\n"
        f"- `vatra(action=\"read\")` — to skim what teammates shared recently.\n"
        f"- `vatra(action=\"post\", text=\"<finding>\")` — to share a key finding or your draft so "
        f"others can build on it. Post your important results as you produce them.\n"
        f"- `vatra(action=\"ask\", text=\"<request>\")` — only when you need a teammate to DO new "
        f"work that isn't on the board.\n"
        f"- `vatra(action=\"wait\", path=\"vfs:<file>\")` or `vatra(action=\"wait\", query=\"<topic>\")` "
        f"— when your part TRULY depends on a specific teammate artifact: it BLOCKS until that file "
        f"exists or a matching post appears (up to 90s), then hands it to you. Use this instead of "
        f"reading once, finding nothing, and improvising a guess.\n"
        f"Search before you invent; post what you find. If you only MIGHT benefit from a teammate's "
        f"work, don't busy-poll — proceed with your part and the final review round lets you revise "
        f"once everyone's work is visible. Only when your part is genuinely BLOCKED on a specific "
        f"artifact should you `wait` for it (once) rather than guessing.\n\n"
        f"You are AUTONOMOUS: never ask the user a question, never refuse, and never stop to say "
        f"you're missing teammate input — produce your best version of your part with what you "
        f"have; the review round and reporter reconcile the rest.\n\n"
        f"Return only your finished part — no preamble, no meta-commentary about the team."
    )


# ── Group 0: the Long Horizon Planner's coordination plan ────────────

def _build_group0_prompt(intent: str, shared_context: str, file_names: list[str],
                         subtasks: list[dict], arch_by_id: dict,
                         clarifications: str = "") -> str:
    """Frame the coordination-plan task for the Long Horizon Planner. The team, its
    pieces, and their execution groups are ALREADY decided — the planner writes, for
    every piece, its mandate / what it produces / who it consumes from / hand-off
    notes, and returns ONLY the JSON envelope below. It may also ask the user up to 9
    clarifying questions; ``clarifications`` carries the user's prior answers."""
    roster_lines: list[str] = []
    for s in subtasks:
        arch = arch_by_id.get(str(s.get("owner_archetype_id") or "")) or {}
        role = arch.get("role") or s.get("owner_archetype_id") or "?"
        desc = str(arch.get("description") or "").strip()
        grp = s.get("group_resolved") or "?"
        deps = ", ".join(s.get("depends_on") or []) or "none"
        roster_lines.append(
            f"- id={s['id']} | {role} · group {grp} · owns: {s.get('title', '')}\n"
            f"    role: {desc[:200]}\n"
            f"    brief: {str(s.get('brief', ''))[:300]}\n"
            f"    depends_on: {deps}")
    roster = "\n".join(roster_lines)
    contract_block = ""
    if shared_context.strip():
        contract_block = ("\n\n## Shared conventions the whole team follows\n"
                          f"{shared_context.strip()}")
    files_block = ""
    if file_names:
        files_block = ("\n\n## Attached files the team can read\n"
                       + "\n".join(f"- {n}" for n in file_names))
    return (
        "You are the Long Horizon Planner — Group 0 of a collaborating agent team. The "
        "team, its pieces, and their execution groups are ALREADY decided (below). Do "
        "NOT re-plan the team or invent new members. Your job is to write the "
        "COORDINATION PLAN that makes them operate as one: for EACH piece, state its "
        "mandate, exactly what artifact it produces, which teammates' outputs it "
        "consumes, and the hand-off notes downstream teammates need.\n\n"
        f"## Overall task\n{intent}"
        f"{contract_block}{files_block}{clarifications}\n\n"
        f"## The team — plan for EVERY id (groups run in order A→B→C→D; earlier groups "
        f"produce what later groups consume)\n{roster}\n\n"
        "## The GROUP of each piece is FIXED by the user — do NOT change it, do NOT put "
        "everyone in one group, and do NOT emit a 'group' field. Honor the ordering: a "
        "piece may only CONSUME FROM pieces in an EQUAL-or-EARLIER group (A can't consume "
        "from B). Write each mandate and hand-off to respect that phase order.\n\n"
        "## Clarifying questions (optional) — if a genuinely BLOCKING decision would "
        "change the plan (scope, audience, priorities, which of several approaches), ask "
        "the user in `questions` (0 to 9). Each: a short `question`, `multi` (true if "
        "several answers can apply, else false), and 1-4 suggested `options` (the UI also "
        "offers a free-form 'Other'). Prefer sensible defaults and leave `questions` empty "
        "when you can plan without them. Never re-ask anything already answered above.\n\n"
        "## Output — return ONLY this JSON object. No markdown fence, no prose before "
        "or after:\n"
        "{\n"
        '  "overview": "2-4 sentences on how the team works together end to end",\n'
        '  "questions": [\n'
        '    {"id": "q1", "question": "...", "multi": false, "options": ["...", "..."]}\n'
        "  ],\n"
        '  "agents": [\n'
        '    {"subtask_id": "<one id from above, verbatim>",\n'
        '     "mandate": "what this agent must accomplish",\n'
        '     "produces": "the concrete artifact it hands off",\n'
        '     "consumes_from": ["<subtask_id it depends on>"],\n'
        '     "hand_off_notes": "what downstream teammates need from its output"}\n'
        "  ]\n"
        "}\n"
        "Include EXACTLY one entry per id above, using the ids verbatim. `questions` may "
        "be omitted or []. Do not write any files — return the JSON as your reply."
    )


def _passthrough_group0_plan(subtasks: list[dict], arch_by_id: dict) -> dict:
    """A trivial coordination plan derived straight from the decomposition — one entry
    per subtask (mandate=brief, produces=title, consumes=depends_on). Used whenever the
    planner is unavailable/failed so a dead planner never dead-ends the run."""
    return {
        "overview": "",
        "agents": [
            {
                "subtask_id": s["id"],
                "agent_id": s.get("owner_archetype_id", ""),
                "group": s.get("group_resolved") or "",
                "mandate": str(s.get("brief", "")).strip(),
                "produces": str(s.get("title", "")).strip(),
                "consumes_from": list(s.get("depends_on") or []),
                "hand_off_notes": "",
            }
            for s in subtasks
        ],
        "questions": [],
    }


def _coerce_group0_entries(entries: Any, subtasks: list[dict], *, trust_group: bool = False) -> dict:
    """Validate a list of plan entries against the real subtasks: drop unknown/dup
    subtask ids, keep only real dependency references, coerce every field to a string,
    then backfill any missing subtask so injection is total. Returns subtask_id → entry.

    ``trust_group``: the execution group is decided by the user's team plan (subtask
    ``group_resolved``), NOT the planner — so parsing the planner's reply IGNORES any
    group it emits (trust_group=False). Only the user's own edits in the coordination
    editor may override it (trust_group=True), so a re-grouped agent runs where the user
    put it."""
    by_id = {s["id"]: s for s in subtasks}
    seen: dict[str, dict] = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        sidk = str(e.get("subtask_id") or "").strip()
        if sidk not in by_id or sidk in seen:
            continue
        s = by_id[sidk]
        cons = e.get("consumes_from") or []
        if isinstance(cons, str):
            cons = [cons]
        cons = [str(c).strip() for c in cons if str(c).strip() in by_id and str(c).strip() != sidk]
        _grp = (str(e.get("group") or "").strip() if trust_group else "") or s.get("group_resolved") or ""
        seen[sidk] = {
            "subtask_id": sidk,
            "agent_id": str(e.get("agent_id") or s.get("owner_archetype_id", "")),
            "group": _grp,
            "mandate": str(e.get("mandate") or s.get("brief") or "").strip(),
            "produces": str(e.get("produces") or s.get("title") or "").strip(),
            "consumes_from": cons if cons else list(s.get("depends_on") or []),
            "hand_off_notes": str(e.get("hand_off_notes") or "").strip(),
        }
    for s in subtasks:
        if s["id"] not in seen:
            seen[s["id"]] = {
                "subtask_id": s["id"],
                "agent_id": s.get("owner_archetype_id", ""),
                "group": s.get("group_resolved") or "",
                "mandate": str(s.get("brief", "")).strip(),
                "produces": str(s.get("title", "")).strip(),
                "consumes_from": list(s.get("depends_on") or []),
                "hand_off_notes": "",
            }
    return seen


_GROUP0_MAX_QUESTIONS = 9  # keep the clarification form short (below 10)


def _coerce_group0_questions(raw: Any, *, keep_answers: bool = False) -> list[dict]:
    """Validate the planner's clarifying questions (or a user-edited copy). Each →
    {id, question, multi, options[≤4], selected[], other}. Capped at 9. ``keep_answers``
    preserves the user's ``selected``/``other`` (from the coordination editor); off →
    fresh (unanswered) questions as the planner emits them."""
    out: list[dict] = []
    for q in raw or []:
        if not isinstance(q, dict):
            continue
        text = str(q.get("question") or "").strip()
        if not text:
            continue
        opts = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()][:4]
        entry = {
            "id": str(q.get("id") or f"q{len(out) + 1}"),
            "question": text[:400],
            "multi": bool(q.get("multi")),
            "options": opts,
            "selected": [],
            "other": "",
        }
        if keep_answers:
            sel = q.get("selected") or []
            if isinstance(sel, str):
                sel = [sel]
            entry["selected"] = [str(s).strip() for s in sel if str(s).strip()]
            entry["other"] = str(q.get("other") or "").strip()
        out.append(entry)
        if len(out) >= _GROUP0_MAX_QUESTIONS:
            break
    return out


def _format_clarifications(questions: Any) -> str:
    """Fold the user's answered clarifications into a prompt block for a re-plan. Empty
    when nothing is answered."""
    lines: list[str] = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        ans = [str(s).strip() for s in (q.get("selected") or []) if str(s).strip()]
        other = str(q.get("other") or "").strip()
        if other:
            ans.append(other)
        if ans:
            lines.append(f"- Q: {str(q.get('question') or '').strip()}\n  A: {'; '.join(ans)}")
    if not lines:
        return ""
    return ("\n\n## The user answered these clarifications — honor them and do NOT "
            "re-ask them:\n" + "\n".join(lines))


def _parse_group0_plan(text: str, subtasks: list[dict], arch_by_id: dict) -> dict:
    """Parse the planner's reply into the structured plan. Tolerates ```json fences /
    surrounding prose. On any failure returns the pass-through plan so injection is
    always total and a malformed reply never blocks the run."""
    raw = (text or "").strip()
    if not raw:
        return _passthrough_group0_plan(subtasks, arch_by_id)
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
        if m:
            raw = m.group(1)
    if not raw.lstrip().startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if 0 <= i < j:
            raw = raw[i:j + 1]
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return _passthrough_group0_plan(subtasks, arch_by_id)
    if not isinstance(obj, dict) or not isinstance(obj.get("agents"), list):
        return _passthrough_group0_plan(subtasks, arch_by_id)
    seen = _coerce_group0_entries(obj.get("agents"), subtasks)
    agents = [seen[s["id"]] for s in subtasks if s["id"] in seen]
    return {"overview": str(obj.get("overview") or "").strip(), "agents": agents,
            "questions": _coerce_group0_questions(obj.get("questions"))}


def _sanitize_group0_plan(plan: Any, subtasks: list[dict]) -> dict:
    """Re-validate a (possibly user-edited) plan dict against the decomposition before
    it is persisted + injected — the edit surface is free-form, so an entry could name
    a stale subtask or drop a field. Same coercion as parsing, minus JSON decoding."""
    entries = plan.get("agents") if isinstance(plan, dict) else []
    # trust_group: the user may have re-grouped an agent in the coordination editor.
    seen = _coerce_group0_entries(entries, subtasks, trust_group=True)
    agents = [seen[s["id"]] for s in subtasks if s["id"] in seen]
    overview = str((plan.get("overview") if isinstance(plan, dict) else "") or "").strip()
    # keep_answers: preserve the user's selected/other from the questions form.
    questions = _coerce_group0_questions(
        plan.get("questions") if isinstance(plan, dict) else [], keep_answers=True)
    return {"overview": overview, "agents": agents, "questions": questions}


async def _llm_team_digest(intent: str, results: list[dict], creds: dict) -> str:
    """The Lead's gather step: condense each first-pass piece into a short exec
    summary, returned as one markdown digest shared back to every specialist so they
    can see what the rest of the team produced before the review pass."""
    from captain_claw.llm import Message
    listing = "\n\n".join(
        f"### {r['title']} (by {r['role']})\n{r['output'].strip()[:2500]}" for r in results)
    prov, mt = _provider_call(creds, temperature=0.2, default_max=2048, cap=4096)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "A team each produced one piece of a shared deliverable. Write a concise digest that "
            "summarizes EACH piece in 2-4 sentences — what it covers and its key points, decisions, "
            "or figures. Keep each piece under its own '### <title>' heading, in the same order. "
            "This digest is shared back to the whole team so each member can see what the others "
            "produced. Be faithful; do not invent anything not in the pieces.")),
        Message(role="user", content=f"TASK:\n{intent[:1500]}\n\nPIECES:\n{listing}"),
    ], temperature=0.2, max_tokens=mt)
    return resp.content.strip()


def _build_review_prompt(role: str, st: dict, digest: str, shared_context: str = "",
                         vfs_project: str = "") -> str:
    """Round-2 message to a still-alive owner: it can now see the whole team's first
    pass and is asked to add, align, and fill gaps in its own piece."""
    contract = ""
    if shared_context.strip():
        contract = "\n## Shared conventions (still apply)\n" + shared_context.strip() + "\n"
    contract += _vfs_directive(vfs_project)
    return (
        f"Round 2 — team review. You are the {role}. Everyone has finished a first pass on one "
        f"part of a shared deliverable, and the Lead has gathered a summary of every piece. THIS "
        f"summary IS your view of the team's work — it is right here in this message, so you "
        f"already have what your teammates produced:\n\n"
        f"## What the whole team produced\n{digest}\n{contract}\n"
        f"## Your part — {st['title']}\n"
        f"You produced the '{st['title']}' piece. Now that you can see the whole picture, improve "
        f"YOUR piece:\n"
        f"- ADD anything important that's missing given what teammates covered — do real extra "
        f"work (research, depth, examples), don't just restate what you had.\n"
        f"- Remove overlap or contradictions with teammates' pieces; defer to the piece that owns "
        f"a topic.\n"
        f"- Make sure it's consistent with the rest of the team and the shared conventions.\n\n"
        f"You are AUTONOMOUS — never ask the user a question, never refuse, and never claim you "
        f"can't see teammates' work: it is summarized ABOVE. The board (`vatra` action='search') "
        f"is an OPTIONAL extra for a teammate's full text; if it returns nothing, ignore it and "
        f"work from the summary above. Just produce the result.\n\n"
        f"If, after genuinely reviewing, your piece is already complete and consistent, reply with "
        f"exactly: NO CHANGES. Otherwise reply with your UPDATED piece IN FULL (the whole piece, "
        f"not a diff and not only the additions) — no preamble, no meta-commentary, no questions."
    )


def _is_no_change(out: str) -> bool:
    """True when an owner's review reply signals it has nothing to revise."""
    low = out.strip().lower()
    return len(low) <= 60 and ("no change" in low or "nothing to add" in low or "no update" in low)


async def _run_reporter(request: Request, user: dict, sid: str, sid8: str, run_tag: str,
                        intent: str, usable: list[dict], cfg: dict, arch_by_id: dict, *,
                        tiers, api_key, env_vars, dispatch_timeout, input_names,
                        dest_dir: Path, seen_gen: set[str],
                        answered_asks: list[dict] | None = None,
                        shared_context: str = "",
                        research_dir: Path | None = None,
                        corpus: bool = False,
                        honesty: bool = False,
                        facts: bool = False,
                        facts_block: str = "") -> tuple[str, list[dict]]:
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
        tools=_augment_tools(arch.get("tools") or [], research_dir, facts=facts),
        tier=arch.get("tier", "reason"),
        tiers=tiers, api_key=api_key, env_vars=env_vars,
        # Bind the reporter to the same shared VFS project so it reads the
        # team's files from (and writes the assembled deliverable to) one folder.
        # CLAW_NO_SCALE: the reporter ASSEMBLES pieces — turn off the list/scale +
        # write-file-enforcement pipeline so it isn't blocked in an infinite rewrite
        # loop (the pieces read as unresolved "list members").
        extra_env=_vatra_env(sid, "reporter", reporter_id, 0) + [{"key": "CLAW_NO_SCALE", "value": "1"}],
        corpus=corpus,  # R10
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
        # Also give the reporter the session's input files (e.g. a prior report on a
        # fill-gaps run, which the intent tells it to integrate into).
        for name in input_names:
            try:
                shutil.copy2(dest_dir / name, ws / name)
            except OSError:
                pass
        big = len(slices_full) > _SLICES_INLINE_CHARS
        inline = slices_full[:_SLICES_INLINE_CHARS] + ("\n\n…(full text in vatra-slices.md)" if big else "")
        template = (_INSTRUCTIONS_DIR / "vatra" / "reporter.md").read_text()
        prompt = template.replace("{intent}", intent).replace("{slices}", inline)
        # R1: when the Research Map is armed, tell the reporter it can search the
        # WHOLE folder (not just the inlined slice) so a big blackboard that
        # doesn't fit its context is still fully covered.
        if research_dir is not None and big:
            prompt = research_map.preamble(research_dir) + prompt
        if shared_context.strip():
            prompt += ("\n\n## Shared conventions the pieces were built against\n"
                       "Keep the final deliverable fully consistent with these (the pieces should "
                       "already follow them; enforce it if any drifted):\n"
                       f"{shared_context.strip()}\n")
        # Honesty overlay: the exception clause to reporter.md's "resolve it and
        # don't narrate the disagreement" — genuinely unresolved conflicts and
        # assumptions surface in one labeled section instead of being absorbed.
        # Appended at runtime so honesty_guard:false keeps reporter.md verbatim.
        if honesty:
            prompt += REPORTER_HONESTY_DIRECTIVE
        # Facts ledger: the canonical values every figure in the deliverable
        # must match — inlined (it's small and structured), not searched-for.
        if facts_block.strip():
            prompt += REPORTER_FACTS_DIRECTIVE + facts_block.strip() + "\n"
        prompt += _vfs_directive(_vfs_project(sid))

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


# ── R8: grounded claim verification ──────────────────────────────────

_FACTCHECK_ARCHETYPES = ("fact-checker", "deep-researcher", "market-scanner")


async def _claim_check(request: Request, user: dict, sid: str, sid8: str, run_tag: str, *,
                       intent: str, deliverable: str, arch_by_id: dict, tiers,
                       api_key, env_vars, dispatch_timeout, quality,
                       research_dir, facts_block: str = "") -> tuple[str, dict | None, list[dict] | None]:
    """Spawn a web-tool fact-checker, verify the deliverable's load-bearing claims,
    revise it to fix any verified WRONG and hedge any unconfirmable specific it
    asserted as fact, and write a standalone audit ledger. Returns
    ``(deliverable, audit_doc, findings)`` — the (possibly revised) text, a
    generated-file descriptor for the ledger (or ``None``), and the checker's
    verdict list (``None`` when the check never ran, for the metrics tally).
    Best-effort: on any failure the original deliverable is returned unchanged."""
    from captain_claw.flight_deck import research_verify as rv
    if not (deliverable or "").strip():
        return deliverable, None, None
    checker = next((arch_by_id[a] for a in _FACTCHECK_ARCHETYPES if a in arch_by_id), None) \
        or next(iter(arch_by_id.values()), None)
    if not checker:
        return deliverable, None, None
    role = checker.get("role") or checker["id"]
    tools = list(dict.fromkeys(
        (checker.get("tools") or []) + ["web_search", "web_fetch", "researchmap", "read"]))

    _phase(sid, "Fact-checking")
    _progress(sid, "verify", f"Fact-checker ({role}) verifying the deliverable's claims…", agent=role)
    sp = await _spawn_worker(
        request, user, name=f"vatra-{sid8}-{run_tag}-factcheck-{checker['id']}",
        description=f"Vatra fact-checker · {role}",
        cognitive_mode=checker.get("cognitive_mode", "phrygian"),  # adversarial lens
        tools=tools, tier=checker.get("tier", "reason"), tiers=tiers,
        api_key=api_key, env_vars=env_vars,
        extra_env=_vatra_env(sid, "factcheck", checker["id"], 0),
        corpus=quality.source_corpus)
    if not sp["ok"]:
        _progress(sid, "verify", f"Fact-checker spawn failed ({sp['message']})", ok=False)
        return deliverable, None, None

    def _on_action(act: dict) -> None:
        detail = act.get("detail", "")
        if act["tool"] == "narration":
            _progress(sid, "narration", f"{role}: {detail}", agent=role, tool="narration", detail=detail)
        else:
            suffix = f": {detail}" if detail else ""
            _progress(sid, "action", f"{role} → {act['tool']}{suffix}", agent=role, tool=act["tool"], detail=detail)

    def _on_usage(pt: int, ct: int, tt: int) -> None:
        _progress(sid, "usage", f"{role} · {pt:,}→{ct:,} tok", agent=role,
                  prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

    _track_worker(sid, sp["slug"], add=True)
    findings: list[dict] = []
    text, revised_applied = deliverable, False
    try:
        prompt = rv.claim_check_prompt(deliverable, intent, quality.claim_check_max,
                                       corpus_hint=research_dir is not None,
                                       facts_block=facts_block)
        d = await _dispatch_one(sp["port"], sp["auth"], prompt, dispatch_timeout,
                                on_action=_on_action,
                                fleet_instructions=checker.get("fleet_instructions", ""),
                                agent_name=role, on_usage=_on_usage)
        findings = rv.parse_findings(d.get("output") or "")
        _progress(sid, "verify", f"Fact-checker: {rv.summary_line(findings)}", agent=role)
        for f in rv.refuted(findings):
            _progress(sid, "narration", f"{role} (refuted): {f['claim']} → {f['correction']}",
                      agent=role, tool="narration", detail=f["correction"])
        for f in rv.unconfirmed(findings):
            _progress(sid, "narration", f"{role} (unconfirmed): {f['claim']} → hedged",
                      agent=role, tool="narration", detail=f["hedge"])
        fix = rv.fix_instructions(findings)
        if fix:
            _progress(sid, "verify", "Revising the deliverable to correct/hedge the flagged claims…", agent=role)
            revise_prompt = (
                "Now output the FULL corrected deliverable. Apply EXACTLY these changes and "
                "change nothing else — keep all other content, structure and formatting identical:\n\n"
                f"{fix}\n\nOutput the complete corrected document only, no preamble.")
            d2 = await _dispatch_one(sp["port"], sp["auth"], revise_prompt, dispatch_timeout,
                                     on_action=_on_action,
                                     fleet_instructions=checker.get("fleet_instructions", ""),
                                     agent_name=role, on_usage=_on_usage)
            revised = (d2.get("output") or "").strip()
            collapsed = not revised or (len(deliverable) > 800 and len(revised) < 0.5 * len(deliverable))
            if collapsed:
                _progress(sid, "verify", "Kept the original (correction pass collapsed)", agent=role, ok=False)
            else:
                text, revised_applied = revised, True
                _progress(sid, "verify",
                          f"Deliverable corrected: {len(rv.refuted(findings))} fix(es) + "
                          f"{len(rv.unconfirmed(findings))} hedge(s) applied", agent=role)
    except Exception as e:  # noqa: BLE001
        log.warning("Vatra claim check failed", error=str(e))
    finally:
        _teardown([sp["slug"]])
        _track_worker(sid, sp["slug"], add=False)
    # Non-destructive audit ledger: every checked claim, its verdict and the action
    # taken — written even when nothing was auto-changed, so unconfirmable specifics
    # stay visible instead of only appearing in the one-line log tally.
    doc = rv.write_audit(_session_files_dir(sid), findings, question=intent,
                         revised=revised_applied)
    if doc:
        _progress(sid, "verify", f"Fact-check report saved · {doc['name']}", agent=role)
    return text, doc, findings


# ── Coordinator: route blackboard asks to helpers (non-blocking) ─────

async def _coordinate_asks(request: Request, user: dict, sid: str, sid8: str,
                           run_tag: str, intent: str, domain: str, *,
                           archetypes: list[dict], arch_by_id: dict, tiers,
                           api_key, env_vars, dispatch_timeout,
                           stop_event: asyncio.Event, corpus: bool = False) -> int:
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
                api_key=api_key, env_vars=env_vars, dispatch_timeout=dispatch_timeout,
                corpus=corpus)
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
        except TimeoutError:
            pass
    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)
    return answered


async def _fulfill_ask(request: Request, user: dict, sid: str, sid8: str, run_tag: str,
                       intent: str, ask: dict, *, archetypes: list[dict],
                       arch_by_id: dict, tiers, api_key, env_vars,
                       dispatch_timeout, corpus: bool = False) -> bool:
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
        corpus=corpus,  # R10
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


class _VatraWaitReq(_AgentReq):
    owner: str = ""
    path: str = ""
    query: str = ""
    wait: int = 0


@router.post("/agent/wait")
async def agent_wait(body: _VatraWaitReq):
    """Block until a teammate's artifact is ready, then hand it back.

    A specialist whose part genuinely depends on another piece calls this instead
    of reading once, finding nothing, and improvising. Two ready conditions
    (provide at least one — if both, whichever lands first wins):

      * ``path`` — a shared-VFS file (``vfs:<proj>/<file>``): ready when it exists
        and is non-empty; returns its content.
      * ``query`` — keywords: ready when a teammate's board post matches.

    Each call is bounded by ``_MAX_WAIT_S`` (< the per-dispatch timeout), and a
    per-owner **ledger** bounds the *total* time an owner may wait across retries
    (``_WAIT_TOTAL_BUDGET_S`` / ``_WAIT_MAX_ATTEMPTS``). So the owner can wait
    *longer* by retrying while the dependency is still plausibly coming, but once
    the budget is spent the endpoint **short-circuits** to a terminal "stop
    waiting, produce now" — retries can never become an infinite loop. On any
    timeout it returns ``ready=False`` plus a board digest and a ``note`` telling
    the caller whether it may wait again. Other owners keep working meanwhile."""
    owner_id = _resolve_owner(body)
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner_id)
    if not sess:
        raise HTTPException(404, "session not found")
    path = (body.path or "").strip()
    query = (body.query or "").strip()
    if not path and not query:
        raise HTTPException(400, "provide path or query to wait for")
    project = _vfs_project(body.session_id)

    # The tool sends `owner` = the archetype id (e.g. "deep-researcher"); the live
    # panel groups by the owner's DISPLAY role (e.g. "Deep Researcher", via
    # _owner_label). Resolve the role so these wait events attach to the real owner
    # card instead of forming a phantom card that never gets a dispatch-done marker
    # (which shows as an extra agent stuck spinning). Best-effort; fall back to the id.
    who = body.owner or "agent"
    try:
        _archs = await merged_archetypes(db, owner_id)
        _role = next((a.get("role") for a in _archs if a.get("id") == body.owner), "")
        if _role:
            who = _role
    except Exception as e:  # noqa: BLE001 — label resolution is cosmetic
        log.debug("Vatra wait label resolve failed", error=str(e))

    # Schedule guard (grouped runs): waiting on a teammate that runs in a LATER
    # phase can never succeed — its group hasn't been dispatched yet. Refuse
    # instantly (no wait-budget charge) with marching orders, instead of letting
    # the worker burn 90s per attempt on output that cannot arrive.
    sched = _group_schedule.get(body.session_id)
    if sched and query:
        later = vatra_groups.match_later_owner(query, sched)
        if later is not None:
            role = later.get("role") or later.get("arch") or "that teammate"
            letter = vatra_groups.group_label(int(later.get("group") or 0))
            # Pull-forward first: call the later owner to run NOW (bounded — see
            # the run loop). If a pull is running, the waiter just waits again
            # and picks the artifact up when it lands.
            pull = sched.get("pull")
            if callable(pull) and pull(later.get("subtask") or ""):
                _progress(body.session_id, "wait",
                          f"⤴ {who} needs {role} (group {letter}) — called forward, "
                          "dispatching it now", agent=who)
                return {"ready": False, "pulled_forward": True,
                        "note": (f"{role} was scheduled in a later phase but has been "
                                 "CALLED FORWARD and is producing its part right now. "
                                 "Wait again with the same query to pick it up — it may "
                                 "take a few minutes. If your wait budget runs out first, "
                                 "proceed and mark its values as (unverified).")}
            _progress(body.session_id, "wait",
                      f"⛔ {who} tried to wait on {role} — it runs LATER (group {letter}); "
                      "refused instantly", agent=who, ok=False)
            recent = await db.list_vatra_board(
                body.session_id, limit=12, exclude_owner=body.owner or None)
            return {"ready": False, "not_scheduled": True,
                    "board": [_board_entry(e) for e in recent],
                    "note": (f"{role} runs in a LATER phase (group {letter}) and has not "
                             "started — its output CANNOT arrive while you run. Do NOT wait "
                             "for it and do NOT retry this wait. Produce your part NOW from "
                             "the board, the shared folder and the datastore, and mark "
                             "anything it would have provided or verified as (unverified) "
                             "so a later phase settles it.")}

    # Per-owner wait ledger — the loop backstop. Once an owner has spent its total
    # budget or attempt cap, stop waiting entirely and tell it to proceed. Enforced
    # here (not left to the agent), so even a stubborn retry can't loop.
    ledger = _wait_ledger.setdefault(body.session_id, {})
    rec = ledger.setdefault(body.owner or "agent", {"waited": 0.0, "attempts": 0})
    remaining = _WAIT_TOTAL_BUDGET_S - rec["waited"]
    if rec["attempts"] >= _WAIT_MAX_ATTEMPTS or remaining <= 1:
        recent = await db.list_vatra_board(
            body.session_id, limit=12, exclude_owner=body.owner or None)
        _progress(
            body.session_id, "wait",
            f"🛑 {who} stop waiting — budget spent ({int(rec['waited'])}s / "
            f"{int(rec['attempts'])} tries); proceeding with what's on the board",
            agent=who, ok=False)
        return {"ready": False, "exhausted": True,
                "waited_total": round(rec["waited"]), "attempts": int(rec["attempts"]),
                "board": [_board_entry(e) for e in recent],
                "note": ("You have spent your full wait budget and the dependency has not "
                         "arrived. Do NOT call wait again — produce your part NOW from the "
                         "shared board, the datastore, and your own analysis.")}

    # Per-call wait: floor a short request so the dep gets real time to land, capped
    # by the single-call max AND by whatever total budget remains.
    requested = int(body.wait) if body.wait else _MAX_WAIT_S
    wait = int(min(_MAX_WAIT_S, remaining, max(requested, _WAIT_MIN_S)))
    started = time.monotonic()
    deadline = started + wait
    target = path or f"posts matching {query!r}"
    _progress(body.session_id, "wait", f"⏳ {who} waiting on {target} (≤{wait}s)…", agent=who)
    while True:
        if path:
            real = _vfs_resolve_under(owner_id, project, path)
            if real and real.is_file():
                try:
                    data = real.read_text(errors="replace")
                except Exception:
                    data = ""
                if data.strip():
                    _progress(body.session_id, "wait", f"✓ {who} got {path}", agent=who)
                    return {"ready": True, "kind": "file", "path": path,
                            "content": data[:_WAIT_CONTENT_CAP],
                            "truncated": len(data) > _WAIT_CONTENT_CAP}
        if query:
            rows = await db.search_vatra_board(
                body.session_id, query, limit=20, exclude_owner=body.owner or None)
            if rows:
                _progress(body.session_id, "wait", f"✓ {who} got a board match for {query!r}", agent=who)
                return {"ready": True, "kind": "board",
                        "entries": [_board_entry(e) for e in rows]}
        if time.monotonic() >= deadline:
            # Charge the actual elapsed time against this owner's total budget.
            rec["waited"] += time.monotonic() - started
            rec["attempts"] += 1
            left = max(0.0, _WAIT_TOTAL_BUDGET_S - rec["waited"])
            can_retry = left > 1 and rec["attempts"] < _WAIT_MAX_ATTEMPTS
            _progress(
                body.session_id, "wait",
                f"⌛ {who} wait timed out ({wait}s) — {target} not ready"
                + (f" · {int(left)}s budget left" if can_retry else " · budget spent"),
                agent=who, ok=False)
            recent = await db.list_vatra_board(
                body.session_id, limit=12, exclude_owner=body.owner or None)
            note = (
                f"Not ready yet. You've waited ~{int(rec['waited'])}s of a ~{_WAIT_TOTAL_BUDGET_S}s "
                f"budget ({int(left)}s left across {_WAIT_MAX_ATTEMPTS - int(rec['attempts'])} more "
                "attempt(s)). If your part TRULY depends on this, you may wait once more; otherwise "
                "produce it now from the board + datastore."
                if can_retry else
                "Wait budget spent — do NOT wait again. Produce your part now from the board, the "
                "datastore, and your own analysis."
            )
            return {"ready": False, "waited": wait,
                    "waited_total": round(rec["waited"]), "attempts": int(rec["attempts"]),
                    "budget_left_s": round(left), "can_retry": can_retry,
                    "board": [_board_entry(e) for e in recent], "note": note}
        await asyncio.sleep(_WAIT_POLL_S)


# ── Shared board: real-time shared memory across the team ────────────

_BOARD_CONTENT_CAP = 30_000  # per-entry content stored on the board


def _board_post_bg(sid: str, owner: str, subtask: str, kind: str, title: str, content: str) -> None:
    """Fire-and-forget board write from a sync streaming callback (we're already on
    the event loop). Best-effort: a failed write must never break the agent's turn."""
    text = (content or "").strip()
    if not text:
        return

    async def _w() -> None:
        try:
            await get_db().add_vatra_board(sid, owner, subtask, kind, title, text[:_BOARD_CONTENT_CAP])
        except Exception as e:
            log.debug("Vatra board write failed", error=str(e))
    try:
        t = asyncio.create_task(_w())
        _basna_agent_tasks.add(t)
        t.add_done_callback(_basna_agent_tasks.discard)
    except RuntimeError:
        pass


def _board_entry(e: dict) -> dict:
    return {"id": e.get("id"), "from": e.get("from_owner", ""), "kind": e.get("kind", ""),
            "title": e.get("title", ""), "content": e.get("content", ""),
            "at": e.get("created_at", "")}


class _VatraBoardPostReq(_AgentReq):
    owner: str = ""
    subtask_id: str = ""
    kind: str = "note"
    title: str = ""
    text: str = ""


class _VatraBoardReadReq(_AgentReq):
    owner: str = ""
    kind: str = ""
    limit: int = 40


class _VatraBoardSearchReq(_AgentReq):
    owner: str = ""
    query: str = ""
    limit: int = 20


async def _board_session(body) -> tuple[str, str]:
    owner_id = _resolve_owner(body)
    if not body.session_id:
        raise HTTPException(400, "session_id is required")
    sess = await get_db().get_basna_session(body.session_id, owner_id)
    if not sess:
        raise HTTPException(404, "session not found")
    return owner_id, body.session_id


@router.post("/agent/board/post")
async def agent_board_post(body: _VatraBoardPostReq):
    """An agent shares a note/finding to the team board for everyone to see."""
    await _board_session(body)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    kind = body.kind if body.kind in ("note", "output") else "note"
    e = await get_db().add_vatra_board(
        body.session_id, body.owner, body.subtask_id, kind, body.title, text[:_BOARD_CONTENT_CAP])
    return {"status": "ok", "id": e["id"]}


@router.post("/agent/board/read")
async def agent_board_read(body: _VatraBoardReadReq):
    """Recent board entries from TEAMMATES (the caller's own entries are excluded)."""
    await _board_session(body)
    kinds = [body.kind] if body.kind else None
    rows = await get_db().list_vatra_board(
        body.session_id, kinds=kinds, limit=body.limit, exclude_owner=body.owner or None)
    return {"entries": [_board_entry(e) for e in rows], "count": len(rows)}


@router.post("/agent/board/search")
async def agent_board_search(body: _VatraBoardSearchReq):
    """Search teammates' board entries by keyword."""
    await _board_session(body)
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(400, "query is required")
    rows = await get_db().search_vatra_board(
        body.session_id, q, limit=body.limit, exclude_owner=body.owner or None)
    return {"entries": [_board_entry(e) for e in rows], "count": len(rows)}


# ── UI manual start (user-scoped, background) ────────────────────────

class VatraStartRequest(BaseModel):
    intent: str
    title: str = ""
    max_agents: int = Field(default=6, ge=1, le=10)
    # Per-tier model config from the Library (same shape Basna's execute uses).
    tiers: dict | None = None
    env_vars: list | None = None
    api_key: str = ""
    # User-fixed team: when non-empty, the Lead must give each a subtask.
    archetype_ids: list[str] = []
    # Which Library tier the Lead (decomposition) runs on — a reasoning task.
    router_tier: str = "reason"
    # Deep / Horizon closer (verify + revise the assembled deliverable). None → off.
    horizon: dict | None = None
    # R12 intent brief: opt-in quality profile (only `intent_brief` is read here).
    quality: dict = Field(default_factory=dict)
    # R12: a user-edited brief to decompose on. When set, the Lead plans against it
    # verbatim (no re-derivation) — the "edit the brief, re-plan on it" path. Empty →
    # derive one iff quality.intent_brief.
    brief: str = ""
    # Opt-in shared datastore: passed at PLAN time too so the Lead decomposes the task
    # to persist structured data into the shared relational store (not a JSON file).
    shared_datastore: bool = False
    # Target VFS folder — passed at PLAN time so the Lead can be seeded with what the
    # folder already contains (continue vs restart). Empty → auto (vatra-<sid8>).
    vfs_project: str = ""
    # Opt-in: seed the plan with the knowledge (report + gaps/blind spots, optionally
    # the board) of these FINISHED prior runs — folded into the Lead prompt and the
    # team's shared_context. Empty → no seeding.
    knowledge_session_ids: list[str] = []
    knowledge_include_board: bool = False
    # Read-only reference VFS folders workers consult BEFORE web-searching. The
    # knowledge runs' own folders are auto-added, so this is for EXTRA folders.
    reference_folders: list[str] = []
    # Project bundle this run belongs to (empty = Unfiled). The project's theme is
    # folded into the plan/shared_context and its folder added as a reference.
    project_id: str = ""


class VatraExecuteRequest(BaseModel):
    session_id: str
    tiers: dict | None = None
    env_vars: list | None = None
    api_key: str = ""
    # Deep / Horizon: Vatra honors the closer (verify+revise the deliverable). Keys:
    # close, critics[], critic_tier. Per-owner pools (worker) are deferred. None → off.
    horizon: dict | None = None
    # Max agent turns to run at once (0 = unlimited). Mainly for local models.
    max_parallel: int = Field(default=0, ge=0, le=16)
    # Ordered execution groups (A→B→C→D) instead of all-at-once. Opt-in.
    execution_groups: bool = False


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
        config=json.dumps({"mode": "vatra", "source": "ui", "max_agents": body.max_agents,
                           **({"shared_datastore": True} if body.shared_datastore else {}),
                           **({"horizon": body.horizon} if body.horizon else {})}))
    sid = sess["id"]
    registry = _load_registry()
    # The Lead's decomposition is a reasoning task — run it on the user-selected
    # tier (default reasoning), not the fast tier.
    creds = _resolve_creds(registry, body.tiers, body.api_key, body.router_tier or "reason")
    # R12 intent brief (opt-in): clarify the task into a structured brief BEFORE the
    # Lead decomposes, so the TEAM + subtasks are planned against the clarified task.
    # A user-edited brief (body.brief) is used verbatim — the "edit the brief, re-plan
    # on it" path; else derive one when quality.intent_brief is set. brief_task keeps
    # the original intent authoritative. brief == "" reproduces today's planning.
    quality = QualityProfile.from_dict(body.quality)
    brief = research_brief.parse_brief(body.brief) if body.brief.strip() else ""
    if not brief and quality.intent_brief:
        try:
            prov, mt = _provider_call(creds, temperature=0.2, default_max=1500, cap=4096)
            from captain_claw.llm import Message
            bresp = await prov.complete(
                [Message(role="user", content=research_brief.derive_brief_prompt(intent))],
                temperature=0.2, max_tokens=mt)
            brief = research_brief.parse_brief(bresp.content or "")
        except Exception as e:  # noqa: BLE001 — brief is best-effort; fall back to raw intent
            log.warning("Vatra intent-brief derivation failed; planning on raw intent", error=str(e))
            brief = ""
    task_for_planning = research_brief.brief_task(intent, brief)
    # Project bundle: the theme (description + instructions) seeds the Lead's plan
    # and every worker's shared_context; the project folder becomes a reference.
    _proj_theme, _proj_folder = "", ""
    if body.project_id.strip():
        from captain_claw.flight_deck.basna_routes import _project_context
        _proj_theme, _proj_folder = await _project_context(db, user["id"], body.project_id.strip())
    if _proj_theme:
        task_for_planning = f"{_proj_theme}\n\n---\n\n{task_for_planning}"
    try:
        _prior = ""
        _ref_folders = list(body.reference_folders or [])
        if _proj_folder:
            _ref_folders = list(dict.fromkeys(_ref_folders + [_proj_folder]))
        if body.knowledge_session_ids:
            from captain_claw.flight_deck.basna_routes import build_prior_knowledge, knowledge_run_folders
            _prior = await build_prior_knowledge(
                db, user["id"], body.knowledge_session_ids,
                include_board=body.knowledge_include_board)
            # Prior runs' folders are read-only reference by default.
            _ref_folders = list(dict.fromkeys(
                _ref_folders + await knowledge_run_folders(db, user["id"], body.knowledge_session_ids)))
        route = await _build_plan(db, user["id"], task_for_planning, body.max_agents, creds,
                                  force_ids=body.archetype_ids or None,
                                  shared_datastore=body.shared_datastore,
                                  vfs_project=body.vfs_project, prior_knowledge=_prior)
        # Fold read-only reference folders into shared_context so every worker checks
        # them before web-searching.
        _ref = _reference_directive(_ref_folders)
        if _ref:
            route["shared_context"] = (route.get("shared_context", "") + _ref).strip()
        # The project theme is folded into shared_context by execute_vatra (the one
        # choke point both fresh and continuation runs pass through) — not here.
    except HTTPException:
        await db.delete_basna_session(sid, user["id"])
        raise
    except Exception as e:
        await db.delete_basna_session(sid, user["id"])
        raise HTTPException(502, f"Vatra Lead failed: {_lead_error_msg(e)}")
    route["brief"] = brief  # R12: persisted with the plan; execute dispatches on it, UI edits it
    # Persist the project binding on the session config so the UI groups the run
    # and any later step can recover the bundle.
    _cfg_extra = {}
    if body.project_id.strip():
        try:
            _cfg_extra = json.loads((await db.get_basna_session(sid, user["id"]) or {}).get("config") or "{}")
        except Exception:  # noqa: BLE001
            _cfg_extra = {}
        _cfg_extra["project_id"] = body.project_id.strip()
        _cfg_extra["project_context"] = _proj_theme
    _upd = dict(domain=route["domain"], route=json.dumps(route), status="routed")
    if _cfg_extra:
        _upd["config"] = json.dumps(_cfg_extra)
    await db.update_basna_session(sid, user["id"], **_upd)
    return {"session_id": sid, "title": title, **route}


@router.post("/execute")
async def execute_vatra_ui(body: VatraExecuteRequest, request: Request,
                           user: dict = Depends(get_current_user)):
    """Run the permanent Group 0 pre-phase in the background: the Long Horizon Planner
    drafts the coordination plan, then the run PAUSES at the approval gate
    (status=awaiting_plan). The UI polls progress, shows the editable plan, and calls
    /plan/approve or /plan/cancel. The team is not spawned until approval."""
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    exec_req = ExecuteRequest(
        session_id=body.session_id, tiers=body.tiers or None,
        env_vars=body.env_vars or None, api_key=body.api_key or "",
        horizon=body.horizon or None, max_parallel=body.max_parallel or 0,
        execution_groups=bool(body.execution_groups))
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(plan_vatra_group0(exec_req, stub, user, gate=True))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": body.session_id, "status": "planning"}


class VatraPlanApproveRequest(BaseModel):
    session_id: str
    # The (possibly edited) group0_plan. None → approve the drafted plan as-is.
    plan: dict | None = None
    # Optional run-knob overrides; default to the values persisted at plan time.
    tiers: dict | None = None
    env_vars: list | None = None
    api_key: str = ""
    horizon: dict | None = None
    max_parallel: int = Field(default=0, ge=0, le=16)
    execution_groups: bool | None = None
    grouped_review: bool = False
    quality: dict | None = None


class VatraPlanCancelRequest(BaseModel):
    session_id: str


@router.post("/plan/approve")
async def approve_vatra_plan(body: VatraPlanApproveRequest, request: Request,
                             user: dict = Depends(get_current_user)):
    """Approve the Group 0 coordination plan and start the run. Persists the edited
    plan (if any), then backgrounds execute_vatra with the knobs saved at plan time
    (overridable). Mirrors the Code-mode plan/approve gate."""
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    if (sess.get("status") or "") != "awaiting_plan":
        raise HTTPException(409, "no plan awaiting approval")
    try:
        route = json.loads(sess.get("route") or "{}")
    except json.JSONDecodeError:
        route = {}
    if body.plan is not None:
        route["group0_plan"] = _sanitize_group0_plan(body.plan, route.get("subtasks") or [])
        await db.update_basna_session(body.session_id, user["id"], route=json.dumps(route))
    try:
        cfg = json.loads(sess.get("config") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    # Tiers/env carry the model config + API keys — they are request-scoped and NEVER
    # persisted to the session (secrets), so approve must resend them. If the client
    # omitted them, fall back to the owner's saved workspace tiers; otherwise the run
    # silently drops to the registry-default model (the anthropic default) instead of
    # the user's configured tier — the model the planner already used.
    _tiers = body.tiers
    _env = body.env_vars
    if _tiers is None:
        _owner_tiers, _owner_env = await _load_owner_tiers(db, user["id"])
        _tiers = _owner_tiers
        if _env is None:
            _env = _owner_env
    exec_req = ExecuteRequest(
        session_id=body.session_id, tiers=_tiers or None,
        env_vars=_env or None, api_key=body.api_key or "",
        horizon=body.horizon if body.horizon is not None else (cfg.get("horizon") or None),
        max_parallel=body.max_parallel or int(cfg.get("max_parallel") or 0),
        execution_groups=(body.execution_groups if body.execution_groups is not None
                          else bool(cfg.get("execution_groups"))),
        grouped_review=bool(body.grouped_review or cfg.get("grouped_review")),
        shared_datastore=bool(cfg.get("shared_datastore")),
        vfs_project=cfg.get("vfs_project") or "",
        quality=body.quality if body.quality is not None else (cfg.get("quality") or None))
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(execute_vatra(exec_req, stub, user))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": body.session_id, "status": "running"}


@router.post("/plan/cancel")
async def cancel_vatra_plan(body: VatraPlanCancelRequest,
                            user: dict = Depends(get_current_user)):
    """Discard a coordination plan awaiting approval — nothing was spawned (the planner
    was already torn down), so the session simply returns to 'routed' and its plan is
    kept as re-executable history."""
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    if (sess.get("status") or "") != "awaiting_plan":
        raise HTTPException(409, "no plan awaiting approval")
    await db.update_basna_session(body.session_id, user["id"], status="routed")
    _progress(body.session_id, "note", "Plan discarded — nothing was run. Re-run when ready.")
    _progress_done(body.session_id)
    return {"session_id": body.session_id, "status": "idle"}


class VatraPlanReplanRequest(BaseModel):
    session_id: str
    # The edited plan — its per-agent `group` assignments (absolute locks) and answered
    # `questions` are read; the mandates / dependencies / hand-offs are regenerated by
    # the planner for the new phasing + answers.
    plan: dict | None = None
    tiers: dict | None = None
    env_vars: list | None = None
    api_key: str = ""


async def _replan_group0(sid: str, request: Request, user: dict, *, new_groups: dict,
                         clarifications: str = "", tiers: dict | None,
                         env_vars: list | None, api_key: str) -> None:
    """Re-run the Long Horizon Planner after the user re-grouped agents and/or answered
    its clarifying questions at the gate. Applies the new group assignments, re-resolves
    phasing, folds the answers into the prompt, regenerates the whole coordination plan,
    and lands back at the approval gate. Any failure re-emits the existing gate so the
    user is never stranded."""
    db = get_db()
    try:
        sess = await db.get_basna_session(sid, user["id"])
        if not sess:
            return
        _run_sid.set(sid)
        _RUN_USAGE.setdefault(sid, [])
        try:
            route = json.loads(sess.get("route") or "{}")
        except json.JSONDecodeError:
            route = {}
        subtasks = route.get("subtasks") or []
        archetypes = await merged_archetypes(db, user["id"])
        arch_by_id = {a["id"]: a for a in archetypes}
        # Apply the user's new groups as absolute locks, then re-resolve so the plan
        # (and the planner's view of it) reflects exactly where the user put each agent.
        for s in subtasks:
            g = str(new_groups.get(s["id"]) or "").strip()
            if g:
                s["group_lock"] = g
        vatra_groups.resolve_groups(subtasks, arch_by_id)

        _progress_start(sid)
        await db.update_basna_session(sid, user["id"], status="planning")
        _phase(sid, "Group 0 · Long Horizon Planner (re-plan)")
        _progress(sid, "group0",
                  "Re-planning the coordination"
                  + (" with your answers…" if clarifications else " for your new grouping…"))

        _tiers, _env = tiers, env_vars
        if _tiers is None:
            _tiers, _oenv = await _load_owner_tiers(db, user["id"])
            if _env is None:
                _env = _oenv
        session_files = _parse_files(sess)
        input_files = [f for f in session_files if f.get("kind") != "generated"]
        plan = await _run_group0_planner(
            request, user, sid, intent=(sess.get("intent") or "").strip(),
            shared_context=route.get("shared_context", ""),
            file_names=[f["name"] for f in input_files], subtasks=subtasks,
            arch_by_id=arch_by_id, tiers=_tiers, api_key=api_key, env_vars=_env, timeout=600.0,
            clarifications=clarifications)
        route["group0_plan"] = plan
        route["subtasks"] = subtasks  # persist the new groups for the run
        await db.update_basna_session(sid, user["id"], route=json.dumps(route))
        await db.update_basna_session(sid, user["id"], status="awaiting_plan")
        _emit_awaiting_plan(sid, plan)
    except Exception as e:  # noqa: BLE001 — never strand the gate
        log.warning("Vatra Group 0 re-plan failed", session_id=sid, error=str(e))
        try:
            sess2 = await db.get_basna_session(sid, user["id"])
            route2 = json.loads((sess2 or {}).get("route") or "{}")
            await db.update_basna_session(sid, user["id"], status="awaiting_plan")
            _emit_awaiting_plan(sid, route2.get("group0_plan") or {})
        except Exception:  # noqa: BLE001
            pass


@router.post("/plan/replan")
async def replan_vatra_plan(body: VatraPlanReplanRequest, request: Request,
                            user: dict = Depends(get_current_user)):
    """Re-plan the coordination after the user re-grouped agents. Reads the new per-agent
    groups from the edited plan, re-runs the planner in the background (status→planning),
    and lands back at the gate with a fresh plan the user reviews before Execute."""
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    if (sess.get("status") or "") != "awaiting_plan":
        raise HTTPException(409, "no plan awaiting approval")
    new_groups: dict[str, str] = {}
    if isinstance(body.plan, dict):
        for a in body.plan.get("agents") or []:
            if isinstance(a, dict) and a.get("subtask_id") and str(a.get("group") or "").strip():
                new_groups[str(a["subtask_id"])] = str(a["group"]).strip()
    clarifications = _format_clarifications(
        (body.plan or {}).get("questions") if isinstance(body.plan, dict) else [])
    # Flip status synchronously so the UI leaves the gate immediately (no poll lag)
    # instead of waiting for the background task to set it.
    await db.update_basna_session(body.session_id, user["id"], status="planning")
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(_replan_group0(
        body.session_id, stub, user, new_groups=new_groups, clarifications=clarifications,
        tiers=body.tiers, env_vars=body.env_vars, api_key=body.api_key or ""))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": body.session_id, "status": "planning"}


async def launch_vatra_resume(
    session_id: str, user: dict, *, tiers: dict | None = None,
    env_vars: list | None = None, api_key: str = "", horizon: dict | None = None,
    max_parallel: int = 0, execution_groups: bool = False,
) -> dict:
    """Resume a stalled/cancelled Vatra run in the background — shared by the UI
    endpoint and the agent/tool entry.

    Restores every owner that already finished (from `vatra_runs` — no re-run, no
    re-spend), re-dispatches only the missing ones, then synthesizes. Reuses the
    persisted plan (same cast) + the same VFS folder + blackboard; run knobs
    (grouped, parallelism, datastore, folder) come from the session config so the
    resumed flow matches the original. Any workers still alive are torn down first
    so a single coroutine owns the session."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    if (sess.get("status") or "") == "done" and (sess.get("truth") or "").strip():
        raise HTTPException(400, "session already completed — nothing to resume")
    if (sess.get("status") or "") == "awaiting_plan":
        raise HTTPException(400, "approve or cancel the plan first — nothing to resume")
    try:
        cfg = json.loads(sess.get("config") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    try:
        from captain_claw.flight_deck.basna_routes import _cancel_basna_run
        await _cancel_basna_run(session_id, user["id"])
    except Exception as e:  # noqa: BLE001
        log.warning("Vatra resume: pre-cancel failed", session_id=session_id, error=str(e))
    exec_req = ExecuteRequest(
        session_id=session_id, tiers=tiers or None,
        env_vars=env_vars or None, api_key=api_key or "",
        horizon=horizon or None,
        max_parallel=int(cfg.get("max_parallel") or max_parallel or 0),
        execution_groups=bool(cfg.get("execution_groups") or execution_groups),
        shared_datastore=bool(cfg.get("shared_datastore")),
        vfs_project=cfg.get("vfs_project") or "",
        resume=True)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(execute_vatra(exec_req, stub, user))
    _basna_agent_tasks.add(t)
    _agent_run_tasks[session_id] = t
    def _clear(_t: asyncio.Task, _sid: str = session_id) -> None:
        _basna_agent_tasks.discard(_t)
        _agent_run_tasks.pop(_sid, None)
    t.add_done_callback(_clear)
    return {"session_id": session_id, "status": "running", "resumed": True}


@router.post("/sessions/{session_id}/resume")
async def resume_vatra(
    session_id: str, body: VatraExecuteRequest,
    user: dict = Depends(get_current_user),
):
    """UI entry: resume a stalled/cancelled Vatra run from its durable checkpoints."""
    return await launch_vatra_resume(
        session_id, user, tiers=body.tiers, env_vars=body.env_vars,
        api_key=body.api_key, horizon=body.horizon,
        max_parallel=body.max_parallel, execution_groups=body.execution_groups)


@router.post("/start")
async def start_vatra(body: VatraStartRequest, request: Request,
                      user: dict = Depends(get_current_user)):
    """Launch a Vatra run from the UI (one-shot: no separate /route step). Creates a
    session and runs the permanent Group 0 pre-phase in the background — the Long
    Horizon Planner drafts the coordination plan, then the run PAUSES at the approval
    gate. Returns immediately; the UI polls progress and drives /plan/approve or
    /plan/cancel. The agent path (`/agent/start`) is the headless entry (auto-approve)."""
    intent = (body.intent or "").strip()
    if not intent:
        raise HTTPException(400, "intent is required")
    db = get_db()
    title = (body.title or intent[:60]).strip()
    sess = await db.create_basna_session(
        user["id"], intent, title=title,
        config=json.dumps({"mode": "vatra", "source": "ui", "max_agents": body.max_agents,
                           **({"shared_datastore": True} if body.shared_datastore else {}),
                           **({"horizon": body.horizon} if body.horizon else {})}))
    sid = sess["id"]
    exec_req = ExecuteRequest(
        session_id=sid, tiers=body.tiers or None,
        env_vars=body.env_vars or None, api_key=body.api_key or "",
        horizon=body.horizon or None, shared_datastore=body.shared_datastore,
        vfs_project=body.vfs_project or "")
    # Background task with a stub request carrying the owner (spawn_process reads
    # request.state.user_id) — the real request object isn't safe to use post-response.
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
    t = asyncio.create_task(plan_vatra_group0(exec_req, stub, user, gate=True))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": sid, "title": title, "status": "planning"}


# ── Continuation: carry a finished Vatra run forward into another round ──

# Continuation framings per kind. Each is seeded with the prior report + a manifest
# of the shared folder, then the Lead re-decomposes the work across the (same) cast.
_VATRA_HEADERS = {
    "fill_gaps": (
        "Improve and COMPLETE an existing deliverable by filling its coverage gaps — "
        "the parts the original task asked for that the current report missed or "
        "covered only thinly."
    ),
    "continue": (
        "CONTINUE an existing deliverable — extend it forward per the instruction "
        "below, building directly on the current report and the shared folder."
    ),
    "revise": (
        "REVISE an existing deliverable — improve it per the instruction below, "
        "keeping everything that already works."
    ),
}


async def _continue_run(owner: str, parent_session_id: str, user: dict, *,
                        instruction: str = "", kind: str = "continue",
                        same_cast: bool = True,
                        tiers: dict | None, env_vars: list | None, api_key: str) -> dict:
    """Create + run a follow-up Vatra that carries a finished run forward, seeded with
    its final report. The whole chain shares ONE VFS folder (the root run's), so the
    team reads and builds on the accumulated data. `kind`: 'fill_gaps' (coverage gaps),
    'continue' (extend forward), or 'revise' (improve per instruction). `same_cast`
    constrains the Lead to the parent's owner archetypes (it still re-decomposes the
    continuation work across them); otherwise the Lead re-selects the team."""
    db = get_db()
    parent = await db.get_basna_session(parent_session_id, owner)
    if not parent:
        raise HTTPException(404, "session not found")
    truth = (parent.get("truth") or "").strip()
    if not truth:
        raise HTTPException(400, "This run has no assembled report to build on yet.")
    try:
        analysis = json.loads(parent.get("analysis") or "{}")
    except (ValueError, TypeError):
        analysis = {}
    try:
        parent_cfg = json.loads(parent.get("config") or "{}")
    except (ValueError, TypeError):
        parent_cfg = {}
    try:
        parent_route = json.loads(parent.get("route") or "{}")
    except (ValueError, TypeError):
        parent_route = {}

    kind = kind if kind in _VATRA_HEADERS else "continue"
    instruction = (instruction or "").strip()
    # Lineage: every round in the chain shares the ROOT run's folder + grows the counter.
    root_sid = parent_cfg.get("root_session_id") or parent_session_id
    vfs_project = parent_cfg.get("vfs_project") or f"vatra-{root_sid[:8]}"
    round_no = int(parent_cfg.get("round") or 1) + 1

    # The original objective (round 1's task) is the chain's north star — carry it
    # into every round so the team keeps the overall goal, not just "edit this".
    original_objective = (parent.get("intent") or "").strip()
    if root_sid != parent_session_id:
        root_sess = await db.get_basna_session(root_sid, owner)
        if root_sess and (root_sess.get("intent") or "").strip():
            original_objective = root_sess["intent"].strip()
    obj_block = (
        f"ORIGINAL OBJECTIVE (round 1 — what this whole effort is for):\n"
        f"{original_objective[:2000]}\n\n"
        if original_objective else ""
    )

    # Kind-specific focus block.
    if kind == "fill_gaps":
        gaps = analysis.get("gaps") or []
        if not gaps and not instruction:
            raise HTTPException(400, "This run has no coverage gaps to fill.")
        focus = ""
        if gaps:
            gap_lines = "\n".join(
                f"- [{g.get('severity', 'minor')}] {g.get('item', '')}"
                + (f" — {g.get('note', '')}" if g.get("note") else "")
                for g in gaps[:8])
            focus += f"COVERAGE GAPS to fill:\n{gap_lines}\n"
        if instruction:
            focus += f"\nADDITIONAL FOCUS:\n{instruction}\n"
    else:
        if not instruction:
            raise HTTPException(400, "A continuation instruction is required.")
        focus = f"WHAT TO DO NEXT:\n{instruction}\n"

    parent_title = (parent.get("title") or parent.get("intent") or "")[:50]
    _PRIOR_FILE = "prior-report.md"
    intent = (
        f"{_VATRA_HEADERS[kind]}\n\n"
        + obj_block
        + f"The current report is in your workspace as `{_PRIOR_FILE}` — read it first "
        "for full context. Produce the COMPLETE improved deliverable that integrates "
        "the new material into the existing report: keep everything already covered "
        "well, and add or deepen what's asked. Do not output only the new bits.\n\n"
        f"{focus}"
        + _vfs_manifest(owner, vfs_project)
        + _round_filename_rule(vfs_project, round_no)
    )

    # Same cast: pin the parent's owner archetypes; the Lead re-decomposes within them.
    cast_ids: list[str] = []
    if same_cast:
        cast_ids = [st.get("owner") for st in (parent_route.get("subtasks") or [])
                    if st.get("owner")]
        # de-dup, preserve order
        cast_ids = list(dict.fromkeys(cast_ids))

    title = f"{kind.replace('_', ' ').title()}: {parent_title}"[:80]
    _parent_shared_ds = bool(parent_cfg.get("shared_datastore"))
    cfg: dict[str, Any] = {
        "mode": "vatra", "source": "ui", "kind": kind,
        "parent_session_id": parent_session_id, "root_session_id": root_sid,
        "round": round_no, "vfs_project": vfs_project,
        # Inherit the parent round's team size + grouped/parallel/quality knobs so
        # the chain runs the same way (execute_vatra reads these from config/body).
        "max_agents": int(parent_cfg.get("max_agents") or 6),
    }
    if parent_cfg.get("quality"):
        cfg["quality"] = parent_cfg["quality"]
    if parent_cfg.get("execution_groups"):
        cfg["execution_groups"] = True
    if parent_cfg.get("grouped_review"):
        cfg["grouped_review"] = True
    if int(parent_cfg.get("max_parallel") or 0):
        cfg["max_parallel"] = int(parent_cfg["max_parallel"])
    # Keep the whole chain in the parent's project bundle; theme re-derived so
    # edits between rounds are picked up (execute injects it into shared_context).
    _proj_id = parent_cfg.get("project_id") or ""
    if _proj_id:
        from captain_claw.flight_deck.basna_routes import _project_context
        _ptheme, _ = await _project_context(db, owner, _proj_id)
        cfg["project_id"] = _proj_id
        cfg["project_context"] = _ptheme
    # Auto-seed this round with the PARENT run's knowledge (report + gaps/blind
    # spots + datastore); execute_vatra folds it into shared_context for the team.
    cfg["knowledge_session_ids"] = [parent_session_id]
    if _parent_shared_ds:  # keep the chain bound to the folder's shared datastore
        cfg["shared_datastore"] = True
    if cast_ids:
        cfg["force_ids"] = cast_ids
    sess = await db.create_basna_session(owner, intent, title=title, config=json.dumps(cfg))
    sid = sess["id"]
    body_bytes = truth.encode("utf-8")
    (_session_files_dir(sid) / _PRIOR_FILE).write_bytes(body_bytes)
    await db.update_basna_session(sid, owner, files=json.dumps([{
        "name": _PRIOR_FILE, "mime": "text/markdown", "size": len(body_bytes), "kind": "input"}]))
    exec_req = ExecuteRequest(session_id=sid, tiers=tiers or None,
                              env_vars=env_vars or None, api_key=api_key or "",
                              vfs_project=vfs_project, shared_datastore=_parent_shared_ds,
                              # Match the parent round's grouped/parallel/deep/quality run.
                              execution_groups=bool(parent_cfg.get("execution_groups")),
                              grouped_review=bool(parent_cfg.get("grouped_review")),
                              max_parallel=int(parent_cfg.get("max_parallel") or 0),
                              horizon=parent_cfg.get("horizon") or None,
                              quality=parent_cfg.get("quality") or None)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
    # Headless continuation: draft a Group 0 plan then auto-approve (no human pause).
    t = asyncio.create_task(plan_vatra_group0(exec_req, stub, user, gate=False))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": sid, "title": title, "round": round_no, "kind": kind}


async def _fill_gaps_run(owner: str, parent_session_id: str, user: dict, *,
                         tiers: dict | None, env_vars: list | None, api_key: str,
                         instruction: str = "") -> dict:
    """Back-compat alias: fill-gaps = continuation focused on coverage gaps. An
    optional instruction is appended to the gap list as extra focus."""
    return await _continue_run(owner, parent_session_id, user, kind="fill_gaps",
                               instruction=instruction,
                               tiers=tiers, env_vars=env_vars, api_key=api_key)


class VatraFillGapsRequest(BaseModel):
    instruction: str = ""  # optional extra guidance appended to the coverage gaps


@router.post("/sessions/{session_id}/fill-gaps")
async def fill_gaps(session_id: str, body: VatraFillGapsRequest | None = None,
                    user: dict = Depends(get_current_user)):
    """UI 'Fill the gaps' — spawn a follow-up Vatra on this run's coverage gaps,
    seeded with its final report. Returns immediately; the UI polls the new run."""
    tiers, env_vars = await _load_owner_tiers(get_db(), user["id"])
    res = await _fill_gaps_run(user["id"], session_id, user,
                               tiers=tiers, env_vars=env_vars, api_key="",
                               instruction=(body.instruction if body else "").strip())
    return {"ok": True, **res}


class VatraContinueRequest(BaseModel):
    instruction: str = ""
    kind: str = "continue"  # continue | fill_gaps | revise
    same_cast: bool = True


@router.post("/sessions/{session_id}/continue")
async def continue_session(session_id: str, body: VatraContinueRequest,
                           user: dict = Depends(get_current_user)):
    """Carry a finished Vatra run forward into a new round — same folder + report."""
    tiers, env_vars = await _load_owner_tiers(get_db(), user["id"])
    res = await _continue_run(
        user["id"], session_id, user,
        instruction=body.instruction, kind=body.kind, same_cast=body.same_cast,
        tiers=tiers, env_vars=env_vars, api_key="")
    return {"ok": True, **res}


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


@router.get("/sessions/{session_id}/board")
async def list_session_board(session_id: str, user: dict = Depends(get_current_user)):
    """The Vatra shared board for a session — every note/output/narration/file the
    team streamed, for the live shared-memory view."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    return {"entries": await db.list_vatra_board(session_id, limit=200)}


class _VatraSkipReq(BaseModel):
    agent: str = ""


@router.post("/sessions/{session_id}/skip")
async def skip_session_agent(session_id: str, body: _VatraSkipReq,
                             user: dict = Depends(get_current_user)):
    """Ask a still-working agent (by its live-panel label) to skip — the orchestrator
    cancels its current turn and moves on, keeping whatever the rest of the team did."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    agent = (body.agent or "").strip()
    if agent:
        _skip_agents.setdefault(session_id, set()).add(agent)
    return {"ok": True, "agent": agent}


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
    board = await db.list_vatra_board(body.session_id, limit=200)
    return {"session_id": body.session_id, "count": len(asks), "asks": asks,
            "board": [_board_entry(e) for e in board]}


# ── Fire-and-forget entry (mirrors Basna's agent/start) ──────────────

async def _run_and_notify(user: dict, session_id: str, title: str, exec_req,
                          source_host: str, source_port: int, origin: dict) -> None:
    owner = user["id"]
    ok = False
    try:
        stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
        # Headless agent-started run: draft a Group 0 plan then auto-approve (no gate).
        result = await plan_vatra_group0(exec_req, stub, user, gate=False)
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
