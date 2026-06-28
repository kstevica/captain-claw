"""Basna REST endpoints for Flight Deck.

Basna is a network-source ensemble mode, sibling to Council. Where Council runs
a multi-round deliberation among already-running agents, Basna routes a single
task to the *minimal* set of specialist archetypes, spawns them fresh, runs them
in parallel, and merges their outputs into one "truth" — weighting each by its
learned per-domain reliability.

This module is Phase 2: the **router**. It classifies a task (domain, difficulty,
merge_kind) and selects the smallest archetype subset that can answer it, scaling
the count to difficulty. Spawn / dispatch / weighted-merge land in later phases.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import shutil
import time
import types
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.horizon_plan import (
    PlanConfig,
    make_llm_dag_planner,
    make_llm_planner,
    make_llm_step_runner,
    make_llm_synthesizer,
    make_llm_verifier,
    run_dag_horizon,
    run_plan_horizon,
)
from captain_claw.flight_deck.horizon_worker import (
    HorizonConfig,
    run_horizon_closer,
    run_worker_horizon,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/basna", tags=["basna"])

_INSTRUCTIONS_DIR = Path(__file__).parent.parent / "instructions"
_VALID_TIERS = {"reason", "balanced", "fast", "longctx", "coding", "vision"}
_VALID_DIFFICULTY = {"trivial", "moderate", "hard"}
_VALID_MERGE = {"converge", "diverge"}

# Agents allowed per difficulty band — the heart of "scale the team to the task".
_DIFFICULTY_CAP = {"trivial": 1, "moderate": 3, "hard": 6}


# ── Registry & routing helpers (pure, unit-testable) ─────────────────

def _load_registry() -> dict:
    """Load the archetype registry, or raise 500 if it's missing/invalid."""
    registry_file = _INSTRUCTIONS_DIR / "archetypes.json"
    if not registry_file.is_file():
        raise HTTPException(500, "Archetype registry not found")
    try:
        return json.loads(registry_file.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Archetype registry is invalid JSON: {e}")


_MAX_FILE_BYTES = 25 * 1024 * 1024
# Merge-step LLM timeouts (seconds) — a slow/hung provider must not stall the run
# forever; on timeout the merge degrades to the best-weighted answer.
_MERGE_TIMEOUT = 90    # fast-tier conflict check
_SYNTH_TIMEOUT = 300   # reason-tier synthesis / analysis


def _safe_name(name: str) -> str:
    base = Path(name or "upload").name.strip() or "upload"
    return base.replace("/", "_").replace("\\", "_")


def _session_files_dir(session_id: str) -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    d = DATA_DIR / "basna_files" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


# Generated files we can fall back to as an agent's "output" when it answered by
# writing a document and left its chat reply empty (the merge reads text, not bytes).
_TEXT_EXTS = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".html",
              ".htm", ".xml", ".yaml", ".yml", ".rst", ".log", ".tex"}
_MAX_TEXT_FALLBACK_BYTES = 4 * 1024 * 1024  # don't slurp a giant artifact as "output"


def _is_texty(name: str, mime: str) -> bool:
    return mime.startswith("text/") or Path(name).suffix.lower() in _TEXT_EXTS


def _parse_files(sess: dict) -> list[dict]:
    try:
        f = json.loads(sess.get("files") or "[]")
        return f if isinstance(f, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _difficulty_cap(difficulty: str, max_agents: int) -> int:
    """How many agents this difficulty may use, never exceeding the request cap."""
    cap = _DIFFICULTY_CAP.get(difficulty, 3)
    return max(1, min(cap, max(1, max_agents)))


def _score_archetypes(intent: str, archetypes: list[dict]) -> list[tuple[int, dict]]:
    """Rank archetypes by keyword/role overlap with the intent (score > 0 only)."""
    low = intent.lower()
    scored: list[tuple[int, dict]] = []
    for a in archetypes:
        score = sum(1 for kw in a.get("keywords", []) if kw.lower() in low)
        score += sum(1 for w in a.get("role", "").lower().split() if len(w) > 3 and w in low)
        if score > 0:
            scored.append((score, a))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _keyword_match(intent: str, archetypes: list[dict], n: int) -> list[dict]:
    """Deterministic fallback router: top `n` archetypes by overlap with the intent.

    Used when the LLM router is unavailable or returns nothing valid, so a route
    always comes back.
    """
    picked = [a for _s, a in _score_archetypes(intent, archetypes)[:n]]
    if not picked:
        # Nothing matched — fall back to the lead generalist if present, else first.
        lead = next((a for a in archetypes if a.get("lead")), None)
        picked = [lead or archetypes[0]] if archetypes else []
    return picked


def _fallback_difficulty(intent: str, breadth: int) -> str:
    """Guess difficulty for the no-LLM path from match breadth and shape.

    Breadth (how many distinct specialists the intent touches) is a better signal
    than raw length: a short multi-part ask is harder than a long single one.
    """
    low = intent.lower()
    if breadth >= 3 or " and " in low or len(intent) > 200:
        return "hard"
    if breadth <= 1 and len(intent) <= 40:
        return "trivial"
    return "moderate"


def _build_catalog(archetypes: list[dict], reliability: dict[str, list[dict]]) -> str:
    """Render the archetype catalog (with learned-reliability hints) for the prompt."""
    lines = ["## Archetype Catalog", ""]
    for a in archetypes:
        rel = reliability.get(a["id"]) or []
        if rel:
            hint = "; ".join(
                f"{r['domain'] or 'general'}={r['weight']:.2f} ({r['runs']} runs)"
                for r in sorted(rel, key=lambda r: r["weight"], reverse=True)[:3]
            )
            rel_str = f" | reliability: {hint}"
        else:
            rel_str = f" | reliability: seed {a.get('reliability_seed', 0.7):.2f} (no track record)"
        lines.append(
            f"- id: {a['id']} — {a['role']} [{a.get('family', '')}]: {a.get('description', '')} "
            f"(keywords: {', '.join(a.get('keywords', []))}; default tier: {a.get('tier', 'balanced')})"
            f"{rel_str}"
        )
    return "\n".join(lines)


def _normalize_route(
    raw: dict, archetypes_by_id: dict[str, dict], cap_for: callable, max_agents: int,
) -> dict:
    """Validate and clamp an LLM (or fallback) route into the canonical shape."""
    difficulty = str(raw.get("difficulty", "")).lower().strip()
    if difficulty not in _VALID_DIFFICULTY:
        difficulty = "moderate"
    merge_kind = str(raw.get("merge_kind", "")).lower().strip()
    if merge_kind not in _VALID_MERGE:
        merge_kind = "converge"
    domain = str(raw.get("domain", "")).lower().strip() or "general"

    cap = cap_for(difficulty, max_agents)
    selected: list[dict] = []
    seen: set[str] = set()
    for item in raw.get("selected", []) or []:
        aid = str(item.get("archetype_id", "")).strip()
        arch = archetypes_by_id.get(aid)
        if not arch or aid in seen:
            continue
        seen.add(aid)
        tier = str(item.get("tier", "")).strip().lower()
        if tier not in _VALID_TIERS:
            tier = arch.get("tier", "balanced")
        selected.append({
            "archetype_id": aid,
            "role": arch.get("role", ""),
            "tier": tier,
            "why": str(item.get("why", "")).strip(),
        })
        if len(selected) >= cap:
            break

    return {
        "domain": domain,
        "difficulty": difficulty,
        "merge_kind": merge_kind,
        "rationale": str(raw.get("rationale", "")).strip(),
        "selected": selected,
    }


# ── Request models ───────────────────────────────────────────────────

def _title_from_intent(intent: str, max_words: int = 8, max_chars: int = 60) -> str:
    """Cheap fallback title from the task text: first sentence/line, trimmed.

    Used when neither the user nor the LLM router supplies a title (e.g. the
    keyword-fallback route path)."""
    text = " ".join((intent or "").split())
    if not text:
        return "Untitled"
    # Stop at the first sentence boundary if it comes early.
    for sep in (". ", "? ", "! ", "\n"):
        idx = text.find(sep)
        if 0 < idx < max_chars:
            text = text[:idx]
            break
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "…"
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    text = text.strip(" .,:;-—")
    return (text[:1].upper() + text[1:]) if text else "Untitled"


class RouteRequest(BaseModel):
    intent: str
    # Optional user-supplied title; auto-generated from the task when blank.
    title: str = ""
    # LLM creds for the router call. Omit to use the fast tier from the registry;
    # api_key falls back to the provider's env var when empty.
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = Field(default=2048, ge=256, le=8192)
    max_agents: int = Field(default=6, ge=1, le=10)
    # Persist into an existing session; omit to create a fresh one.
    session_id: str = ""
    # User-fixed team: when non-empty, the router MUST use exactly these archetypes
    # (all of them, no others) instead of choosing the team itself.
    archetype_ids: list[str] = []


class CreateSessionRequest(BaseModel):
    intent: str
    title: str = ""
    config: str = "{}"


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    intent: str | None = None
    domain: str | None = None
    difficulty: str | None = None
    merge_kind: str | None = None
    status: str | None = None
    route: str | None = None
    truth: str | None = None
    confidence: float | None = None
    config: str | None = None


# ── Session endpoints ────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.list_basna_sessions(user["id"])


@router.post("/sessions")
async def create_session(body: CreateSessionRequest, user: dict = Depends(get_current_user)):
    if not body.intent.strip():
        raise HTTPException(400, "intent is required")
    db = get_db()
    title = body.title.strip() or _title_from_intent(body.intent)
    return await db.create_basna_session(user["id"], body.intent.strip(), body.config, title=title)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    return sess


@router.put("/sessions/{session_id}")
async def update_session(
    session_id: str, body: UpdateSessionRequest, user: dict = Depends(get_current_user),
):
    db = get_db()
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    ok = await db.update_basna_session(session_id, user["id"], **fields)
    if not ok:
        raise HTTPException(404, "session not found or nothing to update")
    return await db.get_basna_session(session_id, user["id"])


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    ok = await db.delete_basna_session(session_id, user["id"])
    if not ok:
        raise HTTPException(404, "session not found")
    shutil.rmtree(_session_files_dir(session_id), ignore_errors=True)
    return {"deleted": True}


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(session_id: str, user: dict = Depends(get_current_user)):
    """Hard-stop an in-flight run (UI Stop button or the arbiter's stop_run)."""
    sess = await get_db().get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    res = await _cancel_basna_run(session_id, user["id"])
    return {"ok": True, **res}


@router.get("/sessions/{session_id}/runs")
async def list_runs(session_id: str, user: dict = Depends(get_current_user)):
    """Per-agent runs for a session — powers the run-trace UI and feedback thumbs."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    return await db.list_basna_runs(session_id, user["id"])


@router.post("/sessions/{session_id}/files")
async def upload_files(
    session_id: str, files: list[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    """Attach files to a session. Stored on disk + recorded on the session; at
    execute they're copied into every spawned agent's workspace."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    d = _session_files_dir(session_id)
    by_name = {f["name"]: f for f in _parse_files(sess)}
    for uf in files:
        content = await uf.read()
        if len(content) > _MAX_FILE_BYTES:
            raise HTTPException(413, f"{uf.filename} exceeds 25MB")
        name = _safe_name(uf.filename or "upload")
        (d / name).write_bytes(content)
        by_name[name] = {"name": name, "mime": uf.content_type or "application/octet-stream",
                         "size": len(content), "kind": "input"}
    merged = list(by_name.values())
    await db.update_basna_session(session_id, user["id"], files=json.dumps(merged))
    return {"files": merged}


@router.get("/sessions/{session_id}/files/{name}")
async def download_file(session_id: str, name: str, user: dict = Depends(get_current_user)):
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    safe = _safe_name(name)
    path = _session_files_dir(session_id) / safe
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=safe, media_type=_guess_mime(safe))


@router.delete("/sessions/{session_id}/files/{name}")
async def delete_file(session_id: str, name: str, user: dict = Depends(get_current_user)):
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    safe = _safe_name(name)
    try:
        (_session_files_dir(session_id) / safe).unlink()
    except OSError:
        pass
    merged = [f for f in _parse_files(sess) if f["name"] != safe]
    await db.update_basna_session(session_id, user["id"], files=json.dumps(merged))
    return {"files": merged}


# ── Agent-facing internal endpoints ──────────────────────────────────
# These let a spawned agent read its OWNER's Basna data via the `basna` tool.
# They carry no user JWT; instead the caller is identified by its web port (the
# same trust model as the flight_deck peer endpoints) and scoped to that agent's
# owner. The agent passes its own `source_port` plus `owner_id` (from FD_OWNER_ID)
# as a fallback for dev setups where the port isn't in the registry.

class _AgentReq(BaseModel):
    source_port: int = 0
    web_auth: str = ""           # the agent's own auth token (primary identity)
    owner_id: str = ""           # last-resort hint (FD_OWNER_ID env)
    session_id: str = ""
    name: str = ""               # for file fetch
    query: str = ""              # for sessions list
    status: str = ""
    limit: int = Field(default=50, ge=1, le=500)
    archetype_id: str = ""       # for a single run's output


def _resolve_owner(body: _AgentReq) -> str:
    """Authoritative owner for an agent request.

    Tries the agent's unique web_auth token first (most reliable — survives a
    spawn-time port reassignment), then its source port, then the FD_OWNER_ID
    hint it sent.
    """
    from captain_claw.flight_deck.server import (
        _resolve_agent_owner,
        _resolve_agent_owner_by_auth,
    )
    owner = ""
    if body.web_auth:
        owner = _resolve_agent_owner_by_auth(body.web_auth) or ""
    if not owner and body.source_port:
        owner = _resolve_agent_owner(int(body.source_port)) or ""
    owner = owner or (body.owner_id or "").strip()
    if not owner:
        raise HTTPException(403, "could not resolve calling agent's owner")
    return owner


def _session_summary(r: dict) -> dict:
    """Compact session view for the agent `list` action (drops big text blobs)."""
    try:
        sel = (json.loads(r.get("route") or "{}").get("selected") or [])
    except json.JSONDecodeError:
        sel = []
    return {
        "id": r.get("id"), "title": r.get("title", ""), "intent": r.get("intent", ""),
        "domain": r.get("domain", ""), "difficulty": r.get("difficulty", ""),
        "merge_kind": r.get("merge_kind", ""), "status": r.get("status", ""),
        "confidence": r.get("confidence", 0.0),
        "n_agents": len(sel), "n_files": len(_parse_files(r)),
        "created_at": r.get("created_at"), "updated_at": r.get("updated_at"),
    }


@router.post("/agent/sessions")
async def agent_list_sessions(body: _AgentReq):
    """List the calling agent's owner's Basna sessions (filtered, summarized)."""
    owner = _resolve_owner(body)
    db = get_db()
    rows = await db.list_basna_sessions(owner)
    if body.status:
        rows = [r for r in rows if r.get("status") == body.status]
    if body.query:
        q = body.query.lower()
        rows = [r for r in rows
                if q in (f"{r.get('title','')} {r.get('intent','')} {r.get('truth','')}").lower()]
    return {"sessions": [_session_summary(r) for r in rows[:body.limit]]}


@router.post("/agent/session")
async def agent_get_session(body: _AgentReq):
    """Full session detail (route, truth, analysis, files) for the owner."""
    owner = _resolve_owner(body)
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner)
    if not sess:
        raise HTTPException(404, "session not found")
    return sess


@router.post("/agent/runs")
async def agent_list_runs(body: _AgentReq):
    """Per-agent runs (output, tool actions, success, latency) for a session."""
    owner = _resolve_owner(body)
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner)
    if not sess:
        raise HTTPException(404, "session not found")
    return {"runs": await db.list_basna_runs(body.session_id, owner)}


@router.post("/agent/file")
async def agent_get_file(body: _AgentReq):
    """Stream a session's file (generated or input) to the owning agent."""
    owner = _resolve_owner(body)
    db = get_db()
    sess = await db.get_basna_session(body.session_id, owner)
    if not sess:
        raise HTTPException(404, "session not found")
    safe = _safe_name(body.name)
    path = _session_files_dir(body.session_id) / safe
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=safe, media_type=_guess_mime(safe))


# ── Agent-initiated runs (v2): start a Basna autonomously ────────────
# An agent hands a task to FD; FD auto-titles, routes, and executes the ensemble
# server-side (no frontend), then reports completion back to the originating
# agent so it can relay the result on the user's channel. Fire-and-forget.

_MAX_AGENT_RUNS_PER_OWNER = 2          # concurrency cap (confirmed with user)
_active_agent_runs: dict[str, set[str]] = {}   # owner_id → {session_id, …}
_basna_agent_tasks: set = set()        # strong refs so tasks aren't GC'd
_run_workers: dict[str, list[str]] = {}        # session_id → spawned worker slugs
_agent_run_tasks: dict[str, Any] = {}          # session_id → asyncio.Task (agent runs)

# Run-rate circuit breaker: stop a runaway burst of agent-started runs (e.g. a
# recursion that slips past other guards) regardless of LLM judgment.
_MAX_AGENT_RUNS_PER_WINDOW = 6         # max agent-started runs …
_AGENT_RUN_WINDOW_SECONDS = 300.0      # … per owner in this rolling window (5 min)
_agent_run_starts: dict[str, list[float]] = {}  # owner_id → [monotonic start times]


async def _cancel_basna_run(session_id: str, owner: str) -> dict:
    """Hard-stop a Basna run: kill its spawned workers (this tears the run's
    compute down — the in-flight dispatch sees the sockets die and unwinds),
    cancel the background task, free the concurrency slot, mark it cancelled.
    Idempotent; safe for UI-started and agent-started runs alike."""
    stopped = 0
    for slug in list(_run_workers.get(session_id, []) or []):
        try:
            from captain_claw.flight_deck.server import _do_stop_process
            _do_stop_process(slug)
            stopped += 1
        except Exception as exc:
            log.warning("cancel: worker stop failed", slug=slug, error=str(exc))
    t = _agent_run_tasks.get(session_id)
    if t is not None and not t.done():
        t.cancel()
    runs = _active_agent_runs.get(owner)
    if runs:
        runs.discard(session_id)
        if not runs:
            _active_agent_runs.pop(owner, None)
    try:
        await get_db().update_basna_session(session_id, owner, status="cancelled")
    except Exception as exc:
        log.warning("cancel: status update failed", session_id=session_id, error=str(exc))
    log.info("Basna run cancelled", session_id=session_id, stopped_workers=stopped)
    return {"stopped_workers": stopped}


class AgentStartReq(_AgentReq):
    task: str = ""                     # the (possibly rephrased) task to run
    title: str = ""                    # optional; auto-generated when blank
    max_agents: int = Field(default=6, ge=1, le=10)
    # Origin channel, so the completion result reaches the user where they asked.
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0
    # Durable origin ({kind,address}) — the channel-agnostic delivery target used
    # for whatsapp/telegram/glasses/channel completions (see delivery_routes).
    origin_kind: str = ""
    origin_address: str = ""
    source_host: str = "localhost"


async def _load_owner_tiers(db, owner_id: str) -> tuple[dict, list]:
    """Return (tiers_map, env_vars) from the owner's saved Library config.

    Reads the `fd:forge-tiers` setting (a `{sets, activeSetId}` blob — the active
    set's `tiers`/`envVars` are what the UI would use). Falls back to the legacy
    single-set shape, then to ({}, []) so execution uses the registry tiers.
    """
    try:
        settings = await db.get_all_settings(owner_id)
    except Exception:
        return {}, []
    raw = settings.get("fd:forge-tiers")
    if not raw:
        return {}, []
    try:
        blob = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}, []
    sets = blob.get("sets") if isinstance(blob, dict) else None
    if isinstance(sets, list) and sets:
        active_id = blob.get("activeSetId")
        chosen = next((s for s in sets if s.get("id") == active_id), sets[0])
        return chosen.get("tiers") or {}, chosen.get("envVars") or []
    # Legacy single-set shape: {tiers, forgeTier} + separate env-vars key.
    if isinstance(blob, dict) and isinstance(blob.get("tiers"), dict):
        env: list = []
        try:
            ev = json.loads(settings.get("fd:forge-env-vars") or "[]")
            if isinstance(ev, list):
                env = ev
        except (json.JSONDecodeError, TypeError):
            pass
        return blob["tiers"], env
    return {}, []


async def _notify_source_agent(
    *, source_host: str, source_port: int, origin: dict,
    title: str, session_id: str, ok: bool, summary: str,
) -> None:
    """Deliver a Basna completion back to the originating agent (delegate-style).

    Opens a WebSocket to the agent's port (auth resolved server-side) and sends a
    `notification` with `trigger_response`, carrying origin channel info so the
    agent's reply lands where the user asked. Best-effort; logs on failure.
    """
    import websockets

    from captain_claw.flight_deck.server import _resolve_agent_auth
    if not source_port:
        return
    auth = _resolve_agent_auth(int(source_port))
    params = f"?token={auth}" if auth else ""
    url = f"ws://{source_host or 'localhost'}:{source_port}/ws{params}"
    verb = "finished successfully" if ok else "ran into an error"
    callback_msg = (
        f"[Basna run '{title}' {verb}] This is the RESULT of the autonomous Basna "
        f"run you started (session {session_id}). Relay it to the user now, "
        f"concisely, in their language. Do NOT start another Basna and do NOT say "
        f"you are still waiting — you already have the outcome below:\n\n{summary}"
    )
    payload: dict = {"type": "notification", "content": callback_msg, "trigger_response": True}
    if origin.get("platform") and origin["platform"] != "web":
        payload["origin_platform"] = origin["platform"]
        payload["origin_user_id"] = origin.get("user_id", "")
        payload["origin_chat_id"] = origin.get("chat_id", 0)
    try:
        async with websockets.connect(url, open_timeout=10, close_timeout=5) as ws:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if welcome.get("type") != "welcome":
                return
            while True:  # skip the session replay before our message
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "replay_done":
                    break
            await ws.send(json.dumps(payload))
            try:
                await asyncio.wait_for(ws.recv(), timeout=30)
            except TimeoutError:
                pass
    except Exception as exc:
        log.warning("Basna completion delivery failed",
                    session_id=session_id, port=source_port, error=str(exc))


async def _run_and_notify(
    user: dict, session_id: str, title: str, exec_req,
    source_host: str, source_port: int, origin: dict,
) -> None:
    """Background: execute the routed Basna, then notify the originating agent."""
    owner = user["id"]
    ok = False
    try:
        # Pass the FULL user record so plan/quota checks (max agents, spawn rate)
        # see the owner's real plan instead of defaulting to free, plus a stub
        # request whose `.state.user_id` is the owner — `spawn_process` reads it
        # to stamp each spawned agent's owner.
        stub_request = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
        result = await execute_route(exec_req, stub_request, user)
        ok = True
        conf = result.get("confidence", 0.0)
        contributors = result.get("contributors") or []
        truth = (result.get("truth") or "").strip()
        summary = (
            f"Confidence {conf:.0%} · {len(contributors)} agent(s).\n\n"
            f"{truth[:1800]}{'…' if len(truth) > 1800 else ''}"
        )
    except Exception as exc:
        log.warning("Agent-started Basna execution failed", session_id=session_id, error=str(exc))
        summary = f"The run could not complete: {exc}"
    finally:
        runs = _active_agent_runs.get(owner)
        if runs:
            runs.discard(session_id)
            if not runs:
                _active_agent_runs.pop(owner, None)
    # Land the result on the channel the request came from. Channel origins
    # (whatsapp/telegram/glasses/channel) go through FD's durable origin sink —
    # the same one cron/scheduled results use — which guarantees delivery to the
    # right address. Web (and any delivery failure) falls back to relaying through
    # the source agent so it still shows up in the web chat.
    kind = str(origin.get("kind") or "").strip().lower()
    address = str(origin.get("address") or "").strip()
    if kind and kind != "web" and address:
        try:
            from captain_claw.flight_deck.delivery_routes import (
                deliver_to_origin as _deliver_origin,
            )
            delivered, note = await _deliver_origin(
                {"kind": kind, "address": address},
                f"🐻🐰 Basna — {title}\n\n{summary}",
            )
            log.info("Basna result delivered to origin",
                     kind=kind, delivered=delivered, note=note)
            if delivered:
                return
            log.warning("Basna origin delivery not sent; relaying via source agent",
                        kind=kind, note=note)
        except Exception as exc:
            log.warning("Basna origin delivery error; relaying via source agent",
                        error=str(exc))
    await _notify_source_agent(
        source_host=source_host, source_port=source_port, origin=origin,
        title=title, session_id=session_id, ok=ok, summary=summary,
    )


@router.post("/agent/start")
async def agent_start(body: AgentStartReq):
    """Start a Basna run on behalf of the calling agent's owner (fire-and-forget).

    Routes synchronously (fast), then executes the ensemble in a background task
    and reports completion back to the agent. Capped per owner.
    """
    owner = _resolve_owner(body)
    task = (body.task or "").strip()
    if not task:
        raise HTTPException(400, "task is required")

    active = _active_agent_runs.setdefault(owner, set())
    if len(active) >= _MAX_AGENT_RUNS_PER_OWNER:
        return {
            "status": "rejected",
            "reason": f"You already have {len(active)} Basna run(s) in progress "
                      f"(limit {_MAX_AGENT_RUNS_PER_OWNER}). Wait for one to finish.",
        }

    # Run-rate circuit breaker: a deterministic guard against a runaway burst
    # (e.g. a recursion that slips past other checks). Trips regardless of LLM
    # judgment, so it's the reliable floor under the arbiter's stop_run action.
    now_mono = time.monotonic()
    starts = _agent_run_starts.setdefault(owner, [])
    starts[:] = [s for s in starts if now_mono - s < _AGENT_RUN_WINDOW_SECONDS]
    if len(starts) >= _MAX_AGENT_RUNS_PER_WINDOW:
        log.warning("Basna run-rate breaker tripped", owner=owner, recent=len(starts))
        return {
            "status": "rejected",
            "reason": f"Run-rate limit hit ({_MAX_AGENT_RUNS_PER_WINDOW} runs / "
                      f"{int(_AGENT_RUN_WINDOW_SECONDS / 60)} min) — cooling down to "
                      f"prevent a runaway loop.",
        }
    starts.append(now_mono)

    db = get_db()
    # Full user record — carries `metadata.plan` so plan/quota checks during
    # spawn see the owner's real plan, not the free-tier default.
    user = await db.get_user_by_id(owner) or {"id": owner}
    tiers, env_vars = await _load_owner_tiers(db, owner)
    fast = (tiers or {}).get("fast", {})

    # 1) Route synchronously (creates + routes the session, auto-titles).
    route = await route_intent(
        RouteRequest(
            intent=task, title=body.title, max_agents=body.max_agents,
            provider=fast.get("provider", ""), model=fast.get("model", ""),
            api_key=fast.get("api_key", ""), base_url=fast.get("base_url", ""),
        ),
        user,
    )
    session_id = route["session_id"]
    title = route.get("title", "") or task[:60]
    selected = route.get("selected", [])

    # Mark the session as agent-started (+ its origin channel) so the UI can
    # badge it in the unified Basna list.
    await db.update_basna_session(
        session_id, owner,
        config=json.dumps({"source": "agent", "origin_platform": body.origin_platform}),
    )

    # 2) Execute in the background; report completion back to the agent.
    active.add(session_id)
    exec_req = ExecuteRequest(
        session_id=session_id, tiers=tiers or None, env_vars=env_vars or None,
    )
    origin = {
        "platform": body.origin_platform, "user_id": body.origin_user_id,
        "chat_id": body.origin_chat_id,
        "kind": body.origin_kind, "address": body.origin_address,
    }
    t = asyncio.create_task(_run_and_notify(
        user, session_id, title, exec_req, body.source_host, body.source_port, origin,
    ))
    _basna_agent_tasks.add(t)
    _agent_run_tasks[session_id] = t

    def _on_done(_t: Any, _sid: str = session_id) -> None:
        _basna_agent_tasks.discard(_t)
        _agent_run_tasks.pop(_sid, None)

    t.add_done_callback(_on_done)

    return {"status": "running", "session_id": session_id, "title": title,
            "n_agents": len(selected)}


# ── Deepen: a follow-up run that resolves a finished run's blind spots ──

async def _deepen_run(owner: str, parent_session_id: str, user: dict) -> dict:
    """Create + route + execute a follow-up run seeded with a finished run's
    compiled truth and its blind spots, focused on resolving those gaps. Returns
    the new session id. Background execution — the result lands on the new session,
    which the UI polls / the agent can read back with the basna tool."""
    db = get_db()
    parent = await db.get_basna_session(parent_session_id, owner)
    if not parent:
        raise HTTPException(404, "session not found")
    truth = (parent.get("truth") or "").strip()
    try:
        analysis = json.loads(parent.get("analysis") or "{}")
    except (ValueError, TypeError):
        analysis = {}
    blind = [str(b).strip() for b in (analysis.get("blind_spots") or []) if str(b).strip()]
    if not blind:
        raise HTTPException(400, "This run has no blind spots to investigate.")
    if not truth:
        raise HTTPException(400, "This run has no compiled result to build on yet.")

    parent_title = (parent.get("title") or parent.get("intent") or "")[:50]
    # Inline only a preview of the prior synthesis — it rides in every worker's
    # message history (resent each internal step × N workers), so the token cost
    # is a per-step multiplier. When the full text is larger, ship it as a
    # workspace file the workers read on demand instead of re-inlining all of it.
    _DEEPEN_TRUTH_CHARS = 16_000
    _PRIOR_FILE = "prior-synthesis.md"
    big = len(truth) > _DEEPEN_TRUTH_CHARS
    file_note = (
        f"\nThe COMPLETE prior synthesis is in your workspace as `{_PRIOR_FILE}` — "
        "read it for full context before addressing the blind spots.\n"
        if big else ""
    )
    intent = (
        "Continue and deepen a prior multi-agent investigation — focus ONLY on its "
        "blind spots, the aspects no prior answer addressed.\n\n"
        f"PRIOR SYNTHESIS{' (preview — full text in the workspace file)' if big else ''}:\n"
        f"{truth[:_DEEPEN_TRUTH_CHARS]}\n"
        + file_note
        + "\nBLIND SPOTS to resolve:\n"
        + "\n".join(f"- {b}" for b in blind[:12])
        + "\n\nInvestigate and resolve these blind spots and extend the synthesis. "
          "Do not repeat what is already settled above."
    )

    tiers, env_vars = await _load_owner_tiers(db, owner)
    fast = (tiers or {}).get("fast", {})
    route = await route_intent(
        RouteRequest(
            intent=intent, title=f"Deepen: {parent_title}"[:80], max_agents=6,
            provider=fast.get("provider", ""), model=fast.get("model", ""),
            api_key=fast.get("api_key", ""), base_url=fast.get("base_url", ""),
        ),
        user,
    )
    sid = route["session_id"]
    update_kwargs: dict[str, Any] = {
        "config": json.dumps({"kind": "deepen", "parent_session_id": parent_session_id}),
    }
    if big:
        # Write the full synthesis as an input file; execute_route copies every
        # non-generated session file into each worker's workspace.
        body_bytes = truth.encode("utf-8")
        (_session_files_dir(sid) / _PRIOR_FILE).write_bytes(body_bytes)
        update_kwargs["files"] = json.dumps([{
            "name": _PRIOR_FILE, "mime": "text/markdown",
            "size": len(body_bytes), "kind": "input",
        }])
    await db.update_basna_session(sid, owner, **update_kwargs)

    exec_req = ExecuteRequest(session_id=sid, tiers=tiers or None, env_vars=env_vars or None)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
    t = asyncio.create_task(execute_route(exec_req, stub, user))
    _basna_agent_tasks.add(t)
    _agent_run_tasks[sid] = t

    def _on_done(_t: Any, _sid: str = sid) -> None:
        _basna_agent_tasks.discard(_t)
        _agent_run_tasks.pop(_sid, None)

    t.add_done_callback(_on_done)
    return {"session_id": sid, "title": route.get("title", "") or f"Deepen: {parent_title}",
            "n_agents": len(route.get("selected", []))}


@router.post("/sessions/{session_id}/deepen")
async def deepen_session(session_id: str, user: dict = Depends(get_current_user)):
    """UI 'Investigate blind spots' — spawn a follow-up run on this run's gaps."""
    res = await _deepen_run(user["id"], session_id, user)
    return {"ok": True, **res}


@router.post("/agent/deepen")
async def agent_deepen(body: _AgentReq):
    """Agent/tool entry: deepen a finished run by its session id."""
    owner = _resolve_owner(body)
    full_user = await get_db().get_user_by_id(owner) or {"id": owner}
    res = await _deepen_run(owner, body.session_id, full_user)
    return {"status": "running", **res}


# ── Router endpoint ──────────────────────────────────────────────────

@router.post("/route")
async def route_intent(body: RouteRequest, user: dict = Depends(get_current_user)):
    """Classify a task and select the minimal archetype subset to handle it.

    Uses a fast-tier LLM with the keyword/reliability-annotated catalog; on any
    LLM failure it falls back to deterministic keyword matching so a route always
    returns. The result is persisted onto a Basna session (created if needed).
    """
    intent = body.intent.strip()
    if not intent:
        raise HTTPException(400, "intent is required")

    db = get_db()
    archetypes = await merged_archetypes(db, user["id"])
    archetypes_by_id = {a["id"]: a for a in archetypes}
    seeds = {a["id"]: float(a.get("reliability_seed", 0.7)) for a in archetypes}

    # Group this user's learned reliability by archetype for the catalog hints.
    rel_rows = await db.get_archetype_reliability(user["id"])
    reliability: dict[str, list[dict]] = {}
    for r in rel_rows:
        reliability.setdefault(r["archetype_id"], []).append(r)

    # Resolve fast-tier creds for the router call.
    registry = _load_registry()
    tiers = registry.get("tiers", {})
    fast = tiers.get("fast", {})
    provider = body.provider or fast.get("provider", "anthropic")
    model = body.model or fast.get("model", "")
    base_url = body.base_url or fast.get("base_url", "")

    system_prompt_file = _INSTRUCTIONS_DIR / "basna" / "router.md"
    if not system_prompt_file.is_file():
        raise HTTPException(500, "Basna router prompt not found")
    system_prompt = system_prompt_file.read_text() + "\n\n" + _build_catalog(archetypes, reliability)
    forced_ids = [a for a in (body.archetype_ids or []) if a in archetypes_by_id]
    if forced_ids:
        forced_list = "\n".join(
            f"- {a}: {archetypes_by_id[a].get('role', '')}" for a in forced_ids)
        user_prompt = (
            f"Task: {intent}\n\n"
            f"The team is FIXED by the user — you MUST use EXACTLY these archetypes, ALL of "
            f"them and NO others, in `selected`:\n{forced_list}\n\n"
            f"For each, write a `why` that instructs it specifically for THIS task (how it should "
            f"contribute). Still choose domain, difficulty, and merge_kind for the task."
        )
    else:
        user_prompt = (
            f"Task: {intent}\n\n"
            f"max_agents: {body.max_agents}. Select the smallest archetype set that "
            f"handles this task well, scaled to its difficulty."
        )

    started = time.monotonic()
    raw: dict | None = None
    source = "llm"
    try:
        from captain_claw.llm import Message, create_provider
        prov = create_provider(
            provider=provider, model=model,
            api_key=body.api_key or None, base_url=base_url or None,
            temperature=0.2, max_tokens=body.max_tokens,
        )
        resp = await prov.complete(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.2, max_tokens=body.max_tokens,
        )
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(
                l for l in content.split("\n") if not l.strip().startswith("```")
            )
        raw = json.loads(content)
    except Exception as e:
        log.warning("Basna router LLM failed; using keyword fallback", error=str(e))
        raw = None
        source = "fallback"

    if not isinstance(raw, dict) or not raw.get("selected"):
        # Deterministic fallback: keyword match, sized by match-breadth difficulty.
        breadth = len(_score_archetypes(intent, archetypes))
        difficulty = _fallback_difficulty(intent, breadth)
        n = _difficulty_cap(difficulty, body.max_agents)
        picked = _keyword_match(intent, archetypes, n)
        raw = {
            "domain": (picked[0].get("family", "general").split(" ")[0].lower() if picked else "general"),
            "difficulty": difficulty,
            "merge_kind": "converge",
            "rationale": "keyword fallback (LLM router unavailable)",
            "selected": [{"archetype_id": a["id"], "tier": a.get("tier", "balanced")} for a in picked],
        }
        if source != "fallback":
            source = "fallback"

    route = _normalize_route(raw, archetypes_by_id, _difficulty_cap, body.max_agents)

    # Force the user-fixed team: use exactly the chosen archetypes (all of them, no
    # others), keeping the LLM's task-specific `why`/tier where it produced one.
    if forced_ids:
        by_sel = {s["archetype_id"]: s for s in route["selected"]}
        route["selected"] = [
            by_sel.get(aid) or {
                "archetype_id": aid,
                "role": archetypes_by_id[aid].get("role", ""),
                "tier": archetypes_by_id[aid].get("tier", "balanced"),
                "why": "",
            }
            for aid in forced_ids
        ]

    # Attach the current learned weight (for the chosen domain) to each pick — the
    # prior the aggregator and learning loop will start from in later phases.
    for s in route["selected"]:
        s["prior_weight"] = await db.get_archetype_weight(
            user["id"], s["archetype_id"], route["domain"], seeds.get(s["archetype_id"], 0.7),
        )
    route["source"] = source
    route["elapsed_ms"] = int((time.monotonic() - started) * 1000)

    # Resolve a session title: explicit user title > the router LLM's title >
    # a cheap heuristic from the task. Computed once, applied below.
    llm_title = (raw.get("title") or "").strip() if isinstance(raw, dict) else ""
    resolved_title = body.title.strip() or llm_title or _title_from_intent(intent)

    # Persist onto a session (create one if the caller didn't supply an id).
    session_id = body.session_id.strip()
    if session_id:
        sess = await db.get_basna_session(session_id, user["id"])
        if not sess:
            raise HTTPException(404, "session not found")
    else:
        sess = await db.create_basna_session(user["id"], intent, title=resolved_title)
        session_id = sess["id"]
    update_fields = dict(
        domain=route["domain"], difficulty=route["difficulty"],
        merge_kind=route["merge_kind"], route=json.dumps(route), status="routed",
    )
    # Backfill the title on an existing session that has none, or honor an
    # explicit user-supplied title; never clobber a previously set title.
    if body.title.strip() or not (sess.get("title") or "").strip():
        update_fields["title"] = resolved_title
    await db.update_basna_session(session_id, user["id"], **update_fields)

    route["title"] = update_fields.get("title", sess.get("title") or "")
    route["session_id"] = session_id
    return route


# ── Phase 3: spawn → dispatch → weighted merge ───────────────────────

_DONE_STATES = {"ready", "idle", "done", "completed"}

# In-memory execution progress, polled by the UI during /execute. Keyed by
# session_id, overwritten each run. Single-process (the FD server), best-effort.
_PROGRESS: dict[str, dict] = {}
_PROGRESS_MAX_SESSIONS = 50


def _progress_start(session_id: str) -> None:
    if len(_PROGRESS) > _PROGRESS_MAX_SESSIONS:
        _PROGRESS.clear()
    _PROGRESS[session_id] = {"events": [], "active": True}


def _progress(session_id: str, stage: str, message: str, **extra) -> None:
    p = _PROGRESS.get(session_id)
    if p is not None:
        p["events"].append({"i": len(p["events"]), "ts": time.time(),
                            "stage": stage, "message": message, **extra})


def _progress_done(session_id: str) -> None:
    p = _PROGRESS.get(session_id)
    if p is not None:
        p["active"] = False


def _phase(session_id: str, label: str, **extra) -> None:
    """Emit one high-level phase banner (Routing / Generating / Merging / Step x/y …)
    so the live UI can always show which stage of the run is active, separate from
    the noisy per-action detail events."""
    _progress(session_id, "phase", label, **extra)


def _build_dispatch_prompt(role: str, intent: str, merge_kind: str,
                           file_names: list[str] | None = None, extra: str = "") -> str:
    """Frame the task for one ephemeral agent.

    The archetype's role + SOP (fleet_instructions) are delivered separately as
    the agent's fleet-level instructions (system prompt) via the peer_agents
    handshake — see _send_chat_and_collect — so this prompt is just the task plus
    a one-shot framing. Agents run blind (cannot see each other), keeping outputs
    independent for the weighted merge. `extra` is per-agent instructions the user
    added in the route editor.
    """
    role = role or "Specialist"
    if merge_kind == "diverge":
        framing = ("Contribute your distinct perspective. Surface options and angles "
                   "others might miss; do not try to be exhaustive on your own.")
    else:
        framing = ("Give your single best answer. Be decisive and concise — one clear "
                   "position, not a survey of possibilities.")
    files_block = ""
    if file_names:
        listed = "\n".join(f"- {n}" for n in file_names)
        files_block = (
            "\n\n## Attached files (in your working directory)\n"
            f"{listed}\n"
            "Use your read / pdf_extract / xlsx_extract / image_vision tools to work "
            "with them as needed.\n"
        )
    extra_block = f"\n\n## Additional instructions\n{extra.strip()}\n" if extra and extra.strip() else ""
    return (
        f"You are the {role}, working as one independent member of a one-shot ensemble. "
        f"{framing}\n\n## Task\n{intent}{files_block}{extra_block}\n\n"
        "You are working alone and cannot see the other members. Return only your final "
        "answer — no preamble, no meta-commentary about the ensemble."
    )


def _norm_text(s: str) -> set[str]:
    return set(s.lower().split())


def _too_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Jaccard word-overlap test for near-duplicate outputs (diverge dedup)."""
    wa, wb = _norm_text(a), _norm_text(b)
    if not wa or not wb:
        return a.strip() == b.strip()
    jaccard = len(wa & wb) / len(wa | wb)
    return jaccard >= threshold


def _mean_weight(good: list[dict]) -> float:
    return sum(r["weight"] for r in good) / len(good) if good else 0.0


def _merge_diverge(good: list[dict]) -> dict:
    """Weighted dedup + concat: keep all distinct contributions, best-weighted first."""
    ranked = sorted(good, key=lambda r: r["weight"], reverse=True)
    kept: list[dict] = []
    for r in ranked:
        if any(_too_similar(r["output"], k["output"]) for k in kept):
            continue
        kept.append(r)
    parts = [f"### {r['role']} (weight {r['weight']:.2f})\n{r['output'].strip()}" for r in kept]
    return {
        "truth": "\n\n".join(parts),
        "confidence": round(min(0.99, _mean_weight(kept)), 3),
        "method": "weighted_dedup",
        "contributors": [r["archetype_id"] for r in kept],
    }


def _usable(r: dict) -> bool:
    """A run contributes to the merge if it produced real content — either its
    chat reply (`ok`) or, when it answered by writing a file, the captured
    artifact text backfilled into `output` (`produced_file`). Without this, an
    agent that delivers a document and ends its turn with an empty reply is
    silently dropped and the merge compiles from nothing."""
    return bool((r.get("ok") or r.get("produced_file"))
                and (r.get("output") or "").strip())


async def _aggregate(
    results: list[dict], merge_kind: str, domain: str, *,
    conflict_fn, synth_fn,
) -> dict:
    """Compile the truth from agent outputs.

    converge: 1 output → take it; many → if they agree, take the highest-weighted
    (Trask's weighted combination); only if they genuinely disagree do we pay for
    an LLM synthesizer to reconcile. diverge: weighted dedup of all contributions.
    `conflict_fn(good) -> bool` and `synth_fn(good) -> str` are injected so the
    merge logic is testable without live models.
    """
    good = [r for r in results if _usable(r)]
    if not good:
        return {"truth": "", "confidence": 0.0, "method": "empty", "contributors": []}

    if merge_kind == "diverge":
        return _merge_diverge(good)

    if len(good) == 1:
        r = good[0]
        return {"truth": r["output"].strip(), "confidence": round(r["weight"], 3),
                "method": "single", "contributors": [r["archetype_id"]]}

    mean_w = _mean_weight(good)
    try:
        agree = await conflict_fn(good)
    except Exception as e:
        log.warning("Basna conflict check failed; taking best-weighted", error=str(e))
        agree = True  # degrade: skip synthesis, take the best-weighted answer
    if not agree:
        try:
            merged = await synth_fn(good)
            return {"truth": merged.strip(),
                    "confidence": round(max(0.05, mean_w - 0.2), 3),
                    "method": "synthesis",
                    "contributors": [r["archetype_id"] for r in good]}
        except Exception as e:
            log.warning("Basna synthesis failed; falling back to best-weighted", error=str(e))
    # Agreement, or a failed/timed-out conflict/synthesis → always return a truth.
    best = max(good, key=lambda r: r["weight"])
    return {"truth": best["output"].strip(),
            "confidence": round(min(0.99, mean_w + 0.1), 3),
            "method": "weighted_pick",
            "contributors": [r["archetype_id"] for r in good]}


def _tier_creds(registry: dict, tier: str, api_key: str) -> dict:
    t = (registry.get("tiers") or {}).get(tier, {})
    return {"provider": t.get("provider", "anthropic"), "model": t.get("model", ""),
            "base_url": t.get("base_url", "") or None, "api_key": api_key or None,
            "output_ctx": int(t.get("output_ctx") or 0)}


def _resolve_merge_creds(body, registry: dict, tier: str) -> dict:
    """LLM creds for a tier: Library config first, registry tier defaults as fallback.

    Shared by the merge/judge closures and the Horizon critic provider (which must
    run on a model *different* from the worker — never the worker grading itself).
    """
    lt = (body.tiers or {}).get(tier)
    if lt and lt.get("model"):
        return {"provider": lt.get("provider", "anthropic"), "model": lt.get("model", ""),
                "base_url": lt.get("base_url") or None,
                "api_key": lt.get("api_key") or body.api_key or None,
                "output_ctx": int(lt.get("output_ctx") or 0)}
    return _tier_creds(registry, tier, body.api_key)


def _provider_call(creds: dict, *, temperature: float, default_max: int, cap: int) -> tuple:
    """Build (create_provider, max_tokens) from merge creds — honoring the tier's
    output_ctx so a long synthesis isn't truncated by a hardcoded cap. Pops the
    non-provider `output_ctx` key before constructing the provider."""
    from captain_claw.llm import create_provider
    c = {k: v for k, v in creds.items() if k != "output_ctx"}
    out = int(creds.get("output_ctx") or 0)
    max_tokens = min(max(out or default_max, default_max), cap)
    return create_provider(temperature=temperature, max_tokens=max_tokens, **c), max_tokens


async def _llm_conflict(good: list[dict], creds: dict) -> bool:
    """Fast-tier check: do these answers substantively agree? Default to disagree."""
    from captain_claw.llm import Message
    listing = "\n\n".join(f"[{i+1}] {r['output'].strip()[:2000]}" for i, r in enumerate(good))
    prov, mt = _provider_call(creds, temperature=0.0, default_max=256, cap=512)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "You compare independent answers to the same task. Reply ONLY with JSON "
            '{"agree": true} if they reach substantively the same conclusion, or '
            '{"agree": false} if they materially disagree on the answer.')),
        Message(role="user", content=listing),
    ], temperature=0.0, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    try:
        return bool(json.loads(content).get("agree", False))
    except (json.JSONDecodeError, AttributeError):
        return False


async def _llm_synthesize(good: list[dict], domain: str, creds: dict) -> str:
    """Reason-tier reconciliation of disagreeing answers, trusting weight."""
    from captain_claw.llm import Message
    listing = "\n\n".join(
        f"### {r['role']} (reliability weight {r['weight']:.2f})\n{r['output'].strip()}"
        for r in good
    )
    prov, mt = _provider_call(creds, temperature=0.3, default_max=8192, cap=32768)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            f"Independent specialists gave conflicting answers in the {domain} domain. "
            "Reconcile them into one correct, complete answer — do not truncate or "
            "abbreviate; if the inputs are long documents, produce the full merged "
            "document. Weigh higher-reliability contributors more, but follow the "
            "evidence over the weight when a lower-weighted contributor is clearly "
            "right. State the resolved answer directly; do not narrate the disagreement.")),
        Message(role="user", content=listing),
    ], temperature=0.3, max_tokens=mt)
    return resp.content.strip()


async def _llm_analysis(good: list[dict], domain: str, creds: dict) -> dict | None:
    """Cross-agent analysis: where the independent answers agree, where they
    differ (attributed), what each uniquely contributed, and what NONE addressed.

    Returns a dict with keys agreement / differences / unique / blind_spots, or
    None if the model output can't be parsed.
    """
    from captain_claw.llm import Message
    listing = "\n\n".join(
        f"### {r['role']}\n{r['output'].strip()[:3500]}" for r in good
    )
    prov, mt = _provider_call(creds, temperature=0.2, default_max=4096, cap=8192)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            f"You are comparing {len(good)} independent expert answers to the same "
            f"task ({domain}). Produce a rigorous comparison. Attribute points to "
            "contributors by their role name. Reply ONLY with JSON of this shape:\n"
            '{"agreement": ["points all/most converge on"],\n'
            ' "differences": [{"point": "what they disagree about", '
            '"positions": [{"by": "Role", "stance": "their position"}]}],\n'
            ' "unique": [{"by": "Role", "insight": "something only this one raised"}],\n'
            ' "blind_spots": ["important aspects NONE of the answers addressed"]}\n'
            "Be specific and concise. Blind spots are the most valuable part — "
            "think about what a careful reviewer would notice is missing.")),
        Message(role="user", content=listing),
    ], temperature=0.2, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    # Tolerate prose wrapped around the JSON object.
    if not content.startswith("{"):
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e > s:
            content = content[s:e + 1]
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    # Normalize key variants so a model that says "key_differences" or
    # "blind spots" still populates the panel.
    def _arr(*keys: str) -> list:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, list):
                return v
        return []

    norm = {
        "agreement": _arr("agreement", "agreements", "consensus", "agreed"),
        "differences": _arr("differences", "key_differences", "keyDifferences", "disagreements"),
        "unique": _arr("unique", "unique_insights", "uniqueInsights"),
        "blind_spots": _arr("blind_spots", "blindspots", "blind spots", "blindSpots", "gaps", "missing"),
    }
    return norm if any(norm.values()) else None


async def _llm_judge(good: list[dict], truth: str, creds: dict) -> list[bool]:
    """Fast-tier per-contribution verdict: did each support the final truth?

    Returns one bool per `good` entry, in order. Raises on unparseable output so
    the caller can leave those runs unscored rather than reward them by default.
    """
    from captain_claw.llm import Message
    listing = "\n\n".join(f"[{i+1}] {r['output'].strip()[:2000]}" for i, r in enumerate(good))
    prov, mt = _provider_call(creds, temperature=0.0, default_max=512, cap=1024)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "Given a FINAL answer and several independent contributions, decide for "
            "each contribution whether it substantively agrees with or supports the "
            "final answer. Reply ONLY with a JSON array of booleans — one per "
            "contribution, in order, same length as the input.")),
        Message(role="user", content=f"FINAL ANSWER:\n{truth[:4000]}\n\nCONTRIBUTIONS:\n{listing}"),
    ], temperature=0.0, max_tokens=mt)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    arr = json.loads(content)
    if not isinstance(arr, list):
        raise ValueError("judge did not return a list")
    out = [bool(x) for x in arr][:len(good)]
    while len(out) < len(good):  # under-length → assume the rest supported the truth
        out.append(True)
    return out


async def _score_runs(results: list[dict], agg: dict, merge_kind: str, *, judge_fn) -> dict:
    """Decide success/fail per archetype against the compiled truth.

    Agents that produced nothing fail. converge: a single survivor is correct;
    many are judged against the truth. diverge: contributors that survived dedup
    succeeded, redundant duplicates did not. If the judge errors, the good runs
    are left unscored (omitted) rather than guessed.
    """
    scores: dict[str, bool] = {}
    good: list[dict] = []
    for r in results:
        if _usable(r):
            good.append(r)
        else:
            scores[r["archetype_id"]] = False
    if not agg.get("truth") or not good:
        return scores
    if merge_kind == "diverge":
        kept = set(agg.get("contributors") or [])
        for r in good:
            scores[r["archetype_id"]] = r["archetype_id"] in kept
        return scores
    if len(good) == 1:
        scores[good[0]["archetype_id"]] = True
        return scores
    try:
        verdicts = await judge_fn(good, agg["truth"])
    except Exception as e:
        log.warning("Basna scoring judge failed; leaving runs unscored", error=str(e))
        return scores
    for r, v in zip(good, verdicts):
        scores[r["archetype_id"]] = bool(v)
    return scores


def _summarize_tool_args(args) -> str:
    """Concise one-line summary of a tool call's arguments for the action log."""
    if isinstance(args, dict) and args:
        return ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:2])[:120]
    if args:
        return str(args)[:120]
    return ""


async def _send_chat_and_collect(
    port: int, token: str, prompt: str, timeout: float, on_action=None,
    fleet_instructions: str = "", agent_name: str = "",
    file_paths: list[str] | None = None, image_paths: list[str] | None = None,
    on_usage=None,
) -> tuple[str, list[dict]]:
    """Connect to an agent's /ws, send one chat, return (final reply, actions).

    `actions` is the agent's tool calls (the `monitor` events Council also shows),
    each {tool, detail}. `on_action(act)` is invoked live as each one arrives.
    `on_usage(prompt, completion, total)` is invoked live on each `turn_usage`
    broadcast — the agent's running cumulative token counts for the current turn,
    so the UI can show LLM usage as it climbs instead of only at the end.
    `fleet_instructions` are delivered via the peer_agents handshake so they land
    in the agent's system prompt (same path the UI uses), not just the message.
    """
    import websockets
    uri = f"ws://localhost:{port}/ws" + (f"?token={token}" if token else "")
    # State persists across reconnects: the task is sent exactly once, and a
    # dropped long-poll socket resumes listening rather than restarting the turn.
    answer = ""
    actions: list[dict] = []
    sent = False
    deadline: float | None = None
    last_err: Exception | None = None

    def _record(kind: str, detail: str) -> None:
        act = {"tool": kind, "detail": detail}
        actions.append(act)
        if on_action:
            try:
                on_action(act)
            except Exception:
                pass

    def _handle(msg: dict) -> bool:
        """Apply one inbound message; return True when the turn is complete."""
        nonlocal answer
        mtype = msg.get("type")
        if mtype == "chat_message" and msg.get("role") == "assistant":
            if msg.get("content"):
                answer = msg["content"]  # keep the latest full reply
        elif mtype == "replay_batch":
            # On reconnect the committed history arrives bundled here; recover the
            # latest assistant reply so a turn that finished while we were
            # disconnected isn't lost. Replayed tool calls are skipped (counted
            # live already) to avoid double-recording.
            for m in (msg.get("messages") or []):
                if (m.get("type") == "chat_message" and m.get("role") == "assistant"
                        and m.get("content")):
                    answer = m["content"]
        elif mtype == "monitor" and not msg.get("replay"):
            _record(str(msg.get("tool_name") or msg.get("tool") or "tool"),
                    _summarize_tool_args(msg.get("arguments")))
        elif mtype == "narration" and str(msg.get("text") or "").strip():
            _record("narration", str(msg["text"]).strip()[:280])
        elif mtype == "turn_usage" and not msg.get("replay"):
            # Live, running cumulative token counts emitted after each internal
            # LLM call within the turn. Surfaced live (not persisted as an action)
            # so the UI shows usage climbing instead of one number at the end.
            if on_usage:
                try:
                    on_usage(int(msg.get("prompt_tokens", 0) or 0),
                             int(msg.get("completion_tokens", 0) or 0),
                             int(msg.get("total_tokens", 0) or 0))
                except Exception:
                    pass
        elif mtype == "usage" and not msg.get("replay"):
            # End-of-turn LLM summary — model + final token counts (recorded once
            # as an action so it's preserved in the run's persisted activity).
            u = msg.get("last") or msg.get("usage") or msg
            model = u.get("model") or msg.get("model") or ""
            it = u.get("input_tokens") or u.get("prompt_tokens")
            ot = u.get("output_tokens") or u.get("completion_tokens")
            tok = f"{it}→{ot} tok" if (it is not None or ot is not None) else ""
            detail = " · ".join(x for x in [str(model), tok] if x)
            if detail:
                _record("llm", detail)
        elif mtype == "status" and str(msg.get("status", "")).lower() in _DONE_STATES:
            # End-of-turn only once the turn actually ran — a fresh agent's boot
            # "ready" carries no actions or answer and must not end us early. An
            # agent that delivered via a file write ends with no answer but real
            # actions, so `actions` is the signal there.
            return bool(actions or answer.strip())
        elif mtype == "replay_done":
            # Reconnected after the turn committed: the replay already carried the
            # final reply, so we're done.
            return bool(answer.strip())
        elif mtype == "error":
            m = str(msg.get("message", "") or "")
            # "busy" means the agent is still finishing OUR turn on the original
            # connection — not fatal. Keep listening; results are broadcast to
            # every client, including this reconnected one.
            if "busy" in m.lower():
                return False
            raise RuntimeError(m or "agent error")
        return False

    # The retry loop covers two cases: the agent's web server may still be booting
    # (initial connect), and the long-poll socket may drop mid-turn. We send the
    # task only on the first connection; reconnects just re-attach and listen, so
    # we never re-trigger the agent's busy guard or discard collected work.
    for attempt in range(10):
        try:
            async with websockets.connect(
                uri, max_size=8 * 1024 * 1024, open_timeout=10,
                ping_interval=20, ping_timeout=30,
            ) as ws:
                await asyncio.wait_for(ws.recv(), timeout=15)  # welcome
                if not sent:
                    # Drain any pre-existing session replay BEFORE sending our task,
                    # so committed history (e.g. a prior turn's reply on a REUSED
                    # agent — autonomous nudges run through the user's live agent)
                    # is never mistaken for this turn's answer. The agent always
                    # emits replay_done after the optional replay_batch, so this is
                    # instant for a fresh Basna agent too. (On a mid-turn reconnect
                    # `sent` is already True, so we skip this and let _handle's
                    # replay recovery rebuild our answer — the intended path there.)
                    try:
                        while True:
                            rd = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                            if rd.get("type") == "replay_done":
                                break
                    except TimeoutError:
                        pass
                    answer = ""  # discard anything observed during the replay
                    if fleet_instructions:
                        # Fleet-level instructions (archetype role + SOP) into the
                        # agent's system prompt before the task turn.
                        await ws.send(json.dumps({
                            "type": "peer_agents", "agents": [],
                            "self": {"name": agent_name or "agent",
                                     "fleet_instructions": fleet_instructions},
                        }))
                    chat_msg: dict = {"type": "chat", "content": prompt}
                    if file_paths:
                        chat_msg["file_paths"] = file_paths
                    if image_paths:
                        chat_msg["image_paths"] = image_paths
                    await ws.send(json.dumps(chat_msg))
                    sent = True
                    deadline = asyncio.get_event_loop().time() + timeout

                while True:
                    rem = deadline - asyncio.get_event_loop().time()
                    if rem <= 0:
                        return answer.strip(), actions  # overall budget spent
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(rem, 30))
                    except TimeoutError:
                        # A quiet socket is not a dead one: the agent may be deep in
                        # an LLM synthesis with nothing to stream. Keep waiting until
                        # the overall deadline rather than tearing the turn down and
                        # re-sending it (which the agent rejects as "busy").
                        continue
                    if _handle(json.loads(raw)):
                        return answer.strip(), actions
        except (ConnectionRefusedError, OSError, websockets.ConnectionClosed) as e:
            # TimeoutError (an OSError subclass) and ConnectionClosed land here for a
            # genuinely dead socket. If we already have the answer, or the budget is
            # spent, stop; otherwise reconnect and resume listening — without
            # re-sending the task.
            last_err = e
            if sent and answer.strip():
                return answer.strip(), actions
            if deadline is not None and asyncio.get_event_loop().time() >= deadline:
                return answer.strip(), actions
            await asyncio.sleep(0.5 * (attempt + 1))
    if sent and (answer.strip() or actions):
        return answer.strip(), actions
    raise RuntimeError(f"could not reach agent on port {port}: {last_err}")


async def _dispatch_one(port: int, token: str, prompt: str, timeout: float, on_action=None,
                        fleet_instructions: str = "", agent_name: str = "",
                        file_paths: list[str] | None = None,
                        image_paths: list[str] | None = None, on_usage=None) -> dict:
    started = time.monotonic()
    try:
        out, actions = await _send_chat_and_collect(
            port, token, prompt, timeout, on_action=on_action,
            fleet_instructions=fleet_instructions, agent_name=agent_name,
            file_paths=file_paths, image_paths=image_paths, on_usage=on_usage)
        return {"ok": True, "output": out, "actions": actions,
                "latency_ms": int((time.monotonic() - started) * 1000)}
    except Exception as e:
        log.warning("Basna dispatch failed", error=str(e))
        return {"ok": False, "output": "", "actions": [], "error": str(e),
                "latency_ms": int((time.monotonic() - started) * 1000)}


class ExecuteRequest(BaseModel):
    session_id: str
    # Per-tier model config from the Library: tier -> {provider, model, api_key,
    # base_url, input_ctx, output_ctx}. Spawned agents and the merge calls resolve
    # their model/key from here by tier; missing entries fall back to the registry
    # tier defaults + the provider env var.
    tiers: dict | None = None
    # Additional env vars / API keys passed to every spawned agent (the Library's
    # "Additional API Keys" — e.g. BRAVE_API_KEY for web search). [{key, value}].
    env_vars: list[dict] | None = None
    # Fallback key when a tier omits one (empty -> provider env var).
    api_key: str = ""
    agent_max_tokens: int = Field(default=8192, ge=512, le=32768)
    dispatch_timeout: float = Field(default=600.0, ge=10.0, le=3600.0)
    # Deep / Horizon mode (opt-in): when set, each worker is driven through the
    # Frontier-Horizon engine (N-sample self-consistency vote + diverse-lens critics
    # + fix loop) instead of a single one-shot dispatch. Keys: samples, fix_attempts,
    # critics[], stakes, agreement_threshold, critic_tier, compute_budget. None → off.
    horizon: dict | None = None


async def _spawn_horizon_member(
    arch: dict, sel: dict, body, request, user, *,
    sid8: str, run_tag: str, input_files: list[dict], src_dir: Path, name_suffix: str,
) -> tuple[int, str, str]:
    """Spawn one Basna-correct ephemeral worker for a Horizon pool → (port, token, slug).

    Same config as the normal Basna spawn — basna tool stripped (no recursion),
    worker env marker, shared VFS project, input files materialized — but returns
    the live (port, token, slug) the engine needs and raises on an unusable spawn.
    """
    from captain_claw.flight_deck.server import (
        DATA_DIR,
        AgentConfig,
        _load_process_registry,
        spawn_process,
    )
    lt = (body.tiers or {}).get(sel["tier"]) or {}
    provider = sel.get("provider") or lt.get("provider")
    model = sel.get("model") or lt.get("model")
    api_key = sel.get("api_key") or lt.get("api_key") or body.api_key or ""
    base_url = sel.get("base_url") or lt.get("base_url") or ""
    max_tokens = int(sel.get("max_tokens") or lt.get("output_ctx") or 0) or 32768
    max_context = int(sel.get("max_context") or lt.get("input_ctx") or 0)
    worker_tools = [t for t in (arch.get("tools") or AgentConfig().tools) if t != "basna"]
    base = dict(
        name=f"basna-{sid8}-{run_tag}-{arch['id']}-{name_suffix}",
        description=f"Basna horizon · {sel.get('role') or arch.get('role', '')}",
        cognitive_mode=sel.get("cognitive_mode") or arch.get("cognitive_mode", "neutra"),
        tools=worker_tools,
        env_vars=(body.env_vars or []) + [
            {"key": "CLAW_BASNA_WORKER", "value": "1"},
            {"key": "CLAW_VFS_PROJECT", "value": f"basna-{sid8}"},
            {"key": "CLAW_AGENT_LABEL",
             "value": sel.get("role") or arch.get("role") or arch["id"]},
        ],
        web_enabled=True, web_port=0,
    )
    if model:
        cfg = AgentConfig(
            **base, tier="", provider=provider or "", model=model,
            provider_api_key=api_key, base_url=base_url,
            max_tokens=max_tokens, max_context=max_context,
        )
    else:
        cfg = AgentConfig(**base, tier=sel["tier"], provider_api_key=api_key)
    res = await spawn_process(cfg, request, user)
    entry = _load_process_registry().get(res.slug) or {}
    port = entry.get("web_port")
    if not res.ok or not port:
        raise RuntimeError(res.message or "spawn failed: no port")
    if input_files:
        ws = DATA_DIR / res.slug / "data" / "workspace"
        for f in input_files:
            try:
                shutil.copy2(src_dir / f["name"], ws / f["name"])
            except OSError as e:
                log.warning("Basna horizon file copy failed", file=f["name"], error=str(e))
    return int(port), entry.get("web_auth", ""), res.slug


async def _dispatch_horizon_workers(
    plan: list[tuple[dict, dict]], sess: dict, merge_kind: str, hcfg: HorizonConfig,
    body, request, user, sid: str, sid8: str, run_tag: str, input_files: list[dict],
    *, critic_provider,
) -> list[dict]:
    """Dispatch each routed worker through the Horizon engine instead of one-shot.

    Each worker self-manages a pool of ``hcfg.samples`` fresh archetype instances
    (independent rollouts for self-consistency); pools are torn down centrally after
    the gather to avoid concurrent process-registry writes. Returns Basna-shaped
    result dicts so merge/learning downstream are untouched.
    """
    from captain_claw.flight_deck.server import (
        DATA_DIR,
        _do_stop_process,
        _load_process_registry,
        _processes,
        _save_process_registry,
    )
    src_dir = _session_files_dir(sid)
    _run_workers.setdefault(sid, [])

    def _register(slugs: list[str]) -> None:
        _run_workers.setdefault(sid, []).extend(slugs)

    async def _noop_stop(_slug: str) -> None:
        return None  # central teardown below — keeps registry writes serial

    async def _one(sel: dict, arch: dict) -> dict:
        role = sel.get("role") or arch.get("role") or arch["id"]
        fleet = sel.get("fleet_instructions") or arch.get("fleet_instructions", "")
        tier = sel["tier"]
        prompt = _build_dispatch_prompt(
            role, sess["intent"], merge_kind,
            [f["name"] for f in input_files], extra=sel.get("extra", ""))

        def _on_action(act: dict) -> None:
            s = act.get("sample")
            tag = f"{role}[s{s}]" if s is not None else role
            if act.get("tool") == "narration":
                _progress(sid, "narration", f"{tag}: {act.get('detail', '')}",
                          agent=role, tool="narration", detail=act.get("detail", ""), sample=s)
            else:
                detail = f": {act['detail']}" if act.get("detail") else ""
                _progress(sid, "action", f"{tag} → {act.get('tool')}{detail}",
                          agent=role, tool=act.get("tool"), detail=act.get("detail", ""), sample=s)

        def _on_usage(pt: int, ct: int, tt: int) -> None:
            _progress(sid, "usage", f"{role} · {pt:,}→{ct:,} tok",
                      agent=role, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

        def _on_event(e: dict) -> None:
            conf = float(e.get("confidence", 0.0) or 0.0)
            mark = "✓ passed" if e.get("passed") else "✗ retry"
            _progress(sid, "attempt",
                      f"{role} · {e.get('kind', '?')} · {e.get('samples', 0)} sample(s) "
                      f"→ conf {conf:.2f} · {mark}",
                      agent=role, kind=e.get("kind"), samples=e.get("samples"),
                      passed=bool(e.get("passed")), confidence=conf)

        async def _spawn_member(name_suffix: str) -> tuple[int, str, str]:
            return await _spawn_horizon_member(
                arch, sel, body, request, user, sid8=sid8, run_tag=run_tag,
                input_files=input_files, src_dir=src_dir, name_suffix=name_suffix)

        res = await run_worker_horizon(
            spawn=_spawn_member, tier=tier, prompt=prompt, cfg=hcfg,
            critic_provider=critic_provider, fleet_instructions=fleet, agent_name=role,
            on_action=_on_action, on_usage=_on_usage, on_event=_on_event,
            on_spawn=_register, stop=_noop_stop, timeout=body.dispatch_timeout,
        )
        mark = "✓" if res["ok"] else "✗"
        err = "" if res.get("ok") else f" — {str(res.get('error', ''))[:160]}"
        _progress(sid, "dispatch",
                  f"{role} {mark} · rung {res.get('rung_reached', 0)} · "
                  f"{res.get('samples_used', 0)} sample(s) · "
                  f"conf {res.get('confidence', 0.0):.0%}{err}",
                  ok=res["ok"], agent=role)
        return {
            "archetype_id": arch["id"],
            "role": role,
            "tier": tier, "provider": "", "model": "",
            "weight": float(sel.get("prior_weight", 0.7)),
            "output": res["output"], "ok": res["ok"],
            "latency_ms": res["latency_ms"], "actions": res.get("actions", []),
        }

    _progress(sid, "spawn",
              f"Deep mode (Horizon) · {len(plan)} worker(s) × {hcfg.samples} sample(s) "
              f"· critics: {', '.join(hcfg.critics) or 'none'}")
    out = await asyncio.gather(*[_one(sel, arch) for sel, arch in plan], return_exceptions=True)

    # Central teardown — serial to avoid concurrent process-registry writes.
    all_slugs = list(_run_workers.get(sid, []))
    for slug in all_slugs:
        try:
            _do_stop_process(slug)
        except Exception as e:  # noqa: BLE001 — best-effort
            log.warning("Basna horizon stop failed", slug=slug, error=str(e))
    if all_slugs:
        try:
            reg = _load_process_registry()
            for slug in all_slugs:
                reg.pop(slug, None)
                _processes.pop(slug, None)
                shutil.rmtree(DATA_DIR / slug, ignore_errors=True)
            _save_process_registry(reg)
        except Exception as e:  # noqa: BLE001 — best-effort
            log.warning("Basna horizon cleanup failed", error=str(e))

    results: list[dict] = []
    for item in out:
        if isinstance(item, Exception):
            log.warning("Basna horizon worker crashed", error=str(item))
            continue
        results.append(item)
    return results


@router.post("/execute")
async def execute_route(
    body: ExecuteRequest, request: Request, user: dict = Depends(get_current_user),
):
    """Spawn the routed archetypes, dispatch in parallel, merge into one truth.

    Agents are spawned fresh, run blind and in parallel, then torn down. Their
    outputs are merged weighted by each archetype's prior reliability; an LLM
    synthesizer is invoked only when converge outputs genuinely disagree.
    """
    db = get_db()
    sess = await db.get_basna_session(body.session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    try:
        route = json.loads(sess.get("route") or "{}")
    except json.JSONDecodeError:
        route = {}
    selected = route.get("selected") or []
    if not selected:
        raise HTTPException(400, "session has no route; call /fd/basna/route first")

    archetypes = await merged_archetypes(db, user["id"])
    arch_by_id = {a["id"]: a for a in archetypes}
    seeds = {a["id"]: float(a.get("reliability_seed", 0.7)) for a in archetypes}
    # Registry tiers — the fallback for merge-step creds when no Library tier is
    # supplied (used by the `_merge_creds` closure below).
    registry = _load_registry()
    domain = route.get("domain", "general")
    merge_kind = route.get("merge_kind", "converge")
    session_files = _parse_files(sess)
    # Only user-supplied inputs are fed to agents; prior generated artifacts are
    # kept on the session but not re-fed (avoids feeding outputs back as inputs).
    input_files = [f for f in session_files if f.get("kind") != "generated"]
    input_names = {f["name"] for f in input_files}
    generated_files: list[dict] = []

    # Lazy import to avoid a circular import (server imports this module).
    from captain_claw.flight_deck.server import (
        DATA_DIR,
        AgentConfig,
        _do_stop_process,
        _load_process_registry,
        _processes,
        _save_process_registry,
        spawn_process,
    )

    await db.update_basna_session(body.session_id, user["id"], status="running")

    sid = body.session_id
    _progress_start(sid)
    sid8 = sid[:8]
    run_tag = format(int(time.time()), "x")[-6:]  # unique per run → no slug collisions on re-run
    plan = [(s, arch_by_id[s["archetype_id"]]) for s in selected if s["archetype_id"] in arch_by_id]
    _phase(sid, "Routing")
    _progress(sid, "route", f"Selected {len(plan)} archetype(s) · {domain} / {merge_kind}")

    # Deep / Horizon mode (opt-in): drive each worker through the Frontier-Horizon
    # engine. Critics must run on a model *different* from the worker (never self-
    # judge), so resolve a separate Library-tier critic provider here.
    _hraw = body.horizon if body.horizon is not None else route.get("horizon")
    horizon_cfg = HorizonConfig.from_dict(_hraw) if _hraw else None
    critic_provider = None
    if horizon_cfg is not None:
        try:
            cc = _resolve_merge_creds(body, registry, horizon_cfg.critic_tier)
            if cc.get("model"):
                critic_provider, _ = _provider_call(
                    cc, temperature=0.7, default_max=1200, cap=2048)
        except Exception as e:  # noqa: BLE001 — no critic model → agreement-only
            log.warning("Basna horizon critic provider unavailable", error=str(e))

    spawned: list[dict] = []  # {sel, arch, slug, port, auth}
    results: list[dict] = []
    try:
        if horizon_cfg is not None and horizon_cfg.worker:
            results = await _dispatch_horizon_workers(
                plan, sess, merge_kind, horizon_cfg, body, request, user,
                sid, sid8, run_tag, input_files, critic_provider=critic_provider)
        else:
            _phase(sid, "Spawning")
            _progress(sid, "spawn", f"Spawning {len(plan)} agent(s)…")
            # 1) Spawn the selected archetypes (spawn_process serializes internally).
            # Resolve each archetype's tier to a concrete model from the Library config
            # when provided; otherwise let the backend resolve the registry tier.
            async def _spawn(sel: dict, arch: dict):
                # Per-agent overrides from the route editor take precedence over the
                # Library tier, which takes precedence over the registry tier default.
                lt = (body.tiers or {}).get(sel["tier"]) or {}
                provider = sel.get("provider") or lt.get("provider")
                model = sel.get("model") or lt.get("model")
                api_key = sel.get("api_key") or lt.get("api_key") or body.api_key or ""
                base_url = sel.get("base_url") or lt.get("base_url") or ""
                max_tokens = int(sel.get("max_tokens") or lt.get("output_ctx") or 0) or 32768
                max_context = int(sel.get("max_context") or lt.get("input_ctx") or 0)
                # No recursion: a Basna worker must never be able to start another
                # Basna. Strip the `basna` tool from the spawn (this also disables the
                # deterministic relay, which is gated on has_tool("basna")) and stamp
                # an env marker the tool double-checks.
                _worker_tools = [
                    t for t in (arch.get("tools") or AgentConfig().tools) if t != "basna"
                ]
                base = dict(
                    name=f"basna-{sid8}-{run_tag}-{arch['id']}",
                    description=f"Basna ephemeral · {sel.get('role') or arch.get('role', '')}",
                    cognitive_mode=sel.get("cognitive_mode") or arch.get("cognitive_mode", "neutra"),
                    tools=_worker_tools,
                    env_vars=(body.env_vars or []) + [
                        {"key": "CLAW_BASNA_WORKER", "value": "1"},
                        # Shared VFS project for all workers in this Basna session.
                        {"key": "CLAW_VFS_PROJECT", "value": f"basna-{sid8}"},
                        # Authorship label for files this worker writes to the VFS.
                        {"key": "CLAW_AGENT_LABEL",
                         "value": sel.get("role") or arch.get("role") or arch["id"]},
                    ],
                    web_enabled=True, web_port=0,
                )
                if model:
                    cfg = AgentConfig(
                        **base, tier="", provider=provider or "", model=model,
                        provider_api_key=api_key, base_url=base_url,
                        max_tokens=max_tokens, max_context=max_context,
                    )
                else:
                    cfg = AgentConfig(**base, tier=sel["tier"], provider_api_key=api_key)
                res = await spawn_process(cfg, request, user)
                # Materialize input files into this agent's workspace.
                if input_files and res.ok:
                    ws = DATA_DIR / res.slug / "data" / "workspace"
                    src = _session_files_dir(body.session_id)
                    for f in input_files:
                        try:
                            shutil.copy2(src / f["name"], ws / f["name"])
                        except OSError as e:
                            log.warning("Basna file copy failed", file=f["name"], error=str(e))
                return sel, arch, res

            spawn_out = await asyncio.gather(
                *[_spawn(sel, arch) for sel, arch in plan], return_exceptions=True,
            )
            proc_reg = _load_process_registry()
            for item in spawn_out:
                if isinstance(item, Exception):
                    log.warning("Basna spawn failed", error=str(item))
                    _progress(sid, "spawn", f"spawn failed: {str(item)[:200]}", ok=False)
                    continue
                sel, arch, res = item
                entry = proc_reg.get(res.slug) or {}
                port = entry.get("web_port")
                if not res.ok or not port:
                    log.warning("Basna spawn unusable", slug=res.slug, ok=res.ok, message=res.message)
                    _progress(sid, "spawn",
                              f"{arch.get('role') or arch['id']}: unusable — {res.message or 'no port'}", ok=False)
                    continue
                spawned.append({"sel": sel, "arch": arch, "slug": res.slug,
                                "port": port, "auth": entry.get("web_auth", "")})
            _progress(sid, "spawn", f"Spawned {len(spawned)}/{len(plan)}; dispatching…")
            # Track workers so a Stop/stop_run can hard-kill this run mid-flight.
            _run_workers[body.session_id] = [sp["slug"] for sp in spawned]

            # 2) Dispatch the task to each agent in parallel; log each tool call live
            # and each agent's completion as it returns.
            async def _dispatch_tracked(sp: dict) -> dict:
                sel, arch = sp["sel"], sp["arch"]
                role = sel.get("role") or arch.get("role") or arch["id"]
                fleet = sel.get("fleet_instructions") or arch.get("fleet_instructions", "")

                # Tag each event with structured fields (agent / tool / detail) so the
                # UI can group the streaming log into live per-agent panels — not just
                # parse the flat message string.
                def _on_action(act: dict) -> None:
                    if act["tool"] == "narration":
                        _progress(sid, "narration", f"{role}: {act['detail']}",
                                  agent=role, tool="narration", detail=act.get("detail", ""))
                    else:
                        detail = f": {act['detail']}" if act.get("detail") else ""
                        _progress(sid, "action", f"{role} → {act['tool']}{detail}",
                                  agent=role, tool=act["tool"], detail=act.get("detail", ""))

                def _on_usage(pt: int, ct: int, tt: int) -> None:
                    _progress(sid, "usage", f"{role} · {pt:,}→{ct:,} tok",
                              agent=role, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

                ws = DATA_DIR / sp["slug"] / "data" / "workspace"
                img_paths = [str(ws / f["name"]) for f in input_files
                             if str(f.get("mime", "")).startswith("image/")]
                doc_paths = [str(ws / f["name"]) for f in input_files
                             if not str(f.get("mime", "")).startswith("image/")]
                d = await _dispatch_one(
                    sp["port"], sp["auth"],
                    _build_dispatch_prompt(role, sess["intent"], merge_kind,
                                           [f["name"] for f in input_files], extra=sel.get("extra", "")),
                    body.dispatch_timeout, on_action=_on_action,
                    fleet_instructions=fleet, agent_name=role,
                    file_paths=doc_paths, image_paths=img_paths, on_usage=_on_usage,
                )
                mark = "✓" if d["ok"] else "✗"
                extra = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
                _progress(sid, "dispatch",
                          f"{role} {mark} · {len(d['actions'])} action(s) ({d['latency_ms'] / 1000:.1f}s){extra}",
                          ok=d["ok"], agent=role)
                return d

            _phase(sid, "Generating")
            _progress(sid, "generate", f"{len(spawned)} agent(s) working the task…")
            dispatched = await asyncio.gather(*[_dispatch_tracked(sp) for sp in spawned])
            for sp, d in zip(spawned, dispatched):
                results.append({
                    "archetype_id": sp["arch"]["id"],
                    "role": sp["sel"].get("role") or sp["arch"].get("role", ""),
                    "tier": sp["sel"]["tier"], "provider": "", "model": "",
                    "weight": float(sp["sel"].get("prior_weight", 0.7)),
                    "output": d["output"], "ok": d["ok"], "latency_ms": d["latency_ms"],
                    "actions": d.get("actions", []),
                })
    finally:
        # 3a) Capture any files the agents generated, BEFORE teardown deletes their
        # workspaces — so generated content is preserved on the session.
        dest_dir = _session_files_dir(body.session_id)
        seen_gen: set[str] = set()
        gen_text_by_idx: dict[int, list[str]] = {}  # spawn index → captured text
        for i, sp in enumerate(spawned):
            ws = DATA_DIR / sp["slug"] / "data" / "workspace"
            if not ws.is_dir():
                continue
            # The agent (role) that produced these files — shown in the UI list.
            agent_role = sp["sel"].get("role") or sp["arch"].get("role") or sp["arch"]["id"]
            for p in sorted(ws.rglob("*")):
                if not p.is_file() or p.name in input_names:
                    continue
                # Disambiguate collisions across agents by prefixing the archetype.
                out_name = p.name
                if out_name in seen_gen:
                    out_name = f"{sp['arch']['id']}__{p.name}"
                seen_gen.add(out_name)
                try:
                    shutil.copy2(p, dest_dir / out_name)
                    generated_files.append({"name": out_name, "mime": _guess_mime(out_name),
                                            "size": p.stat().st_size, "kind": "generated",
                                            "agent": agent_role})
                except OSError as e:
                    log.warning("Basna generated-file capture failed", file=p.name, error=str(e))
                    continue
                # Keep this agent's textual artifacts around as a merge fallback.
                if _is_texty(out_name, _guess_mime(out_name)) and p.stat().st_size <= _MAX_TEXT_FALLBACK_BYTES:
                    try:
                        gen_text_by_idx.setdefault(i, []).append(p.read_text(errors="replace"))
                    except OSError:
                        pass
        if generated_files:
            _progress(sid, "files", f"Captured {len(generated_files)} generated file(s)")

        # When an agent answered by writing a document and left its chat reply
        # empty, fall back to that document so the merge has something to compile.
        # `results` is index-aligned with `spawned`.
        for i, r in enumerate(results):
            if (r.get("output") or "").strip():
                continue
            texts = gen_text_by_idx.get(i)
            if texts:
                r["output"] = "\n\n".join(t.strip() for t in texts if t.strip()).strip()
                r["produced_file"] = True

        # 3b) Always remove the ephemeral agents — fully, not just "stopped", so
        # they don't pile up in the fleet. Their outputs/actions live in basna_runs.
        for sp in spawned:
            try:
                _do_stop_process(sp["slug"])
            except Exception as e:
                log.warning("Basna teardown stop failed", slug=sp["slug"], error=str(e))
        if spawned:
            reg = _load_process_registry()  # reload after the stops above persisted
            for sp in spawned:
                reg.pop(sp["slug"], None)
                _processes.pop(sp["slug"], None)
                try:
                    shutil.rmtree(DATA_DIR / sp["slug"], ignore_errors=True)
                except Exception:
                    pass
            _save_process_registry(reg)
        _run_workers.pop(body.session_id, None)

    # 4) Persist one run per agent (success scored below, once the truth is known).
    run_ids: list[int] = []
    if results:
        run_ids = await db.add_basna_runs(body.session_id, user["id"], [{
            "archetype_id": r["archetype_id"], "role": r["role"], "tier": r["tier"],
            "weight_at_run": r["weight"], "output": r["output"],
            "actions": json.dumps(r.get("actions", [])),
            "latency_ms": r["latency_ms"], "success": None,
        } for r in results])

    # Resolve LLM creds for a tier from the Library config, falling back to the
    # registry tier defaults + env key.
    def _merge_creds(tier: str) -> dict:
        return _resolve_merge_creds(body, registry, tier)

    # 5) Compile the truth (weighted; LLM synthesis only on genuine conflict).
    _phase(sid, "Merging")
    _progress(sid, "merge", "Compiling the truth…")
    agg = await _aggregate(
        results, merge_kind, domain,
        conflict_fn=lambda good: asyncio.wait_for(_llm_conflict(good, _merge_creds("fast")), _MERGE_TIMEOUT),
        synth_fn=lambda good: asyncio.wait_for(_llm_synthesize(good, domain, _merge_creds("reason")), _SYNTH_TIMEOUT),
    )
    _progress(sid, "merge", f"Merged via {agg['method']} · confidence {agg['confidence']:.0%}")

    # 5b) Cross-agent analysis — agreement, attributed differences, unique
    # insights, and blind spots none of the agents covered. Only when ≥2 agents
    # actually contributed (a single answer has nothing to compare against).
    analysis: dict | None = None
    good_results = [r for r in results if _usable(r)]
    if len(good_results) >= 2:
        _progress(sid, "merge", "Analyzing agreement & blind spots…")
        try:
            analysis = await asyncio.wait_for(
                _llm_analysis(good_results, domain, _merge_creds("reason")), _SYNTH_TIMEOUT)
        except Exception as e:
            log.warning("Basna analysis failed", error=str(e))
        if analysis:
            _progress(sid, "merge", "Analysis: {} agreement · {} differences · {} blind spots".format(
                len(analysis.get("agreement") or []), len(analysis.get("differences") or []),
                len(analysis.get("blind_spots") or [])))
        else:
            _progress(sid, "merge", "Analysis: none produced", ok=False)

    # 5c) Horizon closer (Lever B, opt-in): adversarially verify the merged truth and
    # revise once if a diverse-lens critic panel refutes it — the back-edge Basna
    # otherwise lacks. Critics run on a separate Library-tier model (never self-judge).
    if horizon_cfg is not None and horizon_cfg.close and (agg.get("truth") or "").strip():
        _phase(sid, "Verifying")
        _progress(sid, "merge", "Horizon closer: verifying the answer…")
        try:
            cc = _merge_creds(horizon_cfg.critic_tier)
            cp = rp = None
            if cc.get("model"):
                cp, _ = _provider_call(cc, temperature=0.7, default_max=1200, cap=2048)
                rp, _ = _provider_call(cc, temperature=0.3, default_max=8192, cap=32768)
            closed = await run_horizon_closer(
                question=sess["intent"], answer=agg["truth"],
                critic_provider=cp, revise_provider=rp, critics=horizon_cfg.critics,
                on_event=_closer_on_event(sid, "Closer", "merge"),
            )
            if closed["revised"]:
                agg["truth"] = closed["answer"]
                agg["method"] = f"{agg.get('method', 'merge')}+revised"
                _progress(sid, "merge", "Closer revised the answer")
            else:
                _progress(sid, "merge",
                          f"Closer: answer held ({closed['survived']}/{closed['total']})")
        except Exception as e:  # noqa: BLE001 — closer is best-effort
            log.warning("Basna horizon closer failed", error=str(e))

    # 6) Close the learning loop: score each run against the truth and fold the
    # outcome into per-archetype reliability, so the next route's prior_weight
    # reflects what actually worked. This is what makes Basna improve over time.
    _progress(sid, "learn", "Scoring contributions…")
    scores = await _score_runs(
        results, agg, merge_kind,
        judge_fn=lambda good, truth: _llm_judge(good, truth, _merge_creds("fast")),
    )
    learned: list[dict] = []
    for r, rid in zip(results, run_ids):
        succ = scores.get(r["archetype_id"])
        if succ is None:  # judge couldn't decide — don't guess
            continue
        await db.score_basna_run(rid, user["id"], succ)
        rel = await db.record_archetype_outcome(
            user["id"], r["archetype_id"], domain, succ, seeds.get(r["archetype_id"], 0.7),
        )
        learned.append({"archetype_id": r["archetype_id"], "run_id": rid,
                        "success": succ, "weight": rel["weight"]})

    files_by_name = {f["name"]: f for f in session_files}
    for g in generated_files:
        files_by_name[g["name"]] = g
    await db.update_basna_session(
        body.session_id, user["id"], status="done",
        truth=agg["truth"], confidence=agg["confidence"],
        files=json.dumps(list(files_by_name.values())),
        analysis=json.dumps(analysis or {}),
    )
    _phase(sid, "Done")
    _progress(sid, "done", f"Done · {len(results)} agent(s), {len(learned)} learned")
    _progress_done(sid)
    # Persist the progress log so reopening the session shows it.
    await db.update_basna_session(
        sid, user["id"], progress=json.dumps((_PROGRESS.get(sid) or {}).get("events", [])),
    )

    return {
        "session_id": body.session_id, "domain": domain, "merge_kind": merge_kind,
        "truth": agg["truth"], "confidence": agg["confidence"],
        "method": agg["method"], "contributors": agg["contributors"],
        "analysis": analysis,
        "agents": [{"archetype_id": r["archetype_id"], "role": r["role"],
                    "ok": r["ok"], "latency_ms": r["latency_ms"], "weight": r["weight"],
                    "actions": r.get("actions", []),
                    "run_id": run_ids[i] if i < len(run_ids) else None,
                    "success": scores.get(r["archetype_id"])} for i, r in enumerate(results)],
        "learned": learned,
        "spawned": len(spawned), "dispatched": len(results),
    }


# ── Plan-Horizon (Lever C): verify-gated multi-step run ───────────────

class PlanRequest(BaseModel):
    intent: str
    title: str = ""
    tiers: dict | None = None
    env_vars: list[dict] | None = None
    api_key: str = ""
    max_steps: int = Field(default=5, ge=1, le=12)
    max_fix_per_step: int = Field(default=1, ge=0, le=3)
    max_replans: int = Field(default=1, ge=0, le=3)
    min_step_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # The Library tier the planner / step / verifier / synthesizer run on.
    plan_tier: str = "reason"
    # How each step is executed: "llm" = one plan-tier generation (fast); "ensemble"
    # = a full Basna ensemble (route+execute) per step (stronger, much costlier).
    step_mode: str = "llm"
    step_max_agents: int = Field(default=4, ge=1, le=10)
    dispatch_timeout: float = Field(default=600.0, ge=10.0, le=3600.0)
    # User-fixed team: each step's Basna ensemble / Vatra team is staffed from these
    # archetypes. Empty → each step routes its own team freely. (No effect on "llm".)
    archetype_ids: list[str] = Field(default_factory=list)
    # When true, the planner emits a dependency DAG and independent steps run in
    # parallel waves (each step sees only its dependencies' outputs). Else a linear
    # chain with re-plan on failure.
    dag: bool = False


async def _mirror_progress(parent_sid: str, child_sid: str) -> None:
    """Forward a plan-step's child run's live agent activity into the parent plan log.

    A plan step (Basna ensemble / Vatra team) runs as a child session that streams
    into its own ``_PROGRESS[child_sid]``; this tails those events and re-emits the
    meaningful ones (agent narration, tool calls, completions, merges, and token
    usage) under the parent, so the plan log shows the agents working — and the
    per-agent cards show running token usage — not just a step summary. Fully
    exception-safe: it can never break the step it's mirroring.
    """
    seen = 0
    mirror_stages = {"narration", "action", "dispatch", "merge", "usage"}

    def drain() -> None:
        nonlocal seen
        try:
            p = _PROGRESS.get(child_sid)
            if not p:
                return
            evs = p.get("events", [])
            while seen < len(evs):
                e = evs[seen]
                seen += 1
                if e.get("stage") not in mirror_stages:
                    continue
                extra = {"agent": e.get("agent"), "tool": e.get("tool")}
                if e.get("stage") == "usage":  # carry the token counts the cards read
                    extra["prompt_tokens"] = e.get("prompt_tokens")
                    extra["completion_tokens"] = e.get("completion_tokens")
                    extra["total_tokens"] = e.get("total_tokens")
                _progress(parent_sid, e.get("stage", "action"),
                          f"↳ {str(e.get('message', ''))[:280]}",
                          **{k: v for k, v in extra.items() if v is not None})
        except Exception:  # noqa: BLE001 — mirroring is best-effort, never fatal
            pass

    try:
        while True:
            drain()
            await asyncio.sleep(0.8)
    except asyncio.CancelledError:
        drain()  # final catch-up before we stop
        raise


def make_basna_ensemble_step_runner(
    parent_sid: str, body: "PlanRequest", user: dict, *,
    route_fn=None, execute_fn=None, on_step=None,
):
    """A plan-horizon ``step_runner`` where each step is a full Basna ensemble.

    Per step: route+execute a child Basna session on the step goal (with prior
    verified results as context), tag it as a plan-step child of ``parent_sid``, and
    return the ensemble's merged truth. ``route_fn``/``execute_fn`` are injectable so
    the runner unit-tests without spawning agents. Child sessions are real Basna runs
    — their archetype-reliability learning still closes — just hidden from the list.
    """
    route_fn = route_fn or route_intent
    execute_fn = execute_fn or execute_route
    ft = (body.tiers or {}).get("fast") or {}

    async def step_runner(goal: str, context: str) -> str:
        step_intent = f"{context}\n\n# Your step now\n{goal}"
        rr = RouteRequest(
            intent=step_intent, max_agents=body.step_max_agents,
            provider=ft.get("provider", ""), model=ft.get("model", ""),
            api_key=ft.get("api_key", "") or body.api_key, base_url=ft.get("base_url", ""),
            archetype_ids=list(body.archetype_ids or []))  # staff each step from the fixed team
        routed = await route_fn(rr, user)
        child_sid = (routed or {}).get("session_id")
        if not child_sid:
            return ""
        try:  # tag the child as a plan-step (best-effort; hidden from the session list)
            await get_db().update_basna_session(
                child_sid, user["id"],
                config=json.dumps({"mode": "basna", "source": "plan-step", "parent": parent_sid}))
        except Exception:  # noqa: BLE001 — tagging is best-effort
            pass
        er = ExecuteRequest(
            session_id=child_sid, tiers=body.tiers, env_vars=body.env_vars,
            api_key=body.api_key, dispatch_timeout=body.dispatch_timeout)
        stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
        mirror = asyncio.create_task(_mirror_progress(parent_sid, child_sid))
        try:
            result = await execute_fn(er, stub, user)
        finally:
            mirror.cancel()
            try:
                await mirror
            except asyncio.CancelledError:
                pass
        if on_step is not None:
            on_step(child_sid, result or {})
        return ((result or {}).get("truth") or "").strip()

    return step_runner


def make_vatra_team_step_runner(
    parent_sid: str, body: "PlanRequest", user: dict, *,
    create_session_fn=None, execute_fn=None, on_step=None,
):
    """A plan-horizon ``step_runner`` where each step is a full **Vatra team**.

    Per step: create a child Vatra session on the step goal (with prior verified
    results as context), run ``execute_vatra`` (the Lead decomposes, specialists
    collaborate on the blackboard, the reporter assembles), and return the assembled
    deliverable. ``create_session_fn``/``execute_fn`` are injectable for tests.
    ``execute_vatra`` is imported lazily — vatra_routes imports this module.
    """
    async def _default_create(intent: str, config_json: str, title: str):
        return await get_db().create_basna_session(
            user["id"], intent, config_json, title=title)

    async def _default_execute(er, request, u):
        from captain_claw.flight_deck.vatra_routes import execute_vatra
        return await execute_vatra(er, request, u)

    create_session_fn = create_session_fn or _default_create
    execute_fn = execute_fn or _default_execute

    async def step_runner(goal: str, context: str) -> str:
        step_intent = f"{context}\n\n# Your step now\n{goal}"
        config_json = json.dumps({
            "mode": "vatra", "source": "plan-step", "parent": parent_sid,
            "max_agents": body.step_max_agents,
            **({"force_ids": list(body.archetype_ids)} if body.archetype_ids else {})})
        sess = await create_session_fn(step_intent, config_json, goal[:60])
        child_sid = (sess or {}).get("id")
        if not child_sid:
            return ""
        er = ExecuteRequest(
            session_id=child_sid, tiers=body.tiers, env_vars=body.env_vars,
            api_key=body.api_key, dispatch_timeout=body.dispatch_timeout)
        stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
        mirror = asyncio.create_task(_mirror_progress(parent_sid, child_sid))
        try:
            result = await execute_fn(er, stub, user)
        finally:
            mirror.cancel()
            try:
                await mirror
            except asyncio.CancelledError:
                pass
        if on_step is not None:
            on_step(child_sid, result or {})
        return ((result or {}).get("truth") or "").strip()

    return step_runner


def _closer_on_event(sid: str, label: str, stage_name: str = "merge"):
    """Map Horizon-closer events onto the live log so the (slow) critic panel shows
    incremental progress — one line as each diverse-lens critic returns, then the
    verdict, then the revise pass — instead of a single long-running line."""
    def on_event(e: dict) -> None:
        st = e.get("stage")
        if st == "verify_start":
            _progress(sid, stage_name, f"{label}: verifying with {e.get('total')} critic(s)…",
                      agent=label)
        elif st == "critic":
            mark = "refuted" if e.get("refuted") else "held"
            _progress(sid, stage_name,
                      f"{label}: critic {int(e.get('index', 0)) + 1}/{e.get('total')} "
                      f"({e.get('mode', '')}) {mark}", agent=label)
        elif st == "verify":
            _progress(sid, stage_name, f"{label}: {e.get('survived')}/{e.get('total')} critics held",
                      agent=label)
        elif st == "heartbeat":
            ph = "revising" if e.get("phase") == "revise" else "verifying"
            _progress(sid, stage_name, f"{label}: still {ph}… ({e.get('elapsed')}s)", agent=label)
        elif st == "revise":
            _progress(sid, stage_name, f"{label}: revising…", agent=label)
        elif st == "revise_rejected":
            _progress(sid, stage_name, f"{label}: kept original (revision was unusable)",
                      agent=label, ok=False)
    return on_event


def _plan_on_event(sid: str):
    """Map plan-horizon events onto the shared Basna live-progress log."""
    # Captured across events so a step banner can read "Step x/y" (the total is
    # only known once the planner has emitted its plan).
    total = {"n": 0}

    def _label(e: dict) -> str:
        # DAG events carry a step id; linear events carry a 0-based index.
        return str(e.get("id")) if e.get("id") is not None else str(int(e.get("index", 0)) + 1)

    def on_event(e: dict) -> None:
        stage = e.get("stage")
        lbl = _label(e)
        if stage == "plan":
            goals = e.get("goals") or []
            total["n"] = len(goals)
            _phase(sid, f"Planning · {len(goals)} step(s)")
            _progress(sid, "route", f"Planned {len(goals)} step(s)")
            for j, g in enumerate(goals):
                _progress(sid, "route", f"  {j + 1}. {str(g)[:140]}")
        elif stage == "step_start":
            suffix = f"/{total['n']}" if total["n"] else ""
            goal = str(e.get("goal", ""))[:80]
            _phase(sid, f"Step {lbl}{suffix}" + (f": {goal}" if goal else ""))
            _progress(sid, "dispatch", f"Step {lbl}: {str(e.get('goal', ''))[:140]}",
                      agent="plan", ok=True)
        elif stage == "verify":
            # ``confidence`` is the verifier's certainty in its verdict (not the
            # output's quality) — spell out accepted/rejected so a high number next
            # to a retry doesn't read as a contradiction.
            conf = float(e.get("confidence", 0) or 0)
            verdict = (f"accepted ✓ (verifier {conf:.0%} sure)" if e.get("passed")
                       else f"rejected ✗ (verifier {conf:.0%} sure) — retry")
            _progress(sid, "attempt", f"Step {lbl} · verify try {e.get('attempt')} → {verdict}")
        elif stage == "step_done":
            v = "verified" if e.get("verified") else "unverified"
            _progress(sid, "dispatch", f"Step {lbl} {v} · conf {float(e.get('confidence', 0) or 0):.0%}",
                      agent="plan", ok=bool(e.get("verified")))
        elif stage == "replan":
            _progress(sid, "merge", f"Re-planning (#{e.get('replans')}) after step {lbl} failed", ok=False)
        elif stage == "synthesize":
            _phase(sid, "Synthesizing")
            _progress(sid, "merge", f"Synthesizing the deliverable from {e.get('steps', 0)} step(s)…")
    return on_event


async def _run_plan(sid: str, body: PlanRequest, user: dict) -> None:
    """Background plan-horizon run: decompose → verify-gated steps → synthesize."""
    db = get_db()
    registry = _load_registry()
    _progress_start(sid)
    try:
        await db.update_basna_session(sid, user["id"], status="running")
        creds = _resolve_merge_creds(body, registry, body.plan_tier)
        if not creds.get("model"):
            raise RuntimeError(f"plan tier {body.plan_tier!r} has no model configured")
        plan_p, _ = _provider_call(creds, temperature=0.3, default_max=2048, cap=4096)
        ver_p, _ = _provider_call(creds, temperature=0.0, default_max=1024, cap=2048)
        synth_p, _ = _provider_call(creds, temperature=0.3, default_max=8192, cap=32768)
        cfg = PlanConfig(
            max_steps=body.max_steps, max_fix_per_step=body.max_fix_per_step,
            max_replans=body.max_replans, min_step_confidence=body.min_step_confidence)
        # Step execution: a full Basna ensemble, a Vatra team, or a lean generation.
        if body.step_mode == "ensemble":
            def _on_child(csid: str, r: dict) -> None:
                _progress(sid, "dispatch",
                          f"  ↳ ensemble · {len((r or {}).get('agents') or [])} agent(s) "
                          f"· merged conf {float((r or {}).get('confidence', 0) or 0):.0%}",
                          agent="plan")
            step_runner = make_basna_ensemble_step_runner(sid, body, user, on_step=_on_child)
        elif body.step_mode == "vatra":
            def _on_team(csid: str, r: dict) -> None:
                _progress(sid, "dispatch",
                          f"  ↳ team · {len((r or {}).get('subtasks') or [])} subtask(s) "
                          f"· conf {float((r or {}).get('confidence', 0) or 0):.0%}",
                          agent="plan")
            step_runner = make_vatra_team_step_runner(sid, body, user, on_step=_on_team)
        else:
            step_p, _ = _provider_call(creds, temperature=0.5, default_max=8192, cap=32768)
            step_runner = make_llm_step_runner(step_p)
        if body.dag:
            res = await run_dag_horizon(
                body.intent.strip(),
                planner_dag=make_llm_dag_planner(plan_p, max_steps=body.max_steps),
                step_runner=step_runner,
                verifier=make_llm_verifier(ver_p),
                synthesizer=make_llm_synthesizer(synth_p),
                cfg=cfg, on_event=_plan_on_event(sid))
        else:
            res = await run_plan_horizon(
                body.intent.strip(),
                planner=make_llm_planner(plan_p, max_steps=body.max_steps),
                step_runner=step_runner,
                verifier=make_llm_verifier(ver_p),
                synthesizer=make_llm_synthesizer(synth_p),
                cfg=cfg, on_event=_plan_on_event(sid))
        confidence = round(res.completed / max(1, len(res.steps)), 3)
        analysis = {
            "kind": "plan",
            "steps": [{"goal": s["goal"], "verified": s["verified"],
                       "confidence": s["confidence"], "attempts": s["attempts"]}
                      for s in res.steps],
            "replans": res.replans, "stopped_reason": res.stopped_reason,
        }
        await db.update_basna_session(
            sid, user["id"], status="done", truth=res.deliverable,
            confidence=confidence, analysis=json.dumps(analysis))
        note = f" ({res.stopped_reason})" if res.stopped_reason else ""
        _phase(sid, "Done")
        _progress(sid, "done",
                  f"Done · {res.completed}/{len(res.steps)} step(s) verified, "
                  f"{res.replans} re-plan(s){note}")
    except Exception as e:  # noqa: BLE001 — background task: record, don't crash the loop
        log.exception("Basna plan run failed", sid=sid)
        _progress(sid, "error", str(e)[:300])
        await db.update_basna_session(sid, user["id"], status="error")
    finally:
        _progress_done(sid)
        await db.update_basna_session(
            sid, user["id"], progress=json.dumps((_PROGRESS.get(sid) or {}).get("events", [])))


@router.post("/plan")
async def plan_route(body: PlanRequest, request: Request, user: dict = Depends(get_current_user)):
    """Run a verify-gated multi-step Plan-Horizon (Lever C) in the background.

    The planner decomposes the intent into ordered steps; each step is driven to a
    verified result (with a fix loop) before the next; a hard step failure triggers a
    bounded re-plan; the deliverable is synthesized from the verified steps. Returns
    immediately with the session id — the UI polls progress like any other run.
    """
    intent = (body.intent or "").strip()
    if not intent:
        raise HTTPException(400, "intent is required")
    db = get_db()
    title = (body.title or intent[:60]).strip()
    sess = await db.create_basna_session(
        user["id"], intent,
        json.dumps({"mode": "plan", "source": "ui", "max_steps": body.max_steps,
                    "max_replans": body.max_replans, "step_mode": body.step_mode,
                    "dag": body.dag,
                    **({"team": list(body.archetype_ids)} if body.archetype_ids else {})}),
        title=title)
    sid = sess["id"]
    t = asyncio.create_task(_run_plan(sid, body, user))
    _basna_agent_tasks.add(t)
    t.add_done_callback(_basna_agent_tasks.discard)
    return {"session_id": sid, "title": title, "status": "running"}


class RecompileRequest(BaseModel):
    tiers: dict | None = None
    api_key: str = ""


@router.post("/sessions/{session_id}/recompile")
async def recompile_route(
    session_id: str, body: RecompileRequest, user: dict = Depends(get_current_user),
):
    """Recompute the truth + analysis from the already-persisted agent runs — no
    re-spawn, no re-dispatch. Use to recover when the merge stalled or failed, or
    to re-merge after changing tiers. Does not re-score reliability."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    runs = await db.list_basna_runs(session_id, user["id"])
    results = [{
        "archetype_id": r.get("archetype_id", ""), "role": r.get("role", ""),
        "weight": float(r.get("weight_at_run", 0.7) or 0.7),
        "output": r.get("output", "") or "",
        "ok": bool((r.get("output") or "").strip()),
    } for r in runs]
    if not any(_usable(r) for r in results):
        raise HTTPException(400, "no agent outputs to compile")

    registry = _load_registry()
    domain = sess.get("domain") or "general"
    merge_kind = sess.get("merge_kind") or "converge"

    def _merge_creds(tier: str) -> dict:
        return _resolve_merge_creds(body, registry, tier)

    # Reconcile generated files captured to disk but never persisted (a stalled
    # run saves files to the session dir in `finally`, before the final DB save).
    files_out = _parse_files(sess)
    existing_names = {f["name"] for f in files_out}
    fdir = _session_files_dir(session_id)
    for p in sorted(fdir.iterdir()):
        if p.is_file() and p.name not in existing_names:
            files_out.append({"name": p.name, "mime": _guess_mime(p.name),
                              "size": p.stat().st_size, "kind": "generated"})

    agg = await _aggregate(
        results, merge_kind, domain,
        conflict_fn=lambda good: asyncio.wait_for(_llm_conflict(good, _merge_creds("fast")), _MERGE_TIMEOUT),
        synth_fn=lambda good: asyncio.wait_for(_llm_synthesize(good, domain, _merge_creds("reason")), _SYNTH_TIMEOUT),
    )

    # Run the analysis only if the session doesn't already have one.
    try:
        prev = json.loads(sess.get("analysis") or "{}")
    except (json.JSONDecodeError, TypeError):
        prev = {}
    analysis = prev if isinstance(prev, dict) and prev else None
    good_results = [r for r in results if _usable(r)]
    if analysis is None and len(good_results) >= 2:
        try:
            analysis = await asyncio.wait_for(
                _llm_analysis(good_results, domain, _merge_creds("reason")), _SYNTH_TIMEOUT)
        except Exception as e:
            log.warning("Basna recompile analysis failed", error=str(e))

    await db.update_basna_session(
        session_id, user["id"], status="done",
        truth=agg["truth"], confidence=agg["confidence"],
        analysis=json.dumps(analysis or {}), files=json.dumps(files_out),
    )
    return {"session_id": session_id, "truth": agg["truth"], "confidence": agg["confidence"],
            "method": agg["method"], "contributors": agg["contributors"],
            "analysis": analysis, "files": files_out}


@router.get("/sessions/{session_id}/progress")
async def get_progress(session_id: str, user: dict = Depends(get_current_user)):
    """Live execution progress for a session, polled by the UI during /execute."""
    db = get_db()
    sess = await db.get_basna_session(session_id, user["id"])
    if not sess:
        raise HTTPException(404, "session not found")
    return _PROGRESS.get(session_id) or {"events": [], "active": False}


class FeedbackRequest(BaseModel):
    success: bool


@router.post("/runs/{run_id}/feedback")
async def run_feedback(
    run_id: int, body: FeedbackRequest, user: dict = Depends(get_current_user),
):
    """Human override of a run's success — a first-class signal over the auto-score.

    Revises the learned reliability by moving the outcome between buckets (no
    double-count), whether the run was auto-scored, unscored, or already overridden.
    """
    db = get_db()
    run = await db.get_basna_run(run_id, user["id"])
    if not run:
        raise HTTPException(404, "run not found")
    sess = await db.get_basna_session(run["session_id"], user["id"])
    domain = (sess.get("domain") if sess else "") or "general"
    archetypes = await merged_archetypes(db, user["id"])
    seed = next(
        (float(a.get("reliability_seed", 0.7)) for a in archetypes
         if a["id"] == run["archetype_id"]), 0.7,
    )

    old = run["success"]  # 1, 0, or None
    new = 1 if body.success else 0
    if old == new:
        return {"changed": False, "run_id": run_id, "success": body.success}

    await db.score_basna_run(run_id, user["id"], body.success)
    if old is None:
        d_success, d_fail = (1, 0) if new else (0, 1)
    else:
        d_success = (1 if new else 0) - (1 if old else 0)
        d_fail = (0 if new else 1) - (0 if old else 1)
    rel = await db.adjust_archetype_reliability(
        user["id"], run["archetype_id"], domain, d_success, d_fail, seed,
    )
    return {"changed": True, "run_id": run_id, "archetype_id": run["archetype_id"],
            "domain": domain, "success": body.success, "reliability": rel}
