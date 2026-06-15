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
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/basna", tags=["basna"])

_INSTRUCTIONS_DIR = Path(__file__).parent.parent / "instructions"
_VALID_TIERS = {"reason", "balanced", "fast", "longctx"}
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

class RouteRequest(BaseModel):
    intent: str
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


class CreateSessionRequest(BaseModel):
    intent: str
    config: str = "{}"


class UpdateSessionRequest(BaseModel):
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
    return await db.create_basna_session(user["id"], body.intent.strip(), body.config)


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
    registry = _load_registry()
    archetypes = registry.get("archetypes", [])
    archetypes_by_id = {a["id"]: a for a in archetypes}
    seeds = {a["id"]: float(a.get("reliability_seed", 0.7)) for a in archetypes}

    # Group this user's learned reliability by archetype for the catalog hints.
    rel_rows = await db.get_archetype_reliability(user["id"])
    reliability: dict[str, list[dict]] = {}
    for r in rel_rows:
        reliability.setdefault(r["archetype_id"], []).append(r)

    # Resolve fast-tier creds for the router call.
    tiers = registry.get("tiers", {})
    fast = tiers.get("fast", {})
    provider = body.provider or fast.get("provider", "anthropic")
    model = body.model or fast.get("model", "")
    base_url = body.base_url or fast.get("base_url", "")

    system_prompt_file = _INSTRUCTIONS_DIR / "basna" / "router.md"
    if not system_prompt_file.is_file():
        raise HTTPException(500, "Basna router prompt not found")
    system_prompt = system_prompt_file.read_text() + "\n\n" + _build_catalog(archetypes, reliability)
    user_prompt = (
        f"Task: {intent}\n\n"
        f"max_agents: {body.max_agents}. Select the smallest archetype set that "
        f"handles this task well, scaled to its difficulty."
    )

    started = time.monotonic()
    raw: dict | None = None
    source = "llm"
    try:
        from captain_claw.llm import create_provider, Message
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

    # Attach the current learned weight (for the chosen domain) to each pick — the
    # prior the aggregator and learning loop will start from in later phases.
    for s in route["selected"]:
        s["prior_weight"] = await db.get_archetype_weight(
            user["id"], s["archetype_id"], route["domain"], seeds.get(s["archetype_id"], 0.7),
        )
    route["source"] = source
    route["elapsed_ms"] = int((time.monotonic() - started) * 1000)

    # Persist onto a session (create one if the caller didn't supply an id).
    session_id = body.session_id.strip()
    if session_id:
        sess = await db.get_basna_session(session_id, user["id"])
        if not sess:
            raise HTTPException(404, "session not found")
    else:
        sess = await db.create_basna_session(user["id"], intent)
        session_id = sess["id"]
    await db.update_basna_session(
        session_id, user["id"],
        domain=route["domain"], difficulty=route["difficulty"],
        merge_kind=route["merge_kind"], route=json.dumps(route), status="routed",
    )

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
    agree = await conflict_fn(good)
    if agree:
        best = max(good, key=lambda r: r["weight"])
        return {"truth": best["output"].strip(),
                "confidence": round(min(0.99, mean_w + 0.1), 3),
                "method": "weighted_pick",
                "contributors": [r["archetype_id"] for r in good]}
    merged = await synth_fn(good)
    return {"truth": merged.strip(),
            "confidence": round(max(0.05, mean_w - 0.2), 3),
            "method": "synthesis",
            "contributors": [r["archetype_id"] for r in good]}


def _tier_creds(registry: dict, tier: str, api_key: str) -> dict:
    t = (registry.get("tiers") or {}).get(tier, {})
    return {"provider": t.get("provider", "anthropic"), "model": t.get("model", ""),
            "base_url": t.get("base_url", "") or None, "api_key": api_key or None}


async def _llm_conflict(good: list[dict], creds: dict) -> bool:
    """Fast-tier check: do these answers substantively agree? Default to disagree."""
    from captain_claw.llm import create_provider, Message
    listing = "\n\n".join(f"[{i+1}] {r['output'].strip()[:2000]}" for i, r in enumerate(good))
    prov = create_provider(temperature=0.0, max_tokens=256, **creds)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "You compare independent answers to the same task. Reply ONLY with JSON "
            '{"agree": true} if they reach substantively the same conclusion, or '
            '{"agree": false} if they materially disagree on the answer.')),
        Message(role="user", content=listing),
    ], temperature=0.0, max_tokens=256)
    content = resp.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    try:
        return bool(json.loads(content).get("agree", False))
    except (json.JSONDecodeError, AttributeError):
        return False


async def _llm_synthesize(good: list[dict], domain: str, creds: dict) -> str:
    """Reason-tier reconciliation of disagreeing answers, trusting weight."""
    from captain_claw.llm import create_provider, Message
    listing = "\n\n".join(
        f"### {r['role']} (reliability weight {r['weight']:.2f})\n{r['output'].strip()}"
        for r in good
    )
    prov = create_provider(temperature=0.3, max_tokens=4096, **creds)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            f"Independent specialists gave conflicting answers in the {domain} domain. "
            "Reconcile them into one correct answer. Weigh higher-reliability "
            "contributors more, but follow the evidence over the weight when a "
            "lower-weighted contributor is clearly right. State the resolved answer "
            "directly; do not narrate the disagreement.")),
        Message(role="user", content=listing),
    ], temperature=0.3, max_tokens=4096)
    return resp.content.strip()


async def _llm_judge(good: list[dict], truth: str, creds: dict) -> list[bool]:
    """Fast-tier per-contribution verdict: did each support the final truth?

    Returns one bool per `good` entry, in order. Raises on unparseable output so
    the caller can leave those runs unscored rather than reward them by default.
    """
    from captain_claw.llm import create_provider, Message
    listing = "\n\n".join(f"[{i+1}] {r['output'].strip()[:2000]}" for i, r in enumerate(good))
    prov = create_provider(temperature=0.0, max_tokens=512, **creds)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "Given a FINAL answer and several independent contributions, decide for "
            "each contribution whether it substantively agrees with or supports the "
            "final answer. Reply ONLY with a JSON array of booleans — one per "
            "contribution, in order, same length as the input.")),
        Message(role="user", content=f"FINAL ANSWER:\n{truth[:4000]}\n\nCONTRIBUTIONS:\n{listing}"),
    ], temperature=0.0, max_tokens=512)
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
) -> tuple[str, list[dict]]:
    """Connect to an agent's /ws, send one chat, return (final reply, actions).

    `actions` is the agent's tool calls (the `monitor` events Council also shows),
    each {tool, detail}. `on_action(act)` is invoked live as each one arrives.
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
        elif mtype == "usage" and not msg.get("replay"):
            # Each turn's LLM usage — model + token counts (the agent does not
            # broadcast full prompt/response, only this summary).
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
                    except asyncio.TimeoutError:
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
                        image_paths: list[str] | None = None) -> dict:
    started = time.monotonic()
    try:
        out, actions = await _send_chat_and_collect(
            port, token, prompt, timeout, on_action=on_action,
            fleet_instructions=fleet_instructions, agent_name=agent_name,
            file_paths=file_paths, image_paths=image_paths)
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

    registry = _load_registry()
    arch_by_id = {a["id"]: a for a in registry.get("archetypes", [])}
    seeds = {a["id"]: float(a.get("reliability_seed", 0.7)) for a in registry.get("archetypes", [])}
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
        AgentConfig, spawn_process, _do_stop_process, _load_process_registry,
        _save_process_registry, _processes, DATA_DIR,
    )

    await db.update_basna_session(body.session_id, user["id"], status="running")

    sid = body.session_id
    _progress_start(sid)
    sid8 = sid[:8]
    run_tag = format(int(time.time()), "x")[-6:]  # unique per run → no slug collisions on re-run
    plan = [(s, arch_by_id[s["archetype_id"]]) for s in selected if s["archetype_id"] in arch_by_id]
    _progress(sid, "route", f"Selected {len(plan)} archetype(s) · {domain} / {merge_kind}")
    spawned: list[dict] = []  # {sel, arch, slug, port, auth}
    results: list[dict] = []
    try:
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
            base = dict(
                name=f"basna-{sid8}-{run_tag}-{arch['id']}",
                description=f"Basna ephemeral · {sel.get('role') or arch.get('role', '')}",
                cognitive_mode=sel.get("cognitive_mode") or arch.get("cognitive_mode", "neutra"),
                tools=arch.get("tools") or AgentConfig().tools,
                env_vars=body.env_vars or [],
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

        # 2) Dispatch the task to each agent in parallel; log each tool call live
        # and each agent's completion as it returns.
        async def _dispatch_tracked(sp: dict) -> dict:
            sel, arch = sp["sel"], sp["arch"]
            role = sel.get("role") or arch.get("role") or arch["id"]
            fleet = sel.get("fleet_instructions") or arch.get("fleet_instructions", "")

            def _on_action(act: dict) -> None:
                if act["tool"] == "narration":
                    _progress(sid, "narration", f"{role}: {act['detail']}")
                else:
                    detail = f": {act['detail']}" if act.get("detail") else ""
                    _progress(sid, "action", f"{role} → {act['tool']}{detail}")

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
                file_paths=doc_paths, image_paths=img_paths,
            )
            mark = "✓" if d["ok"] else "✗"
            extra = "" if d["ok"] else f" — {str(d.get('error', ''))[:160]}"
            _progress(sid, "dispatch",
                      f"{role} {mark} · {len(d['actions'])} action(s) ({d['latency_ms'] / 1000:.1f}s){extra}", ok=d["ok"])
            return d

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
                                            "size": p.stat().st_size, "kind": "generated"})
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
        lt = (body.tiers or {}).get(tier)
        if lt and lt.get("model"):
            return {"provider": lt.get("provider", "anthropic"), "model": lt.get("model", ""),
                    "base_url": lt.get("base_url") or None,
                    "api_key": lt.get("api_key") or body.api_key or None}
        return _tier_creds(registry, tier, body.api_key)

    # 5) Compile the truth (weighted; LLM synthesis only on genuine conflict).
    _progress(sid, "merge", "Compiling the truth…")
    agg = await _aggregate(
        results, merge_kind, domain,
        conflict_fn=lambda good: _llm_conflict(good, _merge_creds("fast")),
        synth_fn=lambda good: _llm_synthesize(good, domain, _merge_creds("reason")),
    )
    _progress(sid, "merge", f"Merged via {agg['method']} · confidence {agg['confidence']:.0%}")

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
    )
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
        "agents": [{"archetype_id": r["archetype_id"], "role": r["role"],
                    "ok": r["ok"], "latency_ms": r["latency_ms"], "weight": r["weight"],
                    "actions": r.get("actions", []),
                    "run_id": run_ids[i] if i < len(run_ids) else None,
                    "success": scores.get(r["archetype_id"])} for i, r in enumerate(results)],
        "learned": learned,
        "spawned": len(spawned), "dispatched": len(results),
    }


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
    registry = _load_registry()
    seed = next(
        (float(a.get("reliability_seed", 0.7)) for a in registry.get("archetypes", [])
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
