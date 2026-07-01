"""Flight Deck HTTP API for Code mode — agentic coding over VFS project folders.

Each project is a VFS folder on disk (``<fd-data>/vfs/<user>/<project>/``) that
is also a git repo (see :mod:`code_git`). A spawned coding agent's workspace is
pinned to that folder, so it works in a real directory — relative paths, shell,
``npm``/``pytest``/``git`` all behave like a normal checkout — while the files
stay browsable through the existing VFS panel.

A cheap router sizes each request: **small** runs a single archetype directly;
**big** (Phase 2) drives the full Vatra build → Basna review → fix loop. Per-
project chat + run state live under ``<project>/.code/`` (gitignored), so the
conversation is portable with the folder and needs no DB migration.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck import code_git
from captain_claw.flight_deck.basna_routes import (
    _PROGRESS,
    _build_catalog,
    _dispatch_one,
    _fallback_difficulty,
    _load_owner_tiers,
    _load_registry,
    _phase,
    _progress,
    _progress_done,
    _progress_start,
    _score_archetypes,
)
from captain_claw.flight_deck.dubina_agents import (
    spawn_archetype_agent,
    stop_archetype_agent,
)
from captain_claw.flight_deck.vfs_routes import _project_root, _user_root
from captain_claw.logging import get_logger
from captain_claw.vfs import safe_name

log = get_logger(__name__)

router = APIRouter(prefix="/fd/code", tags=["code"])

_INSTRUCTIONS_DIR = Path(__file__).parent.parent / "instructions"
_DISPATCH_TIMEOUT = 900.0  # coding turns can install deps + run tests

_PLANNERS = {"light-planner", "long-horizon-planner", "architect"}
_SMALL = {"quick-dirty", "code-implementer", "debugger"}
_REVIEWERS = ["code-reviewer", "security-reviewer", "qa-engineer"]
_MAX_FIX_ROUNDS = 3

_REPORTS_DIRNAME = ".reports"

# Appended to every agent prompt: keep written reports inside the VFS project
# (committed + downloadable) instead of the agent's throwaway `saved/` tree.
_REPORTS_DIRECTIVE = (
    "\n\nIf you produce a written report, findings document, or summary file, save it "
    "as Markdown under a `.reports/` folder in the project root (create it if needed). "
    "NEVER write reports to `saved/` — that folder is untracked and won't be kept."
)


def _write_report(pdir: Path, name: str, content: str) -> str:
    """Persist a report into ``<project>/.reports/`` (committed with the run). Returns rel path."""
    rd = pdir / _REPORTS_DIRNAME
    rd.mkdir(parents=True, exist_ok=True)
    safe = safe_name(name, fallback="report") + ".md"
    (rd / safe).write_text(content or "")
    return f"{_REPORTS_DIRNAME}/{safe}"


def _persist_trace(project: str, pdir: Path, label: str) -> None:
    """Append this run's live progress events (tools/narration/usage) to
    ``.code/trace.jsonl`` so the coding process survives restarts and can be
    exported. ``_PROGRESS`` is in-memory only; this is its durable record."""
    events = (_PROGRESS.get(project) or {}).get("events") or []
    if not events:
        return
    tf = _code_dir(pdir) / "trace.jsonl"
    with tf.open("a") as fh:
        fh.write(json.dumps({"type": "run", "label": label, "ts": time.time(),
                             "count": len(events)}) + "\n")
        for e in events:
            fh.write(json.dumps({"type": "event", **e}) + "\n")


def _read_trace(pdir: Path) -> list[dict]:
    f = pdir / ".code" / "trace.jsonl"
    if not f.is_file():
        return []
    out: list[dict] = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _export_markdown(pdir: Path, project: str) -> str:
    """Assemble the full coding process — conversation (outputs), tool/narration
    trace, and any reports — into one Markdown document."""
    import time as _t

    def ts(v: float) -> str:
        try:
            return _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(v))
        except (ValueError, OSError):
            return ""

    lines = [f"# Coding process — {project}", f"_Exported {ts(_t.time())}_", ""]

    # 1) Conversation (requests + agent outputs).
    lines.append("## Conversation\n")
    for m in _read_chat(pdir):
        who = "🧑 User" if m.get("role") == "user" else f"🤖 {m.get('archetype') or 'assistant'}"
        meta = " · ".join(x for x in [m.get("kind"), m.get("size"),
                                      (m.get("commit") or "")[:7]] if x)
        lines.append(f"### {who}{(' — ' + meta) if meta else ''}  ·  {ts(m.get('ts', 0))}")
        lines.append((m.get("text") or "").rstrip())
        for f in (m.get("findings") or []):
            lines.append(f"- **{f.get('severity', '')}** · {f.get('title', '')}"
                         + (f" ({f['file']})" if f.get("file") else ""))
        lines.append("")

    # 2) Tool & narration trace (grouped per run).
    trace = _read_trace(pdir)
    if trace:
        lines.append("## Tool & narration trace\n")
        for rec in trace:
            if rec.get("type") == "run":
                lines.append(f"\n### ▶ {rec.get('label', 'run')}  ·  {ts(rec.get('ts', 0))}\n")
                continue
            stage = rec.get("stage", "")
            if stage in ("usage",):
                continue  # token lines are noise in a readable export; kept in JSON
            msg = rec.get("message", "")
            lines.append(f"- `{ts(rec.get('ts', 0))}`  {msg}")
        lines.append("")

    # 3) Inline any reports so the export is self-contained.
    rd = pdir / _REPORTS_DIRNAME
    if rd.is_dir():
        reports = sorted(f for f in rd.glob("*.md") if f.is_file())
        if reports:
            lines.append("## Reports\n")
            for rf in reports:
                lines.append(f"<details><summary>{rf.name}</summary>\n")
                lines.append(rf.read_text().rstrip())
                lines.append("\n</details>\n")

    return "\n".join(lines) + "\n"

# Mirror the frontend's tier inheritance (tierConfig.ts): the `coding`/`vision`
# tiers were added after many Library sets were first seeded, so a saved
# `fd:forge-tiers` blob may not carry an explicit entry for them. Resolve a tier
# to the user's nearest configured tier (with its provider/model/api_key) instead
# of falling through to the keyless registry default.
_TIER_FALLBACK = {"coding": "reason", "vision": "balanced"}


def _resolve_tcfg(tiers_map: dict, tier: str) -> dict:
    """The owner's tier config for ``tier``, inheriting a sibling tier when the
    exact one isn't configured. Returns {} only when the owner has no tiers at all."""
    if tiers_map.get(tier):
        return tiers_map[tier]
    pref = _TIER_FALLBACK.get(tier, "balanced")
    return (tiers_map.get(pref) or tiers_map.get("balanced")
            or tiers_map.get("reason") or next(iter(tiers_map.values()), {}))


# ── per-project storage (under <project>/.code/) ─────────────────────

def _pdir(user_id: str, project: str) -> Path:
    """Absolute on-disk dir for a project; 400 on a bad name."""
    name = safe_name(project, fallback="")
    if not name:
        raise HTTPException(400, "invalid project name")
    return _project_root(user_id, name)


def _code_dir(pdir: Path) -> Path:
    d = pdir / ".code"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_code_project(pdir: Path) -> bool:
    return (pdir / ".code").is_dir()


def _read_chat(pdir: Path) -> list[dict]:
    f = pdir / ".code" / "chat.jsonl"
    if not f.is_file():
        return []
    out: list[dict] = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _append_chat(pdir: Path, role: str, text: str, **meta) -> dict:
    msg = {"id": uuid.uuid4().hex[:12], "role": role, "text": text,
           "ts": time.time(), **meta}
    f = _code_dir(pdir) / "chat.jsonl"
    with f.open("a") as fh:
        fh.write(json.dumps(msg) + "\n")
    return msg


def _read_state(pdir: Path) -> dict:
    f = pdir / ".code" / "state.json"
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return {"status": "idle"}


def _write_state(pdir: Path, state: dict) -> None:
    (_code_dir(pdir) / "state.json").write_text(json.dumps(state, indent=2))


# ── router (small vs big + archetype pick) ───────────────────────────

async def _classify(intent: str, context: str, archetypes: list[dict],
                    tiers_map: dict, registry: dict) -> dict:
    """Size the request and pick the executing archetype(s). LLM with a
    deterministic fallback so a route is always returned."""
    by_id = {a["id"]: a for a in archetypes}
    sys_file = _INSTRUCTIONS_DIR / "code" / "router.md"
    system_prompt = sys_file.read_text() + "\n\n" + _build_catalog(archetypes, {})
    fast = _resolve_tcfg(tiers_map, "fast") or registry.get("tiers", {}).get("fast", {})
    raw: dict | None = None
    try:
        from captain_claw.llm import Message, create_provider
        prov = create_provider(
            provider=fast.get("provider", "anthropic"), model=fast.get("model", ""),
            api_key=fast.get("api_key") or None, base_url=fast.get("base_url") or None,
            temperature=0.1, max_tokens=600,
        )
        user_prompt = (f"{context}\nRequest: {intent}" if context else f"Request: {intent}")
        resp = await prov.complete(messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ], temperature=0.1, max_tokens=600)
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        raw = json.loads(content)
    except Exception as e:  # noqa: BLE001 — any failure → deterministic fallback
        log.warning("code router LLM failed; keyword fallback", error=str(e))
        raw = None

    if not isinstance(raw, dict) or "size" not in raw:
        low = intent.lower()
        breadth = len(_score_archetypes(intent, archetypes))
        difficulty = _fallback_difficulty(intent, breadth)
        is_bug = any(w in low for w in ("bug", "error", "crash", "fix", "broken", "fails"))
        size = "big" if difficulty == "hard" else "small"
        raw = {
            "size": size,
            "planner": "architect" if difficulty == "hard" else "light-planner",
            "small_archetype": "debugger" if is_bug else "code-implementer",
            "domain": "general", "difficulty": difficulty,
            "title": intent[:48], "why": "keyword fallback",
        }

    # Validate / clamp picks to known archetypes.
    raw["size"] = "big" if str(raw.get("size")).lower() == "big" else "small"
    if raw.get("planner") not in _PLANNERS or raw["planner"] not in by_id:
        raw["planner"] = "light-planner" if "light-planner" in by_id else "architect"
    if raw.get("small_archetype") not in _SMALL or raw["small_archetype"] not in by_id:
        raw["small_archetype"] = "code-implementer"
    return raw


# ── single-agent execution (small path; Phase-1 also handles big) ────

def _exec_prompt(intent: str) -> str:
    return (
        "You are working inside a real project directory — it IS your workspace and "
        "current working directory, a git repo. Create and edit files with PLAIN "
        "RELATIVE paths (e.g. `src/main.py`); use your shell to install deps, run, and "
        "verify. Do NOT use any `vfs:` prefix — just work in the directory you're in.\n\n"
        f"Task:\n{intent}\n\n"
        "When finished, briefly summarize what you created/changed and how you "
        "verified it actually runs." + _REPORTS_DIRECTIVE
    )


async def _run_agent(request: Request, user: dict, project: str, pdir: Path,
                     archetype_id: str, prompt: str, by_id: dict,
                     tiers_map: dict, env_vars: list) -> dict:
    """Spawn one archetype anchored at the project dir, dispatch ``prompt``, dispose."""
    arch = by_id[archetype_id]
    role = arch.get("role", archetype_id)
    tier = arch.get("tier", "coding")
    tcfg = _resolve_tcfg(tiers_map, tier)
    suffix = uuid.uuid4().hex[:6]

    def _on_action(act: dict) -> None:
        if act.get("tool") == "narration":
            _progress(project, "narration", f"{role}: {act.get('detail', '')}",
                      agent=role, tool="narration", detail=act.get("detail", ""))
        else:
            detail = f": {act['detail']}" if act.get("detail") else ""
            _progress(project, "action", f"{role} → {act.get('tool')}{detail}",
                      agent=role, tool=act.get("tool"), detail=act.get("detail", ""))

    def _on_usage(pt: int, ct: int, tt: int) -> None:
        _progress(project, "usage", f"{role} · {pt:,}→{ct:,} tok",
                  agent=role, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

    _phase(project, f"{role} working")
    port, token, slug = await spawn_archetype_agent(
        arch, tier, tcfg, request, user, name_suffix=suffix,
        env_vars=env_vars, workspace_path=str(pdir),
    )
    try:
        d = await _dispatch_one(
            port, token, prompt, _DISPATCH_TIMEOUT, on_action=_on_action,
            fleet_instructions=arch.get("fleet_instructions", ""),
            agent_name=role, on_usage=_on_usage,
        )
    finally:
        # Disposal is best-effort: the agent has already done its work, so a
        # stop failure must never discard the result or 500 the request.
        try:
            await stop_archetype_agent(slug)
        except Exception as e:  # noqa: BLE001
            log.warning("code: failed to stop agent", slug=slug, error=str(e))
    return d


# ── big-job prompts (Vatra plan→build, Basna review fan-out, fix loop) ─

def _plan_prompt(intent: str) -> str:
    return (
        "You are planning a coding task in THIS repository — it is your workspace and "
        "current directory. Survey the existing code first (relative paths, your shell), "
        "then produce a clear, scoped implementation plan and WRITE it to `plan.md` in the "
        "project root. The plan drives an implementer next, so make it concrete and ordered. "
        "Do NOT write any other code yet.\n\n"
        f"Task:\n{intent}"
    )


def _build_prompt(intent: str) -> str:
    return (
        "An implementation plan has been approved and saved as `plan.md` in THIS repository "
        "(your workspace and current directory). Implement it fully: create/edit files with "
        "plain relative paths, install deps and run/verify via your shell. Follow the plan; "
        "if you must deviate, say why. Do NOT use any `vfs:` prefix.\n\n"
        f"Original request for context:\n{intent}\n\n"
        "When finished, summarize what you built and how you verified it runs." + _REPORTS_DIRECTIVE
    )


_REVIEW_PROMPTS = {
    "code-reviewer": (
        "Review the CURRENT state of this repository (your workspace) for correctness bugs, "
        "edge cases, error handling, and regressions against the task. Read the files and use "
        "read-only shell (git diff, grep) — do not edit. Report findings ranked by severity "
        "(blocking / major / minor) with file:line and a concrete fix."
    ),
    "security-reviewer": (
        "Security-review the CURRENT state of this repository (your workspace): injection, "
        "auth/authz, secrets in code, unsafe input handling, dependency risks. Read-only — do "
        "not edit. Report CVSS-ranked findings with file:line and remediation."
    ),
    "qa-engineer": (
        "Assess this repository (your workspace) for test coverage and correctness. ACTUALLY "
        "RUN the test suite and/or the program via your shell to verify it works. If there are "
        "no tests, add a minimal test file covering the core path. Report failing tests, "
        "missing coverage, and edge cases as findings ranked by severity."
    ),
}


def _review_prompt(reviewer: str, intent: str) -> str:
    return f"{_REVIEW_PROMPTS[reviewer]}\n\nTask under review:\n{intent}" + _REPORTS_DIRECTIVE


def _fix_prompt(intent: str, fix_instructions: str) -> str:
    return (
        "A code review of THIS repository (your workspace) found issues that must be fixed. "
        "Apply the fixes with relative paths and verify via your shell. Fix ONLY the issues "
        "listed; keep working code intact.\n\n"
        f"Issues to fix:\n{fix_instructions}\n\n"
        f"Original request for context:\n{intent}" + _REPORTS_DIRECTIVE
    )


async def _triage_reviews(reviews: list[dict], intent: str,
                          tiers_map: dict, registry: dict) -> dict:
    """Merge the reviewers' reports into a fix decision (Basna-style verdict)."""
    sys_file = _INSTRUCTIONS_DIR / "code" / "triage.md"
    parts = [f"## {r['role']} report\n{r['output'] or '(no output)'}" for r in reviews]
    user_prompt = f"Task:\n{intent}\n\n" + "\n\n".join(parts)
    tier = _resolve_tcfg(tiers_map, "reason") or registry.get("tiers", {}).get("reason", {})
    try:
        from captain_claw.llm import Message, create_provider
        prov = create_provider(
            provider=tier.get("provider", "anthropic"), model=tier.get("model", ""),
            api_key=tier.get("api_key") or None, base_url=tier.get("base_url") or None,
            temperature=0.1, max_tokens=1500,
        )
        resp = await prov.complete(messages=[
            Message(role="system", content=sys_file.read_text()),
            Message(role="user", content=user_prompt),
        ], temperature=0.1, max_tokens=1500)
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        raw = json.loads(content)
    except Exception as e:  # noqa: BLE001 — be conservative: no auto-fix if triage fails
        log.warning("code triage failed", error=str(e))
        return {"needs_fix": False, "fixer": "code-implementer",
                "summary": "Review complete (triage unavailable — not auto-fixing).",
                "fix_instructions": "", "findings": []}
    raw["needs_fix"] = bool(raw.get("needs_fix"))
    if raw.get("fixer") not in ("debugger", "code-implementer"):
        raw["fixer"] = "code-implementer"
    return raw


async def _run_build_loop(request: Request, user: dict, project: str, pdir: Path,
                          intent: str, by_id: dict, tiers_map: dict,
                          env_vars: list, registry: dict) -> None:
    """Big-job pipeline: build → review fan-out → capped fix loop. Runs in background;
    communicates via progress events, the chat log, and per-phase git commits."""
    _progress_start(project)
    _write_state(pdir, {"status": "running"})
    try:
        # 1) Build from the approved plan.md (Vatra-style: plan → implement).
        _phase(project, "Building")
        d = await _run_agent(request, user, project, pdir, "code-implementer",
                             _build_prompt(intent), by_id, tiers_map, env_vars)
        sha = await code_git.git_commit(pdir, f"[build] code-implementer: {intent[:60]}")
        _append_chat(pdir, "assistant", (d.get("output") or "(build produced no summary)").strip(),
                     kind="build", archetype="code-implementer", ok=bool(d.get("ok")), commit=sha or "")

        # 2) Review → fix loop (Basna-style parallel review fan-out → triage → fix).
        for rnd in range(_MAX_FIX_ROUNDS + 1):
            _phase(project, "Reviewing")
            reviews_raw = await asyncio.gather(*[
                _run_agent(request, user, project, pdir, rv, _review_prompt(rv, intent),
                           by_id, tiers_map, env_vars)
                for rv in _REVIEWERS
            ], return_exceptions=True)
            reviews = []
            for rv, res in zip(_REVIEWERS, reviews_raw):
                role = by_id[rv].get("role", rv)
                out = "" if isinstance(res, Exception) else (res.get("output") or "")
                reviews.append({"role": role, "id": rv, "output": out})
                # Persist each reviewer's full findings into the VFS project.
                if out.strip():
                    _write_report(pdir, f"review-r{rnd}-{rv}", f"# {role} — review r{rnd}\n\n{out}")

            triage = await _triage_reviews(reviews, intent, tiers_map, registry)
            review_summary = triage.get("summary", "Review complete.")
            _write_report(pdir, f"review-r{rnd}-summary", f"# Review summary — r{rnd}\n\n{review_summary}")
            # Commit reviewer reports + any tests qa-engineer added during review.
            await code_git.git_commit(pdir, f"[review r{rnd}] reports + reviewer tests")
            _append_chat(pdir, "assistant", review_summary, kind="review", round=rnd,
                         findings=triage.get("findings", []), needs_fix=triage["needs_fix"])

            if not triage["needs_fix"] or rnd == _MAX_FIX_ROUNDS:
                if triage["needs_fix"]:
                    _append_chat(pdir, "assistant",
                                 f"Reached the fix-round cap ({_MAX_FIX_ROUNDS}); stopping with "
                                 "open findings above.", kind="note")
                break

            # 3) Fix the blocking/major findings, then re-review.
            _phase(project, f"Fixing (round {rnd + 1})")
            fixer = triage["fixer"]
            fd = await _run_agent(request, user, project, pdir, fixer,
                                  _fix_prompt(intent, triage.get("fix_instructions", "")),
                                  by_id, tiers_map, env_vars)
            fsha = await code_git.git_commit(pdir, f"[fix r{rnd + 1}] {fixer}")
            _append_chat(pdir, "assistant", (fd.get("output") or "(fix produced no summary)").strip(),
                         kind="fix", round=rnd + 1, archetype=fixer,
                         ok=bool(fd.get("ok")), commit=fsha or "")

        _write_state(pdir, {"status": "idle"})
    except Exception as e:  # noqa: BLE001 — never leave the project stuck in "running"
        log.warning("code build loop failed", project=project, error=str(e))
        _append_chat(pdir, "assistant", f"⚠️ Build loop failed: {e}", kind="note", ok=False)
        _write_state(pdir, {"status": "idle"})
    finally:
        _persist_trace(project, pdir, f"Build → review → fix: {intent[:80]}")
        _progress_done(project)


# ── endpoints ────────────────────────────────────────────────────────

class CreateProjectReq(BaseModel):
    name: str


class MessageReq(BaseModel):
    project: str
    text: str


class RollbackReq(BaseModel):
    ref: str


@router.get("/projects")
async def list_projects(user: dict = Depends(get_current_user)):
    """List the user's Code projects (VFS folders that carry a ``.code/`` marker)."""
    root = _user_root(user["id"])
    out: list[dict] = []
    if root.is_dir():
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_dir() or not _is_code_project(p):
                continue
            chat = _read_chat(p)
            last = chat[-1] if chat else None
            files = sum(1 for f in p.rglob("*")
                        if f.is_file() and ".code" not in f.parts and ".git" not in f.parts)
            out.append({
                "name": p.name, "files": files,
                "messages": len(chat),
                "last_message": (last or {}).get("text", "")[:120],
                "mtime": p.stat().st_mtime,
                "status": _read_state(p).get("status", "idle"),
            })
    return {"projects": out}


@router.post("/projects")
async def create_project(body: CreateProjectReq, user: dict = Depends(get_current_user)):
    name = safe_name(body.name, fallback="")
    if not name:
        raise HTTPException(400, "invalid project name")
    pdir = _project_root(user["id"], name)
    if _is_code_project(pdir):
        raise HTTPException(409, "project already exists")
    pdir.mkdir(parents=True, exist_ok=True)
    _code_dir(pdir)
    await code_git.git_init(pdir)
    _write_state(pdir, {"status": "idle"})
    return {"name": name, "files": 0, "messages": 0, "status": "idle"}


@router.get("/projects/{project}/chat")
async def get_chat(project: str, user: dict = Depends(get_current_user)):
    pdir = _pdir(user["id"], project)
    if not _is_code_project(pdir):
        raise HTTPException(404, "project not found")
    return {"messages": _read_chat(pdir), "state": _read_state(pdir)}


@router.get("/projects/{project}/progress")
async def get_progress(project: str, user: dict = Depends(get_current_user)):
    p = _PROGRESS.get(safe_name(project, fallback=""))
    if p is None:
        return {"events": [], "active": False}
    return p


@router.get("/projects/{project}/export")
async def export_process(project: str, format: str = "md",
                         user: dict = Depends(get_current_user)):
    """Export the full coding process — conversation (outputs), tool/narration
    trace, and reports — as a downloadable Markdown or JSON document."""
    from fastapi.responses import JSONResponse, PlainTextResponse
    pdir = _pdir(user["id"], project)
    if not _is_code_project(pdir):
        raise HTTPException(404, "project not found")
    name = safe_name(project, fallback="project")
    if format == "json":
        data = {"project": name, "exported_at": time.time(),
                "messages": _read_chat(pdir), "trace": _read_trace(pdir)}
        return JSONResponse(data, headers={
            "Content-Disposition": f'attachment; filename="{name}-process.json"'})
    return PlainTextResponse(
        _export_markdown(pdir, name), media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{name}-process.md"'})


@router.get("/projects/{project}/log")
async def get_log(project: str, user: dict = Depends(get_current_user)):
    pdir = _pdir(user["id"], project)
    # Self-heal projects created before per-folder isolation existed: ensure this
    # folder is its OWN repo (not resolving to an ancestor repo the VFS tree sits in).
    if _is_code_project(pdir):
        await code_git.git_init(pdir)
    return {"commits": await code_git.git_log(pdir)}


@router.get("/projects/{project}/diff")
async def get_diff(project: str, ref_a: str = "", ref_b: str = "",
                   user: dict = Depends(get_current_user)):
    pdir = _pdir(user["id"], project)
    return {"diff": await code_git.git_diff(pdir, ref_a, ref_b)}


@router.get("/projects/{project}/show")
async def show_commit(project: str, sha: str, user: dict = Depends(get_current_user)):
    """Return a single commit's patch for the diff viewer."""
    pdir = _pdir(user["id"], project)
    return {"diff": await code_git.git_show(pdir, sha)}


@router.post("/projects/{project}/rollback")
async def rollback(project: str, body: RollbackReq, user: dict = Depends(get_current_user)):
    pdir = _pdir(user["id"], project)
    if _read_state(pdir).get("status") == "running":
        raise HTTPException(409, "a build is running — wait for it to finish")
    target = next((c for c in await code_git.git_log(pdir) if c["sha"].startswith(body.ref)), None)
    await code_git.git_reset(pdir, body.ref)
    _append_chat(pdir, "assistant",
                 f"↩ Rolled back to {body.ref[:7]}"
                 + (f": {target['message']}" if target else ""), kind="note")
    return {"ok": True, "head": (await code_git.git_log(pdir, 1))[:1]}


@router.post("/message")
async def message(body: MessageReq, request: Request, user: dict = Depends(get_current_user)):
    """Main entry: route the request, run it, commit, and append to the chat."""
    intent = body.text.strip()
    if not intent:
        raise HTTPException(400, "text is required")
    project = safe_name(body.project, fallback="")
    pdir = _pdir(user["id"], project)
    if not _is_code_project(pdir):
        raise HTTPException(404, "project not found — create it first")

    db = get_db()
    archetypes = await merged_archetypes(db, user["id"])
    by_id = {a["id"]: a for a in archetypes}
    registry = _load_registry()
    tiers_map, env_vars = await _load_owner_tiers(db, user["id"])

    _append_chat(pdir, "user", intent)
    _progress_start(project)
    _write_state(pdir, {"status": "running"})
    try:
        _phase(project, "Routing")
        prior = _read_chat(pdir)[-8:]
        context = "\n".join(f"{m['role']}: {m['text'][:200]}" for m in prior[:-1])
        route = await _classify(intent, context, archetypes, tiers_map, registry)
        _progress(project, "route", f"size={route['size']} · {route.get('why', '')}",
                  size=route["size"])

        # ── SMALL: one best-fit archetype runs the edit directly. ──
        if route["size"] == "small":
            executor = route["small_archetype"]
            d = await _run_agent(request, user, project, pdir, executor,
                                 _exec_prompt(intent), by_id, tiers_map, env_vars)
            sha = await code_git.git_commit(
                pdir, f"[edit] {executor}: {route.get('title', intent)[:60]}")
            out = (d.get("output") or "").strip() or "(no output)"
            if not d.get("ok"):
                out = f"⚠️ {executor} failed: {d.get('error', 'unknown error')}"
            assistant = _append_chat(pdir, "assistant", out, archetype=executor,
                                     size="small", ok=bool(d.get("ok")), commit=sha or "",
                                     route=route)
            _write_state(pdir, {"status": "idle", "last_route": route})
            return {"message": assistant, "route": route, "commit": sha}

        # ── BIG: plan first, then stop at the approval gate. ──
        planner = route["planner"]
        _phase(project, f"Planning ({planner})")
        d = await _run_agent(request, user, project, pdir, planner,
                             _plan_prompt(intent), by_id, tiers_map, env_vars)
        plan_md = pdir / "plan.md"
        plan_text = plan_md.read_text() if plan_md.is_file() else (d.get("output") or "").strip()
        sha = await code_git.git_commit(pdir, f"[plan] {planner}: {route.get('title', intent)[:60]}")
        assistant = _append_chat(
            pdir, "assistant", plan_text or "(planner produced no plan)",
            kind="plan", archetype=planner, ok=bool(d.get("ok")), commit=sha or "", route=route)
        _write_state(pdir, {"status": "awaiting_plan", "intent": intent, "route": route})
        return {"message": assistant, "route": route, "commit": sha,
                "status": "awaiting_plan", "plan": plan_text}
    finally:
        _sz = (locals().get("route") or {}).get("size", "")
        _persist_trace(project, pdir, f"{_sz + ': ' if _sz else ''}{intent[:80]}")
        _progress_done(project)


class ApproveReq(BaseModel):
    project: str
    plan: str | None = None  # optional user-edited plan to overwrite plan.md


@router.post("/plan/approve")
async def approve_plan(body: ApproveReq, request: Request,
                       user: dict = Depends(get_current_user)):
    """Approve a big job's plan and kick off the build → review → fix loop in the
    background. Returns immediately; the frontend follows progress + the chat log."""
    project = safe_name(body.project, fallback="")
    pdir = _pdir(user["id"], project)
    if not _is_code_project(pdir):
        raise HTTPException(404, "project not found")
    state = _read_state(pdir)
    if state.get("status") != "awaiting_plan":
        raise HTTPException(409, "no plan awaiting approval")
    intent = state.get("intent", "")

    # Honor a user-edited plan.
    if body.plan and body.plan.strip():
        (pdir / "plan.md").write_text(body.plan)
        await code_git.git_commit(pdir, "[plan] user-edited")

    db = get_db()
    archetypes = await merged_archetypes(db, user["id"])
    by_id = {a["id"]: a for a in archetypes}
    registry = _load_registry()
    tiers_map, env_vars = await _load_owner_tiers(db, user["id"])

    _append_chat(pdir, "user", "✓ Plan approved — building.", kind="approval")
    asyncio.create_task(_run_build_loop(
        request, user, project, pdir, intent, by_id, tiers_map, env_vars, registry))
    return {"status": "running"}
