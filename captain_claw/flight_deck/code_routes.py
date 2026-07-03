"""Flight Deck HTTP API for Code mode — agentic coding over VFS projects.

Model: **project → folders + sessions**.

* A *project* is a namespace dir ``<fd-data>/vfs/<user>/<project>/`` with a
  control dir ``<project>/.code/`` holding ``project.json`` (folder membership)
  and ``sessions/<sid>/{chat,trace,state}`` (one conversation each).
* A *folder* is the actual git repo an agent works in — a VFS sub-dir
  (``<project>/<folder>/``), the project dir itself (legacy flat), or an
  external **linked** folder (same local path can be a folder in many projects).
* A *session* is a conversation that targets one of the project's folders; git
  commits land in that folder's repo, chat/trace/state live under the session.

A cheap router sizes each request: **small** runs one archetype directly;
**big** drives the Vatra build → Basna review → capped fix loop.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck.archetypes import merged_archetypes
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck import code_git
from captain_claw.flight_deck import code_map
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
from captain_claw.flight_deck.vfs_routes import _user_root
from captain_claw.logging import get_logger
from captain_claw.vfs import link_target_at, read_links_at, safe_name

log = get_logger(__name__)

router = APIRouter(prefix="/fd/code", tags=["code"])

_INSTRUCTIONS_DIR = Path(__file__).parent.parent / "instructions"
_DISPATCH_TIMEOUT = 900.0  # coding turns can install deps + run tests

_PLANNERS = {"light-planner", "long-horizon-planner", "architect"}
_GIT = "git-operator"
_SMALL = {"quick-dirty", "code-implementer", "debugger", _GIT}
# git verbs that, in the coding context, mean a version-control operation.
_GIT_WORDS = ("commit", "push", "pull", "fetch", "branch", "checkout", "switch",
              "merge", "rebase", "stash", "git status", "git log", "git diff",
              "revert", "reset", "cherry-pick", " tag", "remote")
_REVIEWERS = ["code-reviewer", "security-reviewer", "qa-engineer"]
_MAX_FIX_ROUNDS = 3

_CARTOGRAPHER = "code-cartographer"
_MAP_WORDS = ("map this", "map the repo", "map the codebase", "index this", "index the code",
              "code map", "codemap", "build the map", "update the map", "refresh the map",
              "cartograph")


def _is_map_intent(intent: str) -> bool:
    low = intent.lower()
    return any(w in low for w in _MAP_WORDS)


_BACKLOG_WORDS = ("continue fixing", "fix the backlog", "fix backlog",
                  "resume fixing", "resume the fixes", "finish the fixes")


def _is_backlog_intent(intent: str) -> bool:
    low = intent.lower()
    return any(w in low for w in _BACKLOG_WORDS)


async def _update_map(repo: Path, tiers_map: dict, registry: dict) -> None:
    """Keep the code map fresh after a commit — reindex (blob-hash gated) + a
    cheap purpose summary for the files that changed."""
    try:
        res = code_map.reindex(repo)
        changed = res.get("changed_files") or []
        if changed:
            creds = _resolve_tcfg(tiers_map, "fast") or registry.get("tiers", {}).get("fast", {})
            await code_map.summarize_changed(repo, changed, creds)
    except Exception as e:  # noqa: BLE001
        log.warning("code map update failed", error=str(e))

# ── per-turn token accounting (P4: cost visibility) ─────────────────
# One entry per progress key; reset at turn start, fed by _run_agent's
# usage callback, summarized into chat at turn end.
_TURN_USAGE: dict[str, dict] = {}


def _usage_reset(pkey: str) -> None:
    _TURN_USAGE[pkey] = {"prompt": 0, "completion": 0, "runs": 0}


def _usage_add(pkey: str, pt: int, ct: int) -> None:
    u = _TURN_USAGE.setdefault(pkey, {"prompt": 0, "completion": 0, "runs": 0})
    u["prompt"] += int(pt or 0)
    u["completion"] += int(ct or 0)
    u["runs"] += 1


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _usage_summary(pkey: str) -> str:
    u = _TURN_USAGE.get(pkey)
    if not u or not u["runs"]:
        return ""
    return (f"{u['runs']} agent run{'s' if u['runs'] != 1 else ''} · "
            f"{_fmt_tok(u['prompt'])} in → {_fmt_tok(u['completion'])} out tokens")


_REPORTS_DIRNAME = ".reports"
_REPORTS_DIRECTIVE = (
    "\n\nIf you produce a written report, findings document, or summary file, save it "
    "as Markdown under a `.reports/` folder in the project root (create it if needed). "
    "NEVER write reports to `saved/` — that folder is untracked and won't be kept."
    "\n\nFILE SAFETY: to change an EXISTING file, use the `edit` tool "
    "(old_string → new_string) — never `write`, which replaces the ENTIRE file. "
    "Reserve `write` for brand-new files. A write that would shrink an existing "
    "file is refused; if you genuinely need a full rewrite, read the file first "
    "and pass overwrite=true."
)


# ── project / folder / session storage ───────────────────────────────

def _proj_dir(user_id: str, project: str) -> Path:
    name = safe_name(project, fallback="")
    if not name:
        raise HTTPException(400, "invalid project name")
    return (_user_root(user_id) / name).resolve()


def _proj_code(user_id: str, project: str) -> Path:
    """``<project>/.code`` — control dir; gitignored so it never enters a repo
    (a legacy-flat project's own dir IS a repo)."""
    d = _proj_dir(user_id, project) / ".code"
    d.mkdir(parents=True, exist_ok=True)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n")
    return d


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _read_project(user_id: str, project: str) -> dict:
    return _read_json(_proj_dir(user_id, project) / ".code" / "project.json", {"folders": []})


def _write_project(user_id: str, project: str, data: dict) -> None:
    (_proj_code(user_id, project) / "project.json").write_text(json.dumps(data, indent=2))


def _folder_meta(user_id: str, project: str, folder: str) -> dict | None:
    for f in _read_project(user_id, project).get("folders", []):
        if f.get("name") == folder:
            return f
    return None


def _folder_repo(user_id: str, project: str, folder: str) -> Path:
    """Resolve a project folder to its on-disk git repo (agent workspace)."""
    meta = _folder_meta(user_id, project, folder)
    if not meta:
        raise HTTPException(404, "folder not found in project")
    proj = _proj_dir(user_id, project)
    kind = meta.get("kind")
    if kind == "link":
        tgt = link_target_at(_user_root(user_id), meta.get("link", ""))
        if tgt is None:
            raise HTTPException(404, "linked folder source is missing")
        return tgt
    if kind == "self":
        return proj
    return (proj / safe_name(folder, fallback="folder")).resolve()


def _read_sessions(user_id: str, project: str) -> list[dict]:
    data = _read_json(_proj_dir(user_id, project) / ".code" / "sessions.json", [])
    return data if isinstance(data, list) else []


def _write_sessions(user_id: str, project: str, sessions: list[dict]) -> None:
    (_proj_code(user_id, project) / "sessions.json").write_text(json.dumps(sessions, indent=2))


def _sget(user_id: str, project: str, sid: str) -> dict | None:
    return next((s for s in _read_sessions(user_id, project) if s.get("id") == sid), None)


def _session_dir(user_id: str, project: str, sid: str) -> Path:
    d = _proj_code(user_id, project) / "sessions" / safe_name(sid, fallback="s")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── chat / state / trace (per session dir) ───────────────────────────

def _read_chat(sdir: Path) -> list[dict]:
    return [json.loads(l) for l in (sdir / "chat.jsonl").read_text().splitlines()
            if l.strip()] if (sdir / "chat.jsonl").is_file() else []


def _append_chat(sdir: Path, role: str, text: str, **meta) -> dict:
    msg = {"id": uuid.uuid4().hex[:12], "role": role, "text": text, "ts": time.time(), **meta}
    with (sdir / "chat.jsonl").open("a") as fh:
        fh.write(json.dumps(msg) + "\n")
    return msg


def _read_state(sdir: Path) -> dict:
    return _read_json(sdir / "state.json", {"status": "idle"})


def _write_state(sdir: Path, state: dict) -> None:
    (sdir / "state.json").write_text(json.dumps(state, indent=2))


def _read_trace(sdir: Path) -> list[dict]:
    f = sdir / "trace.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.is_file() else []


def _persist_trace(pkey: str, sdir: Path, label: str) -> None:
    events = (_PROGRESS.get(pkey) or {}).get("events") or []
    if not events:
        return
    with (sdir / "trace.jsonl").open("a") as fh:
        fh.write(json.dumps({"type": "run", "label": label, "ts": time.time(),
                             "count": len(events)}) + "\n")
        for e in events:
            fh.write(json.dumps({"type": "event", **e}) + "\n")


# ── reports (per folder repo) ────────────────────────────────────────

def _write_report(repo: Path, name: str, content: str) -> str:
    rd = repo / _REPORTS_DIRNAME
    rd.mkdir(parents=True, exist_ok=True)
    safe = safe_name(name, fallback="report") + ".md"
    (rd / safe).write_text(content or "")
    return f"{_REPORTS_DIRNAME}/{safe}"


_PLANS_DIRNAME = ".plans"


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:max_len].strip("-") or "plan"


def _new_plan_rel(repo: Path, intent: str) -> str:
    """A unique, sortable plan path for this turn: ``.plans/<ts>-<slug>.md``.

    Each big turn gets its OWN plan file so a new plan never overwrites the
    prior one — the `.plans/` folder becomes the plan history, committed with
    the repo alongside `.reports/`.
    """
    (repo / _PLANS_DIRNAME).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    rel = f"{_PLANS_DIRNAME}/{ts}-{_slugify(intent)}.md"
    if (repo / rel).exists():  # same-second collision within a session
        rel = f"{_PLANS_DIRNAME}/{ts}-{uuid.uuid4().hex[:6]}-{_slugify(intent)}.md"
    return rel


def _history_preamble(sdir: Path, max_msgs: int = 8, max_chars: int = 700) -> str:
    """Recent conversation so a fresh ephemeral agent has continuity.

    Each turn spawns a brand-new agent that would otherwise see ONLY the current
    message — so follow-up turns ("no, don't use the browser", "the build button
    still does nothing") lose all prior context. This gives the agent the recent
    back-and-forth. The current user message is excluded (it's already in the
    task prompt); long messages (plans, review dumps) are truncated.
    """
    msgs = _read_chat(sdir)[:-1]  # drop the just-appended current user turn
    recent = [m for m in msgs[-max_msgs:] if (m.get("text") or "").strip()]
    if not recent:
        return ""
    lines = []
    for m in recent:
        who = "User" if m.get("role") == "user" else (m.get("archetype") or "Assistant")
        text = " ".join((m.get("text") or "").split())
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"{who}: {text}")
    return (
        "=== Conversation so far (most recent last) — context for continuity. "
        "The CURRENT request is in the task below; treat this as background, and "
        "honor any user corrections/preferences stated here. ===\n"
        + "\n".join(lines)
        + "\n=== end context ===\n\n"
    )


# ── migration from the old folder-level model ────────────────────────

def _import_session(user_id: str, project: str, folder: str, code: Path,
                    sessions: list[dict]) -> None:
    """Copy an old folder-level ``.code`` (chat/trace/state) into a new session."""
    chat = code / "chat.jsonl"
    if not chat.is_file() or not chat.read_text().strip():
        return
    sid = uuid.uuid4().hex[:8]
    sdir = _session_dir(user_id, project, sid)
    for fn in ("chat.jsonl", "trace.jsonl", "state.json"):
        src = code / fn
        if src.is_file():
            shutil.copy2(src, sdir / fn)
    sessions.append({"id": sid, "title": folder, "folder": folder,
                     "created": time.time(), "status": "idle"})


def _ensure_migrated(user_id: str, project: str) -> None:
    """Build ``project.json`` + sessions from the old model on first touch (idempotent)."""
    proj = _proj_dir(user_id, project)
    if (proj / ".code" / "project.json").is_file():
        return
    folders: list[dict] = []
    sessions: list[dict] = []
    subcode = sorted([s for s in proj.iterdir()
                      if s.is_dir() and s.name != ".code" and (s / ".code").is_dir()],
                     key=lambda p: p.name.lower()) if proj.is_dir() else []
    if subcode:
        for s in subcode:                        # container project → each sub is a vfs folder
            folders.append({"name": s.name, "kind": "vfs"})
            _import_session(user_id, project, s.name, s / ".code", sessions)
    else:                                        # legacy flat → the project dir itself is the repo
        folders.append({"name": project, "kind": "self"})
        if (proj / ".code").is_dir():
            _import_session(user_id, project, project, proj / ".code", sessions)
    if not sessions and folders:                 # always leave one session to open
        sessions.append({"id": uuid.uuid4().hex[:8], "title": "Session 1",
                         "folder": folders[0]["name"], "created": time.time(), "status": "idle"})
    _write_project(user_id, project, {"folders": folders})
    _write_sessions(user_id, project, sessions)


def _ensure_links_project(user_id: str) -> None:
    """Surface pre-existing global VFS links as folders of a ``linked`` project so
    they aren't lost in the move to per-project link membership."""
    links = read_links_at(_user_root(user_id))
    if not links:
        return
    proj = _proj_dir(user_id, "linked")
    if (proj / ".code" / "project.json").is_file():
        return
    folders = [{"name": name, "kind": "link", "link": name, "mode": ent.get("mode", "rw")}
               for name, ent in sorted(links.items())]
    _write_project(user_id, "linked", {"folders": folders})
    _write_sessions(user_id, "linked", [
        {"id": uuid.uuid4().hex[:8], "title": "Session 1",
         "folder": folders[0]["name"], "created": time.time(), "status": "idle"}] if folders else [])


def _discover_projects(user_id: str) -> list[str]:
    root = _user_root(user_id)
    names: list[str] = []
    if not root.is_dir():
        return names
    for d in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir():
            continue
        if (d / ".code").is_dir():
            names.append(d.name)
            continue
        try:
            if any(s.is_dir() and (s / ".code").is_dir() for s in d.iterdir()):
                names.append(d.name)
        except OSError:
            pass
    return names


# ── tree building ────────────────────────────────────────────────────

_STATS_SKIP = {".git", ".code", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}


def _count_files(root: Path, cap: int = 20000) -> int:
    n, stack = 0, [root]
    while stack and n < cap:
        try:
            for c in stack.pop().iterdir():
                if c.is_dir():
                    if c.name not in _STATS_SKIP:
                        stack.append(c)
                elif c.is_file():
                    n += 1
                    if n >= cap:
                        break
        except OSError:
            pass
    return n


def _project_tree(user_id: str, project: str) -> dict:
    folders_out = []
    for f in _read_project(user_id, project).get("folders", []):
        name = f.get("name", "")
        try:
            repo = _folder_repo(user_id, project, name)
            exists = repo.is_dir()
        except HTTPException:
            repo, exists = None, False
        folders_out.append({
            "name": name, "kind": f.get("kind", "vfs"),
            "linked": f.get("kind") == "link", "mode": f.get("mode", "rw"),
            "files": _count_files(repo) if (repo and exists) else 0, "missing": not exists,
        })
    sessions_out = []
    for s in _read_sessions(user_id, project):
        sdir = _proj_dir(user_id, project) / ".code" / "sessions" / safe_name(s.get("id", ""), fallback="s")
        chat = _read_chat(sdir) if sdir.is_dir() else []
        sessions_out.append({
            "id": s.get("id"), "title": s.get("title", "Session"), "folder": s.get("folder", ""),
            "messages": len(chat), "status": _read_state(sdir).get("status", "idle"),
            "last_message": (chat[-1]["text"][:120] if chat else ""), "created": s.get("created", 0),
        })
    return {"name": project, "folders": folders_out, "sessions": sessions_out}


# ── router (small vs big + archetype pick) ───────────────────────────

# ... orchestration below is model-agnostic: it takes a progress key (pkey), a
# repo dir (agent workspace / git), and a session dir (sdir) for chat/trace.

_TIER_FALLBACK = {"coding": "reason", "vision": "balanced"}


_GIT_ENV_CACHE: list[dict] | None = None


def _git_env() -> list[dict]:
    """Real-user git identity + config for spawned agents.

    Spawned agents run with ``HOME`` overridden to their own config dir, so a raw
    ``git commit``/``git push`` loses the user's identity, global config, and
    credential helper. Re-inject them: identity via ``GIT_AUTHOR_*``/
    ``GIT_COMMITTER_*``, the user's global config via ``GIT_CONFIG_GLOBAL`` (so
    credential.helper etc. still apply), and an SSH command that uses the real
    known_hosts and auto-accepts new hosts (SSH agent socket is inherited)."""
    global _GIT_ENV_CACHE
    if _GIT_ENV_CACHE is not None:
        return _GIT_ENV_CACHE
    import os
    import subprocess
    home = os.path.expanduser("~")
    env: list[dict] = []
    try:
        name = subprocess.run(["git", "config", "--get", "user.name"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        email = subprocess.run(["git", "config", "--get", "user.email"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        name = email = ""
    if name:
        env += [{"key": "GIT_AUTHOR_NAME", "value": name}, {"key": "GIT_COMMITTER_NAME", "value": name}]
    if email:
        env += [{"key": "GIT_AUTHOR_EMAIL", "value": email}, {"key": "GIT_COMMITTER_EMAIL", "value": email}]
    gc = Path(home) / ".gitconfig"
    if gc.is_file():
        env.append({"key": "GIT_CONFIG_GLOBAL", "value": str(gc)})
    kh = Path(home) / ".ssh" / "known_hosts"
    env.append({"key": "GIT_SSH_COMMAND",
                "value": f"ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile={kh}"})
    _GIT_ENV_CACHE = env
    return env


def _resolve_tcfg(tiers_map: dict, tier: str) -> dict:
    if tiers_map.get(tier):
        return tiers_map[tier]
    pref = _TIER_FALLBACK.get(tier, "balanced")
    return (tiers_map.get(pref) or tiers_map.get("balanced")
            or tiers_map.get("reason") or next(iter(tiers_map.values()), {}))


async def _classify(intent: str, context: str, archetypes: list[dict],
                    tiers_map: dict, registry: dict) -> dict:
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
    except Exception as e:  # noqa: BLE001
        log.warning("code router LLM failed; keyword fallback", error=str(e))
        raw = None
    if not isinstance(raw, dict) or "size" not in raw:
        low = intent.lower()
        breadth = len(_score_archetypes(intent, archetypes))
        difficulty = _fallback_difficulty(intent, breadth)
        is_bug = any(w in low for w in ("bug", "error", "crash", "fix", "broken", "fails"))
        is_git = any(w in low for w in _GIT_WORDS)
        raw = {
            "size": "small" if is_git else ("big" if difficulty == "hard" else "small"),
            "planner": "architect" if difficulty == "hard" else "light-planner",
            "small_archetype": _GIT if is_git else ("debugger" if is_bug else "code-implementer"),
            "domain": "general", "difficulty": difficulty, "title": intent[:48], "why": "keyword fallback",
        }
    raw["size"] = "big" if str(raw.get("size")).lower() == "big" else "small"
    if raw.get("planner") not in _PLANNERS or raw["planner"] not in by_id:
        raw["planner"] = "light-planner" if "light-planner" in by_id else "architect"
    if raw.get("small_archetype") not in _SMALL or raw["small_archetype"] not in by_id:
        raw["small_archetype"] = "code-implementer"
    # A git op is inherently small even if the LLM oversized it.
    if raw.get("small_archetype") == _GIT:
        raw["size"] = "small"
    return raw


# ── agent execution ──────────────────────────────────────────────────

def _codemap_preamble(repo: Path) -> str:
    """Prepend a Code Map hint (+ overview excerpt) so agents query the map
    instead of reading the whole tree. Empty when the repo isn't mapped yet."""
    try:
        st = code_map.stats(repo)
        if not st.get("symbols"):
            return ""
        ov = code_map.read_overview(repo).strip()
        head = (
            "## Code Map available (query it before reading files)\n"
            f"This repo is indexed ({st['files']} files, {st['symbols']} symbols). Use the "
            "`codemap` tool to locate things fast: `overview`, `search <query>`, "
            "`symbol <name>`, `file <path>`, `models`, `ui`. It returns file:line "
            "pointers — read the actual file only when you need the source.\n"
            "READ DISCIPLINE: read specific line RANGES (offset/limit) at the "
            "pointers codemap gives you — do not re-read whole files you already "
            "have in context (identical full re-reads are short-circuited).\n"
        )
        if ov:
            head += "\n### Architecture overview\n" + ov[:2500] + "\n"
        return head + "\n---\n\n"
    except Exception:  # noqa: BLE001
        return ""


_ESCALATE_MARK = "ESCALATE"
_ESCALATE_RE = re.compile(r"(?mi)^\s*ESCALATE\s*[:\-]\s*(.+)$")
_ESCALATE_DIRECTIVE = (
    "\n\nSCOPE CHECK: you were routed as a QUICK edit. If, once you look, this is "
    "actually substantial — many files, an architectural or cross-cutting change, or "
    "something you cannot finish cleanly in a focused edit — do NOT half-do it and "
    "burn your budget. STOP and reply with a single line `ESCALATE: <one-sentence "
    "reason>` (optionally list the files/areas involved). That hands the task to the "
    "full plan→build→review pipeline. Only escalate when it's genuinely bigger than a "
    "quick edit; for a normal small fix, just do it."
)


def _exec_prompt(intent: str) -> str:
    return (
        "You are working inside a real project directory — it IS your workspace and "
        "current working directory, a git repo. Create and edit files with PLAIN "
        "RELATIVE paths (e.g. `src/main.py`); use your shell to install deps, run, and "
        "verify. Do NOT use any `vfs:` prefix — just work in the directory you're in.\n\n"
        f"Task:\n{intent}\n\n"
        "When finished, briefly summarize what you created/changed and how you "
        "verified it actually runs." + _ESCALATE_DIRECTIVE + _REPORTS_DIRECTIVE
    )


def _should_escalate(d: dict | None) -> tuple[bool, str]:
    """Decide whether a small-path run should be promoted to the full pipeline.

    Escalate when the agent explicitly asked (``ESCALATE: reason``), the run
    errored, or it ran out of its iteration budget before finishing — all mean
    a lone quick-edit agent couldn't land the change and a plan→build→review
    pass is warranted. Returns (escalate, human_reason).
    """
    if not d:
        return True, "the quick-edit agent returned no result."
    out = (d.get("output") or "").strip()
    m = _ESCALATE_RE.search(out)
    if m:
        return True, m.group(1).strip()[:300]
    if not d.get("ok"):
        return True, f"the quick edit failed: {d.get('error', 'unknown error')}"
    low = out.lower()
    if ("iteration budget" in low or "wasn't able to fully complete" in low
            or "unable to complete this request" in low or "couldn't fully complete" in low):
        return True, "the quick-edit agent ran out of its iteration budget before finishing."
    return False, ""


def _git_prompt(intent: str) -> str:
    return (
        "You are performing a git / version-control operation in THIS repository "
        "(your workspace and current directory). Use your shell to run `git`. Do "
        "EXACTLY what is asked and nothing more. Inspect state first (`git status`, "
        "`git diff`); for commits write a clear message derived from the real diff. "
        "Never force-push or rewrite published history unless explicitly told; if a "
        "remote/auth isn't configured or a push is rejected, report it plainly.\n\n"
        f"Request:\n{intent}\n\n"
        "Report the commands you ran and the resulting state (commit sha + subject, "
        "branch, push result)."
    )


async def _run_agent(request: Request, user: dict, pkey: str, repo: Path,
                     archetype_id: str, prompt: str, by_id: dict,
                     tiers_map: dict, env_vars: list) -> dict:
    """Spawn one archetype anchored at ``repo``, dispatch ``prompt``, dispose."""
    src = by_id[archetype_id]
    # Every code agent gets the codemap tool (query the repo's blackboard).
    tools = list(src.get("tools") or [])
    if "codemap" not in tools:
        tools = tools + ["codemap"]
    arch = {**src, "tools": tools}
    role = arch.get("role", archetype_id)
    tier = arch.get("tier", "coding")
    tcfg = _resolve_tcfg(tiers_map, tier)
    suffix = uuid.uuid4().hex[:6]
    prompt = _codemap_preamble(repo) + prompt

    def _on_action(act: dict) -> None:
        if act.get("tool") == "narration":
            _progress(pkey, "narration", f"{role}: {act.get('detail', '')}",
                      agent=role, tool="narration", detail=act.get("detail", ""))
        else:
            detail = f": {act['detail']}" if act.get("detail") else ""
            _progress(pkey, "action", f"{role} → {act.get('tool')}{detail}",
                      agent=role, tool=act.get("tool"), detail=act.get("detail", ""))

    def _on_usage(pt: int, ct: int, tt: int) -> None:
        _usage_add(pkey, pt, ct)
        _progress(pkey, "usage", f"{role} · {pt:,}→{ct:,} tok",
                  agent=role, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

    _phase(pkey, f"{role} working")
    # CLAW_WRITE_DIRECT: the write tool otherwise sandboxes every write into
    # saved/tmp/<session>/; code agents must write real files into the repo.
    # CLAW_CODE_AGENT: latches off the agent's list/scale contract machinery
    # (list extraction, scale advisory, task rephrase, coverage gate) — code
    # agents are orchestrated externally and a numbered fix prompt otherwise
    # gets parsed as "list members" the completion gate then blocks on
    # forever (SENKO2 stuck-loop post-mortem).
    # Plus the real-user git identity/config so the agent's own git (esp. the
    # git-operator's commit/push) works despite the spawned HOME override.
    code_env = [
        {"key": "CLAW_WRITE_DIRECT", "value": "1"},
        {"key": "CLAW_CODE_AGENT", "value": "1"},
    ] + _git_env()
    port, token, slug = await spawn_archetype_agent(
        arch, tier, tcfg, request, user, name_suffix=suffix,
        env_vars=list(env_vars or []) + code_env, workspace_path=str(repo),
    )
    try:
        d = await _dispatch_one(
            port, token, prompt, _DISPATCH_TIMEOUT, on_action=_on_action,
            fleet_instructions=arch.get("fleet_instructions", ""),
            agent_name=role, on_usage=_on_usage,
        )
    finally:
        try:
            await stop_archetype_agent(slug)
        except Exception as e:  # noqa: BLE001
            log.warning("code: failed to stop agent", slug=slug, error=str(e))
    return d


async def _run_cartographer(request: Request, user: dict, pkey: str, repo: Path,
                            by_id: dict, tiers_map: dict, env_vars: list, registry: dict) -> dict | None:
    """Refresh the skeleton, then have the cartographer write the semantic layer."""
    if _CARTOGRAPHER not in by_id:
        return None
    _phase(pkey, "Mapping the codebase")
    await _update_map(repo, tiers_map, registry)
    return await _run_agent(
        request, user, pkey, repo, _CARTOGRAPHER,
        "Build or refresh this repository's Code Map. Use the codemap tool to write a "
        "concise architecture overview (set_overview), the data-models map (set_models), "
        "and the UI map (set_ui). Summarize — don't dump code.",
        by_id, tiers_map, env_vars)


def _plan_prompt(intent: str, plan_rel: str = "plan.md") -> str:
    return (
        "You are planning a coding task in THIS repository — it is your workspace and "
        "current directory. Survey the existing code first (relative paths, your shell), "
        f"then produce a clear, scoped implementation plan and WRITE it to `{plan_rel}` "
        "(create the parent folder if needed). Use exactly that path — do NOT write "
        "`plan.md` or any other filename. The plan drives an implementer next, so make it "
        "concrete and ordered. Do NOT write any other code yet.\n\n"
        f"Task:\n{intent}"
    )


def _build_prompt(intent: str, plan_rel: str = "plan.md") -> str:
    return (
        f"An implementation plan has been approved and saved as `{plan_rel}` in THIS "
        "repository (your workspace and current directory). Read that plan file, then "
        "implement it fully: create/edit files with plain relative paths, install deps and "
        "run/verify via your shell. Follow the plan; if you must deviate, say why. Do NOT "
        "use any `vfs:` prefix.\n\n"
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


_DELTA_REVIEW_PROMPTS = {
    "code-reviewer": (
        "DELTA review — a fix round just landed in this repository. Below are the prior "
        "findings and the diff of the fix commit. Your job: (1) verify each prior finding "
        "is actually fixed, (2) review ONLY the changed hunks for new issues the fix may "
        "have introduced. Do NOT re-review unchanged code — your earlier full review "
        "stands. Read-only; report per-finding fixed/not-fixed plus any NEW issues with "
        "file:line."
    ),
    "security-reviewer": (
        "DELTA security review — a fix round just landed. Below are the prior findings "
        "and the diff of the fix commit. Verify the security-relevant fixes and check "
        "ONLY the changed hunks for newly introduced security issues. Do NOT re-audit "
        "unchanged code. Read-only; report fixed/not-fixed + any NEW findings."
    ),
    "qa-engineer": (
        "DELTA QA — a fix round just landed. RE-RUN the existing test suite via your "
        "shell and report pass/fail. Add tests ONLY for the just-fixed findings if they "
        "lack coverage. Do NOT write a fresh full assessment and do NOT rewrite the "
        "suite — your earlier report stands for unchanged code."
    ),
}

# Findings/fix-instructions that mention any of these keep the security
# reviewer in delta rounds; otherwise it sits those rounds out (its full
# round-0 audit stands for unchanged code).
_SECURITY_HINT_RE = re.compile(
    r"(?i)secur|xss|inject|csp\b|auth|secret|credential|sanitiz|escap|csrf|"
    r"vuln|cve-|prototype|eval\(|innerhtml")


def _review_prompt(reviewer: str, intent: str, *, rnd: int = 0,
                   diff: str = "", prior_findings: list | None = None) -> str:
    """Round 0: full review. Rounds 1+: delta review of the fix commit only."""
    if rnd == 0 or not diff:
        return f"{_REVIEW_PROMPTS[reviewer]}\n\nTask under review:\n{intent}" + _REPORTS_DIRECTIVE
    prior = "\n".join(
        f"- [{f.get('severity', '?')}] {f.get('title', '')} ({f.get('file', '') or 'n/a'})"
        for f in (prior_findings or [])
    ) or "(none recorded)"
    return (
        f"{_DELTA_REVIEW_PROMPTS[reviewer]}\n\n"
        f"Task under review:\n{intent}\n\n"
        f"Prior findings (round {rnd - 1}):\n{prior}\n\n"
        f"Diff of the fix commit:\n```diff\n{diff}\n```" + _REPORTS_DIRECTIVE
    )


def _select_reviewers(rnd: int, prior_findings: list | None, fix_instructions: str) -> list[str]:
    """All three on the full round; delta rounds drop the security reviewer
    unless the prior findings / fix instructions touch a security surface."""
    if rnd == 0:
        return list(_REVIEWERS)
    blob = fix_instructions + " " + " ".join(
        f"{f.get('title', '')} {f.get('file', '')}" for f in (prior_findings or []))
    keep_security = bool(_SECURITY_HINT_RE.search(blob))
    return [rv for rv in _REVIEWERS if keep_security or rv != "security-reviewer"]


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
    sys_file = _INSTRUCTIONS_DIR / "code" / "triage.md"
    parts = [f"## {r['role']} report\n{r['output'] or '(no output)'}" for r in reviews]
    user_prompt = f"Task:\n{intent}\n\n" + "\n\n".join(parts)
    tier = _resolve_tcfg(tiers_map, "reason") or registry.get("tiers", {}).get("reason", {})
    try:
        from captain_claw.llm import Message, create_provider
        prov = create_provider(
            provider=tier.get("provider", "anthropic"), model=tier.get("model", ""),
            api_key=tier.get("api_key") or None, base_url=tier.get("base_url") or None,
            temperature=0.1, max_tokens=4000,
        )
        resp = await prov.complete(messages=[
            Message(role="system", content=sys_file.read_text()),
            Message(role="user", content=user_prompt),
        ], temperature=0.1, max_tokens=4000)
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        raw = json.loads(content)
    except Exception as e:  # noqa: BLE001
        log.warning("code triage failed", error=str(e))
        return {"needs_fix": False, "fixer": "code-implementer",
                "summary": "Review complete (triage unavailable — not auto-fixing).",
                "fix_instructions": "", "findings": []}
    raw["needs_fix"] = bool(raw.get("needs_fix"))
    if raw.get("fixer") not in ("debugger", "code-implementer"):
        raw["fixer"] = "code-implementer"
    return raw


_BACKLOG_REPORT = "backlog"


def _write_backlog(repo: Path, triage: dict) -> str:
    """Persist the open findings at the fix-round cap so a follow-up turn can
    resume exactly where the loop stopped."""
    lines = ["# Fix backlog — open findings at the fix-round cap", ""]
    for f in triage.get("findings", []):
        lines.append(f"- [{f.get('severity', '?')}] {f.get('title', '')}"
                     + (f" ({f['file']})" if f.get("file") else ""))
    fi = (triage.get("fix_instructions") or "").strip()
    if fi:
        lines += ["", "## Fix instructions (from triage)", "", fi]
    return _write_report(repo, _BACKLOG_REPORT, "\n".join(lines))


def _backlog_path(repo: Path) -> Path:
    return repo / _REPORTS_DIRNAME / f"{_BACKLOG_REPORT}.md"


async def _fix_commit_diff(repo: Path, fsha: str | None, cap: int = 12000) -> str:
    """The fix commit's patch, truncated — the delta reviewers' whole world."""
    if not fsha:
        return ""
    try:
        patch = await code_git.git_show(repo, fsha)
    except Exception:  # noqa: BLE001
        return ""
    if len(patch) > cap:
        patch = patch[:cap] + f"\n… (truncated at {cap} chars — use `git show {fsha[:10]}` for the rest)"
    return patch


async def _run_build_loop(request: Request, user: dict, pkey: str, repo: Path, sdir: Path,
                          intent: str, by_id: dict, tiers_map: dict,
                          env_vars: list, registry: dict, plan_file: str = "plan.md",
                          seed_fix: str = "") -> None:
    """Big-job pipeline: build → review fan-out → capped fix loop (background).

    ``seed_fix``: backlog continuation — skip the build, apply the seeded fix
    instructions first, then run delta review rounds on the fix diff.
    """
    _progress_start(pkey)
    _usage_reset(pkey)
    _write_state(sdir, {"status": "running"})
    try:
        last_fix_sha: str | None = None
        if seed_fix:
            _phase(pkey, "Fixing (backlog)")
            fd = await _run_agent(request, user, pkey, repo, "debugger",
                                  _fix_prompt(intent, seed_fix), by_id, tiers_map, env_vars)
            last_fix_sha = await code_git.git_commit(repo, "[fix backlog] debugger")
            _append_chat(sdir, "assistant", (fd.get("output") or "(fix produced no summary)").strip(),
                         kind="fix", archetype="debugger", ok=bool(fd.get("ok")),
                         commit=last_fix_sha or "")
        else:
            _phase(pkey, "Building")
            d = await _run_agent(request, user, pkey, repo, "code-implementer",
                                 _build_prompt(intent, plan_file), by_id, tiers_map, env_vars)
            sha = await code_git.git_commit(repo, f"[build] code-implementer: {intent[:60]}")
            _append_chat(sdir, "assistant", (d.get("output") or "(build produced no summary)").strip(),
                         kind="build", archetype="code-implementer", ok=bool(d.get("ok")), commit=sha or "")

        prior_findings: list = []
        prior_fix_instructions: str = seed_fix
        for rnd in range(_MAX_FIX_ROUNDS + 1):
            # Round 0 after a fresh build = full review of everything.
            # Rounds 1+ (and round 0 of a backlog continuation) = DELTA
            # review: verify prior findings + scan the fix diff only —
            # re-reviewing the whole repo every round was 53% of all
            # tokens on the SW3 run.
            diff_text = ""
            if rnd > 0 or seed_fix:
                diff_text = await _fix_commit_diff(repo, last_fix_sha)
            delta = bool(diff_text)          # no diff → fall back to a full review
            eff_rnd = max(rnd, 1) if delta else 0
            reviewers = _select_reviewers(eff_rnd, prior_findings, prior_fix_instructions)
            _phase(pkey, "Reviewing" + (" (delta)" if delta else ""))
            reviews_raw = await asyncio.gather(*[
                _run_agent(request, user, pkey, repo, rv,
                           _review_prompt(rv, intent, rnd=eff_rnd,
                                          diff=diff_text, prior_findings=prior_findings),
                           by_id, tiers_map, env_vars)
                for rv in reviewers
            ], return_exceptions=True)
            reviews = []
            for rv, res in zip(reviewers, reviews_raw):
                role = by_id[rv].get("role", rv)
                out = "" if isinstance(res, Exception) else (res.get("output") or "")
                reviews.append({"role": role, "id": rv, "output": out})
                if out.strip():
                    _write_report(repo, f"review-r{rnd}-{rv}", f"# {role} — review r{rnd}\n\n{out}")

            triage = await _triage_reviews(reviews, intent, tiers_map, registry)
            review_summary = triage.get("summary", "Review complete.")
            _write_report(repo, f"review-r{rnd}-summary", f"# Review summary — r{rnd}\n\n{review_summary}")
            await code_git.git_commit(repo, f"[review r{rnd}] reports + reviewer tests")
            _append_chat(sdir, "assistant", review_summary, kind="review", round=rnd,
                         findings=triage.get("findings", []), needs_fix=triage["needs_fix"])
            prior_findings = triage.get("findings", []) or []
            prior_fix_instructions = triage.get("fix_instructions", "") or ""

            if not triage["needs_fix"]:
                # Clean pass — a stale backlog from an earlier capped run is done.
                _backlog_path(repo).unlink(missing_ok=True)
                break
            if rnd == _MAX_FIX_ROUNDS:
                rel = _write_backlog(repo, triage)
                _append_chat(sdir, "assistant",
                             f"Reached the fix-round cap ({_MAX_FIX_ROUNDS}). Open findings "
                             f"saved to `{rel}` — say **continue fixing** to resume from "
                             "the backlog.", kind="note")
                break

            _phase(pkey, f"Fixing (round {rnd + 1})")
            fixer = triage["fixer"]
            fd = await _run_agent(request, user, pkey, repo, fixer,
                                  _fix_prompt(intent, triage.get("fix_instructions", "")),
                                  by_id, tiers_map, env_vars)
            fsha = await code_git.git_commit(repo, f"[fix r{rnd + 1}] {fixer}")
            last_fix_sha = fsha or last_fix_sha
            _append_chat(sdir, "assistant", (fd.get("output") or "(fix produced no summary)").strip(),
                         kind="fix", round=rnd + 1, archetype=fixer,
                         ok=bool(fd.get("ok")), commit=fsha or "")

        # The repo just changed substantially — (re)build the code map so future
        # sessions/tasks can query it instead of re-reading everything.
        try:
            await _run_cartographer(request, user, pkey, repo, by_id, tiers_map, env_vars, registry)
        except Exception as e:  # noqa: BLE001
            log.warning("cartographer after build failed", error=str(e))
        _u = _usage_summary(pkey)
        if _u:
            _append_chat(sdir, "assistant", f"_Run total: {_u}._", kind="note")
        _write_state(sdir, {"status": "idle"})
    except Exception as e:  # noqa: BLE001
        log.warning("code build loop failed", pkey=pkey, error=str(e))
        _append_chat(sdir, "assistant", f"⚠️ Build loop failed: {e}", kind="note", ok=False)
        _write_state(sdir, {"status": "idle"})
    finally:
        _persist_trace(pkey, sdir, f"Build → review → fix: {intent[:80]}")
        _progress_done(pkey)


def _export_markdown(sdir: Path, repo: Path, title: str) -> str:
    import time as _t

    def ts(v: float) -> str:
        try:
            return _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(v))
        except (ValueError, OSError):
            return ""

    lines = [f"# Coding process — {title}", f"_Exported {ts(_t.time())}_", "", "## Conversation\n"]
    for m in _read_chat(sdir):
        who = "🧑 User" if m.get("role") == "user" else f"🤖 {m.get('archetype') or 'assistant'}"
        meta = " · ".join(x for x in [m.get("kind"), m.get("size"), (m.get("commit") or "")[:7]] if x)
        lines.append(f"### {who}{(' — ' + meta) if meta else ''}  ·  {ts(m.get('ts', 0))}")
        lines.append((m.get("text") or "").rstrip())
        for f in (m.get("findings") or []):
            lines.append(f"- **{f.get('severity', '')}** · {f.get('title', '')}"
                         + (f" ({f['file']})" if f.get("file") else ""))
        lines.append("")
    trace = _read_trace(sdir)
    if trace:
        lines.append("## Tool & narration trace\n")
        for rec in trace:
            if rec.get("type") == "run":
                lines.append(f"\n### ▶ {rec.get('label', 'run')}  ·  {ts(rec.get('ts', 0))}\n")
                continue
            if rec.get("stage") == "usage":
                continue
            lines.append(f"- `{ts(rec.get('ts', 0))}`  {rec.get('message', '')}")
        lines.append("")
    rd = repo / _REPORTS_DIRNAME
    if rd.is_dir():
        reports = sorted(f for f in rd.glob("*.md") if f.is_file())
        if reports:
            lines.append("## Reports\n")
            for rf in reports:
                lines.append(f"<details><summary>{rf.name}</summary>\n")
                lines.append(rf.read_text().rstrip())
                lines.append("\n</details>\n")
    return "\n".join(lines) + "\n"


# ── request models ───────────────────────────────────────────────────

class CreateProjectReq(BaseModel):
    name: str


class NewFolderReq(BaseModel):
    folder: str


class LinkFolderReq(BaseModel):
    name: str                 # folder name within the project
    link: str = ""            # existing VFS link name to reference…
    path: str = ""            # …or create a new link at this absolute path
    mode: str = "rw"


class NewSessionReq(BaseModel):
    title: str = ""
    folder: str = ""


class MessageReq(BaseModel):
    project: str
    session: str
    text: str


class ApproveReq(BaseModel):
    project: str
    session: str
    plan: str | None = None


class CancelReq(BaseModel):
    project: str
    session: str


class RollbackReq(BaseModel):
    ref: str


# ── project / folder / session endpoints ─────────────────────────────

@router.get("/projects")
async def list_projects(user: dict = Depends(get_current_user)):
    """Full tree: each project with its folders and sessions."""
    uid = user["id"]
    _ensure_links_project(uid)
    out = []
    for name in _discover_projects(uid):
        _ensure_migrated(uid, name)
        out.append(_project_tree(uid, name))
    out.sort(key=lambda p: p["name"].lower())
    return {"projects": out}


@router.post("/projects")
async def create_project(body: CreateProjectReq, user: dict = Depends(get_current_user)):
    name = safe_name(body.name, fallback="")
    if not name:
        raise HTTPException(400, "invalid project name")
    proj = _proj_dir(user["id"], name)
    if (proj / ".code" / "project.json").is_file():
        raise HTTPException(409, "project already exists")
    proj.mkdir(parents=True, exist_ok=True)
    _write_project(user["id"], name, {"folders": []})
    _write_sessions(user["id"], name, [])
    return {"project": _project_tree(user["id"], name)}


@router.post("/projects/{project}/folders")
async def add_folder(project: str, body: NewFolderReq, user: dict = Depends(get_current_user)):
    """Create a new VFS folder (git repo) inside a project."""
    uid = user["id"]
    _ensure_migrated(uid, project)
    folder = safe_name(body.folder, fallback="")
    if not folder:
        raise HTTPException(400, "folder name required")
    data = _read_project(uid, project)
    if any(f.get("name") == folder for f in data["folders"]):
        raise HTTPException(409, "folder already exists in project")
    repo = (_proj_dir(uid, project) / folder).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    await code_git.git_init(repo)
    data["folders"].append({"name": folder, "kind": "vfs"})
    _write_project(uid, project, data)
    return {"project": _project_tree(uid, project)}


@router.post("/projects/{project}/link")
async def add_link_folder(project: str, body: LinkFolderReq, user: dict = Depends(get_current_user)):
    """Add a linked external folder to a project. Either references an existing
    VFS link (``link``) or creates one from ``path``. The same link may be added
    to multiple projects."""
    uid = user["id"]
    _ensure_migrated(uid, project)
    name = safe_name(body.name, fallback="")
    if not name:
        raise HTTPException(400, "folder name required")
    link_name = safe_name(body.link, fallback="") or name
    # Create the VFS link if a path was given (reuse the vfs_routes admin path).
    if body.path.strip():
        from captain_claw.flight_deck.vfs_routes import LinkBody, add_link
        await add_link(LinkBody(name=link_name, path=body.path, mode=body.mode), user=user)
    if link_target_at(_user_root(uid), link_name) is None:
        raise HTTPException(404, f"no VFS link named '{link_name}'")
    data = _read_project(uid, project)
    if any(f.get("name") == name for f in data["folders"]):
        raise HTTPException(409, "folder already exists in project")
    data["folders"].append({"name": name, "kind": "link", "link": link_name, "mode": body.mode})
    _write_project(uid, project, data)
    return {"project": _project_tree(uid, project)}


@router.delete("/projects/{project}/folders/{folder}")
async def remove_folder(project: str, folder: str, user: dict = Depends(get_current_user)):
    """Remove a folder from a project (drops membership only — never deletes a
    linked external folder; a VFS folder's files stay on disk)."""
    uid = user["id"]
    data = _read_project(uid, project)
    data["folders"] = [f for f in data["folders"] if f.get("name") != folder]
    _write_project(uid, project, data)
    return {"project": _project_tree(uid, project)}


@router.post("/projects/{project}/sessions")
async def create_session(project: str, body: NewSessionReq, user: dict = Depends(get_current_user)):
    uid = user["id"]
    _ensure_migrated(uid, project)
    folders = _read_project(uid, project).get("folders", [])
    folder = body.folder or (folders[0]["name"] if folders else "")
    if not folder or not _folder_meta(uid, project, folder):
        raise HTTPException(400, "a valid folder is required to start a session")
    sessions = _read_sessions(uid, project)
    sid = uuid.uuid4().hex[:8]
    sess = {"id": sid, "title": body.title.strip() or f"Session {len(sessions) + 1}",
            "folder": folder, "created": time.time(), "status": "idle"}
    sessions.append(sess)
    _write_sessions(uid, project, sessions)
    _session_dir(uid, project, sid)
    return {"project": _project_tree(uid, project), "session": sess}


@router.delete("/projects/{project}/sessions/{session}")
async def delete_session(project: str, session: str, user: dict = Depends(get_current_user)):
    uid = user["id"]
    _write_sessions(uid, project, [s for s in _read_sessions(uid, project) if s.get("id") != session])
    sdir = _proj_dir(uid, project) / ".code" / "sessions" / safe_name(session, fallback="s")
    if sdir.is_dir():
        shutil.rmtree(sdir, ignore_errors=True)
    return {"project": _project_tree(uid, project)}


class SetFolderReq(BaseModel):
    folder: str


@router.put("/projects/{project}/sessions/{session}/folder")
async def set_session_folder(project: str, session: str, body: SetFolderReq,
                             user: dict = Depends(get_current_user)):
    uid = user["id"]
    if not _folder_meta(uid, project, body.folder):
        raise HTTPException(404, "folder not in project")
    sessions = _read_sessions(uid, project)
    sess = next((s for s in sessions if s.get("id") == session), None)
    if not sess:
        raise HTTPException(404, "session not found")
    sess["folder"] = body.folder
    _write_sessions(uid, project, sessions)
    return {"project": _project_tree(uid, project)}


# ── session-scoped context helper ────────────────────────────────────

def _sctx(user_id: str, project: str, session: str):
    """Return (session, repo, sdir, pkey) for a session, or 404."""
    _ensure_migrated(user_id, project)
    sess = _sget(user_id, project, session)
    if not sess:
        raise HTTPException(404, "session not found")
    repo = _folder_repo(user_id, project, sess.get("folder", ""))
    sdir = _session_dir(user_id, project, session)
    return sess, repo, sdir, f"{project}/{session}"


@router.get("/projects/{project}/sessions/{session}/chat")
async def get_chat(project: str, session: str, user: dict = Depends(get_current_user)):
    _sess, _repo, sdir, _pk = _sctx(user["id"], project, session)
    return {"messages": _read_chat(sdir), "state": _read_state(sdir)}


@router.get("/projects/{project}/sessions/{session}/progress")
async def get_progress(project: str, session: str, user: dict = Depends(get_current_user)):
    p = _PROGRESS.get(f"{project}/{session}")
    return p if p is not None else {"events": [], "active": False}


@router.get("/projects/{project}/sessions/{session}/log")
async def get_log(project: str, session: str, user: dict = Depends(get_current_user)):
    _sess, repo, _sdir, _pk = _sctx(user["id"], project, session)
    await code_git.git_init(repo)   # self-heal / init the target repo
    return {"commits": await code_git.git_log(repo)}


@router.get("/projects/{project}/sessions/{session}/diff")
async def get_diff(project: str, session: str, ref_a: str = "", ref_b: str = "",
                   user: dict = Depends(get_current_user)):
    _sess, repo, _sdir, _pk = _sctx(user["id"], project, session)
    return {"diff": await code_git.git_diff(repo, ref_a, ref_b)}


@router.get("/projects/{project}/sessions/{session}/show")
async def show_commit(project: str, session: str, sha: str, user: dict = Depends(get_current_user)):
    _sess, repo, _sdir, _pk = _sctx(user["id"], project, session)
    return {"diff": await code_git.git_show(repo, sha)}


@router.post("/projects/{project}/sessions/{session}/rollback")
async def rollback(project: str, session: str, body: RollbackReq,
                   user: dict = Depends(get_current_user)):
    _sess, repo, sdir, _pk = _sctx(user["id"], project, session)
    if _read_state(sdir).get("status") == "running":
        raise HTTPException(409, "a build is running — wait for it to finish")
    target = next((c for c in await code_git.git_log(repo) if c["sha"].startswith(body.ref)), None)
    await code_git.git_reset(repo, body.ref)
    _append_chat(sdir, "assistant", f"↩ Rolled back to {body.ref[:7]}"
                 + (f": {target['message']}" if target else ""), kind="note")
    return {"ok": True, "head": (await code_git.git_log(repo, 1))[:1]}


@router.get("/projects/{project}/sessions/{session}/export")
async def export_process(project: str, session: str, format: str = "md",
                         user: dict = Depends(get_current_user)):
    from fastapi.responses import JSONResponse, PlainTextResponse
    sess, repo, sdir, _pk = _sctx(user["id"], project, session)
    title = f"{project} / {sess.get('title', 'session')}"
    fname = safe_name(f"{project}-{sess.get('title', 'session')}", fallback="process")
    if format == "json":
        data = {"project": project, "session": sess, "exported_at": time.time(),
                "messages": _read_chat(sdir), "trace": _read_trace(sdir)}
        return JSONResponse(data, headers={
            "Content-Disposition": f'attachment; filename="{fname}-process.json"'})
    return PlainTextResponse(
        _export_markdown(sdir, repo, title), media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{fname}-process.md"'})


# ── code map ─────────────────────────────────────────────────────────

@router.get("/projects/{project}/sessions/{session}/map")
async def get_map(project: str, session: str, user: dict = Depends(get_current_user)):
    """The session folder's Code Map — overview, models, ui, stats."""
    _sess, repo, _sdir, _pk = _sctx(user["id"], project, session)
    return {"overview": code_map.read_overview(repo),
            "models": code_map.read_json_layer(repo, "models"),
            "ui": code_map.read_json_layer(repo, "ui"),
            "stats": code_map.stats(repo)}


@router.get("/projects/{project}/sessions/{session}/map/search")
async def map_search(project: str, session: str, q: str = "",
                     user: dict = Depends(get_current_user)):
    _sess, repo, _sdir, _pk = _sctx(user["id"], project, session)
    return {"results": code_map.search(repo, q) if q.strip() else []}


@router.post("/projects/{project}/sessions/{session}/map/build")
async def map_build(project: str, session: str, request: Request,
                    user: dict = Depends(get_current_user)):
    """(Re)build the map in the background: reindex + cartographer."""
    uid = user["id"]
    sess, repo, sdir, pkey = _sctx(uid, project, session)
    if _read_state(sdir).get("status") == "running":
        raise HTTPException(409, "a run is in progress")
    db = get_db()
    archetypes = await merged_archetypes(db, uid)
    by_id = {a["id"]: a for a in archetypes}
    registry = _load_registry()
    tiers_map, env_vars = await _load_owner_tiers(db, uid)

    # Flip to running SYNCHRONOUSLY before returning so the client can poll
    # /progress immediately without racing the background task's startup
    # (otherwise the first poll can read idle+inactive and stop at once).
    _progress_start(pkey)
    _usage_reset(pkey)
    _write_state(sdir, {"status": "running"})
    _phase(pkey, "Mapping the codebase")

    async def _bg():
        try:
            await _run_cartographer(request, user, pkey, repo, by_id, tiers_map, env_vars, registry)
        except Exception as e:  # noqa: BLE001
            log.warning("map build failed", error=str(e))
        finally:
            _write_state(sdir, {"status": "idle"})
            _progress_done(pkey)

    asyncio.create_task(_bg())
    return {"status": "running"}


# ── the main message + approve flow ──────────────────────────────────

@router.post("/message")
async def message(body: MessageReq, request: Request, user: dict = Depends(get_current_user)):
    intent = body.text.strip()
    if not intent:
        raise HTTPException(400, "text is required")
    uid = user["id"]
    sess, repo, sdir, pkey = _sctx(uid, body.project, body.session)
    await code_git.git_init(repo)

    db = get_db()
    archetypes = await merged_archetypes(db, uid)
    by_id = {a["id"]: a for a in archetypes}
    registry = _load_registry()
    tiers_map, env_vars = await _load_owner_tiers(db, uid)

    _append_chat(sdir, "user", intent)

    # ── Backlog continuation → resume the fix loop (bypass the router). ──
    # Checked BEFORE the try/finally below: the background loop owns
    # state/progress/trace for this turn, so message() must not run its
    # finally-block progress_done/persist against it.
    if _is_backlog_intent(intent) and _backlog_path(repo).is_file():
        seed = _backlog_path(repo).read_text()
        _write_state(sdir, {"status": "running"})
        asyncio.create_task(_run_build_loop(
            request, user, pkey, repo, sdir, intent, by_id, tiers_map,
            env_vars, registry, seed_fix=seed))
        return {"status": "running"}

    _progress_start(pkey)
    _usage_reset(pkey)
    _write_state(sdir, {"status": "running"})
    try:
        # ── Map/index request → the cartographer (bypass the router). ──
        if _is_map_intent(intent):
            d = await _run_cartographer(request, user, pkey, repo, by_id, tiers_map, env_vars, registry)
            st = code_map.stats(repo)
            body_out = (d.get("output") or "Code map refreshed.").strip() if d else \
                "Code map cartographer is unavailable."
            body_out += f"\n\n_Map: {st['files']} files · {st['symbols']} symbols · {st['summarized']} summarized._"
            _u = _usage_summary(pkey)
            body_out += f"\n\n_{_u}._" if _u else ""
            assistant = _append_chat(sdir, "assistant", body_out, archetype=_CARTOGRAPHER, kind="map")
            _write_state(sdir, {"status": "idle"})
            return {"message": assistant}

        _phase(pkey, "Routing")
        prior = _read_chat(sdir)[-8:]
        context = "\n".join(f"{m['role']}: {m['text'][:200]}" for m in prior[:-1])
        route = await _classify(intent, context, archetypes, tiers_map, registry)
        _progress(pkey, "route", f"size={route['size']} · {route.get('why', '')}", size=route["size"])

        hist = _history_preamble(sdir)
        if route["size"] == "small":
            executor = route["small_archetype"]
            is_git = executor == _GIT
            prompt = hist + (_git_prompt(intent) if is_git else _exec_prompt(intent))
            d = await _run_agent(request, user, pkey, repo, executor, prompt,
                                 by_id, tiers_map, env_vars)

            # ── Escalation: a non-git quick edit that proved too big is promoted
            # to the full plan→build→review pipeline (agent said ESCALATE, the run
            # failed, or it exhausted its budget). We commit whatever partial work
            # exists so the planner sees the real state, then FALL THROUGH to the
            # planning branch below instead of returning.
            escalate, why = (False, "") if is_git else _should_escalate(d)
            if escalate:
                await code_git.git_commit(
                    repo, f"[edit] {executor} (partial, escalating): {route.get('title', intent)[:50]}")
                await _update_map(repo, tiers_map, registry)
                _append_chat(sdir, "assistant",
                             f"This is bigger than a quick edit — escalating to a full plan. "
                             f"Reason: {why}", kind="note", archetype=executor)
                _progress(pkey, "route", f"escalated small → big ({executor})", size="big")
                route = {**route, "size": "big",
                         "planner": "architect" if "architect" in by_id else route["planner"]}
                hist = _history_preamble(sdir)  # refresh so the planner sees the escalation reason
                # fall through ↓ to the planning branch
            else:
                if is_git:
                    # The git agent runs commits/pushes itself — don't wrap another
                    # commit; just report where HEAD landed.
                    head = await code_git.git_log(repo, 1)
                    sha = head[0]["sha"] if head else None
                else:
                    sha = await code_git.git_commit(repo, f"[edit] {executor}: {route.get('title', intent)[:60]}")
                    await _update_map(repo, tiers_map, registry)   # keep the map fresh
                out = (d.get("output") or "").strip() or "(no output)"
                if not d.get("ok"):
                    out = f"⚠️ {executor} failed: {d.get('error', 'unknown error')}"
                _u = _usage_summary(pkey)
                if _u:
                    out += f"\n\n_{_u}._"
                assistant = _append_chat(sdir, "assistant", out, archetype=executor,
                                         size="small", ok=bool(d.get("ok")), commit=sha or "", route=route)
                _write_state(sdir, {"status": "idle", "last_route": route})
                return {"message": assistant, "route": route, "commit": sha}

        planner = route["planner"]
        _phase(pkey, f"Planning ({planner})")
        plan_rel = _new_plan_rel(repo, route.get("title", intent))
        d = await _run_agent(request, user, pkey, repo, planner,
                             hist + _plan_prompt(intent, plan_rel), by_id, tiers_map, env_vars)
        # Land the plan at plan_rel no matter what the planner did: it may have
        # written the requested path, fallen back to the legacy plan.md, or only
        # returned the plan as chat text. Never let a new plan clobber an old one.
        plan_abs = repo / plan_rel
        if not plan_abs.is_file():
            legacy = repo / "plan.md"
            if legacy.is_file():
                plan_abs.write_text(legacy.read_text())
                legacy.unlink()  # don't leave a clobbering plan.md around
        plan_text = plan_abs.read_text() if plan_abs.is_file() else (d.get("output") or "").strip()
        if not plan_abs.is_file() and plan_text:
            plan_abs.write_text(plan_text)   # persist chat-only plans so build can read them
        sha = await code_git.git_commit(repo, f"[plan] {planner}: {route.get('title', intent)[:60]}")
        # Usage rides as metadata — the plan text itself feeds the editable
        # approval textarea, so no suffix on it.
        assistant = _append_chat(sdir, "assistant", plan_text or "(planner produced no plan)",
                                 kind="plan", archetype=planner, ok=bool(d.get("ok")), commit=sha or "",
                                 route=route, usage=_usage_summary(pkey))
        _write_state(sdir, {"status": "awaiting_plan", "intent": intent,
                            "route": route, "plan_file": plan_rel})
        return {"message": assistant, "route": route, "commit": sha,
                "status": "awaiting_plan", "plan": plan_text}
    finally:
        _sz = (locals().get("route") or {}).get("size", "")
        _persist_trace(pkey, sdir, f"{_sz + ': ' if _sz else ''}{intent[:80]}")
        _progress_done(pkey)


@router.post("/plan/approve")
async def approve_plan(body: ApproveReq, request: Request, user: dict = Depends(get_current_user)):
    uid = user["id"]
    sess, repo, sdir, pkey = _sctx(uid, body.project, body.session)
    state = _read_state(sdir)
    if state.get("status") != "awaiting_plan":
        raise HTTPException(409, "no plan awaiting approval")
    intent = state.get("intent", "")
    plan_file = state.get("plan_file") or "plan.md"   # legacy sessions used plan.md
    if body.plan and body.plan.strip():
        pa = repo / plan_file
        pa.parent.mkdir(parents=True, exist_ok=True)
        pa.write_text(body.plan)
        await code_git.git_commit(repo, "[plan] user-edited")

    db = get_db()
    archetypes = await merged_archetypes(db, uid)
    by_id = {a["id"]: a for a in archetypes}
    registry = _load_registry()
    tiers_map, env_vars = await _load_owner_tiers(db, uid)

    _append_chat(sdir, "user", "✓ Plan approved — building.", kind="approval")
    asyncio.create_task(_run_build_loop(
        request, user, pkey, repo, sdir, intent, by_id, tiers_map, env_vars, registry,
        plan_file=plan_file))
    return {"status": "running"}


@router.post("/plan/cancel")
async def cancel_plan(body: CancelReq, user: dict = Depends(get_current_user)):
    """Discard a plan awaiting approval — nothing is built. The plan file stays
    in `.plans/` as history; the session returns to idle for a new request."""
    uid = user["id"]
    _sess, _repo, sdir, _pk = _sctx(uid, body.project, body.session)
    state = _read_state(sdir)
    if state.get("status") != "awaiting_plan":
        raise HTTPException(409, "no plan awaiting approval")
    assistant = _append_chat(sdir, "assistant",
                             "Plan discarded — nothing was built. Send a new request when ready.",
                             kind="note")
    _write_state(sdir, {"status": "idle"})
    return {"status": "idle", "message": assistant}
