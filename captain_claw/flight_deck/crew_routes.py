"""Crew — coding CLIs run for a user, on an admin's approval.

  GET    /fd/crew/me                    the caller's approval, and each CLI's state here
  GET    /fd/crew/repos                 where a task can work
  GET    /fd/crew/usage                 each CLI's model and limits as last reported, and a week's use
  GET    /fd/crew/tasks                 the caller's tasks
  POST   /fd/crew/tasks                 a new task
  GET    /fd/crew/tasks/{id}            one task
  GET    /fd/crew/tasks/{id}/events     what its worker did, from ?after=
  GET    /fd/crew/tasks/{id}/diff       what it changed
  POST   /fd/crew/tasks/{id}/files      where its files are, in the VFS (?whole=true for the whole project)
  GET    /fd/crew/tasks/{id}/file       one copied file, to show (?path=)
  POST   /fd/crew/tasks/{id}/link       a page to open, published from that copy
  POST   /fd/crew/tasks/{id}/steer      the next instruction (now: cut the running turn short for it)
  POST   /fd/crew/tasks/{id}/interrupt  what's waiting for this turn to end goes now
  POST   /fd/crew/tasks/{id}/stop       stop the worker, keep its work
  POST   /fd/crew/tasks/{id}/pr         push and open a pull request
  POST   /fd/crew/tasks/{id}/merge      land the work
  POST   /fd/crew/tasks/{id}/close      tear it down (?discard=true for work that hasn't landed)
  GET    /fd/crew/models                what each CLI can run on, which can take a while to ask
  PUT    /fd/crew/models/{harness}      the model a CLI uses when a task names none
  PUT/DELETE /fd/crew/own-model         a model server of their own, for OpenCode and Pi
  POST   /fd/crew/own-model/check       can Flight Deck reach it, and what does it serve
  POST/DELETE /fd/crew/uploads[/{id}]   a file for a task or a message that's still being written
  POST   /fd/crew/projects              a new project, ready for tasks
  POST   /fd/crew/projects/tidy         clear a project's finished tasks off the page
  GET    /fd/crew/envs                  each project's environment: what's installed, its settings
  PUT    /fd/crew/envs/{id}             its settings: test, run, copy, vars, activate
  POST   /fd/crew/envs/{id}/current     what's installed is up to date
  WS     /fd/crew/terminal              a shell in a project (?env=) or a task's worktree (?task=)
  GET/PUT/DELETE /fd/crew/logins[/{name}]   the accounts their workers run on

  /fd/crew/agent/*                      the same, for the `crew` tool; the agent's
                                        owner is who it runs for. An agent can ask
                                        for a merge, not do one, and can't discard.

Every call reads the approval fresh (:mod:`crew_access`); a task of someone else's
is a 404. Admins approve on the Users page (``PUT /fd/admin/users/{id}``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, WebSocket
from pydantic import BaseModel

from captain_claw.flight_deck import (
    crew_access,
    crew_attach,
    crew_env,
    crew_harness,
    crew_logins,
    crew_runner,
    crew_store,
    crew_terminal,
)
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_current_user, get_db
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/crew", tags=["crew"])

_MERGE_DOMAIN = "coding"


class DispatchBody(BaseModel):
    harness: str
    brief: str = ""
    title: str = ""
    kind: str = "ship"
    deliver: str = "branch"
    project: str = ""
    folder: str = ""
    repo_url: str = ""
    repo_path: str = ""
    model: str = ""
    effort: str = ""
    attachments: list[str] = []


class LinkBody(BaseModel):
    name: str = ""


class SteerBody(BaseModel):
    text: str = ""
    now: bool = False
    attachments: list[str] = []


class LoginBody(BaseModel):
    secret: str


class AgentDispatchBody(DispatchBody):
    source_host: str = "localhost"
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0


class AgentCloneBody(BaseModel):
    repo_url: str
    source_host: str = "localhost"
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0


class AgentPlaceBody(BaseModel):
    project: str = ""
    folder: str = ""
    repo_url: str = ""
    repo_path: str = ""


class EnvBody(BaseModel):
    test: str | None = None
    run: str | None = None
    copy_paths: list[str] | str | None = None
    vars: dict[str, str] | str | None = None
    activate: str | None = None


class AgentTaskBody(BaseModel):
    task_id: str
    text: str = ""
    now: bool = False
    whole: bool = False
    attachments: list[str] = []


class ModelBody(BaseModel):
    model: str = ""


class OwnModelBody(BaseModel):
    url: str
    key: str = ""
    models: list[str] | str = ""
    context: int | str = 0


class OwnModelCheck(BaseModel):
    url: str
    key: str = ""


class ProjectBody(BaseModel):
    name: str


class ProjectKey(BaseModel):
    key: str


async def _approved(user: dict) -> tuple[dict, Grant]:
    """The caller, read fresh, and what they may do; refused when that's nothing."""
    uid = user.get("id") or ""
    row = await crew_runner._user_of(uid)
    grant = crew_access.grant_of(row)
    if not row or not grant.allowed:
        raise HTTPException(403, crew_access.NOT_APPROVED if crew_access.offered() else crew_access.NOT_OFFERED)
    return row, grant


def _with_own(own: dict | None, harness: str, listed: list[str]) -> list[str]:
    return list(dict.fromkeys([*(crew_logins.own_model_names(own) if harness in crew_logins.OWN_HARNESSES else []), *listed]))


async def _me(user: dict, *, ask_for_models: bool = True) -> dict[str, Any]:
    """*ask_for_models* off gives what's already known: the page opens on this, and a CLI
    can take half a minute to answer, so the page asks for them apart, at /models."""
    row = await crew_runner._user_of(user.get("id") or "")
    grant = crew_access.grant_of(row)
    out: dict[str, Any] = {"offered": crew_access.offered(), "allowed": grant.allowed}
    if not row or not grant.allowed:
        return out
    db = get_db()
    defaults = await crew_logins.read_models(db, row["id"])
    own = await crew_logins.read_own_model(db, row["id"])

    async def one(harness: str) -> dict[str, Any]:
        info = crew_harness.HARNESSES[harness]
        binary = await asyncio.to_thread(crew_harness.find_binary, harness)
        version = await asyncio.to_thread(crew_harness.version_of, binary) if binary else ""
        listed = (await crew_runner.models_of(row, grant, harness, binary, db) if ask_for_models
                  else crew_runner.models_known(row["id"], harness))
        return {"id": harness, "label": info.label, "installed": bool(binary), "version": version,
                "install": info.install, "login": await crew_logins.login_state(db, row, grant, harness),
                "default_model": crew_logins.effective_model(defaults, own, harness),
                "models": _with_own(own, harness, listed)}

    out.update(
        trusted=grant.trusted, max_parallel=grant.max_parallel, terminal=grant.terminal,
        harnesses=await asyncio.gather(*(one(h) for h in grant.harnesses)),
        logins=await crew_logins.read_meta(db, row["id"]),
        login_names=list(crew_logins.LOGIN_NAMES), free_terms_url=crew_runner.FREE_TERMS_URL,
        own_model=crew_logins.own_model_view(own),
    )
    return out


@router.get("/me")
async def me(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _me(user, ask_for_models=False)


@router.get("/models")
async def models(user: dict = Depends(get_current_user)) -> dict[str, list[str]]:
    row, grant = await _approved(user)
    db = get_db()
    own = await crew_logins.read_own_model(db, row["id"])

    async def one(harness: str) -> tuple[str, list[str]]:
        binary = await asyncio.to_thread(crew_harness.find_binary, harness)
        return harness, _with_own(own, harness, await crew_runner.models_of(row, grant, harness, binary, db))

    return dict(await asyncio.gather(*(one(h) for h in grant.harnesses)))


@router.get("/repos")
async def repos(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    from captain_claw.flight_deck import code_routes

    projects = []
    for name in code_routes._discover_projects(row["id"]):
        tree = code_routes._project_tree(row["id"], name)
        folders = [f.get("name") for f in tree.get("folders", [])]
        paths = {f: crew_runner._vfs_path(row["id"], code_routes._folder_repo(row["id"], name, f)) for f in folders}
        projects.append({"project": name, "key": f"project:{name}", "folders": folders, "paths": paths})
    return {"projects": projects, "recent": crew_store.get_store().recent_repos(row["id"])}


async def _known_repos(row: dict, grant: Grant) -> list[dict[str, Any]]:
    """Every repository this person has here. Environments and shells are only ever
    reached through this list, so nobody can name a folder of someone else's."""
    from captain_claw.flight_deck import code_routes

    uid = row["id"]
    known: list[dict[str, Any]] = []
    for name in code_routes._discover_projects(uid):
        for item in code_routes._project_tree(uid, name).get("folders", []):
            folder = item.get("name")
            # A link is a folder on this computer: an admin's to work in, as for a task.
            if not grant.trusted and (code_routes._folder_meta(uid, name, folder) or {}).get("kind") == "link":
                continue
            known.append({"repo": str(code_routes._folder_repo(uid, name, folder)), "label": f"{name}/{folder}", "kind": "project",
                          "project_key": f"project:{name}", "project_name": name, "place": {"project": name, "folder": folder}})
    clones = crew_store.user_dir(uid) / "repos"
    if clones.is_dir():
        for path in sorted(clones.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                rc, remote = await crew_runner._git(path, "remote", "get-url", "origin")
                url = remote if rc == 0 else ""
                known.append({"repo": str(path), "label": path.name, "kind": "url", "project_key": f"url:{path.name}",
                              "project_name": path.name, "repo_url": url, "place": {"repo_url": url}})
    if grant.trusted:
        for item in crew_store.get_store().recent_repos(uid):
            if item.get("repo_kind") == "path":
                known.append({"repo": item["repo"], "label": item.get("repo_label") or item["repo"], "kind": "path",
                              "project_key": f"path:{item['repo']}", "project_name": item.get("repo_label") or item["repo"],
                              "place": {"repo_path": item["repo"]}})
    return list({entry["repo"]: entry for entry in known}.values())


async def _env_list(row: dict, grant: Grant) -> list[dict[str, Any]]:
    known = await _known_repos(row, grant)
    return list(await asyncio.gather(*(crew_env.status(row["id"], entry) for entry in known)))


async def _known_repo(row: dict, grant: Grant, env_id: str) -> dict[str, Any]:
    for entry in await _known_repos(row, grant):
        if crew_env.env_id(entry["repo"]) == env_id:
            return entry
    raise HTTPException(404, "No such project")


@router.get("/envs")
async def list_envs(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return {"envs": await _env_list(row, grant)}


@router.put("/envs/{env_id}")
async def save_env(env_id: str, body: EnvBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    entry = await _known_repo(row, grant, env_id)
    sent = {k: v for k, v in {"test": body.test, "run": body.run, "copy": body.copy_paths, "vars": body.vars,
                              "activate": body.activate}.items() if v is not None}
    crew_env.save_settings(row["id"], entry["repo"], sent)
    return await crew_env.status(row["id"], entry)


@router.post("/envs/{env_id}/current")
async def env_current(env_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    entry = await _known_repo(row, grant, env_id)
    await crew_env.mark_current(row["id"], entry["repo"])
    return await crew_env.status(row["id"], entry)


async def _terminal_user(fd_token: str) -> dict | None:
    """Who's asking: this deck's JWT, the way the agent chat's socket has it."""
    from captain_claw.flight_deck.auth import decode_access_token

    if not _fd_auth_enabled():
        return await crew_runner._user_of("")
    try:
        payload = decode_access_token(fd_token) if fd_token else None
    except HTTPException:
        payload = None
    return await crew_runner._user_of(str(payload.get("sub") or "")) if payload else None


async def _shell_for(row: dict, grant: Grant, env: str, task: str, cols: int, rows: int) -> Any:
    """The shell for a project or a task: the one already open, or a new one."""
    uid = row["id"]
    target = f"task:{task}" if task else f"env:{env}"
    shell = crew_terminal.existing(uid, target)
    if shell is not None:
        return shell
    if crew_terminal.at_limit(uid):
        raise HTTPException(429, "You have as many terminals open as you can. Close one first.")
    if task:
        found = crew_runner._get(uid, task)
        cwd, repo = found.get("worktree") or "", found["repo"]
        state = found.get("env") or {}
        banner = f"This is the folder of the task \"{found['title'][:60]}\", with its environment."
    else:
        entry = await _known_repo(row, grant, env)
        cwd = repo = entry["repo"]
        state = {"python": [p["path"] for p in (await crew_env.scan(repo))["python"]]}
        banner = "This is the project's main folder. What you install here, new tasks on this project get."
    if not cwd or not Path(cwd).is_dir():
        raise HTTPException(404, "That folder isn't there any more.")
    environment = crew_logins.person_env(row, grant)
    environment.update(crew_env.variables(uid, repo, cwd, state, environment.get("PATH", "")))
    rc_file = crew_env.env_dir(uid, repo) / f"shell-{crew_env.env_id(target)}.rc"
    return crew_terminal.start(uid, target, cwd, environment, rc_file, banner, cols, rows)


@router.websocket("/terminal")
async def terminal(ws: WebSocket, env: str = "", task: str = "", fd_token: str = "",
                   cols: int = Query(100), rows: int = Query(30)) -> None:
    """The browser guard has already turned away other sites' pages (HTTP middleware
    doesn't run here, that guard does). Accepted first, so a refusal has words."""
    await ws.accept()
    row = await _terminal_user(fd_token)
    if not row:
        await ws.close(code=4001, reason="Missing or invalid token")
        return
    grant = crew_access.grant_of(row)
    if not grant.allowed or not grant.terminal:
        await ws.close(code=4403, reason="The terminal isn't approved for this account.")
        return
    if bool(env) == bool(task):
        await ws.close(code=4400, reason="Say which project or task.")
        return
    try:
        shell = await _shell_for(row, grant, env, task, cols, rows)
    except HTTPException as exc:
        await ws.close(code=4429 if exc.status_code == 429 else 4404, reason=str(exc.detail)[:120])
        return
    await crew_terminal.serve(ws, shell)
    try:
        await ws.close()
    except RuntimeError:
        pass


@router.get("/usage")
async def usage(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return crew_runner.usage(row["id"])


@router.post("/projects", status_code=201)
async def create_project(body: ProjectBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    where = await crew_runner._resolve_repo(row, grant, {"project": body.name})
    name = where["repo_label"].split("/", 1)[0]
    return {"project": name, "key": f"project:{name}", "path": crew_runner._vfs_path(row["id"], where["repo"])}


@router.post("/projects/tidy")
async def tidy_project(body: ProjectKey, user: dict = Depends(get_current_user)) -> dict[str, int]:
    row, _ = await _approved(user)
    return await crew_runner.tidy(row["id"], body.key)


@router.get("/tasks")
async def list_tasks(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return {"tasks": [crew_runner.view(t, grant=grant) for t in crew_store.get_store().list_for(row["id"])]}


@router.post("/uploads", status_code=201)
async def upload(file: UploadFile = File(...), user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return await crew_attach.save_upload(row["id"], file)


@router.delete("/uploads/{upload_id}")
async def drop_upload(upload_id: str, user: dict = Depends(get_current_user)) -> dict[str, bool]:
    row, _ = await _approved(user)
    crew_attach.drop_upload(row["id"], upload_id)
    return {"ok": True}


@router.post("/tasks", status_code=201)
async def create_task(body: DispatchBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    task = await crew_runner.dispatch(row, grant, body.model_dump())
    return crew_runner.view(task, grant=grant, full=True)


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    task = crew_runner._get(row["id"], task_id)
    return {**crew_runner.view(task, grant=grant, full=True), "report": crew_runner.report(row["id"], task_id)}


@router.get("/tasks/{task_id}/events")
async def task_events(task_id: str, after: int = 0, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    crew_runner._get(row["id"], task_id)
    return {"events": crew_store.read_events(row["id"], task_id, after=max(0, after))}


@router.get("/tasks/{task_id}/diff")
async def task_diff(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, str]:
    row, _ = await _approved(user)
    return await crew_runner.diff(row["id"], task_id)


@router.post("/tasks/{task_id}/files")
async def task_files(task_id: str, whole: bool = False, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return await crew_runner.files(row["id"], task_id, whole=whole)


@router.get("/tasks/{task_id}/file")
async def task_file(task_id: str, path: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return crew_runner.read_result(row["id"], task_id, path)


@router.post("/tasks/{task_id}/link")
async def task_link(task_id: str, body: LinkBody | None = None, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return await crew_runner.link(row["id"], task_id, body.name if body else "")


@router.post("/tasks/{task_id}/steer")
async def task_steer(task_id: str, body: SteerBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return crew_runner.view(await crew_runner.steer(row["id"], task_id, body.text, now=body.now, attachments=body.attachments),
                            grant=grant, full=True)


@router.post("/tasks/{task_id}/interrupt")
async def task_interrupt(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return crew_runner.view(await crew_runner.interrupt(row["id"], task_id), grant=grant, full=True)


@router.post("/tasks/{task_id}/stop")
async def task_stop(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    # Not behind the approval: someone it was withdrawn from can still stop their own.
    uid = user.get("id") or ""
    return crew_runner.view(await crew_runner.stop(uid, task_id), full=True)


@router.post("/tasks/{task_id}/pr")
async def task_pr(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return crew_runner.view(await crew_runner.open_pr(row, grant, task_id), grant=grant, full=True)


@router.post("/tasks/{task_id}/merge")
async def task_merge(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return crew_runner.view(await crew_runner.merge(row, grant, task_id), grant=grant, full=True)


@router.post("/tasks/{task_id}/close")
async def task_close(task_id: str, discard: bool = False, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return crew_runner.view(await crew_runner.close(row["id"], task_id, discard=discard), grant=grant, full=True)


@router.put("/logins/{name}")
async def put_login(name: str, body: LoginBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    saved = await crew_logins.save_login(get_db(), row["id"], name, body.secret)
    crew_store.get_store().forget_seen(row["id"], name)  # another account has other limits
    return saved


@router.put("/models/{harness}")
async def put_model(harness: str, body: ModelBody, user: dict = Depends(get_current_user)) -> dict[str, str]:
    row, grant = await _approved(user)
    if harness not in grant.harnesses:
        raise HTTPException(404, "Unknown CLI")
    model = crew_runner._clean_axis(body.model, crew_runner._MODEL_RE, "model name")
    await crew_logins.save_model(get_db(), row["id"], harness, model)
    return {"harness": harness, "model": model}


@router.put("/own-model")
async def put_own_model(body: OwnModelBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    own = await crew_logins.save_own_model(get_db(), row["id"], body.url, body.key, body.models, body.context)
    crew_runner._MODELS.pop(row["id"], None)
    return crew_logins.own_model_view(own) or {}


@router.delete("/own-model")
async def delete_own_model(user: dict = Depends(get_current_user)) -> dict[str, bool]:
    row, _ = await _approved(user)
    await crew_logins.delete_own_model(get_db(), row["id"])
    crew_runner._MODELS.pop(row["id"], None)
    return {"ok": True}


@router.post("/own-model/check")
async def check_own_model(body: OwnModelCheck, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return await crew_logins.check_own_model(get_db(), row["id"], body.url, body.key)


@router.delete("/logins/{name}")
async def delete_login(name: str, user: dict = Depends(get_current_user)) -> dict[str, bool]:
    row, _ = await _approved(user)
    if name not in crew_logins.LOGIN_NAMES:
        raise HTTPException(404, "Unknown login")
    await crew_logins.delete_login(get_db(), row["id"], name)
    crew_store.get_store().forget_seen(row["id"], name)
    return {"ok": True}


# ── The `crew` tool ─────────────────────────────────────────────────────


async def _agent_caller(request: Request) -> tuple[dict, Grant, str]:
    """The agent's owner (FD's record, never what the agent says), and their approval."""
    from captain_claw.flight_deck.agent_identity import require_agent_caller

    caller = require_agent_caller(request, what="Crew")
    owner = caller.owner or ("" if _fd_auth_enabled() else "local")
    if not owner:
        raise HTTPException(403, "This agent has no owner on record, so it can't use Crew.")
    row, grant = await _approved({"id": owner})
    return row, grant, caller.slug


@router.get("/agent/status")
async def agent_status(request: Request) -> dict[str, Any]:
    from captain_claw.flight_deck.agent_identity import require_agent_caller

    caller = require_agent_caller(request, what="Crew")
    owner = caller.owner or ("" if _fd_auth_enabled() else "local")
    row = await crew_runner._user_of(owner) if owner else None
    return {"usable": crew_access.grant_of(row).allowed}


@router.get("/agent/options")
async def agent_options(request: Request) -> dict[str, Any]:
    row, grant, _ = await _agent_caller(request)
    return {"me": await _me(row), "repos": await repos(row), "usage": crew_runner.usage(row["id"]),
            "envs": await _env_list(row, grant)}


@router.post("/agent/env")
async def agent_env(body: AgentPlaceBody, request: Request) -> dict[str, Any]:
    """What's prepared for a project, or for all of them when none is named."""
    row, grant, _ = await _agent_caller(request)
    envs = await _env_list(row, grant)
    if not (body.project or body.repo_url or body.repo_path):
        return {"envs": envs}

    def matches(entry: dict) -> bool:
        place = entry["place"]
        if body.repo_url:
            return crew_runner._same_repo(place.get("repo_url") or "", body.repo_url)
        if body.repo_path:
            return place.get("repo_path") == body.repo_path
        return place.get("project") == body.project and (not body.folder or place.get("folder") == body.folder)

    return {"envs": [entry for entry in envs if matches(entry)]}


@router.post("/agent/uploads", status_code=201)
async def agent_upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    row, _, _ = await _agent_caller(request)
    return await crew_attach.save_upload(row["id"], file)


@router.delete("/agent/uploads/{upload_id}")
async def agent_drop_upload(upload_id: str, request: Request) -> dict[str, bool]:
    row, _, _ = await _agent_caller(request)
    crew_attach.drop_upload(row["id"], upload_id)
    return {"ok": True}


@router.post("/agent/dispatch")
async def agent_dispatch(body: AgentDispatchBody, request: Request) -> dict[str, Any]:
    row, grant, slug = await _agent_caller(request)
    fields = body.model_dump()
    fields.update(
        agent_slug=slug, source_host=body.source_host or "localhost",
        origin={"platform": body.origin_platform, "user_id": body.origin_user_id, "chat_id": body.origin_chat_id},
    )
    task = await crew_runner.dispatch(row, grant, fields)
    key = crew_runner.project_of(task)[0]
    return {**crew_runner.view(task, full=True), "unlanded": crew_runner.unlanded_in(row["id"], key, but=task["id"])}


@router.post("/agent/clone")
async def agent_clone(body: AgentCloneBody, request: Request) -> dict[str, Any]:
    """A repository got ready with no worker. It's a task to the page and the callback."""
    row, grant, slug = await _agent_caller(request)
    task = await crew_runner.clone(row, grant, {
        "repo_url": body.repo_url, "agent_slug": slug, "source_host": body.source_host or "localhost",
        "origin": {"platform": body.origin_platform, "user_id": body.origin_user_id, "chat_id": body.origin_chat_id},
    })
    return crew_runner.view(task, full=True)


@router.post("/agent/files")
async def agent_files(body: AgentTaskBody, request: Request) -> dict[str, Any]:
    row, _, _ = await _agent_caller(request)
    return await crew_runner.files(row["id"], body.task_id, whole=body.whole)


@router.get("/agent/tasks")
async def agent_tasks(request: Request) -> dict[str, Any]:
    row, _, _ = await _agent_caller(request)
    return {"tasks": [crew_runner.view(t) for t in crew_store.get_store().list_for(row["id"], limit=30)]}


@router.post("/agent/task")
async def agent_task(body: AgentTaskBody, request: Request) -> dict[str, Any]:
    row, _, _ = await _agent_caller(request)
    task = crew_runner._get(row["id"], body.task_id)
    recent = crew_store.read_events(row["id"], body.task_id, after=max(0, int(task.get("events") or 0) - 12))
    return {**crew_runner.view(task, full=True), "report": crew_runner.report(row["id"], body.task_id),
            "recent": [{"kind": e.get("kind"), "tool": e.get("tool", ""), "text": str(e.get("text") or "")[:600]}
                       for e in recent]}


@router.post("/agent/steer")
async def agent_steer(body: AgentTaskBody, request: Request) -> dict[str, Any]:
    row, _, _ = await _agent_caller(request)
    return crew_runner.view(await crew_runner.steer(row["id"], body.task_id, body.text, now=body.now, attachments=body.attachments),
                            full=True)


@router.post("/agent/stop")
async def agent_stop(body: AgentTaskBody, request: Request) -> dict[str, Any]:
    row, _, _ = await _agent_caller(request)
    return crew_runner.view(await crew_runner.stop(row["id"], body.task_id), full=True)


@router.post("/agent/pr")
async def agent_pr(body: AgentTaskBody, request: Request) -> dict[str, Any]:
    row, grant, _ = await _agent_caller(request)
    return crew_runner.view(await crew_runner.open_pr(row, grant, body.task_id), full=True)


@router.post("/agent/request_merge")
async def agent_request_merge(body: AgentTaskBody, request: Request) -> dict[str, Any]:
    """An agent's way to a merge: a proposal only the person can approve."""
    from captain_claw.flight_deck import owner_notify
    from captain_claw.flight_deck.autonomy import get_store as autonomy_store

    row, _, _ = await _agent_caller(request)
    task = crew_runner._get(row["id"], body.task_id)
    evidence = await crew_runner._evidence(task)
    crew_runner._must_be_landable(task, evidence)
    if task["repo_kind"] == "url" and not task["delivery"].get("pr"):
        # Said now, not after the person has pressed Approve.
        raise HTTPException(409, "This one came from a repository address, so it lands through a pull request. Open it "
                                 "first with crew action=pr.")
    store = autonomy_store()
    for waiting in store.waiting_actions(row["id"]):
        payload = waiting.get("payload") or {}
        if waiting.get("kind") == "crew_merge" and payload.get("task_id") == task["id"]:
            if payload.get("head") == evidence.get("head"):
                return {"status": "already_waiting", "action_id": waiting["id"]}
            store.expire_actions([waiting["id"]], "expired: the work changed after this was asked")
    into = task.get("base_branch") or "its base branch"
    action = store.add_action(
        row["id"], kind="crew_merge", source="crew", risk="normal", domain=_MERGE_DOMAIN, status="awaiting_approval",
        title=f"Merge {task['title'][:100]} into {into}",
        rationale=crew_runner.notice_text({**task, "evidence": evidence}), ref_id=task["id"],
        # The Approve covers these commits and no later ones.
        payload={"task_id": task["id"], "head": evidence.get("head") or ""},
    )
    job = asyncio.create_task(owner_notify.notify_proposal(row["id"], action))
    crew_runner._TASKS.add(job)
    job.add_done_callback(crew_runner._TASKS.discard)
    return {"status": "waiting_for_approval", "action_id": action["id"]}


class AgentUpdateBody(BaseModel):
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0


@router.post("/agent/update_deck")
async def agent_update_deck(body: AgentUpdateBody, request: Request) -> dict[str, Any]:
    """An agent's way to a deck update: a proposal only its admin owner can approve."""
    from captain_claw.flight_deck import deck_update, owner_notify
    from captain_claw.flight_deck.autonomy import get_store as autonomy_store

    row, _, slug = await _agent_caller(request)
    if row.get("role") != "admin":
        raise HTTPException(403, "Only an admin's agent can ask for this. An admin updates the deck in Console → Updates.")
    seen = await deck_update.status(fetch=True)
    if not seen.get("available"):
        raise HTTPException(409, seen.get("why") or "This deck can't update itself.")
    if seen.get("active"):
        raise HTTPException(409, "An update is already under way.")
    if seen.get("fetch_error"):
        raise HTTPException(409, seen["fetch_error"])
    if not seen["behind"]:
        return {"status": "up_to_date", "head": seen["head"], "head_title": seen["head_title"]}
    if seen["blockers"]:
        raise HTTPException(409, " ".join(seen["blockers"]))
    target = seen["target"]
    store = autonomy_store()
    for waiting in store.waiting_actions(row["id"]):
        if waiting.get("kind") == "deck_update":
            if (waiting.get("payload") or {}).get("target") == target:
                return {"status": "already_waiting", "action_id": waiting["id"], "commits": seen["behind"]}
            store.expire_actions([waiting["id"]], "expired: there's a newer version")
    titles = "\n".join(f"- {c['title']}" for c in seen["commits"])
    action = store.add_action(
        row["id"], kind="deck_update", source="crew", risk="normal", domain=_MERGE_DOMAIN, status="awaiting_approval",
        title=f"Update this deck to {target[:9]} ({seen['behind']} new commit{'s' if seen['behind'] != 1 else ''})",
        rationale=f"It stops every agent, pulls {seen['upstream']} and restarts the deck.\n\n{titles}", ref_id=target,
        # The Approve covers this commit and no later one.
        payload={"target": target, "slug": slug, "origin": {
            "platform": body.origin_platform, "user_id": body.origin_user_id, "chat_id": body.origin_chat_id}},
    )
    job = asyncio.create_task(owner_notify.notify_proposal(row["id"], action))
    crew_runner._TASKS.add(job)
    job.add_done_callback(crew_runner._TASKS.discard)
    return {"status": "waiting_for_approval", "action_id": action["id"], "commits": seen["behind"]}
