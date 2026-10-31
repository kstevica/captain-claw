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
  POST   /fd/crew/tasks/{id}/steer      the next instruction
  POST   /fd/crew/tasks/{id}/stop       stop the worker, keep its work
  POST   /fd/crew/tasks/{id}/pr         push and open a pull request
  POST   /fd/crew/tasks/{id}/merge      land the work
  POST   /fd/crew/tasks/{id}/close      tear it down (?discard=true for work that hasn't landed)
  PUT    /fd/crew/models/{harness}      the model a CLI uses when a task names none
  PUT/DELETE /fd/crew/own-model         a model server of their own, for OpenCode and Pi
  POST   /fd/crew/own-model/check       can Flight Deck reach it, and what does it serve
  POST   /fd/crew/projects              a new project, ready for tasks
  POST   /fd/crew/projects/tidy         clear a project's finished tasks off the page
  GET/PUT/DELETE /fd/crew/logins[/{name}]   the accounts their workers run on

  /fd/crew/agent/*                      the same, for the `crew` tool; the agent's
                                        owner is who it runs for. An agent can ask
                                        for a merge, not do one, and can't discard.

Every call reads the approval fresh (:mod:`crew_access`); a task of someone else's
is a 404. Admins approve on the Users page (``PUT /fd/admin/users/{id}``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck import crew_access, crew_harness, crew_logins, crew_runner, crew_store
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_current_user, get_db
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/crew", tags=["crew"])

_MERGE_DOMAIN = "coding"


class DispatchBody(BaseModel):
    harness: str
    brief: str
    title: str = ""
    kind: str = "ship"
    deliver: str = "branch"
    project: str = ""
    folder: str = ""
    repo_url: str = ""
    repo_path: str = ""
    model: str = ""
    effort: str = ""


class SteerBody(BaseModel):
    text: str


class LoginBody(BaseModel):
    secret: str


class AgentDispatchBody(DispatchBody):
    source_host: str = "localhost"
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0


class AgentTaskBody(BaseModel):
    task_id: str
    text: str = ""
    whole: bool = False


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


async def _me(user: dict) -> dict[str, Any]:
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
        return {"id": harness, "label": info.label, "installed": bool(binary), "version": version,
                "install": info.install, "login": await crew_logins.login_state(db, row, grant, harness),
                "default_model": crew_logins.effective_model(defaults, own, harness),
                "models": list(dict.fromkeys([
                    *(crew_logins.own_model_names(own) if harness in crew_logins.OWN_HARNESSES else []),
                    *await crew_runner.models_of(row, grant, harness, binary, db)]))}

    out.update(
        trusted=grant.trusted, max_parallel=grant.max_parallel,
        harnesses=await asyncio.gather(*(one(h) for h in grant.harnesses)),
        logins=await crew_logins.read_meta(db, row["id"]),
        login_names=list(crew_logins.LOGIN_NAMES), free_terms_url=crew_runner.FREE_TERMS_URL,
        own_model=crew_logins.own_model_view(own),
    )
    return out


@router.get("/me")
async def me(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _me(user)


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
async def task_link(task_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, _ = await _approved(user)
    return await crew_runner.link(row["id"], task_id)


@router.post("/tasks/{task_id}/steer")
async def task_steer(task_id: str, body: SteerBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row, grant = await _approved(user)
    return crew_runner.view(await crew_runner.steer(row["id"], task_id, body.text), grant=grant, full=True)


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
    row, _, _ = await _agent_caller(request)
    return {"me": await _me(row), "repos": await repos(row), "usage": crew_runner.usage(row["id"])}


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
    return crew_runner.view(await crew_runner.steer(row["id"], body.task_id, body.text), full=True)


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
