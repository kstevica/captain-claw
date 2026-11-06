"""Custom tools over HTTP: ``/fd/external-tools``. See ``external_tools.py`` and
docs/external-tools-plan.md.

Phase 0: a person works with their own tools only. Making or publishing one
needs Crew (a tool is code a coding agent writes); running, going back to a kept
version, secrets and deleting don't. The whole feature is off until an admin
turns it on (Admin → Settings → Custom tools).
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/external-tools", tags=["external-tools"])

OFF = "Custom tools are off on this deck. An admin turns them on under Admin → Settings."
NEEDS_CREW = ("Making and publishing custom tools needs Crew (coding agents), which your account "
              "isn't approved for. An admin approves it on the Users page.")
DELIVERY_PROJECT = "custom-tools"


async def deck_settings() -> dict[str, bool]:
    """The admin's switches (system config): on/off, and whether sharing a tool
    with everybody waits for an admin."""
    from captain_claw.flight_deck.admin_routes import _load_system_config

    cfg = _load_system_config(await get_db().get_system_setting("fd:system-config"))
    return {
        "enabled": bool(cfg.get("external_tools_enabled")),
        "everybody_needs_approval": bool(cfg.get("external_tools_everybody_approval", True)),
    }


async def _enabled(user: dict = Depends(get_current_user)) -> dict:
    if not (await deck_settings())["enabled"]:
        raise HTTPException(403, OFF)
    return user


async def _can_build(user: dict) -> bool:
    from captain_claw.flight_deck import crew_access, crew_runner

    row = await crew_runner._user_of(user.get("id") or "")
    return bool(row) and crew_access.grant_of(row).allowed


async def _require_builder(user: dict) -> None:
    if not await _can_build(user):
        raise HTTPException(403, NEEDS_CREW)


def _http(exc: xt.ToolError) -> HTTPException:
    return HTTPException(exc.status, exc.message)


def _uid(user: dict) -> str:
    uid = str(user.get("id") or "")
    if not uid:
        raise HTTPException(401, "Sign in first.")
    return uid


def _own(user: dict, slug: str) -> tuple[str, str]:
    """Phase 0: the caller's own tool, or 404 (sharing comes in Phase 3)."""
    uid = _uid(user)
    try:
        slug = xt.check_slug(slug)
        if not xt.exists(uid, slug):
            raise HTTPException(404, f"There's no tool called {slug}.")
    except xt.ToolError as exc:
        raise _http(exc) from None
    return uid, slug


# ── Secrets: each person's own values, sealed in their settings ──────────


async def secret_values(caller: str, owner: str, slug: str) -> dict[str, str]:
    from captain_claw.flight_deck import fd_crypto

    db = get_db()
    out: dict[str, str] = {}
    for s in xt.secret_names(owner, slug):
        token = await db.get_setting(caller, xt.secret_key(owner, slug, s["name"]))
        if not token:
            continue
        try:
            value = fd_crypto.unseal(token)
        except fd_crypto.CryptoUnavailable:
            value = None
        if isinstance(value, str) and value:
            out[s["name"]] = value
    return out


# ── Files in and out ─────────────────────────────────────────────────────


def _file_resolver(uid: str):
    """``vfs:<folder>/<path>`` in the caller's own VFS (Phase 0)."""
    from captain_claw.flight_deck import vfs_routes

    def resolve(ref: str) -> Path:
        ref = str(ref or "").strip()
        if not ref.startswith("vfs:"):
            raise xt.ToolError("files", f"{ref!r}: name a file as vfs:<folder>/<path>.")
        project, _, rel = ref[4:].lstrip("/").partition("/")
        if not project or not rel:
            raise xt.ToolError("files", f"{ref!r}: name a file as vfs:<folder>/<path>.")
        try:
            vfs_routes._refuse_reserved(rel)
            return vfs_routes._resolve(uid, project, rel)
        except HTTPException as exc:
            raise xt.ToolError("files", f"{ref!r}: {exc.detail}") from None

    return resolve


def _deliverer(uid: str, slug: str):
    """A call's files go to the caller's VFS: vfs:custom-tools/<slug>/<day>-<call>/."""
    from captain_claw.flight_deck import vfs_routes

    async def deliver(call_id: str, files: list[tuple[str, Path]]) -> list[dict[str, Any]]:
        folder_rel = f"{slug}/{time.strftime('%Y-%m-%d')}-{call_id}"
        dest = vfs_routes._user_root(uid) / DELIVERY_PROJECT / folder_rel
        out = []

        def copy() -> None:
            for rel, src in files:
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
                out.append({"name": rel, "path": f"vfs:{DELIVERY_PROJECT}/{folder_rel}/{rel}",
                            "size": target.stat().st_size})

        await asyncio.to_thread(copy)
        return out

    return deliver


# ── Routes ───────────────────────────────────────────────────────────────


class CreateBody(BaseModel):
    slug: str = Field(..., max_length=48)
    title: str = Field("", max_length=200)
    description: str = Field("", max_length=2000)


class PublishBody(BaseModel):
    sha: str = Field("", max_length=64)


class LiveBody(BaseModel):
    version: str = Field(..., max_length=12)


class RunBody(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


class SecretsBody(BaseModel):
    values: dict[str, str]


@router.get("/status")
async def status(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Whether custom tools are on here, and whether this person can make them."""
    settings = await deck_settings()
    return {**settings, "can_build": settings["enabled"] and await _can_build(user)}


@router.get("")
async def list_mine(user: dict = Depends(_enabled)) -> dict[str, Any]:
    try:
        return {"tools": xt.list_tools(_uid(user)), "can_build": await _can_build(user)}
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.post("")
async def create(body: CreateBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    await _require_builder(user)
    try:
        return await xt.create_tool(_uid(user), body.slug, title=body.title, description=body.description)
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}")
async def get_one(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    out = xt.detail(owner, slug)
    have = await secret_values(_uid(user), owner, slug)
    out["secrets"] = [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]
    out["can_build"] = await _can_build(user)
    return out


@router.delete("/{slug}")
async def delete(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    try:
        await xt.delete_tool(owner, slug)
    except xt.ToolError as exc:
        raise _http(exc) from None
    # The owner's values for it go with it (Phase 3: everyone's who added it).
    db = get_db()
    prefix = xt.secret_key(owner, slug, "")
    for key in await db.get_all_settings(owner):
        if key.startswith(prefix):
            await db.delete_setting(owner, key)
    return {"ok": True}


@router.post("/{slug}/publish", status_code=202)
async def publish(slug: str, body: PublishBody | None = None, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    await _require_builder(user)
    sha = (body.sha if body else "").strip()
    if sha and not re.fullmatch(r"[0-9a-f]{7,64}", sha):
        raise HTTPException(400, "That isn't a commit id.")
    try:
        return xt.start_publish(owner, slug, sha=sha, by=_uid(user),
                                secret_values=await secret_values(_uid(user), owner, slug))
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}/publish")
async def publish_state(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    return {"publish": xt.read_publish(owner, slug)}


@router.post("/{slug}/live")
async def make_live(slug: str, body: LiveBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    try:
        return {"live": await xt.make_live(owner, slug, body.version)}
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.post("/{slug}/run")
async def run(slug: str, body: RunBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Run the live version as the signed-in person, with their secrets."""
    owner, slug = _own(user, slug)
    caller = _uid(user)
    try:
        return await xt.run_tool(owner, slug, caller, body.input,
                                 secret_values=await secret_values(caller, owner, slug),
                                 resolve_file=_file_resolver(caller), deliver=_deliverer(caller, slug))
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}/secrets")
async def get_secrets(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    have = await secret_values(_uid(user), owner, slug)
    return {"secrets": [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]}


@router.put("/{slug}/secrets")
async def put_secrets(slug: str, body: SecretsBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Set (or, with "", clear) this person's values for the tool's secrets. Values
    are sealed and never sent back."""
    from captain_claw.flight_deck import fd_crypto

    owner, slug = _own(user, slug)
    caller = _uid(user)
    names = {s["name"] for s in xt.secret_names(owner, slug)}
    unknown = sorted(set(body.values) - names)
    if unknown:
        raise HTTPException(400, f"This tool doesn't ask for {', '.join(unknown)}.")
    db = get_db()
    to_set: dict[str, str] = {}
    for name, value in body.values.items():
        if len(value) > 8_000:
            raise HTTPException(400, f"{name} is longer than 8,000 characters.")
        key = xt.secret_key(owner, slug, name)
        if value == "":
            await db.delete_setting(caller, key)
            continue
        try:
            to_set[key] = fd_crypto.seal(value)
        except fd_crypto.CryptoUnavailable as exc:
            raise HTTPException(503, f"Secrets can't be kept safely here: {exc}") from None
    if to_set:
        await db.set_settings(caller, to_set)
    have = await secret_values(caller, owner, slug)
    return {"secrets": [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]}


# ── Agents ───────────────────────────────────────────────────────────────
#
# An agent calls its tools through these, never runs one itself. The person a
# call runs as (whose tools, whose secrets, whose files) is the agent's owner on
# Flight Deck's own record, or on a shared agent the member whose turn it is,
# proved by that turn's speaker grant (speaker_grants.acting_member). Included
# in server.py BEFORE ``router``, so /agent/... never reads as a tool's name.

agent_router = APIRouter(prefix="/fd/external-tools/agent", tags=["external-tools"])

# The routes a member's turn reaches with its grant (speaker_grants.GRANT_AWARE_PATHS).
GRANT_AWARE = frozenset({"/fd/external-tools/agent/tools", "/fd/external-tools/agent/describe",
                         "/fd/external-tools/agent/run"})


@dataclass(frozen=True)
class Runner:
    user_id: str     # whose tools, secrets and files
    agent: str       # the calling agent's slug
    member: bool     # a shared agent's member, not its owner


async def runner_of(request: Request) -> Runner:
    from captain_claw.flight_deck import speaker_grants
    from captain_claw.flight_deck.agent_identity import require_agent_caller
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if speaker_grants.member_request(request):
        m = await speaker_grants.acting_member(request)
        if m is None:  # not reached: a marked request returns a member or raises
            raise HTTPException(403, speaker_grants.NO_TURN_DETAIL)
        return Runner(m.user_id, m.slug, True)
    caller = require_agent_caller(request, what="custom tools")
    owner = caller.owner or ("" if _fd_auth_enabled() else "local")
    if not owner:
        raise HTTPException(403, "This agent has no owner on record, so it can't use custom tools.")
    return Runner(owner, caller.slug, False)


async def _agent_enabled(request: Request) -> Runner:
    runner = await runner_of(request)
    if not (await deck_settings())["enabled"]:
        raise HTTPException(403, OFF)
    return runner


def usable_tools(runner_id: str) -> list[dict[str, Any]]:
    """The tools *runner_id* can run now: their own live ones (Phase 3 adds the
    ones they added)."""
    return [{**t, "ref": t["slug"]} for t in xt.list_tools(runner_id) if t.get("live")]


def resolve_ref(runner_id: str, ref: str) -> tuple[str, str]:
    """``(owner, slug)`` for a tool *runner_id* named, or 404."""
    ref = str(ref or "").strip()
    try:
        slug = xt.check_slug(ref)
    except xt.ToolError:
        raise HTTPException(404, f"There's no custom tool called {ref!r}. custom_tools action=list shows yours.") from None
    if xt.exists(runner_id, slug):
        return runner_id, slug
    raise HTTPException(404, f"There's no custom tool called {slug!r}. custom_tools action=list shows yours.")


class AgentToolBody(BaseModel):
    tool: str = Field(..., max_length=120)


class AgentRunBody(BaseModel):
    tool: str = Field(..., max_length=120)
    input: dict[str, Any] = Field(default_factory=dict)


@agent_router.get("/status")
async def agent_status(request: Request) -> dict[str, Any]:
    """Whether custom tools are on here (the agent shows its tool only then)."""
    runner = await runner_of(request)
    settings = await deck_settings()
    out: dict[str, Any] = {"enabled": settings["enabled"]}
    if settings["enabled"]:
        out["tools"] = [{"ref": t["ref"], "title": t["title"], "description": t["description"]}
                        for t in usable_tools(runner.user_id)][:40]
        out["can_build"] = not runner.member and await _can_build({"id": runner.user_id})
    return out


@agent_router.get("/tools")
async def agent_tools(runner: Runner = Depends(_agent_enabled)) -> dict[str, Any]:
    tools = usable_tools(runner.user_id)
    out = []
    for t in tools:
        have = await secret_values(runner.user_id, t["owner"], t["slug"])
        missing = [s["name"] for s in xt.secret_names(t["owner"], t["slug"]) if s["name"] not in have]
        out.append({"ref": t["ref"], "title": t["title"], "description": t["description"],
                    "version": (t.get("live") or {}).get("version"), "secrets_missing": missing})
    return {"tools": out, "can_build": not runner.member and await _can_build({"id": runner.user_id})}


@agent_router.post("/describe")
async def agent_describe(body: AgentToolBody, runner: Runner = Depends(_agent_enabled)) -> dict[str, Any]:
    owner, slug = resolve_ref(runner.user_id, body.tool)
    detail = xt.detail(owner, slug)
    if not detail.get("manifest"):
        raise HTTPException(409, f"{slug} has no published version yet.")
    have = await secret_values(runner.user_id, owner, slug)
    manifest = detail["manifest"]
    return {
        "ref": body.tool, "version": (detail.get("live") or {}).get("version"), **manifest,
        "secrets": [{**s, "set": s["name"] in have} for s in manifest["secrets"]],
        "file_fields": sorted(xt.file_fields(manifest["input"])),
    }


@agent_router.post("/run")
async def agent_run(body: AgentRunBody, runner: Runner = Depends(_agent_enabled)) -> dict[str, Any]:
    owner, slug = resolve_ref(runner.user_id, body.tool)
    try:
        return await xt.run_tool(owner, slug, runner.user_id, body.input,
                                 secret_values=await secret_values(runner.user_id, owner, slug),
                                 resolve_file=_file_resolver(runner.user_id), deliver=_deliverer(runner.user_id, slug))
    except xt.ToolError as exc:
        raise _http(exc) from None


# ── Building and improving: a Crew task in the tool's own repository ─────


def build_brief(slug: str, title: str, what: str) -> str:
    return (f"Build the custom tool “{slug}” ({title}).\n\nWhat it should do:\n{what.strip()}\n\n"
            "The repository already holds a working starter (an echo tool): replace it with the real tool, and make "
            "tool.json, the tests and the examples match what it now does.")


def improve_brief(slug: str, what: str) -> str:
    return f"Change the custom tool “{slug}”.\n\n{what.strip()}"


async def _build_target(uid: str, slug: str, *, new: bool, title: str, what: str) -> tuple[str, bool]:
    """Check (and for a new tool, make) the tool a task will work on: ``(slug, made)``."""
    if len(what.strip()) < 10:
        raise HTTPException(422, "Say what the tool should do (its input, what it returns, an example), in a few sentences.")
    try:
        slug = xt.check_slug(slug)
    except xt.ToolError as exc:
        raise _http(exc) from None
    if not new:
        if not xt.exists(uid, slug):
            raise HTTPException(404, f"You have no custom tool called {slug}.")
        return slug, False
    if xt.exists(uid, slug):
        raise HTTPException(409, f"You already have a tool called {slug}: improve it, or pick another name.")
    first = re.split(r"(?<=[.!?])\s", " ".join(what.split()), maxsplit=1)[0][:600]
    try:
        await xt.create_tool(uid, slug, title=title or slug, description=first)
    except xt.ToolError as exc:
        raise _http(exc) from None
    return slug, True


async def _drop_if_made(uid: str, slug: str, made: bool) -> None:
    """A tool made for a task that then couldn't start goes again, so asking twice works."""
    if made:
        try:
            await xt.delete_tool(uid, slug)
        except xt.ToolError:
            pass


def _task_out(task: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "title", "status", "harness", "harness_label", "branch", "repo_label", "kind")
    return {k: task.get(k) for k in keys}


class BuildBody(BaseModel):
    slug: str = Field(..., max_length=48)
    title: str = Field("", max_length=200)
    what: str = Field(..., max_length=8000)
    harness: str = Field(..., max_length=40)
    model: str = Field("", max_length=200)


class ImproveBody(BaseModel):
    what: str = Field(..., max_length=8000)
    harness: str = Field(..., max_length=40)
    model: str = Field("", max_length=200)


async def _person_dispatch(user: dict, slug: str, brief: str, title: str, harness: str, model: str) -> dict[str, Any]:
    from captain_claw.flight_deck import crew_routes, crew_runner

    row, grant = await crew_routes._approved(user)
    task = await crew_runner.dispatch(row, grant, {"harness": harness, "brief": brief, "title": title, "kind": "ship",
                                                   "deliver": "branch", "tool": slug, "model": model})
    return crew_runner.view(task, full=True)


@router.post("/build")
async def build(body: BuildBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """A new tool from what it should do: its repository, then a Crew task that writes it."""
    await _require_builder(user)
    uid = _uid(user)
    title = " ".join(body.title.split())[:80]
    slug, made = await _build_target(uid, body.slug, new=True, title=title, what=body.what)
    try:
        task = await _person_dispatch(user, slug, build_brief(slug, title or slug, body.what),
                                      f"Build custom tool {slug}", body.harness, body.model)
    except BaseException:
        await _drop_if_made(uid, slug, made)
        raise
    return {"tool": xt.summary(uid, slug), "task": _task_out(task)}


@router.post("/{slug}/improve")
async def improve(slug: str, body: ImproveBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    await _require_builder(user)
    uid = _uid(user)
    slug, _ = await _build_target(uid, slug, new=False, title="", what=body.what)
    task = await _person_dispatch(user, slug, improve_brief(slug, body.what), f"Improve custom tool {slug}",
                                  body.harness, body.model)
    return {"tool": xt.summary(uid, slug), "task": _task_out(task)}


class AgentBuildBody(BaseModel):
    tool: str = Field(..., max_length=48)
    title: str = Field("", max_length=200)
    what: str = Field(..., max_length=8000)
    harness: str = Field(..., max_length=40)
    model: str = Field("", max_length=200)
    source_host: str = "localhost"
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0


async def _agent_dispatch(request: Request, body: AgentBuildBody, brief: str, title: str) -> dict[str, Any]:
    """Through Crew's own agent route, so the task is the agent's (callback, budget, origin)."""
    from captain_claw.flight_deck import crew_routes

    return await crew_routes.agent_dispatch(crew_routes.AgentDispatchBody(
        harness=body.harness, brief=brief, title=title, kind="ship", deliver="branch", tool=body.tool,
        model=body.model, source_host=body.source_host or "localhost", origin_platform=body.origin_platform,
        origin_user_id=body.origin_user_id, origin_chat_id=body.origin_chat_id,
    ), request)


async def _agent_builder(request: Request) -> Runner:
    runner = await _agent_enabled(request)
    if runner.member:
        raise HTTPException(403, "Only the agent's owner can have custom tools made or changed.")
    if not await _can_build({"id": runner.user_id}):
        raise HTTPException(403, NEEDS_CREW)
    return runner


@agent_router.post("/build")
async def agent_build(body: AgentBuildBody, request: Request) -> dict[str, Any]:
    runner = await _agent_builder(request)
    title = " ".join(body.title.split())[:80]
    slug, made = await _build_target(runner.user_id, body.tool, new=True, title=title, what=body.what)
    body.tool = slug
    try:
        task = await _agent_dispatch(request, body, build_brief(slug, title or slug, body.what), f"Build custom tool {slug}")
    except BaseException:
        await _drop_if_made(runner.user_id, slug, made)
        raise
    return {"tool": xt.summary(runner.user_id, slug), "task": _task_out(task)}


@agent_router.post("/improve")
async def agent_improve(body: AgentBuildBody, request: Request) -> dict[str, Any]:
    runner = await _agent_builder(request)
    slug, _ = await _build_target(runner.user_id, body.tool, new=False, title="", what=body.what)
    body.tool = slug
    task = await _agent_dispatch(request, body, improve_brief(slug, body.what), f"Improve custom tool {slug}")
    return {"tool": xt.summary(runner.user_id, slug), "task": _task_out(task)}
