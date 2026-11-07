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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import external_tool_review as xtr
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
        "require_sandbox": bool(cfg.get("external_tools_require_sandbox")),
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


# ── Sharing: who may use a tool that isn't theirs ────────────────────────
#
# A tool is offered to a person (a resource_shares row, type "external_tool",
# id "<owner>/<slug>") or to everybody (external_tool_listings, approved by an
# admin unless the admin turned that off). An offer is only an offer: nobody's
# agent runs someone else's tool until that person ADDED it, which is their
# consent (a server-owned setting). Taking an offer back ends the use at once.

SHARE_TYPE = "external_tool"
ADDED_PREFIX = "xtool-added:"


def share_id(owner: str, slug: str) -> str:
    return f"{owner}/{slug}"


def added_key(owner: str, slug: str) -> str:
    return f"{ADDED_PREFIX}{owner}/{slug}"


def listing_live(listing: dict | None, settings: dict[str, bool]) -> bool:
    if not listing:
        return False
    status = listing.get("status")
    return status == "approved" or (status == "pending" and not settings["everybody_needs_approval"])


async def offered_to(user_id: str, owner: str, slug: str) -> str:
    """How *owner*'s tool reaches *user_id*: "person", "everybody" or ""."""
    db = get_db()
    if await db.get_share_for_grantee(SHARE_TYPE, share_id(owner, slug), user_id, owner):
        return "person"
    if listing_live(await db.get_tool_listing(owner, slug), await deck_settings()):
        return "everybody"
    return ""


async def can_use(user_id: str, owner: str, slug: str) -> bool:
    """Their own tool, or one offered to them that they added."""
    if owner == user_id:
        return True
    if not await get_db().get_setting(user_id, added_key(owner, slug)):
        return False
    return bool(await offered_to(user_id, owner, slug))


def _declares(owner: str, slug: str) -> dict[str, Any] | None:
    """What the live version says it needs, for consent and update bells."""
    live = xt.read_live(owner, slug)
    vdir = xt._ready(owner, slug, live["version"]) if live else None
    if vdir is None:
        return None
    try:
        m = xt.load_manifest(vdir, slug=slug)
    except xt.ToolError:
        return None
    return {"version": live["version"], "network": m.network, "secrets": [s["name"] for s in m.secrets],
            "timeout_s": m.timeout_s, "memory_mb": m.memory_mb}


def declares_more(old: dict | None, new: dict | None) -> list[str]:
    """What *new* asks for that *old* didn't, in words."""
    if not old or not new:
        return []
    out = []
    if new["network"] and not old["network"]:
        out.append("internet access")
    extra = [n for n in new["secrets"] if n not in old["secrets"]]
    if extra:
        out.append("the secret" + ("s " if len(extra) > 1 else " ") + ", ".join(extra))
    if new["timeout_s"] > old["timeout_s"]:
        out.append(f"up to {new['timeout_s']} s per run")
    if new["memory_mb"] > old["memory_mb"]:
        out.append(f"up to {new['memory_mb']} MB of memory")
    return out


async def _person(uid: str) -> dict[str, str]:
    row = await get_db().get_user_by_id(uid) if uid else None
    if not row:
        return {"id": uid, "email": "", "name": uid}
    return {"id": uid, "email": row.get("email") or "", "name": row.get("display_name") or row.get("email") or uid}


async def _offer_view(user_id: str, owner: str, slug: str, how: str) -> dict[str, Any] | None:
    summary = xt.summary(owner, slug)
    if not summary.get("live"):
        return None
    return {**summary, "owner_person": await _person(owner), "offered": how, "declares": _declares(owner, slug),
            "added": bool(await get_db().get_setting(user_id, added_key(owner, slug)))}


def _use_target(user: dict, slug: str, owner: str) -> tuple[str, str, str]:
    """``(caller, owner, slug)`` before the access check: the caller's own tool when
    *owner* is empty."""
    uid = _uid(user)
    try:
        slug = xt.check_slug(slug)
        owner = xt._check_owner(owner) if owner else uid
    except xt.ToolError:
        raise HTTPException(404, f"There's no tool called {slug}.") from None
    if not xt.exists(owner, slug):
        raise HTTPException(404, f"There's no tool called {slug}.")
    return uid, owner, slug


async def _usable(user: dict, slug: str, owner: str) -> tuple[str, str, str]:
    uid, owner, slug = _use_target(user, slug, owner)
    if not await can_use(uid, owner, slug):
        raise HTTPException(404, f"There's no tool called {slug}.")
    return uid, owner, slug


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
    return {**settings, "can_build": settings["enabled"] and await _can_build(user),
            "sandbox": await asyncio.to_thread(xt.sandbox_info)}


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
async def get_one(slug: str, owner: str = Query(""), user: dict = Depends(_enabled)) -> dict[str, Any]:
    uid, owner, slug = _use_target(user, slug, owner)
    if owner == uid:
        out = xt.detail(owner, slug)
        out["can_build"] = await _can_build(user)
    else:
        how = await offered_to(uid, owner, slug)
        if not how:
            raise HTTPException(404, f"There's no tool called {slug}.")
        detail = xt.detail(owner, slug)
        # Someone else's tool: what it does and needs, never its repository, versions or logs.
        out = {**{k: detail.get(k) for k in ("owner", "slug", "title", "description", "live", "manifest")},
               "owner_person": await _person(owner), "offered": how, "declares": _declares(owner, slug),
               "added": bool(await get_db().get_setting(uid, added_key(owner, slug)))}
    have = await secret_values(uid, owner, slug)
    out["secrets"] = [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]
    return out


@router.delete("/{slug}")
async def delete(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    try:
        await xt.delete_tool(owner, slug)
    except xt.ToolError as exc:
        raise _http(exc) from None
    # Its offers, everyone's consent and everyone's secrets for it go with it.
    db = get_db()
    await db.delete_shares_for_resource(SHARE_TYPE, share_id(owner, slug), owner)
    await db.delete_tool_listing(owner, slug)
    await db.delete_setting_for_all_users(added_key(owner, slug))
    prefix = xt.secret_key(owner, slug, "")
    for uid in {owner, *[u for u in (await _holders_of_prefix(prefix))]}:
        for key in await db.get_all_settings(uid):
            if key.startswith(prefix):
                await db.delete_setting(uid, key)
    return {"ok": True}


async def _holders_of_prefix(prefix: str) -> list[str]:
    """Users with any setting under *prefix* (a tool's secrets)."""
    db = get_db()
    rows = await db._db.execute_fetchall(
        "SELECT DISTINCT user_id FROM user_settings WHERE key LIKE ? ESCAPE '\\'",
        (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",))
    return [str(r["user_id"]) for r in rows]


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
async def run(slug: str, body: RunBody, owner: str = Query(""), user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Run the live version as the signed-in person, with their secrets: their own
    tool, or (with *owner*) one offered to them that they added."""
    caller, owner, slug = await _usable(user, slug, owner)
    try:
        return await xt.run_tool(owner, slug, caller, body.input,
                                 secret_values=await secret_values(caller, owner, slug),
                                 resolve_file=_file_resolver(caller), deliver=_deliverer(caller, slug))
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}/secrets")
async def get_secrets(slug: str, owner: str = Query(""), user: dict = Depends(_enabled)) -> dict[str, Any]:
    uid, owner, slug = _use_target(user, slug, owner)
    if owner != uid and not await offered_to(uid, owner, slug):
        raise HTTPException(404, f"There's no tool called {slug}.")
    have = await secret_values(uid, owner, slug)
    return {"secrets": [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]}


@router.put("/{slug}/secrets")
async def put_secrets(slug: str, body: SecretsBody, owner: str = Query(""),
                      user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Set (or, with "", clear) this person's values for the tool's secrets. Values
    are sealed and never sent back."""
    from captain_claw.flight_deck import fd_crypto

    caller, owner, slug = _use_target(user, slug, owner)
    if owner != caller and not await offered_to(caller, owner, slug):
        raise HTTPException(404, f"There's no tool called {slug}.")
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


async def added_tools(user_id: str) -> list[tuple[str, str]]:
    """``(owner, slug)`` of the tools *user_id* added that are still offered to them."""
    out = []
    for key in await get_db().get_all_settings(user_id):
        if not key.startswith(ADDED_PREFIX):
            continue
        owner, _, slug = key[len(ADDED_PREFIX):].partition("/")
        try:
            if xt.exists(owner, slug) and await offered_to(user_id, owner, slug):
                out.append((owner, slug))
        except xt.ToolError:
            continue
    return sorted(out)


def _owner_label(person: dict[str, str]) -> str:
    base = (person.get("email") or "").split("@")[0] or person.get("id", "")[:8]
    return re.sub(r"[^a-z0-9-]+", "-", base.lower()).strip("-") or "someone"


async def usable_tools(runner_id: str) -> list[dict[str, Any]]:
    """The tools *runner_id* can run now: their own live ones, then the ones they
    added. ``ref`` is how the agent names one: the tool's name, or
    ``<owner>/<name>`` when two would share it."""
    own = [{**t, "ref": t["slug"]} for t in xt.list_tools(runner_id) if t.get("live")]
    taken = {t["slug"] for t in own}
    added = []
    for owner, slug in await added_tools(runner_id):
        summary = xt.summary(owner, slug)
        if not summary.get("live"):
            continue
        person = await _person(owner)
        added.append({**summary, "owner_person": person, "label": _owner_label(person)})
    counts: dict[str, int] = {}
    for t in added:
        counts[t["slug"]] = counts.get(t["slug"], 0) + 1
    for t in added:
        clash = t["slug"] in taken or counts[t["slug"]] > 1
        t["ref"] = f"{t['label']}/{t['slug']}" if clash else t["slug"]
    return own + added


async def resolve_ref(runner_id: str, ref: str) -> tuple[str, str]:
    """``(owner, slug)`` for a tool *runner_id* named, or 404."""
    ref = str(ref or "").strip()
    missing = HTTPException(404, f"There's no custom tool called {ref!r}. custom_tools action=list shows the ones you can use.")
    if "/" not in ref:
        try:
            slug = xt.check_slug(ref)
        except xt.ToolError:
            raise missing from None
        if xt.exists(runner_id, slug):
            return runner_id, slug
    for t in await usable_tools(runner_id):
        if t["ref"] == ref:
            return t["owner"], t["slug"]
    raise missing


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
                        for t in await usable_tools(runner.user_id)][:40]
        out["can_build"] = not runner.member and await _can_build({"id": runner.user_id})
    return out


@agent_router.get("/tools")
async def agent_tools(runner: Runner = Depends(_agent_enabled)) -> dict[str, Any]:
    tools = await usable_tools(runner.user_id)
    out = []
    for t in tools:
        have = await secret_values(runner.user_id, t["owner"], t["slug"])
        missing = [s["name"] for s in xt.secret_names(t["owner"], t["slug"]) if s["name"] not in have]
        out.append({"ref": t["ref"], "title": t["title"], "description": t["description"],
                    "version": (t.get("live") or {}).get("version"), "secrets_missing": missing,
                    **({"by": t["owner_person"]["name"]} if t.get("owner_person") else {})})
    return {"tools": out, "can_build": not runner.member and await _can_build({"id": runner.user_id})}


@agent_router.post("/describe")
async def agent_describe(body: AgentToolBody, runner: Runner = Depends(_agent_enabled)) -> dict[str, Any]:
    owner, slug = await resolve_ref(runner.user_id, body.tool)
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
    owner, slug = await resolve_ref(runner.user_id, body.tool)
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


class ReviewBody(BaseModel):
    slug: str = Field("", max_length=48)
    title: str = Field("", max_length=200)
    what: str = Field(..., max_length=xtr.MAX_SPEC)


@router.post("/review")
async def review_spec(body: ReviewBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Step two of New tool: the person's model reads the spec with what Captain
    is and their tools, and answers with a fuller spec and why. Nothing is kept.
    A name or a spec Flight Deck refuses is an HTTP error (fixed in step one);
    a check that gave no spec is ``{"error": ...}``: the person can still build
    what they wrote."""
    await _require_builder(user)
    uid = _uid(user)
    what = body.what.strip()
    if len(what) < 10:
        raise HTTPException(422, "Say what the tool should do (its input, what it returns, an example), in a few sentences.")
    slug = body.slug.strip()
    if slug:
        try:
            slug = xt.check_slug(slug)
        except xt.ToolError as exc:
            raise _http(exc) from None
        if slug in xt.RESERVED_SLUGS:
            raise HTTPException(400, f"“{slug}” is a name Flight Deck uses itself: pick another.")
        if xt.exists(uid, slug):
            raise HTTPException(409, f"You already have a tool called {slug}: improve it, or pick another name.")
    own = xt.list_tools(uid)
    added = [t for t in await usable_tools(uid) if t.get("owner") != uid]
    try:
        return await xtr.review(uid, slug, " ".join(body.title.split())[:80], what, own, added)
    except xtr.ReviewError as exc:
        return {"error": exc.message}


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


# ── Sharing routes ───────────────────────────────────────────────────────
#
# Fixed paths, on their own router included BEFORE ``router`` in server.py, so
# /shared, /catalog, /added and /approvals never read as a tool's name.

catalog_router = APIRouter(prefix="/fd/external-tools", tags=["external-tools"])


class AddBody(BaseModel):
    owner: str = Field(..., max_length=64)
    slug: str = Field(..., max_length=48)


class EverybodyBody(BaseModel):
    on: bool


class DecideBody(BaseModel):
    approve: bool


@catalog_router.get("/shared")
async def shared_with_me(user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Tools someone offered to this person, live ones only."""
    uid = _uid(user)
    out = []
    for share in await get_db().list_shares_for_grantee(uid, SHARE_TYPE):
        owner, _, slug = str(share.get("resource_id") or "").partition("/")
        try:
            if owner != share.get("owner_id") or not xt.exists(owner, slug):
                continue
        except xt.ToolError:
            continue
        view = await _offer_view(uid, owner, slug, "person")
        if view:
            out.append(view)
    return {"tools": out}


@catalog_router.get("/catalog")
async def catalog(user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Tools offered to everybody on this deck (approved, or waiting while the
    admin's approval is off), live ones only, not the person's own."""
    uid = _uid(user)
    settings = await deck_settings()
    out = []
    for listing in await get_db().list_tool_listings():
        owner, slug = listing["owner_id"], listing["slug"]
        if owner == uid or not listing_live(listing, settings):
            continue
        try:
            if not xt.exists(owner, slug):
                continue
        except xt.ToolError:
            continue
        view = await _offer_view(uid, owner, slug, "everybody")
        if view:
            out.append(view)
    return {"tools": out}


@catalog_router.post("/added")
async def add(body: AddBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """This person's consent: their agents may run this tool, as them, with their
    secrets and files. Recorded with what the tool said it needs."""
    uid = _uid(user)
    try:
        slug = xt.check_slug(body.slug)
        owner = xt._check_owner(body.owner)
    except xt.ToolError:
        raise HTTPException(404, "There's no such tool.") from None
    if owner == uid:
        raise HTTPException(400, "It's your own tool: your agents can use it already.")
    if not xt.exists(owner, slug) or not await offered_to(uid, owner, slug):
        raise HTTPException(404, "There's no such tool offered to you.")
    declares = _declares(owner, slug)
    if declares is None:
        raise HTTPException(409, "It has no published version yet.")
    import json as _json

    await get_db().set_settings(uid, {added_key(owner, slug): _json.dumps({"at": time.time(), **declares})})
    return {"added": True, "tool": await _offer_view(uid, owner, slug, await offered_to(uid, owner, slug))}


@catalog_router.delete("/added")
async def remove(owner: str = Query(...), slug: str = Query(...), user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Take the consent back; this person's secrets for the tool go too."""
    uid = _uid(user)
    db = get_db()
    try:
        key = added_key(xt._check_owner(owner), xt.check_slug(slug))
        prefix = xt.secret_key(owner, slug, "")
    except xt.ToolError:
        raise HTTPException(404, "There's no such tool.") from None
    await db.delete_setting(uid, key)
    for k in await db.get_all_settings(uid):
        if k.startswith(prefix):
            await db.delete_setting(uid, k)
    return {"added": False}


async def _require_admin_user(user: dict = Depends(_enabled)) -> dict:
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if _fd_auth_enabled() and user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user


@catalog_router.get("/approvals")
async def approvals(user: dict = Depends(_require_admin_user)) -> dict[str, Any]:
    """Tools waiting for an admin before everybody is offered them."""
    out = []
    for listing in await get_db().list_tool_listings("pending"):
        owner, slug = listing["owner_id"], listing["slug"]
        try:
            if not xt.exists(owner, slug):
                continue
        except xt.ToolError:
            continue
        detail = xt.detail(owner, slug)
        out.append({"owner": owner, "slug": slug, "owner_person": await _person(owner),
                    "title": detail.get("title"), "description": detail.get("description"),
                    "live": detail.get("live"), "manifest": detail.get("manifest"),
                    "requested_at": listing.get("requested_at"),
                    "publish": {k: (detail.get("publish") or {}).get(k) for k in ("state", "version", "error")}})
    settings = await deck_settings()
    return {"approvals": out, "needs_approval": settings["everybody_needs_approval"]}


@catalog_router.post("/approvals/{owner}/{slug}")
async def decide(owner: str, slug: str, body: DecideBody, user: dict = Depends(_require_admin_user)) -> dict[str, Any]:
    db = get_db()
    status = "approved" if body.approve else "rejected"
    if not await db.decide_tool_listing(owner, slug, status, _uid(user)):
        raise HTTPException(409, "It isn't waiting for a decision any more.")
    try:
        what = "is now offered to everybody" if body.approve else "wasn't approved for everybody"
        await db.add_notification(owner, "custom_tool", f"Your custom tool “{slug}” {what}",
                                  "" if body.approve else "You can still offer it to people one by one.",
                                  ref_type="custom_tool", ref_id=share_id(owner, slug))
    except Exception:
        pass
    return {"listing": await db.get_tool_listing(owner, slug)}


@router.get("/{slug}/sharing")
async def sharing(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Who it's offered to, the everybody offer, and how many people added it."""
    owner, slug = _own(user, slug)
    db = get_db()
    settings = await deck_settings()
    listing = await db.get_tool_listing(owner, slug)
    people = await db.list_shares_for_resource(SHARE_TYPE, share_id(owner, slug), owner)
    holders = await db.users_with_setting(added_key(owner, slug))
    return {"share_id": share_id(owner, slug), "people": people,
            "everybody": ({**listing, "live": listing_live(listing, settings)} if listing else None),
            "needs_approval": settings["everybody_needs_approval"], "added_by": len(holders)}


@router.put("/{slug}/everybody")
async def offer_everybody(slug: str, body: EverybodyBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Offer it to everybody on the deck (waiting for an admin while that approval is
    on), or take that offer back."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    owner, slug = _own(user, slug)
    db = get_db()
    if not body.on:
        await db.delete_tool_listing(owner, slug)
        return {"everybody": None}
    if not _fd_auth_enabled():
        raise HTTPException(400, "This deck has no sign-in, so there's nobody else to offer it to.")
    if not xt.read_live(owner, slug):
        raise HTTPException(409, "Publish it first: only a published tool can be offered.")
    settings = await deck_settings()
    status = "pending" if settings["everybody_needs_approval"] else "approved"
    listing = await db.set_tool_listing(owner, slug, status)
    if status == "pending":
        who = user.get("display_name") or user.get("email") or "Someone"
        for admin_id in await db.list_admin_ids():
            if admin_id == owner:
                continue
            try:
                await db.add_notification(admin_id, "custom_tool", f"{who} asks to offer the custom tool “{slug}” to everybody",
                                          "Approve or decline it under Custom tools → Approvals.",
                                          ref_type="custom_tool_approval", ref_id=share_id(owner, slug))
            except Exception:
                pass
    return {"everybody": {**listing, "live": listing_live(listing, settings)}}


@router.get("/{slug}/usage")
async def usage(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """How the tool is used: calls and failures per day, versions and how many people.
    Never anyone's inputs, outputs or files."""
    owner, slug = _own(user, slug)
    return xt.usage_summary(owner, slug)


# ── After a publish: tell the people who added it what a new version asks for ──


async def after_publish(owner: str, slug: str, old_version: str | None, new_version: str) -> None:
    if not old_version or old_version == new_version:
        return
    old = new = None
    for version, target in ((old_version, "old"), (new_version, "new")):
        vdir = xt._ready(owner, slug, version)
        if vdir is None:
            return
        try:
            m = xt.load_manifest(vdir, slug=slug)
        except xt.ToolError:
            return
        d = {"network": m.network, "secrets": [s["name"] for s in m.secrets], "timeout_s": m.timeout_s,
             "memory_mb": m.memory_mb, "title": m.title}
        if target == "old":
            old = d
        else:
            new = d
    more = declares_more(old, new)
    if not more:
        return
    db = get_db()
    for uid in await db.users_with_setting(added_key(owner, slug)):
        try:
            await db.add_notification(uid, "custom_tool", f"“{new['title']}” updated: it now asks for {', '.join(more)}",
                                      "Your agents use the new version. Remove it under Custom tools if you'd rather not.",
                                      ref_type="custom_tool", ref_id=share_id(owner, slug))
        except Exception:
            pass


xt.AFTER_PUBLISH.append(after_publish)
