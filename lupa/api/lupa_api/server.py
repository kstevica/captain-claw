"""Lupa BFF — FastAPI app.

Single public surface for the product: serves the SPA, proxies auth to Flight
Deck first-party (so FD's refresh cookie works cross-origin-free), verifies
access JWTs locally when ``FD_JWT_SECRET`` is shared (falls back to proxying
``/fd/auth/me``), and drives the commission lifecycle over FD's HTTP API.

FD is expected on loopback (``LUPA_FD_URL``, default http://127.0.0.1:25080)
with auth enabled. This process never imports captain_claw.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import jwt as pyjwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lupa_api.db import LupaDB
from lupa_api.packs import (
    active_pack_slug,
    list_seed_packs,
    load_pack,
    pack_quality,
    row_manifest,
)

def _fd_url() -> str:
    return os.environ.get("LUPA_FD_URL", "http://127.0.0.1:25080")


def _data_dir() -> Path:
    return Path(os.environ.get("LUPA_DATA_DIR", "./lupa-data"))


_FD_COOKIE_PATH = "Path=/fd/auth"
_OUR_COOKIE_PATH = "Path=/api/auth"

log = logging.getLogger("lupa")

# token → (user dict, expiry) for the /fd/auth/me fallback validator.
_me_cache: dict[str, tuple[dict, float]] = {}
_ME_CACHE_TTL = 60.0


def _fd(request: Request) -> httpx.AsyncClient:
    return request.app.state.fd


def _db(request: Request) -> LupaDB:
    return request.app.state.db


def _bearer(request: Request) -> str:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else ""


def _auth_headers(request: Request) -> dict:
    tok = _bearer(request)
    return {"Authorization": f"Bearer {tok}"} if tok else {}


async def require_user(request: Request) -> dict:
    """The product-route auth dependency.

    With ``FD_JWT_SECRET`` shared, verifies the FD-issued access JWT locally
    (no round trip). Without it, validates by proxying ``GET /fd/auth/me``
    (cached briefly) — slower but zero-config.
    """
    token = _bearer(request)
    if not token:
        raise HTTPException(401, "missing bearer token")
    secret = os.environ.get("FD_JWT_SECRET", "")
    if secret:
        try:
            payload = pyjwt.decode(token, secret, algorithms=["HS256"])
        except pyjwt.PyJWTError:
            raise HTTPException(401, "invalid or expired token")
        if payload.get("type") not in (None, "access"):
            raise HTTPException(401, "not an access token")
        return {"id": payload.get("sub", ""), "role": payload.get("role", "user")}
    cached = _me_cache.get(token)
    if cached and cached[1] > time.monotonic():
        return cached[0]
    r = await _fd(request).get("/fd/auth/me",
                               headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        raise HTTPException(401, "invalid or expired token")
    user = r.json()
    _me_cache[token] = (user, time.monotonic() + _ME_CACHE_TTL)
    return user


async def _forward(resp: httpx.Response) -> Response:
    """Pass an FD response through, rewriting the refresh-cookie path so the
    browser scopes it to our proxy prefix instead of FD's."""
    out = Response(content=resp.content, status_code=resp.status_code,
                   media_type=resp.headers.get("content-type", "application/json"))
    for sc in resp.headers.get_list("set-cookie"):
        out.headers.append("set-cookie", sc.replace(_FD_COOKIE_PATH, _OUR_COOKIE_PATH))
    return out


def _fd_error(resp: httpx.Response) -> HTTPException:
    try:
        detail = resp.json().get("detail", resp.text)
    except (ValueError, AttributeError):
        detail = resp.text
    return HTTPException(resp.status_code, detail)


# ── request models ───────────────────────────────────────────────────


class StreamCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    pack: str = ""  # the desk this stream belongs to; "" → the default pack


class PackCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,39}$")
    name: str = Field(min_length=1, max_length=80)
    tagline: str = ""


class PackManifestBody(BaseModel):
    manifest: dict


class GenerateBody(BaseModel):
    instructions: str = Field(min_length=1)


class CreatorAdd(BaseModel):
    user_id: str = Field(min_length=1)


class CommissionCreate(BaseModel):
    brief: str = Field(min_length=1)
    kind: str = "auto"  # auto | initial | continue | revise | fill_gaps
    max_agents: int = Field(default=0, ge=0, le=10)  # 0 → stream/default


class SettingsBody(BaseModel):
    """Stream-level overrides; explicit null resets a knob to the pack default."""
    quality_profile: str | None = None
    same_cast: bool | None = None
    max_agents: int | None = Field(default=None, ge=1, le=10)
    archetype_ids: list[str] | None = None  # pinned house cast; [] clears


class ApproveBody(BaseModel):
    plan: dict | None = None


class BriefBody(BaseModel):
    instruction: str = Field(min_length=1)
    cadence_hours: float = Field(gt=0, le=24 * 30)
    enabled: bool = True


_CONTINUE_KINDS = ("continue", "revise", "fill_gaps")
_QUALITY_PROFILES = ("off", "balanced", "thorough")


def _stream_settings(stream: dict) -> dict:
    raw = stream.get("settings")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}


def _stream_quality(stream: dict, pack: dict) -> dict:
    """The quality profile a commission runs with: stream override, else pack."""
    prof = _stream_settings(stream).get("quality_profile")
    if prof in _QUALITY_PROFILES:
        return {"profile": prof}
    return pack_quality(pack)


# ── standing-brief scheduler (product-side; deliberately NOT FD's
#    global /scheduler/*, which has no user scoping) ─────────────────


def _mint_owner_token(sub: str, secret: str) -> str:
    """A short-lived access token for a brief's owner — the BFF shares
    FD_JWT_SECRET with FD, so scheduled runs authenticate as the user."""
    now = int(time.time())
    return pyjwt.encode({"sub": sub, "role": "user", "type": "access",
                         "iat": now, "exp": now + 900}, secret, algorithm="HS256")


def _brief_instruction(text: str) -> str:
    return ("Standing brief — this is a recurring monitoring round: report ONLY "
            "what is new or changed since the previous round, in a short delta "
            f"format. The standing question: {text}")


async def fire_due_briefs(state) -> int:
    """One scheduler pass; returns how many briefs fired. Kept separate from
    the sleep loop so tests can drive it deterministically.

    Continuation rounds inherit the parent run's quality profile, and both
    `balanced` and `thorough` include the delta_rounds lever — so brief
    rounds get delta behavior without any quality plumbing here."""
    secret = os.environ.get("FD_JWT_SECRET", "")
    if not secret:
        return 0  # minting owner tokens requires the shared secret
    db, fd = state.db, state.fd
    now = datetime.now(timezone.utc)
    fired = 0
    for brief in await db.list_due_briefs(now.isoformat()):
        try:
            rounds = await db.list_rounds(brief["stream_id"])
            if not rounds:
                continue  # a brief needs an initial round to continue from
            headers = {"Authorization":
                       f"Bearer {_mint_owner_token(brief['user_id'], secret)}"}
            tip = rounds[-1]["session_id"]
            r = await fd.get(f"/fd/basna/sessions/{tip}", headers=headers)
            if r.status_code != 200 or r.json().get("status") != "done":
                continue  # tip still running/failed — retried next tick
            stream = await db.get_stream(brief["stream_id"], brief["user_id"])
            if not stream:
                continue
            settings = _stream_settings(stream)
            resp = await fd.post(
                f"/fd/vatra/sessions/{tip}/continue",
                json={"instruction": _brief_instruction(brief["instruction"]),
                      "kind": "continue",
                      "same_cast": bool(settings.get("same_cast", True))},
                headers=headers)
            if resp.status_code != 200:
                log.warning("brief continue failed for stream %s: %s",
                            brief["stream_id"], resp.status_code)
                continue
            sid = resp.json().get("session_id") or ""
            if not sid:
                continue
            await db.add_round(brief["stream_id"], sid, "brief")
            nxt = (now + timedelta(hours=float(brief["cadence_hours"]))).isoformat()
            await db.mark_brief_ran(brief["stream_id"], sid, nxt)
            fired += 1
        except Exception:  # noqa: BLE001 — one bad brief must not stop the rest
            log.exception("brief scheduler pass failed for stream %s",
                          brief.get("stream_id"))
    return fired


async def _brief_loop(state) -> None:
    tick = float(os.environ.get("LUPA_BRIEF_TICK_SECONDS", "60"))
    while True:
        try:
            await fire_due_briefs(state)
        except Exception:  # noqa: BLE001
            log.exception("brief scheduler tick failed")
        await asyncio.sleep(tick)


# ── Kalup factory: headless Vatra runs for generate/evaluate ─────────


def _factory_poll_s() -> float:
    return float(os.environ.get("LUPA_FACTORY_POLL_SECONDS", "2"))


def _factory_timeout() -> float:
    return float(os.environ.get("LUPA_FACTORY_TIMEOUT_SECONDS", "900"))


async def _run_headless_vatra(state, token: str, intent: str,
                              quality: dict) -> tuple[dict | None, str]:
    """Start → auto-approve at the gate → poll to a terminal state.

    The Studio's generate/evaluate runs are unattended, so the plan gate is
    auto-approved (the human gate stays for customer commissions). Returns
    (session detail, "") or (None, error)."""
    headers = {"Authorization": f"Bearer {token}"}
    r = await state.fd.post("/fd/vatra/start",
                            json={"intent": intent, "quality": quality},
                            headers=headers)
    if r.status_code != 200:
        return None, f"start failed ({r.status_code})"
    sid = r.json().get("session_id") or ""
    if not sid:
        return None, "start returned no session id"
    deadline = time.monotonic() + _factory_timeout()
    approved = False
    while time.monotonic() < deadline:
        r = await state.fd.get(f"/fd/basna/sessions/{sid}", headers=headers)
        if r.status_code != 200:
            return None, f"session poll failed ({r.status_code})"
        detail = r.json()
        status = detail.get("status", "")
        if status == "awaiting_plan" and not approved:
            ok = await state.fd.post("/fd/vatra/plan/approve",
                                     json={"session_id": sid, "quality": quality},
                                     headers=headers)
            approved = ok.status_code == 200
        elif status == "done":
            return detail, ""
        elif status in ("error", "cancelled", "rejected"):
            return None, f"run ended with status {status}"
        await asyncio.sleep(_factory_poll_s())
    return None, "factory run timed out"


def _extract_manifest(text: str) -> dict | None:
    """The generated pack manifest from a run's conclusion — a fenced ```json
    block, else the outermost {...}."""
    m = None
    fence = text.find("```json")
    if fence != -1:
        end = text.find("```", fence + 7)
        if end != -1:
            m = text[fence + 7:end]
    if m is None:
        start, stop = text.find("{"), text.rfind("}")
        if start == -1 or stop <= start:
            return None
        m = text[start:stop + 1]
    try:
        out = json.loads(m)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _generate_intent(name: str, instructions: str) -> str:
    return (
        "KALUP PACK DRAFT — you are configuring a vertical research desk, not "
        "writing a report. Produce ONLY a JSON object (in a ```json fence) that "
        "is the desk's pack manifest, with keys: name, tagline, "
        "theme {accent, accent_soft, bg, surface, border, text, text_dim}, "
        "vocabulary {stream, streams, commission, brief, round, report, "
        "plan_gate_title, plan_gate_hint, composer_placeholder, "
        "continue_placeholder, empty_streams, new_stream, receipts_title, "
        "receipts_hint, facts_title, cost_title, brief_title, brief_hint, "
        "brief_placeholder, inbox_title}, intake {types: [{id, label, "
        "description, default_max_agents}]}, quality {profile}, "
        "briefs {presets: [{id, label, hours}]}, roi {analyst_hourly_usd, "
        "analyst_label}, evals [{brief}], onboarding_md. Ground every string "
        f"in this vertical:\nDesk name: {name}\n{instructions}"
    )


def _eval_verdict(analysis: dict) -> tuple[str, dict]:
    """The ship-gate: green iff the golden run's own receipts pass."""
    metrics = (analysis or {}).get("quality_metrics") or {}
    verdict = str((analysis or {}).get("quality_verdict")
                  or metrics.get("quality_verdict") or "")
    green = (verdict.lower() == "pass"
             and int(metrics.get("contract_failed_critical", 0) or 0) == 0
             and int(metrics.get("consistency_critical", 0) or 0) == 0)
    return ("green" if green else "red"), metrics


# ── app ──────────────────────────────────────────────────────────────


def create_app(fd_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db = LupaDB(_data_dir() / "lupa.db")
        await app.state.db.init()
        app.state.fd = httpx.AsyncClient(
            base_url=_fd_url(), transport=fd_transport, timeout=60.0)
        # Seed the registry from repo packs, then serve the default pack FROM
        # the registry — runtime edits win over files from here on.
        for slug, manifest in list_seed_packs().items():
            await app.state.db.upsert_seed_pack(slug, manifest)
        row = await app.state.db.get_pack(active_pack_slug())
        app.state.pack = row_manifest(row) if row else load_pack()
        scheduler = asyncio.create_task(_brief_loop(app.state))
        yield
        scheduler.cancel()
        with suppress(asyncio.CancelledError):
            await scheduler
        await app.state.fd.aclose()
        await app.state.db.close()

    app = FastAPI(title="Lupa", lifespan=lifespan)

    # ── pack ─────────────────────────────────────────────────────────

    @app.get("/api/pack")
    async def get_pack(request: Request):
        return request.app.state.pack

    # ── auth proxy (first-party; cookie path rewritten) ──────────────

    @app.post("/api/auth/{action}")
    async def auth_proxy(action: str, request: Request):
        if action not in ("login", "register", "refresh", "logout"):
            raise HTTPException(404, "unknown auth action")
        body = await request.body()
        headers = {"content-type": request.headers.get("content-type", "application/json")}
        cookie = request.headers.get("cookie", "")
        if cookie:
            headers["cookie"] = cookie
        r = await _fd(request).post(f"/fd/auth/{action}", content=body, headers=headers)
        return await _forward(r)

    @app.get("/api/auth/me")
    async def auth_me(request: Request):
        r = await _fd(request).get("/fd/auth/me", headers=_auth_headers(request))
        return await _forward(r)

    # ── streams ──────────────────────────────────────────────────────

    async def _pack_by_slug(request: Request, slug: str) -> dict:
        """A stream's pack manifest from the registry; default pack fallback."""
        if slug:
            row = await _db(request).get_pack(slug)
            if row:
                return row_manifest(row)
        return request.app.state.pack

    @app.get("/api/streams")
    async def list_streams(request: Request, pack: str = "",
                           user: dict = Depends(require_user)):
        return {"streams": await _db(request).list_streams(
            user["id"], pack=pack or None)}

    @app.post("/api/streams")
    async def create_stream(body: StreamCreate, request: Request,
                            user: dict = Depends(require_user)):
        slug = body.pack.strip() or request.app.state.pack.get("slug", "")
        if body.pack.strip():
            row = await _db(request).get_pack(slug)
            if not row or row["status"] != "published":
                raise HTTPException(404, "unknown desk")
        return await _db(request).create_stream(user["id"], body.title.strip(), slug)

    @app.get("/api/streams/{stream_id}")
    async def get_stream(stream_id: str, request: Request,
                         user: dict = Depends(require_user)):
        db = _db(request)
        stream = await db.get_stream(stream_id, user["id"])
        if not stream:
            raise HTTPException(404, "stream not found")
        stream["rounds"] = await db.list_rounds(stream_id)
        stream["settings"] = _stream_settings(stream)
        return stream

    @app.patch("/api/streams/{stream_id}/settings")
    async def patch_settings(stream_id: str, body: SettingsBody, request: Request,
                             user: dict = Depends(require_user)):
        db = _db(request)
        stream = await db.get_stream(stream_id, user["id"])
        if not stream:
            raise HTTPException(404, "stream not found")
        if body.quality_profile is not None and body.quality_profile != "" \
                and body.quality_profile not in _QUALITY_PROFILES:
            raise HTTPException(400, f"quality_profile must be one of {_QUALITY_PROFILES}")
        settings = _stream_settings(stream)
        sent = body.model_dump(exclude_unset=True)
        for key, value in sent.items():
            if value is None or value == "" or value == []:
                settings.pop(key, None)  # explicit null/empty → back to default
            else:
                settings[key] = value
        await db.set_stream_settings(stream_id, settings)
        return {"settings": settings}

    # ── commissions (rounds) ─────────────────────────────────────────

    @app.post("/api/streams/{stream_id}/commissions")
    async def create_commission(stream_id: str, body: CommissionCreate,
                                request: Request, user: dict = Depends(require_user)):
        db = _db(request)
        stream = await db.get_stream(stream_id, user["id"])
        if not stream:
            raise HTTPException(404, "stream not found")
        rounds = await db.list_rounds(stream_id)
        settings = _stream_settings(stream)
        stream_pack = await _pack_by_slug(request, stream.get("pack") or "")
        quality = _stream_quality(stream, stream_pack)
        kind = body.kind
        if kind == "auto":
            kind = "initial" if not rounds else "continue"
        if kind != "initial" and kind not in _CONTINUE_KINDS:
            raise HTTPException(400, f"kind must be initial or one of {_CONTINUE_KINDS}")

        if kind == "initial":
            if rounds:
                raise HTTPException(409, "stream already has rounds — continue it instead")
            max_agents = body.max_agents or int(settings.get("max_agents", 6))
            payload = {"intent": body.brief, "title": stream["title"],
                       "max_agents": max_agents, "quality": quality}
            cast = settings.get("archetype_ids") or []
            if cast:
                payload["archetype_ids"] = cast  # pinned house team
            r = await _fd(request).post("/fd/vatra/start", json=payload,
                                        headers=_auth_headers(request))
            if r.status_code != 200:
                raise _fd_error(r)
            sid = r.json()["session_id"]
            # FD's default shared folder for a Vatra run; continuation rounds
            # inherit it, so it is the stream's folder from round 1 on.
            await db.set_stream_vfs_project(stream_id, f"vatra-{sid[:8]}")
        else:
            if not rounds:
                raise HTTPException(409, "no prior round to continue from")
            parent_sid = rounds[-1]["session_id"]
            r = await _fd(request).post(
                f"/fd/vatra/sessions/{parent_sid}/continue",
                json={"instruction": body.brief, "kind": kind,
                      "same_cast": bool(settings.get("same_cast", True))},
                headers=_auth_headers(request))
            if r.status_code != 200:
                raise _fd_error(r)
            data = r.json()
            sid = data.get("session_id") or data.get("id") or ""
            if not sid:
                raise HTTPException(502, "FD continue returned no session id")

        rec = await db.add_round(stream_id, sid, kind)
        return {"session_id": sid, "round": rec["round_no"], "kind": kind,
                "status": "planning" if kind == "initial" else "running"}

    @app.post("/api/commissions/{session_id}/approve")
    async def approve_commission(session_id: str, body: ApproveBody, request: Request,
                                 user: dict = Depends(require_user)):
        if not await _db(request).stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        quality = pack_quality(request.app.state.pack)
        r = await _fd(request).post(
            "/fd/vatra/plan/approve",
            json={"session_id": session_id, "plan": body.plan, "quality": quality},
            headers=_auth_headers(request))
        if r.status_code != 200:
            raise _fd_error(r)
        return r.json()

    @app.post("/api/commissions/{session_id}/cancel")
    async def cancel_commission(session_id: str, request: Request,
                                user: dict = Depends(require_user)):
        if not await _db(request).stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        # At the plan gate → discard the plan; running → cancel the run.
        r = await _fd(request).post("/fd/vatra/plan/cancel",
                                    json={"session_id": session_id},
                                    headers=_auth_headers(request))
        if r.status_code != 200:
            r = await _fd(request).post(f"/fd/basna/sessions/{session_id}/cancel",
                                        headers=_auth_headers(request))
            if r.status_code != 200:
                raise _fd_error(r)
        return {"ok": True}

    async def _fd_get(request: Request, path: str, params: dict | None = None):
        r = await _fd(request).get(path, params=params, headers=_auth_headers(request))
        if r.status_code != 200:
            raise _fd_error(r)
        return r.json()

    @app.get("/api/commissions/{session_id}")
    async def commission_detail(session_id: str, request: Request,
                                user: dict = Depends(require_user)):
        if not await _db(request).stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        return await _fd_get(request, f"/fd/basna/sessions/{session_id}")

    @app.get("/api/commissions/{session_id}/progress")
    async def commission_progress(session_id: str, request: Request,
                                  user: dict = Depends(require_user)):
        if not await _db(request).stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        return await _fd_get(request, f"/fd/basna/sessions/{session_id}/progress")

    @app.get("/api/commissions/{session_id}/facts")
    async def commission_facts(session_id: str, request: Request,
                               user: dict = Depends(require_user)):
        if not await _db(request).stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        return await _fd_get(request, f"/fd/basna/sessions/{session_id}/facts")

    async def _fd_try(request: Request, path: str, params: dict | None = None):
        """Best-effort FD read — a missing artifact must not sink the panel."""
        try:
            r = await _fd(request).get(path, params=params,
                                       headers=_auth_headers(request))
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    @app.get("/api/commissions/{session_id}/receipts")
    async def commission_receipts(session_id: str, request: Request,
                                  user: dict = Depends(require_user)):
        """Everything the verification panel renders, in one call: the run's
        quality tally + verdict, consistency summary, coverage gaps, facts
        ledger + conflicts, the constraints contract, and the cost line
        (ledger row matched by run id; effective $/hr recomputed from
        usd/elapsed since the ledger doesn't persist it)."""
        stream = await _db(request).stream_for_session(session_id, user["id"])
        if not stream:
            raise HTTPException(404, "commission not found")

        detail = await _fd_get(request, f"/fd/basna/sessions/{session_id}")
        analysis = detail.get("analysis") or {}
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except ValueError:
                analysis = {}

        facts = await _fd_try(request, f"/fd/basna/sessions/{session_id}/facts") or {}

        contract = None
        if stream.get("vfs_project"):
            raw = await _fd_try(request, "/fd/vfs/read",
                                {"project": stream["vfs_project"],
                                 "path": ".contract.json"})
            if raw and raw.get("text") and not raw.get("binary"):
                try:
                    contract = json.loads(raw["text"])
                except ValueError:
                    contract = None

        cost = None
        rows = await _fd_try(request, "/fd/costs", {"limit": 1000})
        for r in (rows or {}).get("costs", []):
            if r.get("run_id") == session_id:
                usd = r.get("usd")
                elapsed = r.get("elapsed_seconds")
                hourly = (round(float(usd) / (float(elapsed) / 3600.0), 2)
                          if usd is not None and elapsed else None)
                cost = {"usd": usd, "elapsed_seconds": elapsed,
                        "hourly_usd": hourly, "tokens": r.get("usage") or {},
                        "at": r.get("at")}
                break

        metrics = analysis.get("quality_metrics") or {}
        return {
            "status": detail.get("status", ""),
            "verdict": analysis.get("quality_verdict")
                       or metrics.get("quality_verdict") or "",
            "blocking": analysis.get("blocking"),
            "metrics": metrics,
            "consistency": analysis.get("consistency"),
            "gaps": analysis.get("gaps") or [],
            "facts": facts.get("facts") or [],
            "conflicts": facts.get("conflicts") or [],
            "contract": contract,
            "cost": cost,
            "roi": (await _pack_by_slug(request, stream.get("pack") or "")
                    ).get("roi") or {},
        }

    # ── stream workspace files (the report reader's source) ──────────

    @app.get("/api/streams/{stream_id}/files")
    async def stream_files(stream_id: str, request: Request, path: str = "",
                           user: dict = Depends(require_user)):
        stream = await _db(request).get_stream(stream_id, user["id"])
        if not stream:
            raise HTTPException(404, "stream not found")
        if not stream["vfs_project"]:
            return {"entries": []}
        return await _fd_get(request, "/fd/vfs/list",
                             {"project": stream["vfs_project"], "path": path})

    @app.get("/api/streams/{stream_id}/file")
    async def stream_file(stream_id: str, path: str, request: Request,
                          user: dict = Depends(require_user)):
        stream = await _db(request).get_stream(stream_id, user["id"])
        if not stream or not stream["vfs_project"]:
            raise HTTPException(404, "stream not found")
        return await _fd_get(request, "/fd/vfs/read",
                             {"project": stream["vfs_project"], "path": path})

    # ── Kalup pack registry + Studio ─────────────────────────────────

    async def _require_creator(request: Request, user: dict) -> None:
        if user.get("role") == "admin":
            return
        if await _db(request).is_creator(user["id"]):
            return
        raise HTTPException(403, "creator access required")

    def _can_touch(row: dict, user: dict) -> bool:
        return user.get("role") == "admin" or row.get("owner_id") == user["id"]

    def _spawn_bg(state, coro) -> None:
        task = asyncio.create_task(coro)
        state.bg_tasks = getattr(state, "bg_tasks", set())
        state.bg_tasks.add(task)
        task.add_done_callback(state.bg_tasks.discard)

    def _pack_summary(row: dict) -> dict:
        m = row_manifest(row)
        gen = json.loads(row.get("generation") or "{}")
        ev = json.loads(row.get("eval_state") or "{}")
        return {"slug": row["slug"], "status": row["status"],
                "version": row["version"], "owner_id": row["owner_id"],
                "name": m.get("name", row["slug"]),
                "tagline": m.get("tagline", ""),
                "accent": (m.get("theme") or {}).get("accent", ""),
                "generation": gen.get("status", ""),
                "eval": ev.get("verdict", "")}

    @app.get("/api/creators/me")
    async def creators_me(request: Request, user: dict = Depends(require_user)):
        creator = (user.get("role") == "admin"
                   or await _db(request).is_creator(user["id"]))
        return {"creator": creator, "role": user.get("role", "user")}

    @app.post("/api/creators")
    async def add_creator(body: CreatorAdd, request: Request,
                          user: dict = Depends(require_user)):
        if user.get("role") != "admin":
            raise HTTPException(403, "admin only")
        await _db(request).add_creator(body.user_id)
        return {"ok": True}

    @app.get("/api/packs")
    async def list_packs(request: Request, user: dict = Depends(require_user)):
        rows = await _db(request).list_packs()
        out = [
            _pack_summary(r) for r in rows
            if r["status"] == "published" or _can_touch(r, user)
        ]
        return {"packs": out}

    @app.get("/api/packs/{slug}")
    async def get_pack_detail(slug: str, request: Request):
        """Published packs are public branding (the desk must render before
        login); drafts stay owner/admin-only."""
        row = await _db(request).get_pack(slug)
        if not row:
            raise HTTPException(404, "pack not found")
        if row["status"] != "published":
            try:
                user = await require_user(request)
            except HTTPException:
                raise HTTPException(404, "pack not found")
            if not _can_touch(row, user):
                raise HTTPException(404, "pack not found")
        return {"pack": row_manifest(row),
                "summary": _pack_summary(row),
                "generation": json.loads(row.get("generation") or "{}"),
                "eval": json.loads(row.get("eval_state") or "{}")}

    @app.post("/api/packs")
    async def create_pack(body: PackCreate, request: Request,
                          user: dict = Depends(require_user)):
        await _require_creator(request, user)
        db = _db(request)
        if await db.get_pack(body.slug):
            raise HTTPException(409, f"pack '{body.slug}' already exists")
        # Scaffold from the default pack so a fresh draft renders immediately.
        base = {k: v for k, v in request.app.state.pack.items()
                if k not in ("slug", "pack_status", "pack_version", "onboarding_md")}
        base["name"] = body.name.strip()
        base["tagline"] = body.tagline.strip()
        row = await db.create_pack(body.slug, user["id"], base)
        return {"pack": row_manifest(row), "summary": _pack_summary(row)}

    @app.put("/api/packs/{slug}")
    async def update_pack(slug: str, body: PackManifestBody, request: Request,
                          user: dict = Depends(require_user)):
        db = _db(request)
        row = await db.get_pack(slug)
        if not row or not _can_touch(row, user):
            raise HTTPException(404, "pack not found")
        manifest = dict(body.manifest)
        for k in ("slug", "pack_status", "pack_version"):
            manifest.pop(k, None)
        await db.update_pack(slug, manifest=manifest)
        return {"ok": True}

    async def _generate_pack(state, slug: str, name: str, instructions: str,
                             token: str) -> None:
        detail, err = await _run_headless_vatra(
            state, token, _generate_intent(name, instructions),
            {"profile": "balanced"})
        if err:
            await state.db.update_pack(slug, generation={"status": "error",
                                                         "message": err})
            return
        generated = _extract_manifest(str(detail.get("truth") or ""))
        if not generated:
            await state.db.update_pack(
                slug, generation={"status": "error",
                                  "message": "no manifest in the run's conclusion"})
            return
        row = await state.db.get_pack(slug)
        manifest = row_manifest(row) if row else {}
        for k in ("slug", "pack_status", "pack_version"):
            manifest.pop(k, None)
            generated.pop(k, None)
        manifest.update(generated)
        await state.db.update_pack(slug, manifest=manifest,
                                   generation={"status": "done",
                                               "session_id": detail.get("id", "")})

    @app.post("/api/packs/{slug}/generate")
    async def generate_pack(slug: str, body: GenerateBody, request: Request,
                            user: dict = Depends(require_user)):
        db = _db(request)
        row = await db.get_pack(slug)
        if not row or not _can_touch(row, user):
            raise HTTPException(404, "pack not found")
        gen = json.loads(row.get("generation") or "{}")
        if gen.get("status") == "running":
            raise HTTPException(409, "a generation is already running")
        await db.update_pack(slug, generation={"status": "running"})
        secret = os.environ.get("FD_JWT_SECRET", "")
        token = _mint_owner_token(user["id"], secret) if secret else _bearer(request)
        name = row_manifest(row).get("name", slug)
        _spawn_bg(request.app.state,
                  _generate_pack(request.app.state, slug, name,
                                 body.instructions, token))
        return {"status": "running"}

    async def _evaluate_pack(state, slug: str, token: str) -> None:
        row = await state.db.get_pack(slug)
        manifest = row_manifest(row) if row else {}
        evals = manifest.get("evals") or []
        brief = str((evals[0] or {}).get("brief") if evals else "") or (
            f"Golden task: produce a short, source-verified overview of the "
            f"core question a customer of '{manifest.get('name', slug)}' would ask.")
        detail, err = await _run_headless_vatra(
            state, token, brief, pack_quality(manifest) or {"profile": "thorough"})
        if err:
            await state.db.update_pack(slug, eval_state={"status": "error",
                                                         "message": err})
            return
        analysis = detail.get("analysis") or {}
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except ValueError:
                analysis = {}
        verdict, metrics = _eval_verdict(analysis)
        await state.db.update_pack(
            slug, eval_state={"status": "done", "verdict": verdict,
                              "metrics": metrics,
                              "session_id": detail.get("id", "")})

    @app.post("/api/packs/{slug}/evaluate")
    async def evaluate_pack(slug: str, request: Request,
                            user: dict = Depends(require_user)):
        db = _db(request)
        row = await db.get_pack(slug)
        if not row or not _can_touch(row, user):
            raise HTTPException(404, "pack not found")
        ev = json.loads(row.get("eval_state") or "{}")
        if ev.get("status") == "running":
            raise HTTPException(409, "an evaluation is already running")
        await db.update_pack(slug, eval_state={"status": "running"})
        secret = os.environ.get("FD_JWT_SECRET", "")
        token = _mint_owner_token(user["id"], secret) if secret else _bearer(request)
        _spawn_bg(request.app.state,
                  _evaluate_pack(request.app.state, slug, token))
        return {"status": "running"}

    @app.post("/api/packs/{slug}/publish")
    async def publish_pack(slug: str, request: Request,
                           user: dict = Depends(require_user)):
        db = _db(request)
        row = await db.get_pack(slug)
        if not row or not _can_touch(row, user):
            raise HTTPException(404, "pack not found")
        ev = json.loads(row.get("eval_state") or "{}")
        if ev.get("verdict") != "green":
            raise HTTPException(409,
                                "the ship-gate is closed: run an evaluation and "
                                "get a green verdict before publishing")
        await db.update_pack(slug, status="published", bump_version=True)
        return {"ok": True, "status": "published"}

    # ── house style: forge + archetypes ──────────────────────────────

    @app.post("/api/forge")
    async def forge(request: Request, user: dict = Depends(require_user)):
        """Multipart proxy to FD's archetype forge: instructions + documents →
        a batch of UNPERSISTED archetype drafts ("your methodology, encoded").
        The client reviews and saves the keepers via POST /api/archetypes."""
        form = await request.form()
        data = {k: str(form.get(k) or "") for k in
                ("instructions", "provider", "model", "count") if form.get(k)}
        files = []
        for uf in form.getlist("files"):
            if hasattr(uf, "read"):
                files.append(("files", (uf.filename or "doc",
                                        await uf.read(),
                                        uf.content_type or "application/octet-stream")))
        r = await _fd(request).post("/fd/archetypes/forge", data=data,
                                    files=files or None,
                                    headers=_auth_headers(request), timeout=300.0)
        if r.status_code != 200:
            raise _fd_error(r)
        out = r.json()
        drafts = out if isinstance(out, list) else out.get("archetypes") or out.get("drafts") or []
        return {"drafts": drafts}

    @app.get("/api/archetypes")
    async def list_archetypes(request: Request, mine: bool = False,
                              user: dict = Depends(require_user)):
        """The cast pool the picker offers. Default: the MERGED registry —
        base/system archetypes + the user's own house cast + shared — each
        tagged with its `source`. `?mine=true` narrows to the user's own
        (the House-style save target)."""
        path = "/fd/archetypes/mine" if mine else "/fd/archetypes"
        r = await _fd(request).get(path, headers=_auth_headers(request))
        if r.status_code != 200:
            raise _fd_error(r)
        out = r.json()
        items = out if isinstance(out, list) else out.get("archetypes", [])
        slim = [{"id": a.get("id", ""), "role": a.get("role") or a.get("id", ""),
                 "source": a.get("source", "user")}
                for a in items if a.get("id")]
        return {"archetypes": slim}

    @app.post("/api/archetypes")
    async def save_archetype(request: Request, user: dict = Depends(require_user)):
        """Persist one forged draft (PUT upsert on FD, so re-saving is safe)."""
        body = await request.json()
        arch_id = str(body.get("id") or "").strip()
        if not arch_id:
            raise HTTPException(400, "archetype id is required")
        r = await _fd(request).put(f"/fd/archetypes/{arch_id}", json=body,
                                   headers=_auth_headers(request))
        if r.status_code != 200:
            raise _fd_error(r)
        return r.json()

    # ── second opinion (Basna ensemble over the same brief) ──────────

    async def _run_second_opinion(state, vatra_sid: str, basna_sid: str,
                                  token: str) -> None:
        """Background: Basna's execute is synchronous — run it off-request and
        record the outcome. The UI polls the session for truth/confidence."""
        try:
            r = await state.fd.post("/fd/basna/execute",
                                    json={"session_id": basna_sid},
                                    headers={"Authorization": f"Bearer {token}"},
                                    timeout=3600.0)
            await state.db.set_second_opinion_status(
                vatra_sid, "done" if r.status_code == 200 else "error")
        except Exception:  # noqa: BLE001
            log.exception("second-opinion execute failed for %s", vatra_sid)
            await state.db.set_second_opinion_status(vatra_sid, "error")

    @app.post("/api/commissions/{session_id}/second-opinion")
    async def start_second_opinion(session_id: str, request: Request,
                                   user: dict = Depends(require_user)):
        db = _db(request)
        if not await db.stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        existing = await db.get_second_opinion(session_id, user["id"])
        if existing:
            return existing  # idempotent — one ensemble pass per round
        detail = await _fd_get(request, f"/fd/basna/sessions/{session_id}")
        intent = str(detail.get("intent") or "").strip()
        if not intent:
            raise HTTPException(409, "the commission has no brief to re-run")
        r = await _fd(request).post("/fd/basna/route", json={"intent": intent},
                                    headers=_auth_headers(request), timeout=120.0)
        if r.status_code != 200:
            raise _fd_error(r)
        data = r.json()
        basna_sid = str(data.get("session_id") or data.get("id") or "")
        if not basna_sid:
            raise HTTPException(502, "FD route returned no session id")
        rec = await db.create_second_opinion(session_id, basna_sid, user["id"])
        # Prefer a minted token (survives the 15-min access TTL on long runs).
        secret = os.environ.get("FD_JWT_SECRET", "")
        token = _mint_owner_token(user["id"], secret) if secret else _bearer(request)
        state = request.app.state
        task = asyncio.create_task(
            _run_second_opinion(state, session_id, basna_sid, token))
        state.bg_tasks = getattr(state, "bg_tasks", set())
        state.bg_tasks.add(task)
        task.add_done_callback(state.bg_tasks.discard)
        return rec

    @app.get("/api/commissions/{session_id}/second-opinion")
    async def get_second_opinion(session_id: str, request: Request,
                                 user: dict = Depends(require_user)):
        db = _db(request)
        if not await db.stream_for_session(session_id, user["id"]):
            raise HTTPException(404, "commission not found")
        rec = await db.get_second_opinion(session_id, user["id"])
        if not rec:
            return {"second_opinion": None}
        out = {"second_opinion": rec}
        detail = await _fd_try(request,
                               f"/fd/basna/sessions/{rec['basna_session_id']}")
        if detail:
            out["truth"] = detail.get("truth") or ""
            out["confidence"] = detail.get("confidence")
        return out

    # ── standing briefs ──────────────────────────────────────────────

    @app.get("/api/streams/{stream_id}/brief")
    async def get_brief(stream_id: str, request: Request,
                        user: dict = Depends(require_user)):
        if not await _db(request).get_stream(stream_id, user["id"]):
            raise HTTPException(404, "stream not found")
        return {"brief": await _db(request).get_brief(stream_id, user["id"])}

    @app.put("/api/streams/{stream_id}/brief")
    async def put_brief(stream_id: str, body: BriefBody, request: Request,
                        user: dict = Depends(require_user)):
        db = _db(request)
        if not await db.get_stream(stream_id, user["id"]):
            raise HTTPException(404, "stream not found")
        if not await db.list_rounds(stream_id):
            raise HTTPException(409, "run a first round before scheduling a brief")
        brief = await db.upsert_brief(stream_id, user["id"], body.instruction.strip(),
                                      body.cadence_hours, body.enabled)
        return {"brief": brief}

    @app.delete("/api/streams/{stream_id}/brief")
    async def delete_brief(stream_id: str, request: Request,
                           user: dict = Depends(require_user)):
        if not await _db(request).get_stream(stream_id, user["id"]):
            raise HTTPException(404, "stream not found")
        await _db(request).delete_brief(stream_id, user["id"])
        return {"ok": True}

    @app.get("/api/inbox")
    async def inbox(request: Request, user: dict = Depends(require_user)):
        return {"rounds": await _db(request).list_brief_rounds(user["id"])}

    # ── costs ────────────────────────────────────────────────────────

    @app.get("/api/costs")
    async def costs(request: Request, run_kind: str = "", since: str = "",
                    limit: int = 200, user: dict = Depends(require_user)):
        return await _fd_get(request, "/fd/costs",
                             {"run_kind": run_kind, "since": since, "limit": limit})

    # ── SPA ──────────────────────────────────────────────────────────

    static = Path(__file__).resolve().parent.parent / "static"
    if (static / "index.html").is_file():
        from fastapi.responses import FileResponse

        @app.get("/desks/{slug}")
        async def desk_entry(slug: str):
            # Every published desk is the same SPA; the client reads the slug
            # from the path and activates that pack (theme, vocabulary, …).
            return FileResponse(static / "index.html")

        app.mount("/", StaticFiles(directory=static, html=True), name="spa")

    return app


app = create_app()


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Lupa BFF")
    ap.add_argument("--host", default=os.environ.get("LUPA_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LUPA_PORT", "25180")))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
