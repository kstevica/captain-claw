"""Member-facing routes for chat-only shared agents (A1) — see ``agent_sharing``.

* ``GET /fd/shared-agents`` — the agents shared with me that still exist and
  are still their sharer's, with what a member gets on each (A2: process
  agents act with the member's own Google — after their opt-in —, deep memory
  and files during their turns; docker agents stay chat-only). Never an
  agent's token, port, host, pid or container id.
* ``PUT /fd/shared-agents/google`` — a member turns "Let this agent use my
  Google during my chats" on or off for one shared process agent.
* ``WS /fd/agent-ws-shared?ref=&lane=&fd_token=`` — a member's chat socket.
  The browser names the agent by ref only; FD resolves it from its own records,
  connects to ``ws://localhost:{recorded port}/ws`` with the recorded token and a
  signed ``X-FD-Speaker`` assertion, waits for the agent to acknowledge the
  assertion (an agent too old to understand it would treat the member as its
  owner), then relays only the allowlisted member frames (``chat``, ``btw`` and
  the session settings size-capped, ``btw`` at most one a second). Membership,
  the sharer and the member's session are re-checked while the socket is open.
  Each member chat turn on a process agent carries a per-turn grant
  (``speaker_grants``) the agent uses to act as the member; it closes with the
  turn, on revocation, and shortly after the socket that sent the message.

Every rejection happens after ``accept()`` and is preceded by an ``fd_close``
frame, so the browser always sees why (a close before accept surfaces as 1006).
Never logs the agent's token, the assertion or the member's JWT.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import time
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import speaker_grants, tenant_profile
from captain_claw.flight_deck.auth import decode_access_token, get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["agent-sharing"])

_NOT_ALLOWED = {"type": "error", "code": "not_allowed", "message": "Not available on a shared agent"}
_CHAT_OPTIONAL_TYPES = {"rewind_to": str, "no_next_steps": bool, "no_rephrase": bool}
_SPEAKER_NAME_MAX = 120  # = tenant_profile's name cap (the speaker block uses the same)
_SETTING_LABELS = {"session_name": "Session name", "session_description": "Session description",
                   "session_instructions": "Session instructions"}


def _email_local(email: str) -> str:
    return str(email or "").split("@")[0]


# ── Listing ───────────────────────────────────────────────────────────────


async def _google_connected(user_id: str) -> bool:
    """Has this user connected their own Google on this deck? (no network)"""
    try:
        from captain_claw.flight_deck import google_oauth_routes

        return bool(await google_oauth_routes.is_google_connected(user_id))
    except Exception:
        return False


async def _mine(db, uid: str) -> dict:
    """O1 — the caller's own shared agents: members and Google opt-ins per ref."""
    mine: dict[str, dict[str, int]] = {}
    try:
        rows = await db.list_shares_for_owner(uid, sharing.AGENT_RESOURCE)
    except Exception as exc:
        log.warning("Could not list the caller's shared agents", error=type(exc).__name__)
        return mine
    for row in rows:
        ref = str(row.get("resource_id") or "")
        grantee = str(row.get("grantee_id") or "")
        if not ref or not grantee:
            continue
        entry = mine.setdefault(ref, {"members": 0, "google": 0})
        entry["members"] += 1
        if await speaker_grants.google_opted_in(db, grantee, ref, uid):
            entry["google"] += 1
    return mine


@router.get("/fd/shared-agents")
async def list_shared_agents(user: dict = Depends(get_current_user)):
    """Agents other users shared with me, and what I get on each."""
    if not sharing.sharing_active():
        return {"enabled": False, "host_warning": "", "agents": []}
    db = get_db()
    uid = str(user["id"])
    rows = await db.list_shares_for_grantee(uid, sharing.AGENT_RESOURCE)
    google_connected = await _google_connected(uid)
    agents = []
    for row in rows:
        ref = str(row.get("resource_id") or "")
        owner_id = str(row.get("owner_id") or "")
        rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
        if rec is None or rec.owner != owner_id or sharing.check_shareable(rec, owner_id):
            continue
        email = str(row.get("owner_email") or "")
        process = rec.runtime == "process"
        agents.append({
            "agent_ref": ref,
            "runtime": rec.runtime,
            "slug": rec.slug,
            "name": rec.name,
            "description": rec.description,
            "status": "running" if rec.running else "stopped",
            "owner_id": owner_id,
            "owner_name": str(row.get("owner_name") or "") or _email_local(email),
            "owner_email": email,
            "shared_at": str(row.get("created_at") or ""),
            "capabilities": {"google": process, "deep_memory": process, "files": process},
            "google_enabled": bool(process and await speaker_grants.google_opted_in(
                db, uid, ref, owner_id)),
            "google_connected": google_connected,
        })
    return {"enabled": True, "host_warning": sharing.HOST_TRUST_WARNING, "agents": agents,
            "mine": await _mine(db, uid)}


class GoogleOptInBody(BaseModel):
    agent_ref: str
    enabled: bool


@router.put("/fd/shared-agents/google")
async def set_shared_agent_google(body: GoogleOptInBody,
                                  user: dict = Depends(get_current_user)) -> dict:
    """A member turns "Let this agent use my Google during my chats" on or off.
    Stored in the member's own settings with the agent's CURRENT owner, so the
    consent lapses if the agent changes hands. Turning it off needs no grant
    revocation: every grant-aware Google request re-reads it."""
    if not sharing.sharing_active():
        raise HTTPException(400, "Agent sharing is off on this Flight Deck")
    ref = body.agent_ref
    try:
        sharing.parse_ref(ref)
    except ValueError:
        raise HTTPException(400, "Invalid agent reference") from None
    rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
    if rec is None or sharing.check_shareable(rec, rec.owner):
        raise HTTPException(404, "Agent not found")
    uid = str(user["id"])
    db = get_db()
    if uid == rec.owner or not await sharing.member_check(db, ref, rec.owner, uid, max_age=0):
        raise HTTPException(404, "This agent isn't shared with you")
    if rec.runtime != "process":
        raise HTTPException(400, "Google isn't available for this agent in shared chats")
    await speaker_grants.set_google_optin(db, uid, ref, rec.owner, body.enabled)
    # A revoke that landed between the check and the write cleared opt-ins
    # before this one was written: undo it so a later re-share starts off.
    if body.enabled and not await sharing.member_check(db, ref, rec.owner, uid, max_age=0):
        await speaker_grants.clear_google_optins(db, ref, uid)
        raise HTTPException(404, "This agent isn't shared with you")
    log.info("Shared-agent Google opt-in changed", agent=rec.slug, member=uid,
             enabled=bool(body.enabled))
    return {"agent_ref": ref, "google_enabled": body.enabled}


# ── Member socket ─────────────────────────────────────────────────────────


def _decode_jwt(token) -> dict | None:
    if not isinstance(token, str) or not token:
        return None
    try:
        payload = decode_access_token(token)
    except HTTPException:
        return None
    except Exception:
        return None
    return payload if str(payload.get("sub") or "") else None


def _jwt_exp(payload: dict) -> int:
    try:
        return int(payload.get("exp") or 0)
    except (TypeError, ValueError):
        return 0


class _MemberConn:
    """One member socket: the browser side, the upstream agent socket, and the
    single close path every rejection and revocation goes through."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.upstream = None
        self.closed = False
        self.done = asyncio.Event()
        # Set once the browser side is finished with: the close frame went out
        # (or the browser is gone). The handler doesn't return before it, so a
        # close started by another task (DELETE / leave / agent removal) can't
        # be cut short into an abrupt 1006 by the handler returning first.
        self.client_closed = asyncio.Event()
        self.jwt_exp = 0
        self.outstanding: set[str] = set()
        self.last_btw: float | None = None  # monotonic time the last `btw` went upstream
        # Set once the socket is registered (A2: the grant tuple of its turns).
        self.ref = self.sub = self.lane = self.conn_id = self.owner = self.runtime = ""

    async def send(self, frame: dict) -> None:
        await self.ws.send_text(json.dumps(frame))

    async def close(self, code: int, reason: str) -> None:
        """``fd_close`` then close, once; also drops the upstream socket."""
        if self.closed:
            return
        self.closed = True
        try:
            try:
                await self.send({"type": "fd_close", "code": code, "reason": reason})
            except Exception:
                pass
            try:
                await self.ws.close(code=code, reason=reason)
            except Exception:
                pass
        finally:
            self.client_closed.set()
            self.done.set()
        await self._close_upstream()

    async def wait_client_closed(self, timeout: float = 5.0) -> None:
        """Let a close another task started reach the browser first."""
        if not self.closed or self.client_closed.is_set():
            return
        try:
            await asyncio.wait_for(self.client_closed.wait(), timeout=timeout)
        except Exception:
            pass

    async def _close_upstream(self) -> None:
        up = self.upstream
        if up is not None:
            try:
                await up.close()
            except Exception:
                pass

    async def shutdown(self) -> None:
        """The browser went away: no frames to send, just drop the upstream."""
        self.closed = True
        self.client_closed.set()
        self.done.set()
        await self._close_upstream()


@router.websocket("/fd/agent-ws-shared")
async def agent_ws_shared(ws: WebSocket, ref: str = "", lane: str = "", fd_token: str = ""):
    """A member's chat socket to a shared agent. The browser never sends (or
    learns) the agent's token, host or port; any it sends are ignored."""
    await ws.accept()
    conn = _MemberConn(ws)

    if not sharing.sharing_active():
        await conn.close(4503, "Agent sharing is off on this Flight Deck")
        return
    payload = _decode_jwt(fd_token)
    if payload is None:
        await conn.close(4001, "Missing or invalid token")
        return
    sub = str(payload["sub"])
    db = get_db()
    try:
        user = await db.get_user_by_id(sub)
    except Exception:
        user = None
    if not user:
        await conn.close(4001, "User not found")
        return
    conn.jwt_exp = _jwt_exp(payload)

    try:
        sharing.parse_ref(ref)
    except ValueError:
        await conn.close(4400, "Invalid agent reference")
        return
    lane = (lane or "A").strip().upper()
    if lane not in sharing.MEMBER_LANES:
        await conn.close(4400, "Invalid lane")
        return

    rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
    if rec is None:
        await conn.close(4404, "Agent no longer exists")
        return
    reason = sharing.check_shareable(rec, rec.owner)
    if reason:
        await conn.close(4403, reason)
        return
    if sub == rec.owner:  # admins get no bypass either
        await conn.close(4400, "You own this agent — open it from your agents list")
        return
    if not await sharing.member_check(db, ref, rec.owner, sub):
        await conn.close(4403, "Access removed")
        return
    if not rec.running or not rec.port:
        await conn.close(4409, "Agent is stopped")
        return
    # No await between the count and the registration: one event loop, so the
    # cap can't be raced.
    if sharing.live_conn_count(ref, sub) >= sharing.MAX_MEMBER_SOCKETS_PER_AGENT:
        await conn.close(4429, "Too many open connections to this agent")
        return
    conn_id = sharing.register_member_socket(ref, sub, lane, conn.close)
    conn.ref, conn.sub, conn.lane, conn.conn_id = ref, sub, lane, conn_id
    conn.owner, conn.runtime = rec.owner, rec.runtime
    try:
        await _serve_member(conn, db, user, rec, ref, lane, conn_id)
    except Exception as exc:
        log.warning("Shared-agent socket failed", error=type(exc).__name__)
        await conn.close(4502, "Agent connection lost")
    finally:
        sharing.unregister_member_socket(conn_id)
        # The grants of this socket's turns outlive it by ORPHAN_GRACE_S at most.
        speaker_grants.conn_dropped(conn_id)
        await conn.wait_client_closed()
        await conn.shutdown()  # the upstream never outlives the member socket


async def _await_welcome(upstream) -> tuple[str, dict]:
    """The agent's first ``welcome`` (earlier frames are dropped, not relayed)."""
    while True:
        msg = await upstream.recv()
        text = msg if isinstance(msg, str) else bytes(msg).decode("utf-8", "replace")
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("type") == "welcome":
            return text, data


async def _serve_member(conn: _MemberConn, db, user: dict, rec: sharing.AgentRecord,
                        ref: str, lane: str, conn_id: str) -> None:
    import websockets
    from websockets.exceptions import ConnectionClosed

    sub = str(user["id"])
    display = str(user.get("display_name") or "").strip() or _email_local(user.get("email", ""))
    # One line, bounded: display names have no length limit, and an oversized
    # name would push the X-FD-Speaker header past the agent's header limit.
    display = " ".join(display.split())[:_SPEAKER_NAME_MAX]
    owner_name = await tenant_profile.owner_name(db, rec.owner)
    header = sharing.sign_speaker_assertion(rec.web_auth, sharing.speaker_payload(
        speaker_id=sub, name=display, owner=rec.owner, owner_name=owner_name,
        ref=ref, lane=lane, conn=conn_id))
    expected_ack = sharing.speaker_ack_for(header)
    url = f"ws://localhost:{int(rec.port)}/ws?token={quote(rec.web_auth, safe='')}"

    if conn.closed:
        return
    try:
        upstream = await websockets.connect(
            url,
            additional_headers={sharing.SPEAKER_HEADER: header},
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=10,
            proxy=None,  # always the local agent, never through an env proxy
        )
    except Exception as exc:
        log.info("Shared agent unreachable", agent=rec.slug, error=type(exc).__name__)
        await conn.close(4502, "Agent unreachable")
        return
    conn.upstream = upstream
    if conn.closed:  # revoked while connecting
        await conn._close_upstream()
        return

    # Ack gate: proof the agent understood X-FD-Speaker (an older one ignores it).
    try:
        welcome_text, welcome = await asyncio.wait_for(
            _await_welcome(upstream), timeout=sharing.ACK_TIMEOUT_S)
    except TimeoutError:
        await conn.close(4426, "This agent needs a restart before it can be shared")
        return
    except ConnectionClosed as exc:
        code = exc.rcvd.code if exc.rcvd is not None else None
        if code == 4429:
            await conn.close(4429, "This agent is at member capacity")
        else:
            await conn.close(4502, "The agent refused the connection")
        return
    ack = welcome.get("speaker_ack")
    if not isinstance(ack, str) or not hmac.compare_digest(
            ack.encode("utf-8"), expected_ack.encode("ascii")):
        await conn.close(4426, "This agent needs a restart before it can be shared")
        return
    if conn.closed:
        return

    await conn.ws.send_text(welcome_text)
    try:
        full, compact = await tenant_profile.compose_for_speaker(db, sub, rec.owner)
    except Exception as exc:
        log.warning("Could not compose the member profile", error=type(exc).__name__)
        full, compact = "", ""
    try:
        await upstream.send(json.dumps(
            {"type": "fd_speaker_context", "profile_full": full, "profile_compact": compact}))
    except ConnectionClosed:
        await conn.close(4502, "Agent connection lost")
        return

    tasks = [
        asyncio.create_task(_client_to_agent(conn, db, rec, ref, sub)),
        asyncio.create_task(_agent_to_client(conn)),
        asyncio.create_task(_watchdog(conn, db, rec, ref, sub)),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for t in done:
        exc = None if t.cancelled() else t.exception()
        if exc is not None and not conn.closed:  # once closed, a failed send is expected
            log.warning("Shared-agent relay failed", error=type(exc).__name__)
            await conn.close(4502, "Agent connection lost")


async def _client_to_agent(conn: _MemberConn, db, rec: sharing.AgentRecord,
                           ref: str, sub: str) -> None:
    from websockets.exceptions import ConnectionClosed

    while not conn.closed:
        message = await conn.ws.receive()
        if message.get("type") == "websocket.disconnect":
            await conn.shutdown()
            return
        raw = message.get("text")
        if raw is None:
            raw = (message.get("bytes") or b"").decode("utf-8", "replace")
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            await conn.send({"type": "error", "code": "invalid", "message": "Malformed message"})
            continue
        ftype = data.get("type")

        if ftype == "fd_auth":  # a rotated FD token; consumed here, never forwarded
            fresh = _decode_jwt(data.get("fd_token"))
            if fresh is not None and str(fresh.get("sub")) == sub:
                conn.jwt_exp = _jwt_exp(fresh)
                await conn.send({"type": "fd_auth_ok", "exp": conn.jwt_exp})
            else:
                await conn.send({"type": "error", "code": "invalid",
                                 "message": "Token refused"})
            continue

        keys = sharing.MEMBER_FRAME_ALLOWLIST.get(ftype) if isinstance(ftype, str) else None
        if keys is None:
            await conn.send(dict(_NOT_ALLOWED))
            continue
        if not await sharing.member_check(db, ref, rec.owner, sub):
            sharing.invalidate_member_cache(ref, sub)
            speaker_grants.revoke(ref, sub)
            await conn.close(4403, "Access removed")
            return
        frame = {k: data[k] for k in keys if k in data}
        token = ""

        # The agent keeps every `btw` until its next turn ends and persists the
        # session settings into every prompt: both are bounded here, not forwarded
        # when over (no turn_end — neither starts a turn).
        refusal = (_btw_refusal(conn, frame) if ftype == "btw"
                   else _session_settings_refusal(frame) if ftype == "session_settings"
                   else None)
        if refusal is not None:
            await conn.send(refusal)
            continue

        if ftype == "chat":
            content = frame.get("content")
            if not isinstance(content, str) or len(content) > sharing.MAX_CHAT_CONTENT:
                await conn.send({"type": "error", "code": "invalid",
                                 "message": "Message must be text of at most "
                                            f"{sharing.MAX_CHAT_CONTENT:,} characters"})
                await conn.send({"type": "status", "status": "ready",
                                 "turn_end": secrets.token_hex(8)})
                continue
            if len(conn.outstanding) >= sharing.MAX_OUTSTANDING_TURNS_PER_CONN:
                await conn.send({"type": "error", "code": "busy",
                                 "message": "Wait for the current reply"})
                await conn.send({"type": "status", "status": "ready",
                                 "turn_end": secrets.token_hex(8)})
                continue
            # The optional chat keys keep only the types the wire contract gives them.
            for key, kind in _CHAT_OPTIONAL_TYPES.items():
                if key in frame and not isinstance(frame[key], kind):
                    frame.pop(key)
            tid = secrets.token_hex(8)
            frame["_fd_turn"] = tid
            conn.outstanding.add(tid)
            # A2: a member turn on a process agent may act as the member (their
            # Google after opt-in, deep memory, files) through this per-turn
            # grant. Not for slash commands, empty messages or docker agents.
            # A revocation landing during member_check above either closed the
            # conn or bumped the membership generation, so a grant minted anyway
            # is refused (and revoked) on its first use.
            stripped = content.strip()
            if (rec.runtime == "process" and stripped and not stripped.startswith("/")
                    and not conn.closed):
                token = speaker_grants.mint(agent_ref=ref, owner=rec.owner, speaker=sub,
                                            lane=conn.lane, turn=tid, conn_id=conn.conn_id)
                if token:
                    frame[speaker_grants.CHAT_GRANT_FIELD] = token

        try:
            await conn.upstream.send(json.dumps(frame))
        except ConnectionClosed:
            if token:
                speaker_grants.end_turn(ref, sub, conn.lane, frame["_fd_turn"])
            return  # the agent→client side reports the lost connection
        except Exception:
            if token:
                speaker_grants.end_turn(ref, sub, conn.lane, frame["_fd_turn"])
            raise


def _btw_refusal(conn: _MemberConn, frame: dict) -> dict | None:
    """Why this ``btw`` isn't forwarded (an error frame), else None. A forwarded
    one starts this socket's one-second window."""
    content = frame.get("content")
    if not isinstance(content, str) or len(content) > sharing.MAX_CHAT_CONTENT:
        return {"type": "error", "code": "invalid",
                "message": f"A note must be text of at most {sharing.MAX_CHAT_CONTENT:,} characters"}
    now = time.monotonic()
    if conn.last_btw is not None and now - conn.last_btw < sharing.BTW_MIN_INTERVAL_S:
        return {"type": "error", "code": "busy", "message": "One note a second — try again"}
    conn.last_btw = now
    return None


def _session_settings_refusal(frame: dict) -> dict | None:
    """Why these session settings aren't forwarded (an error frame), else None.
    A non-string field is dropped (the agent ignores those anyway)."""
    for key, cap in sharing.SESSION_SETTING_MAX.items():
        if key not in frame:
            continue
        if not isinstance(frame[key], str):
            frame.pop(key)
        elif len(frame[key]) > cap:
            return {"type": "error", "code": "invalid",
                    "message": f"{_SETTING_LABELS[key]} must be at most {cap:,} characters"}
    return None


async def _agent_to_client(conn: _MemberConn) -> None:
    from websockets.exceptions import ConnectionClosed

    try:
        async for msg in conn.upstream:
            text = msg if isinstance(msg, str) else bytes(msg).decode("utf-8", "replace")
            if '"turn_end"' in text:
                try:
                    data = json.loads(text)
                except ValueError:
                    data = None
                tid = data.get("turn_end") if isinstance(data, dict) else None
                if isinstance(tid, str):
                    conn.outstanding.discard(tid)
                    speaker_grants.end_turn(conn.ref, conn.sub, conn.lane, tid)
            try:
                await conn.ws.send_text(text)
            except Exception:
                await conn.shutdown()  # the browser is gone
                return
    except ConnectionClosed:
        pass
    await conn.close(4502, "Agent connection lost")


async def _watchdog(conn: _MemberConn, db, rec: sharing.AgentRecord, ref: str, sub: str) -> None:
    """Re-check the member's session, the agent and the membership every
    ``RECHECK_INTERVAL_S`` (revocations that didn't go through the share
    routes: a deleted user, an owner change, a direct DB edit, JWT expiry)."""
    while not conn.closed:
        try:
            await asyncio.wait_for(conn.done.wait(), timeout=sharing.RECHECK_INTERVAL_S)
            return
        except TimeoutError:
            pass
        if time.time() > conn.jwt_exp + sharing.JWT_GRACE_S:
            await conn.close(4001, "Session expired")
            return
        try:
            current = await asyncio.to_thread(sharing.resolve_agent_record, ref, strict=True)
        except sharing.RecordUnavailable:
            continue  # a failed read isn't a removed agent: check again next tick
        if current is None:
            await _forget_agent_grants(db, ref)
            await conn.close(4404, "Agent no longer exists")
            return
        if current.owner != rec.owner or sharing.check_shareable(current, rec.owner):
            sharing.invalidate_member_cache(ref)
            if current.owner != rec.owner:  # the members consented for the old owner
                await _forget_agent_grants(db, ref)
            else:
                speaker_grants.revoke(ref)
            await conn.close(4403, "Access removed")
            return
        if not await sharing.member_check(db, ref, rec.owner, sub, max_age=0):
            sharing.invalidate_member_cache(ref, sub)
            speaker_grants.revoke(ref, sub)
            await conn.close(4403, "Access removed")
            return


async def _forget_agent_grants(db, ref: str) -> None:
    """The agent is gone or changed hands: close every member's grants on it and
    drop their Google opt-ins (given for the old owner). Never raises."""
    speaker_grants.revoke(ref)
    try:
        await speaker_grants.clear_google_optins(db, ref)
    except Exception as exc:
        log.warning("Could not clear a shared agent's Google opt-ins", error=type(exc).__name__)
