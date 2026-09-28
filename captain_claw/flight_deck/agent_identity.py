"""Gate for agent-facing connector endpoints: which of THIS deck's agents is calling.

Loopback and the shared agent secret only prove the caller runs next to Flight
Deck — a web page in the user's browser is on loopback too, and so is another
deck's agent on the same host. Endpoints that hand out credentials or proxy
connectors (Codex token, MCP tool proxy) therefore also require the caller's
per-agent ``web_auth`` token in ``X-Agent-Auth`` and resolve it against FD's own
records (process registry / managed-container labels). Captain Claw agents send
it via ``fd_client.flight_deck_headers()``; the agent never gets to assert who it
is. Same rule the Google ``/access_token`` endpoint follows.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request

from captain_claw.logging import get_logger

log = get_logger(__name__)


def is_browser_request(request: Request) -> bool:
    """Browsers stamp every fetch / WebSocket with ``Origin`` (cross-origin, and
    any non-GET) and/or the ``Sec-Fetch-*`` metadata headers; captain-claw agents
    (httpx) send none of them."""
    names = {str(k).lower() for k in request.headers.keys()}
    return "origin" in names or any(n.startswith("sec-fetch-") for n in names)


@dataclass(frozen=True)
class AgentCaller:
    slug: str   # FD_AGENT_SLUG the agent was spawned with ("" if unknown)
    owner: str  # owner recorded at spawn ("" if none was)


def require_agent_caller(request: Request, *, what: str) -> AgentCaller:
    """Resolve the calling agent or raise.

    * Browser request → 403 (never a legitimate caller of these endpoints).
    * Transport gate (``server._agent_caller_ok``: this deck's agent secret in
      ``X-Agent-Secret``, or loopback unless ``FD_LOCKDOWN``) → 401.
    * ``X-Agent-Auth`` missing, or not a token this deck issued → 403.
    """
    if is_browser_request(request):
        raise HTTPException(
            status_code=403,
            detail=f"{what[:1].upper()}{what[1:]} is for Flight Deck agents, not browsers",
        )
    from captain_claw.flight_deck.server import _agent_caller_ok, _find_agent_by_auth

    if not _agent_caller_ok(request):
        raise HTTPException(
            status_code=401,
            detail=(
                "Unauthorized agent call — this Flight Deck requires its agent "
                "secret (X-Agent-Secret) from this caller"
            ),
        )
    token = request.headers.get("X-Agent-Auth", "")
    if not token:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Flight Deck can't identify this agent (no X-Agent-Auth) — only "
                f"agents spawned by this Flight Deck can use {what}; respawn it "
                f"from Flight Deck"
            ),
        )
    try:
        matched, owner, slug = _find_agent_by_auth(token)
    except Exception as exc:  # fail closed; never log the token
        log.warning("Agent identity lookup failed: %s", type(exc).__name__)
        matched, owner, slug = False, "", ""
    if not matched:
        raise HTTPException(
            status_code=403,
            detail="Unknown agent — it was not spawned by this Flight Deck",
        )
    return AgentCaller(slug=slug, owner=owner)
