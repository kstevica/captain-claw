"""The `spend` tool — a Bat worker's capped, pre-authorized real-money spend.

A Bat worker uses this to spend real money inside its run, under the owner's cap
(owner decision #2: pre-authorization, auto-approve under a per-item limit,
owner-approve above it, a run cap enforced across restarts). The worker must
DECLARE a purchase and get back 'approved' BEFORE it pays; anything at/above the
per-item limit returns 'requested' and the run pauses for the owner. After the
charge goes through, the worker SETTLES with the actual amount.

Only works inside a Bat run (CLAW_BAT_SESSION is set by the spawn); elsewhere it
refuses. Routes to FD's /fd/bat/agent/spend/* with the hardened Bat transport.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from captain_claw.tools.bat import _agent_secret_headers, _fd_url_pinned
from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class SpendTool(Tool):
    name = "spend"
    description = (
        "Spend real money inside a Bat run, under the owner's cap. ALWAYS declare first and wait for "
        "'approved' before you pay: action 'authorize' (merchant, merchant_domain, amount_usd, "
        "description). A small purchase auto-approves; a larger one returns 'requested' and the run "
        "pauses for the owner — stop and let it resume. After the charge completes, call 'settle' "
        "(spend_id, actual_usd, order_ref). 'void' cancels an unused authorization; 'status' shows the "
        "remaining budget. Never pay without an 'approved' authorization id."
    )
    timeout_seconds = 30.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["authorize", "settle", "void", "status"]},
            "merchant": {"type": "string", "description": "Human name of who is being paid ('authorize')."},
            "merchant_domain": {"type": "string", "description": "The payee's domain, e.g. 'stripe.com' ('authorize')."},
            "amount_usd": {"type": "number", "description": "Amount in USD to authorize ('authorize')."},
            "description": {"type": "string", "description": "What the purchase is for ('authorize')."},
            "spend_id": {"type": "string", "description": "Authorization id from 'authorize' (for 'settle'/'void')."},
            "actual_usd": {"type": "number", "description": "The amount actually charged ('settle')."},
            "order_ref": {"type": "string", "description": "Order/receipt id as evidence ('settle')."},
        },
        "required": ["action"],
    }

    def _session(self) -> str:
        return (os.environ.get("CLAW_BAT_SESSION") or "").strip()

    def _get_fd_url(self, **kwargs: Any) -> str:
        from captain_claw.fd_client import resolve_flight_deck_url
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = (getattr(session, "metadata", {}) or {}) if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        return resolve_flight_deck_url(fd_url)

    def _identity(self, fd_url: str) -> dict:
        from captain_claw.config import get_config
        try:
            cfg = get_config()
            port = int(cfg.web.port or 0)
            auth = cfg.web.auth_token or ""
        except Exception:
            port, auth = 0, ""
        return {
            "web_auth": auth if _fd_url_pinned(fd_url) else "",
            "source_port": port,
            "owner_id": os.environ.get("FD_OWNER_ID", ""),
            "session_id": self._session(),
            "agent": os.environ.get("FD_AGENT_SLUG", ""),
        }

    async def _post(self, fd_url: str, path: str, payload: dict) -> Any:
        import httpx
        body = {**self._identity(fd_url), **payload}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{fd_url}{path}", json=body, headers=_agent_secret_headers(fd_url))
        if resp.status_code == 404:
            return {"_error": "not found"}
        if resp.status_code == 403:
            return {"_error": "this agent's owner could not be resolved (not authorized)"}
        resp.raise_for_status()
        return resp.json()

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not self._session():
            return ToolResult(success=False, error="`spend` is only available inside a Bat run.")
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Bat.")
        try:
            if action == "authorize":
                return await self._authorize(fd_url, **kwargs)
            if action == "settle":
                return await self._settle(fd_url, **kwargs)
            if action == "void":
                d = await self._post(fd_url, "/fd/bat/agent/spend/void", {"spend_id": kwargs.get("spend_id", "")})
                return self._done(d, "Authorization voided.")
            if action == "status":
                d = await self._post(fd_url, "/fd/bat/agent/spend/status", {})
                if isinstance(d, dict) and d.get("_error"):
                    return ToolResult(success=False, error=d["_error"])
                return ToolResult(success=True, content=(
                    f"Spend {'enabled' if d.get('enabled') else 'DISABLED'}: "
                    f"${d.get('committed_usd', 0):.2f} committed of ${d.get('run_usd_cap', 0):.2f} cap, "
                    f"auto-approve ≤ ${d.get('per_item_usd', 0):.2f}. {len(d.get('items', []))} item(s)."))
            return ToolResult(success=False, error=f"Unknown action '{action}'.")
        except Exception as e:  # noqa: BLE001
            log.warning("spend tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Spend request failed: {e}")

    async def _authorize(self, fd_url: str, **kwargs: Any) -> ToolResult:
        amount = float(kwargs.get("amount_usd") or 0.0)
        payload = {"merchant": kwargs.get("merchant", ""), "merchant_domain": kwargs.get("merchant_domain", ""),
                   "amount_usd": amount, "description": kwargs.get("description", "")}
        d = await self._post(fd_url, "/fd/bat/agent/spend/authorize", payload)
        if isinstance(d, dict) and d.get("_error"):
            return ToolResult(success=False, error=d["_error"])
        status = d.get("status")
        if status == "approved":
            return ToolResult(success=True, content=(
                f"APPROVED (id {d.get('id')}): you may spend up to ${amount:.2f} at "
                f"{payload['merchant_domain'] or payload['merchant']}. Settle with the actual amount after paying."))
        if status == "requested":
            return ToolResult(success=True, content=(
                "REQUESTED: this purchase needs the owner's approval and the run is now paused. "
                "Do NOT pay. Stop here and report that you're waiting on approval; the run resumes when they answer."))
        return ToolResult(success=True, content=f"DENIED: {d.get('reason', 'not allowed')}. Do not pay.")

    async def _settle(self, fd_url: str, **kwargs: Any) -> ToolResult:
        payload = {"spend_id": kwargs.get("spend_id", ""), "actual_usd": float(kwargs.get("actual_usd") or 0.0),
                   "order_ref": kwargs.get("order_ref", "")}
        d = await self._post(fd_url, "/fd/bat/agent/spend/settle", payload)
        return self._done(d, "Spend settled.")

    @staticmethod
    def _done(d: Any, ok_msg: str) -> ToolResult:
        if isinstance(d, dict) and d.get("_error"):
            return ToolResult(success=False, error=d["_error"])
        if isinstance(d, dict) and d.get("ok"):
            return ToolResult(success=True, content=ok_msg)
        return ToolResult(success=False, error=(d or {}).get("reason", "failed"))
