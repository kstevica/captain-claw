"""The `ask_human` tool — a Bat worker pauses the run to ask the owner.

When a Bat worker hits something only the human can supply — a verification
code, a 2FA/OTP, a CAPTCHA it must not solve itself, a credential it doesn't
have — it calls this. The run stands down (awaiting_human) and resumes when the
owner answers; the answer is handed back to this step on its next attempt. Use
kind='secret' for codes/passwords so the value is kept out of logs, the bell and
chat channels. After calling this, STOP — don't guess or proceed.

Only works inside a Bat run (CLAW_BAT_SESSION set by the spawn). Routes to FD's
/fd/bat/agent/ask with the hardened Bat transport.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from captain_claw.tools.bat import _agent_secret_headers, _fd_url_pinned
from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class AskHumanTool(Tool):
    name = "ask_human"
    description = (
        "Inside a Bat run only: pause the run and ask the owner for something only they can give — a "
        "verification code, a 2FA/OTP, a CAPTCHA you must not solve, or a credential. Use kind='secret' "
        "for codes/passwords (kept private). After calling this, STOP: the run resumes and re-runs this "
        "step with the answer once the owner replies. Never guess, fabricate, or bypass a CAPTCHA."
    )
    timeout_seconds = 30.0
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What you need from the owner, in one clear sentence."},
            "kind": {"type": "string", "enum": ["input", "secret"],
                     "description": "'secret' for codes/passwords (kept out of logs); 'input' otherwise."},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": "Optional choices for the owner to pick from."},
        },
        "required": ["question"],
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
            port, auth = int(cfg.web.port or 0), (cfg.web.auth_token or "")
        except Exception:
            port, auth = 0, ""
        return {
            "web_auth": auth if _fd_url_pinned(fd_url) else "",
            "source_port": port,
            "owner_id": os.environ.get("FD_OWNER_ID", ""),
            "session_id": self._session(),
            "step_key": os.environ.get("CLAW_BAT_SUBTASK", ""),
        }

    async def execute(self, question: str = "", kind: str = "input", **kwargs: Any) -> ToolResult:
        if not self._session():
            return ToolResult(success=False, error="`ask_human` is only available inside a Bat run.")
        q = (question or "").strip()
        if not q:
            return ToolResult(success=False, error="Provide `question`.")
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Bat.")
        payload = {"kind": "secret" if kind == "secret" else "input", "question": q,
                   "options": kwargs.get("options") or []}
        try:
            import httpx
            body = {**self._identity(fd_url), **payload}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{fd_url}/fd/bat/agent/ask", json=body,
                                         headers=_agent_secret_headers(fd_url))
            if resp.status_code in (403, 404):
                return ToolResult(success=False, error="could not reach the Bat run (not authorized).")
            resp.raise_for_status()
            resp.json()
        except Exception as e:  # noqa: BLE001
            log.warning("ask_human error", error=str(e))
            return ToolResult(success=False, error=f"ask_human failed: {e}")
        return ToolResult(success=True, content=(
            "Asked the owner and PAUSED the run. Stop now — do not continue, guess, or retry. "
            "The run will re-run this step with their answer once they reply."))
