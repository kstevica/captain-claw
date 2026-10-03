"""Admin-only diagnostics for the host's Google subscription CLI.

No credential endpoint: agents must run under the same local OS account.
Checking quota uses CLI commands and does not send a model prompt.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from captain_claw.exceptions import LLMError
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.llm.antigravity import resolve_cli, run_cli, validate_subscription_settings

router = APIRouter(prefix="/fd/antigravity", tags=["antigravity"])


async def require_host_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required for the host subscription.")
    return user


@router.get("/status")
async def subscription_status(_user: dict = Depends(require_host_admin)) -> dict[str, Any]:
    status: dict[str, Any] = {
        "provider": "antigravity-cli", "installed": False, "settings_safe": False,
        "supports_tools": False, "local_only": True,
    }
    try:
        resolve_cli()
        status["installed"] = True
        validate_subscription_settings()
        status["settings_safe"] = True
    except LLMError as exc:
        status["detail"] = str(exc)
    return status


@router.post("/check")
async def check_subscription(_user: dict = Depends(require_host_admin)) -> dict[str, Any]:
    try:
        usage_raw, models_raw = await asyncio.gather(
            run_cli(["-p", "/usage", "--output-format", "json", "--print-timeout", "30s"], timeout=40),
            run_cli(["models"], timeout=40),
        )
        data = json.loads(usage_raw)
        if not isinstance(data, dict) or data.get("status") != "SUCCESS":
            raise LLMError("Sign in with `agy` on this host, then check the connection again.")
        # Do not forward arbitrary CLI logs or settings to the browser.
        report = data.get("response", "")
        if not isinstance(report, str):
            raise LLMError("Antigravity returned an invalid quota report.")
        if any(term in report.lower() for term in ("access_token", "refresh_token", "bearer ", "authorization")):
            raise LLMError("Quota report could not be displayed safely. Check `/usage` in Antigravity.")
        models = [line.split()[0] for line in models_raw.decode("utf-8", "replace").splitlines()
                  if line.startswith("gemini-")]
        return {"connected": True, "quota": report[:8000], "models": models,
                "extra_credits_enabled": False}
    except (LLMError, OSError, ValueError):
        raise HTTPException(
            400, "Connection check failed. Run `agy` on the host, sign in with Google, "
            "disable AI Credit Overages and remove modelProvider from settings."
        ) from None
