"""A secret the person typed for this agent (``request_secret``), by reference.

The agent only ever holds ``sec_…`` and a length. Its browser asks Flight Deck
for the value right when it types it into a field (:func:`release`), and
everything the browser says afterwards is scrubbed of it (:func:`scrub`), so
the value never reaches the model, the session, the tool-call register or the
logs — those only see the reference.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

REF_RE = re.compile(r"^sec_[0-9a-f]{16}$")
SHOWN = "[the secret]"


class SecretError(Exception):
    """Why the value couldn't be had; said to the model as it is."""


async def release(ref: str) -> str:
    """The value behind *ref* for this agent's current person. Raises SecretError."""
    from captain_claw import speaker
    from captain_claw.fd_client import flight_deck_base, flight_deck_headers

    ref = str(ref or "").strip()
    if not REF_RE.match(ref):
        raise SecretError("secret_ref must be the sec_… reference request_secret gave you.")
    base = flight_deck_base()
    if not base:
        raise SecretError("Secrets need Flight Deck.")
    headers = flight_deck_headers()
    try:
        headers.update(speaker.grant_headers())
        params = speaker.grant_params()
    except speaker.SpeakerGrantMissing:
        raise SecretError("This member's turn has no grant, so their secret can't be had.") from None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{base}/fd/asks/agent/secret", headers=headers, params=params,
                                     json={"ref": ref})
    except httpx.HTTPError as exc:
        raise SecretError(f"Flight Deck didn't answer ({type(exc).__name__}).") from None
    if resp.status_code == 404:
        raise SecretError("That secret is gone: it was used up (3 times) or is older than 10 minutes. "
                          "Ask for it again with request_secret.")
    if resp.status_code != 200:
        raise SecretError(f"Flight Deck refused the secret ({resp.status_code}).")
    value = resp.json().get("value")
    if not isinstance(value, str) or not value:
        raise SecretError("Flight Deck gave no value for that secret.")
    return value


def _forms(value: str) -> list[str]:
    """The value as it may show up in text: plain, JSON-escaped, and URL-encoded
    (a captured form post or a query string)."""
    from urllib.parse import quote, quote_plus

    forms = {value, json.dumps(value, ensure_ascii=False)[1:-1], json.dumps(value)[1:-1],
             quote(value, safe=""), quote_plus(value), quote(value)}
    return sorted((f for f in forms if f), key=len, reverse=True)


def scrub_text(text: str, value: str) -> str:
    if not text or not value:
        return text
    for form in _forms(value):
        text = text.replace(form, SHOWN)
    return text


def scrub(result: Any, value: str) -> Any:
    """*result* (a ToolResult) with every trace of *value* replaced."""
    for attr in ("content", "error", "system_hint"):
        current = getattr(result, attr, None)
        if isinstance(current, str) and current:
            setattr(result, attr, scrub_text(current, value))
    return result
