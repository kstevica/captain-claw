"""Signed bundles: which deck made a file.

Each deck has an Ed25519 key of its own (``<FD home>/bundle_signing_key``, mode
0600, made on first use). Every bundle it writes carries ``signature.json``:
the deck's public key, its name, and a signature over the exact bytes of
``manifest.json`` — which holds the sha256 of every other file, so one
signature covers the whole bundle.

On import the signature is checked against the key the file carries:

* **valid**   — made by the deck with that fingerprint, unchanged since;
* **invalid** — changed after it was signed: refused;
* **unsigned** — no signature (made elsewhere, or by an older deck).

A valid signature only says *which* key signed; whether to trust it is the
deck's call: its own key always, and the fingerprints an admin lists. With
"only trusted bundles" on, anything else is refused.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

SIGNATURE = "signature.json"
TRUST_SETTING = "fd:agent-bundle-trust"
_KEY_FILE = "bundle_signing_key"


def _key_path() -> Path:
    """Beside the deck's agent secret (agent_secret.py): per deck, never in an agent's reach."""
    from captain_claw.flight_deck.agent_secret import _real_user_home
    from captain_claw.flight_deck.fd_home import fd_home
    return fd_home(_real_user_home() / ".captain-claw-fd") / _KEY_FILE


def _private_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    path = _key_path()
    try:
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    except FileNotFoundError:
        pass
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    log.info("agent bundles: made this deck's signing key", path=str(path))
    return key


def _raw_public(key) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint(public_raw: bytes) -> str:
    """``ab12 cd34 …`` — eight groups of the public key's sha256."""
    h = hashlib.sha256(public_raw).hexdigest()[:32]
    return " ".join(h[i:i + 4] for i in range(0, 32, 4))


def own_fingerprint() -> str:
    return fingerprint(_raw_public(_private_key()))


def sign(manifest_bytes: bytes, deck_name: str) -> dict[str, Any]:
    key = _private_key()
    return {"alg": "ed25519", "deck": deck_name[:120],
            "public_key": base64.b64encode(_raw_public(key)).decode(),
            "signature": base64.b64encode(key.sign(manifest_bytes)).decode()}


def verify(manifest_bytes: bytes, sig: Any) -> dict[str, Any]:
    """``{"state": valid|invalid|unsigned, "fingerprint", "deck"}``."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if sig is None:
        return {"state": "unsigned", "fingerprint": "", "deck": ""}
    try:
        if not isinstance(sig, dict) or sig.get("alg") != "ed25519":
            raise ValueError("not an ed25519 signature")
        raw = base64.b64decode(str(sig["public_key"]), validate=True)
        Ed25519PublicKey.from_public_bytes(raw).verify(base64.b64decode(str(sig["signature"]), validate=True),
                                                       manifest_bytes)
    except (InvalidSignature, ValueError, KeyError, TypeError):
        return {"state": "invalid", "fingerprint": "", "deck": str((sig or {}).get("deck") or "")[:120]
                if isinstance(sig, dict) else ""}
    return {"state": "valid", "fingerprint": fingerprint(raw), "deck": str(sig.get("deck") or "")[:120]}


async def trust() -> dict[str, Any]:
    """The admin's list of trusted fingerprints and whether only those are taken."""
    from captain_claw.flight_deck.auth import get_db
    try:
        raw = await get_db().get_system_setting(TRUST_SETTING)
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    trusted = [{"fingerprint": norm(t.get("fingerprint")), "label": str(t.get("label") or "")[:80]}
               for t in (data.get("trusted") or []) if isinstance(t, dict) and norm(t.get("fingerprint"))]
    return {"trusted": trusted, "require_trusted": bool(data.get("require_trusted"))}


async def save_trust(trusted: list[dict[str, Any]], require_trusted: bool) -> dict[str, Any]:
    from captain_claw.flight_deck.auth import get_db
    clean = [{"fingerprint": norm(t.get("fingerprint")), "label": str(t.get("label") or "")[:80]}
             for t in trusted if isinstance(t, dict) and norm(t.get("fingerprint"))][:100]
    await get_db().set_system_setting(TRUST_SETTING, json.dumps({"trusted": clean,
                                                                 "require_trusted": bool(require_trusted)}))
    return {"trusted": clean, "require_trusted": bool(require_trusted)}


def norm(value: Any) -> str:
    """A fingerprint as ``ab12 cd34 …`` whatever spacing it was typed with ("" if it isn't one)."""
    h = "".join(ch for ch in str(value or "").lower() if ch in "0123456789abcdef")
    return " ".join(h[i:i + 4] for i in range(0, 32, 4)) if len(h) == 32 else ""


async def judge(state: dict[str, Any]) -> dict[str, Any]:
    """*state* (``verify``'s) with ``own`` and ``trusted`` — and ``refused``,
    the reason this deck won't take the bundle ("" when it will)."""
    t = await trust()
    own = state["state"] == "valid" and state["fingerprint"] == own_fingerprint()
    listed = next((x for x in t["trusted"] if x["fingerprint"] == state["fingerprint"]), None)
    trusted = own or (state["state"] == "valid" and listed is not None)
    refused = ""
    if state["state"] == "invalid":
        refused = "The file was changed after it was signed — not opened."
    elif t["require_trusted"] and not trusted:
        refused = ("This deck only takes bundles signed by a deck its admin trusts"
                   + (f" — this one is signed by {state['fingerprint']}." if state["fingerprint"] else " — this one isn't signed."))
    return {**state, "own": own, "trusted": trusted, "label": (listed or {}).get("label", ""), "refused": refused}
