"""An agent's keys in a backup, sealed with a passphrase.

Only a full backup may carry the agent's keys, and only when its owner gives a
passphrase: ``secrets.enc`` in the bundle is the keys as JSON, encrypted with
AES-256-GCM under a key derived from the passphrase with scrypt. Without the
passphrase the bundle is a bundle without keys; Flight Deck never stores the
passphrase.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from captain_claw.flight_deck.agent_bundle.archive import BundleError

KDF = {"name": "scrypt", "n": 2 ** 15, "r": 8, "p": 1}
MIN_PASSPHRASE = 8


def _key(passphrase: str, salt: bytes, kdf: dict[str, Any]) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    n, r, p = int(kdf.get("n") or 0), int(kdf.get("r") or 0), int(kdf.get("p") or 0)
    if kdf.get("name") != "scrypt" or not (2 ** 14 <= n <= 2 ** 20) or not (1 <= r <= 16) or not (1 <= p <= 4):
        raise BundleError("The bundle's keys are sealed in a way this deck doesn't read.")
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase.encode("utf-8"))


def seal(payload: dict[str, Any], passphrase: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if len(passphrase or "") < MIN_PASSPHRASE:
        raise BundleError(f"The passphrase needs at least {MIN_PASSPHRASE} characters.")
    salt, nonce = os.urandom(16), os.urandom(12)
    sealed = AESGCM(_key(passphrase, salt, KDF)).encrypt(
        nonce, json.dumps(payload, ensure_ascii=False).encode("utf-8"), b"captain-agent-secrets")
    return json.dumps({"kdf": KDF, "salt": base64.b64encode(salt).decode(), "nonce": base64.b64encode(nonce).decode(),
                       "sealed": base64.b64encode(sealed).decode()}).encode("utf-8")


def open_sealed(blob: bytes, passphrase: str) -> dict[str, Any]:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        box = json.loads(blob.decode("utf-8"))
        salt, nonce = base64.b64decode(box["salt"]), base64.b64decode(box["nonce"])
        sealed = base64.b64decode(box["sealed"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise BundleError("The bundle's keys aren't readable.") from exc
    try:
        plain = AESGCM(_key(passphrase or "", salt, box.get("kdf") or {})).decrypt(
            nonce, sealed, b"captain-agent-secrets")
    except InvalidTag:
        raise BundleError("That passphrase doesn't open its keys.") from None
    data = json.loads(plain.decode("utf-8"))
    return data if isinstance(data, dict) else {}
