"""Credential management for the browser tool.

Handles storing, retrieving, encrypting, and decrypting credentials
for automated browser login flows.  Uses Fernet symmetric encryption
from the ``cryptography`` library when available, with a fallback to
Base64 obfuscation (with a warning) if no encryption key is configured.

Encryption key resolution order:
1. ``config.tools.browser.credential_encryption_key``
2. ``CLAW_BROWSER_CREDENTIAL_KEY`` environment variable
3. No key → Base64 obfuscation + warning
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ---------- optional dependency guard ----------------------------------------

try:
    from cryptography.fernet import Fernet, InvalidToken

    _HAS_FERNET = True
except ImportError:  # pragma: no cover
    _HAS_FERNET = False
    Fernet = None  # type: ignore[assignment,misc]
    InvalidToken = Exception  # type: ignore[assignment,misc]


# ---------- CredentialStore --------------------------------------------------


class CredentialStore:
    """Encrypt/decrypt browser credentials.

    The store itself is stateless — encryption operations are pure functions.
    Persistent storage is handled by the SessionManager's
    ``browser_credentials`` table.
    """

    def __init__(self, key: str | None = None) -> None:
        self._key = self._resolve_key(key)
        self._fernet: Any | None = None

        if self._key and _HAS_FERNET:
            try:
                self._fernet = Fernet(self._key.encode())
                log.info("Credential encryption enabled (Fernet)")
            except Exception as e:
                log.warning(
                    "Invalid Fernet key — falling back to obfuscation",
                    error=str(e),
                )
                self._fernet = None
        elif self._key and not _HAS_FERNET:
            log.warning(
                "cryptography package not installed — "
                "credentials will be Base64-obfuscated only. "
                "Install with: pip install cryptography"
            )
        else:
            log.warning(
                "No credential encryption key configured — "
                "credentials will be Base64-obfuscated only. "
                "Set tools.browser.credential_encryption_key in config "
                "or CLAW_BROWSER_CREDENTIAL_KEY env var."
            )

    @staticmethod
    def _resolve_key(explicit_key: str | None) -> str:
        """Resolve the encryption key from config, env var, or empty."""
        if explicit_key:
            return explicit_key.strip()

        cfg = get_config()
        config_key = cfg.tools.browser.credential_encryption_key.strip()
        if config_key:
            return config_key

        env_key = os.environ.get("CLAW_BROWSER_CREDENTIAL_KEY", "").strip()
        return env_key

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key.

        Returns the key as a URL-safe base64 string.
        """
        if not _HAS_FERNET:
            raise RuntimeError(
                "cryptography package is required for key generation. "
                "Install with: pip install cryptography"
            )
        return Fernet.generate_key().decode()

    # -- encrypt / decrypt ----------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string.

        Returns a Fernet token if encryption is available, otherwise a
        Base64-encoded string prefixed with ``b64:``.
        """
        if self._fernet is not None:
            token: bytes = self._fernet.encrypt(plaintext.encode("utf-8"))
            return token.decode("ascii")

        # Fallback: Base64 obfuscation (NOT secure, just avoids plaintext)
        encoded = base64.urlsafe_b64encode(plaintext.encode("utf-8")).decode("ascii")
        return f"b64:{encoded}"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string back to plaintext.

        Handles both Fernet tokens and ``b64:`` fallback encoding.
        """
        if ciphertext.startswith("b64:"):
            # Base64 fallback
            encoded = ciphertext[4:]
            return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")

        if self._fernet is not None:
            try:
                decrypted: bytes = self._fernet.decrypt(ciphertext.encode("ascii"))
                return decrypted.decode("utf-8")
            except InvalidToken:
                log.error("Failed to decrypt credential — key may have changed")
                raise ValueError(
                    "Cannot decrypt credential. The encryption key may have changed "
                    "since the credential was stored. Re-store the credential."
                )

        # No Fernet but not Base64 either — might be an old Fernet token
        raise ValueError(
            "Cannot decrypt credential — cryptography package not installed "
            "and the value is not Base64-encoded. Install cryptography or re-store."
        )

    @property
    def is_encrypted(self) -> bool:
        """Return True if real Fernet encryption is active."""
        return self._fernet is not None

    # -- high-level helpers ---------------------------------------------------

    async def store_credential(
        self,
        app_name: str,
        url: str,
        username: str,
        password: str,
        *,
        auth_type: str = "form",
        login_selector_map: dict[str, str] | None = None,
        notes: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Encrypt password and store credential in the DB.

        Returns a summary dict (password masked).
        """
        from captain_claw.session import get_session_manager

        sm = get_session_manager()

        encrypted_password = self.encrypt(password)
        selector_json = json.dumps(login_selector_map) if login_selector_map else None

        entry = await sm.create_browser_credential(
            app_name=app_name,
            url=url,
            username=username,
            password_encrypted=encrypted_password,
            auth_type=auth_type,
            login_selector_map=selector_json,
            notes=notes,
            source_session=session_id,
        )

        return entry.to_dict(mask_password=True)

    async def get_credential(self, app_name: str) -> dict[str, Any] | None:
        """Retrieve and decrypt a credential.

        Returns a dict with plaintext password, or None if not found.
        """
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        entry = await sm.get_browser_credential(app_name)
        if not entry:
            return None

        try:
            plaintext_password = self.decrypt(entry.password_encrypted)
        except ValueError as e:
            return {
                "app_name": entry.app_name,
                "url": entry.url,
                "username": entry.username,
                "password": None,
                "error": str(e),
                "auth_type": entry.auth_type,
                "login_selector_map": (
                    json.loads(entry.login_selector_map)
                    if entry.login_selector_map else None
                ),
                "cookies": (
                    json.loads(entry.cookies)
                    if entry.cookies else None
                ),
            }

        return {
            "app_name": entry.app_name,
            "url": entry.url,
            "username": entry.username,
            "password": plaintext_password,
            "auth_type": entry.auth_type,
            "login_selector_map": (
                json.loads(entry.login_selector_map)
                if entry.login_selector_map else None
            ),
            "cookies": (
                json.loads(entry.cookies)
                if entry.cookies else None
            ),
        }

    async def list_credentials(self) -> list[dict[str, Any]]:
        """List all stored credentials (passwords masked)."""
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        entries = await sm.list_browser_credentials()
        return [e.to_dict(mask_password=True) for e in entries]

    async def delete_credential(self, app_name: str) -> bool:
        """Delete a stored credential."""
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        return await sm.delete_browser_credential(app_name)

    async def save_cookies(
        self, app_name: str, cookies: list[dict[str, Any]],
    ) -> bool:
        """Save browser cookies for a credential (for session persistence)."""
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        cookies_json = json.dumps(cookies)
        return await sm.update_browser_credential_cookies(app_name, cookies_json)

    async def load_cookies(self, app_name: str) -> list[dict[str, Any]] | None:
        """Load saved cookies for a credential.

        Returns cookie list or None if no cookies saved.
        """
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        entry = await sm.get_browser_credential(app_name)
        if not entry or not entry.cookies:
            return None

        try:
            return json.loads(entry.cookies)
        except json.JSONDecodeError:
            return None
