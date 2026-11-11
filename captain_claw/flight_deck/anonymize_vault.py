"""Anonymization vaults: the real values behind the tokens, kept encrypted.

A vault belongs to a **person** (on a shared agent's member turn, the member —
never the agent's owner) and holds one case: the same person, email or IBAN
gets the same token in every document anonymized into it, and different
vaults never share tokens' meanings.

Store: ``anonymize.db`` in Flight Deck's data dir (``FD_DATA_DIR``), like
``external_memory.db``. Each vault has its own Fernet key, sealed with the deck
data key (:mod:`fd_crypto`, kept apart from the database); every entry — the
original value, its lookup key, the forms it was seen in — is encrypted with
that vault key, so the database alone reveals token names and kinds, nothing
else. A vault's name is sealed too (an agent may well call a case "Müller
v. Acme"); it is found by a keyed hash of the name, with a random key of this
deck's, so the hash can't be checked against a list of names. Deleting a vault
deletes its key and its entries. Events (anonymize,
restore, view, delete …) record counts, never values.

Without the ``cryptography`` package nothing is stored: reversible runs fail
(:class:`fd_crypto.CryptoUnavailable`) and only ``one_way`` works.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.anonymize.engine import Entity, VaultState
from captain_claw.flight_deck import fd_crypto

MAX_VAULTS_PER_OWNER = 1000
MAX_ENTRIES_PER_VAULT = 20000
NAME_MAX = 80

_DDL = """
CREATE TABLE IF NOT EXISTS vaults (
    id            TEXT PRIMARY KEY,
    owner_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    name_key      TEXT NOT NULL,
    key_sealed    TEXT NOT NULL,
    counters      TEXT NOT NULL DEFAULT '{}',
    kinds         TEXT NOT NULL DEFAULT '{}',
    entity_count  INTEGER NOT NULL DEFAULT 0,
    created_by    TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    last_used_at  REAL NOT NULL,
    expires_at    REAL,
    UNIQUE (owner_id, name_key)
);
CREATE INDEX IF NOT EXISTS idx_vaults_owner ON vaults(owner_id, last_used_at);
CREATE TABLE IF NOT EXISTS entries (
    vault_id    TEXT NOT NULL,
    token       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (vault_id, token)
);
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id  TEXT NOT NULL,
    owner_id  TEXT NOT NULL,
    at        REAL NOT NULL,
    actor     TEXT NOT NULL DEFAULT '',
    action    TEXT NOT NULL,
    detail    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_vault ON events(vault_id, id);
CREATE TABLE IF NOT EXISTS meta (
    k  TEXT PRIMARY KEY,
    v  TEXT NOT NULL
);
"""


class VaultError(ValueError):
    """A request about a vault that can't be done (name, limits)."""


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    if not name:
        raise VaultError("a vault needs a name")
    if len(name) > NAME_MAX:
        raise VaultError(f"a vault name is at most {NAME_MAX} characters")
    if any(ord(c) < 32 for c in name):
        raise VaultError("a vault name can't contain control characters")
    return name


def _open_name(sealed: str) -> str:
    try:
        name = fd_crypto.unseal(sealed)
    except fd_crypto.CryptoUnavailable:
        return "(locked)"
    return str(name) if name else "(unreadable)"


def _public(row: dict[str, Any]) -> dict[str, Any]:
    """A vault as anyone may see it: no key, no values."""
    return {
        "id": row["id"], "name": row["name"], "entity_count": row["entity_count"],
        "kinds": json.loads(row["kinds"] or "{}"), "created_by": row["created_by"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "last_used_at": row["last_used_at"], "expires_at": row["expires_at"],
    }


class VaultStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None
        self._conn_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}

    async def _conn(self) -> aiosqlite.Connection:
        async with self._conn_lock:
            if self._db is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                db = await aiosqlite.connect(str(self.path))
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                await db.executescript(_DDL)
                await db.commit()
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
                self._db = db
            return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def lock(self, vault_id: str) -> asyncio.Lock:
        """Held around load → run → save, so two runs into one vault can't
        hand out the same token number."""
        return self._locks.setdefault(vault_id, asyncio.Lock())

    # ── vaults ──────────────────────────────────────────────────────

    async def list_vaults(self, owner: str) -> list[dict[str, Any]]:
        await self.sweep_expired()
        db = await self._conn()
        cur = await db.execute("SELECT * FROM vaults WHERE owner_id=? ORDER BY last_used_at DESC", (owner,))
        rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            r["name"] = _open_name(r["name"])
        return [_public(r) for r in rows]

    async def _pepper(self) -> bytes:
        """This deck's random key for vault-name lookups, sealed in the DB."""
        db = await self._conn()
        cur = await db.execute("SELECT v FROM meta WHERE k='name_pepper'")
        row = await cur.fetchone()
        if row is not None:
            value = fd_crypto.unseal(row[0])
            if not value:
                raise fd_crypto.CryptoUnavailable("the vault-name key can't be opened with this deck's data key")
            return bytes.fromhex(value)
        value = secrets.token_hex(32)
        await db.execute("INSERT OR IGNORE INTO meta (k, v) VALUES ('name_pepper', ?)", (fd_crypto.seal(value),))
        await db.commit()
        return await self._pepper()

    async def _name_key(self, name: str) -> str:
        norm = re.sub(r"\s+", " ", str(name or "")).strip().casefold()
        return hmac.new(await self._pepper(), norm.encode("utf-8"), hashlib.sha256).hexdigest()

    async def _row(self, owner: str, vault: str) -> dict[str, Any] | None:
        """A vault of *owner*'s by id or by name (its ``name`` opened)."""
        db = await self._conn()
        cur = await db.execute("SELECT * FROM vaults WHERE owner_id=? AND id=?", (owner, vault))
        row = await cur.fetchone()
        if row is None:
            cur = await db.execute("SELECT * FROM vaults WHERE owner_id=? AND name_key=?",
                                   (owner, await self._name_key(vault)))
            row = await cur.fetchone()
        if row is None:
            return None
        out = dict(row)
        out["name"] = _open_name(out["name"])
        return out

    async def get(self, owner: str, vault: str) -> dict[str, Any] | None:
        await self.sweep_expired()
        return await self._row(owner, vault)

    async def find_or_create(self, owner: str, name: str, *, created_by: str = "") -> dict[str, Any]:
        await self.sweep_expired()
        existing = await self._row(owner, name)
        if existing is not None:
            return existing
        name = clean_name(name)
        db = await self._conn()
        cur = await db.execute("SELECT COUNT(*) FROM vaults WHERE owner_id=?", (owner,))
        if (await cur.fetchone())[0] >= MAX_VAULTS_PER_OWNER:
            raise VaultError(f"you have {MAX_VAULTS_PER_OWNER} vaults — delete some first")
        from cryptography.fernet import Fernet  # fd_crypto.seal below fails first without it

        sealed = fd_crypto.seal(Fernet.generate_key().decode("ascii"))
        now = time.time()
        vid = "v_" + secrets.token_hex(8)
        try:
            await db.execute(
                """INSERT INTO vaults (id, owner_id, name, name_key, key_sealed, created_by, created_at,
                                       updated_at, last_used_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (vid, owner, fd_crypto.seal(name), await self._name_key(name), sealed, created_by[:120],
                 now, now, now))
            await db.commit()
        except aiosqlite.IntegrityError:
            pass  # created meanwhile by another call
        row = await self._row(owner, name)
        assert row is not None
        if row["id"] == vid:
            await self.log(owner, vid, created_by, "create", {})
        return row

    def _fernet(self, row: dict[str, Any]) -> Any:
        from cryptography.fernet import Fernet

        key = fd_crypto.unseal(row["key_sealed"])
        if not key:
            raise fd_crypto.CryptoUnavailable("this vault's key can't be opened with this deck's data key")
        return Fernet(key.encode("ascii"))

    async def load_state(self, row: dict[str, Any]) -> VaultState:
        f = self._fernet(row)
        db = await self._conn()
        cur = await db.execute("SELECT token, kind, payload FROM entries WHERE vault_id=?", (row["id"],))
        entities = []
        for r in await cur.fetchall():
            try:
                data = json.loads(f.decrypt(r["payload"].encode("ascii")).decode("utf-8"))
            except Exception:
                continue  # an entry this key can't open is skipped, never guessed
            entities.append(Entity(r["token"], r["kind"], str(data.get("key") or ""),
                                   str(data.get("original") or ""), list(data.get("forms") or [])))
        return VaultState(entities, json.loads(row["counters"] or "{}"))

    async def save_run(self, row: dict[str, Any], state: VaultState) -> int:
        """Store what a run added or changed; returns how many entries were written."""
        f = self._fernet(row)
        db = await self._conn()
        now = time.time()
        to_write = [e for e in state.entities if (e.new and e.count > 0) or (e.changed and not e.new)]
        cur = await db.execute("SELECT COUNT(*) FROM entries WHERE vault_id=?", (row["id"],))
        current = (await cur.fetchone())[0]
        if current + sum(1 for e in to_write if e.new) > MAX_ENTRIES_PER_VAULT:
            raise VaultError(f"a vault holds at most {MAX_ENTRIES_PER_VAULT} values — use another vault")
        for e in to_write:
            payload = f.encrypt(json.dumps(
                {"key": e.key, "original": e.original, "forms": e.forms}, ensure_ascii=False).encode("utf-8")
            ).decode("ascii")
            await db.execute(
                """INSERT INTO entries (vault_id, token, kind, payload, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(vault_id, token) DO UPDATE SET payload=excluded.payload,
                       updated_at=excluded.updated_at""",
                (row["id"], e.token, e.kind, payload, now, now))
        cur = await db.execute("SELECT kind, COUNT(*) FROM entries WHERE vault_id=? GROUP BY kind", (row["id"],))
        kinds = {k: n for k, n in await cur.fetchall()}
        await db.execute(
            "UPDATE vaults SET counters=?, kinds=?, entity_count=?, updated_at=?, last_used_at=? WHERE id=?",
            (json.dumps(state.counters), json.dumps(kinds), sum(kinds.values()), now, now, row["id"]))
        await db.commit()
        return len(to_write)

    async def touch(self, row: dict[str, Any]) -> None:
        db = await self._conn()
        await db.execute("UPDATE vaults SET last_used_at=? WHERE id=?", (time.time(), row["id"]))
        await db.commit()

    async def entries(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Every entry with its real value — for the vault's person only."""
        state = await self.load_state(row)
        db = await self._conn()
        cur = await db.execute("SELECT token, created_at, updated_at FROM entries WHERE vault_id=?", (row["id"],))
        times = {r["token"]: (r["created_at"], r["updated_at"]) for r in await cur.fetchall()}
        out = []
        for e in state.entities:
            created, updated = times.get(e.token, (None, None))
            out.append({"token": e.token, "kind": e.kind, "original": e.original, "forms": e.forms,
                        "created_at": created, "updated_at": updated})
        return sorted(out, key=lambda x: (x["kind"], _token_number(x["token"])))

    async def update(self, owner: str, vault: str, *, name: str | None = None,
                     expires_at: float | None | str = "keep") -> dict[str, Any] | None:
        row = await self._row(owner, vault)
        if row is None:
            return None
        db = await self._conn()
        if name is not None:
            name = clean_name(name)
            other = await self._row(owner, name)
            if other is not None and other["id"] != row["id"]:
                raise VaultError(f"you already have a vault named '{name}'")
            await db.execute("UPDATE vaults SET name=?, name_key=?, updated_at=? WHERE id=?",
                             (fd_crypto.seal(name), await self._name_key(name), time.time(), row["id"]))
        if expires_at != "keep":
            await db.execute("UPDATE vaults SET expires_at=?, updated_at=? WHERE id=?",
                             (expires_at, time.time(), row["id"]))
        await db.commit()
        return await self._row(owner, row["id"])

    async def delete(self, owner: str, vault: str, *, actor: str = "") -> bool:
        row = await self._row(owner, vault)
        if row is None:
            return False
        await self._delete_row(row)
        await self.log(owner, row["id"], actor, "delete", {"entities": row["entity_count"]})
        return True

    async def _delete_row(self, row: dict[str, Any]) -> None:
        db = await self._conn()
        await db.execute("DELETE FROM entries WHERE vault_id=?", (row["id"],))
        await db.execute("DELETE FROM vaults WHERE id=?", (row["id"],))
        await db.commit()
        self._locks.pop(row["id"], None)

    async def sweep_expired(self, now: float | None = None) -> int:
        db = await self._conn()
        now = time.time() if now is None else now
        cur = await db.execute("SELECT * FROM vaults WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        rows = [dict(r) for r in await cur.fetchall()]
        for row in rows:
            await self._delete_row(row)
            await self.log(row["owner_id"], row["id"], "deck", "expired", {"entities": row["entity_count"]})
        return len(rows)

    # ── events ──────────────────────────────────────────────────────

    async def log(self, owner: str, vault_id: str, actor: str, action: str, detail: dict[str, Any]) -> None:
        db = await self._conn()
        await db.execute(
            "INSERT INTO events (vault_id, owner_id, at, actor, action, detail) VALUES (?, ?, ?, ?, ?, ?)",
            (vault_id, owner, time.time(), actor[:120], action, json.dumps(detail)))
        await db.commit()

    async def events(self, owner: str, vault_id: str, limit: int = 200) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT at, actor, action, detail FROM events WHERE owner_id=? AND vault_id=? ORDER BY id DESC LIMIT ?",
            (owner, vault_id, max(1, min(limit, 1000))))
        return [{"at": r["at"], "actor": r["actor"], "action": r["action"], "detail": json.loads(r["detail"])}
                for r in await cur.fetchall()]


def _token_number(token: str) -> int:
    m = re.search(r"_(\d+)⟧$", token)
    return int(m.group(1)) if m else 0


# ── the store this deck uses ────────────────────────────────────────

_data_dir: Path | None = None
_store: VaultStore | None = None


def configure(*, data_dir: Path) -> None:
    global _data_dir
    _data_dir = Path(data_dir)


def store() -> VaultStore:
    global _store
    if _store is None:
        base = _data_dir
        if base is None:
            env = (os.environ.get("FD_DATA_DIR") or "").strip()
            if env:
                base = Path(env).expanduser().resolve()
            else:
                from captain_claw.flight_deck.server import DATA_DIR

                base = DATA_DIR
        _store = VaultStore(base / "anonymize.db")
    return _store


def set_store(value: VaultStore | None) -> None:
    """Tests: point the deck at another store."""
    global _store
    _store = value


async def close() -> None:
    if _store is not None:
        await _store.close()
