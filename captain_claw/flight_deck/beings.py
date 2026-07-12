"""Iskra beings store — registry, wallets, conservation ledger, lifecycle.

The life layer's persistence (docs/living-beings-plan.md §10). Sync sqlite3 +
a lock in its own ``beings.db``, mirroring AutonomyStore/ConsciousnessStore.

Conservation by construction: every wallet balance mutation happens inside
exactly one method (:meth:`BeingsStore._apply`) that writes the matching
``token_transfers`` row in the same transaction. Mint rows have no
``from_being``; sink rows (usage spend, metamorphosis burns) have no
``to_being``; so at any instant::

    sum(balances) == sum(mints) - sum(sinks)

— checked by :meth:`conservation`, asserted in tests. The Constitution module
supplies the physics (stage gates, tier weights, preset clamps); this store
enforces them at the only door tokens can pass through.

Clock: methods take an optional ``now`` so the beings loop, tests and backfills
can be deterministic; default is real UTC.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.logging import get_logger

log = get_logger(__name__)

STATES = ("alive", "paused", "torpor", "dead")
TRANSFER_REASONS = (
    "allowance", "usage", "fee", "gift", "trade",
    "procreation", "metamorphosis_burn", "adjust",
)


class BeingError(Exception):
    """Base for being-domain failures; ``status`` maps to HTTP in routes."""

    status = 400

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        if status is not None:
            self.status = status


class BeingNotFound(BeingError):
    status = 404


class InsufficientTokens(BeingError):
    status = 402


class BurnCapExceeded(BeingError):
    status = 429


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "beings.db"
    return Path("~/.captain-claw/beings.db").expanduser()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "iskra"


class BeingsStore:
    """SQLite-backed being registry + wallet. Sync sqlite3 + a lock."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS beings (
                    id                TEXT PRIMARY KEY,
                    owner_id          TEXT NOT NULL,
                    slug              TEXT NOT NULL UNIQUE,
                    name              TEXT NOT NULL,
                    stage             TEXT NOT NULL DEFAULT 'egg',
                    state             TEXT NOT NULL DEFAULT 'alive',
                    genome            TEXT NOT NULL,
                    drives            TEXT NOT NULL DEFAULT '{}',
                    attention_credits INTEGER NOT NULL DEFAULT 3,
                    next_wake_at      TEXT,
                    born_at           TEXT NOT NULL,
                    hatched_at        TEXT,
                    died_at           TEXT,
                    lineage           TEXT NOT NULL DEFAULT '[]',
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_beings_owner ON beings(owner_id);

                CREATE TABLE IF NOT EXISTS being_wallets (
                    being_id         TEXT PRIMARY KEY REFERENCES beings(id),
                    balance_tokens   INTEGER NOT NULL DEFAULT 0,
                    allowance_preset TEXT NOT NULL DEFAULT '2M',
                    period           TEXT NOT NULL DEFAULT 'day',
                    daily_burn_cap   INTEGER,
                    savings_ceiling  INTEGER,
                    reserve_tokens   INTEGER NOT NULL DEFAULT 0,
                    updated_at       TEXT NOT NULL
                );

                -- The conservation ledger. from_being NULL = mint (parent),
                -- to_being NULL = sink (usage spend / metamorphosis burn).
                CREATE TABLE IF NOT EXISTS token_transfers (
                    id         TEXT PRIMARY KEY,
                    owner_id   TEXT NOT NULL,
                    from_being TEXT,
                    to_being   TEXT,
                    tokens     INTEGER NOT NULL,
                    reason     TEXT NOT NULL,
                    job_id     TEXT,
                    note       TEXT,
                    at         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_transfers_from
                    ON token_transfers(from_being, at);
                CREATE INDEX IF NOT EXISTS idx_transfers_to
                    ON token_transfers(to_being, at);
                CREATE INDEX IF NOT EXISTS idx_transfers_owner
                    ON token_transfers(owner_id, at);

                CREATE TABLE IF NOT EXISTS being_events (
                    id       TEXT PRIMARY KEY,
                    being_id TEXT NOT NULL,
                    kind     TEXT NOT NULL,
                    data     TEXT NOT NULL DEFAULT '{}',
                    at       TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_being_events
                    ON being_events(being_id, at);
                """
            )
            self._c().commit()

    # ── Registry ─────────────────────────────────────────────────────

    def conceive(
        self,
        owner_id: str,
        name: str,
        *,
        attributes: dict | None = None,
        preset: str | None = None,
        roll_seed: int | None = None,
        voice_seed: str = "",
        interest_seeds: list[str] | None = None,
        allowance_preset: str = "2M",
        now: datetime | None = None,
    ) -> dict:
        """Point-buy conception (Generation 1). Exactly one of attributes /
        preset / roll_seed selects the sheet; born as an egg with an empty,
        allowance-armed wallet."""
        now = now or _utcnow()
        if attributes is not None:
            errors = genome_mod.validate_point_buy(attributes)
            if errors:
                raise BeingError("; ".join(errors))
            sheet = {a: int(attributes[a]) for a in genome_mod.ATTRS}
        elif preset is not None:
            if preset not in genome_mod.PRESETS:
                raise BeingError(f"unknown preset {preset!r}")
            sheet = dict(genome_mod.PRESETS[preset])
        else:
            import random
            sheet = genome_mod.roll(random.Random(roll_seed))
        if allowance_preset not in constitution.ALLOWANCE_PRESETS:
            raise BeingError(f"unknown allowance preset {allowance_preset!r}")
        g = genome_mod.new_genome(
            sheet, voice_seed=voice_seed, interest_seeds=interest_seeds,
        )
        bid = uuid.uuid4().hex
        slug = f"iskra-{_slugify(name)}-{bid[:4]}"
        with self._lock:
            c = self._c()
            c.execute(
                "INSERT INTO beings (id, owner_id, slug, name, stage, state, genome,"
                " born_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (bid, owner_id, slug, name, "egg", "alive", json.dumps(g),
                 _iso(now), _iso(now), _iso(now)),
            )
            c.execute(
                "INSERT INTO being_wallets (being_id, allowance_preset, updated_at)"
                " VALUES (?,?,?)",
                (bid, allowance_preset, _iso(now)),
            )
            c.commit()
        self.record_event(bid, "conceived", {"sheet": sheet, "name": name}, now=now)
        return self.get(owner_id, slug)

    def _row(self, owner_id: str, slug: str) -> sqlite3.Row:
        row = self._c().execute(
            "SELECT * FROM beings WHERE owner_id = ? AND slug = ?",
            (owner_id, slug),
        ).fetchone()
        if not row:
            raise BeingNotFound(f"no being {slug!r}")
        return row

    def get(self, owner_id: str, slug: str) -> dict:
        b = dict(self._row(owner_id, slug))
        b["genome"] = json.loads(b["genome"])
        b["drives"] = json.loads(b["drives"])
        b["lineage"] = json.loads(b["lineage"])
        return b

    def list(self, owner_id: str) -> list[dict]:
        rows = self._c().execute(
            "SELECT b.slug, b.name, b.stage, b.state, b.born_at, b.hatched_at,"
            " b.died_at, w.balance_tokens, w.allowance_preset"
            " FROM beings b LEFT JOIN being_wallets w ON w.being_id = b.id"
            " WHERE b.owner_id = ? ORDER BY b.born_at",
            (owner_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _update(self, being_id: str, now: datetime, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._c().execute(
                f"UPDATE beings SET {cols}, updated_at = ? WHERE id = ?",
                (*fields.values(), _iso(now), being_id),
            )
            self._c().commit()

    # ── Lifecycle ────────────────────────────────────────────────────

    def hatch(self, owner_id: str, slug: str, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["stage"] != "egg":
            raise BeingError("already hatched")
        self._update(b["id"], now, stage="infant", hatched_at=_iso(now))
        self.record_event(b["id"], "hatched", {}, now=now)
        self.credit_allowance(b["id"], now=now)
        return self.get(owner_id, slug)

    def set_stage(self, owner_id: str, slug: str, stage: str,
                  now: datetime | None = None) -> dict:
        now = now or _utcnow()
        if stage not in constitution.STAGE_ORDER:
            raise BeingError(f"unknown stage {stage!r}")
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("cannot change stage of a dead being")
        if stage == "egg":
            raise BeingError("cannot return to the egg")
        self._update(b["id"], now, stage=stage)
        self.record_event(b["id"], "stage", {"from": b["stage"], "to": stage}, now=now)
        return self.get(owner_id, slug)

    def set_state(self, owner_id: str, slug: str, state: str,
                  now: datetime | None = None) -> dict:
        now = now or _utcnow()
        if state not in STATES:
            raise BeingError(f"unknown state {state!r}")
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("a dead being stays dead")
        fields: dict = {"state": state}
        if state == "dead":
            fields["died_at"] = _iso(now)
        self._update(b["id"], now, **fields)
        self.record_event(b["id"], "state", {"from": b["state"], "to": state}, now=now)
        return self.get(owner_id, slug)

    # ── Wallet (the only door tokens pass through) ───────────────────

    def wallet_view(self, being: dict) -> dict:
        """Effective wallet numbers after Constitution clamps."""
        w = self._c().execute(
            "SELECT * FROM being_wallets WHERE being_id = ?", (being["id"],)
        ).fetchone()
        if not w:
            raise BeingNotFound("wallet missing")
        w = dict(w)
        eff_preset = constitution.clamp_preset(being["stage"], w["allowance_preset"])
        per_day = constitution.ALLOWANCE_PRESETS[eff_preset]
        enforced = per_day is not None
        ceiling = w["savings_ceiling"]
        if ceiling is None:
            ceiling = constitution.savings_ceiling_tokens(
                being["stage"], w["allowance_preset"])
        burn_cap = w["daily_burn_cap"]
        if burn_cap is None:
            burn_cap = per_day
        return {
            "balance_tokens": w["balance_tokens"],
            "allowance_preset": w["allowance_preset"],
            "effective_preset": eff_preset,
            "per_day_tokens": per_day,
            "enforced": enforced,
            "savings_ceiling": ceiling,
            "daily_burn_cap": burn_cap,
            "reserve_tokens": w["reserve_tokens"],
        }

    def set_allowance(
        self, owner_id: str, slug: str, preset: str,
        daily_burn_cap: int | None = None, savings_ceiling: int | None = None,
        now: datetime | None = None,
    ) -> dict:
        now = now or _utcnow()
        if preset not in constitution.ALLOWANCE_PRESETS:
            raise BeingError(f"unknown allowance preset {preset!r}")
        b = self.get(owner_id, slug)
        with self._lock:
            self._c().execute(
                "UPDATE being_wallets SET allowance_preset = ?, daily_burn_cap = ?,"
                " savings_ceiling = ?, updated_at = ? WHERE being_id = ?",
                (preset, daily_burn_cap, savings_ceiling, _iso(now), b["id"]),
            )
            self._c().commit()
        return self.wallet_view(self.get(owner_id, slug))

    def _apply(
        self, owner_id: str, *, tokens: int, reason: str,
        from_being: str | None, to_being: str | None,
        job_id: str | None = None, note: str | None = None,
        now: datetime,
    ) -> None:
        """The single balance-mutation path: ledger row + balance updates,
        one transaction. Callers validate; this conserves."""
        if tokens <= 0:
            raise BeingError("tokens must be positive")
        if reason not in TRANSFER_REASONS:
            raise BeingError(f"unknown transfer reason {reason!r}")
        with self._lock:
            c = self._c()
            c.execute(
                "INSERT INTO token_transfers"
                " (id, owner_id, from_being, to_being, tokens, reason, job_id, note, at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, owner_id, from_being, to_being, tokens,
                 reason, job_id, note, _iso(now)),
            )
            if from_being:
                c.execute(
                    "UPDATE being_wallets SET balance_tokens = balance_tokens - ?,"
                    " updated_at = ? WHERE being_id = ?",
                    (tokens, _iso(now), from_being),
                )
            if to_being:
                c.execute(
                    "UPDATE being_wallets SET balance_tokens = balance_tokens + ?,"
                    " updated_at = ? WHERE being_id = ?",
                    (tokens, _iso(now), to_being),
                )
            c.commit()

    def _being_by_id(self, being_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM beings WHERE id = ?", (being_id,)
        ).fetchone()
        if not row:
            raise BeingNotFound("no such being")
        b = dict(row)
        b["genome"] = json.loads(b["genome"])
        return b

    def credit_allowance(self, being_id: str, now: datetime | None = None) -> int:
        """Daily allowance mint — idempotent per (being, date); clipped at the
        piggy-bank ceiling. Returns tokens actually credited."""
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        if b["state"] == "dead" or b["stage"] == "egg":
            return 0
        view = self.wallet_view(b)
        if not view["enforced"]:
            return 0
        date_key = _iso(now)[:10]
        exists = self._c().execute(
            "SELECT 1 FROM token_transfers WHERE to_being = ? AND reason = 'allowance'"
            " AND note = ? LIMIT 1",
            (being_id, date_key),
        ).fetchone()
        if exists:
            return 0
        headroom = view["per_day_tokens"]
        if view["savings_ceiling"] is not None:
            headroom = min(headroom,
                           max(0, view["savings_ceiling"] - view["balance_tokens"]))
        if headroom <= 0:
            return 0
        self._apply(b["owner_id"], tokens=headroom, reason="allowance",
                    from_being=None, to_being=being_id, note=date_key, now=now)
        return headroom

    def spent_today(self, being_id: str, now: datetime | None = None) -> int:
        now = now or _utcnow()
        row = self._c().execute(
            "SELECT COALESCE(SUM(tokens), 0) AS s FROM token_transfers"
            " WHERE from_being = ? AND reason = 'usage' AND at LIKE ?",
            (being_id, _iso(now)[:10] + "%"),
        ).fetchone()
        return int(row["s"])

    def debit_usage(
        self, being_id: str, tier: str, usage: dict,
        note: str | None = None, now: datetime | None = None,
    ) -> int:
        """Meter one model call against the wallet (Constitution invariant 1).

        Weighted (tier × cache-aware) tokens leave the economy as a sink row.
        Hard-stops on empty wallet or the daily burn cap — the caller decides
        whether that means torpor. Unlimited wallets return 0 unmetered.
        """
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        if b["state"] == "dead":
            raise BeingError("a dead being spends nothing")
        view = self.wallet_view(b)
        if not view["enforced"]:
            return 0
        weighted = constitution.weighted_tokens(usage, tier)
        if weighted <= 0:
            return 0
        if view["balance_tokens"] < weighted:
            raise InsufficientTokens("wallet empty")
        if view["daily_burn_cap"] is not None:
            if self.spent_today(being_id, now=now) + weighted > view["daily_burn_cap"]:
                raise BurnCapExceeded("daily burn cap reached")
        self._apply(b["owner_id"], tokens=weighted, reason="usage",
                    from_being=being_id, to_being=None, note=note or tier, now=now)
        return weighted

    def transfer(
        self, owner_id: str, from_slug: str, to_slug: str, tokens: int,
        reason: str = "trade", job_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Being→being transfer; conserves supply, respects the receiver's
        piggy-bank headroom (no clipped-in-flight tokens)."""
        now = now or _utcnow()
        if reason not in ("trade", "gift", "fee", "procreation"):
            raise BeingError(f"invalid transfer reason {reason!r}")
        src = self.get(owner_id, from_slug)
        dst = self.get(owner_id, to_slug)
        if src["state"] == "dead" or dst["state"] == "dead":
            raise BeingError("the dead neither give nor receive")
        src_view = self.wallet_view(src)
        dst_view = self.wallet_view(dst)
        if src_view["enforced"] and src_view["balance_tokens"] < tokens:
            raise InsufficientTokens("sender cannot afford this")
        if dst_view["savings_ceiling"] is not None:
            if dst_view["balance_tokens"] + tokens > dst_view["savings_ceiling"]:
                raise BeingError("receiver's piggy bank is full")
        self._apply(owner_id, tokens=tokens, reason=reason,
                    from_being=src["id"], to_being=dst["id"],
                    job_id=job_id, now=now)

    def burn(self, being_id: str, tokens: int, reason: str = "metamorphosis_burn",
             note: str | None = None, now: datetime | None = None) -> None:
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        view = self.wallet_view(b)
        if view["enforced"] and view["balance_tokens"] < tokens:
            raise InsufficientTokens("cannot afford this")
        self._apply(b["owner_id"], tokens=tokens, reason=reason,
                    from_being=being_id, to_being=None, note=note, now=now)

    # ── Metamorphosis (§2.1.2 — the paid rite) ───────────────────────

    def metamorphose(
        self, owner_id: str, slug: str, from_attr: str, to_attr: str,
        reason: str, now: datetime | None = None,
    ) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only a living being can change")
        policy = constitution.metamorphosis_policy(b["stage"])
        if policy == "none":
            raise BeingError(f"a {b['stage']} cannot metamorphose yet")
        plan = genome_mod.plan_metamorphosis(b["genome"], from_attr, to_attr, now)
        if not plan["ok"]:
            raise BeingError("; ".join(plan["errors"]))
        self.burn(b["id"], plan["price"], reason="metamorphosis_burn",
                  note=f"{from_attr}->{to_attr}", now=now)
        new_genome = genome_mod.apply_metamorphosis(
            b["genome"], from_attr, to_attr, reason, plan["price"], now)
        self._update(b["id"], now, genome=json.dumps(new_genome))
        self.record_event(b["id"], "metamorphosis", {
            "from": from_attr, "to": to_attr, "reason": reason,
            "price_tokens": plan["price"], "policy": policy,
        }, now=now)
        return self.get(owner_id, slug)

    # ── Views, events, audits ────────────────────────────────────────

    def vitals(self, owner_id: str, slug: str) -> dict:
        b = self.get(owner_id, slug)
        attrs = genome_mod.effective_attributes(b["genome"])
        return {
            "slug": b["slug"], "name": b["name"], "stage": b["stage"],
            "state": b["state"], "born_at": b["born_at"],
            "hatched_at": b["hatched_at"], "died_at": b["died_at"],
            "attention_credits": b["attention_credits"],
            "attributes": attrs,
            "derived": genome_mod.derive(attrs),
            "generation": b["genome"].get("generation", 1),
            "lineage": b["genome"].get("lineage", []),
            "metamorphoses": b["genome"].get("metamorphoses", []),
            "interest_seeds": b["genome"].get("interest_seeds", []),
            "wallet": self.wallet_view(b),
            "spent_today": self.spent_today(b["id"]),
            "capabilities": sorted(constitution.capabilities(b["stage"])),
        }

    def ledger(self, owner_id: str, slug: str, limit: int = 100) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT * FROM token_transfers WHERE from_being = ? OR to_being = ?"
            " ORDER BY at DESC LIMIT ?",
            (b["id"], b["id"], limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def liabilities(self, owner_id: str) -> dict:
        """Outstanding token liabilities — deferred real dollars (plan §5.1)."""
        rows = self._c().execute(
            "SELECT b.slug, b.name, b.state, w.balance_tokens"
            " FROM beings b JOIN being_wallets w ON w.being_id = b.id"
            " WHERE b.owner_id = ? AND b.state != 'dead'",
            (owner_id,),
        ).fetchall()
        per = [dict(r) for r in rows]
        return {"total_tokens": sum(r["balance_tokens"] for r in per), "beings": per}

    def conservation(self, owner_id: str) -> dict:
        """Audit: sum(balances) must equal mints − sinks. ok=False is a bug."""
        c = self._c()
        mints = c.execute(
            "SELECT COALESCE(SUM(tokens),0) AS s FROM token_transfers"
            " WHERE owner_id = ? AND from_being IS NULL", (owner_id,),
        ).fetchone()["s"]
        sinks = c.execute(
            "SELECT COALESCE(SUM(tokens),0) AS s FROM token_transfers"
            " WHERE owner_id = ? AND to_being IS NULL", (owner_id,),
        ).fetchone()["s"]
        balances = c.execute(
            "SELECT COALESCE(SUM(w.balance_tokens),0) AS s FROM beings b"
            " JOIN being_wallets w ON w.being_id = b.id WHERE b.owner_id = ?",
            (owner_id,),
        ).fetchone()["s"]
        return {"mints": mints, "sinks": sinks, "balances": balances,
                "ok": balances == mints - sinks}

    def record_event(self, being_id: str, kind: str, data: dict,
                     now: datetime | None = None) -> None:
        now = now or _utcnow()
        with self._lock:
            self._c().execute(
                "INSERT INTO being_events (id, being_id, kind, data, at)"
                " VALUES (?,?,?,?,?)",
                (uuid.uuid4().hex, being_id, kind, json.dumps(data), _iso(now)),
            )
            self._c().commit()

    def events(self, owner_id: str, slug: str, limit: int = 100) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT kind, data, at FROM being_events WHERE being_id = ?"
            " ORDER BY at DESC LIMIT ?",
            (b["id"], limit),
        ).fetchall()
        return [{"kind": r["kind"], "data": json.loads(r["data"]), "at": r["at"]}
                for r in rows]


_STORE: BeingsStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> BeingsStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = BeingsStore()
        return _STORE
