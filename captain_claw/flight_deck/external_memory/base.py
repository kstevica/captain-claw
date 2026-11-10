"""What every external-memory provider is, whichever service it talks to.

A provider is one external memory service (Membase first). Flight Deck holds
the user's credentials for it and is the only thing that ever talks to it; the
user's agents reach it through ``/fd/external-memory/agent/*`` and share it.

Providers differ in what they can do and in the rules an agent must follow
with them, so neither is fixed here:

* :meth:`ExternalMemoryProvider.actions` — the actions THIS connection offers
  (its own names, descriptions and parameters). The agent tool builds its
  schema from them, so a provider with a ``profile`` or ``ask`` verb gets one
  and a provider without doesn't. Two names mean the same everywhere:
  ``remember`` (save one fact) and ``search`` — every provider has both.
* :meth:`ExternalMemoryProvider.rules` — the rules the agent is told: the
  provider's own (cite the source, confirm before forgetting, …) plus any the
  person set in the service itself.
* :meth:`ExternalMemoryProvider.mirror` — how one item Captain keeps
  (an insight, an intention, a reflection, a lesson, an intuition) is written
  into the service — each provider picks its own verb and format.

What a connection can do is learned once, when its credentials are checked
(:meth:`verify` → :class:`ConnectionInfo`), and kept with the connection, so
listing actions and rules never needs the network.

Providers also differ in how they keep people apart (:attr:`tenancy`):

* ``"key"`` — one credential IS one person's memory (Membase: every end user
  has their own account and key). Each person connects their own.
* ``"scoped"`` — one credential serves many people, and every call names whose
  memory it is (Hindsight: a bank per person). An admin can connect one for the
  whole deck; Flight Deck then gives each user a scope of their own
  (:attr:`ExternalMemoryProvider.scope`) and the provider puts it in every call.
  The agent never chooses it — no action takes a scope.

Some services remember what they are given as it is (Qdrant, ArcadeDB — a
database, not an extractor); for those, :func:`mirror_body` keeps the text
about the person's world and the rest goes in fields beside it, and
:func:`mirror_document_id` / :func:`note_document_id` key what is saved so that
saving the same thing again replaces it instead of piling up.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlsplit

TENANCIES = ("key", "scoped")
_KIND_LABEL = {
    "insight": "Insight",
    "intention": "Intention",
    "reflection": "Reflection",
    "lesson": "Lesson",
    "intuition": "Intuition",
}

# What Captain mirrors into an external memory on its own. ``lesson`` is
# written by Flight Deck (lessons are approved there); the rest come from the
# agents as they store them.
MIRROR_KINDS: tuple[str, ...] = ("insight", "intention", "reflection", "lesson", "intuition")
# What the person's memory gets unless they change it: what is ABOUT them
# (insights, their intentions, the lessons they taught) is on; what is about
# the agent itself (reflections: its notes on how to work better; intuitions:
# its unchecked hunches) is off until they ask for it.
DEFAULT_MIRROR: dict[str, bool] = {
    "insight": True, "intention": True, "lesson": True, "reflection": False, "intuition": False,
}

PARAM_TYPES = frozenset({"string", "integer", "boolean"})

# Every provider: the rules about the memory itself, before the provider's own.
COMMON_RULES: tuple[str, ...] = (
    "This is the person's own external memory, shared by ALL of their agents — what you save here, "
    "their other agents can find, and what they saved, you can.",
    "Save what stays useful beyond this conversation: facts about the person and their work, "
    "decisions, preferences, commitments, contacts. One fact per call, written so it reads on its "
    "own (no 'he', 'this', 'the above').",
    "Never save secrets: passwords, API keys, tokens, card or bank numbers, ID numbers.",
    "Search it before saying you don't know something about the person or their work.",
)


class ProviderError(RuntimeError):
    """The service refused or failed. ``retryable`` — trying again later may
    work (rate limit, outage, timeout); ``status`` — its HTTP status, 0 when
    no answer came."""

    def __init__(
        self, message: str, *, status: int = 0, code: str = "", retryable: bool = False,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.details = details


class ProviderAuthError(ProviderError):
    """The credentials were refused (unknown, expired or revoked)."""


class ProviderInputError(ValueError):
    """The call itself is wrong (missing or bad argument) — nothing was sent."""


class UnknownAction(ProviderInputError):  # noqa: N818 — reads as the condition
    """An action this connection doesn't offer."""


@dataclass(frozen=True)
class CredentialField:
    """One thing the user pastes to connect (an API key, a server address)."""

    name: str
    label: str
    secret: bool = True
    required: bool = True
    placeholder: str = ""
    help: str = ""
    # Shown folded under "More settings" — what most people leave as it is.
    advanced: bool = False


@dataclass(frozen=True)
class ActionParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    enum: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError(f"param {self.name}: type must be one of {sorted(PARAM_TYPES)}")


@dataclass(frozen=True)
class ProviderAction:
    """One thing an agent can ask this connection to do."""

    name: str
    description: str
    params: tuple[ActionParam, ...] = ()
    writes: bool = False
    # Removes something: only after the person agreed, with confirm=true.
    destructive: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["params"] = [
            {k: (list(v) if k == "enum" else v) for k, v in asdict(p).items() if not (k == "enum" and not v)}
            for p in self.params
        ]
        return out


@dataclass
class ConnectionInfo:
    """What a checked connection reaches and may do, kept with it (JSON)."""

    # Provider-side names of what the credential may do (Membase: ``search``,
    # ``memories.add``, …). Empty = unknown, offer everything.
    allowed: list[str] = field(default_factory=list)
    # Human-readable: the memories / spaces / collections it reaches.
    reaches: list[str] = field(default_factory=list)
    # The person's own standing rules from the service, already worded.
    rules: list[str] = field(default_factory=list)
    # Anything else a provider wants to keep (raw capability flags…).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> ConnectionInfo:
        raw = raw if isinstance(raw, dict) else {}

        def _strs(v: Any) -> list[str]:
            return [str(x) for x in v if isinstance(x, (str, int, float))] if isinstance(v, list) else []

        extra = raw.get("extra")
        return cls(
            allowed=_strs(raw.get("allowed")),
            reaches=_strs(raw.get("reaches")),
            rules=_strs(raw.get("rules")),
            extra=extra if isinstance(extra, dict) else {},
        )


@dataclass
class MemoryItem:
    """One item Captain mirrors, provider-neutral."""

    kind: str               # one of MIRROR_KINDS
    source_id: str          # its id where Captain keeps it (unique per kind)
    content: str
    title: str = ""
    tags: list[str] = field(default_factory=list)
    agent: str = ""         # the agent's name, for attribution
    created_at: str = ""    # ISO 8601
    meta: dict[str, Any] = field(default_factory=dict)  # category, why, how_to_apply, …

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MemoryItem:
        tags = raw.get("tags")
        meta = raw.get("meta")
        return cls(
            kind=str(raw.get("kind") or ""),
            source_id=str(raw.get("source_id") or ""),
            content=str(raw.get("content") or ""),
            title=str(raw.get("title") or ""),
            tags=[str(t) for t in tags if str(t).strip()] if isinstance(tags, list) else [],
            agent=str(raw.get("agent") or ""),
            created_at=str(raw.get("created_at") or ""),
            meta=meta if isinstance(meta, dict) else {},
        )


@dataclass
class RecallHit:
    """One thing the memory holds that bears on a message — what automatic
    recall puts in front of the agent."""

    text: str
    id: str = ""
    when: str = ""   # the date it holds for (or was saved), YYYY-MM-DD
    kind: str = ""   # insight, note, world, … — the provider's own word

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionResult:
    """``text`` is what the agent reads; ``data`` the structured answer (the
    Flight Deck UI and tests use it)."""

    text: str
    data: dict[str, Any] = field(default_factory=dict)


class ExternalMemoryProvider(ABC):
    """One external memory service. Instances are cheap and short-lived: one
    per call, built from the stored credentials."""

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str] = ""
    docs_url: ClassVar[str] = ""
    # Where the user gets the credentials.
    credentials_url: ClassVar[str] = ""
    credential_fields: ClassVar[tuple[CredentialField, ...]] = ()
    # "key": a credential is one person's memory; "scoped": one credential
    # serves many people, each call scoped to one (see the module docstring).
    tenancy: ClassVar[str] = "key"
    # The credential field holding the server's address, if the provider has
    # one (checked against private addresses for users who aren't admins).
    url_field: ClassVar[str] = ""
    # Other credential fields holding an address Flight Deck calls (an
    # embeddings endpoint…) — checked the same way.
    extra_url_fields: ClassVar[tuple[str, ...]] = ()
    # What one person's scope is called in the UI ("bank").
    scope_label: ClassVar[str] = ""
    # The credential field where a person's OWN connection may name its scope
    # ("bank"); the deck's connection never keeps one (each person gets theirs).
    scope_field: ClassVar[str] = ""

    def __init__(self, credentials: dict[str, str], info: ConnectionInfo | None = None,
                 *, scope: str = "") -> None:
        self.credentials = {k: str(v) for k, v in (credentials or {}).items() if v is not None}
        self.info = info or ConnectionInfo()
        self.scope = scope

    # ── catalog ─────────────────────────────────────────────────────

    @classmethod
    def catalog_entry(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "label": cls.label,
            "description": cls.description,
            "docs_url": cls.docs_url,
            "credentials_url": cls.credentials_url,
            "credential_fields": [asdict(f) for f in cls.credential_fields],
            "tenancy": cls.tenancy,
            "scope_label": cls.scope_label,
            "scope_field": cls.scope_field,
        }

    @classmethod
    def address_fields(cls) -> tuple[str, ...]:
        """Every credential field holding an address Flight Deck will call."""
        return tuple(f for f in (cls.url_field, *cls.extra_url_fields) if f)

    @classmethod
    def shareable(cls) -> bool:
        """Whether an admin can connect it once for the whole deck."""
        return cls.tenancy == "scoped"

    @classmethod
    def clean_credentials(cls, raw: dict[str, Any]) -> dict[str, str]:
        """The credentials from a form, checked: every required field there,
        nothing a header can't carry. Raises :class:`ProviderInputError`."""
        out: dict[str, str] = {}
        for f in cls.credential_fields:
            value = raw.get(f.name) if isinstance(raw, dict) else None
            value = value.strip() if isinstance(value, str) else ""
            if not value:
                if f.required:
                    raise ProviderInputError(f"Paste the {f.label}")
                continue
            if not (value.isascii() and value.isprintable()) or any(c.isspace() for c in value):
                raise ProviderInputError(
                    f"That {f.label} has characters it can't have (hidden spaces or curly quotes "
                    "from copying) — copy it again. Nothing was saved."
                )
            out[f.name] = value
        return out

    @staticmethod
    def hint(secret: str) -> str:
        """A secret as the UI may show it: its start and its last four."""
        secret = secret or ""
        if len(secret) <= 8:
            return "…" if secret else ""
        return f"{secret[:4]}…{secret[-4:]}"

    def credential_hints(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in self.credential_fields:
            value = self.credentials.get(f.name, "")
            if value:
                out[f.name] = self.hint(value) if f.secret else value
        return out

    # ── the connection ──────────────────────────────────────────────

    @abstractmethod
    async def verify(self) -> ConnectionInfo:
        """Check the credentials with the service and learn what they reach.
        Raises :class:`ProviderAuthError` when refused, :class:`ProviderError`
        when the service can't be asked."""

    @abstractmethod
    def actions(self) -> list[ProviderAction]:
        """The actions this connection offers (from :attr:`info`). Must
        include ``remember`` and ``search`` whenever the credential allows."""

    def rules(self) -> list[str]:
        """Everything the agent is told to follow with this connection."""
        return [*COMMON_RULES, *self.provider_rules(), *self.info.rules]

    def reaches(self) -> list[str]:
        """What this connection reaches, as the agent and the UI are told."""
        return list(self.info.reaches)

    def provider_rules(self) -> list[str]:
        return []

    @abstractmethod
    async def run(self, action: str, args: dict[str, Any]) -> ActionResult:
        """Do one action. Raises :class:`UnknownAction` /
        :class:`ProviderInputError` before anything is sent, and
        :class:`ProviderError` when the service refuses or fails."""

    @abstractmethod
    async def mirror(self, item: MemoryItem) -> str:
        """Write one mirrored item; returns the service's id for it ("" when
        it gave none)."""

    @abstractmethod
    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        """What the memory holds that bears on *query*, most relevant first —
        for automatic recall before a reply, so ONLY what is relevant (often
        nothing), and fast (Flight Deck gives it a few seconds). Raises
        :class:`ProviderError` like :meth:`run`."""

    # ── helpers for subclasses ──────────────────────────────────────

    def action(self, name: str) -> ProviderAction:
        for a in self.actions():
            if a.name == name:
                return a
        offered = ", ".join(a.name for a in self.actions()) or "none"
        raise UnknownAction(f"This memory has no action '{name}' — it offers: {offered}.")

    def manifest(self) -> dict[str, Any]:
        """What an agent needs to use this connection: its actions and rules."""
        return {
            "provider": self.name,
            "label": self.label,
            "actions": [a.to_dict() for a in self.actions()],
            "rules": self.rules(),
            "reaches": self.reaches(),
        }


# ── shared helpers ──────────────────────────────────────────────────


def clip(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def mirror_text(item: MemoryItem, *, limit: int = 20_000) -> tuple[str, str]:
    """``(title, content)`` for one mirrored item, the same for every
    provider: the item, then what Captain knows about it, then who kept it."""
    label = _KIND_LABEL.get(item.kind, item.kind.title() or "Note")
    category = str(item.meta.get("category") or "").strip()
    title = label + (f" · {category}" if category else "") + (f": {item.title}" if item.title else "")
    lines = [item.content.strip()]
    for key, name in (("why", "Why"), ("how_to_apply", "How to apply"), ("body", "Details"),
                      ("status", "Status"), ("source", "Source")):
        value = str(item.meta.get(key) or "").strip()
        if value:
            lines.append(f"{name}: {value}")
    if item.tags:
        lines.append("Tags: " + ", ".join(item.tags))
    by = f"Captain agent “{item.agent}”" if item.agent else "Captain"
    when = f" on {item.created_at[:10]}" if item.created_at else ""
    lines.append(f"(Kept by {by}{when}.)")
    return clip(title, 200), clip("\n".join(lines), limit)


def mirror_body(item: MemoryItem, *, limit: int = 50_000) -> str:
    """Only what is about the person's world: the item and why it holds. Who
    kept it, its category and tags belong in fields beside it — measured on
    Hindsight, provenance lines in the text came back as facts ("Bob Bot is a
    Captain agent"), and in a vector store they blur what the text means."""
    lines = [item.content.strip()]
    for key in ("why", "how_to_apply", "body"):
        value = str(item.meta.get(key) or "").strip()
        if value:
            lines.append(value)
    return clip("\n".join(lines), limit)


def safe_document_id(raw: str) -> str:
    """A document id any service takes, in a URL path or a field."""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw).strip("-")[:200]


def mirror_document_id(item: MemoryItem) -> str:
    """What a mirrored item IS, not which copy: an insight that replaces
    another (same ``entity_key``) gets the same id, so it replaces it."""
    key = str(item.meta.get("entity_key") or "").strip() or item.source_id
    return safe_document_id(f"captain-{item.kind}-{key}")


def note_document_id(content: str) -> str:
    """A fact saved by hand, keyed by its text: saved twice, it replaces itself."""
    digest = hashlib.sha1(" ".join(content.lower().split()).encode()).hexdigest()[:20]
    return f"captain-note-{digest}"


def clean_topics(*sources: Any, limit: int = 12) -> list[str]:
    """Topics (people, projects, places…) from comma-separated text or lists:
    lower-case, trimmed, no repeats, at most *limit*."""
    out: list[str] = []
    for src in sources:
        parts = src if isinstance(src, (list, tuple)) else str(src or "").split(",")
        for part in parts:
            topic = " ".join(str(part or "").strip().lower().split())[:60].strip()
            if topic and topic not in out:
                out.append(topic)
            if len(out) >= limit:
                return out
    return out


def as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "y")


def iso_utc(value: Any = "") -> str:
    """A time as ``YYYY-MM-DDTHH:MM:SSZ`` (UTC) — sortable as text, and what
    datetime indexes take; now when *value* is empty or unreadable."""
    raw = str(value or "").strip()
    if raw:
        try:
            dt = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def rrf(*rankings: list[str], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ranked id lists, best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def scope_id(owner: str, prefix: str = "captain-") -> str:
    """A person's scope (bank, namespace…) on a shared connection: their Flight
    Deck id, in the characters any service takes."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", str(owner or "")).strip("-") or "local"
    return (prefix + safe)[:64]


def _private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified)


def private_address(url: str) -> bool:
    """Whether *url* points into this machine or its network (by name or by any
    address the name resolves to). An unresolvable name counts as private:
    nothing can be checked about it."""
    host = (urlsplit(url).hostname or "").strip("[]").lower()
    if not host:
        return True
    if host in ("localhost", "host.docker.internal") or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        return _private_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for info in infos:
        try:
            if _private_ip(ipaddress.ip_address(info[4][0].split("%")[0])):
                return True
        except ValueError:
            return True
    return False
