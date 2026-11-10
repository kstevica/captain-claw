"""Membase (membase.com) as an external memory.

Membase keeps a person's memory in several *Memories* (their own topics, not
tenants). A developer key (``mbk_…``, minted in the Membase app under
Connect › Developer keys) has an access level (Read, Read & write, Full
access), the Memories it reaches, an optional profile tick, and one *write
target* where its writes land. No request ever names a Memory: reads span
everything the key reaches, each hit naming its ``memory_name``.

REST, ``Authorization: Bearer mbk_…`` (https://docs.membase.com/build/reference/):

    GET  /v1/capabilities                    what the key may do, what it reaches, the person's rules
    POST /v1/search            {q, limit, mode}
    POST /v1/ask               {message}     only for credentials that hold it
    GET  /v1/memories?limit=                 newest learned facts (mem_)
    GET  /v1/memories/{id}
    POST /v1/memories/add      {content, title}         → a note doc_ (learned in the background)
    POST /v1/memories/{id}/forget?confirm=true
    GET  /v1/profile-facts?limit=            standing statements about the person (pf_)
    POST /v1/profile-facts/add {content}
    POST /v1/profile-facts/{id}/forget?confirm=true
    GET  /v1/documents?kind=&limit=          files and notes (doc_)
    GET  /v1/documents/{id}
    POST /v1/documents/add     {content|url, title, custom_id}   same custom_id again = no-op
    POST /v1/documents/{id}/delete?confirm=true
    GET  /v1/media/{id}                      photos and videos (med_)
    POST /v1/media/search      {q, kind, limit}
    POST /v1/media/{id}/delete?confirm=true

Errors carry ``{"error": {code, message, details, retryable, trace_id}}``.
A removal without ``confirm=true`` answers 428 ``confirmation_required`` with
a ``how`` sentence and changes nothing. 429 has no ``Retry-After``; with
``code: quota_exceeded`` (storage full, plan inactive) retrying won't help.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from captain_claw.flight_deck.external_memory.base import (
    ActionParam,
    ActionResult,
    ConnectionInfo,
    CredentialField,
    ExternalMemoryProvider,
    MemoryItem,
    ProviderAction,
    ProviderAuthError,
    ProviderError,
    ProviderInputError,
    RecallHit,
    mirror_text,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.app.membase.com"
# The first search after a quiet spell can take up to a minute.
TIMEOUT_S = 90.0
MAX_CONTENT = 20_000
MAX_TITLE = 200
MAX_PROFILE_FACT = 2_000
MAX_QUESTION = 4_000

# ID prefix → (collection, the verb that removes it, the action name Membase lists).
_FORGET = {
    "mem_": ("memories", "forget", "memories.forget"),
    "doc_": ("documents", "delete", "documents.delete"),
    "pf_": ("profile-facts", "forget", "profile_facts.forget"),
    "med_": ("media", "delete", "media.delete"),
}
_GET = {
    "mem_": ("memories", "memories.get"),
    "doc_": ("documents", "documents.get"),
    "med_": ("media", "media.get"),
}


def _clip(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "y")


def _items(data: Any) -> list[dict[str, Any]]:
    items = data.get("items") if isinstance(data, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _partial_note(data: Any) -> str:
    partial = data.get("partial") if isinstance(data, dict) else None
    if not isinstance(partial, list) or not partial:
        return ""
    names = []
    for p in partial:
        if isinstance(p, dict):
            names.append(str(p.get("memory_name") or "?"))
        elif isinstance(p, str):
            names.append(p)
    if not names:
        return ""
    return (
        "\nNot every Memory answered (" + ", ".join(names) + ") — the answer may be incomplete; "
        "say so rather than concluding nothing is known."
    )


def _where(item: dict[str, Any]) -> str:
    name = str(item.get("memory_name") or "").strip()
    return f"[{name}] " if name else ""


class MembaseProvider(ExternalMemoryProvider):
    name = "membase"
    label = "Membase"
    description = (
        "A personal memory that learns facts from what it is given and finds them again — "
        "shared by all of your agents."
    )
    docs_url = "https://docs.membase.com/build/reference/sdk-quickstart/"
    credentials_url = "https://app.membase.com"
    credential_fields = (
        CredentialField(
            name="api_key",
            label="Membase developer key",
            placeholder="mbk_…",
            help=(
                "Membase app → Connect › Developer keys. Read & write lets agents save; Full access "
                "also lets them forget (only when you agree). Tick “Your profile” to share profile facts."
            ),
        ),
        CredentialField(
            name="base_url",
            label="Server address",
            secret=False,
            required=False,
            placeholder=DEFAULT_BASE_URL,
            help="Leave empty for membase.com. A self-hosted Membase (membase --local serve) is http://127.0.0.1:8787.",
        ),
    )

    tenancy = "key"
    url_field = "base_url"

    def __init__(self, credentials: dict[str, str], info: ConnectionInfo | None = None,
                 *, scope: str = "", transport: httpx.AsyncBaseTransport | None = None,
                 max_retries: int = 2) -> None:
        super().__init__(credentials, info, scope=scope)
        self._transport = transport
        self.max_retries = max(0, int(max_retries))

    # ── transport ───────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        url = (
            self.credentials.get("base_url")
            or os.environ.get("FD_MEMBASE_BASE_URL", "")
            or DEFAULT_BASE_URL
        )
        return url.strip().rstrip("/")

    async def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None, verifying: bool = False,
    ) -> Any:
        key = self.credentials.get("api_key", "")
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "captain-flight-deck",
        }
        attempt = 0
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=TIMEOUT_S, transport=self._transport,
        ) as client:
            while True:
                try:
                    resp = await client.request(method, path, json=json, params=params, headers=headers)
                except httpx.TimeoutException:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError(
                        "Membase didn't answer in time (the first search after a quiet spell can take "
                        "a minute) — try again shortly.", retryable=True,
                    ) from None
                except httpx.HTTPError as exc:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError(
                        f"Could not reach Membase at {self.base_url}: {type(exc).__name__}", retryable=True,
                    ) from None
                if 200 <= resp.status_code < 300:
                    try:
                        return resp.json() if resp.content else {}
                    except ValueError:
                        raise ProviderError("Membase answered with something that isn't JSON",
                                            status=resp.status_code) from None
                if (resp.status_code in (408, 409) or resp.status_code >= 500) and attempt < self.max_retries:
                    attempt += 1
                    continue
                raise self._error(resp, verifying=verifying)

    @staticmethod
    def _error(resp: httpx.Response, *, verifying: bool) -> ProviderError:
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}
        err = body.get("error") if isinstance(body, dict) else None
        err = err if isinstance(err, dict) else {}
        code = str(err.get("code") or "")
        message = str(err.get("message") or (body.get("detail") if isinstance(body, dict) else "") or "")
        details = err.get("details")
        if status == 401 or (status == 403 and verifying):
            return ProviderAuthError(
                f"Membase refused the key (HTTP {status}) — it is unknown, expired or revoked. "
                "Mint a new one under Connect › Developer keys.",
                status=status, code=code or "unauthorized",
            )
        if status == 403:
            return ProviderError(
                "Membase refused this: the key's access level doesn't allow it"
                + (f" ({message})" if message else "")
                + ". The person can raise it under Connect › Developer keys.",
                status=status, code=code,
            )
        if status == 428:
            how = ""
            if isinstance(details, dict):
                how = str(details.get("how") or "")
            return ProviderError(
                (how or message or "Membase needs confirmation for this")
                + " Nothing changed. Ask the person; only if they agree, call again with confirm=true.",
                status=status, code=code or "confirmation_required", details=details,
            )
        if status == 429:
            if code == "quota_exceeded":
                return ProviderError(
                    "Membase is full or the plan is inactive (quota exceeded) — nothing was saved. "
                    "The person needs to free space or renew the plan.",
                    status=status, code=code,
                )
            return ProviderError(
                "Membase is busy (rate limit) — try again in a minute.", status=status, code=code or "rate_limited",
                retryable=True,
            )
        if status == 422 and code == "write_target_required":
            return ProviderError(
                "Membase doesn't know where this key's writes go: it reaches several Memories and has no "
                "write target. The person sets one for the key under Connect › Developer keys. Nothing was saved.",
                status=status, code=code,
            )
        if status == 404:
            return ProviderError(
                "Membase has nothing by that id this key may read (unknown, deleted or out of reach).",
                status=status, code=code or "not_found",
            )
        retryable = status >= 500 or bool(err.get("retryable"))
        text = message or f"HTTP {status}"
        trace = str(err.get("trace_id") or "")
        return ProviderError(
            f"Membase: {text}" + (f" (trace {trace})" if trace else ""),
            status=status, code=code, retryable=retryable, details=details,
        )

    # ── the connection ──────────────────────────────────────────────

    async def capabilities(self, *, verifying: bool = False) -> ConnectionInfo:
        data = await self._request("GET", "/v1/capabilities", verifying=verifying)
        data = data if isinstance(data, dict) else {}
        actions = data.get("actions") if isinstance(data.get("actions"), list) else []
        memories = data.get("memories") if isinstance(data.get("memories"), list) else []
        raw_rules = data.get("rules") if isinstance(data.get("rules"), dict) else {}
        reaches = []
        for m in memories:
            if isinstance(m, dict) and m.get("name"):
                kind = str(m.get("kind") or "").strip()
                reaches.append(f"{m['name']}" + (f" ({kind})" if kind and kind != "text" else ""))
        rules: list[str] = []
        citation = bool(raw_rules.get("citation_required"))
        if citation:
            rules.append(
                "The person requires citations: whenever you use something from this memory, name the "
                "Memory it came from (its memory_name)."
            )
        notice = str(raw_rules.get("retention_notice") or "").strip()
        if notice:
            rules.append(f"The person's retention notice — follow it and relay it when it matters: {notice}")
        return ConnectionInfo(
            allowed=sorted(str(a) for a in actions if isinstance(a, str)),
            reaches=reaches,
            rules=rules,
            extra={
                "citation_required": citation,
                "retention_notice": notice,
                "rules_version": str(raw_rules.get("version") or ""),
                "features": [str(f) for f in data.get("features") or [] if isinstance(f, str)],
                "text_memories": sum(
                    1 for m in memories if isinstance(m, dict) and str(m.get("kind") or "text") == "text"
                ),
            },
        )

    async def verify(self) -> ConnectionInfo:
        return await self.capabilities(verifying=True)

    def _allows(self, *names: str) -> bool:
        if not self.info.allowed:
            return True
        return any(n in self.info.allowed for n in names)

    def actions(self) -> list[ProviderAction]:
        a: list[ProviderAction] = []
        if self._allows("memories.add"):
            a.append(ProviderAction(
                "remember",
                "Save one fact to the person's memory (learned in the background).",
                (
                    ActionParam("content", "string", "The fact, written to read on its own (max 20,000 characters).", True),
                    ActionParam("title", "string", "A short title (optional, max 200 characters)."),
                ),
                writes=True,
            ))
        if self._allows("search"):
            a.append(ProviderAction(
                "search",
                "Find what the person's memory holds on a question or topic, most relevant first.",
                (
                    ActionParam("query", "string", "What to look for, in plain words.", True),
                    ActionParam("limit", "integer", "How many passages (1-50, default 8)."),
                    ActionParam("fast", "boolean", "true = quick retrieval only, no model pass (faster, rougher)."),
                ),
            ))
        if self._allows("memories.list"):
            a.append(ProviderAction(
                "list",
                "The newest facts the memory learned.",
                (ActionParam("limit", "integer", "How many (1-100, default 20)."),),
            ))
        if self._allows(*(v[1] for v in _GET.values())):
            a.append(ProviderAction(
                "get",
                "One item by id: a fact (mem_…), a document (doc_…) or a photo/video (med_…).",
                (ActionParam("id", "string", "The id from search, list or documents.", True),),
            ))
        if self._allows("documents.add"):
            a.append(ProviderAction(
                "add_document",
                "Hand in a longer text, or a public web page by URL, for the memory to learn from.",
                (
                    ActionParam("content", "string", "The text (give content OR url)."),
                    ActionParam("url", "string", "A public web address to fetch (give content OR url)."),
                    ActionParam("title", "string", "A title (optional, max 200 characters)."),
                ),
                writes=True,
            ))
        if self._allows("documents.list"):
            a.append(ProviderAction(
                "documents",
                "The documents handed in, newest first: files/pages and the notes facts were saved as.",
                (
                    ActionParam("kind", "string", "file or note (default both).", enum=("file", "note")),
                    ActionParam("limit", "integer", "How many (1-100, default 20)."),
                ),
            ))
        if self._allows("profile_facts.list"):
            a.append(ProviderAction(
                "profile",
                "Standing statements about the person (name, role, lasting preferences).",
                (ActionParam("limit", "integer", "How many (1-100, default 50)."),),
            ))
        if self._allows("profile_facts.add"):
            a.append(ProviderAction(
                "add_profile_fact",
                "Keep one lasting statement about the person (who they are, a standing preference).",
                (ActionParam("content", "string", "The statement (max 2,000 characters).", True),),
                writes=True,
            ))
        if self._allows("media.search"):
            a.append(ProviderAction(
                "search_media",
                "Find the person's photos and videos by what is in them or said in them.",
                (
                    ActionParam("query", "string", "What to look for.", True),
                    ActionParam("limit", "integer", "How many (1-20, default 5)."),
                ),
            ))
        if self._allows("ask"):
            a.append(ProviderAction(
                "ask",
                "Get one written answer from the memory (with the Memories it came from, not the passages).",
                (ActionParam("question", "string", "The question.", True),),
            ))
        if self._allows(*(v[2] for v in _FORGET.values())):
            a.append(ProviderAction(
                "forget",
                "Remove one item for good: a fact (mem_…), document (doc_…), profile fact (pf_…) or "
                "photo/video (med_…). ONLY after the person clearly agreed — then pass confirm=true.",
                (
                    ActionParam("id", "string", "The id to remove.", True),
                    ActionParam("confirm", "boolean", "true only after the person agreed to remove it."),
                ),
                writes=True,
                destructive=True,
            ))
        a.append(ProviderAction(
            "capabilities",
            "What this connection reaches and may do right now, and the person's standing rules.",
        ))
        return a

    def provider_rules(self) -> list[str]:
        return [
            "Each search hit names the Memory it came from (memory_name); when you use one, say which "
            "Memory it came from.",
            "You never choose a Memory: searches cover every Memory this connection reaches, and saves "
            "land in its one write target.",
            "Saved facts are learned in the background — one you just saved may not show up in search "
            "for a little while. Don't save it again.",
            "The first search after a quiet spell can take up to a minute.",
            "forget removes things for good. Call it with confirm=true ONLY after the person clearly "
            "agreed; without confirm nothing changes and the answer says how to proceed — tell the person.",
        ]

    # ── actions ─────────────────────────────────────────────────────

    async def run(self, action: str, args: dict[str, Any]) -> ActionResult:
        action = (action or "").strip().lower()
        self.action(action)  # UnknownAction for one this connection doesn't offer
        args = args if isinstance(args, dict) else {}
        handler = getattr(self, f"_do_{action}")
        return await handler(args)

    async def _do_remember(self, args: dict[str, Any]) -> ActionResult:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ProviderInputError("remember needs content — the fact to save")
        if len(content) > MAX_CONTENT:
            raise ProviderInputError(
                f"That is {len(content):,} characters; one fact is at most {MAX_CONTENT:,}. "
                "Save the fact itself, or hand in the text with add_document."
            )
        data = await self._request("POST", "/v1/memories/add", json={
            "content": content, "title": _clip(args.get("title"), MAX_TITLE),
        })
        doc = str((data or {}).get("id") or "")
        status = str((data or {}).get("status") or "")
        return ActionResult(
            f"Saved to Membase{f' as {doc}' if doc else ''}"
            f"{f' ({status})' if status else ''}. It is learned in the background, so search may "
            "not find it for a little while.",
            data if isinstance(data, dict) else {},
        )

    async def _do_search(self, args: dict[str, Any]) -> ActionResult:
        query = str(args.get("query") or args.get("q") or "").strip()
        if not query:
            raise ProviderInputError("search needs a query")
        limit = _int(args.get("limit"), 8, 1, 50)
        data = await self._request("POST", "/v1/search", json={
            "q": query, "limit": limit, "mode": "fast" if _truthy(args.get("fast")) else "full",
        })
        items = _items(data)
        if not items:
            return ActionResult(
                f"Membase holds nothing on “{query}”." + _partial_note(data), data if isinstance(data, dict) else {},
            )
        lines = [f"Membase — {len(items)} passage(s) on “{query}”, most relevant first:"]
        for n, it in enumerate(items, 1):
            ref = f" ({it['id']})" if it.get("id") else ""
            title = f"{it['title']}: " if it.get("title") else ""
            lines.append(f"{n}. {_where(it)}{title}{_clip(it.get('content'), 1200)}{ref}")
        return ActionResult("\n".join(lines) + _partial_note(data), data)

    async def _do_list(self, args: dict[str, Any]) -> ActionResult:
        limit = _int(args.get("limit"), 20, 1, 100)
        data = await self._request("GET", "/v1/memories", params={"limit": limit})
        items = _items(data)
        if not items:
            return ActionResult("Membase has learned no facts yet." + _partial_note(data), data or {})
        lines = [f"Membase — the {len(items)} newest fact(s):"]
        for it in items:
            when = str(it.get("recorded_at") or "")[:10]
            lines.append(f"- {it.get('id', '')} {_where(it)}{_clip(it.get('content'), 400)}"
                         + (f" ({when})" if when else ""))
        if data.get("has_more"):
            lines.append("(there are more — search finds older ones)")
        return ActionResult("\n".join(lines) + _partial_note(data), data)

    async def _do_get(self, args: dict[str, Any]) -> ActionResult:
        item_id = str(args.get("id") or "").strip()
        prefix = next((p for p in _GET if item_id.startswith(p)), "")
        if not prefix:
            raise ProviderInputError("get needs an id that starts with mem_, doc_ or med_")
        collection, _name = _GET[prefix]
        data = await self._request("GET", f"/v1/{collection}/{quote(item_id, safe='')}")
        data = data if isinstance(data, dict) else {}
        lines = [f"{item_id} {_where(data)}".strip()]
        for key in ("title", "kind", "filename", "status", "recorded_at", "created_at"):
            if data.get(key):
                lines.append(f"{key}: {data[key]}")
        body = data.get("content") or data.get("summary") or ""
        if body:
            lines.append(_clip(body, 8000))
        learning = data.get("learning")
        if isinstance(learning, list) and learning:
            lines.append("learning: " + ", ".join(
                f"{x.get('memory_name', '?')}={x.get('status', '?')}" for x in learning if isinstance(x, dict)
            ))
        return ActionResult("\n".join(lines), data)

    async def _do_add_document(self, args: dict[str, Any]) -> ActionResult:
        content = str(args.get("content") or "").strip()
        url = str(args.get("url") or "").strip()
        if bool(content) == bool(url):
            raise ProviderInputError("add_document needs content OR url — exactly one of them")
        if url and not url.lower().startswith(("http://", "https://")):
            raise ProviderInputError("url must be a public http(s) address")
        body: dict[str, Any] = {"title": _clip(args.get("title"), MAX_TITLE), "custom_id": ""}
        if content:
            body["content"] = content
        else:
            body["url"] = url
        data = await self._request("POST", "/v1/documents/add", json=body)
        doc = str((data or {}).get("id") or "")
        return ActionResult(
            f"Handed in to Membase{f' as {doc}' if doc else ''} — it is learned in the background.",
            data if isinstance(data, dict) else {},
        )

    async def _do_documents(self, args: dict[str, Any]) -> ActionResult:
        params: dict[str, Any] = {"limit": _int(args.get("limit"), 20, 1, 100)}
        kind = str(args.get("kind") or "").strip().lower()
        if kind in ("file", "note"):
            params["kind"] = kind
        data = await self._request("GET", "/v1/documents", params=params)
        items = _items(data)
        if not items:
            return ActionResult("No documents in Membase yet." + _partial_note(data), data or {})
        lines = [f"Membase — {len(items)} document(s), newest first:"]
        for it in items:
            lines.append(f"- {it.get('id', '')} {_where(it)}{it.get('kind', '')}: "
                         f"{_clip(it.get('title') or '(untitled)', 160)}")
        if data.get("has_more"):
            lines.append("(there are more)")
        return ActionResult("\n".join(lines) + _partial_note(data), data)

    async def _do_profile(self, args: dict[str, Any]) -> ActionResult:
        data = await self._request("GET", "/v1/profile-facts", params={"limit": _int(args.get("limit"), 50, 1, 100)})
        items = _items(data)
        if not items:
            return ActionResult("Membase keeps no profile facts about the person yet.", data or {})
        lines = ["What Membase keeps about the person (newest first):"]
        for it in items:
            src = str(it.get("source_kind") or "")
            lines.append(f"- {it.get('id', '')} {_clip(it.get('content'), 400)}" + (f" ({src})" if src else ""))
        return ActionResult("\n".join(lines), data)

    async def _do_add_profile_fact(self, args: dict[str, Any]) -> ActionResult:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ProviderInputError("add_profile_fact needs content — the statement")
        if len(content) > MAX_PROFILE_FACT:
            raise ProviderInputError(f"A profile fact is at most {MAX_PROFILE_FACT:,} characters")
        data = await self._request("POST", "/v1/profile-facts/add", json={"content": content})
        data = data if isinstance(data, dict) else {}
        if data.get("created") is False:
            return ActionResult("Membase already keeps that about the person — nothing new saved.", data)
        fact_id = str(data.get("id") or "")
        return ActionResult(f"Kept in the person's Membase profile{f' as {fact_id}' if fact_id else ''}.", data)

    async def _do_search_media(self, args: dict[str, Any]) -> ActionResult:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ProviderInputError("search_media needs a query")
        data = await self._request("POST", "/v1/media/search", json={
            "q": query, "limit": _int(args.get("limit"), 5, 1, 20),
        })
        items = _items(data)
        if not items:
            return ActionResult(f"No photos or videos in Membase match “{query}”.", data or {})
        lines = [f"Membase — {len(items)} photo(s)/video(s) for “{query}”:"]
        for it in items:
            lines.append(f"- {it.get('id', '')} {_where(it)}{it.get('kind', '')} {it.get('filename') or ''}: "
                         f"{_clip(it.get('summary'), 300)}")
        return ActionResult("\n".join(lines), data)

    async def _do_ask(self, args: dict[str, Any]) -> ActionResult:
        question = str(args.get("question") or args.get("query") or "").strip()
        if not question:
            raise ProviderInputError("ask needs a question")
        data = await self._request("POST", "/v1/ask", json={"message": _clip(question, MAX_QUESTION)})
        data = data if isinstance(data, dict) else {}
        cites = [str(c.get("memory_name")) for c in data.get("citations") or [] if isinstance(c, dict)]
        text = str(data.get("answer") or "(no answer)")
        if cites:
            text += "\nFrom: " + ", ".join(cites)
        return ActionResult(text + _partial_note(data), data)

    async def _do_forget(self, args: dict[str, Any]) -> ActionResult:
        item_id = str(args.get("id") or "").strip()
        prefix = next((p for p in _FORGET if item_id.startswith(p)), "")
        if not prefix:
            raise ProviderInputError("forget needs an id that starts with mem_, doc_, pf_ or med_")
        collection, verb, _name = _FORGET[prefix]
        confirm = "true" if _truthy(args.get("confirm")) else "false"
        data = await self._request(
            "POST", f"/v1/{collection}/{quote(item_id, safe='')}/{verb}", params={"confirm": confirm},
        )
        done = "Forgotten" if verb == "forget" else "Deleted"
        return ActionResult(f"{done}: {item_id}.", data if isinstance(data, dict) else {})

    async def _do_capabilities(self, args: dict[str, Any]) -> ActionResult:
        info = await self.capabilities()
        self.info = info
        lines = ["Membase connection:"]
        lines.append("Reaches: " + (", ".join(info.reaches) or "no Memories"))
        lines.append("Actions: " + ", ".join(a.name for a in self.actions()))
        for r in info.rules:
            lines.append(f"Rule: {r}")
        return ActionResult("\n".join(lines), {"info": info.to_dict()})

    # ── automatic recall ────────────────────────────────────────────

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        """Membase's own search in its fast mode, most relevant first; nothing
        when the key may not search."""
        query = str(query or "").strip()[:2000]
        if not query or not self._allows("search"):
            return []
        data = await self._request("POST", "/v1/search", json={"q": query, "limit": limit, "mode": "fast"})
        hits = []
        for it in _items(data)[:limit]:
            if it.get("content"):
                title = f"{it['title']}: " if it.get("title") else ""
                hits.append(RecallHit(text=_clip(f"{title}{it['content']}", 600), id=str(it.get("id") or ""),
                                      when=str(it.get("recorded_at") or it.get("created_at") or "")[:10],
                                      kind=str(it.get("kind") or "")))
        return hits

    # ── mirroring ───────────────────────────────────────────────────

    async def mirror(self, item: MemoryItem) -> str:
        title, content = mirror_text(item, limit=MAX_CONTENT)
        if item.kind == "reflection":
            # Long, and rewritten as a whole: a document, kept once per id.
            data = await self._request("POST", "/v1/documents/add", json={
                "content": content, "title": title,
                "custom_id": _clip(f"captain-{item.kind}-{item.source_id}", 200),
            })
        else:
            data = await self._request("POST", "/v1/memories/add", json={"content": content, "title": title})
        return str((data or {}).get("id") or "") if isinstance(data, dict) else ""
