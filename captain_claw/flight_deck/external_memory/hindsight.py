"""Hindsight (vectorize-io/hindsight, MIT) as an external memory.

Hindsight keeps memories in *banks*, each strictly isolated. It extracts facts,
entities and their dates from what it is given (``retain``), finds them again
with four strategies at once — meaning, keywords, the entity graph and time —
(``recall``), and can reason over everything a bank holds (``reflect``).

One server (self-hosted with one Docker command, or Hindsight Cloud) serves
many people, so it is a ``scoped`` provider: every call names a bank, and that
bank is this connection's :attr:`scope` — the person's own. An admin can
connect one Hindsight for the whole deck and each user gets a bank of their
own (``captain-<user id>``); a person can also connect their own server or
Cloud account, with a bank they name.

REST (from Hindsight's own generated client, hindsight-client 1.2.0), with
``Authorization: Bearer <key>`` when the server requires one (self-hosted:
``HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension``
plus ``HINDSIGHT_API_TENANT_API_KEY``)::

    GET    /version                                        api_version, features
    GET    /v1/default/banks                               (used to check the key)
    POST   /v1/default/banks/{bank}/memories               retain {items: [{content, context, timestamp,
                                                           metadata, document_id, tags}], async}
    POST   /v1/default/banks/{bank}/memories/recall        {query, budget, max_tokens, types, tags}
    POST   /v1/default/banks/{bank}/reflect                {query, budget, context}
    GET    /v1/default/banks/{bank}/memories/list          ?limit&offset&q&type
    GET    /v1/default/banks/{bank}/documents              ?limit&offset&q
    GET    /v1/default/banks/{bank}/documents/{id}
    DELETE /v1/default/banks/{bank}/documents/{id}         the document and every fact learned from it
    GET    /v1/default/banks/{bank}/stats

A bank is created the first time it is used. Retaining again with the same
``document_id`` REPLACES that document (old facts deleted, new content
reprocessed), so Captain's mirrors — keyed by what they are — never pile up,
and an insight that replaces another (same ``entity_key``) replaces it here too.
"""

from __future__ import annotations

import os
import re
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
    clip,
    mirror_body,
    mirror_document_id,
    note_document_id,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

TIMEOUT_S = 120.0  # reflect runs an agentic loop of up to ten rounds
MAX_CONTENT = 50_000
MAX_QUERY_CHARS = 1_800  # Hindsight refuses queries over 500 tokens
BUDGETS = ("low", "mid", "high")

# How each mirrored kind is introduced to Hindsight's extractor. Neutral on
# purpose: anything said about Captain itself (agent names, categories, "kept
# by") is extracted as a fact about the person's world — that goes in metadata.
_CONTEXT = {
    "insight": "something learned about the person and their work",
    "intention": "something the person intends to do",
    "lesson": "a standing rule or fact the person gave their assistants",
    "reflection": "the assistant's own notes on how to work better for the person",
    "intuition": "an unconfirmed hunch the assistant had",
}


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


def _day(value: Any) -> str:
    return str(value or "")[:10]


def _when(item: dict[str, Any]) -> str:
    start = _day(item.get("occurred_start") or item.get("date"))
    said = _day(item.get("mentioned_at"))
    if start and said and start != said:
        return f"happened {start}, said {said}"
    if start:
        return start
    return f"said {said}" if said else ""


class HindsightProvider(ExternalMemoryProvider):
    name = "hindsight"
    label = "Hindsight"
    description = (
        "Open-source agent memory (MIT): learns facts, people and dates from what it is given, "
        "finds them four ways at once, and can reason over everything it knows. Self-host it "
        "with one Docker command, or use Hindsight Cloud."
    )
    docs_url = "https://hindsight.vectorize.io/"
    credentials_url = "https://hindsight.vectorize.io/"
    tenancy = "scoped"
    url_field = "base_url"
    scope_label = "bank"
    scope_field = "bank"
    credential_fields = (
        CredentialField(
            name="base_url",
            label="Server address",
            secret=False,
            placeholder="http://127.0.0.1:8888",
            help=(
                "Where Hindsight's API answers: your own server (docker run … ghcr.io/vectorize-io/hindsight, "
                "port 8888) or your Hindsight Cloud API address."
            ),
        ),
        CredentialField(
            name="api_key",
            label="API key",
            required=False,
            placeholder="Leave empty if the server has no key",
            help=(
                "Hindsight Cloud's key, or the HINDSIGHT_API_TENANT_API_KEY your own server was started with. "
                "A server without one lets anyone who can reach it read every bank."
            ),
        ),
        CredentialField(
            name="bank",
            label="Bank",
            secret=False,
            required=False,
            placeholder="Leave empty for one named after you",
            help="Your own connection only: the bank your agents use (letters, digits, - and _).",
        ),
    )

    def __init__(self, credentials: dict[str, str], info: ConnectionInfo | None = None,
                 *, scope: str = "", transport: httpx.AsyncBaseTransport | None = None,
                 max_retries: int = 2) -> None:
        super().__init__(credentials, info, scope=scope)
        self._transport = transport
        self.max_retries = max(0, int(max_retries))

    @classmethod
    def clean_credentials(cls, raw: dict[str, Any]) -> dict[str, str]:
        out = super().clean_credentials(raw)
        url = out.get("base_url", "")
        if not url.lower().startswith(("http://", "https://")):
            raise ProviderInputError("The server address must start with http:// or https://")
        out["base_url"] = url.rstrip("/")
        bank = out.get("bank", "")
        if bank and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", bank):
            raise ProviderInputError("A bank name is letters, digits, - and _ (at most 64).")
        return out

    # ── transport ───────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return (self.credentials.get("base_url") or os.environ.get("FD_HINDSIGHT_BASE_URL", "")).rstrip("/")

    @property
    def bank(self) -> str:
        if not self.scope:
            # Never a default bank: on a shared server that would mix people.
            raise ProviderInputError("This Hindsight connection has no bank for this person.")
        return self.scope

    def _bank_path(self, rest: str = "") -> str:
        return f"/v1/default/banks/{quote(self.bank, safe='')}{rest}"

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None,
                       params: dict[str, Any] | None = None, verifying: bool = False) -> Any:
        if not self.base_url:
            raise ProviderInputError("This Hindsight connection has no server address.")
        headers = {"Accept": "application/json", "User-Agent": "captain-flight-deck"}
        key = self.credentials.get("api_key", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        attempt = 0
        async with httpx.AsyncClient(base_url=self.base_url, timeout=TIMEOUT_S, transport=self._transport) as client:
            while True:
                try:
                    resp = await client.request(method, path, json=json, params=params, headers=headers)
                except httpx.TimeoutException:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError("Hindsight didn't answer in time — try again shortly.", retryable=True) from None
                except httpx.HTTPError as exc:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError(
                        f"Could not reach Hindsight at {self.base_url}: {type(exc).__name__}", retryable=True,
                    ) from None
                if 200 <= resp.status_code < 300:
                    try:
                        return resp.json() if resp.content else {}
                    except ValueError:
                        raise ProviderError("Hindsight answered with something that isn't JSON",
                                            status=resp.status_code) from None
                if resp.status_code >= 500 and attempt < self.max_retries:
                    attempt += 1
                    continue
                raise self._error(resp, verifying=verifying)

    def _error(self, resp: httpx.Response, *, verifying: bool) -> ProviderError:
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}
        detail = body.get("detail") if isinstance(body, dict) else ""
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(str(d.get("msg") or d) for d in detail if isinstance(d, dict)) or str(detail)
        detail = str(detail or (body.get("message") if isinstance(body, dict) else "") or "").strip()
        if status in (401, 403):
            return ProviderAuthError(
                f"Hindsight refused the key (HTTP {status}) — check it, or leave it empty if the server "
                "has none.",
                status=status, code="unauthorized",
            )
        if status == 404:
            return ProviderError(
                "Hindsight has nothing by that id in this bank" + (f" ({detail})" if detail else "") + ".",
                status=status, code="not_found",
            )
        if status == 429:
            return ProviderError("Hindsight is busy — try again in a minute.", status=status,
                                 code="rate_limited", retryable=True)
        if status in (400, 409, 422):
            return ProviderError(f"Hindsight refused the request: {detail or f'HTTP {status}'}", status=status,
                                 code="validation")
        return ProviderError(f"Hindsight: {detail or f'HTTP {status}'}", status=status, retryable=status >= 500)

    # ── the connection ──────────────────────────────────────────────

    async def verify(self) -> ConnectionInfo:
        version: dict[str, Any] = {}
        try:
            data = await self._request("GET", "/version", verifying=True)
            version = data if isinstance(data, dict) else {}
        except ProviderAuthError:
            raise
        except ProviderError as exc:
            if exc.status != 404:
                raise
        # /version may be open even where the API needs a key: a call that
        # needs one proves the key.
        await self._request("GET", "/v1/default/banks", verifying=True)
        features = version.get("features") if isinstance(version.get("features"), dict) else {}
        return ConnectionInfo(
            allowed=[],
            reaches=[],
            rules=[],
            extra={
                "api_version": str(version.get("api_version") or ""),
                "observations": bool(features.get("observations")),
                "store_document_text": features.get("store_document_text"),
            },
        )

    def reaches(self) -> list[str]:
        return [f"Hindsight bank “{self.scope}”"] if self.scope else []

    def actions(self) -> list[ProviderAction]:
        return [
            ProviderAction(
                "remember",
                "Save one fact or note; Hindsight learns the facts, people and dates in it (in the background).",
                (
                    ActionParam("content", "string", "The fact, written to read on its own.", True),
                    ActionParam("context", "string", "Where it comes from, e.g. 'call with Acme' (optional)."),
                ),
                writes=True,
            ),
            ProviderAction(
                "search",
                "Find the facts the memory holds on a question or topic, most relevant first, with their dates.",
                (
                    ActionParam("query", "string", "What to look for, in plain words (keep it short).", True),
                    ActionParam("limit", "integer", "How many facts at most (1-50, default 10)."),
                    ActionParam("budget", "string", "How hard to look: low, mid (default) or high.", enum=BUDGETS),
                ),
            ),
            ProviderAction(
                "reflect",
                "A reasoned answer from everything the memory holds — for 'what have we learned about X', "
                "'what should I keep in mind', patterns and corrections. Slower than search.",
                (
                    ActionParam("question", "string", "The question.", True),
                    ActionParam("budget", "string", "How hard to think: low (default), mid or high.", enum=BUDGETS),
                ),
            ),
            ProviderAction(
                "list",
                "The facts the memory holds, newest first (optionally only those matching words).",
                (
                    ActionParam("limit", "integer", "How many (1-100, default 20)."),
                    ActionParam("query", "string", "Only facts containing these words (optional)."),
                ),
            ),
            ProviderAction(
                "documents",
                "The documents saved into the memory (each fact comes from one), newest first.",
                (ActionParam("limit", "integer", "How many (1-100, default 20)."),),
            ),
            ProviderAction(
                "forget",
                "Remove one saved document and every fact learned from it, for good. ONLY after the person "
                "clearly agreed — then pass confirm=true. Its id comes from search or documents (document_id).",
                (
                    ActionParam("id", "string", "The document id to remove.", True),
                    ActionParam("confirm", "boolean", "true only after the person agreed to remove it."),
                ),
                writes=True,
                destructive=True,
            ),
            ProviderAction(
                "capabilities",
                "What this memory holds right now (facts, documents, pending work) and which server it is.",
            ),
        ]

    def provider_rules(self) -> list[str]:
        return [
            "This memory is the person's own bank: everything you save goes there and searches only see it.",
            "Hindsight learns facts from what you save in the background — something you just saved may "
            "take a moment to show up. Don't save it again.",
            "Facts come with dates (when it happened, when it was said). When two facts disagree, the newer "
            "one usually holds — say which you relied on.",
            "When the person corrects something, save the correction as a whole sentence (what is true now, "
            "and that it replaces what was said before).",
            "Use search for a plain fact; use reflect when the answer needs reasoning over many facts.",
            "forget removes a whole document and everything learned from it, for good: only after the "
            "person agreed, with confirm=true.",
        ]

    # ── actions ─────────────────────────────────────────────────────

    async def run(self, action: str, args: dict[str, Any]) -> ActionResult:
        action = (action or "").strip().lower()
        self.action(action)
        args = args if isinstance(args, dict) else {}
        return await getattr(self, f"_do_{action}")(args)

    async def _retain(self, item: dict[str, Any], *, wait: bool = False) -> dict[str, Any]:
        data = await self._request("POST", self._bank_path("/memories"), json={"items": [item], "async": not wait})
        return data if isinstance(data, dict) else {}

    async def _do_remember(self, args: dict[str, Any]) -> ActionResult:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ProviderInputError("remember needs content — the fact to save")
        if len(content) > MAX_CONTENT:
            raise ProviderInputError(f"That is {len(content):,} characters; save at most {MAX_CONTENT:,} at once.")
        # Keyed by the text: the same fact saved twice replaces itself.
        item: dict[str, Any] = {
            "content": content,
            "document_id": note_document_id(content),
            "tags": ["captain", "note"],
            "metadata": {"source": "captain", "kind": "note"},
        }
        context = clip(args.get("context"), 300)
        if context:
            item["context"] = context
        data = await self._retain(item)
        return ActionResult(
            f"Saved to Hindsight (document {item['document_id']}). It is learned in the background, so search "
            "may not find it for a moment.",
            data,
        )

    async def _do_search(self, args: dict[str, Any]) -> ActionResult:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ProviderInputError("search needs a query")
        if len(query) > MAX_QUERY_CHARS:
            raise ProviderInputError("That query is too long for Hindsight — ask in a sentence or two.")
        budget = str(args.get("budget") or "mid").strip().lower()
        limit = _int(args.get("limit"), 10, 1, 50)
        data = await self._request("POST", self._bank_path("/memories/recall"), json={
            "query": query, "budget": budget if budget in BUDGETS else "mid", "max_tokens": 4096,
        })
        results = [r for r in (data or {}).get("results") or [] if isinstance(r, dict)][:limit]
        if not results:
            return ActionResult(f"Hindsight holds nothing on “{query}”.", data if isinstance(data, dict) else {})
        lines = [f"Hindsight — {len(results)} fact(s) on “{query}”, most relevant first:"]
        for n, r in enumerate(results, 1):
            bits = [b for b in (r.get("type") or "", _when(r)) if b]
            ref = f" [document {r['document_id']}]" if r.get("document_id") else ""
            lines.append(f"{n}. {clip(r.get('text'), 1200)}" + (f" ({', '.join(bits)})" if bits else "") + ref)
        return ActionResult("\n".join(lines), data)

    async def _do_reflect(self, args: dict[str, Any]) -> ActionResult:
        question = str(args.get("question") or args.get("query") or "").strip()
        if not question:
            raise ProviderInputError("reflect needs a question")
        if len(question) > MAX_QUERY_CHARS:
            raise ProviderInputError("That question is too long for Hindsight — ask in a sentence or two.")
        budget = str(args.get("budget") or "low").strip().lower()
        data = await self._request("POST", self._bank_path("/reflect"), json={
            "query": question, "budget": budget if budget in BUDGETS else "low",
        })
        data = data if isinstance(data, dict) else {}
        text = str(data.get("text") or "").strip() or "(Hindsight had no answer)"
        based = data.get("based_on") if isinstance(data.get("based_on"), dict) else {}
        facts = [f for f in based.get("memories") or [] if isinstance(f, dict)]
        if facts:
            text += f"\n\n(Based on {len(facts)} fact(s) in the memory.)"
        return ActionResult(text, data)

    async def _do_list(self, args: dict[str, Any]) -> ActionResult:
        params: dict[str, Any] = {"limit": _int(args.get("limit"), 20, 1, 100), "offset": 0}
        q = str(args.get("query") or "").strip()
        if q:
            params["q"] = q
        data = await self._request("GET", self._bank_path("/memories/list"), params=params)
        items = [i for i in (data or {}).get("items") or [] if isinstance(i, dict)]
        if not items:
            return ActionResult("Hindsight holds no facts yet" + (f" matching “{q}”" if q else "") + ".", data or {})
        total = (data or {}).get("total")
        lines = [f"Hindsight — {len(items)} fact(s)" + (f" of {total}" if total else "") + ":"]
        for i in items:
            when = _when(i)
            lines.append(f"- {clip(i.get('text'), 400)}" + (f" ({when})" if when else "")
                         + (" [invalidated]" if i.get("state") == "invalidated" else ""))
        return ActionResult("\n".join(lines), data)

    async def _do_documents(self, args: dict[str, Any]) -> ActionResult:
        data = await self._request("GET", self._bank_path("/documents"),
                                   params={"limit": _int(args.get("limit"), 20, 1, 100), "offset": 0})
        items = [i for i in (data or {}).get("items") or [] if isinstance(i, dict)]
        if not items:
            return ActionResult("No documents in Hindsight yet.", data or {})
        lines = [f"Hindsight — {len(items)} document(s):"]
        for i in items:
            lines.append(f"- {i.get('id', '')}: {i.get('memory_unit_count', 0)} fact(s), "
                         f"{i.get('text_length', 0)} characters, saved {_day(i.get('updated_at') or i.get('created_at'))}")
        return ActionResult("\n".join(lines), data)

    async def _do_forget(self, args: dict[str, Any]) -> ActionResult:
        doc = str(args.get("id") or "").strip()
        if not doc:
            raise ProviderInputError("forget needs the id of the document to remove")
        if not _truthy(args.get("confirm")):
            # Hindsight deletes on the spot; the person's yes is checked here.
            raise ProviderError(
                f"Forgetting {doc} removes it and everything learned from it, for good. Nothing changed. "
                "Ask the person; only if they agree, call again with confirm=true.",
                status=428, code="confirmation_required",
            )
        data = await self._request("DELETE", self._bank_path(f"/documents/{quote(doc, safe='')}"))
        data = data if isinstance(data, dict) else {}
        n = data.get("memory_units_deleted")
        return ActionResult(f"Forgotten: {doc}" + (f" ({n} fact(s) removed)." if n is not None else "."), data)

    async def _do_capabilities(self, args: dict[str, Any]) -> ActionResult:
        version = await self._request("GET", "/version")
        stats: dict[str, Any] = {}
        try:
            got = await self._request("GET", self._bank_path("/stats"))
            stats = got if isinstance(got, dict) else {}
        except ProviderError as exc:
            if exc.status != 404:
                raise
        lines = [f"Hindsight {str((version or {}).get('api_version') or '').strip() or ''} — bank “{self.scope}”"]
        if stats:
            lines.append(f"Holds {stats.get('total_nodes', 0)} fact(s) from {stats.get('total_documents', 0)} "
                         f"document(s), {stats.get('total_observations', 0)} observation(s); "
                         f"{stats.get('pending_operations', 0)} still being learned.")
        else:
            lines.append("Nothing saved yet.")
        lines.append("Actions: " + ", ".join(a.name for a in self.actions()))
        return ActionResult("\n".join(lines), {"version": version, "stats": stats})

    # ── automatic recall ────────────────────────────────────────────

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        """Hindsight's own recall at its cheapest budget, capped in tokens:
        it ranks (meaning, words, entities, time, then a reranker) and keeps
        only what fits, so what comes back is what it judged relevant."""
        query = clip(query, MAX_QUERY_CHARS)
        if not query:
            return []
        data = await self._request("POST", self._bank_path("/memories/recall"), json={
            "query": query, "budget": "low", "max_tokens": 1500,
        })
        hits = []
        for r in ((data or {}).get("results") or [])[:limit]:
            if isinstance(r, dict) and r.get("text"):
                hits.append(RecallHit(text=clip(r["text"], 600), id=str(r.get("document_id") or ""),
                                      when=_day(r.get("occurred_start") or r.get("mentioned_at")),
                                      kind=str(r.get("type") or "")))
        return hits

    # ── mirroring ───────────────────────────────────────────────────

    @staticmethod
    def mirror_content(item: MemoryItem) -> str:
        """Only what is about the person's world (:func:`base.mirror_body`)."""
        return mirror_body(item, limit=MAX_CONTENT)

    async def mirror(self, item: MemoryItem) -> str:
        content = self.mirror_content(item)
        # What it IS, not which copy: an insight that replaces another (same
        # entity_key) replaces its document here too.
        doc = mirror_document_id(item)
        meta = {"source": "captain", "kind": item.kind, "source_id": item.source_id}
        for k in ("category", "agent"):
            v = item.meta.get(k) if k != "agent" else item.agent
            if v:
                meta[k] = clip(v, 200)
        retain: dict[str, Any] = {
            "content": content,
            "context": _CONTEXT.get(item.kind, "something the person's assistant kept"),
            "document_id": doc,
            "metadata": meta,
            "tags": ["captain", item.kind, *[clip(t, 60) for t in item.tags[:8]]],
        }
        if item.created_at:
            retain["timestamp"] = item.created_at
        data = await self._retain(retain)
        return str(data.get("operation_id") or doc)
