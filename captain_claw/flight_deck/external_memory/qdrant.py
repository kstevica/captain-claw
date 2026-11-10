"""Qdrant (qdrant/qdrant, Apache-2.0) as an external memory.

Qdrant is a vector database: it keeps what it is given as it is and finds it
again by meaning — and, since 1.20, by words too (BM25 over a full-text index),
fused with the meaning in one query. It doesn't make vectors itself, so Flight
Deck makes them (:mod:`.embeddings`: a small model on the deck by default).

One collection serves many people, so it is a ``scoped`` provider. Each person
is a *space* in it — the ``tenant`` field, indexed as Qdrant's own tenant key
(``is_tenant``), and in EVERY filter Flight Deck sends. The agent never names
it. An admin can connect one Qdrant for the whole deck (each user gets
``captain-<user id>``); a person can connect their own, naming the space.

REST (checked against Qdrant 1.20.0), ``api-key`` header when the server has
one::

    GET  /                                        version
    GET  /collections                             (used to check the key)
    GET  /collections/{c}                         vectors config, payload indexes
    PUT  /collections/{c}                         {vectors: {<model>: {size, distance}}, hnsw_config}
    PUT  /collections/{c}/index                   {field_name, field_schema}
    PUT  /collections/{c}/points?wait=true        {points: [{id, vector: {<model>: [...]}, payload}]}
    POST /collections/{c}/points/query            dense, text (BM25) or prefetch both + fusion rrf
    POST /collections/{c}/points/scroll           newest first (order_by created_at)
    POST /collections/{c}/points/count
    POST /collections/{c}/points/delete?wait=true {filter}
    POST /collections/{c}/facet                   {key: topics, filter}

A point's id is derived from its space and document id (uuid5), so saving the
same document again REPLACES it: Captain's mirrors — keyed by what they are —
never pile up, and an insight that replaces another (same ``entity_key``)
replaces it here too. The collection's vector is named after the embedding
model, so a connection that changes model is refused rather than mixing
vectors that can't be compared.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any
from urllib.parse import quote

import httpx

from captain_claw.flight_deck.external_memory import embeddings
from captain_claw.flight_deck.external_memory.base import (
    MIRROR_KINDS,
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
    as_int,
    clean_topics,
    clip,
    iso_utc,
    mirror_body,
    mirror_document_id,
    note_document_id,
    truthy,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

TIMEOUT_S = 30.0
DEFAULT_COLLECTION = "captain_memory"
MAX_CONTENT = 20_000
KINDS = ("note", *MIRROR_KINDS)
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
# Every point is in exactly one space; the id makes the same document in the
# same space the same point.
_ID_NS = uuid.UUID("6f1c9a52-3c1e-4c55-9d43-0b8e9a6f2a11")


def _version_at_least(version: str, major: int, minor: int) -> bool:
    m = re.match(r"(\d+)\.(\d+)", version or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (major, minor)


def _line(p: dict[str, Any]) -> str:
    """One stored item as the agent reads it."""
    bits = [b for b in (p.get("kind") or "", str(p.get("created_at") or "")[:10]) if b]
    topics = p.get("topics") if isinstance(p.get("topics"), list) else []
    if topics:
        bits.append("topics: " + ", ".join(str(t) for t in topics[:8]))
    return (clip(p.get("content"), 1200) + (f" ({'; '.join(bits)})" if bits else "")
            + (f" [id {p['doc_id']}]" if p.get("doc_id") else ""))


class QdrantProvider(ExternalMemoryProvider):
    name = "qdrant"
    label = "Qdrant"
    description = (
        "Open-source vector database (Apache-2.0): keeps what your agents save as it is and finds it by "
        "meaning and by words. Self-host it with one Docker command, or use Qdrant Cloud. Vectors are made "
        "on Flight Deck unless you name an embeddings endpoint."
    )
    docs_url = "https://qdrant.tech/documentation/"
    credentials_url = "https://cloud.qdrant.io/"
    tenancy = "scoped"
    url_field = "base_url"
    extra_url_fields = ("embedding_url",)
    scope_label = "space"
    scope_field = "space"
    credential_fields = (
        CredentialField(
            name="base_url",
            label="Server address",
            secret=False,
            placeholder="http://127.0.0.1:6333",
            help=(
                "Qdrant's REST address: your own server (docker run -p 6333:6333 qdrant/qdrant) or your "
                "Qdrant Cloud cluster URL (https://…cloud.qdrant.io:6333)."
            ),
        ),
        CredentialField(
            name="api_key",
            label="API key",
            required=False,
            placeholder="Leave empty if the server has no key",
            help=(
                "Qdrant Cloud's API key, or the QDRANT__SERVICE__API_KEY your own server was started with. "
                "A server without one lets anyone who can reach it read every collection."
            ),
        ),
        CredentialField(
            name="collection",
            label="Collection",
            secret=False,
            required=False,
            placeholder=DEFAULT_COLLECTION,
            help="Made on first connect if it isn't there (letters, digits, - and _).",
        ),
        CredentialField(
            name="space",
            label="Space",
            secret=False,
            required=False,
            placeholder="Leave empty for one named after you",
            help="Your own connection only: your part of the collection (letters, digits, - and _).",
        ),
        *embeddings.FIELDS,
    )

    def __init__(self, credentials: dict[str, str], info: ConnectionInfo | None = None,
                 *, scope: str = "", transport: httpx.AsyncBaseTransport | None = None,
                 max_retries: int = 2) -> None:
        super().__init__(credentials, info, scope=scope)
        self._transport = transport
        self.max_retries = max(0, int(max_retries))
        self.embedder = embeddings.Embedder(self.credentials, transport=transport)

    @classmethod
    def clean_credentials(cls, raw: dict[str, Any]) -> dict[str, str]:
        out = super().clean_credentials(raw)
        url = out.get("base_url", "")
        if not url.lower().startswith(("http://", "https://")):
            raise ProviderInputError("The server address must start with http:// or https://")
        out["base_url"] = url.rstrip("/")
        for field, what in (("collection", "A collection"), ("space", "A space")):
            if out.get(field) and not _NAME.fullmatch(out[field]):
                raise ProviderInputError(f"{what} name is letters, digits, - and _ (at most 64).")
        return embeddings.clean(out)

    # ── transport ───────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return (self.credentials.get("base_url") or os.environ.get("FD_QDRANT_BASE_URL", "")).rstrip("/")

    @property
    def collection(self) -> str:
        return self.credentials.get("collection") or DEFAULT_COLLECTION

    @property
    def tenant(self) -> str:
        if not self.scope:
            # Never a default space: in a shared collection that would mix people.
            raise ProviderInputError("This Qdrant connection has no space for this person.")
        return self.scope

    @property
    def vector(self) -> str:
        return self.embedder.vector_name

    def _c(self, rest: str = "") -> str:
        return f"/collections/{quote(self.collection, safe='')}{rest}"

    def _filter(self, *conditions: dict[str, Any]) -> dict[str, Any]:
        return {"must": [{"key": "tenant", "match": {"value": self.tenant}}, *conditions]}

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None,
                       params: dict[str, Any] | None = None) -> Any:
        if not self.base_url:
            raise ProviderInputError("This Qdrant connection has no server address.")
        headers = {"Accept": "application/json", "User-Agent": "captain-flight-deck"}
        key = self.credentials.get("api_key", "")
        if key:
            headers["api-key"] = key
        attempt = 0
        async with httpx.AsyncClient(base_url=self.base_url, timeout=TIMEOUT_S, transport=self._transport) as client:
            while True:
                try:
                    resp = await client.request(method, path, json=json, params=params, headers=headers)
                except httpx.TimeoutException:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError("Qdrant didn't answer in time — try again shortly.", retryable=True) from None
                except httpx.HTTPError as exc:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError(
                        f"Could not reach Qdrant at {self.base_url}: {type(exc).__name__}", retryable=True,
                    ) from None
                if 200 <= resp.status_code < 300:
                    try:
                        body = resp.json() if resp.content else {}
                    except ValueError:
                        raise ProviderError("Qdrant answered with something that isn't JSON",
                                            status=resp.status_code) from None
                    return body.get("result", body) if isinstance(body, dict) else body
                if resp.status_code >= 500 and attempt < self.max_retries:
                    attempt += 1
                    continue
                raise self._error(resp)

    @staticmethod
    def _error(resp: httpx.Response) -> ProviderError:
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}
        detail = ""
        if isinstance(body, dict):
            st = body.get("status")
            detail = str((st.get("error") if isinstance(st, dict) else st) or body.get("detail") or "")
        detail = detail.strip() or (resp.text or "").strip()[:300]
        if status in (401, 403):
            return ProviderAuthError(
                f"Qdrant refused the key (HTTP {status}) — check it, or leave it empty if the server has none.",
                status=status, code="unauthorized",
            )
        if status == 404:
            return ProviderError(f"Qdrant: {detail or 'not found'}", status=status, code="not_found")
        if status == 429:
            return ProviderError("Qdrant is busy — try again in a minute.", status=status,
                                 code="rate_limited", retryable=True)
        if status in (400, 409, 422):
            return ProviderError(f"Qdrant refused the request: {detail or f'HTTP {status}'}", status=status,
                                 code="validation")
        return ProviderError(f"Qdrant: {detail or f'HTTP {status}'}", status=status, retryable=status >= 500)

    # ── the collection ──────────────────────────────────────────────

    async def _ensure_collection(self, dims: int, version: str) -> bool:
        """Make the collection and its indexes if missing; refuse one made for
        another model. Returns whether word search (BM25) is on."""
        try:
            info = await self._request("GET", self._c())
        except ProviderError as exc:
            if exc.status != 404:
                raise
            info = None
        if info is None:
            await self._request("PUT", self._c(), json={
                "vectors": {self.vector: {"size": dims, "distance": "Cosine"}},
                # Qdrant's advice for many tenants in one collection: a graph
                # per tenant instead of one global graph.
                "hnsw_config": {"payload_m": 16, "m": 0},
            })
            schema: dict[str, Any] = {}
        else:
            params = ((info.get("config") or {}).get("params") or {}) if isinstance(info, dict) else {}
            vectors = params.get("vectors")
            named = vectors if isinstance(vectors, dict) and "size" not in vectors else {}
            mine = named.get(self.vector) if isinstance(named, dict) else None
            if not isinstance(mine, dict):
                made_for = ", ".join(sorted(named)) if named else "one unnamed vector"
                raise ProviderInputError(
                    f"The collection “{self.collection}” was made for other vectors ({made_for}), not for "
                    f"{self.embedder.model}. Name another collection, or use the embeddings model it was made with."
                )
            if int(mine.get("size") or 0) != dims:
                raise ProviderInputError(
                    f"The collection “{self.collection}” holds {mine.get('size')}-number vectors, but "
                    f"{self.embedder.model} makes {dims}. Name another collection."
                )
            schema = info.get("payload_schema") if isinstance(info.get("payload_schema"), dict) else {}
        wanted: list[tuple[str, dict[str, Any]]] = [
            ("tenant", {"type": "keyword", "is_tenant": True}),
            ("doc_id", {"type": "keyword"}),
            ("kind", {"type": "keyword"}),
            ("topics", {"type": "keyword"}),
            ("created_at", {"type": "datetime"}),
        ]
        for field, field_schema in wanted:
            if field not in schema:
                await self._request("PUT", self._c("/index"), params={"wait": "true"},
                                    json={"field_name": field, "field_schema": field_schema})
        text = schema.get("content") if isinstance(schema.get("content"), dict) else None
        if text is not None:
            return bool((text.get("params") or {}).get("scoring"))
        base = {"type": "text", "tokenizer": "word", "lowercase": True}
        if _version_at_least(version, 1, 20):
            try:
                await self._request("PUT", self._c("/index"), params={"wait": "true"},
                                    json={"field_name": "content", "field_schema": {**base, "scoring": True}})
                return True
            except ProviderError as exc:
                if exc.status not in (400, 422):
                    raise
        await self._request("PUT", self._c("/index"), params={"wait": "true"},
                            json={"field_name": "content", "field_schema": base})
        return False

    # ── the connection ──────────────────────────────────────────────

    async def verify(self) -> ConnectionInfo:
        root = await self._request("GET", "/")
        version = str((root or {}).get("version") or "") if isinstance(root, dict) else ""
        # "/" can be open where the API needs a key: a call that needs one proves it.
        await self._request("GET", "/collections")
        dims = await self.embedder.dimensions()  # before anything is made
        hybrid = await self._ensure_collection(dims, version)
        return ConnectionInfo(
            reaches=[f"Qdrant collection “{self.collection}”"],
            extra={"version": version, "collection": self.collection, "vector": self.vector, "dims": dims,
                   "hybrid": hybrid, "embeddings": self.embedder.label},
        )

    def reaches(self) -> list[str]:
        return [f"Qdrant collection “{self.collection}”, space “{self.scope}”"] if self.scope else []

    def actions(self) -> list[ProviderAction]:
        return [
            ProviderAction(
                "remember",
                "Save one fact or note, as it is. Name the people, projects or places it is about in topics, "
                "so it can be found by them too.",
                (
                    ActionParam("content", "string", "The fact, written to read on its own.", True),
                    ActionParam("topics", "string", "Comma-separated: who and what it is about (optional)."),
                ),
                writes=True,
            ),
            ProviderAction(
                "search",
                "Find what the memory holds on a question or topic, by meaning and by words, most relevant first.",
                (
                    ActionParam("query", "string", "What to look for, in plain words.", True),
                    ActionParam("limit", "integer", "How many at most (1-50, default 10)."),
                    ActionParam("topic", "string", "Only items about this topic (optional)."),
                    ActionParam("kind", "string", "Only this kind of item (optional).", enum=KINDS),
                ),
            ),
            ProviderAction(
                "list",
                "What the memory holds, newest first (optionally only about one topic).",
                (
                    ActionParam("limit", "integer", "How many (1-100, default 20)."),
                    ActionParam("topic", "string", "Only items about this topic (optional)."),
                ),
            ),
            ProviderAction(
                "topics",
                "The topics the memory holds most about (people, projects, places…), with how many items each.",
                (ActionParam("limit", "integer", "How many (1-100, default 30)."),),
            ),
            ProviderAction(
                "forget",
                "Remove one saved item for good. ONLY after the person clearly agreed — then pass confirm=true. "
                "Its id comes from search or list.",
                (
                    ActionParam("id", "string", "The id of the item to remove.", True),
                    ActionParam("confirm", "boolean", "true only after the person agreed to remove it."),
                ),
                writes=True,
                destructive=True,
            ),
            ProviderAction(
                "capabilities",
                "How much this memory holds right now, and which server, collection and model it uses.",
            ),
        ]

    def provider_rules(self) -> list[str]:
        return [
            "This memory is the person's own space in a Qdrant collection: everything you save goes there and "
            "searches only see it.",
            "Qdrant keeps what you save word for word — it doesn't work out facts by itself. Save one whole "
            "fact per call, and name who and what it is about in topics.",
            "Saving the same text again replaces it. When the person corrects something, save the correction "
            "as a whole sentence (what is true now); offer to forget the old item.",
            "Search returns the closest matches even when none really answers — use only what does, and say "
            "you don't know otherwise.",
            "forget removes an item for good: only after the person agreed, with confirm=true.",
        ]

    # ── writing ─────────────────────────────────────────────────────

    def _point_id(self, doc_id: str) -> str:
        return str(uuid.uuid5(_ID_NS, f"{self.tenant}\x1f{doc_id}"))

    async def _upsert(self, doc_id: str, content: str, fields: dict[str, Any]) -> None:
        vec = (await self.embedder.embed([content]))[0]
        dims = int(self.info.extra.get("dims") or 0)
        if dims and len(vec) != dims:
            raise ProviderError(
                f"{self.embedder.model} now makes {len(vec)}-number vectors but the collection holds {dims} — "
                "reconnect with the model it was made with, or another collection.", code="dimension_mismatch",
            )
        now = iso_utc()
        payload = {**fields, "tenant": self.tenant, "doc_id": doc_id, "content": content, "updated_at": now,
                   "created_at": iso_utc(fields.get("created_at")), "source": "captain"}
        body = {"points": [{"id": self._point_id(doc_id), "vector": {self.vector: vec}, "payload": payload}]}
        try:
            await self._request("PUT", self._c("/points"), params={"wait": "true"}, json=body)
        except ProviderError as exc:
            if exc.status != 404:
                raise
            # The collection went away (deleted by hand): make it again, once.
            await self._ensure_collection(len(vec), str(self.info.extra.get("version") or ""))
            await self._request("PUT", self._c("/points"), params={"wait": "true"}, json=body)

    # ── actions ─────────────────────────────────────────────────────

    async def run(self, action: str, args: dict[str, Any]) -> ActionResult:
        action = (action or "").strip().lower()
        self.action(action)
        args = args if isinstance(args, dict) else {}
        return await getattr(self, f"_do_{action}")(args)

    async def _do_remember(self, args: dict[str, Any]) -> ActionResult:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ProviderInputError("remember needs content — the fact to save")
        if len(content) > MAX_CONTENT:
            raise ProviderInputError(f"That is {len(content):,} characters; save at most {MAX_CONTENT:,} at once.")
        doc = note_document_id(content)
        topics = clean_topics(args.get("topics"))
        await self._upsert(doc, content, {"kind": "note", "topics": topics})
        return ActionResult(f"Saved to Qdrant (id {doc})" + (f", about {', '.join(topics)}." if topics else "."),
                            {"id": doc, "topics": topics})

    def _conditions(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        topic = clean_topics(args.get("topic"), limit=1)
        if topic:
            out.append({"key": "topics", "match": {"value": topic[0]}})
        kind = str(args.get("kind") or "").strip().lower()
        if kind in KINDS:
            out.append({"key": "kind", "match": {"value": kind}})
        return out

    async def _do_search(self, args: dict[str, Any]) -> ActionResult:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ProviderInputError("search needs a query")
        limit = as_int(args.get("limit"), 10, 1, 50)
        flt = self._filter(*self._conditions(args))
        vec = (await self.embedder.embed([clip(query, 2000)]))[0]
        dense = {"query": vec, "using": self.vector, "filter": flt}
        data: Any = None
        if self.info.extra.get("hybrid"):
            try:
                data = await self._request("POST", self._c("/points/query"), json={
                    "prefetch": [{**dense, "limit": limit * 3},
                                 {"query": {"text": clip(query, 2000)}, "using": "content", "filter": flt,
                                  "limit": limit * 3}],
                    "query": {"fusion": "rrf"}, "limit": limit, "with_payload": True,
                })
            except ProviderError as exc:
                if exc.status not in (400, 422):
                    raise
                log.info("external memory: qdrant word search refused, meaning only", error=str(exc)[:200])
        if data is None:
            data = await self._request("POST", self._c("/points/query"),
                                       json={**dense, "limit": limit, "with_payload": True})
        points = [p for p in (data or {}).get("points") or [] if isinstance(p, dict)]
        if not points:
            return ActionResult(f"Qdrant holds nothing on “{query}”.", {"points": []})
        lines = [f"Qdrant — {len(points)} item(s) on “{query}”, most relevant first:"]
        for n, p in enumerate(points, 1):
            lines.append(f"{n}. {_line(p.get('payload') or {})}")
        return ActionResult("\n".join(lines), {"points": points})

    async def _do_list(self, args: dict[str, Any]) -> ActionResult:
        limit = as_int(args.get("limit"), 20, 1, 100)
        data = await self._request("POST", self._c("/points/scroll"), json={
            "filter": self._filter(*self._conditions(args)), "limit": limit, "with_payload": True,
            "order_by": {"key": "created_at", "direction": "desc"},
        })
        points = [p for p in (data or {}).get("points") or [] if isinstance(p, dict)]
        if not points:
            return ActionResult("Qdrant holds nothing here yet.", {"points": []})
        lines = [f"Qdrant — {len(points)} item(s), newest first:"]
        lines += [f"- {_line(p.get('payload') or {})}" for p in points]
        return ActionResult("\n".join(lines), {"points": points})

    async def _do_topics(self, args: dict[str, Any]) -> ActionResult:
        data = await self._request("POST", self._c("/facet"), json={
            "key": "topics", "filter": self._filter(), "limit": as_int(args.get("limit"), 30, 1, 100),
        })
        hits = [h for h in (data or {}).get("hits") or [] if isinstance(h, dict)]
        if not hits:
            return ActionResult("No topics in Qdrant yet — items saved with topics show up here.", {"hits": []})
        return ActionResult("Qdrant — topics, most items first: "
                            + ", ".join(f"{h.get('value')} ({h.get('count', 0)})" for h in hits), {"hits": hits})

    async def _do_forget(self, args: dict[str, Any]) -> ActionResult:
        doc = str(args.get("id") or "").strip()
        if not doc:
            raise ProviderInputError("forget needs the id of the item to remove")
        flt = self._filter({"key": "doc_id", "match": {"value": doc}})
        found = await self._request("POST", self._c("/points/count"), json={"filter": flt, "exact": True})
        if not int((found or {}).get("count") or 0):
            raise ProviderError(f"Qdrant has nothing with the id {doc} in this space.", status=404, code="not_found")
        if not truthy(args.get("confirm")):
            raise ProviderError(
                f"Forgetting {doc} removes it for good. Nothing changed. Ask the person; only if they agree, "
                "call again with confirm=true.",
                status=428, code="confirmation_required",
            )
        await self._request("POST", self._c("/points/delete"), params={"wait": "true"}, json={"filter": flt})
        return ActionResult(f"Forgotten: {doc}.", {"id": doc})

    async def _do_capabilities(self, args: dict[str, Any]) -> ActionResult:
        root = await self._request("GET", "/")
        count = await self._request("POST", self._c("/points/count"),
                                    json={"filter": self._filter(), "exact": True})
        n = int((count or {}).get("count") or 0)
        version = str((root or {}).get("version") or "") if isinstance(root, dict) else ""
        words = "meaning and words" if self.info.extra.get("hybrid") else "meaning (word search needs Qdrant 1.20+)"
        lines = [
            f"Qdrant {version} — collection “{self.collection}”, space “{self.scope}”",
            f"Holds {n} item(s). Search by {words}; vectors by {self.embedder.label}.",
            "Actions: " + ", ".join(a.name for a in self.actions()),
        ]
        return ActionResult("\n".join(lines), {"version": version, "count": n})

    # ── automatic recall ────────────────────────────────────────────

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        """Items alike in meaning (at least the connection's threshold) or
        sharing the message's content words (BM25, stopwords dropped — "the"
        would match everything), fused; often none."""
        from captain_claw import retrieval

        query = clip(query, 2000)
        if not query:
            return []
        flt = self._filter()
        vec = (await self.embedder.embed([query]))[0]
        dense = {"query": vec, "using": self.vector, "filter": flt, "limit": limit * 2,
                 "score_threshold": self.embedder.floor}
        terms = retrieval.query_terms(query, max_terms=12, keep_identifiers=True)
        prefetch = [dense]
        if terms and self.info.extra.get("hybrid"):
            prefetch.append({"query": {"text": " ".join(terms)}, "using": "content", "filter": flt,
                             "limit": limit * 2})
        body = {"prefetch": prefetch, "query": {"fusion": "rrf"}, "limit": limit,
                "with_payload": ["doc_id", "content", "kind", "created_at"]}
        try:
            data = await self._request("POST", self._c("/points/query"), json=body)
        except ProviderError as exc:
            if exc.status not in (400, 422) or len(prefetch) == 1:
                raise
            data = await self._request("POST", self._c("/points/query"), json={**body, "prefetch": [dense]})
        hits = []
        for p in (data or {}).get("points") or []:
            pl = p.get("payload") if isinstance(p, dict) else None
            if isinstance(pl, dict) and pl.get("content"):
                hits.append(RecallHit(text=clip(pl["content"], 600), id=str(pl.get("doc_id") or ""),
                                      when=str(pl.get("created_at") or "")[:10], kind=str(pl.get("kind") or "")))
        return hits

    # ── mirroring ───────────────────────────────────────────────────

    async def mirror(self, item: MemoryItem) -> str:
        doc = mirror_document_id(item)
        fields: dict[str, Any] = {
            "kind": item.kind,
            "topics": clean_topics(item.tags),
            "source_id": item.source_id,
            "created_at": item.created_at,
        }
        for key, value in (("title", item.title), ("agent", item.agent), ("category", item.meta.get("category"))):
            if value:
                fields[key] = clip(value, 200)
        await self._upsert(doc, mirror_body(item, limit=MAX_CONTENT), fields)
        return doc
