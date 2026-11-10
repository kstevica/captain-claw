"""ArcadeDB (ArcadeData/arcadedb, Apache-2.0) as an external memory.

ArcadeDB is a multi-model database — graph, documents, full-text and vectors in
one. Here each thing an agent saves is a ``CaptainMemory`` vertex (its text,
full-text indexed, and its vector), and the people, projects and places it is
about are ``CaptainTopic`` vertices joined to it by ``CaptainAbout`` edges — so
besides "find what we know about X" the memory answers "what is connected to
X" (:meth:`ArcadeDBProvider._do_related`), from the graph.

Search runs two queries and fuses their ranks (reciprocal rank fusion): words
through the full-text index (``SEARCH_INDEX``, Lucene) and meaning through
exact cosine similarity over the person's own vectors
(``vectorCosineSimilarity``). ArcadeDB's ANN index (``LSM_VECTOR``) picks
neighbours across the whole type before any filter applies, which in a shared
database would lose a person's results among everyone else's — the exact scan
stays inside their space. Flight Deck makes the vectors (:mod:`.embeddings`).

One database serves many people, so it is a ``scoped`` provider: every vertex
carries its ``tenant`` (the person's *space*), and every query Flight Deck
sends is bound to it — the agent never names it.

HTTP API (checked against ArcadeDB 26.11.1), Basic auth::

    GET  /api/v1/server                    version (and the credentials work)
    GET  /api/v1/exists/{db}
    POST /api/v1/server                    {command: "create database {db}"}
    POST /api/v1/query/{db}                {language: "sql", command, params}   (read-only)
    POST /api/v1/command/{db}              {language: "sql" | "sqlscript", command, params}

A vertex is keyed by (space, document id) — a unique index — and written with
``UPDATE … UPSERT``, so saving the same document again REPLACES it (and its
topics): Captain's mirrors never pile up, and an insight that replaces another
(same ``entity_key``) replaces it here too.
"""

from __future__ import annotations

import os
import re
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
    rrf,
    truthy,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

TIMEOUT_S = 30.0
DEFAULT_DATABASE = "captain_memory"
MAX_CONTENT = 20_000
KINDS = ("note", *MIRROR_KINDS)
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_QUERY_WORDS = 24

# Idempotent: run on every check, so a database made by hand gets what's missing.
SCHEMA = """CREATE VERTEX TYPE CaptainMemory IF NOT EXISTS;
CREATE PROPERTY CaptainMemory.tenant IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.doc_id IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.content IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.title IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.kind IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.agent IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.category IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.source_id IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.created_at IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.updated_at IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.embedder IF NOT EXISTS STRING;
CREATE PROPERTY CaptainMemory.embedding IF NOT EXISTS ARRAY_OF_FLOATS;
CREATE INDEX IF NOT EXISTS ON CaptainMemory (tenant, doc_id) UNIQUE;
CREATE INDEX IF NOT EXISTS ON CaptainMemory (tenant) NOTUNIQUE;
CREATE INDEX IF NOT EXISTS ON CaptainMemory (content) FULL_TEXT;
CREATE VERTEX TYPE CaptainTopic IF NOT EXISTS;
CREATE PROPERTY CaptainTopic.tenant IF NOT EXISTS STRING;
CREATE PROPERTY CaptainTopic.name IF NOT EXISTS STRING;
CREATE INDEX IF NOT EXISTS ON CaptainTopic (tenant, name) UNIQUE;
CREATE EDGE TYPE CaptainAbout IF NOT EXISTS;
RETURN 'ok';"""

_ROW = "doc_id, content, kind, created_at, out('CaptainAbout').name AS topics"
_DROP_ORPHANS = "DELETE FROM CaptainTopic WHERE tenant = :t AND in('CaptainAbout').size() = 0"


def _ft_query(text: str) -> str:
    """Words for the full-text index, each quoted: Lucene would read AND, OR,
    a colon or a stray quote as syntax."""
    words = re.findall(r"\w+", text.lower())[:MAX_QUERY_WORDS]
    return " ".join(f'"{w}"' for w in words)


def _line(row: dict[str, Any]) -> str:
    """One stored item as the agent reads it."""
    bits = [b for b in (row.get("kind") or "", str(row.get("created_at") or "")[:10]) if b]
    topics = row.get("topics") if isinstance(row.get("topics"), list) else []
    if topics:
        bits.append("topics: " + ", ".join(str(t) for t in topics[:8]))
    return (clip(row.get("content"), 1200) + (f" ({'; '.join(bits)})" if bits else "")
            + (f" [id {row['doc_id']}]" if row.get("doc_id") else ""))


class ArcadeDBProvider(ExternalMemoryProvider):
    name = "arcadedb"
    label = "ArcadeDB"
    description = (
        "Open-source multi-model database (Apache-2.0): what your agents save, with the people, projects and "
        "places it is about as a graph — found by words and by meaning, and by what connects to what. "
        "Self-host it with one Docker command. Vectors are made on Flight Deck unless you name an "
        "embeddings endpoint."
    )
    docs_url = "https://docs.arcadedb.com/"
    credentials_url = "https://docs.arcadedb.com/"
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
            placeholder="http://127.0.0.1:2480",
            help=(
                "ArcadeDB's HTTP address (docker run -p 2480:2480 -e JAVA_OPTS=\"-Darcadedb.server.rootPassword=…\" "
                "arcadedata/arcadedb)."
            ),
        ),
        CredentialField(
            name="username",
            label="User",
            secret=False,
            required=False,
            placeholder="root",
            help="An ArcadeDB user that may read and write the database (and create it, the first time).",
        ),
        CredentialField(
            name="password",
            label="Password",
            placeholder="The user's password",
        ),
        CredentialField(
            name="database",
            label="Database",
            secret=False,
            required=False,
            placeholder=DEFAULT_DATABASE,
            help="Made on first connect if it isn't there (letters, digits, - and _).",
        ),
        CredentialField(
            name="space",
            label="Space",
            secret=False,
            required=False,
            placeholder="Leave empty for one named after you",
            help="Your own connection only: your part of the database (letters, digits, - and _).",
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
        for field, what in (("database", "A database"), ("space", "A space")):
            if out.get(field) and not _NAME.fullmatch(out[field]):
                raise ProviderInputError(f"{what} name is letters, digits, - and _ (at most 64).")
        return embeddings.clean(out)

    # ── transport ───────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return (self.credentials.get("base_url") or os.environ.get("FD_ARCADEDB_BASE_URL", "")).rstrip("/")

    @property
    def database(self) -> str:
        return self.credentials.get("database") or DEFAULT_DATABASE

    @property
    def tenant(self) -> str:
        if not self.scope:
            # Never a default space: in a shared database that would mix people.
            raise ProviderInputError("This ArcadeDB connection has no space for this person.")
        return self.scope

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> Any:
        if not self.base_url:
            raise ProviderInputError("This ArcadeDB connection has no server address.")
        auth = (self.credentials.get("username") or "root", self.credentials.get("password") or "")
        headers = {"Accept": "application/json", "User-Agent": "captain-flight-deck"}
        attempt = 0
        async with httpx.AsyncClient(base_url=self.base_url, timeout=TIMEOUT_S, transport=self._transport,
                                     auth=auth) as client:
            while True:
                try:
                    resp = await client.request(method, path, json=json, headers=headers)
                except httpx.TimeoutException:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError("ArcadeDB didn't answer in time — try again shortly.",
                                        retryable=True) from None
                except httpx.HTTPError as exc:
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise ProviderError(
                        f"Could not reach ArcadeDB at {self.base_url}: {type(exc).__name__}", retryable=True,
                    ) from None
                if 200 <= resp.status_code < 300:
                    try:
                        body = resp.json() if resp.content else {}
                    except ValueError:
                        raise ProviderError("ArcadeDB answered with something that isn't JSON",
                                            status=resp.status_code) from None
                    return body.get("result") if isinstance(body, dict) and "result" in body else body
                if resp.status_code >= 500 and attempt < self.max_retries:
                    attempt += 1
                    continue
                raise self._error(resp)

    def _error(self, resp: httpx.Response) -> ProviderError:
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}
        body = body if isinstance(body, dict) else {}
        exception = str(body.get("exception") or "")
        detail = str(body.get("detail") or body.get("error") or "").strip() or (resp.text or "").strip()[:300]
        if status in (401, 403):
            return ProviderAuthError(
                f"ArcadeDB refused the user or password (HTTP {status}).", status=status, code="unauthorized",
            )
        if status == 404 and "DatabaseNotFound" in exception:
            return ProviderError(
                f"ArcadeDB has no database “{self.database}” — check the connection again to make it.",
                status=status, code="no_database",
            )
        if status == 404:
            return ProviderError(f"ArcadeDB: {detail or 'not found'}", status=status, code="not_found")
        if status == 400:
            return ProviderError(f"ArcadeDB refused the request: {clip(detail, 400)}", status=status,
                                 code="validation")
        # Concurrent writes and a busy server come back as 5xx: worth retrying.
        return ProviderError(f"ArcadeDB: {clip(detail, 300) or f'HTTP {status}'}", status=status,
                             retryable=status >= 500)

    async def _query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = await self._request("POST", f"/api/v1/query/{quote(self.database, safe='')}",
                                   json={"language": "sql", "command": sql, "params": params or {}})
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    async def _command(self, sql: str, params: dict[str, Any] | None = None, *,
                       language: str = "sql") -> list[dict[str, Any]]:
        rows = await self._request("POST", f"/api/v1/command/{quote(self.database, safe='')}",
                                   json={"language": language, "command": sql, "params": params or {}})
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    # ── the connection ──────────────────────────────────────────────

    async def verify(self) -> ConnectionInfo:
        server = await self._request("GET", "/api/v1/server")
        version = str((server or {}).get("version") or "").split(" ")[0] if isinstance(server, dict) else ""
        dims = await self.embedder.dimensions()  # before anything is made
        exists = await self._request("GET", f"/api/v1/exists/{quote(self.database, safe='')}")
        if exists is not True:
            try:
                await self._request("POST", "/api/v1/server", json={"command": f"create database {self.database}"})
            except ProviderAuthError:
                raise ProviderInputError(
                    f"ArcadeDB has no database “{self.database}”, and this user may not make one. Make it "
                    "(or connect as a user who may), then connect again."
                ) from None
        await self._command(SCHEMA, language="sqlscript")
        return ConnectionInfo(
            reaches=[f"ArcadeDB database “{self.database}”"],
            extra={"version": version, "database": self.database, "vector": self.embedder.vector_name,
                   "dims": dims, "embeddings": self.embedder.label},
        )

    def reaches(self) -> list[str]:
        return [f"ArcadeDB database “{self.database}”, space “{self.scope}”"] if self.scope else []

    def actions(self) -> list[ProviderAction]:
        return [
            ProviderAction(
                "remember",
                "Save one fact or note, as it is. Name the people, projects or places it is about in topics — "
                "they become the memory's graph, so related finds it by them.",
                (
                    ActionParam("content", "string", "The fact, written to read on its own.", True),
                    ActionParam("topics", "string", "Comma-separated: who and what it is about (optional)."),
                ),
                writes=True,
            ),
            ProviderAction(
                "search",
                "Find what the memory holds on a question or topic, by words and by meaning, most relevant first.",
                (
                    ActionParam("query", "string", "What to look for, in plain words.", True),
                    ActionParam("limit", "integer", "How many at most (1-50, default 10)."),
                    ActionParam("topic", "string", "Only items about this topic (optional)."),
                    ActionParam("kind", "string", "Only this kind of item (optional).", enum=KINDS),
                ),
            ),
            ProviderAction(
                "related",
                "Everything the memory holds about one topic (a person, project, place…), and the topics that "
                "come up together with it — from the graph.",
                (
                    ActionParam("topic", "string", "The topic, as it was saved (e.g. a name).", True),
                    ActionParam("limit", "integer", "How many items at most (1-50, default 15)."),
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
                "Its id comes from search, list or related.",
                (
                    ActionParam("id", "string", "The id of the item to remove.", True),
                    ActionParam("confirm", "boolean", "true only after the person agreed to remove it."),
                ),
                writes=True,
                destructive=True,
            ),
            ProviderAction(
                "capabilities",
                "How much this memory holds right now, and which server, database and model it uses.",
            ),
        ]

    def provider_rules(self) -> list[str]:
        return [
            "This memory is the person's own space in an ArcadeDB database: everything you save goes there and "
            "searches only see it.",
            "ArcadeDB keeps what you save word for word — it doesn't work out facts by itself. Save one whole "
            "fact per call, and name who and what it is about in topics: topics are the memory's graph.",
            "Use related for 'what do we know about <person/project>' and what goes with it; use search for a "
            "question.",
            "Saving the same text again replaces it. When the person corrects something, save the correction "
            "as a whole sentence (what is true now); offer to forget the old item.",
            "Search returns the closest matches even when none really answers — use only what does, and say "
            "you don't know otherwise.",
            "forget removes an item for good: only after the person agreed, with confirm=true.",
        ]

    # ── writing ─────────────────────────────────────────────────────

    async def _upsert(self, doc_id: str, content: str, fields: dict[str, Any], topics: list[str]) -> None:
        vec = (await self.embedder.embed([content]))[0]
        params: dict[str, Any] = {
            "t": self.tenant, "d": doc_id, "c": content,
            "title": fields.get("title") or None, "kind": fields.get("kind") or "note",
            "agent": fields.get("agent") or None, "category": fields.get("category") or None,
            "sid": fields.get("source_id") or None, "created": iso_utc(fields.get("created_at")),
            "updated": iso_utc(), "emb": self.embedder.vector_name, "vec": vec,
        }
        lines = [
            "BEGIN;",
            "LET m = UPDATE CaptainMemory SET tenant = :t, doc_id = :d, content = :c, title = :title, "
            "kind = :kind, agent = :agent, category = :category, source_id = :sid, created_at = :created, "
            "updated_at = :updated, embedder = :emb, embedding = :vec "
            "UPSERT RETURN AFTER WHERE tenant = :t AND doc_id = :d;",
            # Its topics are what this save says, not what an earlier one did.
            "DELETE FROM (SELECT expand(outE('CaptainAbout')) FROM CaptainMemory WHERE tenant = :t AND doc_id = :d);",
        ]
        for i, topic in enumerate(topics):
            params[f"n{i}"] = topic
            lines.append(f"LET t{i} = UPDATE CaptainTopic SET tenant = :t, name = :n{i} "
                         f"UPSERT RETURN AFTER WHERE tenant = :t AND name = :n{i};")
            lines.append(f"CREATE EDGE CaptainAbout FROM $m TO $t{i} IF NOT EXISTS;")
        lines += [f"{_DROP_ORPHANS};", "COMMIT RETRY 3;", "RETURN $m;"]
        await self._command("\n".join(lines), params, language="sqlscript")

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
        await self._upsert(doc, content, {"kind": "note"}, topics)
        return ActionResult(f"Saved to ArcadeDB (id {doc})" + (f", about {', '.join(topics)}." if topics else "."),
                            {"id": doc, "topics": topics})

    def _where(self, args: dict[str, Any], params: dict[str, Any]) -> str:
        """The person's space, plus a topic / kind when asked."""
        params["t"] = self.tenant
        clauses = ["tenant = :t"]
        topic = clean_topics(args.get("topic"), limit=1)
        if topic:
            params["topic"] = topic[0]
            clauses.append(":topic IN out('CaptainAbout').name")
        kind = str(args.get("kind") or "").strip().lower()
        if kind in KINDS:
            params["kind"] = kind
            clauses.append("kind = :kind")
        return " AND ".join(clauses)

    async def _do_search(self, args: dict[str, Any]) -> ActionResult:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ProviderInputError("search needs a query")
        limit = as_int(args.get("limit"), 10, 1, 50)
        pool = limit * 3
        params: dict[str, Any] = {}
        where = self._where(args, params)
        words: list[str] = []
        ft = _ft_query(query)
        if ft:
            try:
                rows = await self._query(
                    f"SELECT doc_id, $score AS score FROM CaptainMemory WHERE "
                    f"SEARCH_INDEX('CaptainMemory[content]', :q) = true AND {where} ORDER BY score DESC LIMIT {pool}",
                    {**params, "q": ft})
                words = [str(r.get("doc_id")) for r in rows if r.get("doc_id")]
            except ProviderError as exc:
                if exc.status != 400:
                    raise
                log.info("external memory: arcadedb word search refused, meaning only", error=str(exc)[:200])
        vec = (await self.embedder.embed([clip(query, 2000)]))[0]
        rows = await self._query(
            f"SELECT doc_id, vectorCosineSimilarity(embedding, :v) AS sim FROM CaptainMemory WHERE {where} "
            f"AND embedder = :emb ORDER BY sim DESC LIMIT {pool}",
            {**params, "v": vec, "emb": self.embedder.vector_name})
        meaning = [str(r.get("doc_id")) for r in rows if r.get("doc_id")]
        ranked = [doc for doc, _score in rrf(words, meaning)][:limit]
        if not ranked:
            return ActionResult(f"ArcadeDB holds nothing on “{query}”.", {"items": []})
        found = await self._query(f"SELECT {_ROW} FROM CaptainMemory WHERE tenant = :t AND doc_id IN :ids",
                                  {"t": self.tenant, "ids": ranked})
        by_id = {str(r.get("doc_id")): r for r in found}
        items = [by_id[d] for d in ranked if d in by_id]
        lines = [f"ArcadeDB — {len(items)} item(s) on “{query}”, most relevant first:"]
        lines += [f"{n}. {_line(r)}" for n, r in enumerate(items, 1)]
        return ActionResult("\n".join(lines), {"items": items, "words": words, "meaning": meaning})

    async def _do_related(self, args: dict[str, Any]) -> ActionResult:
        topic = clean_topics(args.get("topic") or args.get("query"), limit=1)
        if not topic:
            raise ProviderInputError("related needs a topic — a person, project or place")
        name = topic[0]
        limit = as_int(args.get("limit"), 15, 1, 50)
        p = {"t": self.tenant, "n": name}
        about = "SELECT expand(in('CaptainAbout')) FROM CaptainTopic WHERE tenant = :t AND name = :n"
        items = await self._query(f"SELECT {_ROW} FROM ({about}) WHERE tenant = :t "
                                  f"ORDER BY created_at DESC LIMIT {limit}", p)
        if not items:
            top = await self._query("SELECT name FROM CaptainTopic WHERE tenant = :t "
                                    "ORDER BY in('CaptainAbout').size() DESC LIMIT 15", {"t": self.tenant})
            names = [str(r.get("name")) for r in top if r.get("name")]
            return ActionResult(f"ArcadeDB has nothing about “{name}”."
                                + (f" Topics it has: {', '.join(names)}." if names else ""), {"items": []})
        together = await self._query(
            f"SELECT name, count(*) AS shared FROM (SELECT expand(out('CaptainAbout')) FROM ({about})) "
            "WHERE tenant = :t AND name <> :n GROUP BY name ORDER BY shared DESC LIMIT 15", p)
        lines = [f"ArcadeDB — about “{name}”, newest first ({len(items)}):"]
        lines += [f"- {_line(r)}" for r in items]
        if together:
            lines.append("Comes up together with: "
                         + ", ".join(f"{r.get('name')} ({r.get('shared', 0)})" for r in together))
        return ActionResult("\n".join(lines), {"items": items, "together": together})

    async def _do_list(self, args: dict[str, Any]) -> ActionResult:
        limit = as_int(args.get("limit"), 20, 1, 100)
        params: dict[str, Any] = {}
        rows = await self._query(f"SELECT {_ROW} FROM CaptainMemory WHERE {self._where(args, params)} "
                                 f"ORDER BY created_at DESC LIMIT {limit}", params)
        if not rows:
            return ActionResult("ArcadeDB holds nothing here yet.", {"items": []})
        lines = [f"ArcadeDB — {len(rows)} item(s), newest first:"] + [f"- {_line(r)}" for r in rows]
        return ActionResult("\n".join(lines), {"items": rows})

    async def _do_topics(self, args: dict[str, Any]) -> ActionResult:
        rows = await self._query(
            "SELECT name, in('CaptainAbout').size() AS n FROM CaptainTopic WHERE tenant = :t "
            f"ORDER BY n DESC LIMIT {as_int(args.get('limit'), 30, 1, 100)}", {"t": self.tenant})
        if not rows:
            return ActionResult("No topics in ArcadeDB yet — items saved with topics show up here.", {"topics": []})
        return ActionResult("ArcadeDB — topics, most items first: "
                            + ", ".join(f"{r.get('name')} ({r.get('n', 0)})" for r in rows), {"topics": rows})

    async def _do_forget(self, args: dict[str, Any]) -> ActionResult:
        doc = str(args.get("id") or "").strip()
        if not doc:
            raise ProviderInputError("forget needs the id of the item to remove")
        p = {"t": self.tenant, "d": doc}
        found = await self._query("SELECT count(*) AS n FROM CaptainMemory WHERE tenant = :t AND doc_id = :d", p)
        if not (found and int(found[0].get("n") or 0)):
            raise ProviderError(f"ArcadeDB has nothing with the id {doc} in this space.", status=404,
                                code="not_found")
        if not truthy(args.get("confirm")):
            raise ProviderError(
                f"Forgetting {doc} removes it for good. Nothing changed. Ask the person; only if they agree, "
                "call again with confirm=true.",
                status=428, code="confirmation_required",
            )
        # Deleting a vertex deletes its edges; topics nothing points at go too.
        await self._command(
            f"BEGIN;\nDELETE FROM CaptainMemory WHERE tenant = :t AND doc_id = :d;\n{_DROP_ORPHANS};\n"
            "COMMIT RETRY 3;\nRETURN 1;", p, language="sqlscript")
        return ActionResult(f"Forgotten: {doc}.", {"id": doc})

    async def _do_capabilities(self, args: dict[str, Any]) -> ActionResult:
        server = await self._request("GET", "/api/v1/server")
        version = str((server or {}).get("version") or "").split(" ")[0] if isinstance(server, dict) else ""
        t = {"t": self.tenant}
        items = await self._query("SELECT count(*) AS n FROM CaptainMemory WHERE tenant = :t", t)
        topics = await self._query("SELECT count(*) AS n FROM CaptainTopic WHERE tenant = :t", t)
        n_items = int(items[0].get("n") or 0) if items else 0
        n_topics = int(topics[0].get("n") or 0) if topics else 0
        lines = [
            f"ArcadeDB {version} — database “{self.database}”, space “{self.scope}”",
            f"Holds {n_items} item(s) about {n_topics} topic(s). Search by words and meaning; vectors by "
            f"{self.embedder.label}.",
            "Actions: " + ", ".join(a.name for a in self.actions()),
        ]
        return ActionResult("\n".join(lines), {"version": version, "items": n_items, "topics": n_topics})

    # ── automatic recall ────────────────────────────────────────────

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        """Items sharing the message's content words (full-text, stopwords
        dropped) or alike in meaning (at least the connection's threshold),
        fused; often none."""
        from captain_claw import retrieval

        query = clip(query, 2000)
        if not query:
            return []
        t = {"t": self.tenant}
        words: list[str] = []
        ft = _ft_query(" ".join(retrieval.query_terms(query, max_terms=12, keep_identifiers=True)))
        if ft:
            try:
                rows = await self._query(
                    "SELECT doc_id, $score AS score FROM CaptainMemory WHERE "
                    "SEARCH_INDEX('CaptainMemory[content]', :q) = true AND tenant = :t "
                    f"ORDER BY score DESC LIMIT {limit * 2}", {**t, "q": ft})
                words = [str(r.get("doc_id")) for r in rows if r.get("doc_id")]
            except ProviderError as exc:
                if exc.status != 400:
                    raise
        vec = (await self.embedder.embed([query]))[0]
        rows = await self._query(
            "SELECT doc_id, vectorCosineSimilarity(embedding, :v) AS sim FROM CaptainMemory "
            f"WHERE tenant = :t AND embedder = :emb ORDER BY sim DESC LIMIT {limit * 2}",
            {**t, "v": vec, "emb": self.embedder.vector_name})
        meaning = [str(r.get("doc_id")) for r in rows
                   if r.get("doc_id") and float(r.get("sim") or 0) >= self.embedder.floor]
        ranked = [doc for doc, _score in rrf(words, meaning)][:limit]
        if not ranked:
            return []
        found = await self._query(f"SELECT {_ROW} FROM CaptainMemory WHERE tenant = :t AND doc_id IN :ids",
                                  {**t, "ids": ranked})
        by_id = {str(r.get("doc_id")): r for r in found}
        return [RecallHit(text=clip(by_id[d].get("content"), 600), id=d,
                          when=str(by_id[d].get("created_at") or "")[:10], kind=str(by_id[d].get("kind") or ""))
                for d in ranked if d in by_id and by_id[d].get("content")]

    # ── mirroring ───────────────────────────────────────────────────

    async def mirror(self, item: MemoryItem) -> str:
        doc = mirror_document_id(item)
        fields = {
            "kind": item.kind, "source_id": item.source_id, "created_at": item.created_at,
            "title": clip(item.title, 200), "agent": clip(item.agent, 200),
            "category": clip(item.meta.get("category"), 200),
        }
        await self._upsert(doc, mirror_body(item, limit=MAX_CONTENT), fields,
                           clean_topics(item.tags))
        return doc
