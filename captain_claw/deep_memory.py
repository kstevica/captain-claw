"""Typesense-backed deep memory for long-term searchable content.

Deep memory is an *additional* layer on top of the SQLite-backed semantic
memory.  It is NOT a replacement — it is a persistent archive searched only
on demand (when the user explicitly asks to "search deep memory", "find in
archive", etc.).

Content flows in via:
  - The micro-loop ``no_file`` sink (scale loop indexes processed items).
  - The LLM-callable ``typesense`` tool (manual indexing).
  - ``DeepMemoryIndex.index_document()`` / ``index_batch()`` programmatic API.

Content flows out via:
  - ``build_context_note()`` → injected into the LLM prompt when triggered.
  - ``search()`` → returns typed ``DeepMemoryResult`` objects.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass (mirrors SemanticMemoryResult shape)
# ---------------------------------------------------------------------------


@dataclass
class DeepMemoryResult:
    """One deep-memory hit."""

    doc_id: str
    source: str
    reference: str
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    snippet: str
    score: float
    text_score: float
    vector_score: float
    updated_at: int  # unix timestamp
    text_l1: str = ""
    text_l2: str = ""


# ---------------------------------------------------------------------------
# Collection schema
# ---------------------------------------------------------------------------

_COLLECTION_SCHEMA_TEMPLATE: dict[str, Any] = {
    "fields": [
        {"name": "doc_id", "type": "string", "facet": True},
        {"name": "source", "type": "string", "facet": True},
        {"name": "reference", "type": "string", "facet": True},
        {"name": "path", "type": "string"},
        # Tenant key.  Optional so it can be PATCHed onto a collection that
        # predates multi-tenancy; every write from Flight Deck sets it.
        {"name": "owner_id", "type": "string", "facet": True, "optional": True},
        # sha256 of the *source* bytes, identical across every chunk of a
        # document.  Lets a re-index of unchanged content short-circuit before
        # chunking, embedding, or summarising anything.
        {"name": "content_hash", "type": "string", "facet": True, "optional": True},
        {"name": "text", "type": "string"},
        {"name": "text_l1", "type": "string", "optional": True},
        {"name": "text_l2", "type": "string", "optional": True},
        {"name": "chunk_index", "type": "int32"},
        {"name": "start_line", "type": "int32", "optional": True},
        {"name": "end_line", "type": "int32", "optional": True},
        {"name": "tags", "type": "string[]", "facet": True, "optional": True},
        {"name": "updated_at", "type": "int64"},
    ],
    "default_sorting_field": "updated_at",
    "token_separators": [".", "/", "-", "_"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_id(text: str) -> str:
    """Deterministic short hash for document/chunk IDs."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _chunk_text(
    text: str,
    chunk_chars: int = 1_400,
    chunk_overlap_chars: int = 200,
) -> list[dict[str, Any]]:
    """Split *text* into overlapping line-based chunks.

    Uses the same algorithm as ``SemanticMemoryIndex._chunk_document()`` so
    that chunk boundaries are consistent across memory layers.
    """
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(lines):
        end = start
        used = 0
        while end < len(lines):
            line_len = len(lines[end]) + 1
            if used and used + line_len > chunk_chars:
                break
            used += line_len
            end += 1
        if end <= start:
            end = min(len(lines), start + 1)
        chunk_text_str = "\n".join(lines[start:end]).strip()
        if chunk_text_str:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_line": start + 1,
                    "end_line": end,
                    "text": chunk_text_str,
                }
            )
            chunk_index += 1
        if end >= len(lines):
            break
        overlap_lines = 0
        overlap_chars = 0
        idx = end - 1
        while idx >= start and overlap_chars < chunk_overlap_chars:
            overlap_chars += len(lines[idx]) + 1
            overlap_lines += 1
            idx -= 1
        start = max(start + 1, end - overlap_lines) if overlap_lines else end
    return chunks


# ---------------------------------------------------------------------------
# DeepMemoryIndex
# ---------------------------------------------------------------------------


class DeepMemoryIndex:
    """Typesense-backed deep memory for long-term searchable content.

    Designed to mirror the ``SemanticMemoryIndex`` public surface so the
    agent context mixin can use them interchangeably for note generation.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 8108,
        protocol: str = "http",
        api_key: str = "",
        collection_name: str = "captain_claw_deep_memory",
        embedding_dims: int = 0,  # 0 = probe the embedding provider
        auto_embed: bool = True,
        chunk_chars: int = 1_400,
        chunk_overlap_chars: int = 200,
        embedding_chain: Any | None = None,
        layered_summaries: bool = True,
        min_score: float = 0.12,
    ) -> None:
        self._base_url = f"{protocol}://{host}:{port}"
        self._api_key = api_key
        self._collection_name = collection_name
        self._embedding_dims = embedding_dims
        self._auto_embed = auto_embed
        self._chunk_chars = chunk_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._embedding_chain = embedding_chain
        self._layered_summaries = bool(layered_summaries)
        self._min_score = float(min_score)
        self._summarizer: Any = None  # callable(text) -> (l1, l2)
        self._client: httpx.Client | None = None
        self._collection_ensured = False
        # Vector state.  ``_chain_dims`` is the width the embedding provider
        # actually emits (probed once); ``_vectors_disabled`` latches when the
        # collection and the provider disagree and we could not reconcile them.
        self._chain_dims: int = 0
        self._vectors_disabled = False
        self._dim_error_logged = False

    @property
    def collection_name(self) -> str:
        """The Typesense collection name used for deep memory."""
        return self._collection_name

    def set_summarizer(self, fn: Any) -> None:
        """Set a callable ``fn(text: str) -> tuple[str, str]`` returning (L1, L2) summaries."""
        self._summarizer = fn

    # ------------------------------------------------------------------
    # HTTP client (lazy)
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(30.0, connect=5.0),
                headers={
                    "X-TYPESENSE-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _probe_chain_dims(self) -> int:
        """Return the vector width the embedding chain actually emits.

        Probed once and cached.  Returns 0 when no usable chain is wired,
        which is the signal to run keyword-only.
        """
        if self._chain_dims:
            return self._chain_dims
        chain = self._embedding_chain
        if chain is None or not getattr(chain, "enabled", False):
            return 0
        try:
            _key, vectors = chain.embed_batch(["dimension probe"])
        except Exception as exc:
            log.debug("Embedding dimension probe failed", error=str(exc))
            return 0
        if vectors and vectors[0]:
            self._chain_dims = len(vectors[0])
        return self._chain_dims

    def _live_embedding_dims(self, schema: dict[str, Any]) -> int:
        """Return the ``num_dim`` the live collection declares, or 0."""
        for field in schema.get("fields", []):
            if field.get("name") == "embedding":
                return int(field.get("num_dim") or 0)
        return 0

    def _add_missing_fields(self, schema: dict[str, Any]) -> list[str]:
        """PATCH any optional template fields the live collection is missing.

        A collection created before a field existed does not grow one on its
        own, and Typesense answers **400** to a ``filter_by`` naming a field it
        does not have — so an un-migrated collection fails every owner-scoped
        read and delete, not just the new writes. Only *optional* fields can be
        added: a required one has no value for the documents already stored.
        """
        live = {f.get("name") for f in schema.get("fields", [])}
        missing = [
            dict(f)
            for f in _COLLECTION_SCHEMA_TEMPLATE["fields"]
            if f.get("name") not in live and f.get("optional")
        ]
        if not missing:
            return []
        try:
            resp = self._get_client().patch(
                f"{self._base_url}/collections/{self._collection_name}",
                json={"fields": missing},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning(
                "Could not add missing deep memory fields",
                fields=[f["name"] for f in missing],
                error=str(exc)[:300],
            )
            return []
        added = [f["name"] for f in missing]
        log.info(
            "Added missing fields to the deep memory collection",
            collection=self._collection_name,
            fields=added,
        )
        return added

    def _repair_embedding_field(self, target_dims: int, *, replace: bool) -> bool:
        """Declare ``embedding`` at *target_dims* on the live collection.

        With *replace*, the existing field is dropped first.  Typesense refuses
        that alter if any stored document still carries a vector of the old
        width, so it only succeeds while the field is unused — which is exactly
        the state a dimension mismatch leaves the collection in, since every
        vector was discarded before it was ever written.  Documents and their
        ``text`` are preserved either way, so callers can re-embed from what is
        already stored.
        """
        client = self._get_client()
        fields: list[dict[str, Any]] = []
        if replace:
            fields.append({"name": "embedding", "drop": True})
        fields.append(
            {
                "name": "embedding",
                "type": "float[]",
                "num_dim": target_dims,
                "optional": True,
            }
        )
        try:
            resp = client.patch(
                f"{self._base_url}/collections/{self._collection_name}",
                json={"fields": fields},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning(
                "Could not re-declare the deep memory embedding field",
                target_dims=target_dims,
                error=str(exc)[:300],
            )
            return False
        log.info(
            "Re-declared deep memory embedding field",
            collection=self._collection_name,
            num_dim=target_dims,
        )
        return True

    def ensure_collection(self) -> None:
        """Create the deep memory collection, reconciling the vector width.

        The authoritative vector width is the *provider's*, not the config's.
        ``deep_memory.embedding_dims`` is a hand-editable Settings field with
        no way to validate itself against the embedding chain, so a stale
        value there (the 1536 default, against a 256-dim local provider)
        silently discarded every vector at both index and query time and left
        deep memory running keyword-only.  Precedence is therefore:

            live collection schema  >  probed embedding chain  >  config

        A brand-new collection is created at the probed width.  An existing
        collection that disagrees with the provider is repaired in place when
        no vectors are stored yet; otherwise vectors are disabled loudly
        rather than being dropped one batch at a time in the background.
        """
        if self._collection_ensured:
            return
        client = self._get_client()
        chain_dims = self._probe_chain_dims()

        try:
            resp = client.get(
                f"{self._base_url}/collections/{self._collection_name}"
            )
            if resp.status_code == 200:
                schema = resp.json()
                # Bring an older collection up to the current schema before
                # anything tries to filter on a field it doesn't have yet.
                self._add_missing_fields(schema)
                live_dims = self._live_embedding_dims(schema)
                if chain_dims and live_dims != chain_dims:
                    # Covers both a width disagreement and a collection with no
                    # vector field at all (live_dims == 0) — the latter is what
                    # the typesense tool's bootstrap path creates, and it must
                    # gain a field rather than being written off as vectorless.
                    if self._repair_embedding_field(
                        chain_dims, replace=bool(live_dims)
                    ):
                        self._embedding_dims = chain_dims
                    else:
                        self._vectors_disabled = True
                        log.error(
                            "Deep memory vector search is DISABLED — the "
                            "collection and the embedding provider disagree "
                            "on vector width. Re-index the collection at the "
                            "provider's width to enable hybrid search.",
                            collection=self._collection_name,
                            collection_dims=live_dims,
                            provider_dims=chain_dims,
                        )
                elif live_dims:
                    self._embedding_dims = live_dims
                self._collection_ensured = True
                return
        except httpx.HTTPError:
            pass

        # Creating fresh — trust the provider over the configured value.
        if chain_dims and chain_dims != self._embedding_dims:
            log.info(
                "Creating deep memory at the embedding provider's width, "
                "not the configured one",
                configured=self._embedding_dims,
                provider=chain_dims,
            )
            self._embedding_dims = chain_dims

        schema: dict[str, Any] = {
            "name": self._collection_name,
            **_COLLECTION_SCHEMA_TEMPLATE,
        }
        # Copy so we never mutate the module-level template.
        schema["fields"] = list(schema["fields"])
        # Add embedding field if dims are configured.
        if self._embedding_dims and self._embedding_dims > 0:
            schema["fields"].append(
                {
                    "name": "embedding",
                    "type": "float[]",
                    "num_dim": self._embedding_dims,
                    "optional": True,
                }
            )
        try:
            resp = client.post(
                f"{self._base_url}/collections",
                json=schema,
            )
            resp.raise_for_status()
            log.info(
                "Created deep memory collection",
                collection=self._collection_name,
                fields=len(schema["fields"]),
            )
            self._collection_ensured = True
        except httpx.HTTPStatusError as exc:
            # 409 = already exists — that's fine.
            if exc.response.status_code == 409:
                self._collection_ensured = True
            else:
                raise

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_document(
        self,
        doc_id: str,
        text: str,
        *,
        source: str = "manual",
        reference: str = "",
        path: str = "",
        tags: list[str] | None = None,
        owner_id: str = "",
        content_hash: str = "",
        summarize: bool = False,
    ) -> int:
        """Index a single document (auto-chunked). Returns chunk count.

        *summarize* is opt-in **per document**, and deliberately defaults to
        off.  The L1/L2 summariser is one LLM call per ~1400-char chunk, so
        auto-summarising an upload turns a 100 KB file into ~70 calls; that is
        a cost the caller should choose, not inherit.  Without it a chunk still
        carries its full text and both search halves work — only the compact
        prompt-injection layers are absent.
        """
        self.ensure_collection()

        text_bytes = len(text.encode("utf-8"))
        log.info(
            "Indexing document",
            doc_id=doc_id,
            source=source,
            reference=reference or "(none)",
            size_bytes=text_bytes,
        )

        chunks = _chunk_text(
            text,
            chunk_chars=self._chunk_chars,
            chunk_overlap_chars=self._chunk_overlap_chars,
        )
        if not chunks:
            log.info("No chunks produced, skipping", doc_id=doc_id)
            return 0

        total = len(chunks)
        log.info("Chunked document", doc_id=doc_id, total_chunks=total)

        now = int(time.time())
        docs: list[dict[str, Any]] = []
        texts_for_embedding: list[str] = []

        for i, chunk in enumerate(chunks):
            cid = _hash_id(f"{doc_id}:{chunk['chunk_index']}:{chunk['text'][:64]}")
            doc: dict[str, Any] = {
                "id": cid,
                "doc_id": doc_id,
                "source": source,
                "reference": reference,
                "path": path or reference,
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "updated_at": now,
            }
            if owner_id:
                doc["owner_id"] = owner_id
            if content_hash:
                doc["content_hash"] = content_hash
            # Generate L1/L2 summaries only when this document asked for them.
            if summarize and self._layered_summaries and self._summarizer is not None:
                try:
                    l1, l2 = self._summarizer(chunk["text"])
                    doc["text_l1"] = str(l1 or "").strip()
                    doc["text_l2"] = str(l2 or "").strip()
                    log.info(
                        "Summarized chunk",
                        doc_id=doc_id,
                        chunk=f"{i + 1}/{total}",
                    )
                except Exception:
                    log.warning(
                        "Summarization failed for chunk",
                        doc_id=doc_id,
                        chunk=f"{i + 1}/{total}",
                    )
            if tags:
                doc["tags"] = tags
            docs.append(doc)
            texts_for_embedding.append(chunk["text"])

        # Compute embeddings if available.
        log.info("Generating embeddings", doc_id=doc_id, chunks=total)
        embeddings = self._embed(texts_for_embedding)
        if embeddings:
            for doc_dict, vec in zip(docs, embeddings):
                doc_dict["embedding"] = vec
            log.info("Embeddings complete", doc_id=doc_id, count=len(embeddings))
        else:
            log.info("No embeddings generated (BM25 only)", doc_id=doc_id)

        log.info("Upserting to Typesense", doc_id=doc_id, chunks=total)
        count = self._upsert_batch(docs)
        log.info(
            "Indexing complete",
            doc_id=doc_id,
            chunks_indexed=count,
            size_bytes=text_bytes,
        )
        return count

    def index_batch(self, documents: list[dict[str, Any]]) -> int:
        """Batch index pre-built documents. Returns count indexed.

        Each document should already be a flat dict with at minimum:
        ``id``, ``doc_id``, ``text``, ``updated_at``.
        """
        self.ensure_collection()
        if not documents:
            return 0

        # Optionally embed texts that don't already have embeddings.
        texts_to_embed: list[tuple[int, str]] = []
        for i, doc in enumerate(documents):
            if "embedding" not in doc and doc.get("text"):
                texts_to_embed.append((i, doc["text"]))

        if texts_to_embed:
            embeddings = self._embed([t for _, t in texts_to_embed])
            if embeddings:
                for (idx, _), vec in zip(texts_to_embed, embeddings):
                    documents[idx]["embedding"] = vec

        return self._upsert_batch(documents)

    def _upsert_batch(self, docs: list[dict[str, Any]]) -> int:
        """JSONL upsert to Typesense. Returns success count."""
        client = self._get_client()
        lines = [json.dumps(d, ensure_ascii=False) for d in docs]
        body = "\n".join(lines)
        resp = client.post(
            f"{self._base_url}/collections/{self._collection_name}/documents/import",
            params={"action": "upsert"},
            content=body,
            headers={
                "X-TYPESENSE-API-KEY": self._api_key,
                "Content-Type": "text/plain",
            },
        )
        resp.raise_for_status()
        result_lines = resp.text.strip().splitlines()
        ok = sum(
            1
            for line in result_lines
            if line.strip() and json.loads(line).get("success", False)
        )
        if ok < len(docs):
            # Log first failing line for debugging.
            first_error = ""
            for line in result_lines:
                if line.strip():
                    parsed = json.loads(line)
                    if not parsed.get("success", False):
                        first_error = parsed.get("error", "") or parsed.get("document", "")
                        break
            log.warning(
                "Deep memory batch upsert partial failure",
                ok=ok,
                total=len(docs),
                first_error=str(first_error)[:300],
            )
        return ok

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings using the shared embedding chain."""
        if not self._auto_embed or not texts or self._vectors_disabled:
            return []
        chain = self._embedding_chain
        if chain is None or not getattr(chain, "enabled", False):
            return []
        try:
            _key, vectors = chain.embed_batch(texts)
            # Backstop only.  ensure_collection() reconciles the collection
            # against the provider up front, so reaching this branch means a
            # fallback provider (ollama, local_hash) kicked in mid-run at a
            # different width.  Latch and shout — the previous code logged at
            # warning level and returned [], which is how vector search stayed
            # off for the entire life of this feature without anyone noticing.
            expected = self._embedding_dims
            if expected and vectors:
                actual = len(vectors[0]) if vectors[0] else 0
                if actual != expected:
                    self._vectors_disabled = True
                    if not self._dim_error_logged:
                        self._dim_error_logged = True
                        log.error(
                            "Deep memory vector search DISABLED mid-run — the "
                            "embedding provider changed vector width. Deep "
                            "memory has fallen back to keyword-only search.",
                            expected=expected,
                            actual=actual,
                            provider=_key,
                        )
                    return []
            return vectors
        except Exception as exc:
            log.debug("Deep memory embedding failed", error=str(exc))
            return []

    def unowned_count(self) -> int:
        """Documents carrying no ``owner_id``.

        Anything indexed before tenancy existed has no owner, and owner-scoped
        search therefore never returns it — present in the collection, invisible
        to everyone. Surfaced so that state is diagnosable rather than a mystery.
        """
        self.ensure_collection()
        # Typesense has no "field is null/absent" filter — ``owner_id:=null``
        # silently matches the literal string and returns 0, and ``:=*`` is a
        # parse error. Faceting does answer it, in one cheap query: total minus
        # the sum of the per-owner counts is what carries no owner at all.
        try:
            resp = self._get_client().post(
                f"{self._base_url}/multi_search",
                json={"searches": [{
                    "collection": self._collection_name,
                    "q": "*", "query_by": "text",
                    "facet_by": "owner_id", "per_page": 0,
                    "max_facet_values": 1000,
                }]},
            )
            resp.raise_for_status()
            result = (resp.json().get("results") or [{}])[0]
            total = int(result.get("found", 0) or 0)
            facets = result.get("facet_counts") or []
            owned = sum(
                int(c.get("count", 0) or 0)
                for f in facets
                if f.get("field_name") == "owner_id"
                for c in (f.get("counts") or [])
            )
            return max(0, total - owned)
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("Unowned-document count failed", error=str(exc))
            return 0

    def claim_unowned(self, owner_id: str) -> int:
        """Stamp *owner_id* onto every document that has none. Returns the count.

        Deliberately explicit rather than automatic: assigning ownership of
        pre-tenancy data is a judgement call, and on a multi-user Flight Deck
        guessing wrong hands one user another's archive.
        """
        self.ensure_collection()
        client = self._get_client()
        resp = client.get(
            f"{self._base_url}/collections/{self._collection_name}/documents/export",
            params={"include_fields": "id,owner_id"},
            timeout=httpx.Timeout(300.0, connect=5.0),
        )
        resp.raise_for_status()
        ids = [
            doc["id"]
            for line in resp.text.splitlines()
            if line.strip()
            for doc in [json.loads(line)]
            if not doc.get("owner_id")
        ]
        if not ids:
            return 0
        body = "\n".join(
            json.dumps({"id": i, "owner_id": owner_id}, ensure_ascii=False) for i in ids
        )
        r = client.post(
            f"{self._base_url}/collections/{self._collection_name}/documents/import",
            params={"action": "update"},
            content=body,
            headers={"X-TYPESENSE-API-KEY": self._api_key, "Content-Type": "text/plain"},
        )
        r.raise_for_status()
        claimed = sum(
            1 for line in r.text.strip().splitlines()
            if line.strip() and json.loads(line).get("success", False)
        )
        log.info("Claimed unowned deep memory documents", owner=owner_id, claimed=claimed)
        return claimed

    def reembed_all(self, *, batch_size: int = 64) -> tuple[int, int]:
        """Backfill vectors for documents stored without one.

        Every document keeps its ``text``, so a collection indexed while the
        vector half was disabled can be repaired in place — no re-ingestion
        from the original sources, which for scale-loop and web-fetch content
        may no longer be reachable.  Returns ``(scanned, embedded)``.
        """
        self.ensure_collection()
        if self._vectors_disabled or not self._probe_chain_dims():
            log.warning("Cannot re-embed — deep memory has no usable vector provider")
            return (0, 0)

        client = self._get_client()
        resp = client.get(
            f"{self._base_url}/collections/{self._collection_name}/documents/export",
            params={"include_fields": "id,text,embedding"},
            timeout=httpx.Timeout(300.0, connect=5.0),
        )
        resp.raise_for_status()

        pending: list[dict[str, Any]] = []
        scanned = 0
        embedded = 0

        def _flush() -> int:
            if not pending:
                return 0
            vectors = self._embed([p["text"] for p in pending])
            if not vectors:
                return 0
            # action=update patches just this field, leaving text, summaries
            # and timestamps untouched.
            body = "\n".join(
                json.dumps({"id": p["id"], "embedding": vec}, ensure_ascii=False)
                for p, vec in zip(pending, vectors)
            )
            r = client.post(
                f"{self._base_url}/collections/{self._collection_name}/documents/import",
                params={"action": "update"},
                content=body,
                headers={
                    "X-TYPESENSE-API-KEY": self._api_key,
                    "Content-Type": "text/plain",
                },
            )
            r.raise_for_status()
            return sum(
                1
                for line in r.text.strip().splitlines()
                if line.strip() and json.loads(line).get("success", False)
            )

        for line in resp.text.splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            scanned += 1
            if doc.get("embedding") or not doc.get("text"):
                continue
            pending.append({"id": doc["id"], "text": doc["text"]})
            if len(pending) >= batch_size:
                embedded += _flush()
                pending = []
        embedded += _flush()

        log.info("Deep memory re-embed complete", scanned=scanned, embedded=embedded)
        return (scanned, embedded)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_score(hit: dict[str, Any]) -> float:
        """Collapse Typesense's three scoring shapes into one 0..1 number.

        Typesense reports relevance on a different scale depending on which
        halves of the search actually ran, and the scales are not comparable:

            hybrid (q + vector_query) -> hybrid_search_info.rank_fusion_score,
                                         already 0..1 and already what
                                         Typesense ranked by
            vector only (q=*)         -> vector_distance, cosine, 0..2,
                                         lower is better
            text only                 -> text_match, an unbounded ~19-digit int

        The previous implementation did ``max(text_match, 1/(1+distance))``,
        which picked ``text_match`` every single time — a 19-digit integer
        dwarfs anything <= 1 — and then handed it to the LLM formatted as
        ``score=1157451470635796601.00``.  Prefer Typesense's own fused score
        and only synthesise one when it is absent.
        """
        # Deliberately NOT hybrid_search_info.rank_fusion_score.  That number
        # is positional, not absolute — measured against a live collection it
        # decays as 0.3 * (1/rank), so the top hit scores 1.0 whether or not it
        # is any good and the third scores 0.10 whether or not it is excellent.
        # Filtering on it would just re-implement max_results.  Both signals
        # below are absolute: they describe THIS hit against THIS query,
        # independent of what else happened to be in the collection.
        best = 0.0

        # Keyword coverage: how much of the query this document actually
        # contains.  Precise when it fires, so it wins ties.
        info = hit.get("text_match_info", {}) or {}
        matched = int(info.get("tokens_matched", 0) or 0)
        missed = int(info.get("num_tokens_dropped", 0) or 0)
        if matched + missed:
            best = matched / (matched + missed)
        elif hit.get("text_match"):
            best = 1.0

        # Cosine similarity, recovered from Typesense's 0..2 distance.
        distance = hit.get("vector_distance")
        if distance is not None:
            best = max(best, max(0.0, 1.0 - float(distance)))

        return best

    def _passes_relevance(self, score: float, floor: float) -> bool:
        """Reject hits too weak to be worth a slot in the prompt.

        Some floor is necessary: ANN search faithfully returns k neighbours no
        matter how far away they are, so without one every prompt would carry
        deep memory's k nearest strangers.

        Be honest about how much this floor can do, though.  Measured against
        the default provider (model2vec ``potion-base-8M``) on a labelled set
        of related/unrelated query-document pairs, the two distributions
        OVERLAP: related similarities ran 0.13..0.59, unrelated ran -0.11..0.14.
        No threshold separates them cleanly, because a static bag-of-embeddings
        model ranks well but is not calibrated for absolute similarity.  So the
        default floor is deliberately permissive — it clears the obvious junk
        and nothing more.  A true transformer embedder (``nomic-embed-text``
        via Ollama, or ``litellm``) separates the classes far better and is
        what makes a tighter floor worth setting.
        """
        return score >= floor

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        filter_by: str = "",
        vector_query: str = "",
        min_score: float | None = None,
        owner_id: str = "",
    ) -> list[DeepMemoryResult]:
        """Hybrid search (BM25 + optional vector) over deep memory.

        *owner_id* scopes the search to one tenant.  It is ANDed onto any
        caller-supplied *filter_by* rather than replacing it, so a caller
        cannot widen its own scope by passing a filter of its own.
        """
        self.ensure_collection()
        client = self._get_client()
        if owner_id:
            scope = f"owner_id:={self.escape_filter_value(owner_id)}"
            filter_by = f"({filter_by}) && {scope}" if filter_by else scope

        params: dict[str, Any] = {
            "collection": self._collection_name,
            "q": query or "*",
            "query_by": "text",
            "per_page": min(max_results, 250),
            # Never ship the raw vectors back — at 256 dims they dwarf the
            # text we actually came for.
            "exclude_fields": "embedding",
        }
        if filter_by:
            params["filter_by"] = filter_by

        # Auto-generate vector query from embedding chain if available.
        if not vector_query and query and query != "*":
            embeddings = self._embed([query])
            if embeddings and embeddings[0]:
                vec_str = ",".join(f"{v:.6f}" for v in embeddings[0])
                vector_query = f"embedding:([{vec_str}], k:{min(max_results * 10, 200)})"

        if vector_query:
            params["vector_query"] = vector_query

        # POST /multi_search rather than GET /documents/search: Typesense caps
        # a query string at 4000 chars, and an inlined vector spends ~21 of
        # them per dimension, so a GET blows the cap at roughly 190 dims —
        # below even the 256 our default local provider emits.  The GET path
        # only ever worked here because the vector half was silently disabled.
        resp = client.post(
            f"{self._base_url}/multi_search",
            json={"searches": [params]},
        )
        resp.raise_for_status()
        payload = resp.json()
        data = (payload.get("results") or [{}])[0]
        if data.get("error"):
            log.warning(
                "Deep memory search rejected by Typesense",
                error=str(data.get("error"))[:300],
            )
            return []

        floor = self._min_score if min_score is None else float(min_score)
        results: list[DeepMemoryResult] = []
        dropped = 0
        for hit in data.get("hits", []):
            doc = hit.get("document", {})
            text_score = float(hit.get("text_match", 0))
            vector_score = float(hit.get("vector_distance", 0))
            combined = self._normalize_score(hit)
            if not self._passes_relevance(combined, floor):
                dropped += 1
                continue
            results.append(
                DeepMemoryResult(
                    doc_id=doc.get("doc_id", doc.get("id", "?")),
                    source=doc.get("source", ""),
                    reference=doc.get("reference", ""),
                    path=doc.get("path", ""),
                    chunk_index=int(doc.get("chunk_index", 0)),
                    start_line=int(doc.get("start_line", 0)),
                    end_line=int(doc.get("end_line", 0)),
                    snippet=doc.get("text", ""),
                    score=combined,
                    text_score=text_score,
                    vector_score=vector_score,
                    updated_at=int(doc.get("updated_at", 0)),
                    text_l1=doc.get("text_l1", ""),
                    text_l2=doc.get("text_l2", ""),
                )
            )
        if dropped:
            log.debug(
                "Deep memory hits below the relevance floor",
                dropped=dropped,
                kept=len(results),
                floor=floor,
            )
        return results

    # ------------------------------------------------------------------
    # Context note (for LLM injection)
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_layer_text(
        layer: str, *, text: str, text_l1: str, text_l2: str,
    ) -> str:
        """Return the best available text for the requested layer, falling back to deeper layers."""
        if layer == "l1":
            return text_l1 or text_l2 or text
        if layer == "l2":
            return text_l2 or text
        return text  # l3 (default)

    def build_context_note(
        self,
        query: str,
        *,
        max_items: int = 5,
        max_snippet_chars: int = 400,
        layer: str = "l3",
        min_score: float | None = None,
    ) -> tuple[str, str]:
        """Build a context note for LLM prompt injection.

        *layer* controls the snippet granularity:
        ``"l1"`` = one-liner, ``"l2"`` = summary, ``"l3"`` = full text (default).

        Returns ``(note, debug_block)`` — same contract as
        ``SemanticMemoryIndex.build_context_note()``.
        """
        if not query or not query.strip():
            return "", ""
        try:
            results = self.search(
                query, max_results=max_items, min_score=min_score
            )
        except Exception as exc:
            log.debug("Deep memory search for context note failed", error=str(exc))
            return "", ""

        if not results:
            return "", ""

        lines: list[str] = ["Deep memory matches (long-term archive):"]
        debug_lines: list[str] = ["[deep_memory_context]"]
        for r in results[:max_items]:
            snippet = self._pick_layer_text(
                layer, text=r.snippet, text_l1=r.text_l1, text_l2=r.text_l2,
            )
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "..."
            loc = r.path or r.reference or r.doc_id
            if r.start_line:
                loc += f":{r.start_line}"
            source_tag = f"[{r.source}]" if r.source else ""
            if r.updated_at:
                try:
                    created = datetime.fromtimestamp(int(r.updated_at), tz=UTC).isoformat()
                except (ValueError, OSError, OverflowError):
                    created = "unknown"
            else:
                created = "unknown"
            lines.append(
                f"- {source_tag} {loc} (score={r.score:.2f}, created={created}) {snippet}"
            )
            debug_lines.append(
                f"  doc_id={r.doc_id} source={r.source} ref={r.reference} "
                f"score={r.score:.2f} text={r.text_score:.0f} vec={r.vector_score:.4f}"
            )
        note = "\n".join(lines)
        debug = "\n".join(debug_lines)
        return note, debug

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to *doc_id*. Returns count deleted."""
        self.ensure_collection()
        client = self._get_client()
        resp = client.delete(
            f"{self._base_url}/collections/{self._collection_name}/documents",
            params={"filter_by": f"doc_id:={doc_id}"},
        )
        resp.raise_for_status()
        return int(resp.json().get("num_deleted", 0))

    @staticmethod
    def escape_filter_value(value: str) -> str:
        """Quote a value for a Typesense ``filter_by`` expression.

        VFS references contain ``:`` and ``/`` and may contain spaces or commas
        — all of which are structural in Typesense's filter grammar.  Backticks
        quote the literal; any backtick inside the value is stripped, since
        there is no escape for it and a path containing one is not worth
        risking a malformed filter over.
        """
        return "`" + str(value).replace("`", "") + "`"

    def stored_hash(self, reference: str, *, owner_id: str = "") -> str:
        """Return the ``content_hash`` already indexed for *reference*, or ``""``.

        The cheap half of staleness handling: if this equals the hash of the
        bytes on disk, re-indexing is a no-op and the caller can skip chunking,
        embedding and summarising entirely.
        """
        self.ensure_collection()
        filters = [f"reference:={self.escape_filter_value(reference)}"]
        if owner_id:
            filters.append(f"owner_id:={self.escape_filter_value(owner_id)}")
        try:
            resp = self._get_client().post(
                f"{self._base_url}/multi_search",
                json={
                    "searches": [
                        {
                            "collection": self._collection_name,
                            "q": "*",
                            "query_by": "text",
                            "filter_by": " && ".join(filters),
                            "per_page": 1,
                            "include_fields": "content_hash",
                        }
                    ]
                },
            )
            resp.raise_for_status()
            hits = (resp.json().get("results") or [{}])[0].get("hits") or []
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("Deep memory hash lookup failed", error=str(exc))
            return ""
        if not hits:
            return ""
        return str(hits[0].get("document", {}).get("content_hash", "") or "")

    def delete_by_reference(self, reference: str, *, owner_id: str = "") -> int:
        """Delete every chunk indexed for *reference*. Returns count deleted.

        Used both when a file is unlinked and as the first half of a re-index,
        so a shrinking document cannot leave orphaned tail chunks behind.
        """
        filters = [f"reference:={self.escape_filter_value(reference)}"]
        if owner_id:
            filters.append(f"owner_id:={self.escape_filter_value(owner_id)}")
        return self.delete_by_filter(" && ".join(filters))

    def delete_by_filter(self, filter_by: str) -> int:
        """Delete documents matching a Typesense filter expression."""
        self.ensure_collection()
        client = self._get_client()
        resp = client.delete(
            f"{self._base_url}/collections/{self._collection_name}/documents",
            params={"filter_by": filter_by},
        )
        resp.raise_for_status()
        return int(resp.json().get("num_deleted", 0))

    def clear_all(self) -> int:
        """Drop and recreate the Typesense collection, removing all documents.

        Returns the number of documents that were in the collection.
        """
        client = self._get_client()
        count = 0
        try:
            resp = client.delete(
                f"{self._base_url}/collections/{self._collection_name}",
            )
            if resp.status_code == 200:
                count = int(resp.json().get("num_documents", 0))
            elif resp.status_code != 404:
                resp.raise_for_status()
        except Exception as exc:
            log.warning("Deep memory collection drop failed", error=str(exc))
        self._collection_ensured = False
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None
