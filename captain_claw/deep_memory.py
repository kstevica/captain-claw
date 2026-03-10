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


# ---------------------------------------------------------------------------
# Collection schema
# ---------------------------------------------------------------------------

_COLLECTION_SCHEMA_TEMPLATE: dict[str, Any] = {
    "fields": [
        {"name": "doc_id", "type": "string", "facet": True},
        {"name": "source", "type": "string", "facet": True},
        {"name": "reference", "type": "string", "facet": True},
        {"name": "path", "type": "string"},
        {"name": "text", "type": "string"},
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
        embedding_dims: int = 1536,
        auto_embed: bool = True,
        chunk_chars: int = 1_400,
        chunk_overlap_chars: int = 200,
        embedding_chain: Any | None = None,
    ) -> None:
        self._base_url = f"{protocol}://{host}:{port}"
        self._api_key = api_key
        self._collection_name = collection_name
        self._embedding_dims = embedding_dims
        self._auto_embed = auto_embed
        self._chunk_chars = chunk_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._embedding_chain = embedding_chain
        self._client: httpx.Client | None = None
        self._collection_ensured = False

    @property
    def collection_name(self) -> str:
        """The Typesense collection name used for deep memory."""
        return self._collection_name

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

    def ensure_collection(self) -> None:
        """Create the deep memory collection if it doesn't exist."""
        if self._collection_ensured:
            return
        client = self._get_client()
        try:
            resp = client.get(
                f"{self._base_url}/collections/{self._collection_name}"
            )
            if resp.status_code == 200:
                self._collection_ensured = True
                return
        except httpx.HTTPError:
            pass

        schema: dict[str, Any] = {
            "name": self._collection_name,
            **_COLLECTION_SCHEMA_TEMPLATE,
        }
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
    ) -> int:
        """Index a single document (auto-chunked). Returns chunk count."""
        self.ensure_collection()
        chunks = _chunk_text(
            text,
            chunk_chars=self._chunk_chars,
            chunk_overlap_chars=self._chunk_overlap_chars,
        )
        if not chunks:
            return 0

        now = int(time.time())
        docs: list[dict[str, Any]] = []
        texts_for_embedding: list[str] = []

        for chunk in chunks:
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
            if tags:
                doc["tags"] = tags
            docs.append(doc)
            texts_for_embedding.append(chunk["text"])

        # Compute embeddings if available.
        embeddings = self._embed(texts_for_embedding)
        if embeddings:
            for doc_dict, vec in zip(docs, embeddings):
                doc_dict["embedding"] = vec

        return self._upsert_batch(docs)

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
        if not self._auto_embed or not texts:
            return []
        chain = self._embedding_chain
        if chain is None or not getattr(chain, "enabled", False):
            return []
        try:
            _key, vectors = chain.embed_batch(texts)
            # Validate dimensions match the collection schema.
            # Fallback providers (ollama, local_hash) may produce
            # different dimensions than what the collection expects.
            expected = self._embedding_dims
            if expected and vectors:
                actual = len(vectors[0]) if vectors[0] else 0
                if actual != expected:
                    log.warning(
                        "Embedding dimension mismatch — discarding vectors",
                        expected=expected,
                        actual=actual,
                        provider=_key,
                    )
                    return []
            return vectors
        except Exception as exc:
            log.debug("Deep memory embedding failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        filter_by: str = "",
        vector_query: str = "",
    ) -> list[DeepMemoryResult]:
        """Hybrid search (BM25 + optional vector) over deep memory."""
        self.ensure_collection()
        client = self._get_client()

        params: dict[str, Any] = {
            "q": query or "*",
            "query_by": "text",
            "per_page": min(max_results, 250),
        }
        if filter_by:
            params["filter_by"] = filter_by

        # Auto-generate vector query from embedding chain if available.
        if not vector_query and query and query != "*":
            embeddings = self._embed([query])
            if embeddings and embeddings[0]:
                vec_str = ",".join(str(v) for v in embeddings[0])
                vector_query = f"embedding:([{vec_str}], k:{min(max_results * 10, 200)})"

        if vector_query:
            params["vector_query"] = vector_query

        resp = client.get(
            f"{self._base_url}/collections/{self._collection_name}/documents/search",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[DeepMemoryResult] = []
        for hit in data.get("hits", []):
            doc = hit.get("document", {})
            text_score = float(hit.get("text_match", 0))
            vector_score = float(hit.get("vector_distance", 0))
            # Combine scores (higher is better for text_match,
            # lower is better for vector_distance → invert).
            combined = text_score
            if vector_score > 0:
                combined = max(combined, 1.0 / (1.0 + vector_score))
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
                )
            )
        return results

    # ------------------------------------------------------------------
    # Context note (for LLM injection)
    # ------------------------------------------------------------------

    def build_context_note(
        self,
        query: str,
        *,
        max_items: int = 5,
        max_snippet_chars: int = 400,
    ) -> tuple[str, str]:
        """Build a context note for LLM prompt injection.

        Returns ``(note, debug_block)`` — same contract as
        ``SemanticMemoryIndex.build_context_note()``.
        """
        if not query or not query.strip():
            return "", ""
        try:
            results = self.search(query, max_results=max_items)
        except Exception as exc:
            log.debug("Deep memory search for context note failed", error=str(exc))
            return "", ""

        if not results:
            return "", ""

        lines: list[str] = ["Deep memory matches (long-term archive):"]
        debug_lines: list[str] = ["[deep_memory_context]"]
        for r in results[:max_items]:
            snippet = r.snippet
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "..."
            loc = r.path or r.reference or r.doc_id
            if r.start_line:
                loc += f":{r.start_line}"
            source_tag = f"[{r.source}]" if r.source else ""
            lines.append(
                f"- {source_tag} {loc} (score={r.score:.2f}) {snippet}"
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None
