"""Semantic memory index with hybrid retrieval and background sync."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Protocol

import httpx

from captain_claw.logging import get_logger


log = get_logger(__name__)

_DEFAULT_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".csv",
    ".sh",
    ".bash",
    ".zsh",
    ".xml",
    ".html",
    ".css",
}


@dataclass
class SemanticMemoryResult:
    """One semantic-memory hit."""

    chunk_id: str
    source: str
    reference: str
    path: str
    start_line: int
    end_line: int
    snippet: str
    score: float
    text_score: float
    vector_score: float
    updated_at: str


@dataclass
class _Document:
    source: str
    reference: str
    path: str
    signature: str
    text: str
    updated_at: str


@dataclass
class _Chunk:
    chunk_id: str
    chunk_index: int
    start_line: int
    end_line: int
    text: str
    updated_at: str


class _EmbeddingProvider(Protocol):
    provider_id: str
    model: str

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized embedding per text."""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_embedding(values: Iterable[float]) -> list[float]:
    vector = [float(v) if isinstance(v, (float, int)) and math.isfinite(float(v)) else 0.0 for v in values]
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return vector
    return [v / norm for v in vector]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    score = sum(x * y for x, y in zip(a, b, strict=False))
    if not math.isfinite(score):
        return 0.0
    return max(-1.0, min(1.0, score))


def _tokenize_fts(text: str) -> list[str]:
    return [token.strip() for token in re.findall(r"[\w]+", text.lower()) if token.strip()]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _build_fts_query(query: str) -> str | None:
    raw_tokens = _tokenize_fts(query)
    tokens = [token for token in raw_tokens if len(token) >= 3 and token not in _STOPWORDS]
    if not tokens:
        tokens = [token for token in raw_tokens if len(token) >= 2]
    if not tokens:
        return None
    quoted = [f'"{token.replace(chr(34), "")}"' for token in tokens]
    return " OR ".join(quoted)


def _parse_iso_to_timestamp(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return None


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


class _LocalHashEmbeddingProvider:
    provider_id = "local_hash"
    model = "sha1-bow-256"

    def __init__(self, dimensions: int = 256):
        self._dimensions = max(64, int(dimensions))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            bucket = [0.0] * self._dimensions
            tokens = _tokenize_fts(text)
            if not tokens:
                embeddings.append(bucket)
                continue
            for token in tokens:
                digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).digest()
                idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % self._dimensions
                sign = -1.0 if digest[4] % 2 else 1.0
                bucket[idx] += sign
            embeddings.append(_normalize_embedding(bucket))
        return embeddings


class _OllamaEmbeddingProvider:
    provider_id = "ollama"

    def __init__(self, model: str, base_url: str, timeout_seconds: int):
        self.model = str(model).strip() or "nomic-embed-text"
        self._base_url = str(base_url).rstrip("/") or "http://127.0.0.1:11434"
        self._timeout = max(3, int(timeout_seconds))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self._timeout) as client:
            for text in texts:
                response = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                payload = response.json()
                embedding = payload.get("embedding")
                if not isinstance(embedding, list):
                    raise ValueError("Ollama embeddings response missing list 'embedding'")
                vectors.append(_normalize_embedding([float(v) for v in embedding]))
        return vectors


class _LiteLLMEmbeddingProvider:
    provider_id = "litellm"

    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        self.model = str(model).strip()
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip()
        if not self.model:
            raise ValueError("LiteLLM embedding model must be configured")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        from litellm import embedding as litellm_embedding

        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        response = litellm_embedding(**kwargs)
        data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
        if not isinstance(data, list):
            raise ValueError("LiteLLM embedding response missing data")
        vectors: list[list[float]] = []
        for row in data:
            raw = row.get("embedding") if isinstance(row, dict) else getattr(row, "embedding", None)
            if not isinstance(raw, list):
                raise ValueError("LiteLLM embedding row missing list embedding")
            vectors.append(_normalize_embedding([float(v) for v in raw]))
        return vectors


class _EmbeddingProviderChain:
    def __init__(self, providers: list[_EmbeddingProvider]):
        self._providers = providers
        self._active = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._providers)

    @property
    def active_provider_key(self) -> str:
        if not self._providers:
            return ""
        provider = self._providers[self._active]
        return f"{provider.provider_id}:{provider.model}"

    def embed_batch(self, texts: list[str]) -> tuple[str, list[list[float]]]:
        if not self._providers:
            raise RuntimeError("No embedding providers configured")
        providers = list(self._providers)
        with self._lock:
            start = self._active
        errors: list[str] = []
        for offset in range(len(providers)):
            idx = (start + offset) % len(providers)
            provider = providers[idx]
            try:
                vectors = provider.embed_batch(texts)
                with self._lock:
                    self._active = idx
                return f"{provider.provider_id}:{provider.model}", vectors
            except Exception as exc:
                errors.append(f"{provider.provider_id}: {exc}")
                continue
        raise RuntimeError("All embedding providers failed: " + " | ".join(errors))


class SemanticMemoryIndex:
    """SQLite-backed semantic memory index with hybrid retrieval."""

    def __init__(
        self,
        *,
        db_path: Path,
        session_db_path: Path,
        workspace_path: Path,
        index_workspace: bool = True,
        index_sessions: bool = True,
        max_workspace_files: int = 400,
        max_file_bytes: int = 262_144,
        include_extensions: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        chunk_chars: int = 1_400,
        chunk_overlap_chars: int = 200,
        cache_ttl_seconds: int = 45,
        stale_after_seconds: int = 120,
        auto_sync_on_search: bool = True,
        max_results: int = 6,
        candidate_limit: int = 80,
        min_score: float = 0.1,
        vector_weight: float = 0.65,
        text_weight: float = 0.35,
        temporal_decay_enabled: bool = True,
        temporal_half_life_days: float = 21.0,
        embedding_chain: _EmbeddingProviderChain | None = None,
    ):
        self.db_path = Path(db_path).expanduser()
        self.session_db_path = Path(session_db_path).expanduser()
        self.workspace_path = Path(workspace_path).resolve()
        self.index_workspace = bool(index_workspace)
        self.index_sessions = bool(index_sessions)
        self.max_workspace_files = max(1, int(max_workspace_files))
        self.max_file_bytes = max(1024, int(max_file_bytes))
        self.include_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (include_extensions or sorted(_DEFAULT_TEXT_EXTENSIONS))
        }
        self.exclude_dirs = {
            value.strip().lower()
            for value in (exclude_dirs or [".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv"])
            if value.strip()
        }
        self.chunk_chars = max(300, int(chunk_chars))
        self.chunk_overlap_chars = max(0, min(self.chunk_chars // 2, int(chunk_overlap_chars)))
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self.stale_after_seconds = max(5, int(stale_after_seconds))
        self.auto_sync_on_search = bool(auto_sync_on_search)
        self.max_results = max(1, int(max_results))
        self.candidate_limit = max(self.max_results, int(candidate_limit))
        self.min_score = float(min_score)
        self.vector_weight = max(0.0, float(vector_weight))
        self.text_weight = max(0.0, float(text_weight))
        if self.vector_weight == 0 and self.text_weight == 0:
            self.vector_weight = 0.65
            self.text_weight = 0.35
        self.temporal_decay_enabled = bool(temporal_decay_enabled)
        self.temporal_half_life_days = max(1.0, float(temporal_half_life_days))
        self.embedding_chain = embedding_chain or _EmbeddingProviderChain([])
        self._sync_lock = threading.Lock()
        self._db_lock = threading.RLock()
        self._sync_running = False
        self._dirty = False
        self._last_sync_started: float = 0.0
        self._last_sync_completed: float = 0.0
        self._cache: dict[str, tuple[float, list[SemanticMemoryResult]]] = {}
        self._conn: sqlite3.Connection | None = None
        self._closed = False
        self._ensure_db()
        self.schedule_sync("startup")

    def close(self) -> None:
        """Close SQLite resources."""
        self._closed = True
        with self._db_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def schedule_sync(self, reason: str = "manual") -> None:
        """Trigger background sync (non-blocking)."""
        if self._closed:
            return
        self._dirty = True
        with self._sync_lock:
            if self._sync_running:
                return
            self._sync_running = True
            thread = threading.Thread(
                target=self._sync_worker,
                name=f"semantic-memory-sync:{reason}",
                daemon=True,
            )
            thread.start()

    def upsert_text(
        self,
        *,
        source: str,
        reference: str,
        path: str,
        text: str,
        updated_at: str | None = None,
    ) -> None:
        """Insert/update one virtual document, useful for tests and ad-hoc memory."""
        now_iso = updated_at or _utcnow_iso()
        doc = _Document(
            source=str(source).strip() or "manual",
            reference=str(reference).strip() or f"manual:{_hash_text(path + text)}",
            path=str(path).strip() or "manual",
            signature=_hash_text(text),
            text=str(text),
            updated_at=now_iso,
        )
        with self._db_lock:
            self._upsert_document(doc)
            self._conn_or_raise().commit()
            self._clear_cache()

    def search(self, query: str, max_results: int | None = None) -> list[SemanticMemoryResult]:
        """Hybrid search across workspace + session memory."""
        cleaned = str(query or "").strip()
        if not cleaned:
            return []
        if self._closed:
            return []
        effective_max = max(1, int(max_results or self.max_results))
        key = f"{cleaned}::{effective_max}"
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return list(cached[1])

        if self.auto_sync_on_search:
            stale = (now - self._last_sync_completed) >= self.stale_after_seconds
            if stale or self._dirty:
                self.schedule_sync("search")

        keyword_hits = self._keyword_search(cleaned, limit=self.candidate_limit)
        vector_hits = self._vector_search(cleaned, limit=self.candidate_limit)
        merged = self._merge_hybrid(keyword_hits, vector_hits, max_results=effective_max)
        self._cache[key] = (time.time() + self.cache_ttl_seconds, merged)
        return list(merged)

    def build_context_note(
        self,
        query: str,
        *,
        max_items: int = 3,
        max_snippet_chars: int = 360,
    ) -> tuple[str, str]:
        """Format top semantic hits as a prompt note + debug block."""
        results = self.search(query=query, max_results=max_items)
        if not results:
            return "", "semantic_memory: no results"
        lines = ["Semantic memory matches (prior sessions + workspace):"]
        debug = [f"semantic_memory query={query!r}", f"result_count={len(results)}"]
        for item in results[:max_items]:
            snippet = re.sub(r"\s+", " ", item.snippet).strip()
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars].rstrip() + "... [truncated]"
            citation = f"{item.path}:{item.start_line}"
            lines.append(
                f"- [{item.source}] {citation} (score={item.score:.3f}) {snippet}"
            )
            debug.append(
                f"- source={item.source} reference={item.reference} path={item.path} "
                f"line={item.start_line} score={item.score:.3f} "
                f"text={item.text_score:.3f} vector={item.vector_score:.3f}"
            )
        return "\n".join(lines), "\n".join(debug)

    def _sync_worker(self) -> None:
        while True:
            if self._closed:
                with self._sync_lock:
                    self._sync_running = False
                return
            if not self._dirty:
                with self._sync_lock:
                    self._sync_running = False
                return
            self._dirty = False
            self._last_sync_started = time.time()
            try:
                self._sync_once()
                self._last_sync_completed = time.time()
            except Exception as exc:
                log.warning("Semantic memory sync failed", error=str(exc))
                self._last_sync_completed = time.time()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    reference TEXT NOT NULL,
                    path TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_documents_source_ref
                ON memory_documents(source, reference)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reference TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_chunks_doc_id
                ON memory_chunks(doc_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_chunks_source_ref
                ON memory_chunks(source, reference)
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts
                USING fts5(chunk_id, text, path, source, reference)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    provider_key TEXT NOT NULL,
                    dims INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_provider_dims
                ON memory_embeddings(provider_key, dims)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
            self._conn = conn

    def _conn_or_raise(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Semantic memory database is closed")
        return self._conn

    def _sync_once(self) -> None:
        workspace_docs = self._collect_workspace_documents() if self.index_workspace else []
        session_docs = self._collect_session_documents() if self.index_sessions else []
        with self._db_lock:
            self._sync_documents("workspace", workspace_docs)
            self._sync_documents("session", session_docs)
            self._clear_cache()

    def _collect_workspace_documents(self) -> list[_Document]:
        if not self.workspace_path.exists() or not self.workspace_path.is_dir():
            return []
        documents: list[_Document] = []
        for root, dirs, files in os.walk(self.workspace_path):
            dirs[:] = [d for d in dirs if d.strip().lower() not in self.exclude_dirs]
            for filename in files:
                if len(documents) >= self.max_workspace_files:
                    return documents
                file_path = Path(root) / filename
                try:
                    stat = file_path.stat()
                except Exception:
                    continue
                if not file_path.is_file():
                    continue
                if stat.st_size <= 0 or stat.st_size > self.max_file_bytes:
                    continue
                suffix = file_path.suffix.lower()
                if self.include_extensions and suffix not in self.include_extensions:
                    continue
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if not raw.strip():
                    continue
                rel_path = file_path.resolve().relative_to(self.workspace_path).as_posix()
                updated_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                signature = f"{stat.st_size}:{int(stat.st_mtime_ns)}"
                documents.append(
                    _Document(
                        source="workspace",
                        reference=rel_path,
                        path=rel_path,
                        signature=signature,
                        text=raw,
                        updated_at=updated_at,
                    )
                )
        return documents

    def _collect_session_documents(self) -> list[_Document]:
        if not self.session_db_path.exists():
            return []
        rows: list[tuple[str, str, str, str]] = []
        try:
            with sqlite3.connect(str(self.session_db_path)) as session_conn:
                cursor = session_conn.execute(
                    "SELECT id, name, messages, updated_at FROM sessions ORDER BY updated_at DESC"
                )
                rows = [(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in cursor.fetchall()]
        except Exception as exc:
            log.debug("Skipping session-memory sync; cannot read session db", error=str(exc))
            return []

        documents: list[_Document] = []
        for sid, name, raw_messages, updated_at in rows:
            try:
                messages = json.loads(raw_messages)
            except Exception:
                continue
            if not isinstance(messages, list):
                continue
            lines: list[str] = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower() or "unknown"
                content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
                if not content:
                    continue
                lines.append(f"[{role}] {content}")
            if not lines:
                continue
            text = "\n".join(lines)
            signature = f"{updated_at}:{len(lines)}:{_hash_text(text[:12000])}"
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-") or sid
            documents.append(
                _Document(
                    source="session",
                    reference=sid,
                    path=f"sessions/{safe_name}.txt",
                    signature=signature,
                    text=text,
                    updated_at=updated_at or _utcnow_iso(),
                )
            )
        return documents

    def _sync_documents(self, source: str, docs: list[_Document]) -> None:
        conn = self._conn_or_raise()
        existing_rows = conn.execute(
            "SELECT doc_id, reference, signature FROM memory_documents WHERE source = ?",
            (source,),
        ).fetchall()
        existing = {str(row[1]): (str(row[0]), str(row[2])) for row in existing_rows}
        seen_refs: set[str] = set()

        for doc in docs:
            seen_refs.add(doc.reference)
            prev = existing.get(doc.reference)
            if prev and prev[1] == doc.signature:
                continue
            self._upsert_document(doc)

        stale_doc_ids = [
            doc_id
            for ref, (doc_id, _signature) in existing.items()
            if ref not in seen_refs
        ]
        for doc_id in stale_doc_ids:
            self._delete_document(doc_id)

        conn.commit()

    def _upsert_document(self, doc: _Document) -> None:
        conn = self._conn_or_raise()
        doc_id = _hash_text(f"{doc.source}:{doc.reference}")
        conn.execute(
            """
            INSERT INTO memory_documents (doc_id, source, reference, path, signature, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                path=excluded.path,
                signature=excluded.signature,
                updated_at=excluded.updated_at
            """,
            (doc_id, doc.source, doc.reference, doc.path, doc.signature, doc.updated_at),
        )
        self._delete_chunks_for_doc(doc_id)
        chunks = self._chunk_document(doc_id=doc_id, text=doc.text, updated_at=doc.updated_at)
        if not chunks:
            return
        conn.executemany(
            """
            INSERT INTO memory_chunks (
                chunk_id, doc_id, source, reference, path,
                chunk_index, start_line, end_line, text, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    doc_id,
                    doc.source,
                    doc.reference,
                    doc.path,
                    chunk.chunk_index,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.updated_at,
                )
                for chunk in chunks
            ],
        )
        conn.executemany(
            """
            INSERT INTO memory_chunks_fts (chunk_id, text, path, source, reference)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.text,
                    doc.path,
                    doc.source,
                    doc.reference,
                )
                for chunk in chunks
            ],
        )
        self._upsert_embeddings_for_chunks(chunks)

    def _delete_document(self, doc_id: str) -> None:
        conn = self._conn_or_raise()
        self._delete_chunks_for_doc(doc_id)
        conn.execute("DELETE FROM memory_documents WHERE doc_id = ?", (doc_id,))

    def _delete_chunks_for_doc(self, doc_id: str) -> None:
        conn = self._conn_or_raise()
        rows = conn.execute(
            "SELECT chunk_id FROM memory_chunks WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        chunk_ids = [str(row[0]) for row in rows]
        if chunk_ids:
            conn.executemany(
                "DELETE FROM memory_chunks_fts WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
            conn.executemany(
                "DELETE FROM memory_embeddings WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
        conn.execute("DELETE FROM memory_chunks WHERE doc_id = ?", (doc_id,))

    def _chunk_document(self, *, doc_id: str, text: str, updated_at: str) -> list[_Chunk]:
        lines = text.splitlines()
        if not lines:
            return []
        chunks: list[_Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(lines):
            end = start
            used = 0
            while end < len(lines):
                line_len = len(lines[end]) + 1
                if used and used + line_len > self.chunk_chars:
                    break
                used += line_len
                end += 1
            if end <= start:
                end = min(len(lines), start + 1)
            chunk_text = "\n".join(lines[start:end]).strip()
            if chunk_text:
                chunk_id = _hash_text(f"{doc_id}:{chunk_index}:{chunk_text[:64]}")
                chunks.append(
                    _Chunk(
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        start_line=start + 1,
                        end_line=end,
                        text=chunk_text,
                        updated_at=updated_at,
                    )
                )
                chunk_index += 1
            if end >= len(lines):
                break
            overlap_lines = 0
            overlap_chars = 0
            idx = end - 1
            while idx >= start and overlap_chars < self.chunk_overlap_chars:
                overlap_chars += len(lines[idx]) + 1
                overlap_lines += 1
                idx -= 1
            start = max(start + 1, end - overlap_lines) if overlap_lines else end
        return chunks

    def _upsert_embeddings_for_chunks(self, chunks: list[_Chunk]) -> None:
        if not chunks or not self.embedding_chain.enabled:
            return
        conn = self._conn_or_raise()
        batch_size = 24
        now_iso = _utcnow_iso()
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            texts = [chunk.text for chunk in batch]
            try:
                provider_key, vectors = self.embedding_chain.embed_batch(texts)
            except Exception as exc:
                log.warning("Embedding batch failed; keeping keyword index only", error=str(exc))
                return
            if len(vectors) != len(batch):
                log.warning("Embedding provider returned mismatched batch size")
                return
            payload = []
            for chunk, vector in zip(batch, vectors, strict=False):
                payload.append(
                    (
                        chunk.chunk_id,
                        provider_key,
                        len(vector),
                        json.dumps(vector, ensure_ascii=True),
                        now_iso,
                    )
                )
            conn.executemany(
                """
                INSERT INTO memory_embeddings (chunk_id, provider_key, dims, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    provider_key=excluded.provider_key,
                    dims=excluded.dims,
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at
                """,
                payload,
            )

    def _keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        conn = self._conn_or_raise()
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []
        try:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.source,
                    c.reference,
                    c.path,
                    c.start_line,
                    c.end_line,
                    c.text,
                    c.updated_at,
                    bm25(memory_chunks_fts) AS rank
                FROM memory_chunks_fts
                JOIN memory_chunks c ON c.chunk_id = memory_chunks_fts.chunk_id
                WHERE memory_chunks_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except Exception as exc:
            log.debug("FTS search failed; fallbacking to LIKE search", error=str(exc))
            like = f"%{query.strip()}%"
            rows = conn.execute(
                """
                SELECT
                    chunk_id, source, reference, path, start_line, end_line, text, updated_at,
                    999.0 AS rank
                FROM memory_chunks
                WHERE text LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
        hits: list[dict[str, Any]] = []
        for row in rows:
            rank = float(row[8]) if isinstance(row[8], (int, float)) and math.isfinite(float(row[8])) else 999.0
            hits.append(
                {
                    "chunk_id": str(row[0]),
                    "source": str(row[1]),
                    "reference": str(row[2]),
                    "path": str(row[3]),
                    "start_line": int(row[4]),
                    "end_line": int(row[5]),
                    "snippet": str(row[6]),
                    "updated_at": str(row[7]),
                    "text_score": 1.0 / (1.0 + max(0.0, rank)),
                }
            )
        return hits

    def _vector_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.embedding_chain.enabled:
            return []
        conn = self._conn_or_raise()
        try:
            provider_key, vectors = self.embedding_chain.embed_batch([query])
        except Exception as exc:
            log.debug("Query embedding failed; using keyword-only fallback", error=str(exc))
            return []
        if not vectors:
            return []
        query_vec = vectors[0]
        dims = len(query_vec)
        if dims <= 0:
            return []
        rows = conn.execute(
            """
            SELECT
                e.chunk_id,
                c.source,
                c.reference,
                c.path,
                c.start_line,
                c.end_line,
                c.text,
                c.updated_at,
                e.embedding
            FROM memory_embeddings e
            JOIN memory_chunks c ON c.chunk_id = e.chunk_id
            WHERE e.provider_key = ? AND e.dims = ?
            """,
            (provider_key, dims),
        ).fetchall()
        if not rows:
            # Index may still be stale for the active provider.
            self.schedule_sync("vector_provider_mismatch")
            return []
        scored: list[dict[str, Any]] = []
        for row in rows:
            try:
                embedding = json.loads(str(row[8]))
            except Exception:
                continue
            if not isinstance(embedding, list):
                continue
            vector = [float(v) for v in embedding]
            score = _cosine_similarity(query_vec, vector)
            if not math.isfinite(score):
                continue
            scored.append(
                {
                    "chunk_id": str(row[0]),
                    "source": str(row[1]),
                    "reference": str(row[2]),
                    "path": str(row[3]),
                    "start_line": int(row[4]),
                    "end_line": int(row[5]),
                    "snippet": str(row[6]),
                    "updated_at": str(row[7]),
                    "vector_score": max(0.0, score),
                }
            )
        scored.sort(key=lambda item: item["vector_score"], reverse=True)
        return scored[:limit]

    def _merge_hybrid(
        self,
        keyword_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        *,
        max_results: int,
    ) -> list[SemanticMemoryResult]:
        by_id: dict[str, dict[str, Any]] = {}
        for item in keyword_hits:
            by_id[item["chunk_id"]] = {
                **item,
                "text_score": float(item.get("text_score", 0.0)),
                "vector_score": 0.0,
            }
        for item in vector_hits:
            existing = by_id.get(item["chunk_id"])
            if existing is None:
                by_id[item["chunk_id"]] = {
                    **item,
                    "text_score": 0.0,
                    "vector_score": float(item.get("vector_score", 0.0)),
                }
                continue
            existing["vector_score"] = float(item.get("vector_score", 0.0))
            snippet = str(item.get("snippet", "")).strip()
            if snippet:
                existing["snippet"] = snippet

        now = time.time()
        merged: list[SemanticMemoryResult] = []
        for payload in by_id.values():
            text_score = float(payload.get("text_score", 0.0))
            vector_score = float(payload.get("vector_score", 0.0))
            score = (self.vector_weight * vector_score) + (self.text_weight * text_score)
            if self.temporal_decay_enabled:
                timestamp = _parse_iso_to_timestamp(str(payload.get("updated_at", "")))
                if timestamp is not None:
                    age_days = max(0.0, (now - timestamp) / 86400.0)
                    decay_lambda = math.log(2) / self.temporal_half_life_days
                    score *= math.exp(-decay_lambda * age_days)
            if score < self.min_score:
                continue
            merged.append(
                SemanticMemoryResult(
                    chunk_id=str(payload.get("chunk_id", "")),
                    source=str(payload.get("source", "")),
                    reference=str(payload.get("reference", "")),
                    path=str(payload.get("path", "")),
                    start_line=int(payload.get("start_line", 1)),
                    end_line=int(payload.get("end_line", 1)),
                    snippet=str(payload.get("snippet", "")),
                    score=score,
                    text_score=text_score,
                    vector_score=vector_score,
                    updated_at=str(payload.get("updated_at", "")),
                )
            )
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:max_results]

    def _clear_cache(self) -> None:
        self._cache.clear()


def _build_embedding_chain(memory_cfg: Any) -> _EmbeddingProviderChain:
    providers: list[_EmbeddingProvider] = []
    cfg = getattr(memory_cfg, "embeddings", None)
    if cfg is None:
        return _EmbeddingProviderChain([_LocalHashEmbeddingProvider()])

    provider_mode = str(getattr(cfg, "provider", "auto")).strip().lower()
    request_timeout_seconds = int(getattr(cfg, "request_timeout_seconds", 4))
    litellm_model = str(getattr(cfg, "litellm_model", "text-embedding-3-small")).strip()
    litellm_api_key = str(getattr(cfg, "litellm_api_key", "")).strip()
    litellm_base_url = str(getattr(cfg, "litellm_base_url", "")).strip()
    ollama_model = str(getattr(cfg, "ollama_model", "nomic-embed-text")).strip()
    ollama_base_url = str(getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")).strip()
    fallback_to_local_hash = bool(getattr(cfg, "fallback_to_local_hash", True))

    def maybe_add_litellm() -> None:
        try:
            providers.append(
                _LiteLLMEmbeddingProvider(
                    model=litellm_model,
                    api_key=litellm_api_key,
                    base_url=litellm_base_url,
                )
            )
        except Exception as exc:
            log.debug("LiteLLM embedding provider unavailable", error=str(exc))

    def add_ollama() -> None:
        providers.append(
            _OllamaEmbeddingProvider(
                model=ollama_model,
                base_url=ollama_base_url,
                timeout_seconds=request_timeout_seconds,
            )
        )

    if provider_mode == "litellm":
        maybe_add_litellm()
    elif provider_mode == "ollama":
        add_ollama()
    elif provider_mode == "none":
        providers = []
    else:
        maybe_add_litellm()
        add_ollama()

    if fallback_to_local_hash or not providers:
        providers.append(_LocalHashEmbeddingProvider())
    return _EmbeddingProviderChain(providers)


def create_semantic_memory_index(
    *,
    memory_cfg: Any,
    session_db_path: Path,
    workspace_path: Path,
) -> SemanticMemoryIndex:
    """Create semantic memory index from configuration."""
    db_path = Path(str(getattr(memory_cfg, "path", "~/.captain-claw/memory.db"))).expanduser()
    include_extensions = list(getattr(memory_cfg, "include_extensions", [])) or sorted(_DEFAULT_TEXT_EXTENSIONS)
    exclude_dirs = list(getattr(memory_cfg, "exclude_dirs", []))
    search_cfg = getattr(memory_cfg, "search", None)
    return SemanticMemoryIndex(
        db_path=db_path,
        session_db_path=session_db_path,
        workspace_path=workspace_path,
        index_workspace=bool(getattr(memory_cfg, "index_workspace", True)),
        index_sessions=bool(getattr(memory_cfg, "index_sessions", True)),
        max_workspace_files=int(getattr(memory_cfg, "max_workspace_files", 400)),
        max_file_bytes=int(getattr(memory_cfg, "max_file_bytes", 262_144)),
        include_extensions=include_extensions,
        exclude_dirs=exclude_dirs,
        chunk_chars=int(getattr(memory_cfg, "chunk_chars", 1_400)),
        chunk_overlap_chars=int(getattr(memory_cfg, "chunk_overlap_chars", 200)),
        cache_ttl_seconds=int(getattr(memory_cfg, "cache_ttl_seconds", 45)),
        stale_after_seconds=int(getattr(memory_cfg, "stale_after_seconds", 120)),
        auto_sync_on_search=bool(getattr(memory_cfg, "auto_sync_on_search", True)),
        max_results=int(getattr(search_cfg, "max_results", 6)) if search_cfg else 6,
        candidate_limit=int(getattr(search_cfg, "candidate_limit", 80)) if search_cfg else 80,
        min_score=float(getattr(search_cfg, "min_score", 0.1)) if search_cfg else 0.1,
        vector_weight=float(getattr(search_cfg, "vector_weight", 0.65)) if search_cfg else 0.65,
        text_weight=float(getattr(search_cfg, "text_weight", 0.35)) if search_cfg else 0.35,
        temporal_decay_enabled=bool(getattr(search_cfg, "temporal_decay_enabled", True)) if search_cfg else True,
        temporal_half_life_days=float(getattr(search_cfg, "temporal_half_life_days", 21.0)) if search_cfg else 21.0,
        embedding_chain=_build_embedding_chain(memory_cfg),
    )
