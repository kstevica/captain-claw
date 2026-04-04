"""Typesense tool — locked-down deep memory interface.

The LLM can only: index text, search text, delete by ID/filter.
All operations use the configured deep memory collection — the LLM
cannot create collections, choose collection names, or define schemas.

Indexing goes through ``DeepMemoryIndex.index_document()`` which handles
chunking, doc_id generation, timestamps, and (optionally) embeddings.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, TYPE_CHECKING

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

if TYPE_CHECKING:
    from captain_claw.deep_memory import DeepMemoryIndex

log = get_logger(__name__)


class TypesenseTool(Tool):
    """Index, search, and delete documents in deep memory (Typesense)."""

    name = "typesense"
    description = (
        "Index, search, and delete documents in deep memory (Typesense). "
        "All operations use the configured deep memory collection — you cannot "
        "create or choose collections. "
        "Actions: index (store text with auto-chunking and embedding), "
        "search (hybrid keyword + vector), "
        "delete (remove documents by ID or filter)."
    )
    timeout_seconds = 60.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["index", "search", "delete"],
                "description": "The action to perform.",
            },
            "text": {
                "type": "string",
                "description": (
                    "Text content to index (for 'index' action). "
                    "Will be auto-chunked and embedded. "
                    "Use 'file_path' instead for large files."
                ),
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Absolute path to a file to index (for 'index' action). "
                    "The file is read directly from disk — use this for large "
                    "files instead of passing text. Supports .txt, .md, .csv, "
                    ".json, .log, .xml, .html, and other text files."
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "Source label for indexed content (for 'index'). "
                    "E.g. 'manual', 'web_fetch', 'pdf', 'scale_loop'. "
                    "Default: 'manual'."
                ),
            },
            "reference": {
                "type": "string",
                "description": (
                    "Reference identifier (for 'index'). "
                    "E.g. a URL, file path, or descriptive label."
                ),
            },
            "tags": {
                "type": "string",
                "description": (
                    "Comma-separated tags (for 'index'). "
                    "E.g. 'finance, series-a, 2024'."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query text (for 'search' action).",
            },
            "filter_by": {
                "type": "string",
                "description": (
                    "Typesense filter expression (for 'search' or 'delete'). "
                    "E.g. 'source:=web_fetch' or 'tags:=[finance]'."
                ),
            },
            "max_results": {
                "type": "number",
                "description": "Max results to return for search (default: 10).",
            },
            "document_id": {
                "type": "string",
                "description": (
                    "Document ID to delete (for 'delete' action). "
                    "Deletes the document and all its chunks."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self, deep_memory: DeepMemoryIndex | None = None) -> None:
        self._deep_memory = deep_memory
        cfg = get_config().tools.typesense
        self._base_url = f"{cfg.protocol}://{cfg.host}:{cfg.port}"
        self._api_key = cfg.api_key
        self._default_collection = cfg.default_collection
        self._timeout = float(cfg.timeout)
        self._connection_timeout = float(cfg.connection_timeout)
        self._client: httpx.AsyncClient | None = None
        self._collection_ensured = False

    # ------------------------------------------------------------------
    # Collection resolution & bootstrap
    # ------------------------------------------------------------------

    def _get_collection(self) -> str:
        """Return the forced collection name — no LLM choice."""
        if self._deep_memory is not None:
            return self._deep_memory.collection_name
        return self._default_collection

    async def _ensure_collection(self) -> None:
        """Create the deep memory collection if it doesn't exist.

        When ``DeepMemoryIndex`` is available its ``ensure_collection()``
        is used (sync — already called at startup).  Otherwise the tool
        creates the collection itself via async HTTP using the canonical
        schema from ``deep_memory._COLLECTION_SCHEMA_TEMPLATE``.
        """
        if self._collection_ensured:
            return

        # If DeepMemoryIndex is wired in, delegate to it.
        if self._deep_memory is not None:
            self._deep_memory.ensure_collection()
            self._collection_ensured = True
            return

        # Fallback: create the collection ourselves.
        coll = self._get_collection()
        if not coll:
            return

        client = self._get_client()

        # Check if collection already exists.
        try:
            resp = await client.get(f"{self._base_url}/collections/{coll}")
            if resp.status_code == 200:
                self._collection_ensured = True
                return
        except httpx.HTTPError:
            pass

        # Build schema from the canonical template in deep_memory.py.
        from captain_claw.deep_memory import _COLLECTION_SCHEMA_TEMPLATE

        dm_cfg = getattr(get_config(), "deep_memory", None)
        embedding_dims = int(getattr(dm_cfg, "embedding_dims", 1536)) if dm_cfg else 1536

        schema: dict[str, Any] = {
            "name": coll,
            **_COLLECTION_SCHEMA_TEMPLATE,
        }
        # Copy fields list so we don't mutate the module-level template.
        schema["fields"] = list(schema["fields"])
        if embedding_dims and embedding_dims > 0:
            schema["fields"].append({
                "name": "embedding",
                "type": "float[]",
                "num_dim": embedding_dims,
                "optional": True,
            })

        try:
            resp = await client.post(
                f"{self._base_url}/collections",
                json=schema,
            )
            resp.raise_for_status()
            log.info(
                "Created deep memory collection (tool bootstrap)",
                collection=coll,
                fields=len(schema["fields"]),
            )
            self._collection_ensured = True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                # Already exists — fine.
                self._collection_ensured = True
            else:
                log.warning(
                    "Failed to create deep memory collection",
                    collection=coll,
                    status=exc.response.status_code,
                    error=exc.response.text[:200],
                )
                raise

    # ------------------------------------------------------------------
    # HTTP client (lazy, for fallback raw operations)
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._timeout,
                    connect=self._connection_timeout,
                ),
                headers={
                    "X-TYPESENSE-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        # Pop injected runtime kwargs (same pattern as all tools).
        for k in (
            "_runtime_base_path",
            "_saved_base_path",
            "_session_id",
            "_abort_event",
            "_file_registry",
            "_task_id",
        ):
            kwargs.pop(k, None)

        if not self._api_key and self._deep_memory is None:
            return ToolResult(
                success=False,
                error=(
                    "Typesense API key is not configured. "
                    "Set tools.typesense.api_key in config.yaml or "
                    "the TYPESENSE_API_KEY environment variable."
                ),
            )

        coll = self._get_collection()
        if not coll:
            return ToolResult(
                success=False,
                error=(
                    "No deep memory collection configured. "
                    "Set deep_memory.collection_name or "
                    "tools.typesense.default_collection in config.yaml."
                ),
            )

        # Ensure the collection exists with our controlled schema
        # before any operation.
        try:
            await self._ensure_collection()
        except Exception as exc:
            log.warning("Collection bootstrap failed", error=str(exc))
            # Continue anyway — the operation itself will fail with a
            # clear 404 if the collection truly doesn't exist.

        handlers = {
            "index": self._action_index,
            "search": self._action_search,
            "delete": self._action_delete,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Available: index, search, delete.",
            )
        try:
            return await handler(**kwargs)
        except httpx.ConnectError as exc:
            return ToolResult(
                success=False,
                error=f"Cannot connect to Typesense at {self._base_url}: {exc}",
            )
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response else ""
            return ToolResult(
                success=False,
                error=f"Typesense API error ({exc.response.status_code}): {body}",
            )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, error=f"Invalid JSON: {exc}")
        except Exception as exc:
            log.warning("Typesense tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _action_index(
        self,
        text: str = "",
        file_path: str = "",
        source: str = "manual",
        reference: str = "",
        tags: str = "",
        **_kw: Any,
    ) -> ToolResult:
        """Index text into the deep memory collection.

        Text is chunked, assigned a doc_id, timestamped, and optionally
        embedded by ``DeepMemoryIndex.index_document()``.

        If ``file_path`` is provided, the file is read directly from disk
        (bypassing the LLM context window) and its contents are indexed.
        """
        # Read file from disk if file_path is provided.
        if file_path and file_path.strip():
            file_path = file_path.strip()
            if not os.path.isabs(file_path):
                return ToolResult(
                    success=False,
                    error=f"file_path must be an absolute path, got: {file_path}",
                )
            if not os.path.isfile(file_path):
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}",
                )
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError as exc:
                return ToolResult(
                    success=False,
                    error=f"Cannot read file {file_path}: {exc}",
                )
            if not reference:
                reference = file_path
            log.info(
                "Indexing file from disk",
                file_path=file_path,
                size_bytes=len(text.encode("utf-8")),
            )

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="'text' or 'file_path' is required for indexing.",
            )

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        doc_id = hashlib.sha1(
            f"{reference or ''}:{text[:128]}".encode()
        ).hexdigest()[:16]

        if self._deep_memory is not None:
            # Preferred path: route through DeepMemoryIndex for proper
            # chunking, embedding, and schema compliance.
            chunks = self._deep_memory.index_document(
                doc_id=doc_id,
                text=text.strip(),
                source=source.strip() or "manual",
                reference=reference.strip(),
                path=reference.strip(),
                tags=tag_list,
            )
            return ToolResult(
                success=True,
                content=(
                    f"Indexed into deep memory: {chunks} chunk(s), "
                    f"doc_id={doc_id}, source={source.strip() or 'manual'}"
                    + (f", ref={reference.strip()}" if reference.strip() else "")
                    + (f", tags={tag_list}" if tag_list else "")
                ),
            )

        # Fallback: raw HTTP upsert (no DeepMemoryIndex available).
        coll = self._get_collection()
        now = int(time.time())
        doc = {
            "id": doc_id,
            "doc_id": doc_id,
            "source": source.strip() or "manual",
            "reference": reference.strip(),
            "path": reference.strip(),
            "text": text.strip(),
            "chunk_index": 0,
            "updated_at": now,
        }
        if tag_list:
            doc["tags"] = tag_list

        body = json.dumps(doc, ensure_ascii=False)
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/collections/{coll}/documents/import",
            params={"action": "upsert"},
            content=body,
            headers={
                "X-TYPESENSE-API-KEY": self._api_key,
                "Content-Type": "text/plain",
            },
        )
        resp.raise_for_status()
        return ToolResult(
            success=True,
            content=(
                f"Indexed 1 document into deep memory (fallback), "
                f"doc_id={doc_id}, source={source.strip() or 'manual'}"
            ),
        )

    async def _action_search(
        self,
        query: str = "",
        filter_by: str = "",
        max_results: int | float | None = None,
        **_kw: Any,
    ) -> ToolResult:
        """Search the deep memory collection."""
        if not query or not query.strip():
            return ToolResult(success=False, error="'query' is required for search.")

        coll = self._get_collection()
        per_page = min(int(max_results or 10), 250)

        # If we have DeepMemoryIndex, use it for proper hybrid search
        # (auto-generates vector queries from the embedding chain).
        if self._deep_memory is not None:
            results = self._deep_memory.search(
                query.strip(),
                max_results=per_page,
                filter_by=filter_by,
            )
            if not results:
                return ToolResult(
                    success=True,
                    content=f"No results found in deep memory for: {query.strip()}",
                )
            lines = [f"Found {len(results)} result(s) in deep memory:"]
            for r in results:
                snippet = r.snippet
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                loc = r.path or r.reference or r.doc_id
                source_tag = f"[{r.source}] " if r.source else ""
                lines.append(
                    f"  - {source_tag}{loc} (score={r.score:.2f}) {snippet}"
                )
            return ToolResult(success=True, content="\n".join(lines))

        # Fallback: raw HTTP search.
        params: dict[str, Any] = {
            "q": query.strip(),
            "query_by": "text",
            "per_page": per_page,
        }
        if filter_by:
            params["filter_by"] = filter_by

        client = self._get_client()
        resp = await client.get(
            f"{self._base_url}/collections/{coll}/documents/search",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        found = data.get("found", 0)

        if not hits:
            return ToolResult(
                success=True,
                content=f"No results found in deep memory for: {query.strip()}",
            )

        lines = [f"Found {found} result(s) in deep memory (showing {len(hits)}):"]
        for hit in hits:
            doc = hit.get("document", {})
            score = hit.get("text_match", 0)
            doc_id = doc.get("doc_id", doc.get("id", "?"))
            text_preview = str(doc.get("text", ""))
            if len(text_preview) > 300:
                text_preview = text_preview[:300] + "..."
            source_tag = f"[{doc.get('source', '')}] " if doc.get("source") else ""
            ref = doc.get("reference", "") or doc.get("path", "") or doc_id
            lines.append(f"  - {source_tag}{ref} (score={score}) {text_preview}")
        return ToolResult(success=True, content="\n".join(lines))

    async def _action_delete(
        self,
        document_id: str = "",
        filter_by: str = "",
        **_kw: Any,
    ) -> ToolResult:
        """Delete documents from the deep memory collection.

        Either by doc_id (deletes all chunks) or by filter expression.
        Collection deletion is not allowed.
        """
        if not document_id and not filter_by:
            return ToolResult(
                success=False,
                error="Provide 'document_id' or 'filter_by' to specify what to delete.",
            )

        coll = self._get_collection()

        if document_id:
            # Use DeepMemoryIndex if available (deletes all chunks for doc_id).
            if self._deep_memory is not None:
                count = self._deep_memory.delete_document(document_id.strip())
                return ToolResult(
                    success=True,
                    content=f"Deleted {count} chunk(s) for doc_id '{document_id.strip()}'.",
                )
            # Fallback: raw filter-based delete.
            client = self._get_client()
            resp = await client.delete(
                f"{self._base_url}/collections/{coll}/documents",
                params={"filter_by": f"doc_id:={document_id.strip()}"},
            )
            resp.raise_for_status()
            deleted = resp.json().get("num_deleted", 0)
            return ToolResult(
                success=True,
                content=f"Deleted {deleted} document(s) for doc_id '{document_id.strip()}'.",
            )

        # Filter-based delete.
        if self._deep_memory is not None:
            count = self._deep_memory.delete_by_filter(filter_by)
            return ToolResult(
                success=True,
                content=f"Deleted {count} document(s) matching: {filter_by}",
            )
        client = self._get_client()
        resp = await client.delete(
            f"{self._base_url}/collections/{coll}/documents",
            params={"filter_by": filter_by},
        )
        resp.raise_for_status()
        deleted = resp.json().get("num_deleted", 0)
        return ToolResult(
            success=True,
            content=f"Deleted {deleted} document(s) matching: {filter_by}",
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
