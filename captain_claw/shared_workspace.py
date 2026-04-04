"""Shared workspace for intra-orchestration data flow.

Provides a lightweight, namespaced key-value store that lives for the
duration of a single orchestration run. Tasks write named outputs and
downstream tasks read them. Entries are namespaced by task ID so writes
are always attributed while any task can read from any namespace.

Inspired by Open Multi-Agent's SharedMemory pattern.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkspaceEntry:
    """A single entry in the shared workspace."""

    key: str
    value: Any
    task_id: str
    session_id: str = ""
    written_at: float = 0.0
    content_type: str = "text"  # text | json | binary_ref

    def __post_init__(self) -> None:
        if not self.written_at:
            self.written_at = time.monotonic()


class SharedWorkspace:
    """In-memory namespaced key-value store for one orchestration run.

    Key format: ``{namespace}:{key}`` where *namespace* defaults to
    ``task_id``.  Any task can **read** from any namespace; writes are
    always attributed to the calling task.

    Thread-safe via a simple lock — orchestration workers may run in
    separate threads.
    """

    def __init__(
        self,
        orchestration_id: str,
        *,
        on_change: Any | None = None,
    ) -> None:
        self.orchestration_id = orchestration_id
        self._data: dict[str, WorkspaceEntry] = {}
        self._lock = threading.Lock()
        self._on_change = on_change  # optional callback(event_type, entry)
        self._history: list[dict[str, Any]] = []  # append-only log

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        key: str,
        value: Any,
        *,
        task_id: str,
        session_id: str = "",
        namespace: str = "",
        content_type: str = "text",
    ) -> str:
        """Store a value.  Returns the fully-qualified key.

        Parameters
        ----------
        key:
            Logical name (e.g. ``api_spec``, ``review_result``).
        value:
            Anything serialisable — strings, dicts, lists.
        task_id:
            ID of the task performing the write (required for attribution).
        session_id:
            Optional session ID for cross-referencing.
        namespace:
            Explicit namespace. Defaults to *task_id*.
        content_type:
            One of ``text``, ``json``, ``binary_ref``.
        """
        ns = namespace or task_id
        fqkey = f"{ns}:{key}"

        entry = WorkspaceEntry(
            key=fqkey,
            value=value,
            task_id=task_id,
            session_id=session_id,
            written_at=time.monotonic(),
            content_type=content_type,
        )

        with self._lock:
            self._data[fqkey] = entry
            self._history.append(
                {
                    "action": "write",
                    "fqkey": fqkey,
                    "task_id": task_id,
                    "content_type": content_type,
                    "ts": entry.written_at,
                }
            )

        if self._on_change:
            try:
                self._on_change("write", entry)
            except Exception:
                pass

        return fqkey

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, key: str, *, namespace: str = "") -> Any | None:
        """Return a stored value or ``None``.

        If *key* contains a colon it is treated as a fully-qualified key.
        Otherwise *namespace* (or bare key) is tried.
        """
        fqkey = self._resolve_key(key, namespace)
        with self._lock:
            entry = self._data.get(fqkey)
        return entry.value if entry else None

    def read_entry(self, key: str, *, namespace: str = "") -> WorkspaceEntry | None:
        """Return the full :class:`WorkspaceEntry` or ``None``."""
        fqkey = self._resolve_key(key, namespace)
        with self._lock:
            return self._data.get(fqkey)

    # ------------------------------------------------------------------
    # List / query
    # ------------------------------------------------------------------

    def list_keys(
        self,
        *,
        namespace: str = "",
        task_id: str = "",
    ) -> list[str]:
        """Return matching fully-qualified keys."""
        with self._lock:
            keys = list(self._data.keys())
        results: list[str] = []
        for k in keys:
            if namespace and not k.startswith(f"{namespace}:"):
                continue
            if task_id:
                entry = self._data.get(k)
                if entry and entry.task_id != task_id:
                    continue
            results.append(k)
        return sorted(results)

    def list_entries(
        self,
        *,
        namespace: str = "",
        task_id: str = "",
    ) -> list[WorkspaceEntry]:
        """Return matching entries."""
        keys = self.list_keys(namespace=namespace, task_id=task_id)
        with self._lock:
            return [self._data[k] for k in keys if k in self._data]

    # ------------------------------------------------------------------
    # Snapshot / summary (for UI and prompt injection)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Full state dict suitable for JSON serialisation / UI."""
        with self._lock:
            entries = {}
            for fqkey, entry in self._data.items():
                val = entry.value
                # Truncate very large values for the snapshot
                if isinstance(val, str) and len(val) > 2000:
                    val = val[:2000] + "... [truncated]"
                entries[fqkey] = {
                    "value": val,
                    "task_id": entry.task_id,
                    "session_id": entry.session_id,
                    "content_type": entry.content_type,
                }
            return {
                "orchestration_id": self.orchestration_id,
                "entry_count": len(self._data),
                "entries": entries,
            }

    def get_summary(self, *, max_value_len: int = 500) -> str:
        """Markdown-formatted digest grouped by namespace.

        Suitable for injecting into an agent's context window so it
        knows what data is available from prior tasks.
        """
        with self._lock:
            if not self._data:
                return ""
            entries = dict(self._data)

        # Group by namespace (part before first colon).
        by_ns: dict[str, list[tuple[str, WorkspaceEntry]]] = {}
        for fqkey, entry in entries.items():
            parts = fqkey.split(":", 1)
            ns = parts[0] if len(parts) == 2 else "_global"
            by_ns.setdefault(ns, []).append((fqkey, entry))

        lines: list[str] = ["## Shared Workspace"]
        for ns in sorted(by_ns):
            lines.append(f"\n### Namespace: {ns}")
            for fqkey, entry in by_ns[ns]:
                short_key = fqkey.split(":", 1)[-1] if ":" in fqkey else fqkey
                val_preview = _preview_value(entry.value, max_value_len)
                lines.append(
                    f"- **{short_key}** ({entry.content_type})"
                    f" — from task `{entry.task_id}`"
                )
                if val_preview:
                    lines.append(f"  ```\n  {val_preview}\n  ```")
        return "\n".join(lines)

    def get_keys_for_task_prompt(self, task_id: str, depends_on: list[str]) -> str:
        """Build a concise section listing workspace keys available to a task.

        Only includes keys written by tasks in *depends_on* (direct
        upstream dependencies) to keep the prompt focused.
        """
        with self._lock:
            if not self._data:
                return ""
            entries = dict(self._data)

        dep_set = set(depends_on)
        relevant: list[tuple[str, WorkspaceEntry]] = []
        for fqkey, entry in entries.items():
            if entry.task_id in dep_set:
                relevant.append((fqkey, entry))

        if not relevant:
            return ""

        lines: list[str] = [
            "\n## Data from upstream tasks (shared workspace)",
            "You can read these values using the `workspace_read` tool.\n",
        ]
        for fqkey, entry in sorted(relevant, key=lambda x: x[0]):
            val_preview = _preview_value(entry.value, 300)
            lines.append(f"- `{fqkey}` ({entry.content_type})")
            if val_preview:
                lines.append(f"  Preview: {val_preview}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Wipe all entries (e.g. when an orchestration run resets)."""
        with self._lock:
            self._data.clear()
            self._history.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str, namespace: str) -> str:
        """Build a fully-qualified key from parts."""
        if ":" in key:
            return key  # already fully qualified
        if namespace:
            return f"{namespace}:{key}"
        return key


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _preview_value(value: Any, max_len: int = 500) -> str:
    """Create a short text preview of a workspace value."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        import json

        try:
            text = json.dumps(value, indent=2, default=str)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text
