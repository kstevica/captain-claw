"""File location registry for cross-session file access.

Maintains a mapping of logical paths (what the LLM requested) to physical
paths (where the file actually lives on disk, scoped under a session folder).

Used during orchestration so that downstream tasks can access files created
by upstream tasks without knowing session IDs or internal folder structures.
Also useful in single-agent mode for consistent file resolution.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


class FileRegistry:
    """Maps logical file paths to physical on-disk locations.

    Thread-safe.  One registry instance is shared across all tasks
    within a single orchestration run (or a single-agent session).

    Lookup priority:
        1. Exact normalized logical path match
        2. Filename-only match (when exactly one candidate exists)

    This handles the common case where LLMs refer to files inconsistently
    (e.g. ``/output/results.json`` in one task, ``results.json`` in another).
    """

    def __init__(self, orchestration_id: str = "") -> None:
        self._orchestration_id = orchestration_id or ""
        self._lock = threading.Lock()
        # logical_path (normalized) -> physical_path (str)
        self._mappings: dict[str, str] = {}
        # filename -> list of logical_path keys (for fuzzy lookup)
        self._filename_index: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def orchestration_id(self) -> str:
        return self._orchestration_id

    def register(
        self,
        logical_path: str,
        physical_path: str | Path,
        *,
        task_id: str = "",
    ) -> None:
        """Record a logical -> physical mapping.

        Args:
            logical_path: The path the LLM requested (e.g. ``scripts/analysis.py``).
            physical_path: The actual path on disk after session scoping.
            task_id: Optional task identifier for diagnostics.
        """
        normalized = self._normalize(logical_path)
        physical = str(physical_path)
        filename = Path(normalized).name

        with self._lock:
            self._mappings[normalized] = physical
            bucket = self._filename_index.setdefault(filename, [])
            if normalized not in bucket:
                bucket.append(normalized)

        log.debug(
            "FileRegistry: registered",
            logical=normalized,
            physical=physical,
            task_id=task_id,
        )

    def resolve(self, logical_path: str) -> str | None:
        """Resolve a logical path to its physical location.

        Returns:
            Physical path string if found, else ``None``.
        """
        normalized = self._normalize(logical_path)

        with self._lock:
            # 1. Exact match
            exact = self._mappings.get(normalized)
            if exact is not None:
                return exact

            # 2. Try without leading category or "saved/" prefix
            stripped = self._strip_common_prefixes(normalized)
            if stripped != normalized:
                exact = self._mappings.get(stripped)
                if exact is not None:
                    return exact

            # 3. Filename-only fuzzy match (unambiguous only)
            filename = Path(normalized).name
            candidates = self._filename_index.get(filename, [])
            if len(candidates) == 1:
                return self._mappings.get(candidates[0])

        return None

    def resolve_or_passthrough(self, path: str) -> str:
        """Resolve if mapped, otherwise return the original path unchanged."""
        resolved = self.resolve(path)
        return resolved if resolved is not None else path

    def list_files(self, *, task_id: str = "") -> list[dict[str, str]]:
        """Return all registered file mappings.

        Args:
            task_id: If provided, filter to files registered by this task.

        Returns:
            List of dicts with ``logical`` and ``physical`` keys.
        """
        with self._lock:
            entries = []
            for logical, physical in self._mappings.items():
                entries.append({
                    "logical": logical,
                    "physical": physical,
                })
            return entries

    def build_manifest(self) -> str:
        """Build a human-readable manifest of available files.

        Suitable for injection into downstream task prompts so the LLM
        knows which files are available from previous steps.
        """
        with self._lock:
            if not self._mappings:
                return ""

            lines = ["Files available from previous steps:"]
            for logical, physical in sorted(self._mappings.items()):
                # Show logical path (what the LLM should use) and a
                # hint about the physical location.
                lines.append(f"  - {logical}  (on disk: {physical})")
            return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry state for persistence / debugging."""
        with self._lock:
            return {
                "orchestration_id": self._orchestration_id,
                "mappings": dict(self._mappings),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileRegistry:
        """Restore registry from serialized state."""
        registry = cls(orchestration_id=str(data.get("orchestration_id", "")))
        mappings = data.get("mappings", {})
        if isinstance(mappings, dict):
            for logical, physical in mappings.items():
                registry.register(str(logical), str(physical))
        return registry

    def merge_from(self, other: FileRegistry) -> None:
        """Merge another registry's mappings into this one.

        Existing entries are NOT overwritten (first-writer wins).
        """
        with other._lock:
            other_mappings = dict(other._mappings)

        with self._lock:
            for logical, physical in other_mappings.items():
                if logical not in self._mappings:
                    self._mappings[logical] = physical
                    filename = Path(logical).name
                    bucket = self._filename_index.setdefault(filename, [])
                    if logical not in bucket:
                        bucket.append(logical)

    def __len__(self) -> int:
        with self._lock:
            return len(self._mappings)

    def __bool__(self) -> bool:
        return len(self) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(path: str) -> str:
        """Normalize a path for consistent lookup.

        - Strips whitespace
        - Resolves ``..`` / ``.`` components
        - Lowercases on case-insensitive systems (kept as-is here for now)
        - Strips leading ``/`` for uniformity
        """
        raw = str(path or "").strip()
        if not raw:
            return ""
        # Remove quotes the LLM might wrap paths in
        raw = raw.strip("\"'`")
        # Normalize path components
        parts = Path(raw).parts
        # Strip leading "/" so both "/foo/bar" and "foo/bar" match
        clean_parts = [p for p in parts if p not in ("", "/")]
        return "/".join(clean_parts) if clean_parts else ""

    @staticmethod
    def _strip_common_prefixes(normalized: str) -> str:
        """Strip common prefixes like 'saved/' for fallback matching."""
        prefixes_to_strip = ("saved/",)
        result = normalized
        for prefix in prefixes_to_strip:
            if result.startswith(prefix):
                result = result[len(prefix):]
        return result
