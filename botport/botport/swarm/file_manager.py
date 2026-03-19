"""Swarm file manager — stores and serves files for swarm task execution.

Files are organized under:
    workspace-botport/swarm/<swarm_id>/<agent_instance_name>/<filename>

Supports gzip-compressed base64 transfer over the WebSocket protocol.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Root workspace directory (relative to CWD, or override via env).
WORKSPACE_ROOT = Path(os.environ.get(
    "BOTPORT_WORKSPACE", "workspace-botport",
))

# File size limits.
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB raw file limit

# Formats that are already compressed — skip gzip for these.
_PRECOMPRESSED_EXTS = frozenset({
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif",
    ".mp3", ".mp4", ".m4a", ".m4v", ".avi", ".mov", ".mkv", ".webm",
    ".woff", ".woff2",
    ".pdf",  # PDF uses internal compression
})


def _swarm_dir(swarm_id: str) -> Path:
    """Get the workspace directory for a swarm."""
    return WORKSPACE_ROOT / "swarm" / swarm_id


def _agent_dir(swarm_id: str, agent_name: str) -> Path:
    """Get the workspace directory for an agent within a swarm."""
    # Sanitize agent name for filesystem safety.
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in agent_name)
    return _swarm_dir(swarm_id) / safe_name


def _should_compress(filename: str) -> bool:
    """Check if a file should be gzip-compressed for transfer."""
    ext = Path(filename).suffix.lower()
    return ext not in _PRECOMPRESSED_EXTS


def encode_file(data: bytes, filename: str) -> tuple[str, bool]:
    """Encode file data for transfer: optionally gzip, then base64.

    Returns (base64_string, is_compressed).
    """
    compressed = False
    payload = data
    if _should_compress(filename) and len(data) > 256:
        payload = gzip.compress(data, compresslevel=6)
        # Only use compressed version if it's actually smaller.
        if len(payload) < len(data):
            compressed = True
        else:
            payload = data
            compressed = False

    return base64.b64encode(payload).decode("ascii"), compressed


def decode_file(b64_data: str, compressed: bool) -> bytes:
    """Decode file data from transfer: base64 decode, optionally gunzip."""
    raw = base64.b64decode(b64_data)
    if compressed:
        raw = gzip.decompress(raw)
    return raw


def file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file data."""
    return hashlib.sha256(data).hexdigest()


def guess_mime_type(filename: str) -> str:
    """Guess MIME type from filename."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


class FileManager:
    """Manages file storage for swarm workspaces."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._root = workspace_root or WORKSPACE_ROOT

    def _swarm_dir(self, swarm_id: str) -> Path:
        return self._root / "swarm" / swarm_id

    def _agent_dir(self, swarm_id: str, agent_name: str) -> Path:
        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in agent_name
        )
        return self._swarm_dir(swarm_id) / safe_name

    # ── Store ────────────────────────────────────────────────────

    def store_file(
        self,
        swarm_id: str,
        agent_name: str,
        filename: str,
        data: bytes,
        subfolder: str = "",
    ) -> dict[str, Any]:
        """Store a file in the swarm workspace.

        Returns file metadata dict.
        """
        if len(data) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(data)} bytes (max {MAX_FILE_SIZE})"
            )

        # Sanitize filename (keep only the basename, no path traversal).
        safe_filename = Path(filename).name
        if not safe_filename:
            safe_filename = f"file_{uuid.uuid4().hex[:8]}"

        target_dir = self._agent_dir(swarm_id, agent_name)
        if subfolder:
            # Sanitize: resolve to prevent path traversal but preserve nesting.
            safe_sub = Path(subfolder)
            # Block absolute paths and parent references.
            if safe_sub.is_absolute() or ".." in safe_sub.parts:
                safe_sub = Path(safe_sub.name)
            target_dir = target_dir / safe_sub

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_filename

        # Avoid overwriting — add suffix if exists.
        if target_path.exists():
            stem = target_path.stem
            ext = target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{stem}_{counter}{ext}"
                counter += 1

        target_path.write_bytes(data)

        rel_path = str(target_path.relative_to(self._swarm_dir(swarm_id)))
        sha = file_hash(data)
        mime = guess_mime_type(safe_filename)

        meta = {
            "file_id": uuid.uuid4().hex,
            "filename": safe_filename,
            "path": rel_path,
            "size": len(data),
            "sha256": sha,
            "mime_type": mime,
            "agent": agent_name,
            "swarm_id": swarm_id,
        }

        log.info(
            "File stored: %s (%d bytes, %s) -> %s",
            safe_filename, len(data), mime, rel_path,
        )
        return meta

    # ── Retrieve ─────────────────────────────────────────────────

    def get_file(self, swarm_id: str, rel_path: str) -> bytes | None:
        """Retrieve a file by its relative path within the swarm directory."""
        target = self._swarm_dir(swarm_id) / rel_path

        # Security: ensure we don't escape the swarm directory.
        try:
            target.resolve().relative_to(self._swarm_dir(swarm_id).resolve())
        except ValueError:
            log.warning("Path traversal attempt: %s", rel_path)
            return None

        if not target.is_file():
            return None

        return target.read_bytes()

    def get_file_by_id(self, swarm_id: str, file_id: str) -> tuple[bytes | None, dict[str, Any]]:
        """Retrieve a file by ID (searches manifest)."""
        manifest = self.list_files(swarm_id)
        for entry in manifest:
            if entry.get("file_id") == file_id:
                data = self.get_file(swarm_id, entry["path"])
                return data, entry
        return None, {}

    # ── List ─────────────────────────────────────────────────────

    def list_files(
        self,
        swarm_id: str,
        agent_name: str = "",
    ) -> list[dict[str, Any]]:
        """List all files in a swarm workspace.

        If agent_name is provided, only list files from that agent.
        """
        if agent_name:
            search_dir = self._agent_dir(swarm_id, agent_name)
        else:
            search_dir = self._swarm_dir(swarm_id)

        if not search_dir.exists():
            return []

        files = []
        swarm_dir = self._swarm_dir(swarm_id)

        for path in search_dir.rglob("*"):
            if not path.is_file():
                continue

            rel_path = str(path.relative_to(swarm_dir))
            # Derive agent name from path.
            parts = Path(rel_path).parts
            file_agent = parts[0] if parts else ""

            stat = path.stat()
            files.append({
                "file_id": hashlib.md5(
                    rel_path.encode()
                ).hexdigest(),  # Deterministic ID from path.
                "filename": path.name,
                "path": rel_path,
                "size": stat.st_size,
                "mime_type": guess_mime_type(path.name),
                "agent": file_agent,
                "swarm_id": swarm_id,
                "modified_at": stat.st_mtime,
            })

        # Sort by modification time (newest first).
        files.sort(key=lambda f: f.get("modified_at", 0), reverse=True)
        return files

    # ── Delete ───────────────────────────────────────────────────

    def delete_file(self, swarm_id: str, rel_path: str) -> bool:
        """Delete a file by relative path."""
        target = self._swarm_dir(swarm_id) / rel_path
        try:
            target.resolve().relative_to(self._swarm_dir(swarm_id).resolve())
        except ValueError:
            return False

        if target.is_file():
            target.unlink()
            log.info("File deleted: %s/%s", swarm_id[:8], rel_path)
            return True
        return False

    def cleanup_swarm(self, swarm_id: str) -> int:
        """Remove all files for a swarm. Returns count of files deleted."""
        swarm_dir = self._swarm_dir(swarm_id)
        if not swarm_dir.exists():
            return 0

        count = 0
        for path in swarm_dir.rglob("*"):
            if path.is_file():
                path.unlink()
                count += 1

        # Remove empty directories.
        for path in sorted(swarm_dir.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

        try:
            swarm_dir.rmdir()
        except OSError:
            pass

        if count:
            log.info("Cleaned up %d files for swarm %s", count, swarm_id[:8])
        return count

    def workspace_size(self, swarm_id: str) -> int:
        """Get total size of all files in a swarm workspace."""
        swarm_dir = self._swarm_dir(swarm_id)
        if not swarm_dir.exists():
            return 0
        return sum(f.stat().st_size for f in swarm_dir.rglob("*") if f.is_file())
