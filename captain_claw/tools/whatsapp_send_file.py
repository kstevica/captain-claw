"""Send a file the agent saved to a WhatsApp chat as a document.

Agent-generated files live on the agent's own filesystem under
``<workspace>/saved/<category>/<session>/<filename>`` (the same files
shown in the Flight Deck / glasses file list). This tool runs *inside*
the agent process: it resolves one of those files, reads its bytes, and
sends it to a WhatsApp chat as a document via the Meta Cloud API.

Two actions:
  * ``list`` — list the agent's saved files (newest first) so the agent can
    discover which one the user means ("that last report").
  * ``send`` — deliver a file. Identify it by ``path`` (what the agent knows,
    e.g. ``showcase/<session>/report.docx``), by ``filename`` (fuzzy, newest
    match wins), or ``latest`` for the most recently saved file.

Recipient resolution for ``send``:
  * An explicit ``to`` (phone number, digits only, no ``+``) wins.
  * Otherwise the *current* WhatsApp chat — captured per session when the
    conversation arrived over WhatsApp (``session.metadata['whatsapp_waid']``).

Requirements: ``WHATSAPP_ACCESS_TOKEN`` + ``WHATSAPP_PHONE_NUMBER_ID`` in the
environment (inherited from Flight Deck). WhatsApp only permits free-form
documents within 24h of the recipient's last message; out-of-window sends
are rejected by Meta and surfaced here as a clear error.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from captain_claw.tools.registry import Tool, ToolResult

# WhatsApp Cloud API caps documents at 100 MB; stay just under.
_MAX_DOC_BYTES = 95 * 1024 * 1024
_GRAPH_BASE = "https://graph.facebook.com/v18.0"

# Meta's Cloud API rejects any document MIME outside a fixed allowlist
# (error #100), and the host's mimetypes DB can't be trusted to produce
# the exact strings Meta wants (e.g. .pptx). Map the common cases
# explicitly; this is the canonical Meta-accepted set for documents.
_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
}
# Text-based formats Meta doesn't list individually but accepts when sent
# as text/plain documents (the file keeps its name/extension for the user).
_TEXT_EXT = {
    ".md", ".markdown", ".html", ".htm", ".csv", ".tsv", ".json", ".log",
    ".py", ".js", ".ts", ".css", ".xml", ".yaml", ".yml", ".rtf",
}


def _guess_doc_mime(filename: str) -> str:
    """Return a WhatsApp-accepted document MIME for *filename*."""
    ext = Path(filename).suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    if ext in _TEXT_EXT:
        return "text/plain"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _allowed_waids() -> set[str]:
    raw = _env("WHATSAPP_ALLOWED_WAIDS")
    return {p.strip().lstrip("+") for p in raw.split(",") if p.strip()} if raw else set()


class WhatsAppSendFileTool(Tool):
    """List and deliver a saved file to a WhatsApp chat as a document."""

    name = "whatsapp_send_file"
    timeout_seconds = 180.0
    description = (
        "Send a file the agent saved to a WhatsApp chat as a document. Use "
        "this whenever the user wants a file delivered over WhatsApp — e.g. "
        "'send me that report on WhatsApp', 'whatsapp me the document', 'get "
        "me that word document here'. By default it sends into the current "
        "WhatsApp chat; pass 'to' (phone number, digits only, no '+') to send "
        "to a specific number. Identify the file by 'path' (e.g. "
        "'showcase/<session>/report.docx'), by 'filename' (fuzzy match), or "
        "set 'latest' for the most recently saved file. If unsure which file "
        "the user means, call action='list' first. WhatsApp only allows "
        "sending within 24 hours of the recipient's last message."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "list"],
                "description": (
                    "'send' delivers a file (default); 'list' returns the "
                    "agent's saved files so you can pick which to send."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Path of the saved file to send, as the agent knows it "
                    "(e.g. 'showcase/<session>/report.docx' or an absolute path)."
                ),
            },
            "filename": {
                "type": "string",
                "description": (
                    "Filename (or part of it) to find among saved files. "
                    "Fuzzy, case-insensitive; newest match wins."
                ),
            },
            "latest": {
                "type": "boolean",
                "description": "Send the most recently saved file.",
            },
            "to": {
                "type": "string",
                "description": (
                    "Recipient WhatsApp number, digits only, no '+'. Omit to "
                    "reply into the current WhatsApp chat."
                ),
            },
            "caption": {
                "type": "string",
                "description": "Optional caption shown under the document.",
            },
        },
        "required": [],
    }

    # ── entry ─────────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action") or "send").strip().lower()
        saved_base = self._saved_base(kwargs)

        if action == "list":
            files = self._scan(saved_base)
            if not files:
                return ToolResult(success=True, content="No saved files found.")
            lines = [
                f"- {logical}  ({p.stat().st_size} bytes)"
                for p, logical in files[:50]
            ]
            return ToolResult(
                success=True,
                content="Saved files (newest first):\n" + "\n".join(lines),
            )

        return await self._send(kwargs, saved_base)

    # ── send ──────────────────────────────────────────────────────────

    async def _send(self, kwargs: dict[str, Any], saved_base: Path | None) -> ToolResult:
        token = _env("WHATSAPP_ACCESS_TOKEN")
        pid = _env("WHATSAPP_PHONE_NUMBER_ID")
        if not token or not pid:
            return ToolResult(
                success=False,
                error="WhatsApp not configured (WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID).",
            )

        path_arg = str(kwargs.get("path") or "").strip()
        filename = str(kwargs.get("filename") or "").strip()
        latest = bool(kwargs.get("latest"))
        to = str(kwargs.get("to") or "").lstrip("+").strip()
        caption = str(kwargs.get("caption") or "").strip()

        if not path_arg and not filename and not latest:
            return ToolResult(
                success=False,
                error=(
                    "Specify which file to send: 'path', 'filename', or "
                    "latest=true. Use action='list' to see available files."
                ),
            )

        # Resolve recipient: explicit 'to' wins, else the current WhatsApp chat.
        if not to:
            session = kwargs.get("_session")
            meta = getattr(session, "metadata", None) if session is not None else None
            if isinstance(meta, dict):
                to = str(meta.get("whatsapp_waid") or "").lstrip("+").strip()
        if not to:
            return ToolResult(
                success=False,
                error=(
                    "No recipient: this conversation isn't a WhatsApp chat and "
                    "no 'to' number was provided."
                ),
            )
        allowed = _allowed_waids()
        if allowed and to not in allowed:
            return ToolResult(
                success=False,
                error=f"Recipient {to} is not in WHATSAPP_ALLOWED_WAIDS.",
            )

        # Resolve the file on the agent's filesystem.
        resolved = self._resolve_file(kwargs, saved_base, path_arg, filename, latest)
        if resolved is None:
            hint = path_arg or filename or "latest"
            return ToolResult(
                success=False,
                error=f"Could not find a saved file for '{hint}'. Try action='list'.",
            )
        try:
            blob = resolved.read_bytes()
        except Exception as exc:
            return ToolResult(success=False, error=f"Could not read file: {exc}")
        if not blob:
            return ToolResult(success=False, error="File is empty.")
        if len(blob) > _MAX_DOC_BYTES:
            return ToolResult(
                success=False,
                error=f"File is {len(blob)} bytes; WhatsApp documents max ~100 MB.",
            )

        out_name = resolved.name
        mime = _guess_doc_mime(out_name)

        media_id, err = await self._meta_upload(token, pid, blob, out_name, mime)
        if not media_id:
            return ToolResult(success=False, error=f"WhatsApp upload failed: {err}")
        ok, err = await self._meta_send(token, pid, to, media_id, out_name, caption)
        if not ok:
            return ToolResult(success=False, error=f"WhatsApp send failed: {err}")
        return ToolResult(success=True, content=f"Sent '{out_name}' to WhatsApp {to}.")

    # ── filesystem resolution ──────────────────────────────────────────

    @staticmethod
    def _saved_base(kwargs: dict[str, Any]) -> Path | None:
        base = kwargs.get("_saved_base_path")
        if base:
            return Path(base)
        runtime = kwargs.get("_runtime_base_path")
        return (Path(runtime) / "saved") if runtime else None

    def _resolve_file(
        self,
        kwargs: dict[str, Any],
        saved_base: Path | None,
        path_arg: str,
        filename: str,
        latest: bool,
    ) -> Path | None:
        # 1. Explicit path: try the file registry, then plausible bases.
        if path_arg:
            registry = kwargs.get("_file_registry")
            if registry is not None:
                try:
                    physical = registry.resolve(path_arg)
                except Exception:
                    physical = None
                if physical and Path(physical).is_file():
                    return Path(physical)
            cand = Path(path_arg).expanduser()
            if cand.is_absolute() and cand.is_file():
                return cand
            stripped = path_arg[len("saved/"):] if path_arg.startswith("saved/") else path_arg
            for base in (saved_base, kwargs.get("_runtime_base_path")):
                if base:
                    for rel in (path_arg, stripped):
                        p = (Path(base) / rel).resolve()
                        if p.is_file():
                            return p
            # Fall through to filename match on the basename.
            filename = filename or Path(path_arg).name

        files = self._scan(saved_base)
        if not files:
            return None
        # 2. Fuzzy filename match (newest first).
        if filename:
            fl = filename.lower()
            exact = [p for p, _ in files if p.name.lower() == fl]
            if exact:
                return exact[0]
            sub = [p for p, _ in files if fl in p.name.lower()]
            if sub:
                return sub[0]
            return None
        # 3. Latest overall.
        if latest:
            return files[0][0]
        return None

    @staticmethod
    def _scan(saved_base: Path | None) -> list[tuple[Path, str]]:
        """Return (path, logical_path) for saved files, newest first."""
        if not saved_base or not Path(saved_base).is_dir():
            return []
        base = Path(saved_base)
        out: list[tuple[float, Path, str]] = []
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            try:
                mtime = p.stat().st_mtime
                logical = str(p.relative_to(base))
            except (OSError, ValueError):
                continue
            out.append((mtime, p, logical))
        out.sort(key=lambda t: t[0], reverse=True)
        return [(p, logical) for _, p, logical in out]

    # ── Meta Cloud API ─────────────────────────────────────────────────

    @staticmethod
    async def _meta_upload(
        token: str, pid: str, blob: bytes, filename: str, mime: str
    ) -> tuple[str, str]:
        url = f"{_GRAPH_BASE}/{pid}/media"
        files = {
            "file": (filename or "file", blob, mime),
            "messaging_product": (None, "whatsapp"),
            "type": (None, mime),
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    url, headers={"Authorization": f"Bearer {token}"}, files=files
                )
        except Exception as exc:
            return "", f"upload request failed: {exc}"
        if resp.status_code != 200:
            return "", f"upload rejected ({resp.status_code}): {resp.text[:300]}"
        try:
            return str((resp.json() or {}).get("id") or ""), ""
        except Exception:
            return "", "upload returned no id"

    @staticmethod
    async def _meta_send(
        token: str, pid: str, to: str, media_id: str, filename: str, caption: str
    ) -> tuple[bool, str]:
        document: dict[str, Any] = {"id": media_id, "filename": filename or "file"}
        if caption:
            document["caption"] = caption
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "document",
            "document": document,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{_GRAPH_BASE}/{pid}/messages",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except Exception as exc:
            return False, f"send request failed: {exc}"
        if resp.status_code != 200:
            return False, f"send rejected ({resp.status_code}): {resp.text[:400]}"
        return True, ""
