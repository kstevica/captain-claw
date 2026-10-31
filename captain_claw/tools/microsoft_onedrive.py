"""Microsoft OneDrive tool — browse, search, read, download and write files in
the user's OneDrive (personal or OneDrive for Business / SharePoint files
shared with them) through Microsoft Graph.

The OneDrive twin of ``google_drive``'s file actions. OneDrive keeps real
Office files (.docx / .xlsx / .pptx), so ``read`` extracts them with the same
extract tools, and changing one is download → edit the local copy → ``update``
(same item id). Nothing is ever overwritten by ``upload`` / ``create`` (a name
clash gets a new name); only ``update`` replaces a file's content.

Items are addressed by ``file_id`` / ``folder_id`` (with ``drive_id`` for an
item in someone else's drive), by ``path`` inside the user's OneDrive
(``/Documents/report.docx``), or by a OneDrive / SharePoint share URL in
``file_id`` (resolved through Graph's ``/shares`` API — needs
``Files.Read.All``). Scopes: ``Files.Read*`` to read, ``Files.ReadWrite*`` to
write.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from captain_claw.logging import get_logger
from captain_claw.microsoft_graph import (
    FILES_ALL,
    FILES_READ,
    FILES_WRITE,
    GRAPH_API,
    graph_error,
    graph_token,
    limit,
    odata_string,
    pop_runtime,
)
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_WRITE_ACTIONS = frozenset({"upload", "create", "create_folder", "update"})

_ITEM_SELECT = (
    "id,name,size,lastModifiedDateTime,createdDateTime,folder,file,webUrl,"
    "parentReference,remoteItem"
)
_INFO_SELECT = _ITEM_SELECT + ",createdBy,lastModifiedBy,shared,description"

_MAX_READ_CHARS = 500_000
_MAX_DOWNLOAD_BYTES = 50_000_000            # read / download
_MAX_UPLOAD_BYTES = 250 * 1024 * 1024       # upload / update
_SIMPLE_UPLOAD_BYTES = 4 * 1024 * 1024      # Graph's limit for a single PUT
_CHUNK = 320 * 1024 * 32                    # upload-session chunk (a 320 KiB multiple)

_EXTRACT_BY_SUFFIX = {
    ".pdf": "pdf_extract", ".docx": "docx_extract", ".xlsx": "xlsx_extract", ".pptx": "pptx_extract",
}
# Office formats Graph converts to PDF (?format=pdf) so read can extract them.
_PDF_CONVERTIBLE = frozenset({
    ".doc", ".dot", ".dotx", ".docm", ".rtf", ".odt", ".ppt", ".pps", ".ppsx", ".pptm",
    ".odp", ".xls", ".xlsm", ".xlsb", ".ods", ".msg", ".eml",
})
_TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html",
    ".htm", ".log", ".ini", ".cfg", ".toml", ".py", ".js", ".ts", ".tsx", ".jsx", ".css",
    ".sql", ".sh", ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".rb", ".php", ".srt",
    ".vtt", ".ics", ".tex", ".rst",
})


def _seg(value: str) -> str:
    return quote(str(value).strip(), safe="=-_.!~")


def _share_token(url: str) -> str:
    """A share URL as Graph's ``u!`` share id."""
    encoded = base64.urlsafe_b64encode(url.strip().encode("utf-8")).decode("ascii").rstrip("=")
    return f"u!{encoded}"


def _size(n: Any) -> str:
    try:
        b = int(n or 0)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024  # type: ignore[assignment]
    return f"{b} B"


class _Resolved:
    """An item and the drive-qualified path every later call uses."""

    def __init__(self, meta: dict[str, Any]) -> None:
        remote = meta.get("remoteItem") or {}
        src = remote if remote.get("id") else meta
        self.meta = meta
        self.id = str(src.get("id") or "")
        self.drive_id = str((src.get("parentReference") or {}).get("driveId") or "")
        self.name = str(meta.get("name") or src.get("name") or self.id)
        self.is_folder = bool(meta.get("folder") or remote.get("folder"))
        self.size = int((src.get("size") if src.get("size") is not None else meta.get("size")) or 0)
        self.path = (f"/drives/{_seg(self.drive_id)}/items/{_seg(self.id)}"
                     if self.drive_id else f"/me/drive/items/{_seg(self.id)}")


class MicrosoftOneDriveTool(Tool):
    """OneDrive files through Microsoft Graph."""

    name = "microsoft_onedrive"
    description = (
        "Microsoft OneDrive (and OneDrive for Business / SharePoint files shared with the user). "
        "Actions: list (folder contents; folder_id or path, default the root), search (by name or "
        "content), read (content inline — PDF/DOCX/XLSX/PPTX extracted, other Office formats "
        "converted, text files as text), info (metadata + web link), download (save a local copy "
        "and return its path, for scripts / extract tools), upload (local file → OneDrive), create "
        "(new file from text), create_folder, update (replace an existing file's content — same "
        "id). Address files by file_id (IDs printed by list/search; add drive_id when shown), by "
        "path inside OneDrive ('/Documents/report.docx'), or pass a OneDrive/SharePoint share URL "
        "as file_id — never web_fetch/browser a OneDrive link. upload/create never overwrite (a "
        "clashing name gets a new one); to change an existing file: download, edit the copy, then "
        "update with local_path."
    )
    timeout_seconds = 300.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "search", "read", "info", "download",
                         "upload", "create", "create_folder", "update"],
                "description": "The action to perform.",
            },
            "file_id": {
                "type": "string",
                "description": (
                    "Item ID (read, info, download, update) — or a OneDrive / SharePoint share URL."
                ),
            },
            "folder_id": {
                "type": "string",
                "description": "Folder ID (list; upload / create / create_folder target). Default: root.",
            },
            "drive_id": {
                "type": "string",
                "description": "Drive ID for an item outside the user's own drive (list/search print it).",
            },
            "path": {
                "type": "string",
                "description": (
                    "Path inside the user's OneDrive instead of an ID, e.g. '/Documents/Q3' (list, "
                    "upload / create / create_folder target) or '/Documents/report.docx' (read, info, "
                    "download, update)."
                ),
            },
            "query": {"type": "string", "description": "search: words in the file name or content."},
            "max_results": {
                "type": "number",
                "description": "list (default 100) / search (default 20); up to 1000.",
            },
            "output_path": {
                "type": "string",
                "description": (
                    "download: where to save inside the saved/ area (default "
                    "saved/downloads/<session>/<name>). A .pdf path on an Office file saves a PDF conversion."
                ),
            },
            "local_path": {"type": "string", "description": "Local file (upload; update from a file)."},
            "name": {"type": "string", "description": "File or folder name (upload, create, create_folder)."},
            "content": {"type": "string", "description": "Text content (create; update with text)."},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=300.0,
            follow_redirects=True,  # /content 302s to a pre-authenticated URL (httpx drops the token)
            headers={"User-Agent": "Captain Claw/0.1.0 (Microsoft OneDrive Tool)"},
        )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        runtime = pop_runtime(kwargs)
        if kwargs.get("local_path"):
            from captain_claw.tools.google_drive import GoogleDriveTool

            kwargs["local_path"] = GoogleDriveTool._workspace_first(str(kwargs["local_path"]), runtime)
        handlers = {
            "list": self._action_list,
            "search": self._action_search,
            "read": self._action_read,
            "info": self._action_info,
            "download": self._action_download,
            "upload": self._action_upload,
            "create": self._action_create,
            "create_folder": self._action_create_folder,
            "update": self._action_update,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use one of: {', '.join(handlers)}",
            )
        if action == "download":
            kwargs["runtime"] = runtime
        write = action in _WRITE_ACTIONS
        try:
            token, granted = await graph_token(
                FILES_WRITE if write else FILES_READ,
                "OneDrive write access (Files.ReadWrite)" if write else "OneDrive access (Files.Read)",
            )
        except RuntimeError as exc:
            return ToolResult(success=False, error=str(exc))
        try:
            return await handler(token, **kwargs)
        except LookupError as exc:
            return ToolResult(success=False, error=str(exc))
        except httpx.HTTPStatusError as exc:
            via_url = str(kwargs.get("file_id") or "").strip().lower().startswith(("http://", "https://"))
            not_found = "File or folder not found. Check the ID or path."
            if via_url and granted and not granted & FILES_ALL:
                not_found = (
                    "That shared file can't be opened: the Microsoft connection only covers the "
                    "user's own files. An admin enables 'OneDrive (read all you can open)' in Flight "
                    "Deck → Connections → Microsoft, then the user reconnects."
                )
            return graph_error(exc, "OneDrive", not_found=not_found)
        except httpx.HTTPError as exc:
            log.error("OneDrive HTTP error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except Exception as exc:
            log.error("OneDrive tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    # ── addressing ──────────────────────────────────────────────────

    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def _get(self, token: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path if path.startswith("https://") else f"{GRAPH_API}{path}"
        resp = await self._client.get(url, params=params, headers=self._headers(token))
        resp.raise_for_status()
        return resp.json()

    async def _resolve(
        self, token: str, *, item_id: str | None, drive_id: str | None, path: str | None,
        default_root: bool, select: str = _ITEM_SELECT,
    ) -> _Resolved:
        ref = str(item_id or "").strip()
        rel = str(path or "").strip()
        params = {"$select": select}
        if ref.lower().startswith(("http://", "https://")):
            meta = await self._get(token, f"/shares/{_share_token(ref)}/driveItem", params)
        elif ref and ref.lower() != "root":
            base = f"/drives/{_seg(drive_id)}/items" if drive_id else "/me/drive/items"
            meta = await self._get(token, f"{base}/{_seg(ref)}", params)
        elif rel and rel.strip("/"):
            clean = "/".join(quote(p, safe="") for p in rel.strip("/").split("/") if p)
            meta = await self._get(token, f"/me/drive/root:/{clean}", params)
        elif default_root or ref.lower() == "root":
            meta = await self._get(token, "/me/drive/root", params)
        else:
            raise LookupError("Pass file_id (an ID or share URL) or path.")
        return _Resolved(meta)

    def _line(self, item: dict[str, Any], *, show_drive: bool = False) -> str:
        """One item; its drive_id when it lives outside the user's drive (a
        shared item) or *show_drive* (search spans drives)."""
        r = _Resolved(item)
        icon = "📁" if r.is_folder else "📄"
        extra = f"{(item.get('folder') or {}).get('childCount', 0)} items" if r.is_folder else _size(r.size)
        out = (f"  {icon} {r.name}  ({extra}, modified {str(item.get('lastModifiedDateTime') or '')[:16].replace('T', ' ')})"
               f"\n    ID: {r.id}")
        if r.drive_id and (show_drive or item.get("remoteItem")):
            out += f"  |  drive_id: {r.drive_id}"
        return out

    # ── list / search / info ────────────────────────────────────────

    async def _action_list(
        self, token: str, folder_id: str | None = None, drive_id: str | None = None,
        path: str | None = None, max_results: Any = None, **kwargs: Any,
    ) -> ToolResult:
        folder = await self._resolve(token, item_id=folder_id, drive_id=drive_id, path=path, default_root=True)
        if not folder.is_folder:
            return ToolResult(success=False, error=f"'{folder.name}' is a file, not a folder — use read or info.")
        n = limit(max_results, 100, 1000)
        items: list[dict[str, Any]] = []
        url: str | None = f"{folder.path}/children"
        params: dict[str, Any] | None = {"$top": min(n, 200), "$select": _ITEM_SELECT}
        while url and len(items) < n:
            data = await self._get(token, url, params)
            params = None
            items.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        more = bool(url)
        items = items[:n]
        if not items:
            return ToolResult(success=True, content=f"Folder '{folder.name}' is empty.")
        folders = sorted((i for i in items if _Resolved(i).is_folder), key=lambda i: str(i.get("name") or "").lower())
        files = sorted((i for i in items if not _Resolved(i).is_folder),
                       key=lambda i: str(i.get("lastModifiedDateTime") or ""), reverse=True)
        lines = [f"Folder '{folder.name}' ({len(items)} items; folder_id {folder.id}"
                 + (f", drive_id {folder.drive_id}" if drive_id else "") + "):\n"]
        lines.extend(self._line(i) for i in folders + files)
        if more:
            lines.append(f"\n(More items exist — call list again with max_results above {n}.)")
        return ToolResult(success=True, content="\n".join(lines))

    async def _action_search(self, token: str, query: str = "", max_results: Any = None, **kwargs: Any) -> ToolResult:
        q = str(query or "").strip()
        if not q:
            return ToolResult(success=False, error="query is required for search.")
        n = limit(max_results, 20, 1000)
        items: list[dict[str, Any]] = []
        url: str | None = f"/me/drive/root/search(q='{quote(odata_string(q), safe='')}')"
        params: dict[str, Any] | None = {"$top": min(n, 200), "$select": _ITEM_SELECT}
        while url and len(items) < n:
            data = await self._get(token, url, params)
            params = None
            items.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        items = items[:n]
        if not items:
            return ToolResult(success=True, content=f"No OneDrive files found for '{q}'.")
        lines = [f"OneDrive search '{q}' ({len(items)} found):\n"]
        lines.extend(self._line(i, show_drive=True) for i in items)
        return ToolResult(success=True, content="\n".join(lines))

    async def _action_info(self, token: str, file_id: str | None = None, drive_id: str | None = None,
                           path: str | None = None, **kwargs: Any) -> ToolResult:
        r = await self._resolve(token, item_id=file_id, drive_id=drive_id, path=path,
                                default_root=False, select=_INFO_SELECT)
        m = r.meta
        lines = [f"{'Folder' if r.is_folder else 'File'}: {r.name}", f"  ID: {r.id}"]
        if r.drive_id:
            lines.append(f"  drive_id: {r.drive_id}")
        if r.is_folder:
            lines.append(f"  Items: {(m.get('folder') or {}).get('childCount', '?')}")
        else:
            lines.append(f"  Type: {(m.get('file') or {}).get('mimeType', '?')}")
            lines.append(f"  Size: {_size(r.size)}")
        parent = (m.get("parentReference") or {}).get("path")
        if parent:
            lines.append(f"  Location: {parent.split(':', 1)[-1] or '/'}")
        lines.append(f"  Created: {m.get('createdDateTime', '?')}"
                     + (f" by {((m.get('createdBy') or {}).get('user') or {}).get('displayName')}"
                        if ((m.get('createdBy') or {}).get('user') or {}).get('displayName') else ""))
        lines.append(f"  Modified: {m.get('lastModifiedDateTime', '?')}"
                     + (f" by {((m.get('lastModifiedBy') or {}).get('user') or {}).get('displayName')}"
                        if ((m.get('lastModifiedBy') or {}).get('user') or {}).get('displayName') else ""))
        if m.get("shared") is not None:
            lines.append(f"  Shared: {(m.get('shared') or {}).get('scope') or 'yes'}")
        if m.get("webUrl"):
            lines.append(f"  Web link: {m['webUrl']}")
        return ToolResult(success=True, content="\n".join(lines))

    # ── read / download ─────────────────────────────────────────────

    async def _fetch(self, token: str, r: _Resolved, *, as_pdf: bool = False) -> bytes | None:
        """The item's bytes (or its PDF conversion); None past the cap."""
        params = {"format": "pdf"} if as_pdf else None
        async with self._client.stream(
            "GET", f"{GRAPH_API}{r.path}/content", params=params, headers=self._headers(token),
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    return None
                chunks.append(chunk)
        return b"".join(chunks)

    async def _action_read(self, token: str, file_id: str | None = None, drive_id: str | None = None,
                           path: str | None = None, **kwargs: Any) -> ToolResult:
        from captain_claw.tools.google_drive import GoogleDriveTool

        r = await self._resolve(token, item_id=file_id, drive_id=drive_id, path=path, default_root=False)
        if r.is_folder:
            return await self._action_list(token, folder_id=r.id, drive_id=r.drive_id or None)
        if r.size > _MAX_DOWNLOAD_BYTES:
            return ToolResult(success=False, error=f"'{r.name}' is too large to read ({_size(r.size)}; max 50 MB).")
        suffix = Path(r.name).suffix.lower()
        mime = str(((r.meta.get("file") or {}).get("mimeType")) or "")
        header = f"[OneDrive: {r.name}]\n[ID: {r.id}]\n[Size: {_size(r.size)}]\n"

        tool_name = _EXTRACT_BY_SUFFIX.get(suffix)
        as_pdf = False
        if tool_name is None and suffix in _PDF_CONVERTIBLE:
            tool_name, as_pdf = "pdf_extract", True
        if tool_name:
            body = await self._fetch(token, r, as_pdf=as_pdf)
            if body is None:
                return ToolResult(success=False, error=f"'{r.name}' is too large (over 50 MB).")
            extract = GoogleDriveTool._get_extract_tool(tool_name)
            if extract is None:
                return ToolResult(success=False, error=f"Extract tool '{tool_name}' not available.")
            with tempfile.NamedTemporaryFile(suffix=".pdf" if as_pdf else suffix, delete=False) as tmp:
                tmp.write(body)
                tmp_path = tmp.name
            try:
                extra: dict[str, Any] = {"max_chars": _MAX_READ_CHARS} if tool_name == "xlsx_extract" else {}
                result = await extract.execute(path=tmp_path, **extra)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if not result.success:
                return result
            content = result.content
            if len(content) > _MAX_READ_CHARS:
                content = content[:_MAX_READ_CHARS] + "\n\n... [content truncated]"
            note = "[Converted to PDF by OneDrive for reading]\n" if as_pdf else ""
            return ToolResult(success=True, content=header + note + "\n" + content)

        if suffix in _TEXT_SUFFIXES or mime.startswith("text/") or mime in (
                "application/json", "application/xml", "application/x-yaml"):
            body = await self._fetch(token, r)
            if body is None:
                return ToolResult(success=False, error=f"'{r.name}' is too large (over 50 MB).")
            text = body.decode("utf-8", errors="replace")
            if len(text) > _MAX_READ_CHARS:
                text = text[:_MAX_READ_CHARS] + "\n\n... [content truncated]"
            return ToolResult(success=True, content=header + "\n" + text)

        return ToolResult(success=False, error=(
            f"'{r.name}' ({mime or suffix or 'binary'}) can't be shown as text. Use download to save "
            "a local copy (images: then image_vision)."
        ))

    async def _action_download(
        self, token: str, file_id: str | None = None, drive_id: str | None = None,
        path: str | None = None, output_path: str | None = None,
        runtime: dict[str, Any] | None = None, **kwargs: Any,
    ) -> ToolResult:
        from captain_claw import saved_attribution
        from captain_claw import speaker as _speaker
        from captain_claw.tools.google_drive import (
            _READER_BY_SUFFIX,
            GoogleDriveTool,
            _DownloadPathError,
            _safe_filename,
        )

        runtime = runtime or {}
        r = await self._resolve(token, item_id=file_id, drive_id=drive_id, path=path, default_root=False)
        if r.is_folder:
            return ToolResult(success=False, error=(
                f"'{r.name}' is a folder. list it (folder_id='{r.id}') and download files one at a time."
            ))
        if r.size > _MAX_DOWNLOAD_BYTES:
            return ToolResult(success=False, error=f"'{r.name}' is too large ({_size(r.size)}; max 50 MB).")
        suffix = Path(r.name).suffix.lower()
        wanted = Path(str(output_path or "").strip()).suffix.lower()
        as_pdf = wanted == ".pdf" and suffix != ".pdf" and (suffix in _PDF_CONVERTIBLE or suffix in _EXTRACT_BY_SUFFIX)
        default_name = _safe_filename(r.name, r.id)
        if as_pdf:
            default_name = str(Path(default_name).with_suffix(".pdf"))
        try:
            dest = GoogleDriveTool._download_dest(output_path, default_name, ".pdf" if as_pdf else "", runtime)
        except _DownloadPathError as exc:
            return ToolResult(success=False, error=str(exc))
        body = await self._fetch(token, r, as_pdf=as_pdf)
        if body is None:
            return ToolResult(success=False, error=f"'{r.name}' is too large (over 50 MB).")
        prior = saved_attribution.prior_creator(dest)
        member = _speaker.current()
        if (member is not None and dest.exists()
                and not saved_attribution.member_may_change(dest, member.speaker_id)):
            return ToolResult(success=False, error=_speaker.PATH_REFUSED_PREFIX + _speaker.FILE_NOT_YOURS_WHY)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_bytes, body)
        saved_attribution.note_write(dest, prior)
        registry = runtime.get("_file_registry")
        if registry is not None:
            try:
                registry.register(logical_path=str(output_path or "").strip() or dest.name,
                                  physical_path=str(dest), task_id=str(runtime.get("_task_id") or ""))
            except Exception:
                pass
        reader = _READER_BY_SUFFIX.get(dest.suffix.lower(), "read")
        return ToolResult(success=True, content=(
            f"Downloaded '{r.name}' from OneDrive" + (" (converted to PDF)" if as_pdf else "") + ".\n"
            f"  File ID: {r.id}\n  Path: {dest}\n  Size: {_size(len(body))}\n"
            f"Use {reader}(path=\"{dest}\") to view it. This is a local copy — to change the "
            f"OneDrive file, edit the copy and update file_id='{r.id}' with local_path."
        ))

    # ── writes ──────────────────────────────────────────────────────

    async def _upload_bytes(self, token: str, target: str, data: bytes, *, replace: bool) -> dict[str, Any]:
        """PUT *data* to *target* (``<parent>:/<name>:`` or ``<item>``) —
        one request up to 4 MB, else an upload session."""
        conflict = "replace" if replace else "rename"
        if len(data) <= _SIMPLE_UPLOAD_BYTES:
            resp = await self._client.put(
                f"{GRAPH_API}{target}/content",
                params={"@microsoft.graph.conflictBehavior": conflict},
                content=data,
                headers={**self._headers(token), "Content-Type": "application/octet-stream"},
            )
            resp.raise_for_status()
            return resp.json()
        resp = await self._client.post(
            f"{GRAPH_API}{target}/createUploadSession",
            json={"item": {"@microsoft.graph.conflictBehavior": conflict}},
            headers=self._headers(token),
        )
        resp.raise_for_status()
        upload_url = str(resp.json().get("uploadUrl") or "")
        if not upload_url:
            raise RuntimeError("OneDrive returned no upload session URL.")
        total = len(data)
        result: dict[str, Any] = {}
        for start in range(0, total, _CHUNK):
            chunk = data[start:start + _CHUNK]
            end = start + len(chunk) - 1
            # The upload URL is pre-authenticated: no Authorization header.
            put = await self._client.put(
                upload_url, content=chunk,
                headers={"Content-Length": str(len(chunk)), "Content-Range": f"bytes {start}-{end}/{total}"},
            )
            put.raise_for_status()
            if put.status_code in (200, 201):
                result = put.json()
        return result

    async def _parent(self, token: str, folder_id: str | None, drive_id: str | None, path: str | None) -> _Resolved:
        parent = await self._resolve(token, item_id=folder_id, drive_id=drive_id, path=path, default_root=True)
        if not parent.is_folder:
            raise LookupError(f"'{parent.name}' is a file, not a folder.")
        return parent

    @staticmethod
    def _child(parent: _Resolved, name: str) -> str:
        return f"{parent.path}:/{quote(name, safe='')}:"

    @staticmethod
    def _created(verb: str, item: dict[str, Any], parent: _Resolved | None = None) -> str:
        msg = f"{verb} '{item.get('name', '?')}' on OneDrive.\n  ID: {item.get('id', '?')}"
        if parent is not None:
            msg += f"\n  Folder: {parent.name}"
        if item.get("size") is not None:
            msg += f"\n  Size: {_size(item.get('size'))}"
        if item.get("webUrl"):
            msg += f"\n  Link: {item['webUrl']}"
        return msg

    async def _action_upload(
        self, token: str, local_path: str = "", name: str | None = None,
        folder_id: str | None = None, drive_id: str | None = None, path: str | None = None, **kwargs: Any,
    ) -> ToolResult:
        if not local_path:
            return ToolResult(success=False, error="local_path is required for upload.")
        src = Path(local_path).expanduser().resolve()
        if not src.is_file():
            return ToolResult(success=False, error=f"Local file not found: {local_path}")
        if src.stat().st_size > _MAX_UPLOAD_BYTES:
            return ToolResult(success=False, error=f"'{src.name}' is too large to upload (max 250 MB).")
        parent = await self._parent(token, folder_id, drive_id, path)
        file_name = (name or src.name).strip()
        data = await asyncio.to_thread(src.read_bytes)
        item = await self._upload_bytes(token, self._child(parent, file_name), data, replace=False)
        msg = self._created("Uploaded", item, parent)
        if item.get("name") and item["name"] != file_name:
            msg += f"\n  (A file named '{file_name}' already existed, so it was saved as '{item['name']}'.)"
        return ToolResult(success=True, content=msg)

    async def _action_create(
        self, token: str, name: str = "", content: str = "", folder_id: str | None = None,
        drive_id: str | None = None, path: str | None = None, **kwargs: Any,
    ) -> ToolResult:
        file_name = str(name or "").strip()
        if not file_name:
            return ToolResult(success=False, error="name is required for create (with its extension, e.g. notes.md).")
        parent = await self._parent(token, folder_id, drive_id, path)
        item = await self._upload_bytes(token, self._child(parent, file_name),
                                        (content or "").encode("utf-8"), replace=False)
        msg = self._created("Created", item, parent)
        if item.get("name") and item["name"] != file_name:
            msg += f"\n  (A file named '{file_name}' already existed, so this one is '{item['name']}'.)"
        return ToolResult(success=True, content=msg)

    async def _action_create_folder(
        self, token: str, name: str = "", folder_id: str | None = None,
        drive_id: str | None = None, path: str | None = None, **kwargs: Any,
    ) -> ToolResult:
        folder_name = str(name or "").strip()
        if not folder_name:
            return ToolResult(success=False, error="name is required for create_folder.")
        parent = await self._parent(token, folder_id, drive_id, path)
        resp = await self._client.post(
            f"{GRAPH_API}{parent.path}/children",
            json={"name": folder_name, "folder": {}, "@microsoft.graph.conflictBehavior": "rename"},
            headers=self._headers(token),
        )
        resp.raise_for_status()
        return ToolResult(success=True, content=self._created("Created folder", resp.json(), parent))

    async def _action_update(
        self, token: str, file_id: str | None = None, drive_id: str | None = None,
        path: str | None = None, content: str | None = None, local_path: str | None = None, **kwargs: Any,
    ) -> ToolResult:
        if content is not None:
            data = content.encode("utf-8")
        elif local_path:
            src = Path(local_path).expanduser().resolve()
            if not src.is_file():
                return ToolResult(success=False, error=f"Local file not found: {local_path}")
            if src.stat().st_size > _MAX_UPLOAD_BYTES:
                return ToolResult(success=False, error=f"'{src.name}' is too large (max 250 MB).")
            data = await asyncio.to_thread(src.read_bytes)
        else:
            return ToolResult(success=False, error="update needs content or local_path.")
        r = await self._resolve(token, item_id=file_id, drive_id=drive_id, path=path, default_root=False)
        if r.is_folder:
            return ToolResult(success=False, error=f"'{r.name}' is a folder; update replaces a file's content.")
        suffix = Path(r.name).suffix.lower()
        if content is not None and (suffix in _EXTRACT_BY_SUFFIX or suffix in _PDF_CONVERTIBLE):
            return ToolResult(success=False, error=(
                f"Refused: '{r.name}' is a binary document — text content would corrupt it. Download "
                "it, edit the local copy, then update with local_path."
            ))
        if local_path and Path(local_path).suffix.lower() != suffix:
            guess = mimetypes.guess_type(str(local_path))[0] or Path(local_path).suffix
            return ToolResult(success=False, error=(
                f"Refused: '{r.name}' and the local file ({guess}) are different file types — "
                "replacing one with the other would corrupt it. Upload the local file as a new file instead."
            ))
        item = await self._upload_bytes(token, r.path, data, replace=True)
        return ToolResult(success=True, content=self._created("Updated", item or {"name": r.name, "id": r.id}))

    async def close(self) -> None:
        await self._client.aclose()
