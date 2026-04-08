"""Google Drive actions for the gws tool.

Mixin used by :class:`GwsTool`. Provides drive_list, drive_search,
drive_download, drive_info, drive_create, plus the pagination and
recursive-listing helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools._gws_runtime import _MAX_OUTPUT_CHARS, _strip_base64_images
from captain_claw.tools.registry import ToolResult

log = get_logger(__name__)


class GwsDriveMixin:
    """Google Drive actions (list/search/download/info/create)."""

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_file_size(size_str: str) -> str:
        """Convert byte string to human-readable size."""
        try:
            size = int(size_str)
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if size < 1024:
                    return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
                size /= 1024
            return f"{size:.1f} PB"
        except (ValueError, TypeError):
            return str(size_str) if size_str else "—"

    @staticmethod
    def _readable_mime(mime: str) -> str:
        """Convert MIME type to a short human-readable label."""
        _MAP = {
            "application/vnd.google-apps.folder": "Folder",
            "application/vnd.google-apps.document": "Google Doc",
            "application/vnd.google-apps.spreadsheet": "Google Sheet",
            "application/vnd.google-apps.presentation": "Google Slides",
            "application/vnd.google-apps.form": "Google Form",
            "application/vnd.google-apps.drawing": "Google Drawing",
            "application/vnd.google-apps.site": "Google Site",
            "application/pdf": "PDF",
            "text/csv": "CSV",
            "text/plain": "Text",
            "application/json": "JSON",
        }
        if mime in _MAP:
            return _MAP[mime]
        if "image/" in mime:
            return "Image"
        if "video/" in mime:
            return "Video"
        if "audio/" in mime:
            return "Audio"
        if "officedocument" in mime or "msword" in mime:
            return "Office Doc"
        return mime.split("/")[-1].upper() if "/" in mime else mime or "Unknown"

    @staticmethod
    def _format_drive_file_line(f: dict[str, Any]) -> str:
        """Format a single Drive file dict into a text line."""
        name = f.get("name", "Unnamed")
        link = f.get("webViewLink", "—")
        size = GwsDriveMixin._format_file_size(f.get("size", ""))
        ftype = GwsDriveMixin._readable_mime(f.get("mimeType", ""))
        return f"{name}, {link}, {size}, {ftype}"

    # ------------------------------------------------------------------
    # drive_list / drive_search
    # ------------------------------------------------------------------

    async def _drive_list(
        self, binary: str, folder_id: str = "", max_results: int | float | None = None,
        output_file: str = "", recursive: bool = False, **kw: Any,
    ) -> ToolResult:
        """List files in a Google Drive folder with auto-pagination."""
        if recursive:
            if not output_file:
                return ToolResult(
                    success=False,
                    error="recursive=true requires output_file to be set.",
                )
            return await self._drive_recursive_list(binary, output_file, int(max_results or 10000))

        limit = int(max_results or 100)
        page_size = min(limit, 1000)
        params: dict[str, Any] = {"pageSize": page_size}

        if folder_id:
            escaped = folder_id.replace("'", "\\'")
            params["q"] = f"'{escaped}' in parents and trashed = false"
        else:
            params["q"] = "'root' in parents and trashed = false"

        params["orderBy"] = "modifiedTime desc"
        params["fields"] = "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink,parents)"
        params["corpora"] = "allDrives"
        params["supportsAllDrives"] = "true"
        params["includeItemsFromAllDrives"] = "true"

        return await self._drive_paginated_list(binary, params, limit, output_file)

    async def _drive_search(
        self, binary: str, query: str = "", max_results: int | float | None = None,
        output_file: str = "", **kw: Any,
    ) -> ToolResult:
        """Search for files across Google Drive with auto-pagination."""
        if not query:
            return ToolResult(success=False, error="query is required for drive_search.")

        limit = int(max_results or 100)
        page_size = min(limit, 1000)
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        q = f"(name contains '{escaped}' or fullText contains '{escaped}') and trashed = false"

        params: dict[str, Any] = {
            "pageSize": page_size,
            "q": q,
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)",
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        return await self._drive_paginated_list(binary, params, limit, output_file)

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    async def _drive_paginated_list(
        self,
        binary: str,
        params: dict[str, Any],
        limit: int,
        output_file: str = "",
    ) -> ToolResult:
        """Paginate through Drive file list results.

        When *output_file* is set, writes each file line directly to disk
        as it is fetched and returns a short summary. Otherwise returns
        raw JSON (capped by ``_MAX_OUTPUT_CHARS``).
        """
        stream_cb = getattr(self, "_stream_callback", None)
        all_files: list[dict[str, Any]] = []
        page_token: str | None = None
        _max_pages = 50
        page_num = 0

        out_fh = None
        out_path: Path | None = None
        if output_file:
            out_path = Path(output_file).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(out_path, "w", encoding="utf-8")  # noqa: SIM115
            if stream_cb:
                stream_cb(f"Writing results to {out_path}\n")

        try:
            for _ in range(_max_pages):
                page_num += 1
                page_params = dict(params)
                if page_token:
                    page_params["pageToken"] = page_token

                if stream_cb:
                    stream_cb(f"Fetching page {page_num}...\n")

                args = ["drive", "files", "list", "--params", json.dumps(page_params)]
                result = await self._run_gws(binary, args)

                if not result.success:
                    if all_files:
                        break
                    return result

                try:
                    data = json.loads(result.content)
                except (json.JSONDecodeError, TypeError):
                    if all_files:
                        break
                    return result

                files = data.get("files", [])
                if not files:
                    break

                for f in files:
                    if len(all_files) >= limit:
                        break
                    all_files.append(f)
                    if out_fh:
                        out_fh.write(self._format_drive_file_line(f) + "\n")

                if stream_cb:
                    stream_cb(f"  ↳ {len(all_files)} files collected so far\n")

                page_token = data.get("nextPageToken")
                if not page_token or len(all_files) >= limit:
                    break
        finally:
            if out_fh:
                out_fh.close()

        if out_path:
            summary = (
                f"✓ {len(all_files)} files written to {out_path}\n"
                f"  Format: <filename>, <google docs link>, <size>, <type>\n"
                f"  Pages fetched: {page_num}"
            )
            if stream_cb:
                stream_cb(f"\n{summary}\n")
            return ToolResult(success=True, content=summary)

        all_files = all_files[:limit]
        output = json.dumps({"files": all_files, "totalFiles": len(all_files)}, indent=2)
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n\n... [output truncated]"
        return ToolResult(success=True, content=output)

    # ------------------------------------------------------------------
    # Recursive listing (flat-fetch + tree reconstruction)
    # ------------------------------------------------------------------

    async def _drive_recursive_list(
        self,
        binary: str,
        output_file: str,
        limit: int = 10000,
    ) -> ToolResult:
        """Fetch ALL Drive files, reconstruct folder paths, write to file."""
        stream_cb = getattr(self, "_stream_callback", None)

        if stream_cb:
            stream_cb("Phase 1: Fetching all files from Google Drive...\n")

        params: dict[str, Any] = {
            "pageSize": 1000,
            "q": "trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,size,webViewLink,parents)",
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        all_items: list[dict[str, Any]] = []
        page_token: str | None = None
        page_num = 0

        for _ in range(100):  # safety cap
            page_num += 1
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token

            if stream_cb:
                stream_cb(f"  Fetching page {page_num}...\n")

            args = ["drive", "files", "list", "--params", json.dumps(page_params)]
            result = await self._run_gws(binary, args)

            if not result.success:
                if all_items:
                    if stream_cb:
                        stream_cb(f"  ⚠ Page {page_num} failed, continuing with {len(all_items)} items\n")
                    break
                return result

            try:
                data = json.loads(result.content)
            except (json.JSONDecodeError, TypeError):
                if all_items:
                    break
                return result

            files = data.get("files", [])
            if not files:
                break

            all_items.extend(files)
            if stream_cb:
                stream_cb(f"  ↳ {len(all_items)} items collected\n")

            if len(all_items) >= limit:
                all_items = all_items[:limit]
                break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        if not all_items:
            return ToolResult(success=True, content="No files found in Google Drive.")

        if stream_cb:
            stream_cb(f"\nPhase 2: Building folder tree from {len(all_items)} items...\n")

        FOLDER_MIME = "application/vnd.google-apps.folder"
        folder_map: dict[str, str] = {}
        for item in all_items:
            if item.get("mimeType") == FOLDER_MIME:
                folder_map[item["id"]] = item.get("name", "Unnamed")

        _path_cache: dict[str, str] = {}

        def _resolve_path(parent_id: str) -> str:
            if parent_id in _path_cache:
                return _path_cache[parent_id]
            if parent_id not in folder_map:
                _path_cache[parent_id] = "My Drive"
                return "My Drive"
            folder_name = folder_map[parent_id]
            for item in all_items:
                if item.get("id") == parent_id:
                    parents = item.get("parents", [])
                    if parents:
                        parent_path = _resolve_path(parents[0])
                        full = f"{parent_path}/{folder_name}"
                    else:
                        full = folder_name
                    _path_cache[parent_id] = full
                    return full
            _path_cache[parent_id] = folder_name
            return folder_name

        if stream_cb:
            stream_cb("Phase 3: Writing output file...\n")

        out_path = Path(output_file).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_count = 0
        folder_count = 0
        folders_seen: set[str] = set()

        with open(out_path, "w", encoding="utf-8") as fh:
            for item in all_items:
                name = item.get("name", "Unnamed")
                mime = item.get("mimeType", "")
                parents = item.get("parents", [])
                parent_path = _resolve_path(parents[0]) if parents else "My Drive"

                if parents:
                    folders_seen.add(parents[0])

                full_path = f"{parent_path}/{name}"
                link = item.get("webViewLink", "—")
                size = self._format_file_size(item.get("size", ""))
                ftype = self._readable_mime(mime)

                if mime == FOLDER_MIME:
                    folder_count += 1
                else:
                    file_count += 1

                fh.write(f"{full_path}, {link}, {size}, {ftype}\n")

        summary = (
            f"✓ Recursive listing complete → {out_path}\n"
            f"  Total items: {len(all_items)} ({file_count} files, {folder_count} folders)\n"
            f"  Unique folders traversed: {len(folders_seen)}\n"
            f"  Pages fetched: {page_num}\n"
            f"  Format: <full/path/name>, <link>, <size>, <type>"
        )
        if stream_cb:
            stream_cb(f"\n{summary}\n")
        return ToolResult(success=True, content=summary)

    # ------------------------------------------------------------------
    # drive_download / drive_info / drive_create
    # ------------------------------------------------------------------

    async def _drive_download(
        self,
        binary: str,
        file_id: str = "",
        output_path: str = "",
        saved_base: Any = None,
        runtime_base: Any = None,
        **kw: Any,
    ) -> ToolResult:
        """Download/export a file from Google Drive."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for drive_download.")

        meta_args = [
            "drive", "files", "get",
            "--params", json.dumps({"fileId": file_id, "fields": "id,name,mimeType", "supportsAllDrives": "true"}),
        ]
        meta_result = await self._run_gws(binary, meta_args)
        if not meta_result.success:
            return meta_result

        file_name = file_id
        mime_type = ""
        try:
            meta = json.loads(meta_result.content)
            file_name = meta.get("name", file_id)
            mime_type = meta.get("mimeType", "")
        except (json.JSONDecodeError, TypeError):
            pass

        google_export_map = {
            "application/vnd.google-apps.document": ("text/markdown", ".md"),
            "application/vnd.google-apps.spreadsheet": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xlsx",
            ),
            "application/vnd.google-apps.presentation": (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".pptx",
            ),
            "application/vnd.google-apps.drawing": ("image/svg+xml", ".svg"),
        }

        if output_path:
            _out = Path(output_path).expanduser()
            if _out.is_absolute():
                dest = _out.resolve()
            elif runtime_base:
                dest = (Path(str(runtime_base)) / _out).resolve()
            else:
                dest = _out.resolve()
        elif saved_base:
            dest = Path(str(saved_base)) / file_name
        else:
            dest = Path.cwd() / file_name

        dest.parent.mkdir(parents=True, exist_ok=True)

        if mime_type in google_export_map:
            export_mime, ext = google_export_map[mime_type]
            if not dest.suffix:
                dest = dest.with_suffix(ext)

            args = [
                "drive", "files", "export",
                "--params", json.dumps({
                    "fileId": file_id,
                    "mimeType": export_mime,
                    "supportsAllDrives": "true",
                }),
                "--output", str(dest),
            ]
        else:
            args = [
                "drive", "files", "get",
                "--params", json.dumps({"fileId": file_id, "alt": "media", "supportsAllDrives": "true"}),
                "--output", str(dest),
            ]

        result = await self._run_gws(binary, args, json_output=False)
        if result.success:
            if dest.exists() and dest.suffix in (".md", ".txt") and mime_type in google_export_map:
                try:
                    raw = dest.read_text(encoding="utf-8", errors="replace")
                    cleaned = _strip_base64_images(raw)
                    if len(cleaned) < len(raw):
                        dest.write_text(cleaned, encoding="utf-8")
                except OSError:
                    pass
            _stale = Path.cwd() / "download.bin"
            if _stale.exists():
                try:
                    _stale.unlink()
                except OSError:
                    pass
            abs_dest = str(dest.resolve())
            return ToolResult(
                success=True,
                content=(
                    f"Downloaded '{file_name}' to {abs_dest}\n"
                    f"Use read(path=\"{abs_dest}\") to view the file contents."
                ),
            )
        return result

    async def _drive_info(
        self, binary: str, file_id: str = "", **kw: Any,
    ) -> ToolResult:
        """Get metadata for a Drive file."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for drive_info.")

        params = {
            "fileId": file_id,
            "fields": "id,name,mimeType,size,modifiedTime,createdTime,parents,webViewLink,owners,shared,description",
            "supportsAllDrives": "true",
        }
        args = ["drive", "files", "get", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _drive_create(
        self,
        binary: str,
        name: str = "",
        content: str = "",
        mime_type: str = "",
        folder_id: str = "",
        **kw: Any,
    ) -> ToolResult:
        """Create a new file on Google Drive."""
        if not name:
            return ToolResult(success=False, error="name is required for drive_create.")

        body: dict[str, Any] = {"name": name}
        if mime_type:
            body["mimeType"] = mime_type
        if folder_id:
            body["parents"] = [folder_id]

        if content:
            import tempfile
            suffix = ".txt"
            if mime_type and "spreadsheet" in mime_type:
                suffix = ".csv"
            elif mime_type and "document" in mime_type:
                suffix = ".md"

            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                args = ["drive", "+upload", "--file", tmp_path, "--name", name]
                if mime_type:
                    args.extend(["--mime-type", mime_type])
                if folder_id:
                    args.extend(["--parent", folder_id])
                return await self._run_gws(binary, args)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            args = ["drive", "files", "create", "--json", json.dumps(body)]
            return await self._run_gws(binary, args)
