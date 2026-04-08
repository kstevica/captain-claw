"""Google Docs / Sheets / Slides actions for the gws tool.

Mixin used by :class:`GwsTool`. Provides docs_read and docs_append.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools._gws_runtime import _strip_base64_images
from captain_claw.tools.registry import ToolResult

log = get_logger(__name__)


class GwsDocsMixin:
    """Google Docs/Sheets/Slides actions."""

    # Google Workspace types that need binary export (XLSX / PPTX) followed
    # by local extraction rather than plain-text export.
    _BINARY_EXPORT_MAP: dict[str, tuple[str, str]] = {
        "application/vnd.google-apps.spreadsheet": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xlsx",
        ),
        "application/vnd.google-apps.presentation": (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".pptx",
        ),
    }

    async def _docs_read(
        self, binary: str, file_id: str = "", saved_base: Any = None, **kw: Any,
    ) -> ToolResult:
        """Read a Google Doc / Sheet / Slides and return content inline."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for docs_read.")

        # gws export writes a `download.bin` side-effect file in cwd.
        _stale = Path.cwd() / "download.bin"
        if _stale.exists():
            try:
                _stale.unlink()
            except OSError:
                pass

        meta_params = {
            "fileId": file_id,
            "fields": "name,mimeType",
            "supportsAllDrives": "true",
        }
        meta_result = await self._run_gws(
            binary,
            ["drive", "files", "get", "--params", json.dumps(meta_params)],
        )
        doc_name = file_id
        file_mime = ""
        if meta_result.success:
            try:
                _meta = json.loads(meta_result.content)
                doc_name = _meta.get("name", file_id)
                file_mime = _meta.get("mimeType", "")
            except (json.JSONDecodeError, TypeError):
                pass

        if file_mime in self._BINARY_EXPORT_MAP:
            return await self._docs_read_binary_export(
                binary, file_id, doc_name, file_mime,
            )

        params = {
            "fileId": file_id,
            "mimeType": "text/markdown",
            "supportsAllDrives": "true",
        }
        args = ["drive", "files", "export", "--params", json.dumps(params)]
        result = await self._run_gws(binary, args, json_output=False)

        if not result.success:
            params["mimeType"] = "text/plain"
            args = ["drive", "files", "export", "--params", json.dumps(params)]
            result = await self._run_gws(binary, args, json_output=False)

        if _stale.exists():
            try:
                _stale.unlink()
            except OSError:
                pass

        if result.success and result.content:
            cleaned = _strip_base64_images(result.content)
            result = ToolResult(success=True, content=cleaned)

        return result

    async def _docs_read_binary_export(
        self,
        binary: str,
        file_id: str,
        doc_name: str,
        file_mime: str,
    ) -> ToolResult:
        """Export a Google Sheet/Presentation to XLSX/PPTX and extract text inline."""
        import tempfile

        export_mime, suffix = self._BINARY_EXPORT_MAP[file_mime]

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            params = {
                "fileId": file_id,
                "mimeType": export_mime,
                "supportsAllDrives": "true",
            }
            args = [
                "drive", "files", "export",
                "--params", json.dumps(params),
                "--output", str(tmp_path),
            ]
            result = await self._run_gws(binary, args, json_output=False)
            if not result.success:
                return result

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                return ToolResult(
                    success=False,
                    error=f"gws export produced empty file for {doc_name!r}",
                )

            from captain_claw.tools.document_extract import (
                _extract_pptx_markdown,
                _extract_xlsx_markdown,
            )

            if suffix == ".xlsx":
                content = await asyncio.to_thread(
                    _extract_xlsx_markdown, tmp_path, 200,
                )
            else:
                content = await asyncio.to_thread(
                    _extract_pptx_markdown, tmp_path, 200,
                )

            return ToolResult(success=True, content=content)

        except Exception as e:
            log.error("docs_read binary export failed", file_id=file_id, error=str(e))
            return ToolResult(success=False, error=str(e))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            _stale = Path.cwd() / "download.bin"
            if _stale.exists():
                try:
                    _stale.unlink()
                except OSError:
                    pass

    async def _docs_append(
        self, binary: str, file_id: str = "", content: str = "", **kw: Any,
    ) -> ToolResult:
        """Append text to a Google Doc."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for docs_append.")
        if not content:
            return ToolResult(success=False, error="content is required for docs_append.")

        args = ["docs", "+write", "--id", file_id, "--text", content]
        return await self._run_gws(binary, args)
