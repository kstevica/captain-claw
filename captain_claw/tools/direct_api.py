"""Direct API Calls tool — register and execute HTTP API endpoints directly."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH"}


class DirectApiTool(Tool):
    """Register and execute direct HTTP API calls."""

    name = "direct_api"
    description = (
        "Register and execute direct HTTP API calls. Each entry is a single "
        "API endpoint with URL, HTTP method (GET/POST/PUT/PATCH — DELETE "
        "is rejected for safety), payload schemas, and optional auth. "
        "Actions: 'add' to register, 'list' to browse, 'show' for details, "
        "'call' to execute and record stats, 'test' to execute without stats, "
        "'update' to modify, 'remove' to delete, 'auth_from_browser' to "
        "capture auth token from the active browser session."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "add", "list", "show", "call", "test",
                    "update", "remove", "auth_from_browser",
                ],
                "description": "Operation to perform.",
            },
            "name": {
                "type": "string",
                "description": "Friendly name for the API call.",
            },
            "call_id": {
                "type": "string",
                "description": "ID, #index, or name of a registered call.",
            },
            "url": {
                "type": "string",
                "description": "Full URL of the API endpoint.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH"],
                "description": "HTTP method. DELETE is not allowed.",
            },
            "description": {
                "type": "string",
                "description": "What this API does.",
            },
            "input_payload": {
                "type": "string",
                "description": "Input schema docs (JSON/YAML/XML/text).",
            },
            "result_payload": {
                "type": "string",
                "description": "Response schema docs (any format).",
            },
            "headers": {
                "type": "string",
                "description": "JSON dict of extra headers beyond auth.",
            },
            "auth_type": {
                "type": "string",
                "enum": ["bearer", "api_key", "basic", "cookie", "none"],
                "description": "Authentication type.",
            },
            "auth_token": {
                "type": "string",
                "description": "Auth credential value.",
            },
            "app_name": {
                "type": "string",
                "description": "App grouping label.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags.",
            },
            "payload": {
                "type": "string",
                "description": "JSON body to send when executing (for call/test).",
            },
            "query_params": {
                "type": "string",
                "description": "JSON dict of query parameters (for call/test).",
            },
        },
        "required": ["action"],
    }
    timeout_seconds = 60.0

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        sm = get_session_manager()
        try:
            if action == "add":
                return await self._add(sm, **kwargs)
            if action == "list":
                return await self._list(sm, **kwargs)
            if action == "show":
                return await self._show(sm, **kwargs)
            if action in ("call", "test"):
                return await self._execute_call(sm, record=action == "call", **kwargs)
            if action == "update":
                return await self._update(sm, **kwargs)
            if action == "remove":
                return await self._remove(sm, **kwargs)
            if action == "auth_from_browser":
                return await self._auth_from_browser(sm, **kwargs)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("direct_api tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @staticmethod
    async def _add(sm: Any, **kw: Any) -> ToolResult:
        name = (kw.get("name") or "").strip()
        url = (kw.get("url") or "").strip()
        if not name:
            return ToolResult(success=False, error="'name' is required for add.")
        if not url:
            return ToolResult(success=False, error="'url' is required for add.")
        method = (kw.get("method") or "GET").upper()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                success=False,
                error=f"HTTP method '{method}' is not allowed. "
                      f"Allowed: {', '.join(sorted(_ALLOWED_METHODS))}.",
            )
        entry = await sm.create_direct_api_call(
            name=name, url=url, method=method,
            description=kw.get("description") or "",
            input_payload=kw.get("input_payload") or "",
            result_payload=kw.get("result_payload") or "",
            headers=kw.get("headers"),
            auth_type=kw.get("auth_type"),
            auth_token=kw.get("auth_token"),
            auth_source="manual" if kw.get("auth_token") else None,
            app_name=kw.get("app_name"),
            tags=kw.get("tags"),
        )
        return ToolResult(
            success=True,
            content=(
                f"Registered direct API call #{entry.id[:8]}: "
                f"{entry.method} {entry.name} ({entry.url})"
            ),
        )

    @staticmethod
    async def _list(sm: Any, **kw: Any) -> ToolResult:
        app_name = (kw.get("app_name") or "").strip() or None
        items = await sm.list_direct_api_calls(limit=50, app_name=app_name)
        if not items:
            return ToolResult(success=True, content="No direct API calls registered.")
        lines: list[str] = []
        for idx, c in enumerate(items, 1):
            auth = f" [{c.auth_type}]" if c.auth_type else ""
            app = f" ({c.app_name})" if c.app_name else ""
            lines.append(
                f"#{idx} {c.method:5s} {c.name}{auth}{app}  "
                f"url={c.url}  uses={c.use_count}  id={c.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _show(sm: Any, **kw: Any) -> ToolResult:
        call_id = kw.get("call_id") or kw.get("name") or ""
        if not call_id:
            return ToolResult(success=False, error="'call_id' is required for show.")
        entry = await sm.select_direct_api_call(call_id)
        if not entry:
            return ToolResult(success=False, error=f"Not found: {call_id}")
        parts = [
            f"Name: {entry.name}",
            f"ID: {entry.id}",
            f"Method: {entry.method}",
            f"URL: {entry.url}",
        ]
        if entry.description:
            parts.append(f"Description: {entry.description}")
        if entry.auth_type:
            parts.append(f"Auth: {entry.auth_type} (source: {entry.auth_source or 'unknown'})")
        if entry.input_payload:
            parts.append(f"Input payload:\n{entry.input_payload}")
        if entry.result_payload:
            parts.append(f"Result payload:\n{entry.result_payload}")
        if entry.headers:
            parts.append(f"Extra headers: {entry.headers}")
        if entry.app_name:
            parts.append(f"App: {entry.app_name}")
        if entry.tags:
            parts.append(f"Tags: {entry.tags}")
        parts.append(f"Uses: {entry.use_count}")
        if entry.last_used_at:
            parts.append(f"Last used: {entry.last_used_at}")
        if entry.last_status_code is not None:
            parts.append(f"Last status: {entry.last_status_code}")
        if entry.last_response_preview:
            parts.append(f"Last response preview:\n{entry.last_response_preview}")
        parts.append(f"Created: {entry.created_at}")
        return ToolResult(success=True, content="\n".join(parts))

    @staticmethod
    async def _execute_call(sm: Any, record: bool = True, **kw: Any) -> ToolResult:
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        call_id = kw.get("call_id") or kw.get("name") or ""
        if not call_id:
            return ToolResult(success=False, error="'call_id' is required.")
        entry = await sm.select_direct_api_call(call_id)
        if not entry:
            return ToolResult(success=False, error=f"Not found: {call_id}")

        method = entry.method.upper()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                success=False,
                error=f"Method '{method}' is not allowed for execution.",
            )

        # Build auth headers
        auth_headers = ApiReplayEngine.resolve_auth_headers(
            entry.auth_type, entry.auth_token,
        )

        # Parse extra headers from entry
        extra_headers: dict[str, str] = {}
        if entry.headers:
            try:
                extra_headers = json.loads(entry.headers)
            except (json.JSONDecodeError, TypeError):
                pass

        merged_headers = {**extra_headers, **auth_headers}

        # Parse payload and query_params
        body_json: dict[str, Any] | None = None
        raw_payload = kw.get("payload")
        if raw_payload:
            try:
                body_json = json.loads(raw_payload)
            except (json.JSONDecodeError, TypeError):
                body_json = None

        query_params: dict[str, str] | None = None
        raw_qp = kw.get("query_params")
        if raw_qp:
            try:
                query_params = json.loads(raw_qp)
            except (json.JSONDecodeError, TypeError):
                pass

        # Split URL for ApiReplayEngine
        parsed = urlparse(entry.url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        endpoint = parsed.path
        if parsed.query:
            endpoint += f"?{parsed.query}"

        result = await ApiReplayEngine.replay(
            base_url=base_url,
            endpoint=endpoint,
            method=method,
            headers=merged_headers or None,
            query_params=query_params,
            body_json=body_json,
            timeout=30.0,
        )

        # Record usage
        if record and result.status_code:
            preview = (result.response_body or "")[:500]
            await sm.record_direct_api_call_usage(
                entry.id, result.status_code, preview,
            )

        # Format result
        lines = [
            f"{'Executed' if record else 'Tested'}: {entry.method} {entry.url}",
            f"Status: {result.status_code or 'N/A'}",
            f"Elapsed: {result.elapsed_ms:.0f}ms",
        ]
        if result.error:
            lines.append(f"Error: {result.error}")
        if result.response_body:
            body = result.response_body
            if len(body) > 2000:
                body = body[:2000] + f"\n... ({len(result.response_body)} total chars)"
            lines.append(f"Response:\n{body}")
        return ToolResult(success=result.success, content="\n".join(lines))

    @staticmethod
    async def _update(sm: Any, **kw: Any) -> ToolResult:
        call_id = kw.get("call_id") or kw.get("name") or ""
        if not call_id:
            return ToolResult(success=False, error="'call_id' is required for update.")
        entry = await sm.select_direct_api_call(call_id)
        if not entry:
            return ToolResult(success=False, error=f"Not found: {call_id}")

        updates: dict[str, Any] = {}
        for field in (
            "name", "url", "description", "input_payload", "result_payload",
            "headers", "auth_type", "auth_token", "auth_source", "app_name", "tags",
        ):
            val = kw.get(field)
            if val is not None:
                updates[field] = val

        method = kw.get("method")
        if method is not None:
            method = method.upper()
            if method not in _ALLOWED_METHODS:
                return ToolResult(
                    success=False,
                    error=f"Method '{method}' is not allowed.",
                )
            updates["method"] = method

        if not updates:
            return ToolResult(success=False, error="No fields to update.")

        ok = await sm.update_direct_api_call(entry.id, **updates)
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(
            success=True,
            content=f"Updated direct API call #{entry.id[:8]}: {entry.name}",
        )

    @staticmethod
    async def _remove(sm: Any, **kw: Any) -> ToolResult:
        call_id = kw.get("call_id") or kw.get("name") or ""
        if not call_id:
            return ToolResult(success=False, error="'call_id' is required for remove.")
        entry = await sm.select_direct_api_call(call_id)
        if not entry:
            return ToolResult(success=False, error=f"Not found: {call_id}")
        ok = await sm.delete_direct_api_call(entry.id)
        if not ok:
            return ToolResult(success=False, error="Delete failed.")
        return ToolResult(
            success=True,
            content=f"Removed direct API call #{entry.id[:8]}: {entry.name}",
        )

    @staticmethod
    async def _auth_from_browser(sm: Any, **kw: Any) -> ToolResult:
        call_id = kw.get("call_id") or kw.get("name") or ""
        if not call_id:
            return ToolResult(
                success=False,
                error="'call_id' is required for auth_from_browser.",
            )
        entry = await sm.select_direct_api_call(call_id)
        if not entry:
            return ToolResult(success=False, error=f"Not found: {call_id}")

        # Access the browser tool's session
        from captain_claw.tools.registry import get_tool_registry
        registry = get_tool_registry()
        browser_tool = registry.get("browser")
        if browser_tool is None:
            return ToolResult(
                success=False,
                error="Browser tool is not available.",
            )
        session = getattr(browser_tool, "_session", None)
        if session is None or not getattr(session, "is_alive", False):
            return ToolResult(
                success=False,
                error="No active browser session. Open the browser first.",
            )

        domain = urlparse(entry.url).netloc

        # Try network captures first
        network = getattr(session, "network", None) or getattr(session, "_network", None)
        if network:
            from captain_claw.tools.browser_network import _infer_auth_type

            captures = network.get_captures()
            for cap in reversed(captures):
                if urlparse(cap.url).netloc != domain:
                    continue
                auth_type, auth_value = _infer_auth_type(cap.request_headers)
                if auth_type != "none" and auth_value:
                    await sm.update_direct_api_call(
                        entry.id,
                        auth_type=auth_type, auth_token=auth_value,
                        auth_source="browser",
                    )
                    return ToolResult(
                        success=True,
                        content=(
                            f"Captured {auth_type} auth from browser for "
                            f"{domain} → stored on #{entry.id[:8]}"
                        ),
                    )

        # Fallback: extract cookies
        try:
            cookies = await session.save_cookies()
            domain_cookies = [
                c for c in cookies
                if domain in (c.get("domain", "") or "")
            ]
            if domain_cookies:
                cookie_str = "; ".join(
                    f"{c['name']}={c['value']}" for c in domain_cookies
                )
                await sm.update_direct_api_call(
                    entry.id,
                    auth_type="cookie", auth_token=cookie_str,
                    auth_source="browser",
                )
                return ToolResult(
                    success=True,
                    content=(
                        f"Captured cookie auth ({len(domain_cookies)} cookies) "
                        f"from browser for {domain} → stored on #{entry.id[:8]}"
                    ),
                )
        except Exception as e:
            log.warning("Cookie extraction failed", error=str(e))

        return ToolResult(
            success=False,
            error=(
                f"No auth tokens or cookies found for {domain} in the "
                "browser session. Log in first, then retry."
            ),
        )
