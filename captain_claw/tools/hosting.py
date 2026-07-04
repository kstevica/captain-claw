"""`hosting` tool — publish and manage VFS-hosted sites/apps from chat.

The chat-agent sibling of the Hosting page: an agent that just built something
in a VFS folder can put it on the web. Two kinds:

* **static** — files served directly at ``/vfs/<name>/`` (index.html or an
  autoindex when there's none).
* **app** — a process (e.g. ``node server.js``) that Flight Deck starts on an
  internal port and reverse-proxies at ``/vfs-apps/<name>/`` (HTTP + WebSocket).

Unlike the `code`/`basna` tools these calls are synchronous — publishing an app
starts it and returns the live URL in one shot, so the agent can relay the link
immediately (including to WhatsApp / Telegram). Everything is owner-scoped via
the request identity, exactly like the Code and Basna agent tools.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class HostingTool(Tool):
    name = "hosting"
    description = (
        "Publish and manage web hosting for the user's VFS folders — put a static site or a "
        "runnable app on the web and get back a public URL. MANDATORY: when the user asks to "
        "'host', 'publish', 'deploy', 'put online', or 'serve' a site/app you built in a VFS "
        "folder, call this with action='publish' — don't just describe how. "
        "kind='static' serves files at /vfs/<name>/ (needs an index.html, else a file listing); "
        "kind='app' runs a process (give `start_cmd`, e.g. 'node server.js') and reverse-proxies "
        "it at /vfs-apps/<name>/ — apps are started automatically on publish. `name` is the URL "
        "slug (lowercase letters, digits, dashes). `project` is the VFS project/folder the files "
        "live in (as used by Code); `subdir` narrows to a sub-path inside it. Re-publishing a "
        "name you own updates it (and restarts the app). Other actions: 'list' your published "
        "sites, 'start'/'stop' an app, 'status' (URL, running state, recent logs, visit count), "
        "'unpublish' to remove. Returns the public URL — relay it to the user."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["publish", "list", "start", "stop", "status", "unpublish"],
                "description": (
                    "publish — put a VFS folder online (create or, if you own the name, update); "
                    "list — your published sites/apps; start/stop — an app's process; "
                    "status — URL + running state + recent logs + visits; unpublish — remove."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "URL slug (lowercase letters, digits, dashes). The site is reachable at "
                    "/vfs/<name>/ (static) or /vfs-apps/<name>/ (app). Required for every action "
                    "except 'list'."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["static", "app"],
                "description": "publish: 'static' (serve files) or 'app' (run a process). Default 'static'.",
            },
            "project": {
                "type": "string",
                "description": (
                    "publish: the VFS project/folder holding the files (same name you'd use in "
                    "Code). Required for publish."
                ),
            },
            "subdir": {
                "type": "string",
                "description": "publish: optional sub-path inside the project (e.g. 'dist' or 'build').",
            },
            "start_cmd": {
                "type": "string",
                "description": (
                    "publish (app): the command that starts the server, e.g. 'node server.js' or "
                    "'python app.py'. The app must bind the PORT env var Flight Deck injects. "
                    "Required when kind='app'."
                ),
            },
            "auto_start": {
                "type": "boolean",
                "description": "publish (app): start the process immediately. Default true.",
            },
        },
        "required": ["action"],
    }

    # ── identity / transport (same pattern as the code + basna tools) ─────────

    def _get_fd_url(self, **kwargs: Any) -> str:
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        if not fd_url:
            fd_url = os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")
        return fd_url

    def _identity(self) -> dict:
        from captain_claw.config import get_config
        try:
            cfg = get_config()
            port, auth = int(cfg.web.port or 0), cfg.web.auth_token or ""
        except Exception:  # noqa: BLE001
            port, auth = 0, ""
        return {"web_auth": auth, "source_port": port,
                "owner_id": os.environ.get("FD_OWNER_ID", "")}

    async def _post(self, fd_url: str, path: str, payload: dict) -> Any:
        import httpx
        body = {**self._identity(), **payload}
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{fd_url}{path}", json=body)
        if resp.status_code in (400, 403, 404, 409):
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:  # noqa: BLE001
                detail = resp.text
            return {"_error": detail or f"request failed ({resp.status_code})"}
        resp.raise_for_status()
        return resp.json()

    # ── actions ───────────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Hosting.")
        try:
            if action == "publish":
                return await self._publish(fd_url, **kwargs)
            if action == "list":
                return await self._list(fd_url)
            if action in ("start", "stop", "status", "unpublish"):
                return await self._one(fd_url, action, **kwargs)
            return ToolResult(success=False, error=f"Unknown action '{action}'.")
        except Exception as e:  # noqa: BLE001
            log.warning("hosting tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Hosting request failed: {e}")

    @staticmethod
    def _pub_url(view: dict) -> str:
        return view.get("url_abs") or view.get("url") or ""

    async def _publish(self, fd_url: str, **kwargs: Any) -> ToolResult:
        name = (kwargs.get("name") or "").strip().lower()
        project = (kwargs.get("project") or "").strip()
        kind = (kwargs.get("kind") or "static").strip()
        if not name:
            return ToolResult(success=False, error="Provide `name` (the URL slug).")
        if not project:
            return ToolResult(success=False, error="Provide `project` (the VFS folder to serve).")
        payload = {
            "name": name, "kind": kind, "project": project,
            "subdir": (kwargs.get("subdir") or "").strip(),
            "start_cmd": (kwargs.get("start_cmd") or "").strip(),
            "auto_start": bool(kwargs.get("auto_start", True)),
        }
        d = await self._post(fd_url, "/fd/hosting/agent/publish", payload)
        if isinstance(d, dict) and d.get("_error"):
            return ToolResult(success=False, error=d["_error"])
        url = self._pub_url(d)
        verb = "Updated" if d.get("updated") else "Published"
        if kind == "app":
            if d.get("started"):
                return ToolResult(success=True, content=(
                    f"{verb} app **{name}** — running and live at {url}"))
            # Published but the process didn't come up — surface the log so the
            # agent can fix and re-publish (don't claim success it can't see).
            logtail = (d.get("log") or "").strip()
            return ToolResult(success=True, content=(
                f"{verb} app **{name}** at {url}, but it did not start: {d.get('message', '')}.\n"
                + (f"Recent output:\n{logtail}" if logtail else "")
                + "\nFix the start command / code and re-publish, or check status later."))
        return ToolResult(success=True, content=f"{verb} static site **{name}** — live at {url}")

    async def _list(self, fd_url: str) -> ToolResult:
        d = await self._post(fd_url, "/fd/hosting/agent/list", {})
        if isinstance(d, dict) and d.get("_error"):
            return ToolResult(success=False, error=d["_error"])
        entries = d.get("entries", [])
        if not entries:
            return ToolResult(success=True, content="Nothing published yet.")
        lines = []
        for e in entries:
            state = ""
            if e.get("kind") == "app":
                state = " · running" if e.get("running") else " · stopped"
            lines.append(f"• {e['name']} ({e.get('kind')}){state} — {self._pub_url(e)}")
        return ToolResult(success=True, content="\n".join(lines))

    async def _one(self, fd_url: str, action: str, **kwargs: Any) -> ToolResult:
        name = (kwargs.get("name") or "").strip().lower()
        if not name:
            return ToolResult(success=False, error="Provide `name`.")
        d = await self._post(fd_url, f"/fd/hosting/agent/{action}", {"name": name})
        if isinstance(d, dict) and d.get("_error"):
            return ToolResult(success=False, error=d["_error"])
        if action == "unpublish":
            return ToolResult(success=True, content=f"Unpublished **{name}**.")
        if action == "stop":
            return ToolResult(success=True, content=f"Stopped **{name}** — {d.get('message', 'ok')}.")
        if action == "start":
            if d.get("started"):
                return ToolResult(success=True, content=f"Started **{name}** — live at {self._pub_url(d)}")
            logtail = (d.get("log") or "").strip()
            return ToolResult(success=False, error=(
                f"Could not start **{name}**: {d.get('message', '')}."
                + (f"\nRecent output:\n{logtail}" if logtail else "")))
        # status
        parts = [f"**{name}** ({d.get('kind')}) — {self._pub_url(d)}"]
        if d.get("kind") == "app":
            parts.append("running" if d.get("running") else "stopped")
        v = d.get("visits") or {}
        if v.get("count"):
            parts.append(f"{v['count']} visit(s)")
        content = " · ".join(parts)
        logtail = (d.get("log") or "").strip()
        if d.get("kind") == "app" and logtail:
            content += f"\n\nRecent output:\n{logtail}"
        return ToolResult(success=True, content=content)
