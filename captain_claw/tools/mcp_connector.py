"""MCP Connector — discover and proxy tools from remote MCP servers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class MCPProxyTool(Tool):
    """A Captain Claw tool that proxies execution to a remote MCP server tool."""

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_description: str,
        mcp_input_schema: dict[str, Any],
        server_name: str,
        connector: "MCPConnector",
    ):
        # Sanitize: Anthropic requires tool names to match ^[a-zA-Z0-9_-]{1,128}$
        safe_name = mcp_tool_name.replace(".", "_").replace(" ", "_")
        safe_server = server_name.replace(".", "_").replace(" ", "_")
        self.name = f"mcp_{safe_server}_{safe_name}"
        self.description = f"[MCP:{server_name}] {mcp_description or mcp_tool_name}"
        self.parameters = mcp_input_schema or {
            "type": "object",
            "properties": {},
        }
        self.timeout_seconds = 60.0
        self._mcp_tool_name = mcp_tool_name
        self._server_name = server_name
        self._connector = connector

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            # Only pass arguments declared in the MCP tool schema;
            # Captain Claw's registry injects internal kwargs (paths,
            # events, callbacks) that must not be forwarded.
            declared = set(self.parameters.get("properties", {}).keys())
            clean_args = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in kwargs.items()
                if k in declared
            }
            result = await self._connector.call_tool(
                self._mcp_tool_name, clean_args
            )
            return ToolResult(success=True, content=result)
        except Exception as exc:
            log.error(
                "MCP tool execution failed",
                tool=self._mcp_tool_name,
                server=self._server_name,
                error=str(exc),
            )
            return ToolResult(success=False, error=str(exc))


class MCPConnector:
    """Connects to a remote MCP server via Streamable HTTP transport.

    Handles OAuth2 client_credentials authentication and provides
    methods to discover tools and call them.
    """

    def __init__(
        self,
        name: str,
        server_url: str,
        client_id: str = "",
        client_secret: str = "",
        token_endpoint: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.headers = dict(headers or {})
        self._access_token: str | None = None
        self._session_url: str | None = None

        # Resolve token endpoint (absolute or relative to server)
        if token_endpoint:
            if token_endpoint.startswith("http"):
                self.token_endpoint = token_endpoint
            else:
                # Relative path — resolve against server base URL
                from urllib.parse import urlparse

                parsed = urlparse(self.server_url)
                base = f"{parsed.scheme}://{parsed.netloc}"
                self.token_endpoint = (
                    base + "/" + token_endpoint.lstrip("/")
                )
        else:
            self.token_endpoint = ""

    async def _get_access_token(self) -> str:
        """Obtain an OAuth2 access token via client_credentials grant."""
        if self._access_token:
            return self._access_token

        if not self.token_endpoint or not self.client_id:
            return ""

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.token_endpoint,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            log.info(
                "MCP OAuth token acquired",
                server=self.name,
                token_type=data.get("token_type", "unknown"),
            )
            return self._access_token

    async def _build_headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        hdrs = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        hdrs.update(self.headers)

        token = await self._get_access_token()
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        return hdrs

    def _jsonrpc(self, method: str, params: dict | None = None, id: int = 1) -> dict:
        """Build a JSON-RPC 2.0 request."""
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": id,
        }
        if params is not None:
            msg["params"] = params
        return msg

    async def _post_rpc(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request to the MCP server and return the result."""
        headers = await self._build_headers()
        url = self._session_url or self.server_url
        payload = self._jsonrpc(method, params)

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)

            # Capture session URL from Mcp-Session header if present
            session_id = resp.headers.get("mcp-session-id")
            if session_id and not self._session_url:
                self._session_url = self.server_url

            # Handle SSE response (text/event-stream)
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                return self._parse_sse_response(resp.text)

            resp.raise_for_status()
            body = resp.json()

            if "error" in body:
                err = body["error"]
                raise RuntimeError(
                    f"MCP RPC error {err.get('code', '?')}: {err.get('message', str(err))}"
                )
            return body.get("result", body)

    def _parse_sse_response(self, text: str) -> dict:
        """Parse SSE text to extract the JSON-RPC result."""
        for line in text.splitlines():
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    msg = json.loads(data_str)
                    if "result" in msg:
                        return msg["result"]
                    if "error" in msg:
                        err = msg["error"]
                        raise RuntimeError(
                            f"MCP RPC error {err.get('code', '?')}: {err.get('message', str(err))}"
                        )
                except json.JSONDecodeError:
                    continue
        raise RuntimeError(f"No valid JSON-RPC result in SSE response")

    async def initialize(self) -> dict:
        """Send MCP initialize handshake."""
        result = await self._post_rpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "captain-claw",
                "version": "0.4.21",
            },
        })
        log.info(
            "MCP server initialized",
            server=self.name,
            server_info=result.get("serverInfo", {}),
        )
        # Send initialized notification (no id = notification)
        try:
            headers = await self._build_headers()
            url = self._session_url or self.server_url
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json=notification, headers=headers)
        except Exception:
            pass  # Notifications are best-effort
        return result

    async def discover_tools(self) -> list[dict[str, Any]]:
        """List all tools available on the MCP server."""
        result = await self._post_rpc("tools/list")
        tools = result.get("tools", [])
        log.info(
            "MCP tools discovered",
            server=self.name,
            count=len(tools),
            tools=[t.get("name") for t in tools],
        )
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server and return the text result."""
        result = await self._post_rpc("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        # Extract text from MCP content blocks
        content = result.get("content", [])
        is_error = result.get("isError", False)

        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
                else:
                    parts.append(json.dumps(block))
            elif isinstance(block, str):
                parts.append(block)

        text = "\n".join(parts) if parts else json.dumps(result)
        if is_error:
            raise RuntimeError(f"MCP tool error: {text}")
        return text

    def invalidate_token(self) -> None:
        """Force re-authentication on next request."""
        self._access_token = None
        self._session_url = None


async def register_mcp_tools(
    registry: Any,
    servers: list[dict[str, Any]],
) -> list[str]:
    """Connect to MCP servers and register their tools in the Captain Claw registry.

    Args:
        registry: ToolRegistry instance
        servers: List of server configs, each with:
            - name: server identifier
            - url: MCP server URL
            - client_id: OAuth client ID (optional)
            - client_secret: OAuth client secret (optional)
            - token_endpoint: OAuth token endpoint (optional)
            - headers: extra HTTP headers (optional)

    Returns:
        List of registered tool names.
    """
    registered: list[str] = []

    for srv_config in servers:
        name = srv_config.get("name", "default")
        url = srv_config.get("url", "")
        if not url:
            log.warning("MCP server config missing URL, skipping", name=name)
            continue

        connector = MCPConnector(
            name=name,
            server_url=url,
            client_id=srv_config.get("client_id", ""),
            client_secret=srv_config.get("client_secret", ""),
            token_endpoint=srv_config.get("token_endpoint", ""),
            headers=srv_config.get("headers"),
        )

        try:
            await connector.initialize()
            mcp_tools = await connector.discover_tools()

            for tool_def in mcp_tools:
                proxy = MCPProxyTool(
                    mcp_tool_name=tool_def["name"],
                    mcp_description=tool_def.get("description", ""),
                    mcp_input_schema=tool_def.get("inputSchema", {}),
                    server_name=name,
                    connector=connector,
                )
                registry.register(proxy)
                registered.append(proxy.name)
                log.info(
                    "Registered MCP tool",
                    tool=proxy.name,
                    mcp_tool=tool_def["name"],
                    server=name,
                )

        except Exception as exc:
            log.error(
                "Failed to connect to MCP server",
                server=name,
                url=url,
                error=str(exc),
            )

    return registered
