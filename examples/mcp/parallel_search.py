"""Smoke-test Parallel Search through Flight Deck's MCP manager, without an LLM."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

from captain_claw.flight_deck import mcp_storage
from captain_claw.flight_deck.mcp_manager import MCPManager


def result_data(result: dict) -> dict:
    if result.get("isError"):
        raise RuntimeError(f"MCP tool failed: {result}")
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for block in result.get("content", []):
        if block.get("type") == "text":
            data = json.loads(block["text"])
            if isinstance(data, dict):
                return data
    raise RuntimeError("MCP tool returned no JSON result")


async def main() -> None:
    record = json.loads(Path(__file__).with_name("parallel-search.json").read_text())
    previous_path = os.environ.get("CAPTAIN_CLAW_FD_MCP_PATH")
    with tempfile.TemporaryDirectory(prefix="claw-parallel-") as directory:
        os.environ["CAPTAIN_CLAW_FD_MCP_PATH"] = str(Path(directory) / "servers.json")
        manager = MCPManager()
        try:
            await mcp_storage.upsert_server(record)
            tools = await manager.list_tools(record["name"])
            names = {tool["name"] for tool in tools}
            if not {"web_search", "web_fetch"} <= names:
                raise RuntimeError(f"Missing search/fetch tools: {names}")
            session_id = str(uuid.uuid4())
            search = result_data(await manager.call_tool(record["name"], "web_search", {
                "objective": "Find the official Python asyncio documentation.",
                "search_queries": ["Python asyncio official documentation"],
                "session_id": session_id,
            }))
            if not search.get("results") or not any(
                item.get("excerpts") for item in search["results"]
            ):
                raise RuntimeError(f"Search returned no excerpts: {search}")
            url = search["results"][0]["url"]
            fetched = result_data(await manager.call_tool(record["name"], "web_fetch", {
                "urls": [url],
                "objective": "Explain what asyncio is used for.",
                "session_id": session_id,
            }))
            if not fetched.get("results") or not fetched["results"][0].get("excerpts"):
                raise RuntimeError(f"Fetch returned no excerpts: {fetched}")
            print(json.dumps({"search": search, "fetch": fetched}, indent=2))
        finally:
            await manager.close()
            if previous_path is None:
                os.environ.pop("CAPTAIN_CLAW_FD_MCP_PATH", None)
            else:
                os.environ["CAPTAIN_CLAW_FD_MCP_PATH"] = previous_path


if __name__ == "__main__":
    asyncio.run(main())
