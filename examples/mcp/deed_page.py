"""Smoke-test deed.page publishing through Flight Deck's MCP manager, without an LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from captain_claw.flight_deck import mcp_storage
from captain_claw.flight_deck.mcp_manager import MCPManager

SAMPLE_PAGE = (
    "<!doctype html><meta charset='utf-8'><title>Captain Claw report</title>"
    "<h1>Captain Claw report</h1><p>Published by the Flight Deck deed.page example.</p>"
)


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


async def main(html_file: Path | None, keep: bool) -> None:
    record = json.loads(Path(__file__).with_name("deed-page.json").read_text())
    html = html_file.read_text(encoding="utf-8") if html_file else SAMPLE_PAGE
    previous_path = os.environ.get("CAPTAIN_CLAW_FD_MCP_PATH")
    with tempfile.TemporaryDirectory(prefix="claw-deed-page-") as directory:
        os.environ["CAPTAIN_CLAW_FD_MCP_PATH"] = str(Path(directory) / "servers.json")
        manager = MCPManager()
        try:
            await mcp_storage.upsert_server(record)
            tools = await manager.list_tools(record["name"])
            names = {tool["name"] for tool in tools}
            if not {"publish_site", "delete_site"} <= names:
                raise RuntimeError(f"Missing publish/delete tools: {names}")
            arguments = {"files": {"index.html": html}}
            if not keep:
                arguments["ttl_seconds"] = 600
            site = result_data(await manager.call_tool(record["name"], "publish_site", arguments))
            request = urllib.request.Request(site["url"], headers=record["headers"])
            with urllib.request.urlopen(request, timeout=30) as response:
                print(f"Published {site['url']} (HTTP {response.status})")
            if keep:
                print("Kept the page. Save this owner token to update or delete it:")
                print(site.get("token", "(existing token unchanged)"))
                return
            deleted = result_data(await manager.call_tool(record["name"], "delete_site", {
                "slug": site["slug"],
                "token": site["token"],
            }))
            print(json.dumps(deleted, indent=2))
        finally:
            await manager.close()
            if previous_path is None:
                os.environ.pop("CAPTAIN_CLAW_FD_MCP_PATH", None)
            else:
                os.environ["CAPTAIN_CLAW_FD_MCP_PATH"] = previous_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="HTML file to publish (default: a sample page)")
    parser.add_argument("--keep", action="store_true", help="Keep the page instead of deleting it")
    args = parser.parse_args()
    asyncio.run(main(args.file, args.keep))
