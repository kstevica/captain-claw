"""Natural-language → app manifest generation.

A small LLM-driven authoring layer that takes a user's description of
what they want — optionally with the current manifest and the list of
MCP tools available on the agent's MCP server — and produces a
validated :class:`AgentManifest`.

The framework owns this. Renderers call ``POST /fd/apps/generate`` and
get back a manifest they can preview and save.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel

from captain_claw.flight_deck import app_manifests
from captain_claw.flight_deck.app_manifests import AgentManifest
from captain_claw.flight_deck.mcp_manager import MCPServerError, get_manager
from captain_claw.games.remote_provider import RemoteLLMProvider
from captain_claw.llm import Message
from captain_claw.logging import get_logger

log = get_logger(__name__)


class AgentTarget(BaseModel):
    """Minimal target spec — matches the one in games_routes."""
    host: str = "localhost"
    port: int
    auth: str = ""
    name: str = ""


# ── prompt ────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You design **agent-app manifests** for the Captain Claw framework.

An app manifest is a declarative spec for a small SaaS-style app that
binds to an agent's MCP tools. Renderers (web, voice, glasses) consume
the manifest and present the agent's capabilities to a user.

You MUST return a single JSON object matching this schema:

```
{
  "manifest_version": 1,
  "agent": {
    "id": "<filesystem-safe slug>",
    "name": "<human name>",
    "tagline": "<short subtitle>",
    "mcp_server": "<name of the MCP server backing the tools>"
  },
  "entities": {
    "<entity_id>": {
      "id": "<entity_id>",
      "label": "<singular label>",
      "plural": "<plural label>",
      "fields": {
        "<field_name>": {
          "type": "string" | "text" | "number" | "boolean" | "date" | "datetime" | "enum" | "markdown" | "file" | { "ref": "<entity_id>" },
          "label": "<optional human label>",
          "values": ["<for enum>"],
          "primary": true,        // exactly one per entity — the id field
          "title": true,          // exactly one per entity — the human-readable title
          "required": false
        }
      },
      "default_view": "card" | "row" | "summary"
    }
  },
  "feeds": {
    "<feed_id>": {
      "id": "<feed_id>",
      "label": "<human label>",
      "mcp_tool": "<exact tool name on the MCP server>",
      "arguments": { /* static args passed to the tool */ },
      "returns": "<entity_id this feed returns rows of>",
      "surfaces": ["<surface_id>", ...],
      "refresh_seconds": 60,
      "proactive": false
    }
  },
  "actions": {
    "<action_id>": {
      "id": "<action_id>",
      "label": "<button label>",
      "mcp_tool": "<exact tool name>",
      "inputs": {
        "<arg_name>": {
          "type": "string" | "text" | ...,
          "label": "<form label>",
          "required": true
        }
      },
      "nl_aliases": ["short phrases the user might say"],
      "surfaces": ["<surface_id>"],
      "prefill": { "<arg_name>": "$entity.id" },
      "returns": "markdown" | "entity" | "none",
      "prominent": false
    }
  },
  "surfaces": {
    "<surface_id>": {
      "id": "<surface_id>",
      "label": "<tab label>",
      "layout": "dashboard" | "list" | "entity" | "inbox" | "upload",
      "entity": "<entity_id, only for layout=entity>",
      "sources": ["<feed_id>", ...],   // feed ids for inbox/list; ACTION ids for upload
      "sections": [                      // only for layout=dashboard/entity
        { "type": "feed",   "id": "<feed_id>", "filter": {"<arg>": "$entity.id"} },
        { "type": "action", "id": "<action_id>", "prefill": {"<arg>": "$entity.id"}, "prominent": true }
      ],
      "accept": "image/*",               // only for layout=upload — file picker accept filter
      "multiple": false                  // only for layout=upload
    }
  },
  "chat": { "enabled": true, "context_aware": true, "default_actions": ["<action_id>"] },
  "home_surface": "<surface_id to land on>"
}
```

Rules:
- Every `mcp_tool` MUST be an exact tool name from the list provided in the user message. Never invent tools.
- Every `feeds[*].returns` MUST be an entity id you also declare.
- Every `sections[*].id` MUST reference a feed or action you also declare.
- Every `surfaces[*].entity` MUST be a declared entity id.
- `home_surface` MUST be a declared surface id.
- Use `$entity.id` (or `$entity.<field>`) inside `prefill`/`filter` to thread the currently-selected entity into a tool call.
- Prefer feed/action ids that look like snake_case identifiers; use hyphens only in `agent.id`.
- Keep the design tight. A few well-chosen feeds beat a sprawling dashboard.
- For apps that receive uploads (images, PDFs, etc.): declare an action input with `type: "file"`, and add a surface with `layout: "upload"` whose `sources` list the action ids to run on uploaded files. The renderer uploads via the framework's file API and passes back a `file_id` string — your MCP tool should accept that `file_id` and resolve the bytes through Captain Claw's file store.
- When the user has NOT selected an external MCP server, set `agent.mcp_server` to the reserved name `"__framework__"` and bind every `mcp_tool` to one of the framework's built-in tools listed below. Do NOT invent tool names; the only callable tools in this mode are `entities_list`, `entities_get`, `entities_create`, `entities_update`, `entities_delete`, and `files_list`. Pass the entity id you declared via the `entity` argument (e.g. `"arguments": {"entity": "post"}` for a list feed). For create/update actions, declare action inputs for each user-editable field of the entity; the renderer will forward them as flat arguments alongside `entity` (and `id` for update).

Return ONLY the JSON object. No prose, no markdown fence.
"""


# ── tool catalogue ────────────────────────────────────────────────────


async def _list_tools(mcp_server: str) -> list[dict[str, Any]]:
    """Return ``[{name, description, input_schema}]`` for an MCP server,
    or ``[]`` if the server is unknown / unreachable."""
    try:
        tools = await get_manager().list_tools(mcp_server)
    except MCPServerError as exc:
        log.warning("manifest authoring: failed to list tools for %s: %s", mcp_server, exc)
        return []
    out: list[dict[str, Any]] = []
    for t in tools or []:
        # Tools come back as dict-shaped from the MCP manager.
        name = t.get("name") if isinstance(t, dict) else None
        if not name:
            continue
        out.append({
            "name": name,
            "description": (t.get("description") or "").strip() if isinstance(t, dict) else "",
            "input_schema": t.get("inputSchema") if isinstance(t, dict) else None,
        })
    return out


_BUILTIN_TOOL_CATALOGUE = """No external MCP server was selected — use the framework's built-in tool catalogue. Set `agent.mcp_server` to `"__framework__"` and bind each `mcp_tool` to one of:

- `entities_list`     args: { "entity": "<entity_id>" } — returns all records of that entity for this app.
- `entities_get`      args: { "entity": "<entity_id>", "id": "<record_id>" }
- `entities_create`   args: { "entity": "<entity_id>", ...record fields }  — `id`/`created_at`/`updated_at` are auto-filled.
- `entities_update`   args: { "entity": "<entity_id>", "id": "<record_id>", ...changed fields }
- `entities_delete`   args: { "entity": "<entity_id>", "id": "<record_id>" }
- `files_list`        args: {} — lists files uploaded to this app.

For feeds that list an entity: `mcp_tool: "entities_list"`, `arguments: {"entity": "<entity_id>"}`, `returns: "<entity_id>"`.
For "create X" actions: `mcp_tool: "entities_create"`, `inputs` should be one per editable field, plus a hidden/static `arguments: {"entity": "<entity_id>"}`.
For "edit X" actions: `mcp_tool: "entities_update"`, prefill `{"id": "$entity.id"}`, plus `arguments: {"entity": "<entity_id>"}`.
For "delete X" actions: `mcp_tool: "entities_delete"`, prefill `{"id": "$entity.id"}`, plus `arguments: {"entity": "<entity_id>"}`."""


def _tools_block(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return _BUILTIN_TOOL_CATALOGUE
    lines = [f"Available tools on the chosen MCP server:"]
    for t in tools:
        desc = t.get("description") or ""
        if len(desc) > 200:
            desc = desc[:197] + "…"
        lines.append(f"- {t['name']}: {desc}")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────


def _default_model() -> str:
    # Allow override via env, default to a fast Anthropic Claude model.
    return os.environ.get("FD_MANIFEST_MODEL", "anthropic/claude-haiku-4-5-20251001")


async def _call_llm(
    messages: list[dict[str, str]],
    *,
    agent: AgentTarget | None = None,
) -> str:
    """Single LLM call, returning the raw assistant content string.

    If ``agent`` is given, route through that agent's ``/api/llm/complete``
    endpoint via :class:`RemoteLLMProvider`. Otherwise fall back to a
    direct litellm call (requires API keys in the environment).
    """
    if agent is not None:
        provider = RemoteLLMProvider(host=agent.host, port=agent.port, auth=agent.auth, name=agent.name)
        cc_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        resp = await provider.complete(cc_messages, temperature=0.2)
        return resp.content or ""

    # Fallback: direct litellm. Only works if vendor keys are in env.
    import litellm

    model = _default_model()
    def _do_call() -> Any:
        return litellm.completion(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )

    resp = await asyncio.to_thread(_do_call)
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):
        return ""


# ── public API ────────────────────────────────────────────────────────


async def generate(
    description: str,
    *,
    mcp_server: str | None = None,
    current_manifest: dict[str, Any] | None = None,
    agent: AgentTarget | None = None,
) -> dict[str, Any]:
    """Generate (or revise) a manifest from a natural-language description.

    Returns ``{"manifest": <dict>, "errors": [str, ...]}``. When the LLM
    output doesn't validate, ``manifest`` is still the best-effort raw
    JSON (so the caller can show it for debugging), and ``errors`` lists
    validation problems.
    """
    description = (description or "").strip()
    if not description:
        return {"manifest": None, "errors": ["description is required"]}

    tools: list[dict[str, Any]] = []
    if mcp_server:
        tools = await _list_tools(mcp_server)

    user_lines = [
        f"User request:\n{description}",
        "",
        _tools_block(tools),
    ]
    if mcp_server:
        user_lines.append("")
        user_lines.append(f"Use `\"mcp_server\": \"{mcp_server}\"` in the agent block.")
    if current_manifest is not None:
        user_lines.append("")
        user_lines.append("Current manifest to revise (apply the user's changes; preserve the rest):")
        user_lines.append("```json")
        user_lines.append(json.dumps(current_manifest, indent=2))
        user_lines.append("```")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]

    try:
        raw = await _call_llm(messages, agent=agent)
    except Exception as exc:
        log.exception("manifest generation LLM call failed")
        return {"manifest": None, "errors": [f"LLM call failed: {exc}"]}
    if not raw:
        return {"manifest": None, "errors": ["LLM returned no content"]}

    # Strip any accidental fencing the model may add.
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        # Fallback: agents not pinned to JSON-mode may wrap the object in prose.
        # Extract the outermost {...} and retry.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return {"manifest": None, "errors": ["LLM output is not valid JSON"]}
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            return {"manifest": None, "errors": [f"LLM output is not valid JSON: {exc}"]}

    if not isinstance(data, dict):
        return {"manifest": None, "errors": ["LLM output is not a JSON object"]}

    # Validate against pydantic. On error, return the raw dict + errors so the
    # caller can render them and ask the user to retry.
    try:
        AgentManifest.model_validate(data)
        return {"manifest": data, "errors": []}
    except Exception as exc:
        return {"manifest": data, "errors": [str(exc)]}


def save_validated(manifest_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate a manifest dict and persist it. Returns
    ``{"ok": bool, "path": str | None, "errors": list[str]}``."""
    try:
        manifest = AgentManifest.model_validate(manifest_dict)
    except Exception as exc:
        return {"ok": False, "path": None, "errors": [str(exc)]}
    try:
        path = app_manifests.save(manifest)
    except (OSError, ValueError) as exc:
        return {"ok": False, "path": None, "errors": [str(exc)]}
    return {"ok": True, "path": str(path), "errors": []}
