"""Anthropic's own tool search, carried through LiteLLM.

Captain's generic tool search (``captain_claw/tool_search.py``) leaves some
tool definitions out and adds the ones ``search_tools`` loads to the next
request. That changes the tool list, which sits at the front of every request,
so each load rewrites the whole prompt cache.

Claude models that support it get every definition instead, the rarely used
ones marked ``defer_loading``, plus Anthropic's search tool. The model searches
inside the same response, the API adds what it found to the conversation, and
the tool list never changes. See
https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool.

The search comes back as two blocks, ``server_tool_use`` and
``tool_search_tool_result``, which must go back to the API unchanged with the
rest of the assistant message. LiteLLM 1.83 turns the first into an ordinary
tool call (id ``srvtoolu_…``) and drops the second, so:

- ``install`` wraps LiteLLM's two Anthropic response parsers (plain and
  streamed) to keep every ``tool_search_tool_result`` block it sees, by the id
  of the search it answers;
- ``split_tool_calls`` takes the search calls out of a response's tool calls
  (Captain must not run them) and pairs each with its kept result;
- the pairs are stored on the assistant message (``server_blocks``) and sent
  back in its content, which LiteLLM passes through as they are.

A search whose result wasn't kept is dropped with its call: the history stays
valid, the model only has to search again.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

SEARCH_TOOL_TYPE = "tool_search_tool_bm25_20251119"
SEARCH_TOOL_NAME = "tool_search_tool_bm25"
RESULT_TYPE = "tool_search_tool_result"
SERVER_CALL_TYPE = "server_tool_use"
SERVER_ID_PREFIX = "srvtoolu_"


def search_tool() -> dict[str, Any]:
    """The server tool definition, as sent in ``tools``."""
    return {"type": SEARCH_TOOL_TYPE, "name": SEARCH_TOOL_NAME}


def is_search_tool(defn: Any) -> bool:
    return isinstance(defn, dict) and str(defn.get("type", "")).startswith("tool_search_tool_")


# ── Which models ────────────────────────────────────────────────────────

# claude-<family>-<major>[-<minor>]…, e.g. claude-opus-5-5, claude-haiku-4-5-20251001.
# A dated id with no minor (claude-sonnet-4-20250514) has minor 0.
_MODEL_RE = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?=$|[-@.\[])")


def model_supported(model: str) -> bool:
    """Claude models with tool search: Fable, Mythos, and every Opus, Sonnet
    and Haiku from 4.5 on (Opus 4.1 and earlier don't have it)."""
    name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    if name.startswith("claude-mythos-preview"):
        return True
    match = _MODEL_RE.match(name)
    if not match:
        return False
    family, major, minor = match.group(1), int(match.group(2)), int(match.group(3) or 0)
    if major >= 5:
        return True
    return major == 4 and minor >= 5 and family in ("opus", "sonnet", "haiku")


# ── Keeping the search results LiteLLM drops ────────────────────────────

_MAX_KEPT = 256
_kept: OrderedDict[str, dict[str, Any]] = OrderedDict()
_lock = threading.Lock()
_installed: bool | None = None


def _keep(block: Any) -> None:
    if not isinstance(block, dict) or block.get("type") != RESULT_TYPE:
        return
    tool_use_id = str(block.get("tool_use_id") or "")
    if not tool_use_id:
        return
    with _lock:
        _kept[tool_use_id] = dict(block)
        _kept.move_to_end(tool_use_id)
        while len(_kept) > _MAX_KEPT:
            _kept.popitem(last=False)


def _take(tool_use_id: str) -> dict[str, Any] | None:
    with _lock:
        return _kept.pop(tool_use_id, None)


def install() -> bool:
    """Wrap LiteLLM's Anthropic response parsers once. False when this
    LiteLLM doesn't have them (then the generic tool search is used)."""
    global _installed
    if _installed is not None:
        return _installed
    try:
        from litellm.llms.anthropic.chat.handler import ModelResponseIterator
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        extract = AnthropicConfig.extract_response_content
        parse = ModelResponseIterator.chunk_parser

        if not getattr(extract, "_cc_keeps_tool_search", False):
            def extract_response_content(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                response = kwargs.get("completion_response", args[0] if args else None)
                if isinstance(response, dict):
                    for block in response.get("content") or []:
                        _keep(block)
                return extract(self, *args, **kwargs)

            extract_response_content._cc_keeps_tool_search = True  # type: ignore[attr-defined]
            AnthropicConfig.extract_response_content = extract_response_content  # type: ignore[method-assign]

        if not getattr(parse, "_cc_keeps_tool_search", False):
            def chunk_parser(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                chunk = kwargs.get("chunk", args[0] if args else None)
                if isinstance(chunk, dict) and chunk.get("type") == "content_block_start":
                    _keep(chunk.get("content_block"))
                return parse(self, *args, **kwargs)

            chunk_parser._cc_keeps_tool_search = True  # type: ignore[attr-defined]
            ModelResponseIterator.chunk_parser = chunk_parser  # type: ignore[method-assign]
        _installed = True
    except Exception as exc:  # pragma: no cover - depends on the LiteLLM version
        log.warning("Anthropic tool search unavailable: LiteLLM parsers not found", error=str(exc))
        _installed = False
    return _installed


# ── Responses and history ───────────────────────────────────────────────


def split_tool_calls(tool_calls: Iterable[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """``(client_calls, server_blocks)``: the calls Captain runs, and each
    Anthropic search as its ``server_tool_use`` block followed by its result,
    in the order they came."""
    client: list[Any] = []
    blocks: list[dict[str, Any]] = []
    for call in tool_calls:
        call_id = str(getattr(call, "id", "") or "")
        if not call_id.startswith(SERVER_ID_PREFIX):
            client.append(call)
            continue
        result = _take(call_id)
        if result is None:
            log.warning("Anthropic tool search: result not kept, dropping the search",
                        call_id=call_id, name=getattr(call, "name", ""))
            continue
        arguments = getattr(call, "arguments", None)
        blocks.append({
            "type": SERVER_CALL_TYPE,
            "id": call_id,
            "name": str(getattr(call, "name", "") or SEARCH_TOOL_NAME),
            "input": arguments if isinstance(arguments, dict) else {},
        })
        blocks.append(result)
    return client, blocks


def valid_blocks(blocks: Any) -> list[dict[str, Any]]:
    """The complete search pairs in *blocks*, as stored on a message."""
    if not isinstance(blocks, list):
        return []
    out: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == SERVER_CALL_TYPE:
            pending = block
        elif block.get("type") == RESULT_TYPE and pending is not None \
                and block.get("tool_use_id") == pending.get("id"):
            out.extend((pending, block))
            pending = None
    return out


def found_tools(blocks: Any) -> list[str]:
    """The tool names the searches in *blocks* returned."""
    names: list[str] = []
    for block in valid_blocks(blocks):
        if block.get("type") != RESULT_TYPE:
            continue
        content = block.get("content")
        refs = content.get("tool_references") if isinstance(content, dict) else None
        for ref in refs or []:
            name = str(ref.get("tool_name") or "") if isinstance(ref, dict) else ""
            if name and name not in names:
                names.append(name)
    return names
