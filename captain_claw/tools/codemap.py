"""codemap tool — query the repo's Code Map "blackboard".

The Code Map (see ``captain_claw.flight_deck.code_map``) is a per-repo index of
symbols (functions/methods/classes with signatures + file:line), file purposes,
data models, UI map and an architecture overview. Agents call this tool to LOCATE
what they need — a symbol, a model, which file does X — WITHOUT reading and
re-interpreting the whole codebase. It returns concise pointers, never raw code:
read the actual file (at the file:line it gives you) when you need the source.

Read actions: overview | search | symbol | file | models | ui | stats.
Write actions (the cartographer keeps the semantic layer fresh):
set_overview | set_models | set_ui.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class CodeMapTool(Tool):
    name = "codemap"
    description = (
        "Query the repository's Code Map — a fast index of symbols "
        "(functions/methods/classes with signatures + file:line), file purposes, "
        "data models, UI map, and an architecture overview. Use it to LOCATE "
        "things instead of reading the whole codebase: `overview` for the big "
        "picture and entry points; `search <query>` to find symbols/files; "
        "`symbol <name>` for a specific function/class; `file <path>` for a "
        "file's purpose + its symbols; `models`/`ui` for the data and UI maps. It "
        "returns pointers (file:line), not source — read the file at that line "
        "when you need the code. Entries flagged `stale` changed since indexing — "
        "trust the actual file. Cartographer only: `set_overview`/`set_models`/"
        "`set_ui` write the semantic layer."
    )
    timeout_seconds = 20.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["overview", "search", "symbol", "file", "models", "ui", "stats",
                         "set_overview", "set_models", "set_ui"],
                "description": "What to do.",
            },
            "query": {"type": "string", "description": "Search text (action=search)."},
            "name": {"type": "string", "description": "Symbol name (action=symbol)."},
            "path": {"type": "string", "description": "File path relative to repo (action=file)."},
            "content": {"type": "string", "description": "Markdown overview text (action=set_overview)."},
            "data": {"description": "JSON value for the models/ui map (action=set_models/set_ui)."},
            "limit": {"type": "number", "description": "Max results (search/symbol)."},
        },
        "required": ["action"],
    }

    async def execute(self, action: str, query: str = "", name: str = "", path: str = "",
                      content: str = "", data: Any = None, limit: int = 25,
                      **kwargs: Any) -> ToolResult:
        from captain_claw.flight_deck import code_map as cm
        base = kwargs.get("_runtime_base_path")
        if not base:
            return ToolResult(success=False, error="codemap: no workspace root")
        repo = Path(base).expanduser().resolve()
        try:
            if action == "overview":
                ov = cm.read_overview(repo)
                st = cm.stats(repo)
                head = (f"Code Map · {st['files']} files · {st['symbols']} symbols · "
                        f"{st['summarized']} summarized\n\n")
                return ToolResult(success=True, content=head + (ov or
                        "(no architecture overview yet — run the cartographer, or use "
                        "`search`/`file` on the skeleton.)"))

            if action == "search":
                if not query.strip():
                    return ToolResult(success=False, error="search needs a query")
                hits = cm.search(repo, query, int(limit or 25))
                if not hits:
                    return ToolResult(success=True, content=f"No matches for '{query}'.")
                lines = [f"{h['kind']:9} {h['signature']}  —  {h['file']}:{h['line']}"
                         + (f"  · {h['summary']}" if h['summary'] else "") for h in hits]
                return ToolResult(success=True, content=f"{len(hits)} match(es):\n" + "\n".join(lines))

            if action == "symbol":
                if not name.strip():
                    return ToolResult(success=False, error="symbol needs a name")
                rows = cm.symbol(repo, name)
                if not rows:
                    return ToolResult(success=True, content=f"No symbol named '{name}'. Try `search`.")
                out = []
                for r in rows:
                    tag = " [STALE — read the file]" if r.get("stale") else ""
                    scope = f" (in {r['scope']})" if r.get("scope") else ""
                    out.append(f"{r['kind']} {r['signature']}{scope}\n  {r['file']}:{r['line']}{tag}"
                               + (f"\n  {r['summary']}" if r['summary'] else "")
                               + (f"\n  file: {r['file_purpose']}" if r['file_purpose'] else ""))
                return ToolResult(success=True, content="\n\n".join(out))

            if action == "file":
                if not path.strip():
                    return ToolResult(success=False, error="file needs a path")
                fm = cm.file_map(repo, path)
                if not fm["symbols"] and not fm["purpose"]:
                    return ToolResult(success=True, content=f"'{path}' not in the map (not indexed or no symbols).")
                head = f"{fm['path']} [{fm['lang']}]" + (" [STALE]" if fm["stale"] else "")
                if fm["purpose"]:
                    head += f"\nPurpose: {fm['purpose']}"
                syms = [f"  {s['kind']:9} {s['signature']}  :{s['line']}"
                        + (f"  · {s['summary']}" if s['summary'] else "") for s in fm["symbols"]]
                return ToolResult(success=True, content=head + ("\n" + "\n".join(syms) if syms else ""))

            if action in ("models", "ui"):
                layer = cm.read_json_layer(repo, action)
                if layer is None:
                    return ToolResult(success=True, content=f"(no {action} map yet — run the cartographer.)")
                return ToolResult(success=True, content=json.dumps(layer, indent=2))

            if action == "stats":
                return ToolResult(success=True, content=json.dumps(cm.stats(repo), indent=2))

            # ── cartographer writes ──
            if action == "set_overview":
                cm.write_overview(repo, content or "")
                return ToolResult(success=True, content="overview saved.")
            if action in ("set_models", "set_ui"):
                payload = data
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        return ToolResult(success=False, error="data must be valid JSON")
                cm.write_json_layer(repo, "models" if action == "set_models" else "ui", payload)
                return ToolResult(success=True, content=f"{action.split('_')[1]} map saved.")

            return ToolResult(success=False, error=f"unknown action: {action}")
        except Exception as e:  # noqa: BLE001
            log.warning("codemap tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"codemap error: {e}")
