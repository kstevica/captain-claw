"""researchmap tool — query the run's shared Research Map "blackboard".

The Research Map (see ``captain_claw.flight_deck.research_map``) indexes the ONE
shared VFS folder a Basna/Vatra run writes into: the prose artifacts of the whole
team and every prior continuation round. Workers call this tool to LOCATE a prior
finding/source/section instead of re-reading the whole folder; the reporter uses
it to pull material past its inline cap.

It resolves the shared folder from the worker's ``CLAW_VFS_PROJECT`` env (the same
folder ``vfs:`` writes land in), NOT the worker's private workspace.

Read actions: overview | search | files | file | stats.
Write action (the reporter/cartographer keeps the layer fresh): set_overview.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ResearchMapTool(Tool):
    name = "researchmap"
    description = (
        "Query the shared Research Map — a fast index of the team's research folder "
        "(all specialists' findings + every prior round). Use it to FIND prior "
        "material instead of re-reading everything: `overview` for what's been "
        "established, `search <query>` to locate claims/sources, `files` for the "
        "file list, `file <path>` for one file's summary. It returns snippets and "
        "pointers — open the file only for the full passage. `set_overview` (reporter/"
        "cartographer) writes the running synthesis."
    )
    timeout_seconds = 20.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["overview", "search", "files", "file", "stats", "set_overview"],
                "description": "What to do.",
            },
            "query": {"type": "string", "description": "Search text (action=search)."},
            "path": {"type": "string", "description": "File path within the folder (action=file)."},
            "content": {"type": "string", "description": "Markdown overview (action=set_overview)."},
            "limit": {"type": "number", "description": "Max results (action=search)."},
        },
        "required": ["action"],
    }

    def _project(self) -> Path | None:
        from captain_claw import vfs
        try:
            return vfs.project_root()
        except Exception as e:  # noqa: BLE001
            log.warning("researchmap: cannot resolve project root", error=str(e))
            return None

    async def execute(self, action: str, query: str = "", path: str = "",
                      content: str = "", limit: int = 15, **kwargs: Any) -> ToolResult:
        from captain_claw.flight_deck import research_map as rm
        project = self._project()
        if project is None:
            return ToolResult(success=False, error="researchmap: no shared VFS folder bound")
        try:
            if action == "overview":
                st = rm.stats(project)
                ov = rm.read_overview(project)
                head = f"Research Map · {st['files']} files · {st['chunks']} sections\n\n"
                return ToolResult(success=True, content=head + (ov or
                        "(no synthesis written yet — use `search`/`files` on the index.)"))

            if action == "search":
                if not query.strip():
                    return ToolResult(success=False, error="search needs a query")
                hits = rm.search(project, query, int(limit or 15))
                if not hits:
                    return ToolResult(success=True, content=f"No matches for '{query}'.")
                lines = []
                for h in hits:
                    head = f"{h['path']}" + (f" › {h['heading']}" if h["heading"] else "")
                    lines.append(f"- {head}\n    {h['snippet']}")
                return ToolResult(success=True, content=f"{len(hits)} match(es):\n" + "\n".join(lines))

            if action == "files":
                files = rm.list_files(project)
                if not files:
                    return ToolResult(success=True, content="(folder not indexed yet.)")
                lines = [f"- {f['path']}" + (f" — {f['purpose']}" if f["purpose"] else "")
                         for f in files]
                return ToolResult(success=True, content="\n".join(lines))

            if action == "file":
                if not path.strip():
                    return ToolResult(success=False, error="file needs a path")
                hits = rm.search(project, Path(path).stem.replace("-", " ").replace("_", " ") or path, 5)
                same = [h for h in hits if h["path"] == path]
                files = {f["path"]: f["purpose"] for f in rm.list_files(project)}
                if path not in files:
                    return ToolResult(success=True, content=f"'{path}' is not in the index.")
                out = f"{path}" + (f"\nPurpose: {files[path]}" if files[path] else "")
                for h in same[:5]:
                    out += f"\n  › {h['heading']}: {h['snippet']}" if h["heading"] else f"\n  {h['snippet']}"
                return ToolResult(success=True, content=out)

            if action == "stats":
                return ToolResult(success=True, content=json.dumps(rm.stats(project), indent=2))

            if action == "set_overview":
                rm.write_overview(project, content or "")
                return ToolResult(success=True, content="research overview saved.")

            return ToolResult(success=False, error=f"unknown action: {action}")
        except Exception as e:  # noqa: BLE001
            log.warning("researchmap tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"researchmap error: {e}")
