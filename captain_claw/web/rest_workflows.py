"""REST handlers for the workflow browser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.config import get_config

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


def workflows_dir() -> Path:
    """Return the workflows directory (same as SessionOrchestrator)."""
    cfg = get_config()
    ws = cfg.resolved_workspace_path()
    d = ws / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def list_workflow_outputs(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/workflow-browser — list workflows with their outputs."""
    d = workflows_dir()
    workflows: dict[str, dict[str, Any]] = {}
    for p in sorted(d.glob("*.json")):
        name = p.stem
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            wf_name = data.get("workflow_name", name)
            user_input = data.get("user_input", "")
            task_count = len(data.get("tasks", []))
        except Exception:
            wf_name = name
            user_input = ""
            task_count = 0
        workflows[name] = {
            "name": wf_name,
            "filename": name,
            "user_input": user_input,
            "task_count": task_count,
            "outputs": [],
        }

    for p in sorted(d.glob("*-output-*.md")):
        fname = p.name
        stem = p.stem
        matched = False
        for wf_key in workflows:
            if stem.startswith(wf_key + "-output-"):
                ts_part = stem[len(wf_key) + len("-output-"):]
                try:
                    stat = p.stat()
                    size = stat.st_size
                except Exception:
                    size = 0
                workflows[wf_key]["outputs"].append({
                    "filename": fname,
                    "timestamp": ts_part,
                    "size": size,
                })
                matched = True
                break

        if not matched:
            idx = stem.rfind("-output-")
            if idx > 0:
                inferred_key = stem[:idx]
                ts_part = stem[idx + len("-output-"):]
            else:
                inferred_key = stem
                ts_part = ""
            if inferred_key not in workflows:
                workflows[inferred_key] = {
                    "name": inferred_key,
                    "filename": inferred_key,
                    "user_input": "",
                    "task_count": 0,
                    "outputs": [],
                }
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            workflows[inferred_key]["outputs"].append({
                "filename": fname,
                "timestamp": ts_part,
                "size": size,
            })

    for wf in workflows.values():
        wf["outputs"].sort(key=lambda o: o.get("timestamp", ""), reverse=True)

    result = sorted(workflows.values(), key=lambda w: w["name"])
    return web.json_response(result)


async def get_workflow_output(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/workflow-browser/output/{filename} — read an output .md file."""
    filename = request.match_info.get("filename", "")
    if not filename or ".." in filename or "/" in filename:
        return web.json_response({"error": "Invalid filename"}, status=400)

    d = workflows_dir()
    path = d / filename
    if not path.is_file():
        return web.json_response({"error": "File not found"}, status=404)

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response({"filename": filename, "content": content})
