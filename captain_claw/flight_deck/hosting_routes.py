"""Routes for VFS Hosting.

* Management API under ``/fd/hosting/*`` — owner-gated (publish / unpublish /
  start / stop / list). Requires login when auth is enabled.
* Public static serving at ``/vfs/<name>/…`` — no auth; resolves to the
  publisher's VFS folder recorded in the registry.
* Public app reverse-proxy at ``/vfs-apps/<name>/…`` — HTTP (all methods) and
  WebSocket, forwarded to the app's Flight-Deck-assigned localhost port.

The public routes are included before the SPA catch-all, so ``/vfs`` and
``/vfs-apps`` win over ``index.html``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import html as _html

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.responses import Response

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck import vfs_hosting as vh

router = APIRouter(tags=["hosting"])

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

# Hop-by-hop headers never forwarded in either direction (RFC 7230 §6.1) + Host.
_HOP = {
    "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


def _public_url(name: str, kind: str) -> str:
    return f"/vfs-apps/{name}/" if kind == "app" else f"/vfs/{name}/"


def _view(name: str, entry: dict) -> dict:
    e = vh.reconcile(name, entry)
    return {
        "name": name,
        "kind": e.get("kind"),
        "project": e.get("project", ""),
        "subdir": e.get("subdir", ""),
        "start_cmd": e.get("start_cmd", ""),
        "running": bool(e.get("running")),
        "port": e.get("port"),
        "url": _public_url(name, e.get("kind", "static")),
    }


# ── Management (owner-gated) ──────────────────────────────────────────────

class PublishBody(BaseModel):
    name: str
    kind: str  # "static" | "app"
    project: str
    subdir: str = ""
    start_cmd: str = ""


@router.get("/fd/hosting")
async def list_hosting(user: dict = Depends(get_current_user)):
    reg = vh.load_registry()
    mine = [_view(n, e) for n, e in reg.items() if e.get("owner") == user["id"]]
    mine.sort(key=lambda x: x["name"])
    return {"entries": mine}


_SKIP_DIRS = {"node_modules", ".git", ".code", ".codemap", ".captain-claw", "__pycache__", ".venv"}


@router.get("/fd/hosting/folders")
async def list_folders(project: str, user: dict = Depends(get_current_user)):
    """List the subfolders of a VFS project (relative paths; '' = project root)."""
    base = vh.entry_dir({"owner": user["id"], "project": project, "subdir": ""})
    if base is None:
        raise HTTPException(404, "project not found")
    out: list[str] = []
    for root, dnames, _files in os.walk(base):
        # Prune noisy/heavy dirs in place so we don't descend into them.
        dnames[:] = sorted(d for d in dnames if d not in _SKIP_DIRS and not d.startswith("."))
        rel = os.path.relpath(root, base)
        out.append("" if rel == "." else rel.replace(os.sep, "/"))
        if len(out) > 800:
            break
    return {"folders": sorted(set(out), key=lambda p: (p != "", p))}


@router.post("/fd/hosting")
async def publish(body: PublishBody, user: dict = Depends(get_current_user)):
    name = (body.name or "").strip().lower()
    if not vh.valid_name(name):
        raise HTTPException(400, "Name must be 1–63 chars: lowercase letters, digits, dashes.")
    if body.kind not in ("static", "app"):
        raise HTTPException(400, "kind must be 'static' or 'app'")
    if not body.project.strip():
        raise HTTPException(400, "project is required")
    if body.kind == "app" and not body.start_cmd.strip():
        raise HTTPException(400, "a start command is required for apps")

    reg = vh.load_registry()
    if name in reg:
        raise HTTPException(409, f"The name '{name}' is already taken.")

    entry = {
        "kind": body.kind,
        "owner": user["id"],
        "project": body.project.strip(),
        "subdir": body.subdir.strip().strip("/"),
        "start_cmd": body.start_cmd.strip(),
    }
    if vh.entry_dir(entry) is None:
        raise HTTPException(404, "That VFS folder doesn't exist.")

    reg[name] = entry
    vh.save_registry(reg)
    return _view(name, entry)


class EditBody(BaseModel):
    kind: str
    project: str
    subdir: str = ""
    start_cmd: str = ""


@router.put("/fd/hosting/{name}")
async def edit(name: str, body: EditBody, user: dict = Depends(get_current_user)):
    ent = _owned_entry(name, user)
    if body.kind not in ("static", "app"):
        raise HTTPException(400, "kind must be 'static' or 'app'")
    if not body.project.strip():
        raise HTTPException(400, "project is required")
    if body.kind == "app" and not body.start_cmd.strip():
        raise HTTPException(400, "a start command is required for apps")

    updated = {
        **ent,
        "kind": body.kind,
        "project": body.project.strip(),
        "subdir": body.subdir.strip().strip("/"),
        "start_cmd": body.start_cmd.strip(),
    }
    if vh.entry_dir(updated) is None:
        raise HTTPException(404, "That VFS folder doesn't exist.")

    # The config changed — stop any running app so the user restarts cleanly.
    if ent.get("kind") == "app":
        vh.stop_app(name)
    updated["pid"] = None
    updated["port"] = None
    updated.pop("stopped", None)

    reg = vh.load_registry()
    reg[name] = updated
    vh.save_registry(reg)
    return _view(name, updated)


def _owned_entry(name: str, user: dict) -> dict:
    ent = vh.load_registry().get(name)
    if not ent:
        raise HTTPException(404, "not found")
    if ent.get("owner") != user["id"] and user.get("role") != "admin":
        raise HTTPException(403, "not your published entry")
    return ent


@router.delete("/fd/hosting/{name}")
async def unpublish(name: str, user: dict = Depends(get_current_user)):
    ent = _owned_entry(name, user)
    if ent.get("kind") == "app":
        vh.stop_app(name)
    reg = vh.load_registry()
    reg.pop(name, None)
    vh.save_registry(reg)
    return {"ok": True}


@router.post("/fd/hosting/{name}/start")
async def start(name: str, user: dict = Depends(get_current_user)):
    _owned_entry(name, user)
    ok, msg = vh.start_app(name)
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


@router.post("/fd/hosting/{name}/stop")
async def stop(name: str, user: dict = Depends(get_current_user)):
    _owned_entry(name, user)
    ok, msg = vh.stop_app(name)
    return {"ok": ok, "message": msg}


# ── Public static serving ─────────────────────────────────────────────────

def _dir_listing(name: str, rel_path: str, directory) -> HTMLResponse:
    """Render a simple autoindex for a folder that has no index.html."""
    rel = (rel_path or "").strip("/")
    base_url = f"/vfs/{name}/" + (rel + "/" if rel else "")
    rows: list[str] = []
    if rel:
        parent = "/".join(rel.split("/")[:-1])
        up = f"/vfs/{name}/" + (parent + "/" if parent else "")
        rows.append(f'<li><a href="{_html.escape(up)}">../</a></li>')
    for p in sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        label = p.name + ("/" if p.is_dir() else "")
        href = _html.escape(base_url + p.name + ("/" if p.is_dir() else ""))
        rows.append(f'<li><a href="{href}">{_html.escape(label)}</a></li>')
    title = _html.escape(f"/{name}/{rel}".rstrip("/") + "/")
    body = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{title}</title>"
        "<style>body{font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "max-width:820px;margin:40px auto;padding:0 20px;color:#18181b}"
        "h1{font-size:15px;font-weight:600;border-bottom:1px solid #e4e4e7;padding-bottom:8px}"
        "ul{list-style:none;padding:0}li{padding:2px 0}a{color:#6d28d9;text-decoration:none}"
        "a:hover{text-decoration:underline}</style></head>"
        f"<body><h1>Index of {title}</h1><ul>{''.join(rows)}</ul></body></html>"
    )
    return HTMLResponse(body)


def _serve_static(name: str, path: str) -> Response:
    ent = vh.load_registry().get(name)
    if not ent or ent.get("kind") != "static":
        raise HTTPException(404, "not found")
    target = vh.resolve_static_file(ent, path)
    if target is not None and target.is_dir():
        idx = target / "index.html"
        if idx.is_file():
            return FileResponse(idx)
        # No default index — list the folder contents instead of 404ing.
        return _dir_listing(name, path, target)
    if target is not None and target.is_file():
        return FileResponse(target)
    # Missing path: fall back to index.html for extension-less routes so a
    # client-routed SPA build works; hard-404 missing assets (.js/.css/…).
    if "." not in Path(path or "").name:
        base = vh.entry_dir(ent)
        if base is not None:
            idx = base / "index.html"
            if idx.is_file():
                return FileResponse(idx)
    raise HTTPException(404, "not found")


@router.get("/vfs/{name}")
async def serve_static_root(name: str):
    return _serve_static(name, "")


@router.get("/vfs/{name}/{path:path}")
async def serve_static_path(name: str, path: str):
    return _serve_static(name, path)


# ── Public app reverse-proxy (HTTP) ───────────────────────────────────────

async def _proxy_http(name: str, path: str, request: Request) -> Response:
    if not vh.app_is_alive(name):
        return JSONResponse({"error": f"App '{name}' is not running. Start it from the Hosting page."}, status_code=502)
    port = vh.app_port(name)
    if not port:
        return JSONResponse({"error": "app has no port"}, status_code=502)

    import httpx
    target = f"http://127.0.0.1:{port}/{path}"
    if request.url.query:
        target += f"?{request.url.query}"
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(
                request.method, target, content=body or None,
                headers=fwd_headers, follow_redirects=False,
            )
    except httpx.ConnectError:
        return JSONResponse({"error": "app is starting or unreachable"}, status_code=502)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"proxy error: {exc}"}, status_code=502)

    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP}
    media = resp.headers.get("content-type")
    return Response(content=resp.content, status_code=resp.status_code, headers=out_headers, media_type=media)


@router.api_route("/vfs-apps/{name}", methods=_PROXY_METHODS)
async def proxy_app_root(name: str, request: Request):
    return await _proxy_http(name, "", request)


@router.api_route("/vfs-apps/{name}/{path:path}", methods=_PROXY_METHODS)
async def proxy_app_path(name: str, path: str, request: Request):
    return await _proxy_http(name, path, request)


# ── Public app reverse-proxy (WebSocket) ──────────────────────────────────

async def _proxy_ws(ws: WebSocket, name: str, path: str) -> None:
    port = vh.app_port(name)
    if not port or not vh.app_is_alive(name):
        await ws.close(code=1011)
        return
    import websockets

    q = ws.url.query
    upstream = f"ws://127.0.0.1:{port}/{path}" + (f"?{q}" if q else "")
    await ws.accept()
    try:
        async with websockets.connect(upstream, max_size=None, open_timeout=10) as up:
            async def c2u() -> None:
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if msg.get("text") is not None:
                            await up.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await up.send(msg["bytes"])
                except Exception:  # noqa: BLE001
                    pass

            async def u2c() -> None:
                try:
                    async for m in up:
                        if isinstance(m, (bytes, bytearray)):
                            await ws.send_bytes(m)
                        else:
                            await ws.send_text(m)
                except Exception:  # noqa: BLE001
                    pass

            tasks = [asyncio.create_task(c2u()), asyncio.create_task(u2c())]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


@router.websocket("/vfs-apps/{name}")
async def proxy_app_ws_root(ws: WebSocket, name: str):
    await _proxy_ws(ws, name, "")


@router.websocket("/vfs-apps/{name}/{path:path}")
async def proxy_app_ws_path(ws: WebSocket, name: str, path: str):
    await _proxy_ws(ws, name, path)
