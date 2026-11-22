"""No request path climbs out of where it was sent.

2026-11-19/20: scanners sent ``GET /../../../../../../../../../../../../proc/self/cwd/.env``
to a deck and got its ``.env`` — the SPA catch-all served ``STATIC_DIR / path``
for any path, and a raw ``..`` reaches the app unchanged (through the proxy too).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.flight_deck import path_guard, server


@pytest.mark.parametrize("path", [
    "/../etc/passwd",
    "/files/../../../../proc/self/cwd/.env",
    "/cgi-bin/../../../../etc/passwd",
    "/a/%2e%2e/%2e%2e/etc/passwd",
    "/a/%2E%2E%2Fetc",
    "/a/%252e%252e/etc",
    "/..\\..\\windows\\win.ini",
    "/assets/..",
])
def test_climbing_paths_are_seen(path):
    assert path_guard.has_parent_segment(path)


@pytest.mark.parametrize("path", [
    "/", "/assets/index-AbC.js", "/fd/chat/sessions/proc-a/messages",
    "/files/report..final.pdf", "/..hidden/x", "/a.../b", "/s/abc%20def",
])
def test_ordinary_paths_pass(path):
    assert not path_guard.has_parent_segment(path)


def test_inside(tmp_path: Path):
    base = tmp_path / "static"
    (base / "assets").mkdir(parents=True)
    (base / "assets" / "a.js").write_text("x")
    (tmp_path / ".env").write_text("SECRET=1")
    (base / "link").symlink_to(tmp_path / ".env")
    assert path_guard.inside(base, "assets/a.js") == (base / "assets" / "a.js").resolve()
    assert path_guard.inside(base, "../.env") is None
    assert path_guard.inside(base, "/../../.env") is None
    assert path_guard.inside(base, "link") is None          # a symlink out is out
    assert path_guard.inside(base, "") == base.resolve()


async def _asgi(app, path: str, raw: bytes | None = None, kind: str = "http") -> tuple[int, bytes, list]:
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    scope = {"type": kind, "method": "GET", "path": path, "raw_path": raw if raw is not None else path.encode(),
             "query_string": b"", "headers": [(b"host", b"deck.test")], "scheme": "http",
             "server": ("deck.test", 80), "client": ("203.0.113.9", 4242), "root_path": "",
             "http_version": "1.1"}
    await app(scope, receive, send)
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return (start["status"] if start else 0), body, sent


async def test_guard_refuses_before_any_route():
    reached: list[str] = []

    async def inner(scope, receive, send):
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = path_guard.PathTraversalGuard(inner)
    status, body, _ = await _asgi(guard, "/../../../../proc/self/cwd/.env")
    assert status == 400 and body == b"Bad request" and reached == []
    # Decoded path clean, raw path not: still refused.
    status, _, _ = await _asgi(guard, "/x/y", raw=b"/x/%2e%2e/%2e%2e/etc/passwd")
    assert status == 400 and reached == []
    status, body, _ = await _asgi(guard, "/assets/index.js")
    assert status == 200 and reached == ["/assets/index.js"]
    _, _, sent = await _asgi(guard, "/../ws", kind="websocket")
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_the_deck_runs_the_guard_outermost():
    assert server.app.user_middleware[0].cls is path_guard.PathTraversalGuard


async def test_spa_catch_all_never_leaves_static(tmp_path, monkeypatch):
    """The route itself, past the guard: a climbing path gets the app's page."""
    route = next((r for r in server.app.routes if getattr(r, "name", "") == "spa_catch_all"), None)
    if route is None:
        pytest.skip("no static build in this checkout")
    static = server.STATIC_DIR
    secret = static.parent / "path_guard_probe.txt"
    secret.write_text("not for the web")
    try:
        resp = await route.endpoint("../path_guard_probe.txt", None)
        assert Path(resp.path).resolve() == (static / "index.html").resolve()
        ok = await route.endpoint("index.html", None)
        assert Path(ok.path).resolve() == (static / "index.html").resolve()
    finally:
        secret.unlink()
