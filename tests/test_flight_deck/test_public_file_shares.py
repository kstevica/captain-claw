"""Public file links — /fd/agent-file-share (owner) and /s/{token} (anyone).

Pins the promises the Share dialog makes: the public side serves only the
snapshot (never the agent, never a path), names nothing but the file, renders
Markdown with raw HTML escaped, sandboxes anything that can run script, and a
revoked / expired / disabled link is gone at once.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import auth, public_file_shares as pfs
from captain_claw.flight_deck.db import FlightDeckDB

OWNER = {"id": "user-owner", "email": "o@x", "display_name": "Owner Person", "role": "user"}
OTHER = {"id": "user-other", "email": "t@x", "display_name": "Other", "role": "user"}
SRC = "/srv/agents/secret-agent/workspace/saved/downloads/session-42/"
AGENT = "/fd/agent-file-share/localhost/24123"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("FD_PUBLIC_FILE_LINKS", raising=False)
    monkeypatch.delenv("FD_PUBLIC_FILE_LINK_MAX_MB", raising=False)
    monkeypatch.delenv("FD_PUBLIC_URL", raising=False)
    monkeypatch.delenv("FD_ALLOWED_HOSTS", raising=False)
    pfs._origin_cache.clear()
    db = FlightDeckDB(tmp_path / "flight-deck.db")
    asyncio.run(db.init())
    auth.set_auth_db(db)

    files: dict[str, bytes] = {}
    seen: list[httpx.Request] = []

    def agent(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params.get("token") not in ("agent-secret-token", "tok-A", "tok-B"):
            return httpx.Response(401, json={"error": "Unauthorized"})
        body = files.get(request.url.params.get("path", ""))
        if body is None:
            return httpx.Response(404, json={"error": "File not found on disk"})
        return httpx.Response(200, content=body)

    real = httpx.AsyncClient
    monkeypatch.setattr(pfs.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(agent), **kw))
    endpoints = {"tok-A": {"port": 24123, "agent_ref": "proc-secret-agent", "name": "Secret Agent"},
                 "tok-B": {"port": 24124, "agent_ref": "proc-other", "name": "Other"}}
    pfs.configure(data_dir=tmp_path, resolve_auth=lambda port, host=None: "agent-secret-token",
                  resolve_agent_endpoint=endpoints.get)

    app = FastAPI()
    app.include_router(pfs.owner_router)
    app.include_router(pfs.public_router)
    current = {"user": OWNER}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    yield TestClient(app), files, seen, current, db, tmp_path
    asyncio.run(db.close())
    auth.set_auth_db(None)  # type: ignore[arg-type]


def _share(client, path, **extra):
    body = {"path": path, "agent_ref": "proc-secret-agent", "agent_name": "Secret Agent", **extra}
    r = client.post(AGENT, json=body)
    assert r.status_code == 200, r.text
    return r.json()["share"]


def _no_leak(text: str) -> None:
    for secret in ("secret-agent", "Secret Agent", "session-42", "user-owner", "Owner Person",
                   "24123", "agent-secret-token", "/srv/"):
        assert secret not in text, secret


def test_markdown_renders_with_raw_html_escaped(env):
    client, files, seen, *_ = env
    files[SRC + "report.md"] = (b"# Quarterly report\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
                                b"<script>alert(1)</script>\n\n[x](javascript:alert(1)) [ok](https://e.com)")
    share = _share(client, SRC + "report.md")
    assert share["kind"] == "markdown" and share["path"] == f"/s/{share['token']}"
    # The agent was asked with its own token, resolved server-side.
    assert seen[-1].url.params["token"] == "agent-secret-token"

    r = client.get(share["path"])
    assert r.status_code == 200
    page = r.text
    assert "<h1>Quarterly report</h1>" in page and "<table>" in page
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "href=\"javascript:" not in page   # left as plain text, never a link
    assert 'href="https://e.com" target="_blank" rel="noopener noreferrer nofollow"' in page
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "script-src" not in csp
    assert r.headers["x-robots-tag"].startswith("noindex")
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    _no_leak(page)
    # The markdown source itself comes back as inert text.
    raw = client.get(share["path"] + "/raw")
    assert raw.headers["content-type"].startswith("text/plain")
    assert raw.headers["content-security-policy"].startswith("sandbox")


def test_html_runs_sandboxed_on_an_opaque_origin(env):
    client, files, *_ = env
    files[SRC + "deck.html"] = b"<html><body><script>document.cookie</script>hi</body></html>"
    share = _share(client, SRC + "deck.html")
    page = client.get(share["path"]).text
    assert f'<iframe src="{share["path"]}/raw"' in page
    raw = client.get(share["path"] + "/raw")
    assert raw.headers["content-type"].startswith("text/html")
    csp = raw.headers["content-security-policy"]
    assert csp.startswith("sandbox allow-scripts") and "allow-same-origin" not in csp
    assert raw.content == files[SRC + "deck.html"]


@pytest.mark.parametrize("name,ctype", [("logo.svg", "image/svg+xml"), ("feed.xml", "text/plain"),
                                        ("notes.txt", "text/plain"), ("photo.png", "image/png")])
def test_raw_never_serves_a_script_capable_document(env, name, ctype):
    client, files, *_ = env
    files[SRC + name] = b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"
    share = _share(client, SRC + name)
    raw = client.get(share["path"] + "/raw")
    assert raw.headers["content-type"].startswith(ctype)
    assert raw.headers["content-security-policy"].startswith("sandbox;")
    assert raw.headers["x-content-type-options"] == "nosniff"


def test_other_types_download_only(env):
    client, files, *_ = env
    files[SRC + "Pilot Order Form.docx"] = b"PK\x03\x04docx"
    share = _share(client, SRC + "Pilot Order Form.docx")
    page = client.get(share["path"]).text
    assert "can't be previewed" in page and f'href="{share["path"]}/download"' in page
    dl = client.get(share["path"] + "/download")
    assert dl.content == b"PK\x03\x04docx"
    assert dl.headers["content-disposition"].startswith("attachment;")
    assert "Pilot%20Order%20Form.docx" in dl.headers["content-disposition"]


def test_snapshot_is_independent_of_the_agent(env):
    client, files, seen, *_ = env
    files[SRC + "a.txt"] = b"version one"
    share = _share(client, SRC + "a.txt")
    files[SRC + "a.txt"] = b"version two with a new secret"
    calls = len(seen)
    assert client.get(share["path"] + "/raw").content == b"version one"
    assert len(seen) == calls          # the public side never contacts the agent
    # Refreshing keeps the URL and publishes the new copy.
    again = _share(client, SRC + "a.txt", expires_days=7)
    assert again["token"] == share["token"] and again["expires_at"]
    assert client.get(share["path"] + "/raw").content == b"version two with a new secret"
    # No expiry given: the refresh keeps the one the link has.
    assert _share(client, SRC + "a.txt")["expires_at"] == again["expires_at"]
    # "Never" only where the deck allows it (FD_PUBLIC_FILE_LINK_MAX_DAYS=0).
    r = client.post(AGENT, json={"path": SRC + "a.txt", "agent_ref": "proc-secret-agent",
                                 "agent_name": "Secret Agent", "expires_days": 0})
    assert r.status_code == 400


def test_revoke_and_expiry_take_effect_at_once(env):
    client, files, _, _, db, tmp = env
    files[SRC + "a.md"] = b"hello"
    share = _share(client, SRC + "a.md")
    assert client.get(share["path"]).status_code == 200
    assert client.delete(f"/fd/public-shares/{share['token']}").json() == {"ok": True}
    gone = client.get(share["path"])
    assert gone.status_code == 404 and "isn't available" in gone.text
    assert client.get(share["path"] + "/raw").status_code == 404
    assert not (tmp / "public_shares" / share["token"]).exists()

    share = _share(client, SRC + "a.md", expires_days=1)
    row = asyncio.run(db.get_public_file_share(share["token"]))
    row["expires_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    asyncio.run(db.save_public_file_share(row))
    assert client.get(share["path"]).status_code == 404
    assert asyncio.run(db.get_public_file_share(share["token"])) is None


def test_views_counted_and_links_are_the_owners_alone(env):
    client, files, _, current, *_ = env
    files[SRC + "a.md"] = b"hello"
    share = _share(client, SRC + "a.md")
    client.get(share["path"])
    client.get(share["path"])
    mine = client.get("/fd/public-shares").json()["shares"]
    assert [s["view_count"] for s in mine] == [2]

    current["user"] = OTHER
    assert client.get("/fd/public-shares").json()["shares"] == []
    assert client.get(AGENT, params={"path": SRC + "a.md", "agent_ref": "proc-secret-agent"}
                      ).json()["share"] is None
    assert client.delete(f"/fd/public-shares/{share['token']}").status_code == 404
    assert client.get(share["path"]).status_code == 200


def test_limits_and_bad_input(env, monkeypatch):
    client, files, *_ = env
    monkeypatch.setenv("FD_PUBLIC_FILE_LINK_MAX_MB", "1")
    files[SRC + "big.bin"] = b"x" * (1024 * 1024 + 1)
    r = client.post(AGENT, json={"path": SRC + "big.bin"})
    assert r.status_code == 413
    assert client.post(AGENT, json={"path": SRC + "missing.md"}).status_code == 404
    # A wrong agent token is the agent's 401 — never the caller's session.
    files[SRC + "a.md"] = b"x"
    assert client.post(AGENT, json={"path": SRC + "a.md", "agent_auth": "wrong"}).status_code == 502
    assert client.post(AGENT, json={"path": SRC + "a", "expires_days": 3}).status_code == 400
    assert client.get("/s/short").status_code == 404
    assert client.get("/s/" + "A" * 30).status_code == 404


def test_deck_switch_turns_every_link_off(env, monkeypatch):
    client, files, *_ = env
    files[SRC + "a.md"] = b"hello"
    share = _share(client, SRC + "a.md")
    monkeypatch.setenv("FD_PUBLIC_FILE_LINKS", "off")
    assert client.get("/fd/public-shares/config").json()["enabled"] is False
    assert client.get(share["path"]).status_code == 404
    assert client.post(AGENT, json={"path": SRC + "a.md"}).status_code == 403


def test_owner_route_is_port_addressed_for_the_ownership_guard():
    # server._agent_proxy_port is what makes the hardening middleware refuse
    # another user's agent; the share routes must keep that shape.
    from captain_claw.flight_deck import server

    assert server._agent_proxy_port(AGENT) == 24123


# ── Agent routes (the `public_link` tool) ────────────────────────────

@pytest.fixture()
def agents(env, monkeypatch):
    """Two agents of OWNER, identified by X-Agent-Auth as the real check does."""
    from captain_claw.flight_deck import server

    owners = {"tok-A": OWNER["id"], "tok-B": OWNER["id"], "tok-orphan": ""}
    monkeypatch.setattr(server, "_agent_caller_ok", lambda request: True)
    monkeypatch.setattr(server, "_find_agent_by_auth",
                        lambda t: (t in owners, owners.get(t, ""), "slug"))
    return env


def _agent(client, tok, route, **body):
    return client.post(f"/fd/public-shares/agent/{route}", json=body, headers={"X-Agent-Auth": tok})


def test_agent_share_lands_in_the_owners_links(agents, monkeypatch):
    client, files, seen, *_ = agents
    files[SRC + "report.md"] = b"# Hi"
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example.com/")
    r = _agent(client, "tok-A", "share", path=SRC + "report.md", expires_days=7)
    assert r.status_code == 200, r.text
    share = r.json()["share"]
    assert share["url"] == f"https://deck.example.com/s/{share['token']}"
    assert share["local_only"] is False and share["expires_at"]
    # Pulled from the calling agent's own file server, with its own token.
    assert seen[-1].url.port == 24123 and seen[-1].url.params["token"] == "tok-A"
    assert client.get(f"/s/{share['token']}").status_code == 200
    # The owner's Share dialog finds it for that agent + file.
    got = client.get(AGENT, params={"path": SRC + "report.md", "agent_ref": "proc-secret-agent"})
    assert got.json()["share"]["token"] == share["token"]
    # Sharing again (from the UI or the agent) keeps the URL.
    assert _share(client, SRC + "report.md")["token"] == share["token"]


def test_agent_link_without_public_url_is_flagged_local(agents, monkeypatch):
    client, files, *_ = agents
    monkeypatch.delenv("FD_PUBLIC_URL", raising=False)
    files[SRC + "a.md"] = b"x"
    share = client.post("/fd/public-shares/agent/share", json={"path": SRC + "a.md"},
                        headers={"X-Agent-Auth": "tok-A"}, ).json()["share"]
    assert share["url"].startswith("http://testserver/s/")
    assert pfs._private_host("127.0.0.1") and pfs._private_host("localhost")
    assert not pfs._private_host("deck.example.com")


def test_agent_sees_and_stops_only_its_own_links(agents):
    client, files, *_ = agents
    files[SRC + "a.md"] = b"x"
    a = _agent(client, "tok-A", "share", path=SRC + "a.md").json()["share"]
    assert [s["token"] for s in _agent(client, "tok-B", "list").json()["shares"]] == []
    assert [s["token"] for s in _agent(client, "tok-A", "list").json()["shares"]] == [a["token"]]
    assert _agent(client, "tok-B", "unshare", token=a["token"]).status_code == 404
    assert _agent(client, "tok-B", "unshare", path=SRC + "a.md").status_code == 404
    assert client.get(f"/s/{a['token']}").status_code == 200
    r = _agent(client, "tok-A", "unshare", path=SRC + "a.md")
    assert r.json() == {"ok": True, "filename": "a.md"}
    assert client.get(f"/s/{a['token']}").status_code == 404


def test_agent_routes_refuse_browsers_strangers_and_ownerless(agents, monkeypatch):
    client, files, *_ = agents
    files[SRC + "a.md"] = b"x"
    browser = client.post("/fd/public-shares/agent/share", json={"path": SRC + "a.md"},
                          headers={"X-Agent-Auth": "tok-A", "Sec-Fetch-Site": "same-origin"})
    assert browser.status_code == 403
    assert _agent(client, "", "share", path=SRC + "a.md").status_code == 403
    assert _agent(client, "tok-stranger", "share", path=SRC + "a.md").status_code == 403
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    assert _agent(client, "tok-orphan", "share", path=SRC + "a.md").status_code == 403
    monkeypatch.setenv("FD_PUBLIC_FILE_LINKS", "off")
    assert _agent(client, "tok-A", "share", path=SRC + "a.md").status_code == 403


# ── Where an agent's link points (the browser's origin, not FD_URL) ──

def test_agent_link_uses_the_origin_the_owners_browser_reported(agents, monkeypatch):
    client, files, _, current, *_ = agents
    files[SRC + "a.md"] = b"x"
    # Before any browser reported in (and no domain configured): the agent's
    # own base URL.
    first = _agent(client, "tok-A", "share", path=SRC + "a.md").json()["share"]
    assert first["url"].startswith("http://testserver/s/")

    monkeypatch.setenv("FD_ALLOWED_HOSTS", "glasses.example.com")
    r = client.post("/fd/public-shares/origin", headers={"Origin": "https://glasses.example.com"})
    assert r.json() == {"origin": "https://glasses.example.com"}
    share = _agent(client, "tok-A", "share", path=SRC + "a.md").json()["share"]
    assert share["url"] == f"https://glasses.example.com/s/{share['token']}"
    assert share["local_only"] is False
    listed = _agent(client, "tok-A", "list").json()["shares"]
    assert listed[0]["url"] == share["url"]
    # FD_PUBLIC_URL, when set, still wins.
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example.com")
    again = _agent(client, "tok-A", "share", path=SRC + "a.md").json()["share"]
    assert again["url"].startswith("https://deck.example.com/s/")


def test_only_public_origins_this_deck_answers_to_are_learned(agents, monkeypatch):
    client, *_ , db, _ = agents
    for bad in ("http://localhost:8765", "http://127.0.0.1:25080", "http://192.168.1.20:8765",
                "https://evil.example.net", "javascript:alert(1)", ""):
        r = client.post("/fd/public-shares/origin", headers={"Origin": bad} if bad else {})
        assert r.json() == {"origin": None}, bad
    assert asyncio.run(db.get_public_link_origin()) == ""
    monkeypatch.setenv("FD_ALLOWED_HOSTS", ".example.com")
    r = client.post("/fd/public-shares/origin", headers={"Origin": "https://glasses.example.com:8443"})
    assert r.json() == {"origin": "https://glasses.example.com:8443"}


def test_link_base_fallbacks(agents, monkeypatch):
    client, files, _, current, db, _ = agents
    files[SRC + "a.md"] = b"x"
    # Another user's browser was seen on the public address: used when this
    # owner never opened the deck from a browser (WhatsApp-only users).
    asyncio.run(db.set_public_link_origin("someone-else", "https://glasses.example.com"))
    share = _agent(client, "tok-A", "share", path=SRC + "a.md").json()["share"]
    assert share["url"].startswith("https://glasses.example.com/s/")
    # No origin seen at all: a concrete FD_ALLOWED_HOSTS domain, as https.
    asyncio.run(db._db.execute("DELETE FROM public_link_origins"))
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "*.example.com, deck.example.org")
    share = _agent(client, "tok-A", "share", path=SRC + "a.md").json()["share"]
    assert share["url"].startswith("https://deck.example.org/s/")


# ── Every link expires (cohort001: client files on links nobody remembered) ──


def test_new_links_expire_by_default_and_never_is_off(env, monkeypatch):
    client, files, *_ = env
    for name in ("FD_PUBLIC_FILE_LINK_DAYS", "FD_PUBLIC_FILE_LINK_MAX_DAYS"):
        monkeypatch.delenv(name, raising=False)
    files[SRC + "a.txt"] = b"hello"
    cfg = client.get("/fd/public-shares/config").json()
    assert cfg["expiry_days"] == [1, 7, 30] and cfg["default_days"] == 7
    share = _share(client, SRC + "a.txt")
    left = datetime.fromisoformat(share["expires_at"]) - datetime.now(timezone.utc)
    assert timedelta(days=6, hours=23) < left <= timedelta(days=7)
    r = client.post(AGENT, json={"path": SRC + "a.txt", "agent_ref": "proc-secret-agent", "expires_days": 0})
    assert r.status_code == 400 and "never" in r.text


def test_a_deck_may_let_links_never_expire(env, monkeypatch):
    client, files, *_ = env
    monkeypatch.setenv("FD_PUBLIC_FILE_LINK_MAX_DAYS", "0")
    monkeypatch.setenv("FD_PUBLIC_FILE_LINK_DAYS", "0")
    files[SRC + "a.txt"] = b"hello"
    assert client.get("/fd/public-shares/config").json()["expiry_days"] == [1, 7, 30, 0]
    assert _share(client, SRC + "a.txt")["expires_at"] is None


def test_a_cap_below_the_default_wins(monkeypatch):
    monkeypatch.setenv("FD_PUBLIC_FILE_LINK_MAX_DAYS", "3")
    monkeypatch.delenv("FD_PUBLIC_FILE_LINK_DAYS", raising=False)
    assert pfs.default_days() == 3 and pfs.expiry_options() == [1, 3]


def test_old_links_that_never_expired_get_a_date_and_owners_are_told(env, monkeypatch):
    client, files, _seen, _current, db, _tmp = env
    from captain_claw.flight_deck import line

    posted = []

    async def fake_post(user_id, kind, title, body="", **kw):
        posted.append((user_id, title, body))
        return "id"

    monkeypatch.setattr(line, "post", fake_post)
    # Made while the deck let links live for ever.
    monkeypatch.setenv("FD_PUBLIC_FILE_LINK_MAX_DAYS", "0")
    monkeypatch.setenv("FD_PUBLIC_FILE_LINK_DAYS", "0")
    files[SRC + "old.md"] = b"# old"
    files[SRC + "new.md"] = b"# new"
    old = _share(client, SRC + "old.md")
    new = _share(client, SRC + "new.md")
    async def age(token: str, days: int) -> None:
        await db._db.execute("UPDATE public_file_shares SET created_at = ? WHERE token = ?",
                             ((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(), token))
        await db._db.commit()

    asyncio.run(age(old["token"], 60))
    monkeypatch.delenv("FD_PUBLIC_FILE_LINK_MAX_DAYS")
    monkeypatch.delenv("FD_PUBLIC_FILE_LINK_DAYS")
    assert asyncio.run(pfs.cap_open_links(db)) == 2
    now = datetime.now(timezone.utc)
    old_until = datetime.fromisoformat(asyncio.run(db.get_public_file_share(old["token"]))["expires_at"])
    new_until = datetime.fromisoformat(asyncio.run(db.get_public_file_share(new["token"]))["expires_at"])
    # 60 days old: past the 30-day cap, so it gets the week's notice.
    assert timedelta(days=6, hours=23) < old_until - now <= timedelta(days=7)
    assert timedelta(days=29) < new_until - now <= timedelta(days=30)
    assert len(posted) == 1 and posted[0][0] == OWNER["id"]
    assert "2 public links" in posted[0][1] and "old.md" in posted[0][2]
    assert asyncio.run(pfs.cap_open_links(db)) == 0  # once only
