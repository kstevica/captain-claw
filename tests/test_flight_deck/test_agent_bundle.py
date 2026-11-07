"""Agent bundles (phase 1): a full backup of a process agent, and a new agent made of it.

* the bundle carries the agent's own files, memory and conversations — its
  stores snapshotted through SQLite's backup API, so what still sits in a WAL
  file is in it — and Flight Deck's side: instructions, chat log, scheduled tasks;
* it never carries a key, token, the owner marker, the owner's profile, logs or
  the semantic index; secret columns are emptied and the file vacuumed;
* the restored agent is made by the normal spawn (owner, port, web token,
  instructions), its scheduled tasks switched off, its chat log its owner's;
* reading a bundle trusts nothing: unsafe paths, links, files the manifest
  doesn't list, changed bytes and a newer format are refused, and an import
  writes only the agent's own catalogued parts;
* bundles and uploads are their owner's alone.
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import httpx
import pytest
import yaml

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import fd_scheduler as sched
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.agent_bundle import archive, catalog, export, restore, stores
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.insights import InsightsManager

BOB = "user-bob"
CAROL = "user-carol"
HC = catalog.HC


class _FakePopen:
    calls: list[str] = []

    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None,
                 creationflags=0):
        _FakePopen.calls.append(Path(cwd).name)
        self.pid = 424242
        if stdout is not None:
            stdout.close()

    def poll(self):
        return None


def _sessions_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as c:
        c.executescript("""
            CREATE TABLE sessions (id TEXT PRIMARY KEY, name TEXT, messages TEXT);
            CREATE TABLE apis (id TEXT PRIMARY KEY, name TEXT, base_url TEXT, credentials TEXT);
            CREATE TABLE browser_credentials (id TEXT PRIMARY KEY, site TEXT, password_encrypted TEXT);
            CREATE TABLE playbooks (id TEXT PRIMARY KEY, name TEXT, body TEXT);
            CREATE TABLE contacts (id TEXT PRIMARY KEY, name TEXT, email TEXT);
            INSERT INTO contacts VALUES ('c1', 'Ana', 'ana@client.example');
            INSERT INTO sessions VALUES ('s1', 'Monday', '[{"role":"user","content":"hello"}]');
            INSERT INTO apis VALUES ('a1', 'CRM', 'https://crm.example', 'TOPSECRET-API-KEY');
            INSERT INTO browser_credentials VALUES ('b1', 'bank', 'TOPSECRET-PASSWORD');
            INSERT INTO playbooks VALUES ('p1', 'Invoices', 'Check the VAT first');
        """)


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: Bob owns a well-used agent, Carol one of her own."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid in (BOB, CAROL):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.setenv("FD_PORT", "25999")
    monkeypatch.setenv("FD_SPAWN_SETTLE_S", "0")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _FakePopen.calls = []
    monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(sched, "_STORE", sched.SchedulerStore(db_path=tmp_path / "scheduler.db"))

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "description": "Does the books",
                      "web_port": 24101, "web_auth": "bob-tok", "owner": BOB, "pid": None,
                      "provider": "openai", "model": "gpt-x", "archetype": "bookkeeper", "model_tier": "balanced"},
        "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 24102,
                        "web_auth": "carol-tok", "owner": CAROL, "pid": None},
    })
    bob = data / "bob-agent"
    hc = bob / HC
    (hc / "skills" / "vat-check").mkdir(parents=True)
    (hc / "skills" / "vat-check" / "SKILL.md").write_text("# VAT check\n")
    (hc / "reflections").mkdir()
    (hc / "reflections" / "2026-11-01T10-00-00+00-00.md").write_text("- Ask before paying.\n")
    (hc / "personality.md").write_text("Calm and exact.\n")
    (hc / "cognitive_mode.txt").write_text("aeolian")
    (hc / "eco_mode.txt").write_text("off")
    (hc / "lessons.json").write_text('{"lessons": [{"text": "Invoices over 1000 EUR need two signatures"}]}')
    (hc / "tenant_context.md").write_text("Bob Builder, bob@private.example\n")   # the owner's profile
    (bob / "data" / "workspace" / "knowledge").mkdir(parents=True)
    (bob / "data" / "workspace" / "knowledge" / "handbook.md").write_text("# Handbook\n")
    (bob / "data" / "workspace" / "Izvješće.md").write_text("Q3 report\n")
    (bob / "data" / "home-config").mkdir(parents=True)
    (bob / "data" / "home-config" / "memory.db").write_bytes(b"index")
    (bob / "practice.json").write_text("[]")
    (bob / ".fd-owner").write_text(BOB)
    (bob / ".env").write_text("OPENAI_API_KEY=sk-bob-secret\n")
    (bob / "process.log").write_text("log\n")
    (bob / "config.yaml").write_text(yaml.dump({
        "model": {"provider": "openai", "model": "gpt-x", "api_key": "sk-bob-secret", "vision": True,
                  "temperature": 0.3, "max_tokens": 4096},
        "context": {"max_tokens": 100000},
        "tools": {"enabled": ["read", "write", "insights", "google_mail"]},
        "web": {"port": 24101, "auth_token": "bob-tok"},
        "telegram": {"enabled": True, "bot_token": "tg-secret"},
    }))
    _sessions_db(bob / "data" / "sessions" / "sessions.db")
    insights = InsightsManager(hc / "insights.db")
    await insights.add("The client's books close on the 5th in Zagreb", "workflow", entity_key="books-close")
    await insights.add("Bob prefers short answers", "preference")
    await insights.close()

    await server._set_owner_agent_instructions(BOB, "process", "bob-agent", "Be brief.")
    for sid in ("proc-bob-agent", "proc-bob-agent::side"):
        await db.upsert_chat_session(sid, BOB, "proc-bob-agent", "Bob's agent")
        await db._db.execute("INSERT INTO chat_messages (session_id, role, content, created_at)"
                             " VALUES (?, 'user', 'hi from ' || ?, '2026-10-01T09:00:00Z')", (sid, sid))
    await db.upsert_chat_session("proc-carol-agent", CAROL, "proc-carol-agent", "Carol's agent")
    await db._db.commit()
    sched.get_store().create(owner_id=BOB, name="Morning books", schedule="daily 08:00", agent_slug="bob-agent",
                             prompt="What is due today?", delivery_kind="agent", delivery={"chat": True})
    (data / "carol-agent").mkdir()
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)), base_url="http://fd.test")


def _hdr(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _export_and_upload(c: httpx.AsyncClient, uid: str = BOB) -> dict:
    made = await c.post("/fd/agent-bundle/export/bob-agent", json={"preset": "full"}, headers=_hdr(uid))
    assert made.status_code == 200, made.text
    got = await c.get(f"/fd/agent-bundle/exports/{made.json()['id']}/download", headers=_hdr(uid))
    assert got.status_code == 200
    assert got.headers["content-disposition"].endswith('.captain"') or ".captain" in got.headers["content-disposition"]
    up = await c.post("/fd/agent-bundle/upload", files={"file": ("bob.captain", got.content, "application/zip")},
                      headers=_hdr(uid))
    assert up.status_code == 200, up.text
    return {"made": made.json(), "bytes": got.content, "upload": up.json()}


# ── a full backup, restored as a new agent ──────────────────────────────────


@pytest.mark.asyncio
async def test_full_backup_restores_a_working_copy_of_the_agent(deck, tmp_path):
    async with _client() as c:
        out = await _export_and_upload(c)
        summary = out["upload"]
        assert summary["agent"]["name"] == "Bob's agent"
        assert summary["suggested_name"] == "Bob's agent"
        assert summary["counts"]["insights"] == 2 and summary["counts"]["chat_messages"] == 2
        assert summary["counts"]["schedules"] == 1 and summary["model"] == {"provider": "openai", "model": "gpt-x"}
        r = await c.post(f"/fd/agent-bundle/uploads/{summary['upload_id']}/import", json={"name": "Restored"},
                         headers=_hdr(BOB))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "restored" and body["chat_messages"] == 2 and body["schedules"] == 1
    assert any("no key for openai" in n for n in body["notes"])
    assert _FakePopen.calls == ["restored"]

    entry = server._load_process_registry()["restored"]
    assert entry["owner"] == BOB and entry["web_auth"] and entry["web_auth"] != "bob-tok"
    assert entry["archetype"] == "bookkeeper" and entry["model_tier"] == "balanced"
    new = server.DATA_DIR / "restored"
    hc = new / HC
    assert (new / ".fd-owner").read_text().strip() == BOB
    assert (hc / "personality.md").read_text() == "Calm and exact.\n"
    assert (hc / "cognitive_mode.txt").read_text() == "aeolian"
    assert (hc / "eco_mode.txt").read_text() == "off"
    assert (hc / "skills" / "vat-check" / "SKILL.md").is_file()
    assert (hc / "reflections" / "2026-11-01T10-00-00+00-00.md").is_file()
    assert (hc / "lessons.json").is_file() and (new / "practice.json").is_file()
    assert (new / "data" / "workspace" / "knowledge" / "handbook.md").is_file()
    assert (new / "data" / "workspace" / "Izvješće.md").read_text() == "Q3 report\n"
    # never: the owner's profile, the semantic index, the old log, a key or token
    assert not (hc / "tenant_context.md").exists()
    assert not (new / "data" / "home-config" / "memory.db").exists()
    assert not (new / "process.log").exists() or "log" not in (new / "process.log").read_text()
    assert "sk-bob-secret" not in (new / ".env").read_text()
    cfg = yaml.safe_load((new / "config.yaml").read_text())
    assert cfg["model"]["model"] == "gpt-x" and cfg["model"]["api_key"] == "" and cfg["model"]["vision"] is True
    assert cfg["context"]["max_tokens"] == 100000 and cfg["tools"]["enabled"] == ["read", "write", "insights", "google_mail"]
    assert cfg["web"]["auth_token"] == entry["web_auth"] and cfg["telegram"]["bot_token"] == ""

    # memory: the insights and their search index work in the restored store
    insights = InsightsManager(hc / "insights.db")
    try:
        assert await insights.count() == 2
        found = await insights.search("Zagreb")
        assert found and "books close" in found[0]["content"]
    finally:
        await insights.close()
    with sqlite3.connect(new / "data" / "sessions" / "sessions.db") as s:
        assert s.execute("SELECT credentials FROM apis").fetchone() == (None,)
        assert s.execute("SELECT count(*) FROM browser_credentials").fetchone() == (0,)
        assert s.execute("SELECT body FROM playbooks").fetchone() == ("Check the VAT first",)
    raw = (new / "data" / "sessions" / "sessions.db").read_bytes()
    assert b"TOPSECRET" not in raw

    # Flight Deck's side
    assert (await server._owner_agent_instructions(BOB, "process"))["restored"] == "Be brief."
    log = await deck.export_chat_log(BOB, "proc-restored")
    assert sorted(s["id"] for s in log["sessions"]) == ["proc-restored", "proc-restored::side"]
    assert {m["content"] for m in log["messages"]} == {"hi from proc-bob-agent", "hi from proc-bob-agent::side"}
    assert {m["created_at"] for m in log["messages"]} == {"2026-10-01T09:00:00Z"}
    jobs = [j for j in sched.get_store().list(BOB) if j["agent_slug"] == "restored"]
    assert len(jobs) == 1 and jobs[0]["enabled"] == 0 and jobs[0]["prompt"] == "What is due today?"
    # the upload is gone once the agent is made
    assert not any((export.bundles_dir() / "uploads").glob("*.captain"))


@pytest.mark.asyncio
async def test_the_bundle_never_holds_keys_tokens_or_the_owners_profile(deck):
    async with _client() as c:
        out = await _export_and_upload(c)
    with zipfile.ZipFile(io.BytesIO(out["bytes"])) as zf:
        names = set(zf.namelist())
        blob = b"".join(zf.read(n) for n in names)
    assert "manifest.json" in names and "fd/agent.json" in names and "stores/insights.db" in names
    assert not any(n.endswith((".env", ".fd-owner", "process.log", "memory.db", "tenant_context.md",
                               "config.yaml")) for n in names)
    for secret in (b"sk-bob-secret", b"bob-tok", b"tg-secret", b"TOPSECRET", b"bob@private.example"):
        assert secret not in blob


@pytest.mark.asyncio
async def test_a_store_snapshot_holds_what_is_still_in_the_wal(tmp_path):
    db = tmp_path / "live.db"
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE t (x TEXT)")
    writer.execute("INSERT INTO t VALUES ('in the wal')")
    writer.commit()
    try:
        plain = tmp_path / "plain-copy.db"
        shutil.copy2(db, plain)
        with sqlite3.connect(plain) as c:
            assert not c.execute("SELECT name FROM sqlite_master WHERE name = 't'").fetchone()   # a file copy misses it
        snap = tmp_path / "snap.db"
        stores.snapshot(db, snap)
        assert stores.counts(snap) == {"t": 1}
        assert not Path(str(snap) + "-wal").exists()
    finally:
        writer.close()


def test_keep_only_drops_tables_blanks_secrets_and_leaves_no_trace(tmp_path):
    db = tmp_path / "s.db"
    _sessions_db(db)
    stores.keep_only(db, {"sessions", "apis"}, blank={"apis": ("credentials",)})
    assert stores.counts(db) == {"apis": 1, "browser_credentials": 0, "contacts": 0, "playbooks": 0, "sessions": 1}
    assert b"TOPSECRET" not in db.read_bytes() and b"Check the VAT" not in db.read_bytes()


def test_a_store_with_a_view_or_that_is_no_database_is_refused(tmp_path):
    view = tmp_path / "v.db"
    with sqlite3.connect(view) as c:
        c.execute("CREATE TABLE t (x)")
        c.execute("CREATE VIEW v AS SELECT * FROM t")
    with pytest.raises(stores.StoreError, match="view"):
        stores.check(view)
    junk = tmp_path / "j.db"
    junk.write_bytes(b"not a database at all" * 100)
    with pytest.raises(stores.StoreError):
        stores.check(junk)


# ── reading a bundle trusts nothing ─────────────────────────────────────────


def _bundle(tmp_path: Path, files: dict[str, bytes], *, manifest_extra: dict | None = None,
            raw: dict[str, bytes] | None = None) -> Path:
    staging = tmp_path / "staging"
    shutil.rmtree(staging, ignore_errors=True)
    for name, data in files.items():
        (staging / name).parent.mkdir(parents=True, exist_ok=True)
        (staging / name).write_bytes(data)
    out = tmp_path / "b.captain"
    archive.write(staging, {"format": archive.FORMAT, "format_version": 1, **(manifest_extra or {})}, out)
    if raw:
        with zipfile.ZipFile(out, "a") as zf:
            for name, data in raw.items():
                zf.writestr(name, data)
    return out


def test_unsafe_names_are_refused():
    for bad in ("../x", "/etc/passwd", "a/../../b", "a\\b", "a//b", "./a", "a/\x01b", ""):
        assert not archive.safe_name(bad), bad
    for good in ("agent/data/workspace/Izvješće.md", "stores/insights.db", "agent/exams/2026 run (1).json"):
        assert archive.safe_name(good), good


def test_a_bundle_with_a_traversal_a_link_or_an_unlisted_file_is_refused(tmp_path):
    agent = json.dumps({"name": "X"}).encode()
    with pytest.raises(archive.BundleError, match="unsafe path"):
        archive.open_bundle(_bundle(tmp_path, {"fd/agent.json": agent}, raw={"../../evil.sh": b"x"}))
    with pytest.raises(archive.BundleError, match="doesn't list"):
        archive.open_bundle(_bundle(tmp_path, {"fd/agent.json": agent}, raw={"agent/extra.md": b"x"}))
    link = _bundle(tmp_path, {"fd/agent.json": agent})
    with zipfile.ZipFile(link, "a") as zf:
        info = zipfile.ZipInfo("agent/data/workspace/link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(archive.BundleError, match="link"):
        archive.open_bundle(link)
    with pytest.raises(archive.BundleError, match="newer Captain"):
        archive.open_bundle(_bundle(tmp_path, {"fd/agent.json": agent}, manifest_extra={"format_version": 99}))
    not_zip = tmp_path / "n.captain"
    not_zip.write_bytes(b"hello")
    with pytest.raises(archive.BundleError, match="isn't an agent bundle"):
        archive.open_bundle(not_zip)


def test_changed_bytes_are_caught_when_unpacked(tmp_path):
    path = _bundle(tmp_path, {"fd/agent.json": json.dumps({"name": "X"}).encode(),
                              f"agent/{HC}/personality.md": b"Calm.\n"})
    tampered = tmp_path / "t.captain"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.endswith("personality.md"):
                data = b"Evil.\n"          # same size, other bytes
            dst.writestr(info.filename, data)
    with pytest.raises(archive.BundleError, match="doesn't match"):
        restore.unpack(tampered, tmp_path / "agent")


def test_an_import_writes_only_the_agents_own_parts(tmp_path):
    path = _bundle(tmp_path, {
        "fd/agent.json": json.dumps({"name": "X"}).encode(),
        "agent/.fd-owner": b"user-mallory",
        "agent/.env": b"OPENAI_API_KEY=stolen",
        "agent/config.yaml": b"web: {auth_token: x}",
        f"agent/{HC}/tenant_context.md": b"someone else",
        f"agent/{HC}/personality.md": b"Calm.\n",
        "stores/evil.db": b"x",
    })
    target = tmp_path / "agent"
    target.mkdir()
    notes = restore.unpack(path, target)
    written = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert written == [f"{HC}/personality.md"]
    assert any("left out" in n for n in notes)


def test_a_link_inside_the_agents_folder_is_not_followed_on_import(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "agent"
    (target / "data").mkdir(parents=True)
    (target / "data" / "workspace").symlink_to(outside, target_is_directory=True)
    path = _bundle(tmp_path, {"fd/agent.json": b"{}", "agent/data/workspace/new/x.md": b"x"})
    restore.unpack(path, target)
    assert not any(outside.rglob("*"))


# ── whose it is ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bundles_and_uploads_are_their_owners_alone(deck):
    async with _client() as c:
        assert (await c.post("/fd/agent-bundle/export/bob-agent", json={}, headers=_hdr(CAROL))).status_code == 404
        out = await _export_and_upload(c)
        export_id, upload_id = out["made"]["id"], out["upload"]["upload_id"]
        assert (await c.get(f"/fd/agent-bundle/exports/{export_id}/download", headers=_hdr(CAROL))).status_code == 404
        r = await c.post(f"/fd/agent-bundle/uploads/{upload_id}/import", json={}, headers=_hdr(CAROL))
        assert r.status_code == 404
        assert (await c.delete(f"/fd/agent-bundle/uploads/{upload_id}", headers=_hdr(CAROL))).status_code == 404
        assert (await c.delete(f"/fd/agent-bundle/uploads/{upload_id}", headers=_hdr(BOB))).status_code == 200
        assert (await c.post("/fd/agent-bundle/uploads/not-an-id/import", json={}, headers=_hdr(BOB))).status_code == 404
    assert _FakePopen.calls == []


@pytest.mark.asyncio
async def test_an_upload_that_isnt_a_bundle_is_refused_and_not_kept(deck):
    async with _client() as c:
        r = await c.post("/fd/agent-bundle/upload", files={"file": ("x.captain", b"nope", "application/zip")},
                         headers=_hdr(BOB))
    assert r.status_code == 400 and "isn't an agent bundle" in r.json()["detail"]
    assert not any((export.bundles_dir() / "uploads").glob("*.captain"))


# ── phase 2: presets, review, sealed keys, model choice, someone else's import ──

from captain_claw.flight_deck import basna_routes  # noqa: E402


async def _export(c, body: dict, uid: str = BOB) -> bytes:
    made = await c.post("/fd/agent-bundle/export/bob-agent", json=body, headers=_hdr(uid))
    assert made.status_code == 200, made.text
    got = await c.get(f"/fd/agent-bundle/exports/{made.json()['id']}/download", headers=_hdr(uid))
    return got.content


async def _upload(c, data: bytes, uid: str = BOB) -> dict:
    up = await c.post("/fd/agent-bundle/upload", files={"file": ("x.captain", data, "application/zip")},
                      headers=_hdr(uid))
    assert up.status_code == 200, up.text
    return up.json()


def _names(data: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return set(zf.namelist())


def _store_rows(data: bytes, store: str, sql: str, tmp_path: Path) -> list:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        target = tmp_path / f"{store}.db"
        target.write_bytes(zf.read(f"stores/{store}.db"))
    with sqlite3.connect(target) as c:
        return c.execute(sql).fetchall()


@pytest.mark.asyncio
async def test_a_template_carries_who_it_is_and_what_it_was_taught_only(deck):
    async with _client() as c:
        data = await _export(c, {"preset": "template"})
    names = _names(data)
    assert f"agent/{HC}/personality.md" in names and f"agent/{HC}/skills/vat-check/SKILL.md" in names
    assert "agent/data/workspace/knowledge/handbook.md" in names and f"agent/{HC}/lessons.json" in names
    assert not any(n.startswith("stores/") for n in names)
    assert "agent/data/workspace/Izvješće.md" not in names and "fd/chats.json" not in names
    assert "fd/schedules.json" not in names and "fd/connections.json" in names


@pytest.mark.asyncio
async def test_trained_leaves_out_what_is_about_the_person_unless_ticked(deck, tmp_path):
    async with _client() as c:
        r = await c.get("/fd/agent-bundle/review/bob-agent", headers=_hdr(BOB))
        assert r.status_code == 200
        cats = {x["id"]: x for x in r.json()["insights"]}
        assert cats["workflow"]["default"] is True and cats["preference"]["default"] is False
        assert cats["preference"]["items"][0]["content"] == "Bob prefers short answers"
        assert r.json()["contacts"] == 1
        assert (await c.get("/fd/agent-bundle/review/bob-agent", headers=_hdr(CAROL))).status_code == 404
        defaults = [x["id"] for x in r.json()["insights"] if x["default"]]
        data = await _export(c, {"preset": "trained", "insight_categories": defaults})
        assert _store_rows(data, "insights", "SELECT content FROM insights", tmp_path) == [
            ("The client's books close on the 5th in Zagreb",)]
        tables = {t for (t,) in _store_rows(data, "sessions", "SELECT name FROM sqlite_master WHERE type='table'",
                                            tmp_path)}
        assert "playbooks" in tables
        assert _store_rows(data, "sessions", "SELECT count(*) FROM sessions", tmp_path) == [(0,)]
        assert _store_rows(data, "sessions", "SELECT count(*) FROM contacts", tmp_path) == [(0,)]
        assert b"ana@client.example" not in data and b"Bob prefers" not in data
        # one insight left out by hand, contacts ticked
        workflow_id = cats["workflow"]["items"][0]["id"]
        data = await _export(c, {"preset": "trained", "insight_categories": None, "exclude_insights": [workflow_id],
                                 "contacts": True})
        assert _store_rows(data, "insights", "SELECT content FROM insights", tmp_path) == [("Bob prefers short answers",)]
        assert _store_rows(data, "sessions", "SELECT name FROM contacts", tmp_path) == [("Ana",)]


@pytest.mark.asyncio
async def test_a_full_backup_seals_its_keys_with_a_passphrase(deck, monkeypatch):
    async def no_limit(*_a, **_k):
        return None

    monkeypatch.setattr(server, "check_agent_count_limit", no_limit)   # Bob makes two agents here
    pw = "correct horse battery"
    async with _client() as c:
        r = await c.post("/fd/agent-bundle/export/bob-agent", json={"preset": "trained", "passphrase": pw},
                         headers=_hdr(BOB))
        assert r.status_code == 400 and "Only a full backup" in r.json()["detail"]
        r = await c.post("/fd/agent-bundle/export/bob-agent", json={"preset": "full", "passphrase": "short"},
                         headers=_hdr(BOB))
        assert r.status_code == 400 and "at least 8" in r.json()["detail"]
        data = await _export(c, {"preset": "full", "passphrase": pw})
        assert "secrets.enc" in _names(data)
        assert b"sk-bob-secret" not in data and b"TOPSECRET-API-KEY" not in data
        up = await _upload(c, data)
        assert up["has_secrets"] is True and up["secrets"] is True
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import", json={"passphrase": "wrong one!"},
                         headers=_hdr(BOB))
        assert r.status_code == 400 and "passphrase" in r.json()["detail"]
        assert _FakePopen.calls == []
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"name": "Keyed", "passphrase": pw}, headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        assert r.json()["keys_restored"] is True and r.json()["needs_key"] is False
        new = server.DATA_DIR / "keyed"
        assert "OPENAI_API_KEY=sk-bob-secret" in (new / ".env").read_text()
        with sqlite3.connect(new / "data" / "sessions" / "sessions.db") as s:
            assert s.execute("SELECT credentials FROM apis").fetchone() == ("TOPSECRET-API-KEY",)
        # without the passphrase it comes without them, and says so
        up = await _upload(c, data)
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import", json={"name": "Keyless"},
                         headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        assert r.json()["keys_restored"] is False
        assert any("keys stayed in the file" in n for n in r.json()["notes"])
        assert "sk-bob-secret" not in (server.DATA_DIR / "keyless" / ".env").read_text()


@pytest.mark.asyncio
async def test_an_import_can_run_on_one_of_the_importers_tiers(deck, monkeypatch):
    async def tiers(_db, owner):
        assert owner == CAROL
        return ({"balanced": {"provider": "anthropic", "model": "claude-tier", "api_key": "carol-tier-key",
                              "input_ctx": 90000, "output_ctx": 8000}}, [{"key": "BRAVE_API_KEY", "value": "brave-c"}])

    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    async with _client() as c:
        data = await _export(c, {"preset": "trained"})
        up = await _upload(c, data, CAROL)
        assert up["model_tier"] == "balanced"
        r = await c.get("/fd/agent-bundle/tiers", headers=_hdr(CAROL))
        assert r.json()["tiers"] == [{"tier": "balanced", "provider": "anthropic", "model": "claude-tier"}]
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"name": "Books", "model": {"mode": "tier", "tier": "reason"}}, headers=_hdr(CAROL))
        assert r.status_code == 400 and "reason" in r.json()["detail"]
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"name": "Books", "model": {"mode": "tier", "tier": "balanced"}}, headers=_hdr(CAROL))
    assert r.status_code == 200, r.text
    body = r.json()
    cfg = yaml.safe_load((server.DATA_DIR / "books" / "config.yaml").read_text())
    assert cfg["model"]["provider"] == "anthropic" and cfg["model"]["model"] == "claude-tier"
    assert cfg["context"]["max_tokens"] == 90000 and cfg["model"]["max_tokens"] == 8000
    env = (server.DATA_DIR / "books" / ".env").read_text()
    assert "carol-tier-key" in env and "BRAVE_API_KEY=brave-c" in env
    entry = server._load_process_registry()["books"]
    assert entry["owner"] == CAROL and entry["model_tier"] == "balanced"
    # someone else's: the scheduled task delivers to the agent's chat only, and what to connect is listed
    jobs = [j for j in sched.get_store().list(CAROL) if j["agent_slug"] == "books"]
    assert len(jobs) == 1 and jobs[0]["delivery_kind"] == "agent" and jobs[0]["enabled"] == 0
    assert json.loads(jobs[0]["delivery"]) == {"chat": True, "email": False, "telegram": False, "email_to": ""}
    assert {"kind": "google", "label": "Google account (mail, calendar, Drive)"} in body["connections"]
    # the trained bundle had no conversations: no chat log for Carol
    assert (await deck.export_chat_log(CAROL, "proc-books"))["sessions"] == []


def test_sealed_keys_open_only_with_their_passphrase():
    from captain_claw.flight_deck.agent_bundle import secrets_box
    blob = secrets_box.seal({"env": {"A": "1"}}, "a long passphrase")
    assert secrets_box.open_sealed(blob, "a long passphrase") == {"env": {"A": "1"}}
    with pytest.raises(archive.BundleError, match="passphrase"):
        secrets_box.open_sealed(blob, "another passphrase")
    box = json.loads(blob)
    box["kdf"]["n"] = 2 ** 30
    with pytest.raises(archive.BundleError, match="doesn't read"):
        secrets_box.open_sealed(json.dumps(box).encode(), "a long passphrase")


# ── phase 3: into an agent that already exists — reset, append, undo ────────

from captain_claw.flight_deck.agent_bundle import merge  # noqa: E402


async def _carol_agent_with_memory() -> Path:
    hc = server.DATA_DIR / "carol-agent" / HC
    hc.mkdir(parents=True, exist_ok=True)
    (hc / "personality.md").write_text("Carol's own voice.\n")
    insights = InsightsManager(hc / "insights.db")
    await insights.add("Books close on the 5th, in Zagreb, for the client", "workflow", entity_key="books-close")
    await insights.add("Carol's office is in Split", "fact")
    await insights.close()
    return hc


@pytest.mark.asyncio
async def test_append_adds_what_the_agent_hasnt_and_keeps_its_own(deck, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug == "carol-agent")
    monkeypatch.setattr(server, "_do_stop_process", lambda slug: calls.append(f"stop {slug}"))
    monkeypatch.setattr(server, "_do_start_process", lambda slug: calls.append(f"start {slug}"))
    hc = await _carol_agent_with_memory()
    async with _client() as c:
        data = await _export(c, {"preset": "trained", "insight_categories": None})
        up = await _upload(c, data, CAROL)
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "bob-agent", "mode": "append"}},
                         headers=_hdr(CAROL))
        assert r.status_code == 404                     # not hers
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "carol-agent", "mode": "append"}},
                         headers=_hdr(CAROL))
        assert r.status_code == 200, r.text
        body = r.json()
        assert calls == ["stop carol-agent", "start carol-agent"] and body["restarted"] is True
        # Bob's "books close" has the same entity_key as Carol's: hers stays, his preference comes in
        assert body["counts"]["insights"] == {"added": 1, "skipped": 1}
        assert body["safety_backup"]["id"] and _FakePopen.calls == []
        got = await c.get(f"/fd/agent-bundle/safety/{body['safety_backup']['id']}/download", headers=_hdr(CAROL))
        assert got.status_code == 200 and "stores/insights.db" in _names(got.content)
        assert (await c.get(f"/fd/agent-bundle/safety/{body['safety_backup']['id']}/download",
                            headers=_hdr(BOB))).status_code == 404
    assert (hc / "personality.md").read_text() == "Carol's own voice.\n"            # hers wins in append
    assert (hc / "skills" / "vat-check" / "SKILL.md").is_file()                      # his skill added
    assert (await server._owner_agent_instructions(CAROL, "process"))["carol-agent"] == "Be brief."  # she had none
    insights = InsightsManager(hc / "insights.db")
    try:
        assert await insights.count() == 3
        assert (await insights.search("short answers"))[0]["content"] == "Bob prefers short answers"
    finally:
        await insights.close()


@pytest.mark.asyncio
async def test_reset_replaces_what_the_bundle_carries_and_undo_brings_it_back(deck):
    hc = server.DATA_DIR / "bob-agent" / HC
    async with _client() as c:
        backup = await _export(c, {"preset": "full"})
        # after the backup, Bob's agent learns something, chats on, and gets a new task
        insights = InsightsManager(hc / "insights.db")
        await insights.add("Invoices go out on Fridays", "workflow")
        await insights.close()
        await deck.upsert_chat_session("proc-bob-agent::later", BOB, "proc-bob-agent", "Bob's agent")
        sched.get_store().create(owner_id=BOB, name="Evening", schedule="daily 18:00", agent_slug="bob-agent",
                                 prompt="Wrap up", delivery_kind="agent", delivery={"chat": True})
        (server.DATA_DIR / "bob-agent" / "data" / "workspace" / "new-file.md").write_text("made later\n")
        up = await _upload(c, backup)
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "bob-agent", "mode": "reset"}},
                         headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        safety = r.json()["safety_backup"]["id"]
        insights = InsightsManager(hc / "insights.db")
        try:
            assert await insights.count() == 2 and not await insights.search("Fridays")
        finally:
            await insights.close()
        assert not (server.DATA_DIR / "bob-agent" / "data" / "workspace" / "new-file.md").exists()
        assert (server.DATA_DIR / "bob-agent" / "data" / "workspace" / "knowledge" / "handbook.md").is_file()
        log = await deck.export_chat_log(BOB, "proc-bob-agent")
        assert sorted(s["id"] for s in log["sessions"]) == ["proc-bob-agent", "proc-bob-agent::side"]
        jobs = [j for j in sched.get_store().list(BOB) if j["agent_slug"] == "bob-agent"]
        assert [j["name"] for j in jobs] == ["Morning books"] and jobs[0]["enabled"] == 0
        # undo: the agent as it was just before the import
        r = await c.post(f"/fd/agent-bundle/safety/{safety}/restore", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        assert r.json()["undone"] is True
    insights = InsightsManager(hc / "insights.db")
    try:
        assert await insights.count() == 3 and await insights.search("Fridays")
    finally:
        await insights.close()
    assert (server.DATA_DIR / "bob-agent" / "data" / "workspace" / "new-file.md").read_text() == "made later\n"
    log = await deck.export_chat_log(BOB, "proc-bob-agent")
    assert "proc-bob-agent::later" in {s["id"] for s in log["sessions"]}
    assert {j["name"] for j in sched.get_store().list(BOB) if j["agent_slug"] == "bob-agent"} == {
        "Morning books", "Evening"}


@pytest.mark.asyncio
async def test_append_puts_its_chat_log_beside_the_agents_own(deck):
    async with _client() as c:
        data = await _export(c, {"preset": "full"})
        up = await _upload(c, data)
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "bob-agent", "mode": "append"}},
                         headers=_hdr(BOB))
    assert r.status_code == 200, r.text
    ids = sorted(s["id"] for s in (await deck.export_chat_log(BOB, "proc-bob-agent"))["sessions"])
    lane = next(i for i in ids if i.startswith("proc-bob-agent::imported-") and not i.endswith("-side"))
    assert ids == sorted(["proc-bob-agent", "proc-bob-agent::side", lane, f"{lane}-side"])
    # the same task wasn't added twice
    assert [j["name"] for j in sched.get_store().list(BOB) if j["agent_slug"] == "bob-agent"] == ["Morning books"]
    assert r.json()["counts"]["sessions"] == {"added": 0, "skipped": 1}


def test_merge_store_appends_rows_without_taking_another_rows_id(tmp_path):
    def db(path: Path, rows: list[tuple]) -> Path:
        with sqlite3.connect(path) as c:
            c.executescript("CREATE TABLE log (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT);"
                            "CREATE TABLE playbooks (id TEXT PRIMARY KEY, name TEXT, body TEXT);")
            c.executemany("INSERT INTO log (id, text) VALUES (?, ?)", rows)
        return path
    mine = db(tmp_path / "mine.db", [(1, "mine one"), (2, "shared")])
    theirs = db(tmp_path / "theirs.db", [(1, "theirs one"), (2, "shared")])
    with sqlite3.connect(mine) as c:
        c.execute("INSERT INTO playbooks VALUES ('p1', 'Invoices', 'mine')")
    with sqlite3.connect(theirs) as c:
        c.execute("INSERT INTO playbooks VALUES ('p9', ' invoices ', 'theirs')")
        c.execute("INSERT INTO playbooks VALUES ('p2', 'Payroll', 'theirs')")
    got = merge.merge_store(theirs, mine, ["log", "playbooks"], "append")
    assert got == {"log": {"added": 1, "skipped": 1, "ids": {}},          # integer ids aren't tracked
                   "playbooks": {"added": 1, "skipped": 1, "ids": {"p2": ""}}}
    with sqlite3.connect(mine) as c:
        assert c.execute("SELECT id, text FROM log ORDER BY id").fetchall() == [
            (1, "mine one"), (2, "shared"), (3, "theirs one")]
        assert c.execute("SELECT name, body FROM playbooks ORDER BY name").fetchall() == [
            ("Invoices", "mine"), ("Payroll", "theirs")]
    got = merge.merge_store(theirs, mine, ["log"], "reset")
    with sqlite3.connect(mine) as c:
        assert c.execute("SELECT id, text FROM log ORDER BY id").fetchall() == [(1, "theirs one"), (2, "shared")]
        assert c.execute("SELECT count(*) FROM playbooks").fetchone() == (2,)     # not in the list: untouched


def test_merge_record_joins_lists_and_keeps_the_agents_own():
    mine = {"rules": [{"text": "Be exact"}], "facts": 3, "knowledge": [{"doc": "a.md"}]}
    theirs = {"rules": [{"text": " be  EXACT "}, {"text": "Cite the source"}], "facts": 9,
              "knowledge": [{"doc": "b.md"}], "glossary": [{"text": "VAT: tax"}]}
    out = merge.merge_record(mine, theirs)
    assert [r["text"] for r in out["rules"]] == ["Be exact", "Cite the source"]
    assert out["facts"] == 3 and [k["doc"] for k in out["knowledge"]] == ["a.md", "b.md"]
    assert out["glossary"] == [{"text": "VAT: tax"}]


@pytest.mark.asyncio
async def test_append_keeps_the_agents_own_reflection(deck):
    hc = await _carol_agent_with_memory()
    (hc / "reflections").mkdir()
    (hc / "reflections" / "2026-01-01T00-00-00+00-00.md").write_text("- Carol's way.\n")
    async with _client() as c:
        data = await _export(c, {"preset": "trained"})
        up = await _upload(c, data, CAROL)
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "carol-agent", "mode": "append"}},
                         headers=_hdr(CAROL))
    assert r.status_code == 200, r.text
    assert [p.name for p in (hc / "reflections").iterdir()] == ["2026-01-01T00-00-00+00-00.md"]


# ── phase 4: packages, versions, updates ────────────────────────────────────

from captain_claw.flight_deck.agent_bundle import provenance  # noqa: E402


def _sql(db: Path, sql: str, args: tuple = ()) -> list:
    with sqlite3.connect(db) as c:
        rows = c.execute(sql, args).fetchall()
        c.commit()
    return rows


@pytest.mark.asyncio
async def test_each_export_of_a_package_is_its_next_version(deck):
    async with _client() as c:
        a = (await c.post("/fd/agent-bundle/export/bob-agent", json={"preset": "trained", "package": "Bookkeeper"},
                          headers=_hdr(BOB))).json()["manifest"]["package"]
        b = (await c.post("/fd/agent-bundle/export/bob-agent", json={"preset": "trained", "package": " bookkeeper "},
                          headers=_hdr(BOB))).json()["manifest"]["package"]
        r = await c.get("/fd/agent-bundle/packages/bob-agent", headers=_hdr(BOB))
    assert a["version"] == 1 and b["version"] == 2 and a["id"] == b["id"]
    assert r.json()["packages"] == [{"name": "Bookkeeper", "version": 2}]


@pytest.mark.asyncio
async def test_a_client_agent_is_updated_without_losing_what_it_learned(deck, monkeypatch):
    async def no_limit(*_a, **_k):
        return None

    monkeypatch.setattr(server, "check_agent_count_limit", no_limit)
    bob = server.DATA_DIR / "bob-agent"
    (bob / ".owner-setup.json").write_text(json.dumps(
        {"rules": [{"text": "Two signatures over 1000 EUR", "always": True}], "skills": ["vat-check"]}))
    v1_body = {"preset": "trained", "package": "Bookkeeper", "insight_categories": None}
    async with _client() as c:
        # v1: the consultant passes it on; the client makes an agent of it
        up = await _upload(c, await _export(c, v1_body), CAROL)
        assert up["package"]["version"] == 1 and up["updates"] == []
        r = await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import", json={"name": "Books"},
                         headers=_hdr(CAROL))
        assert r.status_code == 200, r.text
        books = server.DATA_DIR / "books"
        rec = json.loads((books / ".owner-setup.json").read_text())
        assert rec["rules"][0]["layer"] == "base" and rec["base_skills"] == ["vat-check"]   # hers to add on, not change
        assert provenance.imported(books)[up["package"]["id"]]["version"] == 1

        # the client works with it: learns something, edits one imported insight, adds a rule of her own
        ins = books / HC / "insights.db"
        insights = InsightsManager(ins)
        await insights.add("Carol's client pays by card", "fact")
        await insights.close()
        (edited_id,) = _sql(ins, "SELECT id FROM insights WHERE entity_key = 'books-close'")[0]
        _sql(ins, "UPDATE insights SET content = 'Books close on the 6th now', updated_at = '2026-12-01' WHERE id = ?",
             (edited_id,))
        rec["rules"].append({"text": "Carol's own rule"})
        (books / ".owner-setup.json").write_text(json.dumps(rec))

        # v2: the consultant drops one insight, adds one, rewrites the skill and the instructions
        bins = bob / HC / "insights.db"
        _sql(bins, "DELETE FROM insights WHERE category = 'preference'")
        insights = InsightsManager(bins)
        await insights.add("Invoices over 5000 EUR need the director", "fact")
        await insights.close()
        (bob / HC / "skills" / "vat-check" / "SKILL.md").write_text("# VAT check v2\n")
        await server._set_owner_agent_instructions(BOB, "process", "bob-agent", "Be brief; cite the invoice.")
        up2 = await _upload(c, await _export(c, v1_body), CAROL)
        assert up2["package"]["version"] == 2
        assert up2["updates"] == [{"slug": "books", "name": "Books", "from_version": 1}]
        r = await c.post(f"/fd/agent-bundle/uploads/{up2['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "books", "mode": "update"}}, headers=_hdr(CAROL))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["from_version"] == 1 and body["package"]["version"] == 2
        assert body["taken_out"]["kept"] >= 1              # her edited insight

        contents = {row[0] for row in _sql(ins, "SELECT content FROM insights")}
        assert "Carol's client pays by card" in contents                # hers
        assert "Books close on the 6th now" in contents                 # her edit of an imported one
        assert "Invoices over 5000 EUR need the director" in contents   # new in v2
        assert "Bob prefers short answers" not in contents              # dropped in v2
        assert (books / HC / "skills" / "vat-check" / "SKILL.md").read_text() == "# VAT check v2\n"
        rules = [x["text"] for x in json.loads((books / ".owner-setup.json").read_text())["rules"]]
        assert sorted(rules) == ["Carol's own rule", "Two signatures over 1000 EUR"]
        assert (await server._owner_agent_instructions(CAROL, "process"))["books"] == "Be brief; cite the invoice."
        jobs = [j for j in sched.get_store().list(CAROL) if j["agent_slug"] == "books"]
        assert len(jobs) == 1 and jobs[0]["enabled"] == 0
        assert provenance.imported(books)[up["package"]["id"]]["version"] == 2

        # the same version again, and an update into an agent that never had the package
        up3 = await _upload(c, await _export(c, {**v1_body}), CAROL)
        r = await c.post(f"/fd/agent-bundle/uploads/{up3['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "carol-agent", "mode": "update"}},
                         headers=_hdr(CAROL))
        assert r.status_code == 400 and "no earlier version" in r.json()["detail"]


@pytest.mark.asyncio
async def test_an_update_leaves_instructions_the_client_changed(deck, monkeypatch):
    async def no_limit(*_a, **_k):
        return None

    monkeypatch.setattr(server, "check_agent_count_limit", no_limit)
    body = {"preset": "template", "package": "Bookkeeper"}
    async with _client() as c:
        up = await _upload(c, await _export(c, body), CAROL)
        assert (await c.post(f"/fd/agent-bundle/uploads/{up['upload_id']}/import", json={"name": "Books"},
                             headers=_hdr(CAROL))).status_code == 200
        await server._set_owner_agent_instructions(CAROL, "process", "books", "Carol's own words.")
        await server._set_owner_agent_instructions(BOB, "process", "bob-agent", "Version two words.")
        up2 = await _upload(c, await _export(c, body), CAROL)
        r = await c.post(f"/fd/agent-bundle/uploads/{up2['upload_id']}/import",
                         json={"target": {"kind": "existing", "slug": "books", "mode": "update"}}, headers=_hdr(CAROL))
    assert r.status_code == 200, r.text
    assert (await server._owner_agent_instructions(CAROL, "process"))["books"] == "Carol's own words."
    assert any("instructions were changed here" in n for n in r.json()["notes"])
