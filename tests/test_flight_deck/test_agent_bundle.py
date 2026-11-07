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
        "tools": {"enabled": ["read", "write", "insights"]},
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
    assert cfg["context"]["max_tokens"] == 100000 and cfg["tools"]["enabled"] == ["read", "write", "insights"]
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
    assert stores.counts(db) == {"apis": 1, "browser_credentials": 0, "playbooks": 0, "sessions": 1}
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
