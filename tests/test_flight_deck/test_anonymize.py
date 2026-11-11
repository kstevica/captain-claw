"""Anonymization in Flight Deck: vault store, agent and person routes, and the
agent tool end to end against the real routes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.anonymize import engine
from captain_claw.flight_deck import agent_secret, anonymize_routes, anonymize_vault, auth, fd_crypto
from captain_claw.flight_deck.db import FlightDeckDB

ALICE = "user-alice"
BOB = "user-bob"
LETTER = "Name: Klaus Müller\nIBAN: DE89370400440532013000\nmail k.mueller@beispiel.de, tel. +49 89 1234567"


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    fd_crypto.reset_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    store = anonymize_vault.VaultStore(tmp_path / "anonymize.db")
    anonymize_vault.set_store(store)
    try:
        yield d
    finally:
        anonymize_vault.set_store(None)
        await store.close()
        agent_secret.reset_cache_for_tests()
        fd_crypto.reset_for_tests()
        auth.set_auth_db(prev_db)
        await d.close()


@pytest.fixture()
def agents(monkeypatch):
    """This deck's agents: web_auth token → (owner, slug)."""
    import captain_claw.flight_deck.server as srv

    registry = {"tok-alice": (ALICE, "alice-bot"), "tok-bob": (BOB, "bob-bot")}

    def find(token):
        if token in registry:
            owner, slug = registry[token]
            return True, owner, slug
        return False, "", ""

    monkeypatch.setattr(srv, "_find_agent_by_auth", find)
    monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth", lambda t: find(t)[:2])
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {})
    return registry


def _app():
    app = FastAPI()
    app.include_router(anonymize_routes.router)
    return app


def _client():
    return TestClient(_app(), client=("127.0.0.1", 50123))


def _bearer(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


BOB_AGENT = {"X-Agent-Auth": "tok-bob"}
ALICE_AGENT = {"X-Agent-Auth": "tok-alice"}


def _db_bytes(tmp_path: Path) -> bytes:
    return b"".join(p.read_bytes() for p in tmp_path.glob("anonymize.db*"))


# ── agent routes ────────────────────────────────────────────────────


class TestAgentRun:
    async def test_tokens_out_values_encrypted_at_rest(self, db, agents, tmp_path):
        c = _client()
        r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                   json={"content": LETTER, "vault": "Case Müller", "source": "file"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["vault"]["name"] == "Case Müller"
        assert data["output"] == ("Name: ⟦PERSON_1⟧\nIBAN: ⟦IBAN_1⟧\nmail ⟦EMAIL_1⟧, tel. ⟦PHONE_1⟧")
        assert data["summary"] == {"PERSON": 1, "IBAN": 1, "EMAIL": 1, "PHONE": 1}
        assert all(e["new"] for e in data["entities"]) and data["verify"]["ok"]
        assert data["already_exposed"] is False
        # Nothing in the response but tokens; nothing in the database but ciphertext.
        assert "Müller" not in json.dumps({k: v for k, v in data.items() if k != "vault"}, ensure_ascii=False)
        raw = _db_bytes(tmp_path)
        for value in (b"k.mueller@beispiel.de", "Müller".encode(), b"DE89370400440532013000", b"1234567"):
            assert value not in raw

    async def test_same_vault_same_tokens_other_vault_independent(self, db, agents):
        c = _client()
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": LETTER, "vault": "A"})
        again = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                       json={"content": "Klaus Müller wrote from k.mueller@beispiel.de", "vault": "a"}).json()
        assert again["output"] == "⟦PERSON_1⟧ wrote from ⟦EMAIL_1⟧"  # vault names ignore case
        assert not any(e["new"] for e in again["entities"])
        other = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                       json={"content": "mail ana@x.hr", "vault": "B"}).json()
        assert other["output"] == "mail ⟦EMAIL_1⟧"

    async def test_one_way_keeps_nothing(self, db, agents):
        c = _client()
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": LETTER, "mode": "one_way", "filename": "Klaus_Muller_contract"}).json()
        assert data["vault"] is None and "⟦PERSON_1⟧" in data["output"]
        assert c.get("/fd/anonymize/agent/vaults", headers=BOB_AGENT).json() == {"vaults": []}

    async def test_formats_and_filename(self, db, agents):
        c = _client()
        content = json.dumps({"customer_name": "Ana Kovač", "email": "ana@x.hr", "total": 120})
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": content, "format": "json", "vault": "J", "filename": "Ana_Kovac_order"}).json()
        assert json.loads(data["output"]) == {"customer_name": "⟦PERSON_1⟧", "email": "⟦EMAIL_1⟧", "total": 120}
        assert data["filename"] == "PERSON_1_order"

    async def test_inline_text_is_marked_exposed(self, db, agents):
        data = _client().post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                              json={"content": "mail ana@x.hr", "source": "text"}).json()
        assert data["already_exposed"] is True and data["vault"]["name"] == "default"

    async def test_restore_and_unknown_tokens(self, db, agents):
        c = _client()
        out = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": LETTER, "vault": "R"}).json()
        r = c.post("/fd/anonymize/agent/restore", headers=BOB_AGENT,
                   json={"content": out["output"] + " ⟦ORG_7⟧", "vault": "R"})
        assert r.status_code == 200
        assert r.json()["output"] == LETTER + " ⟦ORG_7⟧"
        assert r.json()["restored"] == 4 and r.json()["unknown"] == ["⟦ORG_7⟧"]

    async def test_restore_json_keeps_it_valid(self, db, agents):
        c = _client()
        content = json.dumps({"company": 'Acme "Best" d.o.o.'})
        out = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                     json={"content": content, "format": "json", "vault": "Q"}).json()
        back = c.post("/fd/anonymize/agent/restore", headers=BOB_AGENT,
                      json={"content": out["output"], "format": "json", "vault": "Q"}).json()
        assert json.loads(back["output"]) == {"company": 'Acme "Best" d.o.o.'}

    async def test_vaults_are_per_person(self, db, agents):
        c = _client()
        c.post("/fd/anonymize/agent/run", headers=ALICE_AGENT, json={"content": LETTER, "vault": "Secret"})
        assert c.get("/fd/anonymize/agent/vaults", headers=BOB_AGENT).json() == {"vaults": []}
        r = c.post("/fd/anonymize/agent/restore", headers=BOB_AGENT,
                   json={"content": "⟦PERSON_1⟧", "vault": "Secret"})
        assert r.status_code == 404
        mine = c.get("/fd/anonymize/agent/vaults", headers=ALICE_AGENT).json()["vaults"]
        assert [v["name"] for v in mine] == ["Secret"] and "key_sealed" not in mine[0]

    async def test_member_turn_uses_the_members_vault(self, db, agents, monkeypatch):
        from captain_claw.flight_deck import speaker_grants

        async def acting(request, google=False):
            if request.headers.get("X-Test-Member"):
                return speaker_grants.ActingMember(user_id=BOB, owner=ALICE, agent_ref="p:alice-bot",
                                                   lane="web", turn="t1", slug="alice-bot", name="Alice Bot")
            return None

        monkeypatch.setattr(speaker_grants, "acting_member", acting)
        c = _client()
        r = c.post("/fd/anonymize/agent/run", headers={**ALICE_AGENT, "X-Test-Member": "1"},
                   json={"content": LETTER, "vault": "Bob's case"})
        assert r.status_code == 200
        assert c.get("/fd/anonymize/agent/vaults", headers=ALICE_AGENT).json() == {"vaults": []}
        assert [v["name"] for v in c.get("/fd/anonymize/agent/vaults", headers=BOB_AGENT).json()["vaults"]] == ["Bob's case"]

    async def test_who_may_call(self, db, agents):
        c = _client()
        assert c.post("/fd/anonymize/agent/run", json={"content": "x"}).status_code in (401, 403)
        assert c.post("/fd/anonymize/agent/run", headers={"X-Agent-Auth": "other-deck"},
                      json={"content": "x"}).status_code == 403
        browser = c.post("/fd/anonymize/agent/run", headers={**BOB_AGENT, "Origin": "https://evil.example"},
                         json={"content": "x"})
        assert browser.status_code == 403

    async def test_bad_requests(self, db, agents, monkeypatch):
        c = _client()
        assert c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={}).status_code == 422
        assert c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": "x", "kinds": ["SECRETS"]}).status_code == 422
        assert c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": "x", "mode": "maybe"}).status_code == 422
        r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                   json={"content": "{bad", "format": "json"})
        assert r.status_code == 422 and "bad" not in r.json()["detail"]
        monkeypatch.setattr(anonymize_routes, "MAX_CHARS", 10)
        assert c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": "x" * 11}).status_code == 413

    async def test_verify_failure_returns_nothing(self, db, agents, monkeypatch):
        def refuse(*a, **kw):
            raise engine.VerifyFailed({"PERSON": 1})

        monkeypatch.setattr(engine, "run", refuse)
        r = _client().post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": LETTER, "vault": "V"})
        assert r.status_code == 422
        assert "Müller" not in r.text and "output" not in r.json()

    async def test_without_crypto_only_one_way_works(self, db, agents, monkeypatch):
        monkeypatch.setattr(fd_crypto, "available", lambda: False)
        fd_crypto.reset_for_tests()
        c = _client()
        r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": LETTER, "vault": "X"})
        assert r.status_code == 503 and "one_way" in r.json()["detail"]
        ok = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": LETTER, "mode": "one_way"})
        assert ok.status_code == 200


# ── person routes ───────────────────────────────────────────────────


class TestPersonRoutes:
    async def test_vault_detail_events_rename_delete(self, db, agents):
        c = _client()
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": LETTER, "vault": "Case 1"})
        vaults = c.get("/fd/anonymize/vaults", headers=_bearer(BOB)).json()["vaults"]
        assert len(vaults) == 1 and vaults[0]["entity_count"] == 4 and vaults[0]["created_by"] == "agent:bob-bot"
        vid = vaults[0]["id"]

        detail = c.get(f"/fd/anonymize/vaults/{vid}", headers=_bearer(BOB)).json()
        originals = {e["token"]: e["original"] for e in detail["entries"]}
        assert originals["⟦PERSON_1⟧"] == "Klaus Müller" and originals["⟦IBAN_1⟧"] == "DE89370400440532013000"
        # Someone else's vault is not there for them, by id or by name.
        assert c.get(f"/fd/anonymize/vaults/{vid}", headers=_bearer(ALICE)).status_code == 404
        assert c.get("/fd/anonymize/vaults/Case 1", headers=_bearer(ALICE)).status_code == 404

        r = c.post("/fd/anonymize/restore", headers=_bearer(BOB),
                   json={"content": "⟦PERSON_1⟧ / ⟦EMAIL_1⟧", "vault": "Case 1"})
        assert r.json()["output"] == "Klaus Müller / k.mueller@beispiel.de"

        renamed = c.patch(f"/fd/anonymize/vaults/{vid}", headers=_bearer(BOB), json={"name": "Müller GmbH case"})
        assert renamed.json()["vault"]["name"] == "Müller GmbH case"

        events = c.get(f"/fd/anonymize/vaults/{vid}/events", headers=_bearer(BOB)).json()["events"]
        assert [e["action"] for e in events][::-1] == ["create", "anonymize", "view", "restore", "update"]
        assert "Müller" not in json.dumps([e["detail"] for e in events], ensure_ascii=False)

        assert c.delete(f"/fd/anonymize/vaults/{vid}", headers=_bearer(BOB)).json() == {"deleted": vid}
        assert c.get("/fd/anonymize/vaults", headers=_bearer(BOB)).json() == {"vaults": []}
        gone = c.post("/fd/anonymize/agent/restore", headers=BOB_AGENT,
                      json={"content": "⟦PERSON_1⟧", "vault": "Müller GmbH case"})
        assert gone.status_code == 404

    async def test_person_run_and_kinds(self, db, agents):
        c = _client()
        data = c.post("/fd/anonymize/run", headers=_bearer(ALICE),
                      json={"content": "IBAN HR1210010051863000160", "vault": "Mine"}).json()
        assert data["output"] == "IBAN ⟦IBAN_1⟧"
        kinds = c.get("/fd/anonymize/kinds", headers=_bearer(ALICE)).json()
        assert {"kind": "OIB", "label": "OIB"} in kinds["kinds"]
        assert c.post("/fd/anonymize/run", json={"content": "x"}).status_code == 401

    async def test_expired_vaults_are_deleted(self, db, agents):
        c = _client()
        c.post("/fd/anonymize/run", headers=_bearer(BOB), json={"content": LETTER, "vault": "Short"})
        vid = c.get("/fd/anonymize/vaults", headers=_bearer(BOB)).json()["vaults"][0]["id"]
        r = c.patch(f"/fd/anonymize/vaults/{vid}", headers=_bearer(BOB), json={"expires_at": time.time() - 1})
        assert r.status_code == 200
        assert c.get("/fd/anonymize/vaults", headers=_bearer(BOB)).json() == {"vaults": []}

    async def test_bad_names(self, db, agents):
        c = _client()
        r = c.post("/fd/anonymize/run", headers=_bearer(BOB), json={"content": "x", "vault": "n" * 81})
        assert r.status_code == 422
        c.post("/fd/anonymize/run", headers=_bearer(BOB), json={"content": "ana@x.hr", "vault": "One"})
        c.post("/fd/anonymize/run", headers=_bearer(BOB), json={"content": "ana@x.hr", "vault": "Two"})
        two = [v for v in c.get("/fd/anonymize/vaults", headers=_bearer(BOB)).json()["vaults"] if v["name"] == "Two"][0]
        clash = c.patch(f"/fd/anonymize/vaults/{two['id']}", headers=_bearer(BOB), json={"name": "one"})
        assert clash.status_code == 422


# ── the agent tool, end to end ──────────────────────────────────────


@pytest.fixture()
def tool_env(db, agents, monkeypatch, tmp_path):
    """The tool talks to the real routes through an in-process transport."""
    from captain_claw import fd_client

    app = _app()
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.ASGITransport(app=app, client=("127.0.0.1", 50123))
        kw.setdefault("base_url", "http://fd.test")
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "http://fd.test")
    monkeypatch.setattr(fd_client, "flight_deck_headers", lambda: dict(BOB_AGENT))
    work = tmp_path / "workspace"
    (work / "saved").mkdir(parents=True)
    return {"_runtime_base_path": str(work), "_saved_base_path": str(work / "saved"), "_session_id": "s-1"}


class TestTool:
    async def test_file_in_file_out_original_never_in_result(self, tool_env, tmp_path):
        from captain_claw.tools.anonymize import AnonymizeTool

        src = Path(tool_env["_runtime_base_path"]) / "Klaus_Muller_notes.txt"
        src.write_text(LETTER, encoding="utf-8")
        res = await AnonymizeTool().execute(action="anonymize", path="Klaus_Muller_notes.txt", **tool_env)
        assert res.success, res.error
        assert "Müller" not in res.content and "beispiel" not in res.content
        assert '1 person' in res.content and 'vault "chat-s-1"' in res.content
        saved = Path(res.content.split("Saved: ")[1].split("\n")[0])
        assert saved.name == "Klaus_Muller_notes.anonymized.txt" or saved.name.startswith("PERSON_1")
        assert saved.read_text(encoding="utf-8").startswith("Name: ⟦PERSON_1⟧")

        back = await AnonymizeTool().execute(action="restore", path=str(saved), **tool_env)
        assert back.success, back.error
        assert "Müller" not in back.content
        restored = Path(back.content.split("for the user: ")[1].split("\n")[0])
        assert restored.read_text(encoding="utf-8") == LETTER

    async def test_inline_text_says_it_was_seen(self, tool_env):
        from captain_claw.tools.anonymize import AnonymizeTool

        res = await AnonymizeTool().execute(action="anonymize", text="mail ana@x.hr", **tool_env)
        assert res.success and "mail ⟦EMAIL_1⟧" in res.content and "already seen the original" in res.content

    async def test_restore_refused_on_automated_turns(self, tool_env, monkeypatch):
        from captain_claw import mail_authority
        from captain_claw.tools.anonymize import AnonymizeTool

        monkeypatch.setattr(mail_authority, "current", lambda: mail_authority.Authority("automated", "cron"))
        res = await AnonymizeTool().execute(action="restore", text="⟦PERSON_1⟧", **tool_env)
        assert not res.success and "automated turn" in res.error

    async def test_secret_files_and_bad_calls(self, tool_env):
        from captain_claw.tools.anonymize import AnonymizeTool

        env = Path(tool_env["_runtime_base_path"]) / ".env"
        env.write_text("OPENAI_API_KEY=sk-x", encoding="utf-8")
        res = await AnonymizeTool().execute(action="anonymize", path=".env", **tool_env)
        assert not res.success and "Not read" in res.error
        both = await AnonymizeTool().execute(action="anonymize", text="a", url="https://x.example", **tool_env)
        assert not both.success and "exactly one" in both.error

    async def test_vaults_listing(self, tool_env):
        from captain_claw.tools.anonymize import AnonymizeTool

        await AnonymizeTool().execute(action="anonymize", text="mail ana@x.hr", vault="Case 9", **tool_env)
        res = await AnonymizeTool().execute(action="vaults", **tool_env)
        assert res.success and "Case 9 — 1 values (1 EMAIL)" in res.content

    async def test_without_a_deck(self, monkeypatch):
        from captain_claw import fd_client
        from captain_claw.tools.anonymize import AnonymizeTool

        monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")
        res = await AnonymizeTool().execute(action="anonymize", text="x")
        assert not res.success and "Flight Deck" in res.error
