"""Anonymization in Flight Deck: vault store, agent and person routes, and the
agent tool end to end against the real routes."""

from __future__ import annotations

import json
import re
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


# ── the trusted-model lane ──────────────────────────────────────────

BARE = "We met Marko yesterday; write to marko.k@x.hr. The only female surgeon on Vis joined."


@pytest.fixture()
def fake_model(monkeypatch):
    """A trusted model that finds what code can't; records what it was shown."""
    from captain_claw.flight_deck import anonymize_trusted as T

    seen: dict = {"prompts": [], "fail": None, "answer": None}

    def completer(m):
        async def call(system, user):
            seen["prompts"].append(user)
            if seen["fail"]:
                raise RuntimeError("down")
            if seen["answer"] is not None:
                return seen["answer"], 50
            spans = [{"text": t, "kind": k} for t, k in (
                ("Marko", "PERSON"), ("only female surgeon on Vis", "DETAIL"), ("Nobody Here", "PERSON"))
                if t in user]
            return json.dumps({"spans": spans}), 120
        return call

    monkeypatch.setattr(T, "completer", completer)
    return seen


def _set_model(c, **body):
    payload = {"provider": "ollama", "model": "qwen3:8b", "base_url": "http://127.0.0.1:11434", **body}
    return c.put("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin"), json=payload)


class TestTrustedModel:
    async def test_without_a_model_only_code_runs(self, db, agents):
        c = _client()
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE}).json()
        assert data["lane"] == "deterministic" and data["lanes"] == {"deterministic": "ok", "llm": "off"}
        assert "Marko yesterday" in data["output"]  # the bare name is the known gap
        r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "lane": "both"})
        assert r.status_code == 409 and "trusted model" in r.json()["detail"]
        assert c.get("/fd/anonymize/agent/status", headers=BOB_AGENT).json() == {
            "llm": False, "model": "", "default_lane": "deterministic", "name_model": ""}

    async def test_admin_trust_rules(self, db, agents):
        c = _client()
        assert _set_model(c).json()["model"]["basis"] == "local"
        cloud = {"provider": "anthropic", "model": "claude-x", "base_url": "", "api_key": "sk-secret-1"}
        r = c.put("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin"), json=cloud)
        assert r.status_code == 422 and "attested" in r.json()["detail"]
        ok = c.put("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin"), json={**cloud, "attested": True})
        model = ok.json()["model"]
        assert model["basis"] == "attested" and model["has_key"] and "sk-secret-1" not in ok.text
        got = c.get("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin")).json()
        assert got["model"]["label"] == "anthropic/claude-x" and "sk-secret" not in json.dumps(got)
        raw = await db.get_system_setting("anonymize:trusted_model")
        assert "sk-secret-1" not in raw  # sealed
        # An Ollama "-cloud" model runs at Ollama: not local.
        assert _set_model(c, model="gpt-oss:120b-cloud").status_code == 422
        assert c.put("/fd/anonymize/admin/trusted-model", headers=_bearer(BOB),
                     json={"provider": "ollama", "model": "x"}).status_code == 403
        assert c.delete("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin")).json() == {"model": None}

    async def test_both_lanes_model_sees_code_finds_as_tokens(self, db, agents, fake_model):
        c = _client()
        _set_model(c)
        assert c.get("/fd/anonymize/status", headers=_bearer(BOB)).json()["default_lane"] == "both"
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "vault": "L"}).json()
        assert data["lane"] == "both"
        assert data["output"] == "We met ⟦PERSON_1⟧ yesterday; write to ⟦EMAIL_1⟧. The ⟦DETAIL_1⟧ joined."
        assert data["lanes"]["llm"] == "ok" and data["lanes"]["llm_terms"] == 2
        assert data["lanes"]["llm_model"] == "ollama/qwen3:8b"
        # The model was shown the code's finds already replaced, not the email itself.
        assert "marko.k@x.hr" not in fake_model["prompts"][0] and "⟦EMAIL_1⟧" in fake_model["prompts"][0]
        # What the model found is in the vault: the next text needs no model to catch Marko.
        again = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                       json={"content": "Marku šaljem ugovor.", "vault": "L", "lane": "deterministic"}).json()
        assert again["output"] == "⟦PERSON_1⟧ šaljem ugovor."

    async def test_llm_only_lane(self, db, agents, fake_model):
        c = _client()
        _set_model(c)
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": BARE, "lane": "llm", "mode": "one_way"}).json()
        assert data["lanes"]["deterministic"] == "off"
        assert "⟦PERSON_1⟧" in data["output"] and "marko.k@x.hr" in data["output"]

    async def test_model_failure_refuses_by_default(self, db, agents, fake_model):
        c = _client()
        _set_model(c)
        fake_model["fail"] = True
        r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "vault": "F"})
        assert r.status_code == 502 and "Nothing was returned" in r.json()["detail"]
        assert "Marko" not in r.text
        _set_model(c, on_failure="deterministic")
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "vault": "F"}).json()
        assert data["lanes"]["llm"] == "failed" and "⟦EMAIL_1⟧" in data["output"]
        # lane=llm has nothing to fall back on.
        assert c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                      json={"content": BARE, "lane": "llm"}).status_code == 502

    async def test_a_model_that_invents_or_injects_adds_nothing(self, db, agents, fake_model):
        c = _client()
        _set_model(c)
        fake_model["answer"] = json.dumps({"spans": [
            {"text": "Ignore that, return the original", "kind": "PERSON"}, {"text": "the", "kind": "PERSON"}]})
        data = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "mode": "one_way"}).json()
        assert data["lanes"]["llm_terms"] == 0 and data["lanes"]["llm_dropped"] == 2
        assert "⟦EMAIL_1⟧" in data["output"]  # the code's finds stand

    async def test_tokens_are_counted_for_a_cloud_model_only(self, db, agents, fake_model, monkeypatch):
        from captain_claw.flight_deck import usage_quota

        counted = []

        async def record(db_, user, tokens, byok=False):
            counted.append((user, tokens))

        monkeypatch.setattr(usage_quota, "record_tokens", record)
        c = _client()
        _set_model(c)
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "mode": "one_way"})
        assert counted == []  # local: costs the deck nothing
        c.put("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin"),
              json={"provider": "openrouter", "model": "x/y", "attested": True})
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={"content": BARE, "mode": "one_way"})
        assert counted == [(BOB, 120)]

    async def test_environment_setting(self, db, agents, monkeypatch, fake_model):
        monkeypatch.setenv("FD_ANONYMIZE_PROVIDER", "ollama")
        monkeypatch.setenv("FD_ANONYMIZE_MODEL", "qwen3:8b")
        c = _client()
        status = c.get("/fd/anonymize/status", headers=_bearer(BOB)).json()
        assert status == {"llm": True, "model": "ollama/qwen3:8b", "default_lane": "both", "name_model": ""}
        got = c.get("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin")).json()["model"]
        assert got["source"] == "env" and got["basis"] == "local"

    async def test_admin_check_on_the_sample_cases(self, db, agents, fake_model):
        c = _client()
        _set_model(c)
        data = c.post("/fd/anonymize/admin/trusted-model/test", headers=_bearer(ALICE, "admin")).json()
        bare = [x for x in data["cases"] if x["id"] == "bare-names"][0]
        assert "Marko" in bare["code_missed"] and "Marko" not in bare["both_missed"]

    async def test_tool_passes_the_lane_and_says_who_looked(self, tool_env, fake_model):
        from captain_claw.tools.anonymize import AnonymizeTool

        c = _client()
        _set_model(c)
        res = await AnonymizeTool().execute(action="anonymize", text=BARE, **tool_env)
        assert res.success and "code + trusted model ollama/qwen3:8b (2 more found by it)" in res.content
        res = await AnonymizeTool().execute(action="anonymize", text=BARE, lane="deterministic", **tool_env)
        assert "code only" in res.content


# ── whole files (P2) ────────────────────────────────────────────────


def _docx_bytes() -> bytes:
    import io

    import docx

    doc = docx.Document()
    doc.core_properties.author = "Klaus Müller"
    p = doc.add_paragraph("Name: ")
    p.add_run("Klaus Müller").bold = True
    doc.add_paragraph("IBAN: DE89370400440532013000, mail k.mueller@beispiel.de")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _b64(blob: bytes) -> str:
    import base64

    return base64.b64encode(blob).decode("ascii")


class TestFiles:
    async def test_docx_in_docx_out_and_back(self, db, agents):
        import base64
        import io

        import docx

        c = _client()
        r = c.post("/fd/anonymize/file", headers=_bearer(BOB),
                   json={"filename": "Klaus_Muller_contract.docx", "data": _b64(_docx_bytes()), "vault": "K"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["filename"] == "PERSON_1_contract.anonymized.docx" or data["filename"].endswith(".anonymized.docx")
        assert data["content_type"].endswith("wordprocessingml.document")
        out = base64.b64decode(data["data"])
        doc = docx.Document(io.BytesIO(out))
        assert doc.paragraphs[0].text == "Name: ⟦PERSON_1⟧" and doc.paragraphs[0].runs[1].bold
        assert doc.core_properties.author == ""
        assert "Müller" not in data["preview"] and "⟦IBAN_1⟧" in data["preview"]
        back = c.post("/fd/anonymize/file/restore", headers=_bearer(BOB),
                      json={"filename": data["filename"], "data": data["data"], "vault": "K"}).json()
        assert back["filename"].endswith(".restored.docx") and back["restored"] == 3
        restored = docx.Document(io.BytesIO(base64.b64decode(back["data"])))
        assert restored.paragraphs[0].text == "Name: Klaus Müller"

    async def test_agent_file_route_and_text_kinds(self, db, agents):
        import base64

        c = _client()
        r = c.post("/fd/anonymize/agent/file", headers=BOB_AGENT,
                   json={"filename": "people.csv", "data": _b64("ime,email\nIvan,ivan@x.hr\n".encode()),
                         "mode": "one_way"})
        assert r.status_code == 200
        assert base64.b64decode(r.json()["data"]).decode() == "ime,email\n⟦PERSON_1⟧,⟦EMAIL_1⟧\n"
        page = c.post("/fd/anonymize/agent/file", headers=BOB_AGENT,
                      json={"filename": "p.html", "data": _b64(b"<p>Mail ana@x.hr</p>"), "mode": "one_way"}).json()
        assert page["filename"].endswith(".md") and any("markdown" in w for w in page["warnings"])

    async def test_bad_files(self, db, agents, monkeypatch):
        c = _client()
        bad = [({"filename": "x.exe", "data": _b64(b"MZ")}, 415),
               ({"filename": "x.txt", "data": "not base64!"}, 422),
               ({"filename": "x.docx", "data": _b64(b"not a zip")}, 422),
               ({"data": _b64(b"x")}, 422)]
        for body, status in bad:
            assert c.post("/fd/anonymize/file", headers=_bearer(BOB), json=body).status_code == status, body
        monkeypatch.setattr(anonymize_routes, "MAX_FILE_BYTES", 10)
        assert c.post("/fd/anonymize/file", headers=_bearer(BOB),
                      json={"filename": "x.txt", "data": _b64(b"x" * 100)}).status_code == 413

    async def test_tool_sends_a_docx_and_gets_a_docx(self, tool_env):
        import io

        import docx

        from captain_claw.tools.anonymize import AnonymizeTool

        src = Path(tool_env["_runtime_base_path"]) / "contract.docx"
        src.write_bytes(_docx_bytes())
        res = await AnonymizeTool().execute(action="anonymize", path="contract.docx", **tool_env)
        assert res.success, res.error
        assert "Müller" not in res.content and "beispiel" not in res.content
        saved = Path(res.content.split("Saved: ")[1].split("\n")[0])
        assert saved.suffix == ".docx"
        assert docx.Document(io.BytesIO(saved.read_bytes())).paragraphs[0].text == "Name: ⟦PERSON_1⟧"
        back = await AnonymizeTool().execute(action="restore", path=str(saved), **tool_env)
        restored = Path(back.content.split("for the user: ")[1].split("\n")[0])
        assert docx.Document(io.BytesIO(restored.read_bytes())).paragraphs[0].text == "Name: Klaus Müller"


# ── the privacy gate (P3) ───────────────────────────────────────────


def _gate(c, **body):
    return c.put("/fd/anonymize/gate", headers=_bearer(BOB),
                 json={"enabled": True, "groups": ["mail"], "when": "untrusted", **body})


def _decide(c, provider="anthropic", model="claude-x", base_url=""):
    return c.post("/fd/anonymize/agent/gate", headers=BOB_AGENT,
                  json={"provider": provider, "model": model, "base_url": base_url}).json()


class TestGateDecision:
    async def test_off_until_turned_on_and_settings_checked(self, db, agents):
        c = _client()
        got = c.get("/fd/anonymize/gate", headers=_bearer(BOB)).json()
        assert got["gate"]["enabled"] is False and got["vault"] == "Chat"
        assert {g["id"] for g in got["groups"]} >= {"mail", "chats", "files"}
        assert _decide(c) == {"active": False, "reason": "off"}
        assert c.put("/fd/anonymize/gate", headers=_bearer(BOB), json={"groups": ["spam"]}).status_code == 422
        assert c.put("/fd/anonymize/gate", headers=_bearer(BOB), json={"when": "sometimes"}).status_code == 422

    async def test_who_it_applies_to(self, db, agents):
        c = _client()
        _gate(c, lane="both")
        d = _decide(c)
        assert d["active"] and d["groups"] == ["mail"] and d["vault"] == "Chat"
        assert d["lane"] == "deterministic"  # no trusted model on this deck: code only
        assert _decide(c, "ollama", "qwen3:8b")["active"] is False  # local: trusted
        assert _decide(c, "ollama", "gpt-oss:120b-cloud")["active"] is True  # runs at Ollama
        c.put("/fd/anonymize/admin/trusted-model", headers=_bearer(ALICE, "admin"),
              json={"provider": "anthropic", "model": "claude-x", "attested": True})
        assert _decide(c)["active"] is False  # the admin's attested model
        _gate(c, when="always")
        assert _decide(c, "ollama", "qwen3:8b")["active"] is True
        # Alice's own setting is hers: Bob's gate doesn't touch her agent.
        alice = c.post("/fd/anonymize/agent/gate", headers=ALICE_AGENT, json={"provider": "anthropic", "model": "y"})
        assert alice.json()["active"] is False

    async def test_reveal_for_the_chat(self, db, agents):
        c = _client()
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
               json={"content": "mail ana@x.hr", "vault": "Chat", "source": "gate"})
        r = c.post("/fd/anonymize/reveal", headers=_bearer(BOB), json={"content": "Reply to ⟦EMAIL_1⟧"})
        assert r.json()["output"] == "Reply to ana@x.hr"
        assert c.post("/fd/anonymize/reveal", headers=_bearer(ALICE), json={"content": "x"}).status_code == 404


class _FakeTools:
    def __init__(self, result_text: str) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.result_text = result_text

    async def execute(self, name, arguments, **kw):
        from captain_claw.tools.registry import ToolResult

        self.calls.append((name, {k: v for k, v in arguments.items() if not k.startswith("_")}))
        return ToolResult(success=True, content=self.result_text)


def _agent(provider: str, model: str, result_text: str):
    import types

    from captain_claw.agent_guard_mixin import AgentGuardMixin

    class Agent(AgentGuardMixin):
        async def _enforce_guard(self, **kw):
            return True, ""

        def _enforce_blast_radius(self, *a):
            return True, ""

        def _current_session_slug(self):
            return "s-1"

    a = Agent()
    a.provider = types.SimpleNamespace(provider=provider, model=model, base_url="")
    a.tools = _FakeTools(result_text)
    return a


@pytest.fixture()
def gate_env(tool_env, monkeypatch):
    from captain_claw import anonymize_gate_client
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: "http://fd.test"))
    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_headers", staticmethod(lambda as_speaker=True: dict(BOB_AGENT)))
    anonymize_gate_client.reset_for_tests()
    yield tool_env
    anonymize_gate_client.reset_for_tests()


class TestGateOnTheAgent:
    MAIL = "From: Klaus Müller <k.mueller@beispiel.de>\nIBAN: DE89370400440532013000\nPlease call me."

    async def _run(self, agent, name, args):
        return await agent._execute_tool_with_guard(name, args, "test")

    async def test_mail_results_are_tokens_before_the_model(self, gate_env):
        _gate(_client())
        agent = _agent("anthropic", "claude-x", self.MAIL)
        res = await self._run(agent, "google_mail", {"action": "read_message", "id": "m1"})
        assert "Müller" not in res.content and "DE89370400440532013000" not in res.content
        assert "⟦IBAN_1⟧" in res.content and "[Privacy gate:" in res.content
        # A source the person didn't pick passes as it is.
        other = await self._run(agent, "web_fetch", {"url": "https://x.example"})
        assert other.content == self.MAIL

    async def test_outgoing_mail_gets_real_values_and_its_echo_is_gated(self, gate_env):
        _gate(_client())
        agent = _agent("anthropic", "claude-x", self.MAIL)
        first = await self._run(agent, "google_mail", {"action": "read_message", "id": "m1"})
        email_token = re.search(r"⟦EMAIL_\d+⟧", first.content).group(0)
        agent.tools.result_text = "Draft created for k.mueller@beispiel.de"
        res = await self._run(agent, "send_mail", {"to": email_token, "subject": "Re", "body": f"Dear {email_token}"})
        sent = agent.tools.calls[-1][1]
        assert sent["to"] == "k.mueller@beispiel.de" and sent["body"] == "Dear k.mueller@beispiel.de"
        assert "k.mueller@beispiel.de" not in res.content and email_token in res.content

    async def test_a_trusted_model_sees_the_real_thing(self, gate_env):
        _gate(_client())
        agent = _agent("ollama", "qwen3:8b", self.MAIL)
        res = await self._run(agent, "google_mail", {"action": "read_message", "id": "m1"})
        assert res.content == self.MAIL

    async def test_fails_closed(self, gate_env, monkeypatch):
        from captain_claw import anonymize_gate_client

        _gate(_client())
        agent = _agent("anthropic", "claude-x", self.MAIL)
        await anonymize_gate_client.decision(agent)  # the gate is known to be on

        async def down(path, body):
            raise anonymize_gate_client.GateFailed("deck unreachable")

        monkeypatch.setattr(anonymize_gate_client, "_post", down)
        res = await self._run(agent, "google_mail", {"action": "read_message", "id": "m1"})
        assert not res.success and "isn't shown" in res.error and "Müller" not in res.error
        sent = await self._run(agent, "send_mail", {"to": "⟦EMAIL_1⟧", "body": "x"})
        assert not sent.success and "Nothing was sent" in sent.error
        assert all(name != "send_mail" for name, _a in agent.tools.calls)

    async def test_folders_narrow_file_reads(self, gate_env):
        _gate(_client(), groups=["files"], folders=["vfs:clients"])
        agent = _agent("anthropic", "claude-x", self.MAIL)
        inside = await self._run(agent, "read", {"path": "vfs:clients/acme/notes.txt"})
        outside = await self._run(agent, "read", {"path": "vfs:public/readme.txt"})
        assert "Müller" not in inside.content and outside.content == self.MAIL

    async def test_the_anonymize_tool_uses_the_gate_vault_while_it_applies(self, gate_env):
        from captain_claw import anonymize_gate_client
        from captain_claw.tools.anonymize import default_vault

        _gate(_client())
        agent = _agent("anthropic", "claude-x", "")
        assert default_vault({"_agent": agent, "_session_id": "s-1"}) == "chat-s-1"
        await anonymize_gate_client.decision(agent)
        assert default_vault({"_agent": agent, "_session_id": "s-1"}) == "Chat"


class TestGateAndRecall:
    async def test_recall_keeps_raw_files_and_memories_out(self, gate_env):
        import types

        from captain_claw import anonymize_gate_client
        from captain_claw.agent_context_mixin import AgentContextMixin
        from captain_claw.agent_guard_mixin import AgentGuardMixin

        class Agent(AgentGuardMixin, AgentContextMixin):
            pass

        agent = Agent()
        agent.provider = types.SimpleNamespace(provider="anthropic", model="claude-x", base_url="")
        seen: dict = {}

        class Memory:
            def build_semantic_note(self, query, **kw):
                seen.update(kw)
                return "note", ""

        agent.memory = Memory()
        AgentContextMixin._build_semantic_memory_note(agent, "who is the client")
        assert seen["exclude_sources"] == frozenset()  # no gate: everything as before

        _gate(_client(), groups=["files", "memory"])
        await anonymize_gate_client.decision(agent)
        AgentContextMixin._build_semantic_memory_note(agent, "who is the client")
        assert seen["exclude_sources"] == frozenset({"workspace"})
        assert AgentContextMixin._build_deep_memory_note(agent, "who is the client") == ("", "")
        assert AgentContextMixin._build_external_memory_context_notes(agent) == ("", "")


def test_semantic_note_leaves_excluded_sources_out():
    import types

    from captain_claw.semantic_memory import SemanticMemoryIndex

    def hit(source, text):
        return types.SimpleNamespace(source=source, snippet=text, text_l1=text, text_l2=text, path=f"{source}.txt",
                                     start_line=1, updated_at="", score=1.0, reference=source,
                                     text_score=1.0, vector_score=0.0)

    fake = types.SimpleNamespace(
        search=lambda **kw: [hit("workspace", "Tomislav Babić, OIB 52601815906"), hit("session", "⟦PERSON_1⟧ asked")],
        cross_session_retrieval=False,
        _pick_layer_text=lambda layer, text, text_l1, text_l2: text,
    )
    note, _debug = SemanticMemoryIndex.build_context_note(fake, "client", exclude_sources=frozenset({"workspace"}))
    assert "Babić" not in note and "⟦PERSON_1⟧" in note


async def test_background_notes_from_covered_sources_stay_out(gate_env):
    import types

    from captain_claw import anonymize_gate_client
    from captain_claw.agent_context_mixin import AgentContextMixin

    class Agent(AgentContextMixin):
        pass

    agent = Agent()
    agent.provider = types.SimpleNamespace(provider="anthropic", model="claude-x", base_url="")
    _gate(_client(), groups=["contacts", "files"])
    await anonymize_gate_client.decision(agent)
    notes = [("contacts_context", "Ivan Horvat +385…"), ("workspace_manifest", "Ivan_Horvat_CV.pdf"),
             ("insights_context", "likes short replies"), ("datastore_context", "clients table")]
    # The collector's own filter, on what it would have returned:
    from captain_claw.anonymize.gate import NOTE_GROUPS

    gate = anonymize_gate_client.cached_decision(agent)
    blocked = {n for g in gate.groups for n in NOTE_GROUPS.get(g, ())}
    assert [n for n, _t in notes if n not in blocked] == ["insights_context", "datastore_context"]
    import inspect

    assert "NOTE_GROUPS" in inspect.getsource(AgentContextMixin._collect_background_context_notes)


# ── P7: the whole chat ──────────────────────────────────────────────


class TestWholeChat:
    MAIL = TestGateOnTheAgent.MAIL

    async def _run(self, agent, name, args, label="test"):
        return await agent._execute_tool_with_guard(name, args, label)

    async def test_setting_and_decision(self, db, agents):
        c = _client()
        assert c.put("/fd/anonymize/gate", headers=_bearer(BOB), json={"coverage": "all"}).status_code == 422
        _gate(c, coverage="chat")
        assert c.get("/fd/anonymize/gate", headers=_bearer(BOB)).json()["gate"]["coverage"] == "chat"
        d = _decide(c)
        assert d["active"] and d["chat"] is True and d["groups"] == [] and d["vault"] == "Chat"
        _gate(c)  # chosen sources again
        assert _decide(c)["chat"] is False

    async def test_every_tool_is_gated_and_gets_the_real_values(self, gate_env):
        _gate(_client(), coverage="chat")
        agent = _agent("anthropic", "claude-x", self.MAIL)
        res = await self._run(agent, "web_fetch", {"url": "https://x.example"})  # not a "source": gated anyway
        assert "Müller" not in res.content and "⟦IBAN_1⟧" in res.content
        assert "[Privacy gate:" not in res.content  # said once, in the system prompt
        email = re.search(r"⟦EMAIL_\d+⟧", res.content).group(0)
        agent.tools.result_text = "written"
        await self._run(agent, "write", {"path": "notes.txt", "content": f"Mail {email}"})
        assert agent.tools.calls[-1][1]["content"] == "Mail k.mueller@beispiel.de"  # files get real values
        await self._run(agent, "anonymize", {"action": "restore", "text": f"Mail {email}"})
        assert agent.tools.calls[-1][1]["text"] == f"Mail {email}"  # tokens are what it works with
        agent.tools.result_text = self.MAIL
        fd = await self._run(agent, "web_fetch", {"url": "https://x.example"}, label="autonomous-action")
        assert fd.content == self.MAIL  # its result goes to Flight Deck, not to the model

    async def test_what_reaches_the_model(self, gate_env, monkeypatch):
        import types

        from captain_claw import anonymize_gate_client as gate
        from captain_claw.llm import Message

        _gate(_client(), coverage="chat")
        agent = _agent("anthropic", "claude-x", "")
        agent.session = types.SimpleNamespace(messages=[
            {"role": "user", "content": "Write to Klaus Weber at klaus.weber@beispiel.de"},
            {"role": "tool", "content": "cron note: IBAN DE89370400440532013000"},
        ])
        calls = []
        real_post = gate._post

        async def counting(path, body):
            calls.append(path)
            return await real_post(path, body)

        monkeypatch.setattr(gate, "_post", counting)
        picture = object()
        messages = [Message(role="system", content="You help the owner, Ivan Horvat (ivan.horvat@acme.hr)."),
                    *(Message(role=m["role"], content=m["content"]) for m in agent.session.messages),
                    Message(role="user", content="And this picture", images=[picture])]
        out = await agent._gate_outgoing(messages)
        text = "\n".join(m.content for m in out[1:])
        for raw in ("Weber", "beispiel", "DE89370400440532013000"):
            assert raw not in text
        # The system prompt is the agent's own setup: as it is, with the gate's note.
        assert out[0].content == f"{messages[0].content}\n\n{gate.CHAT_NOTE}"
        # Where tokens are, the model is reminded what they are (not redactions).
        assert out[1].content.endswith(gate.CHAT_REMINDER) and out[2].content.endswith(gate.CHAT_REMINDER)
        assert "not redactions" in gate.CHAT_NOTE
        assert out[-1].images == [picture]  # pictures pass
        # The history keeps the tokens …
        assert "Weber" not in agent.session.messages[0]["content"] and "⟦" in agent.session.messages[1]["content"]
        # … and the same texts are never sent again (they read the same: the prompt cache holds).
        runs = [p for p in calls if p.endswith("/agent/run")]
        again = await agent._gate_outgoing(messages)
        assert [p for p in calls if p.endswith("/agent/run")] == runs
        assert [m.content for m in again] == [m.content for m in out]

    async def test_the_persons_message_first_and_fail_closed(self, gate_env, monkeypatch):
        from captain_claw import anonymize_gate_client as gate
        from captain_claw.agent import Agent
        from captain_claw.exceptions import GuardBlockedError
        from captain_claw.llm import Message

        _gate(_client(), coverage="chat")
        agent = _agent("anthropic", "claude-x", "")
        text, refused = await Agent._gate_user_text(agent, "Pozovi Klausa Webera, OIB 52601815906")
        assert not refused and "52601815906" not in text and "⟦OIB_1⟧" in text

        async def down(path, body):
            raise gate.GateFailed("deck unreachable")

        monkeypatch.setattr(gate, "_post", down)
        text, refused = await Agent._gate_user_text(agent, "Ana Kovač, ana@x.hr")
        assert refused.startswith("The privacy gate couldn't check your message")
        with pytest.raises(GuardBlockedError):
            await agent._gate_outgoing([Message(role="user", content="new text, ana@x.hr")])

    async def test_flight_decks_borrowed_calls_go_through_the_gate(self, gate_env, monkeypatch):
        """Consciousness, the arbiter, WhatsApp auto-replies… borrow the agent's
        model through /api/llm/complete: anonymized on the way in, the answer
        restored on the way out."""
        import types

        from captain_claw import anonymize_gate_client as gate
        from captain_claw.llm import LLMResponse
        from captain_claw.web_server import WebServer

        _gate(_client(), coverage="chat")
        seen: list = []

        class Provider:
            provider, model, base_url = "anthropic", "claude-x", ""

            async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
                seen.append(messages)
                tokens = re.findall(r"⟦[A-Z_]+_\d+⟧", messages[-1].content)
                return LLMResponse(content=f"Unread note from {tokens[0]} ({tokens[1]}).", model="m")

        class Req:
            def __init__(self, body):
                self._body = body

            async def json(self):
                return self._body

        server = WebServer.__new__(WebServer)
        server.agent = types.SimpleNamespace(provider=Provider(), session=None)
        monkeypatch.setattr(WebServer, "_scoped_provider", lambda self: self.agent.provider, raising=False)
        body = {"messages": [{"role": "system", "content": "You write one-line notes."},
                             {"role": "user", "content": "Klaus Weber (klaus.weber@beispiel.de) sent a note."}]}
        resp = await server._llm_complete(Req(body))
        data = json.loads(resp.text)
        sent = seen[-1]
        assert "Weber" not in sent[-1].content and sent[-1].content.endswith(gate.CHAT_REMINDER)
        assert sent[0].content.endswith(gate.CHAT_NOTE)
        assert data["ok"] and data["content"] == "Unread note from Klaus Weber (klaus.weber@beispiel.de)."

        async def down(path, payload):
            raise gate.GateFailed("deck unreachable")

        monkeypatch.setattr(gate, "_post", down)
        gate._CLEAN.clear()
        resp = await server._llm_complete(Req({"messages": [{"role": "user", "content": "Ana Kovač, ana@x.hr"}]}))
        assert resp.status == 503 and len(seen) == 1  # nothing went to the model

    async def test_flight_deck_restores_for_the_person_only(self, gate_env):
        from captain_claw.flight_deck import anonymize_chat

        anonymize_chat.reset_for_tests()
        c = _client()
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
               json={"content": "Klaus Weber, klaus.weber@beispiel.de", "vault": "Chat", "source": "gate"})
        frame = json.dumps({"type": "chat_message", "role": "assistant", "content": "Wrote to ⟦PERSON_1⟧ (⟦EMAIL_1⟧)."})
        bob = anonymize_chat.ChatRestorer(BOB)
        assert await bob.frame(frame) == frame  # the gate doesn't cover the whole chat: tokens
        _gate(c, coverage="chat")
        bob = anonymize_chat.ChatRestorer(BOB)
        assert json.loads(await bob.frame(frame))["content"] == "Wrote to Klaus Weber (klaus.weber@beispiel.de)."
        replay = json.dumps({"type": "replay_batch", "messages": [{"role": "assistant", "content": "Hi ⟦PERSON_1⟧"}]})
        assert json.loads(await bob.frame(replay))["messages"][0]["content"] == "Hi Klaus Weber"
        assert "\\u27e6" in frame  # the agent's frames escape non-ASCII: still found
        stream = json.dumps({"type": "response_stream", "text": "⟦PERS"})
        assert await bob.frame(stream) == stream  # deltas pass (a token may be split)
        assert await anonymize_chat.ChatRestorer(ALICE).frame(frame) == frame  # someone else: tokens
        assert await anonymize_chat.ChatRestorer("").frame(frame) == frame

    async def test_the_status_line_says_the_gate_is_working(self, gate_env):
        from captain_claw.agent import Agent
        from captain_claw.llm import Message

        _gate(_client(), coverage="chat")
        agent = _agent("anthropic", "claude-x", TestGateOnTheAgent.MAIL)
        said: list[str] = []
        agent._set_runtime_status = said.append
        await Agent._gate_user_text(agent, "Pozovi Klausa, OIB 52601815906")
        fetched = await self._run(agent, "web_fetch", {"url": "https://x.example"})
        email = re.search(r"⟦EMAIL_\d+⟧", fetched.content).group(0)
        await self._run(agent, "send_mail", {"to": email, "body": "hi"})
        note = Message(role="user", content="[Memory] Owner: Ivan Horvat, ivan.horvat@acme.hr")
        await agent._gate_outgoing([Message(role="system", content="You are Captain."), note])
        await agent._gate_outgoing([Message(role="system", content="You are Captain."), note])
        assert said[0] == "🛡️ Anonymizing your message — code is looking for private data…"
        assert "🛡️ Anonymizing what web_fetch returned — code is looking for private data…" in said
        assert "🛡️ Putting the real values back into send_mail…" in said
        assert said.count("🛡️ Anonymizing your message — code is looking for private data…") == 1
        assert [s for s in said if "before the model reads them" in s] == [
            "🛡️ Anonymizing 1 new text before the model reads them — code is looking for private data…"]  # once
        # Housekeeping after the turn stays quiet (the agent said it is idle).
        before = len(said)
        await agent._gate_outgoing([Message(role="user", content="Reflection on Ana Kovač, ana@x.hr")], "reflection")
        assert len(said) == before

    def test_the_status_names_the_trusted_model(self):
        from captain_claw import anonymize_gate_client as gate

        d = gate.Decision(True, lane="both", chat=True, model="ollama · ornith:9b")
        assert gate.working(d, "your message") == (
            "🛡️ Anonymizing your message — the trusted model (ollama · ornith:9b) is looking for private data…")

    def test_a_failure_says_what_went_wrong(self):
        from captain_claw import anonymize_gate_client as gate

        assert gate.why(httpx.ReadTimeout("")) == f"it took longer than {gate.CALL_TIMEOUT / 60:g} minutes"
        assert gate.why(httpx.ConnectError("")) == "Flight Deck couldn't be reached"
        assert gate.why(gate.GateFailed("")) == "GateFailed"
        assert gate.CALL_TIMEOUT >= 1800  # a local trusted model reads the system prompt in minutes

    async def test_the_trusted_model_call_gets_the_long_cap(self, monkeypatch):
        import types

        from captain_claw import llm
        from captain_claw.flight_deck import anonymize_trusted as trusted

        made = []

        class Provider:
            request_timeout = None

            async def complete(self, **kw):
                made.append(self.request_timeout)
                assert self.log_content is False  # its answer is the private data: never in a log line
                return types.SimpleNamespace(content="[]", usage={"total_tokens": 3})

        kwargs = {}

        def make(**kw):
            kwargs.update(kw)
            return Provider()

        monkeypatch.setattr(llm, "create_provider", make)
        monkeypatch.delenv("FD_ANONYMIZE_MODEL_THINK", raising=False)
        m = trusted.TrustedModel(provider="ollama", model="ornith:9b", base_url="", api_key="",
                                 attested=False, on_failure="refuse", enabled=True, source="settings")
        assert await trusted.completer(m)("system", "user") == ("[]", 3)
        assert made == [trusted.CALL_TIMEOUT] and trusted.CALL_TIMEOUT >= 600
        assert kwargs["max_tokens"] == trusted.MAX_TOKENS >= 32000
        assert kwargs["think"] is False  # off by default: finding names needs no thinking
        monkeypatch.setenv("FD_ANONYMIZE_MODEL_THINK", "model")
        await trusted.completer(m)("system", "user")
        assert kwargs["think"] is None  # the model's own
        monkeypatch.setenv("FD_ANONYMIZE_MODEL_THINK", "low")
        await trusted.completer(m)("system", "user")
        assert kwargs["think"] == "low"

    def test_a_provider_can_keep_model_output_out_of_its_logs(self):
        from captain_claw.llm import OllamaProvider

        p = OllamaProvider(model="ornith:9b")
        assert p._content_preview("Name: Ivan Horvat\nx", 10) == "Name: Ivan"
        p.log_content = False
        assert p._content_preview("Name: Ivan Horvat", 10) == ""

    async def test_an_answer_cut_at_the_limit_is_said_so(self, monkeypatch):
        import types

        from captain_claw import llm
        from captain_claw.anonymize.llm_lane import AnswerCut
        from captain_claw.flight_deck import anonymize_trusted as trusted

        class Provider:
            request_timeout = None

            async def complete(self, **kw):
                return types.SimpleNamespace(content='{"spans": [{"text": "Mar', usage={}, finish_reason="length")

        thinks = []

        def make(**kw):
            thinks.append(kw["think"])
            return Provider()

        monkeypatch.setattr(llm, "create_provider", make)
        m = trusted.TrustedModel(provider="ollama", model="ornith:9b")
        with pytest.raises(AnswerCut):
            await trusted.completer(m)("system", "user")
        assert thinks == [False]  # thinking already off: no second try here
        # With thinking on, a cut answer is asked again without it.
        monkeypatch.setenv("FD_ANONYMIZE_MODEL_THINK", "on")
        thinks.clear()
        with pytest.raises(AnswerCut):
            await trusted.completer(m)("system", "user")
        assert thinks == [True, False]

    async def test_thinking_off_reaches_an_openai_compatible_server(self, monkeypatch):
        """A Qwen on a local OpenAI-compatible server (llama.cpp, vLLM): "off" is
        chat_template_kwargs.enable_thinking=false in the request it gets."""
        import types

        from captain_claw import llm
        from captain_claw.flight_deck import anonymize_trusted as trusted

        sent = []

        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            sent.append(self._request_kwargs(messages, temperature=temperature, max_tokens=max_tokens))
            return types.SimpleNamespace(content='{"spans": []}', usage={}, finish_reason="stop")

        monkeypatch.setattr(llm.LiteLLMProvider, "complete", complete)
        monkeypatch.delenv("FD_ANONYMIZE_MODEL_THINK", raising=False)
        m = trusted.TrustedModel(provider="openai", model="qwen3.8-flash-next-iq3_s",
                                 base_url="http://192.168.1.50:8080/v1", api_key="x")
        await trusted.completer(m)("system", "user")
        assert sent[-1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
        monkeypatch.setenv("FD_ANONYMIZE_MODEL_THINK", "model")  # the model's own: nothing sent
        await trusted.completer(m)("system", "user")
        assert not (sent[-1].get("extra_body") or {}).get("chat_template_kwargs")

    async def test_without_thinking_the_retry_can_finish(self, monkeypatch):
        import types

        from captain_claw import llm
        from captain_claw.flight_deck import anonymize_trusted as trusted

        class Provider:
            request_timeout = None

            def __init__(self, think):
                self.think = think

            async def complete(self, **kw):
                if self.think:  # thinks until the limit
                    return types.SimpleNamespace(content="", usage={}, finish_reason="length")
                return types.SimpleNamespace(content='{"spans": []}', usage={"total_tokens": 9}, finish_reason="stop")

        monkeypatch.setattr(llm, "create_provider", lambda **kw: Provider(kw["think"]))
        monkeypatch.setenv("FD_ANONYMIZE_MODEL_THINK", "on")
        m = trusted.TrustedModel(provider="ollama", model="ornith:9b")
        assert await trusted.completer(m)("system", "user") == ('{"spans": []}', 9)

    async def test_telegram_restores_for_the_owners_chat(self, gate_env):
        from captain_claw import anonymize_gate_client as gate
        from captain_claw.web import telegram

        c = _client()
        agent = _agent("anthropic", "claude-x", "")
        _gate(c)  # chosen sources: no restoring
        await gate.decision(agent)
        assert telegram._tg_restorer(agent) is None
        gate.reset_for_tests()
        _gate(c, coverage="chat")
        await gate.decision(agent)
        c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
               json={"content": "Klaus Müller", "vault": "Chat", "source": "gate", "extra": [{"text": "Klaus Müller"}]})
        restore = telegram._tg_restorer(agent)
        assert await restore("Hi ⟦PERSON_1⟧") == "Hi Klaus Müller"


# ── P4: coarser values and the name-recognition pack ────────────────


class TestP4Routes:
    async def test_generalize_only_one_way(self, db, agents):
        c = _client()
        body = {"content": "Rođen 12. 3. 1985., plaća 4.237,50 EUR", "generalize": True}
        assert c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json=body).status_code == 422
        r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT, json={**body, "mode": "one_way"}).json()
        assert r["output"] == "Rođen 1985, plaća 2.000–5.000 EUR" and r["generalized"] == ["ADDRESS", "AGE", "AMOUNT", "DOB"]
        bad = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                     json={**body, "mode": "one_way", "generalize": ["SALARY"]})
        assert bad.status_code == 422

    async def test_admin_name_pack(self, db, agents, monkeypatch):
        import asyncio as _asyncio

        from captain_claw.anonymize import detectors
        from captain_claw.flight_deck import anonymize_ner

        c = _client()
        admin = _bearer(ALICE, "admin")
        monkeypatch.setattr(anonymize_ner, "runtime_available", lambda: False)
        got = c.get("/fd/anonymize/admin/ner", headers=admin).json()
        assert got["status"] == "off" and {m["id"] for m in got["models"]} == set(anonymize_ner.MODELS)
        assert c.put("/fd/anonymize/admin/ner", headers=admin,
                     json={"enabled": True, "model": "someone/else"}).status_code == 422
        on = c.put("/fd/anonymize/admin/ner", headers=admin, json={"enabled": True}).json()
        assert on["status"] == "not_installed"
        assert c.post("/fd/anonymize/admin/ner/download", headers=admin).status_code == 409
        assert c.get("/fd/anonymize/admin/ner", headers=_bearer(BOB)).status_code == 403

        # With the runtime and a (stand-in) model on disk it loads and runs.
        monkeypatch.setattr(anonymize_ner, "runtime_available", lambda: True)
        monkeypatch.setattr(anonymize_ner, "downloaded", lambda model: True)

        def fake_pipe(text):
            i = text.find("Marko")
            return [{"start": i, "end": i + 5, "entity_group": "PER", "score": 0.99}] if i >= 0 else []

        monkeypatch.setattr(anonymize_ner, "_load_pipeline", lambda model: fake_pipe)
        anonymize_ner._state.update(status="off", model="", error="")
        try:
            await anonymize_ner._bring_up(anonymize_ner.DEFAULT_MODEL, download=False)
            assert c.get("/fd/anonymize/admin/ner", headers=admin).json()["status"] == "ready"
            r = c.post("/fd/anonymize/agent/run", headers=BOB_AGENT,
                       json={"content": "We met Marko yesterday.", "mode": "one_way"}).json()
            assert r["output"] == "We met ⟦PERSON_1⟧ yesterday." and r["name_model"] == anonymize_ner.DEFAULT_MODEL
            assert c.get("/fd/anonymize/status", headers=_bearer(BOB)).json()["name_model"]
        finally:
            detectors.set_ner_backend(None)
            anonymize_ner._state.update(status="off", model="", error="")
        await _asyncio.sleep(0)


# ── shared agents: a member uses the tool as themselves ─────────────


class TestMemberTurns:
    GRANT = "g" * 43

    @pytest.fixture()
    def member(self, tool_env, monkeypatch, tmp_path):
        from captain_claw import speaker
        from captain_claw.flight_deck import speaker_grants

        seen: list[dict] = []

        async def acting(request, google=False):
            if request.headers.get(speaker.GRANT_HEADER) == self.GRANT and request.query_params.get("fd_member") == "1":
                seen.append(dict(request.headers))
                return speaker_grants.ActingMember(user_id=ALICE, owner=BOB, agent_ref="process:bob-bot:1",
                                                   lane="A", turn="t1", slug="bob-bot", name="Bob's agent")
            return None

        monkeypatch.setattr(speaker_grants, "acting_member", acting)
        monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
        p = speaker.Principal(ALICE, "Alice", "Bob", "A", "process:bob-bot:0123456789abcdef")
        t1, t2 = speaker.bind(p), speaker.bind_grant(self.GRANT)
        try:
            yield {"env": tool_env, "seen": seen, "vfs": tmp_path / "vfs"}
        finally:
            speaker.reset_grant(t2)
            speaker.reset(t1)

    async def test_a_members_text_goes_to_their_vault(self, member):
        from captain_claw.tools.anonymize import AnonymizeTool

        res = await AnonymizeTool().execute(action="anonymize", text="mail ana@x.hr", vault="Mine",
                                            **member["env"])
        assert res.success and "⟦EMAIL_1⟧" in res.content and member["seen"]
        c = _client()
        assert [v["name"] for v in c.get("/fd/anonymize/vaults", headers=_bearer(ALICE)).json()["vaults"]] == ["Mine"]
        assert c.get("/fd/anonymize/vaults", headers=_bearer(BOB)).json()["vaults"] == []  # never the owner's

    async def test_a_members_files_land_in_their_own_folder(self, member):
        from captain_claw.tools.anonymize import AnonymizeTool

        src = Path(member["env"]["_saved_base_path"]) / "downloads" / "notes.txt"  # a member reads saved/ or VFS
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(LETTER, encoding="utf-8")
        res = await AnonymizeTool().execute(action="anonymize", path=str(src), **member["env"])
        assert res.success, res.error
        shown = res.content.split("Saved: ")[1].split("\n")[0]
        assert shown.startswith("vfs:") and "/anonymize/anonymized/" in shown
        assert list((member["vfs"] / ALICE).rglob("*.anonymized.txt"))
        assert not list(Path(member["env"]["_saved_base_path"]).rglob("*.anonymized.txt"))  # not the commons

    async def test_a_members_page_fetch_takes_the_public_only_path(self, member, monkeypatch):
        from captain_claw.tools import web_fetch
        from captain_claw.tools.anonymize import AnonymizeTool

        asked = []

        async def member_fetch(url):
            asked.append(url)
            raise PermissionError("private address")

        monkeypatch.setattr(web_fetch, "_member_fetch", member_fetch)
        res = await AnonymizeTool().execute(action="anonymize", url="http://intranet.example/hr", **member["env"])
        assert asked == ["http://intranet.example/hr"]
        assert not res.success and "shared agent" in res.error

    async def test_no_grant_no_call(self, tool_env):
        from captain_claw import speaker
        from captain_claw.tools.anonymize import AnonymizeTool

        p = speaker.Principal(ALICE, "Alice", "Bob", "A", "process:bob-bot:0123456789abcdef")
        t = speaker.bind(p)
        try:
            res = await AnonymizeTool().execute(action="anonymize", text="mail ana@x.hr", **tool_env)
        finally:
            speaker.reset(t)
        assert not res.success and "member's own message" in res.error


def test_members_may_use_the_tool_under_rules():
    from captain_claw import speaker

    assert "anonymize" in speaker.SPEAKER_TOOL_ALLOWLIST_MAX
    assert "anonymize" not in speaker.SPEAKER_TOOL_ALLOWLIST  # process agents only (their own vaults)
    assert speaker.apply_tool_rules("anonymize", {"action": "vaults"}, None)[1] is None
    assert speaker.apply_tool_rules("anonymize", {"action": "delete_everything"}, None)[1] == speaker.NOT_ALLOWED_MESSAGE
    _args, err = speaker.apply_tool_rules("anonymize", {"action": "anonymize", "url": "file:///etc/passwd"}, None)
    assert err
    assert {r.pointer for r in speaker.SPEAKER_PATH_MAP["anonymize"]} == {"/path", "/output_path"}
