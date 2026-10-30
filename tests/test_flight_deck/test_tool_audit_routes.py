"""Flight Deck's tool-call register: agents report every tool call; the owner
sees their agents' calls (a member's parameters hidden), a member their own,
admins everything; rows are chained so a change, an insert or a removal shows,
and an agent's numbering shows a record that never arrived."""

from __future__ import annotations

import asyncio
import csv
import io
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import auth
from captain_claw.flight_deck import tool_audit_routes as tr
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.tool_audit_store import ToolAuditStore

OWNER = {"id": "user-owner", "email": "o@x", "display_name": "Owen", "role": "user"}
MEMBER = {"id": "user-member", "email": "m@x", "display_name": "Mia", "role": "user"}
OTHER = {"id": "user-other", "email": "t@x", "display_name": "Tia", "role": "user"}
ADMIN = {"id": "user-admin", "email": "a@x", "display_name": "Ada", "role": "admin"}
AGENT = {"X-Agent-Auth": "tok-A"}
BOOT = "0123456789abcdef"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from captain_claw.flight_deck import agent_sharing, context_packs, server

    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    db = FlightDeckDB(tmp_path / "flight-deck.db")
    asyncio.run(db.init())
    auth.set_auth_db(db)
    store = ToolAuditStore(tmp_path / "tool_audit.db")
    tr.set_store(store)
    tr._identities.clear()
    agents = {"tok-A": (True, OWNER["id"], "helper")}
    monkeypatch.setattr(server, "_agent_caller_ok", lambda request: True)
    monkeypatch.setattr(server, "_find_agent_by_auth", lambda t: agents.get(t, (False, "", "")))
    monkeypatch.setattr(server, "_find_agent_endpoint_by_auth",
                        lambda t: {"port": 1, "agent_ref": "proc-helper", "name": "Helper"} if t in agents else None)
    monkeypatch.setattr(context_packs, "agent_ref_for_auth", lambda t: "process:helper:0123456789abcdef")
    members = {MEMBER["id"]}

    async def member_check(db_, ref, owner, uid, **kw):
        return ref == "process:helper:0123456789abcdef" and owner == OWNER["id"] and uid in members

    monkeypatch.setattr(agent_sharing, "member_check", member_check)
    app = FastAPI()
    app.include_router(tr.router)
    current = {"user": OWNER}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    yield TestClient(app), current, store, members
    asyncio.run(store.close())
    tr.set_store(None)
    asyncio.run(db.close())
    auth.set_auth_db(None)  # type: ignore[arg-type]


def _rec(seq, tool="shell", phase="full", **kw):
    rec = {"boot": BOOT, "seq": seq, "phase": phase, "started_at": time.time(), "tool": tool,
           "params": {"command": "ls"}, "source": "web", "source_label": "Web chat",
           "outcome": "ok", "duration_ms": 12, "ended_at": time.time()}
    rec.update(kw)
    return rec


def _send(client, *records, headers=AGENT):
    return client.post("/fd/tool-audit/agent/calls", json={"records": list(records)}, headers=headers)


def _calls(client, path="/fd/tool-audit/calls", **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()["calls"]


# ── Ingest ────────────────────────────────────────────────────────────


def test_only_this_decks_agents_can_report(env):
    client, *_ = env
    assert _send(client, _rec(1), headers={"X-Agent-Auth": "nope"}).status_code == 403
    assert _send(client, _rec(1), headers={}).status_code == 403
    r = client.post("/fd/tool-audit/agent/calls", json={"records": [_rec(1)]},
                    headers={**AGENT, "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 403
    assert _send(client, _rec(1, boot="not-hex")).status_code == 422


def test_identity_risk_and_masking_are_flight_decks_own(env):
    client, *_ = env
    r = _send(client, _rec(1, tool="google_mail", params={"action": "send", "to": "x@y", "api_key": "k",
                                                          "body": "token sk-proj-abcdefghijklmnopqrstuv"},
                           risk="read", owner_id="someone-else"))
    assert r.status_code == 200
    (c,) = _calls(client)
    assert c["agent"] == {"slug": "helper", "name": "Helper", "ref": "proc-helper"}
    assert c["owner"]["id"] == OWNER["id"] and c["owner"]["you"]
    assert c["risk"] == "send"
    assert c["params"]["api_key"] == "[masked]" and "sk-proj" not in c["params"]["body"]


def test_repeats_are_ignored_and_outcome_settles_once(env):
    client, *_ = env
    _send(client, _rec(1, phase="start", outcome=""))
    (c,) = _calls(client)
    assert c["outcome"] == "running"
    _send(client, _rec(1, phase="start", tool="other"))       # a repeat
    _send(client, {"boot": BOOT, "seq": 1, "phase": "end", "started_at": 0, "outcome": "blocked",
                   "reason": "nope", "duration_ms": 5, "ended_at": time.time()})
    _send(client, {"boot": BOOT, "seq": 1, "phase": "end", "started_at": 0, "outcome": "ok"})
    (c,) = _calls(client)
    assert (c["tool"], c["outcome"], c["reason"]) == ("shell", "blocked", "nope")


def test_end_without_its_start_still_fills_the_number(env):
    client, *_ = env
    _send(client, {"boot": BOOT, "seq": 3, "phase": "end", "started_at": time.time(), "outcome": "ok"})
    (c,) = _calls(client)
    assert c["tool"] == "(unknown)" and c["seq"] == 3


# ── Who sees what ─────────────────────────────────────────────────────


def test_owner_member_other_and_admin(env):
    client, current, _store, _members = env
    _send(client,
          _rec(1, params={"command": "owner's"}),
          _rec(2, tool="web_search", params={"query": "mia's secret plan"}, reason="",
               speaker_id=MEMBER["id"], speaker_name="Mia", source="member"),
          _rec(3, tool="web_search", params={"query": "fail"}, outcome="error", reason="mia's error",
               speaker_id=MEMBER["id"], speaker_name="Mia", source="member"))

    current["user"] = OWNER
    own = {c["seq"]: c for c in _calls(client)}
    assert set(own) == {1, 2, 3}
    assert own[1]["params"] == {"command": "owner's"} and not own[1]["hidden"]
    assert own[2]["hidden"] and own[2]["params"] is None
    assert own[3]["reason"] == tr.HIDDEN_NOTE
    assert own[2]["speaker"]["label"] == "Mia"

    current["user"] = MEMBER
    mine = {c["seq"]: c for c in _calls(client)}
    assert set(mine) == {2, 3}
    assert mine[2]["params"] == {"query": "mia's secret plan"} and mine[2]["speaker"]["you"]
    assert mine[3]["reason"] == "mia's error"

    current["user"] = OTHER
    assert _calls(client) == []

    current["user"] = ADMIN
    every = {c["seq"]: c for c in _calls(client, "/fd/admin/tool-audit/calls")}
    assert set(every) == {1, 2, 3} and every[2]["params"] == {"query": "mia's secret plan"}
    assert {c["seq"] for c in _calls(client, "/fd/admin/tool-audit/calls", user_id=MEMBER["id"])} == {2, 3}


def test_unverified_member_stays_hidden_from_the_owner(env):
    client, current, _store, members = env
    members.clear()
    _send(client, _rec(1, tool="web_fetch", params={"url": "https://x"},
                       speaker_id=MEMBER["id"], speaker_name="Mia", source="member"))
    (c,) = _calls(client)
    assert c["speaker"]["id"] == "" and "unverified" in c["speaker"]["label"]
    assert c["hidden"]
    current["user"] = MEMBER
    assert _calls(client) == []


def test_telegram_group_turn_is_the_owners(env):
    client, *_ = env
    _send(client, _rec(1, tool="web_search", params={"query": "q"}, speaker_id="tg-group:-100",
                       speaker_name="Team chat"))
    (c,) = _calls(client)
    assert not c["member"] and not c["hidden"] and c["speaker"]["label"] == "Team chat"


def test_admin_routes_are_admins_only(env):
    client, current, *_ = env
    current["user"] = OWNER
    for path in ("/fd/admin/tool-audit/calls", "/fd/admin/tool-audit/verify", "/fd/admin/tool-audit/settings",
                 "/fd/admin/tool-audit/events", "/fd/admin/tool-audit/export"):
        assert client.get(path).status_code == 403, path
    assert client.put("/fd/admin/tool-audit/settings", json={"retention_days": 30}).status_code == 403


def test_filters_and_paging(env):
    client, *_ = env
    _send(client, *[_rec(i, tool="read" if i % 2 else "shell", outcome="ok" if i % 3 else "blocked")
                    for i in range(1, 11)])
    assert {c["tool"] for c in _calls(client, tool="read")} == {"read"}
    assert {c["outcome"] for c in _calls(client, outcome="blocked")} == {"blocked"}
    assert {c["risk"] for c in _calls(client, risk="exec")} == {"exec"}
    first = client.get("/fd/tool-audit/calls", params={"limit": 4}).json()
    assert len(first["calls"]) == 4 and first["next_before"]
    rest = _calls(client, before_id=first["next_before"])
    assert len(rest) == 6
    assert client.get("/fd/tool-audit/calls", params={"risk": "nope"}).status_code == 400
    facets = client.get("/fd/tool-audit/facets").json()
    assert {t["tool"] for t in facets["tools"]} == {"read", "shell"}
    summary = client.get("/fd/tool-audit/summary").json()
    assert summary["total"] == 10 and summary["outcomes"]["blocked"] == 3


# ── Integrity ─────────────────────────────────────────────────────────


def _raw(store: ToolAuditStore, sql: str, *args):
    async def run():
        db = await store._ensure_db()
        await db.execute(sql, args)
        await db.commit()
    asyncio.run(run())


def test_verify_finds_changed_inserted_and_removed_rows(env):
    client, current, store, _ = env
    current["user"] = ADMIN
    _send(client, *[_rec(i) for i in range(1, 6)])
    assert client.get("/fd/admin/tool-audit/verify").json()["ok"]

    _raw(store, "UPDATE tool_calls SET params='{\"command\":\"innocent\"}' WHERE seq=2")
    _raw(store, "UPDATE tool_calls SET outcome='ok', reason='' WHERE seq=3")
    _raw(store, "DELETE FROM tool_calls WHERE seq=4")
    res = client.get("/fd/admin/tool-audit/verify").json()
    kinds = {(p["problem"]) for p in res["problems"]}
    assert not res["ok"] and {"row", "link"} <= kinds
    events = client.get("/fd/admin/tool-audit/events").json()["events"]
    assert events[0]["kind"] == "verified" and events[0]["detail"]["ok"] is False


def test_changed_outcome_is_found(env):
    client, current, store, _ = env
    current["user"] = ADMIN
    _send(client, _rec(1, outcome="blocked", reason="refused"))
    _raw(store, "UPDATE tool_calls SET outcome='ok', reason='' WHERE seq=1")
    res = client.get("/fd/admin/tool-audit/verify").json()
    assert [p["problem"] for p in res["problems"]] == ["outcome"]


def test_gaps_show_records_that_never_arrived(env):
    client, current, store, _ = env
    _send(client, _rec(1), _rec(2), _rec(5))
    gaps = asyncio.run(store.gaps(now=time.time() + 3600))
    assert [(g["agent_slug"], g["missing"]) for g in gaps] == [("helper", 2)]
    assert asyncio.run(store.gaps()) == []        # too fresh to tell yet


def test_retention_prunes_from_an_anchor_and_the_chain_still_verifies(env):
    client, current, store, _ = env
    current["user"] = ADMIN
    _send(client, _rec(1), _rec(2))
    assert client.put("/fd/admin/tool-audit/settings", json={"retention_days": 7}).json()["retention_days"] == 7
    assert client.put("/fd/admin/tool-audit/settings", json={"retention_days": 2}).status_code == 422
    assert asyncio.run(store.prune(now=time.time() + 8 * 86400)) == 2
    _send(client, _rec(3), _rec(4))
    res = client.get("/fd/admin/tool-audit/verify").json()
    assert res["ok"] and res["checked"] == 2 and res["anchor_id"] == 2
    kinds = [e["kind"] for e in client.get("/fd/admin/tool-audit/events").json()["events"]]
    assert "pruned" in kinds and "retention_changed" in kinds


# ── Export ────────────────────────────────────────────────────────────


def test_export_neutralises_formulas_and_hides_members_params(env):
    client, current, *_ = env
    _send(client, _rec(1, params={"command": "=HYPERLINK(\"x\")"}, reason="=cmd|' /C calc'!A0", outcome="error"),
          _rec(2, tool="web_search", params={"query": "mia's"}, speaker_id=MEMBER["id"], speaker_name="Mia",
               source="member"))
    r = client.get("/fd/tool-audit/export", params={"format": "csv"})
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    by = {row["seq"]: row for row in rows}
    assert by["1"]["reason"].startswith("'=")
    assert by["2"]["params"] == ""
    current["user"] = ADMIN
    r = client.get("/fd/admin/tool-audit/export", params={"format": "jsonl"})
    lines = [ln for ln in r.text.splitlines() if ln]
    assert len(lines) == 2 and "mia's" in r.text
    assert client.get("/fd/tool-audit/export", params={"format": "xml"}).status_code == 400


# ── Turn id and result refs (Taskboard) ───────────────────────────────

TURN = "ab" * 16


def test_turn_id_and_refs_are_stored_with_or_without_them(env):
    client, _current, store, _ = env
    assert _send(client,
                 _rec(1, tool="google_mail", params={"action": "send"}, turn_id=TURN,
                      refs={"gmail_message_id": "18c2f", "gmail_thread_id": "18c2e", "body": "not an id"}),
                 _rec(2, tool="google_mail", params={"action": "send"}, turn_id=TURN, outcome="error",
                      refs={"gmail_message_id": "18c30"}),
                 _rec(3)).status_code == 200                       # an older agent's record
    by = {c["seq"]: c for c in _calls(client)}
    assert by[1]["turn_id"] == TURN and by[1]["refs"] == {"gmail_message_id": "18c2f", "gmail_thread_id": "18c2e"}
    assert by[2]["refs"] == {}                                     # only an ok call made something
    assert by[3]["turn_id"] == "" and by[3]["refs"] == {}
    assert {r["seq"] for r in asyncio.run(store.list({"turn_id": TURN}))} == {1, 2}
    assert _send(client, _rec(4, turn_id="not-a-turn")).status_code == 422


def test_a_members_refs_are_hidden_from_the_owner(env):
    client, current, *_ = env
    _send(client, _rec(1, tool="whatsapp_web", params={"action": "send"}, speaker_id=MEMBER["id"],
                       speaker_name="Mia", source="member", turn_id=TURN, refs={"waweb_message_id": "3EB0"}))
    (c,) = _calls(client)
    assert c["hidden"] and c["refs"] is None and c["turn_id"] == TURN
    current["user"] = MEMBER
    (c,) = _calls(client)
    assert c["refs"] == {"waweb_message_id": "3EB0"}


def test_refs_settle_with_the_outcome_under_its_mac(env):
    client, current, store, _ = env
    current["user"] = ADMIN
    _send(client, _rec(1, phase="start", outcome="", turn_id=TURN))
    _send(client, {"boot": BOOT, "seq": 1, "phase": "end", "started_at": 0, "outcome": "ok",
                   "refs": {"calendar_event_id": "ev1"}, "duration_ms": 3, "ended_at": time.time()})
    (c,) = _calls(client, "/fd/admin/tool-audit/calls")
    assert (c["outcome"], c["refs"], c["turn_id"]) == ("ok", {"calendar_event_id": "ev1"}, TURN)
    assert client.get("/fd/admin/tool-audit/verify").json()["ok"]
    _raw(store, "UPDATE tool_calls SET refs='{\"calendar_event_id\":\"other\"}' WHERE seq=1")
    assert [p["problem"] for p in client.get("/fd/admin/tool-audit/verify").json()["problems"]] == ["outcome"]
    _raw(store, "UPDATE tool_calls SET refs='{\"calendar_event_id\":\"ev1\"}', turn_id='' WHERE seq=1")
    assert [p["problem"] for p in client.get("/fd/admin/tool-audit/verify").json()["problems"]] == ["row"]


def test_a_register_from_before_turn_ids_migrates_and_still_verifies(tmp_path):
    import hashlib
    import hmac
    import sqlite3

    from captain_claw.flight_deck import tool_audit_store as tas

    db_path = tmp_path / "tool_audit.db"
    old_ddl = tas._DDL.replace("    turn_id       TEXT NOT NULL DEFAULT '',\n", "").replace(
        "    refs          TEXT NOT NULL DEFAULT '{}',\n", "")
    assert "turn_id" not in old_ddl and "refs" not in old_ddl
    key = tas._load_key(db_path.with_suffix(".key"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript(old_ddl)
    prev = tas.GENESIS
    for seq, private in ((1, 0), (2, 1)):
        # Recorded by the register as it was: its MACs over the fields it had.
        row = {"owner_id": OWNER["id"], "member": 0, "speaker_id": "", "speaker_label": "", "agent_slug": "helper",
               "agent_name": "Helper", "agent_ref": "proc-helper", "boot": BOOT, "seq": seq, "started_at": 1.0,
               "received_at": 2.0, "tool": "shell", "risk": "exec", "params": "{}", "session_id": "",
               "source": "web", "source_label": "Web chat"}
        values = [row[f] for f in tas._CHAIN_FIELDS] + ([["private", 1]] if private else [])
        row_hash = hmac.new(key, prev.encode() + b"\n" + tas._canon(values), hashlib.sha256).hexdigest()
        end = ["ok", "", 5, 3.0]
        end_mac = hmac.new(key, row_hash.encode() + b"\n" + tas._canon(end), hashlib.sha256).hexdigest()
        cols = list(row) + ["private", "outcome", "reason", "duration_ms", "ended_at", "prev_hash", "hash", "end_mac"]
        conn.execute(f"INSERT INTO tool_calls ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [*row.values(), private, *end, prev, row_hash, end_mac])
        prev = row_hash
    conn.commit()
    conn.close()

    store = ToolAuditStore(db_path)

    async def run():
        assert (await store.verify())["ok"]
        db = await store._ensure_db()
        async with db.execute("PRAGMA index_list(tool_calls)") as cur:
            assert "idx_tool_calls_turn" in {r[1] for r in await cur.fetchall()}
        old = await store.list({})
        assert {(r["turn_id"], r["refs"]) for r in old} == {("", "{}")}
        await store.record([{
            "phase": "full", "owner_id": OWNER["id"], "member": 0, "speaker_id": "", "speaker_label": "",
            "agent_slug": "helper", "agent_name": "Helper", "agent_ref": "proc-helper", "boot": BOOT, "seq": 3,
            "started_at": 4.0, "tool": "google_mail", "risk": "send", "params": "{}", "session_id": "",
            "source": "web", "source_label": "Web chat", "outcome": "ok", "reason": "", "duration_ms": 1,
            "ended_at": 5.0, "private": 0, "turn_id": TURN, "refs": '{"gmail_message_id":"m1"}'}])
        res = await store.verify()
        assert res["ok"] and res["checked"] == 3
        (new,) = await store.list({"turn_id": TURN})
        assert new["refs"] == '{"gmail_message_id":"m1"}'
        await store.close()

    asyncio.run(run())
