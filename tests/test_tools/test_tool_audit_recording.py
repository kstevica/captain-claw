"""The agent side of the tool-call register (captain_claw/tool_audit.py): every
call through ToolRegistry.execute is spooled — masked — with how it ended, and
shipped to Flight Deck in batches."""

from __future__ import annotations

import asyncio
import json

import pytest

from captain_claw import tool_audit
from captain_claw.exceptions import ToolBlockedError, ToolExecutionError, ToolNotFoundError
from captain_claw.tools.registry import Tool, ToolPolicy, ToolRegistry, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echo"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        if kwargs.get("fail"):
            return ToolResult(success=False, error="it went wrong")
        if kwargs.get("explode"):
            raise RuntimeError("boom")
        return ToolResult(success=True, content="ok")


class SlowTool(Tool):
    name = "slow"
    description = "Slow"
    parameters = {"type": "object", "properties": {}, "required": []}
    timeout_seconds = 1.0

    async def execute(self, **kwargs):
        await asyncio.sleep(3.0)
        return ToolResult(success=True)


@pytest.fixture()
def spool(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("FD_AGENT_SLUG", "tester")
    monkeypatch.setenv("CLAW_TOOL_AUDIT_SPOOL", str(tmp_path / "spool.db"))
    monkeypatch.delenv("CLAW_TOOL_AUDIT", raising=False)
    monkeypatch.setattr("captain_claw.fd_client._agent_web_auth_token", lambda: "tok")
    monkeypatch.setattr(tool_audit, "_SPOOL", None)
    monkeypatch.setattr(tool_audit, "_SPOOL_FAILED", False)
    monkeypatch.setattr(tool_audit, "_ensure_shipper", lambda: None)
    yield tool_audit
    if tool_audit._SPOOL is not None and tool_audit._SPOOL._conn is not None:
        tool_audit._SPOOL._conn.close()
    tool_audit._SPOOL = None


def _ended(t) -> list[dict]:
    records, _done, _started = t._spool().batch(t._BOOT, 1e12)
    return records


# ── Masking ───────────────────────────────────────────────────────────


def test_mask_params_drops_runtime_keys_and_masks_secrets():
    out = tool_audit.mask_params({
        "_agent": object(), "_session": "whole conversation",
        "command": 'curl -H "Authorization: Bearer abcdefghijklmnop1234" https://me:hunter2@host/x?token=abc&q=1',
        "api_key": "anything", "accessToken": "t", "author": "Ana", "max_tokens": 400,
        "nested": {"client_secret": "s", "items": ["ghp_abcdefghijklmnopqrstuvwxyz12", "plain"]},
        "card": "pay with 4111 1111 1111 1111 please",
        "key": "sk-ant-api03-abcdefghijklmnopqrstuv",
    })
    assert "_agent" not in out and "_session" not in out
    assert "hunter2" not in out["command"] and "abcdefghijklmnop1234" not in out["command"]
    assert "token=[masked]" in out["command"] and "q=1" in out["command"]
    assert out["api_key"] == out["accessToken"] == "[masked]"
    assert out["author"] == "Ana" and out["max_tokens"] == 400
    assert out["nested"]["client_secret"] == "[masked]"
    assert out["nested"]["items"] == ["[masked]", "plain"]
    assert "4111" not in out["card"]
    assert out["key"] == "[masked]"


def test_mask_params_keeps_only_the_start_of_long_values():
    out = tool_audit.mask_params({"content": "x" * 5000})
    assert out["content"]["_len"] == 5000
    assert len(out["content"]["_head"]) == 300
    assert len(out["content"]["_sha256"]) == 64


def test_mask_tree_is_idempotent_and_caps_size():
    once = tool_audit.mask_params({"password": "p", "content": "y" * 9000, "a": "sk-proj-abcdefghijklmnopqrstu"})
    assert tool_audit.mask_tree(once) == once
    big = tool_audit.mask_params({f"k{i}": "z" * 1500 for i in range(40)})
    assert big["_truncated"] is True and len(json.dumps(big)) < 20_000


def test_off_outside_flight_deck(tmp_path, monkeypatch):
    monkeypatch.delenv("FD_URL", raising=False)
    monkeypatch.setenv("CLAW_TOOL_AUDIT_SPOOL", str(tmp_path / "s.db"))
    assert tool_audit.start("echo", {}) is None
    assert not (tmp_path / "s.db").exists()


# ── Recording through the registry ────────────────────────────────────


@pytest.mark.asyncio
async def test_every_outcome_is_recorded(spool):
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(SlowTool())

    await reg.execute("echo", {"msg": "hi", "password": "pw"}, session_id="s1")
    await reg.execute("echo", {"fail": True})
    with pytest.raises(ToolExecutionError):
        await reg.execute("echo", {"explode": True})
    with pytest.raises(ToolBlockedError):
        await reg.execute("echo", {}, task_policy=ToolPolicy(deny=["echo"]))
    with pytest.raises(ToolNotFoundError):
        await reg.execute("nope", {"x": 1})
    with pytest.raises(ToolExecutionError):
        await reg.execute("slow", {})

    recs = _ended(spool)
    assert [r["seq"] for r in recs] == sorted(r["seq"] for r in recs)
    by = [(r["tool"], r["outcome"]) for r in recs]
    assert by == [("echo", "ok"), ("echo", "error"), ("echo", "error"), ("echo", "blocked"),
                  ("nope", "blocked"), ("slow", "timeout")]
    first = recs[0]
    assert first["phase"] == "full" and first["session_id"] == "s1"
    assert first["params"] == {"msg": "hi", "password": "[masked]"}
    assert recs[1]["reason"] == "it went wrong"
    assert "boom" in recs[2]["reason"]
    assert "Blocked by tool policy chain" in recs[3]["reason"]
    assert all(r["duration_ms"] is not None for r in recs)


@pytest.mark.asyncio
async def test_source_comes_from_the_agents_turn(spool):
    class Agent:
        _turn_source = ("whatsapp", "WhatsApp")

    reg = ToolRegistry()
    reg.register(EchoTool())
    await reg.execute("echo", {"_agent": Agent()})
    rec = _ended(spool)[0]
    assert (rec["source"], rec["source_label"]) == ("whatsapp", "WhatsApp")
    assert rec["speaker_id"] == ""


@pytest.mark.asyncio
async def test_a_bound_automation_wins_over_the_last_turns_source(spool):
    from captain_claw import mail_authority

    class Agent:
        _turn_source = ("whatsapp", "WhatsApp")   # the last chat turn's

    reg = ToolRegistry()
    reg.register(EchoTool())
    with mail_authority.bound(mail_authority.automated("fd_scheduler")):
        await reg.execute("echo", {"_agent": Agent()})
    rec = _ended(spool)[0]
    assert rec["source"] == "scheduled"


@pytest.mark.asyncio
async def test_member_turn_names_the_speaker(spool):
    from captain_claw import speaker

    p = speaker.Principal(speaker_id="user-m", display_name="Mia", owner_name="Owen", lane="A",
                          agent_ref="process:a:0123456789abcdef")
    tok = speaker.bind(p)
    try:
        h = tool_audit.start("web_search", {"query": "x"}, member=True)
        tool_audit.finish(h, "ok")
    finally:
        speaker.reset(tok)
    rec = _ended(spool)[0]
    assert (rec["speaker_id"], rec["speaker_name"]) == ("user-m", "Mia")
    assert rec["source"] == "member"


def test_guard_refusal_is_recorded(spool):
    tool_audit.refused("shell", {"command": "rm -rf /"}, "Guard: dangerous")
    rec = _ended(spool)[0]
    assert (rec["tool"], rec["outcome"], rec["reason"]) == ("shell", "blocked", "Guard: dangerous")


# ── The spool's phases ────────────────────────────────────────────────


def test_long_call_is_sent_as_started_then_ended(spool):
    h = spool.start("shell", {"command": "sleep 100"})
    s = spool._spool()
    records, done, started = s.batch(spool._BOOT, 1e12)
    assert [r["phase"] for r in records] == ["start"] and started and not done
    s.settle(done, started)
    assert s.batch(spool._BOOT, 1e12)[0] == []          # not sent twice
    spool.finish(h, "ok")
    records, done, started = s.batch(spool._BOOT, 1e12)
    assert [(r["phase"], r["outcome"]) for r in records] == [("end", "ok")]
    assert "params" not in records[0]
    s.settle(done, started)
    assert not s.pending()


def test_fresh_running_call_waits(spool):
    import time

    spool.start("shell", {"command": "sleep 1"})
    assert spool._spool().batch(spool._BOOT, time.time())[0] == []
    assert spool._spool().pending()


def test_unfinished_call_of_a_previous_process_is_interrupted(spool, monkeypatch):
    spool.start("shell", {"command": "x"})
    monkeypatch.setattr(spool, "_BOOT", "f" * 16)
    records, done, _ = spool._spool().batch(spool._BOOT, 0.0)
    assert [(r["phase"], r["outcome"], r["tool"]) for r in records] == [("full", "interrupted", "shell")]
    assert done


@pytest.mark.asyncio
async def test_ship_loop_sends_and_settles(spool, monkeypatch):
    sent: list[list[dict]] = []

    async def post(client, records):
        sent.append(records)
        return 200

    monkeypatch.setattr(spool, "_post", post)
    monkeypatch.setattr(spool, "_FLUSH_S", 0.01)
    spool.finish(spool.start("echo", {"a": 1}), "ok")
    await asyncio.wait_for(spool._ship_loop(), timeout=5)
    assert len(sent) == 1 and sent[0][0]["tool"] == "echo"
    assert not spool._spool().pending()


@pytest.mark.asyncio
async def test_ship_loop_keeps_records_when_flight_deck_refuses(spool, monkeypatch):
    calls = {"n": 0}
    real_sleep = asyncio.sleep

    async def post(client, records):
        calls["n"] += 1
        return 503

    async def fast_sleep(_s):
        if calls["n"] >= 2:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(spool, "_post", post)
    monkeypatch.setattr(spool.asyncio, "sleep", fast_sleep)
    spool.finish(spool.start("echo", {}), "ok")
    with pytest.raises(asyncio.CancelledError):
        await spool._ship_loop()
    assert spool._spool().pending()


# ── Turn id and result refs (Taskboard) ───────────────────────────────


class MailTool(Tool):
    name = "google_mail"
    description = "Mail"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        if kwargs.get("action") == "search":
            return ToolResult(success=True, content="  Message ID: m-listed\n  Thread ID: t-listed")
        if kwargs.get("fail"):
            return ToolResult(success=False, error="Message ID: m-failed")
        return ToolResult(success=True, content="Email sent.\n  To: ana@example.com\n"
                                                "  Message ID: 18c2f0a1b\n  Thread ID: 18c2e9f00")


@pytest.mark.asyncio
async def test_calls_carry_their_turns_id_and_count_for_it(spool):
    from captain_claw import turn_register

    reg = ToolRegistry()
    reg.register(EchoTool())
    turn = turn_register.Turn(turn_id="ab" * 16, started_at=0.0)
    tokens = (turn_register.current_turn_id.set(turn.turn_id), turn_register._CURRENT.set(turn))
    try:
        await reg.execute("echo", {})
        await reg.execute("echo", {"fail": True})
        with pytest.raises(ToolBlockedError):
            await reg.execute("echo", {}, task_policy=ToolPolicy(deny=["echo"]))
    finally:
        turn_register.current_turn_id.reset(tokens[0])
        turn_register._CURRENT.reset(tokens[1])
    await reg.execute("echo", {})
    recs = _ended(spool)
    assert [r["turn_id"] for r in recs] == ["ab" * 16] * 3 + [""]
    assert (turn.tool_calls, turn.failed_tool_calls) == (3, 1)     # a refusal isn't a failed call


@pytest.mark.asyncio
async def test_an_ok_send_records_the_ids_its_result_named(spool):
    reg = ToolRegistry()
    reg.register(MailTool())
    await reg.execute("google_mail", {"action": "send", "to": "ana@example.com", "body": "hi"})
    await reg.execute("google_mail", {"action": "search", "query": "x"})
    await reg.execute("google_mail", {"action": "send", "fail": True})
    sent, searched, failed = _ended(spool)
    assert sent["refs"] == {"gmail_message_id": "18c2f0a1b", "gmail_thread_id": "18c2e9f00"}
    assert "refs" not in searched and "refs" not in failed


@pytest.mark.parametrize("tool, action, content, refs", [
    ("google_mail", "create_draft", "Draft created.\n  Threaded under: t-9 (reply to m-1)\n  Draft ID: r-5\n"
     "  Message ID: m-6", {"gmail_draft_id": "r-5", "gmail_message_id": "m-6", "gmail_thread_id": "t-9"}),
    ("google_mail", "send", "Email sent.\n  Message ID: ?\n  Thread ID: ?", {}),
    ("whatsapp_web", "send", "Sent to Ana on WhatsApp (message id 3EB0C4F2).\nText as sent:\nMessage ID: no",
     {"waweb_message_id": "3EB0C4F2"}),
    ("whatsapp_send_file", "", "Sent 'report.pdf' to WhatsApp 38591234567 as a document.",
     {"wa_id": "38591234567"}),
    ("google_calendar", "create_event", "Created event 'Lunch' on Google Calendar.\n  ID: ev123\n  When: Fri",
     {"calendar_event_id": "ev123"}),
    ("microsoft_calendar", "update_event", "Updated event 'Sync'.\n  ID: AAMkAD=\n  When: Mon",
     {"calendar_event_id": "AAMkAD="}),
    ("google_calendar", "list_events", "  ID: ev1\n  ID: ev2", {}),
    ("web_fetch", "", "Message ID: x", {}),
])
def test_result_refs(tool, action, content, refs):
    assert tool_audit.result_refs(tool, action, ToolResult(success=True, content=content)) == refs


def test_clean_refs_keeps_known_id_shaped_values():
    assert tool_audit.clean_refs({"gmail_message_id": "18c", "body": "secret text", "wa_id": 385912,
                                  "gmail_thread_id": "has spaces in it", "calendar_event_id": True}) == {
        "gmail_message_id": "18c", "wa_id": "385912"}
    assert tool_audit.clean_refs("nope") == {}


def test_a_spool_an_older_agent_left_still_ships(tmp_path, monkeypatch, spool):
    import sqlite3

    path = tmp_path / "old_spool.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(tool_audit._SPOOL_DDL.replace("    refs        TEXT NOT NULL DEFAULT '',\n", ""))
    conn.execute("INSERT INTO calls (boot, seq, started_at, body, ended, outcome) VALUES (?, 1, 1.0, ?, 1, 'ok')",
                 ("f" * 16, json.dumps({"tool": "echo", "params": {}})))
    conn.commit()
    conn.close()
    monkeypatch.setenv("CLAW_TOOL_AUDIT_SPOOL", str(path))
    spool.finish(spool.start("echo", {}), "ok")
    old_rec, new_rec = _ended(spool)
    assert (old_rec["boot"], old_rec["seq"], old_rec["phase"], old_rec["outcome"]) == ("f" * 16, 1, "full", "ok")
    assert new_rec["boot"] == spool._BOOT and "turn_id" in new_rec
