"""The agent side of the Taskboard's turn register (captain_claw/turn_register.py):
every outer turn through Agent.complete / stream is spooled as an open and a
close record — masked, numbered, tied to its tool calls — and shipped to
Flight Deck one boot per batch."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from captain_claw import tool_audit, turn_register
from captain_claw.agent import Agent
from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

OPEN_KEYS = {"seq", "phase", "turn_id", "at", "session_id", "session_name", "origin", "origin_detail",
             "channel", "source", "source_label", "speaker_id", "speaker_label", "member", "private",
             "opener", "human", "literal", "text_head", "ext_refs", "automation"}
CLOSE_KEYS = {"seq", "phase", "turn_id", "at", "outcome", "reply_head", "deliverables", "tool_calls",
              "failed_tool_calls"}


class EchoTool(Tool):
    name = "echo"
    description = "Echo"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        if kwargs.get("fail"):
            return ToolResult(success=False, error="it went wrong")
        return ToolResult(success=True, content="ok")


class _StubProvider(LLMProvider):
    provider = "openai"
    model = "stub"

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text):
        return len(text.split()) or 1


def _close(module) -> None:
    if module._SPOOL is not None and module._SPOOL._conn is not None:
        module._SPOOL._conn.close()
    module._SPOOL = None


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("FD_AGENT_SLUG", "tester")
    monkeypatch.setenv("CLAW_TOOL_AUDIT_SPOOL", str(tmp_path / "spool.db"))
    monkeypatch.setenv("CLAW_TURN_REGISTER_SPOOL", str(tmp_path / "turns.db"))
    monkeypatch.delenv("CLAW_TOOL_AUDIT", raising=False)
    monkeypatch.delenv("CLAW_TURN_REGISTER", raising=False)
    monkeypatch.setattr("captain_claw.fd_client._agent_web_auth_token", lambda: "tok")
    for module in (tool_audit, turn_register):
        monkeypatch.setattr(module, "_SPOOL", None)
        monkeypatch.setattr(module, "_SPOOL_FAILED", False)
        monkeypatch.setattr(module, "_ensure_shipper", lambda: None)
    yield turn_register
    _close(tool_audit)
    _close(turn_register)


def _turns(t=turn_register) -> list[dict]:
    spool = t._spool()
    return spool.batch(tool_audit._BOOT, time.time())[0] if spool is not None else []


def _calls() -> list[dict]:
    return tool_audit._spool().batch(tool_audit._BOOT, 1e12)[0]


class _FakeSession:
    def __init__(self) -> None:
        self.id, self.name, self.metadata = "sess-1", "Planning", {}


class FakeAgent:
    """What open_turn reads off an agent once its origin is set."""

    def __init__(self, text="Book a table for two at 8", origin=("human", "", "web"),
                 source=("web", "Web chat")) -> None:
        self._turn_origin, self._turn_source, self._turn_user_text = origin, source, text
        self.session = _FakeSession()
        self._last_complete_success = True


def _agent() -> Agent:
    agent = Agent(provider=_StubProvider())
    agent.session = Session(id="s1", name="default")
    agent._build_env_now_text = lambda: ""
    agent._initialized = True
    agent.tools = ToolRegistry()
    agent.tools.register(EchoTool())
    return agent


# ── Records ───────────────────────────────────────────────────────────


def test_open_and_close_records_have_the_contract_shape(reg):
    agent = FakeAgent()
    turn = reg.open_turn(agent)
    assert turn is not None and len(turn.turn_id) == 32
    assert reg.current_turn_id.get() == turn.turn_id and reg.current() is turn
    reg.close_turn(turn, "ok", "Booked: Friday 20:00.", agent=agent)
    assert reg.current_turn_id.get() == "" and reg.current() is None

    opened, closed = _turns()
    assert set(opened) - {"boot"} == OPEN_KEYS and set(closed) - {"boot"} == CLOSE_KEYS
    assert (opened["phase"], closed["phase"]) == ("open", "close")
    assert opened["turn_id"] == closed["turn_id"] == turn.turn_id
    assert closed["seq"] == opened["seq"] + 1
    assert opened["session_id"] == "sess-1" and opened["session_name"] == "Planning"
    assert (opened["origin"], opened["channel"], opened["source"]) == ("human", "web", "web")
    assert opened["opener"] is True and opened["human"] is True and opened["literal"] == ""
    assert opened["member"] is False and opened["private"] is False and opened["speaker_id"] == ""
    assert opened["text_head"] == "Book a table for two at 8"
    assert opened["automation"] == {"kind": "", "id": ""}
    assert opened["origin_detail"] == {"channel": "web"}
    assert closed["outcome"] == "ok" and closed["reply_head"] == "Booked: Friday 20:00."
    assert closed["deliverables"] == [] and closed["tool_calls"] == 0 and closed["failed_tool_calls"] == 0


def test_close_is_once_and_unknown_outcomes_are_errors(reg):
    turn = reg.open_turn(FakeAgent())
    reg.close_turn(turn, "weird")
    reg.close_turn(turn, "ok")
    reg.close_turn(None, "ok")
    assert [r.get("outcome") for r in _turns()] == [None, "error"]


@pytest.mark.parametrize("text, origin, source, channel, literal, human", [
    ("Your last response was completely empty. Write the answer now", ("human", "", "web"),
     ("web", ""), "web", "corrective", True),
    ("[Autonomous nudge] Proactively reach out", ("autonomy", "nudge", None), ("autonomy", ""),
     "autonomy", "autonomy", False),
    ("[LIFE TICK — orient] You are Ada", ("life_tick", "tick", None), ("agent", ""), "", "life_tick", False),
    ("hi", ("human", "", "whatsapp"), ("whatsapp", "WhatsApp"), "whatsapp", "", True),
    ("hi", ("human", "", "device"), ("device:sticks3", "M5Stack StickS3"), "device:sticks3", "", True),
    ("hi", ("mcp_task", "", "mcp"), ("api", ""), "mcp", "", False),
    ("hi", ("human", "", "glasses"), ("app", ""), "web", "", True),
])
def test_origin_literal_and_channel(reg, text, origin, source, channel, literal, human):
    reg.close_turn(reg.open_turn(FakeAgent(text=text, origin=origin, source=source)), "ok")
    opened = _turns()[0]
    assert (opened["channel"], opened["literal"], opened["human"]) == (channel, literal, human)
    assert opened["opener"] is True


def test_member_turn_names_the_member_and_is_private(reg):
    from captain_claw import speaker

    p = speaker.Principal(speaker_id="user-m", display_name="Mia", owner_name="Owen", lane="A",
                          agent_ref="process:a:0123456789abcdef")
    tok = speaker.bind(p)
    try:
        reg.close_turn(reg.open_turn(FakeAgent(source=("member", ""))), "ok")
    finally:
        speaker.reset(tok)
    opened = _turns()[0]
    assert (opened["speaker_id"], opened["speaker_label"], opened["member"]) == ("user-m", "Mia", True)
    assert opened["private"] is True and opened["channel"] == "member"


def test_telegram_group_is_not_a_member_and_names_its_chat(reg):
    from captain_claw import speaker

    tok = speaker.bind(speaker.group_principal(-100123, "Team chat"))
    try:
        reg.close_turn(reg.open_turn(FakeAgent(origin=("human", "", "telegram"), source=("telegram", ""))), "ok")
    finally:
        speaker.reset(tok)
    opened = _turns()[0]
    assert opened["member"] is False and opened["speaker_id"] == "" and opened["speaker_label"] == "Team chat"
    assert opened["ext_refs"] == {"telegram_chat_id": "-100123"}


def test_ext_refs_and_a_scheduled_jobs_id(reg):
    from captain_claw import mail_authority

    agent = FakeAgent(origin=("cron", "fd_scheduler", "whatsapp"), source=("scheduled", ""))
    agent.session.metadata.update(origin={"kind": "whatsapp", "address": "38591"}, whatsapp_waid="38591",
                                  scheduled_task="job-7")
    with mail_authority.bound(mail_authority.automated("fd_scheduler")):
        reg.close_turn(reg.open_turn(agent), "ok")
    opened = _turns()[0]
    assert opened["automation"] == {"kind": "scheduled", "id": "job-7"}
    assert opened["channel"] == "scheduled" and opened["human"] is False
    assert opened["ext_refs"] == {"origin": {"kind": "whatsapp", "address": "38591"}}   # not a WhatsApp turn
    assert opened["origin_detail"]["automation"] == "fd_scheduler"


def test_a_links_turn_carries_the_link(reg):
    from captain_claw import mail_authority

    agent = FakeAgent(origin=("automated", "link", None), source=("link", "CRM"))
    agent.session.metadata["scheduled_task"] = "link-abc123"
    with mail_authority.bound(mail_authority.automated("link")):
        reg.close_turn(reg.open_turn(agent), "ok")
    opened = _turns()[0]
    assert opened["automation"] == {"kind": "link", "id": "abc123"}
    assert opened["ext_refs"]["link_id"] == "abc123"


def test_text_and_reply_are_masked_and_cut(reg):
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
    agent = FakeAgent(text=f"use key {secret} and card 4111 1111 1111 1111 " + "x" * 3000)
    reg.close_turn(reg.open_turn(agent), "ok", f"Done with {secret}. " + "y" * 3000)
    opened, closed = _turns()
    assert secret not in opened["text_head"] and "4111 1111" not in opened["text_head"]
    assert "[masked]" in opened["text_head"]
    assert len(opened["text_head"]) == 1000 and len(closed["reply_head"]) == 1000
    assert secret not in closed["reply_head"]


def test_deliverables_are_this_turns_saved_files(reg, tmp_path):
    saved = tmp_path / "saved"
    (saved / "output" / "sess").mkdir(parents=True)
    (saved / "downloads" / "sess").mkdir(parents=True)
    old = saved / "output" / "sess" / "old.txt"
    old.write_text("before")
    os.utime(old, (time.time() - 3600, time.time() - 3600))

    class Tools:
        def get_saved_base_path(self, create=False):
            return saved

    agent = FakeAgent()
    agent.tools = Tools()
    agent._current_session_slug = lambda: "sess"
    turn = reg.open_turn(agent)
    soon = time.time() + 1
    for made in (saved / "output" / "sess" / "report.docx", saved / "downloads" / "sess" / "input.pdf"):
        made.write_text("made now")
        os.utime(made, (soon, soon))
    reg.close_turn(turn, "ok", "Here is report.docx", agent=agent)
    assert _turns()[1]["deliverables"] == ["output/sess/report.docx"]


def _saved_agent(saved):
    class Tools:
        def get_saved_base_path(self, create=False):
            return saved

    agent = FakeAgent()
    agent.tools = Tools()
    agent._current_session_slug = lambda: "sess"
    return agent


def test_working_dirs_are_passed_over_by_name(reg, tmp_path, monkeypatch):
    """A session's downloads (or any working dir) never use up the walk:
    the report it saved this turn is still found."""
    monkeypatch.setattr(reg, "_WALK_MAX", 50)
    saved = tmp_path / "saved"
    for folder in ("downloads", "scripts", ".cache"):
        (saved / folder / "sess").mkdir(parents=True)
        for i in range(80):
            (saved / folder / "sess" / f"page{i}.html").write_text("x")
    (saved / "reports" / "sess" / "old").mkdir(parents=True)
    for i in range(30):   # files from before the turn: looked at, never checked closely
        old = saved / "reports" / "sess" / "old" / f"draft{i}.md"
        old.write_text("before")
        os.utime(old, (time.time() - 3600, time.time() - 3600))
    agent = _saved_agent(saved)
    turn = reg.open_turn(agent)
    (saved / "reports" / "sess" / "weekly.md").write_text("made now")
    reg.close_turn(turn, "ok", agent=agent)
    assert _turns()[1]["deliverables"] == ["reports/sess/weekly.md"]


@pytest.mark.asyncio
async def test_saved_files_are_listed_off_the_event_loop(reg, tmp_path, monkeypatch):
    import threading

    saved = tmp_path / "saved"
    (saved / "output" / "sess").mkdir(parents=True)
    where: list[str] = []
    real = reg._deliverables

    def listing(base, slug, since):
        where.append(threading.current_thread().name)
        return real(base, slug, since)

    monkeypatch.setattr(reg, "_deliverables", listing)
    agent = _saved_agent(saved)
    turn = reg.open_turn(agent)
    (saved / "output" / "sess" / "notes.md").write_text("made now")
    reg.close_turn(turn, "ok", agent=agent)
    assert reg.current_turn_id.get() == ""   # the turn ends at once; its close follows
    await reg.wait_closes()
    assert where and where[0] != threading.current_thread().name
    assert _turns()[1]["deliverables"] == ["output/sess/notes.md"]


def test_off_outside_flight_deck_and_with_the_switch(tmp_path, monkeypatch, reg):
    monkeypatch.setenv("CLAW_TURN_REGISTER", "off")
    assert reg.open_turn(FakeAgent()) is None
    assert reg.current_turn_id.get() == ""
    monkeypatch.delenv("CLAW_TURN_REGISTER")
    monkeypatch.setenv("CLAW_TOOL_AUDIT", "off")        # off exactly when the tool-call register is
    assert reg.open_turn(FakeAgent()) is None
    monkeypatch.delenv("CLAW_TOOL_AUDIT")
    monkeypatch.delenv("FD_URL")
    assert reg.open_turn(FakeAgent()) is None
    assert not (tmp_path / "turns.db").exists()


def test_never_raises_when_the_spool_cant_be_written(tmp_path, monkeypatch, reg):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder")
    monkeypatch.setenv("CLAW_TURN_REGISTER_SPOOL", str(blocker / "turns.db"))
    agent = FakeAgent()
    assert reg.open_turn(agent) is None
    assert reg.current_turn_id.get() == "" and reg.current() is None
    reg.close_turn(None, "ok", agent=agent)


# ── Through the agent ─────────────────────────────────────────────────


@pytest.fixture()
def fake_turn(monkeypatch):
    """Agent.complete's inner turn, replaced: it makes one ok and one failed
    tool call and says which turn id it ran under."""
    seen: dict = {"ids": [], "raise": None, "reply": "All done.", "success": True}

    async def complete(self, user_input):
        seen["ids"].append(turn_register.current_turn_id.get())
        await self.tools.execute("echo", {"_agent": self})
        await self.tools.execute("echo", {"_agent": self, "fail": True})
        if seen["raise"] is not None:
            raise seen["raise"]
        self._last_complete_success = seen["success"]
        return seen["reply"]

    monkeypatch.setattr(AgentOrchestrationMixin, "complete", complete)
    return seen


@pytest.mark.asyncio
async def test_complete_records_one_turn_tied_to_its_tool_calls(reg, fake_turn):
    agent = _agent()
    assert await agent.complete("what's on my calendar?", channel="web") == "All done."
    await reg.wait_closes()
    opened, closed = _turns()
    assert opened["text_head"] == "what's on my calendar?" and opened["session_id"] == "s1"
    assert closed["outcome"] == "ok" and closed["reply_head"] == "All done."
    assert (closed["tool_calls"], closed["failed_tool_calls"]) == (2, 1)
    assert fake_turn["ids"] == [opened["turn_id"]]
    assert {c["turn_id"] for c in _calls()} == {opened["turn_id"]}
    assert reg.current_turn_id.get() == ""
    # A tool call outside any turn carries none.
    await agent.tools.execute("echo", {})
    assert _calls()[-1]["turn_id"] == ""


@pytest.mark.asyncio
async def test_a_call_after_its_turn_closed_carries_no_turn(reg):
    tools = ToolRegistry()
    tools.register(EchoTool())
    gate = asyncio.Event()
    turn = reg.open_turn(FakeAgent())

    async def later():
        await tools.execute("echo", {})
        await gate.wait()
        await tools.execute("echo", {})

    task = asyncio.create_task(later())       # its context holds the turn
    await asyncio.sleep(0)
    reg.close_turn(turn, "ok")
    gate.set()
    await task
    assert [c["turn_id"] for c in _calls()] == [turn.turn_id, ""]
    assert _turns()[1]["tool_calls"] == 1


@pytest.mark.asyncio
async def test_stream_is_one_turn_though_it_reenters_complete(reg, fake_turn):
    agent = _agent()
    chunks = [c async for c in agent.stream("summarise the thread", channel="web")]
    assert "".join(chunks) == "All done."
    await reg.wait_closes()
    records = _turns()
    assert [r["phase"] for r in records] == ["open", "close"]
    assert records[1]["reply_head"] == "All done." and records[1]["tool_calls"] == 2
    assert fake_turn["ids"] == [records[0]["turn_id"]]


@pytest.mark.asyncio
async def test_nested_complete_opens_nothing(reg, monkeypatch):
    outer, inner = _agent(), _agent()

    async def complete(self, user_input):
        if self is outer:
            await inner.complete("an internal sub-call")
        return "ok"

    monkeypatch.setattr(AgentOrchestrationMixin, "complete", complete)
    await outer.complete("do the thing")
    await reg.wait_closes()
    assert [r["phase"] for r in _turns()] == ["open", "close"]
    assert _turns()[0]["text_head"] == "do the thing"


@pytest.mark.asyncio
async def test_exception_cancellation_and_guard_block(reg, fake_turn):
    agent = _agent()
    fake_turn["raise"] = RuntimeError("model down")
    with pytest.raises(RuntimeError):
        await agent.complete("one")
    fake_turn["raise"] = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await agent.complete("two")
    fake_turn["raise"] = None
    fake_turn.update(success=False, reply="Blocked by input guard: Suspicious content detected.")
    await agent.complete("three")
    fake_turn.update(reply="Sorry, I ran out of steps.")
    await agent.complete("four")
    await reg.wait_closes()
    closes = [r for r in _turns() if r["phase"] == "close"]
    assert [c["outcome"] for c in closes] == ["error", "cancelled", "blocked", "error"]
    assert reg.current() is None


class _StopProvider(_StubProvider):
    """A model the user stops while it thinks: Stop (``cancel_event``)
    pressed during its first answer, which asks for a tool."""

    def __init__(self) -> None:
        self.agent: Agent | None = None
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None, **kw):
        self.calls += 1
        if self.calls == 1:
            self.agent.cancel_event.set()
            return LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="echo", arguments={})])
        return LLMResponse(content="ok")


@pytest.mark.asyncio
async def test_the_users_stop_closes_cancelled_not_error(reg):
    """The tool loop's own Stop (it returns, nothing is raised) is a
    cancelled turn — never a failed task on the board."""
    agent = _agent()
    agent.provider = provider = _StopProvider()
    provider.agent = agent
    reply = await agent.complete("look up the flights", channel="web")
    assert reply == "Request cancelled by user." and agent._last_complete_success is False
    await reg.wait_closes()
    assert [(r["phase"], r.get("outcome")) for r in _turns()] == [("open", None), ("close", "cancelled")]
    # The next turn starts with no Stop of its own.
    agent.provider = _StubProvider()
    await agent.complete("and the hotels?", channel="web")
    await reg.wait_closes()
    assert [r["outcome"] for r in _turns() if r["phase"] == "close"] == ["cancelled", "ok"]


@pytest.mark.asyncio
async def test_a_stream_left_early_closes_cancelled(reg, fake_turn):
    agent = _agent()
    gen = agent.stream("x" * 10, channel="web")
    await gen.__anext__()
    await gen.aclose()
    await reg.wait_closes()
    assert [(r["phase"], r.get("outcome")) for r in _turns()] == [("open", None), ("close", "cancelled")]


@pytest.mark.asyncio
async def test_agent_turn_runs_when_the_spool_cant_be_written(tmp_path, monkeypatch, reg, fake_turn):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder")
    monkeypatch.setenv("CLAW_TURN_REGISTER_SPOOL", str(blocker / "turns.db"))
    agent = _agent()
    assert await agent.complete("still answered") == "All done."
    assert fake_turn["ids"] == [""]


# ── Shipping ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ship_loop_posts_one_boot_per_batch(reg, monkeypatch):
    sent: list[dict] = []

    class Client:
        async def post(self, path, json):
            sent.append({"path": path, "json": json})

            class R:
                status_code = 200
            return R()

        async def close(self):
            pass

    monkeypatch.setattr("captain_claw.fd_client.FDClient", lambda timeout=10.0: Client())
    monkeypatch.setattr(tool_audit, "_FLUSH_S", 0.01)
    monkeypatch.setattr(tool_audit, "_BOOT", "a" * 16)
    reg.close_turn(reg.open_turn(FakeAgent()), "ok")
    monkeypatch.setattr(tool_audit, "_BOOT", "b" * 16)
    reg.close_turn(reg.open_turn(FakeAgent()), "ok")
    await asyncio.wait_for(reg._ship_loop(), timeout=5)

    assert [s["path"] for s in sent] == [reg.INGEST_PATH] * 2
    assert [s["json"]["boot"] for s in sent] == ["a" * 16, "b" * 16]
    for s in sent:
        assert set(s["json"]) == {"boot", "records"}
        assert [r["phase"] for r in s["json"]["records"]] == ["open", "close"]
        assert all("boot" not in r for r in s["json"]["records"])
    assert not reg._spool().pending()


@pytest.mark.asyncio
async def test_records_stay_when_flight_deck_refuses(reg, monkeypatch):
    calls = {"n": 0}
    real_sleep = asyncio.sleep

    async def post(client, records):
        calls["n"] += 1
        return 503

    async def fast_sleep(_s):
        if calls["n"] >= 2:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(reg, "_post", post)
    monkeypatch.setattr(tool_audit.asyncio, "sleep", fast_sleep)
    reg.close_turn(reg.open_turn(FakeAgent()), "ok")
    with pytest.raises(asyncio.CancelledError):
        await reg._ship_loop()
    assert reg._spool().pending()


def test_unreadable_spool_rows_are_dropped_not_retried(reg):
    reg.close_turn(reg.open_turn(FakeAgent()), "ok")
    spool = reg._spool()
    spool._db().execute("UPDATE turns SET body='{not json' WHERE seq=(SELECT min(seq) FROM turns)")
    spool._db().commit()
    records, done, _ = spool.batch("", 0.0)
    assert [r["phase"] for r in records] == ["close"] and len(done) == 1
    spool.settle(done, [])
    assert not spool.pending()
