import re
from typing import Any

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class SingleResponseProvider(LLMProvider):
    def __init__(self, content: str = "ok"):
        self.content = content
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content)

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


class RecordingShellTool(Tool):
    name = "shell"
    description = "Test shell tool"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self):
        self.commands: list[str] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.commands.append(str(kwargs.get("command", "")))
        return ToolResult(success=True, content="ok")


@pytest.mark.asyncio
async def test_input_guard_stop_suspicious_blocks_before_llm_call(monkeypatch: pytest.MonkeyPatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.guards.input.enabled = True
    cfg.guards.input.level = "stop_suspicious"
    set_config(cfg)
    try:
        provider = SingleResponseProvider(content="should-not-be-called")
        agent = Agent(provider=provider)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = ToolRegistry()

        async def fake_guard(guard_type: str, interaction_label: str, content: str, turn_usage=None):
            if guard_type == "input":
                return {"allow": False, "reason": "Suspicious input detected.", "raw": ""}
            return {"allow": True, "reason": "ok", "raw": ""}

        monkeypatch.setattr(agent, "_run_guard_decision", fake_guard)

        result = await agent.complete("dangerous prompt")

        assert "Blocked by input guard" in result
        assert provider.calls == 0
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_output_guard_ask_for_approval_blocks_when_denied(monkeypatch: pytest.MonkeyPatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.guards.output.enabled = True
    cfg.guards.output.level = "ask_for_approval"
    set_config(cfg)
    try:
        approvals: list[str] = []

        def approve(question: str) -> bool:
            approvals.append(question)
            return False

        provider = SingleResponseProvider(content="run rm -rf / now")
        agent = Agent(provider=provider, approval_callback=approve)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = ToolRegistry()

        async def fake_guard(guard_type: str, interaction_label: str, content: str, turn_usage=None):
            if guard_type == "output":
                return {"allow": False, "reason": "Dangerous output.", "raw": ""}
            return {"allow": True, "reason": "ok", "raw": ""}

        monkeypatch.setattr(agent, "_run_guard_decision", fake_guard)

        result = await agent.complete("hello")

        assert "Blocked by output guard (approval denied)" in result
        assert len(approvals) == 1
        assert provider.calls == 1
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_script_tool_guard_blocks_tool_execution(monkeypatch: pytest.MonkeyPatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.guards.script_tool.enabled = True
    cfg.guards.script_tool.level = "stop_suspicious"
    set_config(cfg)
    try:
        provider = SingleResponseProvider()
        shell = RecordingShellTool()
        registry = ToolRegistry()
        registry.register(shell)

        agent = Agent(provider=provider)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = registry

        async def fake_guard(guard_type: str, interaction_label: str, content: str, turn_usage=None):
            if guard_type == "script_tool":
                return {"allow": False, "reason": "Destructive command.", "raw": ""}
            return {"allow": True, "reason": "ok", "raw": ""}

        monkeypatch.setattr(agent, "_run_guard_decision", fake_guard)

        results = await agent._handle_tool_calls(
            [ToolCall(id="c1", name="shell", arguments={"command": "rm -rf /"})]
        )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Blocked by script_tool guard" in results[0]["error"]
        assert shell.commands == []
    finally:
        set_config(old_cfg)


class TestFalseActionClaimRetractionGuard:
    """The false-action-claim gate must not fire on the model's own apology.

    Regression for the apology loop: a weak model's correction
    ("lažno sam tvrdio da sam delegirao, a nisam" — I falsely claimed I
    delegated, but I didn't) contains the stem "delegira", so the claim
    regex re-matched it, the gate re-injected an accusation, and the model
    apologized again — endlessly. A retraction that DENIES the action must
    never count as a fresh claim of that action.
    """

    @pytest.mark.parametrize(
        "text",
        [
            # the two exact apologies from the stuck deepseek-v4-flash session
            "Shvaćam — i prihvaćam. U prvom odgovoru jesam lažno tvrdio da sam "
            "delegirao, a nisam pozvao tool. To sam ispravio u trećoj poruci.",
            "Shit, potpuno si u pravu — ispričavam se. U prošlom koraku nisam "
            "pozvao nijedan tool za delegaciju, samo sam rekao da sam spojio.",
            "U zadnjem odgovoru nisam tvrdio nikakvu delegaciju — samo sam sažeo "
            "što report sadrži.",
            "You're right — that claim is false, I didn't actually delegate it.",
        ],
    )
    def test_delegation_apology_is_not_a_claim(self, text: str) -> None:
        from captain_claw.agent_orchestration_mixin import _claims_delegation

        assert _claims_delegation(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "I apologize, I did not actually search the web — I pulled that from memory.",
            "That claim is false — I didn't fetch anything this turn.",
        ],
    )
    def test_web_apology_is_not_a_claim(self, text: str) -> None:
        from captain_claw.agent_orchestration_mixin import _claims_web_research

        assert _claims_web_research(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "Poslao sam zadatak peer-u, čekam odgovor.",
            "I delegated it to the researcher and I'm waiting for the reply.",
            "Proslijedio sam to MiniMax-u.",
            # genuine claim with an UNRELATED negation must still be caught
            "Poslao sam zadatak, ali nisam dobio odgovor još.",
        ],
    )
    def test_genuine_delegation_claim_still_detected(self, text: str) -> None:
        from captain_claw.agent_orchestration_mixin import _claims_delegation

        assert _claims_delegation(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "I searched the web and found three sources.",
            "Pretražio sam web i pronašao odgovor.",
        ],
    )
    def test_genuine_web_claim_still_detected(self, text: str) -> None:
        from captain_claw.agent_orchestration_mixin import _claims_web_research

        assert _claims_web_research(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            # "Pošaljem na WhatsApp?" is an offer to send the report TO THE USER,
            # not a peer hand-off — the substring "šaljem" inside "Pošaljem" must
            # not trip the delegation pattern (regression: it killed a real
            # 2k-char Genesis findings answer that ended with this offer).
            "Što sad? 📱 Pošaljem na WhatsApp? 🔍 Istražimo dublje?",
            "Mogu ti to poslati — Pošaljem na WhatsApp ili otvorim u browseru?",
            "## Genesis Space Labs — Kompletan pregled\n\nOpis: startup iz Čakovca "
            "(12 ljudi). Report ima 890 linija. **Što sad?** 📱 Pošaljem na WhatsApp?",
        ],
    )
    def test_posaljem_offer_to_user_is_not_a_delegation_claim(self, text: str) -> None:
        from captain_claw.agent_orchestration_mixin import _claims_delegation

        assert _claims_delegation(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "Šaljem zadatak peer-u, čekam odgovor.",
            "Šaljemo to istraživaču odmah.",
        ],
    )
    def test_genuine_saljem_claim_still_detected(self, text: str) -> None:
        from captain_claw.agent_orchestration_mixin import _claims_delegation

        assert _claims_delegation(text) is True


# ── Made-up tool work: false sends, made-up IDs, narrated lookups ────
#
# Regression for the deepseek-v4-flash-glasses failure: asked to send an
# email, the model answered "**Sent** to …" (sometimes with an invented
# Message ID, narrated searches and a forged lane note) without calling
# google_mail at all, and every gate let the turn finish. What a reply is and
# what the model is told is decided in captain_claw.fabrication_guard (unit
# tests in tests/test_fabrication_guard.py); these run the agent loop.

_SEND_INVITE_RE = re.compile(
    r"(?i)action\s*=\s*send|send_draft|\bcall\b[^.]*\bsend\b|\bre-?send\b"
    r"|\bsend\s+(?:it|this|that|the|them|an?|your|now|again)\b|\bgo(?:es)?\s+out\b"
)


class _ScriptedProvider(LLMProvider):
    """Answers with the scripted responses in order, then "ok"."""

    provider = "openai"
    model = "stub"

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


class _StubMailTool(Tool):
    """google_mail that records its calls; ``send`` answers like the real
    tool, with a repeat refusal (of a sent email or of a draft), or with
    another refusal."""

    name = "google_mail"
    description = "Gmail"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "query": {"type": "string"},
            "draft_id": {"type": "string"},
            "thread_id": {"type": "string"},
        },
        "required": ["action"],
    }

    _MATCHES = {
        "sent": {"kind": "sent", "subject": "10:30", "to": "ana@x.co", "date": "12 Oct",
                 "message_id": "m-0"},
        "draft": {"kind": "draft", "subject": "10:30", "to": "ana@x.co", "draft_id": "r-1"},
        "thread_draft": {"kind": "draft", "subject": "Re: 10:30", "to": "ana@x.co",
                         "message_id": "m-7"},
    }

    def __init__(self, *, repeat: str = "", refusal: str = "",
                 search_text: str = "1. To: ana@x.co — 10:30 (12 Oct)  Newest msg: m-0",
                 sent: str = "Email sent.\n  To: ana@x.co\n  Subject: 10:30\n  Message ID: m-1\n"
                             "  Thread ID: t-1"):
        self.actions: list[str] = []
        self.repeat = repeat
        self.refusal = refusal
        self.search_text = search_text
        self.sent = sent

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        from captain_claw import gmail_compose

        self.actions.append(action)
        if action == "send" and self.refusal:
            return ToolResult(success=False, error=self.refusal)
        if action == "send" and self.repeat:
            return ToolResult(success=False, error=gmail_compose.repeat_refusal(
                [self._MATCHES[self.repeat]], lead="Not sent"))
        if action in ("send", "send_draft"):
            return ToolResult(success=True, content=self.sent)
        if action == "search":
            return ToolResult(success=True, content=self.search_text)
        if action == "list_drafts":
            return ToolResult(success=True, content="No drafts found.")
        if action == "get_thread":
            return ToolResult(success=True, content=(
                "Thread t-1 (2 messages)\n  ID: m-ana-1  |  To: ana@x.co — Re: 10:30\n"
                "  ID: m-ana-2  |  From: ana@x.co — Re: 10:30"))
        if action == "get_attachment":
            return ToolResult(success=True, content="Saved to: saved/downloads/s1/invoice.pdf")
        return ToolResult(success=True, content="ok")


class _StubActionTool(Tool):
    """Another tool the turn can call (a calendar, a hand-off, a picture)."""

    description = "stub"
    parameters = {"type": "object", "properties": {"action": {"type": "string"}}}

    def __init__(self, name: str, content: str, success: bool = True):
        self.name = name
        self.content = content
        self.success = success
        self.calls = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        if not self.success:
            return ToolResult(success=False, error=self.content)
        return ToolResult(success=True, content=self.content)


def _send_call(call_id: str = "c1", content: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[ToolCall(
        id=call_id, name="google_mail",
        arguments={"action": "send", "to": "ana@x.co", "subject": "10:30", "body": "10:30 works."},
    )])


def _mail_call(action: str, call_id: str = "s1", content: str = "", **arguments: Any) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[ToolCall(
        id=call_id, name="google_mail", arguments={"action": action, **arguments})])


def _call(name: str, call_id: str = "x1", content: str = "", **arguments: Any) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[ToolCall(id=call_id, name=name,
                                                             arguments=arguments)])


def _calls(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    """One response with several tool calls, as the model batches them."""
    return LLMResponse(content="", tool_calls=[
        ToolCall(id=f"b{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)])


def _mail_agent(responses: list[LLMResponse], tool: _StubMailTool | None = None):
    provider = _ScriptedProvider(responses)
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    mail = tool or _StubMailTool()
    registry = ToolRegistry()
    registry.register(mail)
    agent.tools = registry
    agent._build_env_now_text = lambda: ""
    return agent, provider, mail


def _earlier_mail_lookup(agent: Agent) -> None:
    """google_mail ran a few messages back (the normal state of a mail-heavy
    session): a bare "Sent" alone would read as an email."""
    agent.session.add_message("user", "any mail from Ana?")
    agent.session.add_message("assistant", "", tool_calls=[{
        "id": "m0", "type": "function",
        "function": {"name": "google_mail", "arguments": "{\"action\": \"search\"}"}}])
    agent.session.add_message("tool", "No threads found.", tool_call_id="m0", tool_name="google_mail",
                              tool_arguments={"action": "search"})
    agent.session.add_message("assistant", "Nothing from Ana.")


@pytest.fixture
def _human_turn(monkeypatch: pytest.MonkeyPatch):
    from captain_claw import mail_authority

    for name in ("CLAW_BEING_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
                 "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT", "CLAW_BAT_WORKER"):
        monkeypatch.delenv(name, raising=False)
    assert mail_authority.current().mode == "human"


def _by_detail(agent: Agent, detail: str, role: str) -> list[dict]:
    return [m for m in agent.session.messages
            if m.get("origin_detail") == detail and m["role"] == role]


def _correctives(agent: Agent, detail: str | None = None) -> list[dict]:
    """The gate's correctives (every kind, or *detail*'s); none asks for a send."""
    found = [m for m in agent.session.messages
             if m["role"] == "user" and m.get("origin") == "corrective"
             and m.get("origin_detail") in (
                 (detail,) if detail else ("false_mail_send", "false_calendar_change",
                                           "narrated_lookup", "fabricated_id"))]
    for m in found:
        own = re.sub(r'the tool answered: ".*?" Nothing was sent\.', "", m["content"])
        assert not _SEND_INVITE_RE.search(own), m["content"]
    return found


def _mail_correctives(agent: Agent) -> list[str]:
    return [m["content"] for m in _correctives(agent, "false_mail_send")]


_CORRECTIVE_LEAD = "Your reply says an email was sent, but no send ran in this turn"
_LANE = "[Autonomous Work — from the AUTO lane]"
# The production turns (synthetic): no tool call, "Sent" — 761 798 809 820
# 835 845, then the two Neven turns (867: narrated searches, "Sent to …" and a
# forged "Not sent — …" lane block; 869: "Sending now. / Sent — threaded on …"
# and a forged "Verified — Thread <made-up id>").
_PRODUCTION_TURNS = [
    ("ok, 10:30 is ok",
     "Finding the panel thread.\n\nChecking for an existing reply first.\n\n**Sent** to Ana + "
     "Ivo — confirming 10:30.\n\nNo duplicate existed; footer added."),
    ("answer Luka's two questions, add Ana and Ivo to the mail",
     f"Checking there's no existing reply first.\n\n**Sent to Luka**, CC Ana + Ivo.\n\n{_LANE}\n"
     "Sent — verified in the thread."),
    ("reply to Petra with this doc, ask if ok",
     "Reading the document, then sending.\n\nChecking the thread too.\n\n**Sent to Petra** — "
     f"reply with the portfolio inline.\n\n{_LANE}\nSent — verified in the thread, not just "
     "asserted. ID `1a2b3c4d5e6f7a8b`."),
    ("send additional mail to Luka",
     "Checking for an existing send first, then sending.\n\n**Sent** — the founders note, CC "
     "Ana + Ivo.\n\nID `1a2b3c4d5e6f7a8c`"),
    ("send questions in english to Tom",
     "Checking for an existing send to Tom first.\n\n**Sent to Tom** — panel questions, in "
     f"English.\n\nID `1a2b3c4d5e6f7a8d`.\n\n{_LANE}\nThe send failed. Nothing created or "
     "sent this turn."),
    ("one line ack to Tom",
     "Finding Tom's latest message to reply to.\n\n**Sent** — one-line ack to Tom\n\n"
     "ID `1a2b3c4d5e6f7a8e`"),
    ("send mail to Luka at luka@example.com, tell him to write more about the plugins",
     "Searching for the thread to ground the reply.\n\n**Sent to Luka** — asking him to write "
     f"more about the plugins.\n\nFooter added per your rule.\n\n{_LANE}\n**Not sent — the "
     "only prior contact is a cold inbound.** I checked drafts and sent mail.\n\nSay **go** and "
     "I'll send it now."),
    ("send now",
     "Finding Luka's message to thread the reply.\n\nSending now.\n\n**Sent** — threaded on "
     f"Luka's 30 Sep mail.\n\nFooter added.\n\n{_LANE}\n**Verified — it did go this time.** "
     "Thread `1a2b3c4d5e6f7a8f` now shows your reply as the newest message."),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text, reply", _PRODUCTION_TURNS)
async def test_the_production_false_sends_get_one_corrective(_human_turn, request_text, reply):
    truthful = "Not sent yet — nothing went out in this turn. Want me to send it?"
    agent, provider, mail = _mail_agent([LLMResponse(content=reply), LLMResponse(content=truthful)])
    _earlier_mail_lookup(agent)

    result = await agent.complete(request_text)

    assert mail.actions == []
    assert result.startswith(truthful) and "from the AUTO lane" not in result
    assert len(provider.calls) == 2
    correctives = _correctives(agent)
    assert len(correctives) == 1
    assert correctives[0]["origin_detail"] == "false_mail_send"
    assert correctives[0]["content"].startswith(_CORRECTIVE_LEAD)
    assert "Also: your reply describes lookups" in correctives[0]["content"]
    # The reply as written is stored as rejected, never replayed later.
    assert [m["content"] for m in _by_detail(agent, "false_mail_send", "assistant")] == [reply]
    # Never a forced tool call: the model chooses.
    assert getattr(provider, "_tool_choice_override", None) is None


@pytest.mark.asyncio
async def test_a_false_send_claim_then_the_real_send(_human_turn):
    false_sent = "Checking for an existing reply first.\n\n**Sent** to Ana — confirming 10:30."
    agent, provider, mail = _mail_agent([
        LLMResponse(content=false_sent),
        _send_call(),
        LLMResponse(content="Email sent to Ana — Message ID m-1."),
    ])

    result = await agent.complete("reply to Ana that 10:30 works, send it")

    assert mail.actions == ["send"]
    assert result.startswith("Email sent to Ana — Message ID m-1.")
    assert len(provider.calls) == 3
    rejected = _by_detail(agent, "false_mail_send", "assistant")
    assert [m["content"] for m in rejected] == [false_sent] and rejected[0]["origin"] == "rejected"
    assert len(_mail_correctives(agent)) == 1
    # The retry saw its own false reply and the corrective, in this turn.
    retry = provider.calls[1]
    assert retry[-2].role == "assistant" and retry[-2].content == false_sent
    assert retry[-1].role == "user" and retry[-1].content.startswith(_CORRECTIVE_LEAD)


@pytest.mark.asyncio
async def test_the_real_first_try_sends_are_untouched(_human_turn):
    # Two shapes of the sends that worked: lookups (also in parallel, also a
    # calendar), the pre-send checks, the send, then "Sent".
    calendar = _StubActionTool("google_calendar", "Fri 10:00 NAV review (Tihana)")
    agent, provider, mail = _mail_agent([
        _calls(("google_calendar", {"action": "search_events"}), ("google_mail", {"action": "search"})),
        _calls(("google_calendar", {"action": "list_events"}),
               ("google_mail", {"action": "get_thread", "thread_id": "t-1"})),
        _calls(("google_mail", {"action": "list_drafts"}),
               ("google_mail", {"action": "search", "query": "in:sent to:ana@x.co"})),
        _send_call(),
        LLMResponse(content="Finding the NAV review event.\n\nChecking for an existing reply "
                            "before I send.\n\n**Sent to Tihana** — Friday's NAV review confirmed, "
                            "threaded."),
        _mail_call("search", query="in:sent to:roger"),
        _calls(("google_mail", {"action": "list_drafts"}),
               ("google_calendar", {"action": "create_event"})),
        _mail_call("get_thread", thread_id="t-1"),
        _send_call(),
        LLMResponse(content="**Done.**\n- **Sent to Roger** — you'll attend; footer added.\n"
                            "- Blocker added for Friday."),
    ])
    agent.tools.register(calendar)

    first = await agent.complete("confirm Tihana's Friday NAV")
    second = await agent.complete("confirm Roger I'll attend + calendar blocker")

    assert first.startswith("Finding the NAV review event.\n\nChecking for an existing reply")
    assert second.startswith("**Done.**\n- **Sent to Roger**")
    assert mail.actions.count("send") == 2 and len(provider.calls) == 10
    assert _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("nudge", [
    "are you sure it's sent?", "is it really sent?", "gordana looks like not sent", "not sent",
    "send it",
])
async def test_a_send_after_a_nudge_is_untouched(_human_turn, nudge):
    from captain_claw.fabrication_guard import MAIL_BACKSTOP

    agent, provider, mail = _mail_agent([
        _mail_call("search", query="in:sent to:ana@x.co"),
        _mail_call("get_thread", call_id="s2", thread_id="t-1"),
        _send_call(),
        LLMResponse(content="**Sent to Ana** — confirming 10:30, threaded. Message ID m-1."),
    ])
    _earlier_mail_lookup(agent)
    # The earlier turn: the same false reply shipped with the backstop.
    agent.session.add_message("user", "reply to Ana that 10:30 works")
    agent.session.add_message("assistant", "**Sent to Ana** — confirming 10:30.\n\n" + MAIL_BACKSTOP)

    result = await agent.complete(nudge)

    assert mail.actions == ["search", "get_thread", "send"]
    assert result.startswith("**Sent to Ana** — confirming 10:30, threaded. Message ID m-1.")
    assert len(provider.calls) == 4 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_long_false_send_claim_is_not_exempt_as_a_deliverable(_human_turn):
    long_claim = (
        "Checking there's no existing reply first.\n\n**Sent to Luka**, CC Ana + Ivo.\n\n"
        "- **Format:** he presents, you interject with questions\n- **Language:** English "
        "confirmed\n- **List:** you'll follow up\n- Footer added per your rule\n\nID `1a2b3c4d5e6f7a8b`."
    )
    assert len(long_claim) > 200
    agent, provider, mail = _mail_agent([
        LLMResponse(content=long_claim),
        LLMResponse(content="Not sent yet — want me to send the answers to Luka now?"),
    ])

    result = await agent.complete("answer Luka's questions and add Ana and Ivo to the mail")

    assert mail.actions == []
    assert result.startswith("Not sent yet") and "1a2b3c4d5e6f7a8b" not in result
    corrective = _mail_correctives(agent)
    assert len(corrective) == 1
    assert "Also: no tool returned `1a2b3c4d5e6f7a8b`" in corrective[0]


@pytest.mark.asyncio
async def test_a_model_that_keeps_claiming_gets_the_backstop_line(_human_turn):
    from captain_claw.fabrication_guard import MAIL_BACKSTOP, MAX_FABRICATION_RETRIES

    agent, provider, mail = _mail_agent(
        [LLMResponse(content="**Sent** to Ana — confirming 10:30.")] * (MAX_FABRICATION_RETRIES + 1)
    )

    result = await agent.complete("send Ana the 10:30 confirmation by mail")

    assert mail.actions == []
    assert len(provider.calls) == MAX_FABRICATION_RETRIES + 1
    assert result.rstrip().endswith(MAIL_BACKSTOP)
    assert len(_mail_correctives(agent)) == MAX_FABRICATION_RETRIES
    assert agent.session.messages[-1]["content"].rstrip().endswith(MAIL_BACKSTOP)


@pytest.mark.asyncio
async def test_a_draft_only_request_gets_a_correction_not_a_send(_human_turn):
    agent, provider, mail = _mail_agent([
        LLMResponse(content="**Sent** to Ana — confirming 10:30."),
        LLMResponse(content="Not sent — the reply to Ana is drafted for your review."),
    ])

    result = await agent.complete(
        "Draft the reply to Ana confirming 10:30 but don't send it yet — I want to review it first.")

    assert result.startswith("Not sent — the reply to Ana is drafted")
    assert mail.actions == []
    assert len(_mail_correctives(agent)) == 1


@pytest.mark.asyncio
async def test_a_made_up_id_is_corrected_until_the_backstop(_human_turn):
    reply = "Confirmed with Roger — you'll attend. ID `1a2b3c4d5e6f7a8b`"
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)] * 3)

    result = await agent.complete("confirm Roger I'll attend")

    assert mail.actions == []
    correctives = _correctives(agent, "fabricated_id")
    assert len(correctives) == 2
    # The corrective quotes the ID back; that never makes it a known one.
    assert all("`1a2b3c4d5e6f7a8b`" in c["content"] for c in correctives)
    assert result.rstrip().endswith("Note: `1a2b3c4d5e6f7a8b` was not returned by any tool.")


@pytest.mark.asyncio
async def test_a_send_refused_as_a_repeat_backs_an_already_sent_reply(_human_turn):
    from captain_claw.fabrication_guard import claims_mail_sent

    request = "send Ana the 10:30 confirmation"
    reply = "**Sent** to Ana — no new copy was made."
    # The reply is one the gate reads as a claim: only the refusal backs it.
    assert claims_mail_sent(reply, request, True) is True
    agent, provider, mail = _mail_agent(
        [_send_call(), LLMResponse(content=reply)], tool=_StubMailTool(repeat="sent"),
    )

    result = await agent.complete(request)

    assert mail.actions == ["send"]
    assert result.startswith(reply)
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", ["draft", "thread_draft"])
async def test_a_send_refused_as_a_repeat_of_a_draft_backs_no_sent_claim(_human_turn, repeat):
    agent, provider, mail = _mail_agent([
        _send_call(),
        LLMResponse(content="Sent to Ana ✓"),
        LLMResponse(content="Not sent — that email is still waiting in your Drafts."),
    ], tool=_StubMailTool(repeat=repeat))
    _earlier_mail_lookup(agent)

    result = await agent.complete("looks good, send it")

    assert mail.actions == ["send"]
    assert result.startswith("Not sent — that email is still waiting in your Drafts.")
    corrective = _mail_correctives(agent)
    assert len(corrective) == 1
    # It quotes what the send answered — not what that answer says to do
    # next — and asks only for a true reply (_correctives checks).
    assert "did not go through" in corrective[0] and "repeat of" in corrective[0]
    assert "send_draft" not in corrective[0] and "allow_repeat" not in corrective[0]


@pytest.mark.asyncio
async def test_a_refused_send_gets_a_corrective_that_quotes_the_refusal(_human_turn):
    from captain_claw.fabrication_guard import MAIL_BACKSTOP_REFUSED

    refusal = "Email not sent: Email sending is off for this agent — create a draft instead."
    agent, provider, mail = _mail_agent(
        [_send_call()] + [LLMResponse(content="**Sent** to Ana — confirming 10:30.")] * 3,
        tool=_StubMailTool(refusal=refusal),
    )

    result = await agent.complete("send Ana the 10:30 confirmation by mail")

    assert mail.actions == ["send"]
    assert result.rstrip().endswith(MAIL_BACKSTOP_REFUSED)
    correctives = _mail_correctives(agent)
    assert len(correctives) == 2
    assert f'the tool answered: "Error: {refusal}" Nothing was sent.' in correctives[0]


@pytest.mark.asyncio
async def test_an_automated_turn_gets_the_same_correct_only_corrective(_human_turn):
    from captain_claw import mail_authority

    agent, provider, mail = _mail_agent([
        LLMResponse(content="**Sent** the reply to Ana — 10:30 confirmed."),
        LLMResponse(content="Nothing went out: the reply to Ana is still waiting for you."),
    ])

    with mail_authority.bound(mail_authority.automated("cron", "send Ana the weekly KPI email",
                                                       mail_write="allow")):
        result = await agent.complete("send Ana the weekly KPI email")

    assert mail.actions == []
    assert result.startswith("Nothing went out")
    corrective = _mail_correctives(agent)
    assert len(corrective) == 1 and corrective[0].startswith(_CORRECTIVE_LEAD)


@pytest.mark.asyncio
async def test_a_croatian_mail_claim_gets_the_mail_corrective_not_the_delegation_one(_human_turn):
    agent, provider, mail = _mail_agent([
        LLMResponse(content="Poslao sam mail Ani."),
        _send_call(),
        LLMResponse(content="Poslao sam mail Ani — Message ID m-1."),
    ])

    result = await agent.complete("pošalji mail Ani da 10:30 odgovara")

    assert mail.actions == ["send"]
    assert result.startswith("Poslao sam mail Ani — Message ID m-1.")
    correctives = [m["content"] for m in agent.session.messages
                   if m["role"] == "user" and m.get("origin") == "corrective"]
    assert len(correctives) == 1 and correctives[0].startswith(_CORRECTIVE_LEAD)


@pytest.mark.asyncio
async def test_a_croatian_report_after_a_real_send_is_not_a_handoff(_human_turn):
    agent, provider, mail = _mail_agent([
        _send_call(),
        LLMResponse(content="Mail je poslan Ani, čekam odgovor."),
    ])

    result = await agent.complete("pošalji mail Ani da 10:30 odgovara")

    assert result.startswith("Mail je poslan Ani, čekam odgovor.")
    assert len(provider.calls) == 2
    assert not any("delegated/sent the task to a peer" in str(m["content"])
                   for m in agent.session.messages)


@pytest.mark.asyncio
async def test_a_handoff_claimed_next_to_a_real_send_is_still_checked(_human_turn):
    agent, provider, mail = _mail_agent([
        _send_call(),
        LLMResponse(content="**Sent** to Ana — Message ID m-1.\n\n"
                            "Poslao sam zadatak Researcheru, čekam odgovor."),
        LLMResponse(content="Mail je poslan Ani (m-1); istraživanje radim sam."),
    ])

    await agent.complete("pošalji Ani mail i proslijedi istraživanje Researcheru")

    assert mail.actions == ["send"]
    delegation = [m for m in agent.session.messages
                  if m["role"] == "user" and "delegated/sent the task to a peer" in str(m["content"])]
    assert len(delegation) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_stall_after_the_corrective_is_still_a_stall(_human_turn):
    long_claim = "**Sent to Luka**, CC Ana + Ivo. " + "All four answers are in the mail. " * 6
    assert len(long_claim) > 200
    agent, provider, mail = _mail_agent([
        LLMResponse(content=long_claim),
        LLMResponse(content="I'll send it now."),
        _send_call(),
        LLMResponse(content="Email sent to Luka — Message ID m-1."),
    ])

    result = await agent.complete("answer Luka's questions by mail, send it")

    # The rejected claim is not a deliverable: the announce-and-stop reply
    # after it gets the stall retry.
    assert mail.actions == ["send"]
    assert result.startswith("Email sent to Luka")


class _StubFileSendTool(Tool):
    name = "whatsapp_send_file"
    description = "Send a file to the chat"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content="File sent to the WhatsApp chat.")


class _StubWriteTool(Tool):
    name = "write"
    description = "Write a file"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content="Written saved/one-pager.pdf (2 pages)")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool, call, request_text, reply", [
    (_StubFileSendTool(), ("whatsapp_send_file", {"path": "deck.pdf"}), "send me the deck", "Sent."),
    (_StubWriteTool(), ("write", {"path": "one-pager.pdf"}), "make me a one-pager on Q3",
     "**Sent** — the Q3 one-pager (2 pages)."),
    (_StubActionTool("flight_deck", "Delegated to Researcher (task t-1)."), ("flight_deck", {}),
     "have the Researcher look into Q3 pricing",
     "**Sent** to the Researcher — I'll post the findings when they're back."),
    (_StubActionTool("image_gen", "Image saved to saved/media/cat.png"), ("image_gen", {}),
     "pošalji mi sliku mačke", "Evo, poslano! 🐱"),
    (_StubActionTool("google_calendar", "Event updated: Sync with Ana, 15:00."),
     ("google_calendar", {"action": "update_event"}),
     "move the sync with Ana to 3pm and let everyone know", "Moved to 15:00. **Sent** to Ana and Ivo."),
])
async def test_another_delivery_this_turn_owns_a_bare_sent(_human_turn, tool, call, request_text,
                                                          reply):
    agent, provider, mail = _mail_agent([_call(call[0], **call[1]), LLMResponse(content=reply)])
    agent.tools.register(tool)
    _earlier_mail_lookup(agent)

    result = await agent.complete(request_text)

    assert result.startswith(reply) and "no email was sent" not in result
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_file_from_an_email_sent_to_the_chat_is_not_an_email(_human_turn):
    agent, provider, mail = _mail_agent([
        _mail_call("get_attachment", message_id="m-5", attachment="1"),
        _call("whatsapp_send_file", path="saved/downloads/s1/invoice.pdf"),
        LLMResponse(content="Sent ✅"),
    ])
    agent.tools.register(_StubFileSendTool())

    result = await agent.complete("send me the attachment from Ana's email")

    assert result.startswith("Sent ✅")
    assert mail.actions == ["get_attachment"]
    assert len(provider.calls) == 3 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_an_inbox_digest_is_not_a_send_claim(_human_turn):
    from captain_claw import mail_authority

    digest = (
        "Inbox, last 24h:\n1. Marko: Sent the signed contract, asks for a countersignature.\n"
        "2. **Ana** wrote: \"Sent you the deck.\"\n- **Ivo**: I've sent the signed NDA back.\n"
        "- **Calendly**: Confirmation email sent to Ana for Friday 10:00.\n"
        "- **Mail Delivery Subsystem** — the email sent to ana@x.co bounced\n"
        "- **Mailchimp**: Your campaign email was sent to 1,204 subscribers\n"
        "- Ivo: mail je poslan krivoj adresi\n\n### Sent\n- Ana: deck\n\n"
        "**Sent today:** 3 emails\nReplied (2): Ana, Ivo"
    )
    agent, provider, mail = _mail_agent([
        _mail_call("search", call_id="s0", query="newer_than:1d"),
        LLMResponse(content=digest),
    ])

    with mail_authority.bound(mail_authority.automated("cron", "Summarize my inbox")):
        result = await agent.complete("Summarize my inbox")

    assert result == digest
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_true_report_of_an_earlier_send_is_not_questioned(_human_turn):
    # Turn 1 really sends; later turns mention that email — with its Message
    # ID, or dated before the turn.
    agent, provider, mail = _mail_agent([
        _send_call(),
        LLMResponse(content="**Sent** to Ana — Message ID m-1."),
        LLMResponse(content="Here's the deck link: https://x.co/deck. The email to Ana went out "
                            "(Message ID m-1)."),
        LLMResponse(content="Evo ugovora. The email to Ana already went out this morning."),
    ])

    await agent.complete("send Ana the 10:30 confirmation")
    second = await agent.complete("send me the deck link")
    third = await agent.complete("pošalji mi ugovor")

    assert mail.actions == ["send"]
    assert len(provider.calls) == 4
    assert second.startswith("Here's the deck link") and third.startswith("Evo ugovora")
    assert _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id, backed", [("m-1", True), ("m-0", False)])
async def test_only_a_message_id_a_send_returned_backs_a_claim(_human_turn, message_id, backed):
    # m-1 came back from an earlier "Email sent."; m-0 only from a search.
    claim = f"Yes — the reply to Ana went out, Message ID {message_id}."
    agent, provider, mail = _mail_agent([
        _send_call(), LLMResponse(content="Email sent to Ana — Message ID m-1."),
        _mail_call("search", query="in:sent to:ana@x.co"),
        LLMResponse(content=claim), LLMResponse(content="It went out on 12 Oct (m-0)."),
    ])
    await agent.complete("send Ana the 10:30 confirmation")

    result = await agent.complete("did the reply to Ana go out?")

    assert mail.actions == ["send", "search"]
    assert result.startswith(claim if backed else "It went out on 12 Oct (m-0).")
    assert len(_mail_correctives(agent)) == (0 if backed else 1)


@pytest.mark.asyncio
async def test_the_users_own_send_is_acknowledged_without_a_corrective(_human_turn):
    from captain_claw.fabrication_guard import claims_mail_sent

    request = "I sent your draft to Richard and Ted"
    reply = "OK, the email to Ted was sent. Nothing else is waiting on you."
    # Worded like a claim: only the user's own send makes it an acknowledgement.
    assert claims_mail_sent(reply, request, True) is True
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)])
    _earlier_mail_lookup(agent)

    result = await agent.complete(request)

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_an_invented_id_next_to_a_real_send_is_corrected(_human_turn):
    agent, provider, mail = _mail_agent([
        _send_call(),
        LLMResponse(content="**Sent** both — Ana (ID m-1) and Marko (ID 1a2b3c4d5e6f7a8b)."),
        LLMResponse(content="Sent to Ana (Message ID m-1). The one to Marko did not go out."),
    ])

    result = await agent.complete("send the replies to Ana and Marko")

    assert mail.actions == ["send"]
    assert result.startswith("Sent to Ana (Message ID m-1).")
    corrective = _correctives(agent)
    assert [m["origin_detail"] for m in corrective] == ["fabricated_id"]
    assert corrective[0]["content"].startswith("Your reply quotes `1a2b3c4d5e6f7a8b`")


@pytest.mark.asyncio
async def test_the_models_own_words_this_turn_never_vouch_for_an_id(_human_turn):
    # An ID it wrote in its narration and then put on the "Sent" line was
    # still returned by no tool — the send this turn returned another one.
    agent, provider, mail = _mail_agent([
        _send_call(content="Checking ID 1a2b3c4d5e6f7a8b first."),
        LLMResponse(content="**Sent** to Ana — ID 1a2b3c4d5e6f7a8b."),
        LLMResponse(content="**Sent** to Ana — Message ID m-1."),
    ])

    result = await agent.complete("send Ana the 10:30 confirmation")

    assert mail.actions == ["send"]
    assert result.endswith("**Sent** to Ana — Message ID m-1.")
    corrective = _correctives(agent)
    assert len(corrective) == 1 and "`1a2b3c4d5e6f7a8b`" in corrective[0]["content"]


@pytest.mark.asyncio
async def test_a_send_claimed_next_to_another_call_is_held_and_checked(_human_turn):
    # The claim rides along a calendar call. It is never narrated, folded
    # into the answer or stored with the call.
    claim = "**Sent** to Roger — you'll attend.\n\nAdding the calendar blocker."
    agent, provider, mail = _mail_agent([
        _call("google_calendar", content=claim, action="create_event"),
        LLMResponse(content="Blocker added for Friday 10:00. The confirmation to Roger was not "
                            "sent — want me to send it?"),
    ])
    calendar = _StubActionTool("google_calendar", "Event created: Blocker, Friday 10:00.")
    agent.tools.register(calendar)
    narrated: list[str] = []
    agent.narration_callback = lambda text, iteration: narrated.append(text)

    result = await agent.complete("confirm Roger by mail I'll attend + calendar blocker")

    assert calendar.calls == 1 and mail.actions == []
    assert result.startswith("Blocker added") and "**Sent** to Roger" not in result
    assert narrated == []
    # The call is stored without the claim; the claim as rejected, then the
    # corrective, after the call's result.
    msgs = agent.session.messages
    assert next(m for m in msgs if m.get("tool_calls"))["content"] == ""
    result_at = next(i for i, m in enumerate(msgs) if m.get("tool_name") == "google_calendar")
    rejected_at = next(i for i, m in enumerate(msgs) if m.get("origin") == "rejected")
    assert msgs[rejected_at]["content"] == claim
    assert result_at < rejected_at and msgs[rejected_at + 1].get("origin") == "corrective"
    assert len(_mail_correctives(agent)) == 1


@pytest.mark.asyncio
async def test_a_send_claimed_next_to_its_own_send_stands_once_it_went_out(_human_turn):
    agent, provider, mail = _mail_agent([
        _send_call(content="**Sent** to Ana — confirming 10:30."),
        LLMResponse(content="Message ID m-1."),
    ])
    narrated: list[str] = []
    agent.narration_callback = lambda text, iteration: narrated.append(text)

    await agent.complete("send Ana the 10:30 confirmation by mail")

    assert mail.actions == ["send"]
    # Held while the send ran (stored without it), shown once the send
    # answered "Email sent.", and never questioned.
    assert next(m for m in agent.session.messages if m.get("tool_calls"))["content"] == ""
    assert narrated == ["**Sent** to Ana — confirming 10:30."]
    assert _correctives(agent) == []


@pytest.mark.asyncio
async def test_the_iteration_budget_salvage_never_ships_a_rejected_claim(_human_turn):
    # The gate fires on the last allowed iteration; the salvage must not ship
    # the rejected reply, and what it does ship still claiming a send gets the
    # backstop line.
    from captain_claw.fabrication_guard import MAIL_BACKSTOP

    false_claim = ("**Sent** to Ana — confirming 10:30, with the agenda and the room details "
                   "attached. Message ID `1a2b3c4d5e6f7a8b`.")
    assert len(false_claim) > 100
    search = _mail_call("search", query="in:sent to:ana@x.co")
    agent, provider, mail = _mail_agent(
        [search, search, search, LLMResponse(content=false_claim),
         LLMResponse(content="**Sent** to Ana — confirming 10:30.")],
        tool=_StubMailTool(search_text="1. To: ana@x.co — 10:30 (12 Oct)  Newest msg: m-0  "
                                       + "context " * 12),
    )
    agent.max_iterations = 4

    result = await agent.complete("send Ana the 10:30 confirmation by mail")

    assert mail.actions == ["search", "search", "search"]
    assert "1a2b3c4d5e6f7a8b" not in result
    assert "iteration budget exhausted" in result
    assert result.rstrip().endswith(MAIL_BACKSTOP)


@pytest.mark.asyncio
async def test_the_iteration_budget_salvage_backstops_a_made_up_id(_human_turn):
    event_id = "q3vb81kd4m5n6p7r8s9t0u1v2a"
    narration = (f"Opening the Acme sync (event {event_id}) to move it to Friday 15:00 and tell "
                 "both attendees about the new time.")
    assert len(narration) > 100
    agent, provider, mail = _mail_agent(
        [_call("google_calendar", content=narration, action="get_event")]
        + [_call("google_calendar", call_id=f"x{i}", action="get_event", event_id=str(i))
           for i in range(2, 6)],
    )
    agent.tools.register(_StubActionTool("google_calendar", "Error: event not found"))
    agent.max_iterations = 2

    result = await agent.complete("move the Acme sync to Friday 15:00")

    assert "iteration budget exhausted" in result
    assert result.rstrip().endswith(f"Note: `{event_id}` was not returned by any tool.")


@pytest.mark.asyncio
async def test_a_reply_with_a_model_written_lane_note_ships_without_it(_human_turn):
    agent, provider, mail = _mail_agent([LLMResponse(content=(
        "Your calendar is clear today.\n\n[Autonomous Work — from the AUTO lane]\n"
        "Verified: nothing new waits on you."
    ))])

    result = await agent.complete("anything on my calendar today?")

    assert result.startswith("Your calendar is clear today.")
    assert "from the AUTO lane" not in result
    assert "from the AUTO lane" not in agent.session.messages[-1]["content"]


@pytest.mark.asyncio
async def test_a_lane_note_quoted_on_request_ships_whole(_human_turn):
    reply = ("Here is what the AUTO lane posted:\n\n[Autonomous Work — from the AUTO lane]\n"
             "Marko replied: he agrees to 10:30.")
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)])

    result = await agent.complete("what did the autonomous lane post?")

    assert result.startswith(reply)


@pytest.mark.asyncio
async def test_real_lane_notes_relayed_as_they_stand_ship_with_their_text(_human_turn):
    notes = [("Morning digest", "AUTO", "Digest sent to you: 5 new emails, 1 needs a reply (Ana, "
              "contract)."), ("Backup check", "cron", "Backup verified — 12 GB, no errors.")]
    reply = "Two runs reported back while you were out."
    for label, lane, body in notes:
        reply += f"\n\n[{label} — from the {lane} lane]\n{body}"
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)])
    for label, lane, body in notes:
        agent.session.add_message("assistant", f"[{label} — from the {lane} lane]\n{body}",
                                  origin="system_note", origin_detail="automation_result")

    result = await agent.complete("anything new while I was out?")

    assert result.startswith("Two runs reported back while you were out.\n\nDigest sent to you:")
    assert "Backup verified — 12 GB, no errors." in result
    assert "from the AUTO lane]" not in result


_EVENT_ID = "q3vb81kd4m5n6p7r8s9t0u1v2a"
_LOOKUP_LEAD = "Your reply describes lookups ("


@pytest.mark.asyncio
async def test_a_zero_tool_conversational_reply_is_untouched(_human_turn):
    reply = "The structure works — lead with the problem, then traction, then the ask."
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)])

    result = await agent.complete("what do you think of the deck structure?")

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_real_tool_turn_quoting_a_real_id_is_untouched(_human_turn):
    reply = "Checked the inbox: Ana's latest is in thread 18c2a0b1c2d3e4f5 — she confirms 10:30."
    agent, provider, mail = _mail_agent(
        [_mail_call("search", query="from:ana"), LLMResponse(content=reply)],
        tool=_StubMailTool(search_text="1. Re: 10:30  Thread: 18c2a0b1c2d3e4f5  |  Newest msg: "
                                       "18c2f4a6b8d0e1f3"),
    )

    result = await agent.complete("did Ana answer about 10:30?")

    assert mail.actions == ["search"]
    assert result.startswith(reply)
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_ledger", [True, False])
async def test_an_id_from_a_compacted_away_result_is_known_via_the_ledger(_human_turn, keep_ledger):
    from captain_claw.fabrication_guard import ID_LEDGER_KEY

    reply = f"The Acme sync (event {_EVENT_ID}) is on Friday at 15:00."
    agent, provider, mail = _mail_agent([
        _call("google_calendar", action="create_event"),
        LLMResponse(content="Booked the Acme sync for Friday 15:00."),
        LLMResponse(content=reply),
        LLMResponse(content="The Acme sync is on Friday at 15:00."),
    ])
    agent.tools.register(_StubActionTool("google_calendar", f"Event created: Acme sync\n  ID: {_EVENT_ID}"))
    await agent.complete("book the Acme sync for Friday 15:00")
    # Compaction: the tool result is gone, the summary does not name the ID.
    agent.session.messages = [{
        "role": "assistant", "content": "Conversation summary of earlier messages (compacted "
        "memory):\nThe user booked the Acme sync for Friday 15:00.", "tool_name": "compaction_summary",
        "origin": "system_note", "origin_detail": "compaction",
    }]
    if not keep_ledger:
        agent.session.metadata.pop(ID_LEDGER_KEY)

    result = await agent.complete("when is the Acme sync?")

    if keep_ledger:
        assert result.startswith(reply)
        assert len(provider.calls) == 3 and _correctives(agent) == []
    else:
        assert result.startswith("The Acme sync is on Friday at 15:00.")
        correctives = _correctives(agent)
        assert len(correctives) == 1 and correctives[0]["origin_detail"] == "fabricated_id"
        assert correctives[0]["content"].startswith(f"Your reply quotes `{_EVENT_ID}`")


@pytest.mark.asyncio
async def test_an_id_the_request_gave_the_model_is_known(_human_turn):
    # Recalled memory / intentions ride in the request, not in the session.
    reply = f"Your intention to follow up on the Acme sync (event {_EVENT_ID}) is still open."
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)] * 3)
    build = agent._build_messages

    def _with_recall(*args: Any, **kwargs: Any) -> list[Message]:
        messages = build(*args, **kwargs)
        return [messages[0], Message(role="system", content=f"Intentions: follow up on {_EVENT_ID}"),
                *messages[1:]]

    agent._build_messages = _with_recall

    result = await agent.complete("anything open on Acme?")

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_narrated_lookup_in_a_turn_with_no_tool_call_is_corrected(_human_turn):
    from captain_claw.fabrication_guard import LOOKUP_BACKSTOP

    narrated = "Checking today's calendar.\n\n**Today: Ops 09:30 → IC 10:00.**"
    agent, provider, mail = _mail_agent([
        LLMResponse(content=narrated),
        _call("google_calendar", action="list_events"),
        # Narrating the lookup it just made is the truth.
        LLMResponse(content="Checked the calendar — today: Ops 09:30, nothing else."),
    ])
    agent.tools.register(_StubActionTool("google_calendar", "Today: Ops 09:30"))

    result = await agent.complete("what's on today?")

    assert result.startswith("Checked the calendar — today: Ops 09:30")
    assert LOOKUP_BACKSTOP not in result
    correctives = _correctives(agent)
    assert len(correctives) == 1 and correctives[0]["origin_detail"] == "narrated_lookup"
    assert correctives[0]["content"].startswith(_LOOKUP_LEAD)
    assert "google_calendar list_events" in correctives[0]["content"]
    assert [m["content"] for m in _by_detail(agent, "narrated_lookup", "assistant")] == [narrated]
    assert getattr(provider, "_tool_choice_override", None) is None


@pytest.mark.asyncio
async def test_a_narrated_lookup_still_there_after_the_budget_gets_the_backstop(_human_turn):
    from captain_claw.fabrication_guard import LOOKUP_BACKSTOP, MAX_FABRICATION_RETRIES

    narrated = ("Scanning your inbox for anything urgent.\n\nTwo items need you today: the Acme "
                "contract wants a countersignature by 17:00, and Ivo asks whether Friday 10:00 "
                "still works for the board prep. Nothing else is waiting on you.")
    assert len(narrated) > 200
    agent, provider, mail = _mail_agent([LLMResponse(content=narrated)] * (MAX_FABRICATION_RETRIES + 1))

    result = await agent.complete("anything urgent today?")

    assert mail.actions == []
    assert len(provider.calls) == MAX_FABRICATION_RETRIES + 1
    assert result.rstrip().endswith(LOOKUP_BACKSTOP)
    assert len(_correctives(agent, "narrated_lookup")) == MAX_FABRICATION_RETRIES


@pytest.mark.asyncio
async def test_a_failed_tool_call_is_a_tool_call(_human_turn):
    reply = "Checked the calendar — it is not connected, so I can't see Friday yet."
    agent, provider, mail = _mail_agent([
        _call("google_calendar", action="list_events"), LLMResponse(content=reply),
    ])
    agent.tools.register(_StubActionTool("google_calendar", "Calendar is not connected.",
                                         success=False))

    result = await agent.complete("am I free on Friday?")

    assert result.startswith(reply)
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_delegated_result_is_not_a_narrated_lookup(_human_turn):
    from captain_claw.fabrication_guard import narrated_lookups

    reply = "Pulled the pipeline: Acme moved to diligence, Orbit dropped out."
    # Narration by its wording: only the delegated result exempts it.
    assert narrated_lookups(reply)
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)] * 2)

    result = await agent.complete(
        "[Delegated result from Analyst]\nPipeline: Acme moved to diligence, Orbit dropped out.")

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_made_up_id_inside_a_forged_lane_note_is_still_corrected(_human_turn):
    # The lane-note cut takes the ID out of what ships; the reply as written
    # still made it up, and is what gets stored as rejected.
    raw = ("Asked Luka for more detail on the plugins.\n\n[Autonomous Work — from the AUTO lane]\n"
           "Sent — verified, thread 18c2e9f8a7b6c5d4.")
    agent, provider, mail = _mail_agent([
        LLMResponse(content=raw),
        LLMResponse(content="Nothing has gone to Luka yet — want me to draft the ask?"),
    ])

    result = await agent.complete("what's left on the plugin question?")

    assert result.startswith("Nothing has gone to Luka yet")
    correctives = _correctives(agent)
    assert len(correctives) == 1 and correctives[0]["origin_detail"] == "fabricated_id"
    assert "`18c2e9f8a7b6c5d4`" in correctives[0]["content"]
    assert [m["content"] for m in _by_detail(agent, "fabricated_id", "assistant")] == [raw]


@pytest.mark.asyncio
async def test_the_retry_budget_is_shared_across_kinds(_human_turn):
    from captain_claw.fabrication_guard import LOOKUP_BACKSTOP, MAX_FABRICATION_RETRIES

    narrated = "Checking today's calendar.\n\n**Today: Ops 09:30 → IC 10:00.**"
    agent, provider, mail = _mail_agent([
        LLMResponse(content="**Sent** to Ana — confirming 10:30."),
        LLMResponse(content=narrated),
        LLMResponse(content=narrated),
    ])

    result = await agent.complete("send Ana the 10:30 confirmation by mail")

    assert MAX_FABRICATION_RETRIES == 2
    assert len(provider.calls) == 3
    assert [m["origin_detail"] for m in _correctives(agent)] == [
        "false_mail_send", "narrated_lookup"]
    assert result.rstrip().endswith(LOOKUP_BACKSTOP)


@pytest.mark.asyncio
async def test_a_turn_offered_no_tool_is_not_judged(_human_turn):
    reply = f"Checked the calendar — the Acme sync (event {_EVENT_ID}) is on Friday at 15:00."
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)] * 2)
    agent.tools = ToolRegistry()

    result = await agent.complete("when is the Acme sync?")

    assert result.startswith(reply)
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_a_traced_turn_is_judged_like_any_other(_human_turn):
    # With LLM tracing on, each response is also stored as an llm_trace tool
    # row: it is neither a tool call nor a source of known IDs.
    made_up = "Confirmed with Roger — you'll attend. ID `1a2b3c4d5e6f7a8b`"
    narrated = "Checking today's calendar.\n\n**Today: Ops 09:30 → IC 10:00.**"
    agent, provider, mail = _mail_agent([
        LLMResponse(content=made_up),
        LLMResponse(content=narrated),
        LLMResponse(content="Nothing went to Roger yet, and I haven't looked at today."),
    ])
    agent.monitor_trace_llm = True
    _earlier_mail_lookup(agent)

    result = await agent.complete("confirm Roger I'll attend")

    assert any(m.get("tool_name") == "llm_trace" for m in agent.session.messages)
    assert result.startswith("Nothing went to Roger yet")
    assert [m["origin_detail"] for m in _correctives(agent)] == ["fabricated_id", "narrated_lookup"]


def test_a_turn_with_any_tool_attempt_made_a_tool_call():
    agent, provider, mail = _mail_agent([])
    agent.session.add_message("user", "earlier", origin="human")
    agent.session.add_message("tool", "old result", tool_call_id="o1", tool_name="google_mail")
    start = len(agent.session.messages)
    agent.session.add_message("user", "what's on today?", origin="human")
    agent._malformed_retry_count = 0
    assert agent._turn_made_no_tool_call(start) is True
    # A monitor row is no tool call.
    agent.session.add_message("tool", "[assistant_response]\n…", tool_name="llm_trace")
    assert agent._turn_made_no_tool_call(start) is True
    # A call cut off before it was stored.
    agent._malformed_retry_count = 1
    assert agent._turn_made_no_tool_call(start) is False
    agent._malformed_retry_count = 0
    # A code-run tool's result (a scheduled job's shell step).
    agent.session.add_message("tool", "done", tool_name="shell")
    assert agent._turn_made_no_tool_call(start) is False


@pytest.mark.asyncio
async def test_a_send_compaction_dropped_still_backs_its_message_id(_human_turn):
    claim = "Yes — the reply to Ana went out, Message ID m-1."
    agent, provider, mail = _mail_agent([
        _send_call(),
        LLMResponse(content="Email sent to Ana — Message ID m-1."),
        LLMResponse(content=claim),
        LLMResponse(content=claim),
    ])
    await agent.complete("send Ana the 10:30 confirmation")
    agent.session.messages = [{
        "role": "assistant", "content": "Conversation summary of earlier messages (compacted "
        "memory):\nThe user asked for a 10:30 confirmation to Ana.", "tool_name": "compaction_summary",
        "origin": "system_note", "origin_detail": "compaction",
    }]

    result = await agent.complete("did the reply to Ana go out?")

    assert mail.actions == ["send"]
    assert result.startswith(claim)
    assert len(provider.calls) == 3 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_send_from_before_the_ledger_still_backs_its_message_id(_human_turn):
    reply = "Yes — the reply to Ana went out, Message ID m-9."
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)] * 3)
    # Stored before the ID ledger existed: only the session holds the result.
    agent.session.messages += [
        {"role": "assistant", "content": "", "origin": "model", "tool_calls": [{
            "id": "c9", "type": "function",
            "function": {"name": "google_mail", "arguments": "{\"action\": \"send\"}"}}]},
        {"role": "tool", "tool_call_id": "c9", "tool_name": "google_mail", "origin": "tool",
         "tool_arguments": {"action": "send"}, "content": "Email sent.\n  Message ID: m-9"},
    ]

    result = await agent.complete("did the reply to Ana go out?")

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_send_a_peer_relays_is_not_questioned(_human_turn):
    reply = "Email sent to Ana via the Mailer agent — Message ID 18c2f3a4b5d6e7f1."
    agent, provider, mail = _mail_agent([
        _call("flight_deck", action="consult"), LLMResponse(content=reply)])
    agent.tools.register(_StubActionTool(
        "flight_deck", "Mailer replied: Email sent.\n  To: ana@x.co\n  Message ID: 18c2f3a4b5d6e7f1"))

    result = await agent.complete("Ask the Mailer agent to send the email to Ana confirming 10:30.")

    assert result.startswith(reply)
    assert mail.actions == [] and len(provider.calls) == 2
    assert _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ledgered", [True, False])
async def test_a_send_a_lane_note_relays_is_not_questioned(_human_turn, ledgered):
    reply = "Yes — the reply to Ana went out, Message ID 18c2bb00cc11dd22."
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)] * 3)
    note = {"role": "assistant", "origin": "system_note", "origin_detail": "automation_result",
            "content": "[Autonomous Work — from the AUTO lane]\nEmail sent. To: ana@x.co — Message "
                       "ID 18c2bb00cc11dd22"}
    if ledgered:
        agent.session.add_message(note["role"], note["content"], origin=note["origin"],
                                  origin_detail=note["origin_detail"])
    else:
        # A session from before the ID ledger: the note alone relays the send.
        agent.session.messages.append(note)

    result = await agent.complete("did the reply to Ana go out?")

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


class _EchoMailTool(_StubMailTool):
    """google_mail whose search names its query back, as the real one does
    ("No threads found for: <query>")."""

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        if action == "search":
            self.actions.append(action)
            return ToolResult(success=True, content=f"No threads found for: {kwargs.get('query', '')}")
        return await super().execute(action=action, **kwargs)


@pytest.mark.asyncio
async def test_a_search_naming_a_made_up_id_back_does_not_vouch_for_it(_human_turn):
    from captain_claw.fabrication_guard import ID_LEDGER_KEY

    made_up = "18c2e9f8a7b6c5d4"
    agent, provider, mail = _mail_agent([
        LLMResponse(content=f"Yes — **Sent** to Luka, thread `{made_up}`."),
        _mail_call("search", query=f"in:sent {made_up}"),
        LLMResponse(content=f"It already went out earlier — thread `{made_up}`."),
        LLMResponse(content="No — nothing went to Luka; I found no reply in Sent."),
    ], tool=_EchoMailTool())

    result = await agent.complete("did you reply to Luka?")

    assert mail.actions == ["search"]
    assert result.startswith("No — nothing went to Luka")
    correctives = _correctives(agent)
    assert len(correctives) == 2
    assert all(f"`{made_up}`" in c["content"] for c in correctives)
    assert made_up not in str(agent.session.metadata.get(ID_LEDGER_KEY))


@pytest.mark.asyncio
async def test_a_compaction_summary_never_vouches_for_a_reply_it_quotes(_human_turn):
    made_up = "18c2e9f8a7b6c5d4"
    agent, provider, mail = _mail_agent([
        LLMResponse(content=f"The thread is `{made_up}`."),
        LLMResponse(content="I don't have Ana's thread ID at hand — want me to look it up?"),
    ])
    # The digest of a turn whose reply (shipped with the backstop) made it up.
    agent.session.messages = [{
        "role": "assistant", "tool_name": "compaction_summary", "origin": "system_note",
        "origin_detail": "compaction",
        "content": "Conversation summary of earlier messages (compacted memory):\nDigest of the "
        "earlier part of this conversation:\nBy topic:\n- Mail [mail] — 1 exchange(s), 2026-10-12\n"
        "  last asked: \"did Ana reply?\"\n  last reply: \"Ana replied on Monday — thread "
        f"`{made_up}`. Note: `{made_up}` was not returned by any tool.\"",
    }]

    result = await agent.complete("what's Ana's thread?")

    assert result.startswith("I don't have Ana's thread ID at hand")
    correctives = _correctives(agent)
    assert len(correctives) == 1 and f"`{made_up}`" in correctives[0]["content"]


@pytest.mark.asyncio
async def test_a_summary_sent_inside_the_request_never_vouches_for_a_reply_it_quotes(_human_turn):
    made_up = "18c2e9f8a7b6c5d4"
    agent, provider, mail = _mail_agent([
        LLMResponse(content=f"The thread is `{made_up}`."),
        LLMResponse(content="I don't have Ana's thread ID at hand."),
    ])
    build = agent._build_messages

    def _with_summary(*args: Any, **kwargs: Any) -> list[Message]:
        messages = build(*args, **kwargs)
        return [messages[0], Message(role="user", content=(
            "Conversation summary of earlier messages (compacted memory):\nBy topic:\n"
            f"  last reply: \"Ana replied on Monday — thread {made_up}.\"")), *messages[1:]]

    agent._build_messages = _with_summary

    result = await agent.complete("what's Ana's thread?")

    assert result.startswith("I don't have Ana's thread ID at hand.")
    assert len(_correctives(agent)) == 1


def test_a_turned_down_reply_and_its_corrective_never_feed_a_summary():
    agent, provider, mail = _mail_agent([])
    made_up = "18c2e9f8a7b6c5d4"
    messages = [
        {"role": "user", "content": "did Ana reply?", "origin": "human"},
        {"role": "assistant", "content": f"Yes, thread {made_up}.", "origin": "rejected"},
        {"role": "user", "content": f"Your reply quotes `{made_up}` — invented.",
         "origin": "corrective"},
        {"role": "assistant", "content": "I don't know yet.", "origin": "model"},
    ]
    formatted = agent._format_compaction_messages(messages)
    fallback = agent._fallback_compaction_summary(messages)
    for text in (formatted, fallback):
        assert made_up not in text and "I don't know yet." in text and "did Ana reply?" in text


@pytest.mark.asyncio
async def test_a_true_report_of_an_earlier_send_is_asked_for_its_id_not_retracted(_human_turn):
    # Turn 2 checks the Sent folder, as the prompts ask, and reports turn 1's
    # send without its Message ID: the corrective never says nothing was sent.
    agent, provider, mail = _mail_agent([
        _send_call(), LLMResponse(content="**Sent** to Ana — Message ID m-1."),
        _mail_call("search", query="in:sent to:ana@x.co"),
        LLMResponse(content="Yes — I looked in your Sent folder.\n\nThe email to Ana went out today "
                            "at 10:31."),
        LLMResponse(content="Yes — the email to Ana went out today at 10:31 (Message ID m-1)."),
    ], tool=_StubMailTool(search_text="1. To: ana@x.co — 10:30 — 13 Oct 10:31  Message ID: m-1"))
    await agent.complete("send Ana a mail that 10:30 works")

    result = await agent.complete("are you sure it went out? check")

    assert mail.actions == ["send", "search"]
    assert result.startswith("Yes — the email to Ana went out today at 10:31 (Message ID m-1).")
    assert "no email was sent" not in result and len(provider.calls) == 5
    corrective = _mail_correctives(agent)
    assert len(corrective) == 1 and "nothing was sent" not in corrective[0].lower()


@pytest.mark.asyncio
async def test_flight_decks_duplicate_refusal_backs_an_already_sent_reply(_human_turn):
    refusal = ("Email not sent: Duplicate: this exact email (same recipients, subject and body) "
               "was already sent at 2026-10-13T09:12 (Gmail message 18f3a1b2c3d4e5f6). Not sent "
               "again.")
    reply = "The email to Ana was already sent at 09:12, so there is no second copy."
    agent, provider, mail = _mail_agent([_send_call(), LLMResponse(content=reply)],
                                        tool=_StubMailTool(refusal=refusal))

    result = await agent.complete("send Ana the 10:30 confirmation")

    assert mail.actions == ["send"] and result.startswith(reply)
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_peers_send_relayed_in_its_own_words_is_not_questioned(_human_turn):
    reply = ("Mailer handled it.\n\n**Sent** to Ana — confirming the contract (Message ID "
             "18f3a1b2c3d4e5f6).")
    agent, provider, mail = _mail_agent([
        _call("flight_deck", action="consult", message="email Ana the contract"),
        LLMResponse(content=reply),
    ])
    agent.tools.register(_StubActionTool("flight_deck", (
        "Response from Mailer:\n\n**Sent** to Ana — confirming the contract. Message ID "
        "18f3a1b2c3d4e5f6.")))

    result = await agent.complete("ask Mailer to email Ana the contract")

    assert result.startswith(reply) and "no email was sent" not in result
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text, reply", [
    ("Ask the Researcher agent to summarize Ana's email thread about the contract",
     "**Sent** to Researcher — they'll summarize Ana's thread and report back."),
    ("ask the Researcher to email Ana the summary",
     "**Sent** to the Researcher — they'll email Ana the summary."),
])
async def test_a_handoff_about_mail_is_not_an_email_send(_human_turn, request_text, reply):
    agent, provider, mail = _mail_agent([
        _call("flight_deck", action="delegate", agent_name="Researcher", message="the summary"),
        LLMResponse(content=reply),
    ])
    agent.tools.register(_StubActionTool("flight_deck", "Task delegated to **Researcher**."))
    _earlier_mail_lookup(agent)

    result = await agent.complete(request_text)

    assert result.startswith(reply) and "no email was sent" not in result
    assert len(provider.calls) == 2 and _correctives(agent) == []
    assert not any("delegated/sent the task to a peer" in str(m["content"])
                   for m in agent.session.messages)


@pytest.mark.asyncio
async def test_a_drafted_letter_is_not_a_send_claim(_human_turn):
    reply = ("Draft saved in Gmail. It reads:\n\nHi Marko,\n\nI've just sent the signed contract "
             "over by courier — it should reach you Thursday.\n\nBest,\nK")
    agent, provider, mail = _mail_agent([
        _mail_call("create_draft", to="marko@x.co", body="Hi Marko, …"),
        LLMResponse(content=reply),
    ])

    result = await agent.complete("Draft a reply to Marko letting him know the signed contract "
                                  "is on its way by courier")

    assert mail.actions == ["create_draft"] and result.startswith(reply)
    assert len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_proofreading_pasted_text_is_no_narrated_lookup(_human_turn):
    request = ("Can you proofread this before I send it to Marko?\n\nHi Marko, I wanted to follow "
               "up on our call last week about the pilot. Could you send over the signed NDA and "
               "the pricing sheet by Friday, so legal can start the review? Thanks, K")
    reply = ("Reading through your email now — here are my suggestions:\n\n1. Shorten the "
             "opening.\n2. Put the ask about the NDA in the first paragraph.")
    agent, provider, mail = _mail_agent([LLMResponse(content=reply)])

    result = await agent.complete(request)

    assert result.startswith(reply)
    assert len(provider.calls) == 1 and _correctives(agent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text, call, refusal", [
    ("one line ack to Tom", _mail_call(
        "get_thread", thread_id="t-1",
        content="Finding Tom's latest message.\n\n**Sent** — one-line ack to Tom."), ""),
    ("send Ana the 10:30 confirmation", _send_call(content="**Sent** to Ana — 10:30 confirmed."),
     "Email not sent: Email sending is off for this agent."),
])
async def test_a_send_claim_next_to_a_call_is_judged_once_the_call_is_stored(
        _human_turn, request_text, call, refusal):
    # A fresh session, no mail word in the request: only this response's own
    # call makes the bare "Sent" an email, and it is judged once stored.
    agent, provider, mail = _mail_agent([call, LLMResponse(content="Not sent yet.")],
                                        tool=_StubMailTool(refusal=refusal))

    result = await agent.complete(request_text)

    assert result.startswith("Not sent yet.") and "**Sent**" not in result
    corrective = _mail_correctives(agent)
    assert len(corrective) == 1 and ("did not go through" in corrective[0]) is bool(refusal)


@pytest.mark.asyncio
async def test_a_follow_up_quoting_the_threads_id_is_no_send(_human_turn):
    # Gmail gives a new thread its first message's ID: turn 1's Message ID is
    # also the Thread ID turn 2's false follow-up quotes.
    agent, provider, mail = _mail_agent([
        _send_call(), LLMResponse(content="**Sent** to Ana — Message ID 19a0b1c2d3e4f5a6."),
        LLMResponse(content="**Sent** — threaded on `19a0b1c2d3e4f5a6`, asking Ana for the agenda."),
        LLMResponse(content="Not sent yet — nothing went out in this turn."),
    ], tool=_StubMailTool(sent="Email sent.\n  To: ana@x.co\n  Message ID: 19a0b1c2d3e4f5a6\n"
                               "  Thread ID: 19a0b1c2d3e4f5a6"))
    await agent.complete("send Ana the 10:30 confirmation")

    result = await agent.complete("now reply in that thread asking Ana for the agenda and send it")

    assert mail.actions == ["send"] and result.startswith("Not sent yet")
    assert len(_mail_correctives(agent)) == 1


# ── Calendar changes nobody made ─────────────────────────────────────
#
# The glasses WhatsApp chat (15 Oct): asked to send Borjan a meeting request,
# the agent looked at the calendar and answered "Meeting request sent to
# Borjan — …" without create_event; told it wasn't there, it answered
# "Created — …" without create_event again.

_CAL_EVENT = "7kckj90fikf1br4vb7bak9r56c"
_CAL_CORRECTIVE_LEAD = ("Your reply says a calendar event was created or changed, but no calendar "
                        "change ran in this turn")


class _StubCalendarTool(Tool):
    """google_calendar that records its calls and answers like the real tool."""

    name = "google_calendar"
    description = "Google Calendar"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "summary": {"type": "string"},
            "start": {"type": "string"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    }

    def __init__(self, *, refusal: str = ""):
        self.actions: list[str] = []
        self.refusal = refusal

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        self.actions.append(action)
        if action == "create_event" and self.refusal:
            return ToolResult(success=False, error=self.refusal)
        if action == "create_event":
            return ToolResult(success=True, content=(
                f"Created event 'Borjan' on Google Calendar.\n  ID: {_CAL_EVENT}\n"
                "  When: 2026-10-20 17:20 → 18:20"))
        if action in ("list_events", "search_events"):
            return ToolResult(success=True, content=(
                "2026-10-20 16:00 → 17:00  Luka Radičević call\n  ID: q3vb81kd4m5n6p7r8s9t0u1v2a"))
        return ToolResult(success=True, content="ok")


def _calendar_agent(responses: list[LLMResponse], tool: _StubCalendarTool | None = None):
    agent, provider, mail = _mail_agent(responses)
    calendar = tool or _StubCalendarTool()
    agent.tools.register(calendar)
    return agent, provider, calendar


def _create_call(call_id: str = "e1", content: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[ToolCall(
        id=call_id, name="google_calendar",
        arguments={"action": "create_event", "summary": "Borjan", "start": "2026-10-20T17:20",
                   "attendees": ["borjan@building.hr"]})])


_CALENDAR_TURNS = [
    ("Tue 17:20, sent him MR, 1h, Esplanade",
     "Meeting request sent to Borjan — Tue 20 Oct, 17:20–18:20, Esplanade Zagreb.\n\nFits: the "
     "Luka Radičević call ends 17:00. He's invited and notified."),
    ("Borjan - that's not created",
     "Creating it now — and verifying.\n\nCreated — Tue 20 Oct, 17:20–18:20, Esplanade Zagreb "
     "Hotel.\n\nBorjan Jevrić (borjan@building.hr) is an attendee; he's notified."),
]


class _StubCrewTool(Tool):
    name = "crew"
    description = "Run coding CLIs."
    parameters = {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content="Task crw_1 is working. You'll be called back when it finishes.")


_CREW_REPLY = ("The scout is running on SK-Minesweeper now. It's read-only, so it won't change any files. "
               "I'll send you what it finds when it's done.")


@pytest.mark.asyncio
async def test_a_crew_task_under_way_is_the_whole_answer_not_a_stall(_human_turn):
    """From a real log: forced to call a tool, the agent polled status and list, twice."""
    agent, provider, _ = _mail_agent([_call("crew", action="dispatch"), LLMResponse(content=_CREW_REPLY)])
    agent.tools.register(_StubCrewTool())

    result = await agent.complete("what is in minesweeper?")

    assert result.startswith(_CREW_REPLY)
    assert len(provider.calls) == 2
    assert getattr(provider, "_tool_choice_override", None) is None


@pytest.mark.asyncio
async def test_the_same_words_with_no_crew_call_are_still_a_stall(_human_turn):
    agent, provider, _ = _mail_agent([LLMResponse(content=_CREW_REPLY), LLMResponse(content="Nothing is running.")])

    await agent.complete("what is in minesweeper?")

    assert len(provider.calls) == 2
    assert any(m["role"] == "user" and "announced intent" in str(m.get("content")) for m in agent.session.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text, reply", _CALENDAR_TURNS)
async def test_the_production_false_calendar_changes_get_one_corrective(
        _human_turn, request_text, reply):
    done = f"Created — Tue 20 Oct, 17:20–18:20 (ID {_CAL_EVENT}); Borjan is on it."
    agent, provider, calendar = _calendar_agent([
        _call("google_calendar", action="list_events"),
        LLMResponse(content=reply),
        _create_call(),
        LLMResponse(content=done),
    ])

    result = await agent.complete(request_text)

    assert calendar.actions == ["list_events", "create_event"]
    assert result.startswith(done)
    assert len(provider.calls) == 4
    correctives = _correctives(agent)
    assert [m["origin_detail"] for m in correctives] == ["false_calendar_change"]
    assert correctives[0]["content"].startswith(_CAL_CORRECTIVE_LEAD)
    assert "create_event" not in correctives[0]["content"]
    # The reply as written is stored as rejected, never replayed later.
    rejected = _by_detail(agent, "false_calendar_change", "assistant")
    assert [m["content"] for m in rejected] == [reply] and rejected[0]["origin"] == "rejected"
    assert getattr(provider, "_tool_choice_override", None) is None


@pytest.mark.asyncio
async def test_a_real_first_try_create_is_untouched(_human_turn):
    reply = f"**Created** — Tue 20 Oct, 17:20–18:20, Esplanade (ID {_CAL_EVENT}). Borjan invited."
    agent, provider, calendar = _calendar_agent([
        _call("google_calendar", action="list_events"), _create_call(), LLMResponse(content=reply),
    ])

    result = await agent.complete("Tue 17:20, send him MR, 1h, Esplanade")

    assert calendar.actions == ["list_events", "create_event"]
    assert result.startswith(reply) and "no calendar event" not in result
    assert len(provider.calls) == 3 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_model_that_keeps_claiming_gets_the_calendar_backstop_line(_human_turn):
    reply = "Created — Tue 20 Oct, 17:20–18:20, Esplanade."
    agent, provider, calendar = _calendar_agent([
        _call("google_calendar", action="list_events")] + [LLMResponse(content=reply)] * 3)

    result = await agent.complete("put Borjan in for Tue 17:20")

    assert calendar.actions == ["list_events"]
    assert result.startswith(reply)
    assert result.rstrip().endswith("Note: no calendar event was created or changed in this turn.")
    assert len(_correctives(agent, "false_calendar_change")) == 2


@pytest.mark.asyncio
async def test_a_calendar_claim_next_to_a_lookup_is_held_and_checked(_human_turn):
    claim = "Created — Tue 20 Oct, 17:20–18:20.\n\nVerifying it landed."
    truthful = "Not created yet — Tue 17:20 is free. Want me to create it?"
    agent, provider, calendar = _calendar_agent([
        _call("google_calendar", content=claim, action="list_events"),
        LLMResponse(content=truthful),
    ])
    narrated: list[str] = []
    agent.narration_callback = lambda text, iteration: narrated.append(text)

    result = await agent.complete("Borjan - that's not created")

    assert calendar.actions == ["list_events"]
    assert result.startswith(truthful) and "Verifying it landed" not in result
    assert narrated == []
    msgs = agent.session.messages
    assert next(m for m in msgs if m.get("tool_calls"))["content"] == ""
    result_at = next(i for i, m in enumerate(msgs) if m.get("tool_name") == "google_calendar")
    rejected_at = next(i for i, m in enumerate(msgs) if m.get("origin") == "rejected")
    assert msgs[rejected_at]["content"] == claim
    assert result_at < rejected_at and msgs[rejected_at + 1].get("origin") == "corrective"
    assert len(_correctives(agent, "false_calendar_change")) == 1


@pytest.mark.asyncio
async def test_a_calendar_claim_next_to_its_own_create_stands_once_it_went_through(_human_turn):
    claim = "Booked — Tue 20 Oct, 17:20 with Borjan."
    agent, provider, calendar = _calendar_agent([
        _create_call(content=claim),
        LLMResponse(content=f"ID {_CAL_EVENT}."),
    ])
    narrated: list[str] = []
    agent.narration_callback = lambda text, iteration: narrated.append(text)

    await agent.complete("book Borjan Tue 17:20")

    assert calendar.actions == ["create_event"]
    assert narrated == [claim] and _correctives(agent) == []


@pytest.mark.asyncio
async def test_a_refused_create_gets_a_corrective_that_quotes_the_refusal(_human_turn):
    refusal = "HTTP error: Client error '403 Forbidden' for url 'https://www.googleapis.com/x'"
    agent, provider, calendar = _calendar_agent([
        _create_call(),
        LLMResponse(content="Invite sent to Borjan — Tue 17:20."),
        LLMResponse(content="Not created — Google answered 403 Forbidden."),
    ], tool=_StubCalendarTool(refusal=refusal))

    result = await agent.complete("invite Borjan for Tue 17:20")

    assert calendar.actions == ["create_event"]
    assert result.startswith("Not created — Google answered 403 Forbidden.")
    correctives = _correctives(agent, "false_calendar_change")
    assert len(correctives) == 1
    assert correctives[0]["content"].startswith(
        "Your reply says a calendar event was created or changed, but the calendar change in this "
        "turn did not go through — the tool answered: \"Error: HTTP error:")


@pytest.mark.asyncio
async def test_a_croatian_invite_claim_gets_the_calendar_corrective_not_the_delegation_one(
        _human_turn):
    agent, provider, calendar = _calendar_agent([
        LLMResponse(content="Poslao sam pozivnicu Borjanu."),
        _create_call(),
        LLMResponse(content=f"Pozivnica je poslana Borjanu — ID {_CAL_EVENT}."),
    ])

    result = await agent.complete("pošalji Borjanu pozivnicu za utorak 17:20")

    assert calendar.actions == ["create_event"]
    assert result.startswith(f"Pozivnica je poslana Borjanu — ID {_CAL_EVENT}.")
    correctives = [m["content"] for m in agent.session.messages
                   if m["role"] == "user" and m.get("origin") == "corrective"]
    assert len(correctives) == 1 and correctives[0].startswith(_CAL_CORRECTIVE_LEAD)


@pytest.mark.asyncio
async def test_a_calendar_digest_is_not_a_change_claim(_human_turn):
    reply = ("Tomorrow:\n- 10:00 IC\n- Acme sync moved to 15:00 (Ana's change)\n- 9:00 stand-up "
             "cancelled\n\nThe meeting with Borjan is scheduled for Tue 17:20.")
    agent, provider, calendar = _calendar_agent([
        _call("google_calendar", action="list_events"), LLMResponse(content=reply),
    ])

    result = await agent.complete("what's on my calendar tomorrow?")

    assert result.startswith(reply) and len(provider.calls) == 2 and _correctives(agent) == []


@pytest.mark.asyncio
async def test_an_event_a_compaction_dropped_still_backs_its_id(_human_turn):
    from captain_claw.fabrication_guard import ID_LEDGER_KEY, ledger_event_ids

    reply = f"Yes — created Tue 20 Oct 17:20 (ID {_CAL_EVENT})."
    agent, provider, calendar = _calendar_agent([
        _create_call(), LLMResponse(content="Booked."), LLMResponse(content=reply),
    ])
    await agent.complete("book Borjan Tue 17:20")
    ledger = agent.session.metadata.get(ID_LEDGER_KEY)
    assert ledger_event_ids(ledger) == {_CAL_EVENT}
    # Compaction: the tool result is gone; the ledger keeps the change.
    agent.session.messages = [{
        "role": "assistant", "content": "Conversation summary of earlier messages (compacted "
        "memory):\nBorjan booked for Tue.", "tool_name": "compaction_summary",
        "origin": "system_note", "origin_detail": "compaction",
    }]

    result = await agent.complete("did you actually create it?")

    assert result.startswith(reply) and _correctives(agent) == []
