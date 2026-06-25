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
