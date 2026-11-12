"""Model switching: which of its provider's models an agent can run on
(llm/model_switch.py), and how the agent switches (agent_model_mixin)."""

from __future__ import annotations

import json
import types

import pytest

from captain_claw import agent_model_mixin
from captain_claw.agent_model_mixin import AgentModelMixin
from captain_claw.llm import model_switch


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    home = tmp_path / ".captain-claw"
    home.mkdir()
    monkeypatch.setattr(model_switch, "SWITCH_PATH", home / model_switch.SWITCH_FILENAME)
    monkeypatch.setattr(model_switch, "CHOICES_PATH", home / model_switch.CHOICES_FILENAME)
    monkeypatch.setattr(model_switch, "_custom_cache", {"mtime": None, "path": None, "value": None})
    monkeypatch.setattr(model_switch, "_refresh_state", {"at": 0.0, "key": "", "task": None})
    monkeypatch.delenv("FD_URL", raising=False)
    return home


def _models(options):
    return [o["model"] for o in options]


# ── the choices ──────────────────────────────────────────────────────


def test_a_big_provider_offers_its_catalog_smartest_first():
    options = model_switch.choices("anthropic", "claude-sonnet-5-5", "")
    assert _models(options) == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    opus, sonnet, haiku = options
    assert (opus["label"], sonnet["label"], haiku["label"]) == ("Smartest", "Everyday", "Fastest and cheapest")
    assert sonnet["default"] and not opus["default"] and not haiku["default"]
    assert opus["name"] == "Claude Opus 5.5"
    # Every catalog model has a price (model_prices.json is kept in step).
    assert all(o["price"] and o["price"]["input"] > 0 for o in options)
    # Haiku's tiers stop under its 100k price step; longctx/micro don't count.
    assert haiku["max_context"] == 100000 and opus["max_output"] == 65536


def test_the_provider_s_own_url_spelled_out_is_the_same_endpoint():
    assert _models(model_switch.choices("openai", "gpt-6.1-terra", "https://api.openai.com/v1")) == [
        "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"]


def test_deepseek_is_openai_on_its_own_endpoint():
    options = model_switch.choices("openai", "deepseek-flash", "https://api.deepseek.com")
    assert _models(options) == ["deepseek-v4-pro", "deepseek-flash"]
    assert options[1]["default"]


def test_a_gateway_has_no_catalog_only_its_configured_model():
    options = model_switch.choices("openai", "qwen3.6-35b", "http://10.0.0.5:4000/v1")
    assert _models(options) == ["qwen3.6-35b"] and options[0]["default"]


def test_the_admin_s_list_and_the_agent_s_allowed_models_join_in(_home):
    model_switch.save_custom("openai", "http://10.0.0.5:4000/v1/", [
        {"model": "qwen3.6-35b", "note": "fast"}, {"model": "kimi-k3", "note": "smart"}, "glm-5.3"])
    allowed = [("openai", "mistral-large-3", "http://10.0.0.5:4000/v1", "for French"),
               ("anthropic", "claude-opus-5-5", "", "another endpoint")]
    options = model_switch.choices("openai", "qwen3.6-35b", "http://10.0.0.5:4000/v1", allowed=allowed)
    assert _models(options) == ["qwen3.6-35b", "kimi-k3", "glm-5.3", "mistral-large-3"]
    assert options[0]["default"] and options[0]["note"] == "fast"
    assert options[1]["source"] == "admin" and options[3]["source"] == "config"


def test_another_endpoint_s_list_is_not_offered(_home):
    model_switch.save_custom("openai", "http://other:4000/v1", ["kimi-k3"])
    assert _models(model_switch.choices("openai", "qwen", "http://10.0.0.5:4000/v1")) == ["qwen"]


def test_a_chatgpt_sign_in_model_gets_no_api_key_models():
    assert _models(model_switch.choices("openai", "gpt-5.4-codex", "")) == ["gpt-5.4-codex"]


def test_pretty_names():
    assert model_switch.pretty_name("gpt-6.1-sol") == "GPT 6.1 Sol"
    assert model_switch.pretty_name("anthropic/claude-haiku-5-5") == "Claude Haiku 5.5"
    assert model_switch.pretty_name("gemini-3.5-flash-lite") == "Gemini 3.5 Flash Lite"


# ── naming a choice ──────────────────────────────────────────────────


@pytest.mark.parametrize("said, model", [
    ("opus", "claude-opus-5-5"), ("Claude Haiku 5.5", "claude-haiku-5-5"),
    ("anthropic/claude-sonnet-5-5", None), ("claude-sonnet-5-5", "claude-sonnet-5-5"),
    ("smartest", "claude-opus-5-5"), ("cheapest", "claude-haiku-5-5"), ("cheaper model", "claude-haiku-5-5"),
    ("default", "claude-sonnet-5-5"), ("gpt-6", None),
])
def test_resolve(said, model):
    options = model_switch.choices("anthropic", "claude-sonnet-5-5", "")
    found, _could_be = model_switch.resolve(options, said)
    assert (found["model"] if found else None) == model


def test_an_ambiguous_name_says_what_it_could_be():
    options = model_switch.choices("anthropic", "claude-sonnet-5-5", "")
    found, could_be = model_switch.resolve(options, "claude")
    assert found is None and len(could_be) == 3


def test_smartest_and_cheapest_go_by_price_without_a_catalog(_home):
    model_switch.save_custom("openrouter", "", ["vendor/big-one", "vendor/small-one"])
    options = model_switch.choices("openrouter", "vendor/small-one", "")
    prices = {"vendor/big-one": {"input": 5.0, "output": 25.0}, "vendor/small-one": {"input": 0.1, "output": 0.4}}
    for o in options:
        o["price"] = prices[o["model"]]
    assert model_switch.resolve(options, "smartest")[0]["model"] == "vendor/big-one"
    assert model_switch.resolve(options, "cheapest")[0]["model"] == "vendor/small-one"


@pytest.mark.parametrize("text, asked, loosely", [
    ("switch to opus please", True, True), ("use a cheaper model for this", True, True),
    ("which model are you on?", True, True), ("koji je tvoj model?", True, True),
    ("can you use haiku", True, True), ("prebaci se na jeftiniji", False, True),
    ("use something smarter", False, True), ("find me cheaper flights", False, True),
    ("make the image smaller", False, False), ("napravi to brzo", False, False),
    ("attach the file and send it", False, False), ("", False, False),
])
def test_talks_about_models(text, asked, loosely):
    options = model_switch.choices("anthropic", "claude-sonnet-5-5", "")
    assert model_switch.talks_about_models(text, options) is asked
    assert model_switch.talks_about_models(text, options, loose=True) is loosely


def test_the_self_switch_flag_is_off_unless_turned_on(_home):
    assert model_switch.self_switch_enabled() is False
    model_switch.SWITCH_PATH.write_text("on")
    assert model_switch.self_switch_enabled() is True
    model_switch.SWITCH_PATH.write_text("off")
    assert model_switch.self_switch_enabled() is False


# ── the agent switching ──────────────────────────────────────────────


class FakeProvider(types.SimpleNamespace):
    pass


class FakeAgent(AgentModelMixin):
    def __init__(self, model="claude-sonnet-5-5"):
        self.provider = FakeProvider(provider="anthropic", model=model, base_url="",
                                     temperature=0.2, max_tokens=32768)
        self._provider_override = False
        self._runtime_model_details = {}
        self.saved = 0
        self.session = types.SimpleNamespace(metadata={})

        async def save_session(_session):
            self.saved += 1

        self.session_manager = types.SimpleNamespace(save_session=save_session)


@pytest.fixture
def cfg(monkeypatch):
    built: list[dict] = []
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(provider="anthropic", model="claude-sonnet-5-5", base_url="",
                                    api_key="", temperature=0.2, max_tokens=32768,
                                    tokens_per_minute=0, allowed=[], vision="auto"),
        context=types.SimpleNamespace(max_tokens=160000),
        provider_keys=types.SimpleNamespace(headers_for=lambda _p: None),
    )

    def create_provider(**kw):
        built.append(kw)
        return FakeProvider(provider=kw["provider"], model=kw["model"], base_url=kw.get("base_url") or "",
                            temperature=kw.get("temperature"), max_tokens=kw.get("max_tokens"))

    monkeypatch.setattr(agent_model_mixin, "get_config", lambda: config)
    monkeypatch.setattr(agent_model_mixin, "create_provider", create_provider)
    monkeypatch.setattr(agent_model_mixin, "set_provider", lambda _p: None)
    config.built = built
    return config


async def test_switching_keeps_the_provider_and_key_and_is_kept_with_the_conversation(cfg):
    agent = FakeAgent()
    home = agent.provider
    ok, message, choice = await agent.switch_model("haiku")
    assert ok and choice["model"] == "claude-haiku-5-5", message
    assert agent.provider.model == "claude-haiku-5-5" and agent.provider.provider == "anthropic"
    kw = cfg.built[-1]
    assert kw["api_key"] is None and kw["base_url"] is None  # the configured key and endpoint
    # Haiku's catalog limits lower the agent's: 100k context, 16k out.
    assert kw["num_ctx"] == 100000 and kw["max_tokens"] == 16384
    sel = agent.session.metadata["model_selection"]
    assert sel["switch"] and sel["model"] == "claude-haiku-5-5" and agent.saved == 1
    # Back to the default: the very provider it started on, and nothing kept.
    ok, _m, choice = await agent.switch_model("default")
    assert ok and choice["default"] and agent.provider is home
    assert "model_selection" not in agent.session.metadata


async def test_a_switch_for_the_turn_goes_back_after_it(cfg):
    agent = FakeAgent()
    home = agent.provider
    ok, _m, _c = await agent.switch_model("opus", scope="turn")
    assert ok and agent.provider.model == "claude-opus-5-5"
    assert "model_selection" not in agent.session.metadata
    assert agent.model_switch_wire()["turn_only"] is True
    assert agent._restore_turn_model() is True and agent.provider is home
    assert agent._restore_turn_model() is False


async def test_a_setting_mid_turn_outlasts_the_turn(cfg):
    agent = FakeAgent()
    await agent.switch_model("opus", scope="turn")
    await agent.switch_model("haiku")
    assert agent._restore_turn_model() is False
    assert agent.provider.model == "claude-haiku-5-5"


async def test_an_unknown_model_lists_the_choices(cfg):
    ok, message, _c = await FakeAgent().switch_model("gpt-6.1-sol")
    assert not ok and "claude-opus-5-5" in message


async def test_one_model_means_nothing_to_switch_to(cfg):
    cfg.model.provider, cfg.model.base_url, cfg.model.model = "openai", "http://gw:4000/v1", "qwen"
    agent = FakeAgent(model="qwen")
    assert agent.model_switch_wire() is None
    ok, message, _c = await agent.switch_model("anything")
    assert not ok and "no other model" in message


async def test_a_member_can_t_switch_the_owner_s_model(cfg):
    agent = FakeAgent()
    agent._speaker_scoped = True
    assert agent.model_switch_wire() is None
    ok, _m, _c = await agent.switch_model("opus")
    assert not ok and agent.provider.model == "claude-sonnet-5-5"


def test_the_picker_wire(cfg, _home):
    wire = FakeAgent().model_switch_wire()
    assert wire["current"] == "claude-sonnet-5-5" and wire["default"] == "claude-sonnet-5-5"
    assert wire["self_switch"] is False and wire["turn_only"] is False
    assert [c["model"] for c in wire["choices"]] == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    assert set(wire["choices"][0]) == {"model", "name", "label", "note", "price", "default"}


def test_a_switch_kept_for_another_endpoint_no_longer_applies(cfg):
    agent = FakeAgent()
    agent.session.metadata["model_selection"] = {
        "provider": "openai", "model": "gpt-6.1-sol", "base_url": "", "switch": True}
    assert agent._session_model_selection() is None
    agent.session.metadata["model_selection"] = {
        "provider": "anthropic", "model": "claude-opus-5-5", "base_url": "", "switch": True}
    assert agent._session_model_selection()["model"] == "claude-opus-5-5"
    # An allowed-list pick (no switch mark) is applied as before, wherever it runs.
    agent.session.metadata["model_selection"] = {"provider": "openai", "model": "gpt-6.1-sol", "base_url": ""}
    assert agent._session_model_selection()["model"] == "gpt-6.1-sol"


async def test_refresh_keeps_what_flight_deck_lists(monkeypatch, _home):
    import httpx

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"models": [{"model": "kimi-k3", "note": "smart"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("FD_URL", "http://fd.test")
    assert await model_switch.refresh_custom("openai", "http://gw:4000/v1") is True
    assert seen[0].url.path == "/fd/model-switch/agent/models"
    assert seen[0].url.params["base_url"] == "http://gw:4000/v1"
    kept = json.loads(model_switch.CHOICES_PATH.read_text())
    assert kept["models"] == [{"model": "kimi-k3", "note": "smart"}]
    assert _models(model_switch.choices("openai", "qwen", "http://gw:4000/v1")) == ["qwen", "kimi-k3"]


async def test_refresh_without_flight_deck_does_nothing(_home):
    assert await model_switch.refresh_custom("openai", "http://gw:4000/v1") is False
    assert not model_switch.CHOICES_PATH.exists()
