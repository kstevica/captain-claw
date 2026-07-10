import pytest

import captain_claw.llm as llm_mod
from captain_claw.llm import LiteLLMProvider, Message, OllamaProvider, create_provider


def test_create_provider_supports_ollama():
    provider = create_provider(
        provider="ollama",
        model="llama3.2",
        base_url="http://localhost:11434",
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2"
    assert provider.base_url == "http://localhost:11434"


def test_create_provider_supports_chatgpt_alias():
    provider = create_provider(
        provider="chatgpt",
        model="gpt-4o-mini",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == "openai"
    assert provider.model == "openai/gpt-4o-mini"


def test_create_provider_supports_claude_alias():
    provider = create_provider(
        provider="claude",
        model="claude-3-5-sonnet-latest",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == "anthropic"
    assert provider.model == "anthropic/claude-3-5-sonnet-latest"


def test_create_provider_supports_gemini_alias():
    provider = create_provider(
        provider="google",
        model="gemini-2.0-flash",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == "gemini"
    assert provider.model == "gemini/gemini-2.0-flash"


def test_create_provider_supports_grok():
    provider = create_provider(
        provider="grok",
        model="grok-3",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == "xai"
    assert provider.model == "xai/grok-3"


def test_create_provider_supports_xai_alias():
    provider = create_provider(
        provider="xai",
        model="grok-3-mini",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == "xai"
    assert provider.model == "xai/grok-3-mini"


def test_create_provider_uses_env_xai_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    provider = create_provider(
        provider="grok",
        model="grok-3",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.api_key == "test-xai-key"


def test_create_provider_preserves_prefixed_model():
    provider = create_provider(
        provider="openai",
        model="openai/gpt-4o-mini",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openai/gpt-4o-mini"


def test_create_provider_uses_env_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    provider = create_provider(
        provider="openai",
        model="gpt-4o-mini",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.api_key == "test-openai-key"


def test_create_provider_rejects_unsupported_provider():
    with pytest.raises(ValueError):
        create_provider(provider="cohere", model="command-r")


def test_create_provider_normalizes_gpt5_temperature_to_one():
    provider = create_provider(
        provider="openai",
        model="gpt-5-codex",
        temperature=0.2,
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.temperature == 1.0


def test_litellm_request_kwargs_force_temp_one_for_gpt5_family():
    provider = LiteLLMProvider(
        provider="openai",
        model="gpt-5",
        temperature=0.3,
        max_tokens=123,
    )

    kwargs = provider._request_kwargs(
        messages=[Message(role="user", content="hello")],
        stream=False,
    )

    assert kwargs["temperature"] == 1.0
    assert kwargs["model"] == "openai/gpt-5"


def test_litellm_request_kwargs_keep_temperature_for_non_gpt5():
    provider = LiteLLMProvider(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=123,
    )

    kwargs = provider._request_kwargs(
        messages=[Message(role="user", content="hello")],
        stream=False,
    )

    assert kwargs["temperature"] == 0.3
    assert kwargs["model"] == "openai/gpt-4o-mini"


@pytest.fixture
def _reset_temperature_registry():
    """Isolate the module-level learned-model set between tests."""
    saved = set(llm_mod._TEMPERATURE_UNSUPPORTED_MODELS)
    llm_mod._TEMPERATURE_UNSUPPORTED_MODELS.clear()
    try:
        yield
    finally:
        llm_mod._TEMPERATURE_UNSUPPORTED_MODELS.clear()
        llm_mod._TEMPERATURE_UNSUPPORTED_MODELS.update(saved)


@pytest.mark.asyncio
async def test_acompletion_tolerant_retries_without_temperature(
    monkeypatch: pytest.MonkeyPatch, _reset_temperature_registry
):
    """A "temperature is deprecated" 400 triggers one retry with temperature dropped."""
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise RuntimeError(
                "AnthropicException - `temperature` is deprecated for this model."
            )
        return {"ok": True}

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    result = await llm_mod._acompletion_tolerant(
        {"model": "anthropic/claude-opus-4-8", "temperature": 0.7, "messages": []}
    )

    assert result == {"ok": True}
    assert len(calls) == 2  # first (with temp) failed, retry (no temp) succeeded
    assert "temperature" not in calls[1]
    # The model is remembered so future calls omit temperature up front.
    assert llm_mod._is_temperature_unsupported_model("anthropic", "anthropic/claude-opus-4-8")


@pytest.mark.asyncio
async def test_acompletion_tolerant_reraises_unrelated_error(
    monkeypatch: pytest.MonkeyPatch, _reset_temperature_registry
):
    """An unrelated 400 is not swallowed and does not trigger a retry."""
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("some other 400: max_tokens too large")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(RuntimeError, match="max_tokens"):
        await llm_mod._acompletion_tolerant(
            {"model": "anthropic/claude-opus-4-8", "temperature": 0.7, "messages": []}
        )
    assert len(calls) == 1  # no retry


def test_request_kwargs_omits_temperature_for_learned_model(_reset_temperature_registry):
    """Once a model is learned to reject temperature, kwargs omit it up front."""
    provider = LiteLLMProvider(
        provider="anthropic",
        model="claude-sonnet-5",
        temperature=0.5,
        max_tokens=123,
    )

    kwargs = provider._request_kwargs(
        messages=[Message(role="user", content="hello")],
        stream=False,
    )
    assert kwargs["temperature"] == 0.5  # still sent before discovery

    llm_mod._remember_temperature_unsupported("anthropic/claude-sonnet-5")

    kwargs2 = provider._request_kwargs(
        messages=[Message(role="user", content="hello")],
        stream=False,
    )
    assert "temperature" not in kwargs2  # omitted after discovery
