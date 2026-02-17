import pytest

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
