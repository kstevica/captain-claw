"""Where a model config talks to: the identity of a provider endpoint.

A key belongs to its endpoint — a provider plus its (custom or own) base URL.
Flight Deck's spawn resolution and the agent-side ``flight_deck`` tool both
have to decide whether two model configs are "the same place" before one's key
may travel with the other; they share this one rule so they cannot disagree.

Deliberately dependency-free: the tool imports it inside agents.
"""

from __future__ import annotations

# A provider's own endpoint, as some configs spell it out (Freebie agents always
# write OpenRouter's): the same place as no base_url at all.
PROVIDER_OWN_ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1",),
    "openai": ("https://api.openai.com/v1",),
    "anthropic": ("https://api.anthropic.com", "https://api.anthropic.com/v1"),
    "xai": ("https://api.x.ai/v1",),
}


# The names agents accept for a provider (captain_claw.llm._normalize_provider_name).
_PROVIDER_ALIASES = {"chatgpt": "openai", "claude": "anthropic", "google": "gemini",
                     "googleai": "gemini", "grok": "xai"}


def endpoint_of(provider: str | None, base_url: str | None) -> tuple[str, str]:
    """(provider, endpoint) — the endpoint is "" for the provider's own."""
    name = (provider or "").strip().lower()
    name = _PROVIDER_ALIASES.get(name, name)
    url = str(base_url or "").strip().rstrip("/").lower().replace("://localhost", "://127.0.0.1")
    if url in PROVIDER_OWN_ENDPOINTS.get(name, ()):
        url = ""
    return name, url


def same_endpoint(provider_a: str | None, base_url_a: str | None,
                  provider_b: str | None, base_url_b: str | None) -> bool:
    """Do two model configs talk to the same place?"""
    return endpoint_of(provider_a, base_url_a) == endpoint_of(provider_b, base_url_b)
