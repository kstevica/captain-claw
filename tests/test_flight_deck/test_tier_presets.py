"""Quick start tier sets: the catalog and the key check (tier_presets.py)."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import pricing, tier_presets
from captain_claw.llm import is_supported_provider

# Mirrors of the frontend's chip lists (services/tierConfig.ts): a preset's
# sizes must be ones the tier editor can show.
INPUT_CTX_OPTIONS = {4096, 8192, 16384, 32768, 65536, 100000, 131072, 160000, 262144, 524288, 786432, 1048576}
OUTPUT_CTX_OPTIONS = {1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 131072, 262144}
TIER_ORDER = ["reason", "balanced", "fast", "longctx", "coding", "vision", "micro"]


def _registry_tiers() -> set[str]:
    data = json.loads((tier_presets.PRESETS_FILE.parent / "archetypes.json").read_text(encoding="utf-8"))
    return set(data["tiers"])


# ── the catalog ─────────────────────────────────────────────────────


def test_catalog_offers_the_five_providers():
    ids = [p["id"] for p in tier_presets.load_presets()]
    assert ids == ["anthropic", "openai", "gemini", "xai", "deepseek"]
    assert tier_presets.checked_on()


@pytest.mark.parametrize("preset", tier_presets.load_presets(), ids=lambda p: p["id"])
def test_every_preset_is_complete(preset):
    # Every tier the registry has, and nothing the tier editor can't show.
    assert set(preset["tiers"]) >= _registry_tiers()
    assert set(preset["tiers"]) <= set(TIER_ORDER)
    assert is_supported_provider(preset["provider"])
    assert preset["set_name"] and preset["label"] and preset["blurb"]
    assert preset["key_url"].startswith("https://")
    check = preset["check"]
    assert check["url"].startswith("https://") and check["auth"] in tier_presets._AUTH_STYLES
    for name, tier in preset["tiers"].items():
        assert tier["model"].strip(), name
        assert tier["input_ctx"] in INPUT_CTX_OPTIONS, (name, tier["input_ctx"])
        assert tier["output_ctx"] in OUTPUT_CTX_OPTIONS, (name, tier["output_ctx"])
        assert tier["output_ctx"] < tier["input_ctx"] or name == "micro", name
        # The run's cost is known: every model has a price.
        assert pricing.rate_for(f"{preset['provider']}/{tier['model']}"), tier["model"]


def test_micro_tier_keeps_the_mrav_cap():
    for p in tier_presets.load_presets():
        assert p["tiers"]["micro"]["input_ctx"] == 8192, p["id"]


def test_deepseek_runs_on_its_own_endpoint_with_vision_on_flash():
    ds = next(p for p in tier_presets.load_presets() if p["id"] == "deepseek")
    assert ds["provider"] == "openai" and ds["base_url"] == "https://api.deepseek.com"
    assert ds["tiers"]["vision"] == {**ds["tiers"]["vision"], "model": "deepseek-flash", "vision": True}


def test_the_listing_leaves_out_the_check_urls():
    assert all("check" not in tier_presets._public(p) for p in tier_presets.load_presets())


def test_an_unreadable_catalog_offers_nothing(tmp_path, monkeypatch):
    bad = tmp_path / "tier_presets.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(tier_presets, "PRESETS_FILE", bad)
    assert tier_presets.load_presets() == []


# ── missing models ──────────────────────────────────────────────────


def test_a_dated_snapshot_counts_as_the_model():
    preset = {"tiers": {"a": {"model": "claude-opus-5-5"}, "b": {"model": "gpt-6-luna"}}}
    assert tier_presets.missing_models(preset, ["claude-opus-5-5-20261001", "gpt-6-luna"]) == []
    assert tier_presets.missing_models(preset, ["claude-opus-5-5"]) == ["gpt-6-luna"]
    # A sibling is not the model.
    assert tier_presets.missing_models({"tiers": {"a": {"model": "grok-4"}}}, ["grok-4.7"]) == ["grok-4"]


def test_nothing_listed_proves_nothing():
    assert tier_presets.missing_models({"tiers": {"a": {"model": "x"}}}, []) == []


def test_gemini_names_lose_their_prefix():
    ids = tier_presets._listed_ids({"models": [{"name": "models/gemini-3.8-flash"}]})
    assert ids == ["gemini-3.8-flash"]


# ── the route ───────────────────────────────────────────────────────


@pytest.fixture
def client_for(monkeypatch):
    monkeypatch.setattr(auth_mod, "_fd_auth_enabled", lambda: False)
    real_client = httpx.AsyncClient
    app = FastAPI()
    app.include_router(tier_presets.router)
    seen: list[httpx.Request] = []

    def make(handler):
        def recording(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        def fake_client(**kw):
            return real_client(transport=httpx.MockTransport(recording), **kw)

        monkeypatch.setattr(tier_presets.httpx, "AsyncClient", fake_client)
        return real_client(transport=httpx.ASGITransport(app=app), base_url="http://fd.test"), seen

    return make


async def test_the_catalog_route(client_for):
    client, _ = client_for(lambda r: httpx.Response(500))
    async with client:
        r = await client.get("/fd/tier-presets")
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["presets"]][:2] == ["anthropic", "openai"]
    assert all("check" not in p for p in body["presets"])


async def test_a_good_key_reports_the_models_it_lacks(client_for):
    listed = {"data": [{"id": "gpt-6.1-sol"}, {"id": "gpt-6.1-terra"}]}
    client, seen = client_for(lambda r: httpx.Response(200, json=listed))
    async with client:
        r = await client.post("/fd/tier-presets/check", json={"preset": "openai", "api_key": " sk-proj-abc "})
    assert r.json() == {"ok": True, "checked": True, "missing": ["gpt-6-luna"]}
    assert str(seen[0].url) == "https://api.openai.com/v1/models"
    assert seen[0].headers["authorization"] == "Bearer sk-proj-abc"


async def test_each_provider_gets_its_own_header(client_for):
    client, seen = client_for(lambda r: httpx.Response(200, json={"data": []}))
    async with client:
        await client.post("/fd/tier-presets/check", json={"preset": "anthropic", "api_key": "sk-ant-1"})
        await client.post("/fd/tier-presets/check", json={"preset": "gemini", "api_key": "AIza1"})
    assert seen[0].headers["x-api-key"] == "sk-ant-1" and seen[0].headers["anthropic-version"]
    assert seen[1].headers["x-goog-api-key"] == "AIza1"
    assert "authorization" not in seen[0].headers and "authorization" not in seen[1].headers


async def test_a_refused_key(client_for):
    client, _ = client_for(lambda r: httpx.Response(401, json={"error": "bad key"}))
    async with client:
        r = await client.post("/fd/tier-presets/check", json={"preset": "xai", "api_key": "xai-nope"})
    assert r.json() == {"ok": False, "refused": True, "status": 401}


async def test_a_provider_that_cant_be_reached(client_for):
    def down(_r):
        raise httpx.ConnectError("no route")

    client, _ = client_for(down)
    async with client:
        r = await client.post("/fd/tier-presets/check", json={"preset": "deepseek", "api_key": "sk-1"})
    assert r.json() == {"ok": False, "unreachable": True}


async def test_an_unknown_preset(client_for):
    client, seen = client_for(lambda r: httpx.Response(200, json={"data": []}))
    async with client:
        r = await client.post("/fd/tier-presets/check", json={"preset": "nope", "api_key": "k"})
    assert r.status_code == 404 and seen == []
