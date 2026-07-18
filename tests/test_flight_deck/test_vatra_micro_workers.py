"""S3 / mrav Phase 4: the `micro_workers` quality lever for Vatra.

Opt-in and off-path byte-identical: with the lever off (and no explicit
micro tier) every worker spawns exactly as before — runtime stays "",
models and context untouched. With it on, extract/digest/format-shaped
subtasks spawn the SAME worker process but with runtime="mrav" and the
owner's micro tier; reasoning roles keep their tiers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import vatra_routes as vr
from captain_claw.flight_deck.quality_profile import QualityProfile

# ── the lever ────────────────────────────────────────────────────────


def test_micro_workers_lever_parses_and_defaults_off():
    assert QualityProfile.from_dict(None).micro_workers is False
    assert QualityProfile.from_dict({}).micro_workers is False
    assert QualityProfile.from_dict({"micro_workers": True}).micro_workers is True
    # presets never enable it — explicit opt-in only
    for preset in ("saver", "deep", "max"):
        prof = QualityProfile.from_dict({"profile": preset})
        assert prof.micro_workers is False, preset


# ── suitability wording ──────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "Extract every invoice number from the PDFs",
    "Digest the meeting notes into bullet points",
    "Summarize each source file",
    "Reformat the table as CSV; convert dates",
    "Collect and catalog the API endpoints",
    "Deduplicate the contact list",
    "Normalise the country names",
])
def test_micro_suited_positive(text):
    assert vr._micro_suited(text)


@pytest.mark.parametrize("text", [
    "Design the authentication architecture",
    "Adversarially review the security posture",
    "Decide the go-to-market strategy",
    "",
])
def test_micro_suited_negative(text):
    assert not vr._micro_suited(text)


# ── spawn wiring ─────────────────────────────────────────────────────


TIERS = {
    "fast": {"provider": "openai", "model": "gpt-5-mini", "api_key": "kf",
             "base_url": "", "input_ctx": 200000, "output_ctx": 16384},
    "micro": {"provider": "ollama", "model": "qwen3.5:4b", "api_key": "",
              "base_url": "", "input_ctx": 8192, "output_ctx": 1024},
}


@pytest.fixture()
def spawn_capture(monkeypatch):
    """Fake the server spawn boundary; capture the AgentConfig."""
    from captain_claw.flight_deck import server as srv
    captured: dict = {}

    async def fake_spawn(cfg, request, user):
        captured["cfg"] = cfg
        return SimpleNamespace(ok=True, slug="w1", message="")

    monkeypatch.setattr(srv, "spawn_process", fake_spawn)
    monkeypatch.setattr(srv, "_load_process_registry",
                        lambda: {"w1": {"web_port": 12345, "web_auth": "tok"}})
    return captured


async def _spawn(micro: bool, tier: str = "fast", tiers=TIERS):
    return await vr._spawn_worker(
        object(), {"id": "u1"},
        name="vatra-x-r1-t1-arch", description="Vatra subtask · extractor",
        cognitive_mode="neutra", tools=["read", "write"], tier=tier,
        tiers=tiers, api_key="", env_vars=[], micro=micro)


@pytest.mark.asyncio
async def test_lever_off_is_byte_identical(spawn_capture):
    out = await _spawn(micro=False)
    assert out["ok"] and out["port"] == 12345
    cfg = spawn_capture["cfg"]
    assert cfg.runtime == ""            # classic loop, untouched
    assert cfg.model == "gpt-5-mini"    # tier resolution unchanged
    assert cfg.max_context == 200000 and cfg.max_tokens == 16384


@pytest.mark.asyncio
async def test_micro_spawns_mrav_on_micro_tier(spawn_capture):
    out = await _spawn(micro=True)
    assert out["ok"]
    cfg = spawn_capture["cfg"]
    assert cfg.runtime == "mrav"
    assert cfg.provider == "ollama" and cfg.model == "qwen3.5:4b"
    # these feed mrav.input_cap / output_cap through the spawn yaml
    assert cfg.max_context == 8192 and cfg.max_tokens == 1024


@pytest.mark.asyncio
async def test_micro_without_micro_tier_keeps_model_swaps_loop(spawn_capture):
    tiers = {"fast": TIERS["fast"]}  # no micro tier configured
    out = await _spawn(micro=True, tiers=tiers)
    assert out["ok"]
    cfg = spawn_capture["cfg"]
    assert cfg.runtime == "mrav"        # capped loop still applies
    assert cfg.model == "gpt-5-mini"    # on the tier's own model


@pytest.mark.asyncio
async def test_explicit_micro_tier_always_means_mrav(spawn_capture):
    out = await _spawn(micro=False, tier="micro")
    assert out["ok"]
    cfg = spawn_capture["cfg"]
    assert cfg.runtime == "mrav"
    assert cfg.model == "qwen3.5:4b"
