"""Iskra — export/import a being (move between machines) + hard-remove."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, owner="alice", name="Vesna"):
    b = store.conceive(owner, name, preset="scholar", allowance_preset="5M",
                       birth_letter="Grow well.", voice_seed="I think first.",
                       now=NOW)
    store.hatch(owner, b["slug"], now=NOW)
    store.set_stage(owner, b["slug"], "child", now=NOW)
    being = store.get(owner, b["slug"])
    await life.build_home(being)
    home = life.home_root(being)
    (home / "garden" / "first.md").write_text("# First\nmade a thing\n")
    store.milestone(being["id"], "first_artifact", now=NOW)
    return store.get(owner, b["slug"])


async def test_export_import_roundtrip_preserves_being(store):
    b = await _being(store)
    slug = b["slug"]
    # a judged chore mints savings
    j = store.post_chore("alice", slug, "tidy", 400000, now=NOW)
    store.chore_done("alice", j["id"], "done", now=NOW)
    store.judge_chore("alice", j["id"], True, now=NOW)
    bal = store.wallet_view(store.get("alice", slug))["balance_tokens"]

    manifest = await life.export_being(None, store, store.get("alice", slug))
    assert manifest["format"] == life.EXPORT_FORMAT
    assert manifest["genome"]["voice_seed"] == "I think first."
    assert "garden/first.md" in manifest["home"]
    assert any(e["kind"] == "milestone" for e in manifest["events"])

    # import under another owner (same machine → slug is taken → suffixed)
    manifest["state"] = "paused"          # skip real body spawn in-test
    manifest["model"] = {"provider": "openrouter", "model": "a/b",
                         "api_key": "sk-or-x"}
    res = await life.import_being(None, store, "bob", manifest)
    imp = res["being"]
    assert imp["slug"] != slug
    assert imp["stage"] == "child" and imp["birth_letter"] == "Grow well."
    assert imp["body_config"]["api_key"] == "sk-or-x"
    # wallet re-minted, conservation holds on the target owner
    assert store.wallet_view(imp)["balance_tokens"] == bal
    assert store.conservation("bob")["ok"]
    # home + milestone survived the move
    assert any(f["path"] == "garden/first.md"
               for f in life.list_self_files(imp))
    assert "milestone" in [e["kind"]
                           for e in store.events("bob", imp["slug"], limit=99)]


async def test_import_uses_original_slug_when_free(store):
    b = await _being(store, owner="alice")
    manifest = await life.export_being(None, store, b)
    manifest["state"] = "paused"
    # a DIFFERENT store (fresh machine) has the slug free
    other = BeingsStore(db_path=store.db_path.parent / "other.db")
    res = await life.import_being(None, other, "carol", manifest)
    assert res["being"]["slug"] == b["slug"]     # original slug kept


async def test_purge_only_the_dead_and_leaves_no_orphans(store):
    b = await _being(store)
    slug, bid = b["slug"], b["id"]
    store.post_chore("alice", slug, "x", 1000, now=NOW)   # a row to orphan-check
    with pytest.raises(BeingError, match="dead"):
        store.purge("alice", slug)
    store.set_state("alice", slug, "dead", now=NOW)
    dead = store.get("alice", slug)
    removed = store.purge("alice", slug)
    assert removed["slug"] == slug
    assert life.remove_home(dead) is True
    with pytest.raises(BeingError):
        store.get("alice", slug)
    c = store._c()
    for tbl, col in [("being_wallets", "being_id"), ("being_events", "being_id"),
                     ("being_jobs", "being_id"), ("token_transfers", "to_being")]:
        n = c.execute(f"SELECT COUNT(*) AS n FROM {tbl} WHERE {col} = ?",
                      (bid,)).fetchone()["n"]
        assert n == 0, f"orphan rows in {tbl}"


async def test_imported_being_body_uses_carried_model(store, monkeypatch):
    b = await _being(store)
    manifest = await life.export_being(None, store, b)
    manifest["state"] = "paused"
    manifest["model"] = {"provider": "openrouter", "model": "carried/model",
                         "api_key": "sk-carried", "base_url": "http://x"}
    res = await life.import_being(None, store, "bob", manifest)
    imp = res["being"]

    captured = {}

    async def fake_spawn_process(cfg, request, _):
        captured["provider"] = cfg.provider
        captured["model"] = cfg.model
        captured["key"] = cfg.provider_api_key

    def fake_resolve(slug):
        return (24099, "tok")

    monkeypatch.setattr("captain_claw.flight_deck.server.spawn_process",
                        fake_spawn_process)
    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token",
        fake_resolve)

    async def fake_tiers(db, owner):
        return ({}, [])
    monkeypatch.setattr(
        "captain_claw.flight_deck.basna_routes._load_owner_tiers", fake_tiers)

    await life.spawn_body(None, store, store.get("bob", imp["slug"]))
    assert captured["model"] == "carried/model"
    assert captured["key"] == "sk-carried"
