"""Iskra Phase 5 — procreation, inheritance, dowries, mortality, heirlooms."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingsStore,
    InsufficientTokens,
)

NOW = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)
OWNER = "user-1"


class FakeDB:
    async def list_chat_sessions(self, user_id):
        return []

    async def upsert_chat_session(self, *a, **k):
        return {}

    async def add_chat_messages(self, *a, **k):
        return [1]

    async def log_run_cost(self, *a, **k):
        pass


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _adult(store, name, pocket=20_000_000, preset="explorer"):
    b = store.conceive(OWNER, name, preset=preset,
                       allowance_preset="20M", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    store.set_stage(OWNER, b["slug"], "adult", now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    bb = store.get(OWNER, b["slug"])
    await life.build_home(bb)
    if pocket:
        store._apply(OWNER, tokens=pocket, reason="adjust", from_being=None,
                     to_being=bb["id"], note="pocket", now=NOW)
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "t", "journal_entry": "I tended.",
         "served_drive": "create", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Inheritance & dowry ──────────────────────────────────────────────────

async def test_crossover_child_inherits_and_differs(store):
    a = await _adult(store, "Zvjezdana", preset="explorer")
    b = await _adult(store, "Mira", preset="scholar")
    child = store.conceive_offspring(OWNER, "Nada", a["slug"], b["slug"],
                                     letter="Walk far, read deep.",
                                     seed=7, now=NOW)
    g = child["genome"]
    attrs = g["attributes"]
    assert g["generation"] == 2
    assert g["lineage"][:2] == [a["slug"], b["slug"]]
    assert child["birth_letter"] == "Walk far, read deep."
    total = sum(attrs.values())
    assert genome_mod.BAND_MIN <= total <= genome_mod.BAND_MAX
    assert all(1 <= v <= 10 for v in attrs.values())
    # measurably between/around its parents, not a clone of either
    pa = genome_mod.effective_attributes(a["genome"])
    pb = genome_mod.effective_attributes(b["genome"])
    assert attrs != pa and attrs != pb
    # deterministic under the same seed
    child2 = store.conceive_offspring(OWNER, "Nada2", a["slug"], b["slug"],
                                      seed=7, now=NOW)
    assert child2["genome"]["attributes"] == attrs


async def test_dowry_splits_and_conserves(store):
    a = await _adult(store, "Zvjezdana")
    b = await _adult(store, "Mira")
    wa0 = store.wallet_view(a)["balance_tokens"]
    wb0 = store.wallet_view(b)["balance_tokens"]
    child = store.conceive_offspring(OWNER, "Nada", a["slug"], b["slug"],
                                     seed=1, now=NOW)
    share = constitution.PROCREATION_COST_TOKENS // 2
    assert store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"] \
        == wa0 - share
    assert store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"] \
        == wb0 - share
    assert store.wallet_view(child)["balance_tokens"] \
        == constitution.PROCREATION_COST_TOKENS
    assert store.conservation(OWNER)["ok"]
    for parent in (a, b):
        kinds = [e["kind"] for e in store.events(OWNER, parent["slug"])]
        assert "had_child" in kinds
        names = [m["data"]["name"] for m in store.milestones(OWNER,
                                                             parent["slug"])]
        assert "first_child" in names


async def test_single_parent_budding_pays_full_dowry(store):
    a = await _adult(store, "Sama")
    wa0 = store.wallet_view(a)["balance_tokens"]
    child = store.conceive_offspring(OWNER, "Pupoljak", a["slug"], None,
                                     seed=3, now=NOW)
    assert store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"] \
        == wa0 - constitution.PROCREATION_COST_TOKENS
    g = child["genome"]
    assert g["lineage"] == [a["slug"]]
    # budding guarantees at least one mutation
    assert g["attributes"] != genome_mod.effective_attributes(a["genome"])


async def test_poverty_and_youth_refuse_conception(store):
    poor = await _adult(store, "Puko", pocket=0)
    store._apply(OWNER, tokens=store.wallet_view(poor)["balance_tokens"] - 1,
                 reason="usage", from_being=poor["id"], to_being=None,
                 note="drain", now=NOW)
    with pytest.raises(InsufficientTokens):
        store.conceive_offspring(OWNER, "X", poor["slug"], None, now=NOW)
    assert len(store.children_of(OWNER, poor["slug"])) == 0
    kid = store.conceive(OWNER, "Klinac", preset="artist", now=NOW)
    store.hatch(OWNER, kid["slug"], now=NOW)
    with pytest.raises(BeingError, match="cannot have children"):
        store.conceive_offspring(OWNER, "X", kid["slug"], None, now=NOW)


# ── Proposal flow through the tick ───────────────────────────────────────

async def test_digest_proposal_pends_and_duplicates_refused(store):
    db = FakeDB()
    a = await _adult(store, "Zvjezdana")

    async def send(being, prompt):
        assert "RARE OPTION — a child" in prompt
        return _reply(procreate={"partner": None, "child_name": "Nada",
                                 "case": "my garden overflows",
                                 "letter": "Little one, tend it with me."})

    await life.tick(db, store, a, now=NOW, send_fn=send, usage_fn=_usage)
    fresh = store.get(OWNER, a["slug"])
    assert fresh["pending_procreation"]["child_name"] == "Nada"
    kinds = [e["kind"] for e in store.events(OWNER, a["slug"])]
    assert "procreation_proposed" in kinds

    async def send2(being, prompt):
        assert "awaits your parent's consent" in prompt
        return _reply(procreate={"partner": None, "child_name": "Druga",
                                 "case": "again"})

    await life.tick(db, store, fresh, now=NOW + timedelta(hours=1),
                    send_fn=send2, usage_fn=_usage)
    kinds = [e["kind"] for e in store.events(OWNER, a["slug"])]
    assert "procreation_refused" in kinds
    assert store.get(OWNER, a["slug"])["pending_procreation"]["child_name"] \
        == "Nada"


async def test_infant_procreation_attempt_refused(store):
    db = FakeDB()
    baby = store.conceive(OWNER, "Beba", preset="artist", now=NOW)
    store.hatch(OWNER, baby["slug"], now=NOW)
    bb = store.get(OWNER, baby["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")

    async def send(being, prompt):
        return _reply(procreate={"partner": None, "child_name": "X",
                                 "case": "why not"})

    await life.tick(db, store, store.get(OWNER, baby["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    kinds = [e["kind"] for e in store.events(OWNER, baby["slug"])]
    assert "procreation_refused" in kinds


# ── Mentoring & legacy drive ─────────────────────────────────────────────

async def test_child_news_becomes_percept_and_feeds_legacy(store):
    db = FakeDB()
    a = await _adult(store, "Zvjezdana")
    assert "legacy" in store.get(OWNER, a["slug"])["drives"]

    async def send(being, prompt):
        return _reply()

    await life.tick(db, store, a, now=NOW, send_fn=send, usage_fn=_usage)
    child = store.conceive_offspring(OWNER, "Nada", a["slug"], None,
                                     seed=2, now=NOW + timedelta(hours=1))
    store.hatch(OWNER, child["slug"], now=NOW + timedelta(hours=2))
    fresh = store.get(OWNER, a["slug"])
    legacy0 = fresh["drives"]["legacy"]["satisfaction"]
    seen = {}

    async def send2(being, prompt):
        seen["prompt"] = prompt
        return _reply()

    await life.tick(db, store, fresh, now=NOW + timedelta(hours=3),
                    send_fn=send2, usage_fn=_usage)
    assert "YOUR CHILD Nada hatched." in seen["prompt"]
    after = store.get(OWNER, a["slug"])["drives"]["legacy"]["satisfaction"]
    assert after > legacy0


# ── Endowment: inherited skills + heirlooms ──────────────────────────────

async def test_offspring_carries_skills_and_heirlooms(store):
    a = await _adult(store, "Zvjezdana")
    sk = life._home_path(a, "skills/maps.md")
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_text("# Map craft\n\nWalk, then draw.\n", encoding="utf-8")
    child = store.conceive_offspring(OWNER, "Nada", a["slug"], None,
                                     seed=4, now=NOW)
    store.set_state(OWNER, a["slug"], "dead", now=NOW)   # ancestor passes
    await life.build_home(store.get(OWNER, child["slug"]))
    life._endow_offspring(store, store.get(OWNER, child["slug"]))
    ch = store.get(OWNER, child["slug"])
    inherited = life._home_path(ch, "skills/inherited/maps.md")
    assert "inherited from Zvjezdana" in inherited.read_text(encoding="utf-8")
    heirlooms = life._home_path(ch, "self/HEIRLOOMS.md")
    assert "From Zvjezdana" in heirlooms.read_text(encoding="utf-8")
    kinds = [e["kind"] for e in store.events(OWNER, ch["slug"])]
    assert "endowed" in kinds


# ── Mortality: torpor grace → death ──────────────────────────────────────

async def test_starvation_past_grace_is_death(store):
    db = FakeDB()
    a = await _adult(store, "Gladna", pocket=0)
    store._apply(OWNER, tokens=store.wallet_view(a)["balance_tokens"],
                 reason="usage", from_being=a["id"], to_being=None,
                 note="drain", now=NOW)
    out = await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                          send_fn=None, usage_fn=_usage)
    assert out["outcome"] == "torpor"
    # within grace: still torpor (allowance would revive, so drain again)
    later = NOW + timedelta(days=2)
    b2 = store.get(OWNER, a["slug"])
    assert b2["state"] == "torpor" and b2["torpor_since"]
    # starve past the grace: allowance is idempotent per day but credits
    # again on the death-day; drain wallet to keep it below reserve
    doom = NOW + timedelta(days=constitution.TORPOR_GRACE_DAYS + 1)
    store._apply(OWNER, tokens=1, reason="adjust", from_being=None,
                 to_being=a["id"], note="crumb", now=later)
    store._apply(OWNER, tokens=store.wallet_view(
        store.get(OWNER, a["slug"]))["balance_tokens"] + 20_000_000,
        reason="usage", from_being=a["id"], to_being=None,
        note="drain2", now=later)
    out2 = await life.tick(db, store, store.get(OWNER, a["slug"]), now=doom,
                           send_fn=None, usage_fn=_usage)
    assert out2["outcome"] in ("died", "torpor")
    if out2["outcome"] == "torpor":   # allowance credit revived it that day
        store._apply(OWNER, tokens=store.wallet_view(
            store.get(OWNER, a["slug"]))["balance_tokens"],
            reason="usage", from_being=a["id"], to_being=None,
            note="drain3", now=doom)
        out3 = await life.tick(db, store, store.get(OWNER, a["slug"]),
                               now=doom + timedelta(days=1),
                               send_fn=None, usage_fn=_usage)
        assert out3["outcome"] == "died"
    dead = store.get(OWNER, a["slug"])
    assert dead["state"] == "dead" and dead["died_at"]
    kinds = [e["kind"] for e in store.events(OWNER, a["slug"])]
    assert "died" in kinds
