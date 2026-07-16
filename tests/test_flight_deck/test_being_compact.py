"""Compact mode: the compact instruction set + the lean-body flag.

Compact is a per-being toggle. The full set must stay byte-honest to the
original narrative (covered by the existing prompt tests); these tests pin
what Compact promises: same physics and contracts, fewer words, external
files for both sets, and a graceful fallback when a compact variant of a
one-liner doesn't exist.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_mind as mind
from captain_claw.flight_deck import being_prompts
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _born(store, name="Prva", stage=None, compact=False):
    b = store.conceive(OWNER, name, preset="explorer", allowance_preset="2M",
                       birth_letter="Grow curious and kind.", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage:
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    if compact:
        store.set_compact_mode(OWNER, b["slug"], True, now=NOW)
    return store.get(OWNER, b["slug"])


# ── Store + vitals ───────────────────────────────────────────────────────

def test_set_compact_mode_persists_and_shows_in_vitals(store):
    b = _born(store)
    assert not b.get("compact_mode")
    assert store.vitals(OWNER, b["slug"])["compact_mode"] is False
    store.set_compact_mode(OWNER, b["slug"], True, now=NOW)
    b2 = store.get(OWNER, b["slug"])
    assert b2["compact_mode"] == 1
    assert store.vitals(OWNER, b["slug"])["compact_mode"] is True
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"], limit=10)]
    assert "compact_set" in kinds
    store.set_compact_mode(OWNER, b["slug"], False, now=NOW)
    assert store.vitals(OWNER, b["slug"])["compact_mode"] is False


# ── Template loading ─────────────────────────────────────────────────────

def test_templates_are_read_from_external_files():
    text = being_prompts.load("wake_task.md")
    assert "HONESTY OF RECORD" in text and "{talk_menu}" in text
    compact = being_prompts.load("wake_task.md", compact=True)
    assert "HONESTY OF RECORD" in compact
    assert len(compact) < len(text)


def test_compact_falls_back_to_full_when_no_variant_exists():
    # morning_note.md has no compact sibling — one-liners don't shrink.
    assert (being_prompts.load("morning_note.md", compact=True)
            == being_prompts.load("morning_note.md"))


def test_render_preserves_json_braces():
    b = {"compact_mode": 0}
    out = being_prompts.render(b, "digest_contract.md")
    assert '{"act_kind": ' in out and out.count("```") == 2


# ── Prompts: same contract, fewer words ──────────────────────────────────

def test_compact_tick_prompt_is_smaller_with_same_physics(store):
    full_b = _born(store, name="Puna")
    compact_b = _born(store, name="Sazeta", compact=True)
    p_full = life.compose_tick_prompt(full_b, now=NOW,
                                      wallet=store.wallet_view(full_b))
    p_compact = life.compose_tick_prompt(compact_b, now=NOW,
                                         wallet=store.wallet_view(compact_b))
    # Same contracts and anchors…
    for anchor in ("HONESTY OF RECORD", '"act_kind"', "```json",
                   "RIGHT NOW", "DRIVES"):
        assert anchor in p_full and anchor in p_compact
    # …fewer words.
    assert len(p_compact) < len(p_full)


def test_compact_dream_prompt_keeps_the_dream(store):
    b = _born(store, compact=True)
    p = life.compose_tick_prompt(b, kind="dream", now=NOW,
                                 wallet=store.wallet_view(b))
    assert "This is your DREAM" in p


def test_compact_faculty_prompts_keep_contracts(store):
    b = _born(store, stage="child", compact=True)
    sibs = [{"id": "x", "slug": "iskra-lada-1234", "name": "Lada",
             "stage": "child", "mood": ""}]
    orient = life.compose_orient_prompt(
        b, kind="wake", now=NOW, spent_today=0, wallet=store.wallet_view(b),
        percepts=None, first_of_day=False, siblings=sibs, letters_left=5,
        visitors=None)
    assert '"act_kind"' in orient and '"intent"' in orient
    talk = life.compose_talk_prompt(b, intent="say hi", sib=sibs[0],
                                    siblings=sibs, letters_left=5)
    assert '"letter"' in talk and "Lada" in talk
    journal = life.compose_journal_prompt(b, intent="", act_kind="journal",
                                          changed=[], visitors=None)
    assert '"journal_entry"' in journal
    act = life.compose_act_prompt(b, act_kind="create", intent="a poem",
                                  target="garden/poem.md")
    assert "WRITE the real file" in act


def test_compact_write_gate_still_stops_theater(store):
    b = _born(store, compact=True)
    gate = life.compose_write_gate_prompt(b, {"summary": "wrote a poem"})
    assert "STOP" in gate and "wrote a poem" in gate
    full = life.compose_write_gate_prompt(_born(store, name="Puna2"),
                                          {"summary": "wrote a poem"})
    assert len(gate) <= len(full)


def test_compact_mind_prompts_keep_link_schema(store):
    b = _born(store, compact=True)
    offer = being_prompts.render(b, "mind_link_offer.md")
    assert '"links"' in offer and "grew_from" in offer
    gate = mind.link_gate_prompt(store, b, {"links": []})
    assert "STOP" in gate and "grew_from" in gate
