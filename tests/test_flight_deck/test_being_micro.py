"""Micro cognition: JSON faculties grammar-locked on the micro tier.

cognition='micro' must (1) route orient/talk/journal/connect to
being_micro.faculty_send and NEVER route act there, (2) fall back to the
body loudly when the micro path has nothing to give, (3) debit the wallet
per call like the feet do, and (4) leave 'monolith'/'faculties' beings
byte-identical — the whole point of the opt-in.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_micro
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _born(store, name="Mrvica", cognition=None):
    b = store.conceive(OWNER, name, preset="explorer", allowance_preset="2M",
                       birth_letter="Think small and true.", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if cognition:
        store.set_cognition(OWNER, b["slug"], cognition, now=NOW)
    return store.get(OWNER, b["slug"])


def _run(store, being, send, **kw):
    return life._run_faculties(
        store, being, kind="wake", now=NOW, send=send, senses=[],
        view=store.wallet_view(being), spent_today=0, first_of_day=False,
        siblings=[], letters_left=None, visitors=[], last_refusals=[],
        drives=being.get("drives") or {}, **kw)


# ── store ────────────────────────────────────────────────────────────


def test_set_cognition_micro_accepted_and_junk_rejected(store):
    b = _born(store)
    out = store.set_cognition(OWNER, b["slug"], "micro", now=NOW)
    assert out["cognition"] == "micro"
    # the two existing values keep working
    assert store.set_cognition(OWNER, b["slug"], "faculties",
                               now=NOW)["cognition"] == "faculties"
    assert store.set_cognition(OWNER, b["slug"], "monolith",
                               now=NOW)["cognition"] == "monolith"
    with pytest.raises(BeingError):
        store.set_cognition(OWNER, b["slug"], "nano", now=NOW)


# ── schemas ──────────────────────────────────────────────────────────


def test_schemas_track_act_kinds_and_cover_micro_faculties():
    orient = being_micro.faculty_schema("orient")
    assert orient["properties"]["act_kind"]["enum"] == list(life.ACT_KINDS)
    for faculty in being_micro.MICRO_FACULTIES:
        schema = being_micro.faculty_schema(faculty)
        assert schema is not None and schema["type"] == "object"
        assert schema.get("required"), faculty
        # rare structured moves must be able to ride through
        assert schema.get("additionalProperties") is True
    # act (and anything unknown) is body-only
    assert being_micro.faculty_schema("act") is None
    assert being_micro.faculty_schema("write_gate") is None


# ── routing ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_micro_routes_json_faculties_and_skips_body(store, monkeypatch):
    being = _born(store, cognition="micro")
    micro_calls: list[str] = []

    async def fake_micro(store_, being_, prompt, faculty, now=None):
        micro_calls.append(faculty)
        if faculty == "orient":
            return json.dumps({
                "act_kind": "journal", "served_drive": "grow",
                "intent": "note the quiet day", "next_wake_minutes": 60,
                "message_to_parent": None})
        if faculty == "journal":
            return json.dumps({
                "journal_entry": "A quiet day; I noted what I saw.",
                "mood": "calm", "served_drive": "grow"})
        return "{}"

    monkeypatch.setattr(being_micro, "faculty_send", fake_micro)
    body_calls: list[str] = []

    async def body_send(being_, prompt):
        body_calls.append(prompt)
        return "{}"

    _reply, digest, _changed = await _run(store, being, body_send)
    assert micro_calls == ["orient", "journal"]
    assert body_calls == []  # act_kind=journal → no body step at all
    assert digest["act_kind"] == "journal"
    assert "quiet day" in (digest.get("journal_entry") or "")
    assert digest.get("fallback") is False


@pytest.mark.asyncio
async def test_micro_falls_back_to_body_loudly(store, monkeypatch):
    being = _born(store, cognition="micro")

    async def no_micro(store_, being_, prompt, faculty, now=None):
        return None  # tier missing / provider down

    monkeypatch.setattr(being_micro, "faculty_send", no_micro)
    body_calls: list[str] = []

    async def body_send(being_, prompt):
        body_calls.append(prompt)
        if len(body_calls) == 1:
            return json.dumps({"act_kind": "rest", "served_drive": "survive",
                               "intent": "rest", "next_wake_minutes": 90})
        return json.dumps({"journal_entry": "Rested.", "mood": "soft",
                           "served_drive": "survive"})

    _reply, digest, _changed = await _run(store, being, body_send)
    assert len(body_calls) == 2  # orient + journal served by the body
    assert digest["act_kind"] == "rest"
    assert store.events_of_kind(being["id"], "micro_fallback_body")


@pytest.mark.asyncio
async def test_faculties_cognition_never_touches_micro(store, monkeypatch):
    """The regression that matters: existing beings are byte-identical."""
    being = _born(store, cognition="faculties")

    async def explode(*a, **k):
        raise AssertionError("micro path used by a 'faculties' being")

    monkeypatch.setattr(being_micro, "faculty_send", explode)
    body_calls: list[str] = []

    async def body_send(being_, prompt):
        body_calls.append(prompt)
        if len(body_calls) == 1:
            return json.dumps({"act_kind": "rest", "intent": "rest",
                               "served_drive": "survive",
                               "next_wake_minutes": 60})
        return json.dumps({"journal_entry": "Still here.", "mood": "even",
                           "served_drive": "survive"})

    _reply, digest, _changed = await _run(store, being, body_send)
    assert len(body_calls) == 2
    assert digest["act_kind"] == "rest"


# ── the one-shot itself ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_faculty_send_debits_wallet_and_returns_json(store, monkeypatch):
    being = _born(store, cognition="micro")

    from captain_claw.flight_deck import auth as auth_mod
    monkeypatch.setattr(auth_mod, "get_db", lambda: object())

    from captain_claw.flight_deck import basna_routes
    async def fake_tiers(db, owner_id):
        return ({"micro": {"provider": "ollama", "model": "qwen3.5:4b",
                           "base_url": "", "api_key": ""}}, [])
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", fake_tiers)

    seen: dict = {}

    class FakeProvider:
        async def complete_structured(self, messages, schema,
                                      temperature=None, max_tokens=None):
            seen["schema"] = schema
            seen["system"] = messages[0].content
            seen["user"] = messages[1].content
            return SimpleNamespace(
                content='{"act_kind":"rest","intent":"breathe"}',
                usage={"prompt_tokens": 100, "completion_tokens": 10,
                       "total_tokens": 110})

    from captain_claw import llm as llm_mod
    monkeypatch.setattr(llm_mod, "create_provider", lambda **kw: FakeProvider())

    before = store.wallet_view(being)["balance_tokens"]
    text = await being_micro.faculty_send(
        store, being, "[LIFE TICK — orient] decide.", "orient", now=NOW)
    assert json.loads(text)["act_kind"] == "rest"
    assert seen["schema"]["properties"]["act_kind"]["enum"] == list(life.ACT_KINDS)
    assert being["name"] in seen["system"]
    assert "orient" in seen["user"]

    after = store.wallet_view(being)["balance_tokens"]
    assert before - after == 110  # tier weight defaults to 1.0


@pytest.mark.asyncio
async def test_faculty_send_none_without_db_or_tier(store, monkeypatch):
    being = _born(store, cognition="micro")

    from captain_claw.flight_deck import auth as auth_mod

    def no_db():
        raise AssertionError("FD db not initialized")
    monkeypatch.setattr(auth_mod, "get_db", no_db)
    assert await being_micro.faculty_send(
        store, being, "p", "orient", now=NOW) is None

    # db up but no micro tier configured → also None (body fallback)
    monkeypatch.setattr(auth_mod, "get_db", lambda: object())
    from captain_claw.flight_deck import basna_routes
    async def empty_tiers(db, owner_id):
        return ({}, [])
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", empty_tiers)
    assert await being_micro.faculty_send(
        store, being, "p", "orient", now=NOW) is None
