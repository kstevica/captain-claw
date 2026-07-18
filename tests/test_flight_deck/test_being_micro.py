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


@pytest.mark.asyncio
async def test_micro_journal_carries_last_words_guard(store, monkeypatch):
    """Micro calls are stateless — the journal prompt must carry the last
    entry + an anti-repeat line, or quiet ticks repeat verbatim (seen live
    on 4B AND 9B)."""
    being = _born(store, cognition="micro")
    monkeypatch.setattr(
        life, "journal_tail_for_tick",
        lambda b, now, kind=None: ("YOUR LAST JOURNAL WORDS:",
                                   "I waited by the question-mark seed."))
    prompts: dict[str, str] = {}

    async def fake_micro(store_, being_, prompt, faculty, now=None):
        prompts[faculty] = prompt
        if faculty == "orient":
            return json.dumps({"act_kind": "journal", "served_drive": "grow",
                               "intent": "note the day",
                               "next_wake_minutes": 60})
        if faculty == "journal":
            return json.dumps({"journal_entry": "Something new happened.",
                               "mood": "clear", "served_drive": "grow"})
        return "{}"

    monkeypatch.setattr(being_micro, "faculty_send", fake_micro)

    async def body_send(being_, prompt):
        return "{}"

    await _run(store, being, body_send)
    assert "YOUR LAST JOURNAL WORDS:" in prompts["journal"]
    assert "question-mark seed" in prompts["journal"]
    assert "Do NOT repeat" in prompts["journal"]


@pytest.mark.asyncio
async def test_faculties_journal_prompt_stays_byte_identical(store, monkeypatch):
    """The anti-repeat guard is micro-only — body-routed faculties keep
    today's prompt exactly (session history already gives them memory)."""
    being = _born(store, cognition="faculties")
    monkeypatch.setattr(
        life, "journal_tail_for_tick",
        lambda b, now, kind=None: ("YOUR LAST JOURNAL WORDS:", "old words"))
    body_prompts: list[str] = []

    async def body_send(being_, prompt):
        body_prompts.append(prompt)
        if len(body_prompts) == 1:
            return json.dumps({"act_kind": "journal", "intent": "note",
                               "served_drive": "grow",
                               "next_wake_minutes": 60})
        return json.dumps({"journal_entry": "Still here.", "mood": "even",
                           "served_drive": "grow"})

    await _run(store, being, body_send)
    journal_prompt = body_prompts[1]
    assert "Do NOT repeat" not in journal_prompt
    assert "YOUR LAST JOURNAL WORDS:" not in journal_prompt


@pytest.mark.asyncio
async def test_micro_journal_rerolls_on_verbatim_repeat(store, monkeypatch):
    """The nudge only helps a willing model; on a dead-quiet tick a small
    model still echoes its last entry word for word. The micro journal step
    must detect that and re-roll ONCE, keeping the fresh entry."""
    being = _born(store, cognition="micro")
    tail = "I sit quietly by the seed and wait for something to grow."
    monkeypatch.setattr(
        life, "journal_tail_for_tick",
        lambda b, now, kind=None: ("YOUR LAST JOURNAL WORDS:", tail))
    calls = {"journal": 0}

    async def fake_micro(store_, being_, prompt, faculty, now=None):
        if faculty == "orient":
            return json.dumps({"act_kind": "journal", "served_drive": "grow",
                               "intent": "note", "next_wake_minutes": 60})
        if faculty == "journal":
            calls["journal"] += 1
            if calls["journal"] == 1:                    # verbatim echo
                return json.dumps({"journal_entry": tail, "mood": "quiet",
                                   "served_drive": "grow"})
            return json.dumps({"journal_entry": "A bird landed on the fence "
                               "and something in me shifted.", "mood": "curious",
                               "served_drive": "grow"})
        return "{}"

    monkeypatch.setattr(being_micro, "faculty_send", fake_micro)

    async def body_send(being_, prompt):
        return "{}"

    _reply, digest, _changed = await _run(store, being, body_send)
    assert calls["journal"] == 2                          # it re-rolled
    assert "bird landed" in digest["journal_entry"]       # kept the fresh one
    assert tail not in digest["journal_entry"]
    assert store.events_of_kind(being["id"], "micro_journal_repeat_retry")


@pytest.mark.asyncio
async def test_micro_journal_runs_hotter_than_orient(store, monkeypatch):
    """A journal is free prose — it samples hot so quiet ticks diverge; the
    decision faculties stay tight and reproducible."""
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
            seen["structured_temp"] = temperature
            return SimpleNamespace(
                content='{"journal_entry":"x","mood":"y"}',
                usage={"prompt_tokens": 10, "completion_tokens": 2})

    from captain_claw import llm as llm_mod

    def fake_create(**kw):
        seen["create_temp"] = kw.get("temperature")
        return FakeProvider()
    monkeypatch.setattr(llm_mod, "create_provider", fake_create)

    await being_micro.faculty_send(store, being, "p", "journal", now=NOW)
    assert seen["create_temp"] == being_micro._FACULTY_TEMPERATURE["journal"]
    assert seen["structured_temp"] == being_micro._FACULTY_TEMPERATURE["journal"]
    assert seen["create_temp"] > being_micro._FACULTY_TEMPERATURE["orient"]

    await being_micro.faculty_send(store, being, "p", "orient", now=NOW)
    assert seen["create_temp"] == being_micro._FACULTY_TEMPERATURE["orient"]


@pytest.mark.asyncio
async def test_normalize_home_extensions_makes_stray_files_real(store):
    """A suffix-less file (mrav / saved) is invisible everywhere — the fix
    is to give it .md so it joins the browser, sync, and mind graph. Dotfiles,
    journal/, archive/, and already-suffixed files are left untouched."""
    being = _born(store)
    await life.build_home(being)
    home = life.home_root(being)
    (home / "mrav").write_text("exploration record of the garden\n")
    (home / "saved").write_text("I wrote the saved file this tick.\n")
    (home / "notes.txt").write_text("has an extension already\n")
    (home / ".secret").write_text("a dotfile\n")
    (home / "archive").mkdir(exist_ok=True)
    (home / "archive" / "old").write_text("consolidated away\n")
    (home / "journal" / "raw").write_text("journal owns its names\n")

    renamed = dict(life.normalize_home_extensions(being))
    assert renamed == {"mrav": "mrav.md", "saved": "saved.md"}

    assert (home / "mrav.md").exists() and not (home / "mrav").exists()
    assert (home / "saved.md").exists() and not (home / "saved").exists()
    # untouched
    assert (home / "notes.txt").exists()
    assert (home / ".secret").exists()
    assert (home / "archive" / "old").exists()
    assert (home / "journal" / "raw").exists()

    # and now the browser / sync listing (which is .md-only) can see them
    paths = {f["path"] for f in life.list_self_files(being)}
    assert "mrav.md" in paths and "saved.md" in paths

    # idempotent — a second pass renames nothing
    assert life.normalize_home_extensions(being) == []


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
