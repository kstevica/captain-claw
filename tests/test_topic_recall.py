"""Context engine P3: the topic recall card, pins, and the topics tool.

Every test uses a tmp DB and a tmp HOME (nothing here may reach ~/.captain-claw).
"""

from __future__ import annotations

import types

import pytest

from captain_claw import speaker, topic_recall
from captain_claw.config import get_config
from captain_claw.session import Session
from captain_claw.speaker import Principal

MEMBER = Principal("u-member", "Ana", "Olga", "A", "process:helper:0123456789abcdef")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    cfg = get_config()
    for section, attr in (("memory", "path"), ("session", "path"), ("insights", "db_path"),
                          ("conversation_topics", "db_path"), ("nervous_system", "db_path")):
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    import captain_claw.conversation_topics as ct

    monkeypatch.setattr(ct, "_MANAGER", None)
    return home


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    import captain_claw.conversation_topics as ct

    manager = ct.ConversationTopicsManager(tmp_path / "topics.db")
    monkeypatch.setattr(ct, "_MANAGER", manager)
    monkeypatch.setattr("captain_claw.tools.conversation_topics.get_topics_manager", lambda: manager)
    yield manager
    if manager._conn is not None:
        manager._conn.close()


def _seed(mgr):
    munich = mgr.upsert_topic("Munich trip", summary="Trains and hotels for the Munich trip in May",
                              keywords=["travel", "munich"])
    mgr.add_messages(munich, [
        {"role": "user", "excerpt": "book the Munich hotel near the station", "msg_id": "old-1",
         "ts": "2026-09-12T10:00:00"},
        {"role": "agent", "excerpt": "Hotel Adler is 300 m from the station.", "msg_id": "old-2",
         "ts": "2026-09-12T10:01:00"},
        {"role": "user", "excerpt": "Ana's own Munich note", "msg_id": "ana-1",
         "ts": "2026-09-13T09:00:00", "speaker": "u-member"},
    ])
    brief = mgr.upsert_topic("Weekly brief", summary="Monday portfolio brief", keywords=["report"])
    mgr.add_messages(brief, [{"role": "user", "excerpt": "send the weekly brief", "msg_id": "b-1"}])
    invoices = mgr.upsert_topic("Biserka invoices", summary="Unpaid invoices from Biserka",
                                keywords=["finance"])
    mgr.add_messages(invoices, [{"role": "user", "excerpt": "chase Biserka", "msg_id": "i-1"}])
    return munich


# ── Deciding ───────────────────────────────────────────────────────────


def test_two_matched_words_with_a_clear_lead_recall_the_topic(mgr):
    _seed(mgr)
    d = topic_recall.decide(mgr, "which Munich hotel did we book again?")
    assert d["topic"] == "munich-trip" and d["rule"] == "words"
    assert d["candidates"][0]["matched_terms"]


def test_a_single_loose_word_abstains(mgr):
    _seed(mgr)
    d = topic_recall.decide(mgr, "write me a brief poem about autumn")
    assert d["topic"] is None and d["reason"] in {"no clear match", "no match"}


def test_meaning_alone_needs_a_strong_clear_match(mgr):
    _seed(mgr)

    def embedder(texts):
        return [[1.0, 0.0] if ("Bavaria" in t or "Munich" in t) else [0.0, 1.0] for t in texts]

    d = topic_recall.decide(mgr, "a weekend in Bavaria", embedder=embedder)
    assert d["topic"] == "munich-trip" and d["rule"] == "meaning"

    def flat(texts):
        return [[0.6, 0.8] for _ in texts]          # everything equally close

    assert topic_recall.decide(mgr, "a weekend in Bavaria", embedder=flat)["topic"] is None


def test_a_topic_already_in_the_conversation_is_not_recalled(mgr):
    _seed(mgr)
    d = topic_recall.decide(mgr, "which Munich hotel did we book?", live_ids={"old-2"})
    assert d["topic"] is None and d["reason"] == "already in this conversation"


def test_members_recall_only_topics_they_spoke_in(mgr):
    _seed(mgr)
    assert topic_recall.decide(mgr, "Munich hotel booking", speaker="u-member")["topic"] == "munich-trip"
    assert topic_recall.decide(mgr, "Biserka unpaid invoices", speaker="u-member")["topic"] is None
    assert topic_recall.decide(mgr, "Munich hotel booking", speaker="")["reason"] == "unverified member"


def test_hidden_topics_are_never_recalled(mgr):
    _seed(mgr)
    mgr.set_hidden("munich-trip", True)
    assert topic_recall.decide(mgr, "which Munich hotel did we book again?")["topic"] != "munich-trip"


# ── Rendering and pins ─────────────────────────────────────────────────


def test_the_card_is_short_and_dated(mgr):
    _seed(mgr)
    topic = mgr.get_topic("munich-trip", max_excerpts=3, speaker="")
    card = topic_recall.render_card(topic)
    assert card.startswith(topic_recall.RECALL_LEAD)
    assert "last discussed" in card and "id munich-trip" in card
    assert "Ana's own" not in card                      # the owner sees the owner's excerpts
    assert card.count("\n- [") == 2
    assert "topics action=get topic=munich-trip" in card


def test_a_pin_rides_the_next_turns_then_expires():
    s = Session(id="s1", name="d")
    s.add_message("user", "pin it", origin="human")
    assert topic_recall.pin(s, "munich-trip", 2) == 2
    assert topic_recall.active_pins(s) == ["munich-trip"]       # same turn
    shown = []
    for n in range(4):
        s.add_message("user", f"turn {n}", origin="human")
        shown.append(bool(topic_recall.active_pins(s)))
        shown.append(bool(topic_recall.active_pins(s)))         # a rebuild in the same turn
    assert shown == [True, True, True, True, False, False, False, False]


def test_pins_are_capped_and_can_be_dropped():
    s = Session(id="s1", name="d")
    s.add_message("user", "x", origin="human")
    for tid in ("a", "b", "c", "d"):
        topic_recall.pin(s, tid, 5)
    assert list(s.metadata[topic_recall.PIN_KEY]) == ["b", "c", "d"]
    assert topic_recall.unpin(s, "c") == ["c"]
    assert topic_recall.unpin(s) == ["b", "d"]
    assert topic_recall.active_pins(s) == []


# ── In the per-turn context block ──────────────────────────────────────


class _Stub:
    provider = "openai"
    model = "stub"

    def count_tokens(self, text):
        return len(str(text or "").split()) or 1


def _agent(monkeypatch, mode: str):
    from captain_claw.agent import Agent

    monkeypatch.setattr(get_config().conversation_topics, "recall", mode)
    agent = Agent(provider=_Stub())
    agent.session = Session(id="s1", name="d")
    agent._build_env_now_text = lambda: ""
    return agent


def _turn(agent, text, origin="human"):
    agent.session.add_message("user", text, origin=origin)
    agent._turn_user_text = text
    agent._turn_origin = (origin, "", None)
    agent._turn_context_notes = None
    agent._build_messages(tool_messages_from_index=len(agent.session.messages) - 1, query=text)
    return agent.last_context_window, agent._turn_context_notes[1]


@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
def test_recall_modes(mgr, monkeypatch, mode):
    _seed(mgr)
    agent = _agent(monkeypatch, mode)
    window, notes = _turn(agent, "which Munich hotel did we book again?")
    kinds = [kind for kind, _ in notes]
    assert ("topic_recall" in kinds) == (mode == "on")
    recall = window.get("topic_recall")
    if mode == "off":
        assert recall is None
    else:
        assert recall["topic"] == "munich-trip" and recall["mode"] == mode


def test_automated_turns_recall_nothing(mgr, monkeypatch):
    _seed(mgr)
    agent = _agent(monkeypatch, "on")
    window, notes = _turn(agent, "[SCHEDULED TASK — cron job 7] Munich hotel prices", origin="cron")
    assert "topic_recall" not in [kind for kind, _ in notes]
    assert window.get("topic_recall") is None


def test_a_pinned_topic_rides_the_block_even_with_recall_off(mgr, monkeypatch):
    _seed(mgr)
    agent = _agent(monkeypatch, "off")
    agent.session.add_message("user", "pin the Munich trip", origin="human")
    topic_recall.pin(agent.session, "munich-trip", 3)
    _window, notes = _turn(agent, "what's the weather?")
    pinned = [text for kind, text in notes if kind == "pinned_topic"]
    assert pinned and pinned[0].startswith(topic_recall.PIN_LEAD)


# ── The topics tool ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_recall_pin_and_unpin(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    _seed(mgr)
    agent = types.SimpleNamespace(session=Session(id="s1", name="d"))
    agent.session.add_message("user", "x", origin="human")
    tool = TopicsTool()

    res = await tool.execute(action="recall", query="the Munich hotel booking", _agent=agent)
    assert res.success and "Topic: Munich trip" in res.content

    res = await tool.execute(action="pin", topic="Munich trip", _agent=agent)
    assert res.success and "munich-trip" in agent.session.metadata[topic_recall.PIN_KEY]
    res = await tool.execute(action="unpin", topic="munich-trip", _agent=agent)
    assert res.success and not agent.session.metadata[topic_recall.PIN_KEY]
    res = await tool.execute(action="pin", topic="no such topic", _agent=agent)
    assert not res.success


@pytest.mark.asyncio
async def test_tool_pin_respects_member_scope(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    _seed(mgr)
    agent = types.SimpleNamespace(session=Session(id="s1", name="d"),
                                  _speaker_scoped=True, _speaker_principal=MEMBER)
    agent.session.add_message("user", "x", origin="human")
    res = await TopicsTool().execute(action="pin", topic="Biserka invoices", _agent=agent)
    assert not res.success                     # Ana has no messages in that topic
    tok = speaker.bind(MEMBER)
    try:
        res = await TopicsTool().execute(action="recall", query="Biserka unpaid invoices", _agent=agent)
    finally:
        speaker.reset(tok)
    assert "chase Biserka" not in res.content


# ── Review round ───────────────────────────────────────────────────────


def test_agreement_on_one_tied_word_is_not_enough(mgr):
    for name in ("London conference", "Vienna weekend"):
        mgr.upsert_topic(name, summary=f"{name}: hotel booked", keywords=["travel"])

    def embedder(texts):
        return [[1.0, 0.0] if t == "and the hotel?" else
                [0.535, 0.845] if "London" in t else [0.501, 0.865] for t in texts]

    d = topic_recall.decide(mgr, "and the hotel?", embedder=embedder)
    assert d["topic"] is None
    both = [c for c in d["candidates"] if "fts_rank" in c and "vec_rank" in c]
    assert both and all("cosine" in c and "bm25" in c for c in both)


def test_attachment_paths_do_not_crowd_out_the_question(mgr):
    from captain_claw.conversation_topics import query_terms

    _seed(mgr)
    text = ("[Attached image: /home/u/.captain-claw/workspace/whatsapp/media/aa76a2ff03a4/IMG-20261008.jpg]\n"
            "is this the Biserka invoice?")
    assert query_terms(text) == ["biserka", "invoice"]
    assert topic_recall.decide(mgr, text)["topic"] == "biserka-invoices"


def test_words_inside_other_words_do_not_count(mgr):
    mgr.upsert_topic("Startup Grind Zagreb", summary="Startup meetup in Zagreb")
    mgr.upsert_topic("Car service", summary="Annual car service")
    assert topic_recall.decide(mgr, "the art fair in Zagreb")["topic"] is None
    assert topic_recall.decide(mgr, "ice cream service")["topic"] is None


def test_pins_are_neither_spent_nor_shown_on_automated_turns(mgr, monkeypatch):
    _seed(mgr)
    agent = _agent(monkeypatch, "off")
    agent.session.add_message("user", "pin the Munich trip", origin="human")
    topic_recall.pin(agent.session, "munich-trip", 1)
    _w, notes = _turn(agent, "[SCHEDULED TASK — cron job 7] brief", origin="cron")
    assert notes == []
    _w, notes = _turn(agent, "[Autonomous nudge] check mail", origin="autonomy")
    assert notes == []
    _w, notes = _turn(agent, "what's next?")                      # the one pinned turn
    assert [kind for kind, _ in notes] == ["pinned_topic"]
    _w, notes = _turn(agent, "and then?")
    assert notes == []


def test_a_members_card_is_dated_by_their_own_messages(mgr):
    _seed(mgr)
    topic = mgr.get_topic("munich-trip", max_excerpts=3, speaker="u-member")
    card = topic_recall.render_card(topic, own_excerpts=True)
    assert "last discussed 2026-09-13" in card


@pytest.mark.asyncio
async def test_hidden_topics_cannot_be_pinned_or_ride_a_pin(mgr, monkeypatch):
    from captain_claw.tools.conversation_topics import TopicsTool

    _seed(mgr)
    agent = _agent(monkeypatch, "off")
    agent.session.add_message("user", "x", origin="human")
    topic_recall.pin(agent.session, "munich-trip", 3)
    mgr.set_hidden("munich-trip", True)
    _w, notes = _turn(agent, "what now?")
    assert "pinned_topic" not in [kind for kind, _ in notes]
    res = await TopicsTool().execute(action="pin", topic="munich-trip", _agent=agent)
    assert not res.success and "hidden" in res.error


@pytest.mark.asyncio
async def test_a_members_tool_recall_ranks_among_their_own_topics(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    for n in range(6):
        mgr.upsert_topic(f"Owner hotel booking {n}", summary="hotel booking for the owner")
    ana = mgr.upsert_topic("Ana conference stay", summary="hotel booking for Ana's conference")
    mgr.add_messages(ana, [{"role": "user", "excerpt": "book my hotel", "msg_id": "a-1",
                            "speaker": "u-member"}])
    agent = types.SimpleNamespace(session=Session(id="s1", name="d"),
                                  _speaker_scoped=True, _speaker_principal=MEMBER)
    res = await TopicsTool().execute(action="recall", query="hotel booking", _agent=agent)
    assert "Ana conference stay" in res.content


# ── Integration review round ───────────────────────────────────────────


@pytest.mark.parametrize("flag", ["_suppress_memory_context", "_skip_memory_injection"])
def test_no_recall_on_a_turn_that_asked_for_no_memory(mgr, monkeypatch, flag):
    _seed(mgr)
    agent = _agent(monkeypatch, "on")
    setattr(agent, flag, True)
    window, notes = _turn(agent, "which Munich hotel did we book again?")
    assert "topic_recall" not in [kind for kind, _ in notes]
    if flag == "_suppress_memory_context":
        assert window["topic_recall"]["reason"] == "memory suppressed"


def test_public_visitors_get_nothing_from_the_owners_store(mgr, monkeypatch):
    _seed(mgr)
    agent = _agent(monkeypatch, "on")
    agent._public_scoped = True
    window, notes = _turn(agent, "which Munich hotel did we book again?")
    assert notes == [] and window["topic_recall"] is None


def test_the_trace_says_whether_the_card_was_sent(mgr, monkeypatch):
    _seed(mgr)
    agent = _agent(monkeypatch, "on")
    window, _notes = _turn(agent, "which Munich hotel did we book again?")
    assert window["topic_recall"]["sent"] is True
    assert window["pinned_topics_used"] == 0


def test_small_tiers_get_short_cards_and_one_pin(mgr):
    _seed(mgr)
    topic = mgr.get_topic("munich-trip", max_excerpts=6, speaker="")
    card = topic_recall.render_card(topic, pinned=True, budget_tokens=6000)
    assert card.count("\n- [") == 2
    assert topic_recall.pins_shown(6000) == 1 and topic_recall.pins_shown(200000) == topic_recall.MAX_PINS


def test_the_config_default_matches_the_rule_default():
    import inspect

    from captain_claw.config import ConversationTopicsConfig

    params = inspect.signature(topic_recall.decide).parameters
    cfg = ConversationTopicsConfig()
    assert params["agree_min_cosine"].default == cfg.recall_agree_min_cosine
    assert params["min_cosine"].default == cfg.recall_min_cosine
    assert params["cosine_margin"].default == cfg.recall_cosine_margin
    assert params["bm25_margin"].default == cfg.recall_bm25_margin
