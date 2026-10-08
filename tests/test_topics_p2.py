"""Context engine P2: the topic store keeps only conversation, tracks its
progress in the store, and ranks topics instead of substring-matching.

Every test uses a tmp DB and a tmp HOME (nothing here may reach ~/.captain-claw).
"""

from __future__ import annotations

import json
import sqlite3
import types

import pytest

from captain_claw.config import get_config
from captain_claw.session import Session


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
    monkeypatch.setattr(cfg.conversation_topics, "include_narration", False)
    import captain_claw.conversation_topics as ct

    monkeypatch.setattr(ct, "_MANAGER", None)
    return home


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    import captain_claw.conversation_topics as ct

    manager = ct.ConversationTopicsManager(tmp_path / "topics.db")
    monkeypatch.setattr(ct, "_MANAGER", manager)
    yield manager
    if manager._conn is not None:
        manager._conn.close()


class _Classifier:
    """Stands in for the agent's model: files every message under one topic."""

    def __init__(self, label="Munich trip"):
        self.label = label
        self.prompts: list[str] = []

    async def __call__(self, messages, tools=None, interaction_label="", max_tokens=None):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        batch = prompt.split("NEW messages:\n", 1)[1]
        idxs = [int(line[1:line.index("]")]) for line in batch.splitlines() if line.startswith("[")]
        reply = [{"label": self.label, "summary": "Planning it", "keywords": ["travel"],
                  "messages": idxs}]
        return types.SimpleNamespace(content=json.dumps(reply))


def _agent(session: Session, classifier: _Classifier | None = None):
    return types.SimpleNamespace(session=session, _complete_with_guards=classifier or _Classifier())


def _turn(s: Session, text: str, reply: str, *, channel: str = "web", origin: str = "human") -> None:
    s.add_message("user", text, origin=origin, channel=channel)
    s.add_message("assistant", reply, origin="model")


# ── What goes into a topic ─────────────────────────────────────────────


def test_only_typed_messages_and_final_replies_are_ingested(mgr):
    from captain_claw.conversation_topics import _conversation_items

    s = Session(id="sess-1", name="d")
    surface = ("[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
               "Keep replies short.\nUSER MESSAGE:\n")
    s.add_message("user", surface + "plan the Munich trip", origin="human", channel="whatsapp")
    s.add_message("assistant", "", tool_calls=[{"id": "c1", "type": "function",
                  "function": {"name": "web_search", "arguments": "{}"}}], origin="model")
    s.add_message("tool", "results", tool_call_id="c1", tool_name="web_search", origin="tool")
    s.add_message("assistant", "Let me check the trains:", origin="model")
    s.add_message("user", "You announced intent without acting. Do NOT narrate", origin="corrective")
    s.add_message("assistant", "Trains leave at 9.", origin="model")
    s.add_message("user", "[Flight Deck] Agent 'x' has joined the fleet on port 1.", origin="fleet_notice")
    s.add_message("user", "[SCHEDULED TASK — cron job 7] send the brief", origin="cron")
    s.add_message("assistant", "Brief sent.", origin="model")
    s.add_message("user", "and hotels?", origin="human", channel="web")
    s.add_message("assistant", "Hotel Adler has rooms.", origin="model")

    items = _conversation_items(_agent(s), s.messages)

    assert [(i["role"], i["excerpt"]) for i in items] == [
        ("user", "plan the Munich trip"),
        ("agent", "Trains leave at 9."),
        ("user", "and hotels?"),
        ("agent", "Hotel Adler has rooms."),
    ]
    assert [i["channel"] for i in items] == ["whatsapp", "whatsapp", "web", "web"]
    assert {i["session_id"] for i in items} == {"sess-1"}


def test_progress_survives_a_restart_and_a_compaction(mgr):
    import captain_claw.conversation_topics as ct

    s = Session(id="s1", name="d")
    for n in range(3):
        _turn(s, f"question {n}", f"answer {n}")
    first, _ = ct._collect_new_messages(_agent(s))
    assert len(first) == 6
    mgr.mark_seen([i["msg_id"] for i in first])

    _turn(s, "question 3", "answer 3")
    # A fresh agent object (restart): only the new turn is pending.
    assert [i["excerpt"] for i in ct._collect_new_messages(_agent(s))[0]] == ["question 3", "answer 3"]
    # Compaction drops the old messages: nothing old comes back, nothing stalls.
    s.messages = s.messages[-4:]
    assert [i["excerpt"] for i in ct._pending_items(_agent(s))] == ["question 3", "answer 3"]


def test_a_session_new_to_topics_starts_from_its_newest_messages(mgr):
    import captain_claw.conversation_topics as ct

    s = Session(id="s1", name="d")
    for n in range(10):
        _turn(s, f"question {n}", f"answer {n}")
    pending = ct._pending_items(_agent(s), cap=4)
    assert [i["excerpt"] for i in pending] == ["question 8", "answer 8", "question 9", "answer 9"]


@pytest.mark.asyncio
async def test_a_pass_pages_through_the_backlog_oldest_first(mgr, monkeypatch):
    import captain_claw.conversation_topics as ct

    monkeypatch.setattr(get_config().conversation_topics, "max_messages_per_pass", 5)
    s = Session(id="s1", name="d")
    _turn(s, "question 0", "answer 0")
    classifier = _Classifier()
    agent = _agent(s, classifier)
    await ct.classify_topics(agent)                      # sets the watermark
    for n in range(1, 11):
        _turn(s, f"question {n}", f"answer {n}")          # 20 pending

    await ct.classify_topics(agent)
    assert len(classifier.prompts) == 1 + 3               # three batches of five
    left = [i["excerpt"] for i in ct._pending_items(agent)]
    assert left == ["answer 8", "question 9", "answer 9", "question 10", "answer 10"]
    topic = mgr.get_topic("munich-trip", max_excerpts=200)
    assert topic["messages"][0]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_a_pass_runs_once_enough_conversation_is_pending(mgr, monkeypatch):
    import captain_claw.conversation_topics as ct

    tc = get_config().conversation_topics
    monkeypatch.setattr(tc, "interval_messages", 4)
    monkeypatch.setattr(tc, "cooldown_seconds", 1)
    s = Session(id="s1", name="d")
    _turn(s, "question 0", "answer 0")
    s.add_message("user", "[Flight Deck] Agent 'x' has joined the fleet on port 1.", origin="fleet_notice")
    s.add_message("user", "[Flight Deck] Agent 'y' has joined the fleet on port 2.", origin="fleet_notice")
    agent = _agent(s)
    assert await ct.maybe_classify_topics(agent) is None   # 2 conversation messages
    _turn(s, "question 1", "answer 1")
    assert await ct.maybe_classify_topics(agent) == 1


# ── Ranked search ──────────────────────────────────────────────────────


def _seed(mgr):
    mgr.upsert_topic("Munich trip", summary="Trains and hotels for Munich in May", keywords=["travel"])
    mgr.upsert_topic("Putovanje u Split", summary="Trajekt i smještaj", keywords=["putovanje"])
    mgr.upsert_topic("Biserka invoices", summary="Unpaid invoices from Biserka", keywords=["finance"])
    mgr.upsert_topic("Weekly brief", summary="Monday brief mentions Munich once", keywords=["report"])


def test_label_matches_outrank_summary_mentions(mgr):
    _seed(mgr)
    rows = mgr.rank_topics("what about the munich hotels")
    assert [r["id"] for r in rows][:2] == ["munich-trip", "weekly-brief"]
    assert rows[0]["matched_terms"] == ["munich", "hotels"]
    assert rows[0]["bm25"] < rows[1]["bm25"]


def test_inflected_words_match_by_stem(mgr):
    _seed(mgr)
    assert mgr.rank_topics("kad je trajekt za putovanja")[0]["id"] == "putovanje-u-split"


def test_typed_search_matches_word_prefixes_then_substrings(mgr):
    _seed(mgr)
    assert [r["id"] for r in mgr.search_topics("Bise")] == ["biserka-invoices"]
    assert [r["id"] for r in mgr.search_topics("serka")] == ["biserka-invoices"]   # substring fallback
    assert mgr.search_topics("") and len(mgr.search_topics("", limit=2)) == 2


def test_embedding_leg_finds_topics_without_shared_words(mgr):
    _seed(mgr)
    calls: list[int] = []

    def embedder(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] if ("Bavaria" in t or "Munich" in t) else [0.0, 1.0] for t in texts]

    rows = mgr.rank_topics("weekend in Bavaria", embedder=embedder)
    assert rows[0]["id"] in {"munich-trip", "weekly-brief"} and rows[0]["cosine"] == 1.0
    assert rows[0]["bm25"] is None
    mgr.rank_topics("weekend in Bavaria", embedder=embedder)
    assert calls == [5, 1]            # topic vectors are cached; only the query is embedded again


def test_hidden_topics_stay_out_of_recall_but_stay_listed(mgr):
    _seed(mgr)
    assert mgr.set_hidden("munich-trip", True)
    assert "munich-trip" not in [r["id"] for r in mgr.rank_topics("munich")]
    assert "munich-trip" not in [r["id"] for r in mgr.list_topics()]
    listed = {r["id"]: r["hidden"] for r in mgr.list_topics(include_hidden=True)}
    assert listed["munich-trip"] == 1


# ── Store maintenance ──────────────────────────────────────────────────


def test_machine_text_topics_are_hidden_once(tmp_path):
    import captain_claw.conversation_topics as ct

    path = tmp_path / "old.db"
    m = ct.ConversationTopicsManager(path)
    fleet = m.upsert_topic("Fleet changes", summary="agents joining")
    m.add_messages(fleet, [{"role": "user", "excerpt": f"[Flight Deck] Agent 'a{i}' has joined the fleet on port {i}."}
                           for i in range(3)])
    starred = m.upsert_topic("Cron log", summary="cron")
    m.add_messages(starred, [{"role": "user", "excerpt": "[SCHEDULED TASK — cron job 1] x"} for _ in range(2)])
    m.set_star(starred, True)
    real = m.upsert_topic("Munich trip", summary="trip")
    m.add_messages(real, [{"role": "user", "excerpt": "plan the trip"},
                          {"role": "user", "excerpt": "[Flight Deck] Agent 'b' has joined the fleet on port 9."}])
    m._conn.execute("DELETE FROM topics_meta")       # as if the store predates the sweep
    m._conn.commit()
    m._conn.close()

    m = ct.ConversationTopicsManager(path)
    hidden = {r["id"] for r in m.list_topics(include_hidden=True) if r["hidden"]}
    assert hidden == {"fleet-changes"}
    m.set_hidden("fleet-changes", False)
    m._conn.close()
    m = ct.ConversationTopicsManager(path)              # the user's un-hide sticks
    assert not any(r["hidden"] for r in m.list_topics(include_hidden=True))
    m._conn.close()


def test_an_old_store_gains_the_new_columns(tmp_path):
    import captain_claw.conversation_topics as ct

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE topics (id TEXT PRIMARY KEY, label TEXT NOT NULL, summary TEXT NOT NULL "
        "DEFAULT '', keywords TEXT NOT NULL DEFAULT '', msg_count INTEGER NOT NULL DEFAULT 0, "
        "starred INTEGER NOT NULL DEFAULT 0, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);"
        "CREATE TABLE topic_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_id TEXT NOT NULL, "
        "role TEXT NOT NULL DEFAULT '', channel TEXT NOT NULL DEFAULT '', excerpt TEXT NOT NULL "
        "DEFAULT '', msg_id TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL, speaker TEXT NOT NULL DEFAULT '');"
    )
    conn.commit()
    conn.close()
    m = ct.ConversationTopicsManager(path)
    try:
        cols = {r[1] for r in m._conn.execute("PRAGMA table_info(topics)")}
        tm_cols = {r[1] for r in m._conn.execute("PRAGMA table_info(topic_messages)")}
        assert "hidden" in cols and "session_id" in tm_cols
    finally:
        m._conn.close()


def test_pruning_drops_starred_topics_last(mgr):
    a = mgr.upsert_topic("Old starred")
    mgr.set_star(a, True)
    for n in range(3):
        mgr.upsert_topic(f"Newer {n}")
    mgr.prune_topics(2)
    assert {r["id"] for r in mgr.list_topics(include_hidden=True)} == {"old-starred", "newer-2"}


# ── The classifier's view of existing topics ───────────────────────────


@pytest.mark.asyncio
async def test_classifier_sees_relevant_and_recent_topics_not_all(mgr):
    import captain_claw.conversation_topics as ct

    mgr.upsert_topic("Munich trip", summary="Trains and hotels for Munich. " + "detail " * 60)
    for n in range(40):
        mgr.upsert_topic(f"Filler topic {n}", summary=f"unrelated filler number {n}")
    s = Session(id="s1", name="d")
    _turn(s, "which Munich hotel did we pick?", "Hotel Adler.")
    classifier = _Classifier()
    await ct._classify_and_store(_agent(s, classifier), ct._pending_items(_agent(s)))

    existing = classifier.prompts[0].split("NEW messages:", 1)[0]
    listed = [line for line in existing.splitlines() if line.startswith("- ")]
    assert listed[0].startswith("- Munich trip:")
    assert len(listed[0]) > 300                    # whole summary, not a 160-char stub
    assert len(listed) == 1 + ct._CLASSIFIER_RECENT
    assert "Filler topic 0:" not in existing       # old and unrelated


# ── Review round ───────────────────────────────────────────────────────


def test_a_reply_followed_by_tool_rows_is_still_the_final_reply(mgr):
    from captain_claw.conversation_topics import _conversation_items

    s = Session(id="s1", name="d")
    s.add_message("tool", "incoming", tool_name="telegram", origin="tool")
    s.add_message("user", "question 0", origin="human", channel="telegram")
    s.add_message("assistant", "answer 0", origin="model")
    s.add_message("tool", "outgoing", tool_name="telegram", origin="tool")       # delivery note
    s.add_message("tool", "spoken", tool_name="pocket_tts", origin="tool")
    s.add_message("tool", "prompt start", tool_name="cron", origin="tool")       # cron note
    s.add_message("user", "[SCHEDULED TASK — cron job 1] x", origin="cron")
    s.add_message("assistant", "cron done", origin="model")
    items = _conversation_items(_agent(s), s.messages)
    assert [(i["role"], i["excerpt"]) for i in items] == [("user", "question 0"), ("agent", "answer 0")]


def test_a_relayed_delegated_result_counts_as_the_asking_turns_reply(mgr):
    from captain_claw.conversation_topics import _conversation_items

    s = Session(id="s1", name="d")
    s.add_message("user", "research the Munich hotels", origin="human", channel="whatsapp")
    s.add_message("assistant", "I've asked the researcher; I'll report back.", origin="model")
    s.add_message("user", "[Delegated result from researcher] Adler is best.", origin="delegated_result")
    s.add_message("assistant", "The researcher found Hotel Adler is best.", origin="model")
    items = _conversation_items(_agent(s), s.messages)
    assert [i["excerpt"] for i in items] == [
        "research the Munich hotels", "I've asked the researcher; I'll report back.",
        "The researcher found Hotel Adler is best.",
    ]
    assert items[-1]["channel"] == "whatsapp"


def test_a_turn_filed_by_hand_does_not_skip_the_pending_ones(mgr):
    import captain_claw.conversation_topics as ct

    s = Session(id="s1", name="d")
    _turn(s, "question 0", "answer 0")
    first = ct._pending_items(_agent(s))
    mgr.mark_seen([i["msg_id"] for i in first])
    _turn(s, "question 1", "answer 1")
    _turn(s, "topic chat question", "topic chat answer")
    # What rest_topics.append_turn does: file the newest turn under its real ids.
    tid = mgr.upsert_topic("Munich trip")
    mgr.add_messages(tid, [{"role": "user", "excerpt": "x", "msg_id": s.messages[-2]["message_id"]},
                           {"role": "agent", "excerpt": "y", "msg_id": s.messages[-1]["message_id"]}])
    assert [i["excerpt"] for i in ct._pending_items(_agent(s))] == ["question 1", "answer 1"]


def test_an_old_store_starts_its_watermark_at_what_was_classified(tmp_path):
    import captain_claw.conversation_topics as ct

    path = tmp_path / "old.db"
    m = ct.ConversationTopicsManager(path)
    tid = m.upsert_topic("Munich trip")
    m.add_messages(tid, [{"role": "user", "excerpt": "q", "msg_id": "m-1"}])
    m._conn.execute("DELETE FROM backfill_seen")
    m._conn.execute("DELETE FROM topics_meta WHERE key = 'seen_from_classified_v1'")
    m._conn.commit()
    m._conn.close()
    m = ct.ConversationTopicsManager(path)
    try:
        assert "m-1" in m.seen_msg_ids()
    finally:
        m._conn.close()


def test_meaning_matches_need_to_clear_a_floor(mgr):
    _seed(mgr)

    def embedder(texts):
        def vec(t):
            if t == "zzz":
                return [1.0, 0.0]
            if "Munich trip" in t:
                return [0.9, 0.4359]
            if "Weekly" in t:
                return [0.2, 0.9798]
            return [0.0, 1.0]
        return [vec(t) for t in texts]

    _terms, _fts, vec = mgr.rank_legs("zzz", embedder=embedder)
    assert [r["id"] for r in vec] == ["munich-trip"]          # 0.9; the brief's 0.2 is noise
    assert mgr.search_topics("zzz", embedder=embedder)[0]["related"] is True


def test_typed_search_also_matches_inflected_words(mgr):
    _seed(mgr)
    assert [r["id"] for r in mgr.search_topics("putovanja")] == ["putovanje-u-split"]


def test_new_conversation_brings_a_swept_topic_back_but_not_a_user_hidden_one(mgr):
    import captain_claw.conversation_topics as ct

    _seed(mgr)
    mgr._conn.execute("UPDATE topics SET hidden = ? WHERE id = 'munich-trip'", (ct._HIDDEN_BY_SWEEP,))
    mgr._conn.commit()
    mgr.set_hidden("weekly-brief", True)
    mgr.upsert_topic("Munich trip", summary="more")
    mgr.upsert_topic("Weekly brief", summary="more")
    hidden = {r["id"]: r["hidden"] for r in mgr.list_topics(include_hidden=True)}
    assert hidden["munich-trip"] == 0 and hidden["weekly-brief"] == 1
    mgr.set_star("weekly-brief", True)                          # the panel's undo
    assert {r["id"]: r["hidden"] for r in mgr.list_topics(include_hidden=True)}["weekly-brief"] == 0


def test_starred_topics_do_not_take_the_recent_slots(mgr):
    for n in range(3):
        mgr.set_star(mgr.upsert_topic(f"Old starred {n}"), True)
    mgr.upsert_topic("Fresh topic")
    assert mgr.recent_topics(1)[0]["id"] == "fresh-topic"


def test_alphabetical_search_keeps_starred_first(mgr):
    _seed(mgr)
    mgr.set_star("weekly-brief", True)
    rows = mgr.search_topics("munich", order="alpha")
    assert [r["id"] for r in rows] == ["weekly-brief", "munich-trip"]


def test_refresh_keeps_the_surface_rules_out(mgr):
    import captain_claw.conversation_topics as ct

    s = Session(id="s1", name="d")
    surface = ("[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
               "Keep replies short.\nUSER MESSAGE:\n")
    s.add_message("user", surface + "plan the trip", origin="human", channel="whatsapp")
    tid = mgr.upsert_topic("Trip")
    mgr.add_messages(tid, [{"role": "user", "excerpt": "old", "msg_id": s.messages[0]["message_id"]}])
    ct.refresh_topic(_agent(s), tid)
    assert mgr.get_topic(tid)["messages"][0]["excerpt"] == "plan the trip"
