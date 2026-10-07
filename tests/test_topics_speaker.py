"""A2: topic excerpts are per speaker (contract a2 part 2b §5).

``topic_messages`` gains a ``speaker`` column (owner rows ``''``; a guarded
ALTER for pre-A2 databases). A shared-agent member instance stamps its
member's id on what it feeds the classifier, a member's ``topics get`` shows
only their OWN excerpts (with their own count), an unverified member none,
and pruning is per speaker so one member can't evict anyone else's excerpts.
Labels, summaries and keywords stay the open commons.

Every test uses a tmp DB and a tmp HOME (nothing here may reach ~/.captain-claw).
"""

from __future__ import annotations

import sqlite3
import types

import pytest

from captain_claw import speaker
from captain_claw.config import get_config
from captain_claw.speaker import UNKNOWN_PRINCIPAL, Principal

PRINCIPAL = Principal("u-member", "Ana", "Olga", "A", "process:helper:0123456789abcdef")
OTHER = Principal("u-other", "Bo", "Olga", "B", "process:helper:0123456789abcdef")


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
    from captain_claw import session as _session

    monkeypatch.setattr(_session, "_manager", _session.SessionManager(home / ".captain-claw" / "s.db"))
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


class _Bound:
    def __init__(self, p):
        self.p = p

    def __enter__(self):
        self._t = speaker.bind(self.p)

    def __exit__(self, *exc):
        speaker.reset(self._t)


def _speaker_agent(p=PRINCIPAL):
    return types.SimpleNamespace(_speaker_scoped=True, _speaker_principal=p)


def _columns(path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(topic_messages)").fetchall()}
    finally:
        conn.close()


def _indexes(path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[1] for r in conn.execute("PRAGMA index_list(topic_messages)").fetchall()}
    finally:
        conn.close()


def test_guarded_alter_on_a_pre_a2_db_is_idempotent(tmp_path):
    import captain_claw.conversation_topics as ct

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE topics (id TEXT PRIMARY KEY, label TEXT NOT NULL, summary TEXT NOT NULL "
        "DEFAULT '', keywords TEXT NOT NULL DEFAULT '', msg_count INTEGER NOT NULL DEFAULT 0, "
        "first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);"
        "CREATE TABLE topic_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_id TEXT NOT NULL, "
        "role TEXT NOT NULL DEFAULT '', channel TEXT NOT NULL DEFAULT '', excerpt TEXT NOT NULL "
        "DEFAULT '', msg_id TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL);"
        "INSERT INTO topics (id, label, first_seen, last_seen) VALUES ('t', 'T', 'x', 'x');"
        "INSERT INTO topic_messages (topic_id, role, excerpt, ts) VALUES ('t', 'user', 'old row', 'x');"
    )
    conn.commit()
    conn.close()
    assert "speaker" not in _columns(path)

    for _ in range(2):                      # opening twice is harmless
        m = ct.ConversationTopicsManager(path)
        m._conn.close()
    assert {"speaker", "msg_id"} <= _columns(path)
    assert "idx_tm_speaker" in _indexes(path)
    m = ct.ConversationTopicsManager(path)
    try:
        t = m.get_topic("t", speaker="")
        assert [x["excerpt"] for x in t["messages"]] == ["old row"]    # the owner's
        assert m.get_topic("t", speaker="u-member")["messages"] == []
    finally:
        m._conn.close()


def test_a_fresh_db_has_the_column_and_index(tmp_path):
    import captain_claw.conversation_topics as ct

    m = ct.ConversationTopicsManager(tmp_path / "new.db")
    m._conn.close()
    assert "speaker" in _columns(tmp_path / "new.db")
    assert "idx_tm_speaker" in _indexes(tmp_path / "new.db")


def _session_agent(messages, *, member: Principal | None = None):
    agent = types.SimpleNamespace(session=types.SimpleNamespace(messages=messages))
    if member is not None:
        agent._speaker_scoped = True
        agent._speaker_principal = member
    return agent


def _msgs():
    return [
        {"role": "user", "content": "plan the Munich trip", "message_id": "m1"},
        {"role": "assistant", "content": "Sure — dates?", "message_id": "m2"},
    ]


def test_collected_messages_carry_the_speaker(mgr, monkeypatch):
    from captain_claw.conversation_topics import _collect_new_messages

    monkeypatch.setattr(get_config().conversation_topics, "include_narration", True)
    member = _session_agent(_msgs(), member=PRINCIPAL)
    member._topics_narration_buffer = ["searched the web"]
    items, _ = _collect_new_messages(member, 0, 50)
    assert items and {i["speaker"] for i in items} == {"u-member"}
    assert any(i["role"] == "narration" for i in items)

    owner = _session_agent(_msgs())
    items, _ = _collect_new_messages(owner, 0, 50)
    assert items and {i["speaker"] for i in items} == {""}

    # A post-turn job carries the principal in its context.
    with _Bound(PRINCIPAL):
        items, _ = _collect_new_messages(_session_agent(_msgs()), 0, 50)
    assert {i["speaker"] for i in items} == {"u-member"}


def _seed(mgr):
    tid = mgr.upsert_topic("Munich trip", summary="Planning the Munich trip", keywords=["travel"])
    mgr.add_messages(tid, [{"role": "user", "excerpt": f"OWNER EXCERPT {i}", "speaker": ""}
                           for i in range(3)])
    mgr.add_messages(tid, [{"role": "user", "excerpt": f"ANA EXCERPT {i}", "speaker": "u-member"}
                           for i in range(2)])
    mgr.add_messages(tid, [{"role": "user", "excerpt": "BO EXCERPT", "speaker": "u-other"}])
    return tid


async def test_member_get_shows_only_their_own_excerpts(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    tid = _seed(mgr)
    res = await TopicsTool().execute(action="get", topic=tid, _agent=_speaker_agent())
    assert res.success
    assert "ANA EXCERPT 0" in res.content and "ANA EXCERPT 1" in res.content
    assert "OWNER EXCERPT" not in res.content and "BO EXCERPT" not in res.content
    assert "Your messages in this topic (2)" in res.content
    assert "6 total" not in res.content and "Planning the Munich trip" in res.content

    with _Bound(OTHER):
        res = await TopicsTool().execute(action="get", topic=tid)
    assert "BO EXCERPT" in res.content and "ANA" not in res.content and "OWNER" not in res.content
    assert "Your messages in this topic (1)" in res.content


async def test_unknown_member_sees_no_excerpts_and_no_count(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    tid = _seed(mgr)
    with _Bound(UNKNOWN_PRINCIPAL):
        res = await TopicsTool().execute(action="get", topic=tid)
    assert res.success and "Munich trip" in res.content
    assert "EXCERPT" not in res.content and "6" not in res.content


async def test_owner_sees_every_excerpt(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    tid = _seed(mgr)
    res = await TopicsTool().execute(action="get", topic=tid)
    # PR D (J19): without an owner instance and Flight Deck's roster nobody is
    # a CURRENT member, so members' excerpts are dropped (fail closed); the
    # owner's own always show. Current members' excerpts with a roster:
    # tests/test_member_privacy.py::test_owner_topics_get.
    assert "OWNER EXCERPT 2" in res.content
    for text in ("ANA EXCERPT 1", "BO EXCERPT"):
        assert text not in res.content
    assert "6 total" in res.content


def test_member_messages_never_evict_anyone_elses(mgr):
    tid = mgr.upsert_topic("Busy topic")
    mgr.add_messages(tid, [{"role": "user", "excerpt": f"OWNER {i}"} for i in range(5)], cap=40)
    for i in range(50):
        mgr.add_messages(tid, [{"role": "user", "excerpt": f"ANA {i}", "speaker": "u-member"}],
                         cap=40)
    owner_rows = mgr.get_topic(tid, max_excerpts=200, speaker="")["messages"]
    assert [m["excerpt"] for m in owner_rows] == [f"OWNER {i}" for i in range(5)]
    member_rows = mgr.get_topic(tid, max_excerpts=200, speaker="u-member")["messages"]
    assert len(member_rows) == 40 and member_rows[-1]["excerpt"] == "ANA 49"
    assert mgr.get_topic(tid)["msg_count"] == 55              # the topic total


async def test_member_overview_has_no_message_counts(mgr):
    from captain_claw.tools.conversation_topics import TopicsTool

    _seed(mgr)
    for action, extra in (("list", {}), ("search", {"query": "Munich"})):
        member = await TopicsTool().execute(action=action, _agent=_speaker_agent(), **extra)
        owner = await TopicsTool().execute(action=action, **extra)
        assert "Munich trip" in member.content
        assert "msgs" not in member.content
        assert "6 msgs" in owner.content
