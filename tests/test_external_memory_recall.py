"""The agent reads and writes the person's external memory often:

* before every reply it asks Flight Deck what the memory holds on the message
  (automatic recall) — unless the person's reading priority is ``on_demand``,
  and never on a shared-agent member's turn or in a public per-visitor run;
* the note sits next to the insights (``alongside``) or last of all, kept
  longest and nearest the message, when the memory is the person's main one
  (``first``); what the insights note already says isn't repeated;
* while a memory is connected, facts are picked out of the chat every few
  messages instead of every 20, and right after a message that hands one over.

Flight Deck is never contacted: ``httpx.AsyncClient`` gets a mock transport.
"""

from __future__ import annotations

import json
import types

import httpx
import pytest

from captain_claw import external_memory_client as emc
from captain_claw.config import Config
from captain_claw.llm import LLMProvider, LLMResponse

FD = "http://fd.test"
_RealAsyncClient = httpx.AsyncClient

MANIFEST = {"connected": True, "provider": "qdrant", "label": "Qdrant", "actions": [], "rules": [],
            "mirror": ["insight"], "priority": "alongside", "recall_limit": 5, "recall_timeout": 4.0,
            "capture_every": 5}
HITS = [{"text": "Project Atlas ships in March.", "id": "captain-note-1", "when": "2026-11-09", "kind": "note"},
        {"text": "Ana leads Atlas.", "id": "captain-insight-7", "when": "", "kind": "insight"}]


@pytest.fixture(autouse=True)
def _config(monkeypatch, tmp_path):
    import captain_claw.config as config_mod
    from captain_claw.flight_deck import agent_secret
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.delenv("FD_URL", raising=False)
    agent_secret.reset_cache_for_tests()
    cfg = Config()
    monkeypatch.setattr(config_mod, "_config", cfg)
    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: FD))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    emc._set_manifest({})
    emc._MANIFEST_AT = 0.0
    yield cfg
    emc._set_manifest({})
    emc._MANIFEST_AT = 0.0
    agent_secret.reset_cache_for_tests()


def _connected(**over):
    emc._set_manifest({**MANIFEST, **over})


def _serve(monkeypatch, status=200, body=None):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content or b"null")})
        return httpx.Response(status, json=body if body is not None else
                              {"priority": "alongside", "label": "Qdrant", "hits": HITS})

    monkeypatch.setattr(emc.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=httpx.MockTransport(handler), **kw))
    return seen


# ── the client ──────────────────────────────────────────────────────


class TestClient:
    async def test_recall_asks_flight_deck_with_the_message(self, monkeypatch):
        _connected()
        seen = _serve(monkeypatch)
        out = await emc.recall("when does Atlas ship?")
        assert seen == [{"path": "/fd/external-memory/agent/recall", "body": {"query": "when does Atlas ship?"}}]
        assert out["hits"] == HITS and out["label"] == "Qdrant" and out["priority"] == "alongside"

    async def test_nothing_is_asked_on_demand_unconnected_or_for_others(self, monkeypatch, _config):
        from captain_claw.google_oauth_manager import GoogleOAuthManager

        seen = _serve(monkeypatch)
        assert (await emc.recall("atlas"))["hits"] == [] and emc.priority() == ""  # not connected
        _connected(priority="on_demand")
        assert (await emc.recall("atlas"))["hits"] == []
        _connected()
        assert (await emc.recall("   "))["hits"] == []
        monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: True))
        assert emc.priority() == "" and (await emc.recall("atlas"))["hits"] == []
        monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
        _config.web.public_run = "computer"
        assert (await emc.recall("atlas"))["hits"] == [] and emc.capture_every() == 0
        assert seen == []

    async def test_a_failing_flight_deck_gives_nothing_not_an_error(self, monkeypatch):
        _connected()
        _serve(monkeypatch, status=502, body={"detail": "down"})
        assert (await emc.recall("atlas"))["hits"] == []

        def boom(**kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(emc.httpx, "AsyncClient", boom)
        assert (await emc.recall("atlas"))["hits"] == []

    def test_capture_pace(self):
        assert emc.capture_every() == 0
        _connected()
        assert emc.capture_every() == 5
        _connected(capture_every="x")
        assert emc.capture_every() == 0

    @pytest.mark.parametrize("said,cue", [
        ("Remember that Ana prefers Tuesdays", True),
        ("my daughter starts school in October", True),
        ("Zapamti: Marta radi PDV prijave", True),
        ("moja supruga je alergična na orašaste plodove", True),
        ("we decided to move the launch to May", True),
        ("Odlučili smo prebaciti sastanak na petak", True),
        ("write me a poem about the sea", False),
        ("what's the weather tomorrow?", False),
        ("/new", False),
    ])
    def test_remember_cue(self, said, cue):
        assert emc.remember_cue(said) is cue


# ── the note in the agent's context ─────────────────────────────────


class _StubProvider(LLMProvider):
    provider = "openai"
    model = "stub-model"

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len([part for part in text.split() if part]) or 1


def _agent():
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    agent = Agent(provider=_StubProvider())
    agent.session = Session(id="s1", name="default")
    agent._build_env_now_text = lambda: ""
    return agent


def _kinds(agent, **kw):
    notes = agent._collect_background_context_notes([], query="when does atlas ship", skip_memory=kw.get(
        "skip_memory", False), owner_notes=kw.get("owner_notes", True))
    return [k for k, _ in notes], dict(notes)


class TestNote:
    def test_alongside_sits_next_to_the_insights(self):
        agent = _agent()
        agent._insights_context_cache = [{"content": "Bob likes short answers.", "category": "preference"}]
        agent._build_todo_context_note = lambda: "Pending todos: water the plants"
        agent._external_memory_context_cache = {"priority": "alongside", "label": "Qdrant", "hits": HITS}
        kinds, notes = _kinds(agent)
        assert kinds.index("external_memory_context") == kinds.index("insights_context") + 1
        note = notes["external_memory_context"]
        assert note.startswith("From the person's external memory (Qdrant, shared by all of their agents)")
        assert "- Project Atlas ships in March. (2026-11-09)" in note and "- Ana leads Atlas." in note

    def test_first_is_last_kept_longest_and_trusted(self):
        agent = _agent()
        agent._insights_context_cache = [{"content": "Bob likes short answers.", "category": "preference"}]
        agent._external_memory_context_cache = {"priority": "first", "label": "Qdrant", "hits": HITS}
        kinds, notes = _kinds(agent)
        assert kinds[-1] == "external_memory_context"
        assert "MAIN memory" in notes["external_memory_context"]
        assert "go by it" in notes["external_memory_context"]

    def test_first_survives_a_crowded_prompt(self):
        agent = _agent()
        agent._external_memory_context_cache = {"priority": "first", "label": "Qdrant", "hits": HITS}
        notes = agent._collect_background_context_notes([], query="atlas", skip_memory=False, owner_notes=True)
        notes = [("filler", "word " * 400)] + notes
        size = agent._count_tokens(agent._wrap_internal_context(notes[-1:], "x")) + 5
        kept = agent._fit_context_notes(notes, size, "x")
        assert [k for k, _ in kept] == ["external_memory_context"]

    def test_what_the_insights_note_says_is_not_repeated(self):
        agent = _agent()
        agent._insights_context_cache = [{"content": "Ana leads Atlas.", "category": "fact"}]
        agent._external_memory_context_cache = {"priority": "alongside", "label": "Qdrant", "hits": HITS}
        note = _kinds(agent)[1]["external_memory_context"]
        assert "Project Atlas ships in March." in note and "Ana leads Atlas." not in note
        agent._insights_context_cache = [{"content": h["text"]} for h in HITS]
        assert "external_memory_context" not in _kinds(agent)[0]  # nothing new: no note

    def test_no_note_when_memory_is_skipped_or_not_the_owners(self):
        agent = _agent()
        agent._external_memory_context_cache = {"priority": "first", "label": "Qdrant", "hits": HITS}
        assert "external_memory_context" not in _kinds(agent, skip_memory=True)[0]
        assert "external_memory_context" not in _kinds(agent, owner_notes=False)[0]
        agent._suppress_memory_context = True
        assert "external_memory_context" not in _kinds(agent)[0]

    async def test_refresh(self, monkeypatch):
        agent = _agent()
        seen = _serve(monkeypatch)
        await agent._refresh_external_memory_context_cache(query="atlas")  # not connected
        assert agent._external_memory_context_cache == {} and seen == []
        _connected()
        await agent._refresh_external_memory_context_cache(query="atlas")
        assert agent._external_memory_context_cache["hits"] == HITS
        agent._speaker_scoped = True  # a shared-agent member's turn
        await agent._refresh_external_memory_context_cache(query="atlas")
        assert agent._external_memory_context_cache == {} and len(seen) == 1
        agent._speaker_scoped = False
        _connected(priority="on_demand")
        await agent._refresh_external_memory_context_cache(query="atlas")
        assert agent._external_memory_context_cache == {} and len(seen) == 1

    def test_it_reaches_the_model(self):
        agent = _agent()
        agent._external_memory_context_cache = {"priority": "first", "label": "Qdrant", "hits": HITS}
        turn_start = len(agent.session.messages)
        agent.session.add_message("user", "when does atlas ship")
        messages = agent._build_messages(tool_messages_from_index=turn_start, query="when does atlas ship")
        last = messages[-1].content
        assert "Project Atlas ships in March." in last
        assert last.index("Project Atlas ships") < last.index("[END INTERNAL CONTEXT]") < last.index(
            "when does atlas ship")


# ── writing: facts are picked out of the chat more often ────────────


def _chatting(n_user: int, last: str = "ok"):
    messages = []
    for i in range(n_user):
        messages += [{"role": "user", "content": last if i == n_user - 1 else f"message {i}"},
                     {"role": "assistant", "content": "sure"}]
    return types.SimpleNamespace(session=types.SimpleNamespace(messages=messages))


class TestCapture:
    @pytest.fixture()
    def extracted(self, monkeypatch):
        import captain_claw.insights as ins

        calls = []

        async def fake_extract(agent, *, trigger, tool_context=None):
            calls.append(trigger)
            setattr(agent, ins._ATTR_RUNNING, False)
            return []

        monkeypatch.setattr(ins, "extract_insights", fake_extract)
        return calls

    async def test_every_few_messages_while_connected(self, extracted):
        from captain_claw.insights import maybe_extract_insights

        await maybe_extract_insights(_chatting(5), trigger="periodic")
        assert extracted == []  # Captain's own pace: every 20
        _connected()
        await maybe_extract_insights(_chatting(4), trigger="periodic")
        assert extracted == []
        await maybe_extract_insights(_chatting(5), trigger="periodic")
        assert extracted == ["periodic"]

    async def test_right_after_a_message_that_hands_a_fact_over(self, extracted):
        from captain_claw.insights import maybe_extract_insights

        _connected()
        await maybe_extract_insights(_chatting(2, "Remember that Marta handles the VAT filings."), trigger="periodic")
        assert extracted == ["periodic"]
        await maybe_extract_insights(_chatting(2, "thanks, that's all"), trigger="periodic")
        assert extracted == ["periodic"]
        # Nothing new since the last extraction: the cue alone doesn't run it again.
        agent = _chatting(2, "Remember that Marta handles the VAT filings.")
        import captain_claw.insights as ins
        setattr(agent, ins._ATTR_LAST_MSG_IDX, 2)
        await maybe_extract_insights(agent, trigger="periodic")
        assert extracted == ["periodic"]
