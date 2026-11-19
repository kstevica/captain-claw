"""summarize_files keeps the agent reachable and finishes its summaries; the
side fixes from the same cohort001 read (2026-11-19).

- Parsing a few hundred PDF pages ran on the agent's event loop for ~20 s,
  and the chat link's keepalive timed out three times in five minutes.
- Chunks were 400k characters whatever the window — ~159k tokens of a
  Croatian gazette, the whole window of a 160k model — and a thinking model
  spent its 4096 output tokens thinking: 9 of 14 summaries stopped at the
  limit and the agent called the result unreliable.
- An insights search with a category failed whenever full-text search found
  nothing ("no such column: i.category").
- A failed LLM call wrote no usage row, so every row read error=0.
"""

from __future__ import annotations

import sqlite3
import threading
import types
from pathlib import Path

import pytest

from captain_claw.llm import LLMProvider, Message, reasoning
from captain_claw.tools import summarize_files as sf
from captain_claw.tools.summarize_files import SummarizeFilesTool

# ── summarize_files ───────────────────────────────────────────────────


class _Provider:
    """Records each call; answers with a short summary."""

    provider = "openai"
    model = "deepseek-flash"

    def __init__(self, *, finish_reason: str = "stop", fail: bool = False,
                 chars_per_token: int = 3) -> None:
        self.calls: list[dict] = []
        self.finish_reason = finish_reason
        self.fail = fail
        self.chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        return len(text) // self.chars_per_token

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append({"max_tokens": max_tokens, "call": reasoning._CALL.get(),
                           "chars": len(messages[-1].content)})
        if self.fail:
            raise TimeoutError("provider timed out")
        return types.SimpleNamespace(content="A summary.", usage={}, model=self.model,
                                     finish_reason=self.finish_reason)


@pytest.fixture
def recorded(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(SummarizeFilesTool, "_record_single_usage",
                        staticmethod(lambda **kw: rows.append(kw)))
    return rows


@pytest.fixture
def window(monkeypatch):
    from captain_claw.config import get_config, set_config

    old = get_config()
    cfg = old.model_copy(deep=True)
    cfg.context.max_tokens = 160_000
    set_config(cfg)
    yield cfg
    set_config(old)


async def test_files_are_read_off_the_event_loop(tmp_path, monkeypatch, recorded):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("Natječaj za radno mjesto.\n" * 20)
    provider = _Provider()
    monkeypatch.setattr("captain_claw.llm.get_provider", lambda: provider)
    monkeypatch.setattr(SummarizeFilesTool, "_load_personality_context", lambda self: "")
    threads: list[int] = []
    real_read = SummarizeFilesTool._read_file_content

    def spy(path):
        threads.append(threading.get_ident())
        return real_read(path)

    monkeypatch.setattr(SummarizeFilesTool, "_read_file_content", staticmethod(spy))
    loop_thread = threading.get_ident()
    result = await SummarizeFilesTool().execute(folder=str(folder), _saved_base_path=str(tmp_path))
    assert result.success, result.error
    assert threads and all(t != loop_thread for t in threads)


def test_a_chunk_is_a_share_of_the_window_in_the_texts_own_density(window):
    # 160k window × 0.35 = 56k tokens; 3 chars a token.
    assert SummarizeFilesTool._chunk_chars(_Provider(chars_per_token=3), "x" * 600_000) == pytest.approx(
        168_000, rel=1e-3)
    # Never past the old ceiling, never below the floor.
    assert SummarizeFilesTool._chunk_chars(_Provider(chars_per_token=20), "x" * 600_000) == sf._MAX_INPUT_CHARS
    window.context.max_tokens = 8_000
    assert SummarizeFilesTool._chunk_chars(_Provider(), "x" * 600_000) == sf._MIN_CHUNK_CHARS


async def test_a_long_file_is_summarised_in_window_sized_chunks(window, recorded):
    provider = _Provider()
    text = "\n\n".join(f"Članak {i}. " + "riječ " * 400 for i in range(250))  # ~600k chars
    chunk = SummarizeFilesTool._chunk_chars(provider, text)
    summary, calls = await SummarizeFilesTool()._summarise_single_file(
        provider, Message, Path("NN-130.pdf"), text, "concise", "", "", {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, chunk_chars=chunk)
    assert calls >= 4                                  # 4+ parts and the combine
    parts = provider.calls[:-1]
    assert all(c["chars"] <= chunk + 500 for c in parts)
    assert summary == "A summary."


async def test_summaries_ask_for_thinking_off(recorded):
    provider = _Provider()
    await SummarizeFilesTool._llm_complete(provider, Message, "sys", "user")
    assert provider.calls[0]["call"] == reasoning.QUIET


async def test_a_summary_cut_at_the_limit_says_it_is_incomplete(recorded):
    text, _ = await SummarizeFilesTool._llm_complete(_Provider(finish_reason="length"), Message,
                                                    "sys", "user")
    assert text.startswith("A summary.") and "cut at the output limit" in text


async def test_the_final_combine_has_room_for_the_words_asked_for(recorded):
    provider = _Provider()
    await SummarizeFilesTool()._combine_summaries(provider, Message, [("a", "x"), ("b", "y")],
                                                  3000, "", "", {"prompt_tokens": 0,
                                                                 "completion_tokens": 0,
                                                                 "total_tokens": 0})
    assert provider.calls[-1]["max_tokens"] == 9000


async def test_a_failed_summary_call_is_recorded(recorded):
    with pytest.raises(TimeoutError):
        await SummarizeFilesTool._llm_complete(_Provider(fail=True), Message, "sys", "user")
    assert recorded and isinstance(recorded[-1]["error"], TimeoutError)
    assert recorded[-1]["response"] is None


# ── insights search ───────────────────────────────────────────────────


async def test_a_category_search_that_full_text_misses_falls_back(tmp_path):
    from captain_claw.insights import InsightsManager

    mgr = InsightsManager(tmp_path / "insights.db")
    try:
        assert await mgr.add(content="Munich hotel: Adler near the main station", category="fact")
        assert await mgr.add(content="Answer in Croatian", category="preference")
        # "dler" is inside a word: no full-text prefix match, a LIKE match.
        found = await mgr.search("dler", category="fact")
        assert [r["content"] for r in found] == ["Munich hotel: Adler near the main station"]
        assert await mgr.search("dler", category="preference") == []
    finally:
        await mgr.close()


# ── failed LLM calls are counted ──────────────────────────────────────


class _FailingProvider(LLMProvider):
    provider = "openai"
    model = "deepseek-flash"

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        raise ConnectionError("upstream reset")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    async def complete_with_callback(self, messages, tools=None, max_tokens=None, on_chunk=None, **kw):
        raise ConnectionError("upstream reset")

    def count_tokens(self, text: str) -> int:
        return len(text)


async def test_a_failed_guarded_call_writes_an_error_row(monkeypatch):
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    agent = Agent(provider=_FailingProvider())
    agent._initialized = True
    agent.session = Session(id="s1", name="usage")
    rows: list[dict] = []
    monkeypatch.setattr(agent, "_record_usage_to_db", lambda **kw: rows.append(kw))
    with pytest.raises(ConnectionError):
        await agent._complete_with_guards([Message(role="user", content="hi")],
                                          interaction_label="turn_1")
    assert len(rows) == 1
    assert rows[0]["error"] is True and rows[0]["response"] is None
    assert rows[0]["finish_reason"] == "error:ConnectionError"


async def test_the_spend_ledger_does_not_count_a_failed_call(tmp_path):
    from captain_claw.flight_deck.usage_spend import _read_calls
    from captain_claw.session import SessionManager

    sm = SessionManager(tmp_path / "sessions.db")
    await sm.record_llm_usage(provider="openai", model="deepseek-flash", prompt_tokens=1000,
                              completion_tokens=100, total_tokens=1100)
    await sm.record_llm_usage(provider="openai", model="deepseek-flash", error=True,
                              finish_reason="error:TimeoutError")
    await sm.close()
    calls, newest = _read_calls(tmp_path / "sessions.db", 0)
    assert newest == 2
    assert sum(c["calls"] for c in calls) == 1
    with sqlite3.connect(tmp_path / "sessions.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM llm_usage WHERE error = 1").fetchone()[0] == 1
