"""A tool result is never too big to be seen, and an unseen result is not a "duplicate".

From cohort001 (2026-11-19): with the governor off, a web_fetch of a long
official-gazette PDF went into the session whole, bigger than the room left in
the window. The packer left it out unseen; on the next call the held cut no
longer fit, so it kept only the newest step — every earlier step of the turn
went too (dropped 3, then 11). The model asked for the PDF again, the duplicate
guard refused ("the content has not changed"), and the all-blocked note told it
it already had the data. The same guard refused an ``xlsx check`` / ``pptx
check`` after the file was edited, and the agent said "done" unchecked.
"""

from __future__ import annotations

import os
import types

import pytest

from captain_claw import context_spill
from captain_claw.agent_context_governor_mixin import AgentContextGovernorMixin
from captain_claw.agent_tool_loop_mixin import duplicate_signature
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

# ── the packer without the governor ───────────────────────────────────


class _Packer(AgentContextGovernorMixin):
    def __init__(self):
        self._turn_window = None
        self.forgotten: list[dict] = []

    def _forget_dropped_call_guards(self, messages):
        self.forgotten.extend(messages)
        return []


def _history(n: int, size: int = 10):
    msgs, sidx = [], []
    for i in range(n):
        role = ("user", "assistant", "tool")[i % 3]
        msgs.append({"role": role, "content": f"m{i}", "_t": size})
        sidx.append(i)
    return msgs, sidx


def _with_big_step(n: int = 12, big: int = 500):
    """Small history, then a step whose tool result is bigger than the budget."""
    msgs, sidx = _history(n)
    msgs.append({"role": "assistant", "content": "", "_t": 10,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "web_fetch",
                                              "arguments": '{"url": "https://nn.example/pdf"}'}}]})
    sidx.append(n)
    msgs.append({"role": "tool", "content": "huge", "_t": big, "tool_call_id": "c1"})
    sidx.append(n + 1)
    return msgs, sidx


def _fill(packer, msgs, sidx, budget):
    selected: set[int] = set()
    used, dropped = packer._plain_fill(msgs, sidx, selected, 0, budget, None,
                                       lambda p: msgs[p]["_t"])
    return sorted(selected), used, dropped


def _append_step(msgs, sidx, size=10):
    n = sidx[-1] + 1
    msgs.append({"role": "assistant", "content": "next", "_t": size})
    sidx.append(n)
    msgs.append({"role": "tool", "content": "ok", "_t": size})
    sidx.append(n + 1)


def test_a_step_bigger_than_the_room_is_left_out_and_the_rest_kept():
    packer = _Packer()
    msgs, sidx = _with_big_step()
    kept, used, _ = _fill(packer, msgs, sidx, 200)
    assert 12 not in kept and 13 not in kept            # the oversized step
    assert len(kept) > 2 and used <= 200                 # earlier steps stay
    assert kept == list(range(kept[0], 12))


def test_the_call_after_keeps_the_held_cut_instead_of_wiping_the_turn():
    packer = _Packer()
    msgs, sidx = _with_big_step()
    first, _, dropped_first = _fill(packer, msgs, sidx, 200)
    _append_step(msgs, sidx)                             # the model's next step
    kept, used, dropped = _fill(packer, msgs, sidx, 200)
    # It used to re-cut to the newest step alone (dropped 3 → 11).
    assert kept[0] == first[0]
    assert 14 in kept and 15 in kept
    assert 12 not in kept and 13 not in kept
    assert used <= 200
    assert dropped == dropped_first


def test_the_left_out_calls_are_handed_to_the_duplicate_guard():
    packer = _Packer()
    msgs, sidx = _with_big_step()
    _fill(packer, msgs, sidx, 200)
    names = [tc["function"]["name"] for m in packer.forgotten for tc in (m.get("tool_calls") or [])]
    assert "web_fetch" in names


def test_an_oversized_older_step_does_not_end_the_cut():
    packer = _Packer()
    msgs, sidx = _with_big_step(n=12, big=500)
    for _ in range(3):
        _append_step(msgs, sidx)
    kept, used, _ = _fill(packer, msgs, sidx, 200)
    assert 12 not in kept and 13 not in kept
    # Steps before the big one are still there, not just the ones after it.
    assert any(p < 12 for p in kept)
    assert used <= 200


# ── one result keeps at most its share of the window ──────────────────


def _spill_agent(tmp_path):
    return types.SimpleNamespace(workspace_base_path=str(tmp_path),
                                 session=types.SimpleNamespace(id="s1"))


def test_a_result_that_fits_is_unchanged(tmp_path):
    text = "short page"
    out = context_spill.cap_tool_result(_spill_agent(tmp_path), "web_fetch", {"url": "u"}, text,
                                        cap_tokens=1000, count_tokens=len)
    assert out == text


def test_a_big_result_keeps_its_head_and_a_pointer_to_the_whole_text(tmp_path):
    text = "Č" * 50_000 + "THE END"
    out = context_spill.cap_tool_result(_spill_agent(tmp_path), "web_fetch",
                                        {"url": "https://nn.example/pdf"}, text,
                                        cap_tokens=10_000, count_tokens=len)
    assert out.startswith("[web_fetch https://nn.example/pdf · 50007 chars · Full text saved to ")
    assert len(out) < 12_000
    pointer = out.split(context_spill.POINTER_MARK, 1)[1].split(" ", 1)[0]
    saved = tmp_path / pointer
    assert saved.is_file() and saved.read_text().endswith("THE END")


def test_a_dense_text_is_measured_the_way_the_packer_measures_it(tmp_path):
    text = "x" * 40_000
    # Two tokens per character: 80k tokens, far past a 10k cap.
    out = context_spill.cap_tool_result(_spill_agent(tmp_path), "pdf_extract", {"path": "a.pdf"},
                                        text, cap_tokens=10_000, count_tokens=lambda t: 2 * len(t))
    assert 2 * len(out) < 12_000


def test_a_read_points_at_its_own_file(tmp_path):
    out = context_spill.cap_tool_result(_spill_agent(tmp_path), "read", {"path": "notes/long.md"},
                                        "y" * 30_000, cap_tokens=5_000, count_tokens=len)
    assert f"{context_spill.POINTER_MARK}notes/long.md" in out
    assert not (tmp_path / ".context").exists()


def test_parallel_calls_share_half_the_window():
    one = context_spill.plain_cap_tokens(None, 1)
    four = context_spill.plain_cap_tokens(None, 4)
    assert four < one
    assert 4 * four <= one * 2 + 1


# ── in the tool loop ──────────────────────────────────────────────────


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


class _NoSave:
    async def save_session(self, session) -> None:
        return None


class _Fetch(Tool):
    name = "web_fetch"
    description = "fetch"
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, url: str = "", **kwargs) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, content="Članak " * 60_000)


class _Office(Tool):
    """An xlsx-like tool: ``check`` reads the file, ``edit`` changes it."""

    name = "xlsx"
    description = "spreadsheet"
    parameters = {"type": "object", "properties": {"action": {"type": "string"},
                                                   "path": {"type": "string"}}}

    def __init__(self, base) -> None:
        self.base = base
        self.checks = 0

    async def execute(self, action: str = "", path: str = "", **kwargs) -> ToolResult:
        target = self.base / path if (self.base / path).is_file() else self.base / "saved" / path
        if action == "edit":
            target.write_text(target.read_text() + "+row\n")
            return ToolResult(success=True, content="edited")
        self.checks += 1
        return ToolResult(success=True, content=f"{target.read_text().count('row')} rows")


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from captain_claw.agent import Agent
    from captain_claw.config import get_config, set_config
    from captain_claw.session import Session

    monkeypatch.setenv("CLAW_CTX_GOVERNOR", "off")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    old_cfg = get_config()
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.duplicate_call_max = 1
    cfg.tools.jev.review_blocks = False
    cfg.context.max_tokens = 160_000
    set_config(cfg)

    a = Agent(provider=_Provider())
    a._initialized = True
    a.session = Session(id="s1", name="dropped-results")
    a.session_manager = _NoSave()
    a.workspace_base_path = str(tmp_path)
    registry = ToolRegistry(base_path=tmp_path)
    a.fetch = _Fetch()
    a.office = _Office(tmp_path)
    registry.register(a.fetch)
    registry.register(a.office)
    a.tools = registry
    a._turn_tool_call_counts = {}
    a._turn_tool_call_fps = {}
    a._blind_write_paths = set()
    a._malformed_retry_count = 0
    yield a
    set_config(old_cfg)


def _blocked(results) -> list[str]:
    return [str(r.get("content", "")) for r in results
            if str(r.get("content", "")).startswith("DUPLICATE CALL BLOCKED")]


async def _call(agent, name, **args):
    return await agent._handle_tool_calls([ToolCall(id=f"{name}-{os.urandom(3).hex()}", name=name,
                                                    arguments=args)])


async def test_a_huge_fetch_reaches_the_model_as_a_preview_and_a_pointer(agent):
    await _call(agent, "web_fetch", url="https://nn.example/pdf")
    kept = [m for m in agent.session.messages if m.get("role") == "tool"][-1]["content"]
    assert context_spill.POINTER_MARK in kept.partition("\n")[0]
    # A quarter of the 160k window, in this provider's tokens (1 per char).
    assert len(kept) <= context_spill.plain_cap_tokens(agent, 1) + 2_000


async def test_a_fetch_the_packer_dropped_may_run_again(agent):
    await _call(agent, "web_fetch", url="https://nn.example/pdf")
    assert _blocked(await _call(agent, "web_fetch", url="https://nn.example/pdf"))
    dropped = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "web_fetch", "arguments": '{"url": "https://nn.example/pdf"}'}}]}
    assert agent._forget_dropped_call_guards([dropped]) == [
        duplicate_signature("web_fetch", {"url": "https://nn.example/pdf"})]
    r = await _call(agent, "web_fetch", url="https://nn.example/pdf")
    assert not _blocked(r)
    assert agent.fetch.calls == 2


async def test_a_check_after_the_file_was_edited_is_new(agent, tmp_path):
    (tmp_path / "out.xlsx").write_text("row\n")
    await _call(agent, "xlsx", action="check", path="out.xlsx")
    await _call(agent, "xlsx", action="edit", path="out.xlsx")
    r = await _call(agent, "xlsx", action="check", path="out.xlsx")
    assert not _blocked(r)
    assert agent.office.checks == 2


async def test_a_check_with_nothing_changed_is_still_blocked(agent, tmp_path):
    (tmp_path / "out.xlsx").write_text("row\n")
    await _call(agent, "xlsx", action="check", path="out.xlsx")
    r = await _call(agent, "xlsx", action="check", path="out.xlsx")
    assert _blocked(r)
    assert "The file has not changed since" in _blocked(r)[0]
    assert agent.office.checks == 1


async def test_a_path_under_saved_without_the_prefix_is_followed(agent, tmp_path):
    (tmp_path / "saved" / "showcase").mkdir(parents=True)
    (tmp_path / "saved" / "showcase" / "deck.xlsx").write_text("row\n")
    await _call(agent, "xlsx", action="check", path="showcase/deck.xlsx")
    await _call(agent, "xlsx", action="edit", path="showcase/deck.xlsx")
    r = await _call(agent, "xlsx", action="check", path="showcase/deck.xlsx")
    assert not _blocked(r)
    assert agent.office.checks == 2
