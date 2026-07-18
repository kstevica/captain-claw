"""Mrav micro runtime — ledger, protocol, toolpack, blackboard, loop."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw.mrav.digest import describe_result, digest_text, split_by_tokens
from captain_claw.mrav.ledger import (
    LedgerOverflowError,
    PromptLedger,
    Section,
    estimate_tokens,
    truncate_tokens,
)
from captain_claw.mrav.protocol import (
    parse_json_object,
    parse_plan,
    strip_thinking,
    validate_action,
)
from captain_claw.mrav.runtime import MravRuntime
from captain_claw.mrav.state import Blackboard
from captain_claw.mrav.toolpack import build_toolpack, compact_definition

# ── ledger ───────────────────────────────────────────────────────────


def test_estimate_tokens_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    # chars/3.6 estimates HIGHER than the usual chars/4 heuristic
    text = "x" * 4000
    assert estimate_tokens(text) >= 1000


def test_truncate_tokens_modes():
    text = "HEAD " + ("middle " * 500) + " TAIL"
    for keep, must_have in (("head", "HEAD"), ("tail", "TAIL")):
        out = truncate_tokens(text, 50, keep=keep)
        assert estimate_tokens(out) <= 50 + 5
        assert must_have in out
    out = truncate_tokens(text, 50, keep="split")
    assert "HEAD" in out and "TAIL" in out
    assert truncate_tokens("short", 100) == "short"


def test_ledger_fit_respects_budgets_and_cap():
    ledger = PromptLedger(input_cap=1000, reserve=100)
    sections = [
        Section("a", "x" * 4000, budget=200, keep="head"),
        Section("b", "y" * 4000, budget=300, keep="tail", flex=True),
        Section("c", "z" * 400, budget=300),
    ]
    fitted, report = ledger.fit(sections)
    assert estimate_tokens(fitted["a"]) <= 200
    assert report.total_tokens <= ledger.usable


def test_ledger_squeezes_flex_sections():
    ledger = PromptLedger(input_cap=500, reserve=50)
    sections = [
        Section("frozen", "f" * 900, budget=250, keep="head"),
        Section("flex1", "a" * 2000, budget=250, keep="tail", flex=True),
        Section("flex2", "b" * 2000, budget=250, keep="tail", flex=True),
    ]
    fitted, report = ledger.fit(sections)
    total = sum(estimate_tokens(t) for t in fitted.values())
    assert total <= ledger.usable
    assert report.squeezed  # something had to give


def test_ledger_overflow_when_frozen_exceeds_cap():
    ledger = PromptLedger(input_cap=300, reserve=50)
    sections = [Section("frozen", "f" * 5000, budget=5000, keep="head")]
    with pytest.raises(LedgerOverflowError):
        ledger.fit(sections)


def test_ledger_property_never_exceeds_cap():
    rng = random.Random(42)
    for _ in range(50):
        cap = rng.randint(600, 9000)
        ledger = PromptLedger(input_cap=cap, reserve=cap // 16)
        sections = []
        for i in range(rng.randint(2, 8)):
            budget = rng.randint(32, max(64, cap // 3))
            text = "w" * rng.randint(0, cap * 4)
            sections.append(
                Section(f"s{i}", text, budget=budget, keep=rng.choice(["head", "tail"]), flex=(i > 0))
            )
        try:
            fitted, report = ledger.fit(sections)
        except LedgerOverflowError:
            continue  # legal outcome for adversarial frozen budgets
        assert sum(estimate_tokens(t) for t in fitted.values()) <= ledger.usable


# ── protocol ─────────────────────────────────────────────────────────


def test_strip_thinking_closed_and_unclosed():
    assert strip_thinking("<think>secret</think>{\"a\":1}") == '{"a":1}'
    assert strip_thinking('{"a":1}\n<think>trailing never closed') == '{"a":1}'


@pytest.mark.parametrize(
    "raw",
    [
        '{"action":"final","text":"done"}',
        '```json\n{"action":"final","text":"done"}\n```',
        'Sure! Here is my step:\n{"action":"final","text":"done"} hope that helps',
        '{"action":"final","text":"done",}',  # trailing comma
        '<think>hmm</think>{"action":"final","text":"done"}',
    ],
)
def test_parse_json_object_ladder(raw):
    obj = parse_json_object(raw)
    assert obj and obj["action"] == "final"


def test_parse_json_object_rejects_junk():
    assert parse_json_object("no json here at all") is None
    assert parse_json_object("") is None
    assert parse_json_object("[1,2,3]") is None  # must be an object


def test_parse_json_object_braces_inside_strings():
    obj = parse_json_object('{"action":"tool","tool":"shell","args":{"command":"echo {a} }"}}')
    assert obj and obj["args"]["command"] == "echo {a} }"


def test_validate_action_paths():
    visible = {"read", "shell"}
    all_tools = visible | {"browser"}

    action, err = validate_action({"action": "tool", "tool": "read", "args": {"path": "x"}}, visible, all_tools)
    assert action and action.kind == "tool" and action.args == {"path": "x"} and not err

    action, err = validate_action({"action": "tool", "tool": "browser"}, visible, all_tools)
    assert action is None and "open_tool" in err

    action, err = validate_action({"action": "tool", "tool": "nope"}, visible, all_tools)
    assert action is None and "Unknown tool" in err

    action, err = validate_action({"action": "open_tool", "name": "browser"}, visible, all_tools)
    assert action and action.kind == "open_tool" and action.name == "browser"

    # opening an already-visible tool is a tolerated no-op (2B models do this)
    action, err = validate_action({"action": "open_tool", "name": "read"}, visible, all_tools)
    assert action and action.kind == "open_tool" and not err

    action, err = validate_action({"action": "final", "text": "answer"}, visible, all_tools)
    assert action and action.kind == "final"

    action, err = validate_action({"action": "final", "text": ""}, visible, all_tools)
    assert action is None

    action, err = validate_action({"action": "give_up"}, visible, all_tools)
    assert action and action.reason == "no reason given"

    action, err = validate_action({"action": "dance"}, visible, all_tools)
    assert action is None and "must be one of" in err

    action, err = validate_action(None, visible, all_tools)
    assert action is None


def test_parse_plan():
    assert parse_plan('{"plan":["a","b","", 3]}') == ["a", "b", "3"]
    assert parse_plan("garbage") == []
    assert len(parse_plan(json.dumps({"plan": [f"s{i}" for i in range(10)]}))) == 6


# ── toolpack ─────────────────────────────────────────────────────────


def _defn(name, desc="Does a thing. With details nobody needs here.", props=None, required=None):
    return {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": props or {"path": {"type": "string", "description": "target path"}},
            "required": required if required is not None else ["path"],
        },
    }


def test_compact_definition_marks_required_and_caps_params():
    props = {f"p{i}": {"type": "integer"} for i in range(10)}
    props["must"] = {"type": "string"}
    compact = compact_definition(_defn("t", props=props, required=["must"]))
    assert "must*:str" in compact.param_line
    assert "must" in compact.param_names
    assert len(compact.param_names) <= 7  # required always kept, optionals capped at 6


def test_compact_definition_enum_and_first_sentence():
    compact = compact_definition(
        _defn(
            "t",
            desc="Pick a mode. Long trailing explanation that should be dropped entirely from output.",
            props={"mode": {"type": "string", "enum": ["a", "b", "c"]}},
            required=["mode"],
        )
    )
    assert "mode*:a|b|c" in compact.param_line
    assert "trailing" not in compact.description


def test_build_toolpack_core_index_and_pinning():
    defs = [_defn(n) for n in ("read", "shell", "browser", "gws")]
    pack = build_toolpack(defs)
    assert set(pack.visible) == {"read", "shell"}  # only registered cores are visible
    assert pack.index_names == ["browser", "gws"]
    assert "browser —" in pack.index_text
    assert pack.all_names == {"read", "shell", "browser", "gws"}

    pinned = build_toolpack(defs, pinned=["browser"])
    assert "browser" in pinned.visible
    assert "browser" not in pinned.index_names
    # core render order is stable (prefix caching depends on it)
    assert pinned.defs_text.index("read —") < pinned.defs_text.index("shell —")


def test_compact_params_subset_of_real_core_tools():
    """Drift gate: compaction may only ever narrow the real registry schemas."""
    from captain_claw import tools as tool_module

    checked = 0
    for cls_name in ("ReadTool", "WriteTool", "GlobTool", "GrepTool", "ShellTool",
                     "WebSearchTool", "WebFetchTool", "TodoTool"):
        cls = getattr(tool_module, cls_name, None)
        if cls is None:
            continue
        name = getattr(cls, "name", "")
        params = getattr(cls, "parameters", None)
        if not name or not isinstance(params, dict):
            continue
        real = {"name": name, "description": getattr(cls, "description", ""), "parameters": params}
        compact = compact_definition(real)
        real_props = set((params.get("properties") or {}).keys())
        assert set(compact.param_names) <= real_props, f"{name}: compact params drifted"
        for req in params.get("required") or []:
            if req in real_props:
                assert req in compact.param_names, f"{name}: required '{req}' dropped"
        checked += 1
    assert checked >= 4, "core drift test barely ran — tool exports changed?"


# ── state ────────────────────────────────────────────────────────────


def test_blackboard_roundtrip_and_new_task(tmp_path: Path):
    board = Blackboard()
    board.new_task("first")
    board.facts.append("fact one")
    board.summary = "seen it all"
    board.add_observation("tool", "read(x)", "content here")
    board.pin_tool("browser", max_pinned=2)
    board.save(tmp_path / "s.json")

    loaded = Blackboard.load(tmp_path / "s.json")
    assert loaded.task == "first"
    assert loaded.observations[0].text == "content here"
    assert loaded.pinned_tools == ["browser"]

    loaded.new_task("second")
    assert loaded.task == "second"
    assert loaded.observations == [] and loaded.pinned_tools == []
    assert loaded.facts == ["fact one"] and loaded.summary == "seen it all"


def test_blackboard_pin_lru():
    board = Blackboard()
    for name in ("a", "b", "c", "a", "d"):
        board.pin_tool(name, max_pinned=3)
    assert board.pinned_tools == ["c", "a", "d"]


def test_blackboard_load_corrupt_starts_fresh(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    board = Blackboard.load(path)
    assert board.task == "" and board.observations == []


# ── digest ───────────────────────────────────────────────────────────


def test_split_by_tokens_bounds():
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(2000))
    chunks = split_by_tokens(text, 1000)
    assert 1 < len(chunks) <= 8
    for chunk in chunks[:-1]:
        assert estimate_tokens(chunk) <= 1100


@pytest.mark.asyncio
async def test_digest_text_condenses():
    calls = []

    async def fake_complete(system, user, max_tokens):
        calls.append((system, user, max_tokens))
        assert estimate_tokens(system) + estimate_tokens(user) < 8192
        return "condensed facts"

    out = await digest_text(fake_complete, "task", "shell output", "big " * 20000, target_tokens=200)
    assert out.startswith("condensed facts")
    assert len(calls) >= 2  # chunked map + (maybe) combine
    assert estimate_tokens(out) <= 200


@pytest.mark.asyncio
async def test_digest_text_falls_back_on_empty_model_output():
    async def silent(system, user, max_tokens):
        return ""

    out = await digest_text(silent, "task", "output", "important " * 5000, target_tokens=100)
    assert out and estimate_tokens(out) <= 105


def test_describe_result():
    ok = SimpleNamespace(success=True, content="hello", error=None)
    fail = SimpleNamespace(success=False, content="", error="boom")
    empty = SimpleNamespace(success=True, content="  ", error=None)
    assert describe_result(ok) == "hello"
    assert "TOOL FAILED: boom" in describe_result(fail)
    assert "empty output" in describe_result(empty)


# ── runtime loop (fakes) ─────────────────────────────────────────────


class FakeProvider:
    """Scripted provider; records every call for cap/property assertions."""

    def __init__(self, script: list[str], model: str = "fake-2b"):
        self.script = list(script)
        self.model = model
        self.calls: list[dict] = []

    def _next(self, messages, schema, max_tokens):
        system = messages[0].content if messages else ""
        user = messages[1].content if len(messages) > 1 else ""
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": bool(schema),
                "max_tokens": max_tokens,
                "tokens": estimate_tokens(system) + estimate_tokens(user),
            }
        )
        content = self.script.pop(0) if self.script else '{"action":"give_up","reason":"script empty"}'
        return SimpleNamespace(content=content, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return self._next(messages, None, max_tokens)

    async def complete_structured(self, messages, response_schema, temperature=None, max_tokens=None):
        return self._next(messages, response_schema, max_tokens)


class FakeTools:
    """Duck-typed registry: a couple of core tools + one index-only tool."""

    def __init__(self, results: dict[str, object] | None = None):
        self.results = results or {}
        self.executed: list[tuple[str, dict]] = []

    def get_definitions(self, session_id=None, **_):
        return [
            _defn("read"),
            _defn("shell", props={"command": {"type": "string"}}, required=["command"]),
            _defn("fancy_tool", desc="Rare specialist tool", props={"level": {"type": "integer"}}, required=["level"]),
        ]

    async def execute(self, name, arguments, session_id=None, **_):
        self.executed.append((name, dict(arguments)))
        result = self.results.get(name)
        if isinstance(result, Exception):
            raise result
        if result is None:
            result = SimpleNamespace(success=True, content=f"{name} ok", error=None)
        return result


def _config(**overrides):
    base = dict(
        input_cap=8192,
        output_cap=512,
        observation_cap=400,
        digest_target=120,
        max_steps=10,
        act_retries=2,
        replan_every=0,
        max_pinned_tools=3,
        temperature=0.2,
        escalate=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _runtime(provider, tools, tmp_path, config=None, escalate_provider=None):
    return MravRuntime(
        provider=provider,
        tools=tools,
        config=config or _config(),
        session_key="test-session",
        state_dir=tmp_path,
        escalate_provider=escalate_provider,
    )


@pytest.mark.asyncio
async def test_runtime_happy_path(tmp_path: Path):
    provider = FakeProvider(
        [
            '{"plan":["read the file","answer"]}',
            '{"action":"tool","tool":"read","args":{"path":"a.txt"}}',
            '{"action":"final","text":"the file says: read ok"}',
        ]
    )
    tools = FakeTools()
    runtime = _runtime(provider, tools, tmp_path)
    reply = await runtime.run("what does a.txt say?")

    assert reply == "the file says: read ok"
    assert tools.executed == [("read", {"path": "a.txt"})]
    assert runtime.board.plan == ["read the file", "answer"]
    assert runtime.last_usage["total_tokens"] == 45  # 3 calls x 15
    # every prompt honored the hard cap
    assert all(c["tokens"] <= 8192 for c in provider.calls)
    # ACT prompts carried the frozen prefix sections
    act_user = provider.calls[1]["user"]
    assert "## TOOLS" in act_user and "## TASK" in act_user and "## INDEX" in act_user
    # trace + state persisted
    assert (tmp_path / "test-session.trace.jsonl").is_file()
    assert (tmp_path / "test-session.state.json").is_file()


@pytest.mark.asyncio
async def test_runtime_retries_invalid_then_succeeds(tmp_path: Path):
    provider = FakeProvider(
        [
            '{"plan":["answer"]}',
            "utter garbage, no json",
            '{"action":"final","text":"recovered"}',
        ]
    )
    runtime = _runtime(provider, FakeTools(), tmp_path)
    reply = await runtime.run("say something")
    assert reply == "recovered"
    # retry prompt carried the protocol error back to the model
    assert "## ERROR" in provider.calls[2]["user"]


@pytest.mark.asyncio
async def test_runtime_open_tool_paging(tmp_path: Path):
    provider = FakeProvider(
        [
            '{"plan":["open the fancy tool","use it","answer"]}',
            '{"action":"tool","tool":"fancy_tool","args":{"level":1}}',  # not visible yet → validation error
            '{"action":"open_tool","name":"fancy_tool"}',
            '{"action":"tool","tool":"fancy_tool","args":{"level":1}}',
            '{"action":"final","text":"fancy done"}',
        ]
    )
    tools = FakeTools()
    runtime = _runtime(provider, tools, tmp_path)
    reply = await runtime.run("use the fancy tool")
    assert reply == "fancy done"
    assert ("fancy_tool", {"level": 1}) in tools.executed
    # before opening: fancy_tool only in INDEX; after: schema in TOOLS
    first_act = provider.calls[1]["user"]
    assert "fancy_tool — Rare specialist tool" in first_act.split("## INDEX")[1]
    later_act = provider.calls[4]["user"]
    assert "level*:int" in later_act.split("## INDEX")[0]


@pytest.mark.asyncio
async def test_runtime_digests_oversized_tool_output(tmp_path: Path):
    big = SimpleNamespace(success=True, content="datum " * 5000, error=None)
    provider = FakeProvider(
        [
            '{"plan":["run shell","answer"]}',
            '{"action":"tool","tool":"shell","args":{"command":"dump"}}',
            "digest summary A",  # digest map call(s) — plain text
            "digest summary B",
            "digest combine",
            '{"action":"final","text":"done"}',
        ]
    )
    runtime = _runtime(provider, FakeTools(results={"shell": big}), tmp_path)
    reply = await runtime.run("dump the data")
    assert reply == "done"
    obs = runtime.board.observations
    assert obs and all(o.tokens <= runtime.observation_cap for o in obs)
    assert any(not c["schema"] for c in provider.calls)  # digest ran unstructured


@pytest.mark.asyncio
async def test_runtime_loop_guard_blocks_identical_calls(tmp_path: Path):
    provider = FakeProvider(
        [
            '{"plan":["read"]}',
            '{"action":"tool","tool":"read","args":{"path":"same.txt"}}',
            '{"action":"tool","tool":"read","args":{"path":"same.txt"}}',
            '{"action":"final","text":"ok stopping"}',
        ]
    )
    tools = FakeTools()
    runtime = _runtime(provider, tools, tmp_path)
    reply = await runtime.run("read it")
    assert reply == "ok stopping"
    assert len(tools.executed) == 1  # second identical call never executed
    assert any(o.label == "loop_guard" for o in runtime.board.observations)


@pytest.mark.asyncio
async def test_runtime_open_tool_on_visible_tool_is_forgiving_noop(tmp_path: Path):
    provider = FakeProvider(
        [
            '{"plan":["write the file"]}',
            '{"action":"open_tool","name":"read"}',  # read is already core-visible
            '{"action":"open_tool","name":"read"}',  # identical repeat → loop guard
            '{"action":"tool","tool":"read","args":{"path":"x"}}',
            '{"action":"final","text":"done"}',
        ]
    )
    tools = FakeTools()
    runtime = _runtime(provider, tools, tmp_path)
    reply = await runtime.run("read x")
    assert reply == "done"
    notes = [o for o in runtime.board.observations if o.label == "open_tool"]
    assert notes and "already in TOOLS" in notes[0].text
    assert any(o.label == "loop_guard" for o in runtime.board.observations)
    assert tools.executed == [("read", {"path": "x"})]


@pytest.mark.asyncio
async def test_runtime_gives_up_after_repeated_protocol_failures(tmp_path: Path):
    provider = FakeProvider(['{"plan":["try"]}'] + ["garbage"] * 40)
    runtime = _runtime(provider, FakeTools(), tmp_path, config=_config(act_retries=0, max_steps=8))
    reply = await runtime.run("impossible protocol")
    assert "could not complete" in reply.lower()


@pytest.mark.asyncio
async def test_runtime_escalates_after_failure_streak(tmp_path: Path):
    tools = FakeTools(results={"shell": SimpleNamespace(success=False, content="", error="denied")})
    provider = FakeProvider(
        [
            '{"plan":["shell"]}',
            '{"action":"tool","tool":"shell","args":{"command":"a"}}',
            '{"action":"tool","tool":"shell","args":{"command":"b"}}',
        ]
    )
    escalate = FakeProvider(['{"action":"final","text":"big model finished it"}'], model="fake-70b")
    runtime = _runtime(
        provider,
        tools,
        tmp_path,
        config=_config(escalate=True),
        escalate_provider=escalate,
    )
    reply = await runtime.run("do the thing")
    assert reply == "big model finished it"
    assert len(escalate.calls) == 1  # exactly one escalated step


@pytest.mark.asyncio
async def test_runtime_honest_when_steps_exhausted(tmp_path: Path):
    script = ['{"plan":["loop"]}']
    for i in range(20):
        script.append(f'{{"action":"tool","tool":"shell","args":{{"command":"c{i}"}}}}')
    provider = FakeProvider(script)
    runtime = _runtime(provider, FakeTools(), tmp_path, config=_config(max_steps=3))
    reply = await runtime.run("never finish")
    assert "ran out of steps" in reply.lower()


@pytest.mark.asyncio
async def test_runtime_small_cap_still_fits(tmp_path: Path):
    """Everything must still assemble under a much tighter cap (4k)."""
    provider = FakeProvider(
        [
            '{"plan":["answer"]}',
            '{"action":"final","text":"tiny cap fine"}',
        ]
    )
    runtime = _runtime(provider, FakeTools(), tmp_path, config=_config(input_cap=4096))
    reply = await runtime.run("q " * 3000)  # oversized task gets trimmed, not exploded
    assert reply == "tiny cap fine"
    assert all(c["tokens"] <= 4096 for c in provider.calls)
