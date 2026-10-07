"""Bat Phase 3 — pure helpers in bat_routes (vote/plan parsing, worker tool
strip, prompt + deliverable assembly). The spawn/dispatch/panel live paths are
FD-coupled and verified on a running deck; these cover the deterministic logic.

Importing bat_routes registers the real handlers into bat_loop as a side effect;
that's fine here (test_bat_loop pins its own defaults)."""

from __future__ import annotations

from captain_claw.flight_deck.bat_routes import (
    _assemble_deliverable, _bat_worker_tools, _build_step_prompt, _parse_plan, _parse_vote,
    plan_needs_gate,
)


# ── plan_needs_gate (start-gate classifier) ────────────────────────────

def test_plan_needs_gate_flags_mail_money_accounts():
    assert plan_needs_gate("email the report to the team", [])[0]
    assert plan_needs_gate("reply to that message", [])[0]
    assert plan_needs_gate("buy a domain and pay with the card", [])[0]
    assert plan_needs_gate("subscribe to the newsletter service", [])[0]
    assert plan_needs_gate("sign up for a new account on the site", [])[0]
    # reason names the category
    needs, reason = plan_needs_gate("purchase the plugin", [])
    assert needs and "spend money" in reason


def test_plan_needs_gate_scans_step_titles_too():
    needs, reason = plan_needs_gate("do the research", [{"title": "then email the findings"}])
    assert needs and "send email" in reason


def test_plan_needs_gate_allows_plain_build_research():
    assert not plan_needs_gate("summarize the latest AI papers", [{"title": "read sources"}])[0]
    assert not plan_needs_gate("refactor the auth module and add tests", [])[0]


def test_build_step_prompt_includes_human_input_when_present():
    p = _build_step_prompt({"task": "t"}, {"step_key": "s", "title": "do it"}, [],
                           human_input="the code is 4242")
    assert "4242" in p


def test_live_path_imports_resolve():
    """The planner/attempt_runner/judge/on_finish import FD symbols lazily, so a
    wrong import source only surfaces at run time (as it did with get_db). Import
    the exact live-path symbols here so pytest catches that class of bug without a
    running deck."""
    from captain_claw.flight_deck.auth import get_db  # noqa: F401
    from captain_claw.flight_deck.basna_routes import (  # noqa: F401
        _RUN_USAGE, _dispatch_one, _effective_key, _load_owner_tiers, _provider_call, _run_sid,
    )
    from captain_claw.flight_deck.server import (  # noqa: F401
        AgentConfig, DATA_DIR, _do_stop_process, _load_process_registry, _processes,
        _save_process_registry, spawn_process,
    )
    from captain_claw.flight_deck import pricing
    from captain_claw.llm import Message  # noqa: F401
    assert hasattr(pricing, "summarize")


def test_summarize_tokens_is_a_dict_and_bat_extracts_total():
    """pricing.summarize returns `tokens` as a breakdown DICT, not a scalar — Bat
    must read tokens['total_tokens'], never int(tokens) (which crashed every step
    on the first live run)."""
    from captain_claw.flight_deck import pricing
    cost = pricing.summarize([{"model": "x", "usage":
                               {"total_tokens": 7, "prompt_tokens": 4, "completion_tokens": 3}, "seconds": 1}])
    assert isinstance(cost["tokens"], dict)
    assert int((cost.get("tokens") or {}).get("total_tokens", 0) or 0) == 7


# ── _parse_vote (fail-closed) ──────────────────────────────────────────

def test_parse_vote_reads_vote_and_reason():
    v = _parse_vote("VOTE: AGREE\nREASON: everything checks out")
    assert v["vote"] == "agree" and "checks out" in v["reason"]
    assert _parse_vote("vote: disagree")["vote"] == "disagree"
    assert _parse_vote("VOTE:ABSTAIN")["vote"] == "abstain"


def test_parse_vote_missing_is_abstain():
    # No parseable VOTE line → abstain (which the judge tally treats as non-agree).
    assert _parse_vote("I think it's probably fine?")["vote"] == "abstain"
    assert _parse_vote("")["vote"] == "abstain"


# ── _parse_plan ────────────────────────────────────────────────────────

def test_parse_plan_array_of_strings():
    steps = _parse_plan('["find the data", "write the report"]', "task")
    assert [s["title"] for s in steps] == ["find the data", "write the report"]
    assert [s["step_key"] for s in steps] == ["step-1", "step-2"]
    assert [s["seq"] for s in steps] == [0, 1]


def test_parse_plan_objects_and_fences_and_steps_key():
    assert _parse_plan('```json\n[{"step": "a"}, {"title": "b"}]\n```', "t")[0]["title"] == "a"
    assert [s["title"] for s in _parse_plan('{"steps": ["x", "y"]}', "t")] == ["x", "y"]


def test_parse_plan_invalid_is_empty_and_caps_at_12():
    assert _parse_plan("not json", "t") == []
    assert _parse_plan("", "t") == []
    big = "[" + ",".join(f'"s{i}"' for i in range(20)) + "]"
    assert len(_parse_plan(big, "t")) == 12


# ── _bat_worker_tools (Bat keeps the full toolset) ─────────────────────

def test_worker_tools_strip_bat_and_send_mail():
    tools = ["read", "write", "shell", "browser", "vatra", "basna", "bat", "send_mail", "google_mail"]
    out = _bat_worker_tools(tools, default_tools=[])
    assert "bat" not in out                 # no Bat-in-Bat
    assert "send_mail" not in out           # uncapped mail path barred — FD Gmail gate only
    # the whole point of Bat: it CAN delegate to Vatra/Basna and use everything else
    assert "vatra" in out and "basna" in out and "browser" in out and "shell" in out
    assert "google_mail" in out             # the audited mail path stays


def test_worker_tools_uses_default_when_none():
    assert _bat_worker_tools(None, default_tools=["read", "bat"]) == ["read"]


# ── assembly + prompt ──────────────────────────────────────────────────

def test_assemble_deliverable_joins_done_outputs_only():
    steps = [
        {"status": "done", "title": "A", "output": "alpha"},
        {"status": "failed", "title": "B", "output": "should be skipped"},
        {"status": "done", "title": "C", "output": "   "},  # empty → skipped
        {"status": "done", "title": "D", "output": "delta"},
    ]
    out = _assemble_deliverable(steps)
    assert "alpha" in out and "delta" in out
    assert "should be skipped" not in out and "## A" in out and "## D" in out


def test_build_step_prompt_includes_goal_step_and_prior():
    run = {"task": "build the thing"}
    step = {"step_key": "step-2", "title": "write tests"}
    prior = [{"status": "done", "title": "step-1", "output": "wrote the code"}]
    p = _build_step_prompt(run, step, prior)
    assert "build the thing" in p and "write tests" in p and "wrote the code" in p
