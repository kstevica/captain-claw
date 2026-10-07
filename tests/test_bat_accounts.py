"""Bat Phase 7 — account creation / logins.

The signup and login themselves are browser+LLM driven (prompt-guided, deck
verified). These cover the enforceable pieces: the account grant via the start
gate, the credential-encryption-key requirement, verification extraction, and
the worker→human escalation (ask_human) that routes 2FA/CAPTCHA/credentials to
the owner.
"""

from __future__ import annotations

import types
import uuid

import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, human_ask
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_routes import (
    _AskReq, _build_step_prompt, agent_ask, credential_encryption_ready, extract_verification,
)
from captain_claw.flight_deck.bat_store import BatStore


# ── verification extraction ────────────────────────────────────────────

def test_extract_verification_finds_url_and_code():
    body = ("Welcome! Confirm your email: https://example.com/verify-email?token=abc123xyz\n"
            "Your verification code is 482913 if the link fails.")
    v = extract_verification(body)
    assert v["url"].startswith("https://example.com/verify-email")
    assert v["code"] == "482913"


def test_extract_verification_empty_when_absent():
    v = extract_verification("Thanks for signing up. Nothing to do.")
    assert v == {"url": "", "code": ""}


# ── credential-encryption-key requirement ──────────────────────────────

def test_credential_encryption_ready_env(monkeypatch):
    monkeypatch.setenv("CLAW_BROWSER_CREDENTIAL_KEY", "k")
    assert credential_encryption_ready() is True


def test_credential_encryption_not_ready_without_key(monkeypatch):
    monkeypatch.delenv("CLAW_BROWSER_CREDENTIAL_KEY", raising=False)
    import captain_claw.config as cfg
    stub = types.SimpleNamespace(tools=types.SimpleNamespace(
        browser=types.SimpleNamespace(credential_encryption_key="")))
    monkeypatch.setattr(cfg, "get_config", lambda: stub)
    assert credential_encryption_ready() is False


def test_step_prompt_account_guidance_warns_when_no_key(monkeypatch):
    monkeypatch.delenv("CLAW_BROWSER_CREDENTIAL_KEY", raising=False)
    import captain_claw.config as cfg
    stub = types.SimpleNamespace(tools=types.SimpleNamespace(
        browser=types.SimpleNamespace(credential_encryption_key="")))
    monkeypatch.setattr(cfg, "get_config", lambda: stub)
    run = {"task": "sign up", "config": {"account_allowed": True}}
    p = _build_step_prompt(run, {"step_key": "s", "title": "register"}, [])
    assert "Accounts (approved for this run)" in p
    assert "ask_human" in p and "CLAW_BROWSER_CREDENTIAL_KEY" in p  # the warning


def test_step_prompt_no_account_section_when_not_granted():
    p = _build_step_prompt({"task": "t", "config": {}}, {"step_key": "s", "title": "x"}, [])
    assert "Accounts (approved for this run)" not in p


# ── account grant via the start gate ───────────────────────────────────

@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


@pytest.fixture(autouse=True)
def _wire():
    async def rec(ask):
        pass
    saved = (bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, human_ask._NOTIFY)
    bat_loop.set_gate_check(bat_routes._gate_check)
    bat_loop._PLANNER = bat_loop._default_planner
    bat_loop._JUDGE = bat_loop._default_judge
    human_ask.set_notifier(rec)
    human_ask._SECRETS.clear()
    bat_routes._SECRET_ANSWERS.clear()
    yield
    bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, _old = saved
    human_ask.set_notifier(_old)


async def test_account_run_gated_then_grant_set(store):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t",
                           task="sign up for a new account on example.com",
                           config={"steps": ["register on the site"]}, status="planning")

    async def runner(run, step):
        return {"ok": True, "output": "registered"}

    drv = BatDriver(store, attempt_runner=runner)
    assert await drv.drive(rid) == "awaiting_plan"
    assert "create an account" in (await store.get_run(rid))["config"]["gate_reason"]
    await human_ask.answer(store, (await store.latest_open_ask(rid))["id"], "approve")
    assert await drv.drive(rid) == "done"
    assert (await store.get_run(rid))["config"]["account_allowed"] is True


# ── worker → human escalation endpoint (ask_human) ─────────────────────

@pytest.fixture
def _owner(monkeypatch):
    bat_loop.set_store  # noqa
    monkeypatch.setattr(bat_routes, "_resolve", lambda body: "u1")


async def test_ask_endpoint_pauses_run_and_secret_resumes(store, _owner):
    bat_loop.set_store(store)
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t", task="log in",
                           config={"steps": ["login"]}, status="running")

    d = await agent_ask(_AskReq(owner_id="u1", session_id=rid, step_key="login",
                                kind="secret", question="what is the 2FA code?"))
    assert d["status"] == "requested"
    assert (await store.get_run(rid))["status"] == "awaiting_human"
    ask = await store.latest_open_ask(rid)
    assert ask["kind"] == "secret" and ask["step_key"] == "login" and ask["secret"] is True

    # owner answers the secret; the raw value is kept in memory, redacted in the row
    await human_ask.answer(store, ask["id"], "778899", via="ui")
    assert (await store.get_ask(ask["id"]))["answer"] == human_ask._REDACTED

    # gate_check resume stashes the secret for the step (never into config)
    decision = await bat_routes._gate_check(store, await store.get_run(rid))
    assert decision == "proceed"
    assert bat_routes._SECRET_ANSWERS[(rid, "login")] == "778899"
    assert "778899" not in str((await store.get_run(rid))["config"])


async def test_ask_human_tool_refuses_outside_bat_run(monkeypatch):
    from captain_claw.tools.bat_ask import AskHumanTool
    monkeypatch.delenv("CLAW_BAT_SESSION", raising=False)
    res = await AskHumanTool().execute(question="code?", kind="secret")
    assert not res.success and "only available inside a Bat run" in res.error
