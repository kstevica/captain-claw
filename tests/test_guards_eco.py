"""Guards and helpers that misfired on cohort001 (2026-11-19).

- The fabrication guard took "Had I set up the cron, it would hit the
  CAPTCHA every 15 minutes" (Croatian: "Da sam postavio cron, … bi …") as a
  claim that a cron was set up, and missed "…, niti sam promijenio raspored"
  ("…, nor did I change the schedule") as a negation.
- Task names came back as the instruction ("Name the task", "Nazovi
  zadatak") or "NEMA task", and the last chat's name labelled every later
  call, cron runs included.
- A tool called by its name alone (Eco mode lists some without their
  definitions) got "Missing required argument: action" and nothing else.
- The browser's refusal of a file:// address gave the agent no way forward.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from captain_claw import deck_confine
from captain_claw import fabrication_guard as fg
from captain_claw.exceptions import ToolExecutionError
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.web import chat_handler as ch

# ── schedule claims ───────────────────────────────────────────────────


@pytest.mark.parametrize("reply", [
    "Nisam pokrenuo cron — jer bi bio prazan hod. Da sam postavio cron, svakih 15 minuta "
    "bi udarao u CAPTCHA i trošio vrijeme bez ijednog upisanog retka.",
    "Nisam ga zakazao za ovaj tjedan, niti sam promijenio raspored.",
    "Had I scheduled the daily job, it would have hit the CAPTCHA every time.",
])
def test_a_hypothetical_or_negated_schedule_is_no_claim(reply):
    assert fg._schedule_claims(reply) == []


@pytest.mark.parametrize("reply", [
    "Postavio sam dnevni cron u 7:30.",
    "Potvrđujem da sam postavio cron za svaki dan.",
    "Zakazao sam posao koji bi se pokretao svaki dan.",
    "Ne brini, zakazao sam dnevni cron.",
    "I scheduled the daily job for 7:30.",
])
def test_a_real_schedule_claim_is_still_caught(reply):
    assert len(fg._schedule_claims(reply)) == 1


# ── task names ────────────────────────────────────────────────────────

MESSAGE = "Pronađi najjeftiniji MacBook Air 13 u Zagrebu"


@pytest.mark.parametrize("echo", [
    "Name the task", "Name task", "Nazovi zadatak", "Write task name", "Naziv zadatka",
    "Napiši opis zadatka", "NEMA task", "none", "Task", "",
])
def test_an_echoed_or_empty_name_falls_back_to_the_message(echo):
    assert ch._usable_task_name(echo, MESSAGE) == "Pronađi najjeftiniji MacBook Air 13 u"


@pytest.mark.parametrize("name", [
    "Task tracker setup", "Nema struje u uredu", "Croatia Airlines branding task",
    "Usporedba PC i konzola",
])
def test_a_real_name_is_kept(name):
    assert ch._usable_task_name(name, MESSAGE) == name


@pytest.mark.parametrize("text", ["da", "Može", "nastavi!", "u redu", "hvala", "hvala ti", "samo naprijed"])
def test_croatian_go_ons_are_continuations(text):
    assert ch._is_continuation(text)


def _litellm(monkeypatch, answer: str | None):
    import litellm

    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if answer is None:
            raise AssertionError("no naming call expected")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=answer))])

    monkeypatch.setattr(litellm, "acompletion", fake)
    return calls


@pytest.mark.parametrize("text", ["hvala ti", "da", "test"])
async def test_nothing_to_name_gets_no_naming_call(monkeypatch, text):
    _litellm(monkeypatch, None)
    assert await ch._generate_task_name(text, [], "openai/deepseek-flash", "k") == ""


async def test_two_words_are_still_named(monkeypatch):
    calls = _litellm(monkeypatch, "Check emails")
    assert await ch._generate_task_name("Check emails", [], "openai/deepseek-flash", "k") == "Check emails"
    assert calls


async def test_a_model_echo_becomes_the_message_words(monkeypatch):
    calls = _litellm(monkeypatch, "Name the task")
    name = await ch._generate_task_name(MESSAGE, [], "openai/deepseek-flash", "k")
    assert calls and name == "Pronađi najjeftiniji MacBook Air 13 u"


async def test_a_continuation_falls_back_to_the_task_it_continues(monkeypatch):
    _litellm(monkeypatch, "NEMA task")
    name = await ch._generate_task_name("može", [MESSAGE], "openai/deepseek-flash", "k")
    assert name == "Pronađi najjeftiniji MacBook Air 13 u"


# ── a tool called blind says what it takes ────────────────────────────


class _Link(Tool):
    name = "public_link"
    description = "share a file"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "revoke"]},
            "path": {"type": "string"},
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="ok")


def test_a_missing_argument_names_what_the_tool_takes():
    with pytest.raises(ToolExecutionError) as err:
        _Link().validate_arguments({"path": "saved/a.html"})
    msg = str(err.value)
    assert msg.startswith("Tool 'public_link' failed: Missing required argument: action.")
    assert "Required: action (one of: create, list, revoke); path (string)." in msg


def test_arguments_that_were_not_json_say_so():
    with pytest.raises(ToolExecutionError) as err:
        _Link().validate_arguments({"raw": '{"action": "create", "path": "saved/a'})
    assert "not valid JSON" in str(err.value) and "Required: action" in str(err.value)


# ── a local file in the browser ───────────────────────────────────────


def test_a_file_address_is_refused_with_a_way_forward(tmp_path):
    refusal = deck_confine._check_value("url", "url", "file:///srv/agent/saved/compare.html", {},
                                        Path(tmp_path))
    assert refusal.startswith(deck_confine.PATH_REFUSED + "url must be a web address")
    assert "`read`" in refusal and "public_link" in refusal


def test_a_web_address_still_passes(tmp_path):
    assert deck_confine._check_value("url", "url", "https://example.com/a", {}, Path(tmp_path)) is None


def test_another_scheme_keeps_the_plain_refusal(tmp_path):
    refusal = deck_confine._check_value("url", "url", "chrome://settings", {}, Path(tmp_path))
    assert refusal == deck_confine.PATH_REFUSED + "url must be a web address (http or https)"
