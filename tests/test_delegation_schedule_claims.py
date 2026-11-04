"""Replies that report other agents' work, or schedule changes, that this
turn's tool calls don't show (cohort001, 2026-11): a lead agent wrote six
phases of team results with no flight_deck call, and an agent removed the
wrong cron job by position and said it had cancelled the weekly one."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from captain_claw import fabrication_guard as fg
from captain_claw.fabrication_guard import ToolStep, TurnFacts

_ON = dict(delegation_tools=frozenset({"flight_deck", "consult_peer"}),
           schedule_tools=frozenset({"cron"}), mail_tools=frozenset(),
           calendar_tools=frozenset())

_PANTHER = """Running Phase 4. I assign the objective to Development Lead only — no direct specialist delegation from me.

Development Lead returned a full package claiming five direct contacts. Before I report any of it, I need to corroborate.

# PHASE 4 — FULL TEAM COORDINATION TEST
Development Lead independently contacted **all five** specialists, and Quality Reviewer returned 3 findings."""


def _judge(reply: str, request: str = "Run phase 4", **facts) -> fg.Verdict:
    return fg.judge_reply(reply, reply, TurnFacts(request=request, **{**_ON, **facts}))


# ── delegation ──


def test_team_results_with_no_hand_off_are_false():
    verdict = _judge(_PANTHER)
    assert verdict.kinds == (fg.DELEGATION_CLAIM,)
    assert "no agent was contacted" in verdict.corrective
    assert "call flight_deck now" in verdict.corrective
    assert verdict.backstop == fg.DELEGATION_BACKSTOP
    assert fg.has_backstop(_PANTHER + "\n\n" + verdict.backstop)


@pytest.mark.parametrize("reply", [
    "Development Lead has reported. Now the labelled corroboration consult.",
    "Both specialists confirmed the schema.",
    "I delegated the schema to the Database Engineer and it came back clean.",
    "The delegation to Quality Reviewer failed with a timeout.",
    "Kontaktirao sam Solution Architecta i dobio ugovor API-ja.",
])
def test_other_shapes_of_invented_team_work(reply):
    assert _judge(reply).kinds == (fg.DELEGATION_CLAIM,)


def test_a_peer_named_anything_is_caught_once_the_session_used_it():
    earlier = (ToolStep("flight_deck", "consult", {"agent_name": "mira"}, "Mira: here are 3 leads"),)
    assert _judge("Mira returned the list with 12 rows.", session_steps=earlier).kinds == (
        fg.DELEGATION_CLAIM,)


@pytest.mark.parametrize("reply", [
    "Once Development Lead has reported, I'll consolidate.",
    "When the agents have replied I will merge their work.",
    "Development Lead has not replied yet.",
    "Earlier, Development Lead returned the manifest.",
    "The API returned 404 for that URL.",
    "The reviewer at the journal flagged two issues in your paper.",
    "Here is the plan: I will ask the Solution Architect for the API contract.",
    "Did Development Lead confirm the schema?",
])
def test_plans_questions_negations_and_history_pass(reply):
    assert _judge(reply).kinds == ()


def test_a_real_hand_off_this_turn_makes_it_true():
    steps = (ToolStep("flight_deck", "consult", {"agent_name": "Development Lead"},
                      "Development Lead replied: package attached"),)
    assert _judge(_PANTHER, turn_steps=steps).kinds == ()


def test_a_listing_is_not_a_hand_off():
    steps = (ToolStep("flight_deck", "list_agents", {}, "Development Lead (24114) online"),)
    assert _judge(_PANTHER, turn_steps=steps).kinds == (fg.DELEGATION_CLAIM,)


def test_a_failed_call_may_be_reported_as_failed():
    steps = (ToolStep("flight_deck", "consult", {"agent_name": "Quality Reviewer"},
                      "Error: agent not reachable"),)
    assert _judge("The consultation with Quality Reviewer failed.", turn_steps=steps).kinds == ()


def test_another_agents_turn_and_recall_questions_pass():
    assert _judge(_PANTHER, others_turn=True).kinds == ()
    assert _judge("Development Lead returned the manifest with 4 tasks.",
                  request="What did Development Lead return?").kinds == ()


def test_no_hand_off_tools_no_check():
    assert _judge(_PANTHER, delegation_tools=frozenset()).kinds == ()


# ── schedules ──

_VEDRAN = """Postavljeno — dnevna provjera svaki dan u 6:00.

**Novi raspored:** `daily 04:00 UTC` = **06:00 po hrvatskom vremenu** (CEST).

**Ukinuo sam tjedni ponedjeljak 08:00** da ne dobivaš dupli izvještaj."""
_CREATE = ToolStep("cron", "create", {"action": "create"},
                   "Cron job created.\n  ID:       d41\n  Schedule: daily 04:00")


def _removed(schedule: str) -> ToolStep:
    return ToolStep("cron", "remove", {"action": "remove"},
                    f"Job x1 removed ({schedule} — DNEVNA PROVJERA).\n\nScheduled jobs now (1):")


def test_the_wrong_job_removed_is_caught():
    verdict = _judge(_VEDRAN, request="Ponavljaj mi ovo svako jutro u 6:00",
                     turn_steps=(_CREATE, _removed("daily 04:00")))
    assert verdict.kinds == (fg.SCHEDULE_CLAIM,)
    assert "daily 04:00" in verdict.corrective and "cron action=list" in verdict.corrective
    assert verdict.backstop == fg.SCHEDULE_BACKSTOP_OTHER


def test_the_right_job_removed_passes():
    assert _judge(_VEDRAN, turn_steps=(_CREATE, _removed("weekly mon 08:00"))).kinds == ()


@pytest.mark.parametrize("reply", [
    "I've scheduled a daily check at 09:00 for you.",
    "Done — the weekly report is now scheduled for Fridays.",
    "Zakazao sam dnevnu provjeru u 7:00.",
    "I cancelled the weekly job so you won't get two reports.",
])
def test_schedule_changes_with_no_cron_write_are_false(reply):
    verdict = _judge(reply, turn_steps=(ToolStep("cron", "list", {"action": "list"}, "No scheduled jobs."),))
    assert verdict.kinds == (fg.SCHEDULE_CLAIM,)
    assert verdict.backstop == fg.SCHEDULE_BACKSTOP


@pytest.mark.parametrize("reply", [
    "I haven't changed the schedule.",
    "Want me to schedule this daily?",
    "Your weekly job ran at 08:00 and found nothing new.",
    "The next run is scheduled for Monday 08:00.",
])
def test_schedule_readbacks_and_offers_pass(reply):
    assert _judge(reply).kinds == ()


def test_a_cron_create_makes_the_claim_true():
    assert _judge("I've scheduled a daily check at 04:00 UTC.", turn_steps=(_CREATE,)).kinds == ()


# ── the cron tool acts by ID and says what it changed ──


@dataclass
class _Job:
    id: str
    schedule: dict
    payload: dict = field(default_factory=dict)
    enabled: bool = True
    next_run_at: str = ""
    last_run_at: str | None = None
    last_status: str = ""


class _SM:
    def __init__(self, jobs):
        self.jobs = list(jobs)

    async def load_cron_job(self, job_id):
        return next((j for j in self.jobs if j.id == job_id), None)

    async def list_cron_jobs(self, limit=200, active_only=False):
        return sorted(self.jobs, key=lambda j: j.next_run_at)[:limit]

    async def delete_cron_job(self, job_id):
        self.jobs = [j for j in self.jobs if j.id != job_id]
        return True

    async def update_cron_job(self, job_id, **kw):
        for j in self.jobs:
            if j.id == job_id:
                j.enabled = kw.get("enabled", j.enabled)


@pytest.fixture
def cron():
    from captain_claw.tools.cron_tool import CronTool

    sm = _SM([
        _Job("684107a7c0de", {"type": "weekly", "day": "mon", "hour": 8, "minute": 0},
             {"text": "Provedi redovnu provjeru"}, next_run_at="2026-10-27T08:00"),
        _Job("d41d8cd98f00", {"type": "daily", "hour": 4, "minute": 0},
             {"text": "DNEVNA PROVJERA"}, next_run_at="2026-10-22T04:00"),
    ])
    tool = CronTool()
    tool._get_session_manager = lambda: sm
    return tool, sm


async def test_a_position_is_refused_with_the_ids(cron):
    tool, sm = cron
    result = await tool.execute(action="remove", job_id="1")
    assert not result.success and len(sm.jobs) == 2
    assert "not its position" in result.error
    assert "684107a7c0de" in result.error and "weekly mon 08:00" in result.error


async def test_remove_by_id_start_names_the_job_and_lists_the_rest(cron):
    tool, sm = cron
    result = await tool.execute(action="remove", job_id="684107a7")
    assert result.success, result.error
    assert result.content.startswith("Job 684107a7c0de removed (weekly mon 08:00 — Provedi")
    assert "Scheduled jobs now (1)" in result.content and "daily 04:00" in result.content
    assert [j.id for j in sm.jobs] == ["d41d8cd98f00"]


async def test_an_ambiguous_start_is_refused(cron):
    tool, sm = cron
    sm.jobs.append(_Job("684107ffffff", {"type": "daily", "hour": 1, "minute": 0}))
    result = await tool.execute(action="pause", job_id="6841")
    assert not result.success and "give more of the ID" in result.error
