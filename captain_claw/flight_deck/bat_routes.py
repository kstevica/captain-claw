"""Bat — Flight Deck routes + the real run handlers.

This module turns the Phase-2 durable skeleton (bat_store + bat_loop) into a
live, agent-invocable mode:

  * ``/fd/bat/agent/*`` — the agent-facing endpoints the `bat` tool calls
    (start / status / get / list / cancel), gated by the same loopback-or-
    X-Agent-Secret guard as Basna/Vatra (see server ``_AGENT_GUARD_PREFIXES``).
  * The three handler **seams** registered into ``bat_loop`` at import:
      - planner      → decompose the task into ordered steps (LLM, with a
                       single-step fallback so a run always makes progress);
      - attempt_runner → spawn an ephemeral Bat worker (full toolset incl. the
                       `vatra`/`basna` launchers; only `bat` is stripped — a Bat
                       worker may delegate to Vatra), dispatch the step through
                       the proven ``_dispatch_one`` path, meter spend;
      - judge        → the fail-closed ``bat_judge`` verdict: deterministic checks
                       Captain runs itself (the steps, the run's acceptance contract,
                       the project's tests), then an independent vote panel on a
                       different model from the workers, reading what's really in
                       the run's folder;
      - replanner    → a refusal's findings become the next round's fix steps.
  * ``on_finish`` → deliver the result back to the caller / origin channel.

Phase 3 scope: agents can invoke Bat for build/research tasks. World-action
grants (the run-scoped email exception, capped real-money spend, account
creation) are Phases 5–7; a Bat worker here runs under the Phase-1 hard floor
with the owner's normal toolset.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import types
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Only its pure judges (organize_intent / explicit_mail_intent: a string in, a
# bool out) and JOB_TEXT_MAX — safe in Flight Deck: the plan gate binds, reads
# and changes no mail-authority context state; it only judges the plan the
# way Bat's workers' own guard will.
from captain_claw import mail_authority
from captain_claw.flight_deck.auth import get_current_user

from captain_claw.flight_deck import bat_loop, bat_nested, human_ask
from captain_claw.flight_deck.bat_judge import Check, evaluate as _judge_evaluate
from captain_claw.flight_deck.bat_store import RUNNING_STATES

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/fd/bat", tags=["bat"])

_WORKER_MARKER = "CLAW_BAT_WORKER"
_BAT_MAX_RUNS_PER_OWNER = 3
_STEP_TIMEOUT_S = 1800.0
_PANEL_SIZE = 3
# A panel's refusal gets this many judgements (two fix rounds). After that, more
# rounds won't settle it and the owner decides: Bat never calls a refused run done
# on its own, and a model's opinion alone can't fail one for good either.
_MAX_MODEL_VETO_ROUNDS = 3
_TIE_BREAK_ACCEPT = "Accept it as it is"
_TIE_BREAK_STOP = "Stop the run"
# The LLM ceiling for a run that doesn't name one. FD_BAT_DEFAULT_LLM_USD=0 means none.
_DEFAULT_LLM_USD = 50.0
_MAX_FIX_STEPS = 4
_MAX_CONTRACT_RULES = 6
_UNPRICED_NOTED: set[tuple[str, str]] = set()


def _llm_cap(asked: float) -> float:
    """The ceiling a run gets: what was asked for, or the deck's default."""
    if float(asked or 0.0) > 0:
        return float(asked)
    try:
        return max(0.0, float(os.environ.get("FD_BAT_DEFAULT_LLM_USD", _DEFAULT_LLM_USD)))
    except ValueError:
        return _DEFAULT_LLM_USD


# ── request models ────────────────────────────────────────────────────

class _BatAgentReq(BaseModel):
    source_port: int = 0
    web_auth: str = ""
    owner_id: str = ""
    run_id: str = ""
    query: str = ""
    limit: int = Field(default=30, ge=1, le=200)


class BatStartReq(_BatAgentReq):
    task: str = ""
    title: str = ""
    llm_usd_cap: float = 0.0
    real_usd_cap: float = 0.0       # hard ceiling on real-money spend for the run
    per_item_usd: float = 0.0       # purchases at/below this auto-approve; above → owner
    worker_mode: str = "plain"      # plain = generic full-toolset worker; archetype = specialist per step
    steps: list[str] = Field(default_factory=list)
    origin_platform: str = "web"
    origin_user_id: str = ""
    origin_chat_id: int = 0
    origin_kind: str = ""
    origin_address: str = ""
    source_host: str = "localhost"


# ── pure helpers (unit-tested) ─────────────────────────────────────────

_VOTE_RE = re.compile(r"^\s*VOTE:\s*(AGREE|DISAGREE|ABSTAIN)", re.IGNORECASE | re.MULTILINE)
_REASON_RE = re.compile(r"^\s*REASON:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# raw secret answers injected into a resuming step, kept in memory only
_SECRET_ANSWERS: dict[tuple[str, str], str] = {}

# Start-gate classifier (owner decision #3): a run whose plan would send mail,
# spend money, or create an account pauses for approval before it acts. Pure.
_GATE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("send email", r"\b(e-?mails?|gmail|inbox|send (a )?(message|mail|note)|reply to|forward (the|this|an? ))\b"),
    ("spend money", r"\b(buy|purchase|pay(ment|ing)?|subscribe|subscription|checkout|check out|order|pre-?order|donate|rent|book (a|the|my)|top up|deposit|withdraw)\b"),
    ("create an account", r"\b(sign ?up|create an? (new )?account|register (for|an?|with)|open an account|make an account)\b"),
)
_GATE_COMPILED = tuple((label, re.compile(rx, re.IGNORECASE)) for label, rx in _GATE_PATTERNS)
# Archiving / moving / labeling the owner's mail is its own grant: approving it
# never lets the run write email (see _bat_automation).
_ORGANIZE_GATE = "organize email"


def _plan_text(task: str, steps: list[dict]) -> str:
    """The task and its step titles — the plan the owner is asked to approve."""
    return (task or "") + "\n" + "\n".join(s.get("title", "") for s in (steps or []))


# The plan's sentences: its lines, and what ends a sentence in one.
_PLAN_BREAK_RE = re.compile(r"[.!?;]+(?=\s|$)|\n")
# A sentence that names nothing of its own to act on ("Archive them", "Archive
# the ones you summarized", "arhiviraj ostale"): it means what the plan named
# before it.
_BACK_REF_RE = re.compile(
    r"\b(?:them|it|those|these|the\s+(?:ones?|rest|others)|each|both|all\s+of\s+them"
    r"|ih|njih|ga|ostal\w*|preostal\w*)\b",
    re.IGNORECASE,
)
# How many sentences back a back-referring one may reach for what it means —
# it keeps the gate linear in a long, pronoun-heavy plan.
_BACK_REF_SPAN = 5


def _back_ref_readings(plan: str, room: int = mail_authority.JOB_TEXT_MAX) -> list[str]:
    """Each plan sentence that refers back ("Archive the ones you summarized"),
    read as one sentence with what it refers to — the sentences before it,
    back to the nearest one that names something ("Compile a digest of the
    newsletters I received this week - Archive the ones you summarized") —
    when that reading asks to organize email and the sentence alone doesn't.
    A step that names its own object ("Move the submit button to the footer"
    after "Build a newsletter signup form") is never read with another, and
    nothing is read past that nearest naming sentence — nor more than
    :data:`_BACK_REF_SPAN` sentences back. Readings stop once they would fill
    *room* characters (the job text has no space for more)."""
    sents = [t.strip() for t in _PLAN_BREAK_RE.split(plan) if t.strip()]
    out = []
    for i in range(1, len(sents)):
        if room <= 0:
            break
        if not _BACK_REF_RE.search(sents[i]) or mail_authority.organize_intent(sents[i]):
            continue
        j = i - 1
        while j > max(0, i - _BACK_REF_SPAN) and _BACK_REF_RE.search(sents[j]):
            j -= 1
        reading = " - ".join(sents[j:i + 1])
        if mail_authority.organize_intent(reading):
            out.append(reading)
            room -= len(reading) + 1
    return out


def _organize_text(task: str, steps: list[dict]) -> str:
    """The plan as an organize-only run's workers get it (``job_text``), cut
    where their mail guard cuts it — so the gate judges what they will.

    The guard reads a sentence at a time, and a step is part of the whole
    plan: so the plan is followed by its back-referring steps read with what
    they refer to (:func:`_back_ref_readings`)."""
    plan = _plan_text(task, steps)
    room = mail_authority.JOB_TEXT_MAX - len(plan) - 1
    readings = _back_ref_readings(plan, room) if room > 0 else []
    return "\n".join([plan, *readings])[:mail_authority.JOB_TEXT_MAX]


def plan_needs_gate(task: str, steps: list[dict]) -> tuple[bool, str]:
    """True + a reason when the task/plan would act in the world in a way the
    owner asked to approve first (mail / money / account creation / organizing
    mail). Deterministic keyword scan over the task and the step titles.

    Organizing mail is judged by the workers' own mail guard (mail_authority —
    stdlib-only) on the text they would get — the whole plan, a step that
    refers back read with what it refers to (:func:`_organize_text`): every
    plan it would let organize is offered for approval, so its refusal in an
    unapproved run is true. An
    approved organize-only plan runs under that guard's ``intent`` mode, which
    would also honour an email ask the keywords above miss ("pošalji Ani
    izvještaj mailom") — so such a plan is gated as mail too."""
    blob = _plan_text(task, steps)
    hits = [label for label, rx in _GATE_COMPILED if rx.search(blob)]
    text = _organize_text(task, steps)
    if mail_authority.organize_intent(text):
        if "send email" not in hits and mail_authority.explicit_mail_intent(text):
            hits.insert(0, "send email")
        hits.append(_ORGANIZE_GATE)
    return (bool(hits), ", ".join(hits))


def _bat_automation(run: dict) -> dict:
    """The automation frame for one Bat step — Bat's own dispatch.

    A child run the worker starts (a peer or Flight Deck hand-off, an
    agent-made cron job) doesn't inherit the frame, but under ``allow`` and
    ``intent`` its ``job_text`` is that child's intent source
    (mail_authority.intent_source_text): the child may write or organize
    mail only where its own text asks too (narrower_intent). So a cron job
    "every morning archive the Substack newsletters", made by a worker under
    an approved organize plan, may archive when it fires — as under
    ``allow``. Under ``deny`` nothing is carried.

    Run-scoped email exception (Phase 5): ``allow`` ONLY when the owner
    approved a mail plan for THIS run. An approved organize-only plan may
    archive / move / label but not write email: ``intent``, judged on the plan
    the owner approved — it asks to organize, and (the gate made sure) for no
    email. Otherwise deny (PR #56 default)."""
    cfg = run.get("config") or {}
    task = str(run.get("task", ""))
    if cfg.get("email_allowed"):
        return {"kind": "bat", "job_text": task[:500], "mail_write": "allow"}
    if cfg.get("organize_allowed"):
        return {"kind": "bat", "job_text": str(cfg.get("organize_text") or ""),
                "mail_write": "intent"}
    return {"kind": "bat", "job_text": task[:500], "mail_write": "deny"}


_APPROVE_WORDS = {"approve", "approved", "go", "yes", "ok", "okay", "proceed", "run", "do it"}

# Button/link text that means "this click spends money" — a pre-click guard for
# the browser (the deck-side Playwright wiring uses this; pure + tested here).
_PURCHASE_RE = re.compile(
    r"\b(place (your )?order|pay now|buy now|complete (purchase|order|payment)|"
    r"confirm (and )?pay|checkout|check out|subscribe|start (free )?trial|"
    r"proceed to payment|submit payment|purchase)\b", re.IGNORECASE)


def looks_like_purchase(text: str) -> bool:
    """True if on-screen control text reads as a money-committing action."""
    return bool(_PURCHASE_RE.search(text or ""))


def credential_encryption_ready() -> bool:
    """True when stored browser credentials will be Fernet-encrypted rather than
    base64-obfuscated. A Bat run that creates accounts / logs in should not store
    near-plaintext passwords, so we require a key to be configured."""
    if (os.environ.get("CLAW_BROWSER_CREDENTIAL_KEY") or "").strip():
        return True
    try:
        from captain_claw.config import get_config
        return bool((get_config().tools.browser.credential_encryption_key or "").strip())
    except Exception:
        return False


# A verification link or a short numeric/alphanumeric code in a signup email.
_VERIFY_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+(?:verify|confirm|activate|validation|verify-email|confirm-email)[^\s\"'<>]*",
    re.IGNORECASE)
_VERIFY_CODE_RE = re.compile(
    r"(?:verification code|one[- ]?time (?:code|password|pin)|\bcode\b|\botp\b|\bpin\b)\b"
    r".{0,20}?\b((?=[A-Za-z0-9]*\d)[A-Za-z0-9]{4,8})\b",
    re.IGNORECASE)


def extract_verification(text: str) -> dict:
    """Pull a verification link and/or code out of an email body. (html_to_text
    strips <a href>, so a worker should pass the raw HTML or plain-text part.)"""
    t = text or ""
    url = _VERIFY_URL_RE.search(t)
    code = _VERIFY_CODE_RE.search(t)
    return {"url": url.group(0) if url else "", "code": code.group(1) if code else ""}


def _spend_enabled() -> bool:
    """Deck kill-switch. Real-money spend is OFF unless FD_BAT_SPEND is on."""
    return (os.environ.get("FD_BAT_SPEND", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _spend_policy(run: dict) -> dict:
    """The effective spend policy for a run: enabled only when the deck switch is
    on AND the owner approved a spend plan AND a run cap was set."""
    cfg = run.get("config") or {}
    cap = float(cfg.get("real_usd_cap") or 0.0)
    enabled = _spend_enabled() and bool(cfg.get("spend_allowed")) and cap > 0
    return {"enabled": enabled, "per_item_usd": float(cfg.get("per_item_usd") or 0.0), "run_usd_cap": cap}


def spend_decision(policy: dict, committed_usd: float, amount_usd: float) -> tuple[str, str]:
    """Pure cap logic. Returns (decision, reason):
      denied      — spend disabled, non-positive amount, or over the run cap;
      needs_human — within the cap but above the per-item auto-approve threshold;
      approved    — within the cap and at/below the per-item threshold."""
    if not policy.get("enabled"):
        return "denied", "real-money spend is not enabled for this run"
    amt = float(amount_usd or 0.0)
    if amt <= 0:
        return "denied", "amount must be positive"
    cap = float(policy.get("run_usd_cap") or 0.0)
    if committed_usd + amt > cap + 1e-9:
        return "denied", f"would exceed the run cap (${committed_usd:.2f} committed + ${amt:.2f} > ${cap:.2f})"
    per_item = float(policy.get("per_item_usd") or 0.0)
    if amt <= per_item:
        return "approved", f"within the per-item auto-approve limit (${per_item:.2f})"
    return "needs_human", f"above the per-item auto-approve limit (${per_item:.2f}) — owner approval required"


def _parse_vote(text: str) -> dict:
    """Parse a judge reply. No parseable VOTE line → 'abstain' (which the judge
    tally treats as non-agree; an all-abstain panel is no_quorum = not done)."""
    m = _VOTE_RE.search(text or "")
    vote = m.group(1).lower() if m else "abstain"
    r = _REASON_RE.search(text or "")
    return {"vote": vote, "reason": (r.group(1).strip() if r else "")[:300]}


def _after_of(item: Any, i: int) -> list[str] | None:
    """The step keys a plan item waits for. None when the planner didn't say, which
    means every earlier step. Only earlier steps count, so a plan can't deadlock."""
    raw = (item.get("after", item.get("depends_on")) if isinstance(item, dict) else None)
    if not isinstance(raw, list):
        return None
    out = []
    for a in raw:
        tail = str(a).rsplit("-", 1)[-1]
        if tail.isdigit() and 1 <= int(tail) <= i:
            out.append(f"step-{int(tail)}")
    return out


def _parse_plan(content: str, task: str) -> list[dict]:
    """Parse the planner LLM's JSON into ordered steps. Tolerates ``` fences.
    Each step: {step_key, title, seq, depends_on}. Empty/invalid → [] (caller falls back)."""
    c = (content or "").strip()
    if c.startswith("```"):
        c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
    try:
        raw = json.loads(c)
    except (json.JSONDecodeError, TypeError):
        return []
    items = raw.get("steps") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for i, it in enumerate(items):
        if isinstance(it, str) and it.strip():
            title = it.strip()
        elif isinstance(it, dict):
            title = str(it.get("step") or it.get("title") or it.get("task") or "").strip()
        else:
            continue
        if title:
            out.append({"step_key": f"step-{i + 1}", "title": title[:400], "seq": i,
                        "depends_on": _after_of(it, i)})
    return out[:12]


def _parse_fix_plan(content: str) -> list[dict]:
    """A fix round's new steps: [{title, after}], ``after`` as 1-based places in the
    same list (bat_loop.fix_steps gives them their keys). Invalid → []."""
    c = (content or "").strip()
    if c.startswith("```"):
        c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
    try:
        raw = json.loads(c)
    except (json.JSONDecodeError, TypeError):
        return []
    items = raw.get("steps") if isinstance(raw, dict) else raw
    out: list[dict] = []
    for it in items if isinstance(items, list) else []:
        title = (it if isinstance(it, str) else
                 str(it.get("step") or it.get("title") or "") if isinstance(it, dict) else "").strip()
        if title:
            after = it.get("after") if isinstance(it, dict) else None
            out.append({"title": title[:400],
                        "after": [a for a in after if isinstance(a, int)] if isinstance(after, list) else []})
    return out[:_MAX_FIX_STEPS]


# Tools a Bat worker never gets: the `bat` launcher (blocks Bat-in-Bat) and
# `send_mail` (the direct Mailgun/SendGrid/SMTP path has no allowlist, rate
# limit or audit — Bat's email exception routes ONLY through the FD Gmail gate,
# which keeps the owner's allowlist/daily-limit/audit, so send_mail is barred).
_BAT_TOOL_DENY = frozenset({"bat", "send_mail"})


def _parse_archetype_plan(content: str, valid_ids: set[str]) -> tuple[list[dict], dict[str, str]]:
    """Parse an archetype-mode plan: a JSON array of {step, archetype}. Returns
    the ordered steps and a {step_key -> archetype_id} map (only ids in the
    catalog; an unknown/blank archetype falls back to a plain worker for that
    step). Tolerates ``` fences; invalid JSON → ([], {})."""
    c = (content or "").strip()
    if c.startswith("```"):
        c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
    try:
        raw = json.loads(c)
    except (json.JSONDecodeError, TypeError):
        return [], {}
    items = raw.get("steps") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], {}
    steps: list[dict] = []
    arch_map: dict[str, str] = {}
    for i, it in enumerate(items):
        if isinstance(it, str):
            title, arch = it.strip(), ""
        elif isinstance(it, dict):
            title = str(it.get("step") or it.get("title") or it.get("task") or "").strip()
            arch = str(it.get("archetype") or it.get("archetype_id") or it.get("agent") or "").strip()
        else:
            continue
        if not title:
            continue
        key = f"step-{i + 1}"
        steps.append({"step_key": key, "title": title[:400], "seq": i, "depends_on": _after_of(it, i)})
        if arch and arch in valid_ids:
            arch_map[key] = arch
    return steps[:12], {k: v for k, v in arch_map.items() if k in {s["step_key"] for s in steps[:12]}}


def _bat_worker_tools(tools: list[str] | None, default_tools: list[str]) -> list[str]:
    """Bat workers keep the FULL toolset (including the vatra/basna launchers so
    a Bat worker may delegate to a Vatra run) EXCEPT the few in _BAT_TOOL_DENY."""
    base = list(tools if tools else default_tools)
    return [t for t in base if t not in _BAT_TOOL_DENY]


def _email_receipt_block(sends: list[dict], sid8: str) -> str:
    """Ground truth from the FD Gmail send audit for this run's workers — so the
    judge panel can tell a real 'I emailed X' from a fabricated one."""
    mine = [s for s in sends if str(s.get("agent", "")).startswith(f"bat-{sid8}")]
    if not mine:
        return ("## VERIFIED — email send audit\n"
                "The audited gateway recorded NO emails actually sent by this run. "
                "Treat any claim that an email was sent as unverified.")
    lines = [f"- to {s.get('to', '?')}: {s.get('subject') or '(no subject)'} [{s.get('status', '?')}]"
             for s in mine[:10]]
    return (f"## VERIFIED — email send audit: {len(mine)} email(s) actually sent\n"
            + "\n".join(lines))


def _assemble_deliverable(steps: list[dict]) -> str:
    done = [s for s in steps if s.get("status") == "done" and (s.get("output") or "").strip()]
    parts = []
    for s in done:
        head = s.get("title") or s.get("step_key") or ""
        parts.append(f"## {head}\n\n{s['output'].strip()}" if head else s["output"].strip())
    return "\n\n".join(parts)


def _final_deliverable(steps: list[dict]) -> str:
    """What the judge evaluates for 'done': what the plan ends in, the output of
    every completed step nothing else waits for (the last step of a straight plan;
    every branch and fix of a parallel one) — not the concatenation of every step's
    working notes, which reads as unfinished. Falls back to the full assembly only
    when that's too thin to stand alone."""
    done = [s for s in steps if s.get("status") == "done" and (s.get("output") or "").strip()]
    if not done:
        return ""
    needed = {d for ds in bat_loop.effective_deps(steps).values() for d in ds}
    ends = [s for s in done if s["step_key"] not in needed] or done[-1:]
    out = "\n\n".join((s.get("output") or "").strip() for s in ends)
    return out if len(out) >= 200 else _assemble_deliverable(steps)


_STEP_SYSTEM = (
    "You are a Bat worker — the stubborn finisher. Complete the assigned step no matter how many "
    "approaches it takes; retry, change tactics, and use any tool available to you (including the "
    "browser, computer use, and starting a `vatra` run for a sub-problem). If you start a Basna, Vatra, "
    "Code or Crew run, end your turn once it has started: Captain waits for it and adds its result to "
    "this step, so don't poll it. Be honest: only report "
    "what you actually accomplished and verified — never claim a step is done when it isn't. "
    "If you are blocked on something only the human can provide (a verification code, a 2FA/OTP, a "
    "CAPTCHA, a credential), call the `ask_human` tool (kind='secret' for codes/passwords) and STOP — "
    "never guess or fabricate it, and never try to bypass or solve a CAPTCHA yourself. "
    "Absolutely never run anything that could harm the host or its drives."
)


def _build_step_prompt(run: dict, step: dict, prior: list[dict], human_input: str = "") -> str:
    lines = [_STEP_SYSTEM, f"\n# Goal\n{run.get('task', '')}", f"\n# Your step\n{step.get('title') or step.get('step_key')}"]
    if (step.get("brief") or "").strip():
        lines.append(f"\n# Why this step exists\nThe work was checked and these problems were found. "
                     f"Fix them, and report the whole result, not only what changed.\n{step['brief']}")
    if int(step.get("attempt", 0) or 0) >= 1 and (step.get("error") or "").strip():
        lines.append(f"\n# Your last attempt at this step failed\n{step['error'][:1500]}\n"
                     "Don't repeat what failed. Take a different approach.")
    if run.get("id"):
        lines.append(f"\n# Where files go\nWrite every file to vfs:{_project_of(run)}/<name>. Your own folders, "
                     "saved/ included, are deleted when your step ends, and only this folder counts as delivered.")
    rules = (run.get("config") or {}).get("contract") or []
    if rules:
        lines.append(f"\n# How this run will be checked\nIt's done only when these hold. Captain checks them "
                     f"itself and doesn't take a report's word for it. File paths are inside "
                     f"vfs:{_project_of(run)}/.\n" + "\n".join(f"- {r['text']}" for r in rules))
    if any(s.get("status") == "running" and s.get("step_key") != step.get("step_key") for s in prior):
        lines.append("\n# Others are working at the same time\nOther steps are running in the same folder "
                     "right now. Keep to your own files.")
    done = [s for s in prior if s.get("status") == "done" and (s.get("output") or "").strip()]
    if done:
        lines.append("\n# What earlier steps produced (context)")
        for s in done:
            lines.append(f"\n## {s.get('title') or s.get('step_key')}\n{(s.get('output') or '')[:2000]}")
    if human_input:
        lines.append(f"\n# The human just provided this (use it to continue)\n{human_input}")
    if (run.get("config") or {}).get("account_allowed"):
        note = ("\n# Accounts (approved for this run)\nYou may sign up for and log into sites. To get a "
                "verification link/code, read the owner's inbox with the google_mail tool (the plain-text "
                "or raw-HTML part — the rendered text drops link URLs). For a 2FA/OTP, a CAPTCHA, or a "
                "credential you don't have, use `ask_human` and stop.")
        if not credential_encryption_ready():
            note += (" NOTE: no browser credential-encryption key is configured, so do NOT store "
                     "passwords in the browser credential store — ask the human to set "
                     "CLAW_BROWSER_CREDENTIAL_KEY first, or keep credentials out of storage.")
        lines.append(note)
    lines.append("\nDo this step, and only this step: other steps cover the rest of the goal. Do it fully and "
                 "report the concrete result. If you are truly blocked on "
                 "something only the human can give you, use the `ask_human` tool and stop — do not "
                 "guess or fabricate it.")
    return "\n".join(lines)


_VOTE_SYSTEM = (
    "You are an independent, skeptical reviewer deciding whether a task is GENUINELY and fully done, "
    "judging the delivered result itself — not any claim that it is finished. Default to DISAGREE when "
    "the evidence is thin, partial, or merely describes what should be done rather than showing it was. "
    "A section marked VERIFIED comes from Captain itself: believe it over anything a worker says. When "
    "you disagree, the reason goes to whoever fixes it, so name what's wrong or missing. "
    "Answer with exactly two lines:\nVOTE: AGREE|DISAGREE|ABSTAIN\nREASON: <one sentence>"
)


def _vote_prompt(lens: str, task: str, deliverable: str) -> str:
    return (f"Judge through this lens: {lens}\n\n# Task\n{task[:2000]}\n\n"
            f"# Delivered result\n{deliverable[:16000]}\n\n"
            f"Is the task genuinely and fully done?")


# ── what a run is checked against ──────────────────────────────────────

_CONTRACT_SYSTEM = (
    "You write the acceptance checks for a task an AI team will carry out: the conditions that, once "
    "true, mean it's actually done, not merely that someone said so. List only what the task itself "
    f"asks for, no style preferences, at most {_MAX_CONTRACT_RULES}. Nobody can correct a check once the work "
    "has started, so test what the task asks for and not a format it never mentions (a trailing newline, "
    "exact spacing), and when a script would be fragile, leave the rule to a reviewer. All the team's files "
    "go in one shared folder, and paths are relative to it. Reply ONLY with JSON, no prose:\n"
    '{"constraints": [{"id": "a1", "text": "<one line>", "severity": "critical|major", "check": {...}}]}\n'
    "check shapes, the first that fits:\n"
    '- {"type": "file_exists", "path": "<path>"}: only for a file the task names or clearly requires\n'
    '- {"type": "file_contains", "path": "<path>", "pattern": "<regex>"}: ^ and $ match at each line\n'
    '- {"type": "no_pattern", "pattern": "<regex>", "glob": "<glob>"}: passes when no file matches\n'
    "COMMAND_SHAPE"
    '- {"type": "judge"}: anything else; a reviewer decides from the result\n'
    'If the task states nothing checkable, return {"constraints": []}.'
)
_COMMAND_SHAPE = ('- {"type": "command", "cmd": "<shell command>"}: passes when it exits 0; cheap, '
                  "deterministic and portable (wc -l pads its number on some systems), no servers or network\n")


def _project_of(run: dict) -> str:
    """The run's shared VFS folder, as its workers are told (_bat_env)."""
    return f"bat-{str(run.get('id', ''))[:8]}"


def _run_folder(run: dict) -> Path:
    from captain_claw.flight_deck.vfs_routes import _user_root
    return _user_root(run["owner_id"]) / _project_of(run)


def _owner_has_shell(user: dict | None) -> bool:
    """A contract command and a project's tests run on the deck itself, outside any
    worker's confinement. The same people who already have a shell here (Crew's
    `trusted`, hosting's admin rule) may have them; for anyone else a command rule
    goes to the reviewers instead."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    return not _fd_auth_enabled() or (user or {}).get("role") == "admin"


async def _floor_runner(command: str, cwd: str) -> tuple[bool, str]:
    """Run a check under the same floor as the workers' own commands."""
    from captain_claw import bat_floor
    from captain_claw.dubina.coder import shell_command_runner
    blocked, why = bat_floor.screen("shell", {"command": command})
    if blocked:
        return False, bat_floor.refusal("shell", why)
    return await shell_command_runner(command, cwd)


def _read_rules(reply: str) -> list[dict]:
    """The rules in a model's reply. A weak model often drops a brace, and one drop
    shouldn't cost every rule: when the whole reply won't parse, each rule is read
    on its own, closing what it left open."""
    from captain_claw.flight_deck import code_contract
    rules = code_contract.parse_contract(reply)
    if rules or not reply:
        return rules
    items = []
    starts = [m.start() for m in re.finditer(r'\{\s*"id"\s*:', reply)]
    for start, end in zip(starts, [*starts[1:], len(reply)]):
        text = reply[start:end].strip().removesuffix("```").rstrip().rstrip(",]} \n")
        for closers in ("}", "}}", '"}}'):
            try:
                items.append(json.loads(text + closers))
                break
            except ValueError:
                continue
    return code_contract.parse_contract(json.dumps({"constraints": items}))


async def _ensure_contract(run: dict, creds: dict | None, has_shell: bool) -> None:
    """Derive the run's acceptance contract once and pin it in bat.db, outside any
    folder a worker can write to. No contract just means a run rests on its steps
    and its reviewers, so a failure here never stops one."""
    cfg = dict(run.get("config") or {})
    store = bat_loop.get_store()
    if "contract" in cfg or not creds or store is None:
        return
    from captain_claw.flight_deck.basna_routes import _provider_call
    from captain_claw.llm import Message
    rules: list[dict] = []
    try:
        prov, mt = _provider_call(creds, temperature=0.0, default_max=4096, cap=16384)
        resp = await prov.complete(messages=[
            Message(role="system", content=_CONTRACT_SYSTEM.replace("COMMAND_SHAPE", _COMMAND_SHAPE if has_shell else "")),
            Message(role="user", content=str(run.get("task", ""))[:8000]),
        ], temperature=0.0, max_tokens=mt)
        rules = _read_rules(resp.content or "")[:_MAX_CONTRACT_RULES]
    except Exception as e:  # noqa: BLE001
        log.warning("bat contract derivation failed", run_id=run.get("id"), error=str(e))
    cfg["contract"] = rules
    await store.set_config(run["id"], cfg)
    await store.append_event(run["id"], "note", (
        f"{len(rules)} acceptance check(s) pinned: " + "; ".join(r["text"] for r in rules)[:400] if rules
        else "No acceptance checks came out of the task, so the reviewers alone decide when it's done."))


async def _contract_checks(run: dict, has_shell: bool) -> tuple[list[Check], list[dict], list[dict]]:
    """Run what can be checked without a model. Returns the results, the rules that
    take a reviewer, and the rules a script failed. Only a rule a script decided can
    block a run for good."""
    from captain_claw.flight_deck import code_contract, code_verify
    rules = []
    for r in (run.get("config") or {}).get("contract") or []:
        command_blocked = (r.get("check") or {}).get("type") == "command" and not has_shell
        rules.append({**r, "check": {"type": "judge"}} if command_blocked else r)
    folder = _run_folder(run)
    res = await code_contract.validate(folder, rules, runner=_floor_runner)
    checks = [Check(f"acceptance: {r['text']}", passed=True, critical=r["severity"] != "minor")
              for r in res["passed"]]
    checks += [Check(f"acceptance: {r['text']}", passed=False, critical=r["severity"] != "minor",
                     detail=str(r.get("note", ""))[:600]) for r in res["failed"]]
    if has_shell and folder.is_dir():
        command = code_verify.detect_test_command(folder)
        if command:
            result = await code_verify.run_tests(folder, command, runner=_floor_runner)
            checks.append(Check("the project's tests pass", passed=bool(result["ok"]),
                                detail="" if result["ok"] else f"`{command}`: {str(result['output'])[-600:]}"))
    return checks, res["unresolved"], res["failed"]


_AUDIT_SYSTEM = (
    "Automatic checks failed on an AI team's work. A model wrote those checks before the work existed, so a "
    "check can itself be wrong: a regex or shell quirk, or a format the task never asked for. For EACH rule, "
    "look at the real files and decide whether the work meets it. pass: the files show the rule holds, and "
    "you can name what's wrong with the check. fail: the work doesn't meet it. unclear: you can't tell. "
    'Reply ONLY with a JSON array: [{"id": "<id>", "verdict": "pass|fail|unclear", "note": "<one line>"}]'
)


async def _audit_failed_checks(run: dict, failed: list[dict], creds: dict, folder: Path) -> dict[str, str]:
    """A check written by a model can be wrong, and a wrong one would hold a correct
    run back for good, so a failed one gets a second look before it counts. Returns
    {rule id: why} for the ones whose check is at fault. That doesn't pass them: the
    rule goes to the reviewers instead. Anything unclear stays a failure."""
    from captain_claw.flight_deck import code_contract
    from captain_claw.flight_deck.basna_routes import _provider_call
    from captain_claw.llm import Message
    rules = {str(r["id"]): r for r in (run.get("config") or {}).get("contract") or []}
    listing = "\n".join(
        f"- id \"{f['id']}\": {f['text']}\n  checked by: {json.dumps((rules.get(str(f['id'])) or {}).get('check') or {})}"
        f"\n  it reported: {str(f.get('note', ''))[:600]}" for f in failed)
    evidence = await asyncio.to_thread(_evidence_block, folder, excerpts=5)
    try:
        async with _metered(run["id"], "audit"):
            prov, mt = _provider_call(creds, temperature=0.0, default_max=4096, cap=16384)
            resp = await prov.complete(messages=[
                Message(role="system", content=_AUDIT_SYSTEM),
                Message(role="user", content=f"# Task\n{str(run.get('task', ''))[:4000]}\n\n# Failed checks\n{listing}\n\n{evidence}"),
            ], temperature=0.0, max_tokens=mt)
        return {j["id"]: j.get("note", "") for j in code_contract.parse_judgement(resp.content or "")
                if j["verdict"] == "pass"}
    except Exception as e:  # noqa: BLE001 — no second look means the failure stands
        log.warning("bat check audit failed", run_id=run.get("id"), error=str(e))
        return {}


_TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".html", ".css", ".js", ".ts", ".tsx", ".py", ".yaml",
                  ".yml", ".toml", ".xml", ".sh", ".sql", ".rst", ".tex"}


_NO_FILES = "The folder holds no files."


def _evidence_block(folder: Path, *, max_files: int = 40, excerpts: int = 3, excerpt_chars: int = 1500) -> str:
    """What's really in the run's folder, read by Captain and not reported by a
    worker: the panel judges this, not only the story of it. Hidden folders aren't
    walked and the walk has a limit, since a worker may have cloned anything here."""
    head = "## VERIFIED — files in the run's shared folder"
    found: list[tuple[float, str, Path, int]] = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            p = Path(root) / name
            if name.startswith(".") or p.is_symlink() or len(found) >= 5000:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            found.append((st.st_mtime, p.relative_to(folder).as_posix(), p, st.st_size))
    if not found:
        return f"{head}\n{_NO_FILES} Any claim that a file was written there is false."
    found.sort(key=lambda f: f[0], reverse=True)
    lines = [head, *[f"- {rel} ({size:,} bytes)" for _, rel, _, size in found[:max_files]]]
    if len(found) > max_files:
        lines.append(f"- … and {len(found) - max_files} more")
    shown = 0
    for _, rel, p, size in found:
        if shown >= excerpts:
            break
        if p.suffix.lower() in _TEXT_SUFFIXES and size:
            try:
                body = p.read_text(encoding="utf-8", errors="replace")[:excerpt_chars]
            except OSError:
                continue
            lines.append(f"\n### {rel}\n{body}")
            shown += 1
    return "\n".join(lines)


def _findings(verdict: dict) -> str:
    """What a judgement found, for whoever has to fix it."""
    lines = [f"- Failed check: {c}" for c in verdict.get("det_criticals") or []]
    lines += [f"- Reviewer: {v['reason']}" for v in (verdict.get("panel") or {}).get("votes") or []
              if v.get("vote") != "agree" and v.get("reason")]
    return "\n".join(lines) or f"- {verdict.get('reason', 'not done yet')}"


@asynccontextmanager
async def _metered(run_id: str, label: str):
    """Model calls inside this block count against the run's budget. Each block has
    its own account, so steps running at the same time can't count each other."""
    from captain_claw.flight_deck import pricing
    from captain_claw.flight_deck.basna_routes import _RUN_USAGE, _run_sid
    key = f"{run_id}:{label}"
    token = _run_sid.set(key)
    _RUN_USAGE[key] = []
    try:
        yield
    finally:
        _run_sid.reset(token)
        cost = pricing.summarize(_RUN_USAGE.pop(key, []))
        usd = float(cost.get("usd") or 0.0)
        tokens = int((cost.get("tokens") or {}).get("total_tokens", 0) or 0)
        store = bat_loop.get_store()
        if store is not None and (usd or tokens):
            await store.bump_cost(run_id, usd, tokens)


# ── credential / owner resolution ──────────────────────────────────────

def _creds_for(tiers: dict | None, prefer: tuple[str, ...] = ("reason", "fast")) -> dict | None:
    from captain_claw.flight_deck.basna_routes import _effective_key
    tiers = tiers or {}

    def build(lt: dict) -> dict:
        provider = lt.get("provider", "anthropic")
        return {"provider": provider, "model": lt.get("model", ""),
                "base_url": lt.get("base_url") or None,
                "api_key": _effective_key(provider, lt.get("api_key") or "", lt.get("base_url")),
                "output_ctx": int(lt.get("output_ctx") or 0)}

    for name in prefer:
        lt = tiers.get(name)
        if lt and lt.get("model"):
            return build(lt)
    for lt in tiers.values():
        if isinstance(lt, dict) and lt.get("model"):
            return build(lt)
    return None


def _panel_creds(tiers: dict | None) -> tuple[dict | None, bool]:
    """Who reviews: the best model that isn't the one workers run on, so nobody
    grades their own work. Returns the credentials, and whether it's a different
    model (a deck with one model has no choice)."""
    tiers = tiers or {}
    worker_model = (tiers.get(_pick_tier(tiers)) or {}).get("model", "")
    for name in ("reason", "balanced", "longctx", "fast"):
        model = (tiers.get(name) or {}).get("model", "")
        if model and model != worker_model:
            return _creds_for(tiers, prefer=(name,)), True
    return _creds_for(tiers), False


def _store():
    s = bat_loop.get_store()
    if s is None:
        raise HTTPException(503, "Bat is not enabled on this deck (FD_BAT_DISABLED).")
    return s


# ── the Bat worker spawn (mirrors vatra_routes._spawn_worker) ──────────

def _bat_env(sid: str, step_key: str, owner: str) -> list[dict]:
    project = f"bat-{sid[:8]}"
    return [
        {"key": "CLAW_BAT_SESSION", "value": sid},
        {"key": "CLAW_BAT_SUBTASK", "value": step_key},
        {"key": "CLAW_BAT_OWNER", "value": owner},
        {"key": "CLAW_VFS_PROJECT", "value": project},
    ]


async def _spawn_bat_worker(request, user, *, name: str, tier: str,
                            tiers: dict | None, env_vars: list[dict] | None,
                            extra_env: list[dict]) -> dict:
    from captain_claw.flight_deck.server import AgentConfig, _load_process_registry, spawn_process
    lt = (tiers or {}).get(tier) or {}
    provider = lt.get("provider") or ""
    model = lt.get("model") or ""
    key = lt.get("api_key") or ""
    base_url = lt.get("base_url") or ""
    max_tokens = int(lt.get("output_ctx") or 0) or 32768
    max_context = int(lt.get("input_ctx") or 0)
    worker_tools = _bat_worker_tools(None, AgentConfig().tools)
    worker_env = (env_vars or []) + extra_env + [{"key": _WORKER_MARKER, "value": "1"}]
    base = dict(name=name, description="Bat worker", cognitive_mode="neutra",
                tools=worker_tools, env_vars=worker_env, web_enabled=True, web_port=0)
    if model:
        cfg = AgentConfig(**base, tier="", provider=provider or "", model=model,
                          provider_api_key=key, base_url=base_url,
                          max_tokens=max_tokens, max_context=max_context)
    else:
        cfg = AgentConfig(**base, tier=tier, provider_api_key=key)
    res = await spawn_process(cfg, request, user)
    reg = _load_process_registry()
    entry = reg.get(res.slug) or {}
    port = entry.get("web_port")
    if not res.ok or not port:
        # A failed spawn still left an entry: it would count against the agent limit
        # and come back to life on the deck's next start.
        _teardown([res.slug])
        return {"ok": False, "slug": res.slug, "port": 0, "auth": "", "message": res.message or "no port"}
    return {"ok": True, "slug": res.slug, "port": port, "auth": entry.get("web_auth", ""), "message": ""}


async def _spawn_archetype_bat_worker(stub, user, sid, step, owner, arch_id, tiers, env_vars, bat_env):
    """Spawn a Bat worker from a planner-assigned archetype (its tools/tier/
    persona), stamped CLAW_BAT_WORKER so the floor + ask_human/spend apply.
    Returns (port, auth, slug) or (None, None, None) on failure.

    NOTE: the archetype spawn names the slug `dubina-<arch>-…`, so the email
    send-audit attribution (which matches a `bat-<sid8>` slug) won't tie a send
    to the run in archetype mode — a minor gap for the archetype+email corner."""
    from captain_claw.flight_deck import dubina_agents
    from captain_claw.flight_deck.archetypes import merged_archetypes
    from captain_claw.flight_deck.auth import get_db
    try:
        arch = next((a for a in await merged_archetypes(get_db(), owner) if a["id"] == arch_id), None)
        if not arch:
            return (None, None, None)
        arch_tier = arch.get("tier") or _pick_tier(tiers)
        lt = (tiers or {}).get(arch_tier) or {}
        tcfg = ({"provider": lt.get("provider"), "model": lt.get("model"),
                 "api_key": lt.get("api_key"), "base_url": lt.get("base_url")}
                if lt.get("model") else {})
        env = list(env_vars or []) + list(bat_env) + [{"key": _WORKER_MARKER, "value": "1"}]
        port, token, slug = await dubina_agents.spawn_archetype_agent(
            arch, arch_tier, tcfg, stub, user,
            name_suffix=f"{sid[:8]}-{step['step_key']}", env_vars=env)
        return (port, token, slug)
    except Exception as e:  # noqa: BLE001
        log.warning("bat archetype worker spawn failed", arch=arch_id, error=str(e))
        return (None, None, None)


def _teardown(slugs: list[str]) -> None:
    from captain_claw.flight_deck.server import (
        DATA_DIR, _do_stop_process, _load_process_registry, _processes, _save_process_registry,
    )
    import shutil
    for slug in slugs:
        try:
            _do_stop_process(slug)
        except Exception as e:  # noqa: BLE001
            log.warning("bat teardown stop failed", slug=slug, error=str(e))
    if not slugs:
        return
    try:
        reg = _load_process_registry()
        for slug in slugs:
            reg.pop(slug, None)
            _processes.pop(slug, None)
            try:
                shutil.rmtree(DATA_DIR / slug, ignore_errors=True)
            except Exception:
                pass
        _save_process_registry(reg)
    except Exception as e:  # noqa: BLE001
        log.warning("bat teardown registry update failed", error=str(e))


def step_cost(entries: list[dict], worker_model: str) -> dict:
    """What one step's turn cost. The agent doesn't always say which model answered;
    it is the worker's own, so it is priced as that rather than not at all (the live
    watch already does the same, so the run's total would otherwise drop what the
    budget had seen)."""
    from captain_claw.flight_deck import pricing
    return pricing.summarize([{**e, "model": e.get("model") or worker_model} for e in entries])


def _pick_tier(tiers: dict | None, prefer: tuple[str, ...] = ("reason", "fast")) -> str:
    tiers = tiers or {}
    for name in prefer:
        if tiers.get(name):
            return name
    return next(iter(tiers), "")


# ── the three handlers (registered into bat_loop at import) ────────────

async def _bat_planner(run: dict) -> list[dict]:
    async with _metered(run["id"], "plan"):
        try:
            from captain_claw.flight_deck.basna_routes import _load_owner_tiers
            from captain_claw.flight_deck.auth import get_db
            db = get_db()
            tiers, _ = await _load_owner_tiers(db, run["owner_id"])
            user = await db.get_user_by_id(run["owner_id"])
            await _ensure_contract(run, _creds_for(tiers),
                                   _owner_has_shell(user) and not (run.get("config") or {}).get("rehearsal"))
        except Exception as e:  # noqa: BLE001
            log.warning("bat contract step failed", run_id=run.get("id"), error=str(e))
        store = bat_loop.get_store()
        fresh = await store.get_run(run["id"]) if store is not None else None
        return await _plan_steps(fresh or run)


async def _plan_steps(run: dict) -> list[dict]:
    cfg = run.get("config") or {}
    explicit = cfg.get("steps") or []
    if explicit:
        return [{"step_key": f"step-{i + 1}", "title": str(s)[:400], "seq": i}
                for i, s in enumerate(explicit) if str(s).strip()][:12]
    archetype_mode = (cfg.get("worker_mode") or "plain") == "archetype"
    # Best-effort LLM decomposition; any failure → a single stubborn step.
    try:
        from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.llm import Message
        db = get_db()
        tiers, _ = await _load_owner_tiers(db, run["owner_id"])
        creds = _creds_for(tiers)
        if creds and archetype_mode:
            steps = await _plan_with_archetypes(run, db, creds)
            if steps:
                return steps
            # no usable archetype plan → fall through to the plain decomposition
        if creds:
            prov, mt = _provider_call(creds, temperature=0.2, default_max=4096, cap=16384)
            resp = await prov.complete(messages=[
                Message(role="system", content=(
                    "Break the user's goal into the smallest ordered list of concrete, independently "
                    "checkable steps that together finish it. Return ONLY a JSON array (max 8) of objects "
                    '{"step": "<short step>", "after": [<1-based numbers of the earlier steps it needs>]}. '
                    'A step that needs none gets "after": []. Steps that don\'t need each other run at the '
                    "same time, so say so wherever order doesn't matter, and keep such steps to separate "
                    "files. No prose.")),
                Message(role="user", content=run.get("task", "")),
            ], temperature=0.2, max_tokens=mt)
            steps = _parse_plan(resp.content or "", run.get("task", ""))
            if steps:
                return steps
    except Exception as e:  # noqa: BLE001
        log.warning("bat planner fell back to single step", run_id=run.get("id"), error=str(e))
    task = run.get("task", "")
    return [{"step_key": "main", "title": task[:400] or "complete the task", "seq": 0}]


async def _plan_with_archetypes(run: dict, db, creds: dict) -> list[dict]:
    """Archetype-mode decomposition: assign the best-fit archetype to each step
    and persist the step→archetype map on the run config. Returns the steps (an
    empty list → the caller falls back to plain decomposition)."""
    from captain_claw.flight_deck.archetypes import merged_archetypes
    from captain_claw.flight_deck.basna_routes import _provider_call
    from captain_claw.llm import Message
    archetypes = await merged_archetypes(db, run["owner_id"])
    if not archetypes:
        return []
    valid = {a["id"] for a in archetypes}
    catalog = "\n".join(f"- {a['id']}: {a.get('role', '')} — {str(a.get('description', ''))[:120]}"
                        for a in archetypes[:40])
    prov, mt = _provider_call(creds, temperature=0.2, default_max=4096, cap=16384)
    resp = await prov.complete(messages=[
        Message(role="system", content=(
            "Break the user's goal into the smallest ordered list of concrete, independently checkable "
            "steps. For EACH step pick the best-fit agent from the catalog by its exact id. Return ONLY "
            "a JSON array of objects {\"step\": \"...\", \"archetype\": \"<id from the catalog>\", "
            "\"after\": [<1-based numbers of the earlier steps it needs>]}, max 8, no prose. A step that "
            "needs none gets \"after\": []; steps that don't need each other run at the same time, so keep "
            "those to separate files.\n\n## Agent catalog\n" + catalog)),
        Message(role="user", content=run.get("task", "")),
    ], temperature=0.2, max_tokens=mt)
    steps, arch_map = _parse_archetype_plan(resp.content or "", valid)
    if not steps:
        return []
    store = bat_loop.get_store()
    if store is not None and arch_map:
        cfg = dict(run.get("config") or {})
        cfg["step_archetypes"] = arch_map
        await store.set_config(run["id"], cfg)
        await store.append_event(run["id"], "note", f"{len(arch_map)} step(s) assigned to archetypes")
    return steps


async def _bat_attempt(run: dict, step: dict) -> dict:
    from captain_claw.flight_deck.basna_routes import (
        _RUN_USAGE, _dispatch_one, _finalize_usage, _load_owner_tiers, _run_sid,
    )
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck import pricing

    sid = run["id"]
    owner = run["owner_id"]
    key = step["step_key"]
    db = get_db()
    user = await db.get_user_by_id(owner) or {"id": owner}
    tiers, env_vars = await _load_owner_tiers(db, owner)
    tier = _pick_tier(tiers)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
    bat_env = _bat_env(sid, key, owner)
    rehearsal = str((run.get("config") or {}).get("rehearsal") or "")
    if rehearsal:
        from captain_claw.flight_deck import captain_swarm_rehearsal
        if captain_swarm_rehearsal.parse_tag(rehearsal) is None:
            return {"ok": False, "error": "a rehearsal run with no valid tag", "spawn_refused": True}
        # Every turn this worker takes is a rehearsal: nothing in the world changes.
        bat_env.append({"key": "CLAW_REHEARSAL", "value": rehearsal})
    # Hybrid worker mode: 'archetype' boots the planner-assigned specialist for
    # this step (its tools/tier/persona), still stamped CLAW_BAT_WORKER so the
    # hard floor + the ask_human/spend tools apply; 'plain' (default) boots a
    # generic full-toolset worker.
    mode = (run.get("config") or {}).get("worker_mode") or "plain"
    arch_id = ((run.get("config") or {}).get("step_archetypes") or {}).get(key)
    if mode == "archetype" and arch_id:
        port, auth, slug = await _spawn_archetype_bat_worker(
            stub, user, sid, step, owner, arch_id, tiers, env_vars, bat_env)
        if not slug:
            return {"ok": False, "error": f"could not spawn archetype worker for {arch_id}", "spawn_refused": True}
    else:
        name = f"bat-{sid[:8]}-{key}"[:60]
        try:
            worker = await _spawn_bat_worker(
                stub, user, name=name, tier=tier, tiers=tiers, env_vars=env_vars, extra_env=bat_env)
        except HTTPException as e:  # a plan's agent or spawn limit
            return {"ok": False, "error": f"could not spawn worker: {e.detail}", "spawn_refused": True}
        if not worker["ok"]:
            return {"ok": False, "error": f"could not spawn worker: {worker['message']}", "spawn_refused": True}
        port, auth, slug = worker["port"], worker["auth"], worker["slug"]

    store = bat_loop.get_store()
    bat_nested.worker_started(sid, key, port=port, slug=slug)
    worker_model = (tiers.get(tier) or {}).get("model", "") if tiers else ""
    # What this turn has spent so far: the agent reports it after every model call,
    # so the budget watch sees a long turn while it's happening.
    so_far: dict = {}

    def _spent() -> dict:
        return pricing.summarize([{"model": so_far.get("model") or worker_model, "usage": _finalize_usage(so_far)}])

    def _on_usage(_pt: int, _ct: int, _tt: int) -> None:
        bat_nested.set_live(sid, key, float(_spent().get("usd") or 0.0))

    def _on_action(ev: dict) -> None:
        # _send_chat_and_collect calls this SYNCHRONOUSLY, so it must not be a
        # coroutine — schedule the (async) event write as a background task.
        if store is None:
            return
        try:
            tool = str(ev.get("tool", ""))
            stage = "narration" if tool == "narration" else "action"
            asyncio.get_running_loop().create_task(
                store.append_event(sid, stage, tool[:60],
                                   agent=key, detail=str(ev.get("detail", ""))[:300]))
        except Exception:
            pass

    usage_key = f"{sid}:{key}:{uuid.uuid4().hex[:6]}"
    token = _run_sid.set(usage_key)
    _RUN_USAGE[usage_key] = []
    try:
        prior = await store.list_steps(sid) if store else []
        # A human answer provided while this step was awaiting_human: a secret
        # lives only in memory (consumed once), a non-secret was stashed in config.
        human_input = _SECRET_ANSWERS.pop((sid, key), "") or \
            ((run.get("config") or {}).get("answers") or {}).get(key, "")
        prompt = _build_step_prompt(run, step, prior, human_input=human_input)
        # Mail-write / organize grants come only from the owner's approval of
        # THIS run's plan (see _bat_automation).
        automation = _bat_automation(run)
        res = await _dispatch_one(
            port, auth, prompt, _STEP_TIMEOUT_S,
            on_action=_on_action, on_usage=_on_usage, agent_name=key, automation=automation,
            usage_sink=so_far,
        )
        cost = step_cost(_RUN_USAGE.get(usage_key, []), worker_model)
        tokens = int((cost.get("tokens") or {}).get("total_tokens", 0) or 0)
        if tokens and cost.get("usd") is None and store and float(run.get("llm_usd_cap") or 0) > 0:
            model = str(res.get("model") or worker_model)
            if (sid, model) not in _UNPRICED_NOTED:
                _UNPRICED_NOTED.add((sid, model))
                await store.append_event(sid, "note", f"No price is known for {model}, so the budget can't see its spend.")
        ok = bool(res.get("ok")) and not res.get("timed_out")
        return {
            "ok": ok,
            "output": res.get("output", ""),
            "usd": float(cost.get("usd") or 0.0),
            "tokens": tokens,
            "error": res.get("error", "") or ("timed out" if res.get("timed_out") else ""),
        }
    except asyncio.CancelledError:
        # Stopped mid-turn (a cancel or a spent budget): what it spent is still spent.
        spent = _spent()
        if store is not None:
            await store.bump_cost(sid, float(spent.get("usd") or 0.0),
                                  int((spent.get("tokens") or {}).get("total_tokens", 0) or 0))
        raise
    finally:
        _RUN_USAGE.pop(usage_key, None)
        _run_sid.reset(token)
        bat_nested.worker_ended(sid, key, port=port, slug=slug)
        _teardown([slug])


async def _bat_judge(run: dict, steps: list[dict]) -> dict:
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.llm import Message

    sid = run["id"]
    cfg = run.get("config") or {}
    if cfg.get("tie_break") == "accepted":
        return {"done": True, "overridden": True, "failing": [],
                "reason": f"you accepted it anyway. They said: {cfg.get('tie_break_findings') or 'not done'}"}
    deliverable = _final_deliverable(steps)
    finished = bool(steps) and all(s.get("status") in ("done", "skipped") for s in steps)
    failed = [s for s in steps if s.get("status") == "failed"]
    # Why they failed, not only which: this is what the next round's planner and
    # the person reading the stop reason have to go on.
    det = [Check("all planned steps completed", passed=finished, critical=True,
                 detail=("failed: " + "; ".join(f"{s['step_key']} ({str(s.get('error') or '')[:300]})" for s in failed)
                         if failed else "pending steps remain"))]
    db = get_db()
    # A rehearsal runs nothing on the deck either: its command checks go to the reviewers.
    has_shell = (_owner_has_shell(await db.get_user_by_id(run["owner_id"]))
                 and not (run.get("config") or {}).get("rehearsal"))
    open_rules: list[dict] = []
    evidence = ""
    store = bat_loop.get_store()
    try:
        tiers, _ = await _load_owner_tiers(db, run["owner_id"])
        creds, independent = _panel_creds(tiers)
    except Exception as e:  # noqa: BLE001
        log.warning("bat judge tiers failed; deterministic-only", run_id=sid, error=str(e))
        creds, independent = None, False
    if finished:
        evidence = await asyncio.to_thread(_evidence_block, _run_folder(run))
        try:
            more, open_rules, script_failed = await _contract_checks(run, has_shell)
            cleared = await _audit_failed_checks(run, script_failed, creds, _run_folder(run)) if script_failed and creds else {}
            for rule in script_failed:
                why = cleared.get(str(rule["id"]))
                if why is None:
                    continue
                more = [c for c in more if c.name != f"acceptance: {rule['text']}"]
                open_rules.append(rule)
                # For the rest of the run too: the same check would only fail again.
                cfg = {**cfg, "contract": [{**r, "check": {"type": "judge"}} if str(r["id"]) == str(rule["id"]) else r
                                           for r in cfg.get("contract") or []]}
                if store is not None:
                    await store.set_config(sid, cfg)
                    await store.append_event(sid, "note", f"The check for \"{rule['text']}\" was at fault ({why}). "
                                             "Reviewers decide that rule instead.")
            det += more
        except Exception as e:  # noqa: BLE001 — a broken check goes to the reviewers
            log.warning("bat contract checks failed", run_id=sid, error=str(e))
            open_rules = list(cfg.get("contract") or [])
    # The panel may hold out for MORE work only while more work is possible; once
    # there are no rounds left, a deterministically-complete deliverable must not
    # be blocked by a picky panel.
    work_remaining = (bat_loop._REPLANNER is not None
                      and int(cfg.get("rounds") or 0) + 1 < bat_loop._DEFAULT_MAX_ROUNDS)

    panel_fn = None
    try:
        if store is not None and creds and not independent and not cfg.get("panel_note"):
            await store.append_event(sid, "note", "Reviewers run on the same model as the workers: "
                                     "this deck has no other model set up.")
            cfg = {**cfg, "panel_note": True}
            await store.set_config(sid, cfg)
        # Layer-3 evidence: if this run could send mail, hand the panel the
        # ground-truth send audit so it judges honesty against reality, not the
        # worker's claim.
        receipt = ""
        if cfg.get("email_allowed"):
            try:
                sends = await db.list_gmail_sends(run["owner_id"], limit=50, origin="agent")
                receipt = _email_receipt_block(sends, sid[:8])
            except Exception:
                receipt = ""
        rules = ("## Acceptance rules: every one must hold\n" + "\n".join(f"- {r['text']}" for r in open_rules)
                 if open_rules else "")
        to_judge = deliverable or ("" if _NO_FILES in evidence else evidence)
        if creds and to_judge.strip():
            lenses = ["correctness and completeness",
                      "whether it ACTUALLY did the task vs merely described it",
                      "edge cases, honesty, and unverified claims"]

            async def vote(deliverable_text: str, task: str, idx: int) -> dict:
                lens = lenses[idx % len(lenses)]
                shown = "\n\n".join(p for p in (receipt, rules, deliverable_text[:8000], evidence) if p)
                prov, mt = _provider_call(creds, temperature=0.0, default_max=2048, cap=8192)
                resp = await prov.complete(messages=[
                    Message(role="system", content=_VOTE_SYSTEM),
                    Message(role="user", content=_vote_prompt(lens, task, shown)),
                ], temperature=0.0, max_tokens=mt)
                return _parse_vote(resp.content or "")

            panel_fn = vote
    except Exception as e:  # noqa: BLE001
        log.warning("bat judge panel setup failed; deterministic-only", run_id=sid, error=str(e))

    prior = int(cfg.get("veto_rounds") or 0)
    async with _metered(sid, "judge"):
        verdict = await _judge_evaluate(
            task=run.get("task", ""), deliverable=deliverable or ("" if _NO_FILES in evidence else evidence),
            deterministic=det,
            panel_vote_fn=panel_fn, panel_size=_PANEL_SIZE,
            prior_model_veto_rounds=prior, max_model_veto_rounds=_MAX_MODEL_VETO_ROUNDS,
            work_remaining=work_remaining,
        )
    # Track the model-veto streak for the anti-deadlock override. Kept on the run,
    # so a restart can't reset it.
    if store is not None and not verdict.done and not verdict.det_criticals \
            and verdict.panel.get("verdict") in ("disagree", "tie", "no_quorum"):
        await store.set_config(sid, {**cfg, "veto_rounds": prior + 1})
    now = await store.get_run(sid) if store is not None else None
    # A Captain Swarm rehearsal doesn't ask: its run checks every result's page
    # itself, so the reviewers' objection rides along with the result instead.
    if now and now["status"] != "cancelled" and verdict.overridden and not cfg.get("rehearsal"):
        findings = _findings(verdict.to_dict())
        await store.set_config(sid, {**cfg, "tie_break_findings": findings[:1500]})
        await human_ask.raise_ask(
            store, run_id=sid, owner=run["owner_id"], kind="tie_break",
            question=("The checks pass, but the reviewers won't pass this run, and more fix rounds aren't "
                      f"settling it.\n\nWhat they said:\n{findings}"),
            options=[_TIE_BREAK_ACCEPT, _TIE_BREAK_STOP])
        await store.set_status(sid, "awaiting_human")
        await store.append_event(sid, "awaiting_human", "the reviewers won't pass it: asking you to decide")
        return {**verdict.to_dict(), "done": False, "parked": True, "failing": []}
    # What "the same failure again" means: a step's error, since fix steps are new
    # steps with new names, and a check's name, since its details change every run.
    failing = sorted({f"step failed: {str(s.get('error') or '')[:80]}" for s in failed})
    failing += [c.name for c in det[1:] if c.critical and not c.passed]
    if not finished and not failed:
        failing.append(det[0].name)
    return {**verdict.to_dict(), "failing": failing}


_REPLAN_SYSTEM = (
    "A team's work on a goal was checked and it isn't done yet. Write the smallest set of NEW steps that "
    "fix exactly what the check found. Don't repeat a step that already finished. If an approach failed, "
    "the new step must take a different one. Each step goes to a fresh worker who can see the team's shared "
    f"folder and earlier results. Return ONLY a JSON array, at most {_MAX_FIX_STEPS} items: "
    '{"step": "<what to do>", "after": [<1-based numbers of the new steps it must wait for>]}. Steps that '
    "don't wait for each other run at the same time, so keep those to separate files. No prose."
)


async def _bat_replan(run: dict, steps: list[dict], verdict: dict, round_no: int) -> list[dict]:
    """A refusal is new work. If the model can't plan it, the findings themselves
    are the step: Bat doesn't give up because a planning call failed."""
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.llm import Message

    findings = _findings(verdict)
    plan: list[dict] = []
    try:
        tiers, _ = await _load_owner_tiers(get_db(), run["owner_id"])
        creds = _creds_for(tiers)
        if creds:
            so_far = "\n".join(
                f"- [{s.get('status')}] {s.get('title') or s.get('step_key')}"
                + (f": {(s.get('error') or s.get('output') or '').strip()[:300]}"
                   if (s.get("error") or s.get("output")) else "")
                for s in steps)
            async with _metered(run["id"], f"replan-{round_no}"):
                prov, mt = _provider_call(creds, temperature=0.2, default_max=4096, cap=16384)
                resp = await prov.complete(messages=[
                    Message(role="system", content=_REPLAN_SYSTEM),
                    Message(role="user", content=(f"# Goal\n{str(run.get('task', ''))[:4000]}\n\n"
                                                  f"# The work so far\n{so_far[:8000]}\n\n"
                                                  f"# What the check found\n{findings}")),
                ], temperature=0.2, max_tokens=mt)
            plan = _parse_fix_plan(resp.content or "")
    except Exception as e:  # noqa: BLE001
        log.warning("bat replan fell back to one fix step", run_id=run.get("id"), error=str(e))
    if not plan:
        plan = [{"title": "Fix what the check found", "after": []}]
    return [{**p, "brief": findings[:4000]} for p in plan]


async def _log_own_cost(run: dict) -> None:
    """Bat's line in the cost ledger: its own workers, planning and review. Runs it
    started are in the run's total for the cap, but have their own ledger lines."""
    store = bat_loop.get_store()
    try:
        from captain_claw.flight_deck.auth import get_db
        kids = await store.list_children(run["id"]) if store is not None else []
        usd = max(0.0, float(run.get("cumulative_usd") or 0.0) - sum(float(k["counted_usd"]) for k in kids))
        tokens = max(0, int(run.get("cumulative_tokens") or 0) - sum(int(k["counted_tokens"]) for k in kids))
        if usd or tokens:
            await get_db().log_run_cost(run["owner_id"], "bat", run["id"], {
                "tokens": {"total_tokens": tokens}, "usd": round(usd, 6),
                "elapsed_seconds": max(0.0, float(run.get("updated_at") or 0.0) - float(run.get("created_at") or 0.0)),
            })
    except Exception as e:  # noqa: BLE001 — cost is best-effort, never blocks the result
        log.warning("bat cost ledger failed", run_id=run.get("id"), error=str(e))


async def _bat_on_finish(run: dict) -> None:
    """Deliver the result back where the run came from (bell / channel / caller
    agent), mirroring Basna/Vatra's completion delivery."""
    sid = run["id"]
    await _log_own_cost(run)
    if (run.get("config") or {}).get("rehearsal"):
        return  # a Captain Swarm attempt: its run reads the result; nobody is told
    status = run.get("status")
    ok = status == "done"
    truth = (run.get("truth") or "").strip()
    reason = run.get("stopped_reason") or ""
    head = "✅ Bat finished" if ok else f"⚠️ Bat stopped ({status})"
    summary = f"{head}: {run.get('title') or run.get('task', '')[:60]}"
    if reason and not ok:
        summary += f"\n{reason}"
    if truth:
        summary += f"\n\n{truth[:1800]}{'…' if len(truth) > 1800 else ''}"
    origin = run.get("origin") or {}
    owner = run["owner_id"]
    kind = (origin.get("kind") or "").strip()
    address = (origin.get("address") or "").strip()
    try:
        if kind in ("", "web"):
            from captain_claw.flight_deck.auth import get_db
            await get_db().add_notification(
                owner, "run" if ok else "run_error", run.get("title") or "Bat run",
                summary, "bat", sid)
            return
        # It answers on the channel it came from; the Line keeps a copy.
        from captain_claw.flight_deck import line
        await line.post(owner, "run", run.get("title") or "Bat run", summary,
                        bell_type="run" if ok else "run_error", source=f"via {kind}",
                        ref_type="bat", ref_id=sid, trust="agent", to_phones=False)
        if address:
            from captain_claw.flight_deck import delivery_routes
            delivered = await delivery_routes.deliver_to_origin({"kind": kind, "address": address}, summary)
            if delivered:
                return
    except Exception as e:  # noqa: BLE001
        log.warning("bat delivery failed; trying source agent", run_id=sid, error=str(e))
    # Fallback: relay to the calling agent.
    try:
        from captain_claw.flight_deck.agent_notify import notify_source_agent
        await notify_source_agent(
            source_host=run.get("source_host") or "localhost",
            source_port=int(run.get("source_port") or 0),
            origin=origin, kind="Bat run", title=run.get("title") or "Bat run",
            run_ref=f"run {sid}", ok=ok, summary=summary,
            no_restart_hint="Do NOT start another Bat and ",
        )
    except Exception as e:  # noqa: BLE001
        log.warning("bat source-agent notify failed", run_id=sid, error=str(e))


# What each gate reason grants once the owner approves it.
_GRANT_OF = {"send email": "email_allowed", "spend money": "spend_allowed",
             "create an account": "account_allowed", _ORGANIZE_GATE: "organize_allowed"}


async def _gate_check(store, run: dict) -> str:
    """Human-in-the-loop gate (bat_loop seam). Returns proceed|wait|cancelled|error.

    Two jobs: (1) the START GATE — a fresh run whose plan would send mail, spend,
    create an account or organize mail pauses (awaiting_plan) for owner
    approval, and so does a fix round whose new steps go beyond what was approved;
    (2) RESUME — an awaiting_* run proceeds once its ask is answered,
    is cancelled if the plan is rejected, and errors if the ask expired."""
    from captain_claw.flight_deck import human_ask
    run_id = run["id"]
    status = run["status"]

    if status in ("awaiting_plan", "awaiting_human"):
        ask = await store.latest_ask(run_id)
        if ask is None:
            return "proceed"
        if ask["status"] == "open":
            return "wait"
        if ask["status"] == "expired":
            await store.set_status(run_id, "error", stopped_reason="human_ask_timeout")
            await store.append_event(run_id, "error", "the human did not answer in time")
            return "error"
        if ask["status"] == "cancelled":
            await store.set_status(run_id, "cancelled", stopped_reason="ask_cancelled")
            return "cancelled"
        # answered
        if ask["kind"] == "plan_approval":
            decision = (ask.get("answer") or "").strip().lower()
            if decision in _APPROVE_WORDS:
                cfg = dict(run.get("config") or {})
                cfg["plan_approved"] = True
                # Grant the run-scoped email exception ONLY now, because the
                # owner approved a plan whose gate reason included mail (Phase 5).
                # The grant lives on the run config and is never copied to a child.
                reason = cfg.get("gate_reason") or ""
                if "send email" in reason:
                    cfg["email_allowed"] = True
                if "spend money" in reason:
                    cfg["spend_allowed"] = True
                if "create an account" in reason:
                    cfg["account_allowed"] = True
                if _ORGANIZE_GATE in reason:
                    cfg["organize_allowed"] = True
                await store.set_config(run_id, cfg)
                grants = [g for g, on in (("email", cfg.get("email_allowed")),
                                          ("organize", cfg.get("organize_allowed")),
                                          ("spend", cfg.get("spend_allowed")),
                                          ("accounts", cfg.get("account_allowed"))) if on]
                await store.append_event(
                    run_id, "note",
                    "plan approved — proceeding" + (f" ({'+'.join(grants)} enabled for this run)"
                                                    if grants else ""))
                return "proceed"
            await store.set_status(run_id, "cancelled", stopped_reason="plan_rejected")
            await store.append_event(run_id, "cancelled", "plan rejected by the owner")
            return "cancelled"
        if ask["kind"] == "spend_approval":
            # step_key carries "spend:<id>"; approve/deny the ledger row, then
            # re-run the step (the worker re-authorizes and finds the decision).
            spend_id = (ask.get("step_key") or "").split("spend:", 1)[-1]
            decision = (ask.get("answer") or "").strip().lower()
            approved = decision in _APPROVE_WORDS
            if spend_id:
                await store.set_spend_status(
                    spend_id, "approved" if approved else "denied",
                    decided_by=f"human:{run['owner_id']}")
            await store.append_event(
                run_id, "note", f"spend {'approved' if approved else 'declined'} by the owner")
            return "proceed"
        if ask["kind"] == "tie_break":
            if (ask.get("answer") or "").strip() == _TIE_BREAK_STOP:
                await store.set_status(run_id, "error", truth=bat_loop._assemble_truth(await store.list_steps(run_id)),
                                       stopped_reason=f"stopped by you: the reviewers wouldn't pass it. "
                                                      f"{(run.get('config') or {}).get('tie_break_findings') or ''}".strip())
                await store.append_event(run_id, "error", "stopped by the owner: the reviewers wouldn't pass it")
                return "error"
            await store.set_config(run_id, {**(run.get("config") or {}), "tie_break": "accepted"})
            await store.append_event(run_id, "note", "accepted by the owner over the reviewers' objection")
            return "proceed"
        # input / secret answer → hand it to the resuming step
        text = await human_ask.resolve_answer_text(store, ask)
        step_key = ask.get("step_key") or ""
        if ask.get("secret"):
            if text:
                _SECRET_ANSWERS[(run_id, step_key)] = text
        else:
            cfg = dict(run.get("config") or {})
            answers = dict(cfg.get("answers") or {})
            answers[step_key] = text
            cfg["answers"] = answers
            await store.set_config(run_id, cfg)
        await store.append_event(run_id, "note", "input received — resuming", agent=step_key)
        return "proceed"

    # A fresh run — the start gate — or a fix round's new steps: approval covers the
    # plan the owner saw, so work beyond it asks again.
    cfg = run.get("config") or {}
    if cfg.get("rehearsal"):
        # A Captain Swarm rehearsal can't send, spend, sign up or organize: its
        # workers turn all of that into intents. The owner is asked when the
        # winner is carried out, not now.
        return "proceed"
    steps = await store.list_steps(run_id)
    needs, reason = plan_needs_gate(run.get("task", ""), steps)
    if not needs:
        return "proceed"
    new_asks = [g for g, flag in _GRANT_OF.items() if g in reason and not cfg.get(flag)]
    if cfg.get("plan_approved") and not new_asks:
        return "proceed"
    # Remember what was gated so approval can grant the matching capability
    # (e.g. 'send email' → the run-scoped email exception), and for organizing
    # the plan text the approval covers (the workers' job_text).
    cfg = dict(run.get("config") or {})
    cfg["gate_reason"] = reason
    if _ORGANIZE_GATE in reason:
        cfg["organize_text"] = _organize_text(run.get("task", ""), steps)
    await store.set_config(run_id, cfg)
    existing = await store.latest_open_ask(run_id)
    if not (existing and existing["kind"] == "plan_approval"):
        q = (f"This Bat run would {', '.join(new_asks) or reason}. Approve it to run, or cancel.\n\nPlan:\n"
             + "\n".join(f"- {s.get('title')}" for s in steps if s.get("status") != "skipped"))
        checks = cfg.get("contract") or []
        if checks:
            q += "\n\nIt's done when:\n" + "\n".join(f"- {r['text']}" for r in checks)
        await human_ask.raise_ask(store, run_id=run_id, owner=run["owner_id"],
                                  kind="plan_approval", question=q, options=["approve", "cancel"])
    await store.set_status(run_id, "awaiting_plan")
    await store.append_event(run_id, "awaiting_plan", f"needs approval — would {reason}")
    return "wait"


async def _notify_ask(ask: dict) -> None:
    """Fan an ask out to the owner: the in-app bell always; a WhatsApp nudge for
    a non-secret ask. (Secret asks are UI-only — never put a request for a code
    or password on a chat channel.) The owner answers via the authoritative
    answer route / the Bat page."""
    owner = ask.get("owner_id", "")
    question = ask.get("question", "")
    try:
        from captain_claw.flight_deck.auth import get_db
        await get_db().add_notification(
            owner, "bat_ask", "Bat needs you", question, "bat", ask.get("run_id", ""))
    except Exception as e:  # noqa: BLE001
        log.warning("bat ask bell failed", error=str(e))
    if ask.get("secret"):
        return
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        from captain_claw.flight_deck.fd_dispatch import _nudge_waids
        from captain_claw.flight_deck.whatsapp_bridge import send_text_checked
        waids, _ = _nudge_waids(resolve_config(owner))
        for waid in waids:
            await send_text_checked(waid, f"🦇 Bat needs you: {question[:600]}\n\n(Answer in the Bat page.)")
    except Exception as e:  # noqa: BLE001
        log.warning("bat ask whatsapp nudge failed", error=str(e))


# Register the real handlers so the supervisor drives runs (handlers_ready()).
bat_loop.set_planner(_bat_planner)
bat_loop.set_attempt_runner(_bat_attempt)
bat_loop.set_judge(_bat_judge)
bat_loop.set_replanner(_bat_replan)
bat_loop.set_on_finish(_bat_on_finish)
bat_loop.set_gate_check(_gate_check)
human_ask.set_notifier(_notify_ask)


# ── routes ─────────────────────────────────────────────────────────────

def _resolve(body: _BatAgentReq) -> str:
    from captain_claw.flight_deck.basna_routes import _AgentReq, _resolve_owner
    return _resolve_owner(_AgentReq(source_port=body.source_port, web_auth=body.web_auth,
                                    owner_id=body.owner_id))


@router.post("/agent/start")
async def agent_start(body: BatStartReq):
    owner = _resolve(body)
    task = (body.task or "").strip()
    if not task:
        raise HTTPException(400, "task is required")
    store = _store()
    active = sum(1 for r in await store.list_runs(owner, limit=100) if r["status"] in RUNNING_STATES)
    if active >= _BAT_MAX_RUNS_PER_OWNER:
        return {"status": "rejected",
                "reason": f"You already have {active} Bat run(s) in progress (limit {_BAT_MAX_RUNS_PER_OWNER})."}
    run_id = f"bat_{uuid.uuid4().hex[:12]}"
    title = (body.title or task[:60]).strip()
    await store.create_run(
        run_id=run_id, owner_id=owner, title=title, task=task,
        config={"source": "agent", "origin_platform": body.origin_platform,
                "steps": [str(s) for s in (body.steps or []) if str(s).strip()],
                "real_usd_cap": float(body.real_usd_cap or 0.0),
                "per_item_usd": float(body.per_item_usd or 0.0),
                "worker_mode": _norm_worker_mode(body.worker_mode)},
        origin={"platform": body.origin_platform, "user_id": body.origin_user_id,
                "chat_id": body.origin_chat_id, "kind": body.origin_kind,
                "address": body.origin_address},
        source_host=body.source_host or "localhost", source_port=int(body.source_port or 0),
        llm_usd_cap=_llm_cap(body.llm_usd_cap), status="planning",
    )
    await store.append_event(run_id, "note", "run created")
    try:
        await bat_loop.kick()  # start now instead of waiting for the periodic tick
    except Exception as e:  # noqa: BLE001
        log.warning("bat kick failed (will start on next tick)", run_id=run_id, error=str(e))
    return {"status": "running", "run_id": run_id, "title": title}


@router.post("/agent/status")
async def agent_status(body: _BatAgentReq):
    owner = _resolve(body)
    run = await _store().get_run(body.run_id)
    if not run or run["owner_id"] != owner:
        return {"found": False}
    steps = await _store().list_steps(body.run_id)
    return {
        "found": True, "status": run["status"],
        "total_steps": len(steps),
        "done_steps": sum(1 for s in steps if s["status"] == "done"),
        "cumulative_usd": run.get("cumulative_usd", 0.0),
        "llm_usd_cap": run.get("llm_usd_cap", 0.0),
        "stopped_reason": run.get("stopped_reason", ""),
    }


@router.post("/agent/get")
async def agent_get(body: _BatAgentReq):
    owner = _resolve(body)
    run = await _store().get_run(body.run_id)
    if not run or run["owner_id"] != owner:
        return {"found": False}
    steps = await _store().list_steps(body.run_id)
    events = await _store().list_events(body.run_id, limit=60)
    return {"found": True, "run": run, "steps": steps, "events": events[-60:]}


@router.post("/agent/list")
async def agent_list(body: _BatAgentReq):
    owner = _resolve(body)
    runs = await _store().list_runs(owner, limit=body.limit)
    q = (body.query or "").strip().lower()
    if q:
        runs = [r for r in runs if q in (r.get("title", "") + " " + r.get("task", "")).lower()]
    return {"runs": [{"id": r["id"], "title": r.get("title", ""), "task": r.get("task", ""),
                      "status": r["status"]} for r in runs]}


@router.post("/agent/cancel")
async def agent_cancel(body: _BatAgentReq):
    owner = _resolve(body)
    run = await _store().get_run(body.run_id)
    if not run or run["owner_id"] != owner:
        return {"ok": False}
    ok = await bat_loop.cancel_run(_store(), body.run_id)
    return {"ok": ok}


# ── capped real-money spend (Phase 6) — agent/worker endpoints ─────────

class _SpendReq(_BatAgentReq):
    session_id: str = ""              # the Bat run (worker's CLAW_BAT_SESSION)
    agent: str = ""
    merchant: str = ""
    merchant_domain: str = ""
    amount_usd: float = 0.0
    description: str = ""
    spend_id: str = ""
    actual_usd: float = 0.0
    order_ref: str = ""


class _AskReq(_BatAgentReq):
    session_id: str = ""              # the Bat run (worker's CLAW_BAT_SESSION)
    step_key: str = ""                # the worker's CLAW_BAT_SUBTASK
    kind: str = "input"               # input | secret
    question: str = ""
    options: list[str] = Field(default_factory=list)


async def _owner_run(body) -> tuple[str, dict]:
    """Resolve the calling worker's owner and load its run (body.session_id =
    the worker's CLAW_BAT_SESSION), refusing a mismatch."""
    owner = _resolve(body)
    run = await _store().get_run(body.session_id)
    if not run or run["owner_id"] != owner:
        raise HTTPException(404, "run not found")
    return owner, run


@router.post("/agent/spend/authorize")
async def spend_authorize(body: _SpendReq):
    owner, run = await _owner_run(body)
    store = _store()
    sid = body.session_id
    domain = (body.merchant_domain or body.merchant or "").strip().lower()
    amount = float(body.amount_usd or 0.0)
    # Idempotent: a prior decision for the same merchant+amount is returned as-is
    # (approved → proceed, denied → stop, requested → still pending). Prevents a
    # re-declare after resume from raising a fresh approval ask in a loop.
    existing = await store.find_live_spend(sid, domain, amount)
    if existing:
        return {"status": existing["status"], "id": existing["id"], "reason": "existing authorization"}
    policy = _spend_policy(run)
    committed = await store.committed_usd(sid)
    decision, reason = spend_decision(policy, committed, amount)
    spend_id = f"spend_{uuid.uuid4().hex[:12]}"
    common = dict(spend_id=spend_id, run_id=sid, owner_id=owner, agent=body.agent or "",
                  merchant=body.merchant, merchant_domain=domain,
                  description=body.description, amount_usd_max=amount)
    if decision == "denied":
        await store.create_spend(**common, status="denied", decided_by="policy")
        await store.append_event(sid, "note", f"spend denied (${amount:.2f} at {domain}): {reason}")
        return {"status": "denied", "id": spend_id, "reason": reason}
    if decision == "approved":
        await store.create_spend(**common, status="approved", decided_by="policy")
        await store.append_event(sid, "budget", f"spend auto-approved ${amount:.2f} at {domain}")
        return {"status": "approved", "id": spend_id, "reason": reason}
    # needs_human → record a reservation + raise a spend-approval ask; the run
    # stands down until the owner answers.
    await store.create_spend(**common, status="requested", decided_by="")
    await human_ask.raise_ask(
        store, run_id=sid, owner=owner, kind="spend_approval",
        question=(f"Bat wants to spend ${amount:.2f} at {body.merchant or domain}"
                  + (f" — {body.description}" if body.description else "") + ". Approve or cancel."),
        options=["approve", "cancel"], step_key=f"spend:{spend_id}")
    await store.set_status(sid, "awaiting_human")
    await store.append_event(sid, "awaiting_human", f"spend needs approval: ${amount:.2f} at {domain}")
    return {"status": "requested", "id": spend_id, "reason": reason}


@router.post("/agent/spend/settle")
async def spend_settle(body: _SpendReq):
    owner, run = await _owner_run(body)
    store = _store()
    row = await store.get_spend(body.spend_id)
    if not row or row["owner_id"] != owner or row["run_id"] != body.session_id:
        raise HTTPException(404, "authorization not found")
    if row["status"] != "approved":
        return {"ok": False, "reason": f"not an approved authorization (status={row['status']})"}
    actual = float(body.actual_usd or 0.0) or row["amount_usd_max"]
    await store.set_spend_status(body.spend_id, "consumed", actual_usd=actual,
                                 order_ref=body.order_ref,
                                 evidence={"order_ref": body.order_ref})
    await store.append_event(body.session_id, "budget",
                             f"spend settled ${actual:.2f} at {row['merchant_domain']}"
                             + (f" (order {body.order_ref})" if body.order_ref else ""))
    return {"ok": True}


@router.post("/agent/spend/void")
async def spend_void(body: _SpendReq):
    owner, run = await _owner_run(body)
    store = _store()
    row = await store.get_spend(body.spend_id)
    if not row or row["owner_id"] != owner:
        raise HTTPException(404, "authorization not found")
    await store.set_spend_status(body.spend_id, "voided")
    return {"ok": True}


@router.post("/agent/spend/status")
async def spend_status(body: _SpendReq):
    owner, run = await _owner_run(body)
    store = _store()
    policy = _spend_policy(run)
    return {
        "enabled": policy["enabled"],
        "committed_usd": await store.committed_usd(body.session_id),
        "run_usd_cap": policy["run_usd_cap"],
        "per_item_usd": policy["per_item_usd"],
        "items": [{"id": s["id"], "merchant": s["merchant"], "amount_usd_max": s["amount_usd_max"],
                   "actual_usd": s["actual_usd"], "status": s["status"]}
                  for s in await store.list_spend(body.session_id)],
    }


# ── worker → human escalation (2FA code, CAPTCHA, a credential) ────────

@router.post("/agent/ask")
async def agent_ask(body: _AskReq):
    """A Bat worker raises a question only the human can answer and the run
    stands down until they reply. The worker must stop after calling this."""
    owner, run = await _owner_run(body)
    kind = body.kind if body.kind in ("input", "secret") else "input"
    question = (body.question or "").strip()[:1000]
    if not question:
        raise HTTPException(400, "question is required")
    ask_id = await human_ask.raise_ask(
        _store(), run_id=body.session_id, owner=owner, kind=kind,
        question=question, options=body.options or [], step_key=body.step_key or "",
        secret=(kind == "secret"))
    await _store().set_status(body.session_id, "awaiting_human")
    await _store().append_event(
        body.session_id, "awaiting_human",
        "awaiting a private value" if kind == "secret" else question[:200],
        agent=body.step_key or "")
    return {"status": "requested", "ask_id": ask_id}


# ── owner-authenticated human-in-the-loop (answer path A) ──────────────
# Not under /fd/bat/agent/, so these use the owner's session (get_current_user),
# not the agent guard — a human answers, never the agent.

class _AnswerReq(BaseModel):
    text: str = ""


class _ApproveReq(BaseModel):
    approve: bool = True


def _ui_event(ev: dict) -> dict:
    """Flatten a stored bat_event into the frontend ProgressEvent shape (the
    extra fields live in `data`; the UI reads agent/tool/ok/… at the top level)."""
    d = ev.get("data") or {}
    return {"i": ev.get("i", 0), "ts": ev.get("ts"), "stage": ev.get("stage", ""),
            "message": ev.get("message", ""), **d}


def _norm_worker_mode(v: str) -> str:
    return "archetype" if str(v or "").strip().lower() == "archetype" else "plain"


class _UiStartReq(BaseModel):
    task: str = ""
    title: str = ""
    llm_usd_cap: float = 0.0
    real_usd_cap: float = 0.0
    per_item_usd: float = 0.0
    worker_mode: str = "plain"
    steps: list[str] = Field(default_factory=list)


def _run_summary(r: dict) -> dict:
    cfg = r.get("config") or {}
    return {"id": r["id"], "title": r.get("title", ""), "task": r.get("task", ""),
            "status": r["status"], "created_at": r.get("created_at", 0),
            "updated_at": r.get("updated_at", 0),
            "cumulative_usd": r.get("cumulative_usd", 0.0),
            "llm_usd_cap": r.get("llm_usd_cap", 0.0),
            "real_usd_cap": cfg.get("real_usd_cap", 0.0),
            "stopped_reason": r.get("stopped_reason", "")}


@router.get("/runs")
async def list_runs_ui(user: dict = Depends(get_current_user)):
    runs = await _store().list_runs(user["id"], limit=60)
    return {"runs": [_run_summary(r) for r in runs]}


@router.get("/runs/{run_id}")
async def get_run_ui(run_id: str, user: dict = Depends(get_current_user)):
    run = await _store().get_run(run_id)
    if not run or run["owner_id"] != user["id"]:
        raise HTTPException(404, "run not found")
    spend = await _store().list_spend(run_id)
    return {
        "run": {**_run_summary(run), "truth": run.get("truth", ""),
                "vfs_project": f"bat-{run_id[:8]}",
                "email_allowed": (run.get("config") or {}).get("email_allowed", False),
                "spend_allowed": (run.get("config") or {}).get("spend_allowed", False),
                "account_allowed": (run.get("config") or {}).get("account_allowed", False)},
        "steps": await _store().list_steps(run_id),
        "events": [_ui_event(e) for e in await _store().list_events(run_id, limit=1000)],
        "spend": {"items": spend, "committed_usd": await _store().committed_usd(run_id),
                  "cap": (run.get("config") or {}).get("real_usd_cap", 0.0)},
    }


@router.get("/runs/{run_id}/events")
async def get_events_ui(run_id: str, since: int = 0, user: dict = Depends(get_current_user)):
    run = await _store().get_run(run_id)
    if not run or run["owner_id"] != user["id"]:
        raise HTTPException(404, "run not found")
    return {"events": [_ui_event(e) for e in await _store().list_events(run_id, since=since)],
            "status": run["status"], "cumulative_usd": run.get("cumulative_usd", 0.0)}


@router.post("/start")
async def ui_start(body: _UiStartReq, user: dict = Depends(get_current_user)):
    """Start a Bat run from the UI (owner-authenticated), mirroring the agent
    start path but with the owner from the session."""
    owner = user["id"]
    task = (body.task or "").strip()
    if not task:
        raise HTTPException(400, "task is required")
    store = _store()
    active = sum(1 for r in await store.list_runs(owner, limit=100) if r["status"] in RUNNING_STATES)
    if active >= _BAT_MAX_RUNS_PER_OWNER:
        raise HTTPException(429, f"You already have {active} Bat run(s) in progress (limit {_BAT_MAX_RUNS_PER_OWNER}).")
    run_id = f"bat_{uuid.uuid4().hex[:12]}"
    title = (body.title or task[:60]).strip()
    await store.create_run(
        run_id=run_id, owner_id=owner, title=title, task=task,
        config={"source": "ui", "origin_platform": "web",
                "steps": [str(s) for s in (body.steps or []) if str(s).strip()],
                "real_usd_cap": float(body.real_usd_cap or 0.0),
                "per_item_usd": float(body.per_item_usd or 0.0),
                "worker_mode": _norm_worker_mode(body.worker_mode)},
        origin={"platform": "web", "kind": "", "address": ""},
        source_host="localhost", source_port=0,
        llm_usd_cap=_llm_cap(body.llm_usd_cap), status="planning")
    await store.append_event(run_id, "note", "run created from the Bat page")
    try:
        await bat_loop.kick()
    except Exception:
        pass
    return {"run_id": run_id, "title": title}


@router.get("/asks")
async def list_asks(user: dict = Depends(get_current_user)):
    """The owner's open asks (what Bat is waiting on). Secret asks show their
    question — never a secret value — so the owner knows what to enter."""
    asks = await _store().open_asks_for_owner(user["id"])
    return {"asks": [{"id": a["id"], "run_id": a["run_id"], "kind": a["kind"],
                      "question": a["question"], "options": a["options"],
                      "secret": a["secret"]} for a in asks]}


@router.post("/asks/{ask_id}/answer")
async def answer_ask(ask_id: str, body: _AnswerReq, user: dict = Depends(get_current_user)):
    store = _store()
    ask = await store.get_ask(ask_id)
    if not ask or ask["owner_id"] != user["id"]:
        raise HTTPException(404, "ask not found")
    res = await human_ask.answer(store, ask_id, body.text, via="ui")
    if not res.get("ok"):
        raise HTTPException(409, res.get("reason", "could not answer"))
    try:
        await bat_loop.kick()  # resume the waiting run now
    except Exception:
        pass
    return {"ok": True, "run_id": res["run_id"]}


@router.post("/runs/{run_id}/cancel")
async def cancel_run_ui(run_id: str, user: dict = Depends(get_current_user)):
    store = _store()
    run = await store.get_run(run_id)
    if not run or run["owner_id"] != user["id"]:
        raise HTTPException(404, "run not found")
    ok = await bat_loop.cancel_run(store, run_id)
    return {"ok": ok}


@router.post("/runs/{run_id}/approve-plan")
async def approve_plan(run_id: str, body: _ApproveReq, user: dict = Depends(get_current_user)):
    store = _store()
    run = await store.get_run(run_id)
    if not run or run["owner_id"] != user["id"]:
        raise HTTPException(404, "run not found")
    ask = await store.latest_open_ask(run_id)
    if not ask or ask["kind"] != "plan_approval":
        raise HTTPException(409, "no plan awaiting approval")
    await human_ask.answer(store, ask["id"], "approve" if body.approve else "cancel", via="ui")
    try:
        await bat_loop.kick()
    except Exception:
        pass
    return {"ok": True, "approved": bool(body.approve)}
