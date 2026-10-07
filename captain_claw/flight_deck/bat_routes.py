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
      - judge        → the fail-closed ``bat_judge`` verdict (deterministic
                       "all steps done" + an independent vote panel on a tier
                       resolved separately from the workers).
  * ``on_finish`` → deliver the result back to the caller / origin channel.

Phase 3 scope: agents can invoke Bat for build/research tasks. World-action
grants (the run-scoped email exception, capped real-money spend, account
creation) are Phases 5–7; a Bat worker here runs under the Phase-1 hard floor
with the owner's normal toolset.
"""

from __future__ import annotations

import json
import re
import types
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user

from captain_claw.flight_deck import bat_loop, human_ask
from captain_claw.flight_deck.bat_judge import Check, evaluate as _judge_evaluate
from captain_claw.flight_deck.bat_store import RUNNING_STATES

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/fd/bat", tags=["bat"])

_WORKER_MARKER = "CLAW_BAT_WORKER"
_BAT_MAX_RUNS_PER_OWNER = 3
_STEP_TIMEOUT_S = 1800.0
_PANEL_SIZE = 3
_MAX_MODEL_VETO_ROUNDS = 2

# model-veto streak per run (anti-deadlock; in-memory, resets on restart — the
# override is a soft backstop, not a safety guarantee).
_VETO_ROUNDS: dict[str, int] = {}


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


def plan_needs_gate(task: str, steps: list[dict]) -> tuple[bool, str]:
    """True + a reason when the task/plan would act in the world in a way the
    owner asked to approve first (mail / money / account creation). Deterministic
    keyword scan over the task and the step titles."""
    blob = (task or "") + "\n" + "\n".join(s.get("title", "") for s in (steps or []))
    hits = [label for label, rx in _GATE_COMPILED if rx.search(blob)]
    return (bool(hits), ", ".join(hits))


_APPROVE_WORDS = {"approve", "approved", "go", "yes", "ok", "okay", "proceed", "run", "do it"}


def _parse_vote(text: str) -> dict:
    """Parse a judge reply. No parseable VOTE line → 'abstain' (which the judge
    tally treats as non-agree; an all-abstain panel is no_quorum = not done)."""
    m = _VOTE_RE.search(text or "")
    vote = m.group(1).lower() if m else "abstain"
    r = _REASON_RE.search(text or "")
    return {"vote": vote, "reason": (r.group(1).strip() if r else "")[:300]}


def _parse_plan(content: str, task: str) -> list[dict]:
    """Parse the planner LLM's JSON into ordered steps. Tolerates ``` fences.
    Each step: {step_key, title, seq}. Empty/invalid → [] (caller falls back)."""
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
            out.append({"step_key": f"step-{i + 1}", "title": title[:400], "seq": i})
    return out[:12]


# Tools a Bat worker never gets: the `bat` launcher (blocks Bat-in-Bat) and
# `send_mail` (the direct Mailgun/SendGrid/SMTP path has no allowlist, rate
# limit or audit — Bat's email exception routes ONLY through the FD Gmail gate,
# which keeps the owner's allowlist/daily-limit/audit, so send_mail is barred).
_BAT_TOOL_DENY = frozenset({"bat", "send_mail"})


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


_STEP_SYSTEM = (
    "You are a Bat worker — the stubborn finisher. Complete the assigned step no matter how many "
    "approaches it takes; retry, change tactics, and use any tool available to you (including the "
    "browser, computer use, and starting a `vatra` run for a sub-problem). Be honest: only report "
    "what you actually accomplished and verified — never claim a step is done when it isn't. "
    "Absolutely never run anything that could harm the host or its drives."
)


def _build_step_prompt(run: dict, step: dict, prior: list[dict], human_input: str = "") -> str:
    lines = [f"# Goal\n{run.get('task', '')}", f"\n# Your step\n{step.get('title') or step.get('step_key')}"]
    done = [s for s in prior if s.get("status") == "done" and (s.get("output") or "").strip()]
    if done:
        lines.append("\n# What earlier steps produced (context)")
        for s in done:
            lines.append(f"\n## {s.get('title') or s.get('step_key')}\n{(s.get('output') or '')[:2000]}")
    if human_input:
        lines.append(f"\n# The human just provided this (use it to continue)\n{human_input}")
    lines.append("\nDo this step fully and report the concrete result. If you are truly blocked on "
                 "something only the human can give you (a verification code, a credential, a CAPTCHA), "
                 "say so clearly and stop — do not guess or fabricate it.")
    return "\n".join(lines)


_VOTE_SYSTEM = (
    "You are an independent, skeptical reviewer deciding whether a task is GENUINELY and fully done, "
    "judging the delivered result itself — not any claim that it is finished. Default to DISAGREE when "
    "the evidence is thin, partial, or merely describes what should be done rather than showing it was. "
    "Answer with exactly two lines:\nVOTE: AGREE|DISAGREE|ABSTAIN\nREASON: <one sentence>"
)


def _vote_prompt(lens: str, task: str, deliverable: str) -> str:
    return (f"Judge through this lens: {lens}\n\n# Task\n{task[:2000]}\n\n"
            f"# Delivered result\n{deliverable[:8000]}\n\n"
            f"Is the task genuinely and fully done?")


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
        return {"ok": False, "slug": res.slug, "port": 0, "auth": "", "message": res.message or "no port"}
    return {"ok": True, "slug": res.slug, "port": port, "auth": entry.get("web_auth", ""), "message": ""}


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


def _pick_tier(tiers: dict | None, prefer: tuple[str, ...] = ("reason", "fast")) -> str:
    tiers = tiers or {}
    for name in prefer:
        if tiers.get(name):
            return name
    return next(iter(tiers), "")


# ── the three handlers (registered into bat_loop at import) ────────────

async def _bat_planner(run: dict) -> list[dict]:
    cfg = run.get("config") or {}
    explicit = cfg.get("steps") or []
    if explicit:
        return [{"step_key": f"step-{i + 1}", "title": str(s)[:400], "seq": i}
                for i, s in enumerate(explicit) if str(s).strip()][:12]
    # Best-effort LLM decomposition; any failure → a single stubborn step.
    try:
        from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call
        from captain_claw.flight_deck.db import get_db
        from captain_claw.llm import Message
        db = get_db()
        tiers, _ = await _load_owner_tiers(db, run["owner_id"])
        creds = _creds_for(tiers)
        if creds:
            prov, mt = _provider_call(creds, temperature=0.2, default_max=1024, cap=4096)
            resp = await prov.complete(messages=[
                Message(role="system", content=(
                    "Break the user's goal into the smallest ordered list of concrete, independently "
                    "checkable steps that together finish it. Return ONLY a JSON array of short step "
                    "strings (max 8). No prose.")),
                Message(role="user", content=run.get("task", "")),
            ], temperature=0.2, max_tokens=mt)
            steps = _parse_plan(resp.content or "", run.get("task", ""))
            if steps:
                return steps
    except Exception as e:  # noqa: BLE001
        log.warning("bat planner fell back to single step", run_id=run.get("id"), error=str(e))
    task = run.get("task", "")
    return [{"step_key": "main", "title": task[:400] or "complete the task", "seq": 0}]


async def _bat_attempt(run: dict, step: dict) -> dict:
    from captain_claw.flight_deck.basna_routes import (
        _RUN_USAGE, _dispatch_one, _load_owner_tiers, _run_sid,
    )
    from captain_claw.flight_deck.db import get_db
    from captain_claw.flight_deck import pricing

    sid = run["id"]
    owner = run["owner_id"]
    db = get_db()
    user = await db.get_user_by_id(owner) or {"id": owner}
    tiers, env_vars = await _load_owner_tiers(db, owner)
    tier = _pick_tier(tiers)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
    name = f"bat-{sid[:8]}-{step['step_key']}"[:60]
    worker = await _spawn_bat_worker(
        stub, user, name=name, tier=tier, tiers=tiers,
        env_vars=env_vars, extra_env=_bat_env(sid, step["step_key"], owner),
    )
    if not worker["ok"]:
        return {"ok": False, "error": f"could not spawn worker: {worker['message']}"}

    store = bat_loop.get_store()

    async def _on_action(ev: dict) -> None:
        if store is not None:
            try:
                await store.append_event(sid, "action", str(ev.get("tool", ""))[:60],
                                         agent=step["step_key"], detail=str(ev.get("detail", ""))[:200])
            except Exception:
                pass

    try:
        prior = await store.list_steps(sid) if store else []
        # A human answer provided while this step was awaiting_human: a secret
        # lives only in memory (consumed once), a non-secret was stashed in config.
        human_input = _SECRET_ANSWERS.pop((sid, step["step_key"]), "") or \
            ((run.get("config") or {}).get("answers") or {}).get(step["step_key"], "")
        prompt = _build_step_prompt(run, step, prior, human_input=human_input)
        _run_sid.set(sid)
        _RUN_USAGE.setdefault(sid, [])
        before = len(_RUN_USAGE[sid])
        # Run-scoped email exception (Phase 5): grant mail-write ONLY when the
        # owner approved a mail plan for THIS run. Otherwise deny (PR #56 default).
        # This frame is Bat's own dispatch; it is never carried into a child run.
        email_allowed = bool((run.get("config") or {}).get("email_allowed"))
        automation = {"kind": "bat", "job_text": str(run.get("task", ""))[:500],
                      "mail_write": "allow" if email_allowed else "deny"}
        res = await _dispatch_one(
            worker["port"], worker["auth"], prompt, _STEP_TIMEOUT_S,
            on_action=_on_action, agent_name=step["step_key"], automation=automation,
        )
        new_usage = _RUN_USAGE[sid][before:]
        cost = pricing.summarize(new_usage)
        ok = bool(res.get("ok")) and not res.get("timed_out")
        return {
            "ok": ok,
            "output": res.get("output", ""),
            "usd": float(cost.get("usd") or 0.0),
            "tokens": int(cost.get("tokens") or 0),
            "error": res.get("error", "") or ("timed out" if res.get("timed_out") else ""),
        }
    finally:
        _teardown([worker["slug"]])


async def _bat_judge(run: dict, steps: list[dict]) -> dict:
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call, _RUN_USAGE, _run_sid
    from captain_claw.flight_deck.db import get_db
    from captain_claw.flight_deck import pricing
    from captain_claw.llm import Message

    sid = run["id"]
    deliverable = _assemble_deliverable(steps)
    all_done = bool(steps) and all(s.get("status") == "done" for s in steps)
    failed = [s["step_key"] for s in steps if s.get("status") == "failed"]
    det = [Check("all planned steps completed", passed=all_done, critical=True,
                 detail=(f"failed: {', '.join(failed)}" if failed else "pending steps remain"))]

    panel_fn = None
    try:
        db = get_db()
        tiers, _ = await _load_owner_tiers(db, run["owner_id"])
        creds = _creds_for(tiers)
        # Layer-3 evidence: if this run could send mail, hand the panel the
        # ground-truth send audit so it judges honesty against reality, not the
        # worker's claim.
        receipt = ""
        if (run.get("config") or {}).get("email_allowed"):
            try:
                sends = await db.list_gmail_sends(run["owner_id"], limit=50)
                receipt = _email_receipt_block(sends, sid[:8])
            except Exception:
                receipt = ""
        if creds and deliverable.strip():
            lenses = ["correctness and completeness",
                      "whether it ACTUALLY did the task vs merely described it",
                      "edge cases, honesty, and unverified claims"]

            async def vote(deliverable_text: str, task: str, idx: int) -> dict:
                lens = lenses[idx % len(lenses)]
                _run_sid.set(sid)
                _RUN_USAGE.setdefault(sid, [])
                shown = (receipt + "\n\n" + deliverable_text) if receipt else deliverable_text
                prov, mt = _provider_call(creds, temperature=0.0, default_max=256, cap=1024)
                resp = await prov.complete(messages=[
                    Message(role="system", content=_VOTE_SYSTEM),
                    Message(role="user", content=_vote_prompt(lens, task, shown)),
                ], temperature=0.0, max_tokens=mt)
                return _parse_vote(resp.content or "")

            panel_fn = vote
    except Exception as e:  # noqa: BLE001
        log.warning("bat judge panel setup failed; deterministic-only", run_id=sid, error=str(e))

    prior = _VETO_ROUNDS.get(sid, 0)
    before = len(_RUN_USAGE.get(sid, []))
    verdict = await _judge_evaluate(
        task=run.get("task", ""), deliverable=deliverable, deterministic=det,
        panel_vote_fn=panel_fn, panel_size=_PANEL_SIZE,
        prior_model_veto_rounds=prior, max_model_veto_rounds=_MAX_MODEL_VETO_ROUNDS,
    )
    # Count judge spend against the run's persisted cap.
    store = bat_loop.get_store()
    if store is not None and _RUN_USAGE.get(sid):
        new = _RUN_USAGE[sid][before:]
        if new:
            cost = pricing.summarize(new)
            if cost.get("usd"):
                try:
                    await store.bump_cost(sid, float(cost["usd"]), int(cost.get("tokens") or 0))
                except Exception:
                    pass
    # Track the model-veto streak for the anti-deadlock override.
    if not verdict.done and not verdict.det_criticals and verdict.panel.get("verdict") in ("disagree", "tie", "no_quorum"):
        _VETO_ROUNDS[sid] = prior + 1
    elif verdict.done:
        _VETO_ROUNDS.pop(sid, None)
    return {"done": verdict.done, "reason": verdict.reason, "overridden": verdict.overridden}


async def _bat_on_finish(run: dict) -> None:
    """Deliver the result back where the run came from (bell / channel / caller
    agent), mirroring Basna/Vatra's completion delivery."""
    sid = run["id"]
    _VETO_ROUNDS.pop(sid, None)
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
            from captain_claw.flight_deck.db import get_db
            await get_db().add_notification(
                owner, "run" if ok else "run_error", run.get("title") or "Bat run",
                summary, "bat", sid)
            return
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


async def _gate_check(store, run: dict) -> str:
    """Human-in-the-loop gate (bat_loop seam). Returns proceed|wait|cancelled|error.

    Two jobs: (1) the START GATE — a fresh run whose plan would send mail, spend,
    or create an account pauses (awaiting_plan) for owner approval; (2) RESUME —
    an awaiting_* run proceeds once its ask is answered, is cancelled if the plan
    is rejected, and errors if the ask expired."""
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
                await store.set_config(run_id, cfg)
                await store.append_event(
                    run_id, "note",
                    "plan approved — proceeding" + (" (email enabled for this run)"
                                                    if cfg.get("email_allowed") else ""))
                return "proceed"
            await store.set_status(run_id, "cancelled", stopped_reason="plan_rejected")
            await store.append_event(run_id, "cancelled", "plan rejected by the owner")
            return "cancelled"
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

    # fresh run — start gate
    cfg = run.get("config") or {}
    if cfg.get("plan_approved"):
        return "proceed"
    steps = await store.list_steps(run_id)
    needs, reason = plan_needs_gate(run.get("task", ""), steps)
    if not needs:
        return "proceed"
    # Remember what was gated so approval can grant the matching capability
    # (e.g. 'send email' → the run-scoped email exception).
    cfg = dict(run.get("config") or {})
    cfg["gate_reason"] = reason
    await store.set_config(run_id, cfg)
    existing = await store.latest_open_ask(run_id)
    if not (existing and existing["kind"] == "plan_approval"):
        q = (f"This Bat run would {reason}. Approve it to run, or cancel.\n\nPlan:\n"
             + "\n".join(f"- {s.get('title')}" for s in steps))
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
        from captain_claw.flight_deck.db import get_db
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
                "steps": [str(s) for s in (body.steps or []) if str(s).strip()]},
        origin={"platform": body.origin_platform, "user_id": body.origin_user_id,
                "chat_id": body.origin_chat_id, "kind": body.origin_kind,
                "address": body.origin_address},
        source_host=body.source_host or "localhost", source_port=int(body.source_port or 0),
        llm_usd_cap=float(body.llm_usd_cap or 0.0), status="planning",
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


# ── owner-authenticated human-in-the-loop (answer path A) ──────────────
# Not under /fd/bat/agent/, so these use the owner's session (get_current_user),
# not the agent guard — a human answers, never the agent.

class _AnswerReq(BaseModel):
    text: str = ""


class _ApproveReq(BaseModel):
    approve: bool = True


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
