"""Captain Swarm — the loop that keeps improving a number.

docs/captain-swarm-plan.md §3, Phase 2: one lane, one number, code. A run

  1. seals the scorer (the base commit's copy) and the held-out cases,
  2. measures the starting point several times — its number and its noise,
  3. asks a model for a few different ideas, and for each one, in turn:
     Crew tries it in a worktree cut from the best so far; Captain screens
     the commit (scope, gaming scan), measures it on the open cases, and if it
     beats the best by more than noise without breaking a guardrail, measures
     it on the held-out cases too; a gain that holds there becomes the new
     best, and everything goes into the ledger,
  4. stops at the target, the budget, the time or experiment limit, after
     rounds with no gain, or when the owner says so.

Captain never merges: the best commit sits on ``captain-swarm/<run8>/champion``
and the owner opens a PR from the page.

Phase 4: lanes in parallel. A round's ideas run in *waves* of up to
``lane.parallel`` attempts, all started together from the same best:

  - every lane at once — but never two attempts on one site or account
    (ideas that name the same site wait for a later wave, and the rehearsal
    guard holds a site for the attempt whose browser is on it),
  - a quick look on a slice of the open cases when the kind has one and
    there are three or more to rank; only the better half goes on,
  - the open cases, through the measurement queue (a local scorer is one GPU,
    an agent answers one question at a time),
  - then one keeper per wave: of the attempts that beat the best, the
    strongest that a model reviewer passes and the held-out cases confirm.
    The others that beat it come back once, on top of the new best.

With one lane a wave is one attempt, and the loop is Phase 2's.

Phase 6: learning (:mod:`captain_swarm_learning`). The proposer adds a
reading of the run to every round's ideas; a stall brings one strategy round
before the run gives up; and a run's lessons are written down when it ends,
so the next run of the same kind of task starts from them, with the
strategies that have won there going into the first waves.

Durable like Bat: one driver per run under a lease, every step written down
as it goes, a run re-adopted after a restart; a measurement interrupted by a
restart is run again, never trusted. The lane, the proposer, the measurement
and the reviewer are seams (:func:`set_lane`, :func:`set_proposer`,
:func:`set_measure`, :func:`set_review`), so the whole loop tests without
models or git.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from captain_claw.flight_deck import captain_swarm_learning as learning
from captain_claw.flight_deck import captain_swarm_objective as objective_mod
from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.flight_deck import captain_swarm_screen as screen
from captain_claw.flight_deck.captain_swarm_score import Stats, beats, margin, read_number, read_path, score_command, stats
from captain_claw.flight_deck.captain_swarm_store import ACTIVE_STATES, CaptainSwarmStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

_TICK_S = 15.0
_POLL_S = 10.0
_HEARTBEAT_S = 60.0
_WAIT_LANES_S = 30.0
_LEDGER_FOR_PROPOSER = 14
_REVIEW_CHARS = 20000
# Where an experiment can still be: the rest (EXP_FINAL) are verdicts.
LIVE_EXP = ("proposed", "running", "screening", "measuring", "measured")
_ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SHELL", "TERM",
             "OLLAMA_HOST")


def sealed_dir(run_id: str) -> Path:
    from captain_claw.flight_deck import server as srv
    return srv.DATA_DIR / "captain_swarm" / "sealed" / "runs" / run_id


def champion_branch(run_id: str) -> str:
    return f"captain-swarm/{run_id.removeprefix('cs_')[:8]}/champion"


# ── seams ────────────────────────────────────────────────────────────


class Lane(Protocol):
    async def start(self, run: dict, exp: dict, brief: str) -> dict: ...
    async def poll(self, run: dict, exp: dict) -> dict: ...
    async def stop(self, run: dict, exp: dict) -> None: ...
    async def discard(self, run: dict, exp: dict) -> None: ...


Proposer = Callable[[dict, list[dict], int], Awaitable[list[dict]]]  # (run, ledger, k) → ideas
Measure = Callable[[dict, str, str], Awaitable[dict]]
Seal = Callable[[dict, str], Awaitable[None]]
Screen = Callable[[dict, dict], Awaitable[dict]]
Branch = Callable[[dict, str], Awaitable[None]]
Review = Callable[[dict, dict], Awaitable[dict]]  # (run, exp) → {"ok", "reason"}
Strategy = Callable[[dict, list[dict]], Awaitable[list[dict]]]  # (run, ledger) → directions

_LANE: Lane | None = None
_PROPOSER: Proposer | None = None
_MEASURE: Measure | None = None
_SEAL: Seal | None = None
_SCREEN: Screen | None = None
_BRANCH: Branch | None = None
_REVIEW: Review | None = None
_STRATEGY: Strategy | None = None
_START: Callable[[dict], Awaitable[str]] | None = None
_STORE: CaptainSwarmStore | None = None
_SUPERVISOR: Supervisor | None = None


def set_lane(fn: Lane | None) -> None:
    global _LANE
    _LANE = fn


def set_review(fn: Review | None) -> None:
    global _REVIEW
    _REVIEW = fn


def set_strategy(fn: Strategy | None) -> None:
    global _STRATEGY
    _STRATEGY = fn


def set_proposer(fn: Proposer | None) -> None:
    global _PROPOSER
    _PROPOSER = fn


def set_measure(fn: Measure | None) -> None:
    global _MEASURE
    _MEASURE = fn


def set_seal(fn: Seal | None) -> None:
    global _SEAL
    _SEAL = fn


def set_screen(fn: Screen | None) -> None:
    global _SCREEN
    _SCREEN = fn


def set_branch(fn: Branch | None) -> None:
    global _BRANCH
    _BRANCH = fn


def set_start(fn: Callable[[dict], Awaitable[str]] | None) -> None:
    global _START
    _START = fn


def set_store(store: CaptainSwarmStore | None) -> None:
    global _STORE
    _STORE = store


def get_store() -> CaptainSwarmStore | None:
    return _STORE


# ── the measurement queue ────────────────────────────────────────────


def measure_parallel() -> int:
    """Command measurements at once on this deck: FD_CAPTAIN_SWARM_MEASURE_PARALLEL, 1."""
    return objective_mod.env_int("FD_CAPTAIN_SWARM_MEASURE_PARALLEL", 1, 1, 8)


_SLOTS: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]] = {}
# Run → {experiment ("" is the best itself): "waiting" | "measuring"}, for the page.
_MEASURING: dict[str, dict[str, str]] = {}


def _slot(key: str, size: int) -> asyncio.Semaphore:
    here = asyncio.get_running_loop()
    got = _SLOTS.get(key)
    if got is None or got[0] is not here:
        got = _SLOTS[key] = (here, asyncio.Semaphore(size))
    return got[1]


@contextlib.asynccontextmanager
async def measurement_slot(run: dict, exp_id: str = ""):
    """A turn in the measurement queue of *run*'s kind: deck-wide for a command
    (one GPU), per agent for practice (it answers one question at a time), none
    for an outcome (it only reads pages)."""
    queue = parts_for(str(run["objective"].get("kind") or "code"))["queue"](run)
    mine = _MEASURING.setdefault(run["id"], {})
    mine[exp_id] = "waiting" if queue else "measuring"
    try:
        if queue is None:
            yield
        else:
            async with _slot(*queue):
                mine[exp_id] = "measuring"
                yield
    finally:
        mine.pop(exp_id, None)
        if not mine:
            _MEASURING.pop(run["id"], None)


def measuring(run_id: str) -> dict[str, str]:
    return dict(_MEASURING.get(run_id) or {})


# ── sealing and measuring (the defaults) ─────────────────────────────


async def seal(run: dict, base_sha: str) -> None:
    """The scorer, frozen at the base commit, and the held-out cases, copied
    where attempts never look. Idempotent: a re-adopted run reuses its seal."""
    obj = run["objective"]
    d = sealed_dir(run["id"])
    if (d / "sealed.json").is_file():
        return
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    snap = obj["subject"].get("snapshot") or {}
    await screen.snapshot(obj["subject"]["repo_path"], base_sha, d / "harness",
                          paths=snap.get("paths") or (), exclude=snap.get("exclude") or ())
    hold = d / "holdout.json"
    strings: dict[str, str] = {}
    if obj["holdout"].get("file"):
        shutil.copyfile(obj["holdout"]["file"], hold)
        os.chmod(hold, 0o600)
        strings.update(screen.case_strings(hold, "holdout"))
    dev_file = obj["metric"].get("dev_file") or ""
    if dev_file:
        strings.update(screen.case_strings(d / "harness" / dev_file, "dev"))
    (d / "strings.json").write_text(json.dumps(strings, ensure_ascii=False), encoding="utf-8")
    (d / "sealed.json").write_text(json.dumps({"base_sha": base_sha, "at": time.time()}), encoding="utf-8")


def _scrubbed_env(extra: list[str]) -> dict[str, str]:
    """What the code under test may see of the deck's environment: the basics,
    and the variables the objective named. Never the deck's own secrets."""
    keep = {k: os.environ[k] for k in (*_ENV_KEEP, *extra) if k in os.environ}
    keep["PYTHONDONTWRITEBYTECODE"] = "1"
    return keep


async def measure_commit(run: dict, sha: str, which: str) -> dict:
    """Measure commit *sha* on the open cases (``dev``) or the held-out ones,
    with the sealed scorer, against a clean copy of that commit:
    ``{"value", "stats", "guard", "per_case", "error", "seconds"}``."""
    from captain_claw import bat_floor
    obj = run["objective"]
    d = sealed_dir(run["id"])
    checkout = d / "checkouts" / f"{sha[:12]}-{secrets.token_hex(2)}"
    work = d / "work"
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        snap = obj["subject"].get("snapshot") or {}
        await screen.snapshot(obj["subject"]["repo_path"], sha, checkout,
                              paths=snap.get("paths") or (), exclude=snap.get("exclude") or ())
        template = obj["metric"]["run"] if which == "dev" else obj["holdout"]["run"]
        dev_file = obj["metric"].get("dev_file") or ""
        command = objective_mod.fill(
            template, python=objective_mod.python(), harness=str(d / "harness"), checkout=str(checkout),
            dev=str(d / "harness" / dev_file) if dev_file else "", holdout=str(d / "holdout.json"))
        blocked, why = bat_floor.screen("shell", {"command": command})
        if blocked:
            return {"value": None, "error": f"the floor refused the measurement: {why}"}
        got = await score_command(command, work, read=obj["metric"]["read"],
                                  env=_scrubbed_env(obj["metric"].get("env") or []), env_only=True,
                                  timeout=float(obj["metric"].get("timeout_s") or 7200))
    finally:
        shutil.rmtree(checkout, ignore_errors=True)
    return reading(obj, got, seconds=round(time.monotonic() - started, 1))


def reading(obj: dict, got: dict, *, seconds: float = 0.0) -> dict:
    """What one command's output says: the number, its spread across repeats,
    the guardrail numbers and the per-case rates."""
    value, data = got.get("value"), got.get("data") or {}
    if value is None:
        return {"value": None, "error": got.get("error") or "no number", "seconds": seconds}
    spread_key = obj["metric"].get("spread") or ""
    raw = read_path(data, spread_key) if spread_key else None
    spread = [float(x) for x in raw if isinstance(x, (int, float)) and not isinstance(x, bool)] \
        if isinstance(raw, list) else []
    s = stats(spread or [value])
    guard = {g["name"]: read_number(data, g["read"]) for g in obj.get("guardrails") or []}
    per_case_key = obj["metric"].get("per_case") or ""
    per_case = read_path(data, per_case_key) if per_case_key else None
    return {"value": float(value), "stats": s.to_dict(), "guard": guard,
            "per_case": per_case if isinstance(per_case, dict) else {}, "error": "", "seconds": seconds}


async def screen_commit(run: dict, exp: dict) -> dict:
    """Scope and the gaming scan for *exp*'s commit: ``{"ok", "reason",
    "changed", "diff_stat", "hits"}``."""
    obj = run["objective"]
    repo = obj["subject"]["repo_path"]
    changed = await screen.changed_files(repo, exp["base_sha"], exp["head_sha"])
    problems = screen.scope_problems(changed, obj["subject"]["editable"], obj["subject"]["protected"])
    stat = await screen.diff_stat(repo, exp["base_sha"], exp["head_sha"])
    if problems:
        return {"ok": False, "reason": "out of bounds: " + "; ".join(problems[:6]), "changed": changed,
                "diff_stat": stat, "hits": []}
    try:
        strings = json.loads((sealed_dir(run["id"]) / "strings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        strings = {}
    hits = screen.gaming_hits(await screen.added_lines(repo, exp["base_sha"], exp["head_sha"]), strings)
    if hits:
        peeked = any(h["source"].startswith("holdout") for h in hits)
        what = "; ".join(f"{h['string']!r} ({h['source']}) in {h['file']}" for h in hits[:4])
        return {"ok": False, "reason": ("it carries held-out cases it was never shown: " if peeked
                                        else "it special-cases the eval: ") + what,
                "changed": changed, "diff_stat": stat, "hits": hits}
    return {"ok": True, "reason": "", "changed": changed, "diff_stat": stat, "hits": []}


async def move_champion_branch(run: dict, sha: str) -> None:
    rc, out = await screen.git(run["objective"]["subject"]["repo_path"], "branch", "-f", champion_branch(run["id"]), sha)
    if rc:
        raise RuntimeError(f"couldn't move the champion branch: {out[-200:]}")


# ── judging ──────────────────────────────────────────────────────────


def fmt(value: float | None) -> str:
    """A score as a person reads it; "nothing yet" (an outcome run's start) is a dash."""
    if value is None or abs(float(value)) >= 1e11:
        return "—"
    return f"{float(value):.3f}"


def judge_dev(obj: dict, cand: dict, champ: dict) -> tuple[bool, str]:
    """Does *cand* beat *champ* on the open cases by more than noise, without
    breaking a guardrail? ``(passes, why)``."""
    direction = obj["metric"]["direction"]
    c, b = Stats(**cand["stats"]), Stats(**champ["stats"])
    need = max(margin(b, obj["noise"]["min_gain"]), 2.0 * c.stdev)
    if not beats(c, b, min_gain=obj["noise"]["min_gain"], direction=direction):
        return False, (f"{fmt(c.mean)} ± {c.stdev:.3f} against the best {fmt(b.mean)} ± {b.stdev:.3f}: "
                       f"not a gain of at least {need:.3f}")
    for g in obj.get("guardrails") or []:
        new, old = (cand.get("guard") or {}).get(g["name"]), (champ.get("guard") or {}).get(g["name"])
        if isinstance(new, (int, float)) and isinstance(old, (int, float)) and old > 0:
            rise = (new - old) / old
            if rise > g["max_increase"]:
                return False, f"{g['name']} went up {rise:.0%} (allowed {g['max_increase']:.0%})"
    return True, f"{fmt(c.mean)} ± {c.stdev:.3f} beats the best {fmt(b.mean)} ± {b.stdev:.3f}"


def judge_holdout(obj: dict, cand: dict, champ: dict) -> tuple[bool, str]:
    """Does the gain hold on the held-out cases: no worse than the best, beyond noise?"""
    c, b = Stats(**cand["stats"]), Stats(**champ["stats"])
    slack = margin(b, 0.0)
    ok = c.mean >= b.mean - slack if obj["metric"]["direction"] == "max" else c.mean <= b.mean + slack
    return ok, f"held-out {fmt(c.mean)} against the best's {fmt(b.mean)}" + ("" if ok else ": the gain didn't hold")


def target_met(obj: dict, dev: dict) -> bool:
    target = obj["stop"].get("target")
    if target is None or not dev.get("stats"):
        return False
    mean = float(dev["stats"]["mean"])
    return mean >= target if obj["metric"]["direction"] == "max" else mean <= target


# ── the proposer ─────────────────────────────────────────────────────


_PROPOSE_SYSTEM = (
    "You plan experiments for Captain Swarm, which keeps improving one measured number. Each experiment is one "
    "idea a coding worker will try in its own branch, starting from the best version so far; Captain then measures "
    "it on open cases and on held-out cases it never shows anyone. Propose {k} DIFFERENT ideas: different "
    "strategies, not rewordings. Prefer small, general changes that would help any similar case. Never propose "
    "reading, changing or special-casing the eval, its cases or its files: a change that names a case is thrown "
    "out, and so is one that touches a protected path. Don't repeat an idea the ledger already discarded unless "
    "you say what is different now. Return ONLY a JSON array of objects: "
    '{{"hypothesis": "<what to change, concretely>", "strategy_tag": "<2-4 word kebab-case tag>", '
    '"files": ["<paths you expect to change>"], "why": "<why it should move the number>"}}. No prose.'
)


def proposer_brief(run: dict, ledger: list[dict]) -> str:
    """What the proposer is told: the goal, the bounds, the best's weak spots on
    the open cases, and what has been tried. Never anything held out."""
    obj = run["objective"]
    champ = run.get("champion") or {}
    dev = champ.get("dev") or {}
    weak = sorted(((k, v) for k, v in (dev.get("per_case") or {}).items()
                   if isinstance(v, (int, float)) and v < 1), key=lambda kv: kv[1])
    lines = [f"Goal: {obj['title']}", obj.get("goal") or "",
             f"The number: {obj['metric']['read']} ({'higher' if obj['metric']['direction'] == 'max' else 'lower'} is better).",
             f"May change: {', '.join(obj['subject']['editable'])}",
             f"Never change: {', '.join(obj['subject']['protected']) or '(nothing listed)'}"]
    if dev.get("stats"):
        s = dev["stats"]
        lines.append(f"The best so far: {s['mean']:.3f} ± {s['stdev']:.3f} on the open cases.")
    if weak:
        lines.append("Open cases it still fails sometimes (pass rate): "
                     + ", ".join(f"{k} {v:.2f}" for k, v in weak[:15]))
    if ledger:
        lines.append("\nTried so far (newest last):")
        for e in ledger[-_LEDGER_FOR_PROPOSER:]:
            lines.append(f"- [{e['status']}] ({e.get('strategy_tag') or '-'}) {e['hypothesis'][:300]}"
                         + (f" → {learning.shown_verdict(e)[:200]}" if e.get("verdict_reason") else ""))
    lines.append(run.get("learned") or "")
    return "\n".join(x for x in lines if x is not None)


def parse_proposals(text: str, k: int) -> list[dict]:
    """The ideas in a model's reply, tidied; garbage is dropped, not guessed at."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("hypothesis") or "").strip():
            continue
        tag = re.sub(r"[^a-z0-9-]+", "-", str(item.get("strategy_tag") or "").lower()).strip("-")[:40]
        files = [str(f)[:200] for f in item.get("files") or [] if str(f).strip()][:10]
        sites = sorted({s for s in (rehearsal.site_of(str(x)) for x in item.get("sites") or []
                                    if isinstance(x, str)) if s})[:8]
        out.append({"hypothesis": str(item["hypothesis"]).strip()[:1500], "strategy_tag": tag,
                    "files": files, "why": str(item.get("why") or "").strip()[:600],
                    **({"sites": sites} if sites else {})})
    return out[:k]


def diverse(proposals: list[dict], ledger: list[dict]) -> list[dict]:
    """Drop rewordings: an idea already discarded (unless it says what's
    different — a 'why' longer than the idea was given), or one that repeats an
    earlier idea in the same batch by tag and files."""
    kept: list[dict] = []
    tried = [e for e in ledger if e["status"] in ("discarded", "disqualified", "failed")]
    for p in proposals:
        text = p["hypothesis"].lower()
        if any(difflib.SequenceMatcher(None, text, e["hypothesis"].lower()).ratio() > 0.85 for e in tried):
            continue
        if any(p["strategy_tag"] and p["strategy_tag"] == q["strategy_tag"]
               and (set(p["files"]) & set(q["files"]) or not p["files"]) for q in kept):
            continue
        kept.append(p)
    return kept


async def ask_model(run: dict, system: str, user: str, *, label: str, temperature: float = 0.7,
                    panel: bool = False) -> str:
    """One model call on the owner's tiers, counted against the run's budget.
    *panel*: Bat's reviewer choice — the best model the workers don't run on."""
    from captain_claw.flight_deck import pricing
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _RUN_USAGE, _load_owner_tiers, _provider_call, _run_sid
    from captain_claw.flight_deck.bat_routes import _creds_for, _panel_creds
    from captain_claw.llm import Message
    tiers, _ = await _load_owner_tiers(get_db(), run["owner_id"])
    creds = _panel_creds(tiers)[0] if panel else _creds_for(tiers, prefer=("reason", "balanced", "fast"))
    if not creds:
        raise RuntimeError("no model to plan experiments with: set up a tier set first")
    key = f"{run['id']}:{label}:{secrets.token_hex(3)}"
    token = _run_sid.set(key)
    _RUN_USAGE[key] = []
    try:
        prov, max_tokens = _provider_call(creds, temperature=temperature, default_max=4096, cap=16384)
        resp = await prov.complete(messages=[Message(role="system", content=system), Message(role="user", content=user)],
                                   temperature=temperature, max_tokens=max_tokens)
    finally:
        _run_sid.reset(token)
        cost = pricing.summarize(_RUN_USAGE.pop(key, []))
        if _STORE is not None:
            await _STORE.add_spend(run["id"], float(cost.get("usd") or 0.0),
                                   int((cost.get("tokens") or {}).get("total_tokens", 0) or 0))
    return resp.content or ""


async def propose(run: dict, ledger: list[dict], k: int) -> list[dict]:
    """Up to *k* ideas for code, from one model call."""
    text = await ask_ideas(run, _PROPOSE_SYSTEM.format(k=k), proposer_brief(run, ledger))
    return diverse(parse_proposals(text, k), ledger)


async def ask_ideas(run: dict, system: str, brief: str) -> str:
    """The proposer's one call: its ideas, and a reading of the run so far,
    kept as a lesson of this run."""
    text = await ask_model(run, system + learning.READING_ASK, brief, label="propose")
    if _STORE is not None:
        try:
            await learning.note_reading(_STORE, run, text)
        except Exception:  # noqa: BLE001 — a lost reading never costs the round its ideas
            log.warning("captain swarm: couldn't keep the planner's reading", run_id=run.get("id"), exc_info=True)
    return text


def sites_of(idea: dict) -> list[str]:
    return list(idea.get("sites") or (idea.get("change") or {}).get("sites") or [])


def pick_wave(queue: list[dict], size: int, per_site: int = 1) -> tuple[list[dict], list[dict]]:
    """Up to *size* ideas to start together, and the rest. An idea that would put
    more than *per_site* attempts on one site waits for a later wave (§4.4)."""
    picked: list[dict] = []
    rest: list[dict] = []
    on: dict[str, int] = {}
    for idea in queue:
        sites = sites_of(idea)
        if len(picked) < size and all(on.get(s, 0) < per_site for s in sites):
            picked.append(idea)
            for s in sites:
                on[s] = on.get(s, 0) + 1
        else:
            rest.append(idea)
    return picked, rest


# ── the reviewer ─────────────────────────────────────────────────────


_REVIEW_SYSTEM = (
    "You review one experiment of Captain Swarm, which keeps improving a measured number, before it may become the "
    "best version. Captain has measured it already; you look for what a number can't see. {question} Judge only "
    "what is in front of you. Be strict about gaming and about a result that doesn't answer the task, and lenient "
    'about style. Return ONLY JSON: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}.'
)

REVIEW_QUESTIONS = {
    "code": ("Does the change make the general behaviour better, or does it game the measurement: special-case "
             "the eval's tasks, files or expected answers, read the eval, its cases or the run's environment, "
             "weaken a check, or hard-code an output? Ordinary improvements to prompts and code pass."),
    "practice": ("Are the new rules general rules about KINDS of mail, in the owner's voice, that a person could "
                 "follow on mail they have never seen? Fail rules that describe one particular email, carry "
                 "one-off details (a particular subject, date, amount or thread), or talk about the test itself."),
    "outcome": ("Does the result really answer the task: the very thing asked for (not a kit, a bundle, a part, a "
                "used or refurbished one, or another variant, unless the task allows it), on terms the task "
                "allows, with the number being that thing's own number on the page? Fail anything the page "
                "excerpt doesn't support."),
    "document": ("Judge only three things, which a checklist grade can't see. Fail it if it states facts, figures or "
                 "sources it made up or that its own text contradicts; if it pads itself to tick a checklist (keyword "
                 "lists, empty or repeated headings, filler); or if it answers a different question than the brief "
                 "asks. Gaps, thin parts and evidence it says is weak are NOT reasons to fail: the checklist grades "
                 "already count them, and a document honest about what it couldn't find passes."),
}


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    verdict = str((data or {}).get("verdict") or "").strip().lower() if isinstance(data, dict) else ""
    if verdict not in ("pass", "fail"):
        return None
    return {"ok": verdict == "pass", "reason": " ".join(str(data.get("reason") or "").split())[:400]}


async def code_review_material(run: dict, exp: dict) -> str:
    obj = run["objective"]
    rc, diff = await screen.git(obj["subject"]["repo_path"], "diff", "--no-color", f"{exp['base_sha']}..{exp['head_sha']}")
    if rc:
        raise RuntimeError(f"git diff failed: {diff[-200:]}")
    cut = "" if len(diff) <= _REVIEW_CHARS else f"\n… ({len(diff) - _REVIEW_CHARS} more characters not shown)"
    return "\n".join([f"Goal: {obj['title']}", obj.get("goal") or "",
                      f"May change: {', '.join(obj['subject']['editable'])}",
                      f"Never change: {', '.join(obj['subject']['protected'])}",
                      f"The idea: {exp['hypothesis']}", "", "The change:", diff[:_REVIEW_CHARS] + cut])


async def review(run: dict, exp: dict) -> dict:
    """A model reads what is about to become the best (§5.6), on the best model
    the lanes don't run on: ``{"ok", "reason"}``. Its unreadable answer is asked
    again once; then it's an error, never a pass."""
    kind = str(run["objective"].get("kind") or "code")
    material = await parts_for(kind)["review"](run, exp)
    system = _REVIEW_SYSTEM.format(question=REVIEW_QUESTIONS[kind])
    for _ in range(2):
        got = parse_verdict(await ask_model(run, system, material, label="review", temperature=0.0, panel=True))
        if got:
            return got
    raise RuntimeError("the reviewer's answer couldn't be read")


# ── the Crew lane ────────────────────────────────────────────────────


def lane_brief(run: dict, exp: dict) -> str:
    obj = run["objective"]
    dev = (run.get("champion") or {}).get("dev") or {}
    weak = [f"{k} ({v:.2f})" for k, v in sorted((dev.get("per_case") or {}).items(), key=lambda kv: kv[1])
            if isinstance(v, (int, float)) and v < 1][:12]
    return "\n".join([
        f"You are one experiment in a Captain Swarm run: {obj['title']}.",
        obj.get("goal") or "",
        "",
        f"Try this one idea: {exp['hypothesis']}",
        f"Why it might help: {exp['why']}" if exp.get("why") else "",
        "",
        "Rules:",
        f"- Change only files under: {', '.join(obj['subject']['editable'])}.",
        f"- Never change: {', '.join(obj['subject']['protected'])}.",
        "- You may read the eval to understand it, but improve the general behaviour: a change that names or "
        "special-cases its cases, files or expected answers is thrown out.",
        "- Keep the change small and about this one idea.",
        "- Commit your work on this branch with a clear message. Captain measures the commit; anything left "
        "uncommitted isn't measured.",
        "" if not weak else "\nOpen cases the current best still fails sometimes (pass rate): " + ", ".join(weak),
    ]).strip()


class CrewLane:
    """One experiment = one Crew task on the run's repository, in a worktree cut
    from the best commit so far (``base_ref``)."""

    async def _user(self, run: dict) -> tuple[dict, Any]:
        from captain_claw.flight_deck import crew_access, crew_runner
        user = await crew_runner._user_of(run["owner_id"])
        grant = crew_access.grant_of(user)
        if not grant.allowed or not grant.trusted:
            raise RuntimeError("Crew isn't open to this owner for a local repository (admins only).")
        return user, grant

    async def start(self, run: dict, exp: dict, brief: str) -> dict:
        from captain_claw.flight_deck import crew_runner
        user, grant = await self._user(run)
        lane = run["objective"]["lane"]
        body = {"harness": lane["harness"], "kind": "ship", "deliver": "branch", "brief": brief,
                "repo_path": run["objective"]["subject"]["repo_path"], "base_ref": exp["base_sha"],
                "title": f"Captain Swarm {run['id'][-6:]} #{exp['round']}.{exp['n']}: {exp['hypothesis'][:70]}",
                "source_host": "localhost"}
        if lane.get("model_source"):
            body["model_source"] = lane["model_source"]
        elif lane.get("model"):
            body["model"] = lane["model"]
        task = await crew_runner.dispatch(user, grant, body)
        return {"child_id": task["id"], "branch": task.get("branch") or ""}

    async def poll(self, run: dict, exp: dict) -> dict:
        from captain_claw.flight_deck import crew_store
        user, _ = await self._user(run)
        task = crew_store.get_store().get(user["id"], exp["child_id"])
        if task is None:
            return {"state": "failed", "note": "its Crew task is gone"}
        usd = float(task.get("cost_usd") or 0.0)
        tokens = int(task.get("tokens_in") or 0) + int(task.get("tokens_out") or 0)
        status = task.get("status")
        if status in crew_store.ACTIVE:
            return {"state": "running", "usd": usd, "tokens": tokens}
        ev = task.get("evidence") or {}
        base = {"usd": usd, "tokens": tokens, "head_sha": str(ev.get("head") or ""),
                "commits": int(ev.get("commits") or 0), "dirty": int(ev.get("dirty") or 0),
                "note": str(task.get("note") or "")[:300]}
        if status == "ready":
            return {**base, "state": "done"}
        if status == "waiting":
            return {**base, "state": "failed", "note": "the worker stopped to ask something: " + base["note"]}
        return {**base, "state": "failed" if status in ("failed", "stopped") else "done"}

    async def stop(self, run: dict, exp: dict) -> None:
        from captain_claw.flight_deck import crew_runner
        user, _ = await self._user(run)
        with contextlib.suppress(Exception):
            await crew_runner.stop(user["id"], exp["child_id"], note="Stopped by its Captain Swarm run.")

    async def discard(self, run: dict, exp: dict) -> None:
        from captain_claw.flight_deck import crew_runner
        user, _ = await self._user(run)
        with contextlib.suppress(Exception):
            await crew_runner.close(user["id"], exp["child_id"], discard=True)


# ── what each kind of run brings ─────────────────────────────────────


async def no_seal(run: dict, base: str) -> None:
    """Nothing to seal: an outcome's checks are fresh looks at the pages themselves."""
    return None


async def git_start(run: dict) -> str:
    return await screen.resolve(run["objective"]["subject"]["repo_path"], "HEAD")


async def as_asked(run: dict, want: int) -> int:
    return want


async def crew_capacity(run: dict, want: int) -> int:
    """Crew runs as many of the owner's tasks at once as their grant allows; more would only queue."""
    from captain_claw.flight_deck import crew_access, crew_runner
    grant = crew_access.grant_of(await crew_runner._user_of(run["owner_id"]))
    return min(want, max(1, int(grant.max_parallel or 1)))


def parts_for(kind: str) -> dict[str, Any]:
    """What a kind of run brings: its lane, proposer, measurement, seal, screen,
    best-so-far marker and starting point; how many lanes it can have now
    (``capacity``), its measurement queue (``queue``), the reviewer's material
    (``review``), whether it has a quick look on a slice (``slice``) and whether
    an idea beaten in its wave comes back on the new best (``carry``). ``code``
    (Phase 2), ``practice`` (Phase 3a), ``outcome`` (Phase 3b) or ``document``
    (writing and research)."""
    if kind == "practice":
        from captain_claw.flight_deck import captain_swarm_rules as rules
        return {"lane": rules.RulesLane(), "propose": rules.propose, "measure": rules.measure, "seal": rules.seal,
                "screen": rules.screen, "branch": rules.no_branch, "start": rules.start_point,
                "capacity": as_asked, "queue": rules.measure_queue, "review": rules.review_material,
                "slice": True, "carry": True}
    if kind == "document":
        from captain_claw.flight_deck import captain_swarm_document as document
        from captain_claw.flight_deck import captain_swarm_outcome as outcome
        return {"lane": document.DocumentLane(), "propose": document.propose, "measure": document.measure,
                "seal": document.seal, "screen": document.screen, "branch": outcome.no_branch,
                "start": document.start_point, "capacity": outcome.capacity, "queue": lambda run: None,
                "review": document.review_material, "slice": False, "carry": False}
    if kind == "outcome":
        from captain_claw.flight_deck import captain_swarm_outcome as outcome
        return {"lane": outcome.BatLane(), "propose": outcome.propose, "measure": outcome.measure,
                "seal": no_seal, "screen": outcome.screen, "branch": outcome.no_branch, "start": outcome.start_point,
                "capacity": outcome.capacity, "queue": lambda run: None, "review": outcome.review_material,
                "slice": False, "carry": False}
    return {"lane": CrewLane(), "propose": propose, "measure": measure_commit, "seal": seal,
            "screen": screen_commit, "branch": move_champion_branch, "start": git_start,
            "capacity": crew_capacity, "queue": lambda run: ("deck:command", measure_parallel()),
            "review": code_review_material, "slice": False, "carry": True}


# ── the driver ───────────────────────────────────────────────────────


def report_of(run: dict, reason: str) -> dict:
    """Where a run ended up: the starting point against the best, on both sets."""
    base, champ = run.get("baseline") or {}, run.get("champion") or {}
    out: dict[str, Any] = {"reason": reason, "experiments": run["experiments"],
                           "spent_usd": round(run["spent_usd"], 4), "spent_tokens": run.get("spent_tokens", 0),
                           "champion_sha": champ.get("sha", "")}
    if run["objective"].get("kind", "code") == "code":
        out["champion_branch"] = champion_branch(run["id"])
    for part in ("dev", "holdout"):
        b, c = (base.get(part) or {}).get("stats"), (champ.get(part) or {}).get("stats")
        if b and c:
            out[part] = {"baseline": round(b["mean"], 4), "champion": round(c["mean"], 4),
                         "gain": round(c["mean"] - b["mean"], 4)}
    out["improved"] = bool(champ.get("exp"))
    return out


def wave_of(state: dict) -> list[str]:
    """The attempts under way: a wave. A run from before waves had one, ``current``."""
    return list(state.get("wave") or ([state["current"]] if state.get("current") else []))


def label(exp: dict) -> str:
    return f"#{exp['round']}.{exp['n']}"


class _LostLease(Exception):
    pass


class _StopError(Exception):
    """The whole run ends from inside one attempt: a limit is reached, or the lane can't start."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


class Driver:
    """Drives one run until it stops, or until it has to stand down."""

    def __init__(self, store: CaptainSwarmStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self.lost = False
        self._lock = asyncio.Lock()

    def _bind(self, run: dict) -> None:
        """This run's kind's parts, with any seam a test set in their place."""
        kind = str(run["objective"].get("kind") or "code")
        parts = parts_for(kind)
        self.kind = kind
        self.lane: Lane = _LANE or parts["lane"]
        self.propose: Proposer = _PROPOSER or parts["propose"]
        self.measure: Measure = _MEASURE or parts["measure"]
        self.seal: Seal = _SEAL or parts["seal"]
        self.screen: Screen = _SCREEN or parts["screen"]
        self.branch: Branch = _BRANCH or parts["branch"]
        self.start: Callable[[dict], Awaitable[str]] = _START or parts["start"]
        self.review: Review = _REVIEW or review
        self.strategy: Strategy = _STRATEGY or learning.strategy_round
        self._no_ideas = ""
        # A test's own lane has no grants to count.
        self.capacity: Callable[[dict, int], Awaitable[int]] = parts["capacity"] if _LANE is None else as_asked
        self.slice, self.carry = bool(parts["slice"]), bool(parts["carry"])

    # small helpers

    async def _run(self) -> dict:
        run = await self.store.get_run(self.run_id)
        if run is None:
            raise _LostLease()
        return run

    async def _event(self, stage: str, message: str, detail: str = "") -> None:
        await self.store.append_event(self.run_id, stage, message, detail)

    async def _state(self, **changes: Any) -> dict:
        """Change some of the run's state. Attempts of one wave run side by side,
        so each change is made on the state as it is now, one at a time."""
        async with self._lock:
            run = await self._run()
            state = {**(run.get("state") or {}), **changes}
            await self.store.update_run(self.run_id, state=state)
        return state

    async def _exps(self, ids: list[str]) -> list[dict]:
        return [e for e in [await self.store.get_experiment(i) for i in ids] if e]

    def _limit_reason(self, run: dict) -> str:
        b = run["objective"]["budget"]
        spent = objective_mod.budget_spent(run)
        if spent:
            return spent
        if run["started_at"] and time.time() - run["started_at"] >= b["hours"] * 3600:
            return f"the time limit ({b['hours']:g} h) is up"
        return ""

    async def _check(self) -> dict:
        """Between steps: a lost lease or a cancel ends the drive."""
        if self.lost:
            raise _LostLease()
        run = await self._run()
        if run["status"] not in ACTIVE_STATES:
            raise _LostLease()
        return run

    async def _finish(self, status: str, reason: str) -> str:
        await self._event("stop", reason)
        await self.store.set_status(self.run_id, status, reason)
        await self._state(report=report_of(await self._run(), reason))
        rehearsal.release_sites(self.run_id)
        await remember(self.store, self.run_id)
        return status

    async def _measured(self, exp_id: str, sha: str, which: str) -> dict:
        """One measurement, in its turn in the queue ("" is the best itself)."""
        run = await self._check()
        async with measurement_slot(run, exp_id):
            return await self.measure(await self._check(), sha, which)

    async def _together(self, coros: list[Awaitable[None]]) -> None:
        """Run *coros* side by side. The first to raise ends the others — a lost
        lease first, then a stop — and its exception is raised here."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        if not tasks:
            return
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        errors = [t.exception() for t in done if not t.cancelled() and t.exception() is not None]
        if not errors:
            return
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise (next((e for e in errors if isinstance(e, _LostLease)), None)
               or next((e for e in errors if isinstance(e, _StopError)), None) or errors[0])

    # the drive

    async def drive(self) -> str:
        hb = asyncio.create_task(self._heartbeat())
        try:
            run = await self._run()
            self._bind(run)
            if run["status"] == "baseline":
                outcome = await self._baseline(run)
                if outcome:
                    return outcome
            return await self._loop()
        except _LostLease:
            run = await self.store.get_run(self.run_id)
            if run and run["status"] == "cancelled":
                await self._halt(wave_of(run.get("state") or {}), "the run was stopped")
            return run["status"] if run else "gone"
        except Exception as exc:  # noqa: BLE001 — a crash must leave a reason, not a stuck run
            log.warning("captain swarm: the run failed", run_id=self.run_id, exc_info=True)
            with contextlib.suppress(Exception):
                await self._halt(wave_of((await self._run()).get("state") or {}), "the run broke")
            return await self._finish("failed", f"it broke: {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            hb.cancel()

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_S)
            if not await self.store.heartbeat_lease(self.run_id):
                self.lost = True
                return

    async def _halt(self, ids: list[str], reason: str) -> None:
        """Every attempt in *ids* still going is stopped and closed as failed."""
        run = await self._run()
        for exp in await self._exps(ids):
            if exp["status"] not in LIVE_EXP:
                continue
            if exp["status"] == "running" and exp["child_id"]:
                with contextlib.suppress(Exception):
                    await self.lane.stop(run, exp)
            await self.store.update_experiment(exp["id"], status="failed", verdict_reason=reason[:1000])
            rehearsal.release_sites(self.run_id, exp["id"])

    async def _baseline(self, run: dict) -> str:
        obj = run["objective"]
        state = run.get("state") or {}
        base_sha = state.get("base_sha") or await self.start(run)
        if not base_sha:
            return await self._finish("failed", "there is nothing to start from")
        await self._state(base_sha=base_sha, round=0, queue=[], wave=[], current="", since_keep=0,
                          kept_this_round=False)
        if not run["started_at"]:
            await self.store.update_run(self.run_id, started_at=time.time())
        await self._event("baseline", f"sealing the scorer at {base_sha[:12]}")
        await self.seal(run, base_sha)
        await self._event("baseline", "measuring the starting point on the open cases")
        dev = await self._measured("", base_sha, "dev")
        if dev.get("value") is None:
            return await self._finish("failed", f"the starting point couldn't be measured: {dev.get('error')}")
        await self._event("baseline", f"open cases: {fmt(dev['stats']['mean'])} ± {dev['stats']['stdev']:.3f}")
        hold = await self._measured("", base_sha, "holdout")
        if hold.get("value") is None:
            return await self._finish("failed", f"the held-out cases couldn't be measured: {hold.get('error')}")
        await self._event("baseline", f"held-out cases: {fmt(hold['stats']['mean'])} ± {hold['stats']['stdev']:.3f}")
        point = {"sha": base_sha, "dev": dev, "holdout": hold, "exp": ""}
        await self.store.update_run(self.run_id, baseline=point, champion=point, status="running")
        await self.branch(await self._run(), base_sha)
        if target_met(obj, dev):
            return await self._finish("done", "the starting point already meets the target")
        return ""

    async def _loop(self) -> str:
        while True:
            run = await self._check()
            obj = run["objective"]
            state = run.get("state") or {}
            wave = wave_of(state)
            if wave:
                outcome = await self._wave(wave)
                if outcome:
                    return outcome
                continue
            why = self._limit_reason(run)
            if why:
                return await self._finish("stopped", why)
            room = int(obj["budget"]["max_experiments"]) - int(run["experiments"])
            if room <= 0:
                return await self._finish("stopped", f"the experiment limit ({obj['budget']['max_experiments']}) is reached")
            queue = list(state.get("queue") or [])
            if not queue:
                outcome = await self._new_round(run)
                if outcome:
                    return outcome
                continue
            want = max(1, min(int(obj["lane"].get("parallel") or 1), objective_mod.max_lanes()))
            size = min(await self.capacity(run, want), room)
            if size < 1:
                if not state.get("waiting_for_lanes"):
                    await self._event("wait", "no lane is free right now (other runs are using them): waiting")
                    await self._state(waiting_for_lanes=True)
                await asyncio.sleep(_WAIT_LANES_S)
                continue
            await self._start_wave(run, queue, size)

    async def _start_wave(self, run: dict, queue: list[dict], size: int) -> None:
        """The next ideas that can run at once, as experiments started from the best."""
        state = run.get("state") or {}
        picked, rest = pick_wave(queue, size, rehearsal.per_site())
        champ = run.get("champion") or {}
        n = int(state.get("next_n") or 1)
        ids, made = [], []
        for idea in picked:
            sites = sites_of(idea)
            change = {**(idea.get("change") or {}), **({"sites": sites} if sites else {})}
            exp = await self.store.add_experiment(
                exp_id=f"exp_{secrets.token_hex(6)}", run_id=self.run_id, round_no=int(state.get("round") or 0),
                n=n, hypothesis=idea["hypothesis"], strategy_tag=idea.get("strategy_tag", ""), why=idea.get("why", ""),
                lane=run["objective"]["lane"]["kind"], base_sha=champ["sha"], change=change or None)
            ids.append(exp["id"])
            made.append(exp)
            n += 1
        await self._state(queue=rest, wave=ids, current="", next_n=n, waiting_for_lanes=False)
        for exp in made:
            sites = sites_of(exp)
            await self._event("propose", f"{label(exp)}: {exp['hypothesis'][:200]}"
                              + (f" (sites: {', '.join(sites)})" if sites else ""))
        if len(made) > 1:
            held = f"; {len(rest)} wait{'s' if len(rest) == 1 else ''} for a later wave" if rest else ""
            await self._event("wave", f"{len(made)} attempts at once, all from the best so far{held}")

    async def _new_round(self, run: dict) -> str:
        obj = run["objective"]
        state = run.get("state") or {}
        round_no = int(state.get("round") or 0)
        since_keep = int(state.get("since_keep") or 0)
        strategised = bool(state.get("strategised"))
        if round_no >= 1:
            if state.get("kept_this_round"):
                since_keep, strategised = 0, False     # a gain: the next stall earns a strategy round again
            else:
                since_keep += 1
            stall = int(obj["stop"]["stall_rounds"])
            if since_keep >= stall and not strategised:
                # Stalled: before giving up, one strategy round, then two more rounds (§9).
                await self._strategy_round(run, f"{since_keep} round{'s' if since_keep != 1 else ''} in a row "
                                                "brought no gain")
                strategised = True
            elif since_keep >= stall + 2:
                await self._state(since_keep=since_keep)
                return await self._finish("stalled", f"{since_keep} rounds in a row brought no gain, a strategy "
                                                     "round included")
        round_no += 1
        if round_no > 1 and (round_no - 1) % obj["stop"]["remeasure_every"] == 0:
            outcome = await self._remeasure(run)
            if outcome:
                return outcome
            run = await self._run()
        await self._event("round", f"round {round_no}: asking for ideas")
        ideas = await self._ideas(run)
        if ideas is None:
            return await self._finish("failed", f"no ideas: {self._no_ideas}")
        if not ideas and not strategised:
            # Nothing new to try is a stall too: a strategy round, and one more ask.
            await self._strategy_round(run, "the planner had nothing new to try")
            strategised = True
            ideas = await self._ideas(run)
            if ideas is None:
                return await self._finish("failed", f"no ideas: {self._no_ideas}")
        if not ideas:
            return await self._finish("stalled", "the planner had nothing new to try")
        ideas = await learning.rank(self.store, run, ideas)
        await self._state(round=round_no, queue=ideas, next_n=1, since_keep=since_keep, kept_this_round=False,
                          strategised=strategised)
        return ""

    async def _ideas(self, run: dict) -> list[dict] | None:
        """This round's ideas, with what was learned in the proposer's brief.
        None when the proposer failed (why: ``self._no_ideas``)."""
        run = await self._run()
        ledger = await self.store.list_experiments(self.run_id)
        try:
            run = {**run, "learned": await learning.brief_block(self.store, run)}
        except Exception:  # noqa: BLE001 — a lesson that can't be read never stops a round
            log.warning("captain swarm: couldn't read what earlier runs learned", run_id=self.run_id, exc_info=True)
        try:
            return await self.propose(run, ledger, int(run["objective"]["lane"]["per_round"]))
        except Exception as exc:  # noqa: BLE001
            self._no_ideas = str(exc)[:300]
            return None

    async def _strategy_round(self, run: dict, why: str) -> None:
        """A stalled run, read three ways; the directions go into the next briefs."""
        await self._event("strategy", f"{why}: a strategy round before giving up")
        try:
            directions = await self.strategy(await self._run(), await self.store.list_experiments(self.run_id))
            await learning.keep_directions(self.store, run, directions)
        except Exception as exc:  # noqa: BLE001 — without it the run still gets its two more rounds
            log.warning("captain swarm: the strategy round failed", run_id=self.run_id, exc_info=True)
            await self._event("strategy", f"the strategy round couldn't be done: {str(exc)[:200]}")
            return
        for d in directions:
            await self._event("strategy", f"new direction: {d['direction'][:300]}")
        if not directions:
            await self._event("strategy", "the strategy round named no new direction")

    async def _remeasure(self, run: dict) -> str:
        """The best, measured again: drift (a model update, a warm cache) must not
        be credited to anyone. Big drift means the number can't be trusted."""
        champ = run.get("champion") or {}
        await self._event("remeasure", f"measuring the best ({champ['sha'][:10]}) again")
        fresh = await self._measured("", champ["sha"], "dev")
        if fresh.get("value") is None:
            return await self._finish("failed", f"the best couldn't be measured again: {fresh.get('error')}")
        old = Stats(**champ["dev"]["stats"])
        drift = abs(fresh["stats"]["mean"] - old.mean)
        if drift > max(3 * max(old.stdev, fresh["stats"]["stdev"]), run["objective"]["noise"]["min_gain"]):
            return await self._finish("stopped", f"the measurement drifted by {drift:.3f} on the same code: "
                                                 "the number can't be trusted right now")
        await self.store.update_run(self.run_id, champion={**champ, "dev": fresh})
        await self._event("remeasure", f"the best now reads {fresh['stats']['mean']:.3f} ± {fresh['stats']['stdev']:.3f}")
        return ""

    # one wave

    async def _wave(self, ids: list[str]) -> str:
        """Carry one wave to its verdict, from wherever it stands (a restart
        resumes it; a measurement it was in the middle of is run again)."""
        try:
            await self._together([self._attempt(i) for i in ids])
            if self.slice and len(await self._ready(ids)) >= 3:
                await self._quick_look(ids)
            await self._together([self._measure_open(e["id"]) for e in await self._ready(ids)])
        except _StopError as stop:
            await self._halt(ids, f"stopped: {stop.reason}")
            return await self._finish(stop.status, stop.reason)
        return await self._verdict(ids)

    async def _ready(self, ids: list[str]) -> list[dict]:
        return [e for e in await self._exps(ids) if e["status"] == "measuring"]

    async def _attempt(self, exp_id: str) -> None:
        """One attempt to the end of its lane and through the screen: it ends
        ready to measure, or with a verdict."""
        exp = await self.store.get_experiment(exp_id)
        run = await self._check()
        if exp is None:
            return
        if exp["status"] == "proposed":
            await self._start_lane(run, exp)
            exp = await self.store.get_experiment(exp_id)
        if exp["status"] == "running":
            await self._wait_lane(exp)
            exp = await self.store.get_experiment(exp_id)
        if exp["status"] == "screening":
            got = await self.screen(await self._check(), exp)
            await self.store.update_experiment(exp_id, changed_files=got.get("changed", []), diff_stat=got.get("diff_stat", ""))
            if not got["ok"]:
                await self._close(exp, "disqualified", got["reason"])
                return
            await self.store.update_experiment(exp_id, status="measuring")

    async def _start_lane(self, run: dict, exp: dict) -> None:
        # The budget left is shared by the wave's attempts, each with an equal part.
        left = objective_mod.tokens_left(run)
        share = {} if left is None else {"token_share": left // max(1, len(wave_of(run.get("state") or {})))}
        try:
            brief = lane_brief(run, exp) if self.kind == "code" else ""
            got = await self.lane.start(run, {**exp, **share}, brief)
        except Exception as exc:  # noqa: BLE001
            msg = str(getattr(exc, "detail", "") or exc)[:300]
            same = ((await self._run()).get("state") or {}).get("last_start_error") == msg
            await self._state(last_start_error=msg)
            await self._close(exp, "failed", f"the lane didn't start: {msg}")
            # The same refusal twice is the setup, not the idea: more ideas won't help.
            if same:
                raise _StopError("failed", f"the lane can't start: {msg}") from exc
            return
        await self._state(last_start_error="")
        await self.store.update_experiment(exp["id"], status="running", child_id=got["child_id"], branch=got.get("branch", ""))

    async def _wait_lane(self, exp: dict) -> None:
        counted_usd, counted_tokens = float(exp["cost_usd"]), int(exp["tokens"])
        while True:
            run = await self._check()
            why = self._limit_reason(run)
            if why:
                raise _StopError("stopped", why)
            got = await self.lane.poll(run, exp)
            usd, tokens = float(got.get("usd") or 0.0), int(got.get("tokens") or 0)
            if usd > counted_usd or tokens > counted_tokens:
                await self.store.add_spend(self.run_id, max(0.0, usd - counted_usd), max(0, tokens - counted_tokens))
                counted_usd, counted_tokens = max(usd, counted_usd), max(tokens, counted_tokens)
                await self.store.update_experiment(exp["id"], cost_usd=counted_usd, tokens=counted_tokens)
            if got["state"] == "running":
                await asyncio.sleep(_POLL_S)
                continue
            rehearsal.release_sites(self.run_id, exp["id"])
            if got["state"] == "failed":
                await self._close(exp, "failed", got.get("note") or "the worker failed")
                return
            if not got.get("head_sha") or not got.get("commits"):
                await self._close(exp, "failed", "it committed nothing")
                return
            note = f" (it left {got['dirty']} uncommitted change(s), not measured)" if got.get("dirty") else ""
            await self.store.update_experiment(exp["id"], status="screening", head_sha=got["head_sha"])
            await self._event("lane", f"{label(exp)}: the worker committed {got['head_sha'][:10]}{note}")
            return

    async def _quick_look(self, ids: list[str]) -> None:
        """Stage 1 (§10): each ready attempt on a slice of the open cases; only
        the better half goes on to the full measurement."""

        async def look(exp: dict) -> None:
            if (exp.get("dev") or {}).get("stage") == "slice":
                return
            await self._event("measure", f"{label(exp)}: a quick look at {exp['head_sha'][:10]} on a slice of the open cases")
            got = await self._measured(exp["id"], exp["head_sha"], "slice")
            if got.get("value") is None:
                await self._close(exp, "failed", f"couldn't measure it: {got.get('error')}")
                return
            await self.store.update_experiment(exp["id"], dev={**got, "stage": "slice"})

        await self._together([look(e) for e in await self._ready(ids)])
        looked = [e for e in await self._ready(ids) if (e.get("dev") or {}).get("stage") == "slice"]
        if len(looked) < 2:
            return
        best_first = sorted(looked, key=lambda e: float(e["dev"]["value"]),
                            reverse=(await self._run())["objective"]["metric"]["direction"] == "max")
        lead = best_first[0]
        for exp in best_first[(len(looked) + 1) // 2:]:
            await self._close(exp, "discarded", f"in the slower half on a quick look: {fmt(exp['dev']['value'])} "
                                                f"against {label(lead)}'s {fmt(lead['dev']['value'])}")

    async def _measure_open(self, exp_id: str) -> None:
        exp = await self.store.get_experiment(exp_id)
        if not exp or exp["status"] != "measuring":
            return
        await self._event("measure", f"{label(exp)}: measuring {exp['head_sha'][:10]} on the open cases")
        dev = await self._measured(exp_id, exp["head_sha"], "dev")
        if dev.get("value") is None:
            await self._close(exp, "failed", f"couldn't measure it: {dev.get('error')}")
            return
        quick = exp.get("dev") or {}
        if quick.get("stage") == "slice":
            dev = {**dev, "slice": quick.get("value")}
        await self.store.update_experiment(exp_id, dev=dev, guardrails=dev.get("guard") or {}, status="measured")

    async def _verdict(self, ids: list[str]) -> str:
        """One keeper per wave (§8): of the attempts that beat the best on the open
        cases, the strongest that the reviewer passes and the held-out cases
        confirm. The others that beat it come back once, on top of the new best."""
        run = await self._check()
        obj, champ = run["objective"], run["champion"]
        passing: list[tuple[dict, str]] = []
        for exp in await self._exps(ids):
            if exp["status"] != "measured":
                continue
            ok, why = judge_dev(obj, exp["dev"], champ["dev"])
            if ok:
                passing.append((exp, why))
            else:
                await self._close(exp, "discarded", why, lesson=self._lesson(exp["dev"], champ["dev"], None))
        passing.sort(key=lambda p: float(p[0]["dev"]["stats"]["mean"]), reverse=obj["metric"]["direction"] == "max")
        kept: dict | None = None
        met = False
        for exp, why in passing:
            if kept is not None:
                await self._runner_up(exp, kept, why, champ["dev"])
                continue
            verdict = await self._review(await self._check(), exp)
            if not verdict["ok"]:
                if verdict.get("error"):
                    await self._close(exp, "failed", f"the review couldn't be done: {verdict['reason']}")
                else:
                    await self._close(exp, "disqualified", f"the reviewer: {verdict['reason']}")
                continue
            await self._event("review", f"{label(exp)}: the reviewer passed it: {verdict['reason'][:240]}")
            await self._event("measure", f"{label(exp)}: {why}; checking the held-out cases")
            hold = await self._measured(exp["id"], exp["head_sha"], "holdout")
            if hold.get("value") is None:
                await self._close(exp, "failed", f"couldn't measure it on the held-out cases: {hold.get('error')}")
                continue
            await self.store.update_experiment(exp["id"], holdout={"stats": hold["stats"], "value": hold["value"]})
            run = await self._check()
            held, hwhy = judge_holdout(obj, hold, run["champion"]["holdout"])
            lesson = self._lesson(exp["dev"], champ["dev"], held)
            if not held:
                await self._close(exp, "discarded", f"{why}, but {hwhy}", lesson=lesson)
                continue
            met = await self._keep(run, await self.store.get_experiment(exp["id"]), exp["dev"], hold,
                                   f"{why}; {hwhy}", lesson)
            kept = exp
        await self._state(wave=[], current="")
        if met:
            return await self._finish("done", "the target is met, and the held-out cases confirm it")
        return ""

    async def _review(self, run: dict, exp: dict) -> dict:
        try:
            got = await self.review(run, exp)
        except Exception as exc:  # noqa: BLE001 — no review is no pass
            return {"ok": False, "error": True, "reason": str(exc)[:300]}
        return {"ok": bool(got.get("ok")), "reason": str(got.get("reason") or "")[:400]}

    async def _runner_up(self, exp: dict, kept: dict, why: str, base_dev: dict) -> None:
        """Beaten in its wave by the keeper, but a gain of its own: it comes back
        once, on top of the new best (not for an outcome: there is nothing to build on)."""
        again = self.carry and not (exp.get("change") or {}).get("again_of")
        await self._close(exp, "discarded", f"{why}, but {label(kept)} {learning.RUNNER_UP}"
                          + ("; it comes back on top of that one" if again else ""),
                          lesson=self._lesson(exp["dev"], base_dev, None))
        if again:
            idea = {"hypothesis": exp["hypothesis"], "strategy_tag": exp["strategy_tag"], "why": exp["why"],
                    "files": [], "change": {**(exp.get("change") or {}), "again_of": exp["id"]}}
            queue = ((await self._run()).get("state") or {}).get("queue") or []
            await self._state(queue=[idea, *queue])

    def _lesson(self, dev: dict, champ_dev: dict, held: bool | None) -> str:
        if fmt(champ_dev["stats"]["mean"]) == "—" or fmt(dev["stats"]["mean"]) == "—":
            held_s = "" if held is None else ("; confirmed" if held else "; not confirmed")
            return f"{fmt(dev['stats']['mean'])} where there was {fmt(champ_dev['stats']['mean'])}{held_s}"
        d = dev["stats"]["mean"] - champ_dev["stats"]["mean"]
        g = ", ".join(f"{k} {v:g}" for k, v in (dev.get("guard") or {}).items() if isinstance(v, (int, float)))
        held_s = "" if held is None else ("; held-out confirmed" if held else "; held-out did not confirm")
        return f"{d:+.3f} on the open cases{held_s}" + (f" ({g})" if g else "")

    async def _close(self, exp: dict, status: str, reason: str, *, lesson: str = "") -> None:
        await self.store.update_experiment(exp["id"], status=status, verdict_reason=reason[:1000], lesson=lesson)
        await self._event(status, f"{label(exp)}: {reason[:300]}")
        rehearsal.release_sites(self.run_id, exp["id"])
        if exp.get("child_id"):
            await self.lane.discard(await self._run(), exp)

    async def _keep(self, run: dict, exp: dict, dev: dict, hold: dict, why: str, lesson: str) -> bool:
        """*exp* is the new best. Whether the target is met now."""
        previous = run.get("champion") or {}
        point = {"sha": exp["head_sha"], "dev": dev, "holdout": {"stats": hold["stats"], "value": hold["value"]},
                 "exp": exp["id"]}
        await self.store.update_experiment(exp["id"], status="kept", verdict_reason=why[:1000], lesson=lesson)
        await self.store.update_run(self.run_id, champion=point)
        await self.branch(await self._run(), exp["head_sha"])
        await self._event("kept", f"{label(exp)} is the new best: {why[:300]}")
        if previous.get("exp"):
            # Its commits live on in the new best, which was cut from it.
            old = await self.store.get_experiment(previous["exp"])
            if old and old.get("child_id"):
                await self.lane.discard(await self._run(), old)
        await self._state(kept_this_round=True)
        return target_met(run["objective"], dev)


# ── supervision ──────────────────────────────────────────────────────


class Supervisor:
    """Finds runs nobody is driving, takes their lease, and drives each in a task of its own."""

    def __init__(self, store: CaptainSwarmStore) -> None:
        self.store = store
        self.tasks: dict[str, asyncio.Task] = {}
        # One scan at a time: a kick() landing on the periodic scan would see the
        # run undriven twice, and this process's own lease doesn't stop it.
        self._scan = asyncio.Lock()

    async def tick(self) -> int:
        async with self._scan:
            return await self._tick()

    async def _tick(self) -> int:
        started = 0
        for run in await self.store.adoptable_runs():
            rid = run["id"]
            if rid in self.tasks and not self.tasks[rid].done():
                continue
            if not await self.store.claim_lease(rid):
                continue
            self.tasks[rid] = asyncio.create_task(self._drive(rid))
            started += 1
        return started

    async def _drive(self, run_id: str) -> None:
        try:
            await Driver(self.store, run_id).drive()
        finally:
            with contextlib.suppress(Exception):
                await self.store.release_lease(run_id)

    async def stop(self) -> None:
        for t in self.tasks.values():
            t.cancel()


async def kick() -> int:
    return await _SUPERVISOR.tick() if _SUPERVISOR is not None else 0


async def remember(store: CaptainSwarmStore, run_id: str) -> None:
    """Write down what a run that has ended taught; never in the way of its end."""
    try:
        run = await store.get_run(run_id)
        if run is not None:
            await learning.record_run(store, run)
    except Exception:  # noqa: BLE001
        log.warning("captain swarm: couldn't write down what the run learned", run_id=run_id, exc_info=True)


async def cancel_run(store: CaptainSwarmStore, run_id: str) -> None:
    """The owner's stop: the run ends now, and every lane of its wave is told to stop."""
    run = await store.get_run(run_id)
    if run is None:
        return
    await store.set_status(run_id, "cancelled", "stopped by the owner")
    await store.append_event(run_id, "stop", "stopped by the owner")
    await store.update_run(run_id, state={**(run.get("state") or {}), "report": report_of(run, "stopped by the owner")})
    lane = _LANE or parts_for(str(run["objective"].get("kind") or "code"))["lane"]
    for exp_id in wave_of(run.get("state") or {}):
        exp = await store.get_experiment(exp_id)
        if not exp or exp["status"] not in LIVE_EXP:
            continue
        await store.update_experiment(exp["id"], status="failed", verdict_reason="the run was stopped")
        if exp["child_id"]:
            with contextlib.suppress(Exception):
                await lane.stop(run, exp)
    rehearsal.release_sites(run_id)
    await remember(store, run_id)


async def supervisor_loop(store: CaptainSwarmStore, stop_event: asyncio.Event) -> None:
    global _SUPERVISOR
    sup = Supervisor(store)
    _SUPERVISOR = sup
    try:
        while not stop_event.is_set():
            try:
                await sup.tick()
            except Exception:  # noqa: BLE001
                log.warning("captain swarm supervisor tick failed", exc_info=True)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=_TICK_S)
    finally:
        await sup.stop()
        _SUPERVISOR = None
