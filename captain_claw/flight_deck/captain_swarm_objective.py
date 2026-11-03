"""Captain Swarm — the objective: what a run improves, how it is measured, what it may spend.

docs/captain-swarm-plan.md §3.1. The objective is pinned when a run starts and
an attempt never changes it. This module checks the code kind: measured by a
command Captain runs on the deck (so only an owner with a shell may start
one), improved by Crew, each attempt in a worktree of its own — one at a time,
or several at once (``lane.parallel``, Phase 4).

A command names its places with placeholders, filled in for each measurement:

    {python}    this deck's Python
    {harness}   the sealed copy of the scorer (the base commit's, never an attempt's)
    {checkout}  a clean copy of the commit being measured
    {dev}       the open cases, from the sealed copy
    {holdout}   the held-out cases, copied into the sealed folder at start

:func:`normalize` turns what a person or an agent sent into a complete,
checked objective, or raises :class:`ObjectiveError` saying what's wrong.
"""

from __future__ import annotations

import copy
import os
import re
import sys
from pathlib import Path
from typing import Any

HARNESSES = ("claude", "codex", "opencode", "crush", "pi")
_PLACES = ("python", "harness", "checkout", "dev", "holdout")
_READ_RE = re.compile(r"^[A-Za-z0-9_.]{1,80}$")

# A run's limits. The money one is in tokens — what a person enters and Captain
# counts — never dollars: a token is a token on any model, a dollar isn't.
DEFAULT_BUDGET = {"tokens": 200_000_000, "hours": 8.0, "max_experiments": 40}
_LIMITS = {"tokens": (100_000, 10_000_000_000), "hours": (0.25, 72.0), "max_experiments": (1, 200)}
_ATTEMPT_LIMITS = (100_000, 1_000_000_000)   # what one Bat attempt may use
_TOKENS_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*([kKmMbBgG]?)\s*$")
_TOKEN_UNITS = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "g": 1_000_000_000}


class ObjectiveError(ValueError):
    """What's wrong with an objective, in words for the person who sent it."""


def env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, "") or default)))
    except ValueError:
        return default


def max_lanes() -> int:
    """How many attempts one run may have going at once (§10): FD_CAPTAIN_SWARM_MAX_LANES, 4."""
    return env_int("FD_CAPTAIN_SWARM_MAX_LANES", 4, 1, 8)


def lanes(raw_lane: dict, *, default: int, most: int, per_round_most: int) -> tuple[int, int]:
    """``(parallel, per_round)`` from a lane's settings: attempts at once — cut to
    what this deck and kind allow — and ideas asked for per round, never fewer
    than lanes."""
    parallel = int(_num(raw_lane.get("parallel", default), "Attempts at once", 1, 64))
    parallel = min(parallel, max(1, min(most, max_lanes())))
    per_round = int(_num(raw_lane.get("per_round", 2), "Attempts per round", 1, per_round_most))
    return parallel, max(per_round, parallel)


def mrav_template(repo_path: str, *, model: str = "gemma4:e2b", base_url: str = "http://127.0.0.1:11434",
                  harness: str = "claude", holdout_file: str = "") -> dict[str, Any]:
    """The first objective: Mrav, the small-model runtime, passes more of its eval.
    The dev cases live in the repo; the held-out ones in the deck's data dir."""
    cmd = ("{python} {harness}/scripts/mrav_eval.py --model " + model + " --base-url " + base_url
           + " --repeat 3 --isolate --code-root {checkout} --json --tasks ")
    return {
        "kind": "code",
        "family": "mrav",
        "title": f"Mrav on {model} passes more of its eval",
        "goal": ("Mrav is Captain's runtime for small models (captain_claw/mrav/). Make a small model "
                 "finish more of the eval's everyday tasks — files, counting, searching, editing, running "
                 "scripts, and saying so honestly when it can't — by improving Mrav's prompts, its tool "
                 "pack, its digest or its runtime."),
        "subject": {
            "repo_path": repo_path,
            "editable": ["captain_claw/mrav/**"],
            "protected": ["scripts/mrav_eval.py", "evals/**", "tests/**",
                          "captain_claw/flight_deck/captain_swarm_score.py",
                          "captain_claw/flight_deck/code_contract.py"],
            "snapshot": {"paths": ["captain_claw", "scripts", "evals", "pyproject.toml"],
                         "exclude": ["captain_claw/flight_deck/static"]},
        },
        "metric": {"kind": "command", "run": cmd + "{dev}", "read": "summary.pass_rate",
                   "spread": "summary.per_repeat", "direction": "max", "timeout_s": 7200,
                   "per_case": "summary.per_task", "dev_file": "evals/mrav/dev.json"},
        "holdout": {"run": cmd + "{holdout}", "file": holdout_file},
        "guardrails": [{"name": "tokens per task", "read": "summary.tokens_per_task", "max_increase": 0.25}],
        "noise": {"min_gain": 0.05},
        "lane": {"kind": "crew", "harness": harness, "per_round": 2, "parallel": 1},
        "budget": dict(DEFAULT_BUDGET),
        "stop": {"target": 1.0, "stall_rounds": 3, "remeasure_every": 3},
    }


TEMPLATES = {"mrav": mrav_template}


def parse_tokens(raw: Any) -> Any:
    """A token count from what a person typed — ``80M``, ``500k``, ``1.5m``,
    ``2_000_000`` or a number; anything else comes back as it was, for
    :func:`_num` to refuse in words."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw
    m = _TOKENS_RE.match(str(raw or "").replace("_", "").replace(" ", ""))
    if not m:
        return raw
    return float(m.group(1).replace(",", ".")) * _TOKEN_UNITS[m.group(2).lower()]


def tokens_count(raw: Any, name: str, lo: int, hi: int) -> int:
    return int(_num(parse_tokens(raw), name, lo, hi))


def tokens_text(n: Any) -> str:
    """``24.2M``, ``500k``, ``900`` — a token count the way a person reads it."""
    v = float(n or 0)
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= size:
            return f"{v / size:.1f}".rstrip("0").rstrip(".") + unit
    return str(int(v))


def budget_of(raw: Any, defaults: dict[str, Any], *, hours_most: float = 72.0,
              experiments_most: int = 200) -> dict[str, Any]:
    """A run's limits from what was sent, over *defaults*: tokens, hours and
    experiments. A dollar figure is not a limit any more and is ignored."""
    got = {k: v for k, v in (raw or {}).items() if v not in (None, "")} if isinstance(raw, dict) else {}
    b = {**defaults, **got}
    return {"tokens": tokens_count(b.get("tokens", DEFAULT_BUDGET["tokens"]), "The token budget", *_LIMITS["tokens"]),
            "hours": _num(b.get("hours", DEFAULT_BUDGET["hours"]), "The hours limit", 0.25, hours_most),
            "max_experiments": int(_num(b.get("max_experiments", DEFAULT_BUDGET["max_experiments"]),
                                        "The experiment limit", 1, experiments_most))}


def budget_spent(run: dict[str, Any]) -> str:
    """Why *run* has used its budget, or ``""``. A run started while budgets
    were in dollars keeps its dollar limit."""
    b = (run.get("objective") or {}).get("budget") or {}
    if b.get("tokens"):
        used = int(run.get("spent_tokens") or 0)
        if used >= int(b["tokens"]):
            return f"the token budget is used ({tokens_text(used)} of {tokens_text(b['tokens'])})"
        return ""
    if b.get("usd") and float(run.get("spent_usd") or 0.0) >= float(b["usd"]):
        return f"the budget is spent (${float(run['spent_usd']):.2f} of ${float(b['usd']):.2f})"
    return ""


def tokens_left(run: dict[str, Any]) -> int | None:
    """Tokens the run may still use; ``None`` for a run with a dollar budget."""
    b = (run.get("objective") or {}).get("budget") or {}
    if not b.get("tokens"):
        return None
    return max(0, int(b["tokens"]) - int(run.get("spent_tokens") or 0))


def attempt_tokens(lane: dict[str, Any], default: int) -> int:
    """The most tokens one Bat attempt may use (``lane.attempt_tokens``)."""
    return tokens_count(lane.get("attempt_tokens") or default, "The most tokens one attempt may use", *_ATTEMPT_LIMITS)


def _num(raw: Any, name: str, lo: float, hi: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ObjectiveError(f"{name} must be a number.") from None
    if not lo <= v <= hi:
        raise ObjectiveError(f"{name} must be between {lo:g} and {hi:g}.")
    return v


def _globs(raw: Any, name: str) -> list[str]:
    items = [str(x).strip() for x in (raw or []) if str(x).strip()]
    for g in items:
        if g.startswith("/") or ".." in g.split("/"):
            raise ObjectiveError(f"{name}: {g!r} must be a path inside the repository.")
    return items[:50]


def _command(raw: Any, name: str) -> str:
    cmd = str(raw or "").strip()
    if not cmd:
        raise ObjectiveError(f"{name} needs a command.")
    if "{checkout}" not in cmd:
        raise ObjectiveError(f"{name} must measure {{checkout}}, the commit being measured.")
    for place in re.findall(r"\{([a-z_]+)\}", cmd):
        if place not in _PLACES:
            raise ObjectiveError(f"{name}: unknown place {{{place}}}.")
    return cmd[:2000]


def normalize(raw: dict[str, Any], *, owner_has_shell: bool) -> dict[str, Any]:
    """A complete, checked copy of *raw*. Phase 2 runs code objectives measured
    by a command, and a command runs on the deck: only for an owner with a shell."""
    if not isinstance(raw, dict):
        raise ObjectiveError("An objective is an object.")
    obj = copy.deepcopy(raw)
    if obj.get("kind", "code") != "code":
        raise ObjectiveError("Only code objectives run so far (Phase 2).")
    obj["kind"] = "code"
    from captain_claw.flight_deck.captain_swarm_learning import family_slug
    obj["family"] = family_slug(obj.get("family")) or "code"
    obj["title"] = str(obj.get("title") or "").strip()[:200] or "Captain Swarm run"
    obj["goal"] = str(obj.get("goal") or "").strip()[:4000]

    subject = obj.get("subject") or {}
    repo = str(subject.get("repo_path") or "").strip()
    if not repo or not Path(repo).is_absolute():
        raise ObjectiveError("The repository is a full path on this computer.")
    if not (Path(repo) / ".git").exists():
        raise ObjectiveError(f"{repo} isn't a git repository.")
    editable = _globs(subject.get("editable"), "What attempts may change")
    if not editable:
        raise ObjectiveError("Name what attempts may change (editable paths).")
    protected = _globs(subject.get("protected"), "What attempts may never change")
    snap = subject.get("snapshot") or {}
    obj["subject"] = {"repo_path": repo, "editable": editable, "protected": protected,
                      "snapshot": {"paths": _globs(snap.get("paths"), "Snapshot paths"),
                                   "exclude": _globs(snap.get("exclude"), "Snapshot exclusions")}}

    metric = obj.get("metric") or {}
    if metric.get("kind", "command") != "command":
        raise ObjectiveError("Only command metrics run so far.")
    if not owner_has_shell:
        raise ObjectiveError("A command metric runs on this deck, so only an admin can start this kind of run.")
    read = str(metric.get("read") or "")
    if not _READ_RE.match(read):
        raise ObjectiveError("Say which number to read from the command's JSON (for example summary.pass_rate).")
    direction = str(metric.get("direction") or "max")
    if direction not in ("max", "min"):
        raise ObjectiveError("The direction is max or min.")
    obj["metric"] = {
        "kind": "command", "run": _command(metric.get("run"), "The metric"), "read": read,
        "spread": str(metric.get("spread") or "") if _READ_RE.match(str(metric.get("spread") or "")) else "",
        "per_case": str(metric.get("per_case") or "") if _READ_RE.match(str(metric.get("per_case") or "")) else "",
        "direction": direction, "timeout_s": _num(metric.get("timeout_s", 7200), "The metric's time limit", 60, 86400),
        "dev_file": (_globs([metric.get("dev_file")] if metric.get("dev_file") else [], "The open cases") or [""])[0],
        "env": [str(e) for e in (metric.get("env") or []) if re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", str(e))][:10],
    }

    hold = obj.get("holdout") or {}
    hold_file = str(hold.get("file") or "").strip()
    if hold_file:
        if not Path(hold_file).is_file():
            raise ObjectiveError("The held-out cases are a file on this deck, outside the repository.")
        if Path(hold_file).resolve().is_relative_to(Path(repo).resolve()):
            raise ObjectiveError("The held-out cases must live outside the repository, where attempts can't read them.")
        obj["holdout"] = {"run": _command(hold.get("run"), "The held-out measurement"), "file": hold_file}
    else:
        # No held-out cases (a benchmark, a size, a count of warnings): the check
        # of a gain is the same command run again on a fresh copy.
        run_again = str(hold.get("run") or "").strip() or obj["metric"]["run"]
        if "{holdout}" in run_again:
            raise ObjectiveError("The held-out measurement names {holdout}, but there is no held-out file.")
        obj["holdout"] = {"run": _command(run_again, "The held-out measurement"), "file": ""}

    guards = []
    for g in obj.get("guardrails") or []:
        if not isinstance(g, dict) or not _READ_RE.match(str(g.get("read") or "")):
            raise ObjectiveError("Each guardrail reads one number from the metric's JSON.")
        guards.append({"name": str(g.get("name") or g["read"])[:80], "read": str(g["read"]),
                       "max_increase": _num(g.get("max_increase", 0.25), "A guardrail's allowed increase", 0, 10)})
    obj["guardrails"] = guards[:10]
    obj["noise"] = {"min_gain": _num((obj.get("noise") or {}).get("min_gain", 0.05), "The smallest gain that counts", 0, 1e9)}

    lane = obj.get("lane") or {}
    harness = str(lane.get("harness") or "claude")
    if harness not in HARNESSES:
        raise ObjectiveError(f"The coding worker is one of {', '.join(HARNESSES)}.")
    src = lane.get("model_source") if isinstance(lane.get("model_source"), dict) else None
    parallel, per_round = lanes(lane, default=1, most=8, per_round_most=8)
    obj["lane"] = {"kind": "crew", "harness": harness, "model": str(lane.get("model") or "")[:120],
                   "model_source": src, "per_round": per_round, "parallel": parallel}

    obj["budget"] = budget_of(obj.get("budget"), DEFAULT_BUDGET)

    stop = obj.get("stop") or {}
    target = stop.get("target")
    obj["stop"] = {
        "target": None if target in (None, "") else _num(target, "The target", -1e12, 1e12),
        "stall_rounds": int(_num(stop.get("stall_rounds", 3), "Rounds without a gain", 1, 20)),
        "remeasure_every": int(_num(stop.get("remeasure_every", 3), "Re-measure the best every", 1, 50)),
    }
    return obj


def fill(command: str, **places: str) -> str:
    """*command* with its places filled in; a place left unfilled is an error."""
    out = command
    for name in _PLACES:
        if "{" + name + "}" in out:
            if not places.get(name):
                raise ObjectiveError(f"{{{name}}} has nothing to stand for here.")
            out = out.replace("{" + name + "}", str(places[name]))
    return out


def python() -> str:
    return sys.executable
