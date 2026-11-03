"""Captain Swarm — measurement: the scorers and the noise rule.

docs/captain-swarm-plan.md §5. Captain measures; an attempt never does. Every
scorer here is something Captain runs or computes itself, on what the attempt
left behind — never on what the attempt says about itself.

* :func:`stats`, :func:`margin`, :func:`beats` — the noise rule. A baseline is
  measured several times; a candidate must beat the champion by at least
  ``max(min_gain, 2 × stdev)`` before it counts as better.
* ``command`` (:func:`score_command`) — run a command, read one number from the
  JSON it prints. For code; it runs on the deck, so its caller decides who may
  use it (Bat's ``_owner_has_shell``) and screens it (``bat_floor``).
* ``predicate`` (:func:`check_predicates`) — facts about files and a reply, in
  ``code_contract``'s shapes (``file_exists``, ``file_contains``, ``no_pattern``)
  plus ``file_absent`` and the reply checks. Fails closed: a pattern that won't
  compile is a failed check, never a free pass.
* ``agreement`` (:func:`agreement`) — yes/no predictions against what the owner
  actually did. An unanswered case always counts against.
* ``owner`` (:func:`owner_score`) — the owner's thumbs.

Pure apart from the command's subprocess and the predicates' file reads.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.code_contract import _relpath

KINDS = ("command", "predicate", "agreement", "owner")

_READ_CAP = 2_000_000
_OUTPUT_CAP = 4_000_000
_SCAN_CAP = 5_000
_COMMAND_TIMEOUT_S = 3600.0


# ── the noise rule ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Stats:
    """Repeated measurements of one thing: their count, mean, sample standard
    deviation (0 for fewer than two) and range."""

    n: int
    mean: float
    stdev: float
    low: float
    high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stats(values: Iterable[float | None]) -> Stats:
    """The stats of the numbers in *values*; a missing one (None, NaN) is left out."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return Stats(0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return Stats(len(xs), mean, math.sqrt(var), min(xs), max(xs))


def margin(baseline: Stats, min_gain: float = 0.0) -> float:
    """How much better than *baseline* a candidate has to be to count."""
    return max(float(min_gain), 2.0 * baseline.stdev)


def beats(candidate: Stats, champion: Stats, *, min_gain: float = 0.0, direction: str = "max") -> bool:
    """Does *candidate* beat *champion* by more than noise? Noise is the larger
    of the two spreads, so a candidate measured once can't hide behind a
    steady champion."""
    if candidate.n == 0 or champion.n == 0:
        return False
    need = max(margin(champion, min_gain), 2.0 * candidate.stdev)
    gain = candidate.mean - champion.mean if direction == "max" else champion.mean - candidate.mean
    return gain >= need and gain > 0


# ── command ──────────────────────────────────────────────────────────


def read_path(data: Any, key: str) -> Any:
    """The value at dotted *key* in *data*, or None."""
    cur = data
    for part in [p for p in key.split(".") if p]:
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def read_number(data: Any, key: str) -> float | None:
    """The number at dotted *key* in *data* (``summary.pass_rate``), or None."""
    cur = data
    for part in [p for p in key.split(".") if p]:
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    return float(cur) if not math.isnan(float(cur)) else None


def last_json(text: str) -> dict | None:
    """The JSON object a command printed: the whole output, else the last line
    that is one (progress may come before it)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


async def score_command(command: Sequence[str] | str, cwd: str | Path, *, read: str,
                        env: Mapping[str, str] | None = None, env_only: bool = False,
                        timeout: float = _COMMAND_TIMEOUT_S) -> dict[str, Any]:
    """Run *command* in *cwd* and read the number at *read* from the JSON it
    prints: ``{"value", "data", "exit_code", "error"}``. ``value`` is None when
    the command failed to give one; a non-zero exit with a number still counts
    (an eval that fails a task exits 1 by design). With *env_only*, *env* is the
    whole environment: code under test never sees the deck's own secrets."""
    shell = isinstance(command, str)
    full_env = dict(env or {}) if env_only else {**os.environ, **(env or {})}
    try:
        if shell:
            proc = await asyncio.create_subprocess_shell(
                command, cwd=str(cwd), env=full_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        else:
            proc = await asyncio.create_subprocess_exec(
                *command, cwd=str(cwd), env=full_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except OSError as exc:
        return {"value": None, "data": None, "exit_code": None, "error": f"couldn't start: {exc}"}
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return {"value": None, "data": None, "exit_code": None, "error": f"no result in {int(timeout)} s"}
    stdout = out[:_OUTPUT_CAP].decode("utf-8", errors="replace")
    data = last_json(stdout)
    value = read_number(data, read) if data is not None else None
    error = ""
    if value is None:
        tail = err[-600:].decode("utf-8", errors="replace").strip()
        error = (f"no number at {read!r} in its output" if data is not None else "it printed no JSON") \
            + (f": {tail}" if tail else "")
    return {"value": value, "data": data, "exit_code": proc.returncode, "error": error}


# ── predicate ────────────────────────────────────────────────────────


def _find(root: Path, rel: str, anywhere: bool) -> Path | None:
    """*rel* under *root*; with *anywhere*, else the first file of that name
    anywhere below it (an agent's write tool may put it in a subfolder)."""
    direct = root / rel
    if direct.is_file():
        return direct
    if not anywhere:
        return None
    name = Path(rel).name
    for i, p in enumerate(root.rglob(name)):
        if i > _SCAN_CAP:
            break
        if p.is_file():
            return p
    return None


def _compile(pattern: str, flags: int = 0) -> re.Pattern | None:
    try:
        return re.compile(pattern, flags | re.MULTILINE)
    except re.error:
        return None


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
    except OSError:
        return ""


def check_one(root: Path, reply: str, check: Mapping[str, Any], *, anywhere: bool = False) -> tuple[bool, str]:
    """One predicate: ``(passed, why it failed)``."""
    kind = str(check.get("type") or "").strip()
    rel = _relpath(check.get("path"))
    pattern = str(check.get("pattern") or "")
    if kind in ("file_exists", "file_absent", "file_contains") and not rel:
        return False, f"{kind}: no usable path"
    if kind == "file_exists":
        return (_find(root, rel, anywhere) is not None), f"`{rel}` doesn't exist"
    if kind == "file_absent":
        return (_find(root, rel, anywhere) is None), f"`{rel}` still exists"
    if kind == "file_contains":
        p = _find(root, rel, anywhere)
        if p is None:
            return False, f"`{rel}` is missing"
        rx = _compile(pattern)
        if rx is None:
            return False, f"bad pattern /{pattern}/"
        return bool(rx.search(_read(p))), f"`{rel}` doesn't match /{pattern}/"
    if kind == "no_pattern":
        rx = _compile(pattern)
        if rx is None:
            return False, f"bad pattern /{pattern}/"
        for i, p in enumerate(root.glob(str(check.get("glob") or "**/*"))):
            if i > _SCAN_CAP:
                break
            if p.is_file() and rx.search(_read(p)):
                return False, f"/{pattern}/ found in {p.relative_to(root).as_posix()}"
        return True, ""
    if kind == "reply_matches":
        rx = _compile(pattern, re.IGNORECASE if check.get("ignore_case") else 0)
        if rx is None:
            return False, f"bad pattern /{pattern}/"
        return bool(rx.search(reply or "")), f"the reply doesn't match /{pattern}/"
    if kind == "reply_any":
        texts = [str(t).casefold() for t in check.get("texts") or [] if str(t)]
        low = (reply or "").casefold()
        return any(t in low for t in texts), "the reply says none of: " + ", ".join(texts[:6])
    return False, f"unknown check type {kind!r}"


def check_predicates(root: str | Path, reply: str, checks: Sequence[Mapping[str, Any]], *,
                     anywhere: bool = False) -> list[dict[str, Any]]:
    """Every predicate in *checks* against *root* and *reply*:
    ``[{"check", "ok", "detail"}]``. No checks is not a pass — the caller
    decides what an empty list means."""
    base = Path(root)
    out = []
    for check in checks:
        ok, why = check_one(base, reply, check, anywhere=anywhere)
        out.append({"check": dict(check), "ok": ok, "detail": "" if ok else why})
    return out


# ── agreement ────────────────────────────────────────────────────────


def agreement(pairs: Iterable[tuple[bool | None, bool]]) -> dict[str, Any]:
    """Yes/no predictions against the truth: ``(predicted, actual)`` pairs, the
    positive class being "yes". A prediction of None (no answer, or one that
    couldn't be read) always counts against: a miss on a yes, a false alarm on
    a no."""
    tp = fp = fn = tn = unanswered = 0
    for predicted, actual in pairs:
        if predicted is None:
            unanswered += 1
            if actual:
                fn += 1
            else:
                fp += 1
        elif predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unanswered": unanswered,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / n, 4) if n else 0.0,
            "base_rate": round((tp + fn) / n, 4) if n else 0.0}


# ── owner ────────────────────────────────────────────────────────────


def owner_score(votes: Iterable[bool | int | None]) -> dict[str, Any]:
    """The owner's thumbs: how many up and down, and the share up (None: no
    votes yet). A vote left blank isn't counted."""
    up = down = 0
    for v in votes:
        if v is None:
            continue
        if bool(v):
            up += 1
        else:
            down += 1
    n = up + down
    return {"n": n, "up": up, "down": down, "rate": round(up / n, 4) if n else None}
