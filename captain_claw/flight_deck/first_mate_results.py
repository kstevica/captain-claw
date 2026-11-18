"""First Mate — what a factory's runs made, the result first.

A run's folder holds more than its result: a big list is built by batch, so
next to ``final_leads.csv`` sit the batches it was merged from, temporary
files and the scripts a worker wrote to check its work (its notes, progress
and the extra data it was given are kept apart already, ``bat_routes.WORK_DIRS``).
Each run's files are sorted three ways so the person sees the result first:

* **main**: files the run's done-checks name, else files called a result
  (final, result, report, output, summary, merged…), else the newest file at
  the top of the folder;
* **other**: the rest at the top of the folder (the batches, say);
* **work**: what's in subfolders, scripts and temporary files.

A table says how many rows it has, and each file whether it is in the File
library (``file_library`` picks: only what the person added goes there).
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

MAX_FILES = 300
ROWS_MAX_BYTES = 20_000_000
RUNS_SHOWN = 10
RUNS_PROFILED = 3  # the newest runs' files are read for what they hold; older ones keep what Bat found
_TABLE_EXTS = {".csv": ",", ".tsv": "\t"}


def rows_of(path: Path) -> int | None:
    """How many rows a CSV/TSV has under its header (None for anything else, or one too big)."""
    sep = _TABLE_EXTS.get(path.suffix.lower())
    if sep is None:
        return None
    try:
        if path.stat().st_size > ROWS_MAX_BYTES:
            return None
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            n = sum(1 for row in csv.reader(fh, delimiter=sep) if any(cell.strip() for cell in row))
    except (OSError, csv.Error):
        return None
    return max(0, n - 1)


def named_by_checks(bat_run: dict[str, Any] | None) -> set[str]:
    """The files a run's done-checks name: what it was asked to deliver."""
    out: set[str] = set()
    for rule in ((bat_run or {}).get("config") or {}).get("contract") or []:
        check = rule.get("check") if isinstance(rule, dict) else None
        path = str((check or {}).get("path") or "").strip().lstrip("./")
        if path and (check or {}).get("type") in ("file_exists", "file_contains"):
            out.add(path)
    return out


def sort_files(found: list[tuple[float, str, Path, int]], named: set[str], picked: set[str]) -> dict[str, Any]:
    """``found`` from ``bat_routes._run_files`` (newest first) into main / other / work
    (``deliverable_profile.sort_files``), each with its rows and whether it is in the library."""
    from captain_claw.flight_deck import deliverable_profile as dp
    whole = "" in picked

    def item(entry: tuple[float, str, Path, int]) -> dict[str, Any]:
        mtime, rel, path, size = entry
        return {"path": rel, "name": Path(rel).name, "size": size, "mtime": mtime,
                "rows": rows_of(path), "library": whole or rel in picked}

    groups = dp.sort_files(found, named)
    code = dp.is_code_project(found)
    out = {k: [item(e) for e in v[:60 if code and k == "main" else len(v)]] for k, v in groups.items()}
    return {**out, "code": code, "total": len(found), "bytes": sum(f[3] for f in found)}


def _library_picks(owner_id: str, project: str) -> set[str]:
    try:
        from captain_claw.flight_deck import file_library as fl
        return fl.get_store().picks(fl._vfs_owner(owner_id)).get(project, set())
    except Exception:  # noqa: BLE001 — whether it's in the library is a nicety
        return set()


async def run_results(run: dict[str, Any], *, profile_now: bool = False) -> dict[str, Any] | None:
    """One First Mate run's files, sorted; None when it has no folder."""
    from captain_claw.flight_deck import bat_loop
    from captain_claw.flight_deck.bat_routes import _project_of, _run_files, _run_folder
    child = str(run.get("child_id") or "")
    if run.get("runner") != "bat" or not child:
        return None
    project = _project_of({"id": child})
    found = await asyncio.to_thread(_run_files, _run_folder({"id": child, "owner_id": run["owner_id"]}), limit=MAX_FILES)
    store = bat_loop.get_store()
    bat_run = await store.get_run(child) if store is not None else None
    named = named_by_checks(bat_run)
    sorted_ = await asyncio.to_thread(sort_files, found, named, _library_picks(run["owner_id"], project))
    timing = run.get("timing") or await run_timing(run)
    verdict = ((bat_run or {}).get("analysis") or {}).get("verdict") or {}
    checked = verdict.get("profile")
    if profile_now and found:  # what the files hold now (a going run's too)
        from captain_claw.flight_deck import deliverable_profile as dp
        checked = dp.summary(await asyncio.to_thread(dp.profile, found, named))
    from captain_claw.flight_deck.bat_loop import dissent
    return {"run_id": run["id"], "status": run["status"], "kind": run["kind"], "created_at": run["created_at"],
            "ended_at": run.get("ended_at"), "project": project, "timing": timing,
            "checked": {"flags": (checked or {}).get("flags") or [], "kind": (checked or {}).get("kind")} if checked else None,
            "dissent": dissent(verdict) if verdict else "",
            "checks": _check_results(verdict), **sorted_}


def _check_results(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """The done-checks Captain ran itself on the result, passed or not."""
    out = []
    for c in verdict.get("det_criticals") or []:
        if str(c).startswith("acceptance: "):
            out.append({"text": str(c)[len("acceptance: "):], "passed": False})
    return out


async def factory_results(factory_id: str, *, limit: int = RUNS_SHOWN) -> list[dict[str, Any]]:
    """The factory's newest runs that left files, newest first."""
    out = []
    for run in fms.get_store().list_runs(factory_id, limit=limit * 2):
        got = await run_results(run, profile_now=len(out) < RUNS_PROFILED)
        if got and got["total"]:
            out.append(got)
        if len(out) >= limit:
            break
    return out


async def latest(factory_id: str) -> dict[str, Any] | None:
    """For a factory's card: its newest run's result, if it left one."""
    for run in fms.get_store().list_runs(factory_id, limit=5):
        got = await run_results(run)
        if got and got["main"]:
            top = got["main"][0]
            return {"run_id": got["run_id"], "project": got["project"], "status": got["status"],
                    "name": top["name"], "path": top["path"], "size": top["size"], "rows": top["rows"],
                    "files": got["total"]}
    return None


# ── how long a run took ────────────────────────────────────────────────
#
# From its Bat events: a step's worker starts ("step" with an attempt) and ends
# ("step" with ok). A turn cut off has no end of its own: a stop (cap, owner,
# error) ends it then; a deck restart ("resumed") ends it at the last thing
# the run did before the deck went down. The time a stopped run waited for its
# owner runs from the stop to "continued where it stopped".

_STOPS = ("cancelled", "budget", "awaiting_human", "awaiting_plan")


def _is_stop(ev: dict[str, Any]) -> bool:
    return ev["stage"] in _STOPS or (ev["stage"] == "error" and str(ev.get("message", "")).startswith(("stopped", "planning failed", "planner produced", "driver crashed")))


def _union(spans: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(spans):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def timing_of(events: list[dict[str, Any]], started: float, ended: float | None, now: float) -> dict[str, Any]:
    """``{total_s, working_s, agent_s, waiting_s, other_s, turns}``: in all, while at
    least one agent worked, agent time summed (steps side by side count each),
    stopped waiting for the owner, and the rest (planning, checking, restarts)."""
    end = ended if ended is not None else now
    open_turns: dict[str, float] = {}
    spans: list[tuple[float, float]] = []
    waits: list[tuple[float, float]] = []
    stopped_at: float | None = None
    last_ts = started
    turns = 0

    def close_all(at: float) -> None:
        for start in open_turns.values():
            spans.append((start, max(start, at)))
        open_turns.clear()

    for ev in events:
        ts = float(ev.get("ts") or 0)
        data = ev.get("data") or {}
        msg = str(ev.get("message") or "")
        if ev["stage"] == "step" and data.get("agent"):
            key = str(data["agent"])
            if "ok" in data:
                if key in open_turns:
                    spans.append((open_turns.pop(key), ts))
            else:
                if key in open_turns:  # a new try began without an end to the last
                    spans.append((open_turns.pop(key), last_ts))
                open_turns[key] = ts
                turns += 1
        elif ev["stage"] == "note" and msg.startswith("resumed"):
            close_all(last_ts)  # the deck was down from the last event to this one
        elif ev["stage"] == "note" and msg.startswith("continued where it stopped"):
            close_all(ts)
            if stopped_at is not None:
                waits.append((stopped_at, ts))
                stopped_at = None
        elif ev["stage"] == "phase" and msg == "running" and stopped_at is not None:
            waits.append((stopped_at, ts))  # a plan approved, a question answered: it goes on
            stopped_at = None
        elif _is_stop(ev):
            close_all(ts)
            stopped_at = ts if stopped_at is None else stopped_at
        last_ts = max(last_ts, ts)
    close_all(end if ended is None else min(end, last_ts))  # still at it, or cut off by the end
    if stopped_at is not None and ended is None:
        waits.append((stopped_at, end))  # stopped, and nobody has pressed Continue yet
    total = max(0.0, end - started)
    working = min(total, _union(spans))
    waiting = min(max(0.0, total - working), _union(waits))
    return {"total_s": round(total), "working_s": round(working), "agent_s": round(sum(b - a for a, b in spans)),
            "waiting_s": round(waiting), "other_s": round(max(0.0, total - working - waiting)), "turns": turns}


async def run_timing(run: dict[str, Any]) -> dict[str, Any] | None:
    """How long a First Mate run took, from its creation to its end (or now),
    with what its agents worked; None without a Bat run."""
    import time
    from datetime import datetime

    from captain_claw.flight_deck import bat_loop
    child = str(run.get("child_id") or "")
    store = bat_loop.get_store()
    if run.get("runner") != "bat" or not child or store is None:
        return None
    events = await store.list_events(child, limit=50_000)
    started = datetime.fromisoformat(run["created_at"]).timestamp()
    ended = None
    if run.get("status") in fms.TERMINAL_RUN and run.get("ended_at"):
        ended = datetime.fromisoformat(run["ended_at"]).timestamp()
    return timing_of(events, started, ended, time.time())


# ── in words, for a message ────────────────────────────────────────────

def duration_text(seconds: float | int | None) -> str:
    s = max(0, int(seconds or 0))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    h, m = divmod(s // 60, 60)
    if h < 48:
        return f"{h} h {m} min" if m else f"{h} h"
    d, h = divmod(h, 24)
    return f"{d} d {h} h" if h else f"{d} d"


def size_text(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB" if n >= 1_000_000 else f"{max(1, round(n / 1000))} KB"


def timing_line(timing: dict[str, Any] | None) -> str:
    """"Took 23 h 4 min · its agents worked 9 h 12 min · waiting for you 11 h 30 min"."""
    if not timing or not timing.get("total_s"):
        return ""
    parts = [f"Took {duration_text(timing['total_s'])}", f"its agents worked {duration_text(timing.get('working_s'))}"]
    if timing.get("agent_s", 0) > timing.get("working_s", 0) * 1.2:
        parts[-1] += f" ({duration_text(timing['agent_s'])} of agent time, side by side)"
    if timing.get("waiting_s", 0) >= 60:
        parts.append(f"waiting for you {duration_text(timing['waiting_s'])}")
    return " · ".join(parts)


def worth_a_look(got: dict[str, Any] | None, limit: int = 3) -> str:
    """"Worth a look: 43% of Email are role inboxes …; 1 of 3 reviewers objected: …"."""
    if not got:
        return ""
    bits = [*((got.get("checked") or {}).get("flags") or [])[:limit]]
    failed = [c["text"] for c in got.get("checks") or [] if not c.get("passed")]
    if failed:
        bits.insert(0, "failed checks: " + "; ".join(failed[:3]))
    if got.get("dissent"):
        bits.append(got["dissent"][:300])
    return ("Worth a look: " + " · ".join(bits)) if bits else ""


def results_line(got: dict[str, Any] | None) -> str:
    """"Result: final_leads.csv · 500 rows · 235 KB (and 27 other files)"."""
    if not got or not got.get("main"):
        return ""
    shown = []
    for f in got["main"][:3]:
        rows = f.get("rows")
        bits = [f["name"]] + ([f"{rows:,} row{'' if rows == 1 else 's'}"] if rows is not None else []) + [size_text(f["size"])]
        shown.append(" · ".join(bits))
    rest = got["total"] - len(shown)
    label = "Result" if len(shown) == 1 else "Results"
    return f"{label}: " + "; ".join(shown) + (f" (and {rest} other file{'' if rest == 1 else 's'} in its folder)" if rest > 0 else "")
