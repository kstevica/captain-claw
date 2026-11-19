"""What a factory's runs already checked: a record across runs, so a standing
factory doesn't check the same item twice (docs/funnel-runner-plan.md, phase 4).

A list-building factory runs again and again. Every item a run looked at goes in
its factory's record when the run ends: a funnel run's every item (kept, dropped,
or close), a Bat run's final list (kept). The next run gets the record as
``extra/already-checked.csv``:

* a kept item is never checked again;
* a dropped one isn't either, while the charter asks for the same work (a new
  charter, checks or settings may let it pass);
* one that only came close is checked again each run (the fact it lacked may be
  there now).

The funnel skips those items itself; Bat's workers are told to. Test cases and
carry-outs neither read nor add to it. The owner can see how much it holds, and
forget it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck.funnel import LEDGER_FILE, norm_key

FILE = "_checked.csv"           # in the factory's data folder (it goes with the factory)
SNAPSHOT = "_checked_now.csv"   # what the run starting now is given
RUN_FILE = LEDGER_FILE          # where a run finds it: extra/already-checked.csv
COLS = ["key", "name", "group", "outcome", "reason", "work", "run", "checked_at"]
OUTCOMES = ("kept", "dropped", "maybe")
KINDS = ("run", "rehearsal")    # the runs that read it and add to it
_RANK = {"maybe": 0, "dropped": 1, "kept": 2}  # a firmer outcome replaces a softer one
_SITE = re.compile(r"website|domain|homepage|site\b|url|link", re.I)
_EMAIL = re.compile(r"e-?mail", re.I)


def _folder(factory_id: str) -> Path:
    from captain_claw.flight_deck import first_mate_data
    return first_mate_data._folder(factory_id)


def path(factory_id: str) -> Path:
    return _folder(factory_id) / FILE


def work_key(version: dict[str, Any] | None) -> str:
    """What the charter asks for, as a short hash: its words, runner, settings and
    done-checks. A dropped item is skipped only while this stays the same."""
    v = version or {}
    raw = json.dumps([v.get("charter"), v.get("runner"), v.get("runner_config") or {},
                      (v.get("checks") or {}).get("contract") or []], sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def read(factory_id: str) -> dict[str, dict[str, str]]:
    """The record, by item key."""
    try:
        with path(factory_id).open(encoding="utf-8", newline="") as fh:
            return {r["key"]: r for r in csv.DictReader(fh) if r.get("key")}
    except OSError:
        return {}


def _write(factory_id: str, rows: dict[str, dict[str, str]]) -> None:
    p = path(factory_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r.get("checked_at", ""), r.get("key", ""))))
    tmp.replace(p)


def record(factory_id: str, rows: list[dict[str, Any]], *, run_id: str, work: str) -> int:
    """Adds a run's items. An item already there keeps its place unless the new
    outcome is firmer (close → dropped → kept). Returns how many were added or
    changed."""
    have = read(factory_id)
    now = time.strftime("%Y-%m-%d %H:%M")
    changed = 0
    for r in rows:
        key = norm_key(str(r.get("key") or ""))
        outcome = str(r.get("outcome") or "")
        if not key or outcome not in OUTCOMES:
            continue
        old = have.get(key)
        if old is not None and _RANK.get(old.get("outcome", ""), -1) >= _RANK[outcome] \
                and not (outcome == "dropped" and old.get("outcome") == "dropped" and old.get("work") != work):
            continue
        have[key] = {"key": key, "name": str(r.get("name") or "")[:120], "group": str(r.get("group") or "")[:80],
                     "outcome": outcome, "reason": " ".join(str(r.get("reason") or "").split())[:200],
                     "work": work, "run": run_id, "checked_at": now}
        changed += 1
    if changed:
        _write(factory_id, have)
    return changed


def to_skip(rows: dict[str, dict[str, str]], work: str) -> list[dict[str, str]]:
    """What a run of this work skips: kept items always, dropped ones checked for
    the same work; never one that only came close."""
    return [r for r in rows.values()
            if r.get("outcome") == "kept" or (r.get("outcome") == "dropped" and r.get("work") == work)]


def snapshot(factory_id: str, version: dict[str, Any]) -> tuple[Path, int] | None:
    """The file a starting run is given (``already-checked.csv``), and how many items
    it lists; None when there's nothing to skip."""
    skip = to_skip(read(factory_id), work_key(version))
    if not skip:
        return None
    p = _folder(factory_id) / SNAPSHOT
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "name", "group", "outcome", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(skip)
    return p, len(skip)


def counts(factory_id: str) -> dict[str, Any]:
    rows = read(factory_id)
    by = {o: sum(1 for r in rows.values() if r.get("outcome") == o) for o in OUTCOMES}
    return {"items": len(rows), **by, "runs": len({r.get("run") for r in rows.values() if r.get("run")}),
            "last": max((r.get("checked_at", "") for r in rows.values()), default="")}


def forget(factory_id: str) -> None:
    for name in (FILE, SNAPSHOT):
        try:
            (_folder(factory_id) / name).unlink()
        except OSError:
            pass


# ── what a finished run adds ───────────────────────────────────────────

def rows_from_funnel(folder: Path) -> list[dict[str, str]]:
    """A funnel run's every item: kept, dropped or close (its finishing step's too)."""
    from captain_claw.flight_deck import funnel
    out = []
    for r in funnel._read_csv(folder / funnel.ITEMS):
        status = r.get("status", "")
        outcome = status if status in OUTCOMES else ""
        if outcome:
            out.append({"key": r.get("key", ""), "name": r.get("name", ""), "group": r.get("group", ""),
                        "outcome": outcome, "reason": r.get("reason", "")})
    return out


def _key_column(cols: list[str], unique: list[str]) -> str:
    """The column that names an item as the funnel does: its site first (by domain),
    then a unique column, an email, else the first column."""
    return (next((c for c in cols if _SITE.search(c)), "") or next((c for c in unique if c in cols), "")
            or next((c for c in cols if _EMAIL.search(c)), "") or (cols[0] if cols else ""))


def rows_from_list(folder: Path, bat_run: dict[str, Any] | None) -> list[dict[str, str]]:
    """A Bat run's final list, every row kept. The list is the table its checks name,
    else its main table."""
    from captain_claw.flight_deck import funnel_extract as fx
    contract = list(((bat_run or {}).get("config") or {}).get("contract") or [])
    table, unique, group = "", [], ""
    for r in contract:
        check = r.get("check") or {}
        if check.get("type") == "table":
            table = table or str(check.get("path") or "")
            unique = unique or list(check.get("unique") or [])
            group = group or next(iter(check.get("min_per") or {}), "")
    target = folder / table if table and (folder / table).is_file() else None
    if target is None:
        from captain_claw.flight_deck import code_contract
        target = code_contract._main_file(folder, (".csv", ".tsv", ".xlsx"), ())
    if target is None:
        return []
    try:
        cols, rows = fx.read_table(target)
    except (OSError, ValueError):
        return []
    key = _key_column(cols, unique)
    name = next((c for c in cols if c != key and not _SITE.search(c) and not _EMAIL.search(c)), key)
    return [{"key": r.get(key, ""), "name": r.get(name, ""), "group": r.get(group, "") if group else "",
             "outcome": "kept", "reason": f"on the list {target.name}"} for r in rows if r.get(key)]


async def record_run(run: dict[str, Any]) -> int:
    """Adds a finished First Mate run's items to its factory's record. Nothing for a
    test case, a carry-out, or a runner that isn't Bat's."""
    if run.get("kind") not in KINDS or run.get("runner") not in fms.BAT_RUNNERS or not run.get("child_id"):
        return 0
    import asyncio

    from captain_claw.flight_deck import bat_loop, bat_routes
    folder = bat_routes._run_folder({"id": run["child_id"], "owner_id": run["owner_id"]})
    version = fms.get_store().version(run["factory_id"], run["version"])
    bat_run = None
    if run["runner"] != "funnel":
        store = bat_loop.get_store()
        bat_run = await store.get_run(run["child_id"]) if store is not None else None

    def work() -> int:
        if not folder.is_dir():
            return 0
        rows = rows_from_funnel(folder) if run["runner"] == "funnel" else rows_from_list(folder, bat_run)
        return record(run["factory_id"], rows, run_id=run["id"], work=work_key(version))

    return await asyncio.to_thread(work)
