"""Bat — is a finished step still on course for the run's goal?

The final review used to be the first look anyone took. In a real 500-lead run the
drift was there after the first batch (19 of 52 were university staff, 15 role
inboxes), four batches used different columns, and the merge padded a shortfall
with 61 staff-directory entries: twelve steps of work went the wrong way before
anything noticed. So each step that makes something is checked before it counts:

1. **Scripts, at no token cost**, on what the step added (the files it made or
   changed since it started):
   - the owner's own rules that hold for a part as for the whole (columns, no
     duplicates, links, role-inbox/free-mail/one-domain shares, excluded patterns,
     allowed values; never counts like "500 rows"), from the run's shaped
     done-checks (``bat_routes._shape_contract``);
   - the same columns as the run's other lists, when the run has conventions;
   - items another part already has;
   - what Captain's reading of the files finds worth a look, and a part that came
     back much shorter than its step asked: those are only signals.
2. **A short model review, only when a signal trips**, after the first step writes
   the run's conventions, and before the last step (the merge): on course,
   drifting or off course, with a correction for the next steps.
3. **Acting, without asking anyone** (a factory's run never waits):
   - a broken rule sends the step round again with why (``bat_loop``; its last try
     is kept, with the finding noted);
   - a step that came back short of the amount it was asked for, while its own
     report shows more it could have tried, goes round once more to keep going
     from its progress note (a run's workers had called 4 of 25 an "honest
     shortfall" after 15 minutes, with candidates still on their list);
   - a correction goes into ``shared/corrections.md``, which every later step's
     prompt shows above its conventions, and the final reviewers see too;
   - a note says what was found.

Runs of fewer than three steps, and steps done by a coding run (its tests do
this), aren't checked. ``config.course_check`` false turns it off for a run.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

from captain_claw.flight_deck import bat_loop

log = structlog.get_logger(__name__)

MIN_STEPS = 3
SHORT_SHARE = 0.6          # a part under 60% of what its step asked for came back short
SHORT_MARK = "came back short"
DUP_SHARE = 0.2            # a part whose items are 20%+ elsewhere already repeats work
MERGED_SHARE = 0.8         # a list holding 80%+ of two others is made from them (a merge)
CORRECTIONS = "corrections.md"
CORRECTIONS_MAX = 12
CORRECTIONS_SHOWN = 1500
_COUNT_KEYS = ("min_rows", "max_rows", "min_per", "min_words", "max_words", "min_links", "headings")
_TABLE_EXTS = (".csv", ".tsv")
_DOC_EXTS = (".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".rst")

_REVIEW_SYSTEM = (
    "You check, in the middle of a long job, whether it is still on course for its goal. You see the goal "
    "(the owner's words), the run's shared conventions and earlier corrections, and one point in the run: "
    "what a step reported and what Captain found in the files (VERIFIED: believe it over any report). "
    "Decide: on_course (keep going), drifting (heading somewhere the goal didn't ask for: who or what it "
    "collects or covers, where things come from, quality, format), or off_course (the work no longer serves "
    "the goal). When drifting or off course, write a correction for the NEXT steps: one or two short, "
    "concrete imperative sentences (\"Skip university staff; collect people who publicly promote AI tools.\"). "
    "Don't ask for more than the goal asks, and never restate one of its numbers differently: quote an amount "
    "exactly as the goal's amounts below give it, or leave numbers out. Also say whether the step stopped "
    "early: it was asked for an amount, came back clearly short of it, and its own report shows it could "
    "have gone on (candidates, leads or sources it noted but didn't try), not a real wall (sources that ran "
    "dry, sites that block it). Reply ONLY with JSON: "
    '{"state": "on_course|drifting|off_course", "why": "<one sentence>", "correction": "<for the next steps, or '
    'empty>", "stopped_early": true|false}'
)


# ── what a step added ──────────────────────────────────────────────────

def changed_since(folder: Path, since: float) -> list[tuple[float, str, Path, int]]:
    """The run's files (not its working notes) made or changed since ``since``."""
    from captain_claw.flight_deck.bat_routes import _run_files
    return [f for f in _run_files(folder) if f[0] >= since - 2]


def step_rules(contract: list[dict]) -> list[dict]:
    """The run's scripted table/document rules that hold for one part as for the
    whole: counts and required sections are for the finished result only."""
    out = []
    for r in contract or []:
        check = dict(r.get("check") or {})
        if check.get("type") not in ("table", "document"):
            continue
        for k in _COUNT_KEYS:
            check.pop(k, None)
        if len([k for k in check if k not in ("type", "path")]) == 0:
            continue
        out.append({**r, "check": check})
    return out


def _header(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            row = next(csv.reader(fh, delimiter="\t" if path.suffix.lower() == ".tsv" else ","), [])
    except (OSError, csv.Error):
        return []
    return [c.strip().lower() for c in row if c.strip()]


def _key_values(path: Path) -> set[str]:
    """The values that make an item itself: an email, else a link, else a name."""
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
    except (OSError, csv.Error):
        return set()
    if not rows:
        return set()
    cols = list(rows[0].keys())
    for want in ("mail", "url", "link", "website", "name", "title"):
        col = next((c for c in cols if c and want in c.lower()), None)
        if col:
            return {str(r.get(col) or "").strip().lower() for r in rows if str(r.get(col) or "").strip()}
    return set()


# The amount a step collects: a number after what it does ("add 8 qualifying firms",
# "Find, vet, and record 25 …"), never one that is a range or has a unit ("2-50
# employees", "24 months"). The first number between 10 and 5000 anywhere used to
# count, so "add 8 firms … 2-50 employees" read as 50, and three steps that had
# found all they were asked for went round again as short.
_COLLECT = r"(?:add|find|collect|gather|record|get|list|research|identify|source|append|vet|check)"
_NOT_AN_AMOUNT = (r"(?![\d.,]|\s*[-–/]\s*\d|-|\s*(?:%|percent|employees?|people|staff|persons?|months?|years?|"
                  r"days?|weeks?|hours?|minutes?|words?|characters?|pages?|columns?|sections?|steps?|tiers?|"
                  r"states?|segments?|groups?)\b)")
_AMOUNT = re.compile(r"\b" + _COLLECT + r"\b[^.;:\n]{0,50}?(?<![\d.,/-])\b(\d{1,5})\b" + _NOT_AN_AMOUNT, re.I)
_FROM_TO = re.compile(r"\b(\d{1,5})\s*(?:→|->)\s*(\d{1,5})\b")


def _asked(title: str) -> int:
    """How many items a step's title asks it to add ("add 8 qualifying firms",
    "batch 1 (0→20)"), or 0 when it names no amount of its own."""
    t = title or ""
    m = _FROM_TO.search(t)
    if m and int(m.group(2)) > int(m.group(1)):
        return int(m.group(2)) - int(m.group(1))
    m = _AMOUNT.search(t)
    n = int(m.group(1)) if m else 0
    return n if 1 <= n <= 5000 else 0


def _merged_lists(keys: dict[str, set[str]]) -> set[str]:
    """The lists made from the run's other lists (the final list a merge step builds):
    each holds nearly all of two or more of them and is bigger than each. Repeating
    their items is what they are for, so they aren't "other parts": counted as one,
    the merged list made a step that rewrote its own part look like a repeat of it."""
    out: set[str] = set()
    for name, mine in keys.items():
        if not mine:
            continue
        held = sum(1 for other, theirs in keys.items()
                   if other != name and theirs and len(mine) > len(theirs)
                   and len(mine & theirs) >= max(2, len(theirs) * MERGED_SHARE))
        if held >= 2:
            out.add(name)
    return out


def _conventions_step(step: dict) -> bool:
    from captain_claw.flight_deck.bat_routes import is_conventions_step
    return is_conventions_step(step)


def script_findings(run: dict, step: dict, folder: Path, changed: list[tuple[float, str, Path, int]],
                    last_step: bool) -> tuple[list[str], list[str], str]:
    """``(hard, soft, verified)``: rules the part breaks, signals worth a model's look,
    and what Captain found in its files (for the review)."""
    from captain_claw.flight_deck import code_contract
    from captain_claw.flight_deck import deliverable_profile as dp
    hard: list[str] = []
    soft: list[str] = []
    cfg = run.get("config") or {}
    rules = step_rules(cfg.get("contract") or [])
    has_conventions = (folder / "shared" / "conventions.md").is_file()
    top = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _TABLE_EXTS] if folder.is_dir() else []
    mine = {rel for _, rel, _, _ in changed}
    keys: dict[str, set[str]] | None = None  # each top-level list's items, read once when needed
    verified: list[str] = []
    for _, rel, path, _ in changed:
        ext = path.suffix.lower()
        if ext in _TABLE_EXTS:
            for r in rules:
                if r["check"]["type"] != "table":
                    continue
                ok, note = code_contract._check_table(folder, {**r["check"], "path": rel})
                if not ok:
                    hard.append(f"{rel} breaks \"{r['text']}\": {note.split(': ', 1)[-1]}")
            prof = dp.profile_table(path)
            verified.append(dp.render({"main": [{"path": rel, "size": path.stat().st_size, "profile": prof,
                                                 "flags": dp.flags(prof)}]}))
            soft += [f"{rel}: {f}" for f in dp.flags(prof)]
            asked = 0 if _conventions_step(step) else _asked(step.get("title") or "")
            if asked and not last_step and prof.get("rows", 0) < asked * SHORT_SHARE:
                soft.append(f"{rel} {SHORT_MARK}: {prof.get('rows', 0)} of the {asked} its step asked for")
            if last_step or "/" in rel:
                continue
            others = [p for p in top if p.name not in mine and p.name != rel]
            if others and has_conventions:
                head = _header(path)
                common = Counter(tuple(_header(p)) for p in others).most_common(1)[0][0]
                if head and common and tuple(head) != common:
                    hard.append(f"{rel} has columns ({', '.join(head)}) unlike the run's other lists "
                                f"({', '.join(common)}): follow shared/conventions.md")
            if keys is None:
                keys = {p.name: _key_values(p) for p in top}
            merged = _merged_lists(keys)
            mine_keys = keys.get(rel) if rel in keys else _key_values(path)
            parts = [p for p in others if p.name not in merged]
            if mine_keys and parts and rel not in merged:
                seen: set[str] = set()
                for p in parts:
                    seen |= keys.get(p.name, set())
                again = mine_keys & seen
                if len(again) >= max(3, len(mine_keys) * DUP_SHARE):
                    hard.append(f"{len(again)} of the {len(mine_keys)} items in {rel} are already in other parts: "
                                "check shared/seen.csv before adding")
        elif ext in _DOC_EXTS:
            for r in rules:
                if r["check"]["type"] == "document" and r["check"].get("no_placeholders"):
                    ok, note = code_contract._check_document(folder, {**r["check"], "path": rel})
                    if not ok:
                        hard.append(f"{rel} breaks \"{r['text']}\": {note.split(': ', 1)[-1]}")
            prof = dp.profile_docx(path) if ext == ".docx" else dp.profile_document(path)
            soft += [f"{rel}: {f}" for f in dp.flags(prof) if not f.startswith("only ")]
    return hard, soft, "\n\n".join(v for v in verified if v)


# ── the conventions and the corrections ────────────────────────────────

def corrections_path(folder: Path) -> Path:
    return folder / "shared" / CORRECTIONS


def read_corrections(folder: Path) -> list[str]:
    try:
        text = corrections_path(folder).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]


def add_correction(folder: Path, where: str, text: str) -> bool:
    """One more for the next steps; the same one twice, or past the limit, isn't."""
    text = " ".join(str(text or "").split())[:400]
    if not text:
        return False
    have = read_corrections(folder)
    if any(text.lower() in c.lower() for c in have) or len(have) >= CORRECTIONS_MAX:
        return False
    p = corrections_path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = "" if p.is_file() else "# Corrections from Captain's checks\n\nEvery later step follows these.\n\n"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(f"{head}- ({where}) {text}\n")
    return True


def conventions_findings(run: dict, folder: Path) -> list[str]:
    """Columns the owner's rules name that the conventions don't."""
    try:
        text = (folder / "shared" / "conventions.md").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return []
    want: list[str] = []
    for r in (run.get("config") or {}).get("contract") or []:
        check = r.get("check") or {}
        if check.get("type") == "table":
            want += list(check.get("columns") or []) + list(check.get("link_columns") or [])
    missing = sorted({c for c in want if c.lower() not in text})
    return [f"the conventions don't name the column{'' if len(missing) == 1 else 's'} the rules ask for: "
            f"{', '.join(missing)}"] if missing else []


# ── the model's look ───────────────────────────────────────────────────

async def review(run: dict, what: str, verified: str, signals: list[str]) -> dict[str, Any] | None:
    """On course, drifting or off course, with a correction, and whether the step
    stopped early; None without a model."""
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    from captain_claw.flight_deck.bat_routes import _creds_for, _metered, _run_folder, think_call
    from captain_claw.llm import Message
    try:
        tiers, _ = await _load_owner_tiers(get_db(), run["owner_id"])
    except Exception:  # noqa: BLE001
        return None
    creds = _creds_for(tiers, prefer=("reason", "balanced", "fast"))
    if not creds:
        return None
    folder = _run_folder(run)
    try:
        conv = (folder / "shared" / "conventions.md").read_text(encoding="utf-8", errors="replace")[:2500]
    except OSError:
        conv = ""
    fixes = read_corrections(folder)
    parts = [f"# The goal\n{str(run.get('task', ''))[:4000]}"]
    amounts = [str(r.get("text") or "") for r in (run.get("config") or {}).get("contract") or []
               if re.search(r"\d", str(r.get("text") or ""))]
    if amounts:
        parts.append("# The goal's amounts (quote these exactly, never another number)\n"
                     + "\n".join(f"- {a}" for a in amounts[:8]))
    if conv:
        parts.append(f"# The run's conventions\n{conv}")
    if fixes:
        parts.append("# Corrections already given\n" + "\n".join(f"- {c}" for c in fixes))
    parts.append(f"# Where the run is\n{what}")
    if verified:
        parts.append(verified[:6000])
    if signals:
        parts.append("# What Captain's checks noticed\n" + "\n".join(f"- {s}" for s in signals[:10]))
    async with _metered(run["id"], "course"):
        prov, mt = think_call(creds, temperature=0.0, default_max=8192)  # thinking included
        resp = await prov.complete(messages=[Message(role="system", content=_REVIEW_SYSTEM),
                                             Message(role="user", content="\n\n".join(parts))],
                                   temperature=0.0, max_tokens=mt)
    raw = (resp.content or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        got = json.loads(raw[start:end + 1]) if 0 <= start < end else {}
    except ValueError:
        got = {}
    state = str(got.get("state") or "").strip().lower()
    if state not in ("on_course", "drifting", "off_course"):
        return None
    early = got.get("stopped_early")
    return {"state": state, "why": str(got.get("why") or "").strip()[:300],
            "correction": str(got.get("correction") or "").strip()[:400] if state != "on_course" else "",
            "stopped_early": early is True or str(early).strip().lower() == "true"}


async def _shape_again(run: dict) -> dict:
    """A factory's done-checks that couldn't be turned into Captain's own checks at
    the start get another try before a step is checked against them."""
    from captain_claw.flight_deck import bat_routes
    cfg = run.get("config") or {}
    if (cfg.get("contract_shaped") or not bat_routes._factory_run(run)
            or int(cfg.get("contract_shape_tries") or 0) >= bat_routes.SHAPE_TRIES):
        return run
    try:
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.basna_routes import _load_owner_tiers
        tiers, _ = await _load_owner_tiers(get_db(), run["owner_id"])
        await bat_routes._shape_contract(run, bat_routes._creds_for(tiers), False)
        store = bat_loop.get_store()
        return (await store.get_run(run["id"]) if store is not None else None) or run
    except Exception as e:  # noqa: BLE001 — the step is checked with what there is
        log.warning("bat course: shaping the checks again failed", run_id=run.get("id"), error=str(e))
        return run


# ── the check ──────────────────────────────────────────────────────────

async def check_step(run: dict, step: dict, started_at: float, last_try: bool) -> dict[str, Any] | None:
    """Before a finished step counts. Returns ``{"redo", "why", "state"}`` or None
    when it isn't checked. Never raises (bat_loop also guards it)."""
    from captain_claw.flight_deck.bat_routes import _run_folder
    cfg = run.get("config") or {}
    store = bat_loop.get_store()
    if cfg.get("course_check") is False or store is None or not run.get("owner_id"):
        return None
    key = step.get("step_key", "")
    if ((cfg.get("step_hows") or {}).get(key)) in ("code", "crew"):
        return None
    steps = await store.list_steps(run["id"])
    if len(steps) < MIN_STEPS:
        return None
    run = await _shape_again(run)
    cfg = run.get("config") or {}
    folder = _run_folder(run)
    others = [s for s in steps if s["step_key"] != key]
    last_seq = max(int(s.get("seq") or 0) for s in steps)
    is_last = int(step.get("seq") or 0) >= last_seq and not any(
        s["status"] in ("pending", "running") for s in others)
    changed = await asyncio.to_thread(changed_since, folder, started_at)
    hard, soft, verified = await asyncio.to_thread(script_findings, run, step, folder, changed, is_last)
    title = (step.get("title") or key)[:120]
    conv_file = folder / "shared" / "conventions.md"
    wrote_conventions = conv_file.is_file() and conv_file.stat().st_mtime >= started_at - 2
    if wrote_conventions:
        soft += conventions_findings(run, folder)
    left = [s for s in others if s["status"] in ("pending", "running")]
    before_last = (len(steps) >= 4 and len(left) == 1 and left[0]["status"] == "pending"
                   and int(left[0].get("seq") or 0) >= last_seq)
    verdict: dict[str, Any] | None = None
    if soft or wrote_conventions or before_last:
        if wrote_conventions:
            what = (f"The first step (\"{title}\") just wrote the run's conventions above. Do they fit the goal: the "
                    "columns or outline it asks for, who or what is in and out, where things come from?")
        elif before_last:
            what = (f"Every step but the last is done; the last is \"{left[0].get('title')}\". Is the work so far "
                    "on course for the goal, and what must the last step do or avoid?")
        else:
            what = f"Step \"{title}\" just finished. It reported:\n{str(step.get('output') or '')[:2000]}"
        if before_last and not verified:
            from captain_claw.flight_deck import deliverable_profile as dp
            from captain_claw.flight_deck.bat_routes import _run_files
            verified = dp.render(await asyncio.to_thread(dp.profile, _run_files(folder), set()))
        try:
            verdict = await review(run, what, verified, hard + soft)
        except Exception as e:  # noqa: BLE001 — the scripts' findings still count
            log.warning("bat course review failed", run_id=run.get("id"), error=str(e))
    redo = bool(hard) and not last_try
    # Short of its amount with more it could have tried: once more, from its progress
    # note. Only on a first try, never for the last step (it adds nothing new).
    short = [f for f in soft if SHORT_MARK in f]
    again_short = (bool(short) and bool(verdict and verdict.get("stopped_early")) and not hard and not last_try
                   and not is_last and int(step.get("attempt") or 1) <= 1)
    if verdict and verdict["correction"]:
        if add_correction(folder, f"after {key}", verdict["correction"]):
            await store.append_event(run["id"], "note", f"course check after {key}: {verdict['state'].replace('_', ' ')} "
                                     f"({verdict['why']}). The next steps are told: {verdict['correction']}", agent=key)
    elif verdict:
        await store.append_event(run["id"], "note", f"course check after {key}: on course", agent=key)
    if hard:
        for finding in hard[:2]:  # what the step broke, the others shouldn't
            add_correction(folder, f"found in {key}", f"Don't repeat: {finding}")
        await store.append_event(run["id"], "note", (f"{key} goes round again: " if redo else
                                                     f"{key} kept on its last try, with: ") + "; ".join(hard)[:600],
                                 agent=key)
    if again_short:
        why = "; ".join(short)[:600]
        await store.append_event(run["id"], "note", f"{key} goes round again: it stopped short with more "
                                 f"to try ({why})", agent=key)
        return {"redo": True, "why": why, "state": (verdict or {}).get("state", ""),
                "error": (f"This step came back short of the amount it asks for ({why}), and its own report lists "
                          "candidates or sources it didn't get to. Keep going: pick up from your progress note, work "
                          "through what it left for later and find new candidates, until the step's amount is "
                          "reached or the sources truly run out. Keep the same criteria; add only what meets "
                          "them, and say how many you checked.")}
    return {"redo": redo, "why": "; ".join(hard)[:1500], "state": (verdict or {}).get("state", "")}
