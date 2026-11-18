"""A1 — the acceptance contract for Code's build/fix loop.

``research_contract.py`` extracts a research deliverable's numeric hard rules and
checks them against a facts ledger. Code's deliverable is a *repo*, so the
predicates are boolean facts about the working tree and its commands: a required
test/file exists, a build command exits 0, a file contains a symbol, a forbidden
placeholder is gone. The contract is:

* **Derived once, from the approved plan** — one reason-tier call at plan-approval
  time turns the plan's acceptance criteria into checkable predicates.
* **Persisted, user-editable** — written to ``.contract.json`` in the repo (like
  ``.plans/``/``.reports/``, committed — not gitignored). A follow-up turn LOADS
  it instead of re-deriving; hand-edits survive (re-normalized on load so a bad
  edit can't smuggle a malformed check into the validator).
* **Validated deterministically** after each build/fix round — command runs, file
  reads, regex. Zero model tokens, which is why the ``thorough`` preset can arm
  it. Only the ``judge`` fallback (rules code genuinely can't verify) costs one
  call, folded into the triage-tier budget the loop already spends.

A failed critical/major becomes a synthetic "Acceptance Contract" report fed into
the existing triage alongside the reviewers — the same ground-truth role
``code_verify.as_review_entry`` plays, but for the task's *specific* acceptance
criteria the generic test run can't see.

Pure module (stdlib + the injectable command runner): prompts, parser,
validators, persistence, renderers. The route file owns the model calls.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from captain_claw.dubina.coder import CommandRunner, shell_command_runner
from captain_claw.logging import get_logger

log = get_logger(__name__)

CONTRACT_FILE = ".contract.json"

_SEVERITIES = ("critical", "major", "minor")
_CHECK_TYPES = ("command", "file_exists", "file_contains", "no_pattern", "table", "document", "python_syntax", "judge")
_MAX_CONSTRAINTS = 20
_CMD_TIMEOUT_S = 180.0        # a contract command must not stall the build loop
_CMD_OUTPUT_CAP = 2000        # chars of a failing command's output kept for triage
_FILE_READ_CAP = 200_000      # bytes read per file when scanning content
_SCAN_FILE_CAP = 800          # max files a no_pattern scan will open


# ── derive ───────────────────────────────────────────────────────────

def derive_prompt(intent: str, plan_text: str = "", learned: str = "") -> str:
    plan_block = f"\n\n## Approved plan\n{plan_text[:8000]}" if plan_text.strip() else ""
    return (
        "You are turning a coding task's ACCEPTANCE CRITERIA into checkable "
        "predicates — the conditions that, once true, mean the task is actually "
        "done (not merely that an agent said so). From the task and plan below, "
        "list every concrete, verifiable acceptance rule. Prefer rules a script "
        "can check. Do NOT invent scope the task doesn't state; do NOT list style "
        "preferences.\n\n"
        "Reply ONLY with JSON — no prose:\n"
        '{"constraints": [{"id": "a1", "text": "<the rule, one line>", '
        '"severity": "critical|major", "check": {...}}]}\n'
        "check shapes (pick the FIRST that fits; paths are repo-relative):\n"
        '- {"type": "command", "cmd": "<shell command run in the repo root>"} '
        "— passes when it exits 0 (e.g. a build, a lint, a specific test)\n"
        '- {"type": "file_exists", "path": "<repo-relative path>"}\n'
        '- {"type": "file_contains", "path": "<path>", "pattern": "<regex>"}\n'
        '- {"type": "no_pattern", "pattern": "<regex>", "glob": "<glob, '
        'default **/*>"} — passes when NO file matches (e.g. no leftover TODO)\n'
        '- {"type": "judge"} — only for rules no script can decide (UX wording, '
        "design intent)\n"
        "severity: critical = shipping without it invalidates the task; major = "
        "a serious defect. Keep commands cheap and deterministic (no servers, no "
        'network). If the task states no checkable acceptance rules, return '
        '{"constraints": []}.\n\n'
        f"## Task\n{intent}{plan_block}{learned}"
    )


def parse_contract(output: str) -> list[dict]:
    """Normalize the derive reply into constraint dicts. Tolerant; never raises."""
    if not output:
        return []
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        blob = text[start:end + 1] if 0 <= start < end else None
    if not blob:
        return []
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return []
    out: list[dict] = []
    for i, item in enumerate((raw.get("constraints") or [])[:_MAX_CONSTRAINTS], 1):
        norm = _normalize(item, i)
        if norm:
            out.append(norm)
    return out


# A check script or pattern is kept whole or not at all: cut, it's broken code (a
# run's three Python checks were cut mid-line at 400 characters and could only ever
# fail). One too long for this goes to the reviewers instead.
CMD_MAX = 4000
PATTERN_MAX = 1000
# What scripts used to be cut at: one exactly this long was cut, and can't run.
_OLD_CMD_CUT = 400


def _normalize(item, i: int) -> dict | None:
    if not isinstance(item, dict):
        return None
    text_ = str(item.get("text") or "").strip()[:300]
    if not text_:
        return None
    sev = str(item.get("severity") or "").strip().casefold()
    if sev not in _SEVERITIES:
        sev = "major"
    check = item.get("check") if isinstance(item.get("check"), dict) else {}
    ctype = str(check.get("type") or "").strip().casefold()
    norm: dict = {"type": "judge"}
    if ctype == "command" and str(check.get("cmd") or "").strip():
        cmd = str(check["cmd"]).strip()
        norm = {"type": "command", "cmd": cmd} if len(cmd) <= CMD_MAX else {"type": "judge"}
    elif ctype == "file_exists" and _relpath(check.get("path")):
        norm = {"type": "file_exists", "path": _relpath(check["path"])}
    elif ctype == "file_contains" and _relpath(check.get("path")) \
            and 0 < len(str(check.get("pattern") or "").strip()) <= PATTERN_MAX:
        norm = {"type": "file_contains", "path": _relpath(check["path"]),
                "pattern": str(check["pattern"])}
    elif ctype == "no_pattern" and 0 < len(str(check.get("pattern") or "").strip()) <= PATTERN_MAX:
        norm = {"type": "no_pattern", "pattern": str(check["pattern"]),
                "glob": (str(check.get("glob") or "").strip() or "**/*")[:120]}
    elif ctype == "table":
        norm = _norm_table(check) or {"type": "judge"}
    elif ctype == "document":
        norm = _norm_document(check) or {"type": "judge"}
    elif ctype == "python_syntax":
        norm = {"type": "python_syntax", "glob": (str(check.get("glob") or "").strip() or "**/*.py")[:120]}
    return {"id": str(item.get("id") or f"a{i}").strip()[:20],
            "text": text_, "severity": sev, "check": norm}


# ── a delivered table or document, checked by what it holds ──
#
# For results that aren't code: a list (rows, columns, duplicates, what its email
# and link columns hold) or a document (length, sections, placeholders, links).
# ``path`` '' means the run's main table or document (``deliverable_profile``).

def _num(v, lo: float, hi: float, as_int: bool = False):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if n != n or n < lo or n > hi:  # NaN, or out of range
        return None
    return int(n) if as_int else round(n, 4)


def _names(v, limit: int = 20) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [str(x).strip()[:80] for x in v or [] if str(x).strip()][:limit] if isinstance(v, list) else []


def _norm_table(check: dict) -> dict | None:
    out: dict = {"type": "table", "path": _relpath(check.get("path")) if check.get("path") else ""}
    for key, lo, hi, as_int in (("min_rows", 1, 10_000_000, True), ("max_rows", 1, 10_000_000, True),
                                ("min_link_share", 0, 1, False), ("max_role_share", 0, 1, False),
                                ("max_free_share", 0, 1, False), ("max_domain_share", 0, 1, False)):
        n = _num(check.get(key), lo, hi, as_int)
        if n is not None:
            out[key] = n
    for key in ("columns", "unique", "non_empty", "link_columns"):
        names = _names(check.get(key))
        if names:
            out[key] = names
    if str(check.get("email_column") or "").strip():
        out["email_column"] = str(check["email_column"]).strip()[:80]
    pat = str(check.get("exclude_pattern") or "").strip()
    if pat and len(pat) <= PATTERN_MAX:
        try:
            re.compile(pat)
            out["exclude_pattern"] = pat
        except re.error:
            pass
    per = check.get("min_per") if isinstance(check.get("min_per"), dict) else {}
    per = {str(k).strip()[:80]: n for k, v in list(per.items())[:3]
           if str(k).strip() and (n := _num(v, 1, 10_000_000, True)) is not None}
    if per:
        out["min_per"] = per
    allowed = check.get("allowed") if isinstance(check.get("allowed"), dict) else {}
    allowed = {str(k)[:80]: _names(v, 50) for k, v in list(allowed.items())[:5] if _names(v, 50)}
    if allowed:
        out["allowed"] = allowed
    return out if len(out) > 2 or out.get("path") else None


def _norm_document(check: dict) -> dict | None:
    out: dict = {"type": "document", "path": _relpath(check.get("path")) if check.get("path") else ""}
    for key, hi in (("min_words", 2_000_000), ("max_words", 2_000_000), ("min_links", 10_000)):
        n = _num(check.get(key), 0, hi, True)
        if n is not None:
            out[key] = n
    heads = _names(check.get("headings"))
    if heads:
        out["headings"] = heads
    if check.get("no_placeholders") is True:
        out["no_placeholders"] = True
    return out if len(out) > 2 or out.get("path") else None


def _main_file(repo: Path, exts: tuple[str, ...], skip: tuple[str, ...]) -> Path | None:
    from captain_claw.flight_deck import deliverable_profile as dp
    found = []
    for p in repo.iterdir() if repo.is_dir() else []:
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts and p.name not in skip:
            st = p.stat()
            found.append((st.st_mtime, p.name, p, st.st_size))
    found.sort(key=lambda f: f[0], reverse=True)
    main = dp.sort_files(found, set())["main"]
    return main[0][2] if main else (found[0][2] if found else None)


def _target(repo: Path, check: dict, exts: tuple[str, ...], skip: tuple[str, ...]) -> tuple[Path | None, str]:
    rel = _relpath(check.get("path")) if check.get("path") else ""
    if rel:
        p = repo / rel
        return (p if p.is_file() else None), rel
    p = _main_file(repo, exts, skip)
    return p, (p.name if p else "")


def _check_table(repo: Path, check: dict, skip: tuple[str, ...] = ()) -> tuple[bool, str]:
    from captain_claw.flight_deck import deliverable_profile as dp
    path, rel = _target(repo, check, (".csv", ".tsv"), skip)
    if path is None:
        return False, f"`{rel}` is missing" if rel else "no table (.csv/.tsv) at the top of the folder"
    prof = dp.profile_table(path)
    if prof.get("too_big"):
        return True, "(too big to check here)"
    n, cols = prof.get("rows", 0), prof.get("columns") or []
    low = {c.lower(): c for c in cols}
    bad: list[str] = []
    if check.get("min_rows") and n < check["min_rows"]:
        bad.append(f"{n} rows, fewer than {check['min_rows']}")
    if check.get("max_rows") and n > check["max_rows"]:
        bad.append(f"{n} rows, more than {check['max_rows']}")
    if prof.get("broken_rows"):
        bad.append(f"{prof['broken_rows']} rows have a different number of fields than the header")
    missing = [c for c in check.get("columns") or [] if c.lower() not in low]
    if missing:
        bad.append(f"missing columns: {', '.join(missing)} (has {', '.join(cols)})")
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
    def values(col: str) -> list[str]:
        key = low.get(col.lower())
        return [str(r.get(key) or "").strip() for r in rows] if key else []
    for col in check.get("unique") or []:
        vals = [v.lower() for v in values(col) if v]
        if len(vals) != len(set(vals)):
            bad.append(f"{len(vals) - len(set(vals))} repeated values in {col}")
    for col in check.get("non_empty") or []:
        empty = sum(1 for v in values(col) if not v)
        if empty:
            bad.append(f"{col} is empty in {empty} rows")
    share = check.get("min_link_share", 0.9)
    for col in check.get("link_columns") or []:
        vals = values(col)
        links = sum(1 for v in vals if re.match(r"^https?://\S+$", v, re.I))
        if vals and links < len(vals) * share:
            bad.append(f"only {links} of {len(vals)} {col} values are links (needs {round(share * 100)}%)")
    ecol = check.get("email_column") or next((c for c in cols if "mail" in c.lower()), "")
    emails = [v for v in values(ecol) if v] if ecol else []
    if emails:
        doms = [e.rsplit("@", 1)[-1].lower() for e in emails]
        role = sum(1 for e in emails if dp.ROLE_LOCAL.match(e.split("@")[0]))
        free = sum(1 for d in doms if d in dp.FREE_MAIL)
        if "max_role_share" in check and role > len(emails) * check["max_role_share"]:
            bad.append(f"{role} of {len(emails)} emails are role inboxes (allowed {round(check['max_role_share'] * 100)}%)")
        if "max_free_share" in check and free > len(emails) * check["max_free_share"]:
            bad.append(f"{free} of {len(emails)} emails are free mail (allowed {round(check['max_free_share'] * 100)}%)")
        if "max_domain_share" in check:
            from collections import Counter
            top, k = (Counter(doms).most_common(1) or [("", 0)])[0]
            if k > max(1, len(emails) * check["max_domain_share"]):
                bad.append(f"{k} emails on {top} alone (allowed {round(check['max_domain_share'] * 100)}%)")
    if check.get("exclude_pattern"):
        rx = re.compile(check["exclude_pattern"], re.I)
        hit = [r for r in rows if any(rx.search(str(v or "")) for v in r.values())]
        if hit:
            bad.append(f"{len(hit)} rows match the excluded /{check['exclude_pattern']}/")
    for col, allowed in (check.get("allowed") or {}).items():
        ok = {a.lower() for a in allowed}
        off = [v for v in values(col) if v and v.lower() not in ok]
        if off:
            bad.append(f"{len(off)} {col} values outside {', '.join(allowed)} (e.g. {off[0]})")
    for col, least in (check.get("min_per") or {}).items():
        from collections import Counter
        have = Counter(v.lower() for v in values(col) if v)
        named = next((a for c, a in (check.get("allowed") or {}).items() if c.lower() == col.lower()), None)
        groups = list(named) if named else sorted({v for v in values(col) if v}, key=str.lower)
        short = [f"{g} {have.get(g.lower(), 0)}" for g in groups if have.get(g.lower(), 0) < least]
        if short or not groups:
            bad.append(f"fewer than {least} rows per {col}: " + (", ".join(short) if short else f"no {col} values"))
    return (not bad, f"`{rel}`: " + "; ".join(bad) if bad else "")


def _check_document(repo: Path, check: dict, skip: tuple[str, ...] = ()) -> tuple[bool, str]:
    from captain_claw.flight_deck import deliverable_profile as dp
    path, rel = _target(repo, check, (".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".rst"), skip)
    if path is None:
        return False, f"`{rel}` is missing" if rel else "no document at the top of the folder"
    prof = dp.profile_docx(path) if path.suffix.lower() == ".docx" else dp.profile_document(path)
    if prof.get("too_big") or prof.get("unread"):
        return True, "(couldn't read it here)"
    bad: list[str] = []
    words = prof.get("words", 0)
    if check.get("min_words") and words < check["min_words"]:
        bad.append(f"{words} words, fewer than {check['min_words']}")
    if check.get("max_words") and words > check["max_words"]:
        bad.append(f"{words} words, more than {check['max_words']}")
    heads = [h.lower() for h in prof.get("headings") or []]
    missing = [h for h in check.get("headings") or [] if not any(h.lower() in x for x in heads)]
    if missing:
        bad.append(f"no section on: {', '.join(missing)}")
    if check.get("no_placeholders") and prof.get("placeholders"):
        bad.append("placeholders left: " + ", ".join(f"{h} ×{k}" for h, k in prof["placeholders"]))
    if check.get("min_links") and prof.get("links", 0) < check["min_links"]:
        bad.append(f"{prof.get('links', 0)} links, fewer than {check['min_links']}")
    return (not bad, f"`{rel}`: " + "; ".join(bad) if bad else "")


def _check_python_syntax(repo: Path, check: dict, skip: tuple[str, ...] = ()) -> tuple[bool, str]:
    import ast
    errors: list[str] = []
    scanned = 0
    for p in repo.glob(str(check.get("glob") or "**/*.py")):
        if not p.is_file() or p.suffix != ".py" or _hidden(repo, p) or p.relative_to(repo).parts[0] in skip:
            continue
        scanned += 1
        if scanned > _SCAN_FILE_CAP:
            break
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            errors.append(f"{p.relative_to(repo).as_posix()}:{e.lineno}")
    return (not errors, "syntax errors in " + ", ".join(errors[:8]) if errors else "")


def _relpath(p) -> str:
    """A safe repo-relative path, or "" — no absolutes, no ``..`` escape."""
    s = str(p or "").strip().replace("\\", "/")
    if not s or s.startswith("/") or s.startswith("~"):
        return ""
    parts = [seg for seg in s.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        return ""
    return "/".join(parts)[:300]


# ── persistence (the contract of record, user-editable) ──────────────

def load(repo: Path | str) -> list[dict] | None:
    """The persisted contract, or None when the repo has none / it's unreadable."""
    p = Path(repo) / CONTRACT_FILE
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = raw.get("constraints") if isinstance(raw, dict) else None
        # Re-normalize so a hand-edited file can't smuggle a malformed check in.
        return parse_contract(json.dumps({"constraints": items or []})) or None
    except (OSError, ValueError, TypeError) as e:
        log.warning("code contract load failed", error=str(e))
        return None


def save(repo: Path | str, constraints: list[dict], intent: str) -> None:
    """Best-effort write of the contract of record. Never raises."""
    try:
        Path(repo).mkdir(parents=True, exist_ok=True)
        (Path(repo) / CONTRACT_FILE).write_text(json.dumps({
            "version": 1,
            "derived_from": (intent or "").strip()[:300],
            "created_at": int(time.time()),
            "note": "Acceptance contract. Hand-edit freely — a follow-up turn "
                    "loads this file instead of re-deriving the rules.",
            "constraints": constraints,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        log.warning("code contract save failed", error=str(e))


# ── prompt injection ─────────────────────────────────────────────────

def contract_directive(constraints: list[dict]) -> str:
    """The block injected into build/fix prompts so the agent aims at the
    acceptance criteria (they are validated after the commit either way)."""
    if not constraints:
        return ""
    lines = ["\n\nACCEPTANCE CONTRACT — the task is done only when every rule "
             "below holds. They are checked automatically after you commit:"]
    for i, c in enumerate(constraints, 1):
        lines.append(f"{i}. [{c['severity']}] {c['text']}")
    return "\n".join(lines)


# ── deterministic validation ─────────────────────────────────────────

def measure(repo: Path, constraints: list[dict], skip: tuple[str, ...] = ()) -> dict[str, int]:
    """How far the work has got on the rules that ask for an amount: a table's rows
    (``min_rows``), a document's words (``min_words``), by rule id. What tells a run
    that is still growing from one that is stuck, when the same rule fails round after
    round. Never raises; a rule it can't measure is left out."""
    from captain_claw.flight_deck import deliverable_profile as dp
    out: dict[str, int] = {}
    for c in constraints or []:
        check = c.get("check") or {}
        try:
            if check.get("type") == "table" and check.get("min_rows"):
                path, _ = _target(repo, check, (".csv", ".tsv"), skip)
                if path is not None:
                    prof = dp.profile_table(path)
                    if not prof.get("too_big"):
                        out[str(c["id"])] = int(prof.get("rows", 0))
            elif check.get("type") == "document" and check.get("min_words"):
                path, _ = _target(repo, check, (".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".rst"), skip)
                if path is not None:
                    prof = dp.profile_docx(path) if path.suffix.lower() == ".docx" else dp.profile_document(path)
                    if not (prof.get("too_big") or prof.get("unread")):
                        out[str(c["id"])] = int(prof.get("words", 0))
        except Exception:  # noqa: BLE001 — a measure is a hint, never a failure
            continue
    if not any((c.get("check") or {}).get("type") == "table" and str(c["id"]) in out for c in constraints or []):
        # The amount sits in a script or a reviewer's rule ("leads.csv has 60 rows"): count
        # the rows of the tables the rules name, so growth still shows.
        named: list[str] = []
        for c in constraints or []:
            where = f"{c.get('text') or ''} {(c.get('check') or {}).get('cmd') or ''}"
            named += [n for n in re.findall(r"[\w./-]+\.(?:csv|tsv)\b", where) if n not in named]
        for name in named[:5]:
            rel = _relpath(name)
            path = repo / rel if rel else None
            if path is None or not path.is_file() or rel.split("/", 1)[0] in skip:
                continue
            try:
                prof = dp.profile_table(path)
                if not prof.get("too_big"):
                    out[f"rows:{rel}"] = int(prof.get("rows", 0))
            except Exception:  # noqa: BLE001
                continue
    return out


async def validate(repo: Path, constraints: list[dict], *,
                   runner: CommandRunner | None = None, skip: tuple[str, ...] = ()) -> dict:
    """Deterministic pass over the repo. Returns
    ``{passed, failed, unresolved}`` — ``unresolved`` holds ``judge``-type rules
    (and any that errored), which the caller may send to the LLM judge and fold
    back with :func:`apply_judgement`. ``skip``: top-level folders a scan of the
    whole repo leaves out (a run's working notes). Never raises."""
    res: dict = {"passed": [], "failed": [], "unresolved": []}
    run = runner or shell_command_runner
    for c in constraints or []:
        check = c.get("check") or {}
        ctype = check.get("type")
        entry = {"id": c["id"], "text": c["text"], "severity": c["severity"]}
        # As stored: the old cut came after trimming, so it can end in a space.
        if ctype == "command" and len(str(check.get("cmd") or "")) == _OLD_CMD_CUT:
            entry["how"] = "judge"
            entry["note"] = "its script was cut short when it was written, so the reviewers decide it"
            res["unresolved"].append(entry)
            continue
        try:
            if ctype == "command":
                ok, note = await _check_command(repo, check, run)
            elif ctype == "file_exists":
                ok, note = _check_file_exists(repo, check)
            elif ctype == "file_contains":
                ok, note = _check_file_contains(repo, check)
            elif ctype == "no_pattern":
                ok, note = _check_no_pattern(repo, check, skip)
            elif ctype == "table":
                ok, note = _check_table(repo, check, skip)
            elif ctype == "document":
                ok, note = _check_document(repo, check, skip)
            elif ctype == "python_syntax":
                ok, note = _check_python_syntax(repo, check, skip)
            else:  # judge (or unknown) → let the LLM decide
                entry["how"] = "judge"
                res["unresolved"].append(entry)
                continue
        except Exception as e:  # noqa: BLE001 — a broken check never fails the run
            log.warning("contract check errored", id=c.get("id"), error=str(e))
            entry["how"] = "judge"
            res["unresolved"].append(entry)
            continue
        entry["how"] = f"deterministic:{ctype}"
        if ok:
            res["passed"].append(entry)
        else:
            entry["note"] = note
            res["failed"].append(entry)
    return res


async def _check_command(repo: Path, check: dict, run: CommandRunner) -> tuple[bool, str]:
    import asyncio
    cmd = str(check.get("cmd") or "").strip()
    if not cmd:
        return True, ""
    try:
        ok, output = await asyncio.wait_for(run(cmd, str(repo)), timeout=_CMD_TIMEOUT_S)
    except asyncio.TimeoutError:
        return False, f"`{cmd}` timed out after {int(_CMD_TIMEOUT_S)}s"
    except Exception as e:  # noqa: BLE001
        return False, f"`{cmd}` errored: {e}"
    if ok:
        return True, ""
    tail = (output or "").strip()
    if len(tail) > _CMD_OUTPUT_CAP:
        tail = tail[-_CMD_OUTPUT_CAP:]
    return False, f"`{cmd}` exited non-zero:\n{tail}"


def _check_file_exists(repo: Path, check: dict) -> tuple[bool, str]:
    rel = _relpath(check.get("path"))
    if not rel:
        return True, ""
    return ((repo / rel).exists(), f"expected file `{rel}` does not exist")


def _check_file_contains(repo: Path, check: dict) -> tuple[bool, str]:
    rel = _relpath(check.get("path"))
    pat = str(check.get("pattern") or "")
    if not rel or not pat:
        return True, ""
    p = repo / rel
    if not p.is_file():
        return False, f"`{rel}` is missing (expected to contain /{pat}/)"
    try:
        body = p.read_text(encoding="utf-8", errors="replace")[:_FILE_READ_CAP]
        rx = re.compile(pat, re.MULTILINE)  # as grep reads them: ^ and $ are lines
    except re.error as e:
        return True, f"(unparseable pattern, skipped: {e})"
    return (bool(rx.search(body)), f"`{rel}` does not match /{pat}/")


def _check_no_pattern(repo: Path, check: dict, skip: tuple[str, ...] = ()) -> tuple[bool, str]:
    pat = str(check.get("pattern") or "")
    glob = str(check.get("glob") or "**/*")
    if not pat:
        return True, ""
    try:
        rx = re.compile(pat, re.MULTILINE)
    except re.error as e:
        return True, f"(unparseable pattern, skipped: {e})"
    hits: list[str] = []
    scanned = 0
    for p in repo.glob(glob):
        if not p.is_file() or _hidden(repo, p) or p.relative_to(repo).parts[0] in skip:
            continue
        scanned += 1
        if scanned > _SCAN_FILE_CAP:
            break
        try:
            body = p.read_text(encoding="utf-8", errors="replace")[:_FILE_READ_CAP]
        except OSError:
            continue
        if rx.search(body):
            hits.append(p.relative_to(repo).as_posix())
            if len(hits) >= 5:
                break
    if hits:
        return False, f"/{pat}/ still present in: {', '.join(hits)}"
    return True, ""


def _hidden(repo: Path, p: Path) -> bool:
    return any(part.startswith(".") for part in p.relative_to(repo).parts)


# ── LLM judge fallback (for ``judge``-type + errored rules) ───────────

def judge_prompt(constraints: list[dict], tree: str) -> str:
    listing = "\n".join(f'- id "{c["id"]}": {c["text"]}' for c in constraints)
    return (
        "You are checking a finished code change against acceptance rules that a "
        "script cannot verify. For EACH rule below, decide from the repo's file "
        "list (and your judgement of the task) whether it is satisfied. Reply "
        "ONLY with a JSON array — no prose:\n"
        '[{"id": "<id>", "verdict": "pass|fail|unclear", '
        '"note": "<one line, ONLY for fail/unclear>"}]\n'
        'Use "unclear" when you genuinely cannot tell — do not guess.\n\n'
        f"## Rules\n{listing}\n\n## Files in the repo\n{tree[:6000]}"
    )


def parse_judgement(output: str) -> list[dict]:
    if not output:
        return []
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("["), text.rfind("]")
        blob = text[start:end + 1] if 0 <= start < end else None
    if not blob:
        return []
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("id") or "").strip():
            continue
        verdict = str(item.get("verdict") or "").strip().casefold()
        if verdict not in ("pass", "fail", "unclear"):
            verdict = "unclear"
        out.append({"id": str(item["id"]).strip()[:20], "verdict": verdict,
                    "note": str(item.get("note") or "").strip()[:200]})
    return out


def apply_judgement(result: dict, judgements: list[dict]) -> dict:
    """Fold judge verdicts over the unresolved rules. A rule the judge didn't
    mention (or called unclear) stays unresolved — recorded, never guessed."""
    by_id = {j["id"]: j for j in judgements}
    still: list[dict] = []
    for c in result.get("unresolved", []):
        j = by_id.get(c["id"])
        entry = {**c, "how": "judged", "note": (j or {}).get("note", "")}
        if j and j["verdict"] == "pass":
            result["passed"].append(entry)
        elif j and j["verdict"] == "fail":
            result["failed"].append(entry)
        else:
            still.append(c)
    result["unresolved"] = still
    return result


# ── triage bridge + summary ──────────────────────────────────────────

def as_review_entry(result: dict) -> dict | None:
    """Turn failed acceptance rules into a synthetic reviewer report for triage,
    mirroring ``code_verify.as_review_entry``. Returns None when nothing failed
    (a clean contract shouldn't manufacture work)."""
    failed = [f for f in result.get("failed", [])
              if f.get("severity") in ("critical", "major")]
    if not failed:
        return None
    lines = [
        "GROUND TRUTH — the acceptance contract is NOT satisfied. These rules "
        "were derived from the approved plan and checked automatically; each must "
        "be made true before the task is done:", ""]
    for f in failed:
        lines.append(f"- [{f['severity']}] {f['text']}"
                     + (f"\n  → {f['note']}" if f.get("note") else ""))
    return {"role": "Acceptance Contract", "id": "acceptance-contract",
            "output": "\n".join(lines)}


def summarize(result: dict) -> dict:
    """Compact record for ``analysis``/quality metrics. Failures kept verbatim —
    they're what a follow-up round or the blocking gate acts on."""
    failed = result.get("failed") or []
    return {
        "checked": (len(result.get("passed") or []) + len(failed)
                    + len(result.get("unresolved") or [])),
        "passed": len(result.get("passed") or []),
        "failed_critical": sum(1 for f in failed if f.get("severity") == "critical"),
        "failed_major": sum(1 for f in failed if f.get("severity") == "major"),
        "unresolved": len(result.get("unresolved") or []),
        # ``unclear`` mirrors research_contract.summarize so the shared
        # build_quality_metrics(contract=…) reads the same key from both.
        "unclear": len(result.get("unresolved") or []),
        "failed": [{"id": f["id"], "text": f["text"], "severity": f["severity"],
                    "how": f.get("how", ""), "note": f.get("note", "")}
                   for f in failed],
    }
