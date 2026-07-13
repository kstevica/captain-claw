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

import json
import re
import time
from pathlib import Path

from captain_claw.dubina.coder import CommandRunner, shell_command_runner
from captain_claw.logging import get_logger

log = get_logger(__name__)

CONTRACT_FILE = ".contract.json"

_SEVERITIES = ("critical", "major", "minor")
_CHECK_TYPES = ("command", "file_exists", "file_contains", "no_pattern", "judge")
_MAX_CONSTRAINTS = 20
_CMD_TIMEOUT_S = 180.0        # a contract command must not stall the build loop
_CMD_OUTPUT_CAP = 2000        # chars of a failing command's output kept for triage
_FILE_READ_CAP = 200_000      # bytes read per file when scanning content
_SCAN_FILE_CAP = 800          # max files a no_pattern scan will open


# ── derive ───────────────────────────────────────────────────────────

def derive_prompt(intent: str, plan_text: str = "") -> str:
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
        f"## Task\n{intent}{plan_block}"
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
        norm = {"type": "command", "cmd": str(check["cmd"]).strip()[:400]}
    elif ctype == "file_exists" and _relpath(check.get("path")):
        norm = {"type": "file_exists", "path": _relpath(check["path"])}
    elif ctype == "file_contains" and _relpath(check.get("path")) \
            and str(check.get("pattern") or "").strip():
        norm = {"type": "file_contains", "path": _relpath(check["path"]),
                "pattern": str(check["pattern"])[:300]}
    elif ctype == "no_pattern" and str(check.get("pattern") or "").strip():
        norm = {"type": "no_pattern", "pattern": str(check["pattern"])[:300],
                "glob": (str(check.get("glob") or "").strip() or "**/*")[:120]}
    return {"id": str(item.get("id") or f"a{i}").strip()[:20],
            "text": text_, "severity": sev, "check": norm}


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

async def validate(repo: Path, constraints: list[dict], *,
                   runner: CommandRunner | None = None) -> dict:
    """Deterministic pass over the repo. Returns
    ``{passed, failed, unresolved}`` — ``unresolved`` holds ``judge``-type rules
    (and any that errored), which the caller may send to the LLM judge and fold
    back with :func:`apply_judgement`. Never raises."""
    res: dict = {"passed": [], "failed": [], "unresolved": []}
    run = runner or shell_command_runner
    for c in constraints or []:
        check = c.get("check") or {}
        ctype = check.get("type")
        entry = {"id": c["id"], "text": c["text"], "severity": c["severity"]}
        try:
            if ctype == "command":
                ok, note = await _check_command(repo, check, run)
            elif ctype == "file_exists":
                ok, note = _check_file_exists(repo, check)
            elif ctype == "file_contains":
                ok, note = _check_file_contains(repo, check)
            elif ctype == "no_pattern":
                ok, note = _check_no_pattern(repo, check)
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
        rx = re.compile(pat)
    except re.error as e:
        return True, f"(unparseable pattern, skipped: {e})"
    return (bool(rx.search(body)), f"`{rel}` does not match /{pat}/")


def _check_no_pattern(repo: Path, check: dict) -> tuple[bool, str]:
    pat = str(check.get("pattern") or "")
    glob = str(check.get("glob") or "**/*")
    if not pat:
        return True, ""
    try:
        rx = re.compile(pat)
    except re.error as e:
        return True, f"(unparseable pattern, skipped: {e})"
    hits: list[str] = []
    scanned = 0
    for p in repo.glob(glob):
        if not p.is_file() or _hidden(repo, p):
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
