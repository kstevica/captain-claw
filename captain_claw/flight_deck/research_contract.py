"""Constraints contract — the task's hard rules, derived once and checked always.

R9's rubric answers "is everything COVERED?"; this answers "does the deliverable
BREAK any rule the task states?" — numeric limits, required relationships
between quantities, hard dates, non-negotiables. The DIGIT SPARK failure class
(RC2b/RC3): every run re-derived the rules ad hoc and nobody validated against
them, so a 100%-aid-intensity budget sailed through.

Design:

* **Derived once, persisted, reused.** One reason-tier call extracts the rules
  into ``.contract.json`` in the run's shared VFS folder. Continuation rounds in
  the same folder LOAD it instead of re-deriving; the user can hand-edit the
  file between rounds (it is the contract of record).
* **Deterministic where possible.** A constraint's ``check`` references facts-
  ledger keys (increment 4). When the ledger has the values, validation is pure
  arithmetic via a tiny recursive-descent evaluator — numbers, ledger keys,
  ``+ - * /``, parentheses, and (chained) comparisons. No ``eval()``, no
  attribute access, nothing but arithmetic over a value dict.
* **LLM-judged where not.** Rules that aren't numeric (language, format,
  mandatory sections) or whose keys the ledger doesn't have fall back to one
  judge call against the deliverable text.

Pure module (stdlib only): prompts, parser, evaluator, persistence, renderers.
The route files own the model calls and the wiring.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

CONTRACT_FILE = ".contract.json"

_SEVERITIES = ("critical", "major", "minor")
_CHECK_TYPES = ("range", "equals", "expr", "judge")
_MAX_CONSTRAINTS = 20
_REL_TOL = 0.005  # equality tolerance, mirrors the ledger/consistency modules


# ── derive ───────────────────────────────────────────────────────────

def derive_prompt(intent: str) -> str:
    return (
        "You are extracting the HARD CONSTRAINTS a deliverable must satisfy — "
        "the rules that, if broken, invalidate it. From the task below, list "
        "every explicit or clearly implied checkable rule: numeric limits "
        "(ranges, caps, minimums), required relationships between quantities, "
        "hard dates, and non-negotiable requirements. Do NOT invent policy the "
        "task doesn't state; do NOT list style preferences or coverage items.\n\n"
        "Reply ONLY with JSON — no prose:\n"
        '{"constraints": [{"id": "c1", "text": "<the rule, one line>", '
        '"severity": "critical|major", "check": {...}}]}\n'
        "check shapes (pick the FIRST that fits):\n"
        '- {"type": "range", "key": "<snake_case quantity>", "min": <num>, '
        '"max": <num>} — either bound may be omitted\n'
        '- {"type": "equals", "key": "<snake_case quantity>", "value": <num>}\n'
        '- {"type": "expr", "expr": "<comparison over quantities and numbers, '
        "e.g. 'grant_eur <= 0.5 * total_eligible_cost_eur' or "
        "'80000 <= grant_eur <= 150000'>\"}\n"
        '- {"type": "judge"} — only for rules that cannot be checked '
        "numerically (language, format, required documents)\n"
        "Name quantities with short snake_case keys (they resolve against the "
        "run's shared facts ledger — the same keys workers record values "
        "under). severity: critical = breaking it invalidates the whole "
        "deliverable; major = a serious defect. If the task states no hard "
        'constraints, return {"constraints": []}.\n\n'
        "## Task\n"
        f"{intent}"
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
        if not isinstance(item, dict):
            continue
        text_ = str(item.get("text") or "").strip()[:300]
        if not text_:
            continue
        sev = str(item.get("severity") or "").strip().casefold()
        if sev not in _SEVERITIES:
            sev = "major"
        check = item.get("check") if isinstance(item.get("check"), dict) else {}
        ctype = str(check.get("type") or "").strip().casefold()
        norm: dict = {"type": "judge"}
        if ctype == "range" and str(check.get("key") or "").strip():
            mn, mx = _num(check.get("min")), _num(check.get("max"))
            if mn is not None or mx is not None:
                norm = {"type": "range", "key": _key(check["key"])}
                if mn is not None:
                    norm["min"] = mn
                if mx is not None:
                    norm["max"] = mx
        elif ctype == "equals" and str(check.get("key") or "").strip():
            v = _num(check.get("value"))
            if v is not None:
                norm = {"type": "equals", "key": _key(check["key"]), "value": v}
        elif ctype == "expr" and str(check.get("expr") or "").strip():
            norm = {"type": "expr", "expr": str(check["expr"]).strip()[:300]}
        out.append({"id": str(item.get("id") or f"c{i}").strip()[:20],
                    "text": text_, "severity": sev, "check": norm})
    return out


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _key(k: str) -> str:
    return re.sub(r"[^\w]+", "_", str(k).strip().casefold()).strip("_")[:120]


# ── persistence (the contract of record, user-editable) ──────────────

def load(project: Path | str) -> list[dict] | None:
    """The persisted contract, or None when the folder has none / it's unreadable."""
    p = Path(project) / CONTRACT_FILE
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = raw.get("constraints") if isinstance(raw, dict) else None
        # Re-normalize through the parser path so a hand-edited file can't
        # smuggle a malformed check into the evaluator.
        return parse_contract(json.dumps({"constraints": items or []})) or None
    except (OSError, ValueError, TypeError) as e:
        log.warning("contract load failed", error=str(e))
        return None


def save(project: Path | str, constraints: list[dict], intent: str) -> None:
    """Best-effort write of the contract of record. Never raises."""
    try:
        Path(project).mkdir(parents=True, exist_ok=True)
        (Path(project) / CONTRACT_FILE).write_text(json.dumps({
            "version": 1,
            "derived_from": (intent or "").strip()[:300],
            "created_at": int(time.time()),
            "note": "Hand-edit freely — continuation rounds load this file "
                    "instead of re-deriving the rules.",
            "constraints": constraints,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        log.warning("contract save failed", error=str(e))


# ── prompt injection ─────────────────────────────────────────────────

def contract_directive(constraints: list[dict], ledger: bool = False) -> str:
    """The block injected into worker/reporter prompts."""
    if not constraints:
        return ""
    lines = ["\n\nHARD CONSTRAINTS — the deliverable MUST satisfy every one "
             "(these are validated after assembly):"]
    for i, c in enumerate(constraints, 1):
        lines.append(f"{i}. [{c['severity']}] {c['text']}")
    if ledger:
        lines.append(
            "When your piece establishes one of the quantities these rules "
            "name, record it in the facts ledger under that exact snake_case "
            "key so the rules can be checked deterministically.")
    return "\n".join(lines)


# ── the safe evaluator (arithmetic + comparisons over a value dict) ──

class MissingKey(Exception):
    """A referenced quantity is not in the ledger — the check is unresolvable."""


_TOKEN_RE = re.compile(r"(<=|>=|==|!=|<|>|\+|\-|\*|/|\(|\)|[A-Za-z_]\w*|\d+(?:\.\d+)?)")
_CMP_OPS = ("<=", ">=", "==", "!=", "<", ">")


def _tokenize(expr: str) -> list[str]:
    toks, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"bad token at {expr[pos:pos + 10]!r}")
        toks.append(m.group(1))
        pos = m.end()
    return toks


class _Parser:
    def __init__(self, tokens: list[str], values: dict[str, float]):
        self.t, self.i, self.v = tokens, 0, values

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str | None:
        tok = self.peek()
        self.i += 1
        return tok

    def arith(self) -> float:
        x = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            y = self.term()
            x = x + y if op == "+" else x - y
        return x

    def term(self) -> float:
        x = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            y = self.factor()
            if op == "/":
                if y == 0:
                    raise ValueError("division by zero")
                x = x / y
            else:
                x = x * y
        return x

    def factor(self) -> float:
        tok = self.take()
        if tok == "(":
            x = self.arith()
            if self.take() != ")":
                raise ValueError("missing )")
            return x
        if tok == "-":
            return -self.factor()
        if tok is None:
            raise ValueError("unexpected end of expression")
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            return float(tok)
        if re.fullmatch(r"[A-Za-z_]\w*", tok):
            key = tok.casefold()
            if key not in self.v:
                raise MissingKey(tok)
            return float(self.v[key])
        raise ValueError(f"unexpected {tok!r}")


def _cmp(a: float, op: str, b: float) -> bool:
    if op == "==":
        return abs(a - b) <= _REL_TOL * max(abs(a), abs(b))
    if op == "!=":
        return abs(a - b) > _REL_TOL * max(abs(a), abs(b))
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    return a > b


def eval_predicate(expr: str, values: dict[str, float]) -> bool:
    """Evaluate a comparison (optionally chained: ``80000 <= x <= 150000``) over
    numbers and ledger keys. Raises ``MissingKey`` for an unknown key and
    ``ValueError`` for anything that isn't a pure arithmetic predicate."""
    p = _Parser(_tokenize(expr), {str(k).casefold(): float(v)
                                  for k, v in (values or {}).items()
                                  if v is not None})
    left = p.arith()
    ok, count = True, 0
    while p.peek() in _CMP_OPS:
        op = p.take()
        right = p.arith()
        ok = ok and _cmp(left, op, right)
        left, count = right, count + 1
    if p.peek() is not None:
        raise ValueError(f"unexpected {p.peek()!r}")
    if count == 0:
        raise ValueError("not a predicate (no comparison)")
    return ok


def evaluate_check(check: dict, values: dict[str, float]) -> bool:
    """True/False for a deterministic check. Raises ``MissingKey``/``ValueError``
    when it cannot be decided from the values (caller treats as unresolved)."""
    vals = {str(k).casefold(): float(v) for k, v in (values or {}).items()
            if v is not None}
    ctype = check.get("type")
    if ctype in ("range", "equals"):
        key = str(check.get("key") or "").casefold()
        if key not in vals:
            raise MissingKey(key)
        v = vals[key]
        if ctype == "equals":
            return _cmp(v, "==", float(check["value"]))
        ok = True
        if "min" in check:
            ok = ok and v >= float(check["min"])
        if "max" in check:
            ok = ok and v <= float(check["max"])
        return ok
    if ctype == "expr":
        return eval_predicate(str(check.get("expr") or ""), vals)
    raise ValueError(f"not deterministically checkable: {ctype}")


# ── validation ───────────────────────────────────────────────────────

def validate(constraints: list[dict], ledger_values: dict[str, float]) -> dict:
    """Deterministic pass. Returns ``{passed, failed, unresolved, unclear}`` —
    ``unresolved`` holds the constraints the ledger couldn't decide (judge-type,
    missing keys, invalid expressions); the caller sends those to the LLM judge
    and folds the verdicts back with :func:`apply_judgement`."""
    res: dict = {"passed": [], "failed": [], "unresolved": [], "unclear": []}
    for c in constraints or []:
        try:
            ok = evaluate_check(c.get("check") or {}, ledger_values)
        except (MissingKey, ValueError, TypeError, KeyError):
            res["unresolved"].append(c)
            continue
        bucket = res["passed"] if ok else res["failed"]
        entry = {"id": c["id"], "text": c["text"], "severity": c["severity"],
                 "how": "deterministic"}
        if not ok:
            entry["note"] = "ledger values violate this rule"
        bucket.append(entry)
    return res


def judge_prompt(deliverable: str, constraints: list[dict]) -> str:
    listing = "\n".join(f'- id "{c["id"]}": {c["text"]}' for c in constraints)
    return (
        "You are checking a finished deliverable against hard constraints. For "
        "EACH constraint below, decide from the deliverable alone whether it is "
        "satisfied. Reply ONLY with a JSON array — no prose:\n"
        '[{"id": "<id>", "verdict": "pass|fail|unclear", '
        '"note": "<one line, ONLY for fail/unclear>"}]\n'
        "Use \"unclear\" when the deliverable genuinely doesn't show whether "
        "the rule holds — do not guess.\n\n"
        f"## Constraints\n{listing}\n\n"
        f"## Deliverable\n{deliverable[:8000]}"
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
    """Fold the judge's verdicts over the unresolved constraints. A constraint
    the judge didn't mention (or called unclear) lands in ``unclear`` — recorded,
    never guessed."""
    by_id = {j["id"]: j for j in judgements}
    still: list[dict] = []
    for c in result["unresolved"]:
        j = by_id.get(c["id"])
        entry = {"id": c["id"], "text": c["text"], "severity": c["severity"],
                 "how": "judged", "note": (j or {}).get("note", "")}
        if j and j["verdict"] == "pass":
            result["passed"].append(entry)
        elif j and j["verdict"] == "fail":
            result["failed"].append(entry)
        else:
            result["unclear"].append(entry)
    result["unresolved"] = still
    return result


def summarize(result: dict) -> dict:
    """The compact record for ``analysis.contract`` (failures kept verbatim —
    they're what a follow-up round or the blocking gate acts on)."""
    failed = result.get("failed") or []
    return {
        "checked": (len(result.get("passed") or []) + len(failed)
                    + len(result.get("unclear") or [])
                    + len(result.get("unresolved") or [])),
        "passed": len(result.get("passed") or []),
        "failed_critical": sum(1 for f in failed if f.get("severity") == "critical"),
        "failed_major": sum(1 for f in failed if f.get("severity") != "critical"),
        "unclear": (len(result.get("unclear") or [])
                    + len(result.get("unresolved") or [])),
        "failed": [{"id": f["id"], "text": f["text"], "severity": f["severity"],
                    "how": f["how"], "note": f.get("note", "")} for f in failed],
    }
