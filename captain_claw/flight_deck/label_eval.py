"""Judge-vs-human agreement on Basna/Vatra run labels.

Every scored run carries the LLM judge's automatic label (`basna_runs.success`)
and, once someone votes, the human thumbs label (`human_success`). This module
turns those rows into an exportable eval set and measures how far the judge can
be trusted — Cohen's kappa over the (judge, human) pairs, overall and broken
down by mode / domain / archetype.

Pure functions only (no DB, no I/O), so the export route, the tests and an
offline notebook all compute the same numbers.

Labels are 1 = success, 0 = fail. A pair needs both labels; a run a human voted
on but the judge left unscored is "unpaired" — exported on request, never
counted in the agreement.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

# Column order for the export (CSV header, JSONL key order). The text columns
# are appended only when the caller opts into them — outputs and truths are
# large and may hold sensitive content.
EXPORT_COLUMNS = [
    "run_id", "session_id", "mode", "domain", "merge_kind", "archetype_id",
    "role", "tier", "model", "weight_at_run", "latency_ms", "run_created_at",
    "judge_success", "human_success", "human_feedback_at", "agree",
]
TEXT_COLUMNS = ["intent", "truth", "output"]


def _mode(config: str | None) -> str:
    try:
        cfg = json.loads(config or "{}")
    except (TypeError, json.JSONDecodeError):
        return "basna"
    return "vatra" if isinstance(cfg, dict) and cfg.get("mode") == "vatra" else "basna"


def export_row(row: dict, include_text: bool = False) -> dict:
    """Shape one labeled run (from FlightDeckDB.list_labeled_basna_runs)."""
    judge, human = row.get("judge_success"), row.get("human_success")
    out = {
        "run_id": row.get("id"),
        "session_id": row.get("session_id"),
        "mode": _mode(row.get("config")),
        "domain": row.get("domain") or "general",
        "merge_kind": row.get("merge_kind") or "",
        "archetype_id": row.get("archetype_id") or "",
        "role": row.get("role") or "",
        "tier": row.get("tier") or "",
        "model": row.get("model") or "",
        "weight_at_run": row.get("weight_at_run"),
        "latency_ms": row.get("latency_ms"),
        "run_created_at": row.get("created_at"),
        "judge_success": judge,
        "human_success": human,
        "human_feedback_at": row.get("human_feedback_at"),
        "agree": (judge == human) if judge is not None and human is not None else None,
    }
    if include_text:
        out.update({"intent": row.get("intent") or "", "truth": row.get("truth") or "",
                    "output": row.get("output") or ""})
    return out


def agreement(pairs: Iterable[tuple[int, int]]) -> dict:
    """Observed agreement, chance agreement and Cohen's kappa for (judge, human)
    label pairs, plus the 2×2 confusion counts.

    Kappa is None when it is undefined: no pairs, or both raters gave every run
    the same single label (chance agreement is 1, so there is nothing to beat).
    """
    both_success = both_fail = judge_only = human_only = 0
    for judge, human in pairs:
        if judge and human:
            both_success += 1
        elif not judge and not human:
            both_fail += 1
        elif judge:
            judge_only += 1   # judge said success, human said fail
        else:
            human_only += 1   # judge said fail, human said success
    n = both_success + both_fail + judge_only + human_only
    confusion = {"both_success": both_success, "both_fail": both_fail,
                 "judge_success_human_fail": judge_only,
                 "judge_fail_human_success": human_only}
    if n == 0:
        return {"n": 0, "agreement": None, "expected_agreement": None, "kappa": None,
                "judge_success_rate": None, "human_success_rate": None,
                "confusion": confusion}
    p_o = (both_success + both_fail) / n
    p_judge = (both_success + judge_only) / n
    p_human = (both_success + human_only) / n
    p_e = p_judge * p_human + (1 - p_judge) * (1 - p_human)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else None
    return {"n": n, "agreement": round(p_o, 4), "expected_agreement": round(p_e, 4),
            "kappa": round(kappa, 4) if kappa is not None else None,
            "judge_success_rate": round(p_judge, 4), "human_success_rate": round(p_human, 4),
            "confusion": confusion}


def _pair(r: dict) -> tuple[int, int] | None:
    j, h = r.get("judge_success"), r.get("human_success")
    return (j, h) if j is not None and h is not None else None


def summarize(rows: list[dict]) -> dict:
    """Agreement over export rows: overall, plus per mode / domain / archetype.

    Breakdowns list only groups with at least one pair, largest first.
    """
    paired = [r for r in rows if _pair(r)]

    def _by(key: str) -> list[dict]:
        groups: dict[str, list[tuple[int, int]]] = {}
        for r in paired:
            groups.setdefault(str(r.get(key) or ""), []).append(_pair(r))
        out = [{key: k, **agreement(v)} for k, v in groups.items()]
        return sorted(out, key=lambda g: (-g["n"], g[key]))

    return {
        "overall": agreement(_pair(r) for r in paired),
        "unpaired": len(rows) - len(paired),
        "by_mode": _by("mode"),
        "by_domain": _by("domain"),
        "by_archetype": _by("archetype_id"),
    }
