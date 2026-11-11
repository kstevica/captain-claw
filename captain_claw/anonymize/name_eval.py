"""Scoring for the name-recognition pack (docs/anonymization-plan.md, P4).

Runs the bare-name set (``name_cases.json``) and the shared cases
(``cases.json``) through the engine with whatever name model is on, and counts
what is hidden and what survives. Used by ``scripts/measure_name_models.py``
and the name-model test.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from captain_claw.anonymize import engine

HERE = Path(__file__).parent


def load(name: str) -> list[dict]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))["cases"]


def present(s: str, text: str) -> bool:
    """*s* stands in *text* as whole words (case-sensitive)."""
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text) is not None


def run_one(text: str) -> tuple[str | None, float]:
    """The one-way output (None when verification failed) and the ms it took."""
    t = time.perf_counter()
    try:
        out, _ = engine.anonymize_text(text, engine.VaultState(), engine.Options())
    except engine.VerifyFailed:
        out = None
    return out, (time.perf_counter() - t) * 1000


def score(cases: list[dict]) -> dict:
    tot = {k: 0 for k in ("people", "people_hit", "orgs", "orgs_hit", "keep", "keep_ok", "either", "either_hit")}
    missed, broken, either_hit, failed, by_lang, ms = [], [], [], [], {}, []

    def count(lang: dict, key: str, ok: bool) -> None:
        for bucket in (tot, lang):
            if key in bucket:
                bucket[key] += 1
                bucket[key + ("_ok" if key == "keep" else "_hit")] += ok

    for c in cases:
        out, took = run_one(c["text"])
        ms.append(took)
        if out is None:
            failed.append(c["id"])
            out = ""
        lang = by_lang.setdefault(c["lang"], {"people": 0, "people_hit": 0, "keep": 0, "keep_ok": 0})
        for key, label in (("people", ""), ("orgs", " (org)")):
            for s in c[key]:
                hit = not present(s, out)
                count(lang, key, hit)
                if not hit:
                    missed.append(f"{c['id']}:{s}{label}")
        for s in c["keep"]:
            ok = present(s, out)
            count(lang, "keep", ok)
            if not ok:
                broken.append(f"{c['id']}:{s}")
        for s in c["either"]:
            hit = not present(s, out)
            count(lang, "either", hit)
            if hit:
                either_hit.append(f"{c['id']}:{s}")
    return {**tot, "missed": missed, "broken": broken, "either_hidden": either_hit,
            "verify_failed": failed, "by_lang": by_lang, "ms_per_case": sum(ms) / max(1, len(ms))}


def shared_broken(cases: list[dict]) -> list[str]:
    """Shared-case expectations broken (the cases the deterministic lane holds to)."""
    bad = []
    for c in cases:
        if "deterministic" not in c["lanes"]:
            continue
        out, _ = run_one(c["text"])
        if out is None:
            bad.append(f"{c['id']}: verify failed")
            continue
        bad += [f"{c['id']}: kept {s!r}" for s in c["must_hide"] if s in out]
        bad += [f"{c['id']}: lost {s!r}" for s in c["must_keep"] if s not in out]
    return bad
