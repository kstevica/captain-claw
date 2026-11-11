"""Measure the anonymization name-recognition pack (docs/anonymization-plan.md, P4).

Runs the bare-name set (captain_claw/anonymize/name_cases.json) and the shared
cases (cases.json) through the engine, with code alone or code + one of the
pack's models, at several score cut-offs, and prints what each finds and breaks.

    python scripts/measure_name_models.py code
    python scripts/measure_name_models.py classla/bcms-bertic-ner --json out.json

The model must already be downloaded (the deck's "Download and start", or
``anonymize_ner._download``); this script never downloads. One model per run,
so the memory figure is that model's own.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from captain_claw.anonymize import detectors
from captain_claw.anonymize.name_eval import load, score, shared_broken
from captain_claw.flight_deck import anonymize_ner

CUTOFFS = (0.5, 0.7, 0.8, 0.9)


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="'code' or one of: " + ", ".join(anonymize_ner.MODELS))
    ap.add_argument("--json", help="write the full results here")
    ap.add_argument("--people-only", action="store_true", help="drop the model's company (ORG) labels")
    ap.add_argument("--all-labels", action="store_true", help="take every PER/ORG label, not just the pack's own choice")
    args = ap.parse_args()

    names = load("name_cases.json")
    shared_cases = load("cases.json")
    report: dict = {"model": args.model, "cases": len(names)}

    if args.model == "code":
        detectors.set_ner_backend(None)
        cutoffs: tuple = (None,)
    else:
        if args.model not in anonymize_ner.MODELS:
            ap.error("unknown model")
        if not anonymize_ner.downloaded(args.model):
            print("not downloaded — this script never downloads", file=sys.stderr)
            return 2
        t = time.perf_counter()
        pipe = anonymize_ner._load_pipeline(args.model)
        report["load_s"] = round(time.perf_counter() - t, 1)
        if args.all_labels:
            anonymize_ner.MODEL_LABELS[args.model] = frozenset({"PER", "ORG"})
        backend = anonymize_ner._make_backend(pipe, args.model)
        if args.people_only:
            full = backend
            backend = lambda text: [f for f in full(text) if f[2] != "ORG"]  # noqa: E731
            report["people_only"] = True
        detectors.set_ner_backend(backend)
        backend("Zagreb.")  # warm up
        page = " ".join(c["text"] for c in names)  # ~5 KB of prose
        t = time.perf_counter()
        for _ in range(3):
            backend(page)
        report["ms_per_kb"] = round((time.perf_counter() - t) * 1000 / 3 / (len(page.encode()) / 1024), 1)
        cutoffs = CUTOFFS

    report["runs"] = {}
    for cut in cutoffs:
        if cut is not None:
            detectors.NER_MIN_SCORE = cut
        r = score(names)
        r["shared_broken"] = shared_broken(shared_cases)
        report["runs"][str(cut or "code")] = r
    report["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10))

    print(f"== {args.model}{' (people only)' if args.people_only else ''}  ({len(names)} cases)")
    for k in ("load_s", "ms_per_kb", "max_rss_mb"):
        if k in report:
            print(f"   {k}: {report[k]}")
    for cut, r in report["runs"].items():
        print(f"-- cut-off {cut}: people {pct(r['people_hit'], r['people'])} · orgs {pct(r['orgs_hit'], r['orgs'])}"
              f" · kept {pct(r['keep_ok'], r['keep'])} · public hidden {pct(r['either_hit'], r['either'])}"
              f" · verify failed {len(r['verify_failed'])} · {r['ms_per_case']:.0f} ms/case")
        for lang, v in sorted(r["by_lang"].items()):
            print(f"     {lang}: people {pct(v['people_hit'], v['people'])} · kept {pct(v['keep_ok'], v['keep'])}")
        print("     missed:", ", ".join(r["missed"]) or "-")
        print("     broke:", ", ".join(r["broken"]) or "-")
        print("     shared cases broken:", ", ".join(r["shared_broken"]) or "-")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
