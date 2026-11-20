"""First Mate: a list factory's recipe, pinned.

A list funnel run follows a recipe a model writes from the charter: the rules,
what each item is read for, the checks. Written fresh on every run, it came out
different every time (a rule there or not, an alternative there or not, a filter
that drops everything), so no two runs of one charter compared.

One recipe per charter now: the first run that builds one pins it in the
factory's data folder, and every later run of the same charter and done-checks
follows it. The owner can read it, edit it by hand (its rules, the columns a row
must fill, its filters) and have it rebuilt. A change to the charter or to the
done-checks makes it stale: the next run builds a new one.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

FILE = "_recipe.json"  # in the factory's data folder (it goes with the factory)


def _folder(factory_id: str) -> Path:
    from captain_claw.flight_deck import first_mate_data
    return first_mate_data._folder(factory_id)


def path(factory_id: str) -> Path:
    return _folder(factory_id) / FILE


def work_key(version: dict[str, Any] | None) -> str:
    """What the recipe was written from: the charter's words and its done-checks."""
    v = version or {}
    raw = json.dumps([v.get("charter"), (v.get("checks") or {}).get("contract") or []], sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _read(factory_id: str) -> dict[str, Any] | None:
    try:
        got = json.loads(path(factory_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) and isinstance(got.get("recipe"), dict) else None


def pinned(factory_id: str, version: dict[str, Any] | None) -> dict[str, Any] | None:
    """The factory's pinned recipe with when it was built and whether it was edited by
    hand; None when there is none, or it was written from another charter."""
    got = _read(factory_id)
    if got is None or got.get("work") != work_key(version):
        return None
    return got


def pin(factory_id: str, version: dict[str, Any] | None, recipe: dict[str, Any], *, edited: bool = False,
        run_id: str = "") -> dict[str, Any]:
    out = {"work": work_key(version), "version": (version or {}).get("version"), "recipe": recipe,
           "edited": edited, "built_at": time.strftime("%Y-%m-%d %H:%M"), "run": run_id}
    before = _read(factory_id)
    if edited and before:  # an edit keeps when it was built, and says when it was changed
        out["built_at"], out["run"] = before.get("built_at", out["built_at"]), before.get("run", "")
        out["edited_at"] = time.strftime("%Y-%m-%d %H:%M")
    target = path(factory_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def forget(factory_id: str) -> bool:
    try:
        path(factory_id).unlink()
        return True
    except OSError:
        return False


def _version_of(run: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    cfg = run.get("config") or {}
    fid = str(cfg.get("factory_id") or "")
    if not fid:
        return "", None
    store = fms.get_store()
    no = cfg.get("fm_version")
    return fid, (store.version(fid, int(no)) if no is not None else store.live(fid))


def for_run(run: dict[str, Any], folder: Path) -> tuple[dict[str, Any] | None, str]:
    """The pinned recipe a factory's run follows, and a few words on it; (None, "") for
    a run that isn't a factory's, a factory with none, or one whose recipe names a
    file this run wasn't given."""
    try:
        fid, version = _version_of(run)
        got = pinned(fid, version) if fid else None
    except Exception:  # noqa: BLE001 — the run builds its own
        return None, ""
    if got is None:
        return None, ""
    recipe = got["recipe"]
    for s in recipe.get("sources") or []:
        if s.get("path") and not (folder / s["path"]).is_file():
            return None, ""  # written for a file this run doesn't have
    return recipe, ("the factory's pinned recipe, built " + str(got.get("built_at") or "earlier")
                    + (", edited by hand" if got.get("edited") else ""))


def pin_for_run(run: dict[str, Any], recipe: dict[str, Any]) -> bool:
    """Pin a recipe a factory's run just built, when its charter has none yet."""
    try:
        fid, version = _version_of(run)
        if not fid or version is None or pinned(fid, version) is not None:
            return False
        pin(fid, version, recipe, run_id=str(run.get("id") or ""))
        return True
    except Exception:  # noqa: BLE001 — pinning is for the next run; this one goes on
        return False


def edit(factory_id: str, version: dict[str, Any] | None, patch: dict[str, Any]) -> dict[str, Any]:
    """The owner's hand edit of the pinned recipe: its rules (words, must or prefer, the
    looser reading, what holds when the fact isn't found, the columns), the columns a
    row must fill, and its filters. Anything else in it stays as built."""
    from captain_claw.flight_deck import funnel
    got = pinned(factory_id, version)
    if got is None:
        raise fms.FirstMateError("this charter has no recipe yet: a run builds it")
    recipe = json.loads(json.dumps(got["recipe"]))
    cols = list(recipe.get("columns") or [])
    if isinstance(patch.get("rules"), list):
        was = {r.get("rule"): r for r in recipe.get("rules") or []}
        rules = []
        for r in patch["rules"][:funnel.RULES_MAX]:
            text = " ".join(str((r or {}).get("rule") or "").split())[:300]
            if not text:
                continue
            old = was.get(text) or {}
            looser = " ".join(str(r.get("looser") or "").split())[:300]
            kind = "prefer" if str(r.get("kind") or "").lower() == "prefer" and looser else "must"
            rule: dict[str, Any] = {"rule": text, "kind": kind,
                                    "columns": [c for c in r.get("columns") or old.get("columns") or [] if c in cols]}
            fact = str(r.get("fact") or old.get("fact") or "").lower()
            if fact in funnel.FACTS:
                rule["fact"] = fact
            if old.get("of_group"):
                rule["of_group"] = True
            if kind == "must":
                rule["protected"] = funnel._protected(rule, recipe)
            if looser:
                rule["looser"] = looser
            alt = " ".join(str(r.get("if_not_shown") or "").split())[:300]
            if alt:
                rule["if_not_shown"] = alt
            rules.append(rule)
        recipe["rules"] = rules
    if isinstance(patch.get("required"), list):
        need = [c for c in cols if c in set(map(str, patch["required"]))]
        rest = [c for c in recipe.get("checks") or [] if c.get("op") != "required"]
        recipe["checks"] = rest + [{"column": c, "op": "required"} for c in need]
    if isinstance(patch.get("filters"), list):
        recipe["filters"] = [f for f in patch["filters"] if isinstance(f, dict) and f.get("op") in funnel.FILTER_OPS
                             and str(f.get("column") or "").strip()][:20]
    if isinstance(patch.get("shortage_ok"), bool):
        recipe["shortage_ok"] = patch["shortage_ok"]
    return pin(factory_id, version, recipe, edited=True)


def view(factory_id: str, version: dict[str, Any] | None) -> dict[str, Any]:
    """The pinned recipe as the drawer shows it."""
    got = pinned(factory_id, version)
    if got is None:
        return {"recipe": None, "stale": _read(factory_id) is not None}
    r = got["recipe"]
    return {"recipe": {"item": r.get("item"), "output": r.get("output"), "columns": r.get("columns") or [],
                       "groups": r.get("groups") or {}, "group_by": r.get("group_by") or "",
                       "rules": r.get("rules") or [], "filters": r.get("filters") or [],
                       "required": [c["column"] for c in r.get("checks") or [] if c.get("op") == "required"],
                       "checks": [c for c in r.get("checks") or [] if c.get("op") != "required"],
                       "gather": r.get("gather") or [], "sources": r.get("sources") or [],
                       "shortage_ok": bool(r.get("shortage_ok"))},
            "edited": bool(got.get("edited")), "built_at": got.get("built_at"), "edited_at": got.get("edited_at"),
            "run": got.get("run"), "stale": False}
