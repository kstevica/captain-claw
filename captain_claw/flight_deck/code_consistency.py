"""Phase C — cross-file interface consistency for Code.

``research_consistency.py`` verifies figures agree across a report's sections.
The code analog: does every local import resolve to a real definition? When
parallel slices (Phase B) evolve interfaces independently, the classic drift is
slice A renaming or dropping a symbol that slice B imports — a broken import that
only bites at runtime. This is the deterministic, low-false-positive check for
exactly that failure.

Scope is deliberately conservative — a false BLOCKING finding is worse than no
check at all:

* Python only (no-op for other languages — they get nothing rather than noise).
* Only LOCAL ``from X import a, b`` whose module resolves to a file IN the repo:
  every explicit relative import, and absolute imports whose top package is a
  real top-level package/module of the repo. Third-party/stdlib imports, star
  imports (``import *``), and dynamic attributes are left alone.
* A name is satisfied if the target module defines it (def/class/assignment),
  re-imports it, declares it in ``__all__``, or exposes it as a submodule.
  Anything genuinely absent is a CRITICAL "broken import" finding.

Pure stdlib (``ast`` + filesystem), zero model tokens.
"""

from __future__ import annotations

import ast
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_SKIP_DIRS = {".git", ".code", ".codemap", ".captain-claw", ".uploads", "saved",
              "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
              ".pytest_cache", ".mypy_cache"}
_MAX_FILES = 1200          # a huge repo shouldn't stall the check
_MAX_FINDINGS = 40


def check(repo: Path | str) -> list[dict]:
    """Every broken local import in the repo's Python files. Never raises."""
    repo = Path(repo)
    try:
        files = _py_files(repo)
    except Exception as e:  # noqa: BLE001
        log.warning("interface consistency: file walk failed", error=str(e))
        return []
    # Parse once; cache each module's top-level names + its AST.
    exports: dict[Path, set[str] | None] = {}   # None = unparseable (skip as target)
    trees: dict[Path, ast.Module | None] = {}
    for f in files:
        tree = _parse(f)
        trees[f] = tree
        exports[f] = _top_level_names(tree) if tree is not None else None

    findings: list[dict] = []
    for f in files:
        tree = trees.get(f)
        if tree is None:
            continue                       # importer itself doesn't parse — not our job
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve_module(repo, f, node)
            if target is None:
                continue                   # external / unresolved — never guessed
            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue               # star import — can't check, don't try
                if not _name_available(repo, target, name, exports):
                    rel = f.relative_to(repo).as_posix()
                    tgt_rel = target.relative_to(repo).as_posix()
                    findings.append({
                        "kind": "broken_import", "severity": "critical",
                        "file": rel,
                        "detail": f"{rel} imports `{name}` from `{tgt_rel}`, "
                                  f"but `{tgt_rel}` does not define or export it",
                    })
                    if len(findings) >= _MAX_FINDINGS:
                        return findings
    return findings


# ── file walking + parsing ───────────────────────────────────────────

def _py_files(repo: Path) -> list[Path]:
    out: list[Path] = []
    for p in repo.rglob("*.py"):
        rel = p.relative_to(repo)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        out.append(p)
        if len(out) >= _MAX_FILES:
            break
    return out


def _parse(f: Path) -> ast.Module | None:
    try:
        return ast.parse(f.read_text(encoding="utf-8", errors="replace"), filename=str(f))
    except (SyntaxError, ValueError, OSError):
        return None


def _top_level_names(tree: ast.Module) -> set[str]:
    """Names a module exposes at import time: defs/classes, top-level assignment
    targets, imported/re-exported names, and anything in ``__all__``."""
    names: set[str] = set()
    all_list: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for nm in _assigned_names(t):
                    names.add(nm)
                    if nm == "__all__":
                        all_list |= _string_list(getattr(node, "value", None))
    return names | all_list


def _assigned_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for el in target.elts:
            out += _assigned_names(el)
        return out
    return []


def _string_list(value) -> set[str]:
    if isinstance(value, (ast.List, ast.Tuple)):
        return {el.value for el in value.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)}
    return set()


# ── local-module resolution ──────────────────────────────────────────

def _resolve_module(repo: Path, importer: Path, node: ast.ImportFrom) -> Path | None:
    """The repo file an ``ImportFrom`` targets, or None when it isn't a local
    module (external package, or unresolvable — which we never flag)."""
    level = node.level or 0
    parts = (node.module or "").split(".") if node.module else []
    if level > 0:
        # Relative: walk up `level-1` packages from the importer's directory.
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        if not _within(repo, base):
            return None
        return _module_path(base, parts)
    # Absolute: only local if the top component is a real top-level package/module.
    if not parts:
        return None
    top = parts[0]
    if not ((repo / top).is_dir() and (repo / top / "__init__.py").exists()) \
            and not (repo / f"{top}.py").is_file():
        return None                        # external package — leave it alone
    return _module_path(repo, parts)


def _module_path(base: Path, parts: list[str]) -> Path | None:
    """Resolve dotted ``parts`` under ``base`` to a module file (``pkg/mod.py`` or
    ``pkg/__init__.py``), or None."""
    if not parts:
        pkg = base / "__init__.py"
        return pkg if pkg.is_file() else None
    d = base
    for seg in parts[:-1]:
        d = d / seg
    last = parts[-1]
    cand = d / f"{last}.py"
    if cand.is_file():
        return cand
    pkg = d / last / "__init__.py"
    if pkg.is_file():
        return pkg
    return None


def _within(repo: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(repo.resolve())
        return True
    except (ValueError, OSError):
        return False


def _name_available(repo: Path, target: Path, name: str,
                    exports: dict[Path, set[str] | None]) -> bool:
    """Is ``name`` provided by the ``target`` module? Defined/re-exported/in
    ``__all__``, OR a submodule of the target package. Unknown (unparseable
    target) → treat as available, so we never flag on our own blind spot."""
    names = exports.get(target)
    if names is None:                      # not indexed / didn't parse → don't guess
        # It may be a file we skipped or couldn't read; be safe, don't flag.
        tree = _parse(target)
        names = _top_level_names(tree) if tree is not None else None
        if names is None:
            return True
    if name in names:
        return True
    # A package can expose a submodule by name (`from pkg import sub`).
    if target.name == "__init__.py":
        pkgdir = target.parent
        if (pkgdir / f"{name}.py").is_file() or (pkgdir / name / "__init__.py").is_file():
            return True
    return False


# ── triage bridge + summary (mirrors code_verify / code_contract) ─────

def as_review_entry(findings: list[dict]) -> dict | None:
    """Broken imports → a synthetic ground-truth report for triage. None when
    clean."""
    crit = [f for f in findings if f.get("severity") == "critical"]
    if not crit:
        return None
    lines = ["GROUND TRUTH — cross-file interface check found broken imports. "
             "These reference symbols that don't exist where they're imported "
             "from and will fail at import/runtime:", ""]
    for f in crit[:_MAX_FINDINGS]:
        lines.append(f"- {f['detail']}")
    return {"role": "Interface Consistency", "id": "interface-consistency",
            "output": "\n".join(lines)}


def summarize(findings: list[dict]) -> dict:
    return {
        "checked": True,
        "broken_imports": sum(1 for f in findings if f.get("kind") == "broken_import"),
        "critical": sum(1 for f in findings if f.get("severity") == "critical"),
    }
