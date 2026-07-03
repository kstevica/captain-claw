"""Per-repo Code Map — a queryable "blackboard" of a codebase's structure.

Two layers:
  * **Skeleton** (deterministic, free): every function/method/class with its
    signature + ``file:line``, extracted by a parser. Python via stdlib ``ast``;
    JS/TS/JSX/TSX via a focused line extractor; any other language via
    universal-ctags when installed. Rebuilt incrementally — only files whose git
    blob sha changed are re-parsed.
  * **Semantics** (LLM-authored, layered on top): a one-line purpose per file,
    plus a cartographer-written architecture overview, data-model map and UI map.

Storage lives under ``<repo>/.codemap/`` (gitignored): ``map.db`` (SQLite +
FTS5 for search) and ``overview.md`` / ``models.json`` / ``ui.json``.

Agents query this via the ``codemap`` tool so they can locate symbols, models
and flows without re-reading and re-interpreting the whole tree.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_MAP_DIRNAME = ".codemap"
_MAX_FILE_BYTES = 1_500_000          # skip files bigger than this
_SKIP_DIRS = {".git", ".code", ".codemap", ".captain-claw", "saved", "node_modules",
              ".venv", "venv", "__pycache__", "dist", "build", ".next", ".pytest_cache",
              ".mypy_cache", "vendor", "coverage", ".reports", ".plans"}
_LANG_BY_SUFFIX = {
    ".py": "python", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts", ".vue": "ts", ".svelte": "ts",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby", ".php": "php",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cs": "cs", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "shell", ".sql": "sql",
}


# ── location ─────────────────────────────────────────────────────────

def map_dir(repo: Path | str) -> Path:
    d = Path(repo) / _MAP_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db(repo: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(map_dir(repo) / "map.db")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS files(
        path TEXT PRIMARY KEY, blob TEXT, lang TEXT, purpose TEXT, ts REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS symbols(
        id INTEGER PRIMARY KEY, file TEXT, name TEXT, kind TEXT, line INTEGER,
        signature TEXT, scope TEXT, summary TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name)")
    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
        name, signature, summary, purpose, file, kind UNINDEXED, line UNINDEXED)""")
    return con


# ── git blob hashing (staleness) ─────────────────────────────────────

def _blobs(repo: Path) -> dict[str, str]:
    """Map of rel-path → git blob sha for every candidate file (tracked or not).

    ``git hash-object`` gives the same sha git would store, so an unchanged file
    keeps its sha across runs — the basis for incremental reindex.
    """
    files = _candidate_files(repo)
    if not files:
        return {}
    out: dict[str, str] = {}
    # Batch through git hash-object --stdin-paths for speed.
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "--stdin-paths"],
            input="\n".join(files) + "\n", capture_output=True, text=True, timeout=120)
        shas = proc.stdout.split()
        if len(shas) == len(files):
            return dict(zip(files, shas))
    except Exception:  # noqa: BLE001 — fall back to per-file mtime-size pseudo-hash
        pass
    for f in files:
        try:
            st = (repo / f).stat()
            out[f] = f"nohash-{int(st.st_mtime)}-{st.st_size}"
        except OSError:
            pass
    return out


def _candidate_files(repo: Path) -> list[str]:
    out: list[str] = []
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.relative_to(repo).parts):
            continue
        if p.suffix.lower() not in _LANG_BY_SUFFIX:
            continue
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(p.relative_to(repo).as_posix())
    return out


# ── symbol extraction ────────────────────────────────────────────────

def extract_symbols(path: Path, lang: str) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if lang == "python":
        return _py_symbols(text)
    if lang in ("js", "ts"):
        return _js_symbols(text)
    return _ctags_symbols(path)


def _py_symbols(text: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[dict] = []

    def sig(node) -> str:
        try:
            args = [a.arg for a in node.args.posonlyargs + node.args.args]
            if node.args.vararg:
                args.append("*" + node.args.vararg.arg)
            if node.args.kwarg:
                args.append("**" + node.args.kwarg.arg)
            return f"({', '.join(args)})"
        except Exception:  # noqa: BLE001
            return "()"

    def walk(node, scope: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if scope else "function"
                out.append({"name": child.name, "kind": kind, "line": child.lineno,
                            "signature": child.name + sig(child), "scope": scope})
                walk(child, scope)  # nested defs
            elif isinstance(child, ast.ClassDef):
                out.append({"name": child.name, "kind": "class", "line": child.lineno,
                            "signature": f"class {child.name}", "scope": scope})
                walk(child, child.name)
    walk(tree, "")
    return out


# Focused JS/TS extractor — best-effort (the cartographer backfills the rest).
_JS_PATTERNS = [
    ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z0-9_$]+)")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z0-9_$]+)\s*(\([^)]*\))")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?(\([^)]*\)|[A-Za-z0-9_$]+)\s*=>")),
    ("component", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?function\s+([A-Z][A-Za-z0-9_$]*)\s*(\([^)]*\))")),
    ("method", re.compile(r"^\s{2,}(?:async\s+)?(?:static\s+)?([A-Za-z0-9_$]+)\s*(\([^)]*\))\s*\{")),
    ("type", re.compile(r"^\s*(?:export\s+)?(?:type|interface)\s+([A-Za-z0-9_$]+)")),
]
_JS_METHOD_SKIP = {"if", "for", "while", "switch", "catch", "return", "function", "constructor"}


def _js_symbols(text: str) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > 400:
            continue
        for kind, pat in _JS_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            name = m.group(1)
            if kind == "method" and name in _JS_METHOD_SKIP:
                continue
            if (name, i) in seen:
                continue
            seen.add((name, i))
            sig = (m.group(2) if m.lastindex and m.lastindex >= 2 else "").strip()
            signature = f"{name}{sig}" if sig.startswith("(") else name
            out.append({"name": name, "kind": kind, "line": i,
                        "signature": signature, "scope": ""})
            break
    return out


_CTAGS_BIN: str | None = None


def _ctags_symbols(path: Path) -> list[dict]:
    """Universal-ctags (JSON) for languages without a built-in extractor."""
    global _CTAGS_BIN
    if _CTAGS_BIN is None:
        found = shutil.which("ctags") or ""
        try:
            ver = subprocess.run([found, "--version"], capture_output=True, text=True, timeout=5).stdout
            _CTAGS_BIN = found if "Universal Ctags" in ver else ""
        except Exception:  # noqa: BLE001
            _CTAGS_BIN = ""
    if not _CTAGS_BIN:
        return []
    try:
        proc = subprocess.run(
            [_CTAGS_BIN, "--output-format=json", "--fields=+nKS", "-f", "-", str(path)],
            capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for line in proc.stdout.splitlines():
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if t.get("_type") != "tag":
            continue
        out.append({"name": t.get("name", ""), "kind": t.get("kind", "symbol"),
                    "line": int(t.get("line", 0) or 0),
                    "signature": (t.get("name", "") + (t.get("signature", "") or "")),
                    "scope": t.get("scope", "") or ""})
    return out


# ── incremental reindex ──────────────────────────────────────────────

def reindex(repo: Path | str, changed: list[str] | None = None) -> dict:
    """Rebuild the skeleton for changed files only (blob-hash gated). No LLM.

    Returns ``{indexed, removed, total, changed_files}``.
    """
    repo = Path(repo).resolve()
    con = _db(repo)
    try:
        stored = {r["path"]: r["blob"] for r in con.execute("SELECT path, blob FROM files")}
        current = _blobs(repo)
        if changed is not None:
            changed_set = {Path(c).as_posix() for c in changed}
            targets = {p: current[p] for p in current if p in changed_set}
        else:
            targets = {p: b for p, b in current.items() if stored.get(p) != b}

        # Remove files that vanished.
        gone = [p for p in stored if p not in current]
        for p in gone:
            con.execute("DELETE FROM symbols WHERE file=?", (p,))
            con.execute("DELETE FROM files WHERE path=?", (p,))
            con.execute("DELETE FROM search WHERE file=?", (p,))

        indexed = 0
        for rel, blob in targets.items():
            lang = _LANG_BY_SUFFIX.get(Path(rel).suffix.lower(), "")
            syms = extract_symbols(repo / rel, lang)
            prev = con.execute("SELECT purpose FROM files WHERE path=?", (rel,)).fetchone()
            purpose = prev["purpose"] if prev else ""      # keep prior summary until re-summarized
            con.execute("DELETE FROM symbols WHERE file=?", (rel,))
            con.execute("DELETE FROM search WHERE file=?", (rel,))
            con.execute("INSERT OR REPLACE INTO files(path, blob, lang, purpose, ts) VALUES(?,?,?,?,?)",
                        (rel, blob, lang, purpose, time.time()))
            for s in syms:
                con.execute("""INSERT INTO symbols(file, name, kind, line, signature, scope, summary)
                               VALUES(?,?,?,?,?,?,?)""",
                            (rel, s["name"], s["kind"], s["line"], s["signature"], s["scope"], ""))
                con.execute("""INSERT INTO search(name, signature, summary, purpose, file, kind, line)
                               VALUES(?,?,?,?,?,?,?)""",
                            (s["name"], s["signature"], "", purpose, rel, s["kind"], s["line"]))
            indexed += 1
        con.commit()
        total = con.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
        return {"indexed": indexed, "removed": len(gone), "total": total,
                "changed_files": list(targets.keys())}
    finally:
        con.close()


# ── piggyback summaries ──────────────────────────────────────────────

async def summarize_changed(repo: Path | str, paths: list[str], creds: dict) -> int:
    """One cheap LLM call → a one-line purpose per changed file. Best-effort."""
    repo = Path(repo).resolve()
    paths = [Path(p).as_posix() for p in paths if (repo / p).is_file()][:40]
    if not paths or not creds.get("model"):
        return 0
    con = _db(repo)
    try:
        blocks = []
        for rel in paths:
            syms = con.execute("SELECT name, kind FROM symbols WHERE file=? LIMIT 25", (rel,)).fetchall()
            names = ", ".join(f"{s['kind']} {s['name']}" for s in syms) or "(no symbols)"
            head = (repo / rel).read_text(encoding="utf-8", errors="replace")[:800]
            blocks.append(f"### {rel}\nsymbols: {names}\nhead:\n{head}")
        prompt = (
            "For each file below, write ONE concise sentence describing what it does "
            "(its role/purpose). Reply as JSON: {\"<path>\": \"<one-line purpose>\", ...} "
            "and nothing else.\n\n" + "\n\n".join(blocks))
        from captain_claw.llm import Message, create_provider
        prov = create_provider(
            provider=creds.get("provider", "anthropic"), model=creds.get("model", ""),
            api_key=creds.get("api_key") or None, base_url=creds.get("base_url") or None,
            temperature=0.1, max_tokens=1200)
        resp = await prov.complete(messages=[Message(role="user", content=prompt)],
                                   temperature=0.1, max_tokens=1200)
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        data = json.loads(content)
    except Exception as e:  # noqa: BLE001
        log.warning("codemap summarize failed", error=str(e))
        con.close()
        return 0
    n = 0
    for rel, purpose in data.items():
        rel = Path(str(rel)).as_posix()
        if not isinstance(purpose, str):
            continue
        con.execute("UPDATE files SET purpose=? WHERE path=?", (purpose.strip(), rel))
        con.execute("UPDATE search SET purpose=? WHERE file=?", (purpose.strip(), rel))
        n += 1
    con.commit()
    con.close()
    return n


# ── query API (concise; never returns raw code) ──────────────────────

def _stale(repo: Path, rel: str, con: sqlite3.Connection) -> bool:
    row = con.execute("SELECT blob FROM files WHERE path=?", (rel,)).fetchone()
    if not row:
        return True
    try:
        cur = subprocess.run(["git", "-C", str(repo), "hash-object", str(repo / rel)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return bool(cur) and cur != row["blob"]
    except Exception:  # noqa: BLE001
        return False


def search(repo: Path | str, query: str, limit: int = 25) -> list[dict]:
    repo = Path(repo).resolve()
    con = _db(repo)
    try:
        q = " OR ".join(f'"{w}"*' for w in re.findall(r"[A-Za-z0-9_]+", query)[:8]) or f'"{query}"'
        try:
            rows = con.execute(
                "SELECT name, signature, summary, purpose, file, kind, line FROM search "
                "WHERE search MATCH ? LIMIT ?", (q, limit)).fetchall()
        except sqlite3.OperationalError:
            rows = con.execute(
                "SELECT name, signature, '' summary, '' purpose, file, kind, line FROM symbols "
                "WHERE name LIKE ? LIMIT ?", (f"%{query}%", limit)).fetchall()
        return [{"name": r["name"], "kind": r["kind"], "file": r["file"], "line": r["line"],
                 "signature": r["signature"], "summary": r["summary"] or r["purpose"] or ""}
                for r in rows]
    finally:
        con.close()


def symbol(repo: Path | str, name: str) -> list[dict]:
    repo = Path(repo).resolve()
    con = _db(repo)
    try:
        rows = con.execute(
            "SELECT s.*, f.purpose AS file_purpose FROM symbols s LEFT JOIN files f ON f.path=s.file "
            "WHERE s.name=? LIMIT 20", (name,)).fetchall()
        return [{"name": r["name"], "kind": r["kind"], "file": r["file"], "line": r["line"],
                 "signature": r["signature"], "scope": r["scope"],
                 "summary": r["summary"] or "", "file_purpose": r["file_purpose"] or "",
                 "stale": _stale(repo, r["file"], con)} for r in rows]
    finally:
        con.close()


def file_map(repo: Path | str, rel: str) -> dict:
    repo = Path(repo).resolve()
    rel = Path(rel).as_posix()
    con = _db(repo)
    try:
        f = con.execute("SELECT * FROM files WHERE path=?", (rel,)).fetchone()
        syms = con.execute("SELECT name, kind, line, signature, scope, summary FROM symbols "
                           "WHERE file=? ORDER BY line", (rel,)).fetchall()
        return {"path": rel, "purpose": (f["purpose"] if f else "") or "",
                "lang": (f["lang"] if f else "") or "",
                "stale": _stale(repo, rel, con),
                "symbols": [dict(s) for s in syms]}
    finally:
        con.close()


def stats(repo: Path | str) -> dict:
    repo = Path(repo).resolve()
    con = _db(repo)
    try:
        return {"files": con.execute("SELECT COUNT(*) c FROM files").fetchone()["c"],
                "symbols": con.execute("SELECT COUNT(*) c FROM symbols").fetchone()["c"],
                "summarized": con.execute("SELECT COUNT(*) c FROM files WHERE purpose!=''").fetchone()["c"]}
    finally:
        con.close()


# ── semantic-layer files (cartographer-authored) ─────────────────────

def read_overview(repo: Path | str) -> str:
    p = Path(repo) / _MAP_DIRNAME / "overview.md"
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def write_overview(repo: Path | str, text: str) -> None:
    (map_dir(repo) / "overview.md").write_text(text or "", encoding="utf-8")


def read_json_layer(repo: Path | str, name: str) -> object:
    p = Path(repo) / _MAP_DIRNAME / f"{name}.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def write_json_layer(repo: Path | str, name: str, data: object) -> None:
    if name not in ("models", "ui"):
        raise ValueError("layer must be models or ui")
    (map_dir(repo) / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
