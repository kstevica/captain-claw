"""Tests for Phase C — cross-file interface consistency (broken local imports).

The load-bearing property is LOW FALSE POSITIVES: external/stdlib imports, star
imports, __all__ re-exports, and package submodules must never be flagged. Only a
genuinely absent local symbol is a finding.
"""

from __future__ import annotations

from captain_claw.flight_deck import code_consistency


def _w(tmp_path, rel, body):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


# ── true positives ───────────────────────────────────────────────────

def test_broken_relative_import_flagged(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/models.py", "def foo():\n    return 1\n")
    _w(tmp_path, "pkg/api.py", "from .models import bar\n")   # bar doesn't exist
    findings = code_consistency.check(tmp_path)
    assert len(findings) == 1
    assert findings[0]["kind"] == "broken_import"
    assert "bar" in findings[0]["detail"]
    assert findings[0]["file"] == "pkg/api.py"


def test_broken_absolute_import_within_repo(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/models.py", "class User: pass\n")
    _w(tmp_path, "pkg/api.py", "from pkg.models import Order\n")   # Order missing
    findings = code_consistency.check(tmp_path)
    assert len(findings) == 1 and "Order" in findings[0]["detail"]


# ── true negatives (must NOT flag) ───────────────────────────────────

def test_valid_relative_import_clean(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/models.py", "def foo():\n    return 1\n")
    _w(tmp_path, "pkg/api.py", "from .models import foo\n")
    assert code_consistency.check(tmp_path) == []


def test_external_import_ignored(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/api.py", "from fastapi import APIRouter\nimport os\n")
    assert code_consistency.check(tmp_path) == []


def test_star_import_not_flagged(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/models.py", "x = 1\n")
    _w(tmp_path, "pkg/api.py", "from .models import *\n")
    assert code_consistency.check(tmp_path) == []


def test_reexport_satisfies_import(tmp_path):
    # api imports `foo` from pkg (__init__), which re-exports it from models.
    _w(tmp_path, "pkg/__init__.py", "from .models import foo\n")
    _w(tmp_path, "pkg/models.py", "def foo():\n    return 1\n")
    _w(tmp_path, "pkg/api.py", "from pkg import foo\n")
    assert code_consistency.check(tmp_path) == []


def test_all_declaration_counts(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/models.py", "__all__ = ['Widget']\n")   # name promised via __all__
    _w(tmp_path, "pkg/api.py", "from .models import Widget\n")
    assert code_consistency.check(tmp_path) == []


def test_submodule_import_from_package(tmp_path):
    # `from pkg import sub` where sub is a submodule file, not a name in __init__.
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/sub.py", "y = 2\n")
    _w(tmp_path, "pkg/api.py", "from pkg import sub\n")
    assert code_consistency.check(tmp_path) == []


def test_assignment_and_class_names_count(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/const.py", "TIMEOUT = 30\nclass Cfg: pass\n")
    _w(tmp_path, "pkg/api.py", "from .const import TIMEOUT, Cfg\n")
    assert code_consistency.check(tmp_path) == []


def test_unparseable_target_not_flagged(tmp_path):
    # If the target module has a syntax error we can't index it → never flag.
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/broken.py", "def foo(:\n")   # syntax error
    _w(tmp_path, "pkg/api.py", "from .broken import foo\n")
    assert code_consistency.check(tmp_path) == []


def test_unparseable_importer_skipped(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/api.py", "from .x import y  # and a syntax error below\ndef (:\n")
    # Importer doesn't parse → we can't read its imports → no findings, no crash.
    assert code_consistency.check(tmp_path) == []


def test_non_python_repo_is_noop(tmp_path):
    _w(tmp_path, "src/index.ts", "import { foo } from './bar'\n")
    assert code_consistency.check(tmp_path) == []


def test_skips_vendored_dirs(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "node_modules/dep/api.py", "from .nope import gone\n")
    _w(tmp_path, ".venv/lib/api.py", "from .nope import gone\n")
    assert code_consistency.check(tmp_path) == []


# ── triage bridge ────────────────────────────────────────────────────

def test_as_review_entry_and_summary(tmp_path):
    _w(tmp_path, "pkg/__init__.py", "")
    _w(tmp_path, "pkg/models.py", "def foo():\n    return 1\n")
    _w(tmp_path, "pkg/api.py", "from .models import bar\n")
    findings = code_consistency.check(tmp_path)
    entry = code_consistency.as_review_entry(findings)
    assert entry["id"] == "interface-consistency"
    assert "GROUND TRUTH" in entry["output"] and "bar" in entry["output"]
    s = code_consistency.summarize(findings)
    assert s["broken_imports"] == 1 and s["critical"] == 1


def test_as_review_entry_none_when_clean():
    assert code_consistency.as_review_entry([]) is None
