"""Guard against the bug class that stuck a production Vatra run.

A function-local `import X` (or `from m import X`) silently makes `X` a LOCAL name
for the WHOLE function, so any reference to `X` earlier in that function raises
`UnboundLocalError: cannot access local variable 'X'`. That is exactly what a redundant
`from captain_claw.flight_deck import story_passes` inside `_execute_vatra_inner` did to the
earlier `len(story_passes.ALL_PASSES)` in `_val_est`, crashing every story_integrity run.

ruff's pyflakes rule F823 ("local variable referenced before assignment") detects this
class precisely. Keep the whole package clean of it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _ruff() -> str | None:
    # Prefer the project venv's ruff, fall back to PATH.
    venv_ruff = _REPO / ".venv" / "bin" / "ruff"
    if venv_ruff.is_file():
        return str(venv_ruff)
    return shutil.which("ruff")


def test_no_referenced_before_assignment_f823():
    ruff = _ruff()
    if not ruff:
        pytest.skip("ruff not available")
    proc = subprocess.run(
        [ruff, "check", "--select", "F823", "--output-format", "concise", "captain_claw/"],
        cwd=_REPO, capture_output=True, text=True,
    )
    hits = [ln for ln in proc.stdout.splitlines() if ": F823 " in ln]
    assert not hits, (
        "F823 (local variable referenced before assignment) found — likely a "
        "function-local import shadowing a module-level one:\n" + "\n".join(hits)
    )
