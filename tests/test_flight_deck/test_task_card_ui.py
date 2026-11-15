"""The chat's task card: a ``bat`` tool call that started a run gets a live card
(TaskCard.tsx). ``batRunOf`` picks the run out of the tool's answer, run in
Node with the TypeScript compiler like the other UI tests; skips without them."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_CARD = _FD / "src" / "components" / "agents" / "TaskCard.tsx"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not _TS.is_dir(),
                                reason="needs node and flight-deck/node_modules (npm install)")

_RUN = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const sf = ts.createSourceFile('c.tsx', fs.readFileSync(process.argv[2], 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const fn = sf.statements.find((s) => s.name && s.name.text === 'batRunOf');
const js = ts.transpileModule(fn.getText(sf).replace(/^export /, ''), { compilerOptions: { target: 'ES2020' } }).outputText;
const batRunOf = new Function(js + '; return batRunOf')();
const cases = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(cases.map(([t, o]) => batRunOf(t, o))));
"""


def _run(cases):
    out = subprocess.run(["node", "-e", _RUN, str(_TS), str(_CARD), json.dumps(cases)],
                         capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout)


def test_a_started_bat_run_gets_a_card():
    started = ("Started Bat run **Tidy the shared drive** (run 4f2a9c1e). It will keep working until an "
               "independent judge says it's done; I'll report the result back here when it finishes.")
    assert _run([
        ["bat", started],
        ["bat", "Bat run 4f2a9c1e: **running** — 2/5 steps"],   # a status check: no second card
        ["bat", "Not started — at limit."],
        ["shell", started],
        ["bat", ""],
    ]) == ["4f2a9c1e", None, None, None, None]
