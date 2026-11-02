"""How Crew's activity timeline shows a worker that's quiet. Run in Node.

As in test_crew_attachments_ui.py: the frontend has no JS test runner, so this lifts the pure parts
out of the TypeScript with the compiler from flight-deck/node_modules and runs them in a Node vm.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_SOURCE = _FD / "src" / "components" / "crew" / "CrewActivity.tsx"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

_RUN = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const src = process.argv[2];
const ask = JSON.parse(fs.readFileSync(0, 'utf8'));
const sf = ts.createSourceFile(src, fs.readFileSync(src, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const wanted = (st) => (st.name && st.name.text === 'toBlocks')
  || (st.declarationList && st.declarationList.declarations.some((d) => d.name.text === 'shortPaths'));
const code = sf.statements.filter(wanted).map((st) => st.getText(sf)).join('\n');
const js = ts.transpileModule(code, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
const ctx = vm.createContext({ exports: {} });
vm.runInContext(js + '\nexports.toBlocks = toBlocks;', ctx);
process.stdout.write(JSON.stringify(ctx.exports.toBlocks(ask.events, ask.live)));
"""


def _blocks(events: list[dict], live: bool = True) -> list[dict]:
    numbered = [{"n": i, "at": i, "turn": 1, **e} for i, e in enumerate(events)]
    done = subprocess.run(["node", "-e", _RUN, str(_TS), str(_SOURCE)], capture_output=True, text=True, timeout=60,
                          input=json.dumps({"events": numbered, "live": live}))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


TOOL = {"kind": "tool", "tool": "bash", "text": "ls"}
WORKING = {"kind": "working", "text": "Still working on it: one."}
NEWER = {"kind": "working", "text": "Still working on it: two."}


def test_only_the_newest_word_shows_not_one_a_minute():
    got = _blocks([TOOL, WORKING, NEWER])
    assert [b["type"] for b in got] == ["steps", "event"] and got[-1]["event"]["text"].endswith("two.")


def test_it_doesnt_split_up_what_the_worker_did():
    got = _blocks([TOOL, WORKING, TOOL])
    assert [b["type"] for b in got] == ["steps"] and len(got[0]["steps"]) == 2


def test_it_goes_once_the_worker_says_something():
    got = _blocks([WORKING, {"kind": "text", "text": "Here it is."}])
    assert [b["event"]["kind"] for b in got] == ["text"]


def test_it_goes_when_the_turn_is_over():
    assert _blocks([TOOL, WORKING], live=False)[-1]["type"] == "steps"
