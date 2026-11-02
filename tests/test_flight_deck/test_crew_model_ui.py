"""What Crew's New task form sends for the model a worker runs on. Run in Node.

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
_SOURCE = _FD / "src" / "components" / "crew" / "CrewModelChoice.tsx"
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
const wanted = (st) => (st.name && ['picked', 'sourceOf'].includes(st.name.text))
  || (st.declarationList && st.declarationList.declarations.some((d) => ['fits', 'NOTHING_CHOSEN'].includes(d.name.text)));
const code = sf.statements.filter(wanted).map((st) => st.getText(sf)).join('\n');
const js = ts.transpileModule(code, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
const ctx = vm.createContext({ exports: {} });
vm.runInContext(js, ctx);
const sent = ctx.exports.sourceOf({ ...ctx.exports.NOTHING_CHOSEN, ...ask.choice }, ask.choices, ask.harness);
process.stdout.write(JSON.stringify(sent === undefined ? null : sent));
"""

CHOICES = {
    "tiers": [
        {"id": "vision", "works_with": ["opencode", "pi"]},
        {"id": "balanced", "works_with": ["claude", "opencode", "pi"]},
    ],
    "agents": [{"id": "gone", "works_with": []}, {"id": "mine", "works_with": ["codex"]}],
}


def _sent(choice: dict, harness: str = "claude", choices: dict | None = CHOICES):
    done = subprocess.run(["node", "-e", _RUN, str(_TS), str(_SOURCE)], capture_output=True, text=True, timeout=60,
                          input=json.dumps({"choice": choice, "choices": choices, "harness": harness}))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_its_own_sends_nothing():
    assert _sent({"kind": "own"}) is None


def test_a_tier_or_an_agent_only_goes_if_its_cli_can_use_it():
    assert _sent({"kind": "tier", "tier": "balanced"}) == {"kind": "tier", "tier": "balanced"}
    assert _sent({"kind": "tier", "tier": "vision"}) == {"kind": "tier", "tier": "balanced"}, "the first one it can, not one it can't"
    assert _sent({"kind": "tier"}, "opencode") == {"kind": "tier", "tier": "vision"}
    assert _sent({"kind": "tier"}, "codex") is None
    assert _sent({"kind": "agent", "agent": "mine"}, "codex") == {"kind": "agent", "agent": "mine"}
    assert _sent({"kind": "agent", "agent": "gone"}) is None
    assert _sent({"kind": "tier"}, "claude", None) is None, "before the list has come"


def test_typed_in_needs_its_address_and_name():
    full = {"kind": "typed", "url": " http://box:1/v1 ", "model": " m ", "context": " 8000 ", "key": " k "}
    assert _sent(full) == {"kind": "typed", "url": "http://box:1/v1", "model": "m", "context": "8000", "key": "k"}
    assert _sent({**full, "url": " "}) is None and _sent({**full, "model": ""}) is None
    assert _sent({**full, "context": "", "key": ""})["key"] == "", "the context and the key are its own business"


def test_a_move_it_made_for_them_is_undone_and_one_they_made_isnt():
    text = _SOURCE.read_text()
    effect = text[text.index("const movedForThem"):text.index("const pick =")]
    assert "movedForThem.current = true" in effect and "onChange({ kind: 'own' })" in effect
    assert text[text.index("const pick ="):].splitlines()[1].strip() == "movedForThem.current = false"


def test_nothing_of_a_key_is_kept_in_the_browser():
    text = _SOURCE.read_text()
    assert "localStorage" not in text and "sessionStorage" not in text
    assert 'type="password"' in text and 'autoComplete="new-password"' in text
