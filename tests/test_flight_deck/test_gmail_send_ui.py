"""Email sending UI logic (Connections -> Google), run in Node.

The Flight Deck frontend has no JS test runner, so these tests lift the named
top-level declarations out of the TypeScript source with the TypeScript
compiler from flight-deck/node_modules, transpile them and run them in a Node
vm. They skip when Node or the frontend dependencies aren't installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_STORE = _FD / "src" / "stores" / "googleAuthStore.ts"
_CARD = _FD / "src" / "components" / "connections" / "GoogleConnection.tsx"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

# argv: typescript dir. stdin: {"src": path, "names": [...], "body": js}.
# Runs the named declarations + body in a vm; body sets globalThis.out.
_RUNNER = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const req = JSON.parse(fs.readFileSync(0, 'utf8'));
const sf = ts.createSourceFile(req.src, fs.readFileSync(req.src, 'utf8'),
  ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const parts = [];
for (const st of sf.statements) {
  if (st.name && req.names.includes(st.name.text)) parts.push(st.getText(sf));
}
if (parts.length !== req.names.length) {
  throw new Error('declarations not found: wanted ' + req.names.join(', '));
}
const js = ts.transpileModule(parts.join('\n') + '\n' + req.body, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const ctx = vm.createContext({ exports: {}, out: null });
vm.runInContext(js, ctx);
process.stdout.write(JSON.stringify(ctx.out));
"""


def _run(src: Path, names: list[str], body: str):
    proc = subprocess.run(
        ["node", "-e", _RUNNER, str(_TS)],
        input=json.dumps({"src": str(src), "names": names, "body": body}),
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ── _gmailSendError: a deck that predates the routes ────────────────

_STALE = "This Flight Deck doesn't offer email sending yet — it needs an update and a restart."


def _error_text(status: int, status_text: str, body: str) -> str:
    return _run(_STORE, ["HttpError", "_gmailSendError"], (
        f"out = _gmailSendError(new HttpError({status}, {json.dumps(status_text)}, "
        f"{json.dumps(body)}));"
    ))


def test_spa_catch_all_404_reads_as_needs_update():
    # Any deck serving the UI answers an unknown fd/ GET from spa_catch_all.
    body = json.dumps({"detail": "No route for /fd/google/gmail-send"})
    assert _error_text(404, "Not Found", body) == _STALE


def test_bare_fastapi_404_reads_as_needs_update():
    assert _error_text(404, "Not Found", json.dumps({"detail": "Not Found"})) == _STALE
    assert _error_text(404, "Not Found", "") == _STALE


def test_put_405_on_an_old_deck_reads_as_needs_update():
    # The catch-all is GET-only, so the PUT gets 405 there.
    body = json.dumps({"detail": "Method Not Allowed"})
    assert _error_text(405, "Method Not Allowed", body) == _STALE


def test_other_errors_keep_the_backends_words():
    body = json.dumps({"detail": "daily_limit must be a whole number from 1 to 500"})
    assert _error_text(400, "Bad Request", body) == "daily_limit must be a whole number from 1 to 500"
    assert _error_text(401, "Unauthorized", "{}").startswith("Your Flight Deck session has expired")


# ── the opt-in box while FD_GMAIL_SEND=off ──────────────────────────


def _policy(enabled: bool, deck_disabled: bool) -> str:
    return json.dumps({
        "enabled": enabled, "deck_disabled": deck_disabled,
        "allowed_recipients": [], "daily_limit": 20, "sent_last_24h": 0,
    })


def test_opt_in_locked_only_in_the_turn_on_direction():
    out = _run(_STORE, ["gmailSendOptInLocked"], (
        "out = {"
        f" optedInDeckOff: gmailSendOptInLocked({_policy(True, True)}),"
        f" optedOutDeckOff: gmailSendOptInLocked({_policy(False, True)}),"
        f" optedInDeckOn: gmailSendOptInLocked({_policy(True, False)}),"
        f" optedOutDeckOn: gmailSendOptInLocked({_policy(False, False)}),"
        " noPolicy: gmailSendOptInLocked(null) };"
    ))
    assert out == {
        "optedInDeckOff": False,  # can still opt out
        "optedOutDeckOff": True,  # turning on would do nothing
        "optedInDeckOn": False,
        "optedOutDeckOn": False,
        "noPolicy": False,
    }


_CHECKBOX_DISABLED = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const src = process.argv[2];
const sf = ts.createSourceFile(src, fs.readFileSync(src, 'utf8'),
  ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const found = [];
(function walk(n) {
  if (ts.isJsxSelfClosingElement(n) && n.tagName.getText(sf) === 'input') {
    const attr = (name) => n.attributes.properties.find(
      (p) => ts.isJsxAttribute(p) && p.name.getText(sf) === name);
    const change = attr('onChange');
    if (change && change.getText(sf).includes('setEnabled(')) {
      found.push(attr('disabled').initializer.expression.getText(sf));
    }
  }
  ts.forEachChild(n, walk);
})(sf);
process.stdout.write(JSON.stringify(found));
"""


def test_card_checkbox_lets_an_opted_in_user_uncheck_while_deck_is_off():
    proc = subprocess.run(
        ["node", "-e", _CHECKBOX_DISABLED, str(_TS), str(_CARD)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    exprs = json.loads(proc.stdout)
    assert len(exprs) == 1, exprs
    expr = exprs[0]

    def disabled(enabled: bool, deck_disabled: bool, saving: bool) -> bool:
        policy = _policy(enabled, deck_disabled)
        return _run(_STORE, ["gmailSendOptInLocked"], (
            f"const policy = {policy}; const saving = {json.dumps(saving)};"
            " const deckDisabled = policy.deck_disabled;"
            " const optInLocked = gmailSendOptInLocked(policy);"
            f" out = !!({expr});"
        ))

    assert disabled(enabled=True, deck_disabled=True, saving=False) is False
    assert disabled(enabled=False, deck_disabled=True, saving=False) is True
    assert disabled(enabled=False, deck_disabled=False, saving=False) is False
    assert disabled(enabled=True, deck_disabled=False, saving=True) is True
