"""A page left open outlives its sign-in token, which lasts 15 minutes: its requests
must renew it and go on, not show "Token expired". Run in Node.

The Flight Deck frontend has no JS test runner, so, as in test_profile_ui.py, this
lifts ``api`` out of the TypeScript source with the compiler from
flight-deck/node_modules and runs it in a Node vm with a fake ``fetch``. It skips when
Node or the frontend dependencies aren't installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_SOURCE = _FD / "src" / "services" / "fileLibraryApi.ts"
_BOARD = _FD / "src" / "components" / "crew" / "CrewBoard.tsx"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

# argv: typescript dir, source path. stdin: the scenario. Prints what the fake saw.
_RUN = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const src = process.argv[2];
const scene = JSON.parse(fs.readFileSync(0, 'utf8'));
const sf = ts.createSourceFile(src, fs.readFileSync(src, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const found = sf.statements.find((st) => st.name && st.name.text === 'api');
const js = ts.transpileModule(found.getText(sf), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const state = { token: 'old', authEnabled: scene.authEnabled };
const seen = { sent: [], refreshes: 0 };
const ctx = vm.createContext({
  exports: {},
  useAuthStore: { getState: () => state },
  refreshAccessToken: async () => {
    seen.refreshes += 1;
    if (scene.refreshWorks) state.token = 'new';
    return scene.refreshWorks;
  },
  fetch: async (path, init) => {
    seen.sent.push({ path, auth: init.headers.Authorization || '', body: init.body || '' });
    const [status, body] = scene.answers[Math.min(seen.sent.length - 1, scene.answers.length - 1)];
    return { ok: status < 300, status, statusText: 'x', json: async () => body };
  },
});
vm.runInContext(js, ctx);
ctx.exports.api('/fd/crew/tasks', scene.init || undefined)
  .then((value) => ({ value }), (error) => ({ error: error.message }))
  .then((result) => process.stdout.write(JSON.stringify({ ...seen, ...result })));
"""


def _run(**scene) -> dict:
    scene = {"authEnabled": True, "refreshWorks": True, **scene}
    done = subprocess.run(["node", "-e", _RUN, str(_TS), str(_SOURCE)], input=json.dumps(scene), capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


EXPIRED = [401, {"detail": "Token expired"}]


def test_an_expired_token_is_renewed_and_the_request_goes_through():
    got = _run(answers=[EXPIRED, [200, {"tasks": []}]])
    assert got["value"] == {"tasks": []} and "error" not in got
    assert [s["auth"] for s in got["sent"]] == ["Bearer old", "Bearer new"], "the second try carries the new token"
    assert got["refreshes"] == 1


def test_a_request_with_a_body_is_sent_again_whole():
    got = _run(answers=[EXPIRED, [200, {"ok": True}]], init={"method": "POST", "body": '{"text": "go on"}'})
    assert [s["body"] for s in got["sent"]] == ['{"text": "go on"}'] * 2


def test_when_the_session_is_over_the_refusal_is_said():
    got = _run(answers=[EXPIRED], refreshWorks=False)
    assert got["error"] == "Token expired" and len(got["sent"]) == 1


def test_it_tries_once_and_doesnt_loop():
    got = _run(answers=[EXPIRED, EXPIRED, [200, {}]])
    assert got["error"] == "Token expired" and len(got["sent"]) == 2 and got["refreshes"] == 1


def test_a_deck_with_sign_in_off_has_nothing_to_renew():
    got = _run(answers=[EXPIRED], authEnabled=False)
    assert got["error"] == "Token expired" and got["refreshes"] == 0


def test_other_refusals_and_successes_are_left_alone():
    got = _run(answers=[[200, {"a": 1}]])
    assert got["value"] == {"a": 1} and len(got["sent"]) == 1 and got["refreshes"] == 0
    for status in (403, 404, 500):
        got = _run(answers=[[status, {"detail": "no"}]])
        assert got["error"] == "no" and len(got["sent"]) == 1 and got["refreshes"] == 0, status


def test_a_failed_look_at_a_task_clears_once_a_look_works():
    """Reading the source: a poll's error is its own, and an action's error isn't touched by a poll."""
    text = _BOARD.read_text(encoding="utf-8")
    assert "await load()\n        if (!stopped) setUnseen('')" in text
    assert "if (!stopped) setUnseen(errorOf(e))" in text
    assert "<ErrorLine text={error || unseen} />" in text
    assert "if (!stopped) setError(errorOf(e))" not in text
