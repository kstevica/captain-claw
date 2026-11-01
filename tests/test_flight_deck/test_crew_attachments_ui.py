"""What on a clipboard counts as files to attach, in Crew's two message boxes. Run in Node.

The Flight Deck frontend has no JS test runner, so, as in test_fd_api_token_ui.py, this
lifts ``pastedFiles`` out of the TypeScript source with the compiler from
flight-deck/node_modules and runs it in a Node vm. It skips when Node or the frontend
dependencies aren't installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_SOURCE = _FD / "src" / "components" / "crew" / "CrewAttachments.tsx"
_BOARD = _FD / "src" / "components" / "crew" / "CrewBoard.tsx"
_LAYOUT = _FD / "src" / "components" / "layout" / "SimpleLayout.tsx"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

# argv: typescript dir, source path. stdin: a clipboard. Prints the names of what it takes.
_RUN = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const src = process.argv[2];
const clip = JSON.parse(fs.readFileSync(0, 'utf8'));
const sf = ts.createSourceFile(src, fs.readFileSync(src, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const found = sf.statements.find((st) => st.name && st.name.text === 'pastedFiles');
const js = ts.transpileModule(found.getText(sf), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const ctx = vm.createContext({ exports: {} });
vm.runInContext(js, ctx);
const data = clip === null ? null : {
  types: clip.types,
  files: clip.files.map((name) => ({ name })),
  items: clip.items.map((item) => ({ kind: item.kind, getAsFile: () => (item.name ? { name: item.name } : null) })),
};
process.stdout.write(JSON.stringify(Array.from(ctx.exports.pastedFiles(data)).map((f) => f.name)));
"""


def _taken(clip: dict | None) -> list[str]:
    if clip is not None:
        clip = {"types": [], "files": [], "items": [], **clip}
    done = subprocess.run(["node", "-e", _RUN, str(_TS), str(_SOURCE)], input=json.dumps(clip), capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_screenshot_is_attached():
    assert _taken({"types": ["Files"], "files": ["image.png"]}) == ["image.png"]


def test_files_of_any_kind_are_attached():
    assert _taken({"types": ["Files"], "files": ["a.log", "b.pdf", "c.zip"]}) == ["a.log", "b.pdf", "c.zip"]


def test_rich_text_stays_text_though_it_comes_with_a_picture_of_itself():
    assert _taken({"types": ["text/plain", "text/html", "Files"], "files": ["image.png"]}) == []


def test_a_picture_copied_from_a_page_is_attached():
    assert _taken({"types": ["text/html", "Files"], "files": ["image.png"]}) == ["image.png"]


def test_a_file_copied_on_the_desktop_is_attached():
    assert _taken({"types": ["text/plain", "Files"], "files": ["report.pdf"]}) == ["report.pdf"]


def test_plain_text_is_left_to_the_box():
    assert _taken({"types": ["text/plain"]}) == []
    assert _taken(None) == []


def test_a_browser_that_only_lists_items_still_works():
    clip = {"types": ["Files"], "items": [{"kind": "string"}, {"kind": "file", "name": "shot.png"}, {"kind": "file"}]}
    assert _taken(clip) == ["shot.png"]


def test_both_boxes_take_paste_drop_and_the_paperclip():
    """Reading the source: the task's message box and the new task's, and only Crew's modal is 90%."""
    text = _BOARD.read_text(encoding="utf-8")
    assert text.count("onPaste={attachments.onPaste}") == 2
    assert text.count("<AttachButton onPick={attachments.onPick}") == 2
    assert text.count("<AttachmentList files={attachments.files}") == 2
    assert "onDrop: attachments.onDrop" in text and "onDrop={attachments.onDrop}" in text
    assert "crewApi.steer(task.id, text, running && !later, attachments.ids)" in text
    assert "attachments: attachments.ids" in text
    layout = _LAYOUT.read_text(encoding="utf-8")
    assert layout.count(" roomy>") == 1 and '<KioskWideDialog label="Crew" phone={isMobile} onClose={closeCrew} roomy>' in layout
    assert "wide:h-[90vh] wide:w-[90vw]" in layout
