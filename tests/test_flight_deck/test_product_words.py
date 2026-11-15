"""One word per thing (docs/product-map.md): what people read in Flight Deck
never uses a retired name.

Every .tsx/.ts file under flight-deck/src is parsed with the TypeScript
compiler (from flight-deck/node_modules); the words people read are its JSX
text, string literals and template text — not imports, not code names, not
comments, not class names or ids. A retired word there fails the test with
where it is and the word to use. It skips when Node or the frontend
dependencies aren't installed, like the other UI tests.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FD = _ROOT / "flight-deck"
_SRC = _FD / "src"
_TS = _FD / "node_modules" / "typescript"
_MAP = _ROOT / "docs" / "product-map.md"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

# Retired word (a regex, matched as written) → the word to use.
RETIRED: dict[str, str] = {
    r"\bTaskboard\b": "Tasks",
    r"\bNeeds me\b": "Needs you",
    r"\b[Cc]ron jobs?\b": "Automations (an automation)",
    r"\bCaptain Claw\b": "Captain",
}

# Attributes whose value is never read by a person.
_CODE_ATTRS = ["className", "key", "id", "htmlFor", "name", "type", "role", "href", "src", "to",
               "data-testid", "rel", "target", "autoComplete", "inputMode", "method", "accept",
               "aria-controls", "aria-labelledby", "form", "value"]

# argv: typescript dir, then files. Prints [[file, line, text], ...]: what a
# person may read in each file.
_EXTRACT = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const CODE_ATTRS = new Set(%CODE_ATTRS%);
const out = [];
for (const file of process.argv.slice(2)) {
  const kind = file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const sf = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, kind);
  const say = (node, text) => {
    if (!text || !/[A-Za-z]/.test(text)) return;
    out.push([file, sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1, text]);
  };
  const skip = (node) => {
    for (let p = node.parent; p; p = p.parent) {
      if (ts.isImportDeclaration(p) || ts.isExportDeclaration(p) || ts.isImportTypeNode?.(p)) return true;
      if (ts.isCallExpression(p) && p.expression.kind === ts.SyntaxKind.ImportKeyword) return true;
      if (ts.isJsxAttribute(p)) return CODE_ATTRS.has(p.name.getText(sf));
      if (ts.isPropertyAssignment(p) && ['className', 'key', 'id'].includes(p.name.getText(sf))) return true;
      if (ts.isLiteralTypeNode(p)) return true;
    }
    return false;
  };
  const walk = (node) => {
    if (ts.isJsxText(node)) say(node, node.getText(sf).replace(/\s+/g, ' ').trim());
    else if ((ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) && !skip(node)) say(node, node.text);
    else if ((ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) && !skip(node)) say(node, node.text);
    ts.forEachChild(node, walk);
  };
  walk(sf);
}
process.stdout.write(JSON.stringify(out));
"""


def _copy() -> list[tuple[str, int, str]]:
    files = sorted(str(p) for p in _SRC.rglob("*") if p.suffix in {".ts", ".tsx"} and ".test." not in p.name)
    script = _EXTRACT.replace("%CODE_ATTRS%", json.dumps(_CODE_ATTRS))
    done = subprocess.run(["node", "-e", script, str(_TS), *files], capture_output=True, text=True, timeout=180,
                          check=True)
    return [(f, n, t) for f, n, t in json.loads(done.stdout)]


def test_no_retired_word_in_what_people_read():
    hits = []
    for file, line, text in _copy():
        for pattern, use in RETIRED.items():
            m = re.search(pattern, text)
            if m:
                hits.append(f"{Path(file).relative_to(_ROOT)}:{line}: “{m.group(0)}” → {use}: {text[:90]!r}")
    assert not hits, "Retired words in Flight Deck copy (docs/product-map.md):\n" + "\n".join(hits)


def test_the_map_lists_every_retired_word():
    text = _MAP.read_text(encoding="utf-8")
    for use in RETIRED.values():
        assert use.split(" (")[0] in text
    for pattern in RETIRED:
        word = pattern.replace(r"\b", "").replace("[Cc]", "C").replace("s?", "")
        assert word in text, f"{word} isn't in docs/product-map.md"


def test_the_extractor_reads_copy_not_code(tmp_path):
    """The scanner sees JSX text and literals, never imports or class names."""
    sample = tmp_path / "Sample.tsx"
    sample.write_text(
        "import { openTaskboard } from './taskboard/Taskboard'\n"
        "export function A() { return <div className=\"Taskboard\" title=\"Open Tasks\">Your Tasks {`for ${x} now`}</div> }\n",
        encoding="utf-8")
    script = _EXTRACT.replace("%CODE_ATTRS%", json.dumps(_CODE_ATTRS))
    done = subprocess.run(["node", "-e", script, str(_TS), str(sample)], capture_output=True, text=True, check=True)
    texts = [t for _f, _n, t in json.loads(done.stdout)]
    assert "Your Tasks" in texts and "Open Tasks" in texts and "for " in texts
    assert not any("Taskboard" in t for t in texts)
