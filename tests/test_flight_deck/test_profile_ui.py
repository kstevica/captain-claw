"""Profile page UI logic (Profile page, Admin → Settings), run in Node.

The Flight Deck frontend has no JS test runner, so — as in
test_gmail_send_ui.py — these tests lift named top-level declarations out of
the TypeScript sources with the TypeScript compiler from
flight-deck/node_modules, transpile them and run them in a Node vm, and read
the JSX wiring off the parsed tree. They skip when Node or the frontend
dependencies aren't installed.

Pinned here:

* the preview opens on Compact (every new agent runs in eco mode), and each
  tab names who receives it; the forms carry the "eco agents get a shortened
  version" hint next to their character counters;
* Reload rebases the draft: an untouched field takes the server's newer value,
  an edited one keeps the user's text;
* the kiosk never gets the editable deck-defaults card, even on an auth-off
  deck; an admin edits them in place on the Deck-wide tab, everyone else reads;
* the Password tab is only there on a deck with accounts;
* Admin → Settings keeps the deck-defaults card mounted through a Refresh;
* a dirty profile asks before the browser tab closes or reloads;
* the Personal tab's details draft tells what changed (links normalised, empty
  rows dropped, the name apart) and when there is nothing to research from.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_SERVICE = _FD / "src" / "services" / "profile.ts"
_PAGE = _FD / "src" / "pages" / "ProfilePage.tsx"
_ADMIN = _FD / "src" / "pages" / "AdminPage.tsx"
_DECK_CARD = _FD / "src" / "components" / "profile" / "DeckProfileDefaults.tsx"
_SOURCES = _FD / "src" / "utils" / "profileSources.ts"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

# argv: typescript dir. stdin: {"decls": [{"src", "names"}], "body": js}.
# Runs the named top-level declarations + body in a vm; body sets out.
_LIFT = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const req = JSON.parse(fs.readFileSync(0, 'utf8'));
const kind = (p) => p.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
const declName = (st) => {
  if (st.name) return st.name.text;
  if (ts.isVariableStatement(st)) return st.declarationList.declarations[0].name.getText();
  return null;
};
const parts = [];
for (const d of req.decls) {
  const sf = ts.createSourceFile(d.src, fs.readFileSync(d.src, 'utf8'),
    ts.ScriptTarget.Latest, true, kind(d.src));
  const found = [];
  for (const st of sf.statements) {
    const n = declName(st);
    if (n && d.names.includes(n)) { parts.push(st.getText(sf)); found.push(n); }
  }
  if (found.length !== d.names.length) {
    throw new Error('declarations not found in ' + d.src + ': wanted ' + d.names.join(', '));
  }
}
const js = ts.transpileModule(parts.join('\n') + '\n' + req.body, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const ctx = vm.createContext({ exports: {}, out: null });
vm.runInContext(js, ctx);
process.stdout.write(JSON.stringify(ctx.out));
"""

# argv: typescript dir, source path. stdin: a function body over (ts, sf, src)
# whose return value is printed as JSON.
_QUERY = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const src = process.argv[2];
const sf = ts.createSourceFile(src, fs.readFileSync(src, 'utf8'),
  ts.ScriptTarget.Latest, true, src.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
const code = fs.readFileSync(0, 'utf8');
process.stdout.write(JSON.stringify(new Function('ts', 'sf', 'src', code)(ts, sf, src)));
"""

# JS helpers prepended to every query.
_WALK = r"""
const all = [];
(function walk(n) { all.push(n); ts.forEachChild(n, walk); })(sf);
const tag = (n) => (ts.isJsxSelfClosingElement(n) ? n : ts.isJsxElement(n) ? n.openingElement : null);
const tagName = (n) => { const t = tag(n); return t ? t.tagName.getText(sf) : null; };
const attr = (n, name) => {
  const t = tag(n);
  const a = t && t.attributes.properties.find((p) => ts.isJsxAttribute(p) && p.name.getText(sf) === name);
  if (!a || !a.initializer) return null;
  return ts.isStringLiteral(a.initializer) ? a.initializer.text : a.initializer.expression.getText(sf);
};
const jsxText = (n) => {
  const bits = [];
  (function w(c) { if (ts.isJsxText(c)) bits.push(c.getText(sf)); ts.forEachChild(c, w); })(n);
  return bits.join(' ').replace(/\s+/g, ' ').trim();
};
"""


def _lift(decls: list[tuple[Path, list[str]]], body: str):
    req = {"decls": [{"src": str(src), "names": names} for src, names in decls], "body": body}
    proc = subprocess.run(
        ["node", "-e", _LIFT, str(_TS)],
        input=json.dumps(req), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _query(src: Path, code: str):
    proc = subprocess.run(
        ["node", "-e", _QUERY, str(_TS), str(src)],
        input=_WALK + code, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ── preview: Compact first, and who gets which ─────────────────────


def _preview_initial_state_expr() -> str:
    """The argument of the page's `useState` for previewMode."""
    exprs = _query(_PAGE, r"""
      return all.filter((n) => ts.isVariableDeclaration(n) && ts.isArrayBindingPattern(n.name)
          && n.name.elements.some((e) => e.getText(sf) === 'setPreviewMode'))
        .map((n) => n.initializer.arguments[0].getText(sf));
    """)
    assert len(exprs) == 1, exprs
    return exprs[0]


def test_preview_opens_on_compact():
    expr = _preview_initial_state_expr()
    out = _lift([(_SERVICE, ["DEFAULT_PREVIEW_MODE"]), (_PAGE, ["PREVIEW_MODES"])],
                f"out = {{ initial: ({expr}), tabs: PREVIEW_MODES }};")
    assert out["initial"] == "compact"
    # The default tab comes first in the toggle.
    assert out["tabs"] == ["compact", "full"]


def test_preview_copy_names_who_receives_each_form():
    audience = _lift([(_SERVICE, ["PREVIEW_AUDIENCE"])], "out = PREVIEW_AUDIENCE;")
    assert audience == {
        "full": "agents with eco mode off",
        "compact": "agents in eco mode — the default for new agents — nano agents, "
                   "and workers in multi-agent runs",
    }
    # The preview card's description is driven by the selected tab.
    texts = _query(_PAGE, r"""
      return all.filter((n) => ts.isJsxExpression(n) && n.expression
          && n.expression.getText(sf) === 'PREVIEW_AUDIENCE[previewMode]')
        .map((n) => jsxText(n.parent));
    """)
    assert texts == ["Added to the system prompt of ."], texts


def _hint_beside_counters(src: Path) -> list[str]:
    """Tag names of the siblings of each `{ECO_SHORTENED_HINT}` paragraph."""
    return _query(src, r"""
      return all.filter((n) => ts.isJsxExpression(n) && n.expression
          && n.expression.getText(sf) === 'ECO_SHORTENED_HINT')
        .map((n) => n.parent.parent)  // <p> → its parent element / fragment
        .map((box) => box.children.map(tagName).filter(Boolean));
    """)


def test_eco_hint_sits_beside_the_counters():
    hint = _lift([(_SERVICE, ["ECO_SHORTENED_HINT"])], "out = ECO_SHORTENED_HINT;")
    assert "eco mode" in hint and "default for new agents" in hint and "shortened" in hint
    for src in (_PAGE, _DECK_CARD):
        boxes = _hint_beside_counters(src)
        assert len(boxes) == 1, (src.name, boxes)
        assert "CappedTextarea" in boxes[0], (src.name, boxes)


# ── Reload rebases the draft ────────────────────────────────────────


def _rebase(draft: dict, base: dict, nxt: dict) -> dict:
    return _lift([(_SERVICE, ["rebaseDraft"])],
                 f"out = rebaseDraft({json.dumps(draft)}, {json.dumps(base)}, {json.dumps(nxt)});")


def test_rebase_takes_newer_server_values_for_untouched_fields():
    base = {"about_me": "old me", "company": "old co", "instructions": "old prefs"}
    # Saved elsewhere since this page loaded: company and instructions.
    nxt = {"about_me": "old me", "company": "NEW co", "instructions": "NEW prefs"}
    # The user only edited about_me here.
    draft = {"about_me": "my edit", "company": "old co", "instructions": "old prefs"}
    assert _rebase(draft, base, nxt) == {
        "about_me": "my edit", "company": "NEW co", "instructions": "NEW prefs",
    }


def test_rebase_keeps_an_edit_even_when_the_server_also_changed_it():
    base = {"about_me": "a", "company": "c", "instructions": "i"}
    nxt = {"about_me": "a", "company": "server", "instructions": "i"}
    draft = {"about_me": "a", "company": "mine", "instructions": "i"}
    assert _rebase(draft, base, nxt)["company"] == "mine"
    # An untouched draft simply becomes the server's profile.
    assert _rebase(base, base, nxt) == nxt


def test_reload_button_and_deck_save_both_rebase():
    wiring = _query(_PAGE, r"""
      const load = all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === 'load');
      const reload = all.filter((n) => tagName(n) === 'button' && attr(n, 'title') === 'Reload');
      const deck = all.filter((n) => tagName(n) === 'DeckProfileDefaults');
      return {
        loadBody: load ? load.initializer.getText(sf) : '',
        reload: reload.map((n) => attr(n, 'onClick')),
        deck: deck.map((n) => attr(n, 'onSaved')),
      };
    """)
    assert "rebaseDraft(" in wiring["loadBody"]
    assert wiring["reload"] == ["() => load(true)"]
    assert wiring["deck"] == ["() => load(true)"]


# ── kiosk: deck defaults read-only, even with auth off ──────────────


def test_deck_defaults_editable_only_off_kiosk_for_the_admin_or_the_auth_off_user():
    out = _lift([(_SERVICE, ["deckDefaultsEditable"])], (
        "out = {"
        " offDesk: deckDefaultsEditable(false, false),"
        " offKiosk: deckDefaultsEditable(false, true),"
        " onDesk: deckDefaultsEditable(true, false),"
        " onDeskAdmin: deckDefaultsEditable(true, false, true),"
        " onKiosk: deckDefaultsEditable(true, true),"
        " onKioskAdmin: deckDefaultsEditable(true, true, true),"
        " unknown: deckDefaultsEditable(null, false),"
        " unknownAdmin: deckDefaultsEditable(null, false, true) };"
    ))
    assert out == {"offDesk": True, "offKiosk": False, "onDesk": False, "onDeskAdmin": True,
                   "onKiosk": False, "onKioskAdmin": False, "unknown": False, "unknownAdmin": False}


def test_page_picks_the_editable_card_through_the_kiosk_check():
    conds = _query(_PAGE, r"""
      return all.filter((n) => ts.isConditionalExpression(n)
          && all.some((m) => tagName(m) === 'DeckProfileDefaults'
              && m.pos >= n.whenTrue.pos && m.end <= n.whenTrue.end))
        .map((n) => n.condition.getText(sf));
    """)
    assert len(conds) == 1, conds

    def editable(auth_enabled, kiosk: bool, admin: bool = False) -> bool:
        return _lift([(_SERVICE, ["deckDefaultsEditable"])], (
            f"const authEnabled = {json.dumps(auth_enabled)}; const kiosk = {json.dumps(kiosk)};"
            f" const isAdmin = {json.dumps(admin)};"
            f" out = !!({conds[0]});"
        ))

    assert editable(False, kiosk=True) is False  # a kiosk visitor never edits them
    assert editable(False, kiosk=False) is True
    assert editable(True, kiosk=False) is False
    assert editable(True, kiosk=False, admin=True) is True   # the admin, in place
    assert editable(True, kiosk=True, admin=True) is False   # …but not from the kiosk


# ── Admin → Settings: the card survives a Refresh ───────────────────


def test_admin_deck_card_is_not_behind_the_loading_guard():
    # Every `&&` guard on the way down to <DeckProfileDefaults />.
    guards = _query(_ADMIN, r"""
      const el = all.filter((n) => tagName(n) === 'DeckProfileDefaults');
      if (el.length !== 1) return { count: el.length };
      const out = [];
      for (let p = el[0]; p; p = p.parent) {
        const up = p.parent;
        if (up && ts.isBinaryExpression(up) && up.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken
            && up.right === p) out.push(up.left.getText(sf));
      }
      return { count: 1, guards: out };
    """)
    assert guards["count"] == 1, guards
    cond = " && ".join(f"({g})" for g in guards["guards"]) or "true"

    def mounted(loading: bool, tab: str) -> bool:
        return _lift([], f"const loading = {json.dumps(loading)}; const tab = {json.dumps(tab)};"
                         f" out = !!({cond});")

    assert mounted(loading=True, tab="settings") is True  # the Refresh in flight
    assert mounted(loading=False, tab="settings") is True
    assert mounted(loading=False, tab="users") is False


# ── closing the browser tab with unsaved text ───────────────────────

_FAKE_WINDOW = r"""
const added = [], removed = [];
const target = {
  addEventListener: (t, f) => added.push([t, f]),
  removeEventListener: (t, f) => removed.push([t, f]),
};
"""


def test_guard_unload_only_while_dirty():
    out = _lift([(_SERVICE, ["guardUnload"])], _FAKE_WINDOW + r"""
      const cleanup = guardUnload(target, false);
      cleanup();
      out = { added: added.length, removed: removed.length };
    """)
    assert out == {"added": 0, "removed": 0}


def test_guard_unload_asks_and_cleans_up():
    out = _lift([(_SERVICE, ["guardUnload"])], _FAKE_WINDOW + r"""
      const cleanup = guardUnload(target, true);
      const ev = { prevented: false, returnValue: undefined, preventDefault() { this.prevented = true; } };
      added[0][1](ev);
      cleanup();
      out = {
        types: added.map((a) => a[0]),
        prevented: ev.prevented,
        returnValue: ev.returnValue,
        sameHandlerRemoved: removed.length === 1 && removed[0][0] === 'beforeunload'
          && removed[0][1] === added[0][1],
      };
    """)
    assert out == {"types": ["beforeunload"], "prevented": True, "returnValue": "",
                   "sameHandlerRemoved": True}


def test_page_guards_unload_on_its_dirty_state():
    calls = _query(_PAGE, r"""
      return all.filter((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'useEffect')
        .map((n) => n.arguments.map((a) => a.getText(sf)))
        .filter((args) => args[0].includes('guardUnload('));
    """)
    assert calls == [["() => guardUnload(window, dirty)", "[dirty]"]]


# ── the tabs ─────────────────────────────────────────────────────────


def _guards_down_to(src: Path, tag: str) -> list[str]:
    """Every `&&` guard on the way down to the one <tag /> in *src*."""
    found = _query(src, r"""
      const el = all.filter((n) => tagName(n) === '%s');
      if (el.length !== 1) return { count: el.length };
      const out = [];
      for (let p = el[0]; p; p = p.parent) {
        const up = p.parent;
        if (up && ts.isBinaryExpression(up) && up.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken
            && up.right === p) out.push(up.left.getText(sf));
      }
      return { count: 1, guards: out };
    """ % tag)
    assert found["count"] == 1, found
    return found["guards"]


def test_password_tab_only_on_a_deck_with_accounts():
    cond = " && ".join(f"({g})" for g in _guards_down_to(_PAGE, "ChangePasswordCard"))

    def shown(auth_enabled, tab: str) -> bool:
        return _lift([], f"const authEnabled = {json.dumps(auth_enabled)}; const shownTab = {json.dumps(tab)};"
                         f" out = !!({cond});")

    assert shown(True, "password") is True
    assert shown(False, "password") is False   # auth off: no account, no tab
    assert shown(None, "password") is False
    assert shown(True, "personal") is False


def test_tabs_are_personal_deck_and_password():
    ids = _query(_PAGE, r"""
      const tabs = all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === 'tabs');
      const out = [];
      (function w(c) {
        if (ts.isPropertyAssignment(c) && c.name.getText(sf) === 'id') out.push(c.initializer.getText(sf));
        ts.forEachChild(c, w);
      })(tabs.initializer);
      return out;
    """)
    assert ids == ["'personal'", "'deck'", "'password' as const"], ids
    # The deck-wide card and the password card each sit on their own tab.
    panels = _query(_PAGE, r"""
      return all.filter((n) => attr(n, 'role') === 'tabpanel').map((n) => attr(n, 'id'));
    """)
    assert panels == ["profile-panel-personal", "profile-panel-deck", "profile-panel-password"]


# ── the Personal tab's details draft ────────────────────────────────


def _details(body: str):
    return _lift([(_SOURCES, ["keptLinks", "sameLinks", "detailsPatch", "nameChanged", "detailsDirty", "sourcesEmpty", "detailsOf"])], body)


_SAVED = {
    "name": "Bob", "email": "bob@x.co", "can_rename": True,
    "links": [{"url": "https://github.com/bob", "label": "GitHub"}],
    "company_name": "Bridges Ltd.", "company_links": [], "documents": [],
    "caps": {}, "kinds": [],
}


def test_details_patch_sends_only_what_changed_and_drops_empty_rows():
    out = _details(f"""
      const saved = {json.dumps(_SAVED)};
      const same = detailsOf(saved);
      const edited = {{ ...same, links: [...same.links, {{ url: '', label: '' }}], company_name: ' Bridges Ltd. ' }};
      const changed = {{ ...same, links: [{{ url: ' https://x.com/bob ', label: 'X' }}], company_links: [{{ url: 'bridges.example', label: '' }}] }};
      out = {{
        none: detailsPatch(same, saved),
        edited: detailsPatch(edited, saved),
        editedDirty: detailsDirty(edited, saved),
        changed: detailsPatch(changed, saved),
        changedDirty: detailsDirty(changed, saved),
      }};
    """)
    assert out["none"] == {}
    # An empty row and spaces around the same name are not a change.
    assert out["edited"] == {} and out["editedDirty"] is False
    assert out["changed"] == {
        "links": [{"url": "https://x.com/bob", "label": "X"}],
        "company_links": [{"url": "bridges.example", "label": ""}],
    }
    assert out["changedDirty"] is True


def test_the_name_is_apart_and_only_where_the_account_can_be_renamed():
    out = _details(f"""
      const saved = {json.dumps(_SAVED)};
      const local = {{ ...saved, can_rename: false }};
      const renamed = {{ ...detailsOf(saved), name: 'Robert' }};
      const blank = {{ ...detailsOf(saved), name: '  ' }};
      out = {{
        renamed: nameChanged(renamed, saved), dirty: detailsDirty(renamed, saved),
        patch: detailsPatch(renamed, saved),
        local: nameChanged(renamed, local), blank: nameChanged(blank, saved),
      }};
    """)
    assert out == {"renamed": True, "dirty": True, "patch": {}, "local": False, "blank": False}


def test_sources_empty_means_nothing_to_research_from():
    out = _details(f"""
      const saved = {json.dumps({**_SAVED, "links": [], "company_name": ""})};
      const d = detailsOf(saved);
      out = {{
        empty: sourcesEmpty(d, saved),
        blankRow: sourcesEmpty({{ ...d, links: [{{ url: ' ', label: 'x' }}] }}, saved),
        link: sourcesEmpty({{ ...d, links: [{{ url: 'github.com/bob', label: '' }}] }}, saved),
        company: sourcesEmpty({{ ...d, company_name: 'Bridges' }}, saved),
        doc: sourcesEmpty(d, {{ ...saved, documents: [{{ id: 'a', name: 'cv.pdf', pages: 1, chars: 10, added: 0 }}] }}),
        unloaded: sourcesEmpty(d, null),
      }};
    """)
    assert out == {"empty": True, "blankRow": True, "link": False, "company": False, "doc": False, "unloaded": True}
