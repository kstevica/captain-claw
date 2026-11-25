"""Captain for Chrome connected to several Flight Decks at once
(browser-extension/chrome: lib/api.js, lib/engines.js, hands.js — run in node
with a stand-in for Chrome's APIs).

Pinned here:

* the browser keeps every deck it's connected to, one of them in use; a deck
  connected again gets its new token, not a second entry; disconnecting one
  leaves the others; a connection kept the old way (``conn``, also how Flight
  Deck seeds an agent's own browser) joins the list;
* each deck's search engine and page-read choice stay its own;
* the worker keeps a socket to every deck: each says hello with its own token,
  an act's answer goes back to the deck that asked, a Stop is told only to
  that agent's deck, and a deck disconnected loses its socket while the
  others keep theirs;
* every connected deck's pages are off-limits to every deck's agents.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXTENSION = Path(__file__).resolve().parents[2] / "browser-extension" / "chrome"

# A stand-in for the bits of Chrome the modules touch: storage (with
# onChanged), events that do nothing, and a WebSocket that records what's sent.
_CHROME = r"""
const listeners = []
const area = (name) => {
  const data = {}
  return {
    data,
    async get(keys) {
      if (keys == null) return { ...data }
      const out = {}
      for (const k of typeof keys === 'string' ? [keys] : Array.isArray(keys) ? keys : Object.keys(keys)) if (k in data) out[k] = structuredClone(data[k])
      return out
    },
    async set(obj) {
      const changes = {}
      for (const [k, v] of Object.entries(obj)) { changes[k] = { oldValue: data[k], newValue: structuredClone(v) }; data[k] = structuredClone(v) }
      for (const fn of listeners) fn(changes, name)
    },
    async remove(keys) {
      const changes = {}
      for (const k of typeof keys === 'string' ? [keys] : keys) if (k in data) { changes[k] = { oldValue: data[k] }; delete data[k] }
      if (Object.keys(changes).length) for (const fn of listeners) fn(changes, name)
    },
  }
}
const event = () => ({ addListener() {} })
globalThis.chrome = {
  runtime: { getManifest: () => ({ version: '0.10.0' }), id: 'test' },
  storage: { local: area('local'), session: area('session'), onChanged: { addListener: (fn) => listeners.push(fn) } },
  alarms: { create() {}, onAlarm: event() },
  permissions: { contains: async () => false, onAdded: event(), onRemoved: event() },
  windows: { onFocusChanged: event(), WINDOW_ID_NONE: -1 },
  tabs: { onActivated: event(), onCreated: event(), onRemoved: event(), onUpdated: event(), query: async () => [] },
  debugger: { attach() {}, onDetach: event(), onEvent: event() },
  tabGroups: { update() {}, query: async () => [] },
}
globalThis.sockets = []
globalThis.WebSocket = class {
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3
  constructor(url) { this.url = url; this.readyState = 0; this.sent = []; sockets.push(this) }
  send(text) { this.sent.push(JSON.parse(text)) }
  close() { this.readyState = 3; this.closed = true }
  // what a deck does
  open() { this.readyState = 1; this.onopen?.() }
  say(msg) { this.onmessage?.({ data: JSON.stringify(msg) }) }
}
globalThis.tick = (ms = 30) => new Promise((r) => setTimeout(r, ms))
const root = process.argv[1]
"""


def _node(script: str) -> dict:
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _CHROME + script, EXTENSION.as_uri() + "/"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_browser_keeps_every_deck_it_connects_to():
    got = _node(r"""
const api = await import(root + 'lib/api.js')
const out = {}
out.none = await api.loadDecks()
await api.saveDeck({ url: 'https://work.example.com', token: 't1', user: { name: 'Pat' } })
await api.saveDeck({ url: 'https://home.example.net', token: 't2', user: {} })
let d = await api.loadDecks()
out.two = { urls: d.decks.map((x) => x.url), use: d.use.url }
await api.saveDeck({ url: 'https://work.example.com', token: 't3', user: {} })   // connected again
d = await api.loadDecks()
out.again = { urls: d.decks.map((x) => x.url), tokens: d.decks.map((x) => x.token), use: d.use.url }
await api.useDeck('https://home.example.net')
out.used = (await api.loadConn()).url
out.next = (await api.forgetDeck('https://home.example.net')).url   // the one in use goes: the other takes over
d = await api.loadDecks()
out.left = { urls: d.decks.map((x) => x.url), use: d.use.url }
out.last = await api.forgetDeck('https://work.example.com')
out.empty = await api.loadDecks()
// kept the old way: an update from before, or Flight Deck seeding an agent's own browser
await chrome.storage.local.set({ decks: [{ url: 'https://home.example.net', token: 'old' }] })
await chrome.storage.local.set({ conn: { url: 'https://home.example.net', token: 'seeded', agent: 'researcher' } })
d = await api.loadDecks()
out.seeded = { urls: d.decks.map((x) => x.url), tokens: d.decks.map((x) => x.token), use: d.use.url, conn: 'conn' in chrome.storage.local.data }
out.onDeck = [api.onDeck('https://home.example.net/fd/x', d.decks), api.onDeck('https://home.example.net:8443/', d.decks),
              api.onDeck('https://example.org/', d.decks), api.onDeck('not a url', d.decks)]
console.log(JSON.stringify(out))
""")
    assert got["none"] == {"decks": [], "use": None}
    assert got["two"] == {"urls": ["https://work.example.com", "https://home.example.net"], "use": "https://home.example.net"}
    # Connected again: the new token, still one entry, and in use.
    assert got["again"]["urls"] == ["https://home.example.net", "https://work.example.com"]
    assert got["again"]["tokens"] == ["t2", "t3"] and got["again"]["use"] == "https://work.example.com"
    assert got["used"] == "https://home.example.net"
    assert got["next"] == "https://work.example.com"
    assert got["left"] == {"urls": ["https://work.example.com"], "use": "https://work.example.com"}
    assert got["last"] is None and got["empty"] == {"decks": [], "use": None}
    assert got["seeded"] == {"urls": ["https://home.example.net"], "tokens": ["seeded"], "use": "https://home.example.net", "conn": False}
    assert got["onDeck"] == [True, False, False, False]


def test_each_deck_keeps_its_own_search_engine_and_page_reads():
    got = _node(r"""
const eng = await import(root + 'lib/engines.js')
chrome.permissions.contains = async () => true
await eng.saveSearchPick('https://work.example.com', 'google')
await eng.saveSearchPick('https://home.example.net', 'bing')
await eng.saveSearchPick('https://home.example.net', 'nonsense')   // not an engine: off
const out = {
  work: await eng.loadSearchPick('https://work.example.com'), home: await eng.loadSearchPick('https://home.example.net'),
  served: { work: await eng.servedEngines('https://work.example.com'), home: await eng.servedEngines('https://home.example.net'),
            nobody: await eng.servedEngines('') },
}
console.log(JSON.stringify(out))
""")
    assert got["work"] == "google" and got["home"] == ""
    assert got["served"] == {"work": ["google"], "home": [], "nobody": []}


def test_the_worker_keeps_a_socket_to_every_deck():
    got = _node(r"""
const api = await import(root + 'lib/api.js')
await chrome.storage.local.set({ hands: { on: true, front: false } })
await api.saveDeck({ url: 'https://work.example.com', token: 'tw', user: {} })
await api.saveDeck({ url: 'https://home.example.net', token: 'th', user: {} })
const hands = await import(root + 'hands.js')
await hands.startHands()
await tick()
const out = {}
const byUrl = (u) => sockets.filter((s) => s.url === u && !s.closed)
const work = byUrl('wss://work.example.com/fd/clip/hands/ws')[0]
const home = byUrl('wss://home.example.net/fd/clip/hands/ws')[0]
out.sockets = sockets.map((s) => s.url)
work.open(); home.open()
out.hellos = [work.sent[0], home.sent[0]].map((h) => ({ type: h.type, token: h.token, hands: h.hands }))
work.say({ type: 'welcome', search_pick: '', fetch_mode: '' }); home.say({ type: 'welcome', search_pick: '', fetch_mode: '' })
await tick()
const status = (await chrome.storage.session.get('handsStatus')).handsStatus
out.status = Object.fromEntries(Object.entries(status).map(([k, v]) => [k, v.state]))
// both decks have a "researcher"; each answer goes back to the deck that asked
work.say({ type: 'act', id: 'ha_w1', agent: { slug: 'researcher', name: 'Researcher' }, action: 'tabs', args: {} })
home.say({ type: 'act', id: 'ha_h1', agent: { slug: 'researcher', name: 'Researcher' }, action: 'tabs', args: {} })
await tick(80)
out.results = { work: work.sent.filter((m) => m.type === 'result').map((m) => [m.id, m.ok]),
                home: home.sent.filter((m) => m.type === 'result').map((m) => [m.id, m.ok]) }
const log = (await chrome.storage.session.get('handsLog')).handsLog
out.log = log.map((e) => [e.key, e.deck])
// Stop the work deck's researcher: only that deck hears it, and its next act is refused
await chrome.storage.local.set({ handsStopped: ['https://work.example.com|researcher'] })
await tick()
out.stopped = { work: work.sent.filter((m) => m.type === 'stopped').pop()?.agents, home: home.sent.filter((m) => m.type === 'stopped').pop()?.agents }
work.say({ type: 'act', id: 'ha_w2', agent: { slug: 'researcher', name: 'Researcher' }, action: 'tabs', args: {} })
home.say({ type: 'act', id: 'ha_h2', agent: { slug: 'researcher', name: 'Researcher' }, action: 'tabs', args: {} })
await tick(80)
out.afterStop = { work: work.sent.find((m) => m.id === 'ha_w2')?.ok, home: home.sent.find((m) => m.id === 'ha_h2')?.ok }
// each deck's agent is refused the other deck's pages
home.say({ type: 'act', id: 'ha_h3', agent: { slug: 'researcher', name: 'Researcher' }, action: 'navigate', args: { url: 'https://work.example.com/' } })
await tick(80)
out.offLimits = home.sent.find((m) => m.id === 'ha_h3')?.error || ''
// a stop kept from before decks (a bare slug) holds on every deck
await chrome.storage.local.set({ handsStopped: ['researcher'] })
await tick()
out.bare = { work: work.sent.filter((m) => m.type === 'stopped').pop()?.agents, home: home.sent.filter((m) => m.type === 'stopped').pop()?.agents }
// the home deck disconnected: its socket goes, the work deck's stays
await api.forgetDeck('https://home.example.net')
await tick()
out.after = { homeClosed: !!home.closed, workClosed: !!work.closed, open: sockets.filter((s) => !s.closed).map((s) => s.url) }
const status2 = (await chrome.storage.session.get('handsStatus')).handsStatus
out.status2 = Object.keys(status2)
console.log(JSON.stringify(out))
process.exit(0)   // the worker's ping timers would keep node up
""")
    assert sorted(got["sockets"]) == ["wss://home.example.net/fd/clip/hands/ws", "wss://work.example.com/fd/clip/hands/ws"]
    assert got["hellos"] == [{"type": "hello", "token": "tw", "hands": True}, {"type": "hello", "token": "th", "hands": True}]
    assert got["status"] == {"https://work.example.com": "on", "https://home.example.net": "on"}
    assert got["results"] == {"work": [["ha_w1", True]], "home": [["ha_h1", True]]}
    assert ["https://work.example.com|researcher", "https://work.example.com"] in got["log"]
    assert ["https://home.example.net|researcher", "https://home.example.net"] in got["log"]
    assert got["stopped"] == {"work": ["researcher"], "home": []}
    assert got["afterStop"] == {"work": False, "home": True}
    assert "own pages are off-limits" in got["offLimits"]
    assert got["bare"] == {"work": ["researcher"], "home": ["researcher"]}
    assert got["after"] == {"homeClosed": True, "workClosed": False, "open": ["wss://work.example.com/fd/clip/hands/ws"]}
    assert got["status2"] == ["https://work.example.com"]
