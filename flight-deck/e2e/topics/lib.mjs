// Harness for the chat store's topic dividers: a fake agent socket and a fake
// Flight Deck chat log (/fd/chat/...), with the log's first read held back
// until the test releases it — so a replay can arrive before or after it.
import { fileURLToPath } from 'node:url'
import { advance } from './clock.mjs'
export { advance }

export const SRC = process.env.SRC || fileURLToPath(new URL('../../src', import.meta.url))

let fails = 0
let passes = 0
export function check(label, cond, extra) {
  if (cond) { passes++; console.log('  ok  ', label) }
  else { fails++; console.log('  FAIL', label, extra !== undefined ? JSON.stringify(extra) : '') }
}
export function done() {
  console.log(fails ? `FAILED ${fails} (passed ${passes})` : `PASSED ${passes}`)
  process.exit(fails ? 1 : 0)
}

/** Let pending promises (fetch → json → setState) run. */
export async function settle() {
  for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r))
}

/** The server's chat log: rows as the API returns them. */
export function makeLog(rows = []) {
  let id = 0
  const log = rows.map((r) => ({ id: ++id, created_at: '2026-10-22T08:00:00Z', metadata: '{}', content: '', ...r }))
  return {
    rows: log,
    add(msgs) { return msgs.map((m) => { const row = { id: ++id, created_at: '2026-10-22T09:00:00Z', ...m }; log.push(row); return row.id }) },
    view() {
      return log.map((r) => {
        const meta = JSON.parse(r.metadata || '{}')
        return meta.topic ? `[${meta.topic.action}:${meta.topic.agentSessionId}]` : `${r.role}:${r.content}`
      })
    },
  }
}

export const divider = (action, sid) => ({ role: 'system', content: 'New topic', metadata: JSON.stringify({ topic: { action, agentSessionId: sid, source: 'agent' } }) })
export const turn = (role, content) => ({ role, content, metadata: '{}' })

export async function setup({ auth = true, log = makeLog(), holdHistory = true } = {}) {
  const { useAuthStore } = await import(SRC + '/stores/authStore.ts')
  useAuthStore.setState({ authEnabled: auth, isAuthenticated: auth })
  let releaseHistory
  const historyGate = holdHistory ? new Promise((r) => { releaseHistory = r }) : Promise.resolve()
  globalThis.fetch = async (url, init = {}) => {
    const u = String(url)
    const method = (init.method || 'GET').toUpperCase()
    const json = (body) => ({ ok: true, status: 200, json: async () => body })
    if (u === '/fd/chat/sessions' && method === 'POST') return json({ id: 'x' })
    if (/\/fd\/chat\/sessions\/[^/]+\/messages/.test(u) && method === 'GET') {
      await historyGate
      const q = new URL('http://x' + u).searchParams
      const limit = Number(q.get('limit') || 500)
      let rows = log.rows
      if (q.get('since_id')) rows = rows.filter((r) => r.id >= Number(q.get('since_id'))).slice(0, limit)
      else if (q.get('before')) rows = rows.filter((r) => r.id < Number(q.get('before'))).slice(-limit)
      else rows = rows.slice(-limit)
      return json(rows.map((r) => ({ ...r })))
    }
    if (/\/fd\/chat\/sessions\/[^/]+\/topics$/.test(u) && method === 'GET') {
      // The whole log's dividers, oldest first (db.list_chat_topics), not just a window's.
      const out = []
      for (const r of log.rows) {
        const t = JSON.parse(r.metadata || '{}').topic
        if (t) out.push({ row_id: r.id, agent_session_id: t.agentSessionId, action: t.action, title: t.title || '', started_at: r.created_at, first_user_message: '', message_count: 0 })
      }
      return json(out)
    }
    if (/\/fd\/chat\/sessions\/[^/]+\/messages$/.test(u) && method === 'POST') {
      // As db.add_chat_messages: a healed turn (40+ chars) the log has word for
      // word is not written again; its row id comes back and its index in `known`.
      const body = JSON.parse(init.body)
      const ids = []
      const known = []
      body.messages.forEach((m, i) => {
        const healed = JSON.parse(m.metadata || '{}').healed === true
        const had = healed && (m.role === 'user' || m.role === 'assistant') && m.content.length >= 40
          ? [...log.rows].reverse().find((r) => r.role === m.role && r.content === m.content) : undefined
        if (had) { ids.push(had.id); known.push(i) } else ids.push(log.add([m])[0])
      })
      return json({ ok: true, ids, ...(known.length ? { known } : {}) })
    }
    return { ok: false, status: 404, json: async () => ({}) }
  }
  const { AgentChatWS } = await import(SRC + '/services/agentChat.ts')
  AgentChatWS.prototype.connect = function () {}
  AgentChatWS.prototype.send = function () { return undefined }
  AgentChatWS.prototype.sendJSON = function () {}
  AgentChatWS.prototype.isOwnMessage = function () { return true }
  const mod = await import(SRC + '/stores/chatStore.ts')
  const st = () => mod.useChatStore.getState()
  st().openChat('agent1', 'Captain', 'localhost', 1234, '')
  const key = [...st().sessions.keys()][0]
  const ses = () => st().sessions.get(key)
  const ws = ses().ws
  const emit = (ev, data = {}) => ws.emit(ev, data)
  ws._connected = true
  const ev = {
    connected: () => emit('_connected'),
    disconnected: () => emit('_disconnected'),
    welcome: (sid) => emit('welcome', { session: { id: sid, name: 'default' }, models: [], personalities: [] }),
    replay: (turns) => emit('replay_batch', { messages: turns.map(([role, content]) => ({ type: 'chat_message', role, content, replay: true })) }),
    replayDone: () => emit('replay_done'),
    sessionInfo: (sid, count = 2) => emit('session_info', { id: sid, name: 'default', message_count: count }),
    reply: (text) => emit('chat_message', { role: 'assistant', content: text }),
    user: (text) => emit('chat_message', { role: 'user', content: text, via: 'telegram' }),
    ready: () => emit('status', { status: 'ready' }),
    commandResult: (command, content, sid) => emit('command_result', { command, content, session: { id: sid, name: 'default' } }),
  }
  /** The transcript in short: [action:session] for a divider, role:text otherwise. */
  const view = () => ses().messages.map((m) => (m.topic ? `[${m.topic.action}:${m.topic.agentSessionId}]` : `${m.role}:${m.content}`))
  const flush = async () => { advance(600); await settle() }
  return { st, key, ses, ws, emit, ev, view, log, flush, release: async () => { releaseHistory?.(); await settle() } }
}
