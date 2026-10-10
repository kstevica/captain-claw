// HUD chat session: one live connection to the selected agent (lane A, the
// same conversation as the dashboard), kept open while the wearer browses
// Files / Data so replies are not missed.
//
// Deliberately NOT the dashboard's chatStore: no peer_agents push (it carries
// other agents' tokens), no queue / plan state, no cross-store dependencies.
// The socket goes through Flight Deck's owner-checked proxy with an empty
// agent token (FD resolves it server-side) or, for shared agents, the member
// route by agent_ref.
//
// Protocol notes (captain_claw/web/ws_handler.py, chat_handler.py):
//   - on connect the agent sends welcome → replay_batch (whole session) →
//     replay_done; we REPLACE the transcript with the replay (the dashboard's
//     merge duplicates rows after a reconnect);
//   - the user echo of a live turn carries the raw content (rules block
//     included) and no client_msg_id, so live user echoes are ignored — we
//     show our own local echo instead;
//   - a non-replay assistant chat_message is the end of the turn.

import { create } from 'zustand'
import { AgentChatWS } from '../../services/agentChat'
import { refreshAccessToken, useAuthStore } from '../../stores/authStore'
import { getHudConfig, listMarkdownFiles, type HudAgent, type HudFile } from '../api'
import {
  cleanAssistantText, cleanUserText, isIdleStatus, matchFileRef, newId, prettyStatus,
  speakableText, speechChunks, toMs,
} from './text'

export interface HudMsg {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  /** Epoch ms. */
  ts: number
  /** From the agent's history replay (not live). */
  replay?: boolean
}

export interface NextStep { label: string; action: string }

export interface Approval { id: string; message: string; category: string }

export interface ChatState {
  /** Agent the session is attached to (null = none). */
  agentId: string | null
  connected: boolean
  /** A turn is running on the agent. */
  busy: boolean
  /** Latest status text while busy ("Thinking…", "Using web_search…"). */
  status: string
  /** Latest narration line from the agent (transient progress). */
  narration: string
  messages: HudMsg[]
  nextSteps: NextStep[]
  approval: Approval | null
  /** Last error to show (cleared on the next send). */
  error: string | null
  /** Assistant replies that arrived while the Chat tab was not shown. */
  unread: number
  /** Read replies aloud (speechSynthesis). */
  tts: boolean
  /** Flight Deck closed the connection for good (shared agent: access
   *  revoked, agent gone…). No automatic reconnect. */
  closed: boolean
  /** The composer's unsent text (survives tab switches). */
  draft: string
}

/** Transcript rows kept in memory / on screen (the glasses CPU is slow). */
export const MAX_MSGS = 40
const TTS_KEY = 'hud.tts.v1'
/** Agents wait up to 60 s for an approval, then auto-approve. */
const APPROVAL_TTL_MS = 62_000
/** Longest we hold a first message waiting for the rules block. */
const RULES_WAIT_MS = 4_000

function readTts(): boolean {
  try { return localStorage.getItem(TTS_KEY) === '1' } catch { return false }
}

function sessionDefaults() {
  return {
    agentId: null as string | null,
    connected: false,
    busy: false,
    status: '',
    narration: '',
    messages: [] as HudMsg[],
    nextSteps: [] as NextStep[],
    approval: null as Approval | null,
    error: null as string | null,
    unread: 0,
    closed: false,
    draft: '',
  }
}

export const useHudChat = create<ChatState>(() => ({
  ...sessionDefaults(),
  tts: readTts(),
}))

const get = useHudChat.getState
const set = useHudChat.setState

// ── Session (module state — one connection per page) ──

let ws: AgentChatWS | null = null
let attached: HudAgent | null = null
let offs: Array<() => void> = []
/** The next chat frame must carry the surface rules (new connection, /new). */
let rulesNeeded = true
let chatVisible = false
/** We closed the socket because the page went hidden. */
let pausedHidden = false
let connectedAt = 0
/** The last message we sent and have no reply for yet. */
let inflight: { id: string; text: string; clientMsgId: string; withRules: boolean; at: number } | null = null
/** We sent /new ourselves (its command_result must not clear again). */
let pendingNew = false
let approvalTimer: ReturnType<typeof setTimeout> | null = null
let seq = 0
let lastForcedRefresh = 0

function localId(prefix: string): string {
  seq += 1
  return `${prefix}${seq}`
}

function appendMsg(m: HudMsg): void {
  const msgs = get().messages
  set({ messages: msgs.length >= MAX_MSGS ? [...msgs.slice(msgs.length - MAX_MSGS + 1), m] : [...msgs, m] })
}

function clearApprovalTimer(): void {
  if (approvalTimer) { clearTimeout(approvalTimer); approvalTimer = null }
}

/** Nothing is running as far as we know (a pending approval may remain:
 *  the agent accepts its answer from any socket until it times out). */
function idle(): void {
  set({ busy: false, status: '', narration: '' })
}

/** End of a turn: nothing is running any more. */
function turnOver(): void {
  clearApprovalTimer()
  set({ busy: false, status: '', narration: '', approval: null })
}

// ── Speech ──

function cancelSpeech(): void {
  try { window.speechSynthesis?.cancel() } catch { /* unsupported */ }
}

function speak(markdown: string): void {
  try {
    const synth = window.speechSynthesis
    if (!synth || typeof SpeechSynthesisUtterance === 'undefined') return
    const text = speakableText(markdown, 600)
    if (!text) return
    synth.cancel()
    for (const chunk of speechChunks(text)) {
      const u = new SpeechSynthesisUtterance(chunk)
      u.lang = 'en-US'
      synth.speak(u)
    }
  } catch { /* speech is best-effort */ }
}

// ── Auth freshness ──

/** Seconds-precision check of the access token's `exp` (no verification —
 *  only used to decide whether to refresh before reopening a socket). */
function tokenExpiresWithin(ms: number): boolean {
  const token = useAuthStore.getState().token
  if (!token) return true
  try {
    const part = token.split('.')[1] || ''
    const b64 = part.replace(/-/g, '+').replace(/_/g, '/')
    const json = JSON.parse(atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4))) as { exp?: unknown }
    const exp = Number(json.exp)
    if (!Number.isFinite(exp)) return true
    return exp * 1000 - Date.now() < ms
  } catch {
    return true
  }
}

/** Make sure the next socket open carries a valid fd_token. refreshAccessToken
 *  is single-flight; we only rotate when the token is about to expire (or
 *  after repeated failures, at most once a minute). */
async function ensureFreshToken(withinMs: number, force = false): Promise<void> {
  if (!useAuthStore.getState().authEnabled) return
  const now = Date.now()
  if (force && now - lastForcedRefresh > 60_000) {
    lastForcedRefresh = now
    await refreshAccessToken()
    return
  }
  if (tokenExpiresWithin(withinMs)) await refreshAccessToken()
}

// ── Surface rules ──

let rulesText: string | null = null

function loadRules(): Promise<string | null> {
  if (rulesText !== null) return Promise.resolve(rulesText)
  return getHudConfig()
    .then((c) => {
      rulesText = typeof c?.surface_rules === 'string' ? c.surface_rules : ''
      return rulesText
    })
    .catch(() => null)
}

/** The rules block, or null when it is not available within `ms`. */
function rulesWithin(ms: number): Promise<string | null> {
  if (rulesText !== null) return Promise.resolve(rulesText)
  return Promise.race([
    loadRules(),
    new Promise<null>((resolve) => setTimeout(() => resolve(null), ms)),
  ])
}

// ── Replay reconciliation ──

/**
 * Give replayed rows the ids of identical rows we already show, so a
 * reconnect (e.g. after the display slept) does not remount the transcript
 * and drop the wearer's focus. Returns the rows plus those that are new.
 */
function reconcile(prev: HudMsg[], rows: Omit<HudMsg, 'id'>[]): { msgs: HudMsg[]; fresh: HudMsg[] } {
  const pool = new Map<string, string[]>()
  for (const m of prev) {
    const k = `${m.role}\u0001${m.text}`
    const ids = pool.get(k)
    if (ids) ids.push(m.id)
    else pool.set(k, [m.id])
  }
  const used = new Set<string>()
  const fresh: HudMsg[] = []
  const msgs = rows.map((r) => {
    const reuse = pool.get(`${r.role}\u0001${r.text}`)?.shift()
    if (reuse) { used.add(reuse); return { ...r, id: reuse } }
    const m = { ...r, id: localId('r') }
    fresh.push(m)
    return m
  })
  // Messages we sent on this connection before the replay arrived (they may
  // not be in it yet) stay after the replay.
  for (const m of prev) {
    if (!m.replay && m.role === 'user' && m.ts >= connectedAt && !used.has(m.id)) msgs.push(m)
  }
  return { msgs: msgs.slice(-MAX_MSGS), fresh }
}

// ── Event handlers ──

type Data = Record<string, unknown>

function str(v: unknown): string {
  return typeof v === 'string' ? v : v == null ? '' : String(v)
}

function onReplayBatch(data: Data): void {
  const items = Array.isArray(data.messages) ? (data.messages as unknown[]) : []
  const rows: Omit<HudMsg, 'id'>[] = []
  for (const it of items) {
    if (!it || typeof it !== 'object') continue
    const o = it as Data
    if (o.type !== 'chat_message' || (o.role !== 'user' && o.role !== 'assistant')) continue
    const text = o.role === 'user' ? cleanUserText(o.content) : cleanAssistantText(o.content)
    if (!text) continue
    rows.push({ role: o.role, text, ts: toMs(o.timestamp), replay: true })
  }
  const { msgs, fresh } = reconcile(get().messages, rows.slice(-MAX_MSGS))
  set({ messages: msgs })
  // A reply that finished while we were away (display asleep, network blip):
  // new assistant rows after the message we were waiting on.
  if (inflight) {
    const sentId = inflight.id
    let from = msgs.findIndex((m) => m.id === sentId)
    if (from < 0) from = msgs.map((m) => m.role).lastIndexOf('user')
    const freshIds = new Set(fresh.map((m) => m.id))
    const replies = msgs.slice(from + 1).filter((m) => m.role === 'assistant' && freshIds.has(m.id))
    if (replies.length) {
      inflight = null
      if (!chatVisible) set({ unread: get().unread + replies.length })
      if (get().tts) speak(replies[replies.length - 1].text)
    }
  }
}

function onReplayDone(): void {
  // A message sent on this connection is still running: keep busy.
  if (inflight && inflight.at >= connectedAt) return
  idle()
}

function onChatMessage(data: Data): void {
  const role = data.role
  if (data.replay) {
    if (role !== 'user' && role !== 'assistant') return
    const text = role === 'user' ? cleanUserText(data.content) : cleanAssistantText(data.content)
    if (text) appendMsg({ id: localId('r'), role, text, ts: toMs(data.timestamp), replay: true })
    return
  }
  if (role === 'user') {
    // Live user echoes are ours (already shown) or another surface's — except
    // a "new topic" cue, which the agent turns into /new + this echo.
    if (!data.rotation_cue) return
    const text = cleanUserText(data.content)
    rulesNeeded = true
    const msgs = get().messages
    const last = msgs[msgs.length - 1]
    if (text && !(last && last.role === 'user' && last.text === text)) {
      appendMsg({ id: localId('l'), role: 'user', text, ts: toMs(data.timestamp) })
    }
    return
  }
  if (role !== 'assistant') return
  const text = cleanAssistantText(data.content)
  const automation = !!data.automation_lane
  if (!automation) {
    inflight = null
    turnOver()
  }
  if (!text) return
  appendMsg({ id: localId('a'), role: 'assistant', text, ts: toMs(data.timestamp) })
  if (!chatVisible) set({ unread: get().unread + 1 })
  if (!automation && get().tts) speak(text)
}

function onStatus(data: Data): void {
  const raw = str(data.text || data.status).trim()
  if (!raw) return
  if (isIdleStatus(raw)) {
    inflight = null
    turnOver()
  } else {
    set({ busy: true, status: prettyStatus(raw) })
  }
}

function onError(data: Data): void {
  const retryable = data.retryable === true
  const msgId = str(data.client_msg_id)
  if (retryable && inflight && (!msgId || msgId === inflight.clientMsgId)) {
    // The agent refused the turn (busy): take our echo back and return the
    // text to the composer so one pinch on Send retries it.
    const failed = inflight
    inflight = null
    if (failed.withRules) rulesNeeded = true
    set({
      messages: get().messages.filter((m) => m.id !== failed.id),
      draft: get().draft.trim() ? get().draft : failed.text,
    })
  }
  inflight = null
  pendingNew = false
  turnOver()
  const message = str(data.message || data.text || data.error).trim()
  set({ error: retryable ? 'Agent is busy — try again' : (message || 'Something went wrong').slice(0, 300) })
}

function clearTranscript(): void {
  inflight = null
  clearApprovalTimer()
  set({ messages: [], nextSteps: [], approval: null, error: null, narration: '' })
}

function onCommandResult(data: Data): void {
  const command = str(data.command).trim()
  const content = str(data.content).trim()
  const isNew = /^\/(?:new|clear)\b/i.test(command) || /New session created/i.test(content)
  if (isNew) {
    rulesNeeded = true
    // Our own /new already cleared the screen; don't wipe what the wearer
    // sent since.
    if (pendingNew) pendingNew = false
    else clearTranscript()
  } else if (content) {
    appendMsg({ id: localId('s'), role: 'system', text: content.slice(0, 2000), ts: Date.now() })
  }
  if (inflight && inflight.text.startsWith('/')) inflight = null
  if (!inflight) turnOver()
}

function onNextSteps(data: Data): void {
  const opts = Array.isArray(data.options) ? (data.options as unknown[]) : []
  const steps: NextStep[] = []
  for (const o of opts) {
    if (!o || typeof o !== 'object') continue
    const label = str((o as Data).label).trim()
    const action = str((o as Data).action).trim() || label
    if (label && action) steps.push({ label, action })
  }
  set({ nextSteps: steps })
}

function onApprovalRequest(data: Data): void {
  const id = str(data.id)
  if (!id) return
  clearApprovalTimer()
  set({ approval: { id, message: str(data.message).trim() || 'The agent asks for approval.', category: str(data.category) } })
  approvalTimer = setTimeout(() => {
    approvalTimer = null
    if (get().approval?.id === id) set({ approval: null })
  }, APPROVAL_TTL_MS)
}

function bind(sock: AgentChatWS): void {
  const on = (event: string, fn: (data: Data) => void) => {
    offs.push(sock.on(event, (data) => { if (ws === sock) fn(data) }))
  }
  on('_connected', () => {
    connectedAt = Date.now()
    rulesNeeded = true
    set({ connected: true, closed: false, error: null })
  })
  on('_disconnected', () => {
    set({ connected: false })
    idle()
  })
  on('_reconnecting', (d) => {
    const attempt = Number(d.attempt) || 0
    void ensureFreshToken(120_000, attempt >= 3)
  })
  on('_closed', (d) => {
    set({ connected: false, closed: true, error: str(d.reason).trim() || 'Connection closed by Flight Deck' })
    turnOver()
  })
  on('replay_batch', onReplayBatch)
  on('replay_done', onReplayDone)
  on('chat_message', onChatMessage)
  on('status', onStatus)
  on('narration', (d) => {
    const t = str(d.text).trim()
    if (t) set({ narration: t.length > 200 ? t.slice(0, 199) + '…' : t })
  })
  on('monitor', (d) => {
    if (d.replay) return
    set({ busy: true, status: get().status || 'Working…' })
  })
  on('error', onError)
  on('command_result', onCommandResult)
  on('next_steps', onNextSteps)
  on('approval_request', onApprovalRequest)
  // welcome (session info), response_stream, tool_stream, usage, turn_usage:
  // nothing to show on the glasses.
}

// ── Visibility (Meta: close sockets while hidden) ──

function onVisibility(): void {
  const sock = ws
  if (!sock) return
  if (document.visibilityState === 'hidden') {
    if (pausedHidden) return
    pausedHidden = true
    sock.disconnect()
    set({ connected: false })
    return
  }
  if (!pausedHidden) return
  void (async () => {
    await ensureFreshToken(180_000)
    if (ws !== sock || !pausedHidden || document.visibilityState !== 'visible') return
    pausedHidden = false
    sock.connect() // the agent replays the session on connect
  })()
}

function newSocket(agent: HudAgent): AgentChatWS | null {
  if (agent.kind === 'shared') {
    return agent.ref ? new AgentChatWS(agent.id, '', 0, '', 'A', { sharedRef: agent.ref }) : null
  }
  // Empty agent token: Flight Deck's proxy resolves it and checks ownership.
  return agent.port ? new AgentChatWS(agent.id, 'localhost', agent.port, '', 'A') : null
}

// ── Files mentioned in replies ──

const FILES_TTL_MS = 60_000
const fileCache = new Map<string, { at: number; files: Promise<HudFile[]> }>()

function agentFiles(agent: HudAgent, maxAgeMs: number): { at: number; files: Promise<HudFile[]> } {
  const hit = fileCache.get(agent.id)
  if (hit && Date.now() - hit.at <= maxAgeMs) return hit
  const entry = { at: Date.now(), files: listMarkdownFiles(agent) }
  fileCache.set(agent.id, entry)
  entry.files.catch(() => { if (fileCache.get(agent.id) === entry) fileCache.delete(agent.id) })
  return entry
}

/**
 * The agent's markdown file a reply refers to (listing cached 60 s per
 * agent; a miss on a listing older than a few seconds re-lists once, since
 * the reply may have just created the file). Throws HudApiError on a failed
 * listing.
 */
export async function resolveFileRef(agent: HudAgent, ref: string): Promise<HudFile | null> {
  const entry = agentFiles(agent, FILES_TTL_MS)
  const found = matchFileRef(await entry.files, ref)
  if (found || Date.now() - entry.at < 3_000) return found
  return matchFileRef(await agentFiles(agent, 0).files, ref)
}

// ── Controller ──

export const chat = {
  /** Connect to `agent` (no-op if already attached to it). */
  attach(agent: HudAgent): void {
    const same = attached && attached.id === agent.id && attached.port === agent.port && attached.ref === agent.ref
    if (same && ws && !get().closed) {
      attached = agent
      return
    }
    chat.detach()
    attached = agent
    rulesNeeded = true
    pausedHidden = false
    set({ ...sessionDefaults(), agentId: agent.id })
    void loadRules() // prefetch: the first message should not wait for it
    const sock = newSocket(agent)
    if (!sock) {
      set({ error: 'Agent is not running', closed: true })
      return
    }
    ws = sock
    bind(sock)
    document.addEventListener('visibilitychange', onVisibility)
    if (document.visibilityState === 'hidden') pausedHidden = true
    else sock.connect()
  },

  /** Open a fresh connection to the attached agent (after a final close). */
  reconnect(): void {
    const agent = attached
    if (!agent) return
    chat.detach()
    chat.attach(agent)
  },

  /** Close the connection and clear the session. */
  detach(): void {
    for (const off of offs) off()
    offs = []
    document.removeEventListener('visibilitychange', onVisibility)
    if (ws) ws.disconnect()
    ws = null
    attached = null
    inflight = null
    pendingNew = false
    pausedHidden = false
    clearApprovalTimer()
    cancelSpeech()
    set(sessionDefaults())
  },

  /** Send a user message. Returns false if it could not be sent. */
  send(text: string): boolean {
    const t = (typeof text === 'string' ? text : '').trim().slice(0, 4000)
    const st = get()
    const sock = ws
    if (!t || !sock || !st.connected || st.busy) return false
    cancelSpeech()
    const id = localId('l')
    const clientMsgId = newId()
    // Slash commands must reach the agent as typed (no rules prefix).
    const withRules = rulesNeeded && !t.startsWith('/')
    const at = Date.now()
    inflight = { id, text: t, clientMsgId, withRules, at }
    appendMsg({ id, role: 'user', text: t, ts: at })
    set({ busy: true, status: 'Thinking…', narration: '', nextSteps: [], error: null })
    void (async () => {
      const rules = withRules ? await rulesWithin(RULES_WAIT_MS) : null
      if (ws !== sock) return
      if (!sock.connected) {
        if (inflight?.id === id) {
          inflight = null
          set({ messages: get().messages.filter((m) => m.id !== id), draft: get().draft.trim() ? get().draft : t })
        }
        turnOver()
        set({ error: 'Not sent — connection lost' })
        return
      }
      // The rules block ends with "USER MESSAGE:\n", so plain concatenation.
      sock.sendJSON({ type: 'chat', content: (rules || '') + t, surface: 'glasses', client_msg_id: clientMsgId })
      if (rules) rulesNeeded = false
    })()
    return true
  },

  /** Stop the running turn. */
  cancel(): void {
    if (!ws || !get().connected) return
    ws.sendJSON({ type: 'cancel' })
    set({ status: 'Stopping…' })
  },

  /** Start a fresh agent session (/new). */
  newSession(): void {
    if (!ws || !get().connected) {
      set({ error: 'Not connected' })
      return
    }
    cancelSpeech()
    ws.sendJSON({ type: 'command', command: '/new' })
    pendingNew = true
    rulesNeeded = true
    clearTranscript()
  },

  /** Answer the pending approval request. */
  respondApproval(approved: boolean): void {
    const a = get().approval
    if (!a) return
    clearApprovalTimer()
    set({ approval: null })
    if (!ws || !get().connected) {
      set({ error: 'Not connected — the agent will decide on its own' })
      return
    }
    ws.sendJSON({ type: 'approval_response', id: a.id, approved })
  },

  /** The Chat tab is visible: clear the unread counter. */
  markRead(): void {
    if (get().unread) set({ unread: 0 })
  },

  /** Chat tab shown / hidden (unread counting). */
  setChatVisible(visible: boolean): void {
    chatVisible = visible
    if (visible) chat.markRead()
  },

  setTts(on: boolean): void {
    set({ tts: on })
    try { localStorage.setItem(TTS_KEY, on ? '1' : '0') } catch { /* storage blocked */ }
    if (!on) cancelSpeech()
  },

  /** Keep the composer's unsent text. */
  setDraft(text: string): void {
    set({ draft: text })
  },

  /** The agent the session is attached to (for WebMCP / screens). */
  agent(): HudAgent | null {
    return attached
  },
}
