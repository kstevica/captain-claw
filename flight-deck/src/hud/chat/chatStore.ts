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
//   - for own agents Flight Deck's proxy accepts the socket BEFORE it dials
//     the agent, so the connection counts only from the agent's welcome;
//   - the user echo of a live turn carries the raw content (rules block
//     included) and no client_msg_id, so live user echoes are ignored — we
//     show our own local echo instead;
//   - a non-replay assistant chat_message is the reply; the turn ends with
//     status 'ready' (the agent keeps its lane until then).

import { create } from 'zustand'
import { AgentChatWS } from '../../services/agentChat'
import { refreshAccessToken, useAuthStore } from '../../stores/authStore'
import { getHudConfig, type HudAgent, type HudFile } from '../api'
import { getCachedFiles, isFresh, loadFiles } from '../files/filesCache'
import {
  capReply, cleanAssistantText, cleanUserText, isIdleStatus, matchFileRef, newId, prettyStatus,
  speakableText, speechChunks, syntheticRowText, toMs,
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

export interface Approval {
  id: string
  message: string
  category: string
  /** Epoch ms when the agent stops waiting and approves on its own. */
  expiresAt: number
}

export interface ChatState {
  /** Agent the session is attached to (null = none). */
  agentId: string | null
  /** The agent answered this connection (its welcome arrived). */
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
  /** The draft was written by voice (WebMCP) and not touched since: the
   *  wearer still has to read it and pinch Send. */
  draftStaged: boolean
}

/** Transcript rows kept in memory / on screen (the glasses CPU is slow). */
export const MAX_MSGS = 40
const TTS_KEY = 'hud.tts.v1'
/** How long the card stays up. The agent approves on its own after 15 s for
 *  a playbook on its main lane and after 60 s for a peer consult or on a
 *  shared agent (web_server.py) — a little less here, so Deny is never sent
 *  after the agent stopped listening. */
const PLAYBOOK_APPROVAL_TTL_MS = 14_000
const APPROVAL_TTL_MS = 58_000
/** Longest we hold a first message waiting for the rules block. */
const RULES_WAIT_MS = 4_000
/** Own agent: connection attempts that may end before the agent's welcome
 *  (agent stopped or crashed) before we stop and offer Reconnect. */
const MAX_FAILED_ATTEMPTS = 6
/** Keep the socket this long after the page is hidden: the display sleeps
 *  after ~25 s and every reopen replays the whole session over a ~500 Kbps
 *  link. Longer absences close it (Meta: no sockets while hidden). */
const HIDDEN_GRACE_MS = 90_000
/** Busy with nothing of ours in flight: idle after this much agent silence. */
const QUIET_MS = 30_000
/** After Stop: idle after this much silence (nothing was running). */
const STOP_QUIET_MS = 5_000

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
    draftStaged: false,
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
/** The agent's welcome arrived on the current socket. */
let welcomed = false
/** Own agent: attempts in a row that ended before the welcome. */
let failedAttempts = 0
let retryTimer: ReturnType<typeof setTimeout> | null = null
/** When the page went hidden, and the disconnect pending after the grace. */
let hiddenAt = 0
let hiddenTimer: ReturnType<typeof setTimeout> | null = null
/** The agent session the transcript belongs to (welcome / session_info). */
let sessionId: string | null = null
/** The last message we sent and have no reply for yet. `accepted`: the agent
 *  started on it (a status arrived), so a refusal can't be about it. */
let inflight: {
  id: string; text: string; clientMsgId: string; withRules: boolean; at: number; accepted: boolean
} | null = null
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

function clearRetry(): void {
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null }
}

function clearHiddenTimer(): void {
  if (hiddenTimer) { clearTimeout(hiddenTimer); hiddenTimer = null }
}

// ── Busy watchdog ──
// Some agent work reports a status and never a closing 'ready' (background
// LLM calls after a turn, a Stop when nothing runs). While busy with no
// message of ours in flight, fall back to idle once the agent goes quiet.

let watchdog: ReturnType<typeof setTimeout> | null = null
let lastActivity = 0
let quietLimit = QUIET_MS
/** Stop was pressed: the watchdog may also drop our in-flight message. */
let stopping = false

function clearWatchdog(): void {
  if (watchdog) { clearTimeout(watchdog); watchdog = null }
}

function checkWatchdog(): void {
  watchdog = null
  if (!get().busy || (inflight && !stopping)) return
  const quiet = Date.now() - lastActivity
  if (quiet < quietLimit) {
    watchdog = setTimeout(checkWatchdog, quietLimit - quiet)
    return
  }
  if (stopping) inflight = null
  idle()
}

/** The agent did something: a running turn is alive. */
function touch(): void {
  lastActivity = Date.now()
  quietLimit = QUIET_MS
  if (!watchdog) watchdog = setTimeout(checkWatchdog, QUIET_MS)
}

/** Nothing is running as far as we know (a pending approval may remain:
 *  the agent accepts its answer from any socket until it times out). */
function idle(): void {
  clearWatchdog()
  stopping = false
  set({ busy: false, status: '', narration: '' })
}

/** End of a turn: nothing is running any more. */
function turnOver(): void {
  clearApprovalTimer()
  clearWatchdog()
  stopping = false
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

/** A short spoken cue, queued after whatever is being read. */
function announce(text: string): void {
  try {
    const synth = window.speechSynthesis
    if (!synth || typeof SpeechSynthesisUtterance === 'undefined') return
    const u = new SpeechSynthesisUtterance(text)
    u.lang = 'en-US'
    synth.speak(u)
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
  // Messages sent on this connection can't be in its replay (the agent reads
  // no frame before replay_done): they never match an older identical row
  // ("yes", "continue") and stay after the replay.
  const sentHere = (m: HudMsg) => !m.replay && m.role === 'user' && m.ts >= connectedAt
  const pool = new Map<string, string[]>()
  for (const m of prev) {
    if (sentHere(m)) continue
    const k = `${m.role}\u0001${m.text}`
    const ids = pool.get(k)
    if (ids) ids.push(m.id)
    else pool.set(k, [m.id])
  }
  const fresh: HudMsg[] = []
  const msgs: HudMsg[] = new Array<HudMsg>(rows.length)
  // Newest first: what we show is the end of the conversation, so a repeated
  // text matches its latest copy and older copies count as new.
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i]
    const reuse = pool.get(`${r.role}\u0001${r.text}`)?.pop()
    if (reuse) {
      msgs[i] = { ...r, id: reuse }
    } else {
      msgs[i] = { ...r, id: localId('r') }
      fresh.push(msgs[i])
    }
  }
  for (const m of prev) if (sentHere(m)) msgs.push(m)
  return { msgs: msgs.slice(-MAX_MSGS), fresh }
}

// ── Event handlers ──

type Data = Record<string, unknown>

function str(v: unknown): string {
  return typeof v === 'string' ? v : v == null ? '' : String(v)
}

/**
 * A replayed chat row as the glasses show it, or null to leave it out. The
 * replay's `origin` tells the wearer's own words from rows the agent's
 * machinery wrote (scheduled jobs, results from other agents, nudges to
 * itself), which must not be labelled "You".
 */
function replayRow(o: Data): Omit<HudMsg, 'id'> | null {
  const role = o.role
  const origin = str(o.origin)
  const ts = toMs(o.timestamp)
  if (role === 'assistant') {
    if (origin === 'rejected') return null // a draft the agent threw away
    const text = capReply(cleanAssistantText(o.content))
    return text ? { role, text, ts, replay: true } : null
  }
  if (role !== 'user') return null
  const text = cleanUserText(o.content)
  if (!text) return null
  if (!origin || origin === 'human') return { role, text, ts, replay: true }
  if (origin === 'corrective') return null // the agent correcting itself mid-turn
  return { role: 'system', text: syntheticRowText(origin, text), ts, replay: true }
}

function onReplayBatch(data: Data): void {
  const items = Array.isArray(data.messages) ? (data.messages as unknown[]) : []
  const rows: Omit<HudMsg, 'id'>[] = []
  // Only the last MAX_MSGS rows are kept: clean just those (newest first).
  for (let i = items.length - 1; i >= 0 && rows.length < MAX_MSGS; i--) {
    const it = items[i]
    if (!it || typeof it !== 'object') continue
    const o = it as Data
    if (o.type !== 'chat_message') continue
    const row = replayRow(o)
    if (row) rows.push(row)
  }
  rows.reverse()
  const { msgs, fresh } = reconcile(get().messages, rows)
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
    const row = replayRow(data)
    if (row) appendMsg({ ...row, id: localId('r') })
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
  const text = capReply(cleanAssistantText(data.content))
  const automation = !!data.automation_lane
  // Automation-lane results and proactive notices arrive outside any turn.
  const sideline = automation || data.notification === true || data.proactive === true
  if (!sideline && (inflight || get().busy)) {
    // The reply is in, but the agent keeps its lane until 'ready' (next-step
    // extraction can take one more LLM call): stay busy so a send now isn't
    // refused.
    inflight = null
    touch()
    set({ busy: true, status: 'Finishing…', narration: '' })
  }
  if (!text) return
  appendMsg({ id: localId('a'), role: 'assistant', text, ts: toMs(data.timestamp) })
  if (!chatVisible) set({ unread: get().unread + 1 })
  if (!automation && get().tts) speak(text)
}

/** Statuses of agent work outside any turn: never followed by 'ready'. */
const BACKGROUND_STATUS_RE = /·\s*(?:terminal_watcher|intentions_generator|session_description)\b/i
/** A labelled LLM call ("Calling LLM (model) · label"): inside a turn it
 *  follows 'thinking'; on its own it is background work. */
const LABELLED_LLM_RE = /^Calling LLM\b[^·]*·/i

function onStatus(data: Data): void {
  const raw = str(data.text || data.status).trim()
  if (!raw) return
  if (isIdleStatus(raw)) {
    inflight = null
    turnOver()
    return
  }
  if (BACKGROUND_STATUS_RE.test(raw)) return
  if (!get().busy && !inflight && LABELLED_LLM_RE.test(raw)) return
  if (inflight) inflight.accepted = true
  touch()
  set({ busy: true, status: prettyStatus(raw) })
}

function onError(data: Data): void {
  const message = str(data.message || data.text || data.error).trim()
  if (!welcomed) {
    // Before the agent's welcome an error is about the connection itself
    // (Flight Deck's proxy could not reach the agent), not about a turn —
    // show one steady line instead of the raw exception on every retry.
    const unreachable = !message || /^Connection failed\b/i.test(message)
    set({ error: unreachable ? 'Agent unreachable — retrying…' : message.slice(0, 300) })
    return
  }
  // Own agents mark a busy refusal `retryable`; a shared agent's member path
  // only sends code 'busy'.
  const refused = data.retryable === true || str(data.code) === 'busy'
  const msgId = str(data.client_msg_id)
  if (refused && inflight && !inflight.accepted && (!msgId || msgId === inflight.clientMsgId)) {
    // The agent refused the turn: take our echo back and return the text to
    // the composer so one pinch on Send retries it.
    const failed = inflight
    inflight = null
    if (failed.withRules) rulesNeeded = true
    set({
      messages: get().messages.filter((m) => m.id !== failed.id),
      draft: get().draft.trim() ? get().draft : failed.text,
      draftStaged: false,
    })
  }
  inflight = null
  turnOver()
  set({ error: refused ? 'Agent is busy — try again' : (message || 'Something went wrong').slice(0, 300) })
}

function clearTranscript(): void {
  inflight = null
  clearApprovalTimer()
  set({ messages: [], nextSteps: [], approval: null, error: null, narration: '' })
}

/** The agent now runs a different session (/new or a switch — ours or
 *  another surface's): the transcript and the rules block belong to the old one. */
function sessionChanged(): void {
  rulesNeeded = true
  clearTranscript()
}

function onCommandResult(data: Data): void {
  const command = str(data.command).trim()
  const content = str(data.content).trim()
  const isNew = /^\/(?:new|clear)\b/i.test(command) || /New session created/i.test(content)
  if (isNew) {
    // Only now — a refused /new must not have wiped the screen.
    sessionChanged()
  } else if (content) {
    appendMsg({ id: localId('s'), role: 'system', text: content.slice(0, 2000), ts: Date.now() })
  }
  if (inflight && inflight.text.startsWith('/')) inflight = null
  if (!inflight) turnOver()
}

function onWelcome(data: Data): void {
  welcomed = true
  failedAttempts = 0
  clearRetry()
  const s = data.session && typeof data.session === 'object' ? (data.session as Data) : {}
  const id = str(s.id)
  // A new or switched session since we last saw it, or an empty one (which
  // sends no replay to replace the old rows with).
  if ((id && sessionId && id !== sessionId) || s.message_count === 0) sessionChanged()
  if (id) sessionId = id
  set({ connected: true, closed: false, error: null })
}

function onSessionInfo(data: Data): void {
  // Broadcast after every turn too: only a different id is news.
  const id = str(data.id)
  if (!id) return
  const prev = sessionId
  sessionId = id
  if (!prev || prev === id) return
  sessionChanged()
  // A switch to an existing conversation: reopen so the agent replays it.
  const sock = ws
  if (sock && Number(data.message_count) > 0) {
    welcomed = false
    set({ connected: false })
    sock.disconnect()
    sock.connect()
  }
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

function approvalTtl(category: string): number {
  return category === 'playbook' && attached?.kind !== 'shared' ? PLAYBOOK_APPROVAL_TTL_MS : APPROVAL_TTL_MS
}

/** The agent stopped waiting for the card's answer: take it down and say so. */
function approvalClosed(note: string): void {
  clearApprovalTimer()
  set({ approval: null })
  appendMsg({ id: localId('s'), role: 'system', text: note, ts: Date.now() })
}

function onApprovalRequest(data: Data): void {
  const id = str(data.id)
  if (!id) return
  clearApprovalTimer()
  const category = str(data.category)
  const ttl = approvalTtl(category)
  set({
    approval: {
      id,
      message: str(data.message).trim() || 'The agent asks for approval.',
      category,
      expiresAt: Date.now() + ttl,
    },
  })
  approvalTimer = setTimeout(() => {
    approvalTimer = null
    if (get().approval?.id === id) approvalClosed('No answer in time — the agent went ahead on its own.')
  }, ttl)
  // The card may be on a tab the wearer isn't looking at.
  if (get().tts) announce('Approval needed')
}

function onMonitor(d: Data): void {
  if (d.replay) return
  const args = d.arguments && typeof d.arguments === 'object' ? (d.arguments as Data) : {}
  if (d.tool_name === 'playbook' && args.action === 'approval_resolved' && get().approval?.category === 'playbook') {
    // Answered on another surface, or the agent's own wait ran out.
    approvalClosed(args.approved === false
      ? 'Approval closed — the playbook was declined.'
      : 'Approval closed — the agent went ahead.')
  }
  touch()
  set({ busy: true, status: get().status || 'Working…' })
}

/** Own agent: a connection attempt ended before the agent's welcome. */
function onFailedAttempt(sock: AgentChatWS): void {
  // AgentChatWS restarts its backoff on every open, and Flight Deck's proxy
  // opens even when the agent is down — it would retry every 500 ms forever.
  // Back off here instead, and stop after a few tries (Reconnect chip).
  sock.disconnect()
  failedAttempts += 1
  if (failedAttempts >= MAX_FAILED_ATTEMPTS) {
    set({ closed: true, error: 'Agent unreachable' })
    return
  }
  const delay = Math.min(15_000, 1_000 * 2 ** (failedAttempts - 1))
  clearRetry()
  retryTimer = setTimeout(() => {
    retryTimer = null
    void (async () => {
      await ensureFreshToken(120_000, failedAttempts >= 3)
      if (ws !== sock || pausedHidden || retryTimer || get().closed) return
      sock.connect()
    })()
  }, delay)
}

function bind(sock: AgentChatWS): void {
  const on = (event: string, fn: (data: Data) => void) => {
    offs.push(sock.on(event, (data) => { if (ws === sock) fn(data) }))
  }
  // Socket open (own agent) / agent welcome (shared). `connected` waits for
  // the welcome either way.
  on('_connected', () => {
    connectedAt = Date.now()
    rulesNeeded = true
    welcomed = false
  })
  on('welcome', onWelcome)
  on('_disconnected', () => {
    const live = welcomed
    welcomed = false
    set({ connected: false })
    idle()
    // A live socket that dropped: AgentChatWS reconnects (its backoff works
    // then). Shared sockets only count as open on the welcome anyway.
    if (!live && attached?.kind !== 'shared') onFailedAttempt(sock)
  })
  on('_reconnecting', (d) => {
    const attempt = Number(d.attempt) || 0
    void ensureFreshToken(120_000, attempt >= 3)
  })
  on('_closed', (d) => {
    clearRetry()
    set({ connected: false, closed: true, error: str(d.reason).trim() || 'Connection closed by Flight Deck' })
    turnOver()
  })
  on('replay_batch', onReplayBatch)
  on('replay_done', onReplayDone)
  on('chat_message', onChatMessage)
  on('status', onStatus)
  on('narration', (d) => {
    const t = str(d.text).trim()
    if (!t) return
    touch()
    set({ narration: t.length > 200 ? t.slice(0, 199) + '…' : t })
  })
  on('monitor', onMonitor)
  // Nothing to show, but a streaming turn is alive.
  on('response_stream', touch)
  on('tool_stream', touch)
  on('error', onError)
  on('command_result', onCommandResult)
  on('session_info', onSessionInfo)
  on('next_steps', onNextSteps)
  on('approval_request', onApprovalRequest)
  // usage, turn_usage: nothing to show on the glasses.
}

// ── Visibility (Meta: close sockets while hidden — after a grace period) ──

function pauseHidden(sock: AgentChatWS): void {
  clearHiddenTimer()
  clearRetry()
  pausedHidden = true
  sock.disconnect()
  welcomed = false
  set({ connected: false })
}

function onVisibility(): void {
  const sock = ws
  if (!sock) return
  if (document.visibilityState === 'hidden') {
    // Closed for good: nothing to pause, and waking must not reopen it
    // (the Reconnect chip does).
    if (pausedHidden || hiddenTimer || get().closed) return
    hiddenAt = Date.now()
    hiddenTimer = setTimeout(() => {
      hiddenTimer = null
      if (ws === sock && document.visibilityState === 'hidden') pauseHidden(sock)
    }, HIDDEN_GRACE_MS)
    return
  }
  if (hiddenTimer) {
    // Back within the grace: the socket (and the transcript) are current. A
    // hidden page may have its timers suspended, so check the clock too — a
    // long sleep's socket may be dead without knowing it.
    if (Date.now() - hiddenAt < HIDDEN_GRACE_MS) {
      clearHiddenTimer()
      return
    }
    pauseHidden(sock)
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

/**
 * The agent's markdown file a reply refers to, from the Files tab's cache
 * (one listing serves the chip and the file screen). A miss on a listing
 * older than a few seconds re-lists once, since the reply may have just
 * created the file. Throws HudApiError on a failed listing.
 */
export async function resolveFileRef(agent: HudAgent, ref: string): Promise<HudFile | null> {
  const hit = getCachedFiles(agent.id)
  if (hit && isFresh(hit)) {
    const found = matchFileRef(hit.files, ref)
    if (found || Date.now() - hit.at < 3_000) return found
  }
  return matchFileRef(await loadFiles(agent), ref)
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
    // Prefetch: the first message should not wait for it (shared agents
    // never get it — see send()).
    if (agent.kind !== 'shared') void loadRules()
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
    pausedHidden = false
    welcomed = false
    failedAttempts = 0
    sessionId = null
    stopping = false
    clearRetry()
    clearHiddenTimer()
    clearWatchdog()
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
    // Slash commands must reach the agent as typed (no rules prefix). Shared
    // agents would only absorb and ignore the block: Flight Deck strips
    // `surface` from member frames and member turns run on the 'member'
    // channel, which never gets surface rules.
    const withRules = rulesNeeded && !t.startsWith('/') && attached?.kind !== 'shared'
    const at = Date.now()
    inflight = { id, text: t, clientMsgId, withRules, at, accepted: false }
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
    // The agent answers a Stop only when something was running: give up
    // waiting after a short silence.
    stopping = true
    lastActivity = Date.now()
    quietLimit = STOP_QUIET_MS
    clearWatchdog()
    watchdog = setTimeout(checkWatchdog, STOP_QUIET_MS)
    set({ status: 'Stopping…' })
  },

  /** Start a fresh agent session (/new). */
  newSession(): void {
    if (!ws || !get().connected) {
      set({ error: 'Not connected' })
      return
    }
    // Flight Deck forwards no commands to a shared agent (the dashboard
    // hides New session there too).
    if (attached?.kind === 'shared') {
      set({ error: 'Not available on a shared agent' })
      return
    }
    // /new mid-turn would swap the session under the running turn.
    if (get().busy) {
      set({ error: 'Stop the reply first, then start a new chat' })
      return
    }
    cancelSpeech()
    ws.sendJSON({ type: 'command', command: '/new' })
    // The screen is cleared when the agent confirms (session_info /
    // command_result) — never before, so a refused /new loses nothing.
  },

  /** Answer the pending approval request. */
  respondApproval(approved: boolean): void {
    const a = get().approval
    if (!a) return
    clearApprovalTimer()
    set({ approval: null })
    if (Date.now() > a.expiresAt) {
      set({ error: 'Too late — the agent already went ahead' })
      return
    }
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
    set({ draft: text, draftStaged: false })
  },

  /** Put a message written by voice (WebMCP) in the composer. Never sent
   *  from here: the wearer reads it and pinches Send. */
  stageDraft(text: string): void {
    set({ draft: text.slice(0, 4000), draftStaged: true, error: null })
  },

  /** The agent the session is attached to (for WebMCP / screens). */
  agent(): HudAgent | null {
    return attached
  },
}
