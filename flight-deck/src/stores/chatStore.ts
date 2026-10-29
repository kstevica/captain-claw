import { create } from 'zustand'
import { AgentChatWS, splitImageRefs, type ChatMessage, type TokenUsage, type TopicMark } from '../services/agentChat'
import { useAuthStore, registerSignOutFlush, registerSignOutTeardown } from './authStore'
import { useContainerStore } from './containerStore'
import { useLocalAgentStore } from './localAgentStore'
import { useProcessStore } from './processStore'
import { useTraceStore } from './traceStore'
import { useQuotaStore } from './quotaStore'
import type { QuotaStatus } from '../services/profile'
import type { TraceSpan } from '../types'
import { sanitizeAgentContent } from '../utils/sanitizeAgentContent'
import { scrubUserText } from '../utils/cardScrub'
import { SHARED_PREFIX, sharedContainerId, sharedSliceId } from '../utils/sharedAgent'
import { clampLevel, parsePick, parseProfile, type ReasoningProfile } from '../utils/reasoningEffort'
import { effortFor, useReasoningStore } from './reasoningStore'

// ── Chat persistence helpers ──

function _chatHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function serverUpsertSession(id: string, agentId: string, agentName: string): Promise<void> {
  try {
    await fetch('/fd/chat/sessions', {
      method: 'POST',
      headers: _chatHeaders(),
      credentials: 'include',
      body: JSON.stringify({ id, agent_id: agentId, agent_name: agentName }),
    })
  } catch { /* ignore */ }
}

/** Rows per page of the lane's chat log. */
const HISTORY_PAGE = 500
/** Rows the loaded window may hold. Beyond it the far end is dropped — the
 *  log keeps everything: the oldest rows when the chat grows at the bottom,
 *  the newest when the reader pages back — down to WINDOW_MAX - WINDOW_SLACK,
 *  so a busy chat isn't trimmed on every message. Mounting costs about a
 *  millisecond and 20 DOM nodes per row; 2,000 keeps a day of chat in view. */
const WINDOW_MAX = 2000
const WINDOW_SLACK = 100

interface LogRow { id: number; role: string; content: string; metadata: string; created_at: string }

/** Rows of this lane's Flight Deck chat log, oldest first: the newest page,
 *  the page before `before`, or everything from row `since` on. Null when
 *  the log could not be read (the caller then keeps what the agent replays
 *  rather than treating an empty answer as an empty log). */
async function serverFetchMessages(
  sessionId: string, q: { limit?: number; before?: number; since?: number } = {},
): Promise<ChatMessage[] | null> {
  try {
    const params = new URLSearchParams({ limit: String(q.limit ?? HISTORY_PAGE) })
    if (q.before !== undefined) params.set('before', String(q.before))
    if (q.since !== undefined) params.set('since_id', String(q.since))
    const res = await fetch(`/fd/chat/sessions/${encodeURIComponent(sessionId)}/messages?${params}`, {
      headers: _chatHeaders(),
      credentials: 'include',
    })
    if (!res.ok) return null
    const rows = await res.json() as LogRow[]
    return rows.map(logRowToMessage)
  } catch { return null }
}

function serverLoadMessages(sessionId: string): Promise<ChatMessage[] | null> {
  return serverFetchMessages(sessionId)
}

/** A log row as a transcript message. Log rows are the transcript itself,
 *  not a replay: they carry their row id and are never re-persisted. */
function logRowToMessage(r: LogRow): ChatMessage {
  let meta: Record<string, unknown> = {}
  try { meta = JSON.parse(r.metadata || '{}') } catch { /* ignore */ }
  const role = r.role as ChatMessage['role']
  const topic = meta.topic && typeof meta.topic === 'object' ? meta.topic as TopicMark : undefined
  return {
    id: `hist-${r.id}`,
    rowId: r.id,
    role,
    content: role === 'assistant' ? sanitizeAgentContent(r.content) : r.content,
    timestamp: r.created_at,
    tool_name: meta.tool_name as string | undefined,
    tool_arguments: meta.tool_arguments as Record<string, unknown> | undefined,
    tool_output: meta.tool_output as string | undefined,
    model: meta.model as string | undefined,
    peer_name: meta.peer_name as string | undefined,
    ...(topic ? { topic } : {}),
    ...(meta.via ? { via: String(meta.via) } : {}),
    ...(meta.source_label ? { sourceLabel: String(meta.source_label) } : {}),
    ...(taskMark(meta.task) ? { task: taskMark(meta.task) } : {}),
    ...(meta.automation ? { automation: String(meta.automation) } : {}),
  }
}

/** A scheduled task's marker from a frame or a log row, or undefined. */
function taskMark(raw: unknown): ChatMessage['task'] {
  if (!raw || typeof raw !== 'object') return undefined
  const t = raw as Record<string, unknown>
  const id = String(t.id || '')
  if (!id) return undefined
  return {
    id, name: String(t.name || 'Scheduled task'),
    ...(t.session_id ? { session_id: String(t.session_id) } : {}),
  }
}

/** Message ids withdrawn from the transcript before or while their flush
 *  ran: skipped if still queued, deleted from the log once their row id is
 *  known. */
const _dropped = new Set<string>()

async function serverDeleteMessage(sessionId: string, rowId: number): Promise<void> {
  if (!useAuthStore.getState().authEnabled) return
  try {
    await fetch(`/fd/chat/sessions/${encodeURIComponent(sessionId)}/messages/${rowId}`, {
      method: 'DELETE',
      headers: _chatHeaders(),
      credentials: 'include',
    })
  } catch { /* ignore */ }
}

/** Whether a user frame the agent broadcast is this socket's own turn: the
 *  echoed client_msg_id says so; an agent build that sends neither an id nor
 *  a surface is read as an echo, as before. */
function isOwnTurn(ws: AgentChatWS, data: Record<string, unknown>): boolean {
  const id = String(data.client_msg_id || '')
  if (id) return ws.isOwnMessage(id)
  return !data.via
}

// The agent's new-session cues (config `session.rotation_cues` defaults):
// "new topic: …" opening a message starts a new session, then the rest is
// re-sent as the turn. The composer already echoed the whole line.
const CUE_RE = /^(?:new topic|new conversation|nova tema|novi razgovor)(?:\s*(?:$|\n)|\s*[:\-–—.,!]\s*)/i

function stripCue(text: string): string | null {
  const t = text.trimStart()
  const m = CUE_RE.exec(t)
  return m ? t.slice(m[0].length).trim() : null
}

const foldSpace = (s: string) => s.replace(/\s+/g, ' ').trim()

/** The composer's echo of a "new topic: <rest>" line — sitting above the
 *  divider the cue produced — is withdrawn once the agent re-sends <rest>
 *  as the new topic's first turn, so the message shows once, below. */
function dropCueEcho(key: string, rest: string) {
  const s = useChatStore.getState().sessions.get(key)
  if (!s) return
  const want = foldSpace(rest)
  for (let i = s.messages.length - 1; i >= 0 && i >= s.messages.length - 8; i--) {
    const m = s.messages[i]
    if (m.role !== 'user' || m.replay || m.via) continue
    const stripped = stripCue(m.content)
    if (stripped === null || foldSpace(stripped) !== want) continue
    _dropped.add(m.id)
    useChatStore.setState((state) => {
      const cur = state.sessions.get(key)
      if (!cur) return state
      const sessions = new Map(state.sessions)
      sessions.set(key, { ...cur, messages: cur.messages.filter((x) => x.id !== m.id) })
      return { sessions }
    })
    if (m.rowId !== undefined) {
      _dropped.delete(m.id)
      void serverDeleteMessage(key, m.rowId)
    }
    return
  }
}

// ── Plan-state localStorage persistence ──
//
// planState lives in memory (and on the agent), but a browser refresh would
// otherwise wipe the FD-side view of an in-flight plan. We mirror the slice
// to localStorage keyed by containerId so the PlanCard re-hydrates immediately
// on reload — the agent's own websocket stream then keeps it fresh.

interface PersistedPlanSlice {
  planState: PlanState | null
  planningEnabled: boolean
  planLevel: string
  planCardCollapsed: boolean
}

// Valid plan-mode enrichment levels (cumulative: each adds context on top of
// the previous). Mirrors PLAN_LEVELS in captain_claw/plan_mode.py.
export const PLAN_LEVELS = ['plain', 'enriched', 'insightful', 'complete'] as const
export type PlanLevel = (typeof PLAN_LEVELS)[number]

function _planLSKey(containerId: string): string {
  return `fd.plan.${_sliceId(containerId)}`
}

function savePlanSlice(containerId: string, slice: PersistedPlanSlice): void {
  try {
    if (!slice.planState && !slice.planningEnabled && (!slice.planLevel || slice.planLevel === 'plain')) {
      // Nothing worth persisting — drop any stale entry.
      window.localStorage.removeItem(_planLSKey(containerId))
      return
    }
    window.localStorage.setItem(_planLSKey(containerId), JSON.stringify(slice))
  } catch { /* quota / private mode — ignore */ }
}

function loadPlanSlice(containerId: string): PersistedPlanSlice | null {
  try {
    const raw = window.localStorage.getItem(_planLSKey(containerId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as PersistedPlanSlice
    return parsed
  } catch {
    return null
  }
}

function clearPlanSlice(containerId: string): void {
  try { window.localStorage.removeItem(_planLSKey(containerId)) } catch { /* ignore */ }
}

// ── Queue persistence ──
//
// The fullscreen-mode task queue survives a page refresh. We keep it
// per-container in localStorage so reopening FD picks up exactly where the
// user left off (pending items still pending, done items still ticked). The
// in-flight ("dispatched") status is downgraded to "pending" on hydrate —
// after a refresh we can't know if the agent ever finished it, so re-queue.

interface PersistedQueueSlice {
  queue: QueuedMessage[]
  queueAutoMode: boolean
  // Held by the user (Pause); nothing dispatches until they resume.
  queuePaused?: boolean
  // Auto was switched off because the agent was powered off, so the queue
  // can say why it stopped and offer to switch Auto back on.
  parked?: boolean
}

function _queueLSKey(containerId: string): string {
  return `fd.queue.${_sliceId(containerId)}`
}

function saveQueueSlice(containerId: string, slice: PersistedQueueSlice): void {
  try {
    if (slice.queue.length === 0 && !slice.queueAutoMode && !slice.queuePaused && !slice.parked) {
      window.localStorage.removeItem(_queueLSKey(containerId))
      return
    }
    window.localStorage.setItem(_queueLSKey(containerId), JSON.stringify(slice))
  } catch { /* quota / private mode — ignore */ }
}

/** A loaded slice, plus the task that was in flight and is now pending again
 *  (its turn may still be running on the agent: see SentFrame.resumed). */
type LoadedQueueSlice = PersistedQueueSlice & { resumedId?: string }

function loadQueueSlice(containerId: string): LoadedQueueSlice | null {
  try {
    const raw = window.localStorage.getItem(_queueLSKey(containerId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as PersistedQueueSlice
    if (!parsed || !Array.isArray(parsed.queue)) return null
    // Downgrade any in-flight items back to pending — after a refresh we
    // have no reliable signal that the agent's reply was received, so
    // re-dispatch is safer than abandoning the task. Message ids are
    // regenerated on load, so no item keeps its bubble.
    // Except a task the queue was HOLDING on (you stopped it, it errored, the
    // allowance refused it, the agent gave up on every try): nothing was
    // running, and re-sending it by itself would undo your Stop or step past
    // "nothing runs until you decide". It comes back still held, as the
    // lane's in-flight item (_openSession picks it up).
    let held = false
    let resumedId: string | undefined
    parsed.queue = parsed.queue.map((q) => {
      if (q.status !== 'dispatched') return { ...q, messageId: undefined }
      if (!held && (q.halted || q.stuck)) {
        held = true
        return { ...q, awaitingAnswer: undefined, awaitingSince: undefined, messageId: undefined }
      }
      resumedId ??= q.id
      return {
        ...q, status: 'pending', dispatchedAt: undefined, awaitingAnswer: undefined,
        awaitingSince: undefined, halted: undefined, repliedAt: undefined, messageId: undefined,
      }
    })
    return { ...parsed, resumedId }
  } catch {
    return null
  }
}

// A shared agent is used alike by every member of it, and several members may
// use this browser — unlike a queue on an agent only you own. So a shared
// chat's slices are kept per deck user (`fd.queue.shared:<ref>[::lane]@<uid>`,
// same for plan): a session that ends with no teardown here (expired refresh
// cookie, an admin reset) can't hand the next person the last member's queue,
// nor — with auto-mode on — send it into their own private chat as theirs.
function _sliceId(chatKey: string): string {
  return sharedSliceId(chatKey, useAuthStore.getState().user?.id)
}

// …and they still go when the session does.
function purgeSharedSlices(): void {
  try {
    const prefixes = [`fd.queue.${SHARED_PREFIX}`, `fd.plan.${SHARED_PREFIX}`]
    const doomed: string[] = []
    for (let i = 0; i < window.localStorage.length; i++) {
      const k = window.localStorage.key(i)
      if (k && prefixes.some((p) => k.startsWith(p))) doomed.push(k)
    }
    for (const k of doomed) window.localStorage.removeItem(k)
  } catch { /* storage blocked — nothing persisted to purge */ }
}

/** Drop this user's queue and plan slices of one shared agent, every lane —
 *  for a member leaving it (a later re-share starts clean). */
export function clearSharedSlices(agentRef: string): void {
  clearAgentSlices(sharedContainerId(agentRef))
}

/** Drop the queue and plan slices of one agent, every lane — for an agent
 *  gone for good (a later one of the same name starts clean). */
export function clearAgentSlices(agentId: string): void {
  for (const lane of LANES) {
    const key = laneKey(agentId, lane)
    try {
      window.localStorage.removeItem(_queueLSKey(key))
      window.localStorage.removeItem(_planLSKey(key))
    } catch { /* storage blocked — nothing persisted */ }
  }
}

registerSignOutTeardown(() => {
  purgeSharedSlices()
  // A shared socket can still deliver a reply (and re-save a slice) between
  // here and the unload — purge once more on the way out.
  try { window.addEventListener('pagehide', purgeSharedSlices, { once: true }) } catch { /* ignore */ }
})

// Debounced batch persist
const _msgQueue: Map<string, ChatMessage[]> = new Map()
let _msgTimer: ReturnType<typeof setTimeout> | null = null

function queueMessagePersist(sessionId: string, msg: ChatMessage) {
  if (!useAuthStore.getState().authEnabled) return
  const q = _msgQueue.get(sessionId) || []
  q.push(msg)
  _msgQueue.set(sessionId, q)
  if (_msgTimer) clearTimeout(_msgTimer)
  _msgTimer = setTimeout(_flushMessages, 500)
}

async function _flushMessages() {
  _msgTimer = null
  const batches = new Map(_msgQueue)
  _msgQueue.clear()
  for (const [sessionId, queued] of batches) {
    // Serialize the latest version of each row: a divider may have been
    // titled or renamed since it was queued. A row withdrawn meanwhile (a
    // "new topic:" echo) is not written at all.
    const live = useChatStore.getState().sessions.get(sessionId)?.messages
    const msgs = queued
      .filter((m) => { if (_dropped.has(m.id)) { _dropped.delete(m.id); return false } return true })
      .map((m) => live?.find((x) => x.id === m.id) ?? m)
    if (msgs.length === 0) continue
    try {
      const res = await fetch(`/fd/chat/sessions/${encodeURIComponent(sessionId)}/messages`, {
        method: 'POST',
        headers: _chatHeaders(),
        credentials: 'include',
        body: JSON.stringify({
          messages: msgs.map((m) => ({
            role: m.role,
            content: m.content,
            metadata: JSON.stringify({
              ...(m.tool_name ? { tool_name: m.tool_name } : {}),
              ...(m.tool_arguments ? { tool_arguments: m.tool_arguments } : {}),
              ...(m.tool_output ? { tool_output: m.tool_output } : {}),
              ...(m.model ? { model: m.model } : {}),
              ...(m.peer_name ? { peer_name: m.peer_name } : {}),
              ...(m.topic ? { topic: m.topic } : {}),
              ...(m.via ? { via: m.via } : {}),
              ...(m.sourceLabel ? { source_label: m.sourceLabel } : {}),
              ...(m.task ? { task: m.task } : {}),
              ...(m.automation ? { automation: m.automation } : {}),
            }),
          })),
        }),
      })
      if (!res.ok) continue
      // The log's row ids come back in order: stamp them on the messages so a
      // divider just inserted is jumpable and "earlier" paging knows the edge.
      const body = await res.json().catch(() => null) as { ids?: number[] } | null
      const ids = body?.ids
      if (!ids || ids.length !== msgs.length) continue
      // A row withdrawn while this batch was in flight is deleted now.
      msgs.forEach((m, i) => {
        if (_dropped.has(m.id)) {
          _dropped.delete(m.id)
          void serverDeleteMessage(sessionId, ids[i])
        }
      })
      const rowIds = new Map(msgs.map((m, i) => [m.id, ids[i]] as const))
      useChatStore.setState((state) => {
        const s = state.sessions.get(sessionId)
        if (!s) return state
        let changed = false
        const messages = s.messages.map((m) => {
          const rid = rowIds.get(m.id)
          if (rid === undefined || m.rowId === rid) return m
          changed = true
          return { ...m, rowId: rid }
        })
        if (!changed) return state
        const sessions = new Map(state.sessions)
        sessions.set(sessionId, { ...s, messages })
        return { sessions }
      })
    } catch { /* ignore */ }
  }
}

// Sign-out drains the debounce now, while the outgoing user's token is valid.
registerSignOutFlush(async () => {
  if (_msgTimer) clearTimeout(_msgTimer)
  await _flushMessages()
})

interface AgentModelInfo {
  id: string
  label: string
  selector: string
}

export interface NextStepOption {
  label: string
  action: string
  description?: string
}

// ── Queued message (fullscreen-mode task queue) ──
//
// The chat panel can run in a "fullscreen" mode that exposes a left-hand
// queue. Each item is dispatched to the agent one at a time. The user marks
// items done manually, or — in auto mode — FD marks the currently-dispatched
// item done as soon as the agent emits a non-replay assistant reply, which
// triggers the next pending item.
export interface QueuedMessage {
  id: string
  content: string
  status: 'pending' | 'dispatched' | 'done'
  createdAt: number
  dispatchedAt?: number
  completedAt?: number
  // Auto-mode is holding this item because the agent asked something. The row
  // shows it, and the hold expires (see armQuestionWatch).
  awaitingAnswer?: boolean
  // When the hold above began (ms): the queue counts down from it.
  awaitingSince?: number
  // Re-runs after the agent gave up (1 = the first re-run, so the try in
  // flight is attempts + 1). Unset while the original dispatch runs.
  attempts?: number
  // The agent gave up as often as we're willing to retry, so the queue is
  // holding here rather than advancing past unfinished work.
  stuck?: boolean
  // What the run cost. Accumulated across re-runs, so a task that gave up
  // twice reports what it actually spent, not just the last attempt.
  usage?: TokenUsage
  toolCount?: number
  // How a finished item ended when it wasn't simply done: skipped by the
  // user, moved on from when a question went unanswered, or handed to the
  // turn that was running (Insert now) instead of waiting for its own.
  endedHow?: 'skipped' | 'no-answer' | 'inserted'
  // The queue holds on this item, and this says why: the turn errored, the
  // daily allowance refused it, or the user stopped it. `refused`: the agent
  // (or Flight Deck) never took the item's frame, so no reply that comes in
  // afterwards is the item's — none credits it or lifts the hold.
  halted?: { kind: 'error' | 'quota' | 'stopped'; message?: string; at: number; refused?: boolean }
  // A (non-replay) assistant reply arrived while this item was dispatched.
  repliedAt?: number
  // The user bubble its latest dispatch created ("Show in chat"). Session
  // lifetime only: message ids are regenerated on reload.
  messageId?: string
}

let _queueCounter = 0
function nextQueueId() { return `q-${Date.now()}-${++_queueCounter}` }

// ── Auto-mark watchers ──
//
// When the agent emits a non-replay assistant reply we DON'T mark the queue
// item done immediately — the agent may still be running post-turn work
// (next_steps emission, tool wind-down) and dispatching another message
// while it's still busy triggers the "Agent is busy processing another
// request" rejection.
//
// Instead, we arm a per-container watcher that fires when either:
//   - the `next_steps` event arrives (the explicit end-of-turn signal), OR
//   - a fallback timeout elapses (covers replies that don't emit next_steps).
//
// The watcher captures the dispatched queue-item id so a late-arriving event
// can't mark the wrong item done if the user has already moved on.
interface AutoCompleteWatch {
  itemId: string
  timer: ReturnType<typeof setTimeout>
}
const _autoCompleteWatches = new Map<string, AutoCompleteWatch>()
const AUTO_COMPLETE_FALLBACK_MS = 6000

function armAutoCompleteWatch(containerId: string, itemId: string) {
  cancelAutoCompleteWatch(containerId)
  const timer = setTimeout(() => {
    _autoCompleteWatches.delete(containerId)
    _fireAutoComplete(containerId, itemId)
  }, AUTO_COMPLETE_FALLBACK_MS)
  _autoCompleteWatches.set(containerId, { itemId, timer })
}

function fireAutoCompleteWatch(containerId: string) {
  const watch = _autoCompleteWatches.get(containerId)
  if (!watch) return
  clearTimeout(watch.timer)
  _autoCompleteWatches.delete(containerId)
  _fireAutoComplete(containerId, watch.itemId)
}

function cancelAutoCompleteWatch(containerId: string) {
  const watch = _autoCompleteWatches.get(containerId)
  if (!watch) return
  clearTimeout(watch.timer)
  _autoCompleteWatches.delete(containerId)
}

// ── Waiting on an answer (bounded) ──
//
// When a reply reads as a genuine clarifying question, auto-mode holds the
// item so the user can answer. It used to hold it FOREVER: no timer, no UI
// signal, just a spinner that never stopped and a queue that never moved.
//
// So the hold is now bounded and visible. The item is flagged
// `awaitingAnswer` (the row shows why it stopped), and if no answer arrives
// the queue moves on rather than stranding every item behind it. This timer
// is deliberately separate from the auto-complete watch — `next_steps` fires
// within seconds of a turn ending and would otherwise cut the wait short.
const _questionWatches = new Map<string, ReturnType<typeof setTimeout>>()
const QUESTION_HOLD_MS = 180000   // 3 min
/** How long the queue waits on an unanswered question (the panel counts it down). */
export const QUEUE_QUESTION_HOLD_MS = QUESTION_HOLD_MS

function armQuestionWatch(containerId: string, itemId: string) {
  cancelQuestionWatch(containerId)
  _setAwaitingAnswer(containerId, itemId, true)
  _questionWatches.set(containerId, setTimeout(() => {
    _questionWatches.delete(containerId)
    const state = useChatStore.getState()
    const session = state.sessions.get(containerId)
    if (!session || !session.queueAutoMode) return
    if (session.queueDispatchedId !== itemId) return
    if (session.busy) return          // the user answered and it's working
    state.addLocalNote(containerId,
      'Queue moved on: the agent ended with a question and no answer arrived in 3 minutes.')
    state.markQueueItemDone(containerId, itemId, { how: 'no-answer' })
  }, QUESTION_HOLD_MS))
}

function cancelQuestionWatch(containerId: string) {
  const timer = _questionWatches.get(containerId)
  if (timer) {
    clearTimeout(timer)
    _questionWatches.delete(containerId)
  }
  const session = useChatStore.getState().sessions.get(containerId)
  if (session?.queue.some((q) => q.awaitingAnswer)) {
    updateSession(containerId, {
      queue: session.queue.map((q) => (q.awaitingAnswer
        ? { ...q, awaitingAnswer: false, awaitingSince: undefined } : q)),
    })
  }
}

function _setAwaitingAnswer(containerId: string, itemId: string, value: boolean) {
  const session = useChatStore.getState().sessions.get(containerId)
  if (!session) return
  updateSession(containerId, {
    queue: session.queue.map((q) => (q.id === itemId
      ? { ...q, awaitingAnswer: value, awaitingSince: value ? Date.now() : undefined } : q)),
  })
}


/** Add a turn's cost to the item that is currently dispatched. */
function creditDispatchedItem(
  key: string,
  patch: (prev: QueuedMessage) => Partial<QueuedMessage>,
) {
  const session = useChatStore.getState().sessions.get(key)
  const id = session?.queueDispatchedId
  if (!session || !id) return
  updateSession(key, {
    queue: session.queue.map((q) => (q.id === id ? { ...q, ...patch(q) } : q)),
  })
}

// ── Whose turn is it? ──
//
// One lane carries turns from several senders: the queue, what you type, other
// surfaces (WhatsApp, another tab). The agent runs one turn at a time and
// refuses a frame that arrives meanwhile ("busy", `retryable`); Flight Deck
// refuses one over the day's allowance. So each frame this tab sends is noted
// until the agent takes it (the turn's opening "thinking" echoes its
// client_msg_id) or refuses it (a busy refusal echoes the id too). That tells
// which item a refusal belongs to, and which item a reply is for.
interface SentFrame {
  /** client_msg_id ('' when the socket gave none). */
  id: string
  /** The queued item it was sent for; null when you typed it. */
  itemId: string | null
  /** A forced start (Run now, Try again): it may run on a paused lane. */
  forced: boolean
  /** A slash command: a command_result answers it, not a turn. */
  cmd: boolean
  /** A turn was already running when it went out. */
  wasBusy: boolean
  /** The first send of a task that was in flight when the page loaded (a
   *  reload, the chat opened again): that task's run may still be going. */
  resumed: boolean
  at: number
}
/** Frames sent and not yet taken or refused, oldest first. */
const _sent = new Map<string, SentFrame[]>()
/** The last frame sent on each lane, taken or not (for errors that name none). */
const _lastFrame = new Map<string, SentFrame>()
/** When the turn running on each lane began, as far as we know: when its
 *  frame went out (one of ours), or when the agent said it started. */
const _turns = new Map<string, number>()
/** Per lane, the waiting tasks handed to the running turn (Insert now), in
 *  the order sent, until the agent's /btw command_result answers each: taken
 *  (`busy`), or refused because that turn had just ended — then the task
 *  goes back to the front of the queue and runs on its own. */
const _inserts = new Map<string, QueuedMessage[]>()
/** When each lane last went idle by the agent's word (its "ready"). */
const _idleAt = new Map<string, number>()
/** The task each lane had in flight when the page loaded, until the lane's
 *  first connection sends it again (loadQueueSlice turned it back to waiting). */
const _resumed = new Map<string, string>()
/** Lanes whose agent echoes our frame ids (it said so once). */
const _echoes = new Set<string>()
// An answer comes back in moments: a frame unanswered this long was taken.
const FRAME_TTL_MS = 60000
const MAX_SENT = 20

function noteSent(key: string, frame: SentFrame) {
  const list = (_sent.get(key) ?? []).filter((f) => frame.at - f.at < FRAME_TTL_MS)
  list.push(frame)
  _sent.set(key, list.slice(-MAX_SENT))
  _lastFrame.set(key, frame)
}

/** Take a frame off the unanswered list: the one with this id (null when it
 *  isn't one of ours), or, with no id to go by, the oldest or the newest. */
function takeFrame(key: string, id: string, pick: 'oldest' | 'newest' = 'oldest'): SentFrame | null {
  const list = _sent.get(key)
  if (!list?.length) return null
  const i = id ? list.findIndex((f) => f.id === id) : pick === 'oldest' ? 0 : list.length - 1
  return i < 0 ? null : list.splice(i, 1)[0]
}

/** A command_result answered the oldest command still unanswered. */
function takeCommandFrame(key: string) {
  const list = _sent.get(key)
  const i = list ? list.findIndex((f) => f.cmd) : -1
  if (list && i >= 0) list.splice(i, 1)
}

/** A turn began (status "thinking"). Its id names the frame it took. Without
 *  one (older agents, shared chats) it took the oldest frame still unanswered,
 *  unless a turn is already running (then it is a later "thinking" of that one).
 *  An agent that echoes ids sends a "thinking" without one only mid-turn or for
 *  another surface's turn: that takes none of our frames. */
function noteTurnStart(key: string, id: string) {
  if (!id && _turns.has(key)) return
  if (id) _echoes.add(key)
  const frame = !id && _echoes.has(key) ? null : takeFrame(key, id)
  _turns.set(key, frame?.at ?? Date.now())
  // The agent says a turn runs: its end frees the lane, not our fallback.
  releaseBusyHold(key)
  // The agent took a message you typed here: it resumes a task the queue is
  // holding on (stopped, errored), since you are dealing with it in the chat.
  // Only now, not when you sent it: a message refused while a Stop was still
  // landing never ran, and the stopped turn's reply is no answer to it.
  if (frame && frame.itemId === null && !frame.cmd) {
    const s = useChatStore.getState().sessions.get(key)
    if (s?.queue.find((q) => q.id === s.queueDispatchedId)?.halted) {
      creditDispatchedItem(key, () => ({ halted: undefined }))
    }
  }
}

/** The agent said the lane is free ("ready"): the turn is over. */
function noteTurnEnd(key: string) {
  _turns.delete(key)
  _idleAt.set(key, Date.now())
  _holdRefusals.delete(key)
}

type SendOpts = { fromQueue?: boolean; forced?: boolean }

// A message written while the agent sleeps: Flight Deck wakes it, and the
// frame goes once its welcome arrives (one sent before would be forgotten
// with the old socket's frames — forgetFrames).
const _sendOnWake = new Map<string, { content: string; opts?: SendOpts; wasBusy: boolean }[]>()

/** The frame for a message (its bubble is the caller's), tracked as sent. */
function sendFrame(key: string, content: string, opts: SendOpts | undefined, wasBusy: boolean) {
  const session = useChatStore.getState().sessions.get(key)
  if (!session) return
  // Attached pictures go as image_paths; the bubble keeps their refs.
  const frame = splitImageRefs(content)
  // The chat box's reasoning-effort pick for this agent, fitted to what its
  // model offers (an agent that sent no levels gets nothing).
  const pick = session.reasoning && !session.shared ? effortFor(session.containerId) : 'auto'
  const reasoningEffort = pick !== 'auto' && session.reasoning ? clampLevel(pick, session.reasoning) : undefined
  const clientMsgId = session.ws.send(frame.text, { noNextSteps: !!opts?.fromQueue, imagePaths: frame.imagePaths, reasoningEffort })
  // Whose frame this is (the queue's, for the item in flight, or yours),
  // for the agent's answer to it: taken, or refused.
  const itemId = opts?.fromQueue ? session.queueDispatchedId ?? null : null
  const resumed = !!itemId && _resumed.get(key) === itemId
  if (resumed) _resumed.delete(key)
  noteSent(key, {
    id: clientMsgId ?? '',
    itemId,
    forced: !!opts?.forced,
    cmd: isSlashCommand(content),
    wasBusy,
    resumed,
    at: Date.now(),
  })
}

function forgetFrames(key: string) {
  _sent.delete(key)
  _turns.delete(key)
  _idleAt.delete(key)
  _holdRefusals.delete(key)
  _refusalReadies.delete(key)
}

/** The queued item the lane's running turn counts for: the one in flight,
 *  unless the agent never took its frame, or the turn began before it went
 *  out (the turn you stopped, a message you sent earlier). A turn that began
 *  after it does count, typed ones too: that is you working on it in the chat. */
function turnItemId(key: string): string | null {
  const s = useChatStore.getState().sessions.get(key)
  const d = s?.queueDispatchedId ? s.queue.find((q) => q.id === s.queueDispatchedId) : undefined
  if (!d || d.halted?.refused) return null
  const startedAt = _turns.get(key)
  if (startedAt !== undefined && d.dispatchedAt && startedAt < d.dispatchedAt) return null
  return d.id
}

/** creditDispatchedItem for what a turn did (a reply, tools, tokens): only
 *  when the turn is the item's. */
function creditTurnItem(key: string, patch: (prev: QueuedMessage) => Partial<QueuedMessage>) {
  if (turnItemId(key)) creditDispatchedItem(key, patch)
}

/** The agent refused the in-flight item's frame as busy, so the task never
 *  ran: it goes back to waiting, first in line, and starts once the agent is
 *  free (on a paused lane only when you had forced it). */
function requeueRefused(key: string, frame: SentFrame) {
  cancelAutoCompleteWatch(key)
  cancelCommandWatch(key)
  cancelQuestionWatch(key)
  resetStallNudges(key)
  const s = useChatStore.getState().sessions.get(key)
  const d = s?.queue.find((q) => q.id === s.queueDispatchedId)
  if (!s || !d) return
  const back: QueuedMessage = {
    ...d, status: 'pending', dispatchedAt: undefined, halted: undefined, awaitingAnswer: undefined,
    awaitingSince: undefined, repliedAt: undefined, messageId: undefined,
  }
  const rest = s.queue.filter((q) => q.id !== d.id)
  const firstPending = rest.findIndex((q) => q.status === 'pending')
  const at = firstPending >= 0 ? firstPending : Math.min(s.queue.indexOf(d), rest.length)
  updateSession(key, {
    queue: [...rest.slice(0, at), back, ...rest.slice(at)],
    queueDispatchedId: null,
    queueStartWhenFree: !s.queuePaused || frame.forced,
  })
}

// A task the agent keeps refusing as busy (another surface keeps it busy, or
// something answers our frames oddly) is not sent again and again: every try
// counts against today's allowance. The third refusal in a minute holds it.
const _taskRefusals = new Map<string, { itemId: string; at: number[] }>()
const MAX_TASK_REFUSALS = 3
const TASK_REFUSALS_MS = 60000

function refusedTooOften(key: string, itemId: string): boolean {
  const now = Date.now()
  const prev = _taskRefusals.get(key)
  const at = (prev?.itemId === itemId ? prev.at : []).filter((t) => now - t < TASK_REFUSALS_MS)
  at.push(now)
  _taskRefusals.set(key, { itemId, at })
  return at.length >= MAX_TASK_REFUSALS
}

// Shared chats: Flight Deck and the member's agent end EVERY chat frame with
// one "ready", a refused frame too (right after the refusal). While a turn
// still runs, that "ready" ends the refused frame, not the turn, so the status
// handler lets it pass. It follows within moments, so the wait is short.
const _refusalReadies = new Map<string, { n: number; until: number }>()
const REFUSAL_READY_MS = 5000

function expectRefusalReady(key: string) {
  const now = Date.now()
  const prev = _refusalReadies.get(key)
  _refusalReadies.set(key, { n: (prev && prev.until > now ? prev.n : 0) + 1, until: now + REFUSAL_READY_MS })
}

/** This "ready" is the one a refused frame was owed (not the turn's end). */
function takeRefusalReady(key: string): boolean {
  const owed = _refusalReadies.get(key)
  if (!owed) return false
  if (owed.until <= Date.now()) {
    _refusalReadies.delete(key)
    return false
  }
  owed.n -= 1
  if (owed.n <= 0) _refusalReadies.delete(key)
  return true
}

/** What an error frame means for the lane: which frame it refused (or whose
 *  turn failed), what that does to the queue, and whether the agent is free. */
function settleError(key: string, data: Record<string, unknown>) {
  const s = useChatStore.getState().sessions.get(key)
  if (!s) return
  const quota = data.code === 'quota_exceeded'
  if (data.client_msg_id) _echoes.add(key)
  // "Busy": the agent is on another turn and never took the frame. The agent
  // says so with `retryable`; in a shared chat Flight Deck and the member's
  // own agent say it with code "busy". It is never the running turn failing.
  const busy = data.retryable === true || data.code === 'busy'
  const id = String(data.client_msg_id || '')
  const message = String(data.message || '')
  const turnKnown = _turns.has(key)
  const d = s.queueDispatchedId ? s.queue.find((q) => q.id === s.queueDispatchedId) : undefined
  // Naming no frame while a turn we know of runs: that turn failed (its
  // "Error: …"). The queue holds on the item it counts for.
  if (!id && !quota && !busy && turnKnown) {
    if (turnItemId(key)) {
      cancelAutoCompleteWatch(key)
      cancelQuestionWatch(key)
      creditDispatchedItem(key, () => ({ halted: { kind: 'error', message, at: Date.now() } }))
    }
    updateSession(key, { busy: false })
    return
  }
  // Otherwise it refuses a frame of ours. A busy refusal names it; Flight
  // Deck's allowance refusal (and older agents, and shared chats) don't, so it
  // is the newest one still unanswered — or, with none and a turn still on,
  // the last one sent (an error after the turn ended is about none of them).
  let frame = id ? takeFrame(key, id) : takeFrame(key, '', 'newest')
  if (!frame && !id && !busy && s.busy) frame = _lastFrame.get(key) ?? null
  const mine = !!d && !!frame && frame.itemId === d.id
  // In a shared chat a refused chat frame is owed its own "ready" next.
  const owesReady = !!s.shared && !!frame
  if (busy) {
    // A shared chat also says "busy" to a note (`btw`) sent too soon. No chat
    // frame of ours is waiting for an answer then, and nothing changes.
    if (!frame && data.retryable !== true) return
    // The turn that refused it ended since (its "ready" came after the frame
    // went out, and no turn began after): the agent is free now.
    const ended = !!frame && !turnKnown && (_idleAt.get(key) ?? 0) > frame.at
    // Sent again as the page loaded, and refused while a turn we never saw
    // start runs: most likely this task's own run from before the reload. It
    // stays in flight, and that run's reply counts for it (sending it again
    // later would run it twice).
    const ownRun = mine && !!frame?.resumed && !turnKnown && !ended
    // Otherwise the queue's own task never ran, so it waits for the agent
    // rather than holding as an error (a task you stopped stays stopped, one
    // refused again and again holds). A typed message is yours to send again.
    if (mine && frame && !ownRun) {
      if (d.halted?.kind === 'stopped') {
        creditDispatchedItem(key, (prev) => ({ halted: prev.halted && { ...prev.halted, refused: true } }))
      } else if (refusedTooOften(key, d.id)) {
        cancelAutoCompleteWatch(key)
        cancelQuestionWatch(key)
        creditDispatchedItem(key, () => ({ halted: { kind: 'error', message, at: Date.now(), refused: true } }))
      } else {
        requeueRefused(key, frame)
      }
    }
    if (ended) {
      scheduleQueueKick(key)
      return
    }
    // Either way the agent is busy, whatever we thought: the lane waits for
    // that turn to end. Its reply and "ready" reach this socket too.
    heardFrom(key)
    _holdRefusals.set(key, (_holdRefusals.get(key) ?? 0) + 1)
    if (!s.busy) {
      updateSession(key, {
        busy: true, statusText: ownRun ? 'Thinking...' : 'Busy with another request…', _busyStartedAt: Date.now(),
      })
    }
    if (owesReady) expectRefusalReady(key)
    // Only when we never saw that turn start (or a Stop is still landing) is
    // there a fallback, and each refusal makes its quiet stretch longer.
    if (!turnKnown || _busyHolds.has(key)) holdBusy(key)
    return
  }
  // Refused outright (over the allowance, or the agent couldn't open the
  // turn): the queue's own task holds here, and no reply that comes in later
  // is its. A typed message refused is not the task failing.
  if (mine) {
    cancelAutoCompleteWatch(key)
    cancelQuestionWatch(key)
    creditDispatchedItem(key, () => ({
      halted: { kind: quota ? 'quota' : 'error', message, at: Date.now(), refused: true },
    }))
  }
  // Sent on top of a running turn (a frame sent after Stop is too, the lane
  // stays busy until the stopped turn ends): that turn still runs, so the
  // agent isn't free.
  if (frame ? frame.wasBusy : turnKnown || s.stopping) {
    if (owesReady) expectRefusalReady(key)
    return
  }
  updateSession(key, { busy: false })
}

// ── Busy on our word ──
//
// Twice the lane counts as busy without the agent saying so: after Stop (the
// agent only notices it at its next step, and refuses a frame sent before
// then), and after a busy refusal of a turn we never saw start. That turn's
// reply or "ready" frees the lane; this is the fallback when neither comes.
// While the agent shows signs of work (statuses, tool output, its thinking
// and streamed text) it keeps waiting; after a quiet stretch the lane counts
// as free again. A frame sent then may be refused once more: each busy
// refusal makes the next quiet stretch four times longer, since a frame sent
// into a busy agent is only refused again and still counts against today's
// allowance.
const BUSY_HOLD_MS = 30000
const BUSY_HOLD_MAX_MS = 30 * 60000
const _busyHolds = new Map<string, ReturnType<typeof setTimeout>>()
/** When each lane last showed a sign of a running turn. */
const _heardAt = new Map<string, number>()
/** Busy refusals since the agent last said the lane is free. */
const _holdRefusals = new Map<string, number>()

function heardFrom(key: string) {
  _heardAt.set(key, Date.now())
}

/** How long the lane must stay quiet before it counts as free on our word. */
function holdWindow(key: string): number {
  return Math.min(BUSY_HOLD_MS * 4 ** (_holdRefusals.get(key) ?? 0), BUSY_HOLD_MAX_MS)
}

function holdBusy(key: string, ms = holdWindow(key)) {
  releaseBusyHold(key)
  _busyHolds.set(key, setTimeout(() => {
    _busyHolds.delete(key)
    const s = useChatStore.getState().sessions.get(key)
    if (!s?.busy) return
    const need = holdWindow(key)
    const quiet = Date.now() - (_heardAt.get(key) ?? 0)
    if (quiet < need) {
      holdBusy(key, need - quiet)
      return
    }
    _turns.delete(key)
    updateSession(key, { busy: false, statusText: '' })
  }, ms))
}

function releaseBusyHold(key: string) {
  const t = _busyHolds.get(key)
  if (!t) return
  clearTimeout(t)
  _busyHolds.delete(key)
}

/** A task the allowance refused, now that the allowance reads as back: it
 *  goes again. Callers decide when that's wanted (an Auto lane that isn't
 *  paused, or you carrying on). */
function retryQuotaHold(key: string) {
  const s = useChatStore.getState().sessions.get(key)
  const d = s?.queueDispatchedId ? s.queue.find((q) => q.id === s.queueDispatchedId) : undefined
  if (!d || d.halted?.kind !== 'quota') return
  const q = useQuotaStore.getState().status
  if (q?.blocked && !q.exempt) return
  useChatStore.getState().retryQueueItem(key, d.id)
}

// ── Give-up detection ──
//
// The agent runtime has canned replies for when it cannot proceed — "I got
// stuck and couldn't make further progress…", retries/budget exhausted, empty
// response. A queued item that ends this way did NOT get done, and completing
// it would drop real work on the floor while the queue marched on.
//
// The marker strings are NOT duplicated here: they come from the same endpoint
// the Council UI uses, whose single source is captain_claw/agent_stuck.py.
// Until they load, detection is simply off — never guessed at.
let _stuckMarkers: string[] = []

async function loadStuckMarkers(): Promise<void> {
  if (_stuckMarkers.length) return
  try {
    const res = await fetch('/fd/council/stuck-markers', {
      headers: _chatHeaders(), credentials: 'include',
    })
    if (!res.ok) return
    const data = await res.json()
    if (Array.isArray(data.markers)) {
      _stuckMarkers = data.markers.map((m: string) => String(m).toLowerCase())
    }
  } catch { /* leave detection off rather than guess at the strings */ }
}

function isStuckReply(text: string): boolean {
  if (_stuckMarkers.length === 0) return false
  const t = (text || '').toLowerCase()
  return _stuckMarkers.some((m) => t.includes(m))
}

// Re-runs of a stuck item, keyed by container. Capped: a task the agent
// genuinely cannot do would otherwise retry forever, burning tokens on every
// attempt. On the last failure the queue STOPS rather than skipping ahead —
// the item is real work, and silently advancing past it loses it.
const _stuckRetries = new Map<string, { itemId: string; count: number }>()
const MAX_STUCK_RETRIES = 3
/** Every try a task gets: the original dispatch plus the re-runs. */
export const QUEUE_MAX_TRIES = MAX_STUCK_RETRIES + 1

// Escalation. The first re-run sends the task VERBATIM: give-ups are often
// transient (a failed fetch, one bad tool call), and the plain task is the
// cleanest thing to retry. Only when that fails too do we add a nudge — and
// which nudge depends on how it gave up. Running out of iterations is not the
// same failure as getting stuck, and telling a budget-exhausted agent to
// "work through the obstacle" aims at the wrong problem.
const NUDGE_BUDGET =
  '\n\nNOTE: your previous attempt ran out of its iteration budget. Do NOT start over — ' +
  'keep what you already produced, work in smaller steps, and finish only the part that ' +
  'is still missing. Save partial results as you go so nothing is lost if you run out again.'
const NUDGE_STUCK =
  '\n\nNOTE: your previous attempt ended by reporting that you got stuck. Do not give up ' +
  'and do not repeat that message. Pick up exactly where you left off and work through the ' +
  'obstacle step by step. If a tool call failed, try a different approach or reason it ' +
  'through directly. Deliver the actual result now.'

function nudgeFor(reply: string): string {
  return /budget exhausted/i.test(reply || '') ? NUDGE_BUDGET : NUDGE_STUCK
}

function resetStuckRetries(containerId: string) {
  _stuckRetries.delete(containerId)
}

/** Re-dispatch the stuck item: verbatim first, nudged after. False when spent. */
function retryStuckItem(containerId: string, itemId: string, reply: string): boolean {
  const state = useChatStore.getState()
  const session = state.sessions.get(containerId)
  if (!session) return false
  const item = session.queue.find((q) => q.id === itemId)
  if (!item) return false

  const entry = _stuckRetries.get(containerId)
  const used = entry && entry.itemId === itemId ? entry.count : 0
  if (used >= MAX_STUCK_RETRIES) {
    _stuckRetries.delete(containerId)
    updateSession(containerId, {
      queue: session.queue.map((q) =>
        q.id === itemId ? { ...q, attempts: used + 1, stuck: true } : q),
    })
    state.addLocalNote(containerId,
      `The agent gave up on this task ${used + 1} times. The queue is holding here — ` +
      'edit or remove the item to continue.')
    return false
  }

  _stuckRetries.set(containerId, { itemId, count: used + 1 })
  updateSession(containerId, {
    queue: session.queue.map((q) =>
      q.id === itemId ? { ...q, attempts: used + 1 } : q),
  })
  // First re-run: the task as written. After that: the task plus a nudge
  // aimed at the way it actually failed.
  const nudged = used > 0
  const content = nudged ? item.content + nudgeFor(reply) : item.content
  state.addLocalNote(containerId,
    `The agent gave up — re-running this task${nudged ? ' with a nudge' : ''} ` +
    `(attempt ${used + 2} of ${MAX_STUCK_RETRIES + 1}).`)
  // Small delay so the re-send lands after the turn's trailing events.
  setTimeout(() => {
    const s = useChatStore.getState().sessions.get(containerId)
    if (!s || s.queueDispatchedId !== itemId || s.busy) return
    // Stopped or errored meanwhile: the queue holds, it doesn't re-send.
    if (s.queue.find((q) => q.id === itemId)?.halted) return
    useChatStore.getState().sendMessage(containerId, content, { fromQueue: true })
  }, 800)
  return true
}

// ── Stall detection (client-side safety net) ──
//
// Some replies are "stalls" — the agent announces intent ("Let me research…")
// without actually doing anything in the same turn, or the visible content
// after sanitization is empty (the whole reply was a leaked context block).
//
// PRIMARY DEFENSE lives in Captain Claw: the agent orchestration loop
// detects stalls on a no-tool-calls turn and silently retries up to
// MAX_STALL_RETRIES times with a corrective prompt + tool_choice="required"
// forcing (see captain_claw/agent_orchestration_mixin.py:_looks_like_stall).
// By the time a stall reaches this client, the server has already burned
// its retry budget on it.
//
// This client-side path is the last-line safety net for the residual
// cases where the server's retries also stalled. We send a single
// "Continue the task." follow-up so the queue can move on, capped at
// MAX_STALL_NUDGES per item to avoid infinite loops.
const _stallNudges = new Map<string, { itemId: string; count: number }>()
const MAX_STALL_NUDGES = 1

function isLikelyStall(text: string): boolean {
  const t = (text || '').trim()
  if (!t) return true            // sanitizer stripped everything → empty
  if (t.length > 400) return false
  // Look at the first non-empty line only — long stalls are rare.
  const firstLine = (t.split(/\r?\n/).find((l) => l.trim()) || '').trim()
  if (!firstLine) return true
  const patterns = [
    /^let me\b/i,
    /^let'?s\s+/i,
    /^i'?ll\s+/i,
    /^i\s+will\s+(now\s+|then\s+|next\s+)?/i,
    /^i'?m\s+(going to|about to)\s+/i,
    /^i\s+am\s+(going to|about to)\s+/i,
    /^proceeding\s+(now|with|to)\b/i,
    /^starting\s+(now|the|with)\b/i,
    /^working on\b/i,
    /^one moment\b/i,
    /^one sec\b/i,
  ]
  // Treat as stall only if the WHOLE message is one short stall-shaped
  // line (no follow-up content). A reply like "Let me grab that. Here it
  // is: ..." with 2KB of substance is not a stall.
  if (t.length > 200) return false
  return patterns.some((p) => p.test(firstLine))
}

function resetStallNudges(containerId: string) {
  _stallNudges.delete(containerId)
}

function _fireAutoComplete(containerId: string, itemId: string) {
  const state = useChatStore.getState()
  const session = state.sessions.get(containerId)
  if (!session) return
  // Only fire if auto-mode is still on AND the dispatched item hasn't moved.
  if (!session.queueAutoMode) return
  if (session.queueDispatchedId !== itemId) return
  state.markQueueItemDone(containerId, itemId)
}

// ── Slash commands in the queue ──
//
// A slash command runs synchronously on the server and answers with a
// `command_result` — never an assistant reply, and never `next_steps`. The
// normal completion path (reply → arm watch → next_steps/fallback) therefore
// never fires, so the item stays `dispatched` forever: its row spinner keeps
// turning and `queueDispatchedId` blocks every later item.
//
// Tick it done as soon as the command answers. This runs in BOTH auto and
// manual mode on purpose — a command has no output for the user to judge, and
// leaving it in-flight strands the whole queue.
function isSlashCommand(text: string): boolean {
  return /^\//.test((text || '').trim())
}

/** A queued step that is a slash command (the planner's `/new`), not a task. */
export function isQueueCommand(text: string): boolean { return isSlashCommand(text) }

function completeDispatchedCommand(containerId: string) {
  const state = useChatStore.getState()
  const session = state.sessions.get(containerId)
  const dispatchedId = session?.queueDispatchedId
  if (!session || !dispatchedId) return
  const item = session.queue.find((q) => q.id === dispatchedId)
  if (!item || !isSlashCommand(item.content)) return
  cancelCommandWatch(containerId)
  const force = _forceThrough.delete(containerId)
  if (force) updateSession(containerId, { queueForceThrough: false })
  state.markQueueItemDone(containerId, dispatchedId)
  // A forced run's step is done: its task runs too, even on a paused lane.
  if (force) tryDispatchNext(containerId, { force: true })
}

// Safety net for the handful of commands that answer out-of-band instead of
// with a `command_result` (a deferred `/flow`, a dropped socket frame). Armed
// on dispatch, disarmed by the `command_result` handler.
const _commandWatches = new Map<string, ReturnType<typeof setTimeout>>()
const COMMAND_FALLBACK_MS = 20000

function armCommandWatch(containerId: string) {
  cancelCommandWatch(containerId)
  _commandWatches.set(containerId, setTimeout(() => {
    _commandWatches.delete(containerId)
    // Still busy → the command turned into real work; let the reply path own it.
    if (useChatStore.getState().sessions.get(containerId)?.busy) return
    completeDispatchedCommand(containerId)
  }, COMMAND_FALLBACK_MS))
}

function cancelCommandWatch(containerId: string) {
  const timer = _commandWatches.get(containerId)
  if (!timer) return
  clearTimeout(timer)
  _commandWatches.delete(containerId)
}

// Heuristic: did the agent's reply actually complete the task, or is it
// asking the user something? Auto-mode shouldn't tick a queue item done
// when the agent is mid-flight waiting for a clarification.
//
// "Looks complete" rules:
//  - Strip trailing whitespace and any trailing source-link / footer lines
//    (CC often appends a "Sources:" block at the bottom).
//  - If the last non-trivial line ends with "?" → NOT complete.
//  - If the body contains a confirmation-seeking phrase ("confirm with",
//    "should I proceed", "do you want me to", "would you like me to",
//    "let me know if", "please confirm", "reply with") AND the message is
//    short-ish → NOT complete. We require short-ish to avoid false negatives
//    on long reports that happen to mention "please confirm" in passing.
//  - Otherwise → complete.
function isLikelyTaskComplete(text: string): boolean {
  const trimmed = (text || '').trim()
  if (!trimmed) return false
  // Drop trailing markdown noise: blockquotes, source lists, separators.
  const lines = trimmed.split(/\r?\n/).map((l) => l.trimEnd())
  let lastIdx = lines.length - 1
  while (lastIdx >= 0) {
    const l = lines[lastIdx].trim()
    if (!l) { lastIdx--; continue }
    // Skip trailing source/footer lines so the "?" check sees real prose.
    if (/^(sources?|references?|links?|notes?)\s*:/i.test(l)) { lastIdx--; continue }
    if (/^[-*•]\s/.test(l) && lastIdx < lines.length - 1) break
    break
  }
  const lastLine = lastIdx >= 0 ? lines[lastIdx].trim() : ''
  // Strip trailing punctuation pairings like ?**, ?", ?_
  const lastChar = lastLine.replace(/[\s*_"'`)\]]+$/g, '').slice(-1)
  const lower = trimmed.toLowerCase()
  const SHORT_LIMIT = 1200
  // A trailing "?" blocks only on a SHORT reply — the same gate the
  // confirmation phrases below already use, and for the same reason. An
  // agent that did the work, printed a ten-row table of results, and then
  // offered "Did you mean a different range? Perhaps 910-919?" has finished
  // its task; the question is a courtesy, not a blocker. Treating that as a
  // clarification parked one user's queue indefinitely.
  if (lastChar === '?' && trimmed.length <= SHORT_LIMIT) return false
  // Confirmation-seeking phrases. Only treat as "waiting" if the message is
  // reasonably short — a 5KB report that mentions "let me know if" is fine.
  if (trimmed.length <= SHORT_LIMIT) {
    const patterns = [
      /\bconfirm with\b/,
      /\bplease confirm\b/,
      /\b(should|shall) i (proceed|continue|start|begin|run|do)\b/,
      /\bdo you want me to\b/,
      /\bwould you like me to\b/,
      /\bwant me to (proceed|continue|go ahead|run|do)\b/,
      /\breply with\b/,
      /\bsay ["“'`]/,
      /\b(or say|or reply)\b/,
    ]
    for (const p of patterns) {
      if (p.test(lower)) return false
    }
  }
  return true
}

interface AgentPersonalityInfo {
  id: string
  name: string
  description?: string
}

// ── Plan-mode state ──

export interface PlanStep {
  id: string
  title: string
  step_kind: string
  acceptance_criteria: string
  depends_on: string[]
  status: 'pending' | 'running' | 'completed' | 'verified' | 'failed' | 'revising'
  verificationPassed?: boolean
  verificationNotes?: string
  revisionCount: number
  // Live runtime fields (populated by orchestrator task_* events)
  startedAt?: number          // epoch ms
  endedAt?: number            // epoch ms
  currentPhase?: string       // 'thinking' | 'tool' | etc
  currentTool?: string
  currentText?: string        // latest task_step text
  tokensIn?: number
  tokensOut?: number
  error?: string
}

export interface PlanRevision {
  task_id: string
  revision_count: number
  rationale: string
  revised_description: string
  previous_verification_notes: string
}

export interface PlanState {
  startedAt: string
  startedAtMs: number
  status: 'running' | 'verified' | 'completed' | 'failed' | 'cancelled'
  maxRevisions: number
  steps: PlanStep[]
  revisions: PlanRevision[]
  activeStepId?: string       // currently running step (latest task_started)
  failedStep?: string
  verificationFailedStep?: string
  verificationNotes?: string
  errorMessage?: string
}

// ── Lanes ──
//
// A lane is a parallel context on ONE agent: its own transcript, queue and
// CC session, running at the same time as the others (docs/queue-lanes-plan.md).
//
// The store keys `sessions` by a LANE KEY rather than a container id — but
// lane A's key IS the container id, because lane A is the agent's existing
// context. That single choice is what keeps this change additive: every
// pre-lane call site, every persisted `fd.queue.<id>` entry, and every
// server-side chat row keeps working untouched, and only lanes B and C ever
// see a composite key.
export const LANE_MAIN = 'A'
// AUTO is the agent's automation lane: scheduler, autonomy, plan and peer
// turns run there instead of in lane A (results meant for you are mirrored
// into lane A as well).
export const LANE_AUTO = 'AUTO'
export const LANES = ['A', 'B', 'C', LANE_AUTO] as const
/** The lanes you hand work to (the queue planner): not the automation lane. */
export const WORK_LANES = LANES.filter((l) => l !== LANE_AUTO)

export function laneKey(containerId: string, lane: string = LANE_MAIN): string {
  return !lane || lane === LANE_MAIN ? containerId : `${containerId}::${lane}`
}

/** Split a lane key back into its parts. */
export function parseLaneKey(key: string): { containerId: string; lane: string } {
  const at = key.lastIndexOf('::')
  return at < 0
    ? { containerId: key, lane: LANE_MAIN }
    : { containerId: key.slice(0, at), lane: key.slice(at + 2) }
}

/** What a connected agent is told about its fleet: its peers, and itself —
 *  name, model and the owner's standing instructions. Sent when a chat
 *  connects, and again when the instructions change. */
function peerAgentsFrame(containerId: string): Record<string, unknown> {
  const containers = useContainerStore.getState().containers
  const localAgents = useLocalAgentStore.getState().agents
  const { getForwardingTask: getFwd, getConsultApproval } = useContainerStore.getState()
  const peers: { name: string; description: string; forwardingTask: string; host: string; port: number; auth: string; requireApproval: boolean }[] = []
  for (const c of containers) {
    if (c.id === containerId || c.status !== 'running' || !c.web_port) continue
    peers.push({
      name: c.agent_name || c.name,
      description: c.description || '',
      forwardingTask: getFwd(c.id),
      host: 'localhost',
      port: c.web_port,
      auth: c.web_auth || '',
      requireApproval: getConsultApproval(c.id),
    })
  }
  for (const a of localAgents) {
    if (a.id === containerId || a.status !== 'online') continue
    peers.push({
      name: a.name,
      description: a.description || '',
      forwardingTask: a.forwardingTask || '',
      host: a.host,
      port: a.port,
      auth: a.authToken || '',
      requireApproval: a.consultApproval ?? false,
    })
  }
  const processes = useProcessStore.getState().processes
  const { getForwardingTask: getProcFwd, getConsultApproval: getProcApproval } = useProcessStore.getState()
  for (const p of processes) {
    if (p.slug === containerId || p.status !== 'running' || !p.web_port) continue
    peers.push({
      name: p.name || p.slug,
      description: p.description || '',
      forwardingTask: getProcFwd(p.slug),
      host: 'localhost',
      port: p.web_port,
      auth: '',
      requireApproval: getProcApproval(p.slug),
    })
  }
  // Build self identity so the agent knows who it is in the fleet
  const { getFleetInstructions: getContainerFleetInst } = useContainerStore.getState()
  const { getFleetInstructions: getProcFleetInst } = useProcessStore.getState()
  let selfIdentity: { name: string; description: string; port: number; model?: string; provider?: string; fleet_instructions?: string } | null = null
  const selfContainer = containers.find((c) => c.id === containerId)
  if (selfContainer) {
    selfIdentity = {
      name: selfContainer.agent_name || selfContainer.name,
      description: selfContainer.description || '',
      port: selfContainer.web_port || 0,
      fleet_instructions: getContainerFleetInst(selfContainer.id),
    }
  }
  if (!selfIdentity) {
    const selfProcess = processes.find((p) => `proc-${p.slug}` === containerId || p.slug === containerId)
    if (selfProcess) {
      selfIdentity = {
        name: selfProcess.name || selfProcess.slug,
        description: selfProcess.description || '',
        port: selfProcess.web_port || 0,
        model: selfProcess.model || '',
        provider: selfProcess.provider || '',
        fleet_instructions: getProcFleetInst(selfProcess.slug),
      }
    }
  }
  if (!selfIdentity) {
    const selfLocal = localAgents.find((a) => a.id === containerId)
    if (selfLocal) {
      selfIdentity = {
        name: selfLocal.name,
        description: selfLocal.description || '',
        port: selfLocal.port || 0,
      }
    }
  }

  // Use internal FD URL if available (container mode), otherwise derive from browser
  const { internalFdUrl } = useAuthStore.getState()
  const fdUrl = internalFdUrl || `${window.location.protocol}//${window.location.host}`
  return { type: 'peer_agents', agents: peers, self: selfIdentity, fd_url: fdUrl }
}

interface ChatSession {
  /** The agent's container id — NOT the store key. Use `key` for store calls
   *  and `containerId` for anything that addresses the agent itself. */
  containerId: string
  /** This session's key in the `sessions` map: `containerId` on lane A,
   *  `containerId::LANE` elsewhere. */
  key: string
  lane: string
  /** This lane produced output while you were looking at another lane.
   *  Without this, parallel lanes are worse than serial ones: work finishes
   *  unseen. Cleared when the lane is selected. */
  unread?: boolean
  containerName: string
  host: string
  port: number
  auth: string
  ws: AgentChatWS
  messages: ChatMessage[]
  connected: boolean
  busy: boolean // agent is processing
  /** Runtime only: you pressed Stop and the agent hasn't ended that turn yet
   *  (it notices Stop at its next step). `busy` stays true meanwhile, so
   *  nothing is sent into the lane; the stopped turn's reply or "ready" ends
   *  both (or 30s without a sign of work). Never true while `busy` is false. */
  stopping: boolean
  statusText: string
  models: AgentModelInfo[]
  personalities: AgentPersonalityInfo[]
  activeModel: string
  /** The reasoning-effort levels the agent's model offers (from its session
   *  info); null when it has none to choose from, or for a shared agent. */
  reasoning: ReasoningProfile | null
  activePersonality: string
  // Token generation speed tracking
  _busyStartedAt: number // timestamp (ms) when agent became busy
  lastTokPerSec: number  // last wall-clock generation speed (tok/s)
  avgTokPerSec: number   // running average wall-clock tok/s
  _tokSamples: number    // number of samples for avg calculation
  llmTokPerSec: number   // real LLM generation speed (completion_tokens / llm_latency)
  // Live cumulative token usage for the in-flight turn (input/output/cache),
  // updated on each LLM call; null between turns.
  liveTurnUsage: TokenUsage | null
  // Planning mode
  planningEnabled: boolean   // optimistic mirror of agent.plan_mode_auto
  planLevel: string          // optimistic mirror of agent.plan_mode_level
  planState: PlanState | null
  planCardCollapsed: boolean
  // Suggested next steps emitted by the agent after a response.
  nextStepOptions: NextStepOption[]
  // Queue (fullscreen mode only — but state lives here so it survives
  // toggling fullscreen on/off without losing user input).
  queue: QueuedMessage[]
  queueAutoMode: boolean
  queueDispatchedId: string | null
  /** The user paused the queue: nothing new dispatches until they resume
   *  (the task in flight finishes). Persisted with the queue. */
  queuePaused: boolean
  /** Auto was switched off because the agent was powered off (parkQueues).
   *  Persisted; cleared when Auto goes back on or a task starts. */
  queueParked: boolean
  /** Runtime only: something the user did wanted a task to start, but the
   *  agent was busy, offline or out of allowance — start it once it can. */
  queueStartWhenFree: boolean
  /** Runtime only: a forced run (Run now, Try again) began with a step (the
   *  fresh start), so the task behind it follows even on a paused lane. */
  queueForceThrough?: boolean
  /** An agent another deck user shared with you (containerId `shared:<ref>`):
   *  reached through Flight Deck's member route, chat only. */
  shared?: { agentRef: string; ownerName: string }
  /** Shared chats: Flight Deck closed the socket for good (access removed,
   *  agent stopped…) — the chat shows why instead of reconnecting. */
  closed?: { code: number; reason: string } | null
  /** Shared chats: who the agent knows it is talking to (from `welcome`). */
  speaker?: { id: string; name: string; owner_name: string; lane: string } | null
  /** Why the agent can't be reached while the chat keeps reconnecting (it is
   *  restarting, or gone). Shown on the chat in place of an error per retry. */
  unreachable?: string | null
  /** Flight Deck put the agent to sleep for being idle and holds this socket
   *  for it: a message written now wakes it (`waking` until its welcome). */
  asleep?: boolean
  waking?: boolean
  /** The agent session this lane's context is on — from `welcome`,
   *  `session_info` and command results. Empty until the agent says. */
  agentSessionId: string
  /** Flight Deck's chat log for this lane is loaded (auth on). From then on
   *  the log IS the transcript: the agent's replay is folded into it, and
   *  /new, /session switch and /clear add topic dividers instead of wiping
   *  the screen. */
  historyLoaded: boolean
  /** The loaded window of the log starts after older rows ("Earlier
   *  messages" fetches the page before it). */
  hasOlder: boolean
  /** The window stops before newer rows of the log — the reader jumped to
   *  or paged back into the past ("Later messages" fetches the page after
   *  it). While set, the window is not the end of the transcript: nothing
   *  reconciles against it or merges the agent's replay into it. */
  hasNewer: boolean
  /** With `hasNewer`: the highest log row id of the window's contiguous
   *  block, where "Later messages" continues from. */
  windowEndRowId: number
  /** Until this time, a change in message count must not auto-scroll to the
   *  bottom: older rows were prepended, or the view jumped to a topic. */
  holdScrollUntil: number
}

/** What the internal opener needs; `shared` marks an agent shared with you. */
interface OpenSessionArgs {
  containerId: string
  containerName: string
  host: string
  port: number
  auth: string
  lane?: string
  shared?: { agentRef: string; ownerName: string }
}

interface ChatStore {
  sessions: Map<string, ChatSession>
  activeChatId: string | null
  /** agent container id → the lane currently shown for it. Absent means A. */
  activeLane: Record<string, string>
  chatOpen: boolean
  // Fullscreen mode: hides sidebar/director/main/tool panels and gives the
  // chat panel the full width. The queue panel is only available here.
  chatFullscreen: boolean

  /** Open (or focus) a chat. `lane` picks the parallel context — omit for A,
   *  which is the agent's main context and today's behaviour exactly. */
  openChat: (id: string, name: string, host: string, port: number, auth: string, lane?: string) => void
  /** Open (or focus) a chat with an agent shared with you. Only its agent_ref
   *  travels; Flight Deck connects to the agent itself. */
  openSharedChat: (agentRef: string, name: string, ownerName: string, lane?: string) => void
  /** Internal: the one opener behind openChat and openSharedChat. */
  _openSession: (args: OpenSessionArgs) => void
  /** Which lane the UI is showing for a given agent (agent id → lane). */
  setActiveLane: (containerId: string, lane: string) => void
  closeChat: () => void
  switchChat: (containerId: string) => void
  disconnectChat: (containerId: string) => void
  /** Tell every open chat with this agent who it is again — after its
   *  instructions (or name) changed, so they apply without a reconnect. */
  resendFleetIdentity: (agentId: string) => void
  /** `fromQueue` marks a turn the queue dispatched: no next-step suggestions
   *  (`forced`: a start you forced, Run now / Try again). Returns the user
   *  bubble's message id (undefined when nothing was sent). */
  sendMessage: (containerId: string, content: string, opts?: SendOpts) => string | undefined
  /** Wake the agent of a chat that is held asleep (see ChatSession.asleep). */
  wakeAgent: (key: string) => void
  sendBtw: (containerId: string, content: string) => void
  /** Start a new topic on this lane (the agent's /new): the agent's context
   *  starts fresh; the transcript keeps going, with a divider. */
  newSession: (containerId: string) => void
  /** Put the lane's context back on an earlier topic's agent session (the
   *  agent's /session switch); shown as a "Back on …" divider. */
  resumeTopic: (containerId: string, agentSessionId: string) => void
  /** Prepend the page of the chat log before the loaded window. */
  loadOlder: (containerId: string) => Promise<void>
  /** Append the page of the chat log after the loaded window (after a jump
   *  or paging back trimmed the newest rows). */
  loadNewer: (containerId: string) => Promise<void>
  /** Make the divider with this log row id part of the loaded window —
   *  fetching the log from it on when it isn't — and return that message's
   *  id to scroll to (null when it can't be found). */
  jumpToTopic: (containerId: string, rowId: number) => Promise<string | null>
  /** Name one topic divider (the user's own title; overrides a generated one). */
  renameTopic: (containerId: string, ref: { rowId?: number; messageId?: string }, title: string) => void
  /** Append a local system note to the chat (no agent turn) — e.g. "started flow X". */
  addLocalNote: (containerId: string, content: string) => void
  cancelTask: (containerId: string) => void
  setModel: (containerId: string, selector: string) => void
  setPersonality: (containerId: string, personalityId: string) => void
  respondToApproval: (containerId: string, requestId: string, approved: boolean) => void
  // Plan-mode actions
  setPlanningEnabled: (containerId: string, enabled: boolean) => void
  setPlanLevel: (containerId: string, level: PlanLevel) => void
  togglePlanCardCollapsed: (containerId: string) => void
  dismissPlan: (containerId: string) => void
  // Fullscreen + queue actions
  toggleChatFullscreen: () => void
  enqueueQueueMessage: (containerId: string, content: string) => void
  removeQueueItem: (containerId: string, id: string) => void
  /** `how` records a Skip, or moving on from an unanswered question. Skipping
   *  a waiting item never starts the queue. */
  markQueueItemDone: (containerId: string, id: string, opts?: { how?: 'skipped' | 'no-answer' | 'inserted' }) => void
  toggleQueueAutoMode: (containerId: string) => void
  /** Run the next pending item now, even while the queue is paused (once the
   *  agent is free, when it is busy). */
  runQueueNext: (key: string) => void
  setQueuePaused: (key: string, paused: boolean) => void
  setQueueAutoMode: (key: string, on: boolean) => void
  /** Put an item back to run next — with new content when given — and start it. */
  retryQueueItem: (key: string, id: string, content?: string) => void
  /** Replace a WAITING item's content. False when it already started (or is gone). */
  editQueueItem: (key: string, id: string, content: string) => boolean
  /** Move a waiting item before another waiting one (null: to the end). */
  moveQueueItem: (key: string, id: string, beforeId: string | null) => void
  /** Insert now: hand a waiting task to the turn that is running, as an
   *  extra instruction for its remaining steps, and retire it from the
   *  queue. False when there is no turn to hand it to (the caller runs it
   *  on its own instead). */
  insertQueueItemNow: (key: string, id: string) => boolean
  /** Undo: put saved copies back where they were. */
  restoreQueueItems: (key: string, entries: { item: QueuedMessage; index: number }[]) => void
  /** Restart the 3-minute wait on an unanswered question. */
  extendQueueAnswerWait: (key: string) => void
  /** Switch Auto off on every lane of an AGENT whose queue still has items —
   *  open in this tab or only persisted — so nothing re-runs by itself when
   *  its chat next connects. With `check`, only report whether any would be.
   *  Returns true when a queue was (or would be) parked. */
  parkQueues: (agentId: string, check?: boolean) => boolean
  clearQueue: (containerId: string) => void
  /** Drop the finished items, keeping pending and in-flight ones. */
  clearQueueFinished: (containerId: string) => void
}

let msgCounter = 0
function nextId() { return `msg-${Date.now()}-${++msgCounter}` }

export const useChatStore = create<ChatStore>((set, get) => ({
  sessions: new Map(),
  activeChatId: null,
  activeLane: {},
  chatOpen: false,
  chatFullscreen: false,

  openChat: (containerId, containerName, host, port, auth, lane = LANE_MAIN) =>
    get()._openSession({ containerId, containerName, host, port, auth, lane }),

  openSharedChat: (agentRef, name, ownerName, lane = LANE_MAIN) =>
    get()._openSession({
      containerId: sharedContainerId(agentRef), containerName: name,
      host: '', port: 0, auth: '', lane, shared: { agentRef, ownerName },
    }),

  _openSession: ({ containerId, containerName, host, port, auth, lane = LANE_MAIN, shared }) => {
    const key = laneKey(containerId, lane)
    const existing = get().sessions.get(key)
    if (existing) {
      // Already have a session, just activate it. A socket still waiting for
      // the agent's welcome is left to finish rather than restarted — unless
      // the agent is now somewhere else.
      const moved = !existing.connected && !existing.shared && port > 0
        && (port !== existing.port || host !== existing.host)
      if (moved) {
        existing.ws.retarget(host, port, auth)
        updateSession(key, { host, port, auth: auth || existing.auth })
      }
      if (!existing.connected && (moved || !existing.ws.connecting)) {
        // Clear old replay messages — agent will resend them on reconnect
        const kept = existing.messages.filter((m) => !m.replay)
        if (kept.length !== existing.messages.length) {
          updateSession(key, { messages: kept })
        }
        if (existing.closed) updateSession(key, { closed: null })
        existing.ws.connect()
      } else if (existing.asleep) {
        // Opened again while its agent sleeps: that is asking for it.
        get().wakeAgent(key)
      }
      if (existing.unread) updateSession(key, { unread: false })
      set({ activeChatId: key, chatOpen: true })
      return
    }

    const ws = new AgentChatWS(containerId, host, port, auth, lane,
      shared ? { sharedRef: shared.agentRef } : undefined)
    // Re-hydrate any persisted plan slice so the PlanCard reappears instantly
    // after a page refresh. Live ws events will keep updating it from here.
    // (Shared chats have no plan mode.)
    const persisted = shared ? null : loadPlanSlice(key)
    const persistedQueue = loadQueueSlice(key)
    // Auto-progress is a habit you set for an AGENT, not for one room of it.
    // A fresh lane has no persisted slice, so without this it silently opens
    // in MANUAL while lane A says AUTO — and every queued item on B and C
    // sits dispatched forever waiting for a tick that never comes.
    const sibling = get().sessions.get(containerId)
      ?? [...get().sessions.values()].find((s) => s.containerId === containerId)
    const session: ChatSession = {
      containerId,
      key,
      lane,
      containerName,
      host,
      port,
      auth,
      ws,
      messages: [],
      agentSessionId: '',
      historyLoaded: false,
      hasOlder: false,
      hasNewer: false,
      windowEndRowId: 0,
      holdScrollUntil: 0,
      connected: false,
      busy: false,
      stopping: false,
      statusText: '',
      models: [],
      personalities: [],
      activeModel: '',
      reasoning: null,
      activePersonality: '',
      _busyStartedAt: 0,
      lastTokPerSec: 0,
      avgTokPerSec: 0,
      _tokSamples: 0,
      llmTokPerSec: 0,
      liveTurnUsage: null,
      planningEnabled: persisted?.planningEnabled ?? false,
      planLevel: persisted?.planLevel ?? 'plain',
      planState: persisted?.planState ?? null,
      planCardCollapsed: persisted?.planCardCollapsed ?? false,
      nextStepOptions: [],
      queue: persistedQueue?.queue ?? [],
      queueAutoMode: persistedQueue?.queueAutoMode ?? sibling?.queueAutoMode ?? false,
      // Only a held task stays in flight across a reload (loadQueueSlice).
      queueDispatchedId: persistedQueue?.queue.find((q) => q.status === 'dispatched')?.id ?? null,
      // Pause is a hold on this one lane, so it isn't inherited from a sibling.
      queuePaused: persistedQueue?.queuePaused ?? false,
      queueParked: persistedQueue?.parked ?? false,
      queueStartWhenFree: false,
      ...(shared ? { shared: { ...shared }, closed: null, speaker: null } : {}),
    }

    const sessions = new Map(get().sessions)
    sessions.set(key, session)
    set({ sessions, activeChatId: key, chatOpen: true })
    // A task that was in flight goes again on the first connection, and its
    // run may still be going on the agent (see settleError).
    if (persistedQueue?.resumedId) _resumed.set(key, persistedQueue.resumedId)
    else _resumed.delete(key)

    // The agent session a replay that arrived before the log belongs to: the
    // one welcome named just before it on the same socket.
    let replaySid = ''

    // Server-side chat persistence: upsert session & load history
    if (useAuthStore.getState().authEnabled) {
      serverUpsertSession(key, containerId, containerName).then(() =>
        serverLoadMessages(key).then((history) => {
          // Unreadable log: keep whatever the agent replays, as before.
          if (history === null) return
          // The log is the transcript. A replay that arrived first is folded
          // into it (only what the log is missing gets appended, and
          // persisted); live rows that arrived meanwhile stay at the end.
          const pendingReplay: ChatMessage[] = []
          useChatStore.setState((state) => {
            const s = state.sessions.get(key)
            if (!s || s.historyLoaded) return state
            pendingReplay.push(...s.messages.filter((m) => m.replay))
            const live = s.messages.filter((m) => !m.replay)
            const updated = {
              ...s, messages: [...history, ...live],
              historyLoaded: true, hasOlder: history.length >= HISTORY_PAGE,
            }
            const newSessions = new Map(state.sessions)
            newSessions.set(key, updated)
            return { sessions: newSessions }
          })
          // The lane moved to another session while the log loaded (a /new, a
          // switch elsewhere, a reconnect): the replay is the earlier one's,
          // so its turns go first and the new session's divider after them.
          const sid = useChatStore.getState().sessions.get(key)?.agentSessionId ?? ''
          if (pendingReplay.length && replaySid && replaySid !== sid) {
            mergeReplay(key, pendingReplay, replaySid)
            reconcileTopic(key)
            return
          }
          // The replay is the agent's current session: when the log has no
          // divider for it yet, the divider goes in first and the session's
          // turns under it. (Merged first, they read as an earlier topic with
          // an empty new one below.)
          const replayTurns = pendingReplay.length ? replayTurnsToAdd(key, pendingReplay) : []
          reconcileTopic(key)
          for (const m of replayTurns) addMessage(key, { ...m, replay: false })
        })
      )
    }

    // Give-up markers come from the backend (single source: agent_stuck.py);
    // fetch them now so a stuck reply is recognised on the very first turn.
    void loadStuckMarkers()

    // Wire up event handlers
    ws.on('_connected', () => {
      // A fresh socket: nothing sent before it will be answered on it.
      forgetFrames(key)
      updateSession(key, shared ? { connected: true, closed: null }
        : { connected: true, unreachable: null, asleep: false, waking: false })
      // What was written while the agent slept goes now that it is awake.
      const wakeSends = _sendOnWake.get(key)
      _sendOnWake.delete(key)
      for (const w of wakeSends ?? []) sendFrame(key, w.content, w.opts, w.wasBusy)
      // If we hydrated a queue with pending items and auto-mode is on, kick
      // off the first dispatch now that the socket is live — and the same
      // for a task the user asked to start while the agent was away.
      // (A start the user asked for while paused is a Run now: it still runs.)
      const cur = useChatStore.getState().sessions.get(key)
      if (cur && !cur.queueDispatchedId && (cur.queueAutoMode || cur.queueStartWhenFree)) {
        tryDispatchNext(key, { force: cur.queueStartWhenFree })
      }
      // A task the allowance refused (held across a reload, or the reset came
      // while the socket was down): an Auto lane runs it again once the
      // allowance reads as back. Unknown yet: the quota subscriber does it.
      if (cur?.queueDispatchedId && cur.queueAutoMode && !cur.queuePaused && useQuotaStore.getState().status) {
        retryQuotaHold(key)
      }
      // Only the send just above can be a copy of a run still going from
      // before the page loaded; a later start is a start.
      _resumed.delete(key)
    })

    ws.on('_disconnected', () => {
      // Frames still unanswered won't be now, and the turn we knew of is out
      // of sight.
      forgetFrames(key)
      updateSession(key, { connected: false, busy: false, statusText: '', asleep: false, waking: false })
      // A dropped socket means we won't see the next_steps event. Cancel
      // any pending auto-mark — when the agent reconnects, replay or fresh
      // events will drive the queue forward again.
      cancelAutoCompleteWatch(key)
    })

    // Flight Deck's proxy is holding the socket until the agent answers (it is
    // restarting). The chat says so once; it reconnects by itself.
    ws.on('fd_agent_waiting', () => {
      updateSession(key, { unreachable: 'The agent is not answering yet' })
    })

    // The agent is asleep (stopped for being idle) and the proxy holds the
    // socket for it: opened by the user, the chat already asked it to wake
    // (`waking`); reconnected by itself, it waits for a message or Wake up.
    ws.on('fd_agent_asleep', (data) => {
      // Work waiting for it wakes it too: a message written meanwhile, or a
      // queue set to run (not paused or parked, and today's allowance left).
      const cur = useChatStore.getState().sessions.get(key)
      const quota = useQuotaStore.getState().status
      const queued = !!cur && !cur.queuePaused && !cur.queueParked && (cur.queueAutoMode || cur.queueStartWhenFree)
        && cur.queue.some((q) => q.status === 'pending') && !(quota?.blocked && !quota.exempt)
      const waking = !!data.waking || _sendOnWake.has(key) || queued
      if (waking && !data.waking) ws.wake()
      updateSession(key, { asleep: !waking, waking, unreachable: null })
    })

    // The agent came back on another port (it restarted and its old one was
    // taken) — the socket already follows it; the session keeps up.
    ws.on('fd_agent_endpoint', (data) => {
      const p = Number(data.port)
      if (p > 0) updateSession(key, { port: p })
      // The agent list (files, datastore, cards) still has the old port until
      // its next poll — fetch it now.
      void useProcessStore.getState().fetchProcesses()
    })

    // Shared chats: Flight Deck closed the member socket for good. Say why
    // (the chat shows it) instead of reconnecting.
    ws.on('_closed', (data) => {
      updateSession(key, {
        connected: false, busy: false, statusText: '',
        closed: { code: Number(data.code) || 0, reason: String(data.reason || '') },
      })
    })

    // The agent's session changed under this lane — a /new or /session
    // switch typed on another channel, a "new topic" cue — or just a
    // rename/model change on the same one, or a turn's end (then nothing to
    // mark).
    ws.on('session_info', (data) => {
      const id = String(data.id ?? '')
      if (!id) return
      // A model change arrives here too: the picker follows the new model.
      if (!shared && 'reasoning' in data) updateSession(key, { reasoning: parseProfile(data.reasoning) })
      // A transcript with no divider at all on the session the lane already
      // knows it is on is the agent's replay (no sign-in, or an unreadable
      // log): that session, not a return to an earlier one. Marking it added
      // "Back on an earlier topic" after every reload.
      const s = useChatStore.getState().sessions.get(key)
      const replayOnly = !!s && id === s.agentSessionId && lastTopicIndex(s.messages) < 0
      if (!replayOnly) {
        const count = Number(data.message_count ?? 0)
        markTopic(key, {
          id, name: String(data.name ?? ''),
          action: count > 0 ? 'resume' : 'new', source: 'agent',
        })
      }
      if (data.title) applyTopicTitle(key, id, String(data.title))
    })

    // The agent changed its web chat reasoning effort (its reasoning_effort
    // tool) — as you would in the chat box, so the picker follows.
    ws.on('reasoning_effort', (data) => {
      const pick = parsePick(data.level)
      if (!shared && pick) useReasoningStore.getState().setEffort(containerId, pick)
    })

    // The agent named a session it was asked to leave (a short title from
    // its opening turns): the Topics list and that session's dividers show it.
    ws.on('session_title', (data) => {
      const id = String(data.id ?? '')
      const title = String(data.title ?? '')
      if (id && title) applyTopicTitle(key, id, title)
    })

    ws.on('welcome', (data) => {
      const sessionInfo = data.session as Record<string, unknown> | undefined
      const name = sessionInfo?.name as string || ''
      const agentSessionId = String(sessionInfo?.id ?? '')
      if (shared) {
        // A shared agent offers no model or persona choice, and it must NOT
        // get the peer_agents handshake below — that would hand the member's
        // own agents (and their tokens) to somebody else's agent.
        const sp = data.speaker as Record<string, unknown> | undefined
        const patch: Partial<ChatSession> = {
          models: [], personalities: [], agentSessionId,
          speaker: sp ? {
            id: String(sp.id ?? ''), name: String(sp.name ?? ''),
            owner_name: String(sp.owner_name ?? ''), lane: String(sp.lane ?? ''),
          } : null,
        }
        if (name) patch.statusText = `Session: ${name}`
        updateSession(key, patch)
        reconcileTopic(key)
        return
      }
      const models = (data.models as AgentModelInfo[] || [])
      const personalities = (data.personalities as AgentPersonalityInfo[] || [])
      const reasoning = parseProfile(sessionInfo?.reasoning)
      const patch: Partial<ChatSession> = { models, personalities, agentSessionId, reasoning }
      if (name) patch.statusText = `Session: ${name}`
      updateSession(key, patch)
      reconcileTopic(key)

      // Send peer agent awareness to the connected agent
      ws.sendJSON(peerAgentsFrame(containerId))
    })

    ws.on('next_steps', (data) => {
      const raw = (data.options as Array<Record<string, unknown>>) || []
      const options: NextStepOption[] = raw
        .map((o) => ({
          label: String(o.label ?? '').trim(),
          action: String(o.action ?? '').trim(),
          description: o.description ? String(o.description) : undefined,
        }))
        .filter((o) => o.label && o.action)
      console.debug('[chatStore] next_steps received', { key, rawCount: raw.length, validCount: options.length, options })
      updateSession(key, { nextStepOptions: options })
      // next_steps is the explicit "agent finished its turn" signal — fire
      // any armed auto-complete watcher now instead of waiting for the
      // fallback timer.
      fireAutoCompleteWatch(key)
    })

    ws.on('chat_message', (data) => {
      const rawContent = (data.content as string) || ''
      let via = ''
      let sourceLabel = ''
      if (data.role === 'user' && !data.replay) {
        if (data.rotation_cue) {
          // The rest of a "new topic" message, re-sent after the divider:
          // the composer's echo of the whole line, above it, goes.
          dropCueEcho(key, rawContent)
        } else if (isOwnTurn(ws, data)) {
          // Our own send — already on screen from sendMessage.
          return
        } else {
          // Another surface's turn on this lane (WhatsApp, the stick, another
          // tab): part of the transcript, labelled where it came from.
          const from = String(data.via || '')
          via = from && from !== 'web' ? from : ''
          if (via && data.source_label) sourceLabel = String(data.source_label)
        }
      }
      // Sanitize assistant content: strip thinking blocks, council protocol
      // headers (SUITABILITY/ACTION/TARGET), insight echoes, and other
      // instruction/memory leaks. User content is left as-is.
      const cleanContent = data.role === 'assistant' ? sanitizeAgentContent(rawContent) : rawContent
      const msg: ChatMessage = {
        id: nextId(),
        role: data.role as 'user' | 'assistant' | 'rephrase',
        content: cleanContent,
        timestamp: data.timestamp as string || new Date().toISOString(),
        replay: data.replay as boolean || false,
        model: data.model as string || '',
        ...(via ? { via } : {}),
        ...(sourceLabel ? { sourceLabel } : {}),
        ...(taskMark(data.task) ? { task: taskMark(data.task) } : {}),
        ...(data.automation_lane ? { automation: String(data.automation_label || '') || 'Automated turn' } : {}),
      }
      addMessage(key, msg)
      // A result the automation lane mirrored here is a note: it ends no turn
      // of this lane and is no queued item's reply.
      if (data.automation_lane) return
      if (data.role === 'assistant' && !data.replay) {
        // Whose reply this is, read before anything moves on. Not the queued
        // task's when the agent refused its frame, or when the turn began
        // before the task went out (the turn you stopped ends with "Request
        // cancelled by user."; a message you sent earlier gets its answer).
        const own = turnItemId(key)
        const onScreen = useChatStore.getState().activeChatId === key
        updateSession(key, {
          busy: false, statusText: '', ...(onScreen ? {} : { unread: true }),
        })
        // The dispatched item got a reply ("Your turn" in step-by-step). A
        // reply also means an errored turn went through after all.
        if (own) {
          creditDispatchedItem(key, (prev) => ({
            repliedAt: Date.now(),
            ...(prev.halted?.kind === 'error' ? { halted: undefined } : {}),
          }))
        }
        // Queue auto-mode: decide what this reply means for the dispatched
        // queue item.
        //
        //   (a) Stall (intent without action / sanitized-to-empty): send a
        //       "Continue." nudge so the agent advances. Capped at
        //       MAX_STALL_NUDGES — after that, accept the result.
        //   (b) Clarifying question: do nothing. The user answers in chat,
        //       and the agent's next reply gets re-evaluated.
        //   (c) Real completion: arm the auto-complete watcher (which fires
        //       on the next_steps event or a fallback timer) so we don't
        //       race the agent's post-turn work.
        const cur = useChatStore.getState().sessions.get(key)
        // A task the user stopped holds: its trailing reply isn't judged.
        const stopped = cur?.queue.find((q) => q.id === cur.queueDispatchedId)?.halted?.kind === 'stopped'
        if (cur && cur.queueAutoMode && own && cur.queueDispatchedId === own && !stopped) {
          const dispatchedId = cur.queueDispatchedId
          if (isStuckReply(cleanContent)) {
            // The agent gave up. This item is NOT done — re-run it rather
            // than completing it and marching past unfinished work.
            cancelAutoCompleteWatch(key)
            cancelQuestionWatch(key)
            retryStuckItem(key, dispatchedId, cleanContent)
          } else if (isLikelyStall(cleanContent)) {
            const entry = _stallNudges.get(key)
            const used = entry && entry.itemId === dispatchedId ? entry.count : 0
            if (used < MAX_STALL_NUDGES) {
              _stallNudges.set(key, { itemId: dispatchedId, count: used + 1 })
              // Small delay so the nudge lands after any trailing events
              // (next_steps, status) from the same turn.
              setTimeout(() => {
                const s = useChatStore.getState().sessions.get(key)
                // Bail if user already moved on (toggled auto off, removed
                // the item, or marked it done manually).
                if (!s || !s.queueAutoMode) return
                if (s.queueDispatchedId !== dispatchedId) return
                if (s.busy) return
                // Stopped or errored meanwhile: the queue holds, no nudge.
                if (s.queue.find((q) => q.id === dispatchedId)?.halted) return
                useChatStore.getState().sendMessage(key, 'Continue the task.', { fromQueue: true })
              }, 800)
            } else {
              // Out of nudges — treat as completion so the queue moves on.
              armAutoCompleteWatch(key, dispatchedId)
            }
          } else if (isLikelyTaskComplete(cleanContent)) {
            resetStallNudges(key)
            cancelQuestionWatch(key)
            armAutoCompleteWatch(key, dispatchedId)
          } else {
            // A question. Hold for the user — but bounded and visible, never
            // the silent forever-park this used to be.
            armQuestionWatch(key, dispatchedId)
          }
        }
      }
    })

    ws.on('replay_batch', (data) => {
      const messages = (data.messages as Array<Record<string, unknown>>) || []
      if (messages.length === 0) return
      const replayed = messages.map(replayToMessage)
      // With the lane's log loaded, the log is the transcript: fold in only
      // what it is missing (a reply that finished while no tab was open, a
      // turn from another channel) instead of showing the session twice.
      const cur = useChatStore.getState().sessions.get(key)
      if (cur?.historyLoaded) {
        mergeReplay(key, replayed)
        return
      }
      replaySid = cur?.agentSessionId ?? ''
      // Clear old replay messages before adding new batch (prevents piling on reconnect)
      useChatStore.setState((state) => {
        const s = state.sessions.get(key)
        if (!s) return state
        const kept = s.messages.filter((m) => !m.replay)
        const updated = { ...s, messages: [...replayed, ...kept] }
        const newSessions = new Map(state.sessions)
        newSessions.set(key, updated)
        return { sessions: newSessions }
      })
    })

    ws.on('replay_done', () => {
      updateSession(key, { busy: false, statusText: '' })
    })

    ws.on('status', (data) => {
      const text = data.text as string || data.status as string || ''
      // "ready", "idle", or empty status means agent is done
      const idle = !text || /^(ready|idle|done|completed)$/i.test(text)
      // Shared chats: the "ready" owed to a frame refused while a turn runs
      // ends that frame, not the turn (see expectRefusalReady).
      if (idle && data.turn_end !== undefined && takeRefusalReady(key)) return
      // A turn began: the frame it took (the agent echoes its id). The end
      // of a turn forgets it.
      if (data.status === 'thinking') noteTurnStart(key, String(data.client_msg_id || ''))
      if (idle) noteTurnEnd(key)
      else heardFrom(key)
      const s = useChatStore.getState().sessions.get(key)
      // After Stop the line keeps saying so until the turn ends.
      const patch: Partial<ChatSession> = { busy: !idle, statusText: idle ? '' : s?.stopping ? 'Stopping…' : text }
      // Record when agent first becomes busy (for tok/s calculation)
      if (!idle && s && !s.busy) patch._busyStartedAt = Date.now()
      updateSession(key, patch)
      // "ready" is the turn's last frame (its usage and session_info came
      // first): the queue goes on now rather than on the kick's timer, which
      // a page in the background may run late.
      if (idle) startQueueNow(key)
    })

    // The thinking console's lines, streamed text and tool output: the chat
    // doesn't show them, but they say a turn is at work (a stopped one too),
    // so the lane isn't counted free on a timer meanwhile.
    for (const sign of ['thinking', 'response_stream', 'tool_stream']) {
      ws.on(sign, () => heardFrom(key))
    }

    ws.on('narration', (data) => {
      // Live between-step narration — show it in the flow as a subtle blurb
      // while a long task runs (same as WhatsApp/glasses).
      const text = String(data.text || '').trim()
      if (!text) return
      heardFrom(key)
      addMessage(key, {
        id: nextId(),
        role: 'system',
        content: text,
        timestamp: (data.timestamp as string) || new Date().toISOString(),
        narration: true,
      })
    })

    ws.on('monitor', (data) => {
      const toolName = data.tool_name as string || ''
      const output = data.output as string || ''
      if (toolName && !data.replay) {
        heardFrom(key)
        // Attribute the call to the queued task that is running it.
        creditTurnItem(key, (prev) => ({ toolCount: (prev.toolCount ?? 0) + 1 }))
        const msg: ChatMessage = {
          id: nextId(),
          role: 'tool',
          content: output,
          timestamp: new Date().toISOString(),
          tool_name: toolName,
          tool_arguments: data.arguments as Record<string, unknown>,
          tool_output: output,
        }
        addMessage(key, msg)
      }
      // The monitor event arrives AFTER the tool finished (it carries the
      // output) — setting "Using X..." here would show a stale phase while
      // the agent is already back in the (slow) LLM call. The backend now
      // emits accurate phase statuses ("Using X...", "Calling LLM (...)")
      // via the `status` event, so only keep the busy flag here.
      updateSession(key, { busy: true })
    })

    ws.on('approval_request', (data) => {
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.message as string || 'Approval requested',
        timestamp: new Date().toISOString(),
        approval_request_id: data.id as string,
        approval_category: data.category as string || '',
      }
      addMessage(key, msg)
    })

    ws.on('peer_activity', (data) => {
      const peerName = (data.peer_name as string) || 'Peer agent'
      const activityType = (data.activity_type as string) || ''
      const detail = (data.detail as string) || ''

      // Only show tool usage — skip status, connecting, done, errors
      if (activityType !== 'tool' && activityType !== 'thinking') return
      // For thinking, only show if it mentions a tool
      if (activityType === 'thinking' && !detail.startsWith('Using ')) return

      const toolName = activityType === 'tool' ? detail : detail.replace('Using ', '')
      if (!toolName) return

      const msg: ChatMessage = {
        id: nextId(),
        role: 'tool',
        content: '',
        timestamp: new Date().toISOString(),
        tool_name: toolName,
        peer_name: peerName,
      }
      addMessage(key, msg)
    })

    ws.on('error', (data) => {
      // Flight Deck refused the turn: today's plan allowance is spent. The
      // composer shows it from the totals that come with the refusal.
      if (data.code === 'quota_exceeded' && data.quota) {
        useQuotaStore.getState().set(data.quota as QuotaStatus)
      }
      // Before the agent's welcome an error is about reaching it (Flight
      // Deck's proxy could not connect), not about a turn. Every retry would
      // add one: keep a single line on the chat instead.
      if (!shared && !ws.connected) {
        updateSession(key, { unreachable: String(data.message || 'Connection failed') })
        return
      }
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.message as string || 'Unknown error',
        timestamp: new Date().toISOString(),
      }
      addMessage(key, msg)
      // A frame that isn't a turn was refused (session settings, a model
      // change): nothing about the turn or the queue changes.
      if (data.code === 'not_allowed') return
      // The queued task's turn failed, or a frame was refused: hold the queue
      // on the task when it was the queue's own (and say why), put a task the
      // busy agent never took back to wait, and keep the lane busy while a
      // turn still runs (see settleError).
      settleError(key, data)
    })

    ws.on('command_result', (data) => {
      const command = (data.command as string || '').trim().toLowerCase()
      const content = String(data.content || '')
      // A note for the running turn (a waiting task's Insert now). The turn
      // goes on, so nothing here is answered or freed: taken, the agent's
      // line goes in the chat; refused (`busy: false` — the turn had ended
      // as it was sent), the task goes back to the front of the queue and
      // runs on its own, as the button promised it would go now.
      if (command === '/btw') {
        const insert = _inserts.get(key)?.shift()
        if (data.busy === false) {
          const s = insert && useChatStore.getState().sessions.get(key)
          if (!insert || !s) return
          // Back in front of the first waiting task (behind a fresh-start
          // step ahead of it), whether or not it was cleared meanwhile.
          const back: QueuedMessage = { ...insert, status: 'pending', completedAt: undefined, endedHow: undefined }
          const rest = s.queue.filter((q) => q.id !== back.id)
          const at = rest.findIndex((q) => q.status === 'pending' && !isSlashCommand(q.content))
          updateSession(key, { queue: at < 0 ? [...rest, back] : [...rest.slice(0, at), back, ...rest.slice(at)] })
          tryDispatchNext(key, { force: true })
          return
        }
        addMessage(key, { id: nextId(), role: 'system', content, timestamp: new Date().toISOString() })
        return
      }
      // /new (typed, the New-topic button, or a "new topic" cue the agent
      // acted on), /session new|switch and /clear move or reset the agent's
      // context. The transcript is one continuous timeline: they mark a
      // topic boundary from the session the agent reports it is on now,
      // instead of wiping the screen.
      const created = content.startsWith('New session created')
      const isNew = (command === '/new' || command.startsWith('/new ')) && created
      const isSessionNew = /^\/session new\b/.test(command) && content.startsWith('Created and switched')
      const isSwitch = /^\/session (switch|load)\b/.test(command) && content.startsWith('Switched to session')
      const isClear = command === '/clear'
      const after = data.session as { id?: unknown; name?: unknown; title?: unknown } | undefined
      const afterId = after?.id ? String(after.id) : ''
      let marked = false
      if (afterId && (isNew || isSessionNew || isSwitch || isClear)) {
        marked = markTopic(key, {
          id: afterId, name: String(after?.name ?? ''),
          action: isClear ? 'clear' : isSwitch ? 'resume' : 'new', source: 'web',
        }) !== 'none'
      } else if (isClear || isNew) {
        // An agent build that doesn't report its session: wipe, as before.
        clearMessages(key)
      }
      if (afterId && after?.title) applyTopicTitle(key, afterId, String(after.title))
      // The divider says it; the agent's one-liner would only repeat it.
      if (!marked) {
        addMessage(key, { id: nextId(), role: 'system', content, timestamp: new Date().toISOString() })
      }
      // Slash commands run synchronously in the handler — once the
      // command_result arrives the agent is idle again. Without this
      // the "Thinking..." spinner stays pinned after `/plan`, `/help`,
      // `/clear`, etc., because no later token/usage event clears busy.
      updateSession(key, { busy: false, statusText: '' })
      // The command's frame is answered (no "thinking" ever takes it).
      takeCommandFrame(key)
      // …and if the command came from the queue, retire it so the row stops
      // spinning and the next item can go out. Must run after busy is cleared
      // — tryDispatchNext refuses to send while the session is busy.
      cancelCommandWatch(key)
      completeDispatchedCommand(key)
    })

    // Forward orchestrator trace spans to the trace store for
    // real-time observability in the TraceTimeline component.
    // Per-task lifecycle events also drive the PlanCard live monitor.
    ws.on('orchestrator_event', (data) => {
      const event = String(data.event ?? '')

      if (event === 'trace_span') {
        const span = data as unknown as TraceSpan
        if (span?.span_id) {
          // NOTE: traces stay keyed by agent — lanes share one trace timeline.
          useTraceStore.getState().handleSpanEvent(containerId, span)
        }
        return
      }

      // Plan-mode live monitor: only meaningful while a plan is active.
      const taskId = String(data.task_id ?? '')
      if (!taskId) return

      switch (event) {
        case 'task_started':
          updatePlan(key, (prev) => {
            if (!prev) return prev
            const steps = prev.steps.map((s) =>
              s.id === taskId
                ? {
                    ...s,
                    status: 'running' as const,
                    startedAt: Date.now(),
                    error: undefined,
                  }
                : s,
            )
            return { ...prev, steps, activeStepId: taskId }
          })
          break

        case 'task_step':
          updatePlan(key, (prev) => {
            if (!prev) return prev
            const phase = data.phase ? String(data.phase) : undefined
            const tool = data.tool ? String(data.tool) : undefined
            const text = data.text ? String(data.text) : undefined
            const steps = prev.steps.map((s) =>
              s.id === taskId
                ? {
                    ...s,
                    currentPhase: phase ?? s.currentPhase,
                    currentTool: tool ?? s.currentTool,
                    currentText: text ?? s.currentText,
                  }
                : s,
            )
            return { ...prev, steps }
          })
          break

        case 'task_completed':
          updatePlan(key, (prev) => {
            if (!prev) return prev
            const usage = (data.usage as Record<string, number> | undefined) || undefined
            const steps = prev.steps.map((s) =>
              s.id === taskId
                ? {
                    ...s,
                    status: 'completed' as const,
                    endedAt: Date.now(),
                    currentPhase: undefined,
                    currentTool: undefined,
                    tokensIn: usage?.input_tokens ?? s.tokensIn,
                    tokensOut: usage?.output_tokens ?? s.tokensOut,
                  }
                : s,
            )
            const nextActive = prev.activeStepId === taskId ? undefined : prev.activeStepId
            return { ...prev, steps, activeStepId: nextActive }
          })
          break

        case 'task_failed':
          updatePlan(key, (prev) => {
            if (!prev) return prev
            const error = String(data.error ?? 'task failed')
            const steps = prev.steps.map((s) =>
              s.id === taskId
                ? {
                    ...s,
                    status: 'failed' as const,
                    endedAt: Date.now(),
                    error,
                    currentPhase: undefined,
                    currentTool: undefined,
                  }
                : s,
            )
            const nextActive = prev.activeStepId === taskId ? undefined : prev.activeStepId
            return { ...prev, steps, activeStepId: nextActive }
          })
          break

        case 'task_validation_retry':
          updatePlan(key, (prev) => {
            if (!prev) return prev
            const reason = data.reason ? String(data.reason) : ''
            const steps = prev.steps.map((s) =>
              s.id === taskId
                ? {
                    ...s,
                    currentPhase: 'validating',
                    currentText: reason ? `Validation retry: ${reason}` : 'Validation retry',
                  }
                : s,
            )
            return { ...prev, steps }
          })
          break

        case 'task_validation_failed':
          updatePlan(key, (prev) => {
            if (!prev) return prev
            const notes = data.notes ? String(data.notes) : ''
            const steps = prev.steps.map((s) =>
              s.id === taskId
                ? { ...s, currentText: notes ? `Validation failed: ${notes}` : 'Validation failed' }
                : s,
            )
            return { ...prev, steps }
          })
          break

        default:
          break
      }
    })

    // ── Plan-mode events ──────────────────────────────────────────────
    ws.on('plan_execution_started', (data) => {
      const rawSteps = (data.steps as Array<Record<string, unknown>>) || []
      const steps: PlanStep[] = rawSteps.map((s) => ({
        id: String(s.id ?? ''),
        title: String(s.title ?? ''),
        step_kind: String(s.step_kind ?? 'atomic'),
        acceptance_criteria: String(s.acceptance_criteria ?? ''),
        depends_on: Array.isArray(s.depends_on) ? (s.depends_on as string[]) : [],
        status: 'pending',
        revisionCount: 0,
      }))
      updatePlan(key, () => ({
        startedAt: new Date().toISOString(),
        startedAtMs: Date.now(),
        status: 'running',
        maxRevisions: Number(data.max_revisions ?? 0),
        steps,
        revisions: [],
      }))
      // Auto-expand the plan card when a new plan starts
      updateSession(key, { planCardCollapsed: false })
    })

    ws.on('plan_orchestrate_expanded', (data) => {
      const expansions = (data.expansions as Array<Record<string, unknown>>) || []
      if (expansions.length === 0) return
      updatePlan(key, (prev) => {
        if (!prev) return prev
        // Expansions are appended sub-steps; merge by id when present.
        const byId = new Map(prev.steps.map((s) => [s.id, s]))
        for (const exp of expansions) {
          const subSteps = (exp.steps as Array<Record<string, unknown>>) || []
          for (const s of subSteps) {
            const id = String(s.id ?? '')
            if (!id || byId.has(id)) continue
            byId.set(id, {
              id,
              title: String(s.title ?? ''),
              step_kind: String(s.step_kind ?? 'atomic'),
              acceptance_criteria: String(s.acceptance_criteria ?? ''),
              depends_on: Array.isArray(s.depends_on) ? (s.depends_on as string[]) : [],
              status: 'pending',
              revisionCount: 0,
            })
          }
        }
        return { ...prev, steps: Array.from(byId.values()) }
      })
    })

    ws.on('plan_step_verified', (data) => {
      const taskId = String(data.task_id ?? '')
      const passed = Boolean(data.passed)
      const notes = String(data.notes ?? '')
      updatePlan(key, (prev) => {
        if (!prev) return prev
        const steps = prev.steps.map((s) =>
          s.id === taskId
            ? {
                ...s,
                status: passed ? ('verified' as const) : ('failed' as const),
                verificationPassed: passed,
                verificationNotes: notes,
              }
            : s,
        )
        return { ...prev, steps }
      })
    })

    ws.on('plan_step_revised', (data) => {
      const taskId = String(data.task_id ?? '')
      const revision: PlanRevision = {
        task_id: taskId,
        revision_count: Number(data.revision_count ?? 0),
        rationale: String(data.rationale ?? ''),
        revised_description: String(data.revised_description ?? ''),
        previous_verification_notes: String(data.previous_verification_notes ?? ''),
      }
      updatePlan(key, (prev) => {
        if (!prev) return prev
        const steps = prev.steps.map((s) =>
          s.id === taskId
            ? {
                ...s,
                status: 'revising' as const,
                revisionCount: revision.revision_count,
                verificationPassed: undefined,
                verificationNotes: undefined,
              }
            : s,
        )
        return { ...prev, steps, revisions: [...prev.revisions, revision] }
      })
    })

    ws.on('plan_execution_verified', (data) => {
      const verified = (data.verified as string[]) || []
      updatePlan(key, (prev) => {
        if (!prev) return prev
        const verifiedSet = new Set(verified)
        const steps = prev.steps.map((s) =>
          verifiedSet.has(s.id) && s.status !== 'verified' ? { ...s, status: 'verified' as const } : s,
        )
        return { ...prev, steps, status: 'verified' }
      })
    })

    ws.on('plan_execution_completed', (data) => {
      const completed = (data.completed as string[]) || []
      const verified = (data.verified as string[]) || []
      const hasFailures = Boolean(data.has_failures)
      const cancelled = Boolean(data.cancelled)
      const failedStep = (data.failed_step as string) || (data.verification_failed_step as string) || undefined
      const verificationNotes = (data.verification_notes as string) || undefined
      updatePlan(key, (prev) => {
        if (!prev) return prev
        const completedSet = new Set(completed)
        const verifiedSet = new Set(verified)
        const steps = prev.steps.map((s) => {
          if (s.id === failedStep) return { ...s, status: 'failed' as const, verificationNotes }
          if (verifiedSet.has(s.id)) return { ...s, status: 'verified' as const }
          if (completedSet.has(s.id) && s.status === 'pending') return { ...s, status: 'completed' as const }
          return s
        })
        const nextStatus: PlanState['status'] = cancelled
          ? 'cancelled'
          : hasFailures
            ? 'failed'
            : 'completed'
        return {
          ...prev,
          steps,
          status: nextStatus,
          activeStepId: undefined,
          failedStep: (data.failed_step as string) || prev.failedStep,
          verificationFailedStep: (data.verification_failed_step as string) || prev.verificationFailedStep,
          verificationNotes: verificationNotes ?? prev.verificationNotes,
          errorMessage: cancelled ? 'Plan execution cancelled by user.' : prev.errorMessage,
        }
      })
    })

    ws.on('plan_execution_failed', (data) => {
      const error = String(data.error ?? 'plan execution failed')
      updatePlan(key, (prev) => {
        if (!prev) return prev
        return { ...prev, status: 'failed', errorMessage: error }
      })
    })

    // Mirror /planning on|off command results into local planningEnabled flag.
    // Strips markdown so bold/italic don't break substring matches.
    //
    // The agent emits "Plan-mode auto-routing **enabled**." or
    // "Plan-mode auto-routing **disabled**." — match the leading state word
    // anchored to "auto-routing" so the trailing "Use /planning off to
    // disable" hint can't trip the opposite branch (the previous patterns
    // looked for `auto-route\s+enabled`, which never matched because the
    // word is "auto-routing", and the fallback `/planning\s+off\b/` then
    // matched the help-text mention of `/planning off` and incorrectly
    // flipped state to OFF on every enable).
    ws.on('command_result', (data) => {
      const command = String(data.command ?? '').trim().toLowerCase()
      if (command !== '/planning' && !command.startsWith('/planning ')) return
      const content = String(data.content ?? '')
        .toLowerCase()
        .replace(/[*_`]+/g, '')
      // Three formats the agent emits:
      //   "Plan-mode auto-routing **enabled**."           (after `/planning on`)
      //   "Plan-mode auto-routing **disabled**."          (after `/planning off`)
      //   "Plan-mode auto-routing: **on|off**. ..."       (after bare `/planning` query)
      if (/auto-rout(?:e|ing)\s+enabled|auto-rout(?:e|ing):\s*on\b|currently\s+on\b/.test(content)) {
        updateSession(key, { planningEnabled: true })
      } else if (/auto-rout(?:e|ing)\s+disabled|auto-rout(?:e|ing):\s*off\b|currently\s+off\b/.test(content)) {
        updateSession(key, { planningEnabled: false })
      }
      // Plan-mode level confirmations:
      //   "Plan-mode level set to **<name>**."         (after `/planning level <name>`)
      //   "Plan-mode level: **<name>**. ..."           (after bare `/planning level`)
      //   "Plan-mode auto-routing: **on** (level: **<name>**). ..."
      //                                                (after bare `/planning`)
      // Match all three with one regex over markdown-stripped content.
      const lvlMatch = content.match(/plan-mode\s+(?:level\s+set\s+to|level:|auto-rout(?:e|ing):[^()]*\(level:)\s*(plain|enriched|insightful|complete)\b/)
      if (lvlMatch) {
        updateSession(key, { planLevel: lvlMatch[1] })
      }
    })

    // Token generation speed tracking from usage events
    // Live cumulative token usage for the in-flight turn — drives the
    // input/output/cache counts in the activity-panel header.
    ws.on('turn_usage', (data) => {
      heardFrom(key)
      updateSession(key, {
        liveTurnUsage: {
          prompt_tokens: (data.prompt_tokens as number) || 0,
          completion_tokens: (data.completion_tokens as number) || 0,
          cache_read_input_tokens: (data.cache_read_input_tokens as number) || 0,
          cache_creation_input_tokens: (data.cache_creation_input_tokens as number) || 0,
          total_tokens: (data.total_tokens as number) || 0,
        },
      })
    })

    // Flight Deck counted a message against today's plan allowance.
    ws.on('fd_quota', (data) => {
      if (data.quota) useQuotaStore.getState().set(data.quota as QuotaStatus)
    })

    ws.on('usage', (data) => {
      // The turn's tokens reach Flight Deck from the agent: read the new total.
      useQuotaStore.getState().refreshSoon()
      const last = data.last as Record<string, number> | undefined
      if (!last) return
      // The turn's numbers onto the queued item, ADDED to what it already
      // spent — a task that gave up twice should report its true cost, not
      // just the cost of the attempt that finally worked. Credited BEFORE the
      // auto-complete below: that can mark the item done and dispatch the
      // next one synchronously, which would hand it this turn's tokens. Only
      // when the turn is the item's (not one it was refused behind).
      creditTurnItem(key, (prev) => {
        const p = prev.usage || {}
        return {
          usage: {
            prompt_tokens: (p.prompt_tokens || 0) + (last.prompt_tokens || 0),
            completion_tokens: (p.completion_tokens || 0) + (last.completion_tokens || 0),
            cache_read_input_tokens:
              (p.cache_read_input_tokens || 0) + (last.cache_read_input_tokens || 0),
            cache_creation_input_tokens:
              (p.cache_creation_input_tokens || 0) + (last.cache_creation_input_tokens || 0),
            total_tokens: (p.total_tokens || 0) + (last.total_tokens || 0),
          },
        }
      })
      // Turn-final usage lands exactly where next_steps would have. A
      // queue-dispatched turn suppresses next_steps, so this is what tells
      // the queue the turn is over — without it, every queued item would
      // wait out the 6s fallback timer instead.
      fireAutoCompleteWatch(key)
      // Freeze the turn's final usage onto the most recent tool message so the
      // activity group keeps showing its own token counts after it collapses
      // (and old groups don't all show the latest turn's numbers). Then clear
      // the live counter for the next turn.
      if ((last.prompt_tokens || 0) > 0 || (last.completion_tokens || 0) > 0) {
        const frozen: TokenUsage = {
          prompt_tokens: last.prompt_tokens || 0,
          completion_tokens: last.completion_tokens || 0,
          cache_read_input_tokens: last.cache_read_input_tokens || 0,
          cache_creation_input_tokens: last.cache_creation_input_tokens || 0,
          total_tokens: last.total_tokens || 0,
        }
        useChatStore.setState((state) => {
          const s = state.sessions.get(key)
          if (!s) return state
          const msgs = s.messages.slice()
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'tool') {
              msgs[i] = { ...msgs[i], usage: frozen }
              break
            }
          }
          const next = new Map(state.sessions)
          next.set(key, { ...s, messages: msgs, liveTurnUsage: null })
          return { sessions: next }
        })
      } else {
        updateSession(key, { liveTurnUsage: null })
      }
      const completionTokens = last.completion_tokens || 0
      if (completionTokens <= 0) return
      const s = useChatStore.getState().sessions.get(key)
      if (!s || !s._busyStartedAt) return
      const elapsedSec = (Date.now() - s._busyStartedAt) / 1000
      if (elapsedSec <= 0) return
      const tokPerSec = completionTokens / elapsedSec
      const samples = s._tokSamples + 1
      const newAvg = (s.avgTokPerSec * s._tokSamples + tokPerSec) / samples
      // Real LLM speed from backend latency_ms (pure API call time)
      const latencyMs = last.latency_ms || 0
      const llmTokPerSec = latencyMs > 0 ? (completionTokens / latencyMs) * 1000 : 0
      updateSession(key, {
        lastTokPerSec: tokPerSec,
        avgTokPerSec: newAvg,
        _tokSamples: samples,
        _busyStartedAt: 0,
        llmTokPerSec,
      })
    })

    ws.connect()
  },

  setActiveLane: (containerId, lane) => {
    const target = lane || LANE_MAIN
    const key = laneKey(containerId, target)
    const state = get()
    // Every lane rides on the same agent, so a lane that has never been
    // looked at has no socket yet. Open it on first view rather than holding
    // three connections for a user who only ever uses A.
    const existing = state.sessions.get(key)
    if (!existing) {
      const anyLane = state.sessions.get(containerId)
        || [...state.sessions.values()].find((s) => s.containerId === containerId)
      if (!anyLane) return          // the agent isn't open at all
      if (anyLane.shared) {
        state.openSharedChat(anyLane.shared.agentRef, anyLane.containerName,
                             anyLane.shared.ownerName, target)
      } else {
        state.openChat(containerId, anyLane.containerName, anyLane.host,
                       anyLane.port, anyLane.auth, target)
      }
    } else if (!existing.connected && !existing.ws.connecting) {
      if (existing.closed) updateSession(key, { closed: null })
      existing.ws.connect()
    }
    set((s) => ({
      activeLane: { ...s.activeLane, [containerId]: target },
      activeChatId: key,
    }))
    if (get().sessions.get(key)?.unread) updateSession(key, { unread: false })
  },

  closeChat: () => {
    set({ chatOpen: false })
  },

  switchChat: (containerId) => {
    if (get().sessions.get(containerId)?.unread) updateSession(containerId, { unread: false })
    set({ activeChatId: containerId, chatOpen: true })
  },

  disconnectChat: (containerId) => {
    const session = get().sessions.get(containerId)
    _sendOnWake.delete(containerId)
    if (session) {
      session.ws.disconnect()
      const sessions = new Map(get().sessions)
      sessions.delete(containerId)
      set((s) => ({
        sessions,
        activeChatId: s.activeChatId === containerId ? null : s.activeChatId,
        chatOpen: s.activeChatId === containerId ? false : s.chatOpen,
      }))
    }
  },

  resendFleetIdentity: (agentId) => {
    for (const session of get().sessions.values()) {
      if (session.containerId === agentId && session.connected) session.ws.sendJSON(peerAgentsFrame(agentId))
    }
  },

  sendMessage: (containerId, content, opts) => {
    const session = get().sessions.get(containerId)
    if (!session) return undefined
    // A card number / CVV never reaches the bubble, Flight Deck's chat log or
    // the agent (utils/cardScrub — Captain never needs one).
    content = scrubUserText(content)
    // An answer ends the hold — the agent's next reply is re-evaluated fresh.
    cancelQuestionWatch(containerId)
    // (A message typed in the chat also resumes a task the queue is holding
    // on, once the agent takes it: noteTurnStart.)
    // Add user message to local state
    const msg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }
    addMessage(containerId, msg)
    // Asleep (or waking): the frame goes once the agent's welcome is in.
    const sleeping = !session.connected && (session.asleep || session.waking)
    updateSession(containerId, {
      busy: true,
      statusText: sleeping ? 'Waking up…' : session.stopping ? 'Stopping…' : 'Thinking...',
      _busyStartedAt: Date.now(),
      nextStepOptions: [],
      liveTurnUsage: null,
      ...(sleeping ? { asleep: false, waking: true } : {}),
    })
    if (sleeping) {
      _sendOnWake.set(containerId, [...(_sendOnWake.get(containerId) ?? []), { content, opts, wasBusy: session.busy }])
      session.ws.wake()
      return msg.id
    }
    sendFrame(containerId, content, opts, session.busy)
    return msg.id
  },

  wakeAgent: (key) => {
    const session = get().sessions.get(key)
    if (!session || session.connected || !session.asleep) return
    updateSession(key, { asleep: false, waking: true })
    session.ws.wake()
  },

  sendBtw: (containerId, content) => {
    const session = get().sessions.get(containerId)
    if (session) session.ws.sendBtw(content)
  },

  newSession: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session || session.busy) return
    session.ws.sendJSON({ type: 'command', command: '/new' })
  },

  resumeTopic: (containerId, agentSessionId) => {
    const session = get().sessions.get(containerId)
    if (!session || session.busy || !agentSessionId) return
    if (session.agentSessionId === agentSessionId) return
    session.ws.sendJSON({ type: 'command', command: `/session switch ${agentSessionId}` })
  },

  loadOlder: async (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session || !session.hasOlder) return
    const first = session.messages.find((m) => m.rowId !== undefined)
    if (!first || first.rowId === undefined) {
      updateSession(containerId, { hasOlder: false })
      return
    }
    const rows = await serverFetchMessages(containerId, { before: first.rowId, limit: HISTORY_PAGE })
    if (rows === null) return
    useChatStore.setState((state) => {
      const s = state.sessions.get(containerId)
      if (!s) return state
      const sessions = new Map(state.sessions)
      // Paging back: the window may shed its newest rows to stay bounded.
      sessions.set(containerId, trimWindow({
        ...s, messages: [...rows, ...s.messages],
        hasOlder: rows.length >= HISTORY_PAGE, holdScrollUntil: Date.now() + 1500,
      }, 'oldest'))
      return { sessions }
    })
  },

  loadNewer: async (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session || !session.hasNewer) return
    const rows = await serverFetchMessages(containerId, { since: session.windowEndRowId + 1, limit: HISTORY_PAGE })
    if (rows === null) return
    useChatStore.setState((state) => {
      const s = state.sessions.get(containerId)
      if (!s) return state
      // Rows the window already holds (sent from here after the jump and
      // persisted since) are not repeated; the rest goes in right after the
      // contiguous block, ahead of anything newer that sits at the end.
      const have = new Set(s.messages.map((m) => m.rowId).filter((id): id is number => id !== undefined))
      const fresh = rows.filter((m) => m.rowId !== undefined && !have.has(m.rowId))
      let at = s.messages.length
      for (let i = s.messages.length - 1; i >= 0; i--) {
        const id = s.messages[i].rowId
        if (id !== undefined && id <= s.windowEndRowId) { at = i + 1; break }
      }
      const more = rows.length >= HISTORY_PAGE
      const endRowId = rows.reduce((mx, m) => Math.max(mx, m.rowId ?? 0), s.windowEndRowId)
      const sessions = new Map(state.sessions)
      // Moving forward: the window may shed its oldest rows to stay bounded.
      sessions.set(containerId, trimWindow({
        ...s, messages: [...s.messages.slice(0, at), ...fresh, ...s.messages.slice(at)],
        hasNewer: more, windowEndRowId: more ? endRowId : 0, holdScrollUntil: Date.now() + 1500,
      }, 'newest'))
      return { sessions }
    })
  },

  jumpToTopic: async (containerId, rowId) => {
    const session = get().sessions.get(containerId)
    if (!session) return null
    const have = session.messages.find((m) => m.rowId === rowId)
    if (have) {
      updateSession(containerId, { holdScrollUntil: Date.now() + 1500 })
      return have.id
    }
    const rows = await serverFetchMessages(containerId, { since: rowId, limit: HISTORY_PAGE })
    if (!rows || rows.length === 0) return null
    const more = rows.length >= HISTORY_PAGE
    const endRowId = rows.reduce((mx, m) => Math.max(mx, m.rowId ?? 0), 0)
    useChatStore.setState((state) => {
      const s = state.sessions.get(containerId)
      if (!s) return state
      // The window becomes a page of the log from that divider on; rows
      // that only exist here so far (not persisted yet) stay at the end.
      const live = s.messages.filter((m) => m.rowId === undefined && !m.replay)
      const sessions = new Map(state.sessions)
      sessions.set(containerId, {
        ...s, messages: [...rows, ...live], hasOlder: true,
        hasNewer: more, windowEndRowId: more ? endRowId : 0, holdScrollUntil: Date.now() + 1500,
      })
      return { sessions }
    })
    return rows[0].id
  },

  renameTopic: (containerId, ref, title) => {
    const clean = title.replace(/\s+/g, ' ').trim().slice(0, 120)
    let rowId: number | undefined
    useChatStore.setState((state) => {
      const s = state.sessions.get(containerId)
      if (!s) return state
      let changed = false
      const messages = s.messages.map((m) => {
        if (!m.topic) return m
        const hit = (ref.messageId !== undefined && m.id === ref.messageId)
          || (ref.rowId !== undefined && m.rowId === ref.rowId)
        if (!hit || m.topic.title === clean) return m
        changed = true
        rowId = m.rowId
        return { ...m, topic: { ...m.topic, title: clean } }
      })
      if (!changed) return state
      const sessions = new Map(state.sessions)
      sessions.set(containerId, { ...s, messages })
      return { sessions }
    })
    const persisted = rowId ?? ref.rowId
    if (persisted !== undefined) void serverSetTopicTitle(containerId, persisted, clean)
  },

  addLocalNote: (containerId, content) => {
    if (!get().sessions.get(containerId)) return
    addMessage(containerId, {
      id: nextId(),
      role: 'system',
      content,
      timestamp: new Date().toISOString(),
    })
  },

  cancelTask: (containerId) => {
    const session = get().sessions.get(containerId)
    // Stopping a queued task's turn holds the queue on it ("Stopped") — no
    // auto-complete, nudge or retry may send it on by itself afterwards.
    if (session?.queueDispatchedId && session.busy) {
      creditDispatchedItem(containerId, (prev) => (prev.halted?.kind === 'stopped'
        ? {} : { halted: { kind: 'stopped', at: Date.now() } }))
      cancelAutoCompleteWatch(containerId)
      cancelQuestionWatch(containerId)
      resetStallNudges(containerId)
    }
    if (!session) return
    session.ws.cancel()
    if (session.busy) {
      // The agent only notices Stop at its next step and refuses a frame
      // sent before then, so the lane stays busy until the stopped turn ends
      // (its reply or "ready"): the next task, Skip & continue and Run it
      // again all wait for that. With no sign of it for a while, the lane
      // counts as free (holdBusy).
      updateSession(containerId, { stopping: true, statusText: 'Stopping…' })
      holdBusy(containerId)
    } else {
      updateSession(containerId, { statusText: 'Cancelled' })
    }
  },

  setModel: (containerId, selector) => {
    const session = get().sessions.get(containerId)
    if (session && !session.shared) {
      session.ws.sendJSON({ type: 'set_model', selector })
      updateSession(containerId, { activeModel: selector })
    }
  },

  setPersonality: (containerId, personalityId) => {
    const session = get().sessions.get(containerId)
    if (session && !session.shared) {
      session.ws.sendJSON({ type: 'set_personality', personality_id: personalityId })
      updateSession(containerId, { activePersonality: personalityId })
    }
  },

  respondToApproval: (containerId, requestId, approved) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.sendJSON({ type: 'approval_response', id: requestId, approved })
      // Mark the approval message as resolved
      const messages = session.messages.map((m) =>
        m.approval_request_id === requestId ? { ...m, approval_resolved: true } : m
      )
      updateSession(containerId, { messages })
    }
  },

  setPlanningEnabled: (containerId, enabled) => {
    const session = get().sessions.get(containerId)
    if (!session || session.shared) return  // no plan mode in shared chats
    // Optimistic update — confirmed/corrected by command_result handler.
    updateSession(containerId, { planningEnabled: enabled })
    // Send as a regular chat message so it routes through the slash command
    // handler (which also returns a command_result we listen to).
    session.ws.send(enabled ? '/planning on' : '/planning off')
  },

  setPlanLevel: (containerId, level) => {
    const session = get().sessions.get(containerId)
    if (!session || session.shared) return  // no plan mode in shared chats
    // Optimistic update — confirmed/corrected by command_result handler.
    updateSession(containerId, { planLevel: level })
    session.ws.send(`/planning level ${level}`)
  },

  togglePlanCardCollapsed: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    updateSession(containerId, { planCardCollapsed: !session.planCardCollapsed })
  },

  dismissPlan: (containerId) => {
    updateSession(containerId, { planState: null })
    clearPlanSlice(containerId)
  },

  toggleChatFullscreen: () => {
    set((s) => ({ chatFullscreen: !s.chatFullscreen }))
  },

  enqueueQueueMessage: (containerId, content) => {
    const trimmed = content.trim()
    if (!trimmed) return
    const session = get().sessions.get(containerId)
    if (!session) return
    const item: QueuedMessage = {
      id: nextQueueId(),
      content: trimmed,
      status: 'pending',
      createdAt: Date.now(),
    }
    updateSession(containerId, { queue: [...session.queue, item] })
    // Try to dispatch immediately if nothing else is in flight.
    tryDispatchNext(containerId)
  },

  removeQueueItem: (containerId, id) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    // Removing the in-flight item also clears the dispatched marker — the
    // agent will still finish its work, but FD stops waiting on it.
    const patch: Partial<ChatSession> = {
      queue: session.queue.filter((q) => q.id !== id),
    }
    if (session.queueDispatchedId === id) {
      patch.queueDispatchedId = null
      cancelAutoCompleteWatch(containerId)
      cancelCommandWatch(containerId)
      cancelQuestionWatch(containerId)
      resetStallNudges(containerId)
    }
    updateSession(containerId, patch)
    if (session.queueDispatchedId === id) tryDispatchNext(containerId)
  },

  markQueueItemDone: (containerId, id, opts) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    const queue = session.queue.map((q) =>
      q.id === id && q.status !== 'done'
        ? {
          ...q, status: 'done' as const, completedAt: Date.now(), endedHow: opts?.how,
          halted: undefined, awaitingAnswer: undefined, awaitingSince: undefined,
        }
        : q,
    )
    const patch: Partial<ChatSession> = { queue }
    if (session.queueDispatchedId === id) {
      patch.queueDispatchedId = null
      cancelAutoCompleteWatch(containerId)
      cancelCommandWatch(containerId)
      cancelQuestionWatch(containerId)
      resetStallNudges(containerId)
    }
    updateSession(containerId, patch)
    // Skipping a task that is only waiting tidies the list, and handing one
    // to the running turn keeps that turn going; neither is a reason to
    // start whatever comes next.
    if ((opts?.how === 'skipped' || opts?.how === 'inserted') && session.queueDispatchedId !== id) return
    tryDispatchNext(containerId)
  },

  toggleQueueAutoMode: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    const next = !session.queueAutoMode
    // Auto back on answers the power-off park.
    updateSession(containerId, next ? { queueAutoMode: true, queueParked: false } : { queueAutoMode: false })
    // Turning auto-mode off cancels any pending watch so it doesn't fire
    // after the user has switched to manual.
    if (!next) { cancelAutoCompleteWatch(containerId); cancelQuestionWatch(containerId) }
    // Turning auto-mode on while idle should kick the queue — and a task the
    // allowance refused goes again if the allowance came back meanwhile.
    if (next) {
      if (!session.queuePaused) retryQuotaHold(containerId)
      tryDispatchNext(containerId)
    }
  },

  parkQueues: (agentId, check = false) => {
    let parked = false
    for (const lane of LANES) {
      const key = laneKey(agentId, lane)
      const session = get().sessions.get(key)
      if (session) {
        if (!session.queueAutoMode || !session.queue.some((q) => q.status !== 'done')) continue
        parked = true
        if (!check) {
          get().toggleQueueAutoMode(key)
          updateSession(key, { queueParked: true })
        }
        continue
      }
      // Not open here: the queue a reopened chat would pick up from storage.
      try {
        const raw = window.localStorage.getItem(_queueLSKey(key))
        const slice = raw ? JSON.parse(raw) as PersistedQueueSlice : null
        if (!slice?.queueAutoMode || !Array.isArray(slice.queue)
          || !slice.queue.some((q) => q.status !== 'done')) continue
        parked = true
        if (!check) saveQueueSlice(key, { ...slice, queueAutoMode: false, parked: true })
      } catch { /* unreadable storage — nothing to park */ }
    }
    return parked
  },

  clearQueueFinished: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    // A long run leaves a wall of struck-through items between you and the
    // work still to come. Keep everything not yet finished.
    const queue = session.queue.filter((q) => q.status !== 'done')
    if (queue.length !== session.queue.length) updateSession(containerId, { queue })
  },

  clearQueue: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    // Drop pending items; leave the in-flight one alone so we don't lose
    // track of the agent's current task.
    const queue = session.queue.filter((q) => q.status !== 'pending')
    updateSession(containerId, { queue })
  },

  runQueueNext: (key) => tryDispatchNext(key, { force: true }),

  setQueuePaused: (key, paused) => {
    if (!get().sessions.get(key)) return
    // Pausing also forgets a start the user asked for earlier: paused wins.
    if (paused) _forceThrough.delete(key)
    updateSession(key, { queuePaused: paused, ...(paused ? { queueStartWhenFree: false, queueForceThrough: false } : {}) })
    if (paused) return
    // Resuming carries on, with a task the allowance refused too when the
    // allowance came back while the lane was paused.
    retryQuotaHold(key)
    tryDispatchNext(key)
  },

  setQueueAutoMode: (key, on) => {
    const session = get().sessions.get(key)
    if (!session || session.queueAutoMode === on) return
    get().toggleQueueAutoMode(key)
    // Step by step's reply was already in, waiting on the user's "mark done":
    // Auto now means "carry on" — nothing would judge that reply any more.
    const d = on && session.queueDispatchedId
      ? session.queue.find((q) => q.id === session.queueDispatchedId)
      : undefined
    if (d && d.repliedAt && !session.busy && !d.halted && !d.stuck && !d.awaitingAnswer && !isSlashCommand(d.content)) {
      get().markQueueItemDone(key, d.id)
    }
  },

  retryQueueItem: (key, id, content) => {
    const session = get().sessions.get(key)
    const item = session?.queue.find((q) => q.id === id)
    if (!session || !item || item.status === 'done') return
    const text = content?.trim()
    // A fresh run: what it cost so far (usage, tools) keeps adding up.
    const fresh: QueuedMessage = {
      ...item, ...(text ? { content: text } : {}), status: 'pending',
      dispatchedAt: undefined, completedAt: undefined, stuck: undefined, attempts: undefined,
      halted: undefined, awaitingAnswer: undefined, awaitingSince: undefined,
      repliedAt: undefined, messageId: undefined,
    }
    // It runs next: just before the first waiting item, else where it was.
    const oldIndex = session.queue.findIndex((q) => q.id === id)
    const rest = session.queue.filter((q) => q.id !== id)
    const firstPending = rest.findIndex((q) => q.status === 'pending')
    const at = firstPending >= 0 ? firstPending : Math.min(oldIndex, rest.length)
    const queue = [...rest.slice(0, at), fresh, ...rest.slice(at)]
    const patch: Partial<ChatSession> = { queue }
    if (session.queueDispatchedId === id) {
      patch.queueDispatchedId = null
      cancelAutoCompleteWatch(key)
      cancelCommandWatch(key)
      cancelQuestionWatch(key)
      resetStallNudges(key)
      resetStuckRetries(key)
    }
    updateSession(key, patch)
    // Still busy on the old turn: start-when-free sends it once it's over.
    tryDispatchNext(key, { force: true })
  },

  editQueueItem: (key, id, content) => {
    const session = get().sessions.get(key)
    const text = content.trim()
    const item = session?.queue.find((q) => q.id === id)
    // Only a waiting item: once it's sent, the agent already has the old text.
    if (!session || !item || item.status !== 'pending' || !text) return false
    updateSession(key, { queue: session.queue.map((q) => (q.id === id ? { ...q, content: text } : q)) })
    return true
  },

  moveQueueItem: (key, id, beforeId) => {
    const session = get().sessions.get(key)
    if (!session || id === beforeId) return
    const item = session.queue.find((q) => q.id === id)
    if (!item || item.status !== 'pending') return
    if (beforeId !== null && session.queue.find((q) => q.id === beforeId)?.status !== 'pending') return
    // Array order IS run order for waiting items, so a move is a splice.
    const rest = session.queue.filter((q) => q.id !== id)
    const at = beforeId === null ? rest.length : rest.findIndex((q) => q.id === beforeId)
    updateSession(key, { queue: [...rest.slice(0, at), item, ...rest.slice(at)] })
  },

  insertQueueItemNow: (key, id) => {
    const session = get().sessions.get(key)
    if (!session) return false
    const item = session.queue.find((q) => q.id === id)
    if (!item || item.status !== 'pending' || isSlashCommand(item.content)) return false
    // Only a turn that is on can take it. (`stopping`: the turn is ending
    // on the user's word — a note for its remaining steps would be lost.)
    if (!session.busy || session.stopping || !session.connected) return false
    session.ws.sendBtw(item.content, { runningOnly: true })
    _inserts.set(key, [...(_inserts.get(key) ?? []), item])
    get().markQueueItemDone(key, id, { how: 'inserted' })
    return true
  },

  restoreQueueItems: (key, entries) => {
    const session = get().sessions.get(key)
    if (!session || entries.length === 0) return
    const queue = [...session.queue]
    for (const { item, index } of [...entries].sort((a, b) => a.index - b.index)) {
      // A saved copy of an item that was in flight comes back waiting.
      const copy: QueuedMessage = item.status === 'dispatched'
        ? { ...item, status: 'pending', dispatchedAt: undefined }
        : item
      const at = queue.findIndex((q) => q.id === item.id)
      if (at >= 0) {
        // Still there (a Skip): put the saved copy back in place — unless it
        // started meanwhile, which the undo must not take back.
        if (queue[at].status !== 'dispatched') queue[at] = copy
      } else {
        queue.splice(Math.max(0, Math.min(index, queue.length)), 0, copy)
      }
    }
    updateSession(key, { queue })
    if (session.queueAutoMode && !session.queuePaused) tryDispatchNext(key)
  },

  extendQueueAnswerWait: (key) => {
    const session = get().sessions.get(key)
    const item = session?.queue.find((q) => q.id === session.queueDispatchedId)
    if (item?.awaitingAnswer) armQuestionWatch(key, item.id)
  },
}))

// ── Queue dispatch ──
//
// Pulls the next pending item and pushes it through sendMessage so it appears
// in chat history exactly like a user-typed message. Guarded by:
//  - no item currently dispatched (queueDispatchedId === null)
//  - at least one pending item to send
//  - the queue not paused (`force` — Run next — runs one anyway)
//  - session not busy (don't pile prompts on top of an in-flight one), the
//    socket up, and today's allowance not spent. Any of those only DEFERS:
//    `queueStartWhenFree` remembers the start was wanted, and it happens as
//    soon as the agent is free, back, or allowed again — never a silent stall.
function tryDispatchNext(containerId: string, opts?: { force?: boolean }) {
  const state = useChatStore.getState()
  const session = state.sessions.get(containerId)
  if (!session) return
  if (session.queueDispatchedId) return
  const next = session.queue.find((q) => q.status === 'pending')
  if (!next) {
    if (session.queueStartWhenFree) updateSession(containerId, { queueStartWhenFree: false })
    return
  }
  if (session.queuePaused && !opts?.force) return
  const quota = useQuotaStore.getState().status
  const blocked = !!quota?.blocked && !quota.exempt
  // (`stopping` keeps `busy` on already; named so a stop still landing is
  // plainly a reason to wait.)
  if (session.busy || session.stopping || !session.connected || blocked) {
    if (!session.queueStartWhenFree) updateSession(containerId, { queueStartWhenFree: true })
    // Asleep (stopped for being idle): a task to run wakes it; it starts on the welcome.
    if (session.asleep && !session.connected && !blocked) state.wakeAgent(containerId)
    // Busy with no turn known to run (a stray status or tool frame after the
    // turn's "ready"): nothing would ever free the lane, so the fallback that
    // counts it free after a quiet stretch does.
    if (session.busy && !_turns.has(containerId) && !_busyHolds.has(containerId)) holdBusy(containerId)
    return
  }
  const queue = session.queue.map((q) =>
    q.id === next.id
      ? {
        ...q, status: 'dispatched' as const, dispatchedAt: Date.now(), completedAt: undefined,
        stuck: undefined, attempts: undefined, halted: undefined, awaitingAnswer: undefined,
        awaitingSince: undefined, repliedAt: undefined, endedHow: undefined,
      }
      : q,
  )
  // A forced run (Run now, Try again) that begins with a step (the fresh
  // start) carries through to the task behind it, paused or not.
  const through = !!opts?.force && isSlashCommand(next.content)
  updateSession(containerId, {
    queue, queueDispatchedId: next.id, queueStartWhenFree: false, queueParked: false, queueForceThrough: through,
  })
  if (through) _forceThrough.add(containerId)
  else _forceThrough.delete(containerId)
  // A new item starts with a fresh retry budget.
  resetStuckRetries(containerId)
  if (isSlashCommand(next.content)) armCommandWatch(containerId)
  const mid = state.sendMessage(containerId, next.content, { fromQueue: true, forced: !!opts?.force })
  // Remember its bubble, for "Show in chat".
  if (mid) creditDispatchedItem(containerId, () => ({ messageId: mid }))
}

// Start-when-free: shortly after the agent goes idle with nothing dispatched,
// try the queue again — on Auto, or when the user had asked for a start. The
// short delay lets the turn's trailing events (status, usage) land first.
// Pausing clears `queueStartWhenFree`, and an unforced try never sets it on a
// paused lane — so on a paused lane it can only be a forced start (Run now,
// Try again) and goes through as one.
const _queueKicks = new Map<string, ReturnType<typeof setTimeout>>()

/** The agent said it is free: the queue's next task goes now when it would
 *  go on the kick (Auto, or a start you asked for), and the kick is dropped. */
function startQueueNow(key: string) {
  const s = useChatStore.getState().sessions.get(key)
  if (!s || !s.connected || s.busy || s.queueDispatchedId) return
  if (!(s.queueStartWhenFree || (!s.queuePaused && s.queueAutoMode))) return
  const t = _queueKicks.get(key)
  if (t) {
    clearTimeout(t)
    _queueKicks.delete(key)
  }
  tryDispatchNext(key, { force: s.queueStartWhenFree })
}

function scheduleQueueKick(key: string) {
  const prev = _queueKicks.get(key)
  if (prev) clearTimeout(prev)
  _queueKicks.set(key, setTimeout(() => {
    _queueKicks.delete(key)
    const s = useChatStore.getState().sessions.get(key)
    // Offline: the reconnect (`_connected`) picks it up instead.
    if (s && s.connected && !s.queueDispatchedId && (s.queueStartWhenFree || (!s.queuePaused && s.queueAutoMode))) {
      tryDispatchNext(key, { force: s.queueStartWhenFree })
    }
  }, 800))
}

/** Lanes whose forced run began with a step (a fresh start): when the step
 *  completes, the task behind it is forced through too. */
const _forceThrough = new Set<string>()

// Helpers that update a session inside the map
function updateSession(containerId: string, patch: Partial<ChatSession>) {
  let kick = false
  let freed = false
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, ...patch }
    // Whatever frees the agent (the stopped turn's reply, "ready", a dropped
    // socket) also ends a Stop that was still landing.
    if (!updated.busy && updated.stopping) updated.stopping = false
    // The one place every "agent is free again" passes through (status idle,
    // a reply, an error, a stop that landed, a command result): the queue
    // may go on.
    freed = session.busy && updated.busy === false
    kick = freed && !updated.queueDispatchedId && (updated.queueAutoMode || updated.queueStartWhenFree)
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    // If any persisted-plan field changed, mirror to localStorage.
    if (
      'planState' in patch ||
      'planningEnabled' in patch ||
      'planLevel' in patch ||
      'planCardCollapsed' in patch
    ) {
      savePlanSlice(containerId, {
        planState: updated.planState,
        planningEnabled: updated.planningEnabled,
        planLevel: updated.planLevel,
        planCardCollapsed: updated.planCardCollapsed,
      })
    }
    // Mirror queue changes so they survive a page refresh. Auto-mode flag
    // travels with the queue since it changes the UX meaningfully, and so do
    // Pause and the power-off park.
    if ('queue' in patch || 'queueAutoMode' in patch || 'queuePaused' in patch || 'queueParked' in patch) {
      saveQueueSlice(containerId, {
        queue: updated.queue,
        queueAutoMode: updated.queueAutoMode,
        queuePaused: updated.queuePaused,
        parked: updated.queueParked,
      })
    }
    return { sessions }
  })
  // Free by the agent's word: the fallback for "busy on our word" is moot.
  if (freed) releaseBusyHold(containerId)
  if (kick) scheduleQueueKick(containerId)
}

// Today's allowance came back (the day turned over, a free reset): queues the
// refusal held go on — Auto lanes by themselves, step-by-step lanes only with
// what the user had asked to start.
let _quotaWasBlocked = (() => {
  const q = useQuotaStore.getState().status
  return !!q?.blocked && !q.exempt
})()
useQuotaStore.subscribe((st) => {
  // Signing out clears the status (quotaStore's teardown) while the agent
  // sockets are still open: that is no allowance coming back, and nothing may
  // go out as the user leaves.
  if (!st.status || useAuthStore.getState().signingOut) {
    _quotaWasBlocked = false
    return
  }
  const blocked = !!st.status.blocked && !st.status.exempt
  const lifted = _quotaWasBlocked && !blocked
  _quotaWasBlocked = blocked
  if (blocked) return
  for (const [key, s] of useChatStore.getState().sessions) {
    if (!s.queueDispatchedId) {
      if (lifted && (s.queueStartWhenFree || (!s.queuePaused && s.queueAutoMode))) {
        tryDispatchNext(key, { force: s.queueStartWhenFree })
      }
      continue
    }
    // A task the allowance refused, on an Auto lane: it goes again whenever
    // the allowance reads as back — not only at the moment it lifts, which
    // may have passed while the page was closed. (Paused or step by step:
    // when you resume or switch Auto on.)
    if (s.queueAutoMode && !s.queuePaused) retryQuotaHold(key)
  }
})

function updatePlan(
  containerId: string,
  reducer: (prev: PlanState | null) => PlanState | null,
) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const next = reducer(session.planState)
    if (next === session.planState) return state
    const updated = { ...session, planState: next }
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    savePlanSlice(containerId, {
      planState: next,
      planningEnabled: updated.planningEnabled,
      planLevel: updated.planLevel,
      planCardCollapsed: updated.planCardCollapsed,
    })
    return { sessions }
  })
}

function clearMessages(containerId: string) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, messages: [], nextStepOptions: [] }
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    return { sessions }
  })
}

// ── Topics: the continuous timeline's dividers ──

/** A replayed session message (user/assistant turn or a monitor card) as a
 *  transcript message flagged `replay`. */
function replayToMessage(d: Record<string, unknown>): ChatMessage {
  const type = d.type as string
  const role = d.role as string || ''
  const rawContent = (d.content as string) || ''
  if (type === 'monitor') {
    return {
      id: nextId(),
      role: 'tool' as const,
      content: (d.output as string) || '',
      timestamp: new Date().toISOString(),
      replay: true,
      tool_name: d.tool_name as string,
      tool_arguments: d.arguments as Record<string, unknown>,
      tool_output: (d.output as string) || '',
    }
  }
  // A mirrored automation result: the note line ahead of it ("[Autonomous
  // Work — from the AUTO lane]") is for the model; the chat shows the result
  // as the live frame did, so it also matches its row in the log.
  const automation = String(d.automation_label || '')
  let content = rawContent
  if (automation) {
    const nl = content.indexOf('\n')
    const head = nl < 0 ? '' : content.slice(0, nl)
    if (head.startsWith(`[${automation} — `) && head.endsWith(']')) content = content.slice(nl + 1)
  }
  return {
    id: nextId(),
    role: role as ChatMessage['role'],
    content: role === 'assistant' ? sanitizeAgentContent(content) : content,
    timestamp: (d.timestamp as string) || new Date().toISOString(),
    replay: true,
    model: (d.model as string) || '',
    ...(taskMark(d.task) ? { task: taskMark(d.task) } : {}),
    ...(automation ? { automation } : {}),
  }
}

/** Index of the last divider, or of the last divider for `agentSessionId`
 *  when given; -1 when there is none. */
function lastTopicIndex(messages: ChatMessage[], agentSessionId?: string): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = messages[i].topic
    if (t && (!agentSessionId || t.agentSessionId === agentSessionId)) return i
  }
  return -1
}

function topicLabel(action: TopicMark['action'], name: string): string {
  if (action === 'clear') return 'Context cleared'
  if (action === 'resume') {
    // Agent sessions are mostly named by default ("web-session", lanes):
    // only a name somebody chose is worth repeating.
    const n = name.trim()
    const generic = !n || n === 'web-session' || n === 'default' || /^lane-/i.test(n)
    return generic ? 'Back on an earlier topic' : `Back on: ${n}`
  }
  return 'New topic'
}

/** Mark that the lane's context is now on agent session `id`.
 *  'inserted' — a divider was added; 'present' — the transcript already
 *  ends on a divider for that session; 'deferred' — the log isn't loaded
 *  yet (reconcileTopic adds the divider once it is); 'none' — no session. */
function markTopic(
  key: string,
  t: { id: string; name: string; action: TopicMark['action']; source: TopicMark['source'] },
): 'inserted' | 'present' | 'deferred' | 'none' {
  const s = useChatStore.getState().sessions.get(key)
  if (!s) return 'none'
  if (useAuthStore.getState().authEnabled && !s.historyLoaded) {
    if (s.agentSessionId !== t.id) updateSession(key, { agentSessionId: t.id })
    return 'deferred'
  }
  const last = lastTopicIndex(s.messages)
  if (t.action !== 'clear' && last >= 0 && s.messages[last].topic?.agentSessionId === t.id) {
    if (s.agentSessionId !== t.id) updateSession(key, { agentSessionId: t.id })
    return 'present'
  }
  // A topic resumed or re-entered keeps the name it already has.
  const known = s.messages.find((m) => m.topic?.agentSessionId === t.id && m.topic.title)?.topic?.title
  const topic: TopicMark = {
    action: t.action, agentSessionId: t.id, source: t.source, ...(known ? { title: known } : {}),
  }
  addMessage(key, {
    id: nextId(), role: 'system', content: topicLabel(t.action, t.name),
    timestamp: new Date().toISOString(), topic,
  })
  updateSession(key, { agentSessionId: t.id, nextStepOptions: [] })
  return 'inserted'
}

/** The agent named session `agentSessionId`: every loaded divider of that
 *  session without a title gets it (a title the user typed stays), and the
 *  persisted ones are renamed in the log. */
function applyTopicTitle(key: string, agentSessionId: string, title: string) {
  const clean = title.replace(/\s+/g, ' ').trim().slice(0, 120)
  if (!clean) return
  const rowIds: number[] = []
  useChatStore.setState((state) => {
    const s = state.sessions.get(key)
    if (!s) return state
    let changed = false
    const messages = s.messages.map((m) => {
      if (!m.topic || m.topic.agentSessionId !== agentSessionId || m.topic.title) return m
      changed = true
      if (m.rowId !== undefined) rowIds.push(m.rowId)
      return { ...m, topic: { ...m.topic, title: clean } }
    })
    if (!changed) return state
    const sessions = new Map(state.sessions)
    sessions.set(key, { ...s, messages })
    return { sessions }
  })
  for (const rowId of rowIds) void serverSetTopicTitle(key, rowId, clean)
}

async function serverSetTopicTitle(sessionId: string, rowId: number, title: string): Promise<void> {
  if (!useAuthStore.getState().authEnabled) return
  try {
    await fetch(`/fd/chat/sessions/${encodeURIComponent(sessionId)}/topics/${rowId}`, {
      method: 'PATCH',
      headers: _chatHeaders(),
      credentials: 'include',
      body: JSON.stringify({ title }),
    })
  } catch { /* ignore */ }
}

/** With the log loaded and the agent's session known: when the transcript
 *  doesn't end on a divider for that session (started on another channel,
 *  after a restart, or before dividers existed), add one. */
function reconcileTopic(key: string) {
  if (!useAuthStore.getState().authEnabled) return
  const s = useChatStore.getState().sessions.get(key)
  if (!s || !s.historyLoaded || !s.agentSessionId) return
  // A window into the past says nothing about where the transcript ends.
  if (s.hasNewer) return
  markTopic(key, { id: s.agentSessionId, name: '', action: 'new', source: 'agent' })
}

/** Keep the window within WINDOW_MAX rows by dropping persisted rows from
 *  the far end. `keep: 'newest'` drops from the top (the chat grew at the
 *  bottom, or the reader moved forward) — never the current session's last
 *  divider or anything after it, which the replay merge is anchored on.
 *  `keep: 'oldest'` drops from the bottom (the reader paged back) — never a
 *  row not yet persisted — and marks that newer rows exist. */
function trimWindow(s: ChatSession, keep: 'newest' | 'oldest'): ChatSession {
  const n = s.messages.length
  if (n <= WINDOW_MAX) return s
  const excess = n - (WINDOW_MAX - WINDOW_SLACK)
  if (keep === 'newest') {
    const anchor = s.hasNewer ? -1 : lastTopicIndex(s.messages, s.agentSessionId || undefined)
    let drop = 0
    while (drop < excess && s.messages[drop].rowId !== undefined && (anchor < 0 || drop < anchor)) drop++
    if (drop === 0) return s
    return { ...s, messages: s.messages.slice(drop), hasOlder: true }
  }
  const kept: ChatMessage[] = []
  let dropped = 0
  for (let i = n - 1; i >= 0; i--) {
    const m = s.messages[i]
    if (dropped < excess && m.rowId !== undefined) { dropped++; continue }
    kept.push(m)
  }
  if (dropped === 0) return s
  kept.reverse()
  const endRowId = kept.reduce((mx, m) => Math.max(mx, m.rowId ?? 0), 0)
  return { ...s, messages: kept, hasNewer: true, windowEndRowId: endRowId }
}

/** What identifies a turn when matching the agent's replay against the log:
 *  the role and the words (whitespace folded, attached-image lines dropped —
 *  the agent words those itself). */
function turnKey(m: ChatMessage): string {
  const text = m.content
    .split('\n')
    .filter((line) => !/^\[Attached image:\s*.+\]$/.test(line.trim()))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 400)
  return `${m.role}\n${text}`
}

/** The turns of the agent's replay of a session (the lane's current one
 *  unless `sid` names another) that the loaded log is missing: user/assistant
 *  turns it doesn't have since that session's divider (or anywhere in the
 *  window when it has none). Replays never add tool cards. */
function replayTurnsToAdd(key: string, replayed: ChatMessage[], sid?: string): ChatMessage[] {
  const s = useChatStore.getState().sessions.get(key)
  if (!s) return []
  // A window into the past doesn't hold the current session: merging would
  // re-add (and re-persist) turns the log already has further down.
  if (s.hasNewer) return []
  const start = lastTopicIndex(s.messages, (sid ?? s.agentSessionId) || undefined)
  const seen = new Set<string>()
  for (let i = start + 1; i < s.messages.length; i++) {
    const m = s.messages[i]
    if (m.role === 'user' || m.role === 'assistant') seen.add(turnKey(m))
  }
  const out: ChatMessage[] = []
  for (const m of replayed) {
    if (m.role !== 'user' && m.role !== 'assistant') continue
    const k = turnKey(m)
    if (seen.has(k)) continue
    seen.add(k)
    out.push(m)
  }
  return out
}

/** Fold the agent's replay of its current session into the loaded log: the
 *  turns it is missing are appended as persisted rows, so the log heals. */
function mergeReplay(key: string, replayed: ChatMessage[], sid?: string) {
  for (const m of replayTurnsToAdd(key, replayed, sid)) addMessage(key, { ...m, replay: false })
}

export interface TopicInfo {
  /** Log row id of the divider (absent for one not persisted yet). */
  rowId?: number
  /** The divider's message id in the loaded window, when it is there. */
  messageId?: string
  agentSessionId: string
  action: TopicMark['action']
  title: string
  startedAt: string
  firstUserMessage: string
  messageCount: number
}

/** Topics from the dividers in the loaded window. */
function topicsFromMessages(messages: ChatMessage[]): TopicInfo[] {
  const out: TopicInfo[] = []
  let cur: TopicInfo | null = null
  for (const m of messages) {
    if (m.topic) {
      cur = {
        rowId: m.rowId, messageId: m.id, agentSessionId: m.topic.agentSessionId,
        action: m.topic.action, title: m.topic.title || '', startedAt: m.timestamp,
        firstUserMessage: '', messageCount: 0,
      }
      out.push(cur)
      continue
    }
    if (!cur) continue
    if (m.role === 'user' || m.role === 'assistant') {
      // A typed slash command is a control line, not a turn.
      if (m.role === 'user' && m.content.trimStart().startsWith('/')) continue
      cur.messageCount++
      if (m.role === 'user' && !cur.firstUserMessage) cur.firstUserMessage = m.content
    }
  }
  return out
}

/** The lane's topics, oldest first: from the log when it is loaded (so
 *  topics beyond the loaded window are listed too), plus any divider only
 *  this tab holds so far; otherwise from the loaded messages. */
export async function fetchTopics(key: string): Promise<TopicInfo[]> {
  const s = useChatStore.getState().sessions.get(key)
  if (!s) return []
  const local = topicsFromMessages(s.messages)
  if (!useAuthStore.getState().authEnabled || !s.historyLoaded) return local
  try {
    const res = await fetch(`/fd/chat/sessions/${encodeURIComponent(key)}/topics`, {
      headers: _chatHeaders(), credentials: 'include',
    })
    if (!res.ok) return local
    const rows = await res.json() as Array<Record<string, unknown>>
    const byRow = new Map(local.filter((t) => t.rowId !== undefined).map((t) => [t.rowId, t] as const))
    const listed = rows.map((r): TopicInfo => {
      const rowId = Number(r.row_id)
      return {
        rowId, messageId: byRow.get(rowId)?.messageId,
        agentSessionId: String(r.agent_session_id ?? ''),
        action: (String(r.action ?? 'new') as TopicMark['action']),
        title: String(r.title ?? ''), startedAt: String(r.started_at ?? ''),
        firstUserMessage: String(r.first_user_message ?? ''),
        messageCount: Number(r.message_count ?? 0),
      }
    })
    const known = new Set(listed.map((t) => t.rowId))
    return [...listed, ...local.filter((t) => t.rowId === undefined || !known.has(t.rowId))]
  } catch { return local }
}

function addMessage(containerId: string, msg: ChatMessage) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    // A chat left open for days keeps growing at the bottom: shed the top.
    const updated = trimWindow({ ...session, messages: [...session.messages, msg] }, 'newest')
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    return { sessions }
  })
  // Persist to server (skip replayed messages — they're already on the server)
  if (!msg.replay) {
    queueMessagePersist(containerId, msg)
  }
}
