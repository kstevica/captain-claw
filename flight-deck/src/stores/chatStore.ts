import { create } from 'zustand'
import { AgentChatWS, type ChatMessage } from '../services/agentChat'
import { useAuthStore } from './authStore'
import { useContainerStore } from './containerStore'
import { useLocalAgentStore } from './localAgentStore'
import { useProcessStore } from './processStore'
import { useTraceStore } from './traceStore'
import type { TraceSpan } from '../types'
import { sanitizeAgentContent } from '../utils/sanitizeAgentContent'

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

async function serverLoadMessages(sessionId: string): Promise<ChatMessage[]> {
  try {
    const res = await fetch(`/fd/chat/sessions/${encodeURIComponent(sessionId)}/messages?limit=500`, {
      headers: _chatHeaders(),
      credentials: 'include',
    })
    if (!res.ok) return []
    const rows = await res.json() as { role: string; content: string; metadata: string; created_at: string }[]
    return rows.map((r, i) => {
      let meta: Record<string, unknown> = {}
      try { meta = JSON.parse(r.metadata || '{}') } catch { /* ignore */ }
      const role = r.role as ChatMessage['role']
      return {
        id: `hist-${i}-${Date.now()}`,
        role,
        content: role === 'assistant' ? sanitizeAgentContent(r.content) : r.content,
        timestamp: r.created_at,
        replay: true,
        tool_name: meta.tool_name as string | undefined,
        tool_arguments: meta.tool_arguments as Record<string, unknown> | undefined,
        tool_output: meta.tool_output as string | undefined,
        model: meta.model as string | undefined,
        peer_name: meta.peer_name as string | undefined,
      }
    })
  } catch { return [] }
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
  return `fd.plan.${containerId}`
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
}

function _queueLSKey(containerId: string): string {
  return `fd.queue.${containerId}`
}

function saveQueueSlice(containerId: string, slice: PersistedQueueSlice): void {
  try {
    if (slice.queue.length === 0 && !slice.queueAutoMode) {
      window.localStorage.removeItem(_queueLSKey(containerId))
      return
    }
    window.localStorage.setItem(_queueLSKey(containerId), JSON.stringify(slice))
  } catch { /* quota / private mode — ignore */ }
}

function loadQueueSlice(containerId: string): PersistedQueueSlice | null {
  try {
    const raw = window.localStorage.getItem(_queueLSKey(containerId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as PersistedQueueSlice
    if (!parsed || !Array.isArray(parsed.queue)) return null
    // Downgrade any in-flight items back to pending — after a refresh we
    // have no reliable signal that the agent's reply was received, so
    // re-dispatch is safer than abandoning the task.
    parsed.queue = parsed.queue.map((q) =>
      q.status === 'dispatched' ? { ...q, status: 'pending', dispatchedAt: undefined } : q,
    )
    return parsed
  } catch {
    return null
  }
}

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
  for (const [sessionId, msgs] of batches) {
    try {
      await fetch(`/fd/chat/sessions/${encodeURIComponent(sessionId)}/messages`, {
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
            }),
          })),
        }),
      })
    } catch { /* ignore */ }
  }
}

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
  if (lastChar === '?') return false
  // Confirmation-seeking phrases. Only treat as "waiting" if the message is
  // reasonably short — a 5KB report that mentions "let me know if" is fine.
  const lower = trimmed.toLowerCase()
  const SHORT_LIMIT = 1200
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

interface ChatSession {
  containerId: string
  containerName: string
  host: string
  port: number
  auth: string
  ws: AgentChatWS
  messages: ChatMessage[]
  connected: boolean
  busy: boolean // agent is processing
  statusText: string
  models: AgentModelInfo[]
  personalities: AgentPersonalityInfo[]
  activeModel: string
  activePersonality: string
  // Token generation speed tracking
  _busyStartedAt: number // timestamp (ms) when agent became busy
  lastTokPerSec: number  // last wall-clock generation speed (tok/s)
  avgTokPerSec: number   // running average wall-clock tok/s
  _tokSamples: number    // number of samples for avg calculation
  llmTokPerSec: number   // real LLM generation speed (completion_tokens / llm_latency)
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
}

interface ChatStore {
  sessions: Map<string, ChatSession>
  activeChatId: string | null
  chatOpen: boolean
  // Fullscreen mode: hides sidebar/director/main/tool panels and gives the
  // chat panel the full width. The queue panel is only available here.
  chatFullscreen: boolean

  openChat: (id: string, name: string, host: string, port: number, auth: string) => void
  closeChat: () => void
  switchChat: (containerId: string) => void
  disconnectChat: (containerId: string) => void
  sendMessage: (containerId: string, content: string) => void
  sendBtw: (containerId: string, content: string) => void
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
  markQueueItemDone: (containerId: string, id: string) => void
  toggleQueueAutoMode: (containerId: string) => void
  clearQueue: (containerId: string) => void
}

let msgCounter = 0
function nextId() { return `msg-${Date.now()}-${++msgCounter}` }

export const useChatStore = create<ChatStore>((set, get) => ({
  sessions: new Map(),
  activeChatId: null,
  chatOpen: false,
  chatFullscreen: false,

  openChat: (containerId, containerName, host, port, auth) => {
    const existing = get().sessions.get(containerId)
    if (existing) {
      // Already have a session, just activate it
      if (!existing.connected) {
        // Clear old replay messages — agent will resend them on reconnect
        const kept = existing.messages.filter((m) => !m.replay)
        if (kept.length !== existing.messages.length) {
          updateSession(containerId, { messages: kept })
        }
        existing.ws.connect()
      }
      set({ activeChatId: containerId, chatOpen: true })
      return
    }

    const ws = new AgentChatWS(containerId, host, port, auth)
    // Re-hydrate any persisted plan slice so the PlanCard reappears instantly
    // after a page refresh. Live ws events will keep updating it from here.
    const persisted = loadPlanSlice(containerId)
    const persistedQueue = loadQueueSlice(containerId)
    const session: ChatSession = {
      containerId,
      containerName,
      host,
      port,
      auth,
      ws,
      messages: [],
      connected: false,
      busy: false,
      statusText: '',
      models: [],
      personalities: [],
      activeModel: '',
      activePersonality: '',
      _busyStartedAt: 0,
      lastTokPerSec: 0,
      avgTokPerSec: 0,
      _tokSamples: 0,
      llmTokPerSec: 0,
      planningEnabled: persisted?.planningEnabled ?? false,
      planLevel: persisted?.planLevel ?? 'plain',
      planState: persisted?.planState ?? null,
      planCardCollapsed: persisted?.planCardCollapsed ?? false,
      nextStepOptions: [],
      queue: persistedQueue?.queue ?? [],
      queueAutoMode: persistedQueue?.queueAutoMode ?? false,
      queueDispatchedId: null,
    }

    const sessions = new Map(get().sessions)
    sessions.set(containerId, session)
    set({ sessions, activeChatId: containerId, chatOpen: true })

    // Server-side chat persistence: upsert session & load history
    if (useAuthStore.getState().authEnabled) {
      serverUpsertSession(containerId, containerId, containerName).then(() =>
        serverLoadMessages(containerId).then((history) => {
          if (history.length > 0) {
            useChatStore.setState((state) => {
              const s = state.sessions.get(containerId)
              if (!s) return state
              // Only prepend history if session still has no real messages (avoid duplication)
              if (s.messages.length > 0 && !s.messages[0].replay) return state
              const merged = [...history, ...s.messages.filter((m) => !m.replay)]
              const updated = { ...s, messages: merged }
              const newSessions = new Map(state.sessions)
              newSessions.set(containerId, updated)
              return { sessions: newSessions }
            })
          }
        })
      )
    }

    // Wire up event handlers
    ws.on('_connected', () => {
      updateSession(containerId, { connected: true })
      // If we hydrated a queue with pending items and auto-mode is on, kick
      // off the first dispatch now that the socket is live.
      const cur = useChatStore.getState().sessions.get(containerId)
      if (cur && cur.queueAutoMode && !cur.queueDispatchedId) {
        tryDispatchNext(containerId)
      }
    })

    ws.on('_disconnected', () => {
      updateSession(containerId, { connected: false, busy: false, statusText: '' })
      // A dropped socket means we won't see the next_steps event. Cancel
      // any pending auto-mark — when the agent reconnects, replay or fresh
      // events will drive the queue forward again.
      cancelAutoCompleteWatch(containerId)
    })

    ws.on('welcome', (data) => {
      const sessionInfo = data.session as Record<string, unknown> | undefined
      const name = sessionInfo?.name as string || ''
      const models = (data.models as AgentModelInfo[] || [])
      const personalities = (data.personalities as AgentPersonalityInfo[] || [])
      const patch: Partial<ChatSession> = { models, personalities }
      if (name) patch.statusText = `Session: ${name}`
      updateSession(containerId, patch)

      // Send peer agent awareness to the connected agent
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
      ws.sendJSON({ type: 'peer_agents', agents: peers, self: selfIdentity, fd_url: fdUrl })
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
      console.debug('[chatStore] next_steps received', { containerId, rawCount: raw.length, validCount: options.length, options })
      updateSession(containerId, { nextStepOptions: options })
      // next_steps is the explicit "agent finished its turn" signal — fire
      // any armed auto-complete watcher now instead of waiting for the
      // fallback timer.
      fireAutoCompleteWatch(containerId)
    })

    ws.on('chat_message', (data) => {
      // Skip echoed user messages — we already add them locally in sendMessage
      if (data.role === 'user' && !data.replay) return
      const rawContent = (data.content as string) || ''
      // Sanitize assistant content: strip thinking blocks, council protocol
      // headers (SUITABILITY/ACTION/TARGET), insight echoes, and other
      // instruction/memory leaks. User content is left as-is.
      const cleanContent = data.role === 'assistant' ? sanitizeAgentContent(rawContent) : rawContent
      const msg: ChatMessage = {
        id: nextId(),
        role: data.role as 'user' | 'assistant',
        content: cleanContent,
        timestamp: data.timestamp as string || new Date().toISOString(),
        replay: data.replay as boolean || false,
        model: data.model as string || '',
      }
      addMessage(containerId, msg)
      if (data.role === 'assistant' && !data.replay) {
        updateSession(containerId, { busy: false, statusText: '' })
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
        const cur = useChatStore.getState().sessions.get(containerId)
        if (cur && cur.queueAutoMode && cur.queueDispatchedId) {
          const dispatchedId = cur.queueDispatchedId
          if (isLikelyStall(cleanContent)) {
            const entry = _stallNudges.get(containerId)
            const used = entry && entry.itemId === dispatchedId ? entry.count : 0
            if (used < MAX_STALL_NUDGES) {
              _stallNudges.set(containerId, { itemId: dispatchedId, count: used + 1 })
              // Small delay so the nudge lands after any trailing events
              // (next_steps, status) from the same turn.
              setTimeout(() => {
                const s = useChatStore.getState().sessions.get(containerId)
                // Bail if user already moved on (toggled auto off, removed
                // the item, or marked it done manually).
                if (!s || !s.queueAutoMode) return
                if (s.queueDispatchedId !== dispatchedId) return
                if (s.busy) return
                useChatStore.getState().sendMessage(containerId, 'Continue the task.')
              }, 800)
            } else {
              // Out of nudges — treat as completion so the queue moves on.
              armAutoCompleteWatch(containerId, dispatchedId)
            }
          } else if (isLikelyTaskComplete(cleanContent)) {
            resetStallNudges(containerId)
            armAutoCompleteWatch(containerId, dispatchedId)
          }
          // else: question — do nothing, user will answer.
        }
      }
    })

    ws.on('replay_batch', (data) => {
      const messages = (data.messages as Array<Record<string, unknown>>) || []
      if (messages.length === 0) return
      // Clear old replay messages before adding new batch (prevents piling on reconnect)
      useChatStore.setState((state) => {
        const s = state.sessions.get(containerId)
        if (!s) return state
        const kept = s.messages.filter((m) => !m.replay)
        const replayed: ChatMessage[] = messages.map((d) => {
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
          return {
            id: nextId(),
            role: role as ChatMessage['role'],
            content: role === 'assistant' ? sanitizeAgentContent(rawContent) : rawContent,
            timestamp: (d.timestamp as string) || new Date().toISOString(),
            replay: true,
            model: (d.model as string) || '',
          }
        })
        const updated = { ...s, messages: [...replayed, ...kept] }
        const newSessions = new Map(state.sessions)
        newSessions.set(containerId, updated)
        return { sessions: newSessions }
      })
    })

    ws.on('replay_done', () => {
      updateSession(containerId, { busy: false, statusText: '' })
    })

    ws.on('status', (data) => {
      const text = data.text as string || data.status as string || ''
      // "ready", "idle", or empty status means agent is done
      const idle = !text || /^(ready|idle|done|completed)$/i.test(text)
      const patch: Partial<ChatSession> = { busy: !idle, statusText: idle ? '' : text }
      // Record when agent first becomes busy (for tok/s calculation)
      if (!idle) {
        const s = useChatStore.getState().sessions.get(containerId)
        if (s && !s.busy) patch._busyStartedAt = Date.now()
      }
      updateSession(containerId, patch)
    })

    ws.on('narration', (data) => {
      // Live between-step narration — show it in the flow as a subtle blurb
      // while a long task runs (same as WhatsApp/glasses).
      const text = String(data.text || '').trim()
      if (!text) return
      addMessage(containerId, {
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
        const msg: ChatMessage = {
          id: nextId(),
          role: 'tool',
          content: output,
          timestamp: new Date().toISOString(),
          tool_name: toolName,
          tool_arguments: data.arguments as Record<string, unknown>,
          tool_output: output,
        }
        addMessage(containerId, msg)
      }
      updateSession(containerId, { busy: true, statusText: `Using ${toolName}...` })
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
      addMessage(containerId, msg)
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
      addMessage(containerId, msg)
    })

    ws.on('error', (data) => {
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.message as string || 'Unknown error',
        timestamp: new Date().toISOString(),
      }
      addMessage(containerId, msg)
      updateSession(containerId, { busy: false })
    })

    ws.on('command_result', (data) => {
      const command = (data.command as string || '').trim().toLowerCase()
      // If /clear was executed, wipe local messages first
      if (command === '/clear') {
        clearMessages(containerId)
      }
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.content as string || '',
        timestamp: new Date().toISOString(),
      }
      addMessage(containerId, msg)
      // Slash commands run synchronously in the handler — once the
      // command_result arrives the agent is idle again. Without this
      // the "Thinking..." spinner stays pinned after `/plan`, `/help`,
      // `/clear`, etc., because no later token/usage event clears busy.
      updateSession(containerId, { busy: false, statusText: '' })
    })

    // Forward orchestrator trace spans to the trace store for
    // real-time observability in the TraceTimeline component.
    // Per-task lifecycle events also drive the PlanCard live monitor.
    ws.on('orchestrator_event', (data) => {
      const event = String(data.event ?? '')

      if (event === 'trace_span') {
        const span = data as unknown as TraceSpan
        if (span?.span_id) {
          useTraceStore.getState().handleSpanEvent(containerId, span)
        }
        return
      }

      // Plan-mode live monitor: only meaningful while a plan is active.
      const taskId = String(data.task_id ?? '')
      if (!taskId) return

      switch (event) {
        case 'task_started':
          updatePlan(containerId, (prev) => {
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
          updatePlan(containerId, (prev) => {
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
          updatePlan(containerId, (prev) => {
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
          updatePlan(containerId, (prev) => {
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
          updatePlan(containerId, (prev) => {
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
          updatePlan(containerId, (prev) => {
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
      updatePlan(containerId, () => ({
        startedAt: new Date().toISOString(),
        startedAtMs: Date.now(),
        status: 'running',
        maxRevisions: Number(data.max_revisions ?? 0),
        steps,
        revisions: [],
      }))
      // Auto-expand the plan card when a new plan starts
      updateSession(containerId, { planCardCollapsed: false })
    })

    ws.on('plan_orchestrate_expanded', (data) => {
      const expansions = (data.expansions as Array<Record<string, unknown>>) || []
      if (expansions.length === 0) return
      updatePlan(containerId, (prev) => {
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
      updatePlan(containerId, (prev) => {
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
      updatePlan(containerId, (prev) => {
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
      updatePlan(containerId, (prev) => {
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
      updatePlan(containerId, (prev) => {
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
      updatePlan(containerId, (prev) => {
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
        updateSession(containerId, { planningEnabled: true })
      } else if (/auto-rout(?:e|ing)\s+disabled|auto-rout(?:e|ing):\s*off\b|currently\s+off\b/.test(content)) {
        updateSession(containerId, { planningEnabled: false })
      }
      // Plan-mode level confirmations:
      //   "Plan-mode level set to **<name>**."         (after `/planning level <name>`)
      //   "Plan-mode level: **<name>**. ..."           (after bare `/planning level`)
      //   "Plan-mode auto-routing: **on** (level: **<name>**). ..."
      //                                                (after bare `/planning`)
      // Match all three with one regex over markdown-stripped content.
      const lvlMatch = content.match(/plan-mode\s+(?:level\s+set\s+to|level:|auto-rout(?:e|ing):[^()]*\(level:)\s*(plain|enriched|insightful|complete)\b/)
      if (lvlMatch) {
        updateSession(containerId, { planLevel: lvlMatch[1] })
      }
    })

    // Token generation speed tracking from usage events
    ws.on('usage', (data) => {
      const last = data.last as Record<string, number> | undefined
      if (!last) return
      const completionTokens = last.completion_tokens || 0
      if (completionTokens <= 0) return
      const s = useChatStore.getState().sessions.get(containerId)
      if (!s || !s._busyStartedAt) return
      const elapsedSec = (Date.now() - s._busyStartedAt) / 1000
      if (elapsedSec <= 0) return
      const tokPerSec = completionTokens / elapsedSec
      const samples = s._tokSamples + 1
      const newAvg = (s.avgTokPerSec * s._tokSamples + tokPerSec) / samples
      // Real LLM speed from backend latency_ms (pure API call time)
      const latencyMs = last.latency_ms || 0
      const llmTokPerSec = latencyMs > 0 ? (completionTokens / latencyMs) * 1000 : 0
      updateSession(containerId, {
        lastTokPerSec: tokPerSec,
        avgTokPerSec: newAvg,
        _tokSamples: samples,
        _busyStartedAt: 0,
        llmTokPerSec,
      })
    })

    ws.connect()
  },

  closeChat: () => {
    set({ chatOpen: false })
  },

  switchChat: (containerId) => {
    set({ activeChatId: containerId, chatOpen: true })
  },

  disconnectChat: (containerId) => {
    const session = get().sessions.get(containerId)
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

  sendMessage: (containerId, content) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    // Add user message to local state
    const msg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }
    addMessage(containerId, msg)
    updateSession(containerId, {
      busy: true,
      statusText: 'Thinking...',
      _busyStartedAt: Date.now(),
      nextStepOptions: [],
    })
    session.ws.send(content)
  },

  sendBtw: (containerId, content) => {
    const session = get().sessions.get(containerId)
    if (session) session.ws.sendBtw(content)
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
    if (session) {
      session.ws.cancel()
      updateSession(containerId, { busy: false, statusText: 'Cancelled' })
    }
  },

  setModel: (containerId, selector) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.sendJSON({ type: 'set_model', selector })
      updateSession(containerId, { activeModel: selector })
    }
  },

  setPersonality: (containerId, personalityId) => {
    const session = get().sessions.get(containerId)
    if (session) {
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
    if (!session) return
    // Optimistic update — confirmed/corrected by command_result handler.
    updateSession(containerId, { planningEnabled: enabled })
    // Send as a regular chat message so it routes through the slash command
    // handler (which also returns a command_result we listen to).
    session.ws.send(enabled ? '/planning on' : '/planning off')
  },

  setPlanLevel: (containerId, level) => {
    const session = get().sessions.get(containerId)
    if (!session) return
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
      resetStallNudges(containerId)
    }
    updateSession(containerId, patch)
    if (session.queueDispatchedId === id) tryDispatchNext(containerId)
  },

  markQueueItemDone: (containerId, id) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    const queue = session.queue.map((q) =>
      q.id === id && q.status !== 'done'
        ? { ...q, status: 'done' as const, completedAt: Date.now() }
        : q,
    )
    const patch: Partial<ChatSession> = { queue }
    if (session.queueDispatchedId === id) {
      patch.queueDispatchedId = null
      cancelAutoCompleteWatch(containerId)
      resetStallNudges(containerId)
    }
    updateSession(containerId, patch)
    tryDispatchNext(containerId)
  },

  toggleQueueAutoMode: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    const next = !session.queueAutoMode
    updateSession(containerId, { queueAutoMode: next })
    // Turning auto-mode off cancels any pending watch so it doesn't fire
    // after the user has switched to manual.
    if (!next) cancelAutoCompleteWatch(containerId)
    // Turning auto-mode on while idle should kick the queue.
    if (next) tryDispatchNext(containerId)
  },

  clearQueue: (containerId) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    // Drop pending items; leave the in-flight one alone so we don't lose
    // track of the agent's current task.
    const queue = session.queue.filter((q) => q.status !== 'pending')
    updateSession(containerId, { queue })
  },
}))

// ── Queue dispatch ──
//
// Pulls the next pending item and pushes it through sendMessage so it appears
// in chat history exactly like a user-typed message. Guarded by:
//  - no item currently dispatched (queueDispatchedId === null)
//  - session not busy (don't pile prompts on top of an in-flight one)
//  - at least one pending item to send
function tryDispatchNext(containerId: string) {
  const state = useChatStore.getState()
  const session = state.sessions.get(containerId)
  if (!session) return
  if (session.queueDispatchedId) return
  if (session.busy) return
  if (!session.connected) return
  const next = session.queue.find((q) => q.status === 'pending')
  if (!next) return
  const queue = session.queue.map((q) =>
    q.id === next.id
      ? { ...q, status: 'dispatched' as const, dispatchedAt: Date.now() }
      : q,
  )
  updateSession(containerId, { queue, queueDispatchedId: next.id })
  state.sendMessage(containerId, next.content)
}

// Helpers that update a session inside the map
function updateSession(containerId: string, patch: Partial<ChatSession>) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, ...patch }
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
    // travels with the queue since it changes the UX meaningfully.
    if ('queue' in patch || 'queueAutoMode' in patch) {
      saveQueueSlice(containerId, {
        queue: updated.queue,
        queueAutoMode: updated.queueAutoMode,
      })
    }
    return { sessions }
  })
}

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

function addMessage(containerId: string, msg: ChatMessage) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, messages: [...session.messages, msg] }
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    return { sessions }
  })
  // Persist to server (skip replayed messages — they're already on the server)
  if (!msg.replay) {
    queueMessagePersist(containerId, msg)
  }
}
