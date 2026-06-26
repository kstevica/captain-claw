import { create } from 'zustand'
import { AgentChatWS } from '../services/agentChat'
import { useAuthStore, refreshAccessToken } from './authStore'
import { useContainerStore } from './containerStore'
import { useProcessStore } from './processStore'
import { uploadFileToAgent, transferFile } from '../services/fileTransfer'
import { sanitizeAgentContent, stripThinkingBlocks as sharedStripThinkingBlocks } from '../utils/sanitizeAgentContent'

// Instruction templates (raw text imports)
import btwTemplate from '../instructions/council/btw_context.md?raw'
import turnTemplate from '../instructions/council/turn_prompt.md?raw'
import moderatorSelectTemplate from '../instructions/council/moderator_select.md?raw'
import synthesisTemplate from '../instructions/council/synthesis.md?raw'
import voteTemplate from '../instructions/council/vote.md?raw'
import suitabilityTemplate from '../instructions/council/suitability_check.md?raw'
import tldrTemplate from '../instructions/council/tldr.md?raw'

// ── Types ───────────────────────────────────────────────────────

export type SessionType = 'debate' | 'brainstorm' | 'review' | 'planning' | 'interview' | 'troubleshoot' | 'critique' | 'freeform'
export type Verbosity = 'thought' | 'message' | 'short' | 'medium' | 'long'
export type CouncilStatus = 'setup' | 'active' | 'synthesizing' | 'concluded'
export type ModeratorMode = 'moderator' | 'round-robin'
export type AgentAction = 'answer' | 'respond' | 'challenge' | 'refine' | 'broaden' | 'pass'
export type Vote = 'agree' | 'disagree' | 'abstain'
export type MemoryRounds = 5 | 10 | 20 | 30 | 0  // 0 = indefinite

export interface CouncilArtifact {
  id: number
  sessionId: string
  kind: 'minutes_md' | 'minutes_html' | 'tldr' | 'action_points'
  agentId: string
  agentName: string
  content: string  // for 'action_points': JSON.stringify(ActionPoint[])
  createdAt: string
}

// A single outstanding action point extracted per agent after synthesis. Each is
// self-contained (context embedded) so it can be worked on without the council.
export interface ActionPoint {
  title: string
  kind: 'todo' | 'intent'
  context: string
  task: string
  done_when: string
  refs: string[]
  sent?: boolean   // recorded into the agent's todo/intentions — persisted in the artifact
}

export interface CouncilAgentDef {
  id: string
  name: string
  host: string
  port: number
  auth: string
  muted: boolean
}

export interface CouncilAgent extends CouncilAgentDef {
  ws: AgentChatWS | null
  connected: boolean
  busy: boolean
  statusText: string
  toolHistory: string[]  // last 3 tool names
}

export interface ActivityLogEntry {
  timestamp: string
  agentId: string
  agentName: string
  type: 'tool' | 'status' | 'speaking' | 'done' | 'system' | 'moderator' | 'error' | 'connect' | 'disconnect' | 'narration' | 'reasoning'
  detail: string
}

export interface CouncilMessage {
  id: number
  localId: string
  round: number
  agentId: string
  agentName: string
  role: 'agent' | 'user' | 'system' | 'moderator' | 'synthesis'
  action: AgentAction | 'inject' | 'vote' | ''
  suitability: number
  targetAgentId: string
  content: string
  pinned: boolean
  metadata: Record<string, unknown>
  createdAt: string
}

export interface CouncilVote {
  id: number
  round: number
  agentId: string
  agentName: string
  vote: Vote
  reason: string
}

export interface CouncilSessionSummary {
  id: string
  title: string
  topic: string
  session_type: SessionType
  verbosity: Verbosity
  max_rounds: number
  current_round: number
  status: CouncilStatus
  moderator_mode: ModeratorMode
  moderator_agent: string
  agents: string
  created_at: string
  updated_at: string
}

export interface CouncilSession {
  id: string
  title: string
  topic: string
  sessionType: SessionType
  verbosity: Verbosity
  maxRounds: number
  originalMaxRounds: number   // the initial cap set at creation
  extensions: number[]        // e.g. [5, 10] = two extensions added
  currentRound: number
  status: CouncilStatus
  moderatorMode: ModeratorMode
  moderatorAgentId: string
  autoAdvance: boolean        // auto-proceed to next round after 5s countdown
  agents: CouncilAgent[]
  messages: CouncilMessage[]
  votes: CouncilVote[]
  artifacts: CouncilArtifact[]
  fileRefs: CouncilFileRef[]
  memoryRounds: MemoryRounds
  /**
   * If false, participants are forbidden from using ACTION: pass.
   * The btw_context.md template omits "pass" from the action list,
   * adds explicit "do not pass" guidance, and the speaker turn loop
   * retries once with a hard directive if the model passes anyway.
   * Default: false (passing disabled — every participant must
   * contribute every round). Flip on via the sidebar toggle if you
   * want to allow passing for a specific session.
   */
  allowPass: boolean
  /**
   * Whether agents may delegate / orchestrate during their council turns
   * (flight_deck, task_contract, consult_peer). Default false: the council IS
   * the coordination layer, so sub-delegating inside a turn is redundant and
   * causes no-progress loops. Flip on for sessions where farming out subtasks
   * is the point (e.g. planning).
   */
  allowDelegation: boolean
  createdAt: string           // ISO timestamp of session creation
  concludedAt: string         // ISO timestamp when session concluded (empty if still active)
  config: { firstSpeaker: string }
  /**
   * Slugs of ephemeral agents auto-spawned for this session (the "Auto-assemble"
   * panel). Empty for councils built from already-running, user-picked agents.
   * Persisted in config so the council knows these agents are temporary and can
   * dispose them — and so delete can tear them down.
   */
  spawnedSlugs: string[]
  /** True once the temporary panel has been disposed (torn down). */
  agentsDisposed: boolean
}

export interface CouncilFileRef {
  name: string
  path: string  // absolute path on agent filesystem
  size: number
}

export interface CreateSessionConfig {
  title: string
  topic: string
  sessionType: SessionType
  verbosity: Verbosity
  maxRounds: number
  moderatorMode: ModeratorMode
  moderatorAgentId: string
  agents: CouncilAgentDef[]
  firstSpeaker: string
  files: File[]
  // Auto-assemble: spawn a task-modeled archetype panel instead of using
  // pre-selected running agents. When true, `agents` is ignored and the panel is
  // built server-side from `maxAgents`, scoped by the Library `tiers`/`envVars`.
  autoAssemble?: boolean
  maxAgents?: number
  tiers?: Record<string, unknown>
  envVars?: { key: string; value: string }[]
  // Optional fixed panel — when non-empty the assembler uses exactly these
  // archetype ids; when empty the router picks the panel automatically.
  archetypeIds?: string[]
}

// ── Constants ───────────────────────────────────────────────────

const VERBOSITY_RULES: Record<Verbosity, string> = {
  thought: 'Limit your response to 1-2 sentences. Be extremely concise — just the core thought.',
  message: 'Limit your response to at most 5 sentences (one short paragraph).',
  short: 'Limit your response to at most 3 short paragraphs.',
  medium: 'Limit your response to at most 5 paragraphs.',
  long: 'Limit your response to at most 10 paragraphs.',
}

const SESSION_TYPE_DESC: Record<SessionType, string> = {
  debate: 'This is a structured debate. Take clear positions and defend them with evidence. Challenge weak arguments respectfully.',
  brainstorm: 'This is a brainstorming session. Build on ideas, suggest new angles, and avoid criticism in early rounds. Be creative.',
  review: 'This is a review session. One agent presents work, others provide constructive critique. Be specific about what works and what does not.',
  planning: 'This is a planning session. Break down tasks, assign responsibilities, identify risks, and create actionable steps.',
  interview: 'This is an interview session. One agent asks focused questions, others provide detailed answers. Extract knowledge, clarify assumptions, and explore topics in depth.',
  troubleshoot: 'This is a troubleshooting session. Collaboratively diagnose the problem, propose hypotheses, narrow down root causes, and suggest fixes. Be methodical and evidence-driven.',
  critique: 'This is a critique session. Actively look for flaws, weaknesses, and gaps. Be adversarial but constructive — the goal is to stress-test ideas and strengthen them.',
  freeform: 'This is a freeform discussion. There are no structural constraints — speak naturally, follow tangents, and let the conversation evolve organically.',
}

// Per-agent response timeout. Agents may run long chains of tool calls (deep
// research, document processing) on slower/reasoning models, so this is set
// generously high. The "Restart round" control is the escape hatch if an agent
// genuinely hangs, rather than a short timeout that kills slow-but-valid work.
const AGENT_TIMEOUT_MS = 3_600_000  // 60 minutes

// When an agent's progress detector trips, the server returns a canned "I got
// stuck…" / "budget exhausted" reply. In manual mode a human nudge ("continue")
// usually unblocks it, so the council does the same automatically before
// accepting the turn — and won't let the next agent speak until then.
const MAX_STUCK_NUDGES = 3
const STUCK_NUDGE_PROMPT =
  'You ended your turn reporting that you got stuck and could not make further progress. ' +
  'Do not give up and do not repeat that message. Pick up exactly where you left off, work ' +
  'through the obstacle step by step, and deliver your actual contribution to the discussion. ' +
  'If a tool call failed, try a different approach or reason it through directly. Provide a ' +
  'substantive answer now.'

// Substrings that identify the agent runtime's canned give-up replies. These are
// NOT hardcoded here — they're fetched from the backend (single source:
// captain_claw/agent_stuck.py) so the strings live in exactly one place. Empty
// until loaded; populated on session load/start (well before any round runs).
let _stuckMarkers: string[] = []

async function apiLoadStuckMarkers(): Promise<void> {
  try {
    const res = await _authedFetch('/fd/council/stuck-markers')
    if (!res.ok) return
    const data = await res.json()
    if (Array.isArray(data.markers)) {
      _stuckMarkers = data.markers.map((m: string) => String(m).toLowerCase())
    }
  } catch { /* leave markers as-is; detection simply stays off until loaded */ }
}

// Orchestration/delegation tools denied during council speaker turns unless the
// session opts into delegation. The council itself is the coordination layer, so
// sub-delegating inside a turn is redundant and a common cause of no-progress
// loops (re-contracting/re-gating at flat context). Research tools stay allowed.
const DELEGATION_TOOLS = ['flight_deck', 'task_contract', 'consult_peer', 'botport']

/** Collapse whitespace and clip long narration/reasoning for the activity log. */
function _truncateLog(text: string, max = 160): string {
  const oneLine = text.replace(/\s+/g, ' ').trim()
  return oneLine.length > max ? oneLine.slice(0, max - 1) + '…' : oneLine
}

/** True if a response is one of the agent runtime's canned give-up replies. */
function isStuckResponse(text: string): boolean {
  if (_stuckMarkers.length === 0) return false
  const t = text.toLowerCase()
  return _stuckMarkers.some(m => t.includes(m))
}

/** Simple {placeholder} template renderer. */
function renderTemplate(template: string, vars: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, key) => {
    return key in vars ? String(vars[key]) : match
  }).trim()
}

// ── API helpers ─────────────────────────────────────────────────

function _headers(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

// All council REST calls go through this. Councils run long (60-min agent turns),
// so the access token routinely expires mid-session; without a refresh+retry the
// request just 401s and the write is dropped — silently losing messages, votes,
// and status updates (round 1 saved, then nothing). On 401 we refresh the token
// and retry once. build() re-reads _headers() so the retry uses the new token.
async function _authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const build = (): RequestInit => ({
    ...init,
    headers: { ..._headers(), ...((init.headers as Record<string, string>) || {}) },
    credentials: 'include',
  })
  let res = await fetch(url, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(url, build())
  }
  return res
}

// ── Failed-write retry queue ─────────────────────────────────────
// Belt-and-suspenders on top of _authedFetch's refresh-retry: if a persist STILL
// fails (transient outage, refresh token briefly invalid), the write is queued
// and re-sent on a backing-off timer so data lands late instead of being lost.
// In-memory only (a tab close before drain loses it — but that data was never
// persisted anyway). bodies are JSON strings, so RequestInit is safely reusable.
interface PendingWrite { url: string; init: RequestInit; label: string; tries: number }
const _writeQueue: PendingWrite[] = []
let _drainTimer: ReturnType<typeof setTimeout> | null = null
const _MAX_WRITE_TRIES = 12

function _scheduleDrain(delayMs: number): void {
  if (_drainTimer) return
  _drainTimer = setTimeout(() => { _drainTimer = null; void _drainWrites() }, delayMs)
}

function _enqueueWrite(url: string, init: RequestInit, label: string): void {
  _writeQueue.push({ url, init, label, tries: 0 })
  _scheduleDrain(5000)
}

async function _drainWrites(): Promise<void> {
  if (_writeQueue.length === 0) return
  const batch = _writeQueue.splice(0, _writeQueue.length)
  let anyFailed = false
  for (const op of batch) {
    let ok = false
    try { ok = (await _authedFetch(op.url, op.init)).ok } catch { ok = false }
    if (ok) continue
    op.tries++
    if (op.tries < _MAX_WRITE_TRIES) { _writeQueue.push(op); anyFailed = true }
    else console.error('council: DROPPING write after retries —', op.label)
  }
  if (anyFailed || _writeQueue.length > 0) _scheduleDrain(10000)  // back off
}

/** Authed write that queues itself for retry if it fails (data-loss critical). */
async function _persistWrite(url: string, init: RequestInit, label: string): Promise<Response> {
  const res = await _authedFetch(url, init)
  if (!res.ok) {
    console.error('council: persist failed, queued for retry —', label, res.status)
    _enqueueWrite(url, init, label)
  }
  return res
}

async function apiCreateSession(data: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await _authedFetch('/fd/council/sessions', {
    method: 'POST', body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to create council session')
  return res.json()
}

async function apiListSessions(): Promise<CouncilSessionSummary[]> {
  const res = await _authedFetch('/fd/council/sessions')
  if (!res.ok) return []
  return res.json()
}

async function apiGetSession(id: string): Promise<Record<string, unknown> | null> {
  const res = await _authedFetch(`/fd/council/sessions/${encodeURIComponent(id)}`)
  if (!res.ok) return null
  return res.json()
}

async function apiUpdateSession(id: string, fields: Record<string, unknown>): Promise<void> {
  await _persistWrite(`/fd/council/sessions/${encodeURIComponent(id)}`, {
    method: 'PUT', body: JSON.stringify(fields),
  }, `updateSession(${Object.keys(fields).join(',')})`)
}

async function apiDeleteSession(id: string): Promise<void> {
  await _authedFetch(`/fd/council/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

// An archetype panelist spawned by /fd/council/assemble.
interface AssembledAgent {
  id: string
  slug: string
  name: string
  host: string
  port: number
  auth: string
  archetype_id: string
  role: string
  why: string
  cognitive_mode: string
  fleet_instructions: string
  is_moderator: boolean
}

interface AssembleResult {
  domain: string
  rationale: string
  title: string
  source: string
  agents: AssembledAgent[]
}

async function apiAssemble(body: Record<string, unknown>): Promise<AssembleResult> {
  const res = await _authedFetch('/fd/council/assemble', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = 'Failed to assemble council'
    try { detail = (await res.json()).detail || detail } catch { /* keep default */ }
    throw new Error(detail)
  }
  return res.json()
}

async function apiTeardownAgents(slugs: string[]): Promise<void> {
  if (!slugs.length) return
  try {
    await _authedFetch('/fd/council/teardown', {
      method: 'POST', body: JSON.stringify({ slugs }),
    })
  } catch { /* best-effort cleanup */ }
}

async function apiGetMessages(id: string): Promise<CouncilMessage[]> {
  const res = await _authedFetch(`/fd/council/sessions/${encodeURIComponent(id)}/messages`)
  if (!res.ok) return []
  const rows = await res.json() as Record<string, unknown>[]
  return rows.map(rowToMessage)
}

async function apiAddMessages(id: string, messages: Record<string, unknown>[]): Promise<number[]> {
  const res = await _persistWrite(`/fd/council/sessions/${encodeURIComponent(id)}/messages`, {
    method: 'POST', body: JSON.stringify({ messages }),
  }, `addMessages(${messages.length})`)
  if (!res.ok) return []
  const data = await res.json()
  return data.ids || []
}

async function apiTogglePin(sessionId: string, messageId: number): Promise<void> {
  await _authedFetch(`/fd/council/sessions/${encodeURIComponent(sessionId)}/messages/${messageId}/pin`, {
    method: 'PUT',
  })
}

async function apiDeleteMessages(sessionId: string, round: number): Promise<void> {
  await _authedFetch(`/fd/council/sessions/${encodeURIComponent(sessionId)}/messages?round=${round}`, {
    method: 'DELETE',
  })
}

async function apiUpdateMessage(sessionId: string, messageId: number, fields: Record<string, unknown>): Promise<void> {
  await _persistWrite(`/fd/council/sessions/${encodeURIComponent(sessionId)}/messages/${messageId}`, {
    method: 'PUT', body: JSON.stringify(fields),
  }, `updateMessage(${messageId})`)
}

async function apiAddVotes(id: string, votes: Record<string, unknown>[]): Promise<void> {
  await _persistWrite(`/fd/council/sessions/${encodeURIComponent(id)}/votes`, {
    method: 'POST', body: JSON.stringify({ votes }),
  }, `addVotes(${votes.length})`)
}

async function apiGetVotes(id: string): Promise<CouncilVote[]> {
  const res = await _authedFetch(`/fd/council/sessions/${encodeURIComponent(id)}/votes`)
  if (!res.ok) return []
  const rows = await res.json() as Record<string, unknown>[]
  return rows.map(r => ({
    id: r.id as number,
    round: r.round as number,
    agentId: r.agent_id as string,
    agentName: r.agent_name as string,
    vote: r.vote as Vote,
    reason: r.reason as string,
  }))
}

async function apiGetArtifacts(id: string, kind?: string): Promise<CouncilArtifact[]> {
  const url = new URL(`/fd/council/sessions/${encodeURIComponent(id)}/artifacts`, location.origin)
  if (kind) url.searchParams.set('kind', kind)
  const res = await _authedFetch(url.toString())
  if (!res.ok) return []
  const rows = await res.json() as Record<string, unknown>[]
  return rows.map(r => ({
    id: r.id as number,
    sessionId: r.session_id as string,
    kind: r.kind as CouncilArtifact['kind'],
    agentId: r.agent_id as string,
    agentName: r.agent_name as string,
    content: r.content as string,
    createdAt: r.created_at as string,
  }))
}

async function apiUpsertArtifact(
  sessionId: string, kind: string, agentId: string, agentName: string, content: string,
): Promise<number> {
  const res = await _persistWrite(`/fd/council/sessions/${encodeURIComponent(sessionId)}/artifacts`, {
    method: 'POST',
    body: JSON.stringify({ kind, agent_id: agentId, agent_name: agentName, content }),
  }, `artifact(${kind})`)
  if (!res.ok) return 0
  const data = await res.json()
  return data.id as number || 0
}

function rowToMessage(r: Record<string, unknown>): CouncilMessage {
  return {
    id: r.id as number,
    localId: `srv-${r.id}`,
    round: r.round as number,
    agentId: r.agent_id as string || '',
    agentName: r.agent_name as string || '',
    role: r.role as CouncilMessage['role'],
    action: (r.action as string || '') as CouncilMessage['action'],
    suitability: r.suitability as number || 0,
    targetAgentId: r.target_agent_id as string || '',
    content: r.content as string || '',
    pinned: !!(r.pinned as number),
    metadata: (() => { try { return JSON.parse(r.metadata as string || '{}') } catch { return {} } })(),
    createdAt: r.created_at as string || '',
  }
}

// ── Response parsing ────────────────────────────────────────────

interface ParsedResponse {
  suitability: number
  action: AgentAction
  targetAgent: string
  content: string
}

/**
 * Strip all thinking / reasoning XML blocks that reasoning models emit.
 * Handles <think>, <thinking>, <reasoning>, <reflection>, <inner_monologue>,
 * and similar tags — both self-closing and block forms.
 */
const stripThinkingBlocks = sharedStripThinkingBlocks

// Re-exported from shared util so council and chat use the same sanitizer.
const extractContent = sanitizeAgentContent

function parseAgentResponse(raw: string): ParsedResponse {
  let suitability = 0.5
  let action: AgentAction = 'answer'
  let targetAgent = ''

  // Strip thinking blocks first so headers inside them don't confuse parsing
  const stripped = stripThinkingBlocks(raw)
  // Also strip markdown bold/italic so **SUITABILITY:** etc. are matched
  const plain = stripped.replace(/\*{1,2}|_{1,2}/g, '')

  // Header separator: colon, em-dash, en-dash, or hyphen.
  const SEP = '\\s*[:\\-–—]\\s*'

  const suitMatch = plain.match(new RegExp(`^SUITABILITY${SEP}([\\d.]+)`, 'mi'))
  if (suitMatch) suitability = Math.max(0, Math.min(1, parseFloat(suitMatch[1]) || 0.5))

  const actionMatch = plain.match(new RegExp(`^ACTION${SEP}(answer|respond|challenge|refine|broaden|pass)`, 'mi'))
  if (actionMatch) action = actionMatch[1] as AgentAction

  const targetMatch = plain.match(new RegExp(`^TARGET${SEP}(.+)`, 'mi'))
  if (targetMatch) targetAgent = targetMatch[1].trim()

  const content = extractContent(raw)

  return { suitability, action, targetAgent, content }
}

// ── Prompt templates ────────────────────────────────────────────

function buildBtwPrompt(session: CouncilSession): string {
  const moderatorInfo = session.moderatorMode === 'moderator'
    ? `Moderation: Moderated by ${session.agents.find(a => a.id === session.moderatorAgentId)?.name || 'a moderator'} who selects speakers based on suitability scores.`
    : `Moderation: Round-robin — all participants speak each round, ordered by suitability score. The user approves each new round.`

  // Build file reference section
  let fileInfo = ''
  if (session.fileRefs.length > 0) {
    const refs = session.fileRefs.map(f => `[Attached file: ${f.name} → ${f.path}]`).join('\n')
    fileInfo = `\n\nThe following files have been provided for this council session. Read and reference them as needed:\n${refs}`
  }

  let extensionNote = ''
  if (session.extensions.length > 0) {
    const extList = session.extensions.map(e => `+${e}`).join(', ')
    extensionNote = ` Originally ${session.originalMaxRounds} rounds, extended: ${extList}.`
  }

  // Pass-action wiring. The btw template carries two placeholders:
  //   {passOption}    — appended into the ACTION list line
  //   {passGuidance}  — appended at the end of the ACTION guidance block
  // When allowPass is true we keep the historical behaviour (pass is
  // listed and softly discouraged). When allowPass is false we strip
  // pass from the list entirely and add a hard "you must contribute"
  // line so the model has no easy out.
  const allowPass = session.allowPass !== false
  const passOption = allowPass ? '|pass' : ''
  const passGuidance = allowPass
    ? '\n- **pass** — last resort. Use ONLY if every angle you would raise has already been said by a previous speaker in this round. "I am not an expert" is NOT a valid reason to pass. The whole point of a multi-agent council is that every participant brings their reasoning, even on topics outside their wheelhouse.'
    : '\n\nPassing is DISABLED for this council session. You must contribute every round. If you feel out of your depth, share your reasoning anyway — even a "here is how I would think about this if I had to decide" framing is more valuable than silence.'

  return renderTemplate(btwTemplate, {
    sessionType: session.sessionType.toUpperCase(),
    sessionTypeDesc: SESSION_TYPE_DESC[session.sessionType],
    topic: session.topic,
    agentNames: session.agents.map(a => a.name).join(', '),
    maxRounds: session.maxRounds,
    currentRound: session.currentRound || 1,
    moderatorInfo,
    verbosityRule: VERBOSITY_RULES[session.verbosity],
    extensionNote,
    passOption,
    passGuidance,
  }) + fileInfo
}

function buildTurnPrompt(
  session: CouncilSession,
  round: number,
  contextMessages: CouncilMessage[],
  directive?: string,
  speakerName?: string,
  speakerAgentId?: string,
): string {
  let recentContributions = ''
  if (contextMessages.length > 0) {
    // Group by round for clarity
    const byRound = new Map<number, CouncilMessage[]>()
    for (const m of contextMessages) {
      const arr = byRound.get(m.round) || []
      arr.push(m)
      byRound.set(m.round, arr)
    }
    const sections: string[] = []
    for (const [r, msgs] of [...byRound.entries()].sort((a, b) => a[0] - b[0])) {
      const lines = msgs.map(m => {
        const actionLabel = m.action ? ` [${m.action}]` : ''
        const target = m.targetAgentId ? ` (→ ${m.agentName})` : ''
        return `  - ${m.agentName}${actionLabel}${target}: ${truncate(m.content, 600)}`
      })
      sections.push(`Round ${r}:\n${lines.join('\n')}`)
    }
    recentContributions = '\nDiscussion so far:\n' + sections.join('\n\n')
  }

  // Collect files that other agents wrote earlier in this session and that
  // Flight Deck has copied into the upcoming speaker's own filesystem. Each
  // agent works in an isolated sandbox, so we surface only the destination
  // path that exists on this speaker's side (not the author's source path).
  let filesNudge = ''
  if (speakerAgentId) {
    const entries = _sessionSharedFiles.get(session.id) || []
    const lines: string[] = []
    for (const entry of entries) {
      if (entry.srcAgentId === speakerAgentId) continue
      const localPath = entry.destPaths[speakerAgentId]
      if (!localPath) continue
      lines.push(
        `  - ${localPath} (written by ${entry.srcAgentName} in round ${entry.round}` +
          (localPath === entry.srcPath ? '' : `, originally ${entry.srcPath}`) +
          `)`,
      )
    }
    if (lines.length > 0) {
      filesNudge =
        '\n\nFiles shared in this discussion (already copied into your sandbox by Flight Deck):\n' +
        lines.join('\n') +
        '\nIf any of these are relevant to your angle, use your `read` tool on the path above to read what was actually written before contributing — build on it or push back against it instead of guessing from the summaries.'
    }
  }

  const directiveText = directive ? `\nDirective from the user: ${directive}` : ''

  // Build extension note for agents
  let extensionNote = ''
  if (session.extensions.length > 0) {
    const extList = session.extensions.map(e => `+${e}`).join(', ')
    extensionNote = ` (originally ${session.originalMaxRounds} rounds, extended: ${extList})`
  }

  // Remind the agent who it is and not to echo instructions or address itself
  const identityNote = speakerName
    ? `\nYou are ${speakerName}. Do NOT repeat these instructions, do NOT echo memory retrieval results, and do NOT address or refer to yourself in the third person. Respond directly in character.`
    : ''

  // Keep a discussion turn a discussion (unless delegation is explicitly on).
  const delegationNote = session.allowDelegation
    ? ''
    : '\nThis is a single contribution to the discussion, not a task to orchestrate. Do NOT delegate to other agents, create task contracts, or spin up sub-tasks — the council is already the coordination layer. Give your own analysis and stop. (Research tools like web_search are fine; delegation tools are disabled this turn.)'

  // Tabular output is the most common way agents waste a turn: they write a
  // Python script to build an .xlsx and that fails or produces nothing useful.
  // Forbid it firmly on every turn and point at the two supported paths.
  const tabularNote =
    '\nDo NOT write Python scripts (or any other code) to generate .xlsx / Excel / spreadsheet files. If you need to save data in tabular format, either write a Markdown table to a `.md` file, or use the `datastore` tool to store it in a relational table. Never produce spreadsheets via generated code.'

  return renderTemplate(turnTemplate, {
    round,
    maxRounds: session.maxRounds,
    recentContributions: recentContributions + filesNudge,
    directive: directiveText + identityNote + delegationNote + tabularNote,
    extensionNote,
  })
}

function buildModeratorSelectPrompt(
  scores: { name: string; score: number; spoken: boolean; spokeLastPrevRound?: boolean }[],
  recentMessages: CouncilMessage[],
): string {
  const agentScores = scores
    .map(s => {
      let line = `- ${s.name}: suitability=${s.score.toFixed(2)}, spoken=${s.spoken ? 'yes' : 'no'}`
      if (s.spokeLastPrevRound) line += ' (spoke last in previous round — do NOT pick first)'
      return line
    })
    .join('\n')
  const recentDiscussion = recentMessages
    .slice(-3)
    .map(m => `- ${m.agentName}: ${truncate(m.content, 200)}`)
    .join('\n')

  return renderTemplate(moderatorSelectTemplate, { agentScores, recentDiscussion })
}

function buildSynthesisPrompt(session: CouncilSession): string {
  const transcript = session.messages
    .filter(m => m.role === 'agent' || m.role === 'moderator')
    .map(m => `[Round ${m.round}] ${m.agentName} (${m.action}): ${m.content}`)
    .join('\n\n')

  return renderTemplate(synthesisTemplate, {
    verbosityRule: VERBOSITY_RULES[session.verbosity],
    transcript,
  })
}

function buildVotePrompt(synthesis: string): string {
  return renderTemplate(voteTemplate, { synthesis })
}

function buildSessionTranscript(session: CouncilSession): string {
  return session.messages
    .filter(m => m.role === 'agent' || m.role === 'moderator')
    .map(m => `[Round ${m.round}] ${m.agentName} (${m.action || m.role}): ${m.content}`)
    .join('\n\n')
}

function buildTldrPrompt(session: CouncilSession): string {
  return renderTemplate(tldrTemplate, {
    topic: session.topic,
    sessionType: session.sessionType,
    totalRounds: session.currentRound,
    transcript: buildSessionTranscript(session),
  })
}

// Scoped action-points prompt: deliberately does NOT include the full transcript
// (could be ~100kB). Each agent already participated, so we give it only its own
// contributions + the synthesis + pinned items — a small slice — and ask for
// self-contained action points it should own.
function buildActionPointsPrompt(session: CouncilSession, agentId: string): string {
  const own = session.messages
    .filter(m => m.agentId === agentId && m.role === 'agent')
    .map(m => `[Round ${m.round}] ${m.content}`).join('\n\n') || '(no recorded contributions)'
  const synthesis = session.messages
    .filter(m => m.role === 'synthesis')
    .map(m => m.content).join('\n\n') || '(no synthesis recorded)'
  const pinned = session.messages
    .filter(m => m.pinned)
    .map(m => `${m.agentName}: ${m.content}`).join('\n\n') || '(none)'
  return [
    'REFLECTION REQUEST — this is NOT a task to perform. You are only DESCRIBING work',
    'for later, never doing it. Do NOT begin, attempt, or complete any of these action',
    'points now. Do NOT take any action, do NOT use any tools, do NOT delegate to or',
    'consult other agents, and do NOT call flight_deck, botport, task_contract,',
    'web_fetch, write, or any other tool. Answer DIRECTLY and immediately from the',
    'material below — you already have everything you need; there is nothing to look',
    'up, produce, or coordinate. Your entire response is the JSON list and nothing else.',
    '',
    `The council on "${session.topic}" has concluded. You were one of the participants.`,
    'List YOUR OWN outstanding action points — the next steps that follow from the',
    'discussion and that you, in your role, would own (do not list other agents\'',
    'work). These are descriptions to record, not instructions to execute now. Make',
    'each FULLY SELF-CONTAINED so it can be worked on later without the council',
    'transcript.',
    '',
    'Respond with ONLY a JSON array (0-6 items), no prose, no tool calls. Each item:',
    '{"title": short imperative, "kind": "todo" (concrete near-term) | "intent" (proactive/recurring),',
    ' "context": 2-4 sentences of background/decision motivating this,',
    ' "task": precisely what to do, "done_when": completion criteria,',
    ' "refs": [files/links/names mentioned, or []]}',
    '',
    '## Your contributions',
    own,
    '',
    '## Final synthesis',
    synthesis,
    '',
    '## Pinned items',
    pinned,
  ].join('\n')
}

/** Parse an LLM JSON-array response into ActionPoint[], tolerant of fences/prose. */
function parseActionPoints(raw: string): ActionPoint[] {
  let txt = raw.trim()
  if (txt.startsWith('```')) {
    txt = txt.split('\n').filter(l => !l.trim().startsWith('```')).join('\n')
  }
  const start = txt.indexOf('[')
  const end = txt.lastIndexOf(']')
  if (start === -1 || end === -1 || end < start) return []
  try {
    const arr = JSON.parse(txt.slice(start, end + 1))
    if (!Array.isArray(arr)) return []
    return arr.filter(x => x && typeof x.title === 'string').map(x => ({
      title: String(x.title),
      kind: x.kind === 'intent' ? 'intent' : 'todo',
      context: String(x.context || ''),
      task: String(x.task || ''),
      done_when: String(x.done_when || ''),
      refs: Array.isArray(x.refs) ? x.refs.map(String) : [],
    }))
  } catch { return [] }
}

function generateMinutesMd(session: CouncilSession, tldrs: CouncilArtifact[]): string {
  const lines: string[] = []
  lines.push(`# ${session.title}`)
  lines.push('')
  lines.push(`**Topic:** ${session.topic}`)
  lines.push(`**Type:** ${session.sessionType}`)
  lines.push(`**Mode:** ${session.moderatorMode}`)
  lines.push(`**Participants:** ${session.agents.map(a => a.name).join(', ')}`)
  lines.push(`**Rounds:** ${session.currentRound} / ${session.maxRounds}${session.extensions.length > 0 ? ` (originally ${session.originalMaxRounds}, extended: ${session.extensions.map(e => `+${e}`).join(', ')})` : ''}`)
  lines.push(`**Date:** ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}`)
  lines.push('')
  lines.push('---')
  lines.push('')

  // Group messages by round
  const byRound = new Map<number, CouncilMessage[]>()
  for (const m of session.messages) {
    if (m.role === 'system') continue
    const arr = byRound.get(m.round) || []
    arr.push(m)
    byRound.set(m.round, arr)
  }

  for (const [round, msgs] of [...byRound.entries()].sort((a, b) => a[0] - b[0])) {
    lines.push(`## Round ${round}`)
    lines.push('')
    for (const m of msgs) {
      const action = m.action ? ` [${m.action}]` : ''
      const target = m.targetAgentId ? ` → ${session.agents.find(a => a.id === m.targetAgentId)?.name || ''}` : ''
      if (m.role === 'synthesis') {
        lines.push(`### Synthesis (by ${m.agentName})`)
        lines.push('')
        lines.push(m.content)
      } else if (m.role === 'moderator') {
        lines.push(`> **${m.agentName}** (moderator): ${m.content}`)
      } else {
        lines.push(`### ${m.agentName}${action}${target}`)
        if (m.suitability > 0) lines.push(`*Suitability: ${m.suitability.toFixed(2)}*`)
        lines.push('')
        lines.push(m.content)
      }
      lines.push('')
    }
  }

  // Votes
  if (session.votes.length > 0) {
    lines.push('## Votes')
    lines.push('')
    lines.push('| Agent | Vote | Reason |')
    lines.push('|-------|------|--------|')
    for (const v of session.votes) {
      lines.push(`| ${v.agentName} | ${v.vote} | ${v.reason} |`)
    }
    lines.push('')
  }

  // TL;DRs
  if (tldrs.length > 0) {
    lines.push('## TL;DR')
    lines.push('')
    for (const t of tldrs) {
      lines.push(`**${t.agentName}:** ${t.content}`)
      lines.push('')
    }
  }

  return lines.join('\n')
}

function downloadFile(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '...' : s
}

/** Rough token estimate: ~4 chars per token */
function estimateTokens(s: string): number {
  return Math.ceil(s.length / 4)
}

const COUNCIL_CONTEXT_TOKEN_CAP = 30_000

/**
 * Collect agent messages for context, respecting memoryRounds and token cap.
 * Returns messages from most recent rounds first, trimming older rounds if over budget.
 */
function collectContextMessages(
  allMessages: CouncilMessage[],
  currentRound: number,
  memoryRounds: MemoryRounds,
): CouncilMessage[] {
  // Only agent messages carry useful context
  const agentMsgs = allMessages.filter(m => m.role === 'agent')
  if (agentMsgs.length === 0) return []

  // Determine which rounds to consider
  const oldestRound = memoryRounds === 0
    ? 1
    : Math.max(1, currentRound - memoryRounds + 1)

  // Group messages by round (newest first)
  const rounds = new Map<number, CouncilMessage[]>()
  for (const m of agentMsgs) {
    if (m.round < oldestRound || m.round > currentRound) continue
    const arr = rounds.get(m.round) || []
    arr.push(m)
    rounds.set(m.round, arr)
  }

  // Walk from newest round backward, accumulate within token budget
  const sortedRounds = [...rounds.keys()].sort((a, b) => b - a)
  const included: CouncilMessage[] = []
  let tokenCount = 0

  for (const r of sortedRounds) {
    const msgs = rounds.get(r)!
    // Estimate tokens for this round
    let roundTokens = 0
    for (const m of msgs) {
      roundTokens += estimateTokens(m.content) + 20  // overhead for name/action/formatting
    }

    if (tokenCount + roundTokens > COUNCIL_CONTEXT_TOKEN_CAP && included.length > 0) {
      break  // drop this and older rounds
    }

    included.unshift(...msgs)  // prepend to keep chronological order
    tokenCount += roundTokens
  }

  return included
}

// ── Store ───────────────────────────────────────────────────────

// Auto-advance timer stored outside Zustand (not serialisable)
let _autoAdvanceTimerId: ReturnType<typeof setInterval> | null = null

// Round epoch: incremented whenever the in-flight round must be invalidated
// (e.g. "Restart round"). Round loops capture the epoch at start and bail if it
// changes, so a stale loop can't append messages to a freshly-restarted round.
let _roundEpoch = 0
// When an agent response is being awaited, this unblocks it immediately (used by
// restart so we don't wait out the full agent timeout on an in-flight turn).
let _abortCollect: (() => void) | null = null

// Pending file-write captures per agent, populated by the WS `monitor` handler
// when an agent calls the `write` tool. Drained at the end of each speaker turn,
// transferred to other council agents via Flight Deck's file-transfer endpoint,
// and surfaced to subsequent speakers via buildTurnPrompt.
const _pendingWrittenFiles = new Map<string, string[]>()

// Files written during the current session, keyed by sessionId. Each entry
// records the source path on the author's filesystem plus the post-transfer
// destination path on each peer agent (so we can tell each speaker exactly
// where the file lives in their own sandbox).
interface SharedFileEntry {
  srcPath: string
  srcAgentId: string
  srcAgentName: string
  round: number
  destPaths: Record<string, string>  // peerAgentId -> path on peer's filesystem
}
const _sessionSharedFiles = new Map<string, SharedFileEntry[]>()

/** Extract the destination path from a `write` tool output line. */
function _extractWrittenPath(output: string): string | null {
  if (!output) return null
  const m = /Written\s+\d+\s+chars?\s*\(\d+\s+lines?\)\s+to\s+(\S+)/i.exec(output)
  return m ? m[1] : null
}

/** Merge partial config into the persisted session config JSON */
function _persistConfig(sessionId: string, patch: Record<string, unknown>) {
  // Read current active session to get full config
  const s = useCouncilStore.getState().activeSession
  if (!s) return
  const fullConfig = {
    ...s.config,
    memoryRounds: s.memoryRounds,
    originalMaxRounds: s.originalMaxRounds,
    extensions: s.extensions,
    autoAdvance: s.autoAdvance,
    fileRefs: s.fileRefs,
    allowPass: s.allowPass,
    allowDelegation: s.allowDelegation,
    spawnedSlugs: s.spawnedSlugs,
    agentsDisposed: s.agentsDisposed,
    ...patch,
  }
  apiUpdateSession(sessionId, { config: JSON.stringify(fullConfig) })
}

interface CouncilStore {
  sessions: CouncilSessionSummary[]
  activeSession: CouncilSession | null
  loading: boolean
  speaking: string   // agentId currently speaking, '' if idle
  roundRunning: boolean  // true while a round loop is actively executing in this tab
  pendingFiles: File[]  // files to upload to agents on council start
  generatingArtifact: '' | 'tldr' | 'minutes_md' | 'minutes_html' | 'action_points'
  activityLog: ActivityLogEntry[]
  autoAdvanceCountdown: number  // seconds remaining, 0 = not counting

  // Activity log
  _log: (agentId: string, agentName: string, type: ActivityLogEntry['type'], detail: string) => void

  // Session lifecycle
  loadSessionList: () => Promise<void>
  createSession: (cfg: CreateSessionConfig) => Promise<string>
  loadSession: (id: string) => Promise<void>
  deleteSession: (id: string) => Promise<void>
  cancelSession: (id: string) => Promise<void>
  clearActive: () => void

  // Connection management
  connectAllAgents: () => void
  disconnectAllAgents: () => void
  disposeAgents: () => Promise<void>
  _ensureConnected: (timeoutMs?: number) => Promise<CouncilAgent[]>

  // Council orchestration
  startCouncil: () => Promise<void>
  advanceRound: () => Promise<void>
  restartRound: () => Promise<void>
  requestSynthesis: () => Promise<void>
  concludeSession: () => Promise<void>

  // User interactions
  injectMessage: (content: string) => Promise<void>
  directAddress: (agentId: string, content: string) => Promise<void>
  muteAgent: (agentId: string, muted: boolean) => void
  pinMessage: (messageId: number) => void
  setMemoryRounds: (value: MemoryRounds) => void
  setAllowPass: (value: boolean) => void
  setAllowDelegation: (value: boolean) => void
  setAutoAdvance: (value: boolean) => void
  extendRounds: (amount: number) => Promise<void>
  cancelAutoAdvance: () => void

  // Minutes & TL;DR
  generateTldrs: () => Promise<void>
  generateActionPoints: () => Promise<void>
  sendActionPointToAgent: (agentId: string, item: ActionPoint) => Promise<void>
  exportMinutesMd: () => void

  // Internal
  _sendToAgent: (agentId: string, content: string, type?: 'chat' | 'btw', opts?: { noTools?: boolean; denyTools?: string[] }) => void
  _collectResponse: (agentId: string, onChunk?: (acc: string) => void) => Promise<string>
  _addMessage: (msg: Omit<CouncilMessage, 'id' | 'localId' | 'createdAt'>) => Promise<void>
  _addSystemMessage: (round: number, content: string) => Promise<void>
  _persistSession: () => Promise<void>
  _startAutoAdvanceTimer: () => void
  _cancelAutoAdvanceTimer: () => void
  _runSpeakerTurn: (agentId: string, round: number, recentMsgs: CouncilMessage[], directive?: string) => Promise<CouncilMessage | null>
  _shareWrittenFiles: (srcAgentId: string, srcAgentName: string, round: number, paths: string[]) => Promise<void>
  _runRoundRobin: (round: number) => Promise<void>
  _runModeratorRound: (round: number) => Promise<void>
  _getSuitabilityScores: () => Promise<Map<string, number>>
}

export const useCouncilStore = create<CouncilStore>((set, get) => ({
  sessions: [],
  activeSession: null,
  loading: false,
  speaking: '',
  roundRunning: false,
  pendingFiles: [],
  generatingArtifact: '',
  activityLog: [],
  autoAdvanceCountdown: 0,

  _log: (agentId, agentName, type, detail) => {
    const entry: ActivityLogEntry = {
      timestamp: new Date().toISOString(),
      agentId, agentName, type, detail,
    }
    set(state => ({
      activityLog: [...state.activityLog.slice(-199), entry],  // keep last 200
    }))
  },

  // ── Session lifecycle ──

  loadSessionList: async () => {
    set({ loading: true })
    try {
      const sessions = await apiListSessions()
      set({ sessions })
    } finally {
      set({ loading: false })
    }
  },

  createSession: async (cfg) => {
    let agentDefs: CouncilAgentDef[]
    let moderatorMode = cfg.moderatorMode
    let moderatorAgentId = cfg.moderatorAgentId
    let firstSpeaker = cfg.firstSpeaker
    // Slugs of any ephemeral panelists we spawn — persisted so delete can tear
    // them down (auto-assembled councils own their agents; manual ones don't).
    let spawnedSlugs: string[] = []

    if (cfg.autoAssemble) {
      // Route a task-modeled archetype panel and spawn it server-side, then wire
      // the returned panelists into the council exactly like picked agents.
      const result = await apiAssemble({
        topic: cfg.topic,
        session_type: cfg.sessionType,
        max_agents: cfg.maxAgents || 4,
        tiers: cfg.tiers || {},
        env_vars: (cfg.envVars || []).filter(e => e.key.trim() && e.value.trim()),
        ...(cfg.archetypeIds?.length ? { archetype_ids: cfg.archetypeIds } : {}),
      })
      const procStore = useProcessStore.getState()
      // Push each panelist's task-tailored persona into the process store so the
      // council sends it as `self.fleet_instructions` on connect.
      for (const a of result.agents) {
        if (a.fleet_instructions) procStore.setFleetInstructions(a.slug, a.fleet_instructions)
      }
      // Refresh the process list so connectAllAgents resolves the fresh ports.
      try { await procStore.fetchProcesses() } catch { /* defs carry port/auth too */ }
      agentDefs = result.agents.map(a => ({
        id: a.id, name: a.name, host: a.host, port: a.port, auth: a.auth, muted: false,
      }))
      spawnedSlugs = result.agents.map(a => a.slug)
      const mod = result.agents.find(a => a.is_moderator)
      moderatorMode = mod ? 'moderator' : 'round-robin'
      moderatorAgentId = mod?.id || ''
      firstSpeaker = 'random'
    } else {
      agentDefs = cfg.agents.map(a => ({
        id: a.id, name: a.name, host: a.host, port: a.port, auth: a.auth, muted: false,
      }))
    }

    const data = await apiCreateSession({
      title: cfg.title,
      topic: cfg.topic,
      session_type: cfg.sessionType,
      verbosity: cfg.verbosity,
      max_rounds: cfg.maxRounds,
      moderator_mode: moderatorMode,
      moderator_agent: moderatorAgentId,
      agents: JSON.stringify(agentDefs),
      config: JSON.stringify({ firstSpeaker, memoryRounds: 10, originalMaxRounds: cfg.maxRounds, extensions: [], allowPass: false, allowDelegation: false, spawnedSlugs, agentsDisposed: false }),
    })
    const id = data.id as string
    // Stash files for upload during startCouncil
    if (cfg.files?.length) set({ pendingFiles: cfg.files })
    await get().loadSessionList()
    return id
  },

  loadSession: async (id) => {
    // A freshly loaded session has no round loop running in this tab, even if
    // its status is 'active' (server may have been restarted mid-round). Reset
    // the flag so the UI shows a recover/restart prompt rather than "in session".
    _roundEpoch += 1
    set({ loading: true, roundRunning: false })
    void apiLoadStuckMarkers()  // refresh stuck-detection markers from the backend
    try {
      const raw = await apiGetSession(id)
      if (!raw) return

      const agentDefs: CouncilAgentDef[] = (() => {
        try { return JSON.parse(raw.agents as string || '[]') } catch { return [] }
      })()
      const config = (() => {
        try { return JSON.parse(raw.config as string || '{}') } catch { return {} }
      })()

      const agents: CouncilAgent[] = agentDefs.map(a => ({
        ...a, ws: null, connected: false, busy: false, statusText: '', toolHistory: [],
      }))

      const messages = await apiGetMessages(id)
      const votes = await apiGetVotes(id)
      const artifacts = await apiGetArtifacts(id)

      const session: CouncilSession = {
        id,
        title: raw.title as string,
        topic: raw.topic as string,
        sessionType: raw.session_type as SessionType,
        verbosity: raw.verbosity as Verbosity,
        maxRounds: raw.max_rounds as number,
        originalMaxRounds: (config.originalMaxRounds as number) || (raw.max_rounds as number),
        extensions: (config.extensions as number[]) || [],
        currentRound: raw.current_round as number,
        status: raw.status as CouncilStatus,
        moderatorMode: raw.moderator_mode as ModeratorMode,
        moderatorAgentId: raw.moderator_agent as string,
        autoAdvance: (config.autoAdvance as boolean) ?? false,
        agents,
        messages,
        votes,
        artifacts,
        fileRefs: (config.fileRefs as CouncilFileRef[]) || [],
        memoryRounds: (config.memoryRounds as MemoryRounds) || 10,
        // Hydrate the pass toggle. Default false: passing is
        // disabled unless the user explicitly opts in via the
        // sidebar toggle. Old sessions saved before this field
        // existed will also default to false on reload — flip the
        // sidebar switch back on if you want the old behaviour.
        allowPass: (config.allowPass as boolean | undefined) ?? false,
        allowDelegation: (config.allowDelegation as boolean | undefined) ?? false,
        createdAt: raw.created_at as string || '',
        concludedAt: (config.concludedAt as string) || '',
        config: { firstSpeaker: config.firstSpeaker || '' },
        spawnedSlugs: Array.isArray(config.spawnedSlugs) ? config.spawnedSlugs as string[] : [],
        agentsDisposed: (config.agentsDisposed as boolean | undefined) ?? false,
      }

      set({ activeSession: session })
    } finally {
      set({ loading: false })
    }
  },

  deleteSession: async (id) => {
    // Tear down any ephemeral panelists this (auto-assembled) session spawned,
    // so they don't linger in the fleet after the council is gone.
    try {
      const raw = await apiGetSession(id)
      const cfg = raw ? JSON.parse((raw.config as string) || '{}') : {}
      const slugs = Array.isArray(cfg.spawnedSlugs) ? cfg.spawnedSlugs as string[] : []
      if (slugs.length) await apiTeardownAgents(slugs)
    } catch { /* best-effort — never block the delete on teardown */ }
    await apiDeleteSession(id)
    const { activeSession } = get()
    if (activeSession?.id === id) {
      get().disconnectAllAgents()
      set({ activeSession: null })
    }
    await get().loadSessionList()
  },

  cancelSession: async (id) => {
    await _authedFetch(`/fd/council/sessions/${encodeURIComponent(id)}/cancel`, { method: 'POST' })
    await get().loadSessionList()
  },

  clearActive: () => {
    get().disconnectAllAgents()
    set({ activeSession: null, activityLog: [] })
  },

  // ── Connection management ──

  connectAllAgents: () => {
    const s = get().activeSession
    if (!s) return
    // Temporary panel already torn down — there's nothing to connect to.
    if (s.agentsDisposed) return

    const agents = s.agents.map(a => {
      if (a.ws) {
        // A ws object lingers after disconnect (e.g. on conclude). Revive a
        // dropped socket instead of skipping it — otherwise reconnect attempts
        // (e.g. before action-point extraction) silently no-op and report
        // "no agents reachable".
        if (!a.ws.connected) a.ws.connect()
        return a
      }
      // Resolve the agent's CURRENT connection params from the live container/
      // process stores — mirroring how the Chat panel connects — instead of the
      // host/port/token snapshotted into the session at creation. Those rotate
      // on (re)spawn/restart; a stale token makes the proxy accept the browser
      // socket then fail to auth to the agent, producing a connect/disconnect
      // flap. Fall back to the stored values when no live entry is found.
      const { containers: _liveCs } = useContainerStore.getState()
      const { processes: _livePs } = useProcessStore.getState()
      const _liveC = _liveCs.find(c => c.id === a.id)
      const _liveP = _livePs.find(p => `proc-${p.slug}` === a.id || p.slug === a.id)
      const _host = (_liveC || _liveP) ? 'localhost' : a.host
      const _port = _liveC?.web_port || _liveP?.web_port || a.port
      const _auth = _liveC?.web_auth || _liveP?.web_auth || a.auth
      const ws = new AgentChatWS(a.id, _host, _port, _auth)

      // On welcome, send peer_agents with fleet instructions (mirrors chatStore behavior)
      ws.on('welcome', () => {
        const session = get().activeSession
        if (!session) return

        const { containers } = useContainerStore.getState()
        const { processes } = useProcessStore.getState()
        const { getFleetInstructions: getContainerFleetInst } = useContainerStore.getState()
        const { getFleetInstructions: getProcFleetInst } = useProcessStore.getState()

        // Build peer list (other agents in the council)
        const peers = session.agents
          .filter(p => p.id !== a.id)
          .map(p => ({
            name: p.name,
            description: '',
            host: p.host,
            port: p.port,
            auth: '',
            requireApproval: false,
          }))

        // Build self identity with fleet instructions
        let selfIdentity: { name: string; description: string; port: number; fleet_instructions?: string } | null = null
        const selfContainer = containers.find(c => c.id === a.id)
        if (selfContainer) {
          selfIdentity = {
            name: selfContainer.agent_name || selfContainer.name,
            description: selfContainer.description || '',
            port: selfContainer.web_port || 0,
            fleet_instructions: getContainerFleetInst(selfContainer.id),
          }
        }
        if (!selfIdentity) {
          const selfProcess = processes.find(p => `proc-${p.slug}` === a.id || p.slug === a.id)
          if (selfProcess) {
            selfIdentity = {
              name: selfProcess.name || selfProcess.slug,
              description: selfProcess.description || '',
              port: selfProcess.web_port || 0,
              fleet_instructions: getProcFleetInst(selfProcess.slug),
            }
          }
        }
        if (!selfIdentity) {
          selfIdentity = { name: a.name, description: '', port: a.port }
        }

        const { internalFdUrl } = useAuthStore.getState()
        const fdUrl = internalFdUrl || `${window.location.protocol}//${window.location.host}`
        ws.sendJSON({ type: 'peer_agents', agents: peers, self: selfIdentity, fd_url: fdUrl })
      })

      ws.on('_connected', () => {
        get()._log(a.id, a.name, 'connect', 'Connected')
        set(state => {
          if (!state.activeSession) return state
          return {
            activeSession: {
              ...state.activeSession,
              agents: state.activeSession.agents.map(ag =>
                ag.id === a.id ? { ...ag, connected: true } : ag,
              ),
            },
          }
        })
      })
      ws.on('_disconnected', () => {
        get()._log(a.id, a.name, 'disconnect', 'Disconnected')
        set(state => {
          if (!state.activeSession) return state
          return {
            activeSession: {
              ...state.activeSession,
              agents: state.activeSession.agents.map(ag =>
                ag.id === a.id ? { ...ag, connected: false, busy: false, statusText: '', toolHistory: [] } : ag,
              ),
            },
          }
        })
      })

      // Track tool usage
      ws.on('monitor', (data) => {
        const toolName = data.tool_name as string || ''
        if (!toolName) return
        get()._log(a.id, a.name, 'tool', toolName)
        // Capture file writes so we can nudge the next speaker to read them
        if (toolName.toLowerCase() === 'write') {
          const output = (data.output as string) || ''
          const path = _extractWrittenPath(output)
          if (path) {
            const existing = _pendingWrittenFiles.get(a.id) || []
            if (!existing.includes(path)) {
              existing.push(path)
              _pendingWrittenFiles.set(a.id, existing)
            }
          }
        }
        set(state => {
          if (!state.activeSession) return state
          return {
            activeSession: {
              ...state.activeSession,
              agents: state.activeSession.agents.map(ag => {
                if (ag.id !== a.id) return ag
                const history = [...ag.toolHistory, toolName].slice(-3)
                // Tool output arrives after the tool finished — phase status
                // (statusText) is driven by the backend `status` events now.
                return { ...ag, busy: true, toolHistory: history }
              }),
            },
          }
        })
      })

      // Track status changes
      ws.on('status', (data) => {
        const text = (data.text as string) || (data.status as string) || ''
        const idle = !text || /^(ready|idle|done|completed)$/i.test(text)
        if (!idle) {
          get()._log(a.id, a.name, 'status', text)
        }
        set(state => {
          if (!state.activeSession) return state
          return {
            activeSession: {
              ...state.activeSession,
              agents: state.activeSession.agents.map(ag =>
                ag.id === a.id ? { ...ag, busy: !idle, statusText: idle ? '' : text } : ag,
              ),
            },
          }
        })
      })

      // Surface the model's between-step narration in the activity log — useful
      // for debugging no-progress loops (what the model keeps re-deciding).
      ws.on('narration', (data) => {
        const text = String(data.text || '').trim()
        if (text) get()._log(a.id, a.name, 'narration', _truncateLog(text))
      })
      // Inline reasoning text (distinct from the "thinking" status heartbeat).
      ws.on('thinking', (data) => {
        const text = String(data.text || '').trim()
        // Skip the bare "Thinking…" placeholder and tool-phase echoes; keep real
        // reasoning content (phase 'reasoning').
        if (!text || text === 'Thinking…' || text === 'Thinking...') return
        get()._log(a.id, a.name, 'reasoning', _truncateLog(text))
      })

      ws.connect()
      return { ...a, ws, connected: false, busy: false, statusText: '', toolHistory: [] }
    })

    set({ activeSession: { ...s, agents } })
  },

  disconnectAllAgents: () => {
    const s = get().activeSession
    if (!s) return
    for (const a of s.agents) {
      a.ws?.disconnect()
    }
  },

  // Tear down the ephemeral panel auto-spawned for this session. Disconnects the
  // sockets, removes the agents from the fleet, and marks the session disposed so
  // it never tries to reconnect them. A no-op for manual (picked-agent) councils
  // or one that's already been disposed.
  disposeAgents: async () => {
    const s = get().activeSession
    if (!s || !s.spawnedSlugs?.length || s.agentsDisposed) return
    get().disconnectAllAgents()
    await apiTeardownAgents(s.spawnedSlugs)
    set(state => state.activeSession ? {
      activeSession: {
        ...state.activeSession,
        agentsDisposed: true,
        agents: state.activeSession.agents.map(a => ({
          ...a, connected: false, busy: false, statusText: '', toolHistory: [],
        })),
      },
    } : state)
    _persistConfig(s.id, { agentsDisposed: true })
    get()._log('', 'System', 'system', `Disposed ${s.spawnedSlugs.length} temporary agent(s)`)
  },

  // Connect agents (if needed) and poll until at least one is connected, up to
  // timeoutMs. Returns the currently-connected, unmuted agents. Replaces fragile
  // fixed-delay waits: agents on a slow/concluded session may take a while, and
  // if they're offline this returns [] so callers can fail gracefully instead of
  // proceeding on a stale snapshot.
  _ensureConnected: async (timeoutMs = 12000) => {
    const s = get().activeSession
    if (!s) return []
    if (s.agents.every(a => !a.ws?.connected)) get().connectAllAgents()
    const deadline = Date.now() + timeoutMs
    const live = () => get().activeSession?.agents.filter(a => a.ws?.connected && !a.muted) || []
    while (Date.now() < deadline) {
      if (live().length > 0) break
      await new Promise(r => setTimeout(r, 600))
    }
    return live()
  },

  // ── Orchestration ──

  startCouncil: async () => {
    const s = get().activeSession
    if (!s || s.status !== 'setup') return
    set({ roundRunning: true })
    void apiLoadStuckMarkers()  // ensure stuck-detection markers are loaded

    // Update status
    const updated: CouncilSession = {
      ...s,
      status: 'active',
      currentRound: 1,
      originalMaxRounds: s.originalMaxRounds || s.maxRounds,
      extensions: s.extensions || [],
    }
    set({ activeSession: updated })
    await apiUpdateSession(s.id, { status: 'active', current_round: 1 })

    // Connect all agents, then wait until every (unmuted) one is actually
    // connected before the first speaker runs. Freshly auto-spawned agents take a
    // few seconds to accept the WS + finish the peer handshake; starting on a
    // fixed 1s delay made the first speaker get skipped as "disconnected". Falls
    // through after a generous timeout so an agent that never comes up can't hang
    // the council.
    get().connectAllAgents()
    get()._log('', 'System', 'system', 'Connecting to agents...')
    {
      const deadline = Date.now() + 30000
      const pending = () => get().activeSession?.agents.filter(a => !a.muted && !a.ws?.connected) || []
      while (Date.now() < deadline && pending().length > 0) {
        await new Promise(r => setTimeout(r, 400))
      }
      const stragglers = pending()
      if (stragglers.length > 0) {
        get()._log('', 'System', 'system',
          `Proceeding without ${stragglers.length} agent(s) still connecting: ${stragglers.map(a => a.name).join(', ')}`)
      } else {
        get()._log('', 'System', 'system', 'All agents connected')
      }
    }

    // Upload pending files to all agents in parallel
    const pendingFiles = get().pendingFiles
    const fileRefs: CouncilFileRef[] = []
    if (pendingFiles.length > 0) {
      get()._log('', 'System', 'system', `Uploading ${pendingFiles.length} file(s) to ${updated.agents.length} agents...`)
      const connectedAgents = get().activeSession?.agents.filter(a => a.ws?.connected) || []

      for (const file of pendingFiles) {
        // Upload to all agents in parallel
        const uploads = connectedAgents.map(async (agent) => {
          try {
            const result = await uploadFileToAgent(agent.host, agent.port, agent.auth, file)
            return { agentId: agent.id, path: result.path, ok: true }
          } catch (err) {
            get()._log(agent.id, agent.name, 'error', `Failed to upload ${file.name}: ${err}`)
            return { agentId: agent.id, path: '', ok: false }
          }
        })
        const results = await Promise.all(uploads)
        const successCount = results.filter(r => r.ok).length
        // Use the path from the first successful upload (all agents get the same relative path)
        const firstOk = results.find(r => r.ok)
        if (firstOk) {
          fileRefs.push({ name: file.name, path: firstOk.path, size: file.size })
        }
        get()._log('', 'System', 'system', `Uploaded ${file.name} to ${successCount}/${connectedAgents.length} agents`)
      }

      // Persist file refs in session config and update local state
      const currentSession = get().activeSession
      if (currentSession) {
        set({ activeSession: { ...currentSession, fileRefs }, pendingFiles: [] })
        await apiUpdateSession(currentSession.id, {
          config: JSON.stringify({
            ...currentSession.config,
            memoryRounds: currentSession.memoryRounds,
            originalMaxRounds: currentSession.originalMaxRounds,
            extensions: currentSession.extensions,
            autoAdvance: currentSession.autoAdvance,
            fileRefs,
            allowPass: currentSession.allowPass,
            allowDelegation: currentSession.allowDelegation,
            spawnedSlugs: currentSession.spawnedSlugs,
            agentsDisposed: currentSession.agentsDisposed,
          }),
        })
      }
    }

    // Send BTW context to all agents (includes file references)
    const session = get().activeSession!
    const btwPrompt = buildBtwPrompt(session)
    for (const agent of session.agents) {
      get()._sendToAgent(agent.id, btwPrompt, 'btw')
      get()._log(agent.id, agent.name, 'system', 'Council context sent')
    }
    await new Promise(r => setTimeout(r, 300))

    // Determine first speaker
    let firstSpeakerId = session.config.firstSpeaker
    if (!firstSpeakerId || firstSpeakerId === 'random') {
      const eligible = session.agents.filter(a => a.id !== session.moderatorAgentId)
      firstSpeakerId = eligible[Math.floor(Math.random() * eligible.length)]?.id || ''
    }

    if (!firstSpeakerId) return

    // Send task to first speaker
    await get()._addSystemMessage(1, `Council started. First speaker: ${session.agents.find(a => a.id === firstSpeakerId)?.name || 'Unknown'}`)
    const firstMsg = await get()._runSpeakerTurn(
      firstSpeakerId,
      1,
      [],
      'You are the first speaker — you set the factual baseline for the whole discussion. If your take depends on dates, current events, version numbers, prices, statistics, recent news, named people/projects, or anything you are not 100% confident about, use your tools (especially `web_search` and `web_fetch`) BEFORE answering. Do not speculate when a lookup is cheap. Cite the source domain or title for anything you fetched.',
    )
    if (!firstMsg) return

    // Now run the rest of round 1
    if (session.moderatorMode === 'moderator') {
      await get()._runModeratorRound(1)
    } else {
      await get()._runRoundRobin(1)
    }
  },

  advanceRound: async () => {
    get()._cancelAutoAdvanceTimer()
    const s = get().activeSession
    if (!s || s.status !== 'active') return

    const nextRound = s.currentRound + 1
    if (nextRound > s.maxRounds) {
      await get().requestSynthesis()
      return
    }

    // Mark running up-front so the brief await window before the round loop
    // starts isn't misread as an interrupted round.
    set({ roundRunning: true })

    set({ activeSession: { ...s, currentRound: nextRound } })
    await apiUpdateSession(s.id, { current_round: nextRound })
    await get()._addSystemMessage(nextRound, `Round ${nextRound} begins.`)

    const session = get().activeSession!
    if (session.moderatorMode === 'moderator') {
      await get()._runModeratorRound(nextRound)
    } else {
      await get()._runRoundRobin(nextRound)
    }
  },

  restartRound: async () => {
    const s = get().activeSession
    if (!s || s.status !== 'active') return
    const round = s.currentRound
    if (round < 1) return

    // Invalidate any in-flight round loop and unblock a waiting agent turn so we
    // don't wait out the (long) agent timeout.
    _roundEpoch += 1
    _abortCollect?.()
    get()._cancelAutoAdvanceTimer()
    _pendingWrittenFiles.clear()
    set({ speaking: '', roundRunning: true })

    // Drop this round's messages locally and on the server, then re-run it.
    set({
      activeSession: {
        ...get().activeSession!,
        messages: get().activeSession!.messages.filter(m => m.round !== round),
      },
    })
    await apiDeleteMessages(s.id, round)
    await get()._addSystemMessage(round, `Round ${round} restarted.`)

    const session = get().activeSession!
    if (session.moderatorMode === 'moderator') {
      await get()._runModeratorRound(round)
    } else {
      await get()._runRoundRobin(round)
    }
  },

  requestSynthesis: async () => {
    get()._cancelAutoAdvanceTimer()
    const s = get().activeSession
    if (!s) return

    // Reconnect agents if needed (e.g. a reloaded/concluded session) and pick a
    // reachable synthesizer BEFORE entering the 'synthesizing' state — otherwise
    // a disconnected synth agent leaves the session stuck in 'synthesizing'.
    const connected = await get()._ensureConnected()
    const synthAgent =
      (s.moderatorAgentId ? connected.find(a => a.id === s.moderatorAgentId) : undefined)
      || connected[0]
    if (!synthAgent) {
      get()._log('', 'System', 'system', 'No agents are reachable — start the council\'s agents and try again.')
      return  // status stays 'active' — not stuck
    }
    const synthId = synthAgent.id

    set({ activeSession: { ...get().activeSession!, status: 'synthesizing' } })
    await apiUpdateSession(s.id, { status: 'synthesizing' })
    await get()._addSystemMessage(s.currentRound, 'Synthesizing discussion...')
    get()._log('', 'System', 'system', 'Starting synthesis phase')

    const session = get().activeSession!
    const prompt = buildSynthesisPrompt(session)
    get()._sendToAgent(synthId, prompt)
    set({ speaking: synthId })

    const rawSynthesis = await get()._collectResponse(synthId)
    set({ speaking: '' })

    // Clean thinking blocks from synthesis too
    const response = stripThinkingBlocks(rawSynthesis).trim()

    await get()._addMessage({
      round: session.currentRound,
      agentId: synthId,
      agentName: synthAgent.name,
      role: 'synthesis',
      action: '',
      suitability: 1,
      targetAgentId: '',
      content: response,
      pinned: false,
      metadata: {},
    })

    // Optional: run voting
    await get()._addSystemMessage(session.currentRound, 'Voting on synthesis...')
    const votes: CouncilVote[] = []

    for (const agent of session.agents.filter(a => a.id !== synthId && !a.muted && a.ws?.connected)) {
      const votePrompt = buildVotePrompt(response)
      get()._sendToAgent(agent.id, votePrompt)
      set({ speaking: agent.id })
      const voteRawFull = await get()._collectResponse(agent.id)
      set({ speaking: '' })

      const voteRaw = stripThinkingBlocks(voteRawFull)
      const voteMatch = voteRaw.match(/^VOTE:\s*(AGREE|DISAGREE|ABSTAIN)/mi)
      const reasonMatch = voteRaw.match(/^REASON:\s*(.+)/mi)
      const v: CouncilVote = {
        id: 0,
        round: session.currentRound,
        agentId: agent.id,
        agentName: agent.name,
        vote: (voteMatch?.[1]?.toLowerCase() as Vote) || 'abstain',
        reason: reasonMatch?.[1]?.trim() || '',
      }
      votes.push(v)
    }

    if (votes.length > 0) {
      await apiAddVotes(session.id, votes.map(v => ({
        round: v.round, agent_id: v.agentId, agent_name: v.agentName,
        vote: v.vote, reason: v.reason,
      })))
      const current = get().activeSession
      if (current) set({ activeSession: { ...current, votes: [...current.votes, ...votes] } })
    }

    await get()._addSystemMessage(session.currentRound, 'Synthesis and voting complete.')

    // Move to concluded state and disconnect agents
    const final = get().activeSession
    if (final) {
      const concludedAt = new Date().toISOString()
      set({ activeSession: { ...final, status: 'concluded', concludedAt } })
      await apiUpdateSession(final.id, { status: 'concluded' })
      _persistConfig(final.id, { concludedAt })
      get().disconnectAllAgents()
      await get().loadSessionList()
    }
  },

  concludeSession: async () => {
    get()._cancelAutoAdvanceTimer()
    const s = get().activeSession
    if (!s) return
    const concludedAt = new Date().toISOString()
    set({ activeSession: { ...s, status: 'concluded', concludedAt } })
    await apiUpdateSession(s.id, { status: 'concluded' })
    _persistConfig(s.id, { concludedAt })
    get().disconnectAllAgents()
    await get()._addSystemMessage(s.currentRound, 'Council session concluded.')
    await get().loadSessionList()
  },

  // ── User interactions ──

  injectMessage: async (content) => {
    const s = get().activeSession
    if (!s) return
    await get()._addMessage({
      round: s.currentRound,
      agentId: '',
      agentName: 'User',
      role: 'user',
      action: 'inject',
      suitability: 0,
      targetAgentId: '',
      content,
      pinned: false,
      metadata: {},
    })
  },

  directAddress: async (agentId, content) => {
    const s = get().activeSession
    if (!s) return

    await get().injectMessage(content)

    const agent = s.agents.find(a => a.id === agentId)
    if (!agent?.ws?.connected) return

    const msgs = get().activeSession!.messages
    await get()._runSpeakerTurn(agentId, s.currentRound, msgs.slice(-5), content)
  },

  muteAgent: (agentId, muted) => {
    const s = get().activeSession
    if (!s) return
    const agents = s.agents.map(a => a.id === agentId ? { ...a, muted } : a)
    set({ activeSession: { ...s, agents } })
    // Persist
    const defs = agents.map(({ ws, connected, busy, ...rest }) => rest)
    apiUpdateSession(s.id, { agents: JSON.stringify(defs) })
  },

  pinMessage: (messageId) => {
    const s = get().activeSession
    if (!s) return
    const messages = s.messages.map(m => m.id === messageId ? { ...m, pinned: !m.pinned } : m)
    set({ activeSession: { ...s, messages } })
    apiTogglePin(s.id, messageId)
  },

  setMemoryRounds: (value) => {
    const s = get().activeSession
    if (!s) return
    set({ activeSession: { ...s, memoryRounds: value } })
    _persistConfig(s.id, { memoryRounds: value })
    get()._log('', 'System', 'system', `Council memory set to ${value === 0 ? 'indefinite' : `${value} rounds`}`)
  },

  setAllowPass: (value) => {
    const s = get().activeSession
    if (!s) return
    set({ activeSession: { ...s, allowPass: value } })
    _persistConfig(s.id, { allowPass: value })
    get()._log(
      '',
      'System',
      'system',
      `Council passing ${value ? 'enabled' : 'disabled — participants must contribute every round'}`,
    )
  },

  setAllowDelegation: (value) => {
    const s = get().activeSession
    if (!s) return
    set({ activeSession: { ...s, allowDelegation: value } })
    _persistConfig(s.id, { allowDelegation: value })
    get()._log(
      '',
      'System',
      'system',
      `Agent delegation ${value ? 'enabled — agents may orchestrate during their turn' : 'disabled — agents contribute directly, no sub-delegation'}`,
    )
  },

  setAutoAdvance: (value) => {
    const s = get().activeSession
    if (!s) return
    set({ activeSession: { ...s, autoAdvance: value } })
    _persistConfig(s.id, { autoAdvance: value })
    get()._log('', 'System', 'system', `Auto-advance ${value ? 'enabled' : 'disabled'}`)
    // If disabling and timer is running, cancel it
    if (!value) get()._cancelAutoAdvanceTimer()
  },

  extendRounds: async (amount) => {
    const s = get().activeSession
    if (!s) return
    const newMax = s.maxRounds + amount
    const newExtensions = [...s.extensions, amount]
    set({ activeSession: { ...s, maxRounds: newMax, extensions: newExtensions } })
    await apiUpdateSession(s.id, { max_rounds: newMax })
    _persistConfig(s.id, { extensions: newExtensions })
    get()._log('', 'System', 'system', `Council extended by ${amount} rounds (now ${newMax} total)`)
    // Inform agents about the extension
    await get()._addSystemMessage(
      s.currentRound,
      `Council extended by +${amount} rounds. New limit: ${newMax} rounds (originally ${s.originalMaxRounds}).`,
    )
    // If council was concluded, reactivate it
    if (s.status === 'concluded') {
      const current = get().activeSession!
      set({ activeSession: { ...current, status: 'active' } })
      await apiUpdateSession(current.id, { status: 'active' })
      get().connectAllAgents()
      get()._log('', 'System', 'system', 'Council reactivated after extension')
    }
  },

  cancelAutoAdvance: () => {
    get()._cancelAutoAdvanceTimer()
  },

  // ── Minutes & TL;DR ──

  generateTldrs: async () => {
    const s = get().activeSession
    if (!s) return

    set({ generatingArtifact: 'tldr' })
    get()._log('', 'System', 'system', 'Generating TL;DRs from all agents...')

    // Connect and wait for agents to hold a connection; bail clearly if offline.
    const needReconnect = s.agents.every(a => !a.ws?.connected)
    const connectedAgents = await get()._ensureConnected()
    if (connectedAgents.length === 0) {
      if (needReconnect) get().disconnectAllAgents()
      set({ generatingArtifact: '' })
      get()._log('', 'System', 'system', 'No agents are reachable — start the council\'s agents and try again.')
      return
    }

    const prompt = buildTldrPrompt(s)
    const newArtifacts: CouncilArtifact[] = []

    const currentSession = get().activeSession
    if (!currentSession) { set({ generatingArtifact: '' }); return }

    for (const agent of connectedAgents) {
      get()._log(agent.id, agent.name, 'speaking', 'Generating TL;DR...')
      set({ speaking: agent.id })
      get()._sendToAgent(agent.id, prompt)
      const responseRaw = await get()._collectResponse(agent.id)
      const response = stripThinkingBlocks(responseRaw)
      set({ speaking: '' })
      get()._log(agent.id, agent.name, 'done', 'TL;DR complete')

      const artId = await apiUpsertArtifact(s.id, 'tldr', agent.id, agent.name, response.trim())
      newArtifacts.push({
        id: artId,
        sessionId: s.id,
        kind: 'tldr',
        agentId: agent.id,
        agentName: agent.name,
        content: response.trim(),
        createdAt: new Date().toISOString(),
      })
    }

    // Update local state: replace existing TL;DRs with new ones
    const updated = get().activeSession
    if (updated) {
      const nonTldr = updated.artifacts.filter(a => a.kind !== 'tldr')
      set({ activeSession: { ...updated, artifacts: [...nonTldr, ...newArtifacts] } })
    }

    // Disconnect if we reconnected for this
    if (needReconnect && s.status === 'concluded') get().disconnectAllAgents()
    set({ generatingArtifact: '' })
    get()._log('', 'System', 'system', `TL;DRs generated from ${newArtifacts.length} agents`)
  },

  generateActionPoints: async () => {
    const s = get().activeSession
    if (!s) return

    set({ generatingArtifact: 'action_points' })
    get()._log('', 'System', 'system', 'Extracting action points from all agents...')

    // Connect and wait for agents to actually hold a connection. If none do, the
    // council's agents are offline/unreachable — bail with a clear message rather
    // than reporting "0 agents" after a fixed delay.
    const needReconnect = s.agents.every(a => !a.ws?.connected)
    const connectedAgents = await get()._ensureConnected()
    if (connectedAgents.length === 0) {
      if (needReconnect) get().disconnectAllAgents()
      set({ generatingArtifact: '' })
      get()._log('', 'System', 'system', 'No agents are reachable — start the council\'s agents and try again.')
      return
    }

    const currentSession = get().activeSession
    if (!currentSession) { set({ generatingArtifact: '' }); return }
    const newArtifacts: CouncilArtifact[] = []

    // Map per agent — each call is scoped to that agent's own slice, never the
    // whole transcript, so context stays small regardless of council size.
    for (const agent of connectedAgents) {
      get()._log(agent.id, agent.name, 'speaking', 'Extracting action points...')
      set({ speaking: agent.id })
      // no_tools: hard enforcement — the agent literally cannot call any tool
      // this turn, so extraction can only describe action points, never execute.
      get()._sendToAgent(agent.id, buildActionPointsPrompt(currentSession, agent.id), 'chat', { noTools: true })
      const raw = await get()._collectResponse(agent.id)
      set({ speaking: '' })
      const items = parseActionPoints(stripThinkingBlocks(raw))
      get()._log(agent.id, agent.name, 'done', `${items.length} action point(s)`)
      if (items.length === 0) continue
      const content = JSON.stringify(items)
      const artId = await apiUpsertArtifact(s.id, 'action_points', agent.id, agent.name, content)
      newArtifacts.push({
        id: artId, sessionId: s.id, kind: 'action_points',
        agentId: agent.id, agentName: agent.name, content,
        createdAt: new Date().toISOString(),
      })
    }

    const updated = get().activeSession
    if (updated) {
      const others = updated.artifacts.filter(a => a.kind !== 'action_points')
      set({ activeSession: { ...updated, artifacts: [...others, ...newArtifacts] } })
    }

    if (needReconnect && s.status === 'concluded') get().disconnectAllAgents()
    set({ generatingArtifact: '' })
    get()._log('', 'System', 'system', `Action points extracted from ${newArtifacts.length} agent(s)`)
  },

  sendActionPointToAgent: async (agentId, item) => {
    const s = get().activeSession
    if (!s) return
    const name = s.agents.find(a => a.id === agentId)?.name || 'Agent'

    // Reconnect (revives dropped sockets) and wait specifically for THIS agent.
    await get()._ensureConnected()
    const deadline = Date.now() + 12000
    let agent = get().activeSession?.agents.find(a => a.id === agentId)
    while (!agent?.ws?.connected && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 500))
      agent = get().activeSession?.agents.find(a => a.id === agentId)
    }
    if (!agent?.ws?.connected) {
      get()._log(agentId, name, 'system', 'Could not connect agent to record action point')
      throw new Error('agent not reachable')
    }

    const toolDesc = item.kind === 'intent'
      ? 'a proactive intention (use your `intentions` tool)'
      : 'a todo (use your `todo` tool)'
    const body = [
      `Record the following as ${toolDesc}. Do NOT act on it now — just record it so it persists for you to work on later.`,
      'IMPORTANT: preserve ALL of the detail below verbatim in the recorded item — put the full Task, Done-when, and References into the item\'s prompt/notes/body so you keep complete context when you eventually act on it. Do not summarise it away.',
      '',
      `Title: ${item.title}`,
      `Context: ${item.context}`,
      `Task: ${item.task}`,
      `Done when: ${item.done_when}`,
      item.refs.length ? `References: ${item.refs.join(', ')}` : '',
    ].filter(Boolean).join('\n')

    set({ speaking: agentId })
    get()._sendToAgent(agentId, body)
    await get()._collectResponse(agentId)
    set({ speaking: '' })
    get()._log(agentId, agent.name, 'system', `Recorded action point "${item.title}" → ${agent.name}`)

    // Persist that this item was sent so the "Recorded" state survives a reload
    // and it can't be re-sent (which would create a duplicate todo/intention).
    const cur = get().activeSession
    const art = cur?.artifacts.find(a => a.kind === 'action_points' && a.agentId === agentId)
    if (cur && art) {
      try {
        const items: ActionPoint[] = JSON.parse(art.content)
        const idx = items.findIndex(it => it.title === item.title && it.task === item.task)
        if (idx >= 0 && !items[idx].sent) {
          items[idx] = { ...items[idx], sent: true }
          const content = JSON.stringify(items)
          set({ activeSession: { ...cur, artifacts: cur.artifacts.map(a => a.id === art.id ? { ...a, content } : a) } })
          await apiUpsertArtifact(cur.id, 'action_points', agentId, art.agentName, content)
        }
      } catch { /* ignore — confirmation just won't persist */ }
    }
  },

  exportMinutesMd: () => {
    const s = get().activeSession
    if (!s) return
    const tldrs = s.artifacts.filter(a => a.kind === 'tldr')
    const md = generateMinutesMd(s, tldrs)
    const filename = `council-${s.title.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.md`
    downloadFile(md, filename, 'text/markdown')
    get()._log('', 'System', 'system', 'Exported minutes as Markdown')
  },

  // ── Internal helpers ──

  _sendToAgent: (agentId, content, type = 'chat', opts) => {
    const s = get().activeSession
    if (!s) return
    const agent = s.agents.find(a => a.id === agentId)
    if (!agent?.ws) return
    if (type === 'btw') agent.ws.sendBtw(content)
    else if (opts?.noTools) agent.ws.sendJSON({ type: 'chat', content, no_tools: true })
    else if (opts?.denyTools?.length) agent.ws.sendJSON({ type: 'chat', content, deny_tools: opts.denyTools })
    else agent.ws.send(content)
  },

  _collectResponse: (agentId, onChunk) => {
    return new Promise<string>((resolve) => {
      const s = get().activeSession
      if (!s) { resolve(''); return }
      const agent = s.agents.find(a => a.id === agentId)
      if (!agent?.ws) { resolve(''); return }

      let accumulated = ''
      const unsubs: (() => void)[] = []

      const cleanup = () => {
        unsubs.forEach(u => u())
        clearTimeout(timer)
        if (_abortCollect === abort) _abortCollect = null
      }

      const finish = (val: string) => { cleanup(); resolve(val) }
      // Lets restartRound() unblock this turn immediately instead of waiting out
      // the (long) agent timeout. The caller checks the round epoch afterward.
      const abort = () => finish(accumulated)
      _abortCollect = abort

      const timer = setTimeout(() => {
        finish(accumulated || '(Agent timed out)')
      }, AGENT_TIMEOUT_MS)

      unsubs.push(agent.ws!.on('chat_message', (data) => {
        if (data.role === 'assistant') {
          accumulated += (data.content as string) || ''
          onChunk?.(accumulated)
        }
      }))

      unsubs.push(agent.ws!.on('status', (data) => {
        const text = (data.text as string || data.status as string || '').toLowerCase()
        if (!text || /^(ready|idle|done|completed)$/i.test(text)) {
          finish(accumulated)
        }
      }))

      // If the agent drops mid-turn (e.g. an unreachable/flapping agent), resolve
      // with whatever we have instead of hanging until the long agent timeout.
      unsubs.push(agent.ws!.on('_disconnected', () => finish(accumulated)))
    })
  },

  _addMessage: async (msg) => {
    const s = get().activeSession
    if (!s) return
    const localId = `local-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const newMsg: CouncilMessage = {
      ...msg,
      id: 0,
      localId,
      createdAt: new Date().toISOString(),
    }

    // Add to local state immediately
    set({ activeSession: { ...s, messages: [...s.messages, newMsg] } })

    // Persist to server
    const ids = await apiAddMessages(s.id, [{
      round: msg.round,
      agent_id: msg.agentId,
      agent_name: msg.agentName,
      role: msg.role,
      action: msg.action,
      suitability: msg.suitability,
      target_agent_id: msg.targetAgentId,
      content: msg.content,
      pinned: msg.pinned ? 1 : 0,
      metadata: JSON.stringify(msg.metadata),
    }])

    // Update local message with server ID
    if (ids[0]) {
      set(state => {
        if (!state.activeSession) return state
        return {
          activeSession: {
            ...state.activeSession,
            messages: state.activeSession.messages.map(m =>
              m.localId === localId ? { ...m, id: ids[0] } : m,
            ),
          },
        }
      })
    }
  },

  _addSystemMessage: async (round, content) => {
    await get()._addMessage({
      round,
      agentId: '',
      agentName: 'System',
      role: 'system',
      action: '',
      suitability: 0,
      targetAgentId: '',
      content,
      pinned: false,
      metadata: {},
    })
  },

  _persistSession: async () => {
    const s = get().activeSession
    if (!s) return
    const defs = s.agents.map(({ ws, connected, busy, ...rest }) => rest)
    await apiUpdateSession(s.id, {
      status: s.status,
      current_round: s.currentRound,
      agents: JSON.stringify(defs),
    })
  },

  _startAutoAdvanceTimer: () => {
    // Cancel any existing timer first
    get()._cancelAutoAdvanceTimer()
    const s = get().activeSession
    if (!s || !s.autoAdvance || s.status !== 'active') return
    // Don't start if we've reached max rounds
    if (s.currentRound >= s.maxRounds) return

    set({ autoAdvanceCountdown: 5 })
    get()._log('', 'System', 'system', 'Auto-advance: next round in 5 seconds...')

    _autoAdvanceTimerId = setInterval(() => {
      const countdown = get().autoAdvanceCountdown
      if (countdown <= 1) {
        get()._cancelAutoAdvanceTimer()
        // Trigger advance
        get().advanceRound()
      } else {
        set({ autoAdvanceCountdown: countdown - 1 })
      }
    }, 1000)
  },

  _cancelAutoAdvanceTimer: () => {
    if (_autoAdvanceTimerId !== null) {
      clearInterval(_autoAdvanceTimerId)
      _autoAdvanceTimerId = null
    }
    if (get().autoAdvanceCountdown > 0) {
      set({ autoAdvanceCountdown: 0 })
    }
  },

  _runSpeakerTurn: async (agentId, round, recentMsgs, directive?) => {
    const s = get().activeSession
    if (!s) return null
    const epoch = _roundEpoch
    // Deny orchestration/delegation tools for this turn unless the session opts
    // in — keeps a discussion turn a discussion, not a delegation loop.
    const turnOpts = s.allowDelegation ? undefined : { denyTools: DELEGATION_TOOLS }

    const agent = s.agents.find(a => a.id === agentId)
    if (!agent?.ws?.connected || agent.muted) {
      await get()._addSystemMessage(round, `${agent?.name || 'Agent'} is ${agent?.muted ? 'muted' : 'disconnected'} — skipping.`)
      return null
    }

    const prompt = buildTurnPrompt(s, round, recentMsgs, directive, agent.name, agentId)
    get()._sendToAgent(agentId, prompt, 'chat', turnOpts)
    set({ speaking: agentId })
    get()._log(agentId, agent.name, 'speaking', `Speaking (round ${round})`)

    // Partial-response checkpointing: when a turn runs long, lazily (after a few
    // seconds) persist the raw accumulated text to the server and keep updating
    // it, so a mid-turn server kill preserves what was generated. The row is
    // finalized in place below (or deleted by a restart). Fast turns never hit
    // the threshold and just take the normal add-at-end path.
    let partialId: number | null = null
    let partialCreating = false
    let lastCheckpoint = 0
    const checkpoint = (acc: string) => {
      if (_roundEpoch !== epoch || partialCreating) return
      const now = Date.now()
      if (now - lastCheckpoint < 3000) return
      lastCheckpoint = now
      if (partialId == null) {
        partialCreating = true
        apiAddMessages(s.id, [{
          round, agent_id: agentId, agent_name: agent.name, role: 'agent',
          action: '', suitability: 0, target_agent_id: '', content: acc,
          pinned: 0, metadata: JSON.stringify({ partial: true }),
        }]).then(ids => { partialId = ids[0] ?? null }).finally(() => { partialCreating = false })
      } else {
        apiUpdateMessage(s.id, partialId, { content: acc, metadata: JSON.stringify({ partial: true }) })
      }
    }

    let raw = await get()._collectResponse(agentId, checkpoint)
    set({ speaking: '' })
    // Round was restarted/invalidated while this agent was speaking — drop the
    // turn entirely so it doesn't pollute the re-run.
    if (_roundEpoch !== epoch) return null

    // Agent reported being stuck — nudge it to continue (as a human would in
    // manual mode) and re-collect, blocking this turn until it gives a real
    // answer or we exhaust the nudge budget. The next agent can't speak until
    // _runSpeakerTurn returns, so this naturally holds the round here.
    let nudges = 0
    while (isStuckResponse(stripThinkingBlocks(raw)) && nudges < MAX_STUCK_NUDGES) {
      nudges++
      get()._log(agentId, agent.name, 'system', `Reported stuck — nudging to continue (${nudges}/${MAX_STUCK_NUDGES})`)
      get()._sendToAgent(agentId, STUCK_NUDGE_PROMPT, 'chat', turnOpts)
      set({ speaking: agentId })
      raw = await get()._collectResponse(agentId, checkpoint)
      set({ speaking: '' })
      if (_roundEpoch !== epoch) return null
    }
    if (nudges > 0 && isStuckResponse(stripThinkingBlocks(raw))) {
      get()._log(agentId, agent.name, 'system', 'Still stuck after nudges — accepting response and moving on')
    }

    get()._log(agentId, agent.name, 'done', 'Finished speaking')
    // Clear agent's tool history after turn
    set(state => {
      if (!state.activeSession) return state
      return {
        activeSession: {
          ...state.activeSession,
          agents: state.activeSession.agents.map(ag =>
            ag.id === agentId ? { ...ag, statusText: '', toolHistory: [] } : ag,
          ),
        },
      }
    })

    let parsed = parseAgentResponse(raw)

    // Pass enforcement. If the session disallows passing and the
    // model passed anyway, re-issue the turn ONCE with a hard
    // directive demanding a real contribution. We only retry once —
    // if the model passes a second time, we accept it rather than
    // looping forever, but we tag the message so it's visible in the
    // log that this happened.
    if (parsed.action === 'pass' && s.allowPass === false) {
      get()._log(
        agentId,
        agent.name,
        'system',
        'Passed but session has passing disabled — retrying with hard directive',
      )
      const retryDirective =
        'Your previous response used ACTION: pass, but passing is DISABLED for this council session. ' +
        'You must contribute. Pick one of: answer | respond | challenge | refine | broaden — and share your reasoning, ' +
        'even if the topic is outside your specialty. A "here is how I would think about this if I had to decide" framing ' +
        'is more valuable than silence. Do not pass again.'
      const retryPrompt = buildTurnPrompt(
        s,
        round,
        recentMsgs,
        directive ? `${directive}\n\n${retryDirective}` : retryDirective,
        agent.name,
        agentId,
      )
      get()._sendToAgent(agentId, retryPrompt, 'chat', turnOpts)
      set({ speaking: agentId })
      const retryRaw = await get()._collectResponse(agentId, checkpoint)
      set({ speaking: '' })
      if (_roundEpoch !== epoch) return null
      const retryParsed = parseAgentResponse(retryRaw)
      if (retryParsed.action !== 'pass') {
        // Retry succeeded — use it.
        parsed = retryParsed
      } else {
        // Retry also passed; keep the original but log it.
        get()._log(
          agentId,
          agent.name,
          'system',
          'Retry also passed — accepting pass to avoid infinite loop',
        )
      }
    }

    // Resolve target agent ID from name
    let targetAgentId = ''
    if (parsed.targetAgent) {
      const target = s.agents.find(a =>
        a.name.toLowerCase() === parsed.targetAgent.toLowerCase(),
      )
      targetAgentId = target?.id || ''
    }

    // Drain any file writes that this agent performed during the turn and
    // transfer them to every other council agent via Flight Deck so peers can
    // actually read them in the next round.
    const writtenFiles = _pendingWrittenFiles.get(agentId) || []
    _pendingWrittenFiles.delete(agentId)
    if (writtenFiles.length > 0) {
      await get()._shareWrittenFiles(agentId, agent.name, round, writtenFiles)
    }

    const metadata: Record<string, unknown> = {}
    if (writtenFiles.length > 0) metadata.writtenFiles = writtenFiles

    const msg: Omit<CouncilMessage, 'id' | 'localId' | 'createdAt'> = {
      round,
      agentId,
      agentName: agent.name,
      role: 'agent',
      action: parsed.action,
      suitability: parsed.suitability,
      targetAgentId,
      content: parsed.content,
      pinned: false,
      metadata,
    }

    // If a checkpoint create is still in flight, wait briefly so we finalize
    // that row rather than orphaning it with a second message.
    for (let i = 0; i < 20 && partialCreating; i++) {
      await new Promise(r => setTimeout(r, 50))
    }

    if (partialId != null) {
      // Finalize the checkpointed row in place: clear the partial flag, write the
      // parsed content on the server, and surface it in the local view now.
      await apiUpdateMessage(s.id, partialId, {
        content: parsed.content,
        action: parsed.action,
        suitability: parsed.suitability,
        target_agent_id: targetAgentId,
        metadata: JSON.stringify(metadata),
      })
      const finalMsg: CouncilMessage = {
        ...msg, id: partialId, localId: `srv-${partialId}`, createdAt: new Date().toISOString(),
      }
      set(state => state.activeSession
        ? { activeSession: { ...state.activeSession, messages: [...state.activeSession.messages, finalMsg] } }
        : state)
      return finalMsg
    }

    await get()._addMessage(msg)
    const current = get().activeSession
    return current?.messages[current.messages.length - 1] || null
  },

  _shareWrittenFiles: async (srcAgentId, srcAgentName, round, paths) => {
    const s = get().activeSession
    if (!s) return
    const srcAgent = s.agents.find(a => a.id === srcAgentId)
    if (!srcAgent) return
    const peers = s.agents.filter(
      a => a.id !== srcAgentId && a.connected && !a.muted,
    )
    if (peers.length === 0) return

    const existing = _sessionSharedFiles.get(s.id) || []
    for (const srcPath of paths) {
      const entry: SharedFileEntry = {
        srcPath,
        srcAgentId,
        srcAgentName,
        round,
        destPaths: {},
      }
      // Transfer to each peer in parallel; the dest path comes back from the
      // server (it may differ from srcPath if the peer's sandbox renames it).
      const transfers = peers.map(async (peer) => {
        try {
          const result = await transferFile(
            { id: srcAgent.id, name: srcAgent.name, host: srcAgent.host, port: srcAgent.port, auth: srcAgent.auth },
            { id: peer.id, name: peer.name, host: peer.host, port: peer.port, auth: peer.auth },
            srcPath,
          )
          if (result?.dest_path) {
            entry.destPaths[peer.id] = result.dest_path
          }
        } catch (err) {
          get()._log(
            srcAgentId,
            srcAgentName,
            'error',
            `Failed to share ${srcPath} with ${peer.name}: ${(err as Error).message}`,
          )
        }
      })
      await Promise.all(transfers)
      if (Object.keys(entry.destPaths).length > 0) {
        existing.push(entry)
        const recipients = Object.entries(entry.destPaths).map(([pid, path]) => {
          const peer = s.agents.find(a => a.id === pid)
          return { agentId: pid, agentName: peer?.name || pid, path }
        })
        const fileName = srcPath.split('/').pop() || srcPath
        await get()._addMessage({
          round,
          agentId: '',
          agentName: 'Flight Deck',
          role: 'system',
          action: '',
          suitability: 0,
          targetAgentId: '',
          content: `Shared ${fileName} from ${srcAgentName} to ${recipients.length} agent${recipients.length === 1 ? '' : 's'}.`,
          pinned: false,
          metadata: {
            kind: 'file_share',
            fileName,
            srcPath,
            srcAgentId,
            srcAgentName,
            recipients,
          },
        })
      }
    }
    _sessionSharedFiles.set(s.id, existing)
  },

  _runRoundRobin: async (round) => {
    const s = get().activeSession
    if (!s) return
    const epoch = _roundEpoch
    set({ roundRunning: true })

    // Agents who haven't spoken in this round yet (exclude moderator)
    const roundMessages = s.messages.filter(m => m.round === round && m.role === 'agent')
    const spokenIds = new Set(roundMessages.map(m => m.agentId))
    const eligible = s.agents.filter(a =>
      !spokenIds.has(a.id) && a.id !== s.moderatorAgentId && !a.muted,
    )

    // Use suitability from last message if available, skip separate prompt for speed
    const scores = new Map<string, number>()
    for (const a of eligible) {
      const lastMsg = [...s.messages].reverse().find(m => m.agentId === a.id && m.role === 'agent')
      scores.set(a.id, lastMsg?.suitability ?? 0.5)
    }

    // Sort eligible agents by suitability score (highest first)
    const sorted = [...eligible].sort((a, b) =>
      (scores.get(b.id) || 0.5) - (scores.get(a.id) || 0.5),
    )

    // Ensure the last speaker from the previous round doesn't go first
    if (round > 1 && sorted.length > 1) {
      const prevRoundMsgs = s.messages.filter(m => m.round === round - 1 && m.role === 'agent')
      const lastSpeakerId = prevRoundMsgs.length > 0
        ? prevRoundMsgs[prevRoundMsgs.length - 1].agentId
        : null
      if (lastSpeakerId && sorted[0]?.id === lastSpeakerId) {
        const [first, ...rest] = sorted
        sorted.splice(0, sorted.length, ...rest, first)
      }
    }

    for (const agent of sorted) {
      const currentSession = get().activeSession
      if (!currentSession || currentSession.status !== 'active' || _roundEpoch !== epoch) break

      // Collect context from prior rounds within memory budget
      const contextMsgs = collectContextMessages(
        currentSession.messages, round, currentSession.memoryRounds,
      )
      await get()._runSpeakerTurn(agent.id, round, contextMsgs)
    }

    // A restart bumped the epoch mid-round — the re-run owns completion (and
    // the roundRunning flag), so bail without touching it.
    if (_roundEpoch !== epoch) return
    await get()._addSystemMessage(round, `Round ${round} complete. Approve next round or request synthesis.`)
    set({ roundRunning: false })
    // Start auto-advance countdown if enabled
    get()._startAutoAdvanceTimer()
  },

  _runModeratorRound: async (round) => {
    const s = get().activeSession
    if (!s || !s.moderatorAgentId) return
    const epoch = _roundEpoch
    set({ roundRunning: true })

    const moderator = s.agents.find(a => a.id === s.moderatorAgentId)
    if (!moderator?.ws?.connected) {
      // Fall back to round-robin
      await get()._runRoundRobin(round)
      return
    }

    const totalNonMod = s.agents.filter(a => a.id !== s.moderatorAgentId && !a.muted).length
    const spokenThisRound = new Set<string>()

    // Find last speaker from previous round to avoid picking them first
    let lastSpeakerPrevRound: string | null = null
    if (round > 1) {
      const prevRoundMsgs = s.messages.filter(m => m.round === round - 1 && m.role === 'agent')
      if (prevRoundMsgs.length > 0) {
        lastSpeakerPrevRound = prevRoundMsgs[prevRoundMsgs.length - 1].agentId
      }
    }

    // Let moderator manage turns until all agents have spoken
    for (let turn = 0; turn < totalNonMod; turn++) {
      const currentSession = get().activeSession
      if (!currentSession || currentSession.status !== 'active' || _roundEpoch !== epoch) break

      // Get suitability scores
      const scores = await get()._getSuitabilityScores()

      // Ask moderator to select next speaker
      const scoreSummary = currentSession.agents
        .filter(a => a.id !== s.moderatorAgentId && !a.muted)
        .map(a => ({
          name: a.name,
          score: scores.get(a.id) || 0.5,
          spoken: spokenThisRound.has(a.id),
          spokeLastPrevRound: turn === 0 && a.id === lastSpeakerPrevRound,
        }))

      const contextMsgs = collectContextMessages(
        currentSession.messages, round, currentSession.memoryRounds,
      )
      const modPrompt = buildModeratorSelectPrompt(scoreSummary, contextMsgs)
      get()._sendToAgent(s.moderatorAgentId, modPrompt)
      set({ speaking: s.moderatorAgentId })
      const modResponseRaw = await get()._collectResponse(s.moderatorAgentId)
      set({ speaking: '' })
      if (_roundEpoch !== epoch) break

      // Parse moderator's selection (strip thinking blocks first)
      const modResponse = stripThinkingBlocks(modResponseRaw)
      const selectedName = modResponse.trim().split('\n')[0].trim()
      const selectedAgent = currentSession.agents.find(a =>
        a.name.toLowerCase() === selectedName.toLowerCase() &&
        a.id !== s.moderatorAgentId,
      )

      if (!selectedAgent) {
        await get()._addSystemMessage(round, `Moderator selected "${selectedName}" — agent not found. Skipping.`)
        continue
      }

      get()._log(s.moderatorAgentId, moderator.name, 'moderator', `Selected next speaker: ${selectedAgent.name}`)
      await get()._addMessage({
        round,
        agentId: s.moderatorAgentId,
        agentName: moderator.name,
        role: 'moderator',
        action: '',
        suitability: 1,
        targetAgentId: selectedAgent.id,
        content: `Next speaker: ${selectedAgent.name}`,
        pinned: false,
        metadata: {},
      })

      const freshCtx = collectContextMessages(
        get().activeSession?.messages || [], round, s.memoryRounds,
      )
      await get()._runSpeakerTurn(selectedAgent.id, round, freshCtx)
      spokenThisRound.add(selectedAgent.id)
    }

    // A restart bumped the epoch mid-round — the re-run owns completion (and
    // the roundRunning flag), so bail without touching it.
    if (_roundEpoch !== epoch) return
    await get()._addSystemMessage(round, `Round ${round} complete. Approve next round or request synthesis.`)
    set({ roundRunning: false })
    // Start auto-advance countdown if enabled
    get()._startAutoAdvanceTimer()
  },

  _getSuitabilityScores: async () => {
    const s = get().activeSession
    if (!s) return new Map()

    const scores = new Map<string, number>()
    const eligible = s.agents.filter(a => a.id !== s.moderatorAgentId && !a.muted && a.ws?.connected)

    for (const agent of eligible) {
      const prompt = renderTemplate(suitabilityTemplate, { topic: s.topic })
      get()._sendToAgent(agent.id, prompt)
    }

    // Collect all scores in parallel
    const results = await Promise.all(
      eligible.map(async (agent) => {
        const raw = stripThinkingBlocks(await get()._collectResponse(agent.id))
        const num = parseFloat(raw.trim())
        return { id: agent.id, score: isNaN(num) ? 0.5 : Math.max(0, Math.min(1, num)) }
      }),
    )

    for (const r of results) scores.set(r.id, r.score)
    return scores
  },
}))
