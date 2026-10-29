import { isQueueCommand, QUEUE_MAX_TRIES, QUEUE_QUESTION_HOLD_MS, type QueuedMessage } from '../../../stores/chatStore'
import { fmtTokens } from '../../../utils/format'

// The message queue's derived state: what is running, what waits, what is
// done, and the ONE hero card that says what is happening, what happens next
// by itself, and what the user can do. Pure (no React, no store reads), so
// every panel variant shows the same thing for the same state.

// ── Formatting ──

/** "14:32:07" on the same day, "Jul 21 14:32:07" otherwise. */
export function fmtClock(ms?: number): string {
  if (!ms) return ''
  const d = new Date(ms)
  const time = d.toLocaleTimeString(undefined, { hour12: false })
  const sameDay = new Date().toDateString() === d.toDateString()
  return sameDay
    ? time
    : `${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${time}`
}

/** A duration as 42s / 3m 07s / 1h 04m. */
export function fmtDuration(ms: number): string {
  if (!(ms > 0)) return ''
  const secs = Math.round(ms / 1000)
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const rest = secs % 60
  if (m < 60) return `${m}m ${String(rest).padStart(2, '0')}s`
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}

/** Elapsed as 42s / 3m 07s / 1h 04m. */
export function fmtElapsed(from?: number, to?: number): string {
  if (!from || !to || to < from) return ''
  if (to - from < 500) return '0s'
  return fmtDuration(to - from)
}

/** The first line of a task, whitespace folded, cut at `max`. */
export function firstLine(text: string, max = 60): string {
  const line = (text.split('\n').find((l) => l.trim()) || '').replace(/\s+/g, ' ').trim()
  return line.length > max ? `${line.slice(0, max - 1).trimEnd()}…` : line
}

// ── Attachment refs ──
//
// A queued task carries its files as ref lines (appendAttachmentRefs). They
// are shown as chips and stripped from the text on screen only: what is sent
// to the agent stays byte-identical.
const IMAGE_REF = /^\[Attached image:\s*(.+?)\s*\]$/
const FILE_REF = /^\[Attached file: (.+?) → (.+)\]$/
// The hint earlier builds appended after image refs: kept, never shown.
const LEGACY_HINT = /^\(To view\/understand the image\(s\), call image_vision.*\)$/

export interface AttachedFile { kind: 'image' | 'file'; name: string; path: string; ref: string }

const baseName = (p: string) => p.split(/[\\/]/).filter(Boolean).pop() || p

export function parseAttachmentRefs(content: string): { text: string; refs: string[]; files: AttachedFile[] } {
  const refs: string[] = []
  const files: AttachedFile[] = []
  const kept: string[] = []
  for (const line of (content || '').split('\n')) {
    const t = line.trim()
    const img = t.match(IMAGE_REF)
    if (img) { refs.push(t); files.push({ kind: 'image', name: baseName(img[1]), path: img[1], ref: t }); continue }
    const file = t.match(FILE_REF)
    if (file) { refs.push(t); files.push({ kind: 'file', name: file[1], path: file[2], ref: t }); continue }
    if (LEGACY_HINT.test(t)) { refs.push(t); continue }
    kept.push(line)
  }
  return { text: kept.join('\n').trim(), refs, files }
}

/** Text and refs back into one task, as appendAttachmentRefs writes it. The
 *  old image hint only goes along while a picture does. */
export function joinAttachmentRefs(text: string, refs: string[]): string {
  const hasImage = refs.some((r) => IMAGE_REF.test(r))
  const kept = refs.filter((r) => hasImage || !LEGACY_HINT.test(r))
  if (kept.length === 0) return text
  return text ? `${text}\n\n${kept.join('\n')}` : kept.join('\n')
}

const LIST_MARK = /^\s*([-*•]|\d+[.)])\s+(?=\S)/

/** A pasted list ("- a", "1. b" on every line) as its items, markers
 *  stripped; null when it isn't one, or has fewer than 2 or over 50 items. */
export function splitListItems(text: string): string[] | null {
  const lines = (text || '').split('\n').filter((l) => l.trim())
  if (lines.length < 2 || lines.length > 50) return null
  if (!lines.every((l) => LIST_MARK.test(l))) return null
  return lines.map((l) => l.replace(LIST_MARK, '').trim())
}

// ── Commands ──

export function isFreshStart(content: string): boolean {
  return /^\/new\b/i.test((content || '').trim())
}

function commandName(content: string): string {
  return (content || '').trim().split(/\s+/)[0] || '/'
}

// ── Moving a task ──
//
// A task moves among the other waiting TASKS. The steps between them (the
// planner's fresh starts) stay where they are, so every task keeps the step
// in front of it — "a fresh topic for each task" holds however you reorder
// them. The tasks change places around the steps, as the list shows them.

/** The waiting items' new run order when task `id` moves before task
 *  `beforeId` (null: last); null when that moves nothing. */
export function moveTaskOrder(waiting: QueuedMessage[], id: string, beforeId: string | null): QueuedMessage[] | null {
  const tasks = waiting.filter((q) => !isQueueCommand(q.content))
  const item = tasks.find((q) => q.id === id)
  if (!item || id === beforeId) return null
  const order = tasks.filter((q) => q.id !== id)
  const at = beforeId === null ? order.length : order.findIndex((q) => q.id === beforeId)
  if (at < 0) return null
  order.splice(at, 0, item)
  let n = 0
  return waiting.map((q) => (isQueueCommand(q.content) ? q : order[n++]))
}

/** Brings the waiting items into `target`'s order with the store's own
 *  single moves (item before item), one slot at a time. */
export function applyWaitingOrder(target: QueuedMessage[], read: () => QueuedMessage[], move: (id: string, beforeId: string) => void) {
  for (let idx = 0; idx < target.length; idx++) {
    const cur = read()
    if (!cur[idx]) return
    if (cur[idx].id !== target[idx].id) move(target[idx].id, cur[idx].id)
  }
}

// ── Signal (the toggle's badge, the rail) ──

export type SignalTone = 'needs' | 'running' | 'idle' | 'none'

interface SignalSource {
  queue: QueuedMessage[]
  queueDispatchedId: string | null
  queueAutoMode: boolean
  queuePaused: boolean
  queueParked: boolean
  busy: boolean
  connected: boolean
}

export interface QueueSignal {
  left: number
  tone: SignalTone
  /** The running item has gone quiet (connected, the agent not replying):
   *  a key for this stretch of quiet. Once it has lasted `settleMs`, the card
   *  asks you ("Check it", "No reply") — the hook turns that into 'needs'. */
  quiet: string | null
  /** Its reply, when one came in (the quiet counts from there). */
  quietSince: number
  settleMs: number
}

/** The key for one stretch of quiet of the running item: a new reply starts
 *  a new one. Shared by the panel and the badge, so they agree. */
export function quietKeyOf(d: QueuedMessage): string {
  return `${d.id}:${d.repliedAt ?? 0}`
}

/** The running item has gone quiet: connected, the agent not replying, and
 *  nothing else to say about it (a hold, a give-up, a question has its own
 *  card). Its silence is what the card reads once it settles. */
export function isQuiet(d: QueuedMessage, busy: boolean, connected: boolean): boolean {
  return connected && !busy && !d.halted && !d.stuck && !d.awaitingAnswer
}

/** A task the allowance refused on an Auto lane that isn't paused goes
 *  again by itself once the allowance is back: it doesn't wait for you. */
export function retriesByItself(d: QueuedMessage | null | undefined, auto: boolean, paused: boolean): boolean {
  return d?.halted?.kind === 'quota' && auto && !paused
}

/** Tasks left (waiting + running) and whether the queue needs the user —
 *  as the hero card says it (deriveQueueModel), in the same order.
 *  `quotaSpent`: today's allowance is used up (the quota store's word). */
export function queueSignalOf(s: SignalSource, quotaSpent = false): QueueSignal {
  const d = s.queueDispatchedId ? s.queue.find((q) => q.id === s.queueDispatchedId) : undefined
  const dCmd = !!d && isQueueCommand(d.content)
  const left = s.queue.filter((q) => q.status === 'pending' && !isQueueCommand(q.content)).length + (d && !dCmd ? 1 : 0)
  const needs = d
    // A hold (unless it retries by itself while the allowance is still
    // spent: the card has nothing for you then), a give-up; then, while
    // connected, a question, or Step by step's reply to check.
    ? !!((d.halted && !(retriesByItself(d, s.queueAutoMode, s.queuePaused) && quotaSpent)) || d.stuck
      || (s.connected && (d.awaitingAnswer || (!s.queueAutoMode && !s.busy && d.repliedAt && !dCmd))))
    // Parked: Auto went off with the agent's power, and only you can put it
    // back (the limit card comes first while the allowance is spent).
    : s.queueParked && !s.queueAutoMode && !s.queuePaused && s.connected && !quotaSpent && s.queue.some((q) => q.status === 'pending')
  const tone: SignalTone = needs ? 'needs' : d && s.busy ? 'running' : left ? 'idle' : 'none'
  const quiet = d && isQuiet(d, s.busy, s.connected) ? quietKeyOf(d) : null
  return { left, tone, quiet, quietSince: d?.repliedAt ?? 0, settleMs: queueSettleMs(d ?? null, s.queueAutoMode) }
}

/** Done of all tasks, for the collapsed rail's progress track. */
export function queueProgressOf(s: SignalSource): { done: number; total: number; working: boolean } {
  let done = 0
  let total = 0
  for (const q of s.queue) {
    if (isQueueCommand(q.content)) continue
    total++
    if (q.status === 'done') done++
  }
  return { done, total, working: !!s.queueDispatchedId && s.busy }
}

// ── The model ──

export interface QueueModelInput {
  queue: QueuedMessage[]
  auto: boolean
  dispatchedId: string | null
  paused: boolean
  parked: boolean
  /** The user asked for a start the agent couldn't take yet. */
  wanted: boolean
  /** The running step was started by Run now / Try again: the task behind it
   *  follows even on a paused lane. */
  forceThrough?: boolean
  busy: boolean
  /** You pressed Stop and the agent hasn't ended that turn yet (busy stays on). */
  stopping: boolean
  connected: boolean
  unreachable: boolean
  /** A shared chat that ended: why, and whether Reconnect can help. */
  closed: { message: string; canRetry: boolean } | null
  agent: string
  shared: boolean
  quotaSpent: boolean
  resetsAt?: string | null
  hasConn: boolean
  now: number
  /** When the running item went quiet (its reply, or the panel noticing). */
  idleSince: number | null
  /** The running item has been quiet for its settle time (queueSettleMs):
   *  the panel's timer says so, so no clock has to tick to find out. */
  settled: boolean
}

export type HeroId =
  | 'empty' | 'closed' | 'limit-halted' | 'error' | 'stopped' | 'gave-up' | 'offline'
  | 'needs-answer' | 'working' | 'wrapping-up' | 'your-turn' | 'check-reply' | 'no-reply'
  | 'limit' | 'paused' | 'parked' | 'waiting-agent' | 'ready' | 'all-done'

export type HeroTone = 'violet' | 'emerald' | 'amber' | 'red' | 'zinc'

export type HeroIcon =
  | 'list' | 'alert' | 'gauge' | 'x' | 'square' | 'wifi-off' | 'loader' | 'help'
  | 'reply' | 'unplug' | 'pause' | 'power' | 'clock' | 'play' | 'check'

export type ActionIcon = 'plus' | 'wand' | 'refresh' | 'rotate' | 'skip' | 'pen' | 'message' | 'check' | 'play'

export type HeroActionId =
  | 'focus-composer' | 'open-planner' | 'reconnect' | 'retry' | 'skip' | 'edit-retry' | 'mark-done'
  | 'answer' | 'move-on' | 'keep-waiting' | 'show-in-chat' | 'pause-after' | 'keep-going'
  | 'show-reply' | 'resume' | 'auto-on' | 'run-next' | 'clear-finished'

export interface HeroAction {
  id: HeroActionId
  label: string
  icon?: ActionIcon
  disabled?: boolean
  title?: string
  /** edit-retry: the editor's Save button. */
  saveLabel?: string
}

export interface HeroState {
  id: HeroId
  tone: HeroTone
  icon: HeroIcon
  spin?: boolean
  /** The status pill's label; none on the empty card (an icon tile instead). */
  pill: string | null
  title: string
  body: string
  /** The agent's error message, on a clamped line of its own above the body. */
  quote?: string
  primary?: HeroAction
  secondary: HeroAction[]
  figure?: { kind: 'elapsed' | 'countdown'; text: string }
  /** "Try 2 of 4" while a re-run is in flight. */
  tryPill?: string
  /** Rows about the running item: its text, stats and ⋯ menu show. */
  running: boolean
  progress: 'bar' | 'countdown' | 'none'
  /** Countdown: the share of the 3-minute wait that is left. */
  remaining?: number
}

export interface QueueTotals { time: number; tokens: number; tools: number }

export interface QueueModel {
  tasks: QueuedMessage[]
  N: number
  D: QueuedMessage | null
  Dcmd: boolean
  /** The running task's number (0 for a command). */
  i: number
  pending: QueuedMessage[]
  kAll: number
  k: number
  /** The first waiting task, and the first waiting item of any kind. */
  next: QueuedMessage | null
  nextAny: QueuedMessage | null
  nextNum: number
  /** Done tasks, newest first. */
  done: QueuedMessage[]
  d: number
  s: number
  totals: QueueTotals
  numberOf: Map<string, number>
  left: number
  /** Adding a task now would start it at once. */
  willStart: boolean
  settle: number
  idleFor: number
  first: string
  hero: HeroState
  /** Under the first waiting task: when it starts. */
  caption: string | null
  /** The running item holds the queue until you act on it: an error, a
   *  stop, a give-up, a refusal that won't retry by itself — or it went
   *  quiet and the card asks you about it (Check it, No reply). */
  held: boolean
  /** The running task was refused by the allowance and goes again by itself
   *  once it is back (Auto, not paused). */
  selfRetry: boolean
  /** Nothing has run on this lane yet (none done, none running): a paused
   *  lane's button says Start, not Resume — and so does all the copy. */
  unstarted: boolean
}

/** The status-row copy for the summary/headers: "3m 07s · 12k tokens". */
export function totalsText(t: QueueTotals): string {
  return [fmtDuration(t.time), t.tokens ? `${fmtTokens(t.tokens)} tokens` : ''].filter(Boolean).join(' · ')
}

/** Tokens in/cached/out of one item, for tooltips. */
export function usageTitle(q: QueuedMessage): string | undefined {
  const u = q.usage
  if (!u) return undefined
  const cached = (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0)
  return `In ${u.prompt_tokens || 0} · cached ${cached} · out ${u.completion_tokens || 0} tokens`
}

export function taskTokens(q: QueuedMessage): number {
  return (q.usage?.prompt_tokens || 0) + (q.usage?.completion_tokens || 0)
}

/** When a finished task's work ended: its last reply, not the moment it was
 *  marked done — in Step by step that also counts how long you took to check. */
export function workEndedAt(q: QueuedMessage): number | undefined {
  return q.repliedAt && q.completedAt && q.repliedAt <= q.completedAt ? q.repliedAt : q.completedAt
}

function resetAt(resetsAt?: string | null): string {
  if (!resetsAt) return ''
  const t = new Date(resetsAt)
  if (Number.isNaN(t.getTime())) return ''
  return ` at ${t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

/** A message as a sentence: it ends with a stop. */
function sentence(text: string): string {
  const t = text.trim()
  return !t || /[.!?…]$/.test(t) ? t : `${t}.`
}

function mmss(ms: number): string {
  const secs = Math.max(0, Math.ceil(ms / 1000))
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`
}

/** How long a quiet running item gets before the queue reads its silence:
 *  a command 25s, Auto 10s (its own check may still mark it done), step by
 *  step 1.5s. */
export function queueSettleMs(D: QueuedMessage | null, auto: boolean): number {
  return D && isQueueCommand(D.content) ? 25000 : auto ? 10000 : 1500
}

export function deriveQueueModel(inp: QueueModelInput): QueueModel {
  const { queue, auto, paused, parked, wanted, busy, stopping, connected, agent, quotaSpent, now } = inp
  const tasks = queue.filter((q) => !isQueueCommand(q.content))
  const N = tasks.length
  const numberOf = new Map<string, number>()
  tasks.forEach((q, idx) => numberOf.set(q.id, idx + 1))

  const D = (inp.dispatchedId && queue.find((q) => q.id === inp.dispatchedId)) || null
  const Dcmd = !!D && isQueueCommand(D.content)
  const i = D && !Dcmd ? numberOf.get(D.id) ?? 0 : 0
  const pending = queue.filter((q) => q.status === 'pending')
  const kAll = pending.length
  const pendingTasks = pending.filter((q) => !isQueueCommand(q.content))
  const k = pendingTasks.length
  const next = pendingTasks[0] ?? null
  const nextAny = pending[0] ?? null
  const nextNum = next ? numberOf.get(next.id) ?? 0 : 0
  const done = tasks.filter((q) => q.status === 'done')
    .sort((a, b) => (b.completedAt ?? 0) - (a.completedAt ?? 0))
  const d = done.length
  const s = done.filter((q) => q.endedHow === 'skipped').length
  const ins = done.filter((q) => q.endedHow === 'inserted').length
  const totals: QueueTotals = { time: 0, tokens: 0, tools: 0 }
  for (const q of done) {
    const end = workEndedAt(q)
    if (!q.dispatchedAt || !end || q.endedHow === 'skipped') continue
    totals.time += Math.max(0, end - q.dispatchedAt)
    totals.tokens += taskTokens(q)
    totals.tools += q.toolCount ?? 0
  }
  const left = k + (D && !Dcmd ? 1 : 0)
  const closed = !!inp.closed
  const willStart = !D && kAll === 0 && connected && !busy && !paused && !quotaSpent && !closed
  const settle = queueSettleMs(D, auto)
  const idleFor = inp.idleSince ? Math.max(0, now - inp.idleSince) : 0
  // With nothing running, the next waiting task starts without you: Auto on
  // a lane that isn't paused, or a start you already asked for (that one task
  // only — Step by step, or a paused lane, then waits for you again).
  const selfStart = (auto && !paused) || wanted
  // The running item holds the queue until you act on it — except a task the
  // allowance refused on an Auto lane that isn't paused: that one goes again
  // by itself once the allowance is back.
  const selfRetry = retriesByItself(D, auto, paused)
  // …and so does one gone quiet past its settle time, when nothing else will
  // finish it: on Auto its own check is over (Check it, No reply), and a step
  // not finished by then won't finish by itself (its own fallback is
  // shorter). In Step by step a quiet task waits for you anyway, as the
  // caption already says.
  const quietHeld = !!D && (auto || Dcmd) && inp.settled && isQuiet(D, busy, connected)
  const held = !!D && (((!!D.halted || !!D.stuck) && !selfRetry) || quietHeld)
  // Start or Resume: a task running (or one done) means the lane has started.
  const unstarted = d === 0 && !D

  // Copy tokens.
  const T = !D ? '' : !Dcmd ? `Task ${i}` : isFreshStart(D.content) ? 'The fresh-start step' : `The ${commandName(D.content)} command`
  const t = !D ? '' : !Dcmd ? `task ${i}` : T.charAt(0).toLowerCase() + T.slice(1)
  const waiting = k === 0 ? 'The next step is' : k === 1 ? '1 task is' : `${k} tasks are`
  // …and what follows it agrees with it.
  const they = k > 1 ? 'They' : 'It'
  const them = k > 1 ? 'them' : 'it'
  const reset = resetAt(inp.resetsAt)
  const first = !nextAny ? ''
    : isQueueCommand(nextAny.content) ? (isFreshStart(nextAny.content) ? 'Fresh start' : commandName(nextAny.content))
    : (() => {
      const p = parseAttachmentRefs(nextAny.content)
      return firstLine(p.text) || firstLine(p.files.map((f) => f.name).join(', '))
    })()

  const markDone: HeroAction = { id: 'mark-done', label: 'Mark done', icon: 'check' }
  const skipIt: HeroAction = { id: 'skip', label: 'Skip it' }
  const runAgain: HeroAction = {
    id: 'retry', label: 'Run it again', icon: 'rotate',
    // A stop still landing: the run waits for it, then goes by itself.
    title: stopping ? 'It runs again once it has stopped' : undefined,
  }
  const resume: HeroAction = { id: 'resume', label: unstarted ? 'Start' : 'Resume', icon: 'play' }
  const tryPill = D && (D.attempts ?? 0) >= 1 && !D.stuck ? `Try ${(D.attempts ?? 0) + 1} of ${QUEUE_MAX_TRIES}` : undefined

  const row = (h: Omit<HeroState, 'secondary' | 'running' | 'progress'> & Partial<Pick<HeroState, 'secondary' | 'running' | 'progress'>>): HeroState => ({
    secondary: [], running: !!D, progress: N > 0 ? 'bar' : 'none', ...h,
  })

  // A start you asked for, off Auto (or on a paused lane), runs the next task
  // only: what the rest do after it.
  const onlyNext = wanted && !(auto && !paused) && k > 1
    ? (paused ? ' The queue stays paused after it.' : ' You mark each one done before the next starts.')
    : ''

  // Nothing running, tasks waiting, and something in the way (the agent is
  // away, the allowance is spent): what they do once `when` comes, and what
  // you can do about it now.
  const whenFree = (when: string, startLabel: string): Pick<HeroState, 'body' | 'primary' | 'secondary'> => {
    const lead = `${waiting} waiting. `
    // Step by step (or paused) runs only the next one by itself.
    const one = k > 1 ? `task ${nextNum} starts` : 'it starts'
    if (auto && !paused) {
      return { body: lead + `${they} start${k > 1 ? ' by themselves' : 's by itself'} ${when}.`, secondary: [] }
    }
    if (wanted) {
      return { body: lead + `${one.charAt(0).toUpperCase()}${one.slice(1)} by itself ${when}.${onlyNext}`, secondary: [] }
    }
    if (paused) {
      const runs = auto && k > 1 ? 'they run' : k > 1 ? `task ${nextNum} runs` : 'it runs'
      return {
        body: lead + `The queue is paused. Press ${resume.label}, and ${runs} ${when}.`,
        primary: resume,
        secondary: [],
      }
    }
    return { body: lead + `You can start ${them} ${when}.`, secondary: [{ id: 'run-next', label: startLabel }] }
  }

  const hero: HeroState = (() => {
    // 0 · A shared chat that ended — first, so an empty queue never offers
    // to add a task the closed composer can't take.
    if (inp.closed) {
      const none = queue.length === 0
      return row({
        id: 'closed', tone: 'amber', icon: 'alert', pill: 'Chat closed', running: false,
        title: 'This chat has ended',
        body: `${sentence(inp.closed.message)} ` + (inp.closed.canRetry
          ? (none ? 'You can add tasks once the chat is back.' : 'The tasks here wait until the chat is back.')
          : (none ? "Tasks can't be added to it any more." : "The tasks here can't run in it any more.")),
        // The chat's own banner calls it Retry too.
        primary: inp.closed.canRetry ? { id: 'reconnect', label: 'Retry', icon: 'refresh' } : undefined,
      })
    }
    // 1 · Nothing queued.
    if (queue.length === 0) {
      return row({
        id: 'empty', tone: 'zinc', icon: 'list', pill: null, running: false, progress: 'none',
        title: `Line up work for ${agent}`,
        // Only Auto carries on by itself: Step by step waits for you after each task.
        body: `Add tasks below. ${agent} works through them one at a time, in order` + (auto
          ? '. You can leave it running.'
          : ', and waits for you after each one. Switch to Auto to leave it running.'),
        primary: { id: 'focus-composer', label: 'Add your first task', icon: 'plus' },
        secondary: inp.shared ? [] : [{
          id: 'open-planner', label: 'Plan tasks from one description', icon: 'wand',
          disabled: !inp.hasConn, title: inp.hasConn ? undefined : 'Waiting for the agent connection…',
        }],
      })
    }
    if (D) {
      // 2 · Refused: today's allowance is spent.
      if (D.halted?.kind === 'quota') {
        // Back already (another tab's reset, a new day): nothing to wait for.
        if (!quotaSpent) {
          return row({
            id: 'limit-halted', tone: 'amber', icon: 'gauge', pill: 'Limit lifted',
            title: 'Your allowance is back',
            body: `${T} couldn't start while it was used up. You can run it now.`,
            primary: { id: 'retry', label: 'Try again', icon: 'rotate' },
            secondary: [skipIt],
          })
        }
        return row({
          id: 'limit-halted', tone: 'amber', icon: 'gauge', pill: 'Limit reached',
          title: "Today's allowance is used up",
          body: `${T} couldn't start. ` + (selfRetry
            ? `It runs again by itself when your allowance resets${reset}.`
            : `You can have it start by itself when your allowance resets${reset}.`),
          // Step by step (or paused): arming the retry now means it starts
          // the moment the allowance is back, with nobody watching for it.
          primary: selfRetry
            ? { id: 'retry', label: 'Try again', icon: 'rotate', disabled: true, title: 'Available when your allowance resets' }
            : { id: 'retry', label: 'Retry when it resets', icon: 'rotate', title: 'It starts by itself as soon as your allowance resets' },
          secondary: [skipIt],
        })
      }
      // 3 · The turn errored.
      if (D.halted?.kind === 'error') {
        return row({
          id: 'error', tone: 'red', icon: 'x', pill: 'Error',
          title: `${T} hit an error`,
          quote: D.halted.message ? `“${D.halted.message}”` : undefined,
          body: `${D.halted.message ? '' : 'The agent reported an error. '}The queue is holding here.`,
          primary: { id: 'retry', label: 'Try again', icon: 'rotate' },
          secondary: [
            ...(!Dcmd ? [{ id: 'edit-retry' as const, label: 'Edit task', saveLabel: 'Save & try again' }] : []),
            skipIt,
          ],
        })
      }
      // 4 · The user stopped it. Until the agent ends that turn (it notices
      // Stop at its next step) the card says so; what you pick meanwhile
      // waits for the stop to land, then goes by itself.
      if (D.halted?.kind === 'stopped') {
        const landing = stopping
        const lead = landing ? `${agent} stops at its next step. ` : ''
        return row({
          id: 'stopped', tone: 'zinc', icon: landing ? 'loader' : 'square', spin: landing,
          pill: landing ? 'Stopping' : 'Stopped',
          title: landing ? `Stopping ${t}…` : `You stopped ${t}`,
          body: lead + (!k
            ? 'Run it again, or mark it done.'
            // Paused: a skip only tidies it away — nothing after it starts.
            : paused ? 'The queue is paused. Skip it, or run it again.'
            : landing ? `Skip it to carry on with task ${nextNum} once it has, or run it again.`
            : `The queue is holding. Skip it to carry on with task ${nextNum}, or run it again.`),
          primary: k ? { id: 'skip', label: paused ? 'Skip' : 'Skip & continue', icon: 'skip' } : runAgain,
          secondary: k ? [runAgain, markDone] : [markDone],
        })
      }
      // 5 · The agent gave up on every try.
      if (D.stuck) {
        return row({
          id: 'gave-up', tone: 'red', icon: 'alert', pill: 'Gave up',
          title: `${agent} gave up on ${t}`,
          body: `It tried ${D.attempts ?? QUEUE_MAX_TRIES} times. Nothing after it runs until you decide.`,
          primary: !Dcmd
            ? { id: 'edit-retry', label: 'Edit and try again', icon: 'pen', saveLabel: 'Save & try again' }
            : { id: 'retry', label: 'Try again', icon: 'rotate' },
          secondary: [...(!Dcmd ? [{ id: 'retry' as const, label: 'Try again as is' }] : []), skipIt],
        })
      }
    }
    // 6 · The agent can't be reached.
    if (!connected) {
      const base = {
        id: 'offline', tone: 'zinc', icon: inp.unreachable ? 'wifi-off' : 'loader', spin: !inp.unreachable,
        pill: inp.unreachable ? 'Offline' : 'Connecting',
        title: inp.unreachable ? `Can't reach ${agent}` : `Connecting to ${agent}…`,
      } as const
      if (D) {
        return row({ ...base, body: `${T} was running when the connection dropped. You can check it when ${agent} is back.` })
      }
      if (!kAll) return row({ ...base, body: `You can still add tasks. They wait until ${agent} is back.` })
      return row({ ...base, ...whenFree(`when ${agent} is back`, "Start when it's back") })
    }
    if (D) {
      // 7 · Waiting (bounded) on the user's answer.
      if (D.awaitingAnswer) {
        const leftMs = D.awaitingSince ? D.awaitingSince + QUEUE_QUESTION_HOLD_MS - now : null
        const when = leftMs === null ? 'in about 3 minutes' : leftMs <= 0 ? 'now' : `in ${mmss(leftMs)}`
        // Unanswered, the task is set aside; the next one starts only on a
        // lane that isn't paused.
        const goesOn = k > 0 && !paused
        return row({
          id: 'needs-answer', tone: 'amber', icon: 'help', pill: 'Needs your answer',
          figure: leftMs !== null ? { kind: 'countdown', text: mmss(leftMs) } : undefined,
          title: `${agent} asked you something`,
          body: `Answer in the chat and it carries on with ${t}. Otherwise the queue `
            + (goesOn ? `moves on ${when}.` : `stops waiting ${when}${paused ? ' and stays paused' : ''}.`),
          primary: { id: 'answer', label: 'Answer in chat', icon: 'message' },
          secondary: [
            { id: 'move-on', label: goesOn ? 'Move on now' : 'Stop waiting' },
            { id: 'keep-waiting', label: 'Keep waiting', title: 'Restart the 3-minute wait' },
          ],
          progress: 'countdown',
          remaining: leftMs === null ? 1 : Math.max(0, Math.min(1, leftMs / QUEUE_QUESTION_HOLD_MS)),
        })
      }
      const runningTitle = Dcmd
        ? (isFreshStart(D.content) ? 'Starting a fresh topic' : `Running ${commandName(D.content)}`)
        : `Working on task ${i} of ${N}`
      // 8 · The agent is on it.
      if (busy) {
        return row({
          id: 'working', tone: 'violet', icon: 'loader', spin: true, pill: 'Working', tryPill,
          figure: D.dispatchedAt ? { kind: 'elapsed', text: fmtElapsed(D.dispatchedAt, Math.max(now, D.dispatchedAt)) } : undefined,
          title: runningTitle,
          body: paused
            // A step you started on a paused lane (Run now, Try again) takes
            // the task behind it along; one you paused during doesn't.
            ? (Dcmd
              ? (inp.forceThrough
                ? (k ? 'This takes a moment. The next task follows by itself.' : 'This takes a moment.')
                : (k ? 'This takes a moment. The queue is paused, so the next task waits.' : 'This takes a moment. The queue is paused.'))
              : 'The queue pauses after this one.')
            // A step finishes by itself, in either mode: nothing to check.
            : Dcmd ? (k ? 'This takes a moment. The next task follows by itself.' : 'This takes a moment.')
            : auto
              ? (!k ? 'This is the last one.'
                : quotaSpent ? `${k} more after this. The next one starts by itself when your allowance resets${reset}.`
                : `${k} more after this. The next one starts by itself.`)
              : 'When it replies, check the result and mark it done.',
          secondary: [
            ...(!Dcmd && D.messageId ? [{ id: 'show-in-chat' as const, label: 'Show in chat' }] : []),
            ...(auto && k && !paused ? [{ id: 'pause-after' as const, label: 'Pause after this' }] : []),
            ...(paused ? [{ id: 'keep-going' as const, label: 'Keep going' }] : []),
          ],
        })
      }
      const replyToCheck = !auto && !!D.repliedAt && !Dcmd
      // 9 · Just went quiet: give the turn a moment to settle.
      if (!inp.settled && !replyToCheck) {
        return row({
          id: 'wrapping-up', tone: 'violet', icon: 'loader', spin: true, pill: 'Wrapping up',
          title: Dcmd ? runningTitle : `Wrapping up task ${i}`,
          // Step by step never moves on by itself: nothing is being checked.
          body: Dcmd ? 'Waiting for it to finish.' : auto ? 'Checking the reply before moving on.' : 'Waiting for the reply to come in.',
        })
      }
      // Marking it done starts the next task — unless the lane is paused, or
      // the allowance is spent (then the next one waits for the reset).
      const startsNext = k > 0 && !paused && !quotaSpent
      const doneNext: HeroAction = startsNext
        ? { id: 'mark-done', label: 'Mark done & start next', icon: 'check' }
        : markDone
      // 10 · Step by step: the reply is the user's to judge.
      if (replyToCheck) {
        return row({
          id: 'your-turn', tone: 'emerald', icon: 'reply', pill: 'Your turn',
          title: `${agent} replied to task ${i}`,
          body: 'Check the reply in the chat (answer there if it asked you something), then mark it done'
            + (paused ? '. The queue stays paused.'
              : !k ? ' to finish the queue.'
              : quotaSpent ? `. Task ${nextNum} starts when your allowance resets${reset}.`
              : ` to start task ${nextNum}.`),
          primary: doneNext,
          secondary: [{ id: 'show-reply', label: 'Show reply' }, runAgain],
        })
      }
      // 11 · Auto couldn't tell whether the reply finished it.
      if (auto && D.repliedAt) {
        return row({
          id: 'check-reply', tone: 'amber', icon: 'reply', pill: 'Check it',
          title: `${T} may be finished`,
          body: "A reply came in, but the queue couldn't tell whether the task is done.",
          primary: doneNext,
          secondary: [{ id: 'show-reply', label: 'Show reply' }, runAgain],
        })
      }
      // 12 · Quiet, and no reply at all.
      return row({
        id: 'no-reply', tone: 'amber', icon: 'unplug', pill: 'No reply',
        title: Dcmd ? `${T} didn't answer` : `Task ${i} didn't get a reply`,
        body: 'The agent may have restarted, or the connection dropped while it worked.',
        primary: runAgain,
        secondary: [markDone, skipIt],
      })
    }
    if (kAll) {
      // 13 · Nothing can start before the allowance resets.
      if (quotaSpent) {
        return row({
          id: 'limit', tone: 'amber', icon: 'gauge', pill: 'Limit reached',
          title: "Today's allowance is used up",
          ...whenFree(`when it resets${reset}`, 'Start when it resets'),
        })
      }
      // 14 · Held by the user. (A start you asked for meanwhile still goes:
      // the cards below say when.)
      if (paused && !wanted) {
        return row({
          id: 'paused', tone: 'zinc', icon: 'pause', pill: 'Paused',
          // The pill already says Paused.
          title: k === 1 ? '1 task waiting' : k ? `${k} tasks waiting` : 'Next step waiting',
          body: `Next up: “${first}”`,
          primary: resume,
        })
      }
      // 15 · Auto went off with the agent's power.
      if (parked && !auto) {
        return row({
          id: 'parked', tone: 'zinc', icon: 'power', pill: 'Parked',
          title: `Paused while ${agent} was off`,
          body: 'Switch Auto back on to carry on, or run just the next task.',
          primary: { id: 'auto-on', label: 'Switch Auto back on', icon: 'play' },
          secondary: [{ id: 'run-next', label: 'Run next task only' }],
        })
      }
      // 16 · The agent is busy with a chat message.
      if (busy) {
        return row({
          id: 'waiting-agent', tone: 'zinc', icon: 'clock', pill: stopping ? 'Stopping' : 'Waiting',
          title: `Waiting for ${agent}`,
          // Neutral on purpose: the reply it is finishing may be a typed
          // message's, or a task's that was just skipped or marked done.
          body: (stopping ? "It's stopping its last reply. " : "It's still finishing its last reply. ") + (selfStart
            ? `The next task starts as soon as it's free.${onlyNext}`
            : "You can run the next task when it's free."),
          secondary: selfStart ? [] : [{ id: 'run-next', label: "Start when it's free" }],
        })
      }
      // 17 · Ready to go.
      return row({
        id: 'ready', tone: 'violet', icon: 'play', pill: 'Ready',
        title: k === 1 ? '1 task ready' : k ? `${k} tasks ready` : 'Ready',
        body: `Next up: “${first}”`,
        primary: { id: 'run-next', label: auto ? 'Start' : 'Run next task', icon: 'play' },
        secondary: !auto && k > 1 ? [{ id: 'auto-on', label: 'Run all automatically' }] : [],
      })
    }
    // 18 · Everything is finished.
    const parts = [
      totals.time ? `Took ${fmtDuration(totals.time)} in total` : '',
      totals.tokens ? `${fmtTokens(totals.tokens)} tokens` : '',
      s ? `${s} skipped` : '',
      ins ? `${ins} inserted into a running task` : '',
    ].filter(Boolean)
    return row({
      id: 'all-done', tone: 'emerald', icon: 'check', pill: 'All done',
      title: N === 1 ? 'Task done' : N ? `All ${N} tasks done` : 'All done',
      body: parts.length ? `${parts.join(' · ')}.` : 'Everything in the queue is finished.',
      primary: { id: 'focus-composer', label: 'Add another task', icon: 'plus' },
      secondary: [{ id: 'clear-finished', label: 'Clear finished' }],
    })
  })()

  // When the first waiting task starts, said under it — never more than the
  // queue will do by itself.
  const caption: string | null = !next ? null : (() => {
    // Held on the running item: nothing after it moves until you act.
    if (held) return `Waits until you deal with ${t}`
    if (paused && !wanted) {
      // A step running on a paused lane: the task behind it follows only
      // when the step was started by Run now / Try again.
      if (Dcmd) return inp.forceThrough ? (isFreshStart(D!.content) ? 'Starts after the fresh start' : `Starts after ${t}`) : 'Waits until you resume'
      // (As the paused card's button says: Start, or Resume once one ran.)
      return unstarted ? 'Waits until you start the queue' : 'Waits until you resume'
    }
    if (D) {
      // A step finishes by itself, in either mode.
      if (Dcmd) return isFreshStart(D.content) ? 'Starts after the fresh start' : `Starts after ${t}`
      // Spent: what comes next waits for the reset, whatever ends this one.
      if (!auto) return quotaSpent ? `Starts once you mark task ${i} done and the allowance resets` : `Starts after you mark task ${i} done`
      if (quotaSpent) return `Starts after task ${i}, once the allowance resets`
      if (!connected) return `Starts after task ${i}, once ${agent} is back`
      return `Starts when task ${i} finishes`
    }
    // Nothing running: it starts by itself only when the lane says so.
    if (!selfStart) return !connected || quotaSpent || busy ? 'Waits until you start it' : null
    if (!connected) return `Starts when ${agent} is back`
    if (quotaSpent) return 'Starts when the allowance resets'
    if (busy) return stopping ? `Starts once ${agent} has stopped` : `Starts when ${agent} finishes its reply`
    return null
  })()

  return {
    tasks, N, D, Dcmd, i, pending, kAll, k, next, nextAny, nextNum, done, d, s, totals, numberOf,
    left, willStart, settle, idleFor, first, hero, caption, held, selfRetry, unstarted,
  }
}
