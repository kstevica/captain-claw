import { useState, type ReactNode } from 'react'
import { AlertTriangle, Check, CircleDot, FileDown, History, ListTodo, Loader2, ShieldAlert, Sparkles } from 'lucide-react'
import type { ExportFormat, TaskAction, TaskFacet, TaskItem, TaskboardStatus } from '../../services/taskboardApi'
import { resolveProposal } from '../../services/notifications'
import {
  JUDGE_LABEL, OWES_LABEL, RESULT_LABEL, STATUS_LABEL, agoRelative, channelInfo, foldChar, type DatePreset,
  type DateRange, type TaskNote,
} from './taskboardText'

// Pieces of the Taskboard: search highlighting (the server's folding, so
// what it finds is what lights up), a multi-select filter list, the date
// filter, the chips a task wears, the sync line, quick answers, the
// download menu and the board's switch. Their words and plain helpers are
// in taskboardText.

/** Search terms, folded on both sides: "duric" lights up "Đurić". `terms`
 *  come folded (foldText). */
export function FoldHighlight({ text, terms }: { text: string; terms: string[] }) {
  if (!terms.length || !text) return <>{text}</>
  // The folded text, and where each of its characters came from.
  let folded = ''
  const from: number[] = []
  let i = 0
  for (const ch of text) {
    const f = foldChar(ch)
    for (let k = 0; k < f.length; k++) from.push(i)
    folded += f
    i += ch.length
  }
  const spans: [number, number][] = []
  for (const t of terms) {
    if (!t) continue
    for (let at = folded.indexOf(t); at >= 0; at = folded.indexOf(t, at + t.length)) {
      const last = from[at + t.length - 1]
      spans.push([from[at], last + (text.codePointAt(last)! > 0xffff ? 2 : 1)])
    }
  }
  if (!spans.length) return <>{text}</>
  spans.sort((a, b) => a[0] - b[0])
  const parts: ReactNode[] = []
  let pos = 0
  for (const [s, e] of spans) {
    if (e <= pos) continue
    const start = Math.max(s, pos)
    if (start > pos) parts.push(<span key={`t${pos}`}>{text.slice(pos, start)}</span>)
    parts.push(<mark key={`m${start}`} className="rounded-sm bg-amber-400/30 px-0.5 text-inherit">{text.slice(start, e)}</mark>)
    pos = e
  }
  if (pos < text.length) parts.push(<span key={`t${pos}`}>{text.slice(pos)}</span>)
  return <>{parts}</>
}

export function ChannelIcon({ channel, className = 'h-4 w-4' }: { channel: string; className?: string }) {
  const info = channelInfo(channel)
  const Icon = info?.Icon || CircleDot
  return <Icon className={`${className} shrink-0 ${info?.tone || 'text-zinc-500'}`} />
}

/** A task's own icon: an agent's to-do item is one; anything else wears
 *  the channel it came in on. */
export function TaskIcon({ task, className = 'h-4 w-4' }: { task: Pick<TaskItem, 'kind' | 'channel'>; className?: string }) {
  if (task.kind === 'todo') return <ListTodo className={`${className} shrink-0 text-teal-600 dark:text-teal-400`} />
  return <ChannelIcon channel={task.channel} className={className} />
}

// ── Chips ──

const chip = 'shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium'

const LANE: Record<string, { label: string; tone: string }> = {
  needs_me: { label: 'Needs you', tone: 'bg-amber-500/15 text-amber-700 dark:text-amber-300' },
  waiting: { label: 'Waiting on others', tone: 'bg-sky-500/10 text-sky-700 dark:text-sky-300' },
  working: { label: 'Agent working', tone: 'bg-violet-500/10 text-violet-700 dark:text-violet-300' },
  done: { label: 'Done', tone: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' },
  snoozed: { label: 'Snoozed', tone: 'bg-zinc-800 text-zinc-400' },
}
const STATUS_TONE: Record<string, string> = {
  open: 'bg-sky-500/10 text-sky-700 dark:text-sky-300',
  done: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  failed: 'bg-red-500/10 text-red-700 dark:text-red-300',
  maybe: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  dismissed: 'bg-zinc-800 text-zinc-400',
  not_task: 'bg-zinc-800 text-zinc-500',
}

/** Where a task stands: its lane, or its status when it is in none. */
export function LaneChip({ task }: { task: TaskItem }) {
  const lane = LANE[task.lane]
  if (lane) return <span className={`${chip} ${lane.tone}`}>{lane.label}</span>
  return <StatusChip status={task.status} />
}

export function StatusChip({ status }: { status: string }) {
  return <span className={`${chip} ${STATUS_TONE[status] || STATUS_TONE.dismissed}`}>{STATUS_LABEL[status] || status}</span>
}

/** ✓ a send or delivery was recorded; "agent says" — only its word; nothing. */
export function EvidenceBadge({ evidence }: { evidence: string }) {
  if (evidence === 'verified') {
    return (
      <span title="A send or a delivery was recorded" className={`${chip} flex items-center gap-0.5 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300`}>
        <Check className="h-3 w-3" />Recorded
      </span>
    )
  }
  if (evidence === 'claimed') {
    return <span title="Only the agent's word — nothing recorded a send" className={`${chip} bg-zinc-800 font-normal italic text-zinc-400`}>agent says</span>
  }
  return null
}

const RESULT_TONE: Record<string, string> = {
  success: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  partial: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  failed: 'bg-red-500/10 text-red-700 dark:text-red-300',
  unknown: 'bg-zinc-800 text-zinc-400',
}

export function ResultChip({ result, answeredInChat = false }: { result: string; answeredInChat?: boolean }) {
  if (answeredInChat) return <span className={`${chip} ${RESULT_TONE.success}`}>Answered in chat</span>
  if (!result) return null
  return <span className={`${chip} ${RESULT_TONE[result] || RESULT_TONE.unknown}`}>{RESULT_LABEL[result] || result}</span>
}

export function OwesChip({ owes }: { owes: string }) {
  if (!owes) return null
  const tone = owes === 'me' ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300' : 'bg-zinc-800 text-zinc-400'
  return <span className={`${chip} ${tone}`}>Next: {OWES_LABEL[owes] || owes}</span>
}

/** Someone else's message asks the agent to do something: the board never
 *  suggests a next step for it. `compact` is the row's icon alone. */
export function InstructionsChip({ compact = false }: { compact?: boolean }) {
  const title = 'Contains instructions — no next steps are suggested for it'
  if (compact) {
    return (
      <span title={title} aria-label="Contains instructions" className="shrink-0 text-amber-700 dark:text-amber-400">
        <ShieldAlert className="h-3.5 w-3.5" />
      </span>
    )
  }
  return (
    <span title={title} className={`${chip} flex items-center gap-0.5 bg-amber-500/10 text-amber-700 dark:text-amber-300`}>
      <ShieldAlert className="h-3 w-3" />Contains instructions
    </span>
  )
}

const NOTE_TONE: Record<TaskNote['id'], string> = {
  agent_declined: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  partly_done: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  new_since_done: 'bg-sky-500/10 text-sky-700 dark:text-sky-300',
}

/** What a task's flags add (taskNotes): the agent couldn't do it — why, on
 *  hover — partly done, a new message since it was done. */
export function NoteChip({ note }: { note: TaskNote }) {
  return <span title={note.detail || undefined} className={`${chip} ${NOTE_TONE[note.id]}`}>{note.label}</span>
}

/** Filled in from the last 30 days of chats when the board first opened:
 *  it never gets a next step or a notification. */
export function HistoryChip() {
  return (
    <span
      title="Filled in from your agents' chats of the last 30 days — no next step is suggested for it"
      className={`${chip} flex items-center gap-0.5 bg-zinc-800 text-zinc-400`}
    >
      <History className="h-3 w-3" />From history
    </span>
  )
}

/** Taken in as a task with less certainty: it isn't counted in the badge. */
export function LowConfidenceHint() {
  return (
    <span title="Jev wasn't sure this is a task, so it isn't counted on the Taskboard button" className={`${chip} font-normal italic text-zinc-500`}>
      Low confidence
    </span>
  )
}

const JUDGE_TONE: Record<string, string> = {
  jev: 'bg-teal-500/10 text-teal-700 dark:text-teal-300',
  model: 'bg-violet-500/10 text-violet-700 dark:text-violet-300',
  user: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  rule: 'bg-zinc-800 text-zinc-400',
}

/** Who decided: Jev, a rule, the model, or you. */
export function JudgeChip({ judge }: { judge: string }) {
  return <span className={`${chip} ${JUDGE_TONE[judge] || JUDGE_TONE.rule}`}>{JUDGE_LABEL[judge] || judge}</span>
}

// ── Filters ──

/** One filter, several values at once: a checkbox per value with its count.
 *  A ticked value that no longer has anything still shows (at 0), so it can
 *  be cleared. A long list shows its first few until asked for the rest. */
export function MultiFacet({ title, facets, values, onChange, dot, labelOf, limit = 8 }: {
  title: string; facets: TaskFacet[]; values: string[]
  onChange: (v: string[]) => void; dot?: (id: string) => ReactNode
  /** The label of a value the server names only by its id (or names in a
   *  way the board says otherwise: its own label comes second, the value
   *  itself third). */
  labelOf?: (id: string, label: string, facet: TaskFacet) => string
  limit?: number
}) {
  const [all, setAll] = useState(false)
  // A blank value (no result yet) is nothing a filter can ask for.
  const rows = [
    ...facets.filter((f) => f.id !== ''),
    ...values.filter((v) => !facets.some((f) => f.id === v)).map((id) => ({ id, label: '', count: 0 })),
  ]
  const shown = all || rows.length <= limit + 2 ? rows : rows.filter((r, i) => i < limit || values.includes(r.id))
  const toggle = (id: string) => onChange(values.includes(id) ? values.filter((v) => v !== id) : [...values, id])
  // The board's own words first, so a filter reads like the chips on the
  // rows; the server's label for a value the board has no name for.
  const label = (f: TaskFacet) => {
    const mine = labelOf?.(f.id, f.label, f)
    return mine && mine !== f.id ? mine : f.label || f.id
  }
  return (
    <div>
      <div className="flex items-center gap-2 px-2.5 pb-1">
        <span className="min-w-0 flex-1 truncate text-[11px] font-medium uppercase tracking-wider text-zinc-500">{title}</span>
        {values.length > 0 && (
          <button onClick={() => onChange([])} className="shrink-0 text-[11px] text-teal-600 hover:underline dark:text-teal-400">Clear</button>
        )}
      </div>
      <ul className="flex flex-col gap-0.5">
        {shown.map((f) => {
          const on = values.includes(f.id)
          return (
            <li key={f.id}>
              <button
                role="checkbox"
                aria-checked={on}
                onClick={() => toggle(f.id)}
                className={`flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors ${
                  on ? 'bg-zinc-800 font-medium text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'}`}
              >
                <span className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border ${
                  on ? 'border-teal-600 bg-teal-600 text-white' : 'border-zinc-600'}`}>
                  {on && <Check className="h-2.5 w-2.5" strokeWidth={3} />}
                </span>
                {dot?.(f.id)}
                <span className="min-w-0 flex-1 truncate">{label(f)}</span>
                <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{f.count}</span>
              </button>
            </li>
          )
        })}
      </ul>
      {shown.length < rows.length && (
        <button onClick={() => setAll(true)} className="mt-0.5 px-2.5 text-[11px] text-teal-600 hover:underline dark:text-teal-400">
          Show {rows.length - shown.length} more
        </button>
      )}
    </div>
  )
}

const DATE_PRESETS: { id: DatePreset; label: string }[] = [
  { id: '', label: 'Any time' },
  { id: 'today', label: 'Today' },
  { id: '7d', label: 'Last 7 days' },
  { id: '30d', label: 'Last 30 days' },
  { id: 'custom', label: 'Between…' },
]

export function DateFacet({ value, onChange }: { value: DateRange; onChange: (v: DateRange) => void }) {
  const input = 'min-w-0 flex-1 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 focus:border-teal-500/60 focus:outline-none'
  return (
    <div>
      <div className="px-2.5 pb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Last activity</div>
      <ul className="flex flex-col gap-0.5">
        {DATE_PRESETS.map((p) => {
          const on = value.preset === p.id
          return (
            <li key={p.id || '_any'}>
              <button
                role="radio"
                aria-checked={on}
                onClick={() => onChange({ ...value, preset: p.id })}
                className={`flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors ${
                  on ? 'bg-zinc-800 font-medium text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'}`}
              >
                <span className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border ${on ? 'border-teal-600' : 'border-zinc-600'}`}>
                  {on && <span className="h-1.5 w-1.5 rounded-full bg-teal-600" />}
                </span>
                {p.label}
              </button>
            </li>
          )
        })}
      </ul>
      {value.preset === 'custom' && (
        <div className="mt-1.5 flex flex-col gap-1.5 px-2.5">
          <label className="flex items-center gap-2 text-[11px] text-zinc-500">
            <span className="w-8 shrink-0">From</span>
            <input type="date" value={value.from} max={value.to || undefined} onChange={(e) => onChange({ ...value, from: e.target.value })} className={input} />
          </label>
          <label className="flex items-center gap-2 text-[11px] text-zinc-500">
            <span className="w-8 shrink-0">To</span>
            <input type="date" value={value.to} min={value.from || undefined} onChange={(e) => onChange({ ...value, to: e.target.value })} className={input} />
          </label>
        </div>
      )}
    </div>
  )
}

// ── Sync ──

/** Where the board's read of its sources stands: the last 30 days being
 *  filled in (once), reading, why it stopped, or up to date. */
export function SyncStatus({ status, syncing }: { status: TaskboardStatus | null; syncing: boolean }) {
  const sweep = status?.sweep
  const fill = status?.backfill
  if (fill?.state === 'running') {
    const { agents_done: done = 0, agents_total: total = 0, tasks = 0 } = fill.progress ?? {}
    const more = [total > 0 ? `${Math.min(done, total)} of ${total} agent${total === 1 ? '' : 's'}` : '',
      tasks > 0 ? `${tasks} task${tasks === 1 ? '' : 's'} found` : ''].filter(Boolean).join(' · ')
    return (
      <div className="flex min-w-0 items-center gap-2 text-xs text-zinc-400" title="Your agents' chats of the last 30 days, read once">
        <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-teal-600 dark:text-teal-400" />
        <span className="truncate">Reading the last 30 days…{more ? ` ${more}` : ''}</span>
      </div>
    )
  }
  if (syncing || sweep?.running) {
    return (
      <div className="flex min-w-0 items-center gap-2 text-xs text-zinc-400">
        <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-teal-600 dark:text-teal-400" />
        <span className="truncate">Checking for new tasks…</span>
      </div>
    )
  }
  if (!sweep) return null
  if (sweep.message) {
    return (
      <div className="flex min-w-0 items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400" title={sweep.message}>
        <AlertTriangle className="h-3.5 w-3.5 shrink-0" /><span className="truncate">{sweep.message}</span>
      </div>
    )
  }
  const n = status?.counts?.all ?? 0
  return (
    <div className="flex min-w-0 items-center gap-1.5 text-xs text-zinc-500">
      <Sparkles className="h-3.5 w-3.5 shrink-0 text-teal-600 dark:text-teal-400" />
      <span className="truncate">
        {n === 0 ? 'No tasks yet' : `Up to date · ${n} task${n === 1 ? '' : 's'}`}
        {sweep.last_at ? ` · checked ${agoRelative(sweep.last_at)}` : ''}
      </span>
    </div>
  )
}

// ── A waiting proposal ──

/** Approve sends the next step to your agent; Reject drops it — the same
 *  answer as in the bell. *onReview*: a next step the board wrote has no
 *  Approve here (its words are read, and edited, on its card): Review opens
 *  that instead. */
export function ProposalButtons({ id, onAnswered, onReview }: {
  id: string
  onAnswered: () => void | Promise<void>
  onReview?: () => void
}) {
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null)
  const [error, setError] = useState('')
  const answer = async (approve: boolean) => {
    setBusy(approve ? 'approve' : 'reject')
    setError('')
    try {
      await resolveProposal(id, approve)
      await onAnswered()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const btn = 'flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium disabled:opacity-50'
  return (
    <div onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center gap-1.5">
        {onReview ? (
          <button disabled={busy !== null} onClick={onReview} className={`${btn} bg-teal-600 text-white hover:bg-teal-500`}
            title="Read what it says, change it if you like, and approve it there">
            Review
          </button>
        ) : (
          <button disabled={busy !== null} onClick={() => answer(true)} className={`${btn} bg-emerald-600 text-white hover:bg-emerald-500`}>
            {busy === 'approve' && <Loader2 className="h-3 w-3 animate-spin" />}Approve
          </button>
        )}
        <button disabled={busy !== null} onClick={() => answer(false)} className={`${btn} bg-zinc-800 text-zinc-300 hover:bg-zinc-700`}>
          {busy === 'reject' && <Loader2 className="h-3 w-3 animate-spin" />}Reject
        </button>
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

// ── A quick answer ──

/** The answers to a question the board asks about a task ("Is this a
 *  task?" Yes / No, "Probably done?" Done / Not yet): each a correction,
 *  applied at once; a failure is said under the buttons. */
export function AnswerButtons({ choices, onAct }: {
  choices: { label: string; action: TaskAction; primary?: boolean; title?: string }[]
  onAct: (action: TaskAction) => Promise<void>
}) {
  const [busy, setBusy] = useState<TaskAction | null>(null)
  const [error, setError] = useState('')
  const answer = async (action: TaskAction) => {
    setBusy(action)
    setError('')
    try {
      await onAct(action)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const btn = 'flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium disabled:opacity-50'
  return (
    <div onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center gap-1.5">
        {choices.map((c) => (
          <button
            key={c.action}
            disabled={busy !== null}
            onClick={() => answer(c.action)}
            title={c.title}
            className={`${btn} ${c.primary ? 'bg-emerald-600 text-white hover:bg-emerald-500' : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}`}
          >
            {busy === c.action && <Loader2 className="h-3 w-3 animate-spin" />}{c.label}
          </button>
        ))}
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

// ── Download ──

const EXPORTS: { id: ExportFormat; label: string; detail: string }[] = [
  { id: 'csv', label: 'Spreadsheet (CSV)', detail: 'One row per task — opens in Excel, Numbers or Sheets' },
  { id: 'jsonl', label: 'Data (JSONL)', detail: 'One JSON object per line, every field' },
]

/** The header's download menu: the tasks this tab shows, under the search
 *  and filters, as a file. Only your own tasks are in it. */
export function ExportPanel({ onExport }: { onExport: (format: ExportFormat) => Promise<void> }) {
  const [busy, setBusy] = useState<ExportFormat | null>(null)
  const [error, setError] = useState('')
  const run = async (format: ExportFormat) => {
    setBusy(format)
    setError('')
    try {
      await onExport(format)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  return (
    <div role="menu" aria-label="Download tasks" className="w-64 overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-1.5 shadow-2xl shadow-black/30">
      <p className="px-2.5 pb-1.5 pt-1 text-[11px] leading-relaxed text-zinc-500">The tasks this tab shows, with your search and filters.</p>
      {EXPORTS.map((x) => (
        <button
          key={x.id}
          role="menuitem"
          disabled={busy !== null}
          onClick={() => run(x.id)}
          className="flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-zinc-800/60 disabled:opacity-60"
        >
          {busy === x.id
            ? <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-teal-600 dark:text-teal-400" />
            : <FileDown className="mt-0.5 h-4 w-4 shrink-0 text-teal-600 dark:text-teal-400" />}
          <span className="min-w-0">
            <span className="block text-sm text-zinc-200">{x.label}</span>
            <span className="block text-[11px] leading-snug text-zinc-500">{x.detail}</span>
          </span>
        </button>
      ))}
      {error && <p className="px-2.5 pb-1 pt-0.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

// ── Switch ──

/** On/off, in the board's teal. The label is the row it sits in. */
export function BoardSwitch({ on, onChange, disabled = false, label }: {
  on: boolean; onChange: (on: boolean) => void; disabled?: boolean; label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-60 ${on ? 'bg-teal-600' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${on ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  )
}
