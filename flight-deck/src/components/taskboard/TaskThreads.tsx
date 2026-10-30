import { useEffect, useState } from 'react'
import { GitMerge, Loader2, Search, Split, Undo2, X } from 'lucide-react'
import { taskboardApi, type TaskItem } from '../../services/taskboardApi'
import { FoldHighlight, LaneChip, TaskIcon } from './taskboardParts'
import { agoRelative, foldText } from './taskboardText'

// Threads (v2): one piece of work that came in twice — the same request on
// WhatsApp and by email, a reply the board couldn't tie to its thread — is
// one task once you merge them, and a task that holds two is two once you
// split it. Jev may say "Same as …?"; only you merge. Both can be undone:
// a merged task can be taken out again (Unmerge).

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
const untitled = (t: { title?: string }) => t.title || 'Untitled task'

/** What just happened to the open task (merged, split, taken out), and the
 *  way back or on. */
export interface ThreadNotice {
  taskId: string
  text: string
  action?: { label: string; run: () => Promise<void> | void }
}

export function ThreadNoticeBar({ notice, onDismiss }: { notice: ThreadNotice; onDismiss: () => void }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async () => {
    if (!notice.action) return
    setBusy(true)
    setError('')
    try {
      await notice.action.run()
      // Done what it offered: nothing more to say.
      onDismiss()
    } catch (e) {
      setError(errText(e))
      setBusy(false)
    }
  }
  return (
    <div className="mb-3 rounded-lg border border-teal-500/30 bg-teal-500/5 px-3 py-2 text-xs">
      <div className="flex items-start gap-2">
        <GitMerge className="mt-0.5 h-3.5 w-3.5 shrink-0 text-teal-600 dark:text-teal-400" />
        <span className="min-w-0 flex-1 leading-relaxed text-zinc-300">{notice.text}</span>
        {notice.action && (
          <button onClick={run} disabled={busy} className="flex shrink-0 items-center gap-1 font-medium text-teal-600 hover:underline disabled:opacity-60 dark:text-teal-400">
            {busy && <Loader2 className="h-3 w-3 animate-spin" />}{notice.action.label}
          </button>
        )}
        <button onClick={onDismiss} className="shrink-0 rounded p-0.5 text-zinc-500 hover:text-zinc-200" aria-label="Dismiss">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

/** Jev thinks this is the same task as another one: merge them, look at
 *  the other one first, or say it isn't (no other suggestion comes back
 *  for this task; Undo brings this one back). Only shown — nothing merges
 *  by itself. */
export function SameAsBanner({ sameAs, onMerge, onNotSame, onOpen }: {
  sameAs: NonNullable<TaskItem['flags']['same_as']>
  onMerge: () => Promise<void>
  onNotSame: () => Promise<void>
  onOpen: () => void
}) {
  const [busy, setBusy] = useState<'merge' | 'not_same' | null>(null)
  const [error, setError] = useState('')
  const run = async (which: 'merge' | 'not_same') => {
    setBusy(which)
    setError('')
    try {
      await (which === 'merge' ? onMerge() : onNotSame())
    } catch (e) {
      setError(errText(e))
      setBusy(null)
    }
  }
  const pct = Number.isFinite(sameAs.p) ? Math.round(Math.max(0, Math.min(1, sameAs.p)) * 100) : 0
  return (
    <div className="mt-4 rounded-lg border border-sky-500/30 bg-sky-500/5 px-3 py-2.5">
      <div className="flex items-start gap-1.5 text-xs font-medium text-sky-800 dark:text-sky-300">
        <GitMerge className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span className="min-w-0 break-words">Same as “{sameAs.title || 'another task'}”?</span>
      </div>
      <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
        {pct ? `Jev is ${pct}% sure. ` : ''}Merging puts this task's messages, links and people on that one. You can take it out again.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <button onClick={() => run('merge')} disabled={!!busy} className="flex items-center gap-1 rounded-md bg-sky-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-60">
          {busy === 'merge' && <Loader2 className="h-3 w-3 animate-spin" />}Merge
        </button>
        <button onClick={onOpen} disabled={!!busy} className="rounded-md bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700">
          Open it
        </button>
        <button onClick={() => run('not_same')} disabled={!!busy} className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-60">
          {busy === 'not_same' && <Loader2 className="h-3 w-3 animate-spin" />}Not the same
        </button>
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

/** This task was merged into another: its messages are there now. */
export function MergedIntoBanner({ onOpen, onUnmerge }: { onOpen: () => void; onUnmerge: () => Promise<void> }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const unmerge = async () => {
    setBusy(true)
    setError('')
    try {
      await onUnmerge()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="mt-4 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-300">
        <GitMerge className="h-3.5 w-3.5 shrink-0 text-zinc-500" />Merged into another task
      </div>
      <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">Its messages, links and people are on that task now.</p>
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <button onClick={onOpen} className="rounded-md bg-teal-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-teal-500">Open that task</button>
        <button onClick={unmerge} disabled={busy} className="flex items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 disabled:opacity-60">
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Undo2 className="h-3 w-3" />}Unmerge
        </button>
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

/** The tasks merged into this one: each opens, and each can be taken out. */
export function MergedList({ merged, onOpen, onUnmerge }: {
  merged: { id: string; title: string }[]
  onOpen: (id: string) => void
  onUnmerge: (id: string) => Promise<void>
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const unmerge = async (id: string) => {
    setBusy(id)
    setError('')
    try {
      await onUnmerge(id)
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }
  return (
    <section className="mt-4">
      <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Merged into this task</h4>
      <ul className="mt-1.5 flex flex-col gap-1">
        {merged.map((m) => (
          <li key={m.id} className="flex min-w-0 items-center gap-2 text-xs">
            <GitMerge className="h-3 w-3 shrink-0 text-zinc-500" />
            <button onClick={() => onOpen(m.id)} className="min-w-0 flex-1 truncate text-left text-zinc-300 hover:underline" title={untitled(m)}>
              {untitled(m)}
            </button>
            <button onClick={() => unmerge(m.id)} disabled={busy !== null}
              className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium text-teal-600 hover:bg-teal-500/10 disabled:opacity-60 dark:text-teal-400"
              title="Make it a task of its own again">
              {busy === m.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Undo2 className="h-3 w-3" />}Unmerge
            </button>
          </li>
        ))}
      </ul>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </section>
  )
}

/** "Merge into…": look for the task this one belongs with (any of yours,
 *  by its words), and pick it. */
export function MergePicker({ task, phone, covered, onPick, onClose }: {
  task: TaskItem
  phone: boolean
  /** Something opened over the task: Escape is that one's. */
  covered: boolean
  onPick: (target: TaskItem) => Promise<void>
  onClose: () => void
}) {
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [items, setItems] = useState<TaskItem[] | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), query ? 250 : 0)
    return () => clearTimeout(t)
  }, [query])

  useEffect(() => {
    let gone = false
    taskboardApi.tasks({ view: 'all', q: debounced, sort: 'recent', limit: 20 })
      .then((res) => { if (!gone) { setItems(res.items.filter((t) => t.id !== task.id && !t.merged_into)); setError('') } })
      .catch((e) => { if (!gone) setError(errText(e)) })
    return () => { gone = true }
  }, [debounced, task.id])

  // Escape closes the picker, not the board (it marks the key used).
  useEffect(() => {
    if (covered) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, covered])

  const pick = async (t: TaskItem) => {
    setBusy(t.id)
    setError('')
    try {
      await onPick(t)
    } catch (e) {
      setError(errText(e))
      setBusy(null)
    }
  }
  const terms = foldText(debounced).split(/\s+/).filter(Boolean)
  return (
    <div className="mt-3 rounded-lg border border-zinc-800 bg-zinc-900/60 p-2.5">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Merge into…</span>
        <button onClick={onClose} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close"><X className="h-3.5 w-3.5" /></button>
      </div>
      <div className="relative mt-1.5">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
        <input
          autoFocus={!phone}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }}
          placeholder="Find the task it belongs with…"
          aria-label="Find a task"
          className={`w-full rounded-lg border border-zinc-800 bg-zinc-950 pl-8 pr-2.5 text-zinc-200 placeholder-zinc-500 focus:border-teal-500/60 focus:outline-none ${
            phone ? 'h-11 text-base' : 'py-1.5 text-sm'}`}
        />
      </div>
      <div className="mt-1.5 max-h-64 overflow-y-auto">
        {items === null && !error ? (
          <div className="flex justify-center py-4 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
        ) : items && items.length === 0 ? (
          <p className="py-3 text-center text-xs text-zinc-500">{debounced ? 'No other task matches.' : 'No other tasks yet.'}</p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {(items ?? []).map((t) => (
              <li key={t.id}>
                <button
                  onClick={() => pick(t)}
                  disabled={busy !== null}
                  className={`flex w-full items-center gap-2 rounded-lg px-2 text-left transition-colors hover:bg-zinc-800/60 disabled:opacity-60 ${phone ? 'py-2.5' : 'py-1.5'}`}
                >
                  {busy === t.id ? <Loader2 className="h-4 w-4 shrink-0 animate-spin text-zinc-500" /> : <TaskIcon task={t} />}
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-zinc-200"><FoldHighlight text={untitled(t)} terms={terms} /></span>
                    <span className="block truncate text-[11px] text-zinc-500">
                      {[t.agent_name || t.agent_slug, agoRelative(t.last_event_at ?? t.created_at)].filter(Boolean).join(' · ')}
                    </span>
                  </span>
                  <LaneChip task={t} />
                </button>
              </li>
            ))}
          </ul>
        )}
        {error && <p className="py-2 text-center text-[11px] text-red-600 dark:text-red-400">{error}</p>}
      </div>
      <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-500">
        This task's messages, links and people move to the one you pick. Unmerge takes them back.
      </p>
    </div>
  )
}

/** Splitting: pick the events that are a task of their own. */
export function SplitBar({ picked, total, onSplit, onCancel }: {
  picked: number
  total: number
  onSplit: () => Promise<void>
  onCancel: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const split = async () => {
    setBusy(true)
    setError('')
    try {
      await onSplit()
    } catch (e) {
      setError(errText(e))
      setBusy(false)
    }
  }
  return (
    <div className="mt-2 rounded-lg border border-teal-500/30 bg-teal-500/5 px-3 py-2">
      <p className="text-[11px] leading-relaxed text-zinc-400">
        Tick what belongs to another task; it moves to a new one.
        {picked > 0 && picked >= total ? ' Leave at least one here.' : ''}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <button onClick={split} disabled={busy || picked === 0 || picked >= total}
          className="flex items-center gap-1 rounded-md bg-teal-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-teal-500 disabled:opacity-50">
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Split className="h-3 w-3" />}
          {picked ? `Move ${picked} to a new task` : 'Move to a new task'}
        </button>
        <button onClick={onCancel} disabled={busy} className="rounded-md bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700">
          Cancel
        </button>
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}
