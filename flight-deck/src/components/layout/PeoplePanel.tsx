import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Contact, ListChecks, Loader2, RefreshCw, Search, X } from 'lucide-react'
import { taskboardApi, type PeoplePending, type PeopleSort, type Person, type TaskItem } from '../../services/taskboardApi'

// ── People ───────────────────────────────────────────────────────────
//
// Flight Deck keeps no contact records. What it does know is who is on the
// Taskboard's tasks: the people a task came from, went to or mentions (the
// board's People filter). This lists them — never you, never another deck
// user — with how many tasks each is on, and a person opens to their latest
// tasks, each one a click from the Taskboard. Nobody yet: a friendly word on
// how they get here, and the way to the Taskboard.
//
// The list is the server's (/fd/taskboard/people: everyone, not the board's
// top 50), searched by name or address and sorted by task count or name, a
// page at a time; the sort is remembered on this browser.

const PAGE = 100
const SORT_KEY = 'fd:people-sort'

// The pending filter's choices: how many of a person's tasks are still open.
const PENDING: { id: PeoplePending; label: string; hint: string }[] = [
  { id: '', label: 'All', hint: 'Everyone, however many tasks are pending' },
  { id: '1', label: '1', hint: 'One pending task' },
  { id: '2-5', label: '2–5', hint: 'Two to five pending tasks' },
  { id: '6-9', label: '6–9', hint: 'Six to nine pending tasks' },
  { id: '10+', label: '10+', hint: 'Ten or more pending tasks' },
]

function savedSort(): PeopleSort {
  try { return localStorage.getItem(SORT_KEY) === 'name' ? 'name' : 'count' } catch { return 'count' }
}

function when(epoch: number | null): string {
  if (!epoch) return ''
  const d = new Date(epoch * 1000)
  const days = Math.floor((Date.now() - d.getTime()) / 86400000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function PeoplePanel({ onClose, onOpenTasks, onOpenTask, covered = false }: {
  onClose: () => void
  /** The Taskboard is open over it (a task opened from here): keys are the
   *  board's, and closing the board comes back here, read again. */
  covered?: boolean
  /** The Taskboard, every task. */
  onOpenTasks: () => void
  /** The Taskboard, showing this task. */
  onOpenTask: (taskId: string) => void
}) {
  const [people, setPeople] = useState<Person[] | null>(null)
  const [total, setTotal] = useState(0)
  const [failed, setFailed] = useState(false)
  const [loading, setLoading] = useState(false)
  const [more, setMore] = useState(false)
  const [openId, setOpenId] = useState<string | null>(null)
  const [tasks, setTasks] = useState<Record<string, TaskItem[] | 'loading' | 'failed'>>({})
  // What the box says, and the search it settled on (a moment after typing stops).
  const [query, setQuery] = useState('')
  const [q, setQ] = useState('')
  const [sort, setSort] = useState<PeopleSort>(savedSort)
  const [pending, setPending] = useState<PeoplePending>('')
  // Only the latest load lands: an older answer arriving late is dropped.
  const loadSeq = useRef(0)

  useEffect(() => {
    const t = setTimeout(() => setQ(query.trim()), 200)
    return () => clearTimeout(t)
  }, [query])

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    setFailed(false)
    try {
      const page = await taskboardApi.people({ q, sort, pending, limit: PAGE })
      if (seq !== loadSeq.current) return
      setPeople(page.people)
      setTotal(page.total)
    } catch {
      if (seq === loadSeq.current) setFailed(true)
    } finally {
      if (seq === loadSeq.current) setLoading(false)
    }
  }, [q, sort, pending])
  useEffect(() => { void load() }, [load])

  const showMore = async () => {
    if (!people) return
    const seq = loadSeq.current
    setMore(true)
    try {
      const page = await taskboardApi.people({ q, sort, pending, limit: PAGE, offset: people.length })
      if (seq !== loadSeq.current) return
      setPeople((cur) => [...(cur ?? []), ...page.people])
      setTotal(page.total)
    } catch { /* the button stays: try again */ } finally {
      setMore(false)
    }
  }

  const pickSort = (s: PeopleSort) => {
    setSort(s)
    try { localStorage.setItem(SORT_KEY, s) } catch { /* only this visit */ }
  }

  // Escape closes it, as the nav's other views close (the Line, Scheduled tasks);
  // in the search box with words in it, it clears them first. Not while the
  // Taskboard is over it: that Escape is the board's.
  useEffect(() => {
    if (covered) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, covered])

  // The open person, for effects that read their tasks again.
  const openIdRef = useRef(openId)
  openIdRef.current = openId

  // A person's tasks under them: their latest five, or — filtered by pending
  // tasks — only the pending ones (still in play: open or maybe), newest first.
  const readTasks = useCallback(async (id: string) => {
    setTasks((t) => ({ ...t, [id]: 'loading' }))
    try {
      const page = await taskboardApi.tasks(pending
        ? { view: 'all', party: [id], status: ['open', 'maybe'], sort: 'recent', limit: 50 }
        : { view: 'all', party: [id], sort: 'recent', limit: 5 })
      setTasks((t) => ({ ...t, [id]: page.items }))
    } catch {
      setTasks((t) => ({ ...t, [id]: 'failed' }))
    }
  }, [pending])

  // The filter changed what the lists hold: read the open one again.
  const firstPending = useRef(true)
  useEffect(() => {
    if (firstPending.current) { firstPending.current = false; return }
    setTasks({})
    if (openIdRef.current) void readTasks(openIdRef.current)
  }, [readTasks])

  const toggle = async (id: string) => {
    if (openId === id) { setOpenId(null); return }
    setOpenId(id)
    if (Array.isArray(tasks[id])) return
    await readTasks(id)
  }

  // Back from the Taskboard: what was done there shows here — the counts,
  // and the open person's tasks — with the search, the order and the place
  // in the list kept.
  const wasCovered = useRef(covered)
  useEffect(() => {
    if (wasCovered.current && !covered) {
      const shownNow = people?.length ?? 0
      void (async () => {
        try {
          const page = await taskboardApi.people({ q, sort, pending, limit: Math.max(PAGE, shownNow) })
          setPeople(page.people)
          setTotal(page.total)
        } catch { /* what was there stays */ }
      })()
      setTasks({})
      if (openId) void readTasks(openId)
    }
    wasCovered.current = covered
  }, [covered, people?.length, q, sort, pending, openId, readTasks])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 items-center gap-3 px-6 pb-4 pt-6">
        <div className="min-w-0 flex-1">
          <h2 className="text-xl font-bold tracking-tight text-zinc-100">People</h2>
          <p className="mt-0.5 text-sm text-zinc-500">The people on your agents' tasks — who asked, who was answered, who was mentioned.</p>
        </div>
        <button onClick={() => void load()} disabled={loading} title="Check again" aria-label="Check again"
          className="flex h-9 w-9 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-60">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
        <button onClick={onClose} title="Close" aria-label="Close"
          className="flex h-9 w-9 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
          <X className="h-4 w-4" />
        </button>
      </header>

      {/* Search and sort: shown once there is anyone (or a search) to work on. */}
      {(total > 0 || q || query || pending) && !failed && (
        <div className="flex shrink-0 flex-wrap items-center gap-2 px-6 pb-4">
          <label className="relative min-w-[12rem] flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Escape' && query) { e.preventDefault(); setQuery('') } }}
              placeholder="Search by name or email"
              aria-label="Search people"
              className="h-10 w-full rounded-full border border-[var(--fd-line)] bg-transparent pl-9 pr-9 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-violet-500/50 focus:outline-none"
            />
            {query && (
              <button onClick={() => setQuery('')} title="Clear" aria-label="Clear search"
                className="absolute right-2 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full text-zinc-500 hover:text-zinc-200">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </label>
          <div role="group" aria-label="Sort people" className="flex shrink-0 rounded-full border border-[var(--fd-line)] p-0.5">
            {([['count', 'Most tasks'], ['name', 'Name A–Z']] as const).map(([key, label]) => (
              <button key={key} onClick={() => pickSort(key)} aria-pressed={sort === key}
                className={`h-8 rounded-full px-3 text-xs font-medium transition-colors ${
                  sort === key ? 'bg-[var(--fd-selected)] text-zinc-100' : 'text-zinc-500 hover:text-zinc-200'}`}>
                {label}
              </button>
            ))}
          </div>
          <div role="group" aria-label="Pending tasks" className="flex shrink-0 items-center rounded-full border border-[var(--fd-line)] p-0.5">
            <span className="px-2 text-xs text-zinc-500">Pending</span>
            {PENDING.map((o) => (
              <button key={o.id || 'all'} onClick={() => setPending(o.id)} aria-pressed={pending === o.id} title={o.hint}
                className={`h-8 min-w-[2rem] rounded-full px-2.5 text-xs font-medium tabular-nums transition-colors ${
                  pending === o.id ? 'bg-[var(--fd-selected)] text-zinc-100' : 'text-zinc-500 hover:text-zinc-200'}`}>
                {o.label}
              </button>
            ))}
          </div>
          <span className="w-full text-xs text-zinc-500 wide:w-auto" aria-live="polite">
            {q || pending ? `${total} match${total === 1 ? '' : 'es'}` : `${total} ${total === 1 ? 'person' : 'people'}`}
          </span>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
        {people === null && !failed ? (
          <div className="flex h-40 items-center justify-center text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
        ) : failed ? (
          <Empty
            title="Couldn't reach your tasks just now"
            text="People come from your tasks, and Tasks didn't answer. Give it another try in a moment."
            action="Try again"
            onAction={() => void load()}
          />
        ) : people!.length === 0 && (q || pending) ? (
          <Empty
            title="No one matches"
            text={[
              q && `Nobody on your tasks has “${q}” in their name or address`,
              pending && `${q ? 'with' : 'Nobody has'} ${PENDING.find((o) => o.id === pending)?.label} pending task${pending === '1' ? '' : 's'}`,
            ].filter(Boolean).join(' ') + '.'}
            action={q && pending ? 'Clear search and filter' : q ? 'Clear search' : 'Show everyone'}
            onAction={() => { setQuery(''); setPending('') }}
          />
        ) : people!.length === 0 ? (
          <Empty
            title="No people yet"
            text="When someone emails or messages one of your agents, or a task names them, they'll show up here with everything they've asked for."
            action="Open Tasks"
            onAction={onOpenTasks}
          />
        ) : (
          <>
          <ul className="space-y-2">
            {people!.map((p) => {
              const open = openId === p.id
              const list = tasks[p.id]
              return (
                <li key={p.id} className="fd-card overflow-hidden">
                  <button onClick={() => void toggle(p.id)} aria-expanded={open}
                    className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-[var(--fd-hover)]">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--fd-accent-soft)] text-sm font-semibold text-[var(--fd-accent-text)]">
                      {(p.label || '?').trim().slice(0, 1).toUpperCase()}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[15px] font-semibold text-zinc-100">{p.label || p.id}</span>
                      <span className="block text-xs text-zinc-500">
                        {p.count} task{p.count === 1 ? '' : 's'}
                        <span className="text-zinc-600"> · </span>
                        <span className={p.pending > 0 ? 'text-amber-700 dark:text-amber-400' : undefined}>{p.pending} pending</span>
                        <span className="text-zinc-600"> · </span>
                        <span className={p.done > 0 ? 'text-emerald-700 dark:text-emerald-400' : undefined}>{p.done} done</span>
                      </span>
                    </span>
                    {open ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
                  </button>
                  {open && (
                    <div className="border-t border-[var(--fd-line)] px-4 py-2">
                      {list === 'loading' || list === undefined ? (
                        <div className="flex items-center gap-2 py-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> {pending ? 'Their pending tasks…' : 'Their latest tasks…'}</div>
                      ) : list === 'failed' ? (
                        <p className="py-2 text-xs text-zinc-500">Their tasks didn't load. Close and open them again to retry.</p>
                      ) : list.length === 0 ? (
                        <p className="py-2 text-xs text-zinc-500">{pending ? 'No pending tasks with them right now.' : 'No tasks with them right now.'}</p>
                      ) : (
                        <ul>
                          {list.map((t) => (
                            <li key={t.id}>
                              <button onClick={() => onOpenTask(t.id)}
                                className="flex w-full items-center gap-3 rounded-xl px-2 py-2 text-left transition-colors hover:bg-[var(--fd-hover)]">
                                <ListChecks className="h-4 w-4 shrink-0 text-zinc-500" />
                                <span className="min-w-0 flex-1 truncate text-sm text-zinc-200">{t.title || t.ask || 'Untitled task'}</span>
                                <span className="shrink-0 text-xs text-zinc-500">{when(t.last_event_at ?? t.created_at)}</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
          {people!.length < total && (
            <div className="mt-4 flex justify-center">
              <button onClick={() => void showMore()} disabled={more} className="fd-btn-quiet h-9 px-4 text-xs">
                {more && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                Show more ({total - people!.length} left)
              </button>
            </div>
          )}
          </>
        )}
      </div>
    </div>
  )
}

function Empty({ title, text, action, onAction }: { title: string; text: string; action: string; onAction: () => void }) {
  return (
    <div className="flex min-h-[320px] flex-col items-center justify-center text-center">
      <div className="fd-empty-mark mb-5"><Contact className="h-6 w-6" /></div>
      <h3 className="fd-empty-title text-lg">{title}</h3>
      <p className="fd-empty-text mt-1.5 max-w-sm text-sm leading-relaxed">{text}</p>
      <button onClick={onAction} className="fd-btn-primary mt-6 h-10 px-5 text-sm">{action}</button>
    </div>
  )
}
