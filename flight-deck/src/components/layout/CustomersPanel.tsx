import { useCallback, useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, Contact, ListChecks, Loader2, RefreshCw, X } from 'lucide-react'
import { taskboardApi, type TaskFacet, type TaskItem } from '../../services/taskboardApi'

// ── Customers ────────────────────────────────────────────────────────
//
// Flight Deck keeps no customer records. What it does know is who is on the
// Taskboard's tasks: the people a task came from, went to or mentions (the
// board's People filter). This lists them — never you, never another deck
// user — with how many tasks each is on, and a person opens to their latest
// tasks, each one a click from the Taskboard. Nobody yet: a friendly word on
// how they get here, and the way to the Taskboard.

/** Another deck user is a colleague, not a customer. */
const isCustomer = (p: TaskFacet) => !p.me && !p.id.startsWith('fd_user:')

function when(epoch: number | null): string {
  if (!epoch) return ''
  const d = new Date(epoch * 1000)
  const days = Math.floor((Date.now() - d.getTime()) / 86400000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function CustomersPanel({ onClose, onOpenTasks, onOpenTask }: {
  onClose: () => void
  /** The Taskboard, every task. */
  onOpenTasks: () => void
  /** The Taskboard, showing this task. */
  onOpenTask: (taskId: string) => void
}) {
  const [people, setPeople] = useState<TaskFacet[] | null>(null)
  const [failed, setFailed] = useState(false)
  const [loading, setLoading] = useState(false)
  const [openId, setOpenId] = useState<string | null>(null)
  const [tasks, setTasks] = useState<Record<string, TaskItem[] | 'loading' | 'failed'>>({})

  const load = useCallback(async () => {
    setLoading(true)
    setFailed(false)
    try {
      const page = await taskboardApi.tasks({ view: 'all', limit: 1 })
      setPeople((page.facets?.party ?? []).filter(isCustomer))
    } catch {
      setFailed(true)
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { void load() }, [load])

  const toggle = async (id: string) => {
    if (openId === id) { setOpenId(null); return }
    setOpenId(id)
    if (Array.isArray(tasks[id])) return
    setTasks((t) => ({ ...t, [id]: 'loading' }))
    try {
      const page = await taskboardApi.tasks({ view: 'all', party: [id], sort: 'recent', limit: 5 })
      setTasks((t) => ({ ...t, [id]: page.items }))
    } catch {
      setTasks((t) => ({ ...t, [id]: 'failed' }))
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 items-center gap-3 px-6 pb-4 pt-6">
        <div className="min-w-0 flex-1">
          <h2 className="text-xl font-bold tracking-tight text-zinc-100">Customers</h2>
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

      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
        {people === null && !failed ? (
          <div className="flex h-40 items-center justify-center text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
        ) : failed ? (
          <Empty
            title="Couldn't reach your tasks just now"
            text="Customers come from your Taskboard, and it didn't answer. Give it another try in a moment."
            action="Try again"
            onAction={() => void load()}
          />
        ) : people!.length === 0 ? (
          <Empty
            title="No customers yet"
            text="When someone emails or messages one of your agents, or a task names them, they'll show up here with everything they've asked for."
            action="Open Tasks"
            onAction={onOpenTasks}
          />
        ) : (
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
                      <span className="block text-xs text-zinc-500">{p.count} task{p.count === 1 ? '' : 's'}</span>
                    </span>
                    {open ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
                  </button>
                  {open && (
                    <div className="border-t border-[var(--fd-line)] px-4 py-2">
                      {list === 'loading' || list === undefined ? (
                        <div className="flex items-center gap-2 py-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Their latest tasks…</div>
                      ) : list === 'failed' ? (
                        <p className="py-2 text-xs text-zinc-500">Their tasks didn't load. Close and open them again to retry.</p>
                      ) : list.length === 0 ? (
                        <p className="py-2 text-xs text-zinc-500">No tasks with them right now.</p>
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
