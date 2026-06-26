import { useEffect, useState } from 'react'
import { CornerDownRight, ArrowRight, Check, X, Loader2, Users, Share2, ClipboardList } from 'lucide-react'
import { apiListVatraAsks, apiListVatraBoard, type VatraAsk, type VatraBoardEntry, type VatraSubtask } from '../stores/basnaStore'

// Vatra collaboration views, split into two pieces so the page can place them in
// different spots: the team PLAN (the Lead's decomposition — who owns what) up top
// where Basna's route plan would be, and the delegation BLACKBOARD (cross-agent
// asks) lower down, between the working agents and the progress log.

function Chip({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`rounded-full border border-zinc-700/60 bg-zinc-800/60 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

// Theme-aware: violet is a non-zinc accent, so it needs explicit light defaults
// plus dark: overrides (zinc would auto-invert, violet does not).
const PANEL = 'rounded-lg border border-violet-300/70 bg-violet-50/60 p-4 dark:border-violet-800/40 dark:bg-violet-950/10'

// ── Team plan: the decomposition (owner · piece), shown for review + while running ──

export function VatraTeamPlan({ subtasks, sharedContext }: { subtasks?: VatraSubtask[]; sharedContext?: string }) {
  return (
    <div className={PANEL}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Users className="h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Team plan</span>
        <Chip className="text-violet-600 dark:text-violet-300">vatra · collaborative</Chip>
        {!!subtasks?.length && <Chip className="text-zinc-400">{subtasks.length} piece(s)</Chip>}
      </div>
      {sharedContext?.trim() && (
        <div className="mb-3 rounded-md border border-violet-300/50 bg-violet-100/40 p-2.5 dark:border-violet-800/40 dark:bg-violet-900/20">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-violet-700 dark:text-violet-300">Shared contract — every piece follows this</div>
          <p className="whitespace-pre-wrap text-[11px] leading-snug text-zinc-600 dark:text-zinc-400">{sharedContext.trim()}</p>
        </div>
      )}
      {subtasks?.length ? (
        <div className="flex flex-wrap gap-1.5">
          {subtasks.map((s) => (
            <span
              key={s.id}
              title={s.brief || ''}
              className="rounded-md border border-zinc-800 bg-zinc-900/50 px-2 py-1 text-[11px] text-zinc-300"
            >
              <span className="text-violet-700 dark:text-violet-300/80">{s.owner_archetype_id}</span>
              <span className="mx-1 text-zinc-700">·</span>
              {s.title}
            </span>
          ))}
        </div>
      ) : (
        <p className="text-xs text-zinc-500">The Lead is decomposing the task into complementary pieces…</p>
      )}
    </div>
  )
}

// ── Blackboard: cross-agent asks (owner → helper), live while the run is active ──

const STATUS_STYLE: Record<VatraAsk['status'], { label: string; cls: string }> = {
  open: { label: 'open', cls: 'text-amber-600 dark:text-amber-300' },
  claimed: { label: 'in progress', cls: 'text-sky-600 dark:text-sky-300' },
  answered: { label: 'answered', cls: 'text-emerald-600 dark:text-emerald-300' },
  dropped: { label: 'dropped', cls: 'text-rose-600 dark:text-rose-300' },
}

function StatusIcon({ status }: { status: VatraAsk['status'] }) {
  if (status === 'answered') return <Check className="h-3 w-3 text-emerald-500" />
  if (status === 'dropped') return <X className="h-3 w-3 text-rose-500" />
  if (status === 'claimed') return <Loader2 className="h-3 w-3 animate-spin text-sky-500" />
  return <Loader2 className="h-3 w-3 text-amber-500" />
}

export function VatraBlackboard({
  sessionId,
  subtasks,
  active,
}: {
  sessionId: string
  subtasks?: VatraSubtask[]
  active?: boolean
}) {
  const [asks, setAsks] = useState<VatraAsk[]>([])
  const [board, setBoard] = useState<VatraBoardEntry[]>([])
  const [open, setOpen] = useState<Record<number, boolean>>({})
  const [openBoard, setOpenBoard] = useState<Record<number, boolean>>({})

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const [a, b] = await Promise.all([apiListVatraAsks(sessionId), apiListVatraBoard(sessionId)])
      if (!cancelled) { setAsks(a); setBoard(b) }
    }
    load()
    if (!active) return () => { cancelled = true }
    const t = setInterval(load, 2500)
    return () => { cancelled = true; clearInterval(t) }
  }, [sessionId, active])

  // Notes + outputs are the shared-memory entries worth showing; narration is noisy.
  const boardShown = board.filter((e) => e.kind === 'note' || e.kind === 'output' || e.kind === 'file')

  const counts = asks.reduce(
    (acc, a) => { acc[a.status] = (acc[a.status] || 0) + 1; return acc },
    {} as Record<string, number>,
  )
  const ownerTitle = (id: string) => subtasks?.find((s) => s.owner_archetype_id === id)?.title

  return (
    <div className={PANEL}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ClipboardList className="h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Shared board</span>
        {boardShown.length > 0 && <Chip className="text-zinc-400">{boardShown.length} entries</Chip>}
        {asks.length > 0 && (
          <span className="ml-auto flex items-center gap-2 text-[11px] text-zinc-500">
            {counts.answered ? <span className="text-emerald-500">{counts.answered} answered</span> : null}
            {(counts.open || counts.claimed) ? <span className="text-amber-500">{(counts.open || 0) + (counts.claimed || 0)} pending</span> : null}
            {counts.dropped ? <span className="text-rose-500">{counts.dropped} dropped</span> : null}
          </span>
        )}
      </div>

      {/* Shared memory — notes & outputs the team streamed, newest first. */}
      {boardShown.length === 0 ? (
        <p className="text-xs text-zinc-600">
          Nothing shared yet — teammates' notes and finished pieces stream here as they work, and
          each can search/read it to build on the others.
        </p>
      ) : (
        <div className="space-y-1.5">
          {[...boardShown].reverse().slice(0, 40).map((e) => {
            const isOpen = !!openBoard[e.id]
            const kindCls = e.kind === 'output'
              ? 'text-emerald-600 dark:text-emerald-300'
              : e.kind === 'file' ? 'text-zinc-400' : 'text-sky-600 dark:text-sky-300'
            return (
              <div key={e.id} className="rounded-md border border-zinc-800 bg-zinc-900/40 p-2">
                <button
                  onClick={() => setOpenBoard((o) => ({ ...o, [e.id]: !o[e.id] }))}
                  className="flex w-full items-center gap-1.5 text-left"
                >
                  <Chip className={kindCls}>{e.kind}</Chip>
                  <span className="truncate text-[11px] text-zinc-300">{ownerTitle(e.from_owner) || e.from_owner}</span>
                  {e.title && <span className="truncate text-[11px] text-zinc-500">· {e.title}</span>}
                </button>
                <p className={`mt-1 pl-1 text-xs text-zinc-400 ${isOpen ? '' : 'line-clamp-2'}`}>{e.content}</p>
              </div>
            )
          })}
        </div>
      )}

      {asks.length > 0 && (
        <div className="mt-4 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <Share2 className="h-3 w-3 text-violet-600 dark:text-violet-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Delegation (asks)</span>
          </div>
          {asks.map((a) => {
            const st = STATUS_STYLE[a.status]
            const isOpen = !!open[a.id]
            const fromLabel = ownerTitle(a.from_owner) || a.from_owner || 'a specialist'
            return (
              <div key={a.id} className="rounded-md border border-zinc-800 bg-zinc-900/40 p-2">
                <button
                  onClick={() => setOpen((o) => ({ ...o, [a.id]: !o[a.id] }))}
                  className="flex w-full items-center gap-1.5 text-left"
                >
                  <StatusIcon status={a.status} />
                  <span className="font-mono text-[10px] text-zinc-500">#{a.id}</span>
                  <span className="truncate text-[11px] text-zinc-400">{a.from_owner}</span>
                  <ArrowRight className="h-3 w-3 shrink-0 text-zinc-600" />
                  <span className="truncate text-[11px] text-zinc-300">{a.answered_by || '—'}</span>
                  {a.depth > 0 && <Chip className="text-zinc-500">d{a.depth}</Chip>}
                  <span className={`ml-auto shrink-0 text-[10px] ${st.cls}`}>{st.label}</span>
                </button>
                <p className="mt-1 line-clamp-2 pl-5 text-xs text-zinc-400" title={a.text}>
                  <span className="text-zinc-600">{fromLabel} asked:</span> {a.text}
                </p>
                {isOpen && (
                  <div className="mt-2 space-y-1 border-t border-zinc-800 pl-5 pt-2">
                    {a.answer ? (
                      <p className="flex gap-1.5 text-xs text-zinc-300">
                        <CornerDownRight className="mt-0.5 h-3 w-3 shrink-0 text-emerald-500" />
                        <span><span className="text-zinc-600">{a.answered_by} answered:</span> {a.answer}</span>
                      </p>
                    ) : (
                      <p className="text-xs text-zinc-600">{a.note || 'No answer recorded.'}</p>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
