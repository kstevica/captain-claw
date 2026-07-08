import { useEffect, useState } from 'react'
import { CornerDownRight, ArrowRight, Check, X, Loader2, Users, Share2, ClipboardList, Download, Pencil } from 'lucide-react'
import { apiListVatraAsks, apiListVatraBoard, type VatraAsk, type VatraBoardEntry, type VatraSubtask } from '../stores/basnaStore'

// Render the shared board as a markdown document — the substantive entries
// (notes/outputs/files), chronological, with author + title headings. Narration is
// the live activity feed, not document content, so it's left out of the export.
function boardToMarkdown(entries: VatraBoardEntry[], titleFor: (id: string) => string | undefined): string {
  const out = ['# Vatra — shared board', '']
  for (const e of entries) {
    if (e.kind === 'narration') continue
    const who = titleFor(e.from_owner) || e.from_owner
    out.push(`## [${e.kind}] ${who}${e.title ? ` — ${e.title}` : ''}`, '', (e.content || '').trim(), '')
  }
  return out.join('\n').trim() + '\n'
}

function downloadMarkdown(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

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

const _GROUPS = ['A', 'B', 'C', 'D'] as const

export function VatraTeamPlan({
  subtasks, sharedContext,
  editable = false, groupInstructions,
  onUpdateSubtask, onRemoveSubtask, onSetGroupInstruction,
}: {
  subtasks?: VatraSubtask[]
  sharedContext?: string
  editable?: boolean
  groupInstructions?: Record<string, string>
  onUpdateSubtask?: (id: string, patch: Partial<VatraSubtask>) => void
  onRemoveSubtask?: (id: string) => void
  onSetGroupInstruction?: (letter: string, text: string) => void
}) {
  // Groups the user has explicitly assigned — only these get an instruction field.
  const assignedGroups = Array.from(
    new Set((subtasks || []).map((s) => (s.group || '').trim()).filter(Boolean)),
  ).sort()
  // The agent whose instructions modal is open (editable mode).
  const [editId, setEditId] = useState<string | null>(null)
  const editing = editId ? (subtasks || []).find((s) => s.id === editId) : undefined

  return (
    <div className={PANEL}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Users className="h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Team plan</span>
        <Chip className="text-violet-600 dark:text-violet-300">vatra · collaborative</Chip>
        {!!subtasks?.length && <Chip className="text-zinc-400">{subtasks.length} piece(s)</Chip>}
        {editable && <Chip className="text-emerald-600 dark:text-emerald-300">editable</Chip>}
      </div>
      {sharedContext?.trim() && (
        <div className="mb-3 rounded-md border border-violet-300/50 bg-violet-100/40 p-2.5 dark:border-violet-800/40 dark:bg-violet-900/20">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-violet-700 dark:text-violet-300">Shared contract — every piece follows this</div>
          <p className="whitespace-pre-wrap text-[11px] leading-snug text-zinc-600 dark:text-zinc-400">{sharedContext.trim()}</p>
        </div>
      )}

      {!subtasks?.length ? (
        <p className="text-xs text-zinc-500">The Lead is decomposing the task into complementary pieces…</p>
      ) : !editable ? (
        <div className="flex flex-wrap gap-1.5">
          {subtasks.map((s) => (
            <span key={s.id} title={s.brief || ''} className="rounded-md border border-zinc-800 bg-zinc-900/50 px-2 py-1 text-[11px] text-zinc-300">
              {s.group && <span className="mr-1 rounded bg-violet-500/20 px-1 text-[9px] font-semibold text-violet-700 dark:text-violet-300">{s.group}</span>}
              <span className="text-violet-700 dark:text-violet-300/80">{s.owner_archetype_id}</span>
              <span className="mx-1 text-zinc-700">·</span>
              {s.title}
            </span>
          ))}
        </div>
      ) : (
        <>
          <div className="space-y-2">
            {subtasks.map((s) => (
              <div key={s.id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                <div className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-200">
                    <span className="text-violet-700 dark:text-violet-300/90">{s.owner_archetype_id}</span>
                    <span className="mx-1 text-zinc-700">·</span>{s.title}
                  </span>
                  {/* Group selector — Auto (Lead's default) or A/B/C/D */}
                  <div className="inline-flex shrink-0 rounded-md border border-zinc-700 bg-zinc-950/50 p-0.5">
                    {(['', ...(_GROUPS as readonly string[])]).map((g) => (
                      <button
                        key={g || 'auto'}
                        onClick={() => onUpdateSubtask?.(s.id, { group: g })}
                        title={g ? `Run in group ${g}` : 'Auto (Lead default phase)'}
                        className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                          (s.group || '') === g ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      >
                        {g || 'Auto'}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => onRemoveSubtask?.(s.id)}
                    title="Remove this agent from the run"
                    className="shrink-0 rounded p-1 text-zinc-500 hover:bg-red-500/20 hover:text-red-300"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <button
                  onClick={() => setEditId(s.id)}
                  className="mt-1.5 flex w-full items-center gap-2 rounded border border-zinc-700 bg-zinc-950/40 px-2 py-1.5 text-left hover:border-violet-500/50"
                >
                  <Pencil className="h-3 w-3 shrink-0 text-zinc-500" />
                  <span className={`min-w-0 flex-1 truncate text-[11px] ${s.brief?.trim() ? 'text-zinc-300' : 'italic text-zinc-600'}`}>
                    {s.brief?.trim() || 'Add instructions…'}
                  </span>
                  <span className="shrink-0 text-[10px] text-violet-500">Edit</span>
                </button>
              </div>
            ))}
          </div>

          {assignedGroups.length > 0 && (
            <div className="mt-3 space-y-1.5">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Per-group instructions</div>
              {assignedGroups.map((g) => (
                <div key={g} className="flex items-start gap-2">
                  <span className="mt-1 rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-violet-700 dark:text-violet-300">{g}</span>
                  <textarea
                    value={groupInstructions?.[g] || ''}
                    onChange={(e) => onSetGroupInstruction?.(g, e.target.value)}
                    rows={2}
                    placeholder={`Extra instructions for every Group ${g} agent…`}
                    className="w-full resize-y rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                  />
                </div>
              ))}
            </div>
          )}

          <p className="mt-2 text-[10px] text-zinc-600">
            Groups run in order A→B→C→D (barrier between); agents in the same group run in parallel. Auto = the Lead's default phase. Edits apply on Run.
          </p>
          {editing && (
            <InstructionModal
              subtask={editing}
              onSave={(brief) => { onUpdateSubtask?.(editing.id, { brief }); setEditId(null) }}
              onClose={() => setEditId(null)}
            />
          )}
        </>
      )}
    </div>
  )
}

function InstructionModal({ subtask, onSave, onClose }: {
  subtask: VatraSubtask
  onSave: (brief: string) => void
  onClose: () => void
}) {
  const [draft, setDraft] = useState(subtask.brief || '')
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="flex max-h-[85vh] w-[560px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <ClipboardList className="h-4 w-4 shrink-0 text-violet-400" />
            <h2 className="truncate text-sm font-semibold text-zinc-100">
              Instructions — <span className="text-violet-700 dark:text-violet-300">{subtask.owner_archetype_id}</span>
              <span className="mx-1 text-zinc-700">·</span>{subtask.title}
            </h2>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"><X className="h-4 w-4" /></button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-4">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            autoFocus
            rows={12}
            placeholder="What this agent should produce, how it should approach it, anything it must source from teammates…"
            className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3">
          <button onClick={onClose} className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800">Cancel</button>
          <button onClick={() => onSave(draft)} className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500">
            <Check className="h-3.5 w-3.5" /> Save
          </button>
        </div>
      </div>
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

  // Show everything the team streamed — notes/outputs/files plus live narration, so
  // the board has content from the first moments of the run (outputs arrive later).
  const boardShown = board

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
        <span className="ml-auto flex items-center gap-2 text-[11px] text-zinc-500">
          {asks.length > 0 && (
            <>
              {counts.answered ? <span className="text-emerald-500">{counts.answered} answered</span> : null}
              {(counts.open || counts.claimed) ? <span className="text-amber-500">{(counts.open || 0) + (counts.claimed || 0)} pending</span> : null}
              {counts.dropped ? <span className="text-rose-500">{counts.dropped} dropped</span> : null}
            </>
          )}
          {board.some((e) => e.kind !== 'narration') && (
            <button
              onClick={() => downloadMarkdown('vatra-shared-board.md', boardToMarkdown(board, (id) => subtasks?.find((s) => s.owner_archetype_id === id)?.title))}
              title="Export the shared board (notes & outputs) as Markdown"
              className="flex items-center gap-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
          )}
        </span>
      </div>

      {/* Shared memory — notes & outputs the team streamed, newest first. */}
      {boardShown.length === 0 ? (
        <p className="text-xs text-zinc-600">
          Nothing shared yet — teammates' notes and finished pieces stream here as they work, and
          each can search/read it to build on the others.
        </p>
      ) : (
        <div className="max-h-80 space-y-1.5 overflow-auto pr-1">
          {[...boardShown].reverse().slice(0, 40).map((e) => {
            const isOpen = !!openBoard[e.id]
            const kindCls = e.kind === 'output'
              ? 'text-emerald-600 dark:text-emerald-300'
              : e.kind === 'note' ? 'text-sky-600 dark:text-sky-300'
              : 'text-zinc-500'  // narration / file — live activity, subtler
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
