import { useEffect, useRef, useState } from 'react'
import { ArrowRight, HelpCircle, Loader2, X } from 'lucide-react'
import { fmtWhen } from '../../services/schedulerApi'
import { taskboardApi, type TaskItem, type TaskWhy } from '../../services/taskboardApi'
import { JudgeChip } from './taskboardParts'
import { agoRelative, answerText } from './taskboardText'

// "Why is this here?" — the decisions that put a task where it is (Jev's,
// a rule's, the model's, yours), latest first, each with its reason in plain
// words and the move it made. Read only when asked for. A popover on a
// desktop; on a phone it opens in place, under its button.

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

export function WhyThisHere({ task, phone }: { task: TaskItem; phone: boolean }) {
  const [open, setOpen] = useState(false)
  const [why, setWhy] = useState<TaskWhy | null>(null)
  const [error, setError] = useState('')
  const box = useRef<HTMLDivElement>(null)

  // Read when opened, and again when the task moves while it is open.
  useEffect(() => {
    if (!open) return
    let gone = false
    taskboardApi.why(task.id)
      .then((w) => { if (!gone) { setWhy(w); setError('') } })
      .catch((e) => { if (!gone) setError(errText(e)) })
    return () => { gone = true }
  }, [open, task.id, task.updated_at])

  // The popover closes on Escape (marking the key used, so the board stays
  // open) and on a click anywhere else.
  useEffect(() => {
    if (!open || phone) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setOpen(false) } }
    const onDown = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false) }
    window.addEventListener('keydown', onKey)
    document.addEventListener('mousedown', onDown)
    return () => {
      window.removeEventListener('keydown', onKey)
      document.removeEventListener('mousedown', onDown)
    }
  }, [open, phone])

  const list = <WhyList why={why} error={error} fallback={task.flags?.why || ''} />
  return (
    <div ref={box} className="relative mt-2.5">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        title={task.flags?.why || 'The decisions that put this task where it is'}
        className="flex items-center gap-1 text-[11px] font-medium text-teal-600 hover:underline dark:text-teal-400"
      >
        <HelpCircle className="h-3.5 w-3.5" />Why is this here?
      </button>
      {open && (phone ? (
        <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2.5">{list}</div>
      ) : (
        <div
          role="dialog"
          aria-label="Why is this here?"
          className="absolute left-0 right-0 top-full z-20 mt-1.5 max-h-80 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 shadow-2xl shadow-black/30"
        >
          <div className="mb-2 flex items-center gap-2">
            <span className="flex-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Why is this here?</span>
            <button onClick={() => setOpen(false)} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          {list}
        </div>
      ))}
    </div>
  )
}

/** The decisions, latest first, under the server's summary (or the task's
 *  own few words when there is none). */
function WhyList({ why, error, fallback }: { why: TaskWhy | null; error: string; fallback: string }) {
  if (error && !why) return <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
  if (!why) return <div className="flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Loading…</div>
  const items = why.items ?? []
  const summary = why.summary || fallback
  return (
    <div className="flex flex-col gap-2.5">
      {summary && <p className="text-xs leading-relaxed text-zinc-300">{summary}</p>}
      {items.length === 0 ? (
        <p className="text-xs text-zinc-500">No decisions yet</p>
      ) : (
        <ol className="flex flex-col gap-2.5">
          {items.map((d, i) => {
            const answers = Object.entries(d.answers ?? {}).map(([q, a]) => answerText(q, a))
            return (
              <li key={d.id ?? `${d.at}:${i}`} className={i > 0 || summary ? 'border-t border-zinc-800/70 pt-2.5' : ''}>
                <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
                  <JudgeChip judge={d.judge} />
                  <span title={fmtWhen(d.at)}>{agoRelative(d.at)}</span>
                </div>
                {d.why && <p className="mt-1 break-words text-xs leading-relaxed text-zinc-300">{d.why}</p>}
                {d.applied && (
                  <p className="mt-0.5 break-words text-[11px] text-zinc-400">
                    <ArrowRight className="mr-1 inline h-3 w-3 align-[-2px]" />{d.applied}
                  </p>
                )}
                {answers.length > 0 && <p className="mt-0.5 break-words text-[10px] text-zinc-500">{answers.join(' · ')}</p>}
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
