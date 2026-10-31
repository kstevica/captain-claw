import { useEffect, useState } from 'react'
import { Ban, Bot, Check, CheckCheck, Clock, Loader2, Minus, RotateCcw, Square, UsersRound, X } from 'lucide-react'
import type { HandOverBatch, TaskAction, TaskItem } from '../../services/taskboardApi'
import { KeyHint } from './taskboardParts'
import { HandOverForm } from './TaskNextStep'
import { OWES_LABEL, SNOOZES, TASK_KEYS, batchLine, fitsBatch, snoozeUntil, usualAgent, type BatchMove } from './taskboardText'

// Several tasks at once, like Gmail: tick them (or `x`), then one move for all.
// A correction applies to each as if made on its own, with Undo for the lot. Handing
// them to an agent takes one set of words for all, and it goes through them one at a
// time (taskboard_handover). Approving a next step is never here: each is read on its own.

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
const SHOWN = 4  // tasks named in the form, the rest counted
const MOVES = ['me', 'agent', 'them', 'nobody', 'unclear'] as const

/** A row's tick box: the whole left edge is its target, so a phone can hit it. */
export function PickBox({ on, mixed = false, head = false, label, onPick }: {
  on: boolean
  mixed?: boolean
  /** The list's own, on its heading line. */
  head?: boolean
  label: string
  /** Shift held: everything since the last tick goes the same way. */
  onPick: (range: boolean) => void
}) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={mixed ? 'mixed' : on}
      aria-label={label}
      title={head ? label : `${label} (${TASK_KEYS.pick})`}
      onMouseDown={(e) => { if (e.shiftKey) e.preventDefault() }}
      onClick={(e) => { e.stopPropagation(); onPick(e.shiftKey) }}
      className={`group flex w-11 shrink-0 justify-center focus:outline-none wide:w-9 ${head ? 'h-7 items-center' : 'self-stretch pt-[17px]'}`}
    >
      <span className={`flex h-4 w-4 items-center justify-center rounded border transition-colors group-focus-visible:ring-1 group-focus-visible:ring-teal-500/60 ${
        on || mixed ? 'border-teal-600 bg-teal-600 text-white' : 'border-zinc-600 group-hover:border-zinc-400'}`}>
        {mixed ? <Minus className="h-3 w-3" /> : on ? <Check className="h-3 w-3" /> : null}
      </span>
    </button>
  )
}

/** What can be done to the picked tasks. A button says how many it reaches when
 *  that isn't all of them; the keys are the open task's, for these instead. */
export function BatchBar({ tasks, phone, keysRef, heldRef, onApply, onHandOver, onClear }: {
  tasks: TaskItem[]
  phone: boolean
  /** Filled with what the bar answers to (true: the key was used). */
  keysRef: { current: ((key: string) => boolean) | null }
  /** Filled with whether something of it is open: moving around waits. */
  heldRef: { current: boolean }
  onApply: (ids: string[], action: TaskAction, value?: string | number) => Promise<void>
  onHandOver: (ids: string[], agent: string, words: string) => Promise<void>
  onClear: () => void
}) {
  const [open, setOpen] = useState<'' | 'snooze' | 'move' | 'agent'>('')
  const [busy, setBusy] = useState(false)
  const [words, setWords] = useState('')
  const [error, setError] = useState('')
  const fit = (move: BatchMove) => tasks.filter((t) => fitsBatch(move, t))
  const said = (text: string, n: number) => (n < tasks.length ? `${text} (${n})` : text)

  const apply = async (move: BatchMove, action: TaskAction, value?: string | number) => {
    setBusy(true)
    setError('')
    try {
      await onApply(fit(move).map((t) => t.id), action, value)
    } catch (e) {
      setError(errText(e))
      setBusy(false)
    }
  }
  const pickMove = (o: (typeof MOVES)[number]) => {
    if (o === 'agent') setOpen('agent')
    else void apply('move', 'owes_next', o)
  }

  // Escape closes what is open here, not the board (it marks the key used).
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setOpen('') } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  useEffect(() => {
    heldRef.current = !!open
    keysRef.current = (key) => {
      if (busy) return true
      if (open === 'agent') return false
      if (open === 'snooze' || open === 'move') {
        const at = Number(key) - 1
        if (open === 'snooze' && SNOOZES[at]) void apply('snooze', 'snooze', snoozeUntil(SNOOZES[at].id))
        else if (open === 'move' && MOVES[at]) pickMove(MOVES[at])
        else if (key === TASK_KEYS.snooze || key === TASK_KEYS.move) setOpen('')
        else return false
        return true
      }
      const moves: Record<string, [number, () => void]> = {
        [TASK_KEYS.agent]: [fit('agent').length, () => setOpen('agent')],
        [TASK_KEYS.done]: [fit('done').length, () => void apply('done', 'done')],
        [TASK_KEYS.snooze]: [fit('snooze').length, () => setOpen('snooze')],
        [TASK_KEYS.move]: [fit('move').length, () => setOpen('move')],
        [TASK_KEYS.notTask]: [fit('not_task').length, () => void apply('not_task', 'not_task')],
        [TASK_KEYS.reopen]: [fit('reopen').length, () => void apply('reopen', 'reopen')],
        [TASK_KEYS.gotIt]: [fit('got_it').length, () => void apply('got_it', 'ack_failed')],
      }
      const move = moves[key]
      if (!move?.[0]) return false
      move[1]()
      return true
    }
    return () => { keysRef.current = null; heldRef.current = false }
  })

  const btn = 'flex items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50'
  const primary = 'flex items-center gap-1.5 rounded-lg bg-teal-600 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-teal-500 disabled:opacity-60'
  const icon = 'h-3.5 w-3.5'
  const going = fit('agent')

  return (
    <div className={`border-b border-zinc-800 bg-teal-500/5 px-3 py-2 wide:px-4 ${open === 'agent' ? 'max-h-[70%] overflow-y-auto' : ''}`} data-batch-bar>
      {open === 'agent' ? (
        <HandOverForm
          taskAgent={usualAgent(going)}
          phone={phone}
          words={words}
          onWords={setWords}
          count={going.length}
          onCancel={() => setOpen('')}
          onHandOver={(agent, text) => onHandOver(going.map((t) => t.id), agent, text)}
        >
          <ul className="flex flex-col gap-0.5 text-xs text-zinc-400">
            {going.slice(0, SHOWN).map((t) => <li key={t.id} className="truncate">{t.title || 'Untitled task'}</li>)}
            {going.length > SHOWN && <li className="text-zinc-500">and {going.length - SHOWN} more</li>}
            {going.length < tasks.length && (
              <li className="text-zinc-500">{tasks.length - going.length} of what you picked can't go: closed, or already in line.</li>
            )}
          </ul>
        </HandOverForm>
      ) : open === 'snooze' ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-zinc-400">Snooze {fit('snooze').length} until</span>
          {SNOOZES.map((s, i) => (
            <button key={s.id} onClick={() => void apply('snooze', 'snooze', snoozeUntil(s.id))} disabled={busy} className={btn}>
              {s.label}<KeyHint k={String(i + 1)} />
            </button>
          ))}
          <button onClick={() => setOpen('')} disabled={busy} className={btn}>Cancel<KeyHint k="Esc" /></button>
          {busy && <Loader2 className={`${icon} animate-spin text-zinc-500`} />}
        </div>
      ) : open === 'move' ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-zinc-400">Next move on {fit('move').length}</span>
          {MOVES.map((o, i) => (
            <button key={o} onClick={() => pickMove(o)} disabled={busy} className={btn}>
              {OWES_LABEL[o]}{o === 'agent' ? '…' : ''}<KeyHint k={String(i + 1)} />
            </button>
          ))}
          <button onClick={() => setOpen('')} disabled={busy} className={btn}>Cancel<KeyHint k="Esc" /></button>
          {busy && <Loader2 className={`${icon} animate-spin text-zinc-500`} />}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 flex items-center gap-1 text-xs font-medium text-zinc-200">
            {busy && <Loader2 className={`${icon} animate-spin text-zinc-500`} />}{tasks.length} picked
          </span>
          {going.length > 0 && (
            <button onClick={() => setOpen('agent')} disabled={busy} className={primary} title="One set of words for all: it takes them one at a time">
              <Bot className={icon} />{said('Hand to agent', going.length)}<KeyHint k={TASK_KEYS.agent} onPrimary />
            </button>
          )}
          {fit('done').length > 0 && (
            <button onClick={() => void apply('done', 'done')} disabled={busy} className={btn}>
              <CheckCheck className={icon} />{said('Done', fit('done').length)}<KeyHint k={TASK_KEYS.done} />
            </button>
          )}
          {fit('snooze').length > 0 && (
            <button onClick={() => setOpen('snooze')} disabled={busy} className={btn}>
              <Clock className={icon} />{said('Snooze', fit('snooze').length)}<KeyHint k={TASK_KEYS.snooze} />
            </button>
          )}
          {fit('move').length > 0 && (
            <button onClick={() => setOpen('move')} disabled={busy} className={btn}>
              <UsersRound className={icon} />{said('Next move', fit('move').length)}<KeyHint k={TASK_KEYS.move} />
            </button>
          )}
          {fit('not_task').length > 0 && (
            <button onClick={() => void apply('not_task', 'not_task')} disabled={busy} className={btn} title="Nothing anyone had to do: they leave the board">
              <Ban className={icon} />{said('Not a task', fit('not_task').length)}<KeyHint k={TASK_KEYS.notTask} />
            </button>
          )}
          {fit('reopen').length > 0 && (
            <button onClick={() => void apply('reopen', 'reopen')} disabled={busy} className={btn}>
              <RotateCcw className={icon} />{said('Reopen', fit('reopen').length)}<KeyHint k={TASK_KEYS.reopen} />
            </button>
          )}
          {fit('got_it').length > 0 && (
            <button onClick={() => void apply('got_it', 'ack_failed')} disabled={busy} className={btn} title="Seen: off Needs me">
              <Check className={icon} />{said('Got it', fit('got_it').length)}<KeyHint k={TASK_KEYS.gotIt} />
            </button>
          )}
          <span className="flex-1" />
          <button onClick={onClear} disabled={busy} className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200" title="Pick none (Esc)">
            <X className={icon} />Clear
          </button>
        </div>
      )}
      {error && <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

/** How a batch stands, while its agent works through it and a little after. */
export function BatchProgress({ batch, onStop, onOpen, onDismiss }: {
  batch: HandOverBatch
  onStop: () => Promise<void>
  onOpen: (taskId: string) => void
  onDismiss: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const going = !batch.ended_at
  const through = batch.total - batch.waiting - (batch.on ? 1 : 0)
  const stop = async () => {
    setBusy(true)
    setError('')
    try {
      await onStop()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="border-b border-zinc-800 bg-sky-500/5 px-3 py-2 text-xs wide:px-4" data-batch-progress={batch.id}>
      <div className="flex items-start gap-2">
        {going ? <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-sky-600 dark:text-sky-400" />
          : <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-600 dark:text-sky-400" />}
        <div className="min-w-0 flex-1 leading-relaxed">
          <p className="text-zinc-200">{batchLine(batch)}{!going && ' Look them over in Needs me.'}</p>
          {batch.on && (
            <button onClick={() => onOpen(batch.on!.id)} className="block max-w-full truncate text-left text-sky-700 hover:underline dark:text-sky-400">
              On “{batch.on.title || 'Untitled task'}”
            </button>
          )}
          <p className="truncate text-zinc-500" title={batch.words}>You asked: {batch.words}</p>
          {error && <p className="text-[11px] text-red-600 dark:text-red-400">{error}</p>}
        </div>
        {going ? (
          batch.waiting > 0 && (
            <button onClick={() => void stop()} disabled={busy} className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1 font-medium text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50"
              title="What still waits doesn't run and comes back to you. The one it is on finishes.">
              {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />}Stop the rest
            </button>
          )
        ) : (
          <button onClick={onDismiss} className="shrink-0 rounded p-0.5 text-zinc-500 hover:text-zinc-200" aria-label="Dismiss"><X className="h-3.5 w-3.5" /></button>
        )}
      </div>
      {going && (
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-zinc-800" role="progressbar" aria-valuemin={0} aria-valuemax={batch.total} aria-valuenow={through}>
          <div className="h-full rounded-full bg-sky-500 transition-all" style={{ width: `${(through / Math.max(batch.total, 1)) * 100}%` }} />
        </div>
      )}
    </div>
  )
}
